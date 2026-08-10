use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, bail};

use crate::ast::{
    Ast, EnumVariant, ExprId, Expression, Literal, NamedExpr, Parameter,
    Pattern, PatternId, Range32, ReturnKind, ReturnSignature, SignatureId,
    Statement, StmtId, StructField, SwitchCase, Symbol, TokenSpan,
    live_subject,
};
use crate::ast_display::{display_expr, display_stmt};
use crate::ir::{
    BlockId, EnumLayout, EnumVariantLayout, FieldLayout, IrBinOp, IrBlock,
    IrConstant, IrExtern, IrFunction, IrLocal, IrModule, IrOperand, IrRvalue,
    IrStatement, IrTerminator, IrUnOp, LocalId, StructLayout,
};
use crate::lexer::Position;
use crate::parser::Operator;
use crate::types::{Type, spelled};

// The names the compiler reads as its own wherever they are written: the calls
// it answers, and the three constants a `when` chooses on. What makes them
// reserved is that a program may not declare one, and that is asked in one
// place.
pub const COMPILER_NAMES: &[&str] = &[
    "alignof",
    "assert",
    "cast",
    "flags_has",
    "live_slots",
    "ptr_cast",
    "ptr_to",
    "sizeof",
    "slice_from",
    "slice_len",
    "str_len",
    "TARGET_LINUX",
    "TARGET_MACOS",
    "TARGET_WINDOWS",
    "type_id",
    "typename",
    "wrap_add",
    "wrap_mul",
    "wrap_sub",
];

// The liveness bookkeeping a generational container carries beside its
// generations and free list, named here because the synthesis writes it, the
// library maintains it and the `for` walk reads it.
pub const LIVE_WORDS: &str = "live_words";
pub const LIVE_COUNT: &str = "live_count";
const SLOTS_PER_WORD: usize = 64;

// One word per sixty-four slots, and never a zero-length array, since a
// container of no capacity still has to have the field to be laid out.
pub fn live_word_count(capacity: usize) -> usize {
    capacity.div_ceil(SLOTS_PER_WORD).max(1)
}

// A field of a columns container that belongs to the container rather than to
// the element, so scattering an element into its slot passes over it.
pub fn is_columns_bookkeeping(field: &str) -> bool {
    matches!(field, "generations" | "free_list" | "free_count")
        || field == LIVE_WORDS
        || field == LIVE_COUNT
}

struct FunctionSignature {
    parameters: Vec<Type>,
    return_type: Type,
}

// A generic struct declaration as its parameter names and written fields, and
// a generic enum's the same with one field list per variant that carries one.
type GenericStructDefs = HashMap<String, (Vec<String>, Vec<(String, Type)>)>;
type GenericEnumVariants = Vec<(String, Option<Vec<(String, Type)>>)>;
type GenericEnumDefs = HashMap<String, (Vec<String>, GenericEnumVariants)>;

// One function to lower: where its pieces live in the arena.
struct FunctionSource {
    parameters: Range32,
    return_sig: SignatureId,
    body: Range32,
}

pub struct IrBuilder {
    signatures: HashMap<String, FunctionSignature>,
    structs: HashMap<String, StructLayout>,
    enums: HashMap<String, EnumLayout>,
    // The bits each `flags` type names, by type name. A flags type is a
    // distinct type over an integer, so nothing but this table tells one apart
    // from any other distinct type, and only two questions need to: what
    // `InitFlags::Video` is, and which operators a bit set answers to.
    flags: HashMap<String, FlagsLayout>,
    // The values each type names under itself, by type name. A named value is
    // its expression wherever it is written, read at the type its declaration
    // gives it, which is what makes `Key::Left` a `Key` and not the number the
    // declaration wrote.
    type_values: HashMap<String, TypeValues>,
    constants: HashMap<String, ExprId>,
    // A function a `where` bound may name, by the one expression its body is.
    // A bound holds a type to what it is, and the vocabulary answers that of a
    // type directly; a program that wants to ask several things at once, or to
    // give the question a name, writes an ordinary function over the same
    // vocabulary and names it where a predicate would stand.
    bound_functions: HashMap<String, (String, ExprId)>,
    generic_functions: HashMap<String, GenericFunction>,
    generic_struct_defs: GenericStructDefs,
    linear: HashSet<String>,
    // Callback registrations, by name.
    registrations: HashMap<String, crate::lower::callbacks::CallbackShape>,
    // A number per type, handed out in the order `type_id` first asks for one.
    // What it is for is a table keyed by type in a program that decides at run
    // time what it holds: a component registry knows a type at the call that
    // registers it and an index only afterwards, and this is what ties the two
    // together. The numbers are this build's own and mean nothing outside it.
    type_ids: std::cell::RefCell<HashMap<String, i64>>,
    anon_counter: std::cell::Cell<usize>,
}

// A flags declaration, as the two things the rest of the compiler asks it for.
struct FlagsLayout {
    repr: Type,
    bits: HashMap<String, i64>,
}

// The values one type names under itself: the type each of them is, and the
// names in the order the declaration wrote them, so a near name suggested for
// one that is not there is chosen over a stable set.
struct TypeValues {
    declared: Type,
    values: Vec<(String, ExprId)>,
}

// An arm nothing reaches, whatever it was that took its values first.
const UNREACHABLE_CASE: &str =
    "this case is covered by an earlier one, so nothing reaches it";

// Whether `spans` between them hold every number from `low` to `high`.
//
// Walked rather than merged: stand at the lowest number wanted, jump to the far
// end of the widest span covering it, and repeat. A step that finds no span
// standing on its number is a number nobody covered, and every step that does
// find one moves strictly right, so this ends.
fn covers(spans: &[(i64, i64)], low: i64, high: i64) -> bool {
    let mut at = low;
    loop {
        let mut reach: Option<i64> = None;
        for (from, to) in spans {
            if *from <= at && at <= *to && reach.is_none_or(|held| *to > held) {
                reach = Some(*to);
            }
        }
        let Some(reach) = reach else { return false };
        if reach >= high {
            return true;
        }
        at = reach + 1;
    }
}

// What a match's arms compare against. An enum leaves the tag and the enum's
// name, anything else leaves the value and its type, and which of the two a
// pattern reaches for is what tells a pattern apart from the value it was
// written against.
struct Scrutinee<'a> {
    tag: Option<&'a IrOperand>,
    enum_name: Option<&'a String>,
    scalar: Option<&'a (IrOperand, Type)>,
}

struct AnonRequest {
    name: String,
    parameters: Range32,
    return_sig: SignatureId,
    body: Range32,
    // The module whose lowering produced this literal, carried for the same
    // reason `Specialization` carries it. A generic instantiated from inside an
    // anonymous function is work that module would have to do.
    requested_by: u32,
}

fn locate<T>(result: Result<T>, position: Position) -> Result<T> {
    result.map_err(|error| {
        let text =
            crate::modules::imports::demangle_private_names(&error.to_string());
        if position == Position::default() || text.starts_with("at ") {
            anyhow::anyhow!("{text}")
        } else {
            anyhow::anyhow!("at {}: {text}", position.describe())
        }
    })
}

pub fn build_module(
    ast: &mut Ast,
    roots: &[StmtId],
    linear: &HashSet<String>,
) -> Result<IrModule> {
    strict(build_module_inner(ast, roots, linear, false)?)
}

/// Lower every function, reporting one failure per function rather than
/// stopping at the first: unknown names are the most common fault while a
/// file is being edited, and one of them should not mask the rest. The outer
/// error is a whole-program fault, a generic struct that cannot expand or a
/// constant cycle, below which no function can be lowered at all. A failed
/// function contributes no IR and its pending specializations drop with it;
/// the module holds what lowered, and a backend only ever sees it when the
/// diagnostics list is empty.
///
/// `per_module` emits a specialization once per module that instantiates it
/// rather than once per program. Only correct when the result is about to be
/// split into one object per module, since two definitions of a name in a
/// single object is a duplicate symbol. Split, they are module-local and a
/// module's object is self-contained.
pub fn build_module_recovering(
    ast: &mut Ast,
    roots: &[StmtId],
    linear: &HashSet<String>,
    per_module: bool,
) -> Result<(IrModule, Vec<crate::diagnostic::Diagnostic>)> {
    build_module_inner(ast, roots, linear, per_module)
}

fn strict(
    lowered: (IrModule, Vec<crate::diagnostic::Diagnostic>),
) -> Result<IrModule> {
    let (module, diagnostics) = lowered;
    if diagnostics.is_empty() {
        return Ok(module);
    }
    let reports: Vec<String> = diagnostics
        .iter()
        .map(|held| held.message.clone())
        .collect();
    Err(anyhow::anyhow!(reports.join("\n")))
}

// A collected report: the position anchors the failing function for a caller
// that wants structure, and the message is the fully rendered text, located
// the way `locate` renders it, so the strict join reads exactly as the old
// first-failure error did.
fn report(
    diagnostics: &mut Vec<crate::diagnostic::Diagnostic>,
    position: Position,
    error: &anyhow::Error,
) {
    let text =
        crate::modules::imports::demangle_private_names(&error.to_string());
    let message = if position == Position::default() || text.starts_with("at ")
    {
        text
    } else {
        format!("at {}: {text}", position.describe())
    };
    diagnostics.push(crate::diagnostic::Diagnostic::new(position, message));
}

// What a type measures, for a constant asking before the program is emitted.
// The declaration may spell an enum as a struct, which is the name the layout
// pass files it under, so both tables are asked.
fn measured(
    ty: &Type,
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
    named: &str,
) -> Option<Literal> {
    let held = match ty {
        Type::Struct(name)
            if !structs.contains_key(name) && enums.contains_key(name) =>
        {
            Type::Enum(name.clone())
        }
        other => other.clone(),
    };
    match named {
        "sizeof" => size_and_align(&held, structs, enums)
            .map(|(size, _)| Literal::Integer(size as i64)),
        "alignof" => size_and_align(&held, structs, enums)
            .map(|(_, align)| Literal::Integer(align as i64)),
        "field_count" => match &held {
            Type::Struct(name) => structs
                .get(name)
                .map(|layout| Literal::Integer(layout.fields.len() as i64)),
            _ => None,
        },
        "typename" => Some(Literal::String(
            crate::modules::imports::demangle_private_names(&held.to_string()),
        )),
        _ => None,
    }
}

// The same expression with every measurement a type answers standing in place
// of the call that asked for it, or nothing where none was asked. Only the
// shapes a constant's value is built from are walked; one this misses is left
// as it was, which is the constant staying refused rather than answering wrong.
fn with_layout_answers(
    ast: &mut Ast,
    expression: ExprId,
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
) -> Option<ExprId> {
    let span = ast.expr_span(expression);
    match ast.expr(expression).clone() {
        Expression::Call(callee, arguments) => {
            let name = match ast.expr(callee) {
                Expression::Identifier(named) => ast.name(*named).to_string(),
                _ => String::new(),
            };
            let written: Vec<ExprId> = ast.exprs_in(arguments).to_vec();
            if let [only] = written.as_slice()
                && let Expression::TypeValue(ty) = ast.expr(*only).clone()
                && let Some(literal) = measured(&ty, structs, enums, &name)
            {
                return Some(ast.push_expr(Expression::Literal(literal), span));
            }
            let mut answered = false;
            let mut held = Vec::with_capacity(written.len());
            for argument in written {
                match with_layout_answers(ast, argument, structs, enums) {
                    Some(settled) => {
                        answered = true;
                        held.push(settled);
                    }
                    None => held.push(argument),
                }
            }
            if !answered {
                return None;
            }
            let arguments = ast.add_expr_list(&held);
            Some(ast.push_expr(Expression::Call(callee, arguments), span))
        }
        Expression::Infix(left, operator, right) => {
            let settled_left = with_layout_answers(ast, left, structs, enums);
            let settled_right = with_layout_answers(ast, right, structs, enums);
            if settled_left.is_none() && settled_right.is_none() {
                return None;
            }
            let left = settled_left.unwrap_or(left);
            let right = settled_right.unwrap_or(right);
            Some(ast.push_expr(Expression::Infix(left, operator, right), span))
        }
        Expression::Prefix(operator, inner) => {
            let settled = with_layout_answers(ast, inner, structs, enums)?;
            Some(ast.push_expr(Expression::Prefix(operator, settled), span))
        }
        _ => None,
    }
}

// Every constant whose value asks a type what it measures, worked out here. A
// layout is what the types answer once they have been read, and a constant is
// settled before they are, so the two compile-time answer sites cannot see each
// other and this is where they meet: the measurements stand in place of the
// calls that asked for them, the value is worked out over the tree the program
// is already built as, and what it answers stands in the program the way every
// other constant's answer does.
//
// A length is read while the types are read, so a length may not ask. That is
// the one rule the two positions differ by, and it is about when each is read
// rather than about what either means.
fn settle_layout_constants(
    ast: &mut Ast,
    roots: &[StmtId],
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
) -> Result<()> {
    let asking: Vec<(StmtId, Symbol, ExprId)> = roots
        .iter()
        .filter_map(|statement| match ast.stmt(*statement) {
            Statement::Constant(name, value)
                if !matches!(
                    ast.expr(*value),
                    Expression::Function(..) | Expression::Proc(..)
                ) =>
            {
                Some((*statement, *name, *value))
            }
            _ => None,
        })
        .collect();
    // What the constants already settled stand for, so one asking a layout may
    // be written in terms of them. They are literals by now, which is what the
    // writeback before the parse left behind.
    let mut known: HashMap<String, crate::const_eval::Value> = HashMap::new();
    for (_, name, value) in &asking {
        let held = match ast.expr(*value) {
            Expression::Literal(Literal::Integer(held)) => {
                crate::const_eval::Value::Integer(*held)
            }
            Expression::Literal(Literal::Boolean(held)) => {
                crate::const_eval::Value::Boolean(*held)
            }
            Expression::Literal(Literal::String(held)) => {
                crate::const_eval::Value::Text(std::rc::Rc::new(held.clone()))
            }
            _ => continue,
        };
        known.insert(ast.name(*name).to_string(), held);
    }
    for (statement, name, value) in asking {
        let Some(settled) = with_layout_answers(ast, value, structs, enums)
        else {
            continue;
        };
        let mut folder = crate::const_eval::Folder::over_tree(ast, roots);
        let answered = match folder.expression(ast, settled, &known) {
            Ok(answered) => answered,
            Err(reason) => {
                let position = ast.position_of(ast.expr_span(value));
                return locate(Err(anyhow::anyhow!("{reason}")), position);
            }
        };
        let span = ast.expr_span(value);
        let written = write_value_back(ast, &answered, span);
        ast.statements[statement.0 as usize] =
            Statement::Constant(name, written);
    }
    Ok(())
}

// What a worked-out value is written as where the constant naming it stands.
// The same shape the parse writes back, over the tree the program is built as
// rather than over the one declaration being read.
fn write_value_back(
    ast: &mut Ast,
    value: &crate::const_eval::Value,
    span: TokenSpan,
) -> ExprId {
    let expression = match value {
        crate::const_eval::Value::Integer(held) => {
            Expression::Literal(Literal::Integer(*held))
        }
        crate::const_eval::Value::Boolean(held) => {
            Expression::Literal(Literal::Boolean(*held))
        }
        crate::const_eval::Value::Text(held) => {
            Expression::Literal(Literal::String(held.to_string()))
        }
        crate::const_eval::Value::Array(items) => {
            let elements: Vec<ExprId> = items
                .iter()
                .map(|item| write_value_back(ast, item, span))
                .collect();
            Expression::Literal(Literal::Array(ast.add_expr_list(&elements)))
        }
        crate::const_eval::Value::Record(name, fields) => {
            let initializers: Vec<crate::ast::NamedExpr> = fields
                .iter()
                .map(|(field, held)| {
                    let value = write_value_back(ast, held, span);
                    crate::ast::NamedExpr {
                        name: ast.intern(field),
                        value,
                    }
                })
                .collect();
            let name = ast.intern(name);
            Expression::StructInit(name, ast.add_named_exprs(&initializers))
        }
    };
    ast.push_expr(expression, span)
}

// Every function a `where` bound may name: one compile-time type parameter, and
// a body that is one expression. The expression is the bound, written once under
// a name instead of at every declaration that asks it, and it is read by the
// same reader, so what a bound may say is the same either way.
//
// A body of more than one expression is not one of these. What a bound does is
// answer a question about a type, and everything the answer can be built from
// is an expression; a body with statements in it would be asking for a language
// the bound reader does not have.
fn collect_bound_functions(
    ast: &Ast,
    roots: &[StmtId],
) -> HashMap<String, (String, ExprId)> {
    let mut found = HashMap::new();
    for statement in roots {
        let Statement::Constant(name, value) = ast.stmt(*statement) else {
            continue;
        };
        let (Expression::Function(parameters, _, body)
        | Expression::Proc(parameters, _, body)) = ast.expr(*value)
        else {
            continue;
        };
        let written = ast.params_in(*parameters);
        let [only] = written else {
            continue;
        };
        if !is_type_parameter(ast, only) {
            continue;
        }
        let statements = ast.stmts_in(*body);
        let [held] = statements else {
            continue;
        };
        let (Statement::Expression(answer) | Statement::Return(answer)) =
            ast.stmt(*held)
        else {
            continue;
        };
        found.insert(
            ast.name(*name).to_string(),
            (ast.name(only.name).to_string(), *answer),
        );
    }
    found
}

fn build_module_inner(
    ast: &mut Ast,
    roots: &[StmtId],
    linear: &HashSet<String>,
    per_module: bool,
) -> Result<(IrModule, Vec<crate::diagnostic::Diagnostic>)> {
    let synthetic_structs = expand_generic_structs(ast, roots)?;
    let mut layout_roots: Vec<StmtId> = roots.to_vec();
    layout_roots.extend(synthetic_structs.iter().copied());
    let (structs, enums) = compute_layouts(ast, &layout_roots);
    settle_layout_constants(ast, roots, &structs, &enums)?;
    let mut constants = HashMap::new();
    for statement in roots {
        if let Statement::Constant(name, value) = ast.stmt(*statement)
            && !matches!(
                ast.expr(*value),
                Expression::Function(..) | Expression::Proc(..)
            )
        {
            constants.insert(ast.name(*name).to_string(), *value);
        }
    }
    check_constant_cycles(ast, &constants)?;
    let bound_functions = collect_bound_functions(ast, roots);
    let mut generic_functions = HashMap::new();
    for statement in roots {
        if let Statement::Constant(name, value) = ast.stmt(*statement)
            && let Expression::Function(parameters, return_sig, body)
            | Expression::Proc(parameters, return_sig, body) =
                ast.expr(*value)
            && function_is_generic(ast, *parameters)
        {
            let type_params = function_type_params(ast, *parameters);
            generic_functions.insert(
                ast.name(*name).to_string(),
                GenericFunction {
                    type_params,
                    parameters: *parameters,
                    return_sig: *return_sig,
                    body: *body,
                },
            );
        }
    }

    let mut generic_struct_defs = HashMap::new();
    for statement in roots {
        if let Statement::Struct(name, type_params, fields) =
            ast.stmt(*statement)
            && !type_params.is_empty()
        {
            let params: Vec<String> = ast
                .symbols_in(*type_params)
                .iter()
                .map(|param| ast.name(*param).to_string())
                .collect();
            let fields: Vec<(String, Type)> = ast
                .fields_in(*fields)
                .iter()
                .map(|field| {
                    (ast.name(field.name).to_string(), field.field_type.clone())
                })
                .collect();
            generic_struct_defs
                .insert(ast.name(*name).to_string(), (params, fields));
        }
    }

    let mut flags: HashMap<String, FlagsLayout> = HashMap::new();
    for statement in roots {
        if let Statement::Flags(name, repr, bits) = ast.stmt(*statement) {
            flags.insert(
                ast.name(*name).to_string(),
                FlagsLayout {
                    repr: repr.clone(),
                    bits: ast
                        .flag_bits_in(*bits)
                        .iter()
                        .map(|bit| (ast.name(bit.name).to_string(), bit.value))
                        .collect(),
                },
            );
        }
    }

    // The type each name declares, for the declarations that may name values
    // under themselves. A distinct type carries the code it was resolved to,
    // so `Key::Left` reads at the `Key` a use of the name reads at.
    let mut declared_types: HashMap<String, Type> = HashMap::new();
    for statement in roots {
        match ast.stmt(*statement) {
            Statement::TypeAlias(name, ty) => {
                declared_types.insert(ast.name(*name).to_string(), ty.clone());
            }
            Statement::Struct(name, ..) => {
                let name = ast.name(*name).to_string();
                declared_types.insert(name.clone(), Type::Struct(name));
            }
            Statement::Enum(name, ..) => {
                let name = ast.name(*name).to_string();
                declared_types.insert(name.clone(), Type::Enum(name));
            }
            _ => {}
        }
    }
    let mut type_values: HashMap<String, TypeValues> = HashMap::new();
    for entry in &ast.type_values {
        let type_name = ast.name(entry.type_name).to_string();
        let Some(declared) = declared_types.get(&type_name) else {
            continue;
        };
        type_values
            .entry(type_name)
            .or_insert_with(|| TypeValues {
                declared: declared.clone(),
                values: Vec::new(),
            })
            .values
            .push((ast.name(entry.name).to_string(), entry.value));
    }

    // The concrete types beside the written ones, so a place reached through an
    // instantiation has a type. `Vec<File>` is where `storage` is a run of
    // resources; `Vec` alone says only that it is a run of whatever `T` stands
    // for, and nothing can be told about a resource from that. A call that
    // answers with an instantiation makes one without anyone writing its name,
    // which is why this is read from what specialization forms rather than from
    // what the source spells out.
    let mut with_instances: Vec<StmtId> = roots.to_vec();
    with_instances.extend(layout_roots.iter().skip(roots.len()).copied());

    let mut builder = IrBuilder {
        signatures: HashMap::new(),
        structs,
        enums,
        flags,
        type_values,
        constants,
        bound_functions,
        generic_functions,
        generic_struct_defs,
        linear: linear_with_holders(linear, ast, &with_instances),
        registrations: crate::lower::callbacks::callback_registrations(
            ast, roots,
        ),
        type_ids: std::cell::RefCell::new(HashMap::new()),
        anon_counter: std::cell::Cell::new(0),
    };
    builder.collect_signatures(ast, roots);

    let ownership =
        crate::check::ownership::specializations(ast, &with_instances, linear);

    let mut functions = Vec::new();
    let mut externs = Vec::new();
    let mut declared = Vec::new();
    let mut top_level = Vec::new();
    let mut has_main = false;
    let mut pending: Vec<Specialization> = Vec::new();
    let mut pending_anon: Vec<AnonRequest> = Vec::new();
    let mut diagnostics: Vec<crate::diagnostic::Diagnostic> = Vec::new();

    for statement in roots {
        let position = ast.stmt_position(*statement);
        match ast.stmt(*statement).clone() {
            Statement::Constant(name, value)
                if matches!(
                    ast.expr(value),
                    Expression::Function(..) | Expression::Proc(..)
                ) =>
            {
                let (Expression::Function(parameters, return_sig, body)
                | Expression::Proc(parameters, return_sig, body)) =
                    ast.expr(value).clone()
                else {
                    unreachable!()
                };
                if function_is_generic(ast, parameters) {
                    continue;
                }
                let name = ast.name(name).to_string();
                if name == "main" {
                    has_main = true;
                }
                // Expansion time runs over every body, not only a
                // specialization's: a walk over a type's fields is decided by
                // a declaration rather than by a call, so an ordinary function
                // may write one.
                let body = match expand_compile_time(
                    ast,
                    body,
                    None,
                    parameters,
                    ExpansionContext {
                        structs: &builder.structs,
                        subst: &HashMap::new(),
                        linear,
                    },
                ) {
                    Ok(body) => body,
                    Err(error) => {
                        report(&mut diagnostics, position, &error);
                        continue;
                    }
                };
                // The ownership rules again, over the types specialization
                // forms rather than only the ones the source writes down. A
                // call that answers with an instantiation makes one without
                // anyone naming it, so `held := option_some($File, ...)` left
                // `Option<File>` ordinary data and the obligation on the
                // resource inside it went in and did not come out.
                if let Some(first) =
                    ownership.check(ast, parameters, body).first()
                {
                    let message =
                        crate::modules::imports::demangle_private_names(
                            &format!("at {}: {first}", position.describe()),
                        );
                    diagnostics.push(crate::diagnostic::Diagnostic {
                        position,
                        message,
                        related: Vec::new(),
                    });
                    continue;
                }
                let (function, requests, anon) = match builder.lower_function(
                    ast,
                    &name,
                    FunctionSource {
                        parameters,
                        return_sig,
                        body,
                    },
                ) {
                    Ok(lowered) => lowered,
                    Err(error) => {
                        report(&mut diagnostics, position, &error);
                        continue;
                    }
                };
                functions.push(in_module(function, position.file));
                pending.extend(requested_by(requests, position.file));
                pending_anon.extend(anon_requested_by(anon, position.file));
            }
            Statement::Extern {
                name,
                params,
                return_type,
                ..
            } => {
                let name = ast.name(name).to_string();
                let return_type = return_type.unwrap_or(Type::Void);
                let return_layout = builder.c_layout(&return_type);
                let param_layouts = match ast
                    .params_in(params)
                    .iter()
                    .map(|parameter| {
                        if parameter.mode != crate::parser::ParamMode::Value {
                            return Ok(None);
                        }
                        let Some(ty) = &parameter.type_annotation else {
                            bail!(
                                "the parameter '{}' of the extern '{name}' is written 'value' but has no type",
                                ast.name(parameter.name)
                            );
                        };
                        let Some(layout) = builder.c_layout(ty) else {
                            bail!(
                                "'{}' of the extern '{name}' is written 'value', but '{ty}' is not an aggregate; a scalar already goes to C by value and needs no mode",
                                ast.name(parameter.name)
                            );
                        };
                        Ok(Some(layout))
                    })
                    .collect::<Result<Vec<_>>>()
                {
                    Ok(layouts) => layouts,
                    Err(error) => {
                        report(&mut diagnostics, position, &error);
                        continue;
                    }
                };
                externs.push(IrExtern {
                    name,
                    params: extern_parameter_types(ast, params),
                    param_layouts,
                    return_type,
                    return_layout,
                });
            }
            // A function some other object defines. It contributes a
            // declaration so calls can be typed and emitted, and no body. The
            // module it came from is not being rebuilt.
            Statement::Declared {
                name,
                params,
                return_sig,
            } => {
                let name = ast.name(name).to_string();
                declared
                    .push(declared_function(ast, &name, params, return_sig));
            }
            Statement::Struct(..)
            | Statement::Enum(..)
            | Statement::Flags(..)
            | Statement::TypeAlias(..)
            | Statement::Import(..) => {}
            _ => top_level.push(*statement),
        }
    }

    if !has_main && !top_level.is_empty() {
        let mut body = top_level.clone();
        let ends_in_expression = matches!(
            body.last().map(|statement| ast.stmt(*statement)),
            Some(Statement::Expression(_))
        );
        if !ends_in_expression && let Some(last) = body.last().copied() {
            let span = ast.stmt_span(last);
            let zero =
                ast.push_expr(Expression::Literal(Literal::Integer(0)), span);
            body.push(ast.push_stmt(Statement::Expression(zero), span));
        }
        let body = ast.add_stmt_list(&body);
        let return_sig = ast.push_signature(ReturnSignature::plain(
            ReturnKind::Single(Type::I64),
        ));
        match builder.lower_function(
            ast,
            "main",
            FunctionSource {
                parameters: Range32::EMPTY,
                return_sig,
                body,
            },
        ) {
            Ok((function, requests, anon)) => {
                functions.push(in_module(function, 0));
                // Synthesized `main` from loose top-level statements, which
                // belong to the entry file.
                pending.extend(requested_by(requests, 0));
                pending_anon.extend(anon_requested_by(anon, 0));
            }
            Err(error) => {
                report(&mut diagnostics, Position::default(), &error);
            }
        }
    }

    // Which modules instantiate each specialization. Recorded for every
    // request, including the ones the global dedup then discards, because the
    // question it answers is what *separate* compilation would cost and that is
    // exactly the requests a single-object build throws away.
    let mut instantiated_by: HashMap<String, std::collections::HashSet<u32>> =
        HashMap::new();
    // Keyed by name alone for one object, and by module and name when each
    // module gets its own.
    let mut emitted: std::collections::HashSet<(u32, String)> =
        std::collections::HashSet::new();
    loop {
        if let Some(specialization) = pending.pop() {
            instantiated_by
                .entry(specialization.mangled_name.clone())
                .or_default()
                .insert(specialization.requested_by);
            let key = if per_module {
                specialization.requested_by
            } else {
                0
            };
            // The output is one object, so a specialization is emitted once no
            // matter how many modules ask for it. Per-module copies become
            // possible only when each module emits its own object.
            if !emitted.insert((key, specialization.mangled_name.clone())) {
                continue;
            }
            let generic = builder
                .generic_functions
                .get(&specialization.generic_name)
                .expect("specialization references a known generic function")
                .clone();
            let mut parameters: Vec<Parameter> = ast
                .params_in(generic.parameters)
                .iter()
                .filter(|parameter| {
                    !is_type_parameter(ast, parameter) && !parameter.pack
                })
                .map(|parameter| Parameter {
                    name: parameter.name,
                    type_annotation: parameter.type_annotation.as_ref().map(
                        |ty| {
                            specialized_param_type(
                                ty,
                                &specialization.subst,
                                parameter.mode,
                            )
                        },
                    ),
                    mutable: parameter.mutable,
                    mode: parameter.mode,
                    compile_time_signature: None,
                    compile_time_default: None,
                    pack: false,
                    // The word travels with the parameter. A body that hands
                    // its own literal on to another `format` parameter is
                    // handing on one a caller already wrote, and the
                    // specialization is where that body is read.
                    format: parameter.format,
                })
                .collect();
            // The bound was checked at the call that asked for this
            // specialization, so the specialized signature carries none.
            let generic_signature = ast.signature(generic.return_sig).clone();
            let return_sig = ReturnSignature {
                bound: None,
                bound_text: String::new(),
                kind: match ast.signature_to_type(&generic_signature) {
                    Some(ty) => ReturnKind::Single(substitute_type(
                        &ty,
                        &specialization.subst,
                    )),
                    None => ReturnKind::None,
                },
                uses: generic_signature
                    .uses
                    .iter()
                    .map(|ty| substitute_type(ty, &specialization.subst))
                    .collect(),
            };
            // A compile-time list becomes ordinary parameters here, one per
            // element the call gave it, each with the type that argument had.
            // That is what makes an element a value evaluated once and a name
            // the unrolled body can use.
            if let Some((_, elements)) = &specialization.pack {
                for element in elements {
                    let PackElement::Value(name, ty) = element else {
                        continue;
                    };
                    parameters.push(Parameter {
                        name: ast.intern(name),
                        type_annotation: Some(ty.clone()),
                        mutable: false,
                        mode: crate::parser::ParamMode::Read,
                        compile_time_signature: None,
                        compile_time_default: None,
                        pack: false,
                        format: false,
                    });
                }
            }
            let parameters = ast.add_parameters(parameters);
            let return_sig = ast.push_signature(return_sig);
            let body =
                substitute_block(ast, generic.body, &specialization.subst);
            // Expansion time: a `for` over the list unrolls, `list[K]` becomes
            // the Kth element, and an `if` over a type predicate keeps the one
            // branch that survives. All three are decided here, where the types
            // are known, and none of them exists afterwards.
            let body = match expand_compile_time(
                ast,
                body,
                specialization.pack.as_ref(),
                parameters,
                ExpansionContext {
                    structs: &builder.structs,
                    subst: &specialization.subst,
                    linear,
                },
            ) {
                Ok(body) => body,
                Err(error) => {
                    let error =
                        locate_instantiation_error(&error, &specialization);
                    report(
                        &mut diagnostics,
                        specialization.requested_at,
                        &error,
                    );
                    continue;
                }
            };
            // The ownership rules, asked of the body that really exists. The
            // template's own says nothing: its parameters are bound to nothing,
            // so no type there is a resource and a list has no elements to
            // unroll. This is the first and only point where both are true.
            let complaints = ownership.check(ast, parameters, body);
            if let Some(first) = complaints.first() {
                // The prefix an import gives a private name is nothing the
                // reader wrote, so it comes back off the way it does in every
                // other diagnostic.
                //
                // The call, and only the call. A complaint raised inside the
                // instance carries the template's place, and putting the call
                // in front of it left two in one report: the renderer reads the
                // first and the second stays in the words, so the claim a
                // reader saw began with a file name.
                // A complaint that carried a place from inside the
                // instance keeps it: that is the line to change. The call is
                // the place to show it at when the complaint has none.
                let placed = match first.starts_with("at ") {
                    true => first.clone(),
                    false => format!(
                        "at {}: {first}",
                        specialization.requested_at.describe()
                    ),
                };
                let message =
                    crate::modules::imports::demangle_private_names(&placed);
                diagnostics.push(crate::diagnostic::Diagnostic {
                    position: specialization.requested_at,
                    message,
                    related: Vec::new(),
                });
                continue;
            }
            let (function, requests, anon) = match locate_instantiation(
                builder.lower_function(
                    ast,
                    &specialization.mangled_name,
                    FunctionSource {
                        parameters,
                        return_sig,
                        body,
                    },
                ),
                &specialization,
            ) {
                Ok(lowered) => lowered,
                Err(error) => {
                    report(
                        &mut diagnostics,
                        specialization.requested_at,
                        &error,
                    );
                    continue;
                }
            };
            let function =
                local_to_module(function, specialization.requested_by);
            functions.push(IrFunction {
                instantiated: Some(crate::ir::Instantiation {
                    name: specialization.display.clone(),
                    at: specialization.requested_at,
                }),
                ..function
            });
            // A generic that instantiates another generic is still work the
            // asking module would have to do, so the attribution carries down,
            // and so does the call site. The inner call was written inside a
            // template, and the line the reader wrote is the outer one.
            pending.extend(requested_at(
                requested_by(requests, specialization.requested_by),
                specialization.requested_at,
            ));
            pending_anon
                .extend(anon_requested_by(anon, specialization.requested_by));
        } else if let Some(request) = pending_anon.pop() {
            let (function, requests, anon) = match builder.lower_function(
                ast,
                &request.name,
                FunctionSource {
                    parameters: request.parameters,
                    return_sig: request.return_sig,
                    body: request.body,
                },
            ) {
                Ok(lowered) => lowered,
                Err(error) => {
                    report(&mut diagnostics, Position::default(), &error);
                    continue;
                }
            };
            functions.push(local_to_module(function, request.requested_by));
            pending.extend(requested_by(requests, request.requested_by));
            pending_anon.extend(anon_requested_by(anon, request.requested_by));
        } else {
            break;
        }
    }

    report_module_specializations(&instantiated_by);
    Ok((
        IrModule {
            functions,
            externs,
            imported: declared,
        },
        diagnostics,
    ))
}

// The shape a declared function needs to have for a backend to emit a call to
// it: parameter types, a return type, and no blocks. It rides in `imported`,
// which already means "declared here, defined in another object", and which the
// backends already declare with the same signature builder that builds a
// definition.
fn declared_function(
    ast: &Ast,
    name: &str,
    params: Range32,
    return_sig: SignatureId,
) -> IrFunction {
    let locals: Vec<crate::ir::IrLocal> = ast
        .params_in(params)
        .iter()
        .map(|parameter| {
            let ty = parameter_type(parameter);
            crate::ir::IrLocal {
                size: ty.size_of(),
                ty,
                name: Some(ast.name(parameter.name).to_string()),
                in_memory: false,
                linear: false,
                position: Position::default(),
            }
        })
        .collect();
    IrFunction {
        name: name.to_string(),
        param_count: locals.len(),
        // A declaration from another object contributes no body, so nothing
        // here collects a parameter. Only the object that defines it does.
        param_layouts: vec![None; locals.len()],
        return_type: ast
            .signature_to_type(ast.signature(return_sig))
            .unwrap_or(Type::Void),
        locals,
        blocks: Vec::new(),
        entry: 0,
        instantiated: None,
        module: 0,
        local: false,
        keeps_name: false,
    }
}

// What an extern's parameters are once C sees them. For a registration these
// are not what the declaration says literally. The `$handler` parameter is the
// callback pointer, and the context is passed as an address, because the library
// keeps it past the call.
fn extern_parameter_types(ast: &Ast, params: Range32) -> Vec<Type> {
    let params = ast.params_in(params);
    let shape = crate::lower::callbacks::callback_shape(params);
    params
        .iter()
        .enumerate()
        .map(|(index, parameter)| match &shape {
            Some(shape) if index == shape.handler => parameter
                .compile_time_signature
                .clone()
                .unwrap_or(Type::Ptr(Box::new(Type::U8))),
            Some(shape) if index == shape.context => {
                Type::Ptr(Box::new(shape.context_type.clone()))
            }
            _ => parameter_type(parameter),
        })
        .collect()
}

fn in_module(function: IrFunction, module: u32) -> IrFunction {
    IrFunction { module, ..function }
}

// A specialization or an anonymous literal: private to the object that holds
// it, because the module that produced it is the only one that calls it.
fn local_to_module(function: IrFunction, module: u32) -> IrFunction {
    IrFunction {
        module,
        local: true,
        ..function
    }
}

// A specialization asked for from inside another one was written in a template,
// so it inherits the call site of the outer one, which is a line the reader
// wrote.
fn requested_at(
    requests: Vec<Specialization>,
    position: Position,
) -> Vec<Specialization> {
    requests
        .into_iter()
        .map(|request| Specialization {
            requested_at: position,
            ..request
        })
        .collect()
}

// A lowering error from inside a stamped-out body names a line in the template.
// The call that asked for the specialization goes first, because that is the
// line the reader wrote and the one they can change.
fn locate_instantiation<T>(
    result: Result<T>,
    specialization: &Specialization,
) -> Result<T> {
    result.map_err(|error| locate_instantiation_error(&error, specialization))
}

fn locate_instantiation_error(
    error: &anyhow::Error,
    specialization: &Specialization,
) -> anyhow::Error {
    let text =
        crate::modules::imports::demangle_private_names(&error.to_string());
    // A fault that carried a place from inside the instance keeps it: that is
    // the line to change, and the instance it went wrong for is named in the
    // claim. The call is the place to show it at when the fault has none.
    if text.starts_with("at ")
        || specialization.requested_at == Position::default()
    {
        anyhow::anyhow!("{text}")
    } else {
        anyhow::anyhow!("at {}: {text}", specialization.requested_at.describe())
    }
}

fn requested_by(
    requests: Vec<Specialization>,
    module: u32,
) -> Vec<Specialization> {
    requests
        .into_iter()
        .map(|request| Specialization {
            requested_by: module,
            ..request
        })
        .collect()
}

fn anon_requested_by(
    requests: Vec<AnonRequest>,
    module: u32,
) -> Vec<AnonRequest> {
    requests
        .into_iter()
        .map(|request| AnonRequest {
            requested_by: module,
            ..request
        })
        .collect()
}

// What separate compilation would cost in duplicated code, measured rather than
// guessed at. The design gives each module its own copy of every specialization
// it instantiates, because cranelift has no weak or COMDAT linkage to fold them
// with, and whether that matters is this number.
//
// Off unless `FROST_MODULE_REPORT` is set, and it prints to stderr so it never
// reaches emitted output.
fn report_module_specializations(
    instantiated_by: &HashMap<String, std::collections::HashSet<u32>>,
) {
    if !std::env::var("FROST_MODULE_REPORT").is_ok_and(|value| value != "0") {
        return;
    }
    let total = instantiated_by.len();
    let copies: usize =
        instantiated_by.values().map(|modules| modules.len()).sum();
    let shared = instantiated_by
        .values()
        .filter(|modules| modules.len() > 1)
        .count();

    let mut per_module: HashMap<u32, usize> = HashMap::new();
    for modules in instantiated_by.values() {
        for module in modules {
            *per_module.entry(*module).or_default() += 1;
        }
    }
    let mut rows: Vec<(String, usize)> = per_module
        .into_iter()
        .map(|(module, count)| {
            let name = crate::source_map::name_of(module)
                .unwrap_or_else(|| "(entry file)".to_string());
            (name, count)
        })
        .collect();
    rows.sort();

    eprintln!(
        "frost: {total} specialization(s) emitted, {copies} would be emitted per-module ({shared} instantiated by more than one module)"
    );
    for (name, count) in rows {
        eprintln!("frost:   {name} instantiates {count}");
    }
}

/// Refuses a constant that is defined in terms of itself.
///
/// A constant is its value wherever it is named, so substituting one that
/// reaches itself never finishes: the compiler recurses until the stack runs
/// out and says so from a thousand frames down, naming nothing.
///
/// Checking the whole table once, here, is what lets every site that follows a
/// constant do it plainly. There are several of them, they are reached from
/// different directions, and a guard at each would be four chances to forget
/// one; a table with no cycle in it cannot be walked forever from anywhere.
fn check_constant_cycles(
    ast: &Ast,
    constants: &HashMap<String, ExprId>,
) -> Result<()> {
    let mut settled: HashSet<String> = HashSet::new();
    let mut path: Vec<String> = Vec::new();
    let mut names: Vec<&String> = constants.keys().collect();
    names.sort();
    for name in names {
        walk_constant(ast, name, constants, &mut settled, &mut path)?;
    }
    Ok(())
}

fn walk_constant(
    ast: &Ast,
    name: &str,
    constants: &HashMap<String, ExprId>,
    settled: &mut HashSet<String>,
    path: &mut Vec<String>,
) -> Result<()> {
    if settled.contains(name) {
        return Ok(());
    }
    if let Some(at) = path.iter().position(|held| held == name) {
        let mut cycle: Vec<String> = path[at..].to_vec();
        cycle.push(name.to_string());
        bail!(
            "the constant '{name}' is defined in terms of itself: {}",
            cycle.join(" names ")
        );
    }
    path.push(name.to_string());
    let mut referenced = Vec::new();
    if let Some(value) = constants.get(name) {
        crate::modules::interface_names::names_in_expression(
            ast,
            *value,
            &mut referenced,
        );
    }
    for reference in referenced {
        if constants.contains_key(&reference) {
            walk_constant(ast, &reference, constants, settled, path)?;
        }
    }
    path.pop();
    settled.insert(name.to_string());
    Ok(())
}

impl IrBuilder {
    fn collect_signatures(&mut self, ast: &Ast, roots: &[StmtId]) {
        for statement in roots {
            match ast.stmt(*statement) {
                Statement::Constant(name, value)
                    if matches!(
                        ast.expr(*value),
                        Expression::Function(..) | Expression::Proc(..)
                    ) =>
                {
                    let (Expression::Function(parameters, return_sig, _)
                    | Expression::Proc(parameters, return_sig, _)) =
                        ast.expr(*value)
                    else {
                        unreachable!()
                    };
                    if function_is_generic(ast, *parameters) {
                        continue;
                    }
                    self.signatures.insert(
                        ast.name(*name).to_string(),
                        FunctionSignature {
                            parameters: ast
                                .params_in(*parameters)
                                .iter()
                                .map(parameter_type)
                                .collect(),
                            return_type: ast
                                .signature_to_type(ast.signature(*return_sig))
                                .unwrap_or(Type::Void),
                        },
                    );
                }
                Statement::Extern {
                    name,
                    params,
                    return_type,
                    ..
                } => {
                    self.signatures.insert(
                        ast.name(*name).to_string(),
                        FunctionSignature {
                            parameters: extern_parameter_types(ast, *params),
                            return_type: return_type
                                .clone()
                                .unwrap_or(Type::Void),
                        },
                    );
                }
                Statement::Declared {
                    name,
                    params,
                    return_sig,
                } => {
                    self.signatures.insert(
                        ast.name(*name).to_string(),
                        FunctionSignature {
                            parameters: ast
                                .params_in(*params)
                                .iter()
                                .map(parameter_type)
                                .collect(),
                            return_type: ast
                                .signature_to_type(ast.signature(*return_sig))
                                .unwrap_or(Type::Void),
                        },
                    );
                }
                _ => {}
            }
        }
    }

    fn lower_function(
        &self,
        ast: &mut Ast,
        name: &str,
        source: FunctionSource,
    ) -> Result<(IrFunction, Vec<Specialization>, Vec<AnonRequest>)> {
        let FunctionSource {
            parameters,
            return_sig,
            body,
        } = source;
        let return_type = ast
            .signature_to_type(ast.signature(return_sig))
            .unwrap_or(Type::Void);
        let declared_parameters: Vec<Parameter> =
            ast.params_in(parameters).to_vec();
        let mut function =
            FunctionLowering::new(self, ast, return_type.clone());

        // A parameter is bound before any statement runs, so it would carry no
        // position and a type error about one would name a function and nothing
        // else. The body's first statement is where a reader looks.
        if let Some(first) = function.ast.stmts_in(body).first().copied() {
            function.current_position = function.ast.stmt_position(first);
        }
        for parameter in &declared_parameters {
            let ty = parameter_type(parameter);
            let parameter_name = function.ast.name(parameter.name).to_string();
            if parameter.format {
                function.forwarded_format = Some(parameter_name.clone());
            }
            function.parameter_names.push(parameter_name.clone());
            let local = function.fresh_local(ty, Some(parameter_name.clone()));
            function.define_variable(&parameter_name, local);
        }

        let has_defers = function.ast.stmts_in(body).iter().any(|s| {
            matches!(
                function.ast.stmt(*s),
                Statement::Defer(_) | Statement::ErrDefer(_)
            )
        });
        if has_defers {
            function.lower_body_with_defers(body, &return_type)?;
        } else {
            let (value, value_type) =
                function.lower_block(body, Some(&return_type))?;
            if !function.current_is_terminated() {
                if matches!(return_type, Type::Void) {
                    function.set_terminator(IrTerminator::Return(None));
                } else {
                    function.check_answer(&value_type, &return_type)?;
                    let operand =
                        function.coerce(value, &value_type, &return_type)?;
                    function
                        .set_terminator(IrTerminator::Return(Some(operand)));
                }
            }
        }

        let specializations = std::mem::take(&mut function.specializations);
        let anonymous = std::mem::take(&mut function.anonymous);
        let (locals, blocks) = function.finish();
        // Which parameters C hands over as the struct itself. The declaration
        // says `value`, and the type is the one it was written with, since the
        // mode lowering has already turned it into a borrow by here.
        let param_layouts = declared_parameters
            .iter()
            .map(|parameter| {
                if parameter.mode != crate::parser::ParamMode::Value {
                    return None;
                }
                let ty = parameter.type_annotation.as_ref()?;
                let ty = match ty {
                    Type::Ref(inner) | Type::RefMut(inner) => inner.as_ref(),
                    other => other,
                };
                self.c_layout(ty)
            })
            .collect();
        Ok((
            IrFunction {
                name: name.to_string(),
                param_count: declared_parameters.len(),
                param_layouts,
                return_type,
                locals,
                blocks,
                entry: 0,
                instantiated: None,
                // Stamped by the caller, which is the only place that knows
                // whether this is a module's own function or a specialization
                // some other module asked for.
                module: 0,
                local: false,
                keeps_name: ast.is_exported_symbol(name),
            },
            specializations,
            anonymous,
        ))
    }

    fn signature(&self, name: &str) -> Option<&FunctionSignature> {
        self.signatures.get(name)
    }

    // An aggregate flattened to what a C ABI asks about: its size, and every
    // scalar leaf with its offset. `None` for anything C returns as a scalar,
    // which needs no classification. See src/c_abi.rs.
    fn c_layout(&self, ty: &Type) -> Option<crate::c_abi::CLayout> {
        if !matches!(
            ty,
            Type::Struct(_)
                | Type::Enum(_)
                | Type::Array(_, _)
                | Type::Str
                | Type::Slice(_)
        ) {
            return None;
        }
        let mut scalars = Vec::new();
        self.flatten_scalars(ty, 0, &mut scalars)?;
        let (size, align) = self.size_and_align_of(ty)?;
        Some(crate::c_abi::CLayout {
            name: ty.to_string(),
            size,
            align,
            scalars,
        })
    }

    fn size_and_align_of(&self, ty: &Type) -> Option<(usize, usize)> {
        match ty {
            Type::Struct(name) => match self.structs.get(name) {
                Some(layout) => Some((layout.size, layout.align)),
                None => self
                    .enums
                    .get(name)
                    .map(|layout| (layout.size, layout.align)),
            },
            Type::Enum(name) => self
                .enums
                .get(name)
                .map(|layout| (layout.size, layout.align)),
            Type::Array(inner, count) => {
                let (size, align) = self.size_and_align_of(inner)?;
                Some((size * count, align))
            }
            Type::Str | Type::Slice(_) => Some((16, 8)),
            Type::Distinct(_, inner) => self.size_and_align_of(inner),
            other => Some((other.size_of(), other.align_of())),
        }
    }

    // Every variant of an enum is flattened, not just one, because an enum is
    // the only union-like shape here and a union classifies as the combination
    // of everything that could occupy each byte.
    fn flatten_scalars(
        &self,
        ty: &Type,
        offset: usize,
        out: &mut Vec<crate::c_abi::CScalar>,
    ) -> Option<()> {
        match ty {
            Type::Struct(name) => {
                if let Some(layout) = self.structs.get(name) {
                    for field in &layout.fields {
                        self.flatten_scalars(
                            &field.ty,
                            offset + field.offset,
                            out,
                        )?;
                    }
                    return Some(());
                }
                self.flatten_enum(name, offset, out)
            }
            Type::Enum(name) => self.flatten_enum(name, offset, out),
            Type::Array(inner, count) => {
                let (size, _) = self.size_and_align_of(inner)?;
                for index in 0..*count {
                    self.flatten_scalars(inner, offset + index * size, out)?;
                }
                Some(())
            }
            Type::Str | Type::Slice(_) => {
                out.push(crate::c_abi::CScalar {
                    offset,
                    ty: Type::Ptr(Box::new(Type::U8)),
                });
                out.push(crate::c_abi::CScalar {
                    offset: offset + 8,
                    ty: Type::I64,
                });
                Some(())
            }
            Type::Distinct(_, inner) => {
                self.flatten_scalars(inner, offset, out)
            }
            Type::Void | Type::Unknown => None,
            other => {
                out.push(crate::c_abi::CScalar {
                    offset,
                    ty: other.clone(),
                });
                Some(())
            }
        }
    }

    fn flatten_enum(
        &self,
        name: &str,
        offset: usize,
        out: &mut Vec<crate::c_abi::CScalar>,
    ) -> Option<()> {
        let layout = self.enums.get(name)?;
        out.push(crate::c_abi::CScalar {
            offset,
            ty: Type::U32,
        });
        for variant in &layout.variants {
            for field in &variant.fields {
                self.flatten_scalars(&field.ty, offset + field.offset, out)?;
            }
        }
        Some(())
    }

    fn struct_layout(&self, name: &str) -> Option<&StructLayout> {
        self.structs.get(name)
    }

    // The number this type goes by, made the first time it is asked for.
    fn type_id(&self, ty: &Type) -> i64 {
        let written = ty.to_string();
        let mut held = self.type_ids.borrow_mut();
        let next = held.len() as i64;
        *held.entry(written).or_insert(next)
    }

    fn enum_layout(&self, name: &str) -> Option<&EnumLayout> {
        self.enums.get(name)
    }

    // Whether a type names this value under itself. Asked at `Type::Name`
    // before the reading that makes one a variant, so a type carrying both
    // answers for each name with the thing it declared under it.
    fn names_a_value(&self, type_name: &str, value_name: &str) -> bool {
        self.type_values.get(type_name).is_some_and(|held| {
            held.values.iter().any(|(name, _)| name == value_name)
        })
    }

    fn byte_size(&self, ty: &Type) -> usize {
        size_and_align(ty, &self.structs, &self.enums)
            .map(|(size, _)| size)
            .unwrap_or(0)
    }

    /// The width of a type, or nothing where none was worked out. `byte_size`
    /// answers zero there, which a program cannot tell from a real zero, so
    /// anything reporting a width to a reader asks this instead.
    fn measured_size(&self, ty: &Type) -> Option<usize> {
        size_and_align(ty, &self.structs, &self.enums).map(|(size, _)| size)
    }

    fn measured_align(&self, ty: &Type) -> Option<usize> {
        size_and_align(ty, &self.structs, &self.enums).map(|(_, align)| align)
    }

    fn type_is_linear(&self, ty: &Type) -> bool {
        ty.is_linear_with(&self.linear)
    }
}

/// Every type that has to be consumed: the ones declared `linear`, and the ones
/// holding such a value, since a struct holding a resource is a resource and an
/// enum with one in a variant's payload is too. Run once per module, and only
/// when something is declared linear at all, so a program with no resources pays
/// nothing for it.
///
/// The statements are the ones specialization forms as well as the ones the
/// source writes, since a call that answers with an instantiation makes one
/// without anyone naming it.
pub fn linear_with_holders(
    declared: &HashSet<String>,
    ast: &Ast,
    statements: &[StmtId],
) -> HashSet<String> {
    let mut held = declared.clone();
    if held.is_empty() {
        return held;
    }
    // The instantiations, which the declarations alone cannot answer for: a
    // generic's field is a parameter bound to nothing here, so `Slab` holds no
    // resource while `Slab<Node, 2>` does.
    let instances =
        crate::check::linear_instances::collect_instances(ast, statements);
    let templates =
        crate::check::linear_instances::declared_structs(ast, statements);
    loop {
        let mut grew = false;
        for statement in statements {
            // A variant's payload is held by the enum exactly as a field is held
            // by a struct, so an enum carrying a resource is one. Reading only
            // the structs left an option holding a file ordinary data, and the
            // obligation went in and did not come out.
            let (name, field_types): (&str, Vec<&Type>) =
                match ast.stmt(*statement) {
                    Statement::Struct(name, _, fields) => (
                        ast.name(*name),
                        ast.fields_in(*fields)
                            .iter()
                            .map(|field| &field.field_type)
                            .collect(),
                    ),
                    Statement::Enum(name, _, variants) => (
                        ast.name(*name),
                        ast.variants_in(*variants)
                            .iter()
                            .filter_map(|variant| variant.fields)
                            .flat_map(|fields| ast.fields_in(fields))
                            .map(|field| &field.field_type)
                            .collect(),
                    ),
                    _ => continue,
                };
            // The name itself as well as the template it came from. These
            // statements include the instantiations specialization forms, whose
            // names carry their arguments, so a guard reading only the template
            // asks about a name that was never going to be inserted and reports
            // growth on every round for ever.
            if held.contains(name) || held.contains(Type::template_of(name)) {
                continue;
            }
            // The struct a return type list stands for is left out. It is the
            // one aggregate a program cannot hold: the lowering builds it at a
            // `return`, takes it apart at the binding that reads it, and reads
            // every field exactly once on the way. So its obligation is the sum
            // of its fields' obligations, and each of those lands on a name the
            // binding introduced, which is tracked. Counted as a resource it
            // was one nothing consumes, and a list carrying a `linear` value
            // was refused however correctly the caller consumed it.
            if ast.is_multi_return_struct(name) {
                continue;
            }
            let holds = field_types.iter().any(|ty| ty.is_linear_with(&held));
            if holds && held.insert(name.to_string()) {
                grew = true;
            }
        }
        // In the same loop as the holders, since an instance is a resource
        // because of a field and a struct is one because of an instance in a
        // field of its own.
        if crate::check::linear_instances::note_linear_instances(
            &templates, &instances, &mut held,
        ) {
            grew = true;
        }
        if !grew {
            return held;
        }
    }
}

fn parameter_type(parameter: &Parameter) -> Type {
    parameter.type_annotation.clone().unwrap_or(Type::I64)
}

#[derive(Clone)]
struct GenericFunction {
    type_params: Vec<String>,
    parameters: Range32,
    return_sig: SignatureId,
    body: Range32,
}

struct Specialization {
    generic_name: String,
    mangled_name: String,
    subst: HashMap<String, Type>,
    // Which module asked for this one, as a source map file id. A
    // specialization requested from inside another specialization inherits the
    // module that asked for the outer one, since that is the module that would
    // have to emit both once modules are compilation units.
    //
    // Nothing downstream uses this yet and the emitted code does not depend on
    // it. It exists to answer an open question: separate compilation gives
    // each module its own copy of a specialization it instantiates, and whether that
    // duplication is worth caring about is a measurement, not an opinion.
    requested_by: u32,
    // Where the call that asked for this one was written, and how it reads
    // there. A diagnostic from inside the stamped-out body names a line in the
    // template. This is the line the reader wrote.
    requested_at: Position,
    display: String,
    // The compile-time list this call bound, in order. None for a function
    // that has no list.
    pack: Option<(String, Vec<PackElement>)>,
}

// One element of a compile-time list. A call may hand it a value or a type, and
// the two are not the same thing afterwards: a value becomes an ordinary
// parameter of the specialization, evaluated once at the call, and a type
// becomes a name the body writes where a type belongs and nothing at run time.
#[derive(Clone)]
enum PackElement {
    Value(String, Type),
    Type(Type),
}

impl PackElement {
    // The element as an argument, for a call that hands a whole list on.
    fn as_argument(&self, ast: &mut Ast, span: TokenSpan) -> ExprId {
        match self {
            PackElement::Value(name, _) => {
                let symbol = ast.intern(name);
                ast.push_expr(Expression::Identifier(symbol), span)
            }
            PackElement::Type(ty) => {
                ast.push_expr(Expression::TypeValue(ty.clone()), span)
            }
        }
    }

    // How the element reads in a specialization's name and in a diagnostic.
    fn written(&self) -> String {
        match self {
            PackElement::Value(_, ty) => ty.to_string(),
            PackElement::Type(ty) => format!("${ty}"),
        }
    }
}

fn function_type_params(ast: &Ast, parameters: Range32) -> Vec<String> {
    let mut names = Vec::new();
    for parameter in ast.params_in(parameters) {
        collect_type_params(&parameter_type(parameter), &mut names);
    }
    names
}

fn function_is_generic(ast: &Ast, parameters: Range32) -> bool {
    !function_type_params(ast, parameters).is_empty()
        || ast
            .params_in(parameters)
            .iter()
            .any(|parameter| parameter.pack)
}

// The name the nth element of a list takes in a specialization. A list becomes
// ordinary parameters there, so each element is a name the unrolled body can
// use and a value the call evaluates once.
fn pack_element_name(pack: &str, index: usize) -> String {
    format!("{pack}__{index}")
}

// Whether an argument bound to a read-mode `$T` parameter should be passed by
// value. A scalar literal counts even though `probe_type` cannot name it, since
// it is not a place and has no address to borrow in the first place.
fn argument_is_copy_value(
    probed: Option<&Type>,
    ast: &Ast,
    argument: ExprId,
) -> bool {
    if let Some(ty) = probed {
        return ty.is_copy();
    }
    matches!(
        ast.expr(argument),
        // `true` and `false` parse as their own expression rather than as a
        // literal, and leaving that out here made `f($bool, true)` pass a
        // boolean by address: a read-mode `$T` stays a reference until the
        // argument says the type is copy, and a bare `true` said nothing.
        Expression::Boolean(_)
            | Expression::Literal(
                Literal::Integer(_)
                    | Literal::Float(_)
                    | Literal::Float32(_)
                    | Literal::Boolean(_)
            )
    )
}

// The type a specialized parameter has, which is not the substituted
// one. `lower_param_modes` turns a read-mode parameter into `Ref(T)` only when
// `T` is not copy, and it has to guess for `$T` because the answer arrives with
// the call. Guessing "reference" is the safe direction there, since it can be
// dropped here once the substitution says the type is copy, whereas a missing
// reference could not be added back.
fn specialized_param_type(
    declared: &Type,
    subst: &HashMap<String, Type>,
    mode: crate::parser::ParamMode,
) -> Type {
    let substituted = substitute_type(declared, subst);
    match (&substituted, declared) {
        (Type::Ref(inner), Type::Ref(under))
            if mode == crate::parser::ParamMode::Read
                && matches!(under.as_ref(), Type::TypeParam(_))
                && inner.is_copy() =>
        {
            inner.as_ref().clone()
        }
        _ => substituted,
    }
}

fn is_type_parameter(ast: &Ast, parameter: &Parameter) -> bool {
    matches!(
        &parameter.type_annotation,
        Some(Type::TypeParam(name)) if name.as_str() == ast.name(parameter.name)
    )
}

/// Which value parameter settles each compile-time parameter, where one does.
///
/// A compile-time parameter named in the declared type of a value parameter
/// after it is bound by unifying against that argument, so the call does not
/// write it: `vec_push(v, 3)` reads `T` off `v: Vec<T>`. One named nowhere else
/// is written, because nothing else says what it is: `vec_new($i64, 8)` answers
/// with a `Vec<T>` and takes a count.
///
/// Decided from the signature and nothing else, so a call has one spelling
/// rather than two. Both compilers run this same walk, over the same parameter
/// list, in the same order.
///
/// A bundle parameter is settled the same way a type is, which is what puts a
/// capability in a value's type: `$ops: Hashing<K>` named in `m: Map<K, V, ops>`
/// is read off the map, so every operation over one map hashes it the way it was
/// built. Only a value parameter settles one. A `$f: fn(T, T) -> bool` names `T`
/// in its own declared type, and that is not an argument whose type can be
/// unified against: what arrives for it is a name, picked at the call.
pub fn settled_by(ast: &Ast, parameters: &[Parameter]) -> Vec<Option<Symbol>> {
    parameters
        .iter()
        .enumerate()
        .map(|(index, parameter)| {
            if !is_type_parameter(ast, parameter)
                || matches!(
                    parameter.compile_time_signature,
                    Some(Type::Proc(..))
                )
            {
                return None;
            }
            let name = ast.name(parameter.name);
            parameters[index + 1..]
                .iter()
                .filter(|later| !is_type_parameter(ast, later) && !later.pack)
                .find(|later| {
                    later.type_annotation.as_ref().is_some_and(|declared| {
                        mentions_parameter(declared, name)
                    })
                })
                .map(|later| later.name)
        })
        .collect()
}

/// Which argument each parameter takes at a call, and `None` for a
/// compile-time parameter a value parameter settles, which takes none. Every
/// pass reading a call against a declaration lines the two up through this, so
/// an argument is weighed against the parameter it was written for rather than
/// the one beside it.
pub(crate) fn argument_slots(
    ast: &Ast,
    parameters: &[Parameter],
) -> Vec<Option<usize>> {
    let mut slots = Vec::with_capacity(parameters.len());
    let mut consumed = 0usize;
    for held in settled_by(ast, parameters) {
        if held.is_some() {
            slots.push(None);
            continue;
        }
        slots.push(Some(consumed));
        consumed += 1;
    }
    slots
}

/// The name of the parameter each argument of a call is written for, in order.
pub(crate) fn argument_names(
    ast: &Ast,
    parameters: &[Parameter],
) -> Vec<String> {
    parameters
        .iter()
        .zip(argument_slots(ast, parameters))
        .filter(|(_, slot)| slot.is_some())
        .map(|(parameter, _)| ast.name(parameter.name).to_string())
        .collect()
}

/// Where a call binds one compile-time parameter from.
pub(crate) enum GenericBinding {
    /// The argument at this position, written `$T` at the call.
    Written(usize),
    /// The type of the argument at this position, read through the declared
    /// type of the value parameter that takes it.
    Settled(usize, Type),
}

/// Every compile-time parameter of a signature, in declaration order, with
/// where a call binds it from. The positions count arguments rather than
/// parameters: a settled parameter takes no argument, so everything written
/// after it moves up one.
///
/// The passes that resolve types read a call through this, so each of them
/// lines a call's arguments up against the parameters that take them the same
/// way the lowering does.
pub(crate) fn generic_bindings(
    ast: &Ast,
    parameters: &[Parameter],
) -> Vec<(String, GenericBinding)> {
    let settled = settled_by(ast, parameters);
    let positions = argument_slots(ast, parameters);
    let mut bindings = Vec::new();
    for (index, parameter) in parameters.iter().enumerate() {
        if !is_type_parameter(ast, parameter) {
            continue;
        }
        let name = ast.name(parameter.name).to_string();
        let Some(by) = settled[index] else {
            if let Some(at) = positions[index] {
                bindings.push((name, GenericBinding::Written(at)));
            }
            continue;
        };
        let Some(settler) = parameters.iter().position(|held| held.name == by)
        else {
            continue;
        };
        if let Some(at) = positions[settler]
            && let Some(pattern) = parameters[settler].type_annotation.clone()
        {
            bindings.push((name, GenericBinding::Settled(at, pattern)));
        }
    }
    bindings
}

/// Whether a declared type names this compile-time parameter, anywhere inside
/// it. A parameter written straight through is a `TypeParam`; one written
/// inside a generic instance's argument list is a plain name there, since
/// `Vec<T>` is one type name until something binds `T`, and that is the shape
/// most of the library declares.
fn mentions_parameter(ty: &Type, name: &str) -> bool {
    match ty {
        Type::TypeParam(held) => held == name,
        Type::Struct(held) | Type::Enum(held) => {
            if held == name {
                return true;
            }
            let Some((_, arguments)) = split_instance(held) else {
                return false;
            };
            arguments.iter().any(|argument| {
                crate::parser::type_from_string(argument)
                    .is_ok_and(|held| mentions_parameter(&held, name))
            })
        }
        Type::Ptr(inner)
        | Type::Ref(inner)
        | Type::RefMut(inner)
        | Type::Slice(inner)
        | Type::Array(inner, _)
        | Type::ArrayGeneric(inner, _)
        | Type::Handle(inner)
        | Type::Distinct(_, inner) => mentions_parameter(inner, name),
        Type::Proc(parameters, answer) => {
            parameters
                .iter()
                .any(|parameter| mentions_parameter(parameter, name))
                || mentions_parameter(answer, name)
        }
        _ => false,
    }
}

fn collect_type_params(ty: &Type, out: &mut Vec<String>) {
    match ty {
        Type::TypeParam(name) => {
            if !out.contains(name) {
                out.push(name.clone());
            }
        }
        Type::Struct(name) if is_generic_instance(name) => {
            if let Some((_, arguments)) = split_instance(name) {
                for argument in arguments {
                    if let Ok(argument_type) =
                        crate::parser::type_from_string(&argument)
                    {
                        collect_type_params(&argument_type, out);
                    }
                }
            }
        }
        Type::Proc(params, ret) => {
            for param in params {
                collect_type_params(param, out);
            }
            collect_type_params(ret, out);
        }
        _ => {
            if let Some(inner) = single_inner(ty) {
                collect_type_params(inner, out);
            }
        }
    }
}

fn single_inner(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Ptr(inner)
        | Type::Ref(inner)
        | Type::RefMut(inner)
        | Type::Array(inner, _)
        | Type::Slice(inner)
        | Type::Handle(inner)
        | Type::Distinct(_, inner) => Some(inner),
        _ => None,
    }
}

pub(crate) fn substitute_type(
    ty: &Type,
    subst: &HashMap<String, Type>,
) -> Type {
    match ty {
        Type::Struct(name) if is_generic_instance(name) => {
            if let Some((base, arguments)) = split_instance(name) {
                let substituted: Vec<String> = arguments
                    .iter()
                    .map(|argument| {
                        match crate::parser::type_from_string(argument) {
                            Ok(argument_type) => {
                                substitute_type(&argument_type, subst)
                                    .to_string()
                            }
                            Err(_) => argument.clone(),
                        }
                    })
                    .collect();
                return Type::Struct(format!(
                    "{}<{}>",
                    base,
                    substituted.join(", ")
                ));
            }
            ty.clone()
        }
        Type::TypeParam(name) | Type::Struct(name) => {
            if let Some(concrete) = subst.get(name) {
                return concrete.clone();
            }
            ty.clone()
        }
        Type::Ptr(inner) => Type::Ptr(Box::new(substitute_type(inner, subst))),
        Type::Ref(inner) => Type::Ref(Box::new(substitute_type(inner, subst))),
        Type::RefMut(inner) => {
            Type::RefMut(Box::new(substitute_type(inner, subst)))
        }
        Type::Array(inner, size) => {
            Type::Array(Box::new(substitute_type(inner, subst)), *size)
        }
        Type::ArrayGeneric(inner, size) => {
            let inner = substitute_type(inner, subst);
            // Every name in the length has to be bound before it is worked out:
            // a length half known is still a length nobody can lay out.
            let known = size
                .evaluate(&|name| match subst.get(name) {
                    Some(Type::ConstUsize(size)) => i64::try_from(*size).ok(),
                    _ => None,
                })
                .and_then(|value| usize::try_from(value).ok());
            match known {
                Some(known) => Type::Array(Box::new(inner), known),
                None => Type::ArrayGeneric(Box::new(inner), size.clone()),
            }
        }
        Type::Slice(inner) => {
            Type::Slice(Box::new(substitute_type(inner, subst)))
        }
        Type::Handle(inner) => {
            Type::Handle(Box::new(substitute_type(inner, subst)))
        }
        Type::Distinct(name, inner) => Type::Distinct(
            name.clone(),
            Box::new(substitute_type(inner, subst)),
        ),
        Type::Proc(params, ret) => Type::Proc(
            params.iter().map(|p| substitute_type(p, subst)).collect(),
            Box::new(substitute_type(ret, subst)),
        ),
        other => other.clone(),
    }
}

pub(crate) fn infer_subst_into(
    pattern: &Type,
    concrete: &Type,
    type_params: &[String],
    subst: &mut HashMap<String, Type>,
) {
    // A read of an aggregate travels as a borrow, so a parameter forwarded from
    // one generic to another arrives here as `ref T` where the caller wrote
    // `T`. The surface has no borrow type, so a type parameter never stands for
    // one: what it binds to is the type the borrow names.
    let named = match concrete {
        Type::Ref(inner) | Type::RefMut(inner) => inner.as_ref(),
        other => other,
    };
    match pattern {
        Type::TypeParam(name) => {
            subst.entry(name.clone()).or_insert_with(|| named.clone());
            return;
        }
        Type::Struct(name) if type_params.contains(name) => {
            subst.entry(name.clone()).or_insert_with(|| named.clone());
            return;
        }
        _ => {}
    }
    // A reference parameter matched against a value argument. That is the
    // auto-borrow at a call site, where the caller hands over a place and the
    // callee takes its address, so inference has to look through the reference
    // to see the type the parameter is generic over.
    if let Type::Ref(pattern_inner) | Type::RefMut(pattern_inner) = pattern
        && !matches!(concrete, Type::Ref(_) | Type::RefMut(_) | Type::Ptr(_))
    {
        infer_subst_into(pattern_inner, concrete, type_params, subst);
        return;
    }
    // A `str` is a run of bytes, so a `[]$T` parameter given one binds its
    // element to `u8`. Without this the element of a `str` is nothing any
    // signature can name, and a body over bytes has to be written twice.
    if let Type::Slice(pattern_inner) = pattern
        && matches!(concrete, Type::Str)
    {
        infer_subst_into(pattern_inner, &Type::U8, type_params, subst);
        return;
    }
    if let (Some(pattern_inner), Some(concrete_inner)) =
        (single_inner(pattern), single_inner(concrete))
    {
        infer_subst_into(pattern_inner, concrete_inner, type_params, subst);
    } else if let (Type::Proc(pp, pr), Type::Proc(cp, cr)) = (pattern, concrete)
    {
        for (pattern_param, concrete_param) in pp.iter().zip(cp) {
            infer_subst_into(pattern_param, concrete_param, type_params, subst);
        }
        infer_subst_into(pr, cr, type_params, subst);
    } else if let (Type::Struct(pattern_name), Type::Struct(concrete_name)) =
        (pattern, concrete)
        && let (
            Some((pattern_base, pattern_args)),
            Some((concrete_base, concrete_args)),
        ) = (split_instance(pattern_name), split_instance(concrete_name))
        && pattern_base == concrete_base
        && pattern_args.len() == concrete_args.len()
    {
        for (pattern_arg, concrete_arg) in
            pattern_args.iter().zip(&concrete_args)
        {
            if let (Ok(pattern_type), Ok(concrete_type)) = (
                crate::parser::type_from_string(pattern_arg),
                crate::parser::type_from_string(concrete_arg),
            ) {
                infer_subst_into(
                    &pattern_type,
                    &concrete_type,
                    type_params,
                    subst,
                );
            }
        }
    }
}

fn sanitize_identifier(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect()
}

// A `.Name` written where the type is left out. There is one spelling for a
// value named under a type and the type is part of it, so this says what to
// write rather than filling it in.
//
// The type the context expects is named where there is one, which is what
// turns the report into the edit. Where there is none there is nothing to
// name, and the message says that instead.
fn refuse_inferred_variant(
    ast: &Ast,
    variant: Symbol,
    expected: Option<&Type>,
) -> anyhow::Error {
    let variant = ast.name(variant);
    let Some(Type::Enum(name) | Type::Struct(name) | Type::Distinct(name, _)) =
        expected
    else {
        return anyhow::anyhow!(
            "a value named under a type is written with the type in front of it, and there is no type here to name"
        );
    };
    let name = crate::modules::imports::demangle_private_names(name);
    anyhow::anyhow!(
        "a value named under a type is written with the type in front of it, so this one is written `{name}::{variant}`"
    )
}

// The struct a `{ x = 1 }` builds, taken from what the context expects.
fn name_inferred_literal(
    ast: &mut Ast,
    expression: ExprId,
    fields: Range32,
    expected: Option<&Type>,
) -> Result<ExprId> {
    let Some(Type::Struct(name) | Type::Enum(name)) = expected else {
        bail!(
            "a `{{ ... }}` literal takes its type from what the context expects, and here there is nothing to take it from; name the struct"
        );
    };
    let name = name.clone();
    let span = ast.expr_span(expression);
    let name = ast.intern(&name);
    Ok(ast.push_expr(Expression::StructInit(name, fields), span))
}

// A distinct type is built only from itself. Another distinct type over the
// same representation will not do, and neither will the representation, which
// is the whole point of declaring one. Reading one as its representation is
// allowed and is what a call into C is: a Meters is an i64 in memory, and
// nothing is at stake going that way.
//
// A literal is exempt. It has no type of its own until the context gives it
// one, which is what makes `m : Meters = 3` read the way it should.
//
// A flags type is not exempt. Its whole content is the names, so a number
// written where one belongs is the thing the declaration exists to replace.
fn distinct_mismatch(
    ast: &Ast,
    value: ExprId,
    from: &Type,
    to: &Type,
    flags: &HashMap<String, FlagsLayout>,
) -> bool {
    let Type::Distinct(name, _) = to else {
        return false;
    };
    let literal = matches!(
        ast.expr(value),
        Expression::Literal(Literal::Integer(_) | Literal::Float(_))
    );
    // A literal takes the type the context expects, so by the time the types
    // are compared they agree. For a flags type the question is what was
    // written rather than what it typed as.
    if flags.contains_key(name) {
        return literal || from != to;
    }
    if from == to {
        return false;
    }
    !literal
}

// How to describe what was written, and which rule it broke. A number written
// where a flags value belongs has taken the flags type by the time the types
// are compared, so saying what it typed as would name the same type twice and
// explain nothing. What the reader wrote is a number, and that is what this
// says.
fn nominal_reason(
    ast: &Ast,
    value: ExprId,
    to: &Type,
    flags: &HashMap<String, FlagsLayout>,
) -> (&'static str, &'static str) {
    let flagged =
        matches!(to, Type::Distinct(name, _) if flags.contains_key(name));
    if !flagged {
        return ("", "a distinct type is not its representation");
    }
    if matches!(
        ast.expr(value),
        Expression::Literal(Literal::Integer(_) | Literal::Float(_))
    ) {
        return (
            "a number",
            "a set of bits is built from the names declared under it, and a number is not one of them",
        );
    }
    (
        "",
        "a set of bits is built only from the names declared under it",
    )
}

// The two halves as the message writes them: what the value is, and why it does
// not fit.
fn nominal_words(
    ast: &Ast,
    value: ExprId,
    value_type: &Type,
    to: &Type,
    flags: &HashMap<String, FlagsLayout>,
) -> (String, String) {
    let (described, note) = nominal_reason(ast, value, to, flags);
    let described = if described.is_empty() {
        format!("a '{}'", spelled(value_type))
    } else {
        described.to_string()
    };
    (described, note.to_string())
}

fn mangle_type(ty: &Type) -> String {
    match ty {
        Type::I8 => "i8".to_string(),
        Type::I16 => "i16".to_string(),
        Type::I32 => "i32".to_string(),
        Type::I64 => "i64".to_string(),
        Type::Isize => "isize".to_string(),
        Type::U8 => "u8".to_string(),
        Type::U16 => "u16".to_string(),
        Type::U32 => "u32".to_string(),
        Type::U64 => "u64".to_string(),
        Type::Usize => "usize".to_string(),
        Type::F32 => "f32".to_string(),
        Type::F64 => "f64".to_string(),
        Type::Bool => "bool".to_string(),
        Type::Struct(name) | Type::Enum(name) => sanitize_identifier(name),
        Type::Ptr(inner) => format!("p_{}", mangle_type(inner)),
        Type::Ref(inner) => format!("r_{}", mangle_type(inner)),
        Type::RefMut(inner) => format!("rm_{}", mangle_type(inner)),
        Type::Array(inner, size) => format!("a{}_{}", size, mangle_type(inner)),
        Type::Handle(inner) => format!("h_{}", mangle_type(inner)),
        Type::Proc(_, _) => "proc".to_string(),
        Type::ConstFn(name) | Type::ConstValue(name) => {
            sanitize_identifier(name)
        }
        other => format!("{other}"),
    }
}

// `add<Point>`: the specialization named the way the reader wrote the call,
// rather than the mangled symbol, which is a compiler artifact.
fn describe_specialization(
    name: &str,
    type_params: &[String],
    subst: &HashMap<String, Type>,
) -> String {
    if type_params.is_empty() {
        return name.to_string();
    }
    let arguments: Vec<String> = type_params
        .iter()
        .map(|type_param| match subst.get(type_param) {
            Some(concrete) => concrete.to_string(),
            None => type_param.clone(),
        })
        .collect();
    format!("{name}<{}>", arguments.join(", "))
}

fn mangle_specialization(
    name: &str,
    type_params: &[String],
    subst: &HashMap<String, Type>,
) -> String {
    let mut mangled = name.to_string();
    for type_param in type_params {
        mangled.push_str("__");
        match subst.get(type_param) {
            Some(concrete) => mangled.push_str(&mangle_type(concrete)),
            None => mangled.push_str("unknown"),
        }
    }
    mangled
}

// A `where` bound asks what the compiler already knows about a type. The
// vocabulary is fixed and closed, and every one of these is a question the
// compiler answers for itself anyway, to decide whether to emit an integer or
// a floating point instruction, whether a value travels by address, and how
// wide it is. So a bound is a precondition, not a set of operations a type
// registers into: nothing implements it, nothing is named, and there is
// nothing to resolve.
//
// What is deliberately absent is any predicate keyed by a string, such as
// asking whether a type has a field of a given name. A string literal does not
// grep back to the declaration it names, which is the one thing the flat
// namespace is for.
fn type_predicate(
    name: &str,
    ty: &Type,
    linear: &HashSet<String>,
) -> Option<bool> {
    let answer = match name {
        "is_numeric" => ty.is_integer() || ty.is_float(),
        "is_integer" => ty.is_integer(),
        "is_float" => ty.is_float(),
        "is_struct" => matches!(ty, Type::Struct(_) | Type::Enum(_)),
        "is_array" => matches!(ty, Type::Array(..) | Type::ArrayGeneric(..)),
        "is_slice" => matches!(ty, Type::Slice(_) | Type::Str),
        "is_pointer" => {
            matches!(ty, Type::Ptr(_) | Type::Ref(_) | Type::RefMut(_))
        }
        // Whether a value of this type has to be consumed exactly once. A
        // container that reaches an element by a number cannot say which
        // element it took, so the generic ones hold themselves to types where
        // there is nothing to say: `where !is_linear(T)` is how a function
        // that would otherwise drop or duplicate a resource refuses the
        // binding at the call rather than leaking inside a body nobody wrote.
        "is_linear" => ty.is_linear_with(linear),
        _ => return None,
    };
    Some(answer)
}

const BOUND_VOCABULARY: &str = "is_numeric, is_integer, is_float, is_struct, is_array, is_slice, is_pointer, is_linear";

// What a formatted value may be: a number, a yes or no, or text. Read through
// any name a type carries, the same way the bounds vocabulary reads one.
// What a distinct type is represented by, which is what a rule about shapes
// asks about. The name it carries is a question for the rules about names.
fn through_distinct(ty: &Type) -> &Type {
    match ty {
        Type::Distinct(_, inner) => through_distinct(inner),
        other => other,
    }
}

// A type spelled the way a reader writes one.
//
// `Display` is the name the compiler files a type under, and it round-trips
// through `type_from_string`, which is what monomorphization reads a
// specialization's arguments back out of. A borrow has no spelling there to
// round-trip and takes `&T` and `&mut T`, which are two forms the surface
// dropped: what a reader writes is `ref T`. So a report spells one and the
// table keeps the other.
fn writable_by_format(ty: &Type) -> bool {
    match ty {
        Type::Distinct(_, inner) => writable_by_format(inner),
        Type::Bool | Type::Str => true,
        other => other.is_integer() || other.is_float(),
    }
}

// The argument a `format` parameter took, read where the call is written. How
// many values a line names is settled here, so the literal is read here too:
// the holes it opens are counted against the list that follows, and a type
// nothing can write is refused against the line the reader wrote rather than
// inside a body they never saw.
fn check_format(
    ast: &Ast,
    argument: ExprId,
    elements: &[PackElement],
) -> Result<()> {
    let Expression::Literal(Literal::String(text)) = ast.expr(argument) else {
        bail!(
            "a format string is written as a literal, since how many values follow it is settled where the call is written"
        )
    };
    let bytes = text.as_bytes();
    let mut holes = 0usize;
    let mut at = 0usize;
    while at < bytes.len() {
        if bytes[at] == b'{' {
            match bytes.get(at + 1) {
                Some(b'}') => holes += 1,
                Some(b'{') => {}
                _ => bail!(
                    "a '{{' in a format string opens a hole or stands for one brace, so write '{{}}' or '{{{{'"
                ),
            }
            at += 2;
            continue;
        }
        if bytes[at] == b'}' && bytes.get(at + 1) == Some(&b'}') {
            at += 2;
            continue;
        }
        at += 1;
    }
    if holes != elements.len() {
        bail!(
            "this format string opens {holes} hole(s) and the call gives {} value(s)",
            elements.len()
        )
    }
    for element in elements {
        let PackElement::Value(_, ty) = element else {
            bail!("a format string writes a value, and a type is not one")
        };
        if !writable_by_format(ty) {
            bail!(
                "a format string writes a number, a yes or no, or a str, and this is a {}",
                spelled(ty)
            )
        }
    }
    Ok(())
}

/// How deep a bound naming a function may reach. One that reaches itself is
/// caught by name where a call is, and this is what catches a long chain.
const BOUND_DEPTH: usize = 32;

/// What a bound reads besides the types it was given: the functions it may name
/// and the layouts a measurement in one is answered from.
#[derive(Clone, Copy)]
struct Bounding<'a> {
    bounds: &'a HashMap<String, (String, ExprId)>,
    linear: &'a HashSet<String>,
    structs: &'a HashMap<String, StructLayout>,
    enums: &'a HashMap<String, EnumLayout>,
}

/// A number a bound compares against: one written down, or what a type
/// measures. A measurement is the same question `sizeof` answers anywhere, and
/// the types have been laid out by the time a call is checked against a bound.
fn bound_number(
    ast: &Ast,
    expression: ExprId,
    subst: &HashMap<String, Type>,
    context: Bounding<'_>,
) -> Result<i64> {
    match ast.expr(expression) {
        Expression::Literal(Literal::Integer(held)) => Ok(*held),
        Expression::Infix(left, operator, right) => {
            let left = bound_number(ast, *left, subst, context)?;
            let right = bound_number(ast, *right, subst, context)?;
            match operator {
                crate::parser::Operator::Add => Ok(left + right),
                crate::parser::Operator::Subtract => Ok(left - right),
                crate::parser::Operator::Multiply => Ok(left * right),
                crate::parser::Operator::Divide if right != 0 => {
                    Ok(left / right)
                }
                crate::parser::Operator::Modulo if right != 0 => {
                    Ok(left % right)
                }
                other => bail!(
                    "a number in a bound is arithmetic over what a type measures, and '{other}' is not part of that"
                ),
            }
        }
        Expression::Call(callee, arguments) => {
            let Expression::Identifier(named) = ast.expr(*callee) else {
                bail!("a number in a bound is what a type measures")
            };
            let named = ast.name(*named).to_string();
            let [only] = ast.exprs_in(*arguments) else {
                bail!("'{named}' measures one type")
            };
            let ty = match ast.expr(*only) {
                // `sizeof(T)` reads its argument as a type, so the parameter
                // arrives as a name of one and what the call bound it to is
                // what it measures.
                Expression::TypeValue(held) => substitute_type(held, subst),
                Expression::Identifier(parameter) => {
                    let parameter = ast.name(*parameter);
                    let Some(held) = subst.get(parameter) else {
                        bail!(
                            "the bound names '{parameter}', which is not a compile-time parameter of this function"
                        )
                    };
                    held.clone()
                }
                _ => bail!("'{named}' measures a type"),
            };
            let Some(Literal::Integer(held)) =
                measured(&ty, context.structs, context.enums, &named)
            else {
                bail!(
                    "'{named}' is not a measurement a bound can be written over, which are sizeof, alignof and field_count"
                )
            };
            Ok(held)
        }
        _ => bail!(
            "a number in a bound is what a type measures, or arithmetic over one"
        ),
    }
}

fn evaluate_bound(
    ast: &Ast,
    expression: ExprId,
    subst: &HashMap<String, Type>,
    context: Bounding<'_>,
    depth: usize,
) -> Result<bool> {
    match ast.expr(expression) {
        // A bound may weigh what a type measures against a number: a container
        // that packs its element into a word is written for the types that fit
        // in one, and saying so is what a bound is for.
        Expression::Infix(left, operator, right)
            if matches!(
                operator,
                crate::parser::Operator::LessThan
                    | crate::parser::Operator::LessThanOrEqual
                    | crate::parser::Operator::GreaterThan
                    | crate::parser::Operator::GreaterThanOrEqual
                    | crate::parser::Operator::Equal
                    | crate::parser::Operator::NotEqual
            ) =>
        {
            let left = bound_number(ast, *left, subst, context)?;
            let right = bound_number(ast, *right, subst, context)?;
            Ok(match operator {
                crate::parser::Operator::LessThan => left < right,
                crate::parser::Operator::LessThanOrEqual => left <= right,
                crate::parser::Operator::GreaterThan => left > right,
                crate::parser::Operator::GreaterThanOrEqual => left >= right,
                crate::parser::Operator::Equal => left == right,
                _ => left != right,
            })
        }
        Expression::Infix(left, operator, right) => {
            let left = evaluate_bound(ast, *left, subst, context, depth)?;
            let right = evaluate_bound(ast, *right, subst, context, depth)?;
            match operator {
                crate::parser::Operator::And => Ok(left && right),
                crate::parser::Operator::Or => Ok(left || right),
                other => bail!(
                    "a `where` bound combines its terms with `&&`, `||` and `!`, and '{other}' is none of those"
                ),
            }
        }
        Expression::Prefix(crate::parser::Operator::Not, inner) => {
            Ok(!evaluate_bound(ast, *inner, subst, context, depth)?)
        }
        Expression::Call(callee, arguments) => {
            let Expression::Identifier(predicate) = ast.expr(*callee) else {
                bail!(
                    "a `where` bound is a predicate applied to a compile-time parameter"
                )
            };
            let predicate = ast.name(*predicate);
            if arguments.len() != 1 {
                bail!(
                    "'{predicate}' takes one compile-time parameter, and {} were given",
                    arguments.len()
                )
            }
            let Expression::Identifier(parameter) =
                ast.expr(ast.exprs_in(*arguments)[0])
            else {
                bail!("'{predicate}' takes a compile-time parameter by name")
            };
            let parameter = ast.name(*parameter);
            let Some(ty) = subst.get(parameter) else {
                bail!(
                    "the bound names '{parameter}', which is not a compile-time parameter of this function"
                )
            };
            if let Some(answer) = type_predicate(predicate, ty, context.linear)
            {
                return Ok(answer);
            }
            // A function the program declares, asked the same question. Its
            // body is one expression over this same vocabulary, read by this
            // same reader with its own parameter standing for the type, so a
            // bound written once under a name says what it would have said
            // written out. One that reaches itself is caught by the depth this
            // recursion is held to.
            if let Some((parameter, body)) = context.bounds.get(predicate) {
                if depth >= BOUND_DEPTH {
                    bail!(
                        "'{predicate}' reaches itself, and a bound is answered by reading it, which never ends"
                    )
                }
                let mut held: HashMap<String, Type> = HashMap::new();
                held.insert(parameter.clone(), ty.clone());
                return evaluate_bound(ast, *body, &held, context, depth + 1);
            }
            // A name that is neither a bound nor a function of one is a mistake
            // in the declaration, so the fault lands on the predicate rather
            // than on the call, which chose a type and nothing else.
            locate(
                Err(anyhow::anyhow!(
                    "'{predicate}' is not one of the bounds a type can be held to, which are: {BOUND_VOCABULARY}, and no function of this program answers it"
                )),
                ast.position_of(ast.expr_span(*callee)),
            )
        }
        _ => bail!(
            "a `where` bound is a predicate applied to a compile-time parameter, and '{}' is not one",
            display_expr(ast, expression)
        ),
    }
}

// The bound, and what to say when it does not hold. The binding is named, since
// the reader chose it at the call and the template is not theirs.
fn check_bound(
    ast: &Ast,
    signature: &ReturnSignature,
    subst: &HashMap<String, Type>,
    callee: &str,
    context: Bounding<'_>,
) -> Result<()> {
    let Some(bound) = signature.bound else {
        return Ok(());
    };
    if evaluate_bound(ast, bound, subst, context, 0)? {
        return Ok(());
    }
    let written = &signature.bound_text;
    let mut bindings: Vec<String> = subst
        .iter()
        .map(|(name, ty)| format!("{name} = {ty}"))
        .collect();
    bindings.sort();
    bail!(
        "'{callee}' is declared `where {written}`, and that does not hold for {}",
        bindings.join(", ")
    )
}

// ---------------------------------------------------------------------------
// Expansion time.
//
// Two constructs are decided while a specialization is being made rather than
// when it runs: a `for` over a compile-time pack, which unrolls into one copy
// of its body per element, and an `if` over a type predicate, which keeps the
// branch that survives and drops the other before anything checks it.
//
// Both iterate a list whose length is known once the generic is instantiated.
// There is no recursion, no unbounded loop, and nothing that reads the world,
// so what this costs is bounded by the program's own text. That is the whole
// difference between this and a compile-time interpreter.
// ---------------------------------------------------------------------------

struct Expansion<'a> {
    // The pack this specialization bound: its name, and the parameters that
    // took its elements.
    pack: Option<&'a (String, Vec<PackElement>)>,
    // What each parameter of the specialization is, so a predicate over one
    // has an answer.
    types: HashMap<String, Type>,
    // Every struct's layout, which is what a walk over a type's fields reads.
    // The compiler laid these out to emit the program. A layout table is the
    // same numbers, written where the reader can use them.
    structs: &'a HashMap<String, StructLayout>,
    // The type arguments this specialization was made for, so `fields(T)` in a
    // generic body names the type the call chose.
    subst: &'a HashMap<String, Type>,
    // The fields in force: a `for` over `fields(T)` binds its name to one of
    // them per copy of the body. A field is not a value, so the only things
    // that read this are `offset_of`, `sizeof` and the type predicates.
    fields: HashMap<String, (usize, Type)>,
    // The types that have to be consumed, so `is_linear` answers here the way
    // it answers in a `where` bound. Both ask the same question of the same
    // set, so an expansion-time `if` and a call-site bound cannot disagree.
    linear: &'a HashSet<String>,
}

// What expansion reads that does not change while it runs: the layouts a walk
// over a type's fields reports, the arguments this specialization was made for,
// and the types that have to be consumed.
struct ExpansionContext<'a> {
    structs: &'a HashMap<String, StructLayout>,
    subst: &'a HashMap<String, Type>,
    linear: &'a HashSet<String>,
}

fn expand_compile_time(
    ast: &mut Ast,
    body: Range32,
    pack: Option<&(String, Vec<PackElement>)>,
    parameters: Range32,
    context: ExpansionContext<'_>,
) -> Result<Range32> {
    let ExpansionContext {
        structs,
        subst,
        linear,
    } = context;
    let mut types = HashMap::new();
    for parameter in ast.params_in(parameters) {
        if let Some(ty) = &parameter.type_annotation {
            types.insert(ast.name(parameter.name).to_string(), ty.clone());
        }
    }
    let expansion = Expansion {
        pack,
        types,
        structs,
        subst,
        fields: HashMap::new(),
        linear,
    };
    expansion.block(ast, body)
}

impl Expansion<'_> {
    fn block(&self, ast: &mut Ast, block: Range32) -> Result<Range32> {
        let expanded = self.statements(ast, block)?;
        Ok(ast.add_stmt_list(&expanded))
    }

    fn statements(&self, ast: &mut Ast, block: Range32) -> Result<Vec<StmtId>> {
        let statements: Vec<StmtId> = ast.stmts_in(block).to_vec();
        let mut expanded = Vec::with_capacity(statements.len());
        for statement in statements {
            // A `for` over the pack, and an `if` whose condition is answered
            // here, both stand for several statements or none, so they are
            // spliced rather than replaced.
            // A `for` over a type's fields: the body is written once and
            // compiled once per field, with the loop's name standing for that
            // field. The list is the struct's own field list, so its length is
            // fixed by a declaration rather than by anything this walks.
            if let Statement::For(variable, None, iterable, body) =
                ast.stmt(statement).clone()
                && let Some(layout) = self.fields_named(ast, iterable)
            {
                let variable = ast.name(variable).to_string();
                let fields: Vec<(usize, Type)> = layout
                    .fields
                    .iter()
                    .map(|field| (field.offset, field.ty.clone()))
                    .collect();
                for field in fields {
                    let bound = self.with_field(&variable, field);
                    expanded.extend(bound.statements(ast, body)?);
                }
                continue;
            }
            if let Statement::For(variable, None, iterable, body) =
                ast.stmt(statement).clone()
                && self.pack_named(ast, iterable).is_some()
            {
                let variable = ast.name(variable).to_string();
                let elements = self
                    .pack_named(ast, iterable)
                    .expect("the pack was just matched")
                    .clone();
                for element in &elements {
                    // A value element is a parameter of this specialization, so
                    // the loop's name stands for that parameter. A type element
                    // is not a value at all: the loop's name is a type, and
                    // what the body wrote it in are type positions.
                    match element {
                        PackElement::Value(name, _) => {
                            let bound = substitute_identifier(
                                ast, body, &variable, name,
                            );
                            expanded.extend(self.statements(ast, bound)?);
                        }
                        PackElement::Type(ty) => {
                            let one =
                                HashMap::from([(variable.clone(), ty.clone())]);
                            let bound = substitute_block(ast, body, &one);
                            let inner = self.with_type(&variable, ty.clone());
                            expanded.extend(inner.statements(ast, bound)?);
                        }
                    }
                }
                continue;
            }
            if let Statement::Expression(value) = ast.stmt(statement)
                && let Expression::If(condition, consequence, alternative) =
                    ast.expr(*value).clone()
                && let Some(taken) = self.answer(ast, condition)?
            {
                let kept = if taken {
                    Some(consequence)
                } else {
                    alternative
                };
                if let Some(kept) = kept {
                    expanded.extend(self.statements(ast, kept)?);
                }
                continue;
            }
            expanded.push(self.statement(ast, statement)?);
        }
        Ok(expanded)
    }

    // The struct a `fields(...)` names, when this expression is one. The
    // argument is a type: a type parameter this specialization bound, or a
    // struct named outright.
    fn fields_named(
        &self,
        ast: &Ast,
        expression: ExprId,
    ) -> Option<&StructLayout> {
        let Expression::Call(callee, arguments) = ast.expr(expression) else {
            return None;
        };
        let Expression::Identifier(named) = ast.expr(*callee) else {
            return None;
        };
        if ast.name(*named) != "fields" || arguments.len() != 1 {
            return None;
        }
        self.structs
            .get(&self.named_type(ast, ast.exprs_in(*arguments)[0])?)
    }

    // The name of the type an expression names, following the type arguments
    // this specialization was made for.
    fn named_type(&self, ast: &Ast, expression: ExprId) -> Option<String> {
        let named = match ast.expr(expression) {
            Expression::Identifier(named) => ast.name(*named).to_string(),
            Expression::TypeValue(Type::Struct(named)) => named.clone(),
            _ => return None,
        };
        match self.subst.get(&named) {
            Some(Type::Struct(concrete)) => Some(concrete.clone()),
            Some(_) => None,
            None => Some(named),
        }
    }

    // The field a name is bound to, for a name a `for` over `fields(T)` bound.
    fn field_named(
        &self,
        ast: &Ast,
        expression: ExprId,
    ) -> Option<&(usize, Type)> {
        match ast.expr(expression) {
            Expression::Identifier(named) => self.fields.get(ast.name(*named)),
            _ => None,
        }
    }

    // This expansion with one more field in force.
    // One argument of a `g(T) for T in list`: the template with the element's
    // name standing for that element, expanded as an ordinary expression.
    fn mapped(
        &self,
        ast: &mut Ast,
        element: &PackElement,
        variable: &str,
        body: ExprId,
    ) -> Result<ExprId> {
        match element {
            PackElement::Value(name, _) => {
                let bound = substitute_identifier_in_expression(
                    ast, body, variable, name,
                );
                self.expression(ast, bound)
            }
            PackElement::Type(ty) => {
                let one = HashMap::from([(variable.to_string(), ty.clone())]);
                let bound = substitute_expression(ast, body, &one);
                let inner = self.with_type(variable, ty.clone());
                inner.expression(ast, bound)
            }
        }
    }

    // A `for` over a list of types binds its name to one of them per copy of
    // the body. The name is a type there, so what reads it is a type predicate,
    // `sizeof`, and every position that names a type.
    fn with_type(&self, name: &str, ty: Type) -> Expansion<'_> {
        let mut types = self.types.clone();
        types.insert(name.to_string(), ty);
        Expansion {
            pack: self.pack,
            types,
            structs: self.structs,
            subst: self.subst,
            fields: self.fields.clone(),
            linear: self.linear,
        }
    }

    fn with_field(&self, name: &str, field: (usize, Type)) -> Expansion<'_> {
        let mut fields = self.fields.clone();
        fields.insert(name.to_string(), field);
        Expansion {
            pack: self.pack,
            types: self.types.clone(),
            structs: self.structs,
            subst: self.subst,
            fields,
            linear: self.linear,
        }
    }

    // A call this answers at expansion time: how many fields a type has, where
    // a field sits, and how wide it is. Every one of them is a number the
    // compiler worked out to lay the type out.
    fn constant_call(
        &self,
        ast: &Ast,
        expression: ExprId,
    ) -> Result<Option<i64>> {
        let Expression::Call(callee, arguments) = ast.expr(expression) else {
            return Ok(None);
        };
        let Expression::Identifier(named) = ast.expr(*callee) else {
            return Ok(None);
        };
        let named = ast.name(*named);
        if arguments.len() != 1 {
            return Ok(None);
        }
        let argument = ast.exprs_in(*arguments)[0];
        if named == "field_count"
            && let Some(name) = self.named_type(ast, argument)
            && let Some(layout) = self.structs.get(&name)
        {
            return Ok(Some(layout.fields.len() as i64));
        }
        if named == "offset_of" {
            // The fault is what was written inside the parentheses, so it
            // lands there. Left unplaced it carried the position of whatever
            // declaration was being read, which is the head of the function a
            // reader would then go looking through.
            let Some((offset, _)) = self.field_named(ast, argument) else {
                return locate(
                    Err(anyhow::anyhow!(
                        "offset_of names a field of a type, which is what a `for` over `fields(T)` binds"
                    )),
                    ast.position_of(ast.expr_span(argument)),
                );
            };
            return Ok(Some(*offset as i64));
        }
        Ok(None)
    }

    // The elements of the pack, when this expression names it.
    fn pack_named(
        &self,
        ast: &Ast,
        expression: ExprId,
    ) -> Option<&Vec<PackElement>> {
        let (name, elements) = self.pack?;
        match ast.expr(expression) {
            Expression::Identifier(named) if ast.name(*named) == name => {
                Some(elements)
            }
            _ => None,
        }
    }

    // Whether a condition is one this can answer, and what it answers. `None`
    // means it is an ordinary condition and stays one.
    fn answer(&self, ast: &Ast, condition: ExprId) -> Result<Option<bool>> {
        match ast.expr(condition) {
            Expression::Prefix(crate::parser::Operator::Not, inner) => {
                Ok(self.answer(ast, *inner)?.map(|held| !held))
            }
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(predicate) = ast.expr(*callee)
                else {
                    return Ok(None);
                };
                if arguments.len() != 1 {
                    return Ok(None);
                }
                let Expression::Identifier(subject) =
                    ast.expr(ast.exprs_in(*arguments)[0])
                else {
                    return Ok(None);
                };
                let subject = ast.name(*subject);
                // A field the `for` around this bound, a parameter of the
                // specialization, or the type argument it was made for. The
                // third is what a `where` bound asks about, and the same
                // question in an `if` had no answer, so one vocabulary read two
                // ways depending on which of the two positions it was written
                // in.
                let ty = match self.fields.get(subject) {
                    Some((_, ty)) => ty,
                    None => match self.types.get(subject) {
                        Some(ty) => ty,
                        None => match self.subst.get(subject) {
                            Some(ty) => ty,
                            None => return Ok(None),
                        },
                    },
                };
                Ok(type_predicate(ast.name(*predicate), ty, self.linear))
            }
            _ => Ok(None),
        }
    }

    fn statement(&self, ast: &mut Ast, statement: StmtId) -> Result<StmtId> {
        let span = ast.stmt_span(statement);
        let expanded = match ast.stmt(statement).clone() {
            Statement::Let {
                name,
                type_annotation,
                value,
                mutable,
            } => Statement::Let {
                name,
                type_annotation,
                value: self.expression(ast, value)?,
                mutable,
            },
            Statement::Constant(name, value) => {
                Statement::Constant(name, self.expression(ast, value)?)
            }
            Statement::Return(value) => {
                Statement::Return(self.expression(ast, value)?)
            }
            Statement::Expression(value) => {
                Statement::Expression(self.expression(ast, value)?)
            }
            Statement::Assignment(place, value) => Statement::Assignment(
                self.expression(ast, place)?,
                self.expression(ast, value)?,
            ),
            Statement::Defer(inner) => {
                Statement::Defer(self.statement(ast, inner)?)
            }
            Statement::ErrDefer(inner) => {
                Statement::ErrDefer(self.statement(ast, inner)?)
            }
            Statement::For(variable, second, iterable, body) => Statement::For(
                variable,
                second,
                self.expression(ast, iterable)?,
                self.block(ast, body)?,
            ),
            Statement::While(condition, body) => Statement::While(
                self.expression(ast, condition)?,
                self.block(ast, body)?,
            ),
            Statement::With(capability, body) => {
                Statement::With(capability, self.block(ast, body)?)
            }
            _ => return Ok(statement),
        };
        Ok(ast.push_stmt(expanded, span))
    }

    fn expression(&self, ast: &mut Ast, expression: ExprId) -> Result<ExprId> {
        let span = ast.expr_span(expression);
        // `offset_of(field)` and `field_count(T)` are numbers this works out
        // here, where the layout is known and nothing has been emitted yet.
        if let Some(value) = self.constant_call(ast, expression)? {
            return Ok(ast.push_expr(
                Expression::Literal(Literal::Integer(value)),
                span,
            ));
        }
        let node = ast.expr(expression).clone();
        // `sizeof(field)` is the width of what that field holds,
        // `alignof(field)` what it is aligned to, and `type_id(field)` its
        // number. A field reads as a named type to the parser, which is what
        // makes this the place that tells the two apart. A name a `for` over a
        // list of types bound is a type here and nowhere else, and resolves the
        // same way.
        if let Expression::Call(callee, arguments) = &node
            && let Expression::Identifier(named) = ast.expr(*callee)
            && matches!(ast.name(*named), "sizeof" | "alignof" | "type_id")
            && arguments.len() == 1
        {
            let callee = *callee;
            let argument = ast.exprs_in(*arguments)[0];
            let resolved = match ast.expr(argument) {
                Expression::TypeValue(Type::Struct(written)) => {
                    match self.fields.get(written) {
                        Some((_, ty)) => Some(ty.clone()),
                        None => self.types.get(written).cloned(),
                    }
                }
                _ => None,
            };
            if let Some(ty) = resolved {
                let argument = ast.push_expr(Expression::TypeValue(ty), span);
                let arguments = ast.add_expr_list(&[argument]);
                return Ok(
                    ast.push_expr(Expression::Call(callee, arguments), span)
                );
            }
        }
        // A type predicate is a question this answers wherever it is asked, not
        // only in the condition of an `if`, so a table may carry the answer as
        // an ordinary field.
        if let Some(held) = self.answer(ast, expression)? {
            return Ok(ast.push_expr(Expression::Boolean(held), span));
        }
        // A field is not a value. Naming one anywhere else is a mistake worth
        // catching here rather than as an unknown variable later.
        if let Expression::Identifier(named) = &node
            && self.fields.contains_key(ast.name(*named))
        {
            bail!(
                "'{named}' is a field of a type, so it is asked about with `offset_of`, `sizeof` and the type predicates, and is not a value",
                named = ast.name(*named)
            )
        }
        // `pack[K]` is the Kth element, which is a parameter of this
        // specialization. Anything else that names the pack is an error: a
        // compile-time list is not a value.
        if let Expression::Index(base, index) = &node
            && let Some(elements) = self.pack_named(ast, *base)
        {
            let Expression::Literal(Literal::Integer(at)) = ast.expr(*index)
            else {
                bail!(
                    "a compile-time list is indexed by a literal, since which element it is has to be known here"
                )
            };
            let at = *at;
            let Some(element) =
                usize::try_from(at).ok().and_then(|at| elements.get(at))
            else {
                bail!(
                    "this call gave {} element(s) to the list, so there is no element {at}",
                    elements.len()
                )
            };
            return Ok(match element {
                PackElement::Value(name, _) => {
                    let name = name.clone();
                    let symbol = ast.intern(&name);
                    ast.push_expr(Expression::Identifier(symbol), span)
                }
                PackElement::Type(ty) => {
                    ast.push_expr(Expression::TypeValue(ty.clone()), span)
                }
            });
        }
        if self.pack_named(ast, expression).is_some() {
            bail!(
                "a compile-time list is iterated with `for` or indexed by a literal, and is not a value of its own"
            )
        }
        let expanded = match node {
            Expression::Prefix(operator, inner) => {
                Expression::Prefix(operator, self.expression(ast, inner)?)
            }
            Expression::Infix(left, operator, right) => {
                let left = self.expression(ast, left)?;
                let right = self.expression(ast, right)?;
                Expression::Infix(left, operator, right)
            }
            Expression::If(condition, consequence, alternative) => {
                Expression::If(
                    self.expression(ast, condition)?,
                    self.block(ast, consequence)?,
                    match alternative {
                        Some(block) => Some(self.block(ast, block)?),
                        None => None,
                    },
                )
            }
            Expression::Call(callee, arguments) => {
                let argument_ids: Vec<ExprId> =
                    ast.exprs_in(arguments).to_vec();
                let mut expanded_arguments =
                    Vec::with_capacity(argument_ids.len());
                for argument in argument_ids {
                    // An argument list is the one place a compile-time list
                    // stands for several things at once. Naming it hands over
                    // its elements, which is how one list is passed on to
                    // another. `g(T) for T in list` hands over the template
                    // once per element, which is how a call gets an arity the
                    // list decides.
                    if self.pack_named(ast, argument).is_some() {
                        let elements = self
                            .pack_named(ast, argument)
                            .expect("the pack was just matched")
                            .clone();
                        let argument_span = ast.expr_span(argument);
                        for element in &elements {
                            expanded_arguments
                                .push(element.as_argument(ast, argument_span));
                        }
                        continue;
                    }
                    if let Expression::PackMap(body, variable, list) =
                        ast.expr(argument).clone()
                    {
                        let variable = ast.name(variable).to_string();
                        let list = ast.name(list).to_string();
                        let elements = match self.pack {
                            Some((name, elements)) if *name == list => {
                                elements.clone()
                            }
                            _ => bail!(
                                "`for {variable} in {list}` walks a compile-time list, and '{list}' is not one here"
                            ),
                        };
                        for element in &elements {
                            expanded_arguments.push(
                                self.mapped(ast, element, &variable, body)?,
                            );
                        }
                        continue;
                    }
                    expanded_arguments.push(self.expression(ast, argument)?);
                }
                let callee = self.expression(ast, callee)?;
                Expression::Call(callee, ast.add_expr_list(&expanded_arguments))
            }
            Expression::Index(base, index) => {
                let base = self.expression(ast, base)?;
                let index = self.expression(ast, index)?;
                Expression::Index(base, index)
            }
            Expression::FieldAccess(base, field) => {
                Expression::FieldAccess(self.expression(ast, base)?, field)
            }
            Expression::AddressOf(inner) => {
                Expression::AddressOf(self.expression(ast, inner)?)
            }
            Expression::Borrow(inner) => {
                Expression::Borrow(self.expression(ast, inner)?)
            }
            Expression::BorrowMut(inner) => {
                Expression::BorrowMut(self.expression(ast, inner)?)
            }
            Expression::Dereference(inner) => {
                Expression::Dereference(self.expression(ast, inner)?)
            }
            Expression::Try(inner) => {
                Expression::Try(self.expression(ast, inner)?)
            }
            Expression::Unsafe(body) => {
                Expression::Unsafe(self.block(ast, body)?)
            }
            Expression::UnsafeFn(inner) => {
                Expression::UnsafeFn(self.expression(ast, inner)?)
            }
            Expression::StructInit(name, fields) => {
                let entries: Vec<NamedExpr> = ast.named_in(fields).to_vec();
                let mut expanded_fields = Vec::with_capacity(entries.len());
                for entry in entries {
                    expanded_fields.push(NamedExpr {
                        name: entry.name,
                        value: self.expression(ast, entry.value)?,
                    });
                }
                Expression::StructInit(
                    name,
                    ast.add_named_exprs(&expanded_fields),
                )
            }
            Expression::EnumVariantInit(name, variant, fields) => {
                let entries: Vec<NamedExpr> = ast.named_in(fields).to_vec();
                let mut expanded_fields = Vec::with_capacity(entries.len());
                for entry in entries {
                    expanded_fields.push(NamedExpr {
                        name: entry.name,
                        value: self.expression(ast, entry.value)?,
                    });
                }
                Expression::EnumVariantInit(
                    name,
                    variant,
                    ast.add_named_exprs(&expanded_fields),
                )
            }
            Expression::Tuple(items) => {
                let item_ids: Vec<ExprId> = ast.exprs_in(items).to_vec();
                let mut expanded_items = Vec::with_capacity(item_ids.len());
                for item in item_ids {
                    expanded_items.push(self.expression(ast, item)?);
                }
                Expression::Tuple(ast.add_expr_list(&expanded_items))
            }
            Expression::Range(start, end, inclusive) => {
                let start = self.expression(ast, start)?;
                let end = self.expression(ast, end)?;
                Expression::Range(start, end, inclusive)
            }
            Expression::Switch(scrutinee, cases) => {
                let case_entries: Vec<SwitchCase> =
                    ast.cases_in(cases).to_vec();
                let mut expanded_cases = Vec::with_capacity(case_entries.len());
                for case in case_entries {
                    expanded_cases.push(SwitchCase {
                        pattern: case.pattern,
                        body: self.block(ast, case.body)?,
                    });
                }
                let scrutinee = self.expression(ast, scrutinee)?;
                Expression::Switch(scrutinee, ast.add_cases(&expanded_cases))
            }
            Expression::ArrayRepeat(value, count) => {
                Expression::ArrayRepeat(self.expression(ast, value)?, count)
            }
            Expression::Literal(Literal::Array(elements)) => {
                let element_ids: Vec<ExprId> = ast.exprs_in(elements).to_vec();
                let mut expanded_elements =
                    Vec::with_capacity(element_ids.len());
                for element in element_ids {
                    expanded_elements.push(self.expression(ast, element)?);
                }
                Expression::Literal(Literal::Array(
                    ast.add_expr_list(&expanded_elements),
                ))
            }
            _ => return Ok(expression),
        };
        Ok(ast.push_expr(expanded, span))
    }
}

// One name for another, through a block. This is what binds a `for` variable to
// the element the copy is for.
fn substitute_identifier(
    ast: &mut Ast,
    block: Range32,
    from: &str,
    to: &str,
) -> Range32 {
    let mut subst = HashMap::new();
    subst.insert(from.to_string(), to.to_string());
    rename_block(ast, block, &subst)
}

fn substitute_identifier_in_expression(
    ast: &mut Ast,
    expression: ExprId,
    from: &str,
    to: &str,
) -> ExprId {
    let mut subst = HashMap::new();
    subst.insert(from.to_string(), to.to_string());
    rename_expression(ast, expression, &subst)
}

fn rename_block(
    ast: &mut Ast,
    block: Range32,
    subst: &HashMap<String, String>,
) -> Range32 {
    let statements: Vec<StmtId> = ast.stmts_in(block).to_vec();
    let mut renamed = Vec::with_capacity(statements.len());
    for statement in statements {
        renamed.push(rename_statement(ast, statement, subst));
    }
    ast.add_stmt_list(&renamed)
}

fn rename_statement(
    ast: &mut Ast,
    statement: StmtId,
    subst: &HashMap<String, String>,
) -> StmtId {
    let span = ast.stmt_span(statement);
    let renamed = match ast.stmt(statement).clone() {
        Statement::Let {
            name,
            type_annotation,
            value,
            mutable,
        } => Statement::Let {
            name,
            type_annotation,
            value: rename_expression(ast, value, subst),
            mutable,
        },
        Statement::Constant(name, value) => {
            Statement::Constant(name, rename_expression(ast, value, subst))
        }
        Statement::Return(value) => {
            Statement::Return(rename_expression(ast, value, subst))
        }
        Statement::Expression(value) => {
            Statement::Expression(rename_expression(ast, value, subst))
        }
        Statement::Assignment(place, value) => Statement::Assignment(
            rename_expression(ast, place, subst),
            rename_expression(ast, value, subst),
        ),
        Statement::Defer(inner) => {
            Statement::Defer(rename_statement(ast, inner, subst))
        }
        Statement::ErrDefer(inner) => {
            Statement::ErrDefer(rename_statement(ast, inner, subst))
        }
        Statement::For(variable, second, iterable, body) => Statement::For(
            variable,
            second,
            rename_expression(ast, iterable, subst),
            rename_block(ast, body, subst),
        ),
        Statement::While(condition, body) => Statement::While(
            rename_expression(ast, condition, subst),
            rename_block(ast, body, subst),
        ),
        Statement::With(capability, body) => {
            Statement::With(capability, rename_block(ast, body, subst))
        }
        _ => return statement,
    };
    ast.push_stmt(renamed, span)
}

fn rename_expression(
    ast: &mut Ast,
    expression: ExprId,
    subst: &HashMap<String, String>,
) -> ExprId {
    let span = ast.expr_span(expression);
    let renamed = match ast.expr(expression).clone() {
        Expression::Identifier(name) => match subst.get(ast.name(name)) {
            Some(renamed) => {
                let renamed = renamed.clone();
                let symbol = ast.intern(&renamed);
                Expression::Identifier(symbol)
            }
            None => return expression,
        },
        Expression::Prefix(operator, inner) => {
            Expression::Prefix(operator, rename_expression(ast, inner, subst))
        }
        Expression::Infix(left, operator, right) => {
            let left = rename_expression(ast, left, subst);
            let right = rename_expression(ast, right, subst);
            Expression::Infix(left, operator, right)
        }
        Expression::If(condition, consequence, alternative) => Expression::If(
            rename_expression(ast, condition, subst),
            rename_block(ast, consequence, subst),
            alternative.map(|block| rename_block(ast, block, subst)),
        ),
        Expression::Call(callee, arguments) => {
            let callee = rename_expression(ast, callee, subst);
            let argument_ids: Vec<ExprId> = ast.exprs_in(arguments).to_vec();
            let mut renamed_arguments = Vec::with_capacity(argument_ids.len());
            for argument in argument_ids {
                renamed_arguments.push(rename_expression(ast, argument, subst));
            }
            Expression::Call(callee, ast.add_expr_list(&renamed_arguments))
        }
        Expression::Index(base, index) => {
            let base = rename_expression(ast, base, subst);
            let index = rename_expression(ast, index, subst);
            Expression::Index(base, index)
        }
        Expression::FieldAccess(base, field) => {
            Expression::FieldAccess(rename_expression(ast, base, subst), field)
        }
        Expression::AddressOf(inner) => {
            Expression::AddressOf(rename_expression(ast, inner, subst))
        }
        Expression::Borrow(inner) => {
            Expression::Borrow(rename_expression(ast, inner, subst))
        }
        Expression::BorrowMut(inner) => {
            Expression::BorrowMut(rename_expression(ast, inner, subst))
        }
        Expression::Dereference(inner) => {
            Expression::Dereference(rename_expression(ast, inner, subst))
        }
        Expression::Try(inner) => {
            Expression::Try(rename_expression(ast, inner, subst))
        }
        Expression::Unsafe(body) => {
            Expression::Unsafe(rename_block(ast, body, subst))
        }
        Expression::UnsafeFn(inner) => {
            Expression::UnsafeFn(rename_expression(ast, inner, subst))
        }
        Expression::StructInit(name, fields) => {
            let entries: Vec<NamedExpr> = ast.named_in(fields).to_vec();
            let mut renamed_fields = Vec::with_capacity(entries.len());
            for entry in entries {
                renamed_fields.push(NamedExpr {
                    name: entry.name,
                    value: rename_expression(ast, entry.value, subst),
                });
            }
            Expression::StructInit(name, ast.add_named_exprs(&renamed_fields))
        }
        Expression::EnumVariantInit(name, variant, fields) => {
            let entries: Vec<NamedExpr> = ast.named_in(fields).to_vec();
            let mut renamed_fields = Vec::with_capacity(entries.len());
            for entry in entries {
                renamed_fields.push(NamedExpr {
                    name: entry.name,
                    value: rename_expression(ast, entry.value, subst),
                });
            }
            Expression::EnumVariantInit(
                name,
                variant,
                ast.add_named_exprs(&renamed_fields),
            )
        }
        Expression::Tuple(items) => {
            let item_ids: Vec<ExprId> = ast.exprs_in(items).to_vec();
            let mut renamed_items = Vec::with_capacity(item_ids.len());
            for item in item_ids {
                renamed_items.push(rename_expression(ast, item, subst));
            }
            Expression::Tuple(ast.add_expr_list(&renamed_items))
        }
        Expression::Range(start, end, inclusive) => {
            let start = rename_expression(ast, start, subst);
            let end = rename_expression(ast, end, subst);
            Expression::Range(start, end, inclusive)
        }
        Expression::Switch(scrutinee, cases) => {
            let scrutinee = rename_expression(ast, scrutinee, subst);
            let case_entries: Vec<SwitchCase> = ast.cases_in(cases).to_vec();
            let mut renamed_cases = Vec::with_capacity(case_entries.len());
            for case in case_entries {
                renamed_cases.push(SwitchCase {
                    pattern: case.pattern,
                    body: rename_block(ast, case.body, subst),
                });
            }
            Expression::Switch(scrutinee, ast.add_cases(&renamed_cases))
        }
        Expression::ArrayRepeat(value, count) => {
            Expression::ArrayRepeat(rename_expression(ast, value, subst), count)
        }
        Expression::Literal(Literal::Array(elements)) => {
            let element_ids: Vec<ExprId> = ast.exprs_in(elements).to_vec();
            let mut renamed_elements = Vec::with_capacity(element_ids.len());
            for element in element_ids {
                renamed_elements.push(rename_expression(ast, element, subst));
            }
            Expression::Literal(Literal::Array(
                ast.add_expr_list(&renamed_elements),
            ))
        }
        _ => return expression,
    };
    ast.push_expr(renamed, span)
}

fn is_generic_instance(name: &str) -> bool {
    name.contains('<')
}

fn split_instance(name: &str) -> Option<(String, Vec<String>)> {
    let open = name.find('<')?;
    if !name.ends_with('>') {
        return None;
    }
    let base = name[..open].to_string();
    let inner = &name[open + 1..name.len() - 1];
    let mut arguments = Vec::new();
    let mut depth = 0usize;
    let mut current = String::new();
    for character in inner.chars() {
        match character {
            '<' => {
                depth += 1;
                current.push(character);
            }
            '>' => {
                depth -= 1;
                current.push(character);
            }
            ',' if depth == 0 => {
                arguments.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(character),
        }
    }
    if !current.trim().is_empty() {
        arguments.push(current.trim().to_string());
    }
    Some((base, arguments))
}

fn collect_instances_in_type(ty: &Type, out: &mut Vec<String>) {
    if let Type::Struct(name) = ty
        && is_generic_instance(name)
        && !out.contains(name)
    {
        out.push(name.clone());
    }
    if let Some(inner) = single_inner(ty) {
        collect_instances_in_type(inner, out);
    } else if let Type::Proc(params, ret) = ty {
        for param in params {
            collect_instances_in_type(param, out);
        }
        collect_instances_in_type(ret, out);
    }
}

fn collect_instances_in_block(
    ast: &Ast,
    block: Range32,
    out: &mut Vec<String>,
) {
    for statement in ast.stmts_in(block) {
        collect_instances_in_statement(ast, *statement, out);
    }
}

fn collect_instances_in_statement(
    ast: &Ast,
    statement: StmtId,
    out: &mut Vec<String>,
) {
    match ast.stmt(statement) {
        Statement::Let {
            type_annotation,
            value,
            ..
        } => {
            if let Some(ty) = type_annotation {
                collect_instances_in_type(ty, out);
            }
            collect_instances_in_expression(ast, *value, out);
        }
        Statement::Return(expression) | Statement::Expression(expression) => {
            collect_instances_in_expression(ast, *expression, out);
        }
        Statement::Assignment(target, value) => {
            collect_instances_in_expression(ast, *target, out);
            collect_instances_in_expression(ast, *value, out);
        }
        Statement::For(_, _, range, body) => {
            collect_instances_in_expression(ast, *range, out);
            collect_instances_in_block(ast, *body, out);
        }
        Statement::While(condition, body) => {
            collect_instances_in_expression(ast, *condition, out);
            collect_instances_in_block(ast, *body, out);
        }
        Statement::Defer(inner) | Statement::ErrDefer(inner) => {
            collect_instances_in_statement(ast, *inner, out);
        }
        // A constant whose value is not a function is that value wherever it is
        // named, so an instance it builds is asked for here. A function
        // constant's body is walked with its parameter types in scope instead.
        Statement::Constant(_, value)
            if !matches!(
                ast.expr(*value),
                Expression::Function(..) | Expression::Proc(..)
            ) =>
        {
            collect_instances_in_expression(ast, *value, out);
        }
        _ => {}
    }
}

fn collect_instances_in_expression(
    ast: &Ast,
    expression: ExprId,
    out: &mut Vec<String>,
) {
    match ast.expr(expression) {
        Expression::TypeValue(ty) => collect_instances_in_type(ty, out),
        Expression::Prefix(_, operand)
        | Expression::AddressOf(operand)
        | Expression::Borrow(operand)
        | Expression::BorrowMut(operand)
        | Expression::Dereference(operand) => {
            collect_instances_in_expression(ast, *operand, out);
        }
        Expression::Infix(left, _, right) => {
            collect_instances_in_expression(ast, *left, out);
            collect_instances_in_expression(ast, *right, out);
        }
        Expression::If(condition, consequence, alternative) => {
            collect_instances_in_expression(ast, *condition, out);
            collect_instances_in_block(ast, *consequence, out);
            if let Some(block) = alternative {
                collect_instances_in_block(ast, *block, out);
            }
        }
        Expression::Call(callee, arguments) => {
            collect_instances_in_expression(ast, *callee, out);
            for argument in ast.exprs_in(*arguments) {
                collect_instances_in_expression(ast, *argument, out);
            }
        }
        Expression::Index(base, index) => {
            collect_instances_in_expression(ast, *base, out);
            collect_instances_in_expression(ast, *index, out);
        }
        Expression::FieldAccess(base, _) => {
            collect_instances_in_expression(ast, *base, out);
        }
        // A literal that says which instance it is asks for that instance, the
        // same as a type annotation naming one. Without this the only literals
        // that reached an instance were the ones a name elsewhere had already
        // built.
        Expression::StructInit(name, fields)
            if is_generic_instance(ast.name(*name))
                && !out.iter().any(|held| held == ast.name(*name)) =>
        {
            out.push(ast.name(*name).to_string());
            for field in ast.named_in(*fields) {
                collect_instances_in_expression(ast, field.value, out);
            }
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => {
            for field in ast.named_in(*fields) {
                collect_instances_in_expression(ast, field.value, out);
            }
        }
        Expression::Range(start, end, _) => {
            collect_instances_in_expression(ast, *start, out);
            collect_instances_in_expression(ast, *end, out);
        }
        Expression::Tuple(elements) => {
            for element in ast.exprs_in(*elements) {
                collect_instances_in_expression(ast, *element, out);
            }
        }
        Expression::Switch(scrutinee, cases) => {
            collect_instances_in_expression(ast, *scrutinee, out);
            for case in ast.cases_in(*cases) {
                collect_instances_in_block(ast, case.body, out);
            }
        }
        _ => {}
    }
}

struct Discovery<'a> {
    functions: &'a HashMap<String, GenericFunction>,
    structs: &'a GenericStructDefs,
}

fn infer_struct_instance_shallow(
    ast: &Ast,
    struct_name: &str,
    field_inits: Range32,
    env: &HashMap<String, Type>,
    discovery: &Discovery,
) -> Option<String> {
    let (type_params, fields) = discovery.structs.get(struct_name)?;
    let mut subst: HashMap<String, Type> = HashMap::new();
    for entry in ast.named_in(field_inits) {
        if let Some((_, field_type)) = fields
            .iter()
            .find(|(field_name, _)| field_name == ast.name(entry.name))
            && let Some(value_type) =
                infer_expr_type_shallow(ast, entry.value, env, discovery)
        {
            infer_subst_into(field_type, &value_type, type_params, &mut subst);
        }
    }
    let rendered: Vec<String> = type_params
        .iter()
        .map(|type_param| {
            subst
                .get(type_param)
                .map(|ty| ty.to_string())
                .unwrap_or_else(|| type_param.clone())
        })
        .collect();
    Some(format!("{struct_name}<{}>", rendered.join(", ")))
}

fn infer_expr_type_shallow(
    ast: &Ast,
    expression: ExprId,
    env: &HashMap<String, Type>,
    discovery: &Discovery,
) -> Option<Type> {
    match ast.expr(expression) {
        Expression::Literal(Literal::Integer(_)) => Some(Type::I64),
        Expression::Literal(Literal::Float(_)) => Some(Type::F64),
        Expression::Literal(Literal::Float32(_)) => Some(Type::F32),
        Expression::Boolean(_) | Expression::Literal(Literal::Boolean(_)) => {
            Some(Type::Bool)
        }
        Expression::Identifier(name) => env.get(ast.name(*name)).cloned(),
        Expression::StructInit(name, fields) => {
            let name = ast.name(*name);
            if discovery.structs.contains_key(name) {
                infer_struct_instance_shallow(
                    ast, name, *fields, env, discovery,
                )
                .map(Type::Struct)
            } else {
                Some(Type::Struct(name.to_string()))
            }
        }
        Expression::EnumVariantInit(name, _, _) => {
            Some(Type::Enum(ast.name(*name).to_string()))
        }
        Expression::Borrow(inner) => {
            infer_expr_type_shallow(ast, *inner, env, discovery)
                .map(|inner| Type::Ref(Box::new(inner)))
        }
        Expression::BorrowMut(inner) => {
            infer_expr_type_shallow(ast, *inner, env, discovery)
                .map(|inner| Type::RefMut(Box::new(inner)))
        }
        Expression::Call(callee, arguments) => {
            let Expression::Identifier(name) = ast.expr(*callee) else {
                return None;
            };
            let generic = discovery.functions.get(ast.name(*name))?;
            let subst = infer_call_subst(
                ast,
                generic,
                ast.exprs_in(*arguments),
                env,
                discovery,
            );
            ast.signature_to_type(ast.signature(generic.return_sig))
                .map(|ty| substitute_type(&ty, &subst))
        }
        _ => None,
    }
}

fn infer_call_subst(
    ast: &Ast,
    generic: &GenericFunction,
    arguments: &[ExprId],
    env: &HashMap<String, Type>,
    discovery: &Discovery,
) -> HashMap<String, Type> {
    let mut subst = HashMap::new();
    let parameters = ast.params_in(generic.parameters);
    for (parameter, slot) in
        parameters.iter().zip(argument_slots(ast, parameters))
    {
        // A compile-time parameter a value parameter settles takes no argument
        // of its own. It is bound when that value parameter is walked, out of
        // the type of what was handed to it.
        let Some(argument) = slot.and_then(|slot| arguments.get(slot)) else {
            continue;
        };
        if is_type_parameter(ast, parameter)
            && let Expression::TypeValue(ty) = ast.expr(*argument)
        {
            subst.insert(ast.name(parameter.name).to_string(), ty.clone());
            continue;
        }
        if let Some(argument_type) =
            infer_expr_type_shallow(ast, *argument, env, discovery)
        {
            infer_subst_into(
                &parameter_type(parameter),
                &argument_type,
                &generic.type_params,
                &mut subst,
            );
        }
    }
    subst
}

fn collect_call_instances_in_block(
    ast: &Ast,
    block: Range32,
    env: &mut HashMap<String, Type>,
    discovery: &Discovery,
    out: &mut Vec<String>,
) {
    for statement in ast.stmts_in(block) {
        collect_call_instances_in_statement(
            ast, *statement, env, discovery, out,
        );
    }
}

fn collect_call_instances_in_statement(
    ast: &Ast,
    statement: StmtId,
    env: &mut HashMap<String, Type>,
    discovery: &Discovery,
    out: &mut Vec<String>,
) {
    match ast.stmt(statement) {
        Statement::Let {
            name,
            type_annotation,
            value,
            ..
        } => {
            collect_call_instances_in_expression(
                ast, *value, env, discovery, out,
            );
            let inferred = type_annotation.clone().or_else(|| {
                infer_expr_type_shallow(ast, *value, env, discovery)
            });
            if let Some(ty) = inferred {
                env.insert(ast.name(*name).to_string(), ty);
            }
        }
        Statement::Return(expression) | Statement::Expression(expression) => {
            collect_call_instances_in_expression(
                ast,
                *expression,
                env,
                discovery,
                out,
            );
        }
        Statement::Assignment(target, value) => {
            collect_call_instances_in_expression(
                ast, *target, env, discovery, out,
            );
            collect_call_instances_in_expression(
                ast, *value, env, discovery, out,
            );
        }
        Statement::For(variable, _, range, body) => {
            collect_call_instances_in_expression(
                ast, *range, env, discovery, out,
            );
            env.insert(ast.name(*variable).to_string(), Type::I64);
            collect_call_instances_in_block(ast, *body, env, discovery, out);
        }
        Statement::While(condition, body) => {
            collect_call_instances_in_expression(
                ast, *condition, env, discovery, out,
            );
            collect_call_instances_in_block(ast, *body, env, discovery, out);
        }
        Statement::Defer(inner) | Statement::ErrDefer(inner) => {
            collect_call_instances_in_statement(
                ast, *inner, env, discovery, out,
            );
        }
        _ => {}
    }
}

fn collect_call_instances_in_expression(
    ast: &Ast,
    expression: ExprId,
    env: &mut HashMap<String, Type>,
    discovery: &Discovery,
    out: &mut Vec<String>,
) {
    match ast.expr(expression) {
        Expression::Call(callee, arguments) => {
            if let Expression::Identifier(name) = ast.expr(*callee)
                && let Some(generic) = discovery.functions.get(ast.name(*name))
            {
                let subst = infer_call_subst(
                    ast,
                    generic,
                    ast.exprs_in(*arguments),
                    env,
                    discovery,
                );
                if let Some(return_type) =
                    ast.signature_to_type(ast.signature(generic.return_sig))
                {
                    collect_instances_in_type(
                        &substitute_type(&return_type, &subst),
                        out,
                    );
                }
                for parameter in ast.params_in(generic.parameters) {
                    collect_instances_in_type(
                        &substitute_type(&parameter_type(parameter), &subst),
                        out,
                    );
                }
            }
            collect_call_instances_in_expression(
                ast, *callee, env, discovery, out,
            );
            for argument in ast.exprs_in(*arguments) {
                collect_call_instances_in_expression(
                    ast, *argument, env, discovery, out,
                );
            }
        }
        Expression::StructInit(name, fields) => {
            if discovery.structs.contains_key(ast.name(*name))
                && let Some(instance) = infer_struct_instance_shallow(
                    ast,
                    ast.name(*name),
                    *fields,
                    env,
                    discovery,
                )
            {
                out.push(instance);
            }
            for field in ast.named_in(*fields) {
                collect_call_instances_in_expression(
                    ast,
                    field.value,
                    env,
                    discovery,
                    out,
                );
            }
        }
        Expression::Prefix(_, operand)
        | Expression::AddressOf(operand)
        | Expression::Borrow(operand)
        | Expression::BorrowMut(operand)
        | Expression::Dereference(operand) => {
            collect_call_instances_in_expression(
                ast, *operand, env, discovery, out,
            );
        }
        Expression::Infix(left, _, right) => {
            collect_call_instances_in_expression(
                ast, *left, env, discovery, out,
            );
            collect_call_instances_in_expression(
                ast, *right, env, discovery, out,
            );
        }
        Expression::If(condition, consequence, alternative) => {
            collect_call_instances_in_expression(
                ast, *condition, env, discovery, out,
            );
            let mut branch_env = env.clone();
            collect_call_instances_in_block(
                ast,
                *consequence,
                &mut branch_env,
                discovery,
                out,
            );
            if let Some(block) = alternative {
                let mut branch_env = env.clone();
                collect_call_instances_in_block(
                    ast,
                    *block,
                    &mut branch_env,
                    discovery,
                    out,
                );
            }
        }
        Expression::Index(base, index) => {
            collect_call_instances_in_expression(
                ast, *base, env, discovery, out,
            );
            collect_call_instances_in_expression(
                ast, *index, env, discovery, out,
            );
        }
        Expression::FieldAccess(base, _) => {
            collect_call_instances_in_expression(
                ast, *base, env, discovery, out,
            );
        }
        Expression::EnumVariantInit(_, _, fields) => {
            for field in ast.named_in(*fields) {
                collect_call_instances_in_expression(
                    ast,
                    field.value,
                    env,
                    discovery,
                    out,
                );
            }
        }
        Expression::Switch(scrutinee, cases) => {
            collect_call_instances_in_expression(
                ast, *scrutinee, env, discovery, out,
            );
            for case in ast.cases_in(*cases) {
                let mut branch_env = env.clone();
                collect_call_instances_in_block(
                    ast,
                    case.body,
                    &mut branch_env,
                    discovery,
                    out,
                );
            }
        }
        _ => {}
    }
}

fn expand_generic_structs(
    ast: &mut Ast,
    roots: &[StmtId],
) -> Result<Vec<StmtId>> {
    let mut generic_structs: GenericStructDefs = HashMap::new();
    let mut generic_enums: GenericEnumDefs = HashMap::new();
    for statement in roots {
        if let Statement::Struct(name, type_params, fields) =
            ast.stmt(*statement)
            && !type_params.is_empty()
        {
            let params: Vec<String> = ast
                .symbols_in(*type_params)
                .iter()
                .map(|param| ast.name(*param).to_string())
                .collect();
            let fields: Vec<(String, Type)> = ast
                .fields_in(*fields)
                .iter()
                .map(|field| {
                    (ast.name(field.name).to_string(), field.field_type.clone())
                })
                .collect();
            generic_structs
                .insert(ast.name(*name).to_string(), (params, fields));
        }
        if let Statement::Enum(name, type_params, variants) =
            ast.stmt(*statement)
            && !type_params.is_empty()
        {
            let params: Vec<String> = ast
                .symbols_in(*type_params)
                .iter()
                .map(|param| ast.name(*param).to_string())
                .collect();
            let variants: GenericEnumVariants = ast
                .variants_in(*variants)
                .iter()
                .map(|variant| {
                    (
                        ast.name(variant.name).to_string(),
                        variant.fields.map(|fields| {
                            ast.fields_in(fields)
                                .iter()
                                .map(|field| {
                                    (
                                        ast.name(field.name).to_string(),
                                        field.field_type.clone(),
                                    )
                                })
                                .collect()
                        }),
                    )
                })
                .collect();
            generic_enums
                .insert(ast.name(*name).to_string(), (params, variants));
        }
    }
    // No early return on empty templates. A `columns<T, N>` is synthesized here
    // too and needs no user template, so the instance walk below must still run.

    let mut generic_functions: HashMap<String, GenericFunction> =
        HashMap::new();
    for statement in roots {
        if let Statement::Constant(name, value) = ast.stmt(*statement)
            && let Expression::Function(parameters, return_sig, body)
            | Expression::Proc(parameters, return_sig, body) =
                ast.expr(*value)
            && function_is_generic(ast, *parameters)
        {
            generic_functions.insert(
                ast.name(*name).to_string(),
                GenericFunction {
                    type_params: function_type_params(ast, *parameters),
                    parameters: *parameters,
                    return_sig: *return_sig,
                    body: *body,
                },
            );
        }
    }

    let discovery = Discovery {
        functions: &generic_functions,
        structs: &generic_structs,
    };
    let mut queue: Vec<String> = Vec::new();
    for statement in roots {
        if let Statement::Constant(_, value) = ast.stmt(*statement)
            && let Expression::Function(parameters, _, body)
            | Expression::Proc(parameters, _, body) = ast.expr(*value)
        {
            let mut env: HashMap<String, Type> = HashMap::new();
            for parameter in ast.params_in(*parameters) {
                if let Some(ty) = &parameter.type_annotation {
                    env.insert(
                        ast.name(parameter.name).to_string(),
                        ty.clone(),
                    );
                }
            }
            collect_call_instances_in_block(
                ast, *body, &mut env, &discovery, &mut queue,
            );
        }
        collect_instances_in_statement(ast, *statement, &mut queue);
        if let Statement::Struct(_, _, fields) = ast.stmt(*statement) {
            for field in ast.fields_in(*fields) {
                collect_instances_in_type(&field.field_type, &mut queue);
            }
        }
        if let Statement::Enum(_, _, variants) = ast.stmt(*statement) {
            for variant in ast.variants_in(*variants) {
                if let Some(fields) = variant.fields {
                    for field in ast.fields_in(fields) {
                        collect_instances_in_type(
                            &field.field_type,
                            &mut queue,
                        );
                    }
                }
            }
        }
        if let Statement::Extern {
            params,
            return_type,
            ..
        } = ast.stmt(*statement)
        {
            for parameter in ast.params_in(*params) {
                if let Some(ty) = &parameter.type_annotation {
                    collect_instances_in_type(ty, &mut queue);
                }
            }
            if let Some(ty) = return_type {
                collect_instances_in_type(ty, &mut queue);
            }
        }
        if let Statement::Constant(_, value) = ast.stmt(*statement)
            && let Expression::Function(parameters, return_sig, body)
            | Expression::Proc(parameters, return_sig, body) =
                ast.expr(*value)
        {
            for parameter in ast.params_in(*parameters) {
                if let Some(ty) = &parameter.type_annotation {
                    collect_instances_in_type(ty, &mut queue);
                }
            }
            if let Some(ty) = ast.signature_to_type(ast.signature(*return_sig))
            {
                collect_instances_in_type(&ty, &mut queue);
            }
            collect_instances_in_block(ast, *body, &mut queue);
        }
    }

    // The non-generic struct definitions, so a `columns<T, N>` can reflect over
    // T's fields to synthesize one array per field. T is required to be a plain
    // struct, the same restriction the self-hosted compiler has.
    let concrete_structs: HashMap<String, Vec<(String, Type)>> = roots
        .iter()
        .filter_map(|statement| match ast.stmt(*statement) {
            Statement::Struct(name, type_params, fields)
                if type_params.is_empty() =>
            {
                Some((
                    ast.name(*name).to_string(),
                    ast.fields_in(*fields)
                        .iter()
                        .map(|field| {
                            (
                                ast.name(field.name).to_string(),
                                field.field_type.clone(),
                            )
                        })
                        .collect(),
                ))
            }
            _ => None,
        })
        .collect();

    let mut done: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let mut synthetic = Vec::new();
    while let Some(instance) = queue.pop() {
        if !done.insert(instance.clone()) {
            continue;
        }
        let Some((base, argument_strings)) = split_instance(&instance) else {
            continue;
        };
        if let Some((type_params, variants)) = generic_enums.get(&base) {
            let subst = instance_substitution(
                &base,
                type_params,
                &argument_strings,
                &instance,
                "enum",
            )?;
            let concrete_variants: GenericEnumVariants = variants
                .iter()
                .map(|(variant_name, fields)| {
                    (
                        variant_name.clone(),
                        fields.as_ref().map(|fields| {
                            fields
                                .iter()
                                .map(|(field_name, field_type)| {
                                    (
                                        field_name.clone(),
                                        substitute_type(field_type, &subst),
                                    )
                                })
                                .collect()
                        }),
                    )
                })
                .collect();
            for (_, fields) in &concrete_variants {
                if let Some(fields) = fields {
                    for (_, field_type) in fields {
                        collect_instances_in_type(field_type, &mut queue);
                    }
                }
            }
            let variant_entries: Vec<EnumVariant> = concrete_variants
                .iter()
                .map(|(variant_name, fields)| {
                    let name = ast.intern(variant_name);
                    let fields = fields.as_ref().map(|fields| {
                        let entries: Vec<StructField> = fields
                            .iter()
                            .map(|(field_name, field_type)| StructField {
                                name: ast.intern(field_name),
                                field_type: field_type.clone(),
                                align: None,
                            })
                            .collect();
                        ast.add_struct_fields(entries)
                    });
                    EnumVariant { name, fields }
                })
                .collect();
            let name = ast.intern(&instance);
            let variants = ast.add_enum_variants(&variant_entries);
            synthetic.push(ast.push_stmt(
                Statement::Enum(name, Range32::EMPTY, variants),
                TokenSpan::NONE,
            ));
            continue;
        }
        if base == "columns" {
            // `columns<T, N>` is the SoA container: one `[N]field` array per
            // field of T (named after the field) plus the generational
            // bookkeeping a slab carries. The layout cannot be written in
            // library Frost, so it is reflected from T's fields here. Only a
            // CONCRETE instance is synthesized. The generic template form
            // `columns<T, N>` in a library signature is skipped, since it is
            // monomorphized to a concrete instance where it is used.
            if argument_strings.len() != 2 {
                continue;
            }
            let Ok(count) = argument_strings[1].trim().parse::<usize>() else {
                continue;
            };
            let Ok(element) =
                crate::parser::type_from_string(&argument_strings[0])
            else {
                continue;
            };
            let Type::Struct(element_name) = &element else {
                continue;
            };
            let Some(element_fields) = concrete_structs.get(element_name)
            else {
                continue;
            };
            let mut columns_fields: Vec<(String, Type)> = element_fields
                .iter()
                .map(|(field_name, field_type)| {
                    (
                        field_name.clone(),
                        Type::Array(Box::new(field_type.clone()), count),
                    )
                })
                .collect();
            columns_fields.push((
                "generations".to_string(),
                Type::Array(Box::new(Type::I64), count),
            ));
            columns_fields.push((
                "free_list".to_string(),
                Type::Array(Box::new(Type::I64), count),
            ));
            columns_fields.push(("free_count".to_string(), Type::I64));
            // Which slots hold an element, one bit each, so `for i in live_slots(c)`
            // walks them in slot order and skips sixty-four dead slots at a
            // time. The free list says which slots are free but not in an order
            // that can be walked, and a generation of zero is a slot that was
            // never filled as much as one that is. Appended, so every column
            // above keeps the offset it had.
            columns_fields.push((
                LIVE_WORDS.to_string(),
                Type::Array(Box::new(Type::I64), live_word_count(count)),
            ));
            columns_fields.push((LIVE_COUNT.to_string(), Type::I64));
            for (_, field_type) in &columns_fields {
                collect_instances_in_type(field_type, &mut queue);
            }
            let entries: Vec<StructField> = columns_fields
                .iter()
                .map(|(field_name, field_type)| StructField {
                    name: ast.intern(field_name),
                    field_type: field_type.clone(),
                    align: None,
                })
                .collect();
            let name = ast.intern(&instance);
            let fields = ast.add_struct_fields(entries);
            synthetic.push(ast.push_stmt(
                Statement::Struct(name, Range32::EMPTY, fields),
                TokenSpan::NONE,
            ));
            continue;
        }
        let Some((type_params, fields)) = generic_structs.get(&base) else {
            continue;
        };
        let subst = instance_substitution(
            &base,
            type_params,
            &argument_strings,
            &instance,
            "struct",
        )?;
        let concrete_fields: Vec<(String, Type)> = fields
            .iter()
            .map(|(field_name, field_type)| {
                (field_name.clone(), substitute_type(field_type, &subst))
            })
            .collect();
        for (_, field_type) in &concrete_fields {
            collect_instances_in_type(field_type, &mut queue);
        }
        let entries: Vec<StructField> = concrete_fields
            .iter()
            .map(|(field_name, field_type)| StructField {
                name: ast.intern(field_name),
                field_type: field_type.clone(),
                align: None,
            })
            .collect();
        let name = ast.intern(&instance);
        let fields = ast.add_struct_fields(entries);
        synthetic.push(ast.push_stmt(
            Statement::Struct(name, Range32::EMPTY, fields),
            TokenSpan::NONE,
        ));
    }
    Ok(synthetic)
}

// What each type parameter of a generic declaration is bound to for one
// instance, from the argument names parsed out of `Name<A, B>`.
fn instance_substitution(
    base: &str,
    type_params: &[String],
    argument_strings: &[String],
    instance: &str,
    kind: &str,
) -> Result<HashMap<String, Type>> {
    if type_params.len() != argument_strings.len() {
        bail!(
            "generic {kind} '{base}' expects {} type argument(s) but {} were given",
            type_params.len(),
            argument_strings.len()
        );
    }
    let mut subst = HashMap::new();
    for (type_param, argument) in type_params.iter().zip(argument_strings) {
        let argument_type = crate::parser::type_from_string(argument)
            .with_context(|| {
                format!("type argument '{argument}' of '{instance}'")
            })?;
        subst.insert(type_param.clone(), argument_type);
    }
    Ok(subst)
}

fn substitute_block(
    ast: &mut Ast,
    block: Range32,
    subst: &HashMap<String, Type>,
) -> Range32 {
    let statements: Vec<StmtId> = ast.stmts_in(block).to_vec();
    let mut copied = Vec::with_capacity(statements.len());
    for statement in statements {
        copied.push(substitute_statement(ast, statement, subst));
    }
    ast.add_stmt_list(&copied)
}

fn substitute_statement(
    ast: &mut Ast,
    statement: StmtId,
    subst: &HashMap<String, Type>,
) -> StmtId {
    let span = ast.stmt_span(statement);
    let substituted = match ast.stmt(statement).clone() {
        Statement::Let {
            name,
            type_annotation,
            value,
            mutable,
        } => Statement::Let {
            name,
            type_annotation: type_annotation
                .as_ref()
                .map(|ty| substitute_type(ty, subst)),
            value: substitute_expression(ast, value, subst),
            mutable,
        },
        Statement::Return(expression) => {
            Statement::Return(substitute_expression(ast, expression, subst))
        }
        Statement::Expression(expression) => {
            Statement::Expression(substitute_expression(ast, expression, subst))
        }
        Statement::Assignment(target, value) => Statement::Assignment(
            substitute_expression(ast, target, subst),
            substitute_expression(ast, value, subst),
        ),
        Statement::For(variable, second, range, body) => Statement::For(
            variable,
            second,
            substitute_expression(ast, range, subst),
            substitute_block(ast, body, subst),
        ),
        Statement::While(condition, body) => Statement::While(
            substitute_expression(ast, condition, subst),
            substitute_block(ast, body, subst),
        ),
        Statement::Defer(inner) => {
            Statement::Defer(substitute_statement(ast, inner, subst))
        }
        Statement::ErrDefer(inner) => {
            Statement::ErrDefer(substitute_statement(ast, inner, subst))
        }
        Statement::Constant(name, value) => {
            Statement::Constant(name, substitute_expression(ast, value, subst))
        }
        Statement::LetMultiple(bindings, value) => Statement::LetMultiple(
            bindings,
            substitute_expression(ast, value, subst),
        ),
        Statement::With(name, body) => {
            Statement::With(name, substitute_block(ast, body, subst))
        }
        _ => return statement,
    };
    ast.push_stmt(substituted, span)
}

fn substitute_expression(
    ast: &mut Ast,
    expression: ExprId,
    subst: &HashMap<String, Type>,
) -> ExprId {
    let span = ast.expr_span(expression);
    // A call through a compile-time function parameter is a call to the
    // function that parameter was given. There is nothing left to dispatch on
    // by the time the specialized body is lowered: the comparator ends up
    // inlined into the loop rather than called through a pointer.
    // A value parameter stands for its integer everywhere the body names it, not
    // only in a type. `while (i < N)` has to mean the capacity.
    if let Expression::Identifier(name) = ast.expr(expression)
        && let Some(Type::ConstUsize(value)) = subst.get(ast.name(*name))
    {
        return ast.push_expr(
            Expression::Literal(Literal::Integer(*value as i64)),
            span,
        );
    }
    if let Expression::Identifier(name) = ast.expr(expression)
        && let Some(Type::ConstValue(target)) = subst.get(ast.name(*name))
    {
        let target = target.clone();
        let symbol = ast.intern(&target);
        return ast.push_expr(Expression::Identifier(symbol), span);
    }
    if let Expression::Call(callee, arguments) = ast.expr(expression)
        && let Expression::Identifier(name) = ast.expr(*callee)
        && let Some(Type::ConstFn(target)) = subst.get(ast.name(*name))
    {
        let target = target.clone();
        let callee_span = ast.expr_span(*callee);
        let argument_ids: Vec<ExprId> = ast.exprs_in(*arguments).to_vec();
        let symbol = ast.intern(&target);
        let callee = ast.push_expr(Expression::Identifier(symbol), callee_span);
        let mut substituted = Vec::with_capacity(argument_ids.len());
        for argument in argument_ids {
            substituted.push(substitute_expression(ast, argument, subst));
        }
        let arguments = ast.add_expr_list(&substituted);
        return ast.push_expr(Expression::Call(callee, arguments), span);
    }
    let node = match ast.expr(expression).clone() {
        Expression::Prefix(operator, operand) => Expression::Prefix(
            operator,
            substitute_expression(ast, operand, subst),
        ),
        Expression::Infix(left, operator, right) => {
            let left = substitute_expression(ast, left, subst);
            let right = substitute_expression(ast, right, subst);
            Expression::Infix(left, operator, right)
        }
        Expression::If(condition, consequence, alternative) => Expression::If(
            substitute_expression(ast, condition, subst),
            substitute_block(ast, consequence, subst),
            alternative.map(|block| substitute_block(ast, block, subst)),
        ),
        Expression::Call(callee, arguments) => {
            let callee = substitute_expression(ast, callee, subst);
            let argument_ids: Vec<ExprId> = ast.exprs_in(arguments).to_vec();
            let mut substituted = Vec::with_capacity(argument_ids.len());
            for argument in argument_ids {
                substituted.push(substitute_expression(ast, argument, subst));
            }
            Expression::Call(callee, ast.add_expr_list(&substituted))
        }
        Expression::Index(base, index) => {
            let base = substitute_expression(ast, base, subst);
            let index = substitute_expression(ast, index, subst);
            Expression::Index(base, index)
        }
        Expression::FieldAccess(base, field) => Expression::FieldAccess(
            substitute_expression(ast, base, subst),
            field,
        ),
        Expression::AddressOf(inner) => {
            Expression::AddressOf(substitute_expression(ast, inner, subst))
        }
        Expression::Borrow(inner) => {
            Expression::Borrow(substitute_expression(ast, inner, subst))
        }
        Expression::BorrowMut(inner) => {
            Expression::BorrowMut(substitute_expression(ast, inner, subst))
        }
        Expression::Dereference(inner) => {
            Expression::Dereference(substitute_expression(ast, inner, subst))
        }
        Expression::StructInit(name, fields) => {
            let entries: Vec<NamedExpr> = ast.named_in(fields).to_vec();
            let mut substituted = Vec::with_capacity(entries.len());
            for entry in entries {
                substituted.push(NamedExpr {
                    name: entry.name,
                    value: substitute_expression(ast, entry.value, subst),
                });
            }
            Expression::StructInit(name, ast.add_named_exprs(&substituted))
        }
        Expression::EnumVariantInit(name, variant, fields) => {
            let entries: Vec<NamedExpr> = ast.named_in(fields).to_vec();
            let mut substituted = Vec::with_capacity(entries.len());
            for entry in entries {
                substituted.push(NamedExpr {
                    name: entry.name,
                    value: substitute_expression(ast, entry.value, subst),
                });
            }
            Expression::EnumVariantInit(
                name,
                variant,
                ast.add_named_exprs(&substituted),
            )
        }
        // `[value; N]` becomes the array it always meant, now that N is a
        // number. A count still unbound is one the enclosing generic passes on
        // to a further instantiation, so the form is carried along.
        Expression::ArrayRepeat(value, count) => {
            let value = substitute_expression(ast, value, subst);
            match subst.get(ast.name(count)) {
                Some(Type::ConstUsize(size)) => {
                    let elements = vec![value; *size];
                    Expression::Literal(Literal::Array(
                        ast.add_expr_list(&elements),
                    ))
                }
                _ => Expression::ArrayRepeat(value, count),
            }
        }
        // A compile-time argument handed on to another generic. Without this a
        // `$T` or a `$f` forwarded from one generic to the next arrived as the
        // parameter's own name rather than what it was bound to.
        Expression::TypeValue(ty) => {
            Expression::TypeValue(substitute_type(&ty, subst))
        }
        Expression::Range(start, end, inclusive) => {
            let start = substitute_expression(ast, start, subst);
            let end = substitute_expression(ast, end, subst);
            Expression::Range(start, end, inclusive)
        }
        Expression::Tuple(elements) => {
            let element_ids: Vec<ExprId> = ast.exprs_in(elements).to_vec();
            let mut substituted = Vec::with_capacity(element_ids.len());
            for element in element_ids {
                substituted.push(substitute_expression(ast, element, subst));
            }
            Expression::Tuple(ast.add_expr_list(&substituted))
        }
        Expression::Switch(scrutinee, cases) => {
            let scrutinee = substitute_expression(ast, scrutinee, subst);
            let case_entries: Vec<SwitchCase> = ast.cases_in(cases).to_vec();
            let mut substituted = Vec::with_capacity(case_entries.len());
            for case in case_entries {
                substituted.push(SwitchCase {
                    pattern: case.pattern,
                    body: substitute_block(ast, case.body, subst),
                });
            }
            Expression::Switch(scrutinee, ast.add_cases(&substituted))
        }
        // The body of an `unsafe` block is ordinary code, so a type parameter
        // used inside one substitutes the same as anywhere else. Missing this
        // left `sizeof(T)` reading as zero inside `unsafe { }`.
        Expression::Unsafe(body) => {
            Expression::Unsafe(substitute_block(ast, body, subst))
        }
        Expression::UnsafeFn(inner) => {
            Expression::UnsafeFn(substitute_expression(ast, inner, subst))
        }
        _ => return expression,
    };
    ast.push_expr(node, span)
}

// What a struct or an enum is called, looking through a borrow of one, or
// nothing for anything else. Two aggregates are the same type when they carry
// the same name; everything else about a type is checked elsewhere, and a name
// is what survives being turned into an address.
fn aggregate_name(ty: &Type) -> Option<&str> {
    match ty {
        Type::Struct(name) | Type::Enum(name) => Some(name.as_str()),
        Type::Ref(inner) | Type::RefMut(inner) => aggregate_name(inner),
        _ => None,
    }
}

/// The element a target wants a slice of, where it wants one.
///
/// A `str` is a `[]u8`, so an array of bytes reaching one becomes a slice of
/// itself exactly as an array of anything else does. Every site that built a
/// slice for a target read `Type::Slice` alone, so `data : str = data` was
/// refused, and so was handing a `[4]u8` to a `str` parameter, while the same
/// value reached a `str` in two steps through a `[]u8`. The self-hosted compiler
/// holds the two as one type and builds all of them.
fn slice_element_wanted(target: &Type) -> Option<Type> {
    match target {
        Type::Slice(element) => Some((**element).clone()),
        Type::Str => Some(Type::U8),
        _ => None,
    }
}

fn needs_memory(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Struct(_)
            | Type::Array(_, _)
            | Type::Enum(_)
            | Type::Str
            | Type::Slice(_)
    )
}

// A place expression names storage: a variable, a field, an element, or a
// dereference. Anything else is a value expression, whose result has to be
// spilled to memory before its address can be taken.
fn is_place_expression(ast: &Ast, expression: ExprId) -> bool {
    matches!(
        ast.expr(expression),
        Expression::Identifier(_)
            | Expression::FieldAccess(_, _)
            | Expression::Index(_, _)
            | Expression::Dereference(_)
    )
}

// A str and a slice share the same fat-pointer layout: a data pointer at offset
// 0 and a usize length at offset 8.
const STR_PTR_OFFSET: usize = 0;
const STR_LEN_OFFSET: usize = 8;
const SLICE_PTR_OFFSET: usize = 0;
const SLICE_LEN_OFFSET: usize = 8;

fn str_byte_ptr_type() -> Type {
    Type::Ptr(Box::new(Type::U8))
}

fn array_element_type(
    ast: &Ast,
    annotation: Option<&Type>,
    elements: &[ExprId],
    signatures: &HashMap<String, FunctionSignature>,
) -> Type {
    match annotation {
        Some(Type::Array(inner, _)) | Some(Type::Slice(inner)) => {
            return (**inner).clone();
        }
        _ => {}
    }
    match elements.first().map(|element| ast.expr(*element)) {
        Some(Expression::Literal(Literal::Integer(_))) => Type::I64,
        Some(Expression::Literal(Literal::Float(_))) => Type::F64,
        Some(Expression::Literal(Literal::Float32(_))) => Type::F32,
        Some(Expression::Literal(Literal::Boolean(_)))
        | Some(Expression::Boolean(_)) => Type::Bool,
        Some(Expression::StructInit(name, _)) => {
            Type::Struct(ast.name(*name).to_string())
        }
        Some(Expression::EnumVariantInit(name, _, _)) => {
            Type::Enum(ast.name(*name).to_string())
        }
        Some(Expression::Identifier(name))
            if let Some(signature) = signatures.get(ast.name(*name)) =>
        {
            Type::Proc(
                signature.parameters.clone(),
                Box::new(signature.return_type.clone()),
            )
        }
        Some(
            Expression::Function(parameters, return_sig, _)
            | Expression::Proc(parameters, return_sig, _),
        ) => Type::Proc(
            ast.params_in(*parameters)
                .iter()
                .map(parameter_type)
                .collect(),
            Box::new(
                ast.signature_to_type(ast.signature(*return_sig))
                    .unwrap_or(Type::Void),
            ),
        ),
        Some(Expression::Literal(Literal::Array(inner))) => Type::Array(
            Box::new(array_element_type(
                ast,
                None,
                ast.exprs_in(*inner),
                signatures,
            )),
            inner.len(),
        ),
        _ => Type::I64,
    }
}

// Whether a `return` hands back a failure rather than an answer: the `Err` of an
// enum the failure-set lowering made. The `?` form writes one of these and so
// does a program handing a failure back itself, which is what makes this the
// question rather than where the `return` came from.
fn returns_a_failure(ast: &Ast, expression: ExprId) -> bool {
    let Expression::EnumVariantInit(name, variant, _) = ast.expr(expression)
    else {
        return false;
    };
    ast.name(*variant) == "Err" && ast.is_failure_result(ast.name(*name))
}

// A deferred statement is lowered again at every exit and its names are resolved
// there, so a name it mentions that is bound again after the `defer` reads as
// that later binding rather than the one in scope where the `defer` was written.
// Refused where it is written, because neither reading is the one the line has,
// and one of them is a binding the path taken never reached. The self-hosted
// compiler refuses the same programs, walking the deferred statement's tokens
// against the locals the function bound.
fn check_defer_names(
    ast: &Ast,
    deferred: StmtId,
    rest: &[StmtId],
) -> Result<()> {
    let mut mentioned = Vec::new();
    crate::modules::interface_names::names_in_statement(
        ast,
        deferred,
        &mut mentioned,
    );
    let mut rebound = HashSet::new();
    for statement in rest {
        crate::modules::import_visibility::bound_in_statement(
            ast,
            *statement,
            &mut rebound,
        );
    }
    for name in mentioned {
        if rebound.contains(&name) {
            bail!(
                "'{name}' is bound again below this `defer`, which is lowered at every exit, so the copy would read that binding rather than the one in scope here"
            );
        }
    }
    Ok(())
}

type LayoutMaps = (HashMap<String, StructLayout>, HashMap<String, EnumLayout>);

fn compute_layouts(ast: &Ast, statements: &[StmtId]) -> LayoutMaps {
    let struct_defs: Vec<(&str, Range32)> = statements
        .iter()
        .filter_map(|statement| match ast.stmt(*statement) {
            Statement::Struct(name, _, fields) => {
                Some((ast.name(*name), *fields))
            }
            _ => None,
        })
        .collect();
    let enum_defs: Vec<(&str, Range32)> = statements
        .iter()
        .filter_map(|statement| match ast.stmt(*statement) {
            Statement::Enum(name, _, variants) => {
                Some((ast.name(*name), *variants))
            }
            _ => None,
        })
        .collect();

    let mut structs: HashMap<String, StructLayout> = HashMap::new();
    let mut enums: HashMap<String, EnumLayout> = HashMap::new();
    loop {
        let mut progress = false;
        for (name, fields) in &struct_defs {
            if structs.contains_key(*name) {
                continue;
            }
            if let Some(layout) = try_struct_layout(
                ast,
                ast.fields_in(*fields),
                &structs,
                &enums,
                ast.is_packed_struct(name),
            ) {
                structs.insert((*name).to_string(), layout);
                progress = true;
            }
        }
        for (name, variants) in &enum_defs {
            if enums.contains_key(*name) {
                continue;
            }
            if let Some(layout) = try_enum_layout(
                ast,
                ast.variants_in(*variants),
                &structs,
                &enums,
            ) {
                enums.insert((*name).to_string(), layout);
                progress = true;
            }
        }
        if !progress {
            break;
        }
    }
    (structs, enums)
}

fn try_struct_layout(
    ast: &Ast,
    fields: &[StructField],
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
    packed: bool,
) -> Option<StructLayout> {
    let mut offset = 0;
    let mut align = 1;
    let mut field_layouts = Vec::with_capacity(fields.len());
    for field in fields {
        let (field_size, natural) =
            size_and_align(&field.field_type, structs, enums)?;
        // Packed means every field at the next byte. A stated alignment is what
        // the field starts at a multiple of instead of what its type would ask
        // for; anything else is the type's own answer.
        let field_align = match (packed, field.align) {
            (true, _) => 1,
            (false, Some(stated)) => stated,
            (false, None) => natural,
        };
        offset = round_up(offset, field_align);
        field_layouts.push(FieldLayout {
            name: ast.name(field.name).to_string(),
            ty: field.field_type.clone(),
            offset,
        });
        offset += field_size;
        align = align.max(field_align);
    }
    Some(StructLayout {
        size: round_up(offset, align),
        align,
        fields: field_layouts,
    })
}

fn try_enum_layout(
    ast: &Ast,
    variants: &[EnumVariant],
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
) -> Option<EnumLayout> {
    let tag_size = 4;
    let mut payload_align = 1;
    for variant in variants {
        if let Some(fields) = variant.fields {
            for field in ast.fields_in(fields) {
                let (_, field_align) =
                    size_and_align(&field.field_type, structs, enums)?;
                payload_align = payload_align.max(field_align);
            }
        }
    }
    let payload_offset = round_up(tag_size, payload_align);

    let mut variant_layouts = Vec::with_capacity(variants.len());
    let mut max_end = payload_offset;
    for (index, variant) in variants.iter().enumerate() {
        let mut offset = payload_offset;
        let mut field_layouts = Vec::new();
        if let Some(fields) = variant.fields {
            for field in ast.fields_in(fields) {
                let (field_size, field_align) =
                    size_and_align(&field.field_type, structs, enums)?;
                offset = round_up(offset, field_align);
                field_layouts.push(FieldLayout {
                    name: ast.name(field.name).to_string(),
                    ty: field.field_type.clone(),
                    offset,
                });
                offset += field_size;
            }
        }
        max_end = max_end.max(offset);
        variant_layouts.push(EnumVariantLayout {
            name: ast.name(variant.name).to_string(),
            tag: index as u32,
            fields: field_layouts,
        });
    }

    let align = payload_align.max(tag_size);
    Some(EnumLayout {
        size: round_up(max_end, align),
        align,
        variants: variant_layouts,
    })
}

fn size_and_align(
    ty: &Type,
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
) -> Option<(usize, usize)> {
    match ty {
        // A named type from an annotation parses as `Struct`, even when it
        // names an enum, so fall back to the enum registry.
        Type::Struct(name) => structs
            .get(name)
            .map(|layout| (layout.size, layout.align))
            .or_else(|| {
                enums.get(name).map(|layout| (layout.size, layout.align))
            }),
        Type::Enum(name) => {
            enums.get(name).map(|layout| (layout.size, layout.align))
        }
        Type::Array(inner, count) => {
            let (size, align) = size_and_align(inner, structs, enums)?;
            Some((size * count, align))
        }
        other => Some((other.size_of(), other.align_of())),
    }
}

fn round_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    value.div_ceil(align) * align
}

struct BlockUnderConstruction {
    statements: Vec<IrStatement>,
    terminator: Option<IrTerminator>,
}

struct LoopTargets {
    continue_block: BlockId,
    break_block: BlockId,
}

struct FunctionLowering<'a> {
    builder: &'a IrBuilder,
    ast: &'a mut Ast,
    locals: Vec<IrLocal>,
    blocks: Vec<BlockUnderConstruction>,
    scopes: Vec<HashMap<String, LocalId>>,
    loops: Vec<LoopTargets>,
    current: BlockId,
    return_type: Type,
    // Each cleanup and whether it runs only where the function leaves through
    // its failure set. One list rather than two, so they run in the order they
    // were written, last first, whichever kind they are.
    active_defers: Vec<(StmtId, bool)>,
    current_position: Position,
    specializations: Vec<Specialization>,
    anonymous: Vec<AnonRequest>,
    // What the enclosing declaration named its `format` parameter, and every
    // parameter name in the order they were written. A body that hands its own
    // literal on together with its own trailing parameters is handing on what a
    // caller wrote and what that caller gave, so the count was checked where
    // they were written and there is nothing left to count here.
    forwarded_format: Option<String>,
    parameter_names: Vec<String>,
}

impl<'a> FunctionLowering<'a> {
    fn new(
        builder: &'a IrBuilder,
        ast: &'a mut Ast,
        return_type: Type,
    ) -> Self {
        let entry = BlockUnderConstruction {
            statements: Vec::new(),
            terminator: None,
        };
        FunctionLowering {
            builder,
            ast,
            locals: Vec::new(),
            blocks: vec![entry],
            scopes: vec![HashMap::new()],
            loops: Vec::new(),
            current: 0,
            return_type,
            active_defers: Vec::new(),
            current_position: Position::default(),
            specializations: Vec::new(),
            anonymous: Vec::new(),
            forwarded_format: None,
            parameter_names: Vec::new(),
        }
    }

    // Whether this call hands on the literal and the values the enclosing
    // declaration was given, rather than writing its own. `print` is the case:
    // it takes a `format` parameter and a list and passes both to `write`, so
    // the holes were counted against the values where the reader wrote them,
    // and counting again here would be counting a name.
    //
    // Both halves are required. A body forwarding the literal beside values of
    // its own would be handing over a count nothing had checked, which is the
    // hole the word exists to close. The values are held to being this
    // declaration's own trailing parameters in the order it declared them,
    // which is what a forwarded list expands to and what writing a list out by
    // hand is not.
    fn forwards_its_own_format(
        &self,
        argument: ExprId,
        arguments: &[ExprId],
    ) -> bool {
        let Some(format) = &self.forwarded_format else {
            return false;
        };
        let Expression::Identifier(name) = self.ast.expr(argument) else {
            return false;
        };
        if self.ast.name(*name) != format {
            return false;
        }
        let Some(at) = arguments.iter().position(|held| *held == argument)
        else {
            return false;
        };
        let mut handed: Vec<&str> = Vec::new();
        for held in &arguments[at + 1..] {
            let Expression::Identifier(other) = self.ast.expr(*held) else {
                return false;
            };
            handed.push(self.ast.name(*other));
        }
        // What this declaration has left to give past its own literal. A call
        // that hands over exactly those, in that order, is handing on what it
        // was given, and a call giving anything else is writing a list of its
        // own that nothing has counted. A `print("\n")` forwards an empty list,
        // which is the two being equal at zero.
        let Some(mine) =
            self.parameter_names.iter().position(|held| held == format)
        else {
            return false;
        };
        let trailing = &self.parameter_names[mine + 1..];
        if trailing.len() == handed.len()
            && trailing
                .iter()
                .zip(handed.iter())
                .all(|(declared, given)| declared == given)
        {
            return true;
        }
        // The list under its own name, which is how the body writes it before
        // the elements have taken a parameter each. One name stands for all of
        // them, and they are the ones named after it.
        handed.len() == 1
            && trailing.iter().enumerate().all(|(index, declared)| {
                *declared == pack_element_name(handed[0], index)
            })
    }

    fn fresh_local(&mut self, ty: Type, name: Option<String>) -> LocalId {
        let id = self.locals.len();
        let size = self.builder.byte_size(&ty);
        let in_memory = needs_memory(&ty);
        let linear = self.builder.type_is_linear(&ty);
        self.locals.push(IrLocal {
            ty,
            name,
            in_memory,
            size,
            linear,
            position: self.current_position,
        });
        id
    }

    fn mark_in_memory(&mut self, local: LocalId) {
        self.locals[local].in_memory = true;
    }

    fn new_block(&mut self) -> BlockId {
        let id = self.blocks.len();
        self.blocks.push(BlockUnderConstruction {
            statements: Vec::new(),
            terminator: None,
        });
        id
    }

    fn switch_to(&mut self, block: BlockId) {
        self.current = block;
    }

    fn current_is_terminated(&self) -> bool {
        self.blocks[self.current].terminator.is_some()
    }

    fn emit(&mut self, statement: IrStatement) {
        if self.current_is_terminated() {
            let block = self.new_block();
            self.switch_to(block);
        }
        self.blocks[self.current].statements.push(statement);
    }

    fn set_terminator(&mut self, terminator: IrTerminator) {
        if self.blocks[self.current].terminator.is_none() {
            self.blocks[self.current].terminator = Some(terminator);
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define_variable(&mut self, name: &str, local: LocalId) {
        self.scopes
            .last_mut()
            .unwrap()
            .insert(name.to_string(), local);
    }

    fn resolve_variable(&self, name: &str) -> Option<LocalId> {
        for scope in self.scopes.iter().rev() {
            if let Some(local) = scope.get(name) {
                return Some(*local);
            }
        }
        None
    }

    fn finish(self) -> (Vec<IrLocal>, Vec<IrBlock>) {
        let blocks = self
            .blocks
            .into_iter()
            .map(|block| IrBlock {
                statements: block.statements,
                terminator: block
                    .terminator
                    .unwrap_or(IrTerminator::Unreachable),
            })
            .collect();
        (self.locals, blocks)
    }

    fn lower_block(
        &mut self,
        block: Range32,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        self.push_scope();
        let mut result = (unit_operand(), Type::Void);
        let statements: Vec<StmtId> = self.ast.stmts_in(block).to_vec();
        for (index, statement) in statements.iter().enumerate() {
            let is_last = index + 1 == statements.len();
            let position = self.ast.stmt_position(*statement);
            self.current_position = position;
            if is_last
                && let Statement::Expression(expression) =
                    self.ast.stmt(*statement)
            {
                let expression = *expression;
                result = locate(
                    self.lower_expression(expression, expected),
                    position,
                )?;
            } else {
                locate(self.lower_statement(*statement), position)?;
            }
        }
        self.pop_scope();
        Ok(result)
    }

    // A body's last expression is its answer, so it is held to what a `return`
    // is held to. Left out, a function could answer with a distinct type by
    // writing its representation and dropping the word `return`, which is how
    // the null of every handle type in the bindings was written.
    //
    // The value itself is not in hand here, so a written number cannot be told
    // from a name. It does not need to be: a literal has taken the answer type
    // by now, and the two agree.
    fn check_answer(
        &self,
        value_type: &Type,
        return_type: &Type,
    ) -> Result<()> {
        if value_type == return_type {
            return Ok(());
        }
        let Type::Distinct(name, _) = return_type else {
            return Ok(());
        };
        let note = if self.builder.flags.contains_key(name) {
            "a set of bits is built only from the names declared under it"
        } else {
            "a distinct type is not its representation"
        };
        bail!(
            "this returns a '{}' and the function answers with a '{return_type}'; {note}",
            spelled(value_type)
        )
    }

    fn lower_body_with_defers(
        &mut self,
        body: Range32,
        return_type: &Type,
    ) -> Result<()> {
        let outer_defers = self.active_defers.len();
        self.push_scope();
        let statements: Vec<StmtId> = self.ast.stmts_in(body).to_vec();
        for (index, statement) in statements.iter().enumerate() {
            let is_last = index + 1 == statements.len();
            let position = self.ast.stmt_position(*statement);
            self.current_position = position;
            match self.ast.stmt(*statement).clone() {
                Statement::Defer(inner) | Statement::ErrDefer(inner) => {
                    let on_failure = matches!(
                        self.ast.stmt(*statement),
                        Statement::ErrDefer(_)
                    );
                    // An `errdefer` in a function that cannot fail names an
                    // exit that function does not have.
                    if on_failure && !self.answers_with_a_failure_set() {
                        locate(
                            Err(anyhow::anyhow!(
                                "`errdefer` runs where a function leaves through its failure set, and this one has none; write `-> T ! E`, or `defer` to run it however the function leaves"
                            )),
                            position,
                        )?;
                    }
                    locate(
                        check_defer_names(
                            self.ast,
                            inner,
                            &statements[index + 1..],
                        ),
                        position,
                    )?;
                    self.active_defers.push((inner, on_failure));
                }
                Statement::Expression(expression) if is_last => {
                    let (value, value_type) = locate(
                        self.lower_expression(expression, Some(return_type)),
                        position,
                    )?;
                    locate(
                        self.check_answer(&value_type, return_type),
                        position,
                    )?;
                    if !self.current_is_terminated() {
                        let operand = if matches!(return_type, Type::Void) {
                            None
                        } else {
                            Some(self.coerce(
                                value,
                                &value_type,
                                return_type,
                            )?)
                        };
                        self.emit_return(operand)?;
                    }
                }
                _ => locate(self.lower_statement(*statement), position)?,
            }
            if self.current_is_terminated() {
                break;
            }
        }
        self.pop_scope();

        if !self.current_is_terminated() {
            self.emit_return(None)?;
        }
        self.active_defers.truncate(outer_defers);
        Ok(())
    }

    fn emit_return(&mut self, operand: Option<IrOperand>) -> Result<()> {
        self.run_active_defers(false)?;
        self.set_terminator(IrTerminator::Return(operand));
        Ok(())
    }

    // The exit a failure takes: the `return` a `?` writes where the call it
    // guards answered `Err`, and one a program writes itself.
    fn emit_failure_return(
        &mut self,
        operand: Option<IrOperand>,
    ) -> Result<()> {
        self.run_active_defers(true)?;
        self.set_terminator(IrTerminator::Return(operand));
        Ok(())
    }

    fn run_active_defers(&mut self, failing: bool) -> Result<()> {
        let defers = self.active_defers.clone();
        for (deferred, on_failure) in defers.iter().rev() {
            if *on_failure && !failing {
                continue;
            }
            self.lower_statement(*deferred)?;
        }
        Ok(())
    }

    // Whether this function has a failure exit at all, which is what an
    // `errdefer` names. The failure-set pass has already turned `-> T ! E` into
    // the enum the function answers with, and recorded the ones it made.
    fn answers_with_a_failure_set(&self) -> bool {
        matches!(&self.return_type, Type::Enum(name) | Type::Struct(name)
            if self.ast.is_failure_result(name))
    }

    fn lower_statement(&mut self, statement: StmtId) -> Result<()> {
        match self.ast.stmt(statement).clone() {
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                let name = self.ast.name(name).to_string();
                if let Expression::StructInit(struct_name, field_inits) =
                    self.ast.expr(value).clone()
                {
                    let struct_name = self.ast.name(struct_name).to_string();
                    let layout_name = match &type_annotation {
                        // `p : Point = { x = 1, y = 2 }`: the annotation is the
                        // only place the struct is named.
                        Some(Type::Struct(annotated))
                            if struct_name.is_empty()
                                || is_generic_instance(annotated) =>
                        {
                            annotated.clone()
                        }
                        _ if self
                            .builder
                            .generic_struct_defs
                            .contains_key(&struct_name) =>
                        {
                            let Some(instance) = self
                                .generic_instance_of(&struct_name, field_inits)
                            else {
                                bail!(
                                    "'{struct_name}' is generic and nothing here says which instance this literal is: write the arguments on the literal, as in '{struct_name}<i64> {{ ... }}', or give the binding a declared type that names them"
                                );
                            };
                            instance
                        }
                        _ => struct_name.clone(),
                    };
                    if layout_name.is_empty() {
                        bail!(
                            "a `{{ ... }}` literal takes its type from what the context expects, and this binding has no type to take it from; annotate it or name the struct"
                        );
                    }
                    let ty = Type::Struct(layout_name.clone());
                    let local = self.fresh_local(ty, Some(name.clone()));
                    self.init_struct(local, &layout_name, field_inits)?;
                    self.define_variable(&name, local);
                    return Ok(());
                }
                if let Expression::Literal(Literal::Array(elements)) =
                    self.ast.expr(value)
                {
                    let elements: Vec<ExprId> =
                        self.ast.exprs_in(*elements).to_vec();
                    let element_type = array_element_type(
                        self.ast,
                        type_annotation.as_ref(),
                        &elements,
                        &self.builder.signatures,
                    );
                    let ty = Type::Array(
                        Box::new(element_type.clone()),
                        elements.len(),
                    );
                    let local = self.fresh_local(ty, Some(name.clone()));
                    self.init_array(local, &element_type, &elements)?;
                    self.define_variable(&name, local);
                    return Ok(());
                }
                if let Expression::EnumVariantInit(
                    enum_name,
                    variant_name,
                    field_inits,
                ) = self.ast.expr(value).clone()
                    // A value a type names under itself reads as a variant and
                    // is a value, so it is bound the way every other value is
                    // rather than by building a tagged one here.
                    && !self.binds_a_named_value(
                        type_annotation.as_ref(),
                        self.ast.name(enum_name),
                        self.ast.name(variant_name),
                    )
                {
                    let enum_name = self.ast.name(enum_name).to_string();
                    let variant_name = self.ast.name(variant_name).to_string();
                    // A binding is the one place a `.Name` never reached the
                    // walk that refuses one, since the enum is built here
                    // rather than lowered as an expression. It is refused here
                    // in the same words, naming the annotation's type.
                    if enum_name.is_empty() {
                        let held = self.ast.intern(&variant_name);
                        return Err(refuse_inferred_variant(
                            self.ast,
                            held,
                            type_annotation.as_ref(),
                        ));
                    }
                    // `o : Option<i64> = Option::Some { value = 3 }`: the
                    // annotation says which instance, the literal does not
                    // carry arguments, so the annotation is what names the
                    // layout. Same rule as a generic struct literal above.
                    let layout_name = match &type_annotation {
                        Some(
                            Type::Enum(annotated) | Type::Struct(annotated),
                        ) if is_generic_instance(annotated)
                            && annotated
                                .starts_with(&format!("{enum_name}<")) =>
                        {
                            annotated.clone()
                        }
                        _ => enum_name.clone(),
                    };
                    let ty = Type::Enum(layout_name.clone());
                    let local = self.fresh_local(ty, Some(name.clone()));
                    self.init_enum(
                        local,
                        &layout_name,
                        &variant_name,
                        field_inits,
                    )?;
                    self.define_variable(&name, local);
                    return Ok(());
                }
                let (operand, value_type) =
                    self.lower_expression(value, type_annotation.as_ref())?;
                if let Some(annotated) = &type_annotation
                    && distinct_mismatch(
                        self.ast,
                        value,
                        &value_type,
                        annotated,
                        &self.builder.flags,
                    )
                {
                    let (described, note) = nominal_words(
                        self.ast,
                        value,
                        &value_type,
                        annotated,
                        &self.builder.flags,
                    );
                    bail!(
                        "this binding is a '{}' and the value is {described}; {note}",
                        spelled(annotated)
                    );
                }
                // A value written into a declared type it does not fit. The
                // same question the IR typechecker asks, asked here while both
                // types are still spelled the way the reader wrote them: an
                // aggregate travels by address, so the coercion below takes one
                // and every check after it agrees about a pointer, and the
                // complaint lands naming the lowered local rather than the
                // binding.
                //
                // Two bridges the coercion builds and `fits` does not carry: an
                // array reaching a slice becomes a view of the whole of itself,
                // and a bare number takes the width the binding declares, which
                // is what `held : f64 = 0` means.
                // A number takes the width and the kind the binding declares,
                // which is what `held : f64 = mantissa` means and what the
                // coercion below builds.
                let numeric = |ty: &Type| ty.is_integer() || ty.is_float();
                if let Some(annotated) = &type_annotation
                    && !(numeric(&value_type) && numeric(annotated))
                    && !slice_element_wanted(annotated).is_some_and(|element| {
                        matches!(&value_type, Type::Array(held, _)
                            if **held == element)
                    })
                    && !crate::ir::typecheck::fits(&value_type, annotated)
                {
                    bail!(
                        "this binding is a '{}' and the value is a '{}'",
                        spelled(annotated),
                        spelled(&value_type)
                    );
                }
                if matches!(value_type, Type::Void) {
                    bail!(
                        "cannot bind '{name}' to a void value; this expression produces no value"
                    );
                }
                // Binding a borrowed aggregate *by name* is a copy of the
                // value rather than a second name for the caller's storage.
                // Naming a parameter always means the caller's value, so the
                // binding takes what it holds. Only a name: a call that answers
                // with a `ref T` handed out a borrow on purpose, and `ref x :=
                // place` asks for one, so both keep what they were given.
                let borrowed_aggregate = match &value_type {
                    Type::Ref(inner) | Type::RefMut(inner)
                        if needs_memory(inner)
                            && matches!(
                                self.ast.expr(value),
                                Expression::Identifier(_)
                            ) =>
                    {
                        Some(inner.as_ref().clone())
                    }
                    _ => None,
                };
                if let Some(inner) = &borrowed_aggregate
                    && self.builder.type_is_linear(inner)
                {
                    bail!(
                        "'{name}' would be a second owner of a '{inner}', which is consumed exactly once; bind a `ref` to read it in place"
                    );
                }
                let declared = match (&type_annotation, &borrowed_aggregate) {
                    (Some(annotated), _) => annotated.clone(),
                    (None, Some(inner)) => inner.clone(),
                    (None, None) => value_type.clone(),
                };
                let coerced = self.coerce(operand, &value_type, &declared)?;
                let local =
                    self.fresh_local(declared.clone(), Some(name.clone()));
                self.emit(IrStatement::Assign(local, IrRvalue::Use(coerced)));
                self.define_variable(&name, local);
                Ok(())
            }
            Statement::Constant(name, value) => {
                let name = self.ast.name(name).to_string();
                let (operand, value_type) =
                    self.lower_expression(value, None)?;
                let local = self.fresh_local(value_type, Some(name.clone()));
                self.emit(IrStatement::Assign(local, IrRvalue::Use(operand)));
                self.define_variable(&name, local);
                Ok(())
            }
            Statement::Assignment(target, value) => {
                self.lower_assignment(target, value)
            }
            Statement::Return(expression) => {
                let return_type = self.return_type.clone();
                if matches!(return_type, Type::Void) {
                    self.emit_return(None)?;
                } else {
                    let (operand, value_type) =
                        self.lower_expression(expression, Some(&return_type))?;
                    if distinct_mismatch(
                        self.ast,
                        expression,
                        &value_type,
                        &return_type,
                        &self.builder.flags,
                    ) {
                        let (described, note) = nominal_words(
                            self.ast,
                            expression,
                            &value_type,
                            &return_type,
                            &self.builder.flags,
                        );
                        bail!(
                            "this returns {described} and the function answers with a '{return_type}'; {note}"
                        );
                    }
                    let coerced =
                        self.coerce(operand, &value_type, &return_type)?;
                    if returns_a_failure(self.ast, expression) {
                        self.emit_failure_return(Some(coerced))?;
                    } else {
                        self.emit_return(Some(coerced))?;
                    }
                }
                Ok(())
            }
            Statement::Expression(expression) => {
                let (_, answered) = self.lower_expression(expression, None)?;
                // A call that can fail answers with which of the two happened,
                // and a statement reads neither. Left alone it is a failure the
                // program stepped over, which is the one thing a failure set
                // exists to stop. `_ :=` is how a caller says the answer was
                // meant to go unread.
                if let Type::Enum(name) = &answered
                    && self.ast.is_failure_result(name)
                {
                    bail!(
                        "this can fail and nothing reads whether it did; write `?` to hand the failure up, `match` to answer it here, or `_ :=` to say it was meant to go unread"
                    );
                }
                Ok(())
            }
            Statement::While(condition, body) => {
                self.lower_while(condition, body)
            }
            Statement::For(variable, second, range, body) => {
                let variable = self.ast.name(variable).to_string();
                let second = second.map(|held| self.ast.name(held).to_string());
                self.lower_for(&variable, second.as_deref(), range, body)
            }
            // Only the top level of a function body collects a `defer`, so one
            // reaching here is written inside a block. Named rather than left to
            // the catch-all below, which says a statement is unsupported and
            // gives a reader nothing to do about it.
            Statement::Defer(_) | Statement::ErrDefer(_) => {
                bail!(
                    "a `defer` belongs at the top level of a body, since it runs where the function leaves rather than where this block does"
                )
            }
            Statement::Break => {
                let Some(targets) = self.loops.last() else {
                    bail!("break outside loop");
                };
                self.set_terminator(IrTerminator::Jump(targets.break_block));
                Ok(())
            }
            Statement::Continue => {
                let Some(targets) = self.loops.last() else {
                    bail!("continue outside loop");
                };
                self.set_terminator(IrTerminator::Jump(targets.continue_block));
                Ok(())
            }
            _ => bail!(
                "unsupported statement: {}",
                display_stmt(self.ast, statement)
            ),
        }
    }

    fn lower_while(&mut self, condition: ExprId, body: Range32) -> Result<()> {
        let header = self.new_block();
        let body_block = self.new_block();
        let exit = self.new_block();

        self.set_terminator(IrTerminator::Jump(header));
        self.switch_to(header);
        let (condition_operand, _) =
            self.lower_expression(condition, Some(&Type::Bool))?;
        self.set_terminator(IrTerminator::Branch {
            condition: condition_operand,
            then_block: body_block,
            else_block: exit,
        });

        self.switch_to(body_block);
        self.loops.push(LoopTargets {
            continue_block: header,
            break_block: exit,
        });
        self.lower_block(body, None)?;
        self.loops.pop();
        self.set_terminator(IrTerminator::Jump(header));

        self.switch_to(exit);
        Ok(())
    }

    // `for item in items` over a slice, a fixed array, or a `str`. This is the
    // index-and-bound loop written out, not an iterator: there is no protocol,
    // nothing is called per element, and the emitted code is what the same loop
    // written by hand produces. The element binds the way a parameter of its
    // type would, so an aggregate is borrowed and a scalar is copied.
    fn lower_for_sequence(
        &mut self,
        variable: &str,
        second: Option<&str>,
        iterable: ExprId,
        body: Range32,
    ) -> Result<()> {
        // The sequence is evaluated once, into a local the loop owns, so
        // `for x in make()` calls `make` once and a body that appends to the
        // same container does not walk what it just added.
        let (sequence, sequence_type) =
            self.lower_expression(iterable, None)?;
        let element = match &sequence_type {
            Type::Array(element, _) => (**element).clone(),
            Type::Slice(element) => (**element).clone(),
            Type::Str => Type::U8,
            other => bail!(
                "a `for` walks a range, a slice, an array or a `str`, and '{other}' is none of those"
            ),
        };
        let IrOperand::Local(held) = sequence else {
            bail!("the sequence a `for` walks is not a place");
        };
        let sequence_name = format!("__for_sequence_{held}");
        self.define_variable(&sequence_name, held);
        let span = self.ast.expr_span(iterable);
        let sequence_symbol = self.ast.intern(&sequence_name);
        let walked = self
            .ast
            .push_expr(Expression::Identifier(sequence_symbol), span);

        let (length, length_type) = match &sequence_type {
            Type::Array(_, count) => (
                IrOperand::Constant(IrConstant::Integer(
                    *count as i64,
                    Type::I64,
                )),
                Type::I64,
            ),
            Type::Str => self.lower_str_len(&[walked])?,
            _ => self.lower_slice_len(&[walked])?,
        };
        let bound = self.fresh_local(Type::I64, None);
        let coerced = self.coerce(length, &length_type, &Type::I64)?;
        self.emit(IrStatement::Assign(bound, IrRvalue::Use(coerced)));

        let index_name = second.map(|_| variable);
        let index = self
            .fresh_local(Type::I64, index_name.map(|name| name.to_string()));
        self.emit(IrStatement::Assign(
            index,
            IrRvalue::Use(IrOperand::Constant(IrConstant::Integer(
                0,
                Type::I64,
            ))),
        ));

        let header = self.new_block();
        let body_block = self.new_block();
        let step_block = self.new_block();
        let exit = self.new_block();

        self.set_terminator(IrTerminator::Jump(header));
        self.switch_to(header);
        let condition = self.fresh_local(Type::Bool, None);
        self.emit(IrStatement::Assign(
            condition,
            IrRvalue::Binary(
                IrBinOp::LessThan,
                IrOperand::Local(index),
                IrOperand::Local(bound),
            ),
        ));
        self.set_terminator(IrTerminator::Branch {
            condition: IrOperand::Local(condition),
            then_block: body_block,
            else_block: exit,
        });

        self.switch_to(body_block);
        self.push_scope();
        if let Some(item) = second {
            self.define_variable(variable, index);
            self.bind_sequence_element(item, walked, index, &element)?;
        } else {
            self.bind_sequence_element(variable, walked, index, &element)?;
        }
        self.loops.push(LoopTargets {
            continue_block: step_block,
            break_block: exit,
        });
        self.lower_block(body, None)?;
        self.loops.pop();
        self.pop_scope();
        self.set_terminator(IrTerminator::Jump(step_block));

        self.switch_to(step_block);
        self.emit(IrStatement::Assign(
            index,
            IrRvalue::Binary(
                IrBinOp::Add,
                IrOperand::Local(index),
                IrOperand::Constant(IrConstant::Integer(1, Type::I64)),
            ),
        ));
        self.set_terminator(IrTerminator::Jump(header));

        self.switch_to(exit);
        Ok(())
    }

    // The element at the index, bound under `name`. An aggregate binds as a
    // borrow of where it sits, so walking a run of structs copies nothing. A
    // scalar binds as the value, which is what a parameter of that type is.
    fn bind_sequence_element(
        &mut self,
        name: &str,
        iterable: ExprId,
        index: LocalId,
        element: &Type,
    ) -> Result<()> {
        let index_name = format!("__for_index_{index}");
        let span = self.ast.expr_span(iterable);
        let index_symbol = self.ast.intern(&index_name);
        let index_expression = self
            .ast
            .push_expr(Expression::Identifier(index_symbol), span);
        let indexed = self
            .ast
            .push_expr(Expression::Index(iterable, index_expression), span);
        // The index is a local the loop owns rather than something the reader
        // wrote, so it is named into scope only for this lookup.
        self.define_variable(&index_name, index);
        let (address, _) = self.element_address_of(indexed)?;
        if needs_memory(element) {
            let local = self.fresh_local(
                Type::Ref(Box::new(element.clone())),
                Some(name.to_string()),
            );
            self.emit(IrStatement::Assign(local, IrRvalue::Use(address)));
            self.define_variable(name, local);
            return Ok(());
        }
        let (value, value_type) = self.load_from(address, element.clone())?;
        let local =
            self.fresh_local(value_type.clone(), Some(name.to_string()));
        self.emit(IrStatement::Assign(local, IrRvalue::Use(value)));
        self.define_variable(name, local);
        Ok(())
    }

    fn element_address_of(
        &mut self,
        indexed: ExprId,
    ) -> Result<(IrOperand, Type)> {
        let Expression::Index(base, index) = self.ast.expr(indexed) else {
            bail!("expected an index expression");
        };
        let (base, index) = (*base, *index);
        self.element_address(base, index)
    }

    // `for slot in live_slots(c)`: the slots of a generational container that hold an
    // element, in slot order. Which ones those are is a bit each in
    // `live_words`, so a word of zeroes passes over sixty-four slots on one
    // test, and a word with bits set gives up its lowest one at a time. No slot
    // is asked whether it holds an element and no empty slot is reached, which
    // is the whole difference from walking the capacity and testing.
    //
    // The slot is a number, so the body indexes columns with it directly and
    // pays no generation check: the walk answered that question by finding the
    // bit set. `for rank, slot in live_slots(c)` counts the elements as it goes,
    // which is what compacting into a packed buffer wants.
    fn lower_for_live(
        &mut self,
        variable: &str,
        second: Option<&str>,
        container: ExprId,
        body: Range32,
    ) -> Result<()> {
        // A name, or a field of one. The walk reads the container's liveness
        // where it stands rather than binding it, so a subject that has to be
        // worked out would be worked out once a word.
        if !matches!(
            self.ast.expr(container),
            Expression::Identifier(_) | Expression::FieldAccess(..)
        ) {
            bail!(
                "`live_slots` walks a container that is named, not one that is worked out; bind it first and walk the name"
            );
        }
        // A container by shape rather than by name, the way a slab and a
        // columns container are recognized everywhere else: what makes a walk
        // possible is the record of which slots are filled, and a `Slab<T, N>`
        // carries one as much as a `columns<T, N>` does.
        let struct_name = match self.probe_type(container) {
            Some(Type::Struct(name))
                if self
                    .builder
                    .struct_layout(&name)
                    .and_then(|layout| layout.field(LIVE_WORDS))
                    .is_some() =>
            {
                name
            }
            _ => bail!(
                "`live_slots` walks a generational container, and this is not one; write `live_slots(c)` where `c` is a `columns<T, N>` or a `Slab<T, N>`"
            ),
        };
        let (words_offset, word_count) = {
            let layout =
                self.builder.struct_layout(&struct_name).ok_or_else(|| {
                    anyhow::anyhow!("unknown columns '{struct_name}'")
                })?;
            let words = layout.field(LIVE_WORDS).ok_or_else(|| {
                anyhow::anyhow!("columns has no '{LIVE_WORDS}' field")
            })?;
            let Type::Array(_, count) = &words.ty else {
                bail!("columns '{LIVE_WORDS}' is not an array");
            };
            (words.offset, *count)
        };
        let (struct_address, _) = self.struct_place(container)?;

        // With two names the first counts the elements and the second is the
        // slot, which is the order `for index, name in` already reads in.
        let slot_name = second.unwrap_or(variable);
        let rank = match second {
            Some(_) => {
                let rank = self.fresh_local(Type::I64, None);
                self.emit(IrStatement::Assign(
                    rank,
                    IrRvalue::Use(IrOperand::Constant(IrConstant::Integer(
                        0,
                        Type::I64,
                    ))),
                ));
                Some((variable, rank))
            }
            None => None,
        };

        let word = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(
            word,
            IrRvalue::Use(IrOperand::Constant(IrConstant::Integer(
                0,
                Type::I64,
            ))),
        ));

        let word_header = self.new_block();
        let word_body = self.new_block();
        let sparse_header = self.new_block();
        let sparse_body = self.new_block();
        let word_step = self.new_block();
        let exit = self.new_block();

        self.set_terminator(IrTerminator::Jump(word_header));

        self.switch_to(word_header);
        let more = self.fresh_local(Type::Bool, None);
        self.emit(IrStatement::Assign(
            more,
            IrRvalue::Binary(
                IrBinOp::LessThan,
                IrOperand::Local(word),
                IrOperand::Constant(IrConstant::Integer(
                    word_count as i64,
                    Type::I64,
                )),
            ),
        ));
        self.set_terminator(IrTerminator::Branch {
            condition: IrOperand::Local(more),
            then_block: word_body,
            else_block: exit,
        });

        self.switch_to(word_body);
        let word_address = self.slab_field_element_address(
            struct_address,
            words_offset,
            &Type::I64,
            IrOperand::Local(word),
        );
        let bits = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(
            bits,
            IrRvalue::Load {
                address: word_address,
                ty: Type::I64,
            },
        ));
        let base = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(
            base,
            IrRvalue::Binary(
                IrBinOp::Multiply,
                IrOperand::Local(word),
                IrOperand::Constant(IrConstant::Integer(
                    SLOTS_PER_WORD as i64,
                    Type::I64,
                )),
            ),
        ));
        self.set_terminator(IrTerminator::Jump(sparse_header));

        self.switch_to(sparse_header);
        let any = self.fresh_local(Type::Bool, None);
        self.emit(IrStatement::Assign(
            any,
            IrRvalue::Binary(
                IrBinOp::NotEqual,
                IrOperand::Local(bits),
                IrOperand::Constant(IrConstant::Integer(0, Type::I64)),
            ),
        ));
        self.set_terminator(IrTerminator::Branch {
            condition: IrOperand::Local(any),
            then_block: sparse_body,
            else_block: word_step,
        });

        self.switch_to(sparse_body);
        let bit = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(
            bit,
            IrRvalue::Unary(IrUnOp::TrailingZeros, IrOperand::Local(bits)),
        ));
        // `bits & (bits - 1)` drops the bit just taken. The subtraction wraps
        // because a word whose only live slot is the sixty-fourth is the
        // lowest i64, and taking one from it is the overflow ordinary
        // arithmetic refuses.
        let lower = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(
            lower,
            IrRvalue::Binary(
                IrBinOp::WrappingSubtract,
                IrOperand::Local(bits),
                IrOperand::Constant(IrConstant::Integer(1, Type::I64)),
            ),
        ));
        self.emit(IrStatement::Assign(
            bits,
            IrRvalue::Binary(
                IrBinOp::BitwiseAnd,
                IrOperand::Local(bits),
                IrOperand::Local(lower),
            ),
        ));
        let sparse_slot =
            self.fresh_local(Type::I64, Some(slot_name.to_string()));
        self.emit(IrStatement::Assign(
            sparse_slot,
            IrRvalue::Binary(
                IrBinOp::Add,
                IrOperand::Local(base),
                IrOperand::Local(bit),
            ),
        ));
        self.lower_live_body(
            (slot_name, sparse_slot),
            rank,
            body,
            LoopTargets {
                continue_block: sparse_header,
                break_block: exit,
            },
        )?;
        self.set_terminator(IrTerminator::Jump(sparse_header));

        self.switch_to(word_step);
        self.emit(IrStatement::Assign(
            word,
            IrRvalue::Binary(
                IrBinOp::Add,
                IrOperand::Local(word),
                IrOperand::Constant(IrConstant::Integer(1, Type::I64)),
            ),
        ));
        self.set_terminator(IrTerminator::Jump(word_header));

        self.switch_to(exit);
        Ok(())
    }

    // The body of a live walk, lowered once per path with the slot and the
    // running count in scope. The count is taken and advanced before the body
    // for the same reason the dense path advances its step there.
    fn lower_live_body(
        &mut self,
        slot: (&str, LocalId),
        rank: Option<(&str, LocalId)>,
        body: Range32,
        targets: LoopTargets,
    ) -> Result<()> {
        let (slot_name, slot) = slot;
        self.push_scope();
        self.define_variable(slot_name, slot);
        if let Some((rank_name, rank)) = rank {
            let taken =
                self.fresh_local(Type::I64, Some(rank_name.to_string()));
            self.emit(IrStatement::Assign(
                taken,
                IrRvalue::Use(IrOperand::Local(rank)),
            ));
            self.emit(IrStatement::Assign(
                rank,
                IrRvalue::Binary(
                    IrBinOp::Add,
                    IrOperand::Local(rank),
                    IrOperand::Constant(IrConstant::Integer(1, Type::I64)),
                ),
            ));
            self.define_variable(rank_name, taken);
        }
        self.loops.push(targets);
        self.lower_block(body, None)?;
        self.loops.pop();
        self.pop_scope();
        Ok(())
    }

    fn lower_for(
        &mut self,
        variable: &str,
        second: Option<&str>,
        range: ExprId,
        body: Range32,
    ) -> Result<()> {
        // `live_slots(c)` is the subject of a `for` and nothing else, so it is read
        // here rather than as an expression that could be held or handed on.
        if let Some(container) = live_subject(self.ast, range) {
            return self.lower_for_live(variable, second, container, body);
        }
        let Expression::Range(start, end, inclusive) =
            self.ast.expr(range).clone()
        else {
            return self.lower_for_sequence(variable, second, range, body);
        };
        if let Some(second) = second {
            bail!(
                "a `for` over a range binds one name, and this one names '{variable}' and '{second}'; two names are for walking a sequence, where the first is the position"
            );
        }

        let (start_operand, start_type) =
            self.lower_expression(start, Some(&Type::I64))?;
        let index =
            self.fresh_local(start_type.clone(), Some(variable.to_string()));
        let start_coerced =
            self.coerce(start_operand, &start_type, &start_type)?;
        self.emit(IrStatement::Assign(index, IrRvalue::Use(start_coerced)));

        let (end_operand, end_type) =
            self.lower_expression(end, Some(&start_type))?;
        let end_local = self.fresh_local(end_type.clone(), None);
        let end_coerced = self.coerce(end_operand, &end_type, &start_type)?;
        self.emit(IrStatement::Assign(end_local, IrRvalue::Use(end_coerced)));

        let header = self.new_block();
        let body_block = self.new_block();
        let step_block = self.new_block();
        let exit = self.new_block();

        self.set_terminator(IrTerminator::Jump(header));
        self.switch_to(header);
        let condition = self.fresh_local(Type::Bool, None);
        let compare = if inclusive {
            IrBinOp::LessThanOrEqual
        } else {
            IrBinOp::LessThan
        };
        self.emit(IrStatement::Assign(
            condition,
            IrRvalue::Binary(
                compare,
                IrOperand::Local(index),
                IrOperand::Local(end_local),
            ),
        ));
        self.set_terminator(IrTerminator::Branch {
            condition: IrOperand::Local(condition),
            then_block: body_block,
            else_block: exit,
        });

        self.switch_to(body_block);
        self.push_scope();
        self.define_variable(variable, index);
        self.loops.push(LoopTargets {
            continue_block: step_block,
            break_block: exit,
        });
        self.lower_block(body, None)?;
        self.loops.pop();
        self.pop_scope();
        self.set_terminator(IrTerminator::Jump(step_block));

        self.switch_to(step_block);
        let one =
            IrOperand::Constant(IrConstant::Integer(1, start_type.clone()));
        self.emit(IrStatement::Assign(
            index,
            IrRvalue::Binary(IrBinOp::Add, IrOperand::Local(index), one),
        ));
        self.set_terminator(IrTerminator::Jump(header));

        self.switch_to(exit);
        Ok(())
    }

    // Where an expression is written, for a fault about one the walk is not
    // lowering: an argument weighed against the parameter it lands in is read
    // by the call around it, so nothing has put the argument's own place on.
    fn at_expression(&self, expression: ExprId) -> Position {
        self.ast.position_of(self.ast.expr_span(expression))
    }

    fn lower_expression(
        &mut self,
        expression: ExprId,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        // `{ x = 1, y = 2 }` names its type nowhere, so the type the context
        // expects is what says what it is. Filling it in here covers every
        // position that carries one: an argument, a field, a return, an
        // assignment and an element.
        //
        // A value named under a type is not one of these. It has one spelling
        // and the type is part of it, so a `.Name` is refused here with the
        // type the context expects named, which is the edit the reader makes.
        let expression = match self.ast.expr(expression).clone() {
            Expression::EnumVariantInit(name, variant, _)
                if self.ast.name(name).is_empty() =>
            {
                return locate(
                    Err(refuse_inferred_variant(self.ast, variant, expected)),
                    self.at_expression(expression),
                );
            }
            Expression::StructInit(name, fields)
                if self.ast.name(name).is_empty() =>
            {
                name_inferred_literal(self.ast, expression, fields, expected)?
            }
            _ => expression,
        };
        // An array that names storage becomes a slice of that storage, rather
        // than of a copy of it. Reading the name first and slicing what came
        // back is the same thing for a local, whose value is the storage, and a
        // different thing for a parameter, which param-mode lowering turned
        // into a pointer: reading one copies the caller's array into the frame,
        // so a write through the slice reached the copy and a slice handed back
        // pointed into a frame that was gone. The self-hosted compiler slices
        // in place, and this is what agrees with it.
        if let Some(wanted) = expected
            && let Some(element) = slice_element_wanted(wanted)
            && is_place_expression(self.ast, expression)
            && let Some(Type::Array(held, count)) = self.probe_type(expression)
            && *held == element
        {
            let (address, _) = self.place_address(expression)?;
            let slice = self.build_slice_from_address(address, &held, count);
            // The type the context asked for, which is `str` where it wanted
            // one. Rebuilding `[]u8` from the element instead reported a type
            // the annotation never named.
            return Ok((slice, wanted.clone()));
        }
        match self.ast.expr(expression).clone() {
            Expression::Literal(literal) => {
                self.lower_literal(&literal, expected)
            }
            Expression::Boolean(value) => {
                Ok((IrOperand::Constant(IrConstant::Bool(value)), Type::Bool))
            }
            Expression::Identifier(name) => {
                let name = self.ast.name(name).to_string();
                if let Some(local) = self.resolve_variable(&name) {
                    if self.locals[local].linear {
                        self.emit(IrStatement::Consume(local));
                    }
                    return Ok((
                        IrOperand::Local(local),
                        self.type_of_local(local),
                    ));
                }
                if let Some(signature) = self.builder.signature(&name) {
                    let proc_type = Type::Proc(
                        signature.parameters.clone(),
                        Box::new(signature.return_type.clone()),
                    );
                    let result = self.fresh_local(proc_type.clone(), None);
                    self.emit(IrStatement::Assign(
                        result,
                        IrRvalue::FunctionAddress(name.clone()),
                    ));
                    return Ok((IrOperand::Local(result), proc_type));
                }
                if let Some(value) = self.builder.constants.get(&name).copied()
                {
                    return self.lower_expression(value, expected);
                }
                // Where the name is written, not where the statement is. A
                // statement may name two things that are not there.
                locate(
                    Err(anyhow::anyhow!("unknown variable '{name}'")),
                    self.at_expression(expression),
                )?
            }
            Expression::Function(parameters, return_sig, body)
            | Expression::Proc(parameters, return_sig, body) => {
                let id = self.builder.anon_counter.get();
                self.builder.anon_counter.set(id + 1);
                let name = format!("__anon_{id}");
                let param_types: Vec<Type> = self
                    .ast
                    .params_in(parameters)
                    .iter()
                    .map(parameter_type)
                    .collect();
                let return_type = self
                    .ast
                    .signature_to_type(self.ast.signature(return_sig))
                    .unwrap_or(Type::Void);
                let proc_type = Type::Proc(param_types, Box::new(return_type));
                self.anonymous.push(AnonRequest {
                    name: name.clone(),
                    parameters,
                    return_sig,
                    body,
                    requested_by: 0,
                });
                let result = self.fresh_local(proc_type.clone(), None);
                self.emit(IrStatement::Assign(
                    result,
                    IrRvalue::FunctionAddress(name),
                ));
                Ok((IrOperand::Local(result), proc_type))
            }
            // A negated literal is a literal, and folding it here is what lets
            // it be range-checked at the type it is written at. Left as a
            // negation it was a computed value, so `d : u8 = -1` went round the
            // check and was quietly 255.
            Expression::Prefix(crate::parser::Operator::Negate, operand)
                if matches!(
                    self.ast.expr(operand),
                    Expression::Literal(Literal::Integer(_))
                ) =>
            {
                let Expression::Literal(Literal::Integer(value)) =
                    self.ast.expr(operand)
                else {
                    unreachable!()
                };
                let value = *value;
                self.lower_literal(&Literal::Integer(-value), expected)
            }
            Expression::Prefix(operator, operand) => {
                // The mark and what it is applied to, which is what a fault
                // about the pair is about.
                let at = self.at_expression(expression);
                locate(self.lower_prefix(operator, operand, expected), at)
            }
            Expression::Infix(left, operator, right) => {
                self.lower_infix(left, operator, right, expected)
            }
            Expression::If(condition, consequence, alternative) => {
                self.lower_if(condition, consequence, alternative, expected)
            }
            Expression::Call(callee, arguments) => {
                if let Expression::Identifier(name) = self.ast.expr(callee)
                    && matches!(
                        self.ast.name(*name),
                        "columns_new" | "slab_new"
                    )
                    && self.resolve_variable(self.ast.name(*name)).is_none()
                    && self.builder.signature(self.ast.name(*name)).is_none()
                    && !self
                        .builder
                        .generic_functions
                        .contains_key(self.ast.name(*name))
                {
                    let called = self.ast.name(*name).to_string();
                    return self.lower_columns_new(&called, expected);
                }
                // `live_slots(c)` reaching here is one written somewhere other than
                // after the `in` of a `for`, where it is the subject of the
                // walk. There is no value it could be: the slots it names are
                // walked, never held.
                if let Expression::Identifier(name) = self.ast.expr(callee)
                    && self.ast.name(*name) == "live_slots"
                {
                    bail!(
                        "`live_slots(c)` is the subject of a `for` and nothing else; write `for slot in live_slots(c)`"
                    );
                }
                if let Some(answered) =
                    self.lower_type_builtin(callee, arguments, expected)?
                {
                    return Ok(answered);
                }
                self.lower_call(callee, arguments)
            }
            Expression::Borrow(inner) => {
                self.lower_address_of(inner, RefKind::Ref)
            }
            Expression::BorrowMut(inner) => {
                self.lower_address_of(inner, RefKind::RefMut)
            }
            Expression::AddressOf(inner) => {
                self.lower_address_of(inner, RefKind::Ptr)
            }
            Expression::Dereference(inner) => self.lower_dereference(inner),
            // An `unsafe` block is a block. It changes nothing about the code
            // it holds. It is where `check_unsafety` allows the three unchecked
            // operations, and that check has already run by the time lowering
            // sees this. So the marker is discharged before here and this is a
            // plain block that answers with its last expression.
            Expression::Unsafe(body) => self.lower_block(body, expected),
            Expression::FieldAccess(base, field) => {
                let field = self.ast.name(field).to_string();
                self.lower_field_read(base, &field)
            }
            Expression::Index(base, index) => {
                let (address, element_type) =
                    self.element_address(base, index)?;
                self.load_from(address, element_type)
            }
            Expression::Switch(scrutinee, cases) => {
                self.lower_match(scrutinee, cases, expected)
            }
            Expression::StructInit(struct_name, field_inits) => {
                let struct_name = self.ast.name(struct_name).to_string();
                let ty = match expected {
                    Some(Type::Struct(instance))
                        if is_generic_instance(instance)
                            && instance
                                .starts_with(&format!("{struct_name}<")) =>
                    {
                        Type::Struct(instance.clone())
                    }
                    // A template names no instance, so which one this is comes
                    // off the values written for the fields that declare its
                    // parameters. Taking the template's own name instead typed
                    // the value by the declaration, whose fields are parameters,
                    // and the reader was told a field held a `$T`.
                    _ if self
                        .builder
                        .generic_struct_defs
                        .contains_key(&struct_name) =>
                    {
                        let Some(instance) =
                            self.generic_instance_of(&struct_name, field_inits)
                        else {
                            bail!(
                                "'{struct_name}' is generic and nothing here says which instance this literal is: write the arguments on the literal, as in '{struct_name}<i64> {{ ... }}', or give the binding a declared type that names them"
                            );
                        };
                        Type::Struct(instance)
                    }
                    _ => Type::Struct(struct_name.clone()),
                };
                let temp = self.fresh_local(ty.clone(), None);
                self.materialize_aggregate(temp, expression)?;
                Ok((IrOperand::Local(temp), ty))
            }
            // `InitFlags::Video` reads as a variant and is one bit of a flags
            // type: a constant of that type rather than a value carrying a tag.
            Expression::EnumVariantInit(type_name, bit, fields)
                if self
                    .builder
                    .flags
                    .contains_key(self.ast.name(type_name)) =>
            {
                let type_name = self.ast.name(type_name).to_string();
                let bit = self.ast.name(bit).to_string();
                let layout = &self.builder.flags[&type_name];
                if !fields.is_empty() {
                    bail!(
                        "'{type_name}::{bit}' is a bit of a set, so it carries nothing"
                    );
                }
                let Some(value) = layout.bits.get(&bit).copied() else {
                    bail!("'{type_name}' names no bit called '{bit}'");
                };
                let ty = Type::Distinct(
                    type_name.clone(),
                    Box::new(layout.repr.clone()),
                );
                Ok((
                    IrOperand::Constant(IrConstant::Integer(value, ty.clone())),
                    ty,
                ))
            }
            // `Key::Left` where `Key` names values under itself. The value is
            // its expression, read at the type the declaration gives it, so a
            // number written under a distinct type comes out as that type
            // rather than as the number.
            Expression::EnumVariantInit(type_name, value_name, fields)
                if self.builder.names_a_value(
                    self.ast.name(type_name),
                    self.ast.name(value_name),
                ) =>
            {
                let type_name = self.ast.name(type_name).to_string();
                let value_name = self.ast.name(value_name).to_string();
                if !fields.is_empty() {
                    bail!(
                        "'{type_name}::{value_name}' is a value named under a type, so it carries nothing"
                    );
                }
                let held = &self.builder.type_values[&type_name];
                let declared = held.declared.clone();
                let value = held
                    .values
                    .iter()
                    .find(|(name, _)| name == &value_name)
                    .map(|(_, value)| *value)
                    .expect("a value the type names");
                let (operand, actual) =
                    self.lower_expression(value, Some(&declared))?;
                if actual != declared {
                    bail!(
                        "a value named under a type is a value of that type, so '{type_name}::{value_name}' is a {declared} and this is a {actual}"
                    );
                }
                Ok((operand, declared))
            }
            // A name under a type that names values, where no value and no
            // variant answers to it. The type's own values are the set a near
            // name is looked for in, the way a name that is not a variable is
            // looked for among the names in scope.
            Expression::EnumVariantInit(type_name, value_name, _)
                if self
                    .builder
                    .type_values
                    .contains_key(self.ast.name(type_name))
                    && self
                        .builder
                        .enum_layout(self.ast.name(type_name))
                        .is_none() =>
            {
                let type_name = self.ast.name(type_name).to_string();
                let value_name = self.ast.name(value_name).to_string();
                let held = &self.builder.type_values[&type_name];
                let names: Vec<&str> =
                    held.values.iter().map(|(name, _)| name.as_str()).collect();
                let suggestion =
                    match crate::tools::api::nearest(&value_name, &names) {
                        Some(near) => format!(" (did you mean '{near}'?)"),
                        None => String::new(),
                    };
                bail!(
                    "'{type_name}' names no value called '{value_name}'{suggestion}"
                )
            }
            Expression::EnumVariantInit(enum_name, _, _) => {
                let enum_name = self.ast.name(enum_name).to_string();
                // A generic enum is written `Option::Some { value = 3 }` with no
                // arguments on it, so which instance it is comes from what the
                // context expects, exactly as for a generic struct literal.
                let ty = match expected {
                    Some(Type::Enum(instance) | Type::Struct(instance))
                        if is_generic_instance(instance)
                            && instance
                                .starts_with(&format!("{enum_name}<")) =>
                    {
                        Type::Enum(instance.clone())
                    }
                    _ => Type::Enum(enum_name.clone()),
                };
                let temp = self.fresh_local(ty.clone(), None);
                self.materialize_aggregate(temp, expression)?;
                Ok((IrOperand::Local(temp), ty))
            }
            // A repeat count that no generic bound. Written outside one, or
            // naming something that is not a parameter of the generic it is
            // written in, so there is no number to expand it to.
            Expression::ArrayRepeat(_, count) => {
                bail!(
                    "'{count}' is not a constant or a value parameter, so there is no count for this array literal",
                    count = self.ast.name(count)
                )
            }
            _ => {
                bail!(
                    "unsupported expression: {}",
                    display_expr(self.ast, expression)
                )
            }
        }
    }

    fn lower_literal(
        &mut self,
        literal: &Literal,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        match literal {
            Literal::Integer(value) => {
                let ty = match expected {
                    Some(ty) if ty.is_integer() => ty.clone(),
                    _ => Type::I64,
                };
                // The type the literal is being read at is in hand here, which
                // is the whole of what a range check needs. Nothing used to
                // look, so `a : u8 = 300` was quietly 44.
                if !fits_in(*value, &ty) {
                    let (low, high) = range_of(&ty).expect("integer type");
                    bail!(
                        "{value} does not fit in a {ty}, which holds {low} to {high}"
                    );
                }
                Ok((
                    IrOperand::Constant(IrConstant::Integer(
                        *value,
                        ty.clone(),
                    )),
                    ty,
                ))
            }
            Literal::Float(value) => {
                let ty = match expected {
                    Some(Type::F32) => Type::F32,
                    _ => Type::F64,
                };
                Ok((
                    IrOperand::Constant(IrConstant::Float(*value, ty.clone())),
                    ty,
                ))
            }
            Literal::Float32(value) => Ok((
                IrOperand::Constant(IrConstant::Float(
                    *value as f64,
                    Type::F32,
                )),
                Type::F32,
            )),
            Literal::Boolean(value) => {
                Ok((IrOperand::Constant(IrConstant::Bool(*value)), Type::Bool))
            }
            Literal::String(value) => {
                if matches!(expected, Some(Type::Ptr(_))) {
                    return Ok((
                        IrOperand::Constant(IrConstant::CString(value.clone())),
                        Type::Ptr(Box::new(Type::I8)),
                    ));
                }
                let local = self.fresh_local(Type::Str, None);
                self.build_str_value(local, value);
                Ok((IrOperand::Local(local), Type::Str))
            }
            // An array written where a value is wanted rather than as the
            // initializer of something already declared: a constant named here,
            // an argument, the thing an index is taken of. It goes in a
            // temporary, which is where an array bound to a name lives too, so
            // nothing after this can tell the two apart.
            Literal::Array(elements) => {
                let elements: Vec<ExprId> =
                    self.ast.exprs_in(*elements).to_vec();
                let element = array_element_type(
                    self.ast,
                    expected,
                    &elements,
                    &self.builder.signatures,
                );
                let ty = Type::Array(Box::new(element.clone()), elements.len());
                let temp = self.fresh_local(ty.clone(), None);
                self.init_array(temp, &element, &elements)?;
                Ok((IrOperand::Local(temp), ty))
            }
        }
    }

    fn lower_prefix(
        &mut self,
        operator: Operator,
        operand: ExprId,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        match operator {
            Operator::Negate => {
                let (value, ty) = self.lower_expression(operand, expected)?;
                // A vector negates lane by lane, which is zero minus it, the
                // same shape every other elementwise operation takes.
                if let Some((element, _)) = lanes_of(&ty) {
                    let zero = self.zero_of(&element);
                    if let Some(answered) = self.lower_elementwise(
                        IrBinOp::Subtract,
                        (&zero, &element),
                        (&value, &ty),
                    )? {
                        return Ok(answered);
                    }
                }
                let result = self.fresh_local(ty.clone(), None);
                self.emit(IrStatement::Assign(
                    result,
                    IrRvalue::Unary(IrUnOp::Negate, value),
                ));
                Ok((IrOperand::Local(result), ty))
            }
            Operator::Not => {
                // `!` answers the opposite of a yes or no and takes one. A
                // number is not one: reading `!count` as `count == 0` is a
                // conversion nothing wrote, and a corpus full of `started == 0`
                // over an i64 flag means what it says.
                if let Some(ty) = self.probe_type(operand)
                    && !matches!(through_distinct(&ty), Type::Bool)
                {
                    bail!(
                        "'!' answers the opposite of a yes or no, and this is a '{}'",
                        spelled(&ty)
                    );
                }
                let (value, _) =
                    self.lower_expression(operand, Some(&Type::Bool))?;
                let result = self.fresh_local(Type::Bool, None);
                self.emit(IrStatement::Assign(
                    result,
                    IrRvalue::Unary(IrUnOp::Not, value),
                ));
                Ok((IrOperand::Local(result), Type::Bool))
            }
            other => {
                bail!("unsupported prefix operator: {other}")
            }
        }
    }

    // The tag of an enum value, which sits at offset zero. Comparing two of
    // them is comparing which variant each is.
    fn load_enum_tag(
        &mut self,
        operand: IrOperand,
        ty: &Type,
    ) -> Result<IrOperand> {
        // A borrow already is the address, which is what a parameter of enum
        // type holds.
        if matches!(ty, Type::Ref(_) | Type::RefMut(_) | Type::Ptr(_)) {
            let tag = self.fresh_local(Type::I32, None);
            self.emit(IrStatement::Assign(
                tag,
                IrRvalue::Load {
                    address: operand,
                    ty: Type::I32,
                },
            ));
            return Ok(IrOperand::Local(tag));
        }
        let IrOperand::Local(local) = operand else {
            bail!("enum value is not addressable");
        };
        self.mark_in_memory(local);
        let address = self.address_of_local(local, ty);
        let tag = self.fresh_local(Type::I32, None);
        self.emit(IrStatement::Assign(
            tag,
            IrRvalue::Load {
                address,
                ty: Type::I32,
            },
        ));
        Ok(IrOperand::Local(tag))
    }

    fn lower_infix(
        &mut self,
        left: ExprId,
        operator: Operator,
        right: ExprId,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        if matches!(operator, Operator::And | Operator::Or) {
            return self.lower_logical(left, operator, right);
        }

        let binop = binop_of(operator)?;
        if binop.is_comparison() {
            // A bare number takes its type from what it is compared against,
            // whichever side it is written on. Lowering the left with nothing
            // to go on gives a float literal the widest float there is, and
            // `0.6 == x` then widens an `f32` to compare it against a number
            // no `f32` holds: true for the values a float represents exactly
            // and false for the rest, which is the shape of a wrong answer
            // that looks right in a test written with round numbers.
            let (left_operand, left_type, right_operand, right_type) =
                if is_bare_number(self.ast, left)
                    && !is_bare_number(self.ast, right)
                {
                    let (right_operand, right_type) =
                        self.lower_expression(right, None)?;
                    let (left_operand, left_type) =
                        self.lower_expression(left, Some(&right_type))?;
                    (left_operand, left_type, right_operand, right_type)
                } else {
                    let (left_operand, left_type) =
                        self.lower_expression(left, None)?;
                    let (right_operand, right_type) =
                        self.lower_expression(right, Some(&left_type))?;
                    (left_operand, left_type, right_operand, right_type)
                };
            self.check_flags_operator(
                binop,
                (left, &left_type),
                (right, &right_type),
            )?;
            self.check_distinct_comparison(&left_type, &right_type)?;
            // Two enum values are compared by their tags, which for an enum
            // whose variants carry nothing is the whole value. A variant with
            // fields makes the question ambiguous, since `.Some { value = 1 }`
            // and `.Some { value = 2 }` are the same variant and different
            // values, so that one is a `match` rather than a guess about which
            // was meant.
            // A name reaches here as a struct until something resolves it, so
            // the enum is asked for by name rather than read off the type.
            // A parameter of enum type is a borrow, and a borrow of an enum
            // is still that enum. Without looking through it a comparison
            // against a variant fell through to the ordinary path, which has
            // two aggregates and nothing to compare.
            let compared = match &left_type {
                Type::Ref(inner) | Type::RefMut(inner) => {
                    self.enum_name_of(inner)
                }
                other => self.enum_name_of(other),
            };
            if let Some(name) = compared
                && matches!(binop, IrBinOp::Equal | IrBinOp::NotEqual)
            {
                let Some(layout) = self.builder.enum_layout(&name) else {
                    bail!("unknown enum '{name}'");
                };
                if let Some(carrying) = layout
                    .variants
                    .iter()
                    .find(|variant| !variant.fields.is_empty())
                {
                    let readable =
                        crate::modules::imports::demangle_private_names(&name);
                    bail!(
                        "'{readable}' cannot be compared with == because its variant '.{}' carries fields, so two values of it can be the same variant and different values; match on it instead",
                        carrying.name
                    );
                }
                let left_tag = self.load_enum_tag(left_operand, &left_type)?;
                let right_tag =
                    self.load_enum_tag(right_operand, &right_type)?;
                let result = self.fresh_local(Type::Bool, None);
                self.emit(IrStatement::Assign(
                    result,
                    IrRvalue::Binary(binop, left_tag, right_tag),
                ));
                return Ok((IrOperand::Local(result), Type::Bool));
            }
            // Two vectors are compared lane by lane nowhere: what a
            // comparison answers is one yes or no, and a vector of them is a
            // mask, which is a type this language does not have.
            if lanes_of(&left_type).is_some() || lanes_of(&right_type).is_some()
            {
                bail!(
                    "'{}' is not something two vectors answer; a vector takes '+', '-', '*' and '/', and one of whole numbers takes '%', '&', '|', '<<' and '>>' as well",
                    operator_text(binop)
                );
            }
            let operand_type = unify(&left_type, &right_type);
            let left_final =
                self.coerce(left_operand, &left_type, &operand_type)?;
            let right_final =
                self.coerce(right_operand, &right_type, &operand_type)?;
            let result = self.fresh_local(Type::Bool, None);
            self.emit(IrStatement::Assign(
                result,
                IrRvalue::Binary(binop, left_final, right_final),
            ));
            return Ok((IrOperand::Local(result), Type::Bool));
        }

        // A number written beside a vector is that number in every lane, so it
        // takes the element's type rather than the vector's, whichever side it
        // is written on. Without this a bare `2.0` was an `f64` beside a
        // `[4]f32` and the two had nothing in common.
        // A number written beside a vector is that number in every lane, so it
        // takes the element's type rather than the vector's, whichever side it
        // is written on. Where nothing is expected here the right side is
        // lowered first, since only it can say what the number should be;
        // where something is expected, that is the answer and the order stays
        // as it was.
        let wanted = expected.map(lane_type);
        let (left_operand, left_type, right_operand, right_type) = if expected
            .is_none()
            && is_bare_number(self.ast, left)
            && !is_bare_number(self.ast, right)
        {
            let (right_operand, right_type) =
                self.lower_expression(right, None)?;
            let held = lane_type(&right_type);
            let (left_operand, left_type) =
                self.lower_expression(left, Some(&held))?;
            (left_operand, left_type, right_operand, right_type)
        } else {
            let (left_operand, left_type) =
                self.lower_expression(left, wanted.as_ref())?;
            let held = lane_type(&left_type);
            let (right_operand, right_type) =
                self.lower_expression(right, Some(&held))?;
            (left_operand, left_type, right_operand, right_type)
        };
        self.check_flags_operator(
            binop,
            (left, &left_type),
            (right, &right_type),
        )?;
        if let Some(answered) = self.lower_elementwise(
            binop,
            (&left_operand, &left_type),
            (&right_operand, &right_type),
        )? {
            return Ok(answered);
        }
        let result_type = unify(&left_type, &right_type);
        // An expression built out of literals is a literal, so it folds here
        // and the number that comes out takes the same range check a written
        // one does.
        if let (
            IrOperand::Constant(IrConstant::Integer(left_value, _)),
            IrOperand::Constant(IrConstant::Integer(right_value, _)),
        ) = (&left_operand, &right_operand)
            && result_type.is_integer()
            && let Some(folded) =
                fold_integers(binop, *left_value, *right_value)
        {
            let operand =
                IrOperand::Constant(IrConstant::Integer(folded, Type::I64));
            return Ok((
                self.coerce(operand, &Type::I64, &result_type)?,
                result_type,
            ));
        }
        let left_final = self.coerce(left_operand, &left_type, &result_type)?;
        let right_final =
            self.coerce(right_operand, &right_type, &result_type)?;
        let result = self.fresh_local(result_type.clone(), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Binary(binop, left_final, right_final),
        ));
        Ok((IrOperand::Local(result), result_type))
    }

    // Elementwise arithmetic over a fixed array of numbers, which is what a
    // vector register holds. `a + b` over two `[4]f32` is four adds, so what an
    // element does is what a number does: a float lane rounds as one number
    // does, and an integer lane aborts on overflow and says where.
    //
    // Answers nothing when neither side is a vector, so an ordinary operator
    // goes on meaning what it did.
    fn lower_elementwise(
        &mut self,
        binop: IrBinOp,
        left: (&IrOperand, &Type),
        right: (&IrOperand, &Type),
    ) -> Result<Option<(IrOperand, Type)>> {
        let (left_operand, left_type) = left;
        let (right_operand, right_type) = right;
        let shape = lanes_of(left_type).or_else(|| lanes_of(right_type));
        let Some((element, count)) = shape else {
            return Ok(None);
        };
        let vector = Type::Array(Box::new(element.clone()), count);
        for held in [left_type, right_type] {
            let held = through_borrow(held);
            if held != &vector && held != &element {
                bail!(
                    "{} and {} do not go together; elementwise arithmetic is over two of one vector type, or a vector and a number of its element type",
                    describe_operand(&vector),
                    describe_operand(held)
                );
            }
        }
        self.check_vector_shape(&element, count)?;
        if !matches!(
            binop,
            IrBinOp::Add
                | IrBinOp::Subtract
                | IrBinOp::Multiply
                | IrBinOp::Divide
        ) && !(element.is_integer()
            && matches!(
                binop,
                IrBinOp::Modulo
                    | IrBinOp::BitwiseAnd
                    | IrBinOp::BitwiseOr
                    | IrBinOp::ShiftLeft
                    | IrBinOp::ShiftRight
            ))
        {
            bail!(
                "'{}' is not something two vectors answer; a vector takes '+', '-', '*' and '/', and one of whole numbers takes '%', '&', '|', '<<' and '>>' as well",
                operator_text(binop)
            );
        }
        let width = self.builder.byte_size(&element);
        let result = self.fresh_local(vector.clone(), None);
        self.mark_in_memory(result);
        let left_lanes = self.lane_source(left_operand, left_type, &element)?;
        let right_lanes =
            self.lane_source(right_operand, right_type, &element)?;
        for lane in 0..count {
            let left_value = self.lane_value(&left_lanes, lane, &element)?;
            let right_value = self.lane_value(&right_lanes, lane, &element)?;
            let computed = self.fresh_local(element.clone(), None);
            self.emit(IrStatement::Assign(
                computed,
                IrRvalue::Binary(binop, left_value, right_value),
            ));
            let destination =
                self.fresh_local(Type::Ptr(Box::new(element.clone())), None);
            self.emit(IrStatement::Assign(
                destination,
                IrRvalue::AddressOf {
                    local: result,
                    offset: lane * width,
                },
            ));
            self.emit(IrStatement::Store {
                address: IrOperand::Local(destination),
                value: IrOperand::Local(computed),
            });
        }
        Ok(Some((IrOperand::Local(result), vector)))
    }

    // The number nothing, at a given type, which is what a lane is subtracted
    // from to negate it.
    fn zero_of(&self, ty: &Type) -> IrOperand {
        if ty.is_float() {
            return IrOperand::Constant(IrConstant::Float(0.0, ty.clone()));
        }
        IrOperand::Constant(IrConstant::Integer(0, ty.clone()))
    }

    // What a vector has to be for its lanes to be a register's worth: a length
    // that is a power of two, and a width a machine has a register for.
    fn check_vector_shape(&self, element: &Type, count: usize) -> Result<()> {
        if !count.is_power_of_two() {
            bail!(
                "elementwise arithmetic is over a vector whose length is a power of two, and {count} is not one"
            );
        }
        let bytes = self.builder.byte_size(element) * count;
        if bytes > VECTOR_LIMIT {
            bail!(
                "elementwise arithmetic is over a vector of at most {VECTOR_LIMIT} bytes, which is a register's worth, and this one is {bytes}"
            );
        }
        Ok(())
    }

    // Where a side's lanes are read from: the address of its storage, or the
    // number itself where a number stands for every lane.
    fn lane_source(
        &mut self,
        operand: &IrOperand,
        ty: &Type,
        element: &Type,
    ) -> Result<LaneSource> {
        if through_borrow(ty) == element {
            return Ok(LaneSource::Broadcast(operand.clone()));
        }
        match ty {
            Type::Array(..) => {
                let IrOperand::Local(local) = operand else {
                    bail!(
                        "a vector is read out of a place, and this is not one"
                    );
                };
                let address = self.address_of_local(*local, ty);
                Ok(LaneSource::At(address))
            }
            Type::Ref(inner) | Type::RefMut(inner) | Type::Ptr(inner)
                if matches!(**inner, Type::Array(..)) =>
            {
                Ok(LaneSource::At(operand.clone()))
            }
            held => bail!("'{held}' is not a vector"),
        }
    }

    fn lane_value(
        &mut self,
        source: &LaneSource,
        lane: usize,
        element: &Type,
    ) -> Result<IrOperand> {
        match source {
            LaneSource::Broadcast(operand) => Ok(operand.clone()),
            LaneSource::At(address) => {
                let width = self.builder.byte_size(element);
                let at = self
                    .fresh_local(Type::Ptr(Box::new(element.clone())), None);
                self.emit(IrStatement::Assign(
                    at,
                    IrRvalue::FieldAddress {
                        base: address.clone(),
                        offset: lane * width,
                    },
                ));
                let (value, _) =
                    self.load_from(IrOperand::Local(at), element.clone())?;
                Ok(value)
            }
        }
    }

    // The flags type either side of an operator names, looking through a
    // borrow the way an enum comparison does.
    fn flags_name_of<'t>(&self, ty: &'t Type) -> Option<&'t str> {
        match ty {
            Type::Distinct(name, _)
                if self.builder.flags.contains_key(name) =>
            {
                Some(name)
            }
            Type::Ref(inner) | Type::RefMut(inner) => self.flags_name_of(inner),
            _ => None,
        }
    }

    // A comparison reads both sides, and a distinct type read as its
    // representation is the thing its declaration says it is not. The rule is
    // asked of a binding, a return, an argument and an assignment; this was the
    // site it was missing from, so `n == Key::Left` with `n` an `i64` was a
    // question anyone could ask and nothing answered it.
    //
    // A written number is exempt for the reason it is exempt everywhere: it
    // takes the other side's type before the two are compared, so by here they
    // agree. A flags type says this in its own words and has already spoken.
    fn check_distinct_comparison(
        &self,
        left: &Type,
        right: &Type,
    ) -> Result<()> {
        let left = through_borrow(left);
        let right = through_borrow(right);
        if left == right {
            return Ok(());
        }
        let named = match (left, right) {
            (Type::Distinct(name, _), _) | (_, Type::Distinct(name, _)) => name,
            _ => return Ok(()),
        };
        if self.builder.flags.contains_key(named) {
            return Ok(());
        }
        let readable = crate::modules::imports::demangle_private_names(named);
        bail!(
            "'{readable}' is compared only with itself, and this is a '{}' against a '{}'",
            spelled(left),
            spelled(right)
        )
    }

    // A set of bits answers to union, intersection and whether it is the same
    // set. Adding two of them, or ordering them, or shifting one along, is a
    // question about the number underneath rather than about the set, and the
    // declaration exists to say that the number is not what this is. Reading
    // one as its representation is still allowed, so a program that means the
    // arithmetic writes the conversion and gets it.
    fn check_flags_operator(
        &self,
        binop: IrBinOp,
        left: (ExprId, &Type),
        right: (ExprId, &Type),
    ) -> Result<()> {
        let (left, left_type) = left;
        let (right, right_type) = right;
        let named = self
            .flags_name_of(left_type)
            .or_else(|| self.flags_name_of(right_type));
        let Some(name) = named else {
            return Ok(());
        };
        let readable = crate::modules::imports::demangle_private_names(name);
        if !matches!(
            binop,
            IrBinOp::BitwiseOr
                | IrBinOp::BitwiseAnd
                | IrBinOp::Equal
                | IrBinOp::NotEqual
        ) {
            bail!(
                "'{readable}' is a set of bits, and '{}' is not something two sets answer; combine them with '|', narrow them with '&', and compare them with '==' or 'flags_has'",
                operator_text(binop)
            );
        }
        // A written number takes the other side's type, so by the time the two
        // are compared they agree and what was written is the question.
        if matches!(
            self.ast.expr(left),
            Expression::Literal(Literal::Integer(_))
        ) || matches!(
            self.ast.expr(right),
            Expression::Literal(Literal::Integer(_))
        ) {
            bail!(
                "'{readable}' is a set of bits, built from the names declared under it, and a number is not one of them"
            );
        }
        // Two sets combine when they are the same set. Otherwise the answer
        // would be a number wearing one of the two names.
        if left_type != right_type {
            bail!(
                "'{readable}' combines only with itself, and this is a '{}' against a '{}'",
                spelled(left_type),
                spelled(right_type)
            );
        }
        Ok(())
    }

    fn lower_logical(
        &mut self,
        left: ExprId,
        operator: Operator,
        right: ExprId,
    ) -> Result<(IrOperand, Type)> {
        let result = self.fresh_local(Type::Bool, None);
        // Each side is read the way an operand of any other operator is, which
        // for a borrow of a yes or no is the value it borrows. Left alone, the
        // branch tested the address the borrow holds, which is never zero, and
        // the assignment of the other side put an address in a `bool`.
        let (left_value, left_type) =
            self.lower_expression(left, Some(&Type::Bool))?;
        let left_read = unify(&left_type, &Type::Bool);
        let left_operand = self.coerce(left_value, &left_type, &left_read)?;

        let evaluate_right = self.new_block();
        let shortcut = self.new_block();
        let merge = self.new_block();

        match operator {
            Operator::And => self.set_terminator(IrTerminator::Branch {
                condition: left_operand,
                then_block: evaluate_right,
                else_block: shortcut,
            }),
            _ => self.set_terminator(IrTerminator::Branch {
                condition: left_operand,
                then_block: shortcut,
                else_block: evaluate_right,
            }),
        }

        self.switch_to(evaluate_right);
        let (right_value, right_type) =
            self.lower_expression(right, Some(&Type::Bool))?;
        let right_read = unify(&right_type, &Type::Bool);
        let right_operand =
            self.coerce(right_value, &right_type, &right_read)?;
        self.emit(IrStatement::Assign(result, IrRvalue::Use(right_operand)));
        self.set_terminator(IrTerminator::Jump(merge));

        self.switch_to(shortcut);
        let shortcut_value = matches!(operator, Operator::Or);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Use(IrOperand::Constant(IrConstant::Bool(
                shortcut_value,
            ))),
        ));
        self.set_terminator(IrTerminator::Jump(merge));

        self.switch_to(merge);
        Ok((IrOperand::Local(result), Type::Bool))
    }

    fn lower_if(
        &mut self,
        condition: ExprId,
        consequence: Range32,
        alternative: Option<Range32>,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        let (condition_operand, _) =
            self.lower_expression(condition, Some(&Type::Bool))?;

        let then_block = self.new_block();
        let else_block = self.new_block();
        let merge = self.new_block();

        self.set_terminator(IrTerminator::Branch {
            condition: condition_operand,
            then_block,
            else_block,
        });

        self.switch_to(then_block);
        let (then_value, then_type) =
            self.lower_block(consequence, expected)?;

        let result_type = match expected {
            Some(ty) if !matches!(ty, Type::Void) => ty.clone(),
            _ => then_type.clone(),
        };
        let produces_value =
            !matches!(result_type, Type::Void) && alternative.is_some();

        let result = if produces_value {
            Some(self.fresh_local(result_type.clone(), None))
        } else {
            None
        };

        // A branch that ends in a statement answers with nothing, and an `if`
        // answers with a value only when both of its branches do. The
        // assignment is skipped for a branch that answered with nothing rather
        // than coercing a unit into the other branch's type, which for a struct
        // is not a value at all: `if (c) { spawn(w) } else { g() }` is written
        // for what it does, and what it would have yielded is read by nothing.
        //
        // An assignment goes in before the branch is terminated, since a
        // statement emitted after a terminator starts a block nothing jumps to.
        let mut then_answered = true;
        if let Some(result_local) = result {
            if matches!(then_type, Type::Void) {
                then_answered = false;
            } else {
                let coerced =
                    self.coerce(then_value, &then_type, &result_type)?;
                self.emit(IrStatement::Assign(
                    result_local,
                    IrRvalue::Use(coerced),
                ));
            }
        }
        self.set_terminator(IrTerminator::Jump(merge));

        self.switch_to(else_block);
        let mut else_answered = true;
        if let Some(alternative) = alternative {
            let (else_value, else_type) =
                self.lower_block(alternative, expected)?;
            if let Some(result_local) = result {
                if matches!(else_type, Type::Void) {
                    else_answered = false;
                } else {
                    let coerced =
                        self.coerce(else_value, &else_type, &result_type)?;
                    self.emit(IrStatement::Assign(
                        result_local,
                        IrRvalue::Use(coerced),
                    ));
                }
            }
        }
        self.set_terminator(IrTerminator::Jump(merge));

        self.switch_to(merge);
        match result {
            Some(result_local) if then_answered && else_answered => {
                Ok((IrOperand::Local(result_local), result_type))
            }
            _ => Ok((unit_operand(), Type::Void)),
        }
    }

    // There is no trampoline: the handler's context parameter is `mut`, so
    // it is already a pointer in the signature, and a Frost function and a C
    // one use the same calling convention. So `on_event` compiled for Frost is
    // bit for bit the `void (*)(void*, ...)` the library wants, and the cast the
    // design set out to hide inside generated code turns out not to exist. What
    // is left is passing the handler's address and the context's address.
    fn lower_registration_call(
        &mut self,
        name: &str,
        arguments: &[ExprId],
    ) -> Result<(IrOperand, Type)> {
        let shape = self
            .builder
            .registrations
            .get(name)
            .cloned()
            .expect("a registration the caller just looked up");
        let mut rewritten = arguments.to_vec();
        let Some(handler) = rewritten.get(shape.handler).copied() else {
            bail!(
                "'{name}' registers a callback and needs one as its argument {}",
                shape.handler + 1
            );
        };
        let Expression::TypeValue(Type::Struct(handler_name)) =
            self.ast.expr(handler)
        else {
            bail!(
                "argument {} of '{name}' is the callback and has to be written '$name'",
                shape.handler + 1
            );
        };
        let handler_name = handler_name.clone();
        let handler_span = self.ast.expr_span(handler);
        let handler_symbol = self.ast.intern(&handler_name);
        rewritten[shape.handler] = self
            .ast
            .push_expr(Expression::Identifier(handler_symbol), handler_span);
        let Some(context) = rewritten.get(shape.context).copied() else {
            bail!(
                "'{name}' registers a callback and needs its context as argument {}",
                shape.context + 1
            );
        };
        let context_span = self.ast.expr_span(context);
        let ptr_to_symbol = self.ast.intern("ptr_to");
        let callee = self
            .ast
            .push_expr(Expression::Identifier(ptr_to_symbol), context_span);
        let call_arguments = self.ast.add_expr_list(&[context]);
        rewritten[shape.context] = self
            .ast
            .push_expr(Expression::Call(callee, call_arguments), context_span);
        self.lower_direct_call(name, &rewritten)
    }

    // `sizeof(T)`, `alignof(T)`, `typename(T)` and `type_id(T)` are calls the
    // parser committed to these names, carrying the type as their one
    // argument. Each is a constant the compiler already knows, answered here so
    // nothing downstream sees a call.
    fn lower_type_builtin(
        &mut self,
        callee: ExprId,
        arguments: Range32,
        expected: Option<&Type>,
    ) -> Result<Option<(IrOperand, Type)>> {
        let Expression::Identifier(name) = self.ast.expr(callee) else {
            return Ok(None);
        };
        let name = self.ast.name(*name);
        if !matches!(name, "sizeof" | "alignof" | "typename" | "type_id") {
            return Ok(None);
        }
        let name = name.to_string();
        let arguments = self.ast.exprs_in(arguments);
        if arguments.len() != 1 {
            return Ok(None);
        }
        let Expression::TypeValue(ty) = self.ast.expr(arguments[0]) else {
            return Ok(None);
        };
        // `$P` on a concrete type reads as a constant named at a call, since
        // that is what a `$` argument is everywhere else. Where the name is a
        // type the program declared, it is that type: `sizeof($P)` measured
        // zero and `sizeof(P)` measured sixteen for the same struct.
        let ty = match ty.clone() {
            Type::TypeParam(named) | Type::ConstValue(named)
                if self.builder.struct_layout(&named).is_some() =>
            {
                Type::Struct(named)
            }
            Type::TypeParam(named) | Type::ConstValue(named)
                if self.builder.enum_layout(&named).is_some() =>
            {
                Type::Enum(named)
            }
            held => held,
        };
        Ok(Some(match name.as_str() {
            "sizeof" => {
                // A type nothing was laid out for measured zero, which a
                // program cannot tell from a real zero. `sizeof($P)` on an
                // ordinary struct answered 0 while `sizeof(P)` answered 16.
                let Some(size) = self.builder.measured_size(&ty) else {
                    bail!(
                        "`sizeof` has no layout for '{ty}', so there is no width to give"
                    );
                };
                let size = size as i64;
                (
                    IrOperand::Constant(IrConstant::Integer(size, Type::I64)),
                    Type::I64,
                )
            }
            // What the type is aligned to, which an allocator handing out a run
            // of it has to start that run on. Refused for the same reason
            // `sizeof` is: a type nothing was laid out for would answer zero,
            // and a caller dividing by it would divide by zero rather than be
            // told the type had no layout.
            "alignof" => {
                let Some(align) = self.builder.measured_align(&ty) else {
                    bail!(
                        "`alignof` has no layout for '{ty}', so there is no alignment to give"
                    );
                };
                let align = align as i64;
                (
                    IrOperand::Constant(IrConstant::Integer(align, Type::I64)),
                    Type::I64,
                )
            }
            "type_id" => {
                let id = self.builder.type_id(&ty);
                (
                    IrOperand::Constant(IrConstant::Integer(id, Type::I64)),
                    Type::I64,
                )
            }
            _ => {
                let written = crate::modules::imports::demangle_private_names(
                    &ty.to_string(),
                );
                if matches!(expected, Some(Type::Ptr(_))) {
                    (
                        IrOperand::Constant(IrConstant::CString(written)),
                        Type::Ptr(Box::new(Type::I8)),
                    )
                } else {
                    let local = self.fresh_local(Type::Str, None);
                    self.build_str_value(local, &written);
                    (IrOperand::Local(local), Type::Str)
                }
            }
        }))
    }

    fn lower_call(
        &mut self,
        callee: ExprId,
        arguments: Range32,
    ) -> Result<(IrOperand, Type)> {
        let arguments: Vec<ExprId> = self.ast.exprs_in(arguments).to_vec();
        let callee_name = match self.ast.expr(callee) {
            Expression::Identifier(name) => {
                Some(self.ast.name(*name).to_string())
            }
            _ => None,
        };
        if let Some(name) = &callee_name
            && name == "assert"
            && self.resolve_variable(name).is_none()
            && (self.builder.signature("frost_rt_assert_at").is_some()
                || self.builder.signature("frost_rt_assert").is_some())
        {
            // The position is the reader's line, and the runtime prints it, so
            // a failed assertion names where it was written rather than only
            // which test it was in. Programs that declare the older one-argument
            // `frost_rt_assert` themselves still work.
            if self.builder.signature("frost_rt_assert_at").is_some() {
                let mut located = arguments.clone();
                let span = self.ast.expr_span(callee);
                let described = self.current_position.describe();
                located.push(self.ast.push_expr(
                    Expression::Literal(Literal::String(described)),
                    span,
                ));
                return self.lower_direct_call("frost_rt_assert_at", &located);
            }
            return self.lower_direct_call("frost_rt_assert", &arguments);
        }
        if let Some(name) = &callee_name
            && name == "str_len"
            && self.resolve_variable(name).is_none()
            && self.builder.signature(name).is_none()
            && !self.builder.generic_functions.contains_key(name)
        {
            return self.lower_str_len(&arguments);
        }
        if let Some(name) = &callee_name
            && name == "slice_len"
            && self.resolve_variable(name).is_none()
            && self.builder.signature(name).is_none()
            && !self.builder.generic_functions.contains_key(name)
        {
            return self.lower_slice_len(&arguments);
        }
        if let Some(name) = &callee_name
            && name == "flags_has"
            && self.resolve_variable(name).is_none()
            && self.builder.signature(name).is_none()
            && !self.builder.generic_functions.contains_key(name)
        {
            return self.lower_flags_has(&arguments);
        }
        if let Some(name) = &callee_name
            && self.resolve_variable(name).is_none()
            && self.builder.signature(name).is_none()
            && !self.builder.generic_functions.contains_key(name)
        {
            match name.as_str() {
                "ptr_to" => return self.lower_ptr_to(&arguments),
                "cast" => return self.lower_cast(&arguments),
                "ptr_cast" => return self.lower_ptr_cast(&arguments),
                "slice_from" => return self.lower_slice_from(&arguments),
                "wrap_add" => {
                    return self
                        .lower_wrapping(IrBinOp::WrappingAdd, &arguments);
                }
                "wrap_sub" => {
                    return self
                        .lower_wrapping(IrBinOp::WrappingSubtract, &arguments);
                }
                "wrap_mul" => {
                    return self
                        .lower_wrapping(IrBinOp::WrappingMultiply, &arguments);
                }
                _ => {}
            }
        }
        if let Some(name) = &callee_name
            && self.resolve_variable(name).is_none()
        {
            if self.builder.generic_functions.contains_key(name) {
                return self.lower_generic_call(name, callee, &arguments);
            }
            if self.builder.registrations.contains_key(name) {
                return self.lower_registration_call(name, &arguments);
            }
            if self.builder.signature(name).is_some() {
                return self.lower_direct_call(name, &arguments);
            }
        }
        if let Some(target) = self.bundle_field_function(callee) {
            return self.lower_direct_call(&target, &arguments);
        }
        // A bare name in callee position that is neither a variable nor a
        // function is a call to something that is not there, and saying it is
        // an unknown variable describes a reading of the line nobody wrote.
        if let Some(name) = &callee_name
            && self.resolve_variable(name).is_none()
            && !self.builder.constants.contains_key(name)
        {
            return locate(
                Err(anyhow::anyhow!("call to undefined function '{name}'")),
                self.at_expression(callee),
            );
        }
        self.lower_indirect_call(callee, &arguments)
    }

    // Whether `Type::Name`, or a `.Name` an annotation names the type of,
    // belongs to the values a type names under itself rather than to its
    // variants. A binding of one holds that value, at the type its declaration
    // gives it, so it is bound the way every other value is.
    //
    // A name a type with such a block does not name answers here too, wherever
    // the type has no variants for it to be one of. What it is is settled
    // where every other name under a type is, which is what puts the near name
    // in the reader's hands instead of a complaint about an enum.
    fn binds_a_named_value(
        &self,
        annotation: Option<&Type>,
        type_name: &str,
        value_name: &str,
    ) -> bool {
        let named = if type_name.is_empty() {
            match annotation {
                Some(
                    Type::Enum(named)
                    | Type::Struct(named)
                    | Type::Distinct(named, _),
                ) => named.as_str(),
                _ => return false,
            }
        } else {
            type_name
        };
        if self.builder.names_a_value(named, value_name) {
            return true;
        }
        self.builder.type_values.contains_key(named)
            && self.builder.enum_layout(named).is_none()
    }

    // The function a bundle's field names, for a bundle that is a constant.
    // `ops.less(a, b)` where `ops` is a constant whose `less` field names a
    // function is a call to that function: there is one value the field can
    // hold and it is known here, so nothing is loaded and nothing is called
    // through a pointer.
    fn bundle_field_function(&self, callee: ExprId) -> Option<String> {
        let Expression::FieldAccess(base, field) = self.ast.expr(callee) else {
            return None;
        };
        let Expression::Identifier(name) = self.ast.expr(*base) else {
            return None;
        };
        let name = self.ast.name(*name);
        if self.resolve_variable(name).is_some() {
            return None;
        }
        let constant = self.builder.constants.get(name).copied()?;
        let Expression::StructInit(_, fields) = self.ast.expr(constant) else {
            return None;
        };
        let entry = self
            .ast
            .named_in(*fields)
            .iter()
            .find(|held| held.name == *field)?;
        let Expression::Identifier(target) = self.ast.expr(entry.value) else {
            return None;
        };
        let target = self.ast.name(*target);
        self.builder
            .signature(target)
            .is_some()
            .then(|| target.to_string())
    }

    /// Bind one compile-time parameter to what the call gives it, or to the
    /// default its declaration carries where the call gives nothing. The two
    /// arrive at the same place: a `$T` argument and a `= Heap` default are
    /// both a type written where a type belongs.
    fn lower_generic_call(
        &mut self,
        name: &str,
        callee: ExprId,
        arguments: &[ExprId],
    ) -> Result<(IrOperand, Type)> {
        let generic = self
            .builder
            .generic_functions
            .get(name)
            .expect("generic function exists")
            .clone();
        let generic_parameters: Vec<Parameter> =
            self.ast.params_in(generic.parameters).to_vec();

        // A compile-time list takes every argument past the parameters written
        // before it, so a call may give more arguments than there are
        // parameters, and one fewer when the list is empty.
        let packed = generic_parameters.iter().any(|parameter| parameter.pack);
        let fixed = generic_parameters.len() - usize::from(packed);

        // Which argument each parameter written before the list takes. A
        // compile-time parameter that a value parameter settles takes none: it
        // is bound by unifying that parameter's declared type against the
        // argument's, further down. `None` is that case, and it is why
        // `vec_push(v, 3)` is the whole call.
        let settled = settled_by(self.ast, &generic_parameters);
        let wanted = generic_parameters.len()
            - settled.iter().filter(|held| held.is_some()).count();

        // A call writing a compile-time argument the signature settles says
        // twice what the argument says once. Counted rather than matched
        // against a position: a written argument lands on whichever
        // compile-time parameter is still open, and the count is what says one
        // too many arrived.
        if !packed
            && let Some((held, by)) = generic_parameters
                .iter()
                .zip(&settled)
                .find_map(|(held, by)| by.map(|by| (held, by)))
        {
            let open = generic_parameters
                .iter()
                .zip(&settled)
                .filter(|(parameter, by)| {
                    is_type_parameter(self.ast, parameter) && by.is_none()
                })
                .count();
            let written = arguments
                .iter()
                .filter(|argument| {
                    matches!(
                        self.ast.expr(**argument),
                        Expression::TypeValue(_)
                    )
                })
                .count();
            if written > open {
                bail!(
                    "'{}' of '{name}' is settled by the type of '{}', so it is not written at the call",
                    self.ast.name(held.name),
                    self.ast.name(by)
                );
            }
        }

        // Which argument each parameter written before the list takes. Two
        // parameters take none: one a value parameter settles, and a
        // compile-time one with a declared default the call wrote no `$`
        // argument for. The second is aligned by the sigil rather than by
        // counting, since a `$` argument binds a compile-time parameter and
        // anything else binds a value one.
        let mut aligned: Vec<Option<usize>> = Vec::with_capacity(fixed);
        let mut consumed = 0usize;
        for (index, parameter) in
            generic_parameters.iter().take(fixed).enumerate()
        {
            if settled[index].is_some() {
                aligned.push(None);
                continue;
            }
            if parameter.compile_time_default.is_some()
                && !arguments.get(consumed).is_some_and(|argument| {
                    matches!(self.ast.expr(*argument), Expression::TypeValue(_))
                })
            {
                aligned.push(None);
                continue;
            }
            if consumed >= arguments.len() {
                bail!(
                    "generic function '{name}' expects {wanted} argument(s) but {} were given",
                    arguments.len()
                );
            }
            aligned.push(Some(consumed));
            consumed += 1;
        }
        if (packed && consumed > arguments.len())
            || (!packed && consumed != arguments.len())
        {
            bail!(
                "generic function '{name}' expects {wanted} argument(s) but {} were given",
                arguments.len()
            );
        }

        enum ArgPlan {
            Value(IrOperand, Type),
            Borrow(usize),
        }

        let mut subst: HashMap<String, Type> = HashMap::new();
        let mut plans: Vec<ArgPlan> = Vec::new();
        // Checked after the loop rather than inside it. A declared signature
        // may name type parameters that other arguments bind, and value
        // arguments are what bind most of them, so `subst` is not complete
        // until every argument has been walked.
        let mut signature_checks: Vec<(&Parameter, String)> = Vec::new();
        // The same, for a parameter declared with a bundle type: the constant
        // the argument names has to be of that type.
        let mut bundle_checks: Vec<(&Parameter, String)> = Vec::new();
        for (parameter, slot) in generic_parameters.iter().zip(&aligned) {
            // The list is last, and what it took is lowered below. Nothing
            // after it is a parameter of its own.
            if parameter.pack {
                break;
            }
            // A compile-time parameter takes no argument in two cases, and
            // they end differently. One a value parameter settles is bound by
            // the walk of that parameter, so there is nothing to do here; one
            // the call wrote nothing for stands for the default its declaration
            // gave it, which lands where a written argument would.
            if is_type_parameter(self.ast, parameter) {
                let ty = match slot {
                    Some(index) => {
                        let Expression::TypeValue(ty) =
                            self.ast.expr(arguments[*index])
                        else {
                            bail!(
                                "type parameter '{parameter_name}' of '{name}' requires a type argument like '${parameter_name}'",
                                parameter_name = self.ast.name(parameter.name)
                            );
                        };
                        ty.clone()
                    }
                    None => match &parameter.compile_time_default {
                        Some(default) => default.clone(),
                        None => continue,
                    },
                };
                // `$f` where f is a function rather than a type is a
                // compile-time function argument. It reads as a named type
                // here, so which one it is comes from whether the name is a
                // function this program declares.
                let bound = match &ty {
                    Type::Struct(named)
                        if self.builder.signature(named).is_some() =>
                    {
                        Type::ConstFn(named.clone())
                    }
                    // A name that is a constant is that constant. This is how a
                    // capability bundle travels: the body names the constant
                    // wherever it named the parameter.
                    Type::Struct(named)
                        if self.builder.constants.contains_key(named) =>
                    {
                        Type::ConstValue(named.clone())
                    }
                    // A name that is neither a declared type nor a declared
                    // function is caught here rather than deep inside the
                    // specialized body, where the reader would be looking at
                    // code they did not write.
                    Type::Struct(named)
                        if self.builder.struct_layout(named).is_none()
                            && self.builder.enum_layout(named).is_none()
                            && !is_generic_instance(named) =>
                    {
                        bail!(
                            "'{named}' given to '{}' as the compile-time argument '{}' names neither a type nor a function",
                            name,
                            self.ast.name(parameter.name)
                        );
                    }
                    other => other.clone(),
                };
                match parameter.compile_time_signature.as_ref() {
                    Some(Type::Proc(..)) => {
                        let Type::ConstFn(target) = &bound else {
                            bail!(
                                "'{}' of '{name}' is declared as a function, so it needs a function as its argument, not the type '{}'",
                                self.ast.name(parameter.name),
                                bound
                            );
                        };
                        signature_checks.push((parameter, target.clone()));
                    }
                    Some(_) => {
                        let Type::ConstValue(target) = &bound else {
                            bail!(
                                "'{}' of '{name}' is declared as a bundle, so it needs a constant of that type as its argument, not '{}'",
                                self.ast.name(parameter.name),
                                bound
                            );
                        };
                        bundle_checks.push((parameter, target.clone()));
                    }
                    None => {}
                }
                subst.insert(self.ast.name(parameter.name).to_string(), bound);
                continue;
            }
            // A value parameter always takes an argument: only a compile-time
            // one stands for anything the call did not write.
            let Some(index) = *slot else {
                continue;
            };
            let argument = &arguments[index];
            // A read-mode `$T` parameter became `Ref(T)` before `T` was known.
            // Had `T` been written out as a copy type it would have stayed a
            // value, so once the argument says it is one, this parameter is a
            // value too. Otherwise the body ends up holding a reference where
            // the concrete function would hold the value, which is a type error
            // the moment it is stored anywhere.
            let param_ty = match parameter_type(parameter) {
                Type::Ref(inner)
                    if matches!(
                        inner.as_ref(),
                        Type::TypeParam(param)
                            if generic.type_params.contains(param)
                    ) && argument_is_copy_value(
                        self.probe_type(*argument).as_ref(),
                        self.ast,
                        *argument,
                    ) =>
                {
                    *inner
                }
                other => other,
            };
            // Auto-borrow. A value place passed to a `read`/`mut` reference
            // parameter has its address taken. An argument that is already a
            // reference is forwarded as-is. The type parameter is inferred from
            // the pointee against the place's type.
            if let Type::Ref(inner) | Type::RefMut(inner) = &param_ty {
                let already_reference = matches!(
                    self.ast.expr(*argument),
                    Expression::Borrow(_)
                        | Expression::BorrowMut(_)
                        | Expression::AddressOf(_)
                ) || matches!(
                    self.probe_type(*argument),
                    Some(Type::Ref(_) | Type::RefMut(_) | Type::Ptr(_))
                );
                if already_reference {
                    let (operand, value_type) =
                        self.lower_expression(*argument, None)?;
                    infer_subst_into(
                        &param_ty,
                        &value_type,
                        &generic.type_params,
                        &mut subst,
                    );
                    plans.push(ArgPlan::Value(operand, value_type));
                } else {
                    // A call is not a place, so what it answers with is read
                    // off its signature. Without it a `mut` parameter bound
                    // nothing from an argument that was itself a call, and the
                    // type parameter that argument settles stayed a name.
                    if let Some(place_type) = self.answer_type(*argument) {
                        infer_subst_into(
                            inner,
                            &place_type,
                            &generic.type_params,
                            &mut subst,
                        );
                    }
                    plans.push(ArgPlan::Borrow(index));
                }
            } else {
                // The declared parameter type with everything bound so far
                // substituted into it. That is what tells a bare generic
                // literal which instance it is: without it
                // `Pair { first = 3, second = 4 }` lowers as the template
                // `Pair`, which has no layout, rather than as `Pair<i64>`.
                //
                // Only when the result is concrete. A type argument written
                // after the value it parameterizes has not been bound yet, and
                // an expected type still naming a type parameter would say less
                // than nothing.
                let substituted = substitute_type(&param_ty, &subst);
                let mut unresolved = Vec::new();
                collect_type_params(&substituted, &mut unresolved);
                let expected = unresolved.is_empty().then_some(substituted);
                let (operand, value_type) =
                    self.lower_expression(*argument, expected.as_ref())?;
                infer_subst_into(
                    &param_ty,
                    &value_type,
                    &generic.type_params,
                    &mut subst,
                );
                plans.push(ArgPlan::Value(operand, value_type));
            }
        }

        // A bundle a value parameter settles arrives as the name unified out of
        // that parameter's type, and the body names the constant wherever it
        // names the parameter. This is what a written `$ops` argument becomes
        // further up, reached from the other direction.
        for (parameter, held) in generic_parameters.iter().zip(&settled) {
            if held.is_none() || parameter.compile_time_signature.is_none() {
                continue;
            }
            let parameter_name = self.ast.name(parameter.name).to_string();
            let Some(Type::Struct(named) | Type::TypeParam(named)) =
                subst.get(&parameter_name)
            else {
                continue;
            };
            if !self.builder.constants.contains_key(named) {
                continue;
            }
            let bound = Type::ConstValue(named.clone());
            subst.insert(parameter_name, bound);
        }

        // The bound, before the body is specialized, so a type that cannot
        // work is refused here rather than inside code the reader never wrote.
        //
        // At the call, which is where the reader chose the type the bound is
        // asked about. The statement holding it may hold several.
        let signature = self.ast.signature(generic.return_sig).clone();
        locate(
            check_bound(
                self.ast,
                &signature,
                &subst,
                name,
                Bounding {
                    bounds: &self.builder.bound_functions,
                    linear: &self.builder.linear,
                    structs: &self.builder.structs,
                    enums: &self.builder.enums,
                },
            ),
            self.ast.position_of(self.ast.expr_span(callee)),
        )?;

        for (parameter, target) in bundle_checks {
            let Some(declared) = parameter.compile_time_signature.as_ref()
            else {
                continue;
            };
            let expected = substitute_type(declared, &subst);
            let constant = self.builder.constants.get(&target).copied();
            let Some(Expression::StructInit(actual, _)) =
                constant.map(|held| self.ast.expr(held))
            else {
                bail!(
                    "'{target}' given to '{name}' as '{}' is not a struct constant, and '{}' is declared as '{expected}'",
                    self.ast.name(parameter.name),
                    self.ast.name(parameter.name)
                );
            };
            let actual = self.ast.name(*actual).to_string();
            if Type::Struct(actual.clone()) != expected {
                bail!(
                    "'{target}' given to '{name}' as '{}' is a '{actual}', but '{}' is declared as '{expected}'",
                    self.ast.name(parameter.name),
                    self.ast.name(parameter.name)
                );
            }
        }

        for (parameter, target) in signature_checks {
            let Some(declared) = parameter.compile_time_signature.as_ref()
            else {
                continue;
            };
            let expected = substitute_type(declared, &subst);
            let Some(signature) = self.builder.signature(&target) else {
                continue;
            };
            let actual = Type::Proc(
                signature.parameters.clone(),
                Box::new(signature.return_type.clone()),
            );
            if actual != expected {
                bail!(
                    "'{}' given to '{name}' as '{}' has the signature '{}', but '{}' is declared as '{}'",
                    target,
                    self.ast.name(parameter.name),
                    spelled(&actual),
                    self.ast.name(parameter.name),
                    spelled(&expected)
                );
            }
        }

        // Every argument the list took, lowered as a value. The specialization
        // takes one ordinary parameter for each, so each is evaluated once
        // however many times the unrolled body names it.
        let pack_name = generic_parameters
            .iter()
            .find(|parameter| parameter.pack)
            .map(|parameter| self.ast.name(parameter.name).to_string());
        let mut pack_elements: Vec<PackElement> = Vec::new();
        if let Some(pack_name) = &pack_name {
            for (index, argument) in arguments[consumed..].iter().enumerate() {
                // `$Position` in the list is a type rather than a value. It
                // takes no parameter and is evaluated nowhere: what it leaves
                // behind is a name the body writes where a type belongs.
                if let Expression::TypeValue(ty) = self.ast.expr(*argument) {
                    let ty = ty.clone();
                    pack_elements
                        .push(PackElement::Type(substitute_type(&ty, &subst)));
                    continue;
                }
                let (operand, value_type) =
                    self.lower_expression(*argument, None)?;
                // An element of a list is a value like any other, so a borrow
                // of a scalar reaching one reads through. Nothing had asked for
                // a type here, which is the one road into the compiler that
                // takes an expression without saying what it wants, so the
                // borrow travelled as itself: the element's type became a
                // reference, the body that unrolls it was handed an address
                // where its number belongs, and a format string was told to
                // write out a `&mut i64`.
                let (operand, value_type) = match borrowed_value(&value_type) {
                    Some(inner) if !needs_memory(inner) => {
                        let held = inner.clone();
                        let read = self.coerce(operand, &value_type, &held)?;
                        (read, held)
                    }
                    // An aggregate travels by address either way, so a borrow
                    // of one is already what a value of it would be here and
                    // only the name it goes under changes.
                    Some(inner) => (operand, inner.clone()),
                    None => (operand, value_type),
                };
                pack_elements.push(PackElement::Value(
                    pack_element_name(pack_name, index),
                    value_type.clone(),
                ));
                plans.push(ArgPlan::Value(operand, value_type));
            }
        }

        for (parameter, slot) in generic_parameters.iter().zip(&aligned) {
            let Some(index) = *slot else {
                continue;
            };
            if parameter.format
                && index < arguments.len()
                && !self.forwards_its_own_format(arguments[index], arguments)
            {
                locate(
                    check_format(self.ast, arguments[index], &pack_elements),
                    self.at_expression(arguments[index]),
                )?;
            }
        }

        let mut value_parameter_types: Vec<Type> = generic_parameters
            .iter()
            .filter(|parameter| {
                !is_type_parameter(self.ast, parameter) && !parameter.pack
            })
            .map(|parameter| {
                substitute_type(&parameter_type(parameter), &subst)
            })
            .collect();
        for element in &pack_elements {
            if let PackElement::Value(_, ty) = element {
                value_parameter_types.push(ty.clone());
            }
        }
        let return_type = self
            .ast
            .signature_to_type(self.ast.signature(generic.return_sig))
            .map(|ty| substitute_type(&ty, &subst))
            .unwrap_or(Type::Void);
        let mut mangled_name =
            mangle_specialization(name, &generic.type_params, &subst);
        // What the list was given is part of what makes this specialization the
        // one it is, so its element types are part of the name.
        for element in &pack_elements {
            mangled_name.push('_');
            mangled_name.push_str(&sanitize_identifier(&element.written()));
        }
        let mut display =
            describe_specialization(name, &generic.type_params, &subst);
        if !pack_elements.is_empty() {
            let written: Vec<String> =
                pack_elements.iter().map(PackElement::written).collect();
            display.push_str(&format!("({})", written.join(", ")));
        }

        // What this specialization's types really are is known here and nowhere
        // earlier: a generic's own signature names parameters bound to nothing,
        // so `heap_slice` holds no resource while `heap_slice<File>` does. The
        // rules about where a resource may live are asked of the concrete
        // types, which is why they are asked here rather than of the written
        // ones alone: a program that never writes `Vec<File>` down still makes
        // one.
        if !self.builder.linear.is_empty() {
            let templates: crate::check::linear_instances::Templates<'_> = self
                .builder
                .generic_struct_defs
                .iter()
                .map(|(held, (params, fields))| {
                    (
                        held.as_str(),
                        (
                            params
                                .iter()
                                .map(String::as_str)
                                .collect::<Vec<&str>>(),
                            fields
                                .iter()
                                .map(|(field_name, field_type)| {
                                    (field_name.as_str(), field_type)
                                })
                                .collect::<Vec<(&str, &Type)>>(),
                        ),
                    )
                })
                .collect();
            for concrete in value_parameter_types
                .iter()
                .chain(std::iter::once(&return_type))
            {
                if let Some(report) =
                    crate::check::linear_instances::pooled_resource_in(
                        concrete,
                        &templates,
                        &self.builder.linear,
                    )
                {
                    bail!("{report}");
                }
            }
        }

        self.specializations.push(Specialization {
            generic_name: name.to_string(),
            mangled_name: mangled_name.clone(),
            requested_at: self.current_position,
            display,
            subst,
            // Stamped by whoever drains these, which is the only place that
            // knows which module's lowering produced them.
            requested_by: 0,
            pack: pack_name.map(|pack_name| (pack_name, pack_elements.clone())),
        });

        let mut lowered = Vec::with_capacity(plans.len());
        for (plan, target) in plans.into_iter().zip(&value_parameter_types) {
            // A parameter whose type is still the template's own name is one
            // this call did not pin down, and the argument is what says how it
            // travels. A `str` handed to such a parameter is an aggregate and
            // goes by address. Passing it in a register is what the backend
            // then refuses.
            let held;
            let target = match (target, &plan) {
                (Type::TypeParam(_), ArgPlan::Value(_, value_type)) => {
                    held = value_type.clone();
                    &held
                }
                _ => target,
            };
            match plan {
                ArgPlan::Value(operand, value_type) => {
                    // A value reaching a borrow parameter goes by address. The
                    // plan reads the template's parameter, where a read of a
                    // generic is a borrow whatever the type turns out to be, so
                    // an argument that is already a borrow in the caller is
                    // planned as a value and arrives here as the aggregate
                    // itself. Passing it in a register is what the backend then
                    // refuses: a `str` forwarded from one generic to another is
                    // this.
                    if let Type::Ref(inner) | Type::RefMut(inner) = target
                        && !matches!(
                            value_type,
                            Type::Ref(_) | Type::RefMut(_) | Type::Ptr(_)
                        )
                    {
                        if matches!(target, Type::Ref(_))
                            && !needs_memory(inner)
                        {
                            let coerced =
                                self.coerce(operand, &value_type, inner)?;
                            lowered.push(coerced);
                            continue;
                        }
                        let IrOperand::Local(local) = operand else {
                            bail!(
                                "this argument is a '{}' with no storage, and '{name}' borrows it here",
                                spelled(&value_type)
                            );
                        };
                        lowered.push(self.address_of_local(local, inner));
                        continue;
                    }
                    // An array reaching a `[]T` parameter becomes a slice of
                    // the whole of itself first. Without this the callee is
                    // handed the array's own address and reads its first two
                    // elements as a pointer and a length.
                    if let (Some(element), Type::Array(held, count)) =
                        (slice_element_wanted(target), &value_type)
                        && **held == element
                    {
                        let IrOperand::Local(local) = operand else {
                            bail!(
                                "an array argument to a generic call is not a place"
                            );
                        };
                        let base = self.address_of_local(local, &value_type);
                        let slice = self
                            .build_slice_from_address(base, &element, *count);
                        let IrOperand::Local(view) = slice else {
                            bail!("slice construction did not yield a place");
                        };
                        lowered.push(self.address_of_local(view, target));
                    } else if needs_memory(target) {
                        let IrOperand::Local(local) = operand else {
                            bail!(
                                "aggregate argument to generic call is not a place"
                            );
                        };
                        // A local already holding the address of the aggregate
                        // is that address. A read parameter of struct type is
                        // one, so handing it on by value passes what it holds
                        // rather than where it is held.
                        if let Type::Ref(inner)
                        | Type::RefMut(inner)
                        | Type::Ptr(inner) = self.type_of_local(local)
                            && *inner == *target
                        {
                            lowered.push(IrOperand::Local(local));
                        } else {
                            lowered.push(self.address_of_local(local, target));
                        }
                    } else {
                        lowered.push(self.coerce(
                            operand,
                            &value_type,
                            target,
                        )?);
                    }
                }
                ArgPlan::Borrow(index) => {
                    let pointee = match target {
                        Type::Ref(inner) | Type::RefMut(inner) => {
                            (**inner).clone()
                        }
                        _ => target.clone(),
                    };
                    // A read of a generic is a borrow in the template, since
                    // nothing there knows whether the type is copied. Once it
                    // is known, a scalar travels in a register and has no
                    // address to hand over, so the value goes instead: this is
                    // `map_insert(m, old_keys[i])` with an i64 key. A `mut`
                    // parameter is not this: it writes back through the address
                    // whatever the width.
                    if matches!(target, Type::Ref(_)) && !needs_memory(&pointee)
                    {
                        let (operand, value_type) = self.lower_expression(
                            arguments[index],
                            Some(&pointee),
                        )?;
                        let coerced =
                            self.coerce(operand, &value_type, &pointee)?;
                        lowered.push(coerced);
                        continue;
                    }
                    let address = self.aggregate_argument_address(
                        arguments[index],
                        &pointee,
                        false,
                    )?;
                    lowered.push(address);
                }
            }
        }

        let result = self.fresh_local(return_type.clone(), None);
        if needs_memory(&return_type) {
            self.mark_in_memory(result);
        }
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Call {
                function: mangled_name,
                arguments: lowered,
            },
        ));
        Ok((IrOperand::Local(result), return_type))
    }

    fn lower_direct_call(
        &mut self,
        name: &str,
        arguments: &[ExprId],
    ) -> Result<(IrOperand, Type)> {
        let signature = self.builder.signature(name).unwrap();
        let parameter_types = signature.parameters.clone();
        let return_type = signature.return_type.clone();

        if arguments.len() != parameter_types.len() {
            bail!(
                "function '{name}' expects {} argument(s) but {} were given",
                parameter_types.len(),
                arguments.len()
            );
        }

        let mut lowered = Vec::with_capacity(arguments.len());
        for (index, argument) in arguments.iter().enumerate() {
            let held_target;
            let expected = parameter_types.get(index);
            // A function value is its signature, and two that differ are two
            // different functions. Compared here, where both are still spelled
            // the way the reader wrote them and where the self-hosted compiler
            // compares them too. The IR check below does catch it, in lowered
            // terms and in a sentence of its own, so the two compilers named
            // one fault two ways. The rule is that check's, reused rather than
            // written again.
            if let Some(wanted @ Type::Proc(..)) = expected
                && let Some(given) = self.value_signature(*argument)
                && matches!(given, Type::Proc(..))
                && !crate::ir::typecheck::fits(&given, wanted)
            {
                return locate(
                    Err(anyhow::anyhow!(
                        "this argument is a '{}' and a '{}' is what is wanted here",
                        spelled(&given),
                        spelled(wanted)
                    )),
                    self.at_expression(*argument),
                );
            }
            // An argument stands where its parameter's type is written. Asked
            // here, ahead of the address-taking below, because an aggregate
            // parameter is a reference by the time it reaches this and the
            // address is taken without a word about what it points at; the IR
            // check that follows sees two pointers, and a pointer fits every
            // other. That is how an 'Other' reached a 'Point' parameter and the
            // callee read one layout as the other. The rule is that check's,
            // asked while both sides are still spelled the way they were
            // written, and the borrow a mode added is read through.
            if let Some(target) = expected {
                let wanted = match target {
                    Type::Ref(inner) | Type::RefMut(inner) => inner.as_ref(),
                    other => other,
                };
                // Text written down is a run of bytes, and what it reaches is a
                // run of them or the address a call into C reads. Asked of the
                // expression rather than of a type, since being a literal is
                // what decides it: the bytes the compiler wrote down are the
                // ones it terminated. Its type alone is an address, and an
                // address reaches a whole number, which is how `show("abc")`
                // filled an `i64` parameter with a pointer.
                if matches!(
                    self.ast.expr(*argument),
                    Expression::Literal(Literal::String(_))
                ) && !matches!(
                    wanted,
                    Type::Str
                        | Type::Slice(_)
                        | Type::Ptr(_)
                        | Type::Array(..)
                        | Type::TypeParam(_)
                        | Type::Unknown
                ) {
                    return locate(
                        Err(anyhow::anyhow!(
                            "this argument is a 'str' and a '{}' is what is wanted here",
                            spelled(wanted)
                        )),
                        self.at_expression(*argument),
                    );
                }
            }
            if let Some(target) = expected
                && let Some(given) = self.answer_type(*argument)
            {
                let wanted = match target {
                    Type::Ref(inner) | Type::RefMut(inner) => inner.as_ref(),
                    other => other,
                };
                if self.type_is_settled(&given)
                    && self.type_is_settled(wanted)
                    && !crate::ir::typecheck::fits(&given, wanted)
                {
                    return locate(
                        Err(anyhow::anyhow!(
                            "this argument is a '{}' and a '{}' is what is wanted here",
                            spelled(&given),
                            spelled(wanted)
                        )),
                        self.at_expression(*argument),
                    );
                }
            }
            // Auto-borrow. A `read`/`mut` parameter is a reference, and a plain
            // value place passed to it takes its address here. An argument that
            // is already a reference (a reference-typed local passed onward) or
            // an explicit borrow is left alone, so nothing is double-referenced.
            if let Some(reference @ (Type::Ref(inner) | Type::RefMut(inner))) =
                expected
            {
                let already_reference =
                    matches!(
                        self.ast.expr(*argument),
                        Expression::Borrow(_)
                            | Expression::BorrowMut(_)
                            | Expression::AddressOf(_)
                    ) || self.probe_type(*argument).as_ref() == Some(reference);
                if !already_reference {
                    let pointee = (**inner).clone();
                    let address = self.aggregate_argument_address(
                        *argument, &pointee, false,
                    )?;
                    lowered.push(address);
                    continue;
                }
            }
            // A parameter whose type is still the template's own name is one
            // this call did not pin down, and then the argument says how it
            // travels: an aggregate goes by address whatever the parameter was
            // written as. A `str` passed on from one generic to another is
            // this, and passing it in a register is what the backend refuses.
            let expected = match expected {
                Some(Type::TypeParam(_)) => match self.probe_type(*argument) {
                    Some(ty) if needs_memory(&ty) => {
                        held_target = ty;
                        Some(&held_target)
                    }
                    _ => expected,
                },
                _ => expected,
            };
            if let Some(target) = expected
                && needs_memory(target)
            {
                let address =
                    self.aggregate_argument_address(*argument, target, true)?;
                lowered.push(address);
                continue;
            }
            let (operand, value_type) =
                self.lower_expression(*argument, expected)?;
            if let Some(Type::Ref(inner) | Type::RefMut(inner)) = expected
                && needs_memory(&value_type)
                && value_type == **inner
            {
                bail!(
                    "a '{}' is wanted here as a borrow and this is the value; a parameter's mode is what borrows, so declare the one this reaches as `read` or `mut`",
                    spelled(&value_type)
                );
            }
            if let Some(target) = expected
                && distinct_mismatch(
                    self.ast,
                    *argument,
                    &value_type,
                    target,
                    &self.builder.flags,
                )
            {
                let (described, note) = nominal_words(
                    self.ast,
                    *argument,
                    &value_type,
                    target,
                    &self.builder.flags,
                );
                // The sentence every argument mismatch is reported in, with
                // the rule that was broken after it. Naming the callee instead
                // read one way here and another at every other argument, and
                // the two compilers said different things about one program.
                bail!(
                    "this argument is {described} and a '{}' is what is wanted here; {note}",
                    spelled(target)
                );
            }
            let coerced = match expected {
                Some(target) => self.coerce(operand, &value_type, target)?,
                None => operand,
            };
            lowered.push(coerced);
        }

        let result = self.fresh_local(return_type.clone(), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Call {
                function: name.to_string(),
                arguments: lowered,
            },
        ));
        Ok((IrOperand::Local(result), return_type))
    }

    fn lower_indirect_call(
        &mut self,
        callee: ExprId,
        arguments: &[ExprId],
    ) -> Result<(IrOperand, Type)> {
        let (callee_operand, callee_type) =
            self.lower_expression(callee, None)?;
        let Type::Proc(parameter_types, return_type) = callee_type else {
            bail!("cannot call a value that is not a function pointer");
        };
        let return_type = *return_type;
        if arguments.len() != parameter_types.len() {
            bail!(
                "function pointer expects {} argument(s) but {} were given",
                parameter_types.len(),
                arguments.len()
            );
        }

        let mut lowered = Vec::with_capacity(arguments.len());
        for (index, argument) in arguments.iter().enumerate() {
            let held_target;
            let expected = parameter_types.get(index);
            // A function value is its signature, and two that differ are two
            // different functions. Compared here, where both are still spelled
            // the way the reader wrote them and where the self-hosted compiler
            // compares them too. The IR check below does catch it, in lowered
            // terms and in a sentence of its own, so the two compilers named
            // one fault two ways. The rule is that check's, reused rather than
            // written again.
            if let Some(wanted @ Type::Proc(..)) = expected
                && let Some(given) = self.value_signature(*argument)
                && matches!(given, Type::Proc(..))
                && !crate::ir::typecheck::fits(&given, wanted)
            {
                return locate(
                    Err(anyhow::anyhow!(
                        "this argument is a '{}' and a '{}' is what is wanted here",
                        spelled(&given),
                        spelled(wanted)
                    )),
                    self.at_expression(*argument),
                );
            }
            // An argument stands where its parameter's type is written. Asked
            // here, ahead of the address-taking below, because an aggregate
            // parameter is a reference by the time it reaches this and the
            // address is taken without a word about what it points at; the IR
            // check that follows sees two pointers, and a pointer fits every
            // other. That is how an 'Other' reached a 'Point' parameter and the
            // callee read one layout as the other. The rule is that check's,
            // asked while both sides are still spelled the way they were
            // written, and the borrow a mode added is read through.
            if let Some(target) = expected {
                let wanted = match target {
                    Type::Ref(inner) | Type::RefMut(inner) => inner.as_ref(),
                    other => other,
                };
                // Text written down is a run of bytes, and what it reaches is a
                // run of them or the address a call into C reads. Asked of the
                // expression rather than of a type, since being a literal is
                // what decides it: the bytes the compiler wrote down are the
                // ones it terminated. Its type alone is an address, and an
                // address reaches a whole number, which is how `show("abc")`
                // filled an `i64` parameter with a pointer.
                if matches!(
                    self.ast.expr(*argument),
                    Expression::Literal(Literal::String(_))
                ) && !matches!(
                    wanted,
                    Type::Str
                        | Type::Slice(_)
                        | Type::Ptr(_)
                        | Type::Array(..)
                        | Type::TypeParam(_)
                        | Type::Unknown
                ) {
                    return locate(
                        Err(anyhow::anyhow!(
                            "this argument is a 'str' and a '{}' is what is wanted here",
                            spelled(wanted)
                        )),
                        self.at_expression(*argument),
                    );
                }
            }
            if let Some(target) = expected
                && let Some(given) = self.answer_type(*argument)
            {
                let wanted = match target {
                    Type::Ref(inner) | Type::RefMut(inner) => inner.as_ref(),
                    other => other,
                };
                if self.type_is_settled(&given)
                    && self.type_is_settled(wanted)
                    && !crate::ir::typecheck::fits(&given, wanted)
                {
                    return locate(
                        Err(anyhow::anyhow!(
                            "this argument is a '{}' and a '{}' is what is wanted here",
                            spelled(&given),
                            spelled(wanted)
                        )),
                        self.at_expression(*argument),
                    );
                }
            }
            // Auto-borrow. A `read`/`mut` parameter is a reference, and a plain
            // value place passed to it takes its address here. An argument that
            // is already a reference (a reference-typed local passed onward) or
            // an explicit borrow is left alone, so nothing is double-referenced.
            if let Some(reference @ (Type::Ref(inner) | Type::RefMut(inner))) =
                expected
            {
                let already_reference =
                    matches!(
                        self.ast.expr(*argument),
                        Expression::Borrow(_)
                            | Expression::BorrowMut(_)
                            | Expression::AddressOf(_)
                    ) || self.probe_type(*argument).as_ref() == Some(reference);
                if !already_reference {
                    let pointee = (**inner).clone();
                    let address = self.aggregate_argument_address(
                        *argument, &pointee, false,
                    )?;
                    lowered.push(address);
                    continue;
                }
            }
            // A parameter whose type is still the template's own name is one
            // this call did not pin down, and then the argument says how it
            // travels: an aggregate goes by address whatever the parameter was
            // written as. A `str` passed on from one generic to another is
            // this, and passing it in a register is what the backend refuses.
            let expected = match expected {
                Some(Type::TypeParam(_)) => match self.probe_type(*argument) {
                    Some(ty) if needs_memory(&ty) => {
                        held_target = ty;
                        Some(&held_target)
                    }
                    _ => expected,
                },
                _ => expected,
            };
            if let Some(target) = expected
                && needs_memory(target)
            {
                let address =
                    self.aggregate_argument_address(*argument, target, true)?;
                lowered.push(address);
                continue;
            }
            let (operand, value_type) =
                self.lower_expression(*argument, expected)?;
            if let Some(Type::Ref(inner) | Type::RefMut(inner)) = expected
                && needs_memory(&value_type)
                && value_type == **inner
            {
                bail!(
                    "a '{}' is wanted here as a borrow and this is the value; a parameter's mode is what borrows, so declare the one this reaches as `read` or `mut`",
                    spelled(&value_type)
                );
            }
            let coerced = match expected {
                Some(target) => self.coerce(operand, &value_type, target)?,
                None => operand,
            };
            lowered.push(coerced);
        }

        let result = self.fresh_local(return_type.clone(), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::CallIndirect {
                callee: callee_operand,
                arguments: lowered,
                parameter_types,
                return_type: return_type.clone(),
            },
        ));
        Ok((IrOperand::Local(result), return_type))
    }

    fn aggregate_argument_address(
        &mut self,
        argument: ExprId,
        target: &Type,
        consume: bool,
    ) -> Result<IrOperand> {
        // An aggregate travels by address, and an address is a machine word:
        // once this has taken one, nothing downstream can tell a pointer to one
        // struct from a pointer to another. So the two are compared here, which
        // is the one place every way of passing an aggregate goes through, and
        // while both types are still spelled out.
        //
        // A `Vec3` passed where a `Mat4` was declared read sixty-four bytes off
        // a twelve-byte value, and every check after this agreed, because what
        // they were checking was a pointer.
        if let Some(given) = self.probe_type(argument)
            && let (Some(from), Some(into)) =
                (aggregate_name(&given), aggregate_name(target))
            && from != into
        {
            return locate(
                Err(anyhow::anyhow!(
                    "this argument is a '{}' and a '{}' is what is wanted here",
                    spelled(&given),
                    spelled(target)
                )),
                self.at_expression(argument),
            );
        }
        // A raw pointer where a slice is wanted. A slice is an address and a
        // length; a pointer is the address alone, so the callee reads whatever
        // sat beside it for the length. Caught here, while both types are still
        // spelled the way the reader wrote them: taken as an address like any
        // other aggregate argument, the complaint lands in the IR typechecker
        // and quotes the lowered `^^i8` of a parameter the source calls a `str`.
        if let Some(given) = self.probe_type(argument)
            && slice_element_wanted(target).is_some()
            && matches!(given, Type::Ptr(_))
        {
            return locate(
                Err(anyhow::anyhow!(
                    "this argument is a '{}' and a '{}' is what is wanted here",
                    spelled(&given),
                    spelled(target)
                )),
                self.at_expression(argument),
            );
        }
        // Passing a `[N]T` array where a `[]T` slice is wanted. Build the slice
        // view and hand over its address, rather than the array's.
        if let Some(element) = slice_element_wanted(target)
            && let Some(Type::Array(array_element, count)) =
                self.probe_type(argument)
            && *array_element == element
        {
            // Any array place, not only a bare variable: a struct field (such as
            // a columns column `c.x`), an index, a deref. `probe_type` reads the
            // place chain's type and `place_address` walks it, so the slice
            // carries the right base and length instead of collapsing to a bare
            // pointer.
            let (base, _) = self.place_address(argument)?;
            let slice = self.build_slice_from_address(base, &element, count);
            let IrOperand::Local(slice_local) = slice else {
                bail!("slice construction did not yield a place");
            };
            return Ok(self.address_of_local(slice_local, target));
        }
        match self.ast.expr(argument).clone() {
            Expression::Identifier(name) => {
                let name = self.ast.name(name).to_string();
                if consume
                    && let Some(local) = self.resolve_variable(&name)
                    && self.locals[local].linear
                {
                    self.emit(IrStatement::Consume(local));
                }
                // A name already holding the address of the aggregate is that
                // address. A read parameter of struct type is one: it arrived
                // as a borrow, so what it holds is where the value is, and
                // taking its address again would hand over the address of the
                // pointer.
                if let Some(local) = self.resolve_variable(&name)
                    && let Type::Ref(inner)
                    | Type::RefMut(inner)
                    | Type::Ptr(inner) = self.type_of_local(local)
                    && needs_memory(&inner)
                    && *inner == *target
                {
                    return Ok(IrOperand::Local(local));
                }
                let (address, _) = self.place_address(argument)?;
                Ok(address)
            }
            Expression::FieldAccess(..)
            | Expression::Index(..)
            | Expression::Dereference(_) => {
                let (address, _) = self.place_address(argument)?;
                Ok(address)
            }
            // A run written out where a slice is wanted holds that slice's
            // element, the way one written into a declared array does, and its
            // own length is the slice's. It is built as the array it is and
            // handed over as a view of that, which is the road a named array
            // already takes. Given the slice's own type instead, the temp was a
            // slice holding elements and the reader was told an array literal
            // had a type that is not an array.
            Expression::Literal(Literal::Array(elements))
                if slice_element_wanted(target).is_some() =>
            {
                let element = slice_element_wanted(target)
                    .expect("a slice element, just asked for");
                let count = elements.len();
                let held = Type::Array(Box::new(element.clone()), count);
                let temp = self.fresh_local(held.clone(), None);
                self.materialize_aggregate(temp, argument)?;
                let base = self.address_of_local(temp, &element);
                let slice =
                    self.build_slice_from_address(base, &element, count);
                let IrOperand::Local(slice_local) = slice else {
                    bail!("slice construction did not yield a place");
                };
                Ok(self.address_of_local(slice_local, target))
            }
            Expression::StructInit(..)
            | Expression::EnumVariantInit(..)
            | Expression::Literal(Literal::Array(_)) => {
                let temp = self.fresh_local(target.clone(), None);
                self.materialize_aggregate(temp, argument)?;
                Ok(self.address_of_local(temp, target))
            }
            _ => {
                let (operand, _) =
                    self.lower_expression(argument, Some(target))?;
                let IrOperand::Local(local) = operand else {
                    bail!("cannot pass this value as an aggregate argument");
                };
                // A borrowed parameter is handed an address, so an aggregate
                // that is not already in memory is put there first: the value
                // an expression answered with lives in a register until
                // something needs to point at it. A scalar reaching a borrow is
                // not this. That one is passed as the value it is, where the
                // plan is read.
                if needs_memory(target) && !self.locals[local].in_memory {
                    let held = self.fresh_local(target.clone(), None);
                    self.mark_in_memory(held);
                    self.emit(IrStatement::Assign(
                        held,
                        IrRvalue::Use(IrOperand::Local(local)),
                    ));
                    return Ok(self.address_of_local(held, target));
                }
                Ok(self.address_of_local(local, target))
            }
        }
    }

    fn address_of_local(&mut self, local: LocalId, ty: &Type) -> IrOperand {
        // Taking a local's address is what "in memory" means, so it is said
        // here rather than left to each caller to remember. One that forgot is
        // how `t.field = held` came to be refused with "aggregate local is not
        // in memory": the copy asked for the source's address and nothing had
        // given it a slot to have one of.
        self.mark_in_memory(local);
        let result = self.fresh_local(Type::Ptr(Box::new(ty.clone())), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::AddressOf { local, offset: 0 },
        ));
        IrOperand::Local(result)
    }

    fn materialize_aggregate(
        &mut self,
        local: LocalId,
        expression: ExprId,
    ) -> Result<()> {
        match self.ast.expr(expression).clone() {
            Expression::StructInit(name, fields) => {
                let name = self.ast.name(name).to_string();
                let layout_name = match self.type_of_local(local) {
                    Type::Struct(instance)
                        if name.is_empty()
                            || is_generic_instance(&instance) =>
                    {
                        instance
                    }
                    _ => name.clone(),
                };
                self.init_struct(local, &layout_name, fields)
            }
            // A value a type names under itself reads as a variant and is a
            // value, so it is lowered and stored rather than built here. Left
            // to the arm below, `Mat4::IDENTITY` written into a field asked the
            // enum table for a struct and was told there is no such enum.
            Expression::EnumVariantInit(name, variant, _)
                if self.builder.names_a_value(
                    self.ast.name(name),
                    self.ast.name(variant),
                ) =>
            {
                let held = self.type_of_local(local);
                let (operand, value_type) =
                    self.lower_expression(expression, Some(&held))?;
                let coerced = self.coerce(operand, &value_type, &held)?;
                self.emit(IrStatement::Assign(local, IrRvalue::Use(coerced)));
                Ok(())
            }
            Expression::EnumVariantInit(name, variant, fields) => {
                let name = self.ast.name(name).to_string();
                let variant = self.ast.name(variant).to_string();
                // The local's type already names the instance when the context
                // resolved one, and that is the layout to write into.
                let layout_name = match self.type_of_local(local) {
                    Type::Enum(instance) | Type::Struct(instance)
                        if name.is_empty()
                            || is_generic_instance(&instance) =>
                    {
                        instance
                    }
                    _ => name.clone(),
                };
                // An argument and an element reach a `.Name` here rather than
                // through the walk that refuses one, since what they build is
                // written straight into the place that holds it. The local's
                // type is what names the enum, and the report carries it.
                if name.is_empty() && !self.ast.is_failure_result(&layout_name)
                {
                    let held = self.type_of_local(local);
                    let variant = self.ast.intern(&variant);
                    return Err(refuse_inferred_variant(
                        self.ast,
                        variant,
                        Some(&held),
                    ));
                }
                self.init_enum(local, &layout_name, &variant, fields)
            }
            Expression::Literal(Literal::Array(elements)) => {
                let Type::Array(element, _) = self.type_of_local(local) else {
                    bail!("array literal has non-array type");
                };
                let elements: Vec<ExprId> =
                    self.ast.exprs_in(elements).to_vec();
                self.init_array(local, &element, &elements)
            }
            _ => {
                bail!("cannot materialize this aggregate")
            }
        }
    }

    fn lower_assignment(
        &mut self,
        target: ExprId,
        value: ExprId,
    ) -> Result<()> {
        if let Expression::Identifier(name) = self.ast.expr(target) {
            let name = self.ast.name(*name).to_string();
            let Some(local) = self.resolve_variable(&name) else {
                // One sentence for one fault, wherever the name stands. What
                // the reader was doing with it is on the line the caret is on.
                return locate(
                    Err(anyhow::anyhow!("unknown variable '{name}'")),
                    self.at_expression(target),
                );
            };
            let target_type = self.type_of_local(local);
            // A name holding a borrow of a scalar names the storage it borrows,
            // so writing to it writes through. Left alone the address itself
            // was overwritten, and the place the reader meant kept its old
            // value with nothing said.
            if let Some(inner) = borrowed_value(&target_type)
                && !needs_memory(inner)
            {
                let inner = inner.clone();
                let (operand, value_type) =
                    self.lower_expression(value, Some(&inner))?;
                let coerced = self.coerce(operand, &value_type, &inner)?;
                self.emit(IrStatement::Store {
                    address: IrOperand::Local(local),
                    value: coerced,
                });
                return Ok(());
            }
            let (operand, value_type) =
                self.lower_expression(value, Some(&target_type))?;
            if distinct_mismatch(
                self.ast,
                value,
                &value_type,
                &target_type,
                &self.builder.flags,
            ) {
                let (described, note) = nominal_words(
                    self.ast,
                    value,
                    &value_type,
                    &target_type,
                    &self.builder.flags,
                );
                bail!(
                    "'{name}' is a '{target_type}' and the value is {described}; {note}"
                );
            }
            let coerced = self.coerce(operand, &value_type, &target_type)?;
            self.emit(IrStatement::Assign(local, IrRvalue::Use(coerced)));
            return Ok(());
        }

        // `c[h] = value`: scatter the whole element into the columns' per-field
        // arrays at the handle's slot. It cannot go through `place_address`,
        // which yields one address. The scatter is inherently multi-store.
        if let Expression::Index(container, index_expr) =
            self.ast.expr(target).clone()
            && let Some(struct_name) = self.columns_shaped_base(container)
        {
            let (index_operand, index_type) =
                self.lower_expression(index_expr, None)?;
            if matches!(index_type, Type::Handle(_)) {
                return self.columns_scatter(
                    container,
                    &struct_name,
                    index_operand,
                    value,
                );
            }
        }

        let (address, pointee) = self.place_address(target)?;
        let (operand, value_type) =
            self.lower_expression(value, Some(&pointee))?;
        // The same nominal rule the named local above is held to. A field, an
        // element and a place behind a pointer are assignments too, and a
        // distinct type reached through one of them was taking its
        // representation without a `cast` saying so: `h.usage = plain` put a
        // number into a set of bits.
        if distinct_mismatch(
            self.ast,
            value,
            &value_type,
            &pointee,
            &self.builder.flags,
        ) {
            let (described, note) = nominal_words(
                self.ast,
                value,
                &value_type,
                &pointee,
                &self.builder.flags,
            );
            bail!(
                "this place is a '{pointee}' and the value is {described}; {note}"
            );
        }
        if needs_memory(&pointee) {
            let IrOperand::Local(source_local) = operand else {
                bail!("aggregate assignment from a non-place value");
            };
            // A local already holding the address of the value is where to copy
            // from. A read parameter of struct type is one, so storing it copies
            // what it points at rather than the pointer.
            let source = match self.type_of_local(source_local) {
                Type::Ref(inner) | Type::RefMut(inner) | Type::Ptr(inner)
                    if *inner == pointee =>
                {
                    IrOperand::Local(source_local)
                }
                _ => self.address_of_local(source_local, &pointee),
            };
            let size = self.builder.byte_size(&pointee);
            self.emit(IrStatement::Copy {
                destination: address,
                source,
                size,
            });
            return Ok(());
        }
        let coerced = self.coerce(operand, &value_type, &pointee)?;
        self.emit(IrStatement::Store {
            address,
            value: coerced,
        });
        Ok(())
    }

    fn lower_address_of(
        &mut self,
        inner: ExprId,
        kind: RefKind,
    ) -> Result<(IrOperand, Type)> {
        let (address, pointee) = self.place_address(inner)?;
        let result_type = match kind {
            RefKind::Ref => Type::Ref(Box::new(pointee)),
            RefKind::RefMut => Type::RefMut(Box::new(pointee)),
            RefKind::Ptr => Type::Ptr(Box::new(pointee)),
        };
        Ok((address, result_type))
    }

    fn lower_dereference(
        &mut self,
        pointer: ExprId,
    ) -> Result<(IrOperand, Type)> {
        let (address, pointee) = self.place_address_of_deref(pointer)?;
        self.load_from(address, pointee)
    }

    fn lower_field_read(
        &mut self,
        base: ExprId,
        field: &str,
    ) -> Result<(IrOperand, Type)> {
        let (address, field_type) = self.field_address(base, field)?;
        self.load_from(address, field_type)
    }

    fn load_from(
        &mut self,
        address: IrOperand,
        ty: Type,
    ) -> Result<(IrOperand, Type)> {
        if needs_memory(&ty) {
            let temp = self.fresh_local(ty.clone(), None);
            self.mark_in_memory(temp);
            let destination = self.address_of_local(temp, &ty);
            let size = self.builder.byte_size(&ty);
            self.emit(IrStatement::Copy {
                destination,
                source: address,
                size,
            });
            return Ok((IrOperand::Local(temp), ty));
        }
        let result = self.fresh_local(ty.clone(), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Load {
                address,
                ty: ty.clone(),
            },
        ));
        Ok((IrOperand::Local(result), ty))
    }

    fn build_str_value(&mut self, local: LocalId, text: &str) {
        self.mark_in_memory(local);
        let ptr_slot =
            self.fresh_local(Type::Ptr(Box::new(str_byte_ptr_type())), None);
        self.emit(IrStatement::Assign(
            ptr_slot,
            IrRvalue::AddressOf {
                local,
                offset: STR_PTR_OFFSET,
            },
        ));
        self.emit(IrStatement::Store {
            address: IrOperand::Local(ptr_slot),
            value: IrOperand::Constant(IrConstant::CString(text.to_string())),
        });
        let len_slot = self.fresh_local(Type::Ptr(Box::new(Type::Usize)), None);
        self.emit(IrStatement::Assign(
            len_slot,
            IrRvalue::AddressOf {
                local,
                offset: STR_LEN_OFFSET,
            },
        ));
        self.emit(IrStatement::Store {
            address: IrOperand::Local(len_slot),
            value: IrOperand::Constant(IrConstant::Integer(
                text.len() as i64,
                Type::Usize,
            )),
        });
    }

    // Best-effort static type of a place expression, without lowering it. Used to
    // recognize a raw-pointer base for indexing. Handles the identifier, deref,
    // and field-access chains that a pointer flows through.
    // The type of an expression used as a function value. `probe_type` answers
    // for locals, and a function named as a value is not one: it resolves to a
    // declaration, the way the identifier arm of `lower_expression` resolves it
    // when it takes the function's address.
    fn value_signature(&self, expression: ExprId) -> Option<Type> {
        if let Some(held) = self.probe_type(expression) {
            return Some(held);
        }
        let Expression::Identifier(name) = self.ast.expr(expression) else {
            return None;
        };
        let name = self.ast.name(*name).to_string();
        if self.resolve_variable(&name).is_some() {
            return None;
        }
        self.builder.signature(&name).map(|signature| {
            Type::Proc(
                signature.parameters.clone(),
                Box::new(signature.return_type.clone()),
            )
        })
    }

    /// Whether every name a type is built from is one this call has settled. A
    /// generic's declared answer still carries its own parameter's name, and
    /// `arena_at` written as answering `ref T` is a borrow of that name rather
    /// than of the element the call chose. Nothing is comparable until the name
    /// resolves, so an argument whose type mentions one is left alone.
    fn type_is_settled(&self, ty: &Type) -> bool {
        match ty {
            Type::TypeParam(_) | Type::Unknown => false,
            Type::Struct(name) | Type::Enum(name) => {
                self.builder.struct_layout(name).is_some()
                    || self.builder.enum_layout(name).is_some()
            }
            Type::Ptr(inner)
            | Type::Ref(inner)
            | Type::RefMut(inner)
            | Type::Slice(inner)
            | Type::Array(inner, _)
            | Type::ArrayGeneric(inner, _)
            | Type::Handle(inner)
            | Type::Distinct(_, inner) => self.type_is_settled(inner),
            Type::Proc(parameters, answer) => {
                parameters.iter().all(|held| self.type_is_settled(held))
                    && self.type_is_settled(answer)
            }
            _ => true,
        }
    }

    fn probe_type(&self, expression: ExprId) -> Option<Type> {
        match self.ast.expr(expression) {
            Expression::Identifier(name) => self
                .resolve_variable(self.ast.name(*name))
                .map(|local| self.type_of_local(local)),
            Expression::Dereference(inner) => {
                deref_target(&self.probe_type(*inner)?).ok()
            }
            // An element of an array, a slice or a raw pointer. Without this an
            // index whose base is itself an index answered nothing, so
            // `pair[0][0]` where `pair` is `[2][]i64` fell past the slice path
            // to the array one and was refused for not naming an array. The
            // self-hosted compiler builds that program and prints the right
            // number, which is what says what the answer is rather than whether
            // there should be one.
            Expression::Index(base, _) => {
                let held = match self.probe_type(*base)? {
                    Type::Ref(inner) | Type::RefMut(inner) => *inner,
                    other => other,
                };
                match held {
                    Type::Array(element, _)
                    | Type::Slice(element)
                    | Type::Ptr(element) => Some(*element),
                    _ => None,
                }
            }
            Expression::FieldAccess(base, field) => {
                let base_type = self.probe_type(*base)?;
                let struct_name = match base_type {
                    Type::Struct(name) => name,
                    Type::Ref(inner)
                    | Type::RefMut(inner)
                    | Type::Ptr(inner) => match *inner {
                        Type::Struct(name) => name,
                        _ => return None,
                    },
                    _ => return None,
                };
                self.builder
                    .struct_layout(&struct_name)?
                    .field(self.ast.name(*field))
                    .map(|field| field.ty.clone())
            }
            _ => None,
        }
    }

    /// What an expression answers with, before anything is lowered, including
    /// through a call. A generic's answer is its declared one with what the
    /// call binds substituted in, which is what says that
    /// `sort(ops, vec_slice(v))` sorts a run of `i64`: the element comes off
    /// the argument and nothing at the call writes it.
    ///
    /// Beside `probe_type` rather than inside it. That one answers about a
    /// place, and a call is a value: the two are read together where a value's
    /// type is what is wanted, and `probe_type` alone where an address is.
    fn answer_type(&self, expression: ExprId) -> Option<Type> {
        if let Some(held) = self.probe_type(expression) {
            return Some(held);
        }
        match self.ast.expr(expression) {
            Expression::Call(..) => self.call_answer_type(expression),
            Expression::StructInit(name, _) => {
                Some(Type::Struct(self.ast.name(*name).to_string()))
            }
            Expression::EnumVariantInit(name, _, _) => {
                Some(Type::Enum(self.ast.name(*name).to_string()))
            }
            Expression::Borrow(inner) => {
                Some(Type::Ref(Box::new(self.answer_type(*inner)?)))
            }
            Expression::BorrowMut(inner) => {
                Some(Type::RefMut(Box::new(self.answer_type(*inner)?)))
            }
            Expression::Dereference(inner) => {
                deref_target(&self.answer_type(*inner)?).ok()
            }
            Expression::Index(base, _) => {
                match through_borrow(&self.answer_type(*base)?).clone() {
                    Type::Array(element, _)
                    | Type::Slice(element)
                    | Type::Ptr(element) => Some(*element),
                    Type::Str => Some(Type::U8),
                    _ => None,
                }
            }
            Expression::FieldAccess(base, field) => {
                let Type::Struct(name) =
                    through_borrow(&self.answer_type(*base)?).clone()
                else {
                    return None;
                };
                self.builder
                    .struct_layout(&name)?
                    .field(self.ast.name(*field))
                    .map(|held| held.ty.clone())
            }
            _ => None,
        }
    }

    fn call_answer_type(&self, expression: ExprId) -> Option<Type> {
        let Expression::Call(callee, arguments) = self.ast.expr(expression)
        else {
            return None;
        };
        let Expression::Identifier(name) = self.ast.expr(*callee) else {
            return None;
        };
        let name = self.ast.name(*name);
        let arguments = self.ast.exprs_in(*arguments);
        let Some(generic) = self.builder.generic_functions.get(name) else {
            return self
                .builder
                .signature(name)
                .map(|held| held.return_type.clone());
        };
        let declared = self
            .ast
            .signature_to_type(self.ast.signature(generic.return_sig))?;
        let mut bound = HashMap::new();
        for (parameter, binding) in
            generic_bindings(self.ast, self.ast.params_in(generic.parameters))
        {
            match binding {
                GenericBinding::Written(at) => {
                    if let Some(Expression::TypeValue(ty)) = arguments
                        .get(at)
                        .map(|argument| self.ast.expr(*argument))
                    {
                        bound.insert(parameter, ty.clone());
                    }
                }
                GenericBinding::Settled(at, pattern) => {
                    if let Some(held) = arguments
                        .get(at)
                        .and_then(|argument| self.answer_type(*argument))
                    {
                        infer_subst_into(
                            &pattern,
                            &held,
                            &generic.type_params,
                            &mut bound,
                        );
                    }
                }
            }
        }
        Some(substitute_type(&declared, &bound))
    }

    fn raw_pointer_element_address(
        &mut self,
        base: ExprId,
        pointee: Type,
        index_operand: IrOperand,
        index_type: Type,
    ) -> Result<(IrOperand, Type)> {
        let (base_pointer, _) = self.lower_expression(base, None)?;
        let element_size = self.builder.byte_size(&pointee);
        let index = self.coerce(index_operand, &index_type, &Type::I64)?;
        let result =
            self.fresh_local(Type::Ptr(Box::new(pointee.clone())), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::ElementAddress {
                base: base_pointer,
                index,
                element_size,
            },
        ));
        Ok((IrOperand::Local(result), pointee))
    }

    // A `str` is a `[]u8` (3.2), so `str_len` reads the length of either. Asking
    // for `Type::Str` alone refused `str_len(bytes)` here and took it in the
    // self-hosted compiler, which holds the two as one type. That is the same
    // fault `slice_value_address` below had, at the fourth site in this compiler
    // to be told that a str is a byte slice.
    fn str_value_address(&mut self, expression: ExprId) -> Result<IrOperand> {
        if is_byte_run(self.probe_type(expression).as_ref()) {
            let (address, _) = self.place_address(expression)?;
            return Ok(address);
        }
        let (operand, value_type) =
            self.lower_expression(expression, Some(&Type::Str))?;
        if !is_byte_run(Some(&value_type)) {
            bail!("expected a str value, found {value_type}");
        }
        let IrOperand::Local(local) = operand else {
            bail!("str value is not addressable");
        };
        self.mark_in_memory(local);
        Ok(self.address_of_local(local, &Type::Str))
    }

    fn str_field(
        &mut self,
        base: IrOperand,
        offset: usize,
        field_type: Type,
    ) -> IrOperand {
        let slot =
            self.fresh_local(Type::Ptr(Box::new(field_type.clone())), None);
        self.emit(IrStatement::Assign(
            slot,
            IrRvalue::FieldAddress { base, offset },
        ));
        let result = self.fresh_local(field_type.clone(), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Load {
                address: IrOperand::Local(slot),
                ty: field_type,
            },
        ));
        IrOperand::Local(result)
    }

    fn lower_str_len(
        &mut self,
        arguments: &[ExprId],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 1 {
            bail!("str_len expects one argument");
        }
        let base = self.str_value_address(arguments[0])?;
        let length = self.str_field(base, STR_LEN_OFFSET, Type::I64);
        Ok((length, Type::I64))
    }

    fn str_byte_address(
        &mut self,
        base: ExprId,
        index_operand: IrOperand,
        index_type: Type,
    ) -> Result<(IrOperand, Type)> {
        let str_address = self.str_value_address(base)?;
        let data = self.str_field(
            str_address.clone(),
            STR_PTR_OFFSET,
            str_byte_ptr_type(),
        );
        let length = self.str_field(str_address, STR_LEN_OFFSET, Type::Usize);
        let index = self.coerce(index_operand, &index_type, &Type::I64)?;
        let length = self.coerce(length, &Type::Usize, &Type::I64)?;
        let check = self.fresh_local(Type::Void, None);
        self.emit(IrStatement::Assign(
            check,
            IrRvalue::Call {
                function: "frost_rt_bounds_check".to_string(),
                arguments: vec![index.clone(), length],
            },
        ));
        let element = self.fresh_local(str_byte_ptr_type(), None);
        self.emit(IrStatement::Assign(
            element,
            IrRvalue::ElementAddress {
                base: data,
                index,
                element_size: 1,
            },
        ));
        Ok((IrOperand::Local(element), Type::U8))
    }

    // Build a `[]T` fat pointer viewing the whole of an in-memory `[N]T` array:
    // the data pointer is the array's address, the length is the element count.
    fn build_slice_from_address(
        &mut self,
        base: IrOperand,
        element: &Type,
        count: usize,
    ) -> IrOperand {
        let slice_type = Type::Slice(Box::new(element.clone()));
        let slice_local = self.fresh_local(slice_type, None);
        self.mark_in_memory(slice_local);
        let ptr_type = Type::Ptr(Box::new(element.clone()));
        let ptr_slot = self.fresh_local(Type::Ptr(Box::new(ptr_type)), None);
        self.emit(IrStatement::Assign(
            ptr_slot,
            IrRvalue::AddressOf {
                local: slice_local,
                offset: SLICE_PTR_OFFSET,
            },
        ));
        self.emit(IrStatement::Store {
            address: IrOperand::Local(ptr_slot),
            value: base,
        });
        let len_slot = self.fresh_local(Type::Ptr(Box::new(Type::Usize)), None);
        self.emit(IrStatement::Assign(
            len_slot,
            IrRvalue::AddressOf {
                local: slice_local,
                offset: SLICE_LEN_OFFSET,
            },
        ));
        self.emit(IrStatement::Store {
            address: IrOperand::Local(len_slot),
            value: IrOperand::Constant(IrConstant::Integer(
                count as i64,
                Type::Usize,
            )),
        });
        IrOperand::Local(slice_local)
    }

    fn slice_value_address(&mut self, expression: ExprId) -> Result<IrOperand> {
        // A slice that lives somewhere addressable, reached by any place chain:
        // a local, a struct field holding one, or a `mut` parameter, which
        // param-mode lowering turns into a deref of a pointer to the slice.
        // `place_address` walks all three, so recognizing the slice is what was
        // missing, not addressing it.
        // A `str` is a `[]u8` (3.2), so its length is read the same way. Asking
        // for `Type::Slice` alone refused `slice_len(text)` here and took it in
        // the self-hosted compiler, which holds the two as one type.
        if matches!(
            self.probe_type(expression),
            Some(Type::Slice(_) | Type::Str)
        ) {
            let (address, _) = self.place_address(expression)?;
            return Ok(address);
        }
        let (operand, value_type) = self.lower_expression(expression, None)?;
        let (Type::Slice(_) | Type::Str) = value_type else {
            bail!("expected a slice value, found {value_type}");
        };
        let IrOperand::Local(local) = operand else {
            bail!("slice value is not addressable");
        };
        self.mark_in_memory(local);
        Ok(self.address_of_local(local, &value_type))
    }

    fn slice_element_of(&self, base: ExprId) -> Option<Type> {
        match self.probe_type(base) {
            Some(Type::Slice(element)) => Some(*element),
            _ => None,
        }
    }

    fn slice_element_address(
        &mut self,
        base: ExprId,
        index_operand: IrOperand,
        index_type: Type,
        element: Type,
    ) -> Result<(IrOperand, Type)> {
        let slice_address = self.slice_value_address(base)?;
        self.slice_element_address_from(
            slice_address,
            index_operand,
            index_type,
            element,
        )
    }

    // Index a slice given the address of the slice value itself. Both a slice
    // held in a place and one produced by an expression reach here, since the
    // element lives behind the data pointer either way.
    fn slice_element_address_from(
        &mut self,
        slice_address: IrOperand,
        index_operand: IrOperand,
        index_type: Type,
        element: Type,
    ) -> Result<(IrOperand, Type)> {
        let element_ptr = Type::Ptr(Box::new(element.clone()));
        let data = self.str_field(
            slice_address.clone(),
            SLICE_PTR_OFFSET,
            element_ptr.clone(),
        );
        let length =
            self.str_field(slice_address, SLICE_LEN_OFFSET, Type::Usize);
        let index = self.coerce(index_operand, &index_type, &Type::I64)?;
        let length = self.coerce(length, &Type::Usize, &Type::I64)?;
        let check = self.fresh_local(Type::Void, None);
        self.emit(IrStatement::Assign(
            check,
            IrRvalue::Call {
                function: "frost_rt_bounds_check".to_string(),
                arguments: vec![index.clone(), length],
            },
        ));
        let element_size = self.builder.byte_size(&element);
        let result = self.fresh_local(element_ptr, None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::ElementAddress {
                base: data,
                index,
                element_size,
            },
        ));
        Ok((IrOperand::Local(result), element))
    }

    fn lower_slice_len(
        &mut self,
        arguments: &[ExprId],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 1 {
            bail!("slice_len expects one argument");
        }
        // A fixed array carries its length in its type, so this is a number
        // worked out here rather than a field read. The self-hosted compiler
        // answered it and the bootstrap refused, and an array that coerces to a
        // slice everywhere else has a length here too.
        if let Some(Type::Array(_, count)) = self.probe_type(arguments[0]) {
            return Ok((
                IrOperand::Constant(IrConstant::Integer(
                    count as i64,
                    Type::I64,
                )),
                Type::I64,
            ));
        }
        let base = self.slice_value_address(arguments[0])?;
        let length = self.str_field(base, SLICE_LEN_OFFSET, Type::I64);
        Ok((length, Type::I64))
    }

    // `flags_has(chosen, InitFlags::Video)`: whether every bit on the right is
    // on in the left. Written here rather than in a library because it is the
    // one question a set of bits is asked, and a library function would have to
    // be generic over a type that has no operations to be generic over.
    //
    // Both sides being the same flags type is the check that makes it a
    // question about one set rather than about two numbers, so a window flag
    // asked of an init flag is refused.
    fn lower_flags_has(
        &mut self,
        arguments: &[ExprId],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 2 {
            bail!(
                "flags_has takes the set and the bits to look for, as in 'flags_has(chosen, InitFlags::Video)'"
            );
        }
        let (set, set_type) = self.lower_expression(arguments[0], None)?;
        let (wanted, wanted_type) =
            self.lower_expression(arguments[1], Some(&set_type))?;
        if distinct_mismatch(
            self.ast,
            arguments[1],
            &wanted_type,
            &set_type,
            &self.builder.flags,
        ) {
            let (described, note) = nominal_words(
                self.ast,
                arguments[1],
                &wanted_type,
                &set_type,
                &self.builder.flags,
            );
            bail!(
                "flags_has looks for a '{set_type}' and this is {described}; {note}"
            );
        }
        let Some(name) = self.flags_name_of(&set_type) else {
            bail!(
                "flags_has asks a set of bits what it holds, and a '{set_type}' is not one"
            );
        };
        if self.flags_name_of(&wanted_type) != Some(name) {
            let readable =
                crate::modules::imports::demangle_private_names(name);
            bail!(
                "flags_has looks for bits of the set it is given, and a '{wanted_type}' is not a '{readable}'"
            );
        }
        let repr = match &set_type {
            Type::Distinct(_, inner) => inner.as_ref().clone(),
            other => other.clone(),
        };
        let narrowed = self.fresh_local(repr.clone(), None);
        self.emit(IrStatement::Assign(
            narrowed,
            IrRvalue::Binary(IrBinOp::BitwiseAnd, set, wanted.clone()),
        ));
        let result = self.fresh_local(Type::Bool, None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Binary(
                IrBinOp::Equal,
                IrOperand::Local(narrowed),
                wanted,
            ),
        ));
        Ok((IrOperand::Local(result), Type::Bool))
    }

    // A first-class raw pointer to a place. `&x` is a second-class reference.
    // ptr_to gives the same address as a `^T` that may be stored and returned.
    fn lower_ptr_to(
        &mut self,
        arguments: &[ExprId],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 1 {
            bail!("ptr_to expects one place argument");
        }
        let (address, pointee) = self.place_address(arguments[0])?;
        Ok((address, Type::Ptr(Box::new(pointee))))
    }

    // Reinterpret a pointer value as `^T`. A pointer is a pointer at the ABI, so
    // this is a retype with no runtime cost.
    // `slice_from($T, pointer, length)`: a `[]T` view over a run of `T` the
    // caller vouches for. This is the primitive a heap-backed container is
    // built on, since a slice is the safe, bounds-checked face it presents over
    // its own raw storage. Trusting the pointer and length is unchecked, which
    // is why forming one is a gated operation like ptr_cast.
    fn lower_slice_from(
        &mut self,
        arguments: &[ExprId],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 3 {
            bail!(
                "slice_from expects a type, a pointer and a length, as in slice_from($T, p, n)"
            );
        }
        let Expression::TypeValue(element) = self.ast.expr(arguments[0]) else {
            bail!("slice_from's first argument must be a type, as in $Entity");
        };
        let element = element.clone();
        let (pointer, _) = self.lower_expression(arguments[1], None)?;
        let (length, length_type) =
            self.lower_expression(arguments[2], None)?;
        // Checked while it is still signed. The bounds check every later access
        // goes through compares unsigned, so a negative length would read as
        // enormous there and let every index past the end of the run.
        let length = self.coerce(length, &length_type, &Type::I64)?;
        let checked = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(
            checked,
            IrRvalue::Call {
                function: "frost_rt_check_length".to_string(),
                arguments: vec![length],
            },
        ));
        let length =
            self.coerce(IrOperand::Local(checked), &Type::I64, &Type::Usize)?;
        let slice_type = Type::Slice(Box::new(element.clone()));
        let slice_local = self.fresh_local(slice_type.clone(), None);
        self.mark_in_memory(slice_local);
        let ptr_slot = self.fresh_local(
            Type::Ptr(Box::new(Type::Ptr(Box::new(element.clone())))),
            None,
        );
        self.emit(IrStatement::Assign(
            ptr_slot,
            IrRvalue::AddressOf {
                local: slice_local,
                offset: SLICE_PTR_OFFSET,
            },
        ));
        self.emit(IrStatement::Store {
            address: IrOperand::Local(ptr_slot),
            value: pointer,
        });
        let len_slot = self.fresh_local(Type::Ptr(Box::new(Type::Usize)), None);
        self.emit(IrStatement::Assign(
            len_slot,
            IrRvalue::AddressOf {
                local: slice_local,
                offset: SLICE_LEN_OFFSET,
            },
        ));
        self.emit(IrStatement::Store {
            address: IrOperand::Local(len_slot),
            value: length,
        });
        Ok((IrOperand::Local(slice_local), slice_type))
    }

    /// `wrap_add(a, b)`, `wrap_sub(a, b)`, `wrap_mul(a, b)`: arithmetic that
    /// keeps the low bits of its type and drops the rest.
    ///
    /// Ordinary arithmetic refuses a result that does not fit, because a count
    /// that overflowed is a wrong number that keeps going. A hash is the case
    /// where leaving the range is the point, and it is spelled rather than
    /// assumed, so a reader can tell the two apart at the site.
    fn lower_wrapping(
        &mut self,
        op: IrBinOp,
        arguments: &[ExprId],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 2 {
            bail!("this takes two numbers, as in wrap_mul(a, b)");
        }
        let (left, left_type) = self.lower_expression(arguments[0], None)?;
        let (right, right_type) =
            self.lower_expression(arguments[1], Some(&left_type))?;
        let right = self.coerce(right, &right_type, &left_type)?;
        let result = self.fresh_local(left_type.clone(), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Binary(op, left, right),
        ));
        Ok((IrOperand::Local(result), left_type))
    }

    /// `cast($T, value)`: a conversion the reader asked for.
    ///
    /// It is written the way its neighbours are, `ptr_cast($T, p)` and
    /// `slice_from($T, p, n)` and `sizeof(T)`, so it needs no keyword, no new
    /// precedence level, and no parsing that did not already exist.
    ///
    /// What it is for is the other half: without it, a conversion that loses
    /// something has no spelling, so refusing one would leave no way to say you
    /// meant it. With it, `held : u8 = wide` is refused and
    /// `held : u8 = cast($u8, wide)` is what you write instead.
    fn lower_cast(
        &mut self,
        arguments: &[ExprId],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 2 {
            bail!("cast expects a type and a value, as in cast($u8, n)");
        }
        let Expression::TypeValue(target) = self.ast.expr(arguments[0]) else {
            bail!("cast's first argument must be a type, as in $u8");
        };
        let target = target.clone();
        let (value, from) = self.lower_expression(arguments[1], None)?;
        if !(is_numeric(&from) && is_numeric(&target))
            && !names_a_distinct(&from, &target)
        {
            bail!(
                "cast converts between numbers, or names a distinct type for a value of its representation, and this is asked to turn a {from} into a {target}"
            );
        }
        if from == target {
            return Ok((value, target));
        }
        let result = self.fresh_local(target.clone(), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::Cast(value, target.clone()),
        ));
        Ok((IrOperand::Local(result), target))
    }

    fn lower_ptr_cast(
        &mut self,
        arguments: &[ExprId],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 2 {
            bail!(
                "ptr_cast expects a type and a pointer, as in ptr_cast($T, p)"
            );
        }
        let Expression::TypeValue(target) = self.ast.expr(arguments[0]) else {
            bail!("ptr_cast's first argument must be a type, as in $Entity");
        };
        // A function type is already an address, so naming one asks for that
        // function type rather than for a pointer to it. This is what lets a
        // table of callbacks hold one shape while each registration is written
        // against the state it belongs to.
        let target = match target {
            Type::Proc(_, _) => target.clone(),
            _ => Type::Ptr(Box::new(target.clone())),
        };
        let (pointer, _) = self.lower_expression(arguments[1], None)?;
        let result = self.fresh_local(target.clone(), None);
        self.emit(IrStatement::Assign(result, IrRvalue::Use(pointer)));
        Ok((IrOperand::Local(result), target))
    }

    fn place_address(&mut self, place: ExprId) -> Result<(IrOperand, Type)> {
        match self.ast.expr(place).clone() {
            Expression::Identifier(name) => {
                let name = self.ast.name(name).to_string();
                let Some(local) = self.resolve_variable(&name) else {
                    // A constant has no storage of its own, so the address of
                    // one is the address of the copy built here. This is what a
                    // bundle passed at runtime, rather than as a compile-time
                    // argument, travels as.
                    if let Some(value) =
                        self.builder.constants.get(&name).copied()
                    {
                        // A constant naming a place is that place. One naming a
                        // value has none, so the copy built here is what the
                        // address is of: `fs_read(PATH)` with `PATH :: "x"` is
                        // this, and a string constant is the common case.
                        if is_place_expression(self.ast, value) {
                            return self.place_address(value);
                        }
                        let (operand, ty) =
                            self.lower_expression(value, None)?;
                        if let IrOperand::Local(local) = operand {
                            self.mark_in_memory(local);
                            let address = self.address_of_local(local, &ty);
                            return Ok((address, ty));
                        }
                        let held = self.fresh_local(ty.clone(), None);
                        self.mark_in_memory(held);
                        self.emit(IrStatement::Assign(
                            held,
                            IrRvalue::Use(operand),
                        ));
                        let address = self.address_of_local(held, &ty);
                        return Ok((address, ty));
                    }
                    return locate(
                        Err(anyhow::anyhow!("unknown variable '{name}'")),
                        self.at_expression(place),
                    );
                };
                self.mark_in_memory(local);
                let pointee = self.type_of_local(local);
                let result = self
                    .fresh_local(Type::Ptr(Box::new(pointee.clone())), None);
                self.emit(IrStatement::Assign(
                    result,
                    IrRvalue::AddressOf { local, offset: 0 },
                ));
                Ok((IrOperand::Local(result), pointee))
            }
            Expression::FieldAccess(base, field) => {
                let field = self.ast.name(field).to_string();
                self.field_address(base, &field)
            }
            Expression::Index(base, index) => self.element_address(base, index),
            Expression::Dereference(pointer) => {
                self.place_address_of_deref(pointer)
            }
            Expression::StructInit(..)
            | Expression::EnumVariantInit(..)
            | Expression::Literal(Literal::Array(_)) => {
                let (operand, ty) = self.lower_expression(place, None)?;
                let IrOperand::Local(local) = operand else {
                    bail!("cannot take the address of this value");
                };
                Ok((self.address_of_local(local, &ty), ty))
            }
            _ => {
                bail!(
                    "expression is not an assignable place: {}",
                    display_expr(self.ast, place)
                )
            }
        }
    }

    fn element_address(
        &mut self,
        base: ExprId,
        index: ExprId,
    ) -> Result<(IrOperand, Type)> {
        // A constant is its value wherever it is named, and every question
        // below asks what the base is before it asks where it lives: whether it
        // is a string, a slice, a raw pointer. A name none of them can resolve
        // reaches the array path and comes back as an unknown variable, so the
        // value goes in ahead of them and they see a string literal rather than
        // a name. Before the index is lowered, so it is lowered once.
        if let Expression::Identifier(name) = self.ast.expr(base) {
            let name = self.ast.name(*name).to_string();
            if self.resolve_variable(&name).is_none()
                && let Some(value) = self.builder.constants.get(&name).copied()
            {
                // The value stands where the name is written, so a refusal
                // about it points there rather than at the declaration it was
                // read from, which is a line the reader did not index.
                let written = self.at_expression(base);
                return self.element_address(value, index).map_err(|error| {
                    match error
                        .downcast_ref::<crate::diagnostic::LocatedError>()
                    {
                        Some(held) => anyhow::Error::new(
                            crate::diagnostic::LocatedError {
                                position: written,
                                message: held.message.clone(),
                            },
                        ),
                        None => error,
                    }
                });
            }
        }
        let (index_operand, index_type) = self.lower_expression(index, None)?;
        if matches!(index_type, Type::Handle(_)) {
            if let Some(struct_name) = self.slab_shaped_base(base) {
                return self.slab_place_deref(
                    base,
                    &struct_name,
                    index_operand,
                );
            }
            bail!(
                "indexing by a Handle needs a slab-shaped struct, one with a 'storage' array and a parallel 'generations' array; see std/slab.frost"
            );
        }
        // The literal as well as a place holding one, because a string constant
        // is the literal it was written as by the time it gets here.
        if matches!(self.probe_type(base), Some(Type::Str))
            || matches!(
                self.ast.expr(base),
                Expression::Literal(Literal::String(_))
            )
        {
            return self.str_byte_address(base, index_operand, index_type);
        }
        if let Some(element) = self.slice_element_of(base) {
            return self.slice_element_address(
                base,
                index_operand,
                index_type,
                element,
            );
        }
        if let Some(Type::Ptr(pointee)) = self.probe_type(base) {
            return self.raw_pointer_element_address(
                base,
                *pointee,
                index_operand,
                index_type,
            );
        }
        // A slice produced by a value expression rather than held in a place,
        // such as a container's `vec_slice(v)`. Its data pointer aliases real
        // storage, so indexing the spilled temporary reads and writes there.
        // Only a non-place base reaches here without a matched probe, so this
        // does not lower a place twice.
        let (base_pointer, element_type, length) =
            if !is_place_expression(self.ast, base) {
                let (value, value_type) = self.lower_expression(base, None)?;
                let IrOperand::Local(local) = value else {
                    return Err(self.not_a_run(base, &value_type));
                };
                match value_type {
                    Type::Slice(element) => {
                        self.mark_in_memory(local);
                        let slice_address = self.address_of_local(
                            local,
                            &Type::Slice(element.clone()),
                        );
                        return self.slice_element_address_from(
                            slice_address,
                            index_operand,
                            index_type,
                            *element,
                        );
                    }
                    // An array with no place of its own: a constant written out
                    // here, or what a call handed back. Spilling it gives the index
                    // something to be an offset from, and the count it was declared
                    // with is still known, so it is bounds-checked like any other.
                    Type::Array(element, count) => {
                        self.mark_in_memory(local);
                        let result = self.fresh_local(
                            Type::Ptr(Box::new((*element).clone())),
                            None,
                        );
                        self.emit(IrStatement::Assign(
                            result,
                            IrRvalue::AddressOf { local, offset: 0 },
                        ));
                        (IrOperand::Local(result), *element, Some(count))
                    }
                    other => return Err(self.not_a_run(base, &other)),
                }
            } else {
                self.array_base_pointer(base)?
            };
        let element_size = self.builder.byte_size(&element_type);
        let index_operand =
            self.coerce(index_operand, &index_type, &Type::I64)?;
        if let Some(length) = length {
            let check_result = self.fresh_local(Type::Void, None);
            self.emit(IrStatement::Assign(
                check_result,
                IrRvalue::Call {
                    function: "frost_rt_bounds_check".to_string(),
                    arguments: vec![
                        index_operand.clone(),
                        IrOperand::Constant(IrConstant::Integer(
                            length as i64,
                            Type::I64,
                        )),
                    ],
                },
            ));
        }
        let result =
            self.fresh_local(Type::Ptr(Box::new(element_type.clone())), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::ElementAddress {
                base: base_pointer,
                index: index_operand,
                element_size,
            },
        ));
        Ok((IrOperand::Local(result), element_type))
    }

    // A struct is "slab-shaped" when it has a `storage` array and a parallel
    // `generations` array, the layout of a generational pool. Indexing such a
    // struct by a Handle is a validated place-deref, generated inline.
    fn slab_shaped_base(&self, base: ExprId) -> Option<String> {
        let Type::Struct(name) = self.probe_type(base)? else {
            return None;
        };
        let layout = self.builder.struct_layout(&name)?;
        let is_array_field = |field: &str| {
            layout
                .field(field)
                .is_some_and(|field| matches!(field.ty, Type::Array(..)))
        };
        if is_array_field("storage") && is_array_field("generations") {
            Some(name)
        } else {
            None
        }
    }

    // Address of `struct.field[index]` where `field` is an array at `offset`.
    fn slab_field_element_address(
        &mut self,
        struct_address: IrOperand,
        field_offset: usize,
        element: &Type,
        index: IrOperand,
    ) -> IrOperand {
        let field_address =
            self.fresh_local(Type::Ptr(Box::new(element.clone())), None);
        self.emit(IrStatement::Assign(
            field_address,
            IrRvalue::FieldAddress {
                base: struct_address,
                offset: field_offset,
            },
        ));
        let element_size = self.builder.byte_size(element);
        let result =
            self.fresh_local(Type::Ptr(Box::new(element.clone())), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::ElementAddress {
                base: IrOperand::Local(field_address),
                index,
                element_size,
            },
        ));
        IrOperand::Local(result)
    }

    fn slab_place_deref(
        &mut self,
        base: ExprId,
        struct_name: &str,
        handle: IrOperand,
    ) -> Result<(IrOperand, Type)> {
        let (storage_offset, element, count, generations_offset) = {
            let layout =
                self.builder.struct_layout(struct_name).ok_or_else(|| {
                    anyhow::anyhow!("unknown slab '{struct_name}'")
                })?;
            let storage = layout.field("storage").ok_or_else(|| {
                anyhow::anyhow!("slab has no 'storage' field")
            })?;
            let generations = layout.field("generations").ok_or_else(|| {
                anyhow::anyhow!("slab has no 'generations' field")
            })?;
            let Type::Array(inner, count) = &storage.ty else {
                bail!("slab 'storage' is not an array");
            };
            (
                storage.offset,
                (**inner).clone(),
                *count,
                generations.offset,
            )
        };

        let (struct_address, _) = self.struct_place(base)?;

        // The handle is a `Handle<T>`, opaque and non-numeric. Reinterpret it as
        // the i64 it is at the ABI before taking it apart.
        let raw_handle = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(raw_handle, IrRvalue::Use(handle)));
        let raw_handle = IrOperand::Local(raw_handle);

        let index = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(
            index,
            IrRvalue::Binary(
                IrBinOp::BitwiseAnd,
                raw_handle.clone(),
                IrOperand::Constant(IrConstant::Integer(
                    0xffff_ffff,
                    Type::I64,
                )),
            ),
        ));
        let generation = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(
            generation,
            IrRvalue::Binary(
                IrBinOp::ShiftRight,
                raw_handle,
                IrOperand::Constant(IrConstant::Integer(32, Type::I64)),
            ),
        ));

        let bounds = self.fresh_local(Type::Void, None);
        self.emit(IrStatement::Assign(
            bounds,
            IrRvalue::Call {
                function: "frost_rt_bounds_check".to_string(),
                arguments: vec![
                    IrOperand::Local(index),
                    IrOperand::Constant(IrConstant::Integer(
                        count as i64,
                        Type::I64,
                    )),
                ],
            },
        ));

        let generation_slot = self.slab_field_element_address(
            struct_address.clone(),
            generations_offset,
            &Type::I64,
            IrOperand::Local(index),
        );
        let stored = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(
            stored,
            IrRvalue::Load {
                address: generation_slot,
                ty: Type::I64,
            },
        ));
        let generation_check = self.fresh_local(Type::Void, None);
        self.emit(IrStatement::Assign(
            generation_check,
            IrRvalue::Call {
                function: "frost_rt_generation_check".to_string(),
                arguments: vec![
                    IrOperand::Local(stored),
                    IrOperand::Local(generation),
                ],
            },
        ));

        let element_address = self.slab_field_element_address(
            struct_address,
            storage_offset,
            &element,
            IrOperand::Local(index),
        );
        Ok((element_address, element))
    }

    // A columns container, the SoA sibling of a slab, recognized by its
    // synthesized `columns<...>` name. Its data is one array per field of the
    // element rather than one `storage` array, so a Handle deref picks a column
    // before indexing.
    fn columns_shaped_base(&self, base: ExprId) -> Option<String> {
        match self.probe_type(base)? {
            Type::Struct(name) if name.starts_with("columns<") => Some(name),
            _ => None,
        }
    }

    // The address of one element of one column, named by `c[handle].field`. The
    // same handle validation as a slab (`frost_rt_bounds_check` +
    // `frost_rt_generation_check`), but the checked index scales into the field's
    // own `[N]field` column rather than a single storage run.
    fn columns_place_deref(
        &mut self,
        base: ExprId,
        struct_name: &str,
        handle: IrOperand,
        field: &str,
    ) -> Result<(IrOperand, Type)> {
        let (column_offset, column_element, count, generations_offset) = {
            let layout =
                self.builder.struct_layout(struct_name).ok_or_else(|| {
                    anyhow::anyhow!("unknown columns '{struct_name}'")
                })?;
            let column = layout.field(field).ok_or_else(|| {
                anyhow::anyhow!(
                    "columns '{struct_name}' has no field '{field}'"
                )
            })?;
            let generations = layout.field("generations").ok_or_else(|| {
                anyhow::anyhow!("columns has no 'generations' field")
            })?;
            let Type::Array(_, count) = &generations.ty else {
                bail!("columns 'generations' is not an array");
            };
            let Type::Array(element, _) = &column.ty else {
                bail!("columns field '{field}' is not a column array");
            };
            (
                column.offset,
                (**element).clone(),
                *count,
                generations.offset,
            )
        };

        let (struct_address, _) = self.struct_place(base)?;

        let raw_handle = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(raw_handle, IrRvalue::Use(handle)));
        let raw_handle = IrOperand::Local(raw_handle);
        let index = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(
            index,
            IrRvalue::Binary(
                IrBinOp::BitwiseAnd,
                raw_handle.clone(),
                IrOperand::Constant(IrConstant::Integer(
                    0xffff_ffff,
                    Type::I64,
                )),
            ),
        ));
        let generation = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(
            generation,
            IrRvalue::Binary(
                IrBinOp::ShiftRight,
                raw_handle,
                IrOperand::Constant(IrConstant::Integer(32, Type::I64)),
            ),
        ));

        let bounds = self.fresh_local(Type::Void, None);
        self.emit(IrStatement::Assign(
            bounds,
            IrRvalue::Call {
                function: "frost_rt_bounds_check".to_string(),
                arguments: vec![
                    IrOperand::Local(index),
                    IrOperand::Constant(IrConstant::Integer(
                        count as i64,
                        Type::I64,
                    )),
                ],
            },
        ));

        let generation_slot = self.slab_field_element_address(
            struct_address.clone(),
            generations_offset,
            &Type::I64,
            IrOperand::Local(index),
        );
        let stored = self.fresh_local(Type::I64, None);
        self.emit(IrStatement::Assign(
            stored,
            IrRvalue::Load {
                address: generation_slot,
                ty: Type::I64,
            },
        ));
        let generation_check = self.fresh_local(Type::Void, None);
        self.emit(IrStatement::Assign(
            generation_check,
            IrRvalue::Call {
                function: "frost_rt_generation_check".to_string(),
                arguments: vec![
                    IrOperand::Local(stored),
                    IrOperand::Local(generation),
                ],
            },
        ));

        let element_address = self.slab_field_element_address(
            struct_address,
            column_offset,
            &column_element,
            IrOperand::Local(index),
        );
        Ok((element_address, column_element))
    }

    // `c[h] = value` scatters the element's fields into the columns' per-field
    // arrays at the handle's validated slot, one field at a time. The handle is
    // re-validated per field, aborting identically if it is stale.
    fn columns_scatter(
        &mut self,
        base: ExprId,
        struct_name: &str,
        handle: IrOperand,
        value: ExprId,
    ) -> Result<()> {
        let column_fields: Vec<String> = {
            let layout =
                self.builder.struct_layout(struct_name).ok_or_else(|| {
                    anyhow::anyhow!("unknown columns '{struct_name}'")
                })?;
            layout
                .fields
                .iter()
                .filter(|field| !is_columns_bookkeeping(&field.name))
                .map(|field| field.name.clone())
                .collect()
        };

        let (value_operand, value_type) = self.lower_expression(value, None)?;
        let IrOperand::Local(value_local) = value_operand else {
            bail!("a columns scatter needs an addressable element");
        };
        let value_address = self.address_of_local(value_local, &value_type);
        let Type::Struct(value_name) = &value_type else {
            bail!("a columns element value must be a struct");
        };

        for field in &column_fields {
            let (field_offset, field_ty) = {
                let value_layout = self
                    .builder
                    .struct_layout(value_name)
                    .ok_or_else(|| {
                        anyhow::anyhow!("unknown element struct '{value_name}'")
                    })?;
                let layout_field =
                    value_layout.field(field).ok_or_else(|| {
                        anyhow::anyhow!(
                            "element '{value_name}' has no field '{field}'"
                        )
                    })?;
                (layout_field.offset, layout_field.ty.clone())
            };
            let (destination, _) = self.columns_place_deref(
                base,
                struct_name,
                handle.clone(),
                field,
            )?;
            let source =
                self.fresh_local(Type::Ptr(Box::new(field_ty.clone())), None);
            self.emit(IrStatement::Assign(
                source,
                IrRvalue::FieldAddress {
                    base: value_address.clone(),
                    offset: field_offset,
                },
            ));
            if needs_memory(&field_ty) {
                let size = self.builder.byte_size(&field_ty);
                self.emit(IrStatement::Copy {
                    destination,
                    source: IrOperand::Local(source),
                    size,
                });
            } else {
                let loaded = self.fresh_local(field_ty.clone(), None);
                self.emit(IrStatement::Assign(
                    loaded,
                    IrRvalue::Load {
                        address: IrOperand::Local(source),
                        ty: field_ty.clone(),
                    },
                ));
                self.emit(IrStatement::Store {
                    address: destination,
                    value: IrOperand::Local(loaded),
                });
            }
        }
        Ok(())
    }

    // `columns_new()`: a zeroed columns container of the type the context wants.
    // Zeroing sets every generation and free slot to 0. `columns_reset` lays out
    // the free list before use, the same "construct then reset" contract a slab
    // has.
    fn lower_columns_new(
        &mut self,
        called: &str,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        let slab = called == "slab_new";
        let wanted = if slab { "Slab<T, N>" } else { "columns<T, N>" };
        let Some(Type::Struct(name)) = expected else {
            bail!(
                "{called}() needs its type from the context, e.g. `mut c : {wanted} = {called}()`"
            );
        };
        // A columns container is known by its name, since the compiler is what
        // made it. A slab is known by its shape, the way it is everywhere else,
        // and by the name the standard library gives it, since an instance
        // whose element is itself an instance has no layout to read yet at the
        // point this is lowered.
        let recognized = if slab {
            name.starts_with("Slab<")
                || self.builder.struct_layout(name).is_some_and(|layout| {
                    layout.field("storage").is_some()
                        && layout.field("generations").is_some()
                })
        } else {
            name.starts_with("columns<")
        };
        if !recognized {
            bail!("{called}() initializes a `{wanted}`, not '{name}'");
        }
        let ty = Type::Struct(name.clone());
        let size = self.builder.byte_size(&ty) as i64;
        let local = self.fresh_local(ty.clone(), None);
        self.mark_in_memory(local);
        let address = self.address_of_local(local, &ty);
        let cleared = self.fresh_local(Type::Void, None);
        self.emit(IrStatement::Assign(
            cleared,
            IrRvalue::Call {
                function: "frost_rt_mem_set".to_string(),
                arguments: vec![
                    address,
                    IrOperand::Constant(IrConstant::Integer(0, Type::I64)),
                    IrOperand::Constant(IrConstant::Integer(size, Type::I64)),
                ],
            },
        ));
        Ok((IrOperand::Local(local), ty))
    }

    // What indexing asks of its base, said the same way wherever it is asked.
    // The caret is on the base, so naming what it turned out to be is what the
    // reader has not got.
    fn not_a_run(&self, base: ExprId, ty: &Type) -> anyhow::Error {
        anyhow::Error::new(crate::diagnostic::LocatedError {
            position: self.at_expression(base),
            message: format!(
                "indexing reads an element out of a run, and this is a {ty}"
            ),
        })
    }

    fn array_base_pointer(
        &mut self,
        base: ExprId,
    ) -> Result<(IrOperand, Type, Option<usize>)> {
        match self.ast.expr(base).clone() {
            Expression::Identifier(name) => {
                let name = self.ast.name(name).to_string();
                let Some(local) = self.resolve_variable(&name) else {
                    return locate(
                        Err(anyhow::anyhow!("unknown variable '{name}'")),
                        self.at_expression(base),
                    );
                };
                match self.type_of_local(local) {
                    Type::Array(element, count) => {
                        self.mark_in_memory(local);
                        let result = self.fresh_local(
                            Type::Ptr(Box::new((*element).clone())),
                            None,
                        );
                        self.emit(IrStatement::Assign(
                            result,
                            IrRvalue::AddressOf { local, offset: 0 },
                        ));
                        Ok((IrOperand::Local(result), *element, Some(count)))
                    }
                    Type::Ref(inner)
                    | Type::RefMut(inner)
                    | Type::Ptr(inner)
                        if matches!(*inner, Type::Array(_, _)) =>
                    {
                        let Type::Array(element, count) = *inner else {
                            unreachable!()
                        };
                        Ok((IrOperand::Local(local), *element, Some(count)))
                    }
                    other => Err(self.not_a_run(base, &other)),
                }
            }
            Expression::FieldAccess(inner, field) => {
                let field = self.ast.name(field).to_string();
                let (address, field_type) =
                    self.field_address(inner, &field)?;
                let Type::Array(element, count) = field_type else {
                    bail!("field '{field}' is not an array");
                };
                Ok((address, *element, Some(count)))
            }
            // An array reached through a pointer, which is what a `mut [N]T`
            // parameter is once the mode pass has rewritten it: the pointer
            // already holds the array's address, so there is nothing here to
            // take the address of.
            //
            // Every array parameter in the tree is a slice, so nothing had ever
            // written this and it type-checked and then died at lowering. The
            // self-hosted compiler compiles it correctly, which is what says
            // what the answer is rather than whether there should be one.
            Expression::Dereference(pointer) => {
                let (operand, pointer_type) =
                    self.lower_expression(pointer, None)?;
                let (Type::Ptr(inner) | Type::Ref(inner) | Type::RefMut(inner)) =
                    pointer_type.clone()
                else {
                    return Err(self.not_a_run(base, &pointer_type));
                };
                let held = (*inner).clone();
                let Type::Array(element, count) = *inner else {
                    return Err(self.not_a_run(base, &held));
                };
                Ok((operand, *element, Some(count)))
            }
            Expression::Index(inner, index) => {
                let (address, element_type) =
                    self.element_address(inner, index)?;
                let held = element_type.clone();
                let Type::Array(element, count) = element_type else {
                    return Err(self.not_a_run(base, &held));
                };
                Ok((address, *element, Some(count)))
            }
            _ => {
                let held = self.answer_type(base).unwrap_or(Type::Unknown);
                Err(self.not_a_run(base, &held))
            }
        }
    }

    fn init_array(
        &mut self,
        local: LocalId,
        element_type: &Type,
        elements: &[ExprId],
    ) -> Result<()> {
        self.mark_owned(local);
        let element_size = self.builder.byte_size(element_type);
        for (index, element) in elements.iter().enumerate() {
            let address = self
                .fresh_local(Type::Ptr(Box::new(element_type.clone())), None);
            self.emit(IrStatement::Assign(
                address,
                IrRvalue::AddressOf {
                    local,
                    offset: index * element_size,
                },
            ));
            if needs_memory(element_type) {
                let source =
                    self.aggregate_field_source(*element, element_type)?;
                self.emit(IrStatement::Copy {
                    destination: IrOperand::Local(address),
                    source,
                    size: element_size,
                });
            } else {
                let (operand, value_type) =
                    self.lower_expression(*element, Some(element_type))?;
                let coerced =
                    self.coerce(operand, &value_type, element_type)?;
                self.emit(IrStatement::Store {
                    address: IrOperand::Local(address),
                    value: coerced,
                });
            }
        }
        Ok(())
    }

    fn place_address_of_deref(
        &mut self,
        pointer: ExprId,
    ) -> Result<(IrOperand, Type)> {
        let (pointer_operand, pointer_type) =
            self.lower_expression(pointer, None)?;
        let pointee = deref_target(&pointer_type)?;
        Ok((pointer_operand, pointee))
    }

    fn field_address(
        &mut self,
        base: ExprId,
        field: &str,
    ) -> Result<(IrOperand, Type)> {
        // A columns element field `c[h].field`: the field selects a column, the
        // handle a validated slot in it.
        if let Expression::Index(container, index_expr) =
            self.ast.expr(base).clone()
            && let Some(struct_name) = self.columns_shaped_base(container)
        {
            let (index_operand, index_type) =
                self.lower_expression(index_expr, None)?;
            if matches!(index_type, Type::Handle(_)) {
                return self.columns_place_deref(
                    container,
                    &struct_name,
                    index_operand,
                    field,
                );
            }
        }
        let (base_pointer, struct_name) = self.struct_place(base)?;
        let layout =
            self.builder.struct_layout(&struct_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "'{struct_name}' is not a type this program declares"
                )
            })?;
        let field_layout = layout.field(field).ok_or_else(|| {
            anyhow::anyhow!("struct '{struct_name}' has no field '{field}'")
        })?;
        let field_type = field_layout.ty.clone();
        let offset = field_layout.offset;
        let result =
            self.fresh_local(Type::Ptr(Box::new(field_type.clone())), None);
        self.emit(IrStatement::Assign(
            result,
            IrRvalue::FieldAddress {
                base: base_pointer,
                offset,
            },
        ));
        Ok((IrOperand::Local(result), field_type))
    }

    fn struct_place(&mut self, base: ExprId) -> Result<(IrOperand, String)> {
        match self.ast.expr(base).clone() {
            Expression::Identifier(name) => {
                let name = self.ast.name(name).to_string();
                let Some(local) = self.resolve_variable(&name) else {
                    // A top-level constant is its value wherever it is named,
                    // so a field of one is a field of that value.
                    if let Some(value) =
                        self.builder.constants.get(&name).copied()
                    {
                        return self.struct_place(value);
                    }
                    return locate(
                        Err(anyhow::anyhow!("unknown variable '{name}'")),
                        self.at_expression(base),
                    );
                };
                match self.type_of_local(local) {
                    Type::Struct(struct_name) => {
                        self.mark_in_memory(local);
                        let result = self.fresh_local(
                            Type::Ptr(Box::new(Type::Struct(
                                struct_name.clone(),
                            ))),
                            None,
                        );
                        self.emit(IrStatement::Assign(
                            result,
                            IrRvalue::AddressOf { local, offset: 0 },
                        ));
                        Ok((IrOperand::Local(result), struct_name))
                    }
                    Type::Ref(inner)
                    | Type::RefMut(inner)
                    | Type::Ptr(inner)
                        if matches!(*inner, Type::Struct(_)) =>
                    {
                        let Type::Struct(struct_name) = *inner else {
                            unreachable!()
                        };
                        Ok((IrOperand::Local(local), struct_name))
                    }
                    other => bail!("'{name}' is not a struct (found {other})"),
                }
            }
            Expression::FieldAccess(inner, field) => {
                let field = self.ast.name(field).to_string();
                let (address, field_type) =
                    self.field_address(inner, &field)?;
                let Type::Struct(struct_name) = field_type else {
                    bail!("field '{field}' is not a struct");
                };
                Ok((address, struct_name))
            }
            Expression::Dereference(pointer) => {
                let (pointer_operand, pointer_type) =
                    self.lower_expression(pointer, None)?;
                let pointee = deref_target(&pointer_type)?;
                let Type::Struct(struct_name) = pointee else {
                    bail!("dereference is not a struct");
                };
                Ok((pointer_operand, struct_name))
            }
            Expression::Index(inner, index) => {
                let (address, element_type) =
                    self.element_address(inner, index)?;
                let Type::Struct(struct_name) = element_type else {
                    bail!("indexed element is not a struct");
                };
                Ok((address, struct_name))
            }
            // Any other expression that yields a borrow or pointer to a struct
            // names that struct. The operand is its address. This is what lets a
            // borrow-returning accessor be written to, as in `at(b, i).field = x`.
            _ => {
                let (operand, ty) = self.lower_expression(base, None)?;
                match ty {
                    Type::Ref(inner)
                    | Type::RefMut(inner)
                    | Type::Ptr(inner)
                        if matches!(*inner, Type::Struct(_)) =>
                    {
                        let Type::Struct(struct_name) = *inner else {
                            unreachable!()
                        };
                        Ok((operand, struct_name))
                    }
                    // A struct value that is not already a place, such as a
                    // literal or what a call answered, is read out of where it
                    // was built.
                    Type::Struct(struct_name) => {
                        let IrOperand::Local(local) = operand else {
                            bail!(
                                "not a struct place: {}",
                                display_expr(self.ast, base)
                            );
                        };
                        self.mark_in_memory(local);
                        let address = self.address_of_local(
                            local,
                            &Type::Struct(struct_name.clone()),
                        );
                        Ok((address, struct_name))
                    }
                    _ => bail!(
                        "not a struct place: {}",
                        display_expr(self.ast, base)
                    ),
                }
            }
        }
    }

    fn mark_owned(&mut self, local: LocalId) {
        if self.locals[local].linear {
            self.emit(IrStatement::Own(local));
        }
    }

    fn init_struct(
        &mut self,
        local: LocalId,
        struct_name: &str,
        field_inits: Range32,
    ) -> Result<()> {
        self.mark_owned(local);
        let field_inits: Vec<NamedExpr> =
            self.ast.named_in(field_inits).to_vec();
        let fields: Vec<(String, usize, Type)> = {
            let layout =
                self.builder.struct_layout(struct_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "'{struct_name}' is not a type this program declares"
                    )
                })?;
            layout
                .fields
                .iter()
                .map(|field| {
                    (field.name.clone(), field.offset, field.ty.clone())
                })
                .collect()
        };

        // A field left out is storage that is never written, and reading it
        // afterwards reads whatever was on the stack. Nothing downstream could
        // catch that, so the literal has to be complete.
        let missing: Vec<&str> = fields
            .iter()
            .map(|(name, _, _)| name.as_str())
            .filter(|name| {
                !field_inits
                    .iter()
                    .any(|given| self.ast.name(given.name) == *name)
            })
            .collect();
        if !missing.is_empty() {
            bail!(
                "struct '{struct_name}' is missing {} {}; a field left out would be read uninitialized",
                if missing.len() == 1 {
                    "field"
                } else {
                    "fields"
                },
                missing
                    .iter()
                    .map(|name| format!("'{name}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        for given in &field_inits {
            let field_name = self.ast.name(given.name).to_string();
            let Some((_, offset, field_type)) =
                fields.iter().find(|(name, _, _)| *name == field_name)
            else {
                bail!("struct '{struct_name}' has no field '{field_name}'");
            };
            let address =
                self.fresh_local(Type::Ptr(Box::new(field_type.clone())), None);
            self.emit(IrStatement::Assign(
                address,
                IrRvalue::AddressOf {
                    local,
                    offset: *offset,
                },
            ));
            if needs_memory(field_type) {
                let source =
                    self.aggregate_field_source(given.value, field_type)?;
                self.emit(IrStatement::Copy {
                    destination: IrOperand::Local(address),
                    source,
                    size: self.builder.byte_size(field_type),
                });
            } else {
                let (operand, value_type) =
                    self.lower_expression(given.value, Some(field_type))?;
                let coerced = self.coerce(operand, &value_type, field_type)?;
                self.emit(IrStatement::Store {
                    address: IrOperand::Local(address),
                    value: coerced,
                });
            }
        }
        Ok(())
    }

    // The address to copy an aggregate field's value out of. An expression that
    // already names a place is copied straight from it, which matters for a
    // borrowed parameter: the local there holds the caller's address, so taking
    // the address of the local again would copy the pointer rather than what it
    // points at.
    fn aggregate_field_source(
        &mut self,
        expression: ExprId,
        field_type: &Type,
    ) -> Result<IrOperand> {
        match self.ast.expr(expression) {
            Expression::StructInit(..)
            | Expression::EnumVariantInit(..)
            | Expression::Literal(Literal::Array(_)) => {
                let temp = self.fresh_local(field_type.clone(), None);
                self.materialize_aggregate(temp, expression)?;
                Ok(self.address_of_local(temp, field_type))
            }
            _ => {
                // A borrowed parameter's local holds the caller's address
                // already, so that value is the source. Taking the address of
                // the local would copy the pointer instead of the aggregate.
                if let Some(
                    Type::Ref(inner) | Type::RefMut(inner) | Type::Ptr(inner),
                ) = self.probe_type(expression)
                    && inner.as_ref() == field_type
                {
                    let (operand, _) =
                        self.lower_expression(expression, None)?;
                    return Ok(operand);
                }
                // A value written into a field is moved there, so a linear one
                // is consumed by the literal that holds it.
                self.aggregate_argument_address(expression, field_type, true)
            }
        }
    }

    fn init_enum(
        &mut self,
        local: LocalId,
        enum_name: &str,
        variant_name: &str,
        field_inits: Range32,
    ) -> Result<()> {
        self.mark_owned(local);
        let field_inits: Vec<NamedExpr> =
            self.ast.named_in(field_inits).to_vec();
        let (tag, fields): (u32, Vec<(String, usize, Type)>) = {
            let layout = self
                .builder
                .enum_layout(enum_name)
                .ok_or_else(|| anyhow::anyhow!("unknown enum '{enum_name}'"))?;
            let variant = layout.variant(variant_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "enum '{enum_name}' has no variant '{variant_name}'"
                )
            })?;
            (
                variant.tag,
                variant
                    .fields
                    .iter()
                    .map(|field| {
                        (field.name.clone(), field.offset, field.ty.clone())
                    })
                    .collect(),
            )
        };

        let tag_address =
            self.fresh_local(Type::Ptr(Box::new(Type::I32)), None);
        self.emit(IrStatement::Assign(
            tag_address,
            IrRvalue::AddressOf { local, offset: 0 },
        ));
        self.emit(IrStatement::Store {
            address: IrOperand::Local(tag_address),
            value: IrOperand::Constant(IrConstant::Integer(
                tag as i64,
                Type::I32,
            )),
        });

        // The same hole a struct literal has. A payload field left out is
        // storage a `match` will happily bind and read.
        let missing: Vec<&str> = fields
            .iter()
            .map(|(name, _, _)| name.as_str())
            .filter(|name| {
                !field_inits
                    .iter()
                    .any(|given| self.ast.name(given.name) == *name)
            })
            .collect();
        if !missing.is_empty() {
            bail!(
                "variant '{variant_name}' is missing {} {}; a field left out would be read uninitialized",
                if missing.len() == 1 {
                    "field"
                } else {
                    "fields"
                },
                missing
                    .iter()
                    .map(|name| format!("'{name}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        for given in &field_inits {
            let field_name = self.ast.name(given.name).to_string();
            let Some((_, offset, field_type)) =
                fields.iter().find(|(name, _, _)| *name == field_name)
            else {
                bail!(
                    "enum variant '{variant_name}' has no field '{field_name}'"
                );
            };
            let address =
                self.fresh_local(Type::Ptr(Box::new(field_type.clone())), None);
            self.emit(IrStatement::Assign(
                address,
                IrRvalue::AddressOf {
                    local,
                    offset: *offset,
                },
            ));
            if needs_memory(field_type) {
                let source =
                    self.aggregate_field_source(given.value, field_type)?;
                self.emit(IrStatement::Copy {
                    destination: IrOperand::Local(address),
                    source,
                    size: self.builder.byte_size(field_type),
                });
            } else {
                let (operand, value_type) =
                    self.lower_expression(given.value, Some(field_type))?;
                let coerced = self.coerce(operand, &value_type, field_type)?;
                self.emit(IrStatement::Store {
                    address: IrOperand::Local(address),
                    value: coerced,
                });
            }
        }
        Ok(())
    }

    /// Every variant of the matched enum has to be covered, or a `case _`
    /// has to say what the rest do.
    ///
    /// Without this a missing variant falls through to whatever the match was
    /// going to answer with anyway, so adding a variant to an enum silently
    /// changes the meaning of every match on it rather than pointing at the
    /// places that now have a case to write.
    fn check_exhaustive(
        &self,
        enum_name: &str,
        cases: &[SwitchCase],
    ) -> Result<()> {
        let catches_rest = cases.iter().any(|case| {
            matches!(self.ast.pattern(case.pattern), Pattern::Wildcard)
        });
        if catches_rest {
            return Ok(());
        }
        let Some(layout) = self.builder.enum_layout(enum_name) else {
            return Ok(());
        };
        let mut covered: HashSet<String> = HashSet::new();
        for case in cases {
            self.gather_variants(case.pattern, &mut covered);
        }
        let missing = layout
            .variants
            .iter()
            .map(|variant| variant.name.as_str())
            .filter(|name| !covered.contains(*name))
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return Ok(());
        }
        let named = missing
            .iter()
            .map(|name| format!("'.{name}'"))
            .collect::<Vec<_>>()
            .join(", ");
        let readable =
            crate::modules::imports::demangle_private_names(enum_name);
        bail!(
            "match on '{readable}' does not cover {named}; add the case or a `case _` for the rest"
        )
    }

    /// Every variant one arm names, looking through the alternatives of an
    /// or-pattern, since what such an arm covers is their union.
    fn gather_variants(&self, pattern: PatternId, into: &mut HashSet<String>) {
        match self.ast.pattern(pattern) {
            Pattern::EnumVariant { variant_name, .. } => {
                into.insert(self.ast.name(*variant_name).to_string());
            }
            Pattern::Or(alternatives) => {
                for held in self.ast.patterns_in(*alternatives).to_vec() {
                    self.gather_variants(held, into);
                }
            }
            _ => {}
        }
    }

    /// The spans of whole numbers one arm covers. A literal is a span of one,
    /// and an alternative list contributes each of its own.
    fn gather_spans(&self, pattern: PatternId, into: &mut Vec<(i64, i64)>) {
        match self.ast.pattern(pattern) {
            Pattern::Literal(Literal::Integer(value)) => {
                into.push((*value, *value));
            }
            Pattern::Range {
                low,
                high,
                inclusive,
            } => {
                let last = if *inclusive { *high } else { *high - 1 };
                into.push((*low, last));
            }
            Pattern::Or(alternatives) => {
                for held in self.ast.patterns_in(*alternatives).to_vec() {
                    self.gather_spans(held, into);
                }
            }
            _ => {}
        }
    }

    /// An arm every value of which the arms above it already take.
    ///
    /// What an arm covers is the union of what its alternatives name, and what
    /// it is read against is the union of every arm above it. That is the
    /// question a reader asks looking down the arms, so it is the one the
    /// compiler answers: `case 1..5:`, `case 5..10:`, `case 3..7:` refuses the
    /// third, because between them the first two take every value it has.
    ///
    /// `_` covers everything, so an arm below one is refused by this rule
    /// rather than by a rule of its own.
    fn check_reachable(&self, cases: &[SwitchCase]) -> Result<()> {
        let mut variants: HashSet<String> = HashSet::new();
        let mut spans: Vec<(i64, i64)> = Vec::new();
        let mut everything = false;
        for case in cases {
            if everything {
                bail!(UNREACHABLE_CASE);
            }
            if matches!(self.ast.pattern(case.pattern), Pattern::Wildcard) {
                everything = true;
                continue;
            }

            let mut mine = HashSet::new();
            self.gather_variants(case.pattern, &mut mine);
            if !mine.is_empty()
                && mine.iter().all(|name| variants.contains(name))
            {
                bail!(UNREACHABLE_CASE);
            }
            variants.extend(mine);

            let mut ours = Vec::new();
            self.gather_spans(case.pattern, &mut ours);
            if !ours.is_empty()
                && ours.iter().all(|(low, high)| covers(&spans, *low, *high))
            {
                bail!(UNREACHABLE_CASE);
            }
            spans.extend(ours);
        }
        Ok(())
    }

    /// The comparisons one arm's pattern stands for, written out as the chain a
    /// reader would write by hand: control reaches `success` where the pattern
    /// covers the value and `failure` where it does not.
    fn emit_case_test(
        &mut self,
        pattern: PatternId,
        subject: &Scrutinee<'_>,
        success: BlockId,
        failure: BlockId,
    ) -> Result<()> {
        let Scrutinee {
            tag: tag_operand,
            enum_name,
            scalar,
        } = *subject;
        match self.ast.pattern(pattern).clone() {
            Pattern::Wildcard => {
                self.set_terminator(IrTerminator::Jump(success));
            }
            Pattern::Literal(literal) => {
                let Some((value, value_type)) = scalar else {
                    bail!("literal pattern requires a scalar match value");
                };
                let value = value.clone();
                let value_type = value_type.clone();
                let (literal_operand, _) =
                    self.lower_literal(&literal, Some(&value_type))?;
                let condition = self.fresh_local(Type::Bool, None);
                self.emit(IrStatement::Assign(
                    condition,
                    IrRvalue::Binary(IrBinOp::Equal, value, literal_operand),
                ));
                self.set_terminator(IrTerminator::Branch {
                    condition: IrOperand::Local(condition),
                    then_block: success,
                    else_block: failure,
                });
            }
            Pattern::Range {
                low,
                high,
                inclusive,
            } => {
                let Some((value, value_type)) = scalar else {
                    bail!("a case range needs a whole number to compare with");
                };
                let value = value.clone();
                let value_type = value_type.clone();
                let (from, _) = self
                    .lower_literal(&Literal::Integer(low), Some(&value_type))?;
                let above = self.fresh_local(Type::Bool, None);
                self.emit(IrStatement::Assign(
                    above,
                    IrRvalue::Binary(
                        IrBinOp::GreaterThanOrEqual,
                        value.clone(),
                        from,
                    ),
                ));
                let upper = self.new_block();
                self.set_terminator(IrTerminator::Branch {
                    condition: IrOperand::Local(above),
                    then_block: upper,
                    else_block: failure,
                });
                self.switch_to(upper);
                let (to, _) = self.lower_literal(
                    &Literal::Integer(high),
                    Some(&value_type),
                )?;
                let ordering = if inclusive {
                    IrBinOp::LessThanOrEqual
                } else {
                    IrBinOp::LessThan
                };
                let below = self.fresh_local(Type::Bool, None);
                self.emit(IrStatement::Assign(
                    below,
                    IrRvalue::Binary(ordering, value, to),
                ));
                self.set_terminator(IrTerminator::Branch {
                    condition: IrOperand::Local(below),
                    then_block: success,
                    else_block: failure,
                });
            }
            Pattern::EnumVariant {
                enum_name: written,
                variant_name,
                ..
            } => {
                let variant_name = self.ast.name(variant_name).to_string();
                let Some(tag) = tag_operand else {
                    bail!("enum variant pattern requires an enum match value");
                };
                let enum_name = enum_name.unwrap();
                // An arm is a value named under a type the way every other
                // mention of one is, and the type is part of the spelling. The
                // enum the subject settled on is named here, which is the edit.
                //
                // A failure set's enum is named by the compiler and a program
                // may not write that name, so `.Ok` and `.Err` are the only
                // spelling there is and nothing is being left out of them.
                if written.is_none() && !self.ast.is_failure_result(enum_name) {
                    let readable =
                        crate::modules::imports::demangle_private_names(
                            enum_name,
                        );
                    bail!(
                        "a value named under a type is written with the type in front of it, so this one is written `{readable}::{variant_name}`"
                    );
                }
                let variant_tag = self
                    .builder
                    .enum_layout(enum_name)
                    .and_then(|layout| layout.variant(&variant_name))
                    .map(|variant| variant.tag)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "enum '{enum_name}' has no variant '{variant_name}'"
                        )
                    })?;
                let tag = tag.clone();
                let condition = self.fresh_local(Type::Bool, None);
                self.emit(IrStatement::Assign(
                    condition,
                    IrRvalue::Binary(
                        IrBinOp::Equal,
                        tag,
                        IrOperand::Constant(IrConstant::Integer(
                            variant_tag as i64,
                            Type::I32,
                        )),
                    ),
                ));
                self.set_terminator(IrTerminator::Branch {
                    condition: IrOperand::Local(condition),
                    then_block: success,
                    else_block: failure,
                });
            }
            // Each alternative is tried in turn, and the one that covers the
            // value reaches the body. What follows a failed test is the next
            // alternative, so the last one's failure is the arm's.
            Pattern::Or(alternatives) => {
                let alternatives = self.ast.patterns_in(alternatives).to_vec();
                let last = alternatives.len() - 1;
                for (index, held) in alternatives.iter().enumerate() {
                    let following = if index == last {
                        failure
                    } else {
                        self.new_block()
                    };
                    self.emit_case_test(*held, subject, success, following)?;
                    if index != last {
                        self.switch_to(following);
                    }
                }
            }
            // A tuple pattern matches a tuple, which `lower_tuple_match` has
            // already taken. Reaching here means the pattern has parts and the
            // value being matched does not, so this is a mismatch to report
            // rather than a feature to miss.
            Pattern::Tuple(patterns) => {
                let parts = patterns.len();
                let described = match (enum_name, scalar) {
                    (Some(name), _) => {
                        crate::modules::imports::demangle_private_names(name)
                    }
                    (None, Some((_, ty))) => ty.to_string(),
                    (None, None) => "the matched value".to_string(),
                };
                bail!(
                    "a `case` of {parts} parts matches a tuple, and this match is on '{described}', which has none; match on `(a, b)` to compare several values at once"
                );
            }
        }
        Ok(())
    }

    fn lower_match(
        &mut self,
        scrutinee: ExprId,
        cases: Range32,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        let cases: Vec<SwitchCase> = self.ast.cases_in(cases).to_vec();
        if cases.is_empty() {
            bail!("match with no cases");
        }

        if let Expression::Tuple(elements) = self.ast.expr(scrutinee) {
            let elements: Vec<ExprId> = self.ast.exprs_in(*elements).to_vec();
            return self.lower_tuple_match(&elements, &cases, expected);
        }

        let enum_info = self.enum_scrutinee_address(scrutinee)?;

        let (enum_address, enum_name, tag_operand, scalar) =
            if let Some((name, address)) = enum_info {
                let tag = self.fresh_local(Type::I32, None);
                self.emit(IrStatement::Assign(
                    tag,
                    IrRvalue::Load {
                        address: address.clone(),
                        ty: Type::I32,
                    },
                ));
                (Some(address), Some(name), Some(IrOperand::Local(tag)), None)
            } else {
                let (value, value_type) =
                    self.lower_expression(scrutinee, None)?;
                if let Some(name) = self.enum_name_of(&value_type) {
                    let IrOperand::Local(local) = value else {
                        bail!("enum match value is not a place");
                    };
                    self.mark_in_memory(local);
                    let address = self.address_of_local(local, &value_type);
                    let tag = self.fresh_local(Type::I32, None);
                    self.emit(IrStatement::Assign(
                        tag,
                        IrRvalue::Load {
                            address: address.clone(),
                            ty: Type::I32,
                        },
                    ));
                    (
                        Some(address),
                        Some(name),
                        Some(IrOperand::Local(tag)),
                        None,
                    )
                } else {
                    (None, None, None, Some((value, value_type)))
                }
            };

        if let Some(name) = &enum_name {
            self.check_exhaustive(name, &cases)?;
        }
        self.check_reachable(&cases)?;

        // Taking a linear value apart is consuming it: every arm names what it
        // held, and what the arm does with those is the arm's obligation. This
        // is what lets a fallible function hand back a resource, since the
        // result carrying one is linear too.
        let consumed = self.linear_scrutinee(scrutinee, &scalar);

        let merge = self.new_block();
        let mut result_local: Option<LocalId> = None;
        let mut result_type = Type::Void;

        for case in &cases {
            let case_block = self.new_block();
            let next_block = self.new_block();

            self.emit_case_test(
                case.pattern,
                &Scrutinee {
                    tag: tag_operand.as_ref(),
                    enum_name: enum_name.as_ref(),
                    scalar: scalar.as_ref(),
                },
                case_block,
                next_block,
            )?;

            self.switch_to(case_block);
            if let Some(local) = consumed {
                self.emit(IrStatement::Consume(local));
            }
            self.push_scope();
            self.bind_pattern(
                case.pattern,
                enum_address.as_ref(),
                enum_name.as_deref(),
            )?;
            let (value, value_type) = self.lower_block(case.body, expected)?;
            // An arm that returns, breaks, or continues yields no value and has
            // already set its terminator, so it contributes nothing to merge.
            if self.current_is_terminated() {
                self.pop_scope();
                self.switch_to(next_block);
                continue;
            }
            if result_local.is_none() {
                result_type = match expected {
                    Some(ty) if !matches!(ty, Type::Void) => ty.clone(),
                    _ => value_type.clone(),
                };
                result_local =
                    Some(self.fresh_local(result_type.clone(), None));
            }
            let target = result_local.unwrap();
            let coerced = self.coerce(value, &value_type, &result_type)?;
            self.emit(IrStatement::Assign(target, IrRvalue::Use(coerced)));
            self.pop_scope();
            self.set_terminator(IrTerminator::Jump(merge));

            self.switch_to(next_block);
        }

        // The block reached when no arm matched. A match on an enum covers
        // every variant so nothing arrives here, but it is a path the linear
        // check walks, and the value is taken apart on it too.
        if let Some(local) = consumed {
            self.emit(IrStatement::Consume(local));
        }
        let target = result_local
            .unwrap_or_else(|| self.fresh_local(result_type.clone(), None));
        if !needs_memory(&result_type) {
            let zero = zero_operand(&result_type);
            self.emit(IrStatement::Assign(target, IrRvalue::Use(zero)));
        }
        self.set_terminator(IrTerminator::Jump(merge));

        self.switch_to(merge);
        Ok((IrOperand::Local(target), result_type))
    }

    fn enum_name_of(&self, ty: &Type) -> Option<String> {
        match ty {
            Type::Enum(name) => Some(name.clone()),
            Type::Struct(name) if self.builder.enum_layout(name).is_some() => {
                Some(name.clone())
            }
            _ => None,
        }
    }

    fn enum_scrutinee_address(
        &mut self,
        scrutinee: ExprId,
    ) -> Result<Option<(String, IrOperand)>> {
        if matches!(
            self.ast.expr(scrutinee),
            Expression::FieldAccess(..)
                | Expression::Index(..)
                | Expression::Dereference(..)
        ) {
            let (address, ty) = self.place_address(scrutinee)?;
            if let Some(enum_name) = self.enum_name_of(&ty) {
                return Ok(Some((enum_name, address)));
            }
            return Ok(None);
        }
        let Expression::Identifier(name) = self.ast.expr(scrutinee) else {
            return Ok(None);
        };
        let name = self.ast.name(*name).to_string();
        let Some(local) = self.resolve_variable(&name) else {
            return Ok(None);
        };
        let ty = self.type_of_local(local);

        if let Some(enum_name) = self.enum_name_of(&ty) {
            self.mark_in_memory(local);
            let address = self.fresh_local(
                Type::Ptr(Box::new(Type::Enum(enum_name.clone()))),
                None,
            );
            self.emit(IrStatement::Assign(
                address,
                IrRvalue::AddressOf { local, offset: 0 },
            ));
            return Ok(Some((enum_name, IrOperand::Local(address))));
        }

        if let Type::Ref(inner) | Type::RefMut(inner) | Type::Ptr(inner) = &ty
            && let Some(enum_name) = self.enum_name_of(inner)
        {
            return Ok(Some((enum_name, IrOperand::Local(local))));
        }

        Ok(None)
    }

    fn lower_tuple_match(
        &mut self,
        elements: &[ExprId],
        cases: &[SwitchCase],
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        let mut values = Vec::with_capacity(elements.len());
        for element in elements {
            values.push(self.lower_expression(*element, None)?);
        }

        // A tuple arm naming `_` in every part covers everything, the same as a
        // bare `case _`, so the arms below one are the same unreachable arms
        // `check_reachable` refuses in a match on one value.
        let mut everything = false;
        for case in cases {
            if everything {
                bail!(UNREACHABLE_CASE);
            }
            let parts: Vec<PatternId> = match self.ast.pattern(case.pattern) {
                Pattern::Tuple(patterns) => {
                    self.ast.patterns_in(*patterns).to_vec()
                }
                _ => Vec::new(),
            };
            everything = parts.iter().all(|held| {
                matches!(self.ast.pattern(*held), Pattern::Wildcard)
            });
        }

        let merge = self.new_block();
        let mut result_local: Option<LocalId> = None;
        let mut result_type = Type::Void;

        for case in cases {
            let case_block = self.new_block();
            let next_block = self.new_block();

            let patterns: Vec<PatternId> = match self.ast.pattern(case.pattern)
            {
                Pattern::Tuple(patterns) => {
                    self.ast.patterns_in(*patterns).to_vec()
                }
                Pattern::Wildcard => Vec::new(),
                _ => {
                    let written = crate::ast_display::display_pattern(
                        self.ast,
                        case.pattern,
                    );
                    bail!("unsupported tuple match pattern: {written}")
                }
            };

            let mut condition: Option<LocalId> = None;
            for (pattern, (value, value_type)) in
                patterns.iter().zip(values.iter())
            {
                if let Pattern::Literal(literal) =
                    self.ast.pattern(*pattern).clone()
                {
                    let value = value.clone();
                    let value_type = value_type.clone();
                    let (literal_operand, _) =
                        self.lower_literal(&literal, Some(&value_type))?;
                    let test = self.fresh_local(Type::Bool, None);
                    self.emit(IrStatement::Assign(
                        test,
                        IrRvalue::Binary(
                            IrBinOp::Equal,
                            value,
                            literal_operand,
                        ),
                    ));
                    condition = Some(match condition {
                        None => test,
                        Some(previous) => {
                            let combined = self.fresh_local(Type::Bool, None);
                            self.emit(IrStatement::Assign(
                                combined,
                                IrRvalue::Binary(
                                    IrBinOp::BitwiseAnd,
                                    IrOperand::Local(previous),
                                    IrOperand::Local(test),
                                ),
                            ));
                            combined
                        }
                    });
                }
            }

            match condition {
                Some(local) => self.set_terminator(IrTerminator::Branch {
                    condition: IrOperand::Local(local),
                    then_block: case_block,
                    else_block: next_block,
                }),
                None => self.set_terminator(IrTerminator::Jump(case_block)),
            }

            self.switch_to(case_block);
            self.push_scope();
            let (value, value_type) = self.lower_block(case.body, expected)?;
            // An arm that returns, breaks, or continues yields no value and has
            // already set its terminator, so it contributes nothing to merge.
            if self.current_is_terminated() {
                self.pop_scope();
                self.switch_to(next_block);
                continue;
            }
            if result_local.is_none() {
                result_type = match expected {
                    Some(ty) if !matches!(ty, Type::Void) => ty.clone(),
                    _ => value_type.clone(),
                };
                result_local =
                    Some(self.fresh_local(result_type.clone(), None));
            }
            let target = result_local.unwrap();
            let coerced = self.coerce(value, &value_type, &result_type)?;
            self.emit(IrStatement::Assign(target, IrRvalue::Use(coerced)));
            self.pop_scope();
            self.set_terminator(IrTerminator::Jump(merge));

            self.switch_to(next_block);
        }

        let target = result_local
            .unwrap_or_else(|| self.fresh_local(result_type.clone(), None));
        if !needs_memory(&result_type) {
            let zero = zero_operand(&result_type);
            self.emit(IrStatement::Assign(target, IrRvalue::Use(zero)));
        }
        self.set_terminator(IrTerminator::Jump(merge));

        self.switch_to(merge);
        Ok((IrOperand::Local(target), result_type))
    }

    // The linear local a match takes apart, if it takes one apart. A named
    // value is found by its name, and a value the match itself produced (the
    // answer of a call) is the local it landed in.
    fn linear_scrutinee(
        &self,
        scrutinee: ExprId,
        scalar: &Option<(IrOperand, Type)>,
    ) -> Option<LocalId> {
        if let Expression::Identifier(name) = self.ast.expr(scrutinee)
            && let Some(local) = self.resolve_variable(self.ast.name(*name))
            && self.locals[local].linear
        {
            return Some(local);
        }
        if let Some((IrOperand::Local(local), _)) = scalar
            && self.locals[*local].linear
        {
            return Some(*local);
        }
        None
    }

    fn bind_pattern(
        &mut self,
        pattern: PatternId,
        enum_address: Option<&IrOperand>,
        enum_name: Option<&str>,
    ) -> Result<()> {
        match self.ast.pattern(pattern).clone() {
            Pattern::EnumVariant {
                variant_name,
                bindings,
                ..
            } => {
                let variant_name = self.ast.name(variant_name).to_string();
                let (Some(address), Some(enum_name)) =
                    (enum_address, enum_name)
                else {
                    bail!("enum pattern on a non-enum match value");
                };
                let fields: Vec<(String, usize, Type)> = self
                    .builder
                    .enum_layout(enum_name)
                    .and_then(|layout| layout.variant(&variant_name))
                    .map(|variant| {
                        variant
                            .fields
                            .iter()
                            .map(|field| {
                                (
                                    field.name.clone(),
                                    field.offset,
                                    field.ty.clone(),
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let bindings: Vec<crate::ast::PatternBinding> =
                    self.ast.pattern_bindings_in(bindings).to_vec();
                for binding in &bindings {
                    let field_name = self.ast.name(binding.field).to_string();
                    let bound_name = self.ast.name(binding.binding).to_string();
                    let Some((_, offset, field_type)) =
                        fields.iter().find(|(name, _, _)| *name == field_name)
                    else {
                        bail!(
                            "variant '{variant_name}' has no field '{field_name}'"
                        );
                    };
                    let field_address = self.fresh_local(
                        Type::Ptr(Box::new(field_type.clone())),
                        None,
                    );
                    self.emit(IrStatement::Assign(
                        field_address,
                        IrRvalue::FieldAddress {
                            base: address.clone(),
                            offset: *offset,
                        },
                    ));
                    let bound = self.fresh_local(
                        field_type.clone(),
                        Some(bound_name.clone()),
                    );
                    if needs_memory(field_type) {
                        let destination =
                            self.address_of_local(bound, field_type);
                        self.emit(IrStatement::Copy {
                            destination,
                            source: IrOperand::Local(field_address),
                            size: self.builder.byte_size(field_type),
                        });
                    } else {
                        self.emit(IrStatement::Assign(
                            bound,
                            IrRvalue::Load {
                                address: IrOperand::Local(field_address),
                                ty: field_type.clone(),
                            },
                        ));
                    }
                    self.define_variable(&bound_name, bound);
                    // A binding takes the field out of the value being
                    // matched, so it holds whatever that field held. Without
                    // this a linear field could not be consumed by the arm
                    // that named it, which is the only way to consume one.
                    self.mark_owned(bound);
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn type_of_local(&self, local: LocalId) -> Type {
        self.locals[local].ty.clone()
    }

    fn shallow_value_type(&self, expression: ExprId) -> Option<Type> {
        match self.ast.expr(expression) {
            Expression::Literal(Literal::Integer(_)) => Some(Type::I64),
            Expression::Literal(Literal::Float(_)) => Some(Type::F64),
            Expression::Literal(Literal::Float32(_)) => Some(Type::F32),
            Expression::Boolean(_)
            | Expression::Literal(Literal::Boolean(_)) => Some(Type::Bool),
            Expression::Identifier(name) => self
                .resolve_variable(self.ast.name(*name))
                .map(|local| self.type_of_local(local)),
            Expression::Borrow(inner) => self
                .shallow_value_type(*inner)
                .map(|ty| Type::Ref(Box::new(ty))),
            Expression::BorrowMut(inner) => self
                .shallow_value_type(*inner)
                .map(|ty| Type::RefMut(Box::new(ty))),
            Expression::StructInit(name, fields) => {
                let name = self.ast.name(*name);
                if self.builder.generic_struct_defs.contains_key(name) {
                    self.generic_instance_of(name, *fields).map(Type::Struct)
                } else {
                    Some(Type::Struct(name.to_string()))
                }
            }
            _ => None,
        }
    }

    /// The instance a bare `Name { ... }` literal makes, worked out by matching
    /// each field's written value against the field's declared type. A generic
    /// call takes its `$T` in writing; a literal takes it from what it is given.
    fn generic_instance_of(
        &self,
        struct_name: &str,
        field_inits: Range32,
    ) -> Option<String> {
        let (type_params, fields) =
            self.builder.generic_struct_defs.get(struct_name)?;
        let mut subst: HashMap<String, Type> = HashMap::new();
        for entry in self.ast.named_in(field_inits) {
            if let Some((_, field_type)) = fields
                .iter()
                .find(|(field_name, _)| field_name == self.ast.name(entry.name))
                && let Some(value_type) = self.shallow_value_type(entry.value)
            {
                infer_subst_into(
                    field_type,
                    &value_type,
                    type_params,
                    &mut subst,
                );
            }
        }
        // Every parameter, or none of them. Rendering an unbound one as its own
        // name made `Pair<T>`, a type nothing declares, and the reader was told
        // it was unknown rather than that the literal had not said which
        // instance it is.
        let rendered: Vec<String> = type_params
            .iter()
            .map(|type_param| Some(subst.get(type_param)?.to_string()))
            .collect::<Option<Vec<String>>>()?;
        Some(format!("{struct_name}<{}>", rendered.join(", ")))
    }

    /// Puts a value into the type the place it is going has.
    ///
    /// Fallible because of one case: a literal that does not fit. The type it
    /// is going into is in hand right here, which is the whole of what a range
    /// check needs, and nothing used to look at it, so `a : u8 = 300` was
    /// quietly 44 and `b : i8 = 200` was quietly -56. Both compilers agreed
    /// about it, which is exactly why the differential oracle could not see it.
    fn coerce(
        &mut self,
        operand: IrOperand,
        from: &Type,
        to: &Type,
    ) -> Result<IrOperand> {
        if from == to || matches!(to, Type::Void | Type::Unknown) {
            return Ok(operand);
        }
        // Reading a borrow reads what it borrows. A `ref T` stays one where a
        // `ref T` is wanted, which is a parameter that takes a borrow and an
        // answer declared as one; everywhere else it is the value, and a scalar
        // has no field access to read through it the way an aggregate does.
        if let Some(inner) = borrowed_value(from)
            && borrowed_value(to).is_none()
            && !needs_memory(inner)
        {
            let held = inner.clone();
            let read = self.fresh_local(held.clone(), None);
            self.emit(IrStatement::Assign(
                read,
                IrRvalue::Load {
                    address: operand,
                    ty: held.clone(),
                },
            ));
            return self.coerce(IrOperand::Local(read), &held, to);
        }
        if let (Type::Array(from_element, count), Some(to_element)) =
            (from, &slice_element_wanted(to))
            && from_element.as_ref() == to_element
            && let IrOperand::Local(array_local) = operand
        {
            self.mark_in_memory(array_local);
            let array_type = Type::Array(from_element.clone(), *count);
            let base = self.address_of_local(array_local, &array_type);
            return Ok(self.build_slice_from_address(
                base,
                from_element,
                *count,
            ));
        }
        Ok(match &operand {
            IrOperand::Constant(IrConstant::Integer(value, _))
                if to.is_integer() =>
            {
                if !fits_in(*value, to) {
                    let (low, high) = range_of(to).expect("integer type");
                    bail!(
                        "{value} does not fit in a {to}, which holds {low} to {high}"
                    );
                }
                IrOperand::Constant(IrConstant::Integer(*value, to.clone()))
            }
            IrOperand::Constant(IrConstant::Float(value, _))
                if matches!(to, Type::F32 | Type::F64) =>
            {
                IrOperand::Constant(IrConstant::Float(*value, to.clone()))
            }
            _ if needs_cast(from, to) => {
                if is_narrowing(from, to) {
                    bail!(
                        "this is a {from} and a {to} is wanted, which cannot hold all of one; write cast(${to}, ...) to say the loss is meant"
                    );
                }
                let result = self.fresh_local(to.clone(), None);
                self.emit(IrStatement::Assign(
                    result,
                    IrRvalue::Cast(operand, to.clone()),
                ));
                IrOperand::Local(result)
            }
            _ => operand,
        })
    }
}

/// What an integer type holds, or `None` when it is not one.
///
/// `usize` and `isize` are the word this compiler targets, which is sixty-four
/// bits everywhere it runs.
fn range_of(ty: &Type) -> Option<(i128, i128)> {
    let held = match ty {
        Type::Distinct(_, inner) => return range_of(inner),
        other => other,
    };
    match held {
        Type::I8 => Some((i8::MIN as i128, i8::MAX as i128)),
        Type::I16 => Some((i16::MIN as i128, i16::MAX as i128)),
        Type::I32 => Some((i32::MIN as i128, i32::MAX as i128)),
        Type::I64 | Type::Isize => Some((i64::MIN as i128, i64::MAX as i128)),
        Type::U8 => Some((0, u8::MAX as i128)),
        Type::U16 => Some((0, u16::MAX as i128)),
        Type::U32 => Some((0, u32::MAX as i128)),
        // A literal is read as an i64, so the largest one that can be written
        // is i64::MAX and the top half of u64 is not reachable from a literal
        // at all. A negative one is: it is the same sixty-four bits, and with
        // no hex literals it is the only way to write a sentinel of all ones.
        // Nothing is lost, which is what this check is about, so it is allowed
        // here and refused for the narrower unsigned types, where bits do go.
        Type::U64 | Type::Usize => Some((i64::MIN as i128, i64::MAX as i128)),
        _ => None,
    }
}

fn fits_in(value: i64, ty: &Type) -> bool {
    match range_of(ty) {
        Some((low, high)) => {
            let value = value as i128;
            value >= low && value <= high
        }
        None => true,
    }
}

#[derive(Clone, Copy)]
enum RefKind {
    Ref,
    RefMut,
    Ptr,
}

fn deref_target(pointer_type: &Type) -> Result<Type> {
    match pointer_type {
        Type::Ref(inner) | Type::RefMut(inner) | Type::Ptr(inner) => {
            Ok((**inner).clone())
        }
        other => {
            bail!("cannot dereference a value of type {other}")
        }
    }
}

fn unit_operand() -> IrOperand {
    IrOperand::Constant(IrConstant::Unit)
}

fn zero_operand(ty: &Type) -> IrOperand {
    match ty {
        Type::F32 | Type::F64 => {
            IrOperand::Constant(IrConstant::Float(0.0, ty.clone()))
        }
        Type::Bool => IrOperand::Constant(IrConstant::Bool(false)),
        _ if ty.is_integer() => {
            IrOperand::Constant(IrConstant::Integer(0, ty.clone()))
        }
        _ => IrOperand::Constant(IrConstant::Integer(0, Type::I64)),
    }
}

// A `bool` is one byte, so widening one to the width something takes is the
// same instruction as widening any narrow integer. It is not `is_integer`,
// because arithmetic on a bool is not a thing the language has, but one handed
// to the `%lld` writer every integer reaches has to arrive at its width. C
// widened it for free, so only the native backend saw this, as a verifier
// error rather than a wrong answer.
fn is_castable_integer(ty: &Type) -> bool {
    ty.is_integer() || matches!(ty, Type::Bool)
}

// `cast($Adapter, p)` where `Adapter` is a distinct type over `^u8` and `p` is
// one: the value takes the name its declaration gives that representation.
//
// A distinct type over a number is reached this way already, since both sides
// are numbers and the gate above lets them through. One over anything else had
// no spelling at all, so the only such value a program could hold was one an
// extern was declared to answer with, and a function writing the null of such a
// type leaned on the answer position not being checked.
fn names_a_distinct(from: &Type, target: &Type) -> bool {
    let Type::Distinct(_, repr) = target else {
        return false;
    };
    from == repr.as_ref()
}

fn is_numeric(ty: &Type) -> bool {
    match ty {
        Type::Distinct(_, inner) => is_numeric(inner),
        other => {
            other.is_integer()
                || other.is_float()
                || matches!(other, Type::Bool)
        }
    }
}

/// A conversion that can lose what it is given: a narrower integer, or a float
/// becoming one.
///
/// Widening is not one of these and stays implicit, because nothing is lost and
/// requiring a cast for it would be noise. What this refuses is the case where
/// the value may come out different from the one written: an i64 of 300 read
/// at a u8 is 44, and 3.9 read at an i64 is 3.
fn is_narrowing(from: &Type, to: &Type) -> bool {
    let from = match from {
        Type::Distinct(_, inner) => inner.as_ref(),
        other => other,
    };
    let to = match to {
        Type::Distinct(_, inner) => inner.as_ref(),
        other => other,
    };
    if from.is_float() && to.is_integer() {
        return true;
    }
    if is_castable_integer(from) && is_castable_integer(to) {
        return to.size_of() < from.size_of();
    }
    if from.is_float() && to.is_float() {
        return to.size_of() < from.size_of();
    }
    false
}

// A binary operation over two integers already known, at the full width a
// literal is read at. Division and remainder by zero, and a shift past the
// word, have no value to answer with and stay as they are written.
fn fold_integers(binop: IrBinOp, left: i64, right: i64) -> Option<i64> {
    Some(match binop {
        IrBinOp::Add => left.wrapping_add(right),
        IrBinOp::Subtract => left.wrapping_sub(right),
        IrBinOp::Multiply => left.wrapping_mul(right),
        IrBinOp::Divide if right != 0 => left.wrapping_div(right),
        IrBinOp::Modulo if right != 0 => left.wrapping_rem(right),
        IrBinOp::BitwiseAnd => left & right,
        IrBinOp::BitwiseOr => left | right,
        IrBinOp::ShiftLeft if (0..64).contains(&right) => {
            left.wrapping_shl(right as u32)
        }
        IrBinOp::ShiftRight if (0..64).contains(&right) => {
            left.wrapping_shr(right as u32)
        }
        _ => return None,
    })
}

fn needs_cast(from: &Type, to: &Type) -> bool {
    (is_castable_integer(from) && is_castable_integer(to))
        || (from.is_float() && to.is_float())
        || (is_castable_integer(from) && to.is_float())
        || (from.is_float() && to.is_integer())
}

// The type a binary operation is computed at: the narrower operand widens to
// the wider. A literal has no width of its own and has already taken the other
// side's by the time this runs, since the right operand is lowered with the
// left's type as its expectation.
/// How wide a vector may be, in bytes. A register's worth: `[16]f32` is one
/// AVX-512 register and `[4]f64` is half of one, and a length past this is a
/// loop the reader would not see written down.
const VECTOR_LIMIT: usize = 64;

// Where the lanes of one side of an elementwise operation come from.
enum LaneSource {
    /// A number, which stands in every lane.
    Broadcast(IrOperand),
    /// The address the vector's storage begins at.
    At(IrOperand),
}

// The element type and length of a vector, looking through a borrow the way
// every other question about a type does. Answers nothing for anything that is
// not a fixed array of numbers, which is what makes an ordinary operator go on
// meaning what it did.
fn lanes_of(ty: &Type) -> Option<(Type, usize)> {
    let held = match ty {
        Type::Ref(inner) | Type::RefMut(inner) | Type::Ptr(inner) => inner,
        held => held,
    };
    let Type::Array(element, count) = held else {
        return None;
    };
    if !element.is_integer() && !element.is_float() {
        return None;
    }
    Some(((**element).clone(), *count))
}

// What the other side of an operator is expected to be: a vector's element
// type, so a number beside one is that number in every lane, and otherwise the
// type itself.
fn lane_type(ty: &Type) -> Type {
    match lanes_of(ty) {
        Some((element, _)) => element,
        None => ty.clone(),
    }
}

// A type as one side of an elementwise operation reads: a vector by its length
// and element, anything else by its own name. Written out here rather than by
// each compiler's type renderer, since the two render a fixed array's length
// differently and a diagnostic is one sentence in one language.
fn describe_operand(ty: &Type) -> String {
    match ty {
        Type::Array(element, count) => {
            format!("a vector of {count} {element}")
        }
        held => format!("a {held}"),
    }
}

// Whether a type is a run of bytes, which is what `str_len` is asked about. A
// `str` is a `[]u8` (3.2), so both spellings answer yes and nothing else does.
fn is_byte_run(ty: Option<&Type>) -> bool {
    match ty {
        Some(Type::Str) => true,
        Some(Type::Slice(element)) => **element == Type::U8,
        _ => false,
    }
}

fn through_borrow(ty: &Type) -> &Type {
    match ty {
        Type::Ref(inner) | Type::RefMut(inner) | Type::Ptr(inner) => inner,
        held => held,
    }
}

fn unify(left: &Type, right: &Type) -> Type {
    // A borrow of a value put beside that value is the two of them, which is
    // what the operator is about, and two borrows of one type are two of them
    // as well. Left alone, `at(bag, 2) == 9` compared the address the borrow
    // holds against nine and answered no for every bag, and read before the
    // equality below, `at(bag, 1) < at(bag, 2)` compared two addresses and
    // answered about where the numbers sit rather than about the numbers.
    match (borrowed_value(left), borrowed_value(right)) {
        (Some(inner), _) => return unify(inner, right),
        (_, Some(inner)) => return unify(left, inner),
        _ => {}
    }
    if left == right {
        return left.clone();
    }
    match (left, right) {
        (wide, narrow) | (narrow, wide)
            if wide.is_integer()
                && narrow.is_integer()
                && wide.size_of() > narrow.size_of() =>
        {
            wide.clone()
        }
        (Type::F64, Type::F32) | (Type::F32, Type::F64) => Type::F64,
        (Type::Unknown, other) | (other, Type::Unknown) => other.clone(),
        _ => left.clone(),
    }
}

// What a borrow borrows, for the one type that is a borrow rather than holds
// one. A raw pointer is not this: it is a value the reader wrote `^` to read
// through, and reading one without the `^` is the address on purpose.
fn borrowed_value(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Ref(inner) | Type::RefMut(inner) => Some(inner),
        _ => None,
    }
}

// An operator as the source writes it, for a diagnostic that has the IR's name
// for it and needs the reader's.
fn operator_text(binop: IrBinOp) -> &'static str {
    match binop {
        IrBinOp::Add => "+",
        IrBinOp::Subtract => "-",
        IrBinOp::Multiply => "*",
        IrBinOp::WrappingAdd => "wrap_add",
        IrBinOp::WrappingSubtract => "wrap_sub",
        IrBinOp::WrappingMultiply => "wrap_mul",
        IrBinOp::Divide => "/",
        IrBinOp::Modulo => "%",
        IrBinOp::BitwiseAnd => "&",
        IrBinOp::BitwiseOr => "|",
        IrBinOp::ShiftLeft => "<<",
        IrBinOp::ShiftRight => ">>",
        IrBinOp::Equal => "==",
        IrBinOp::NotEqual => "!=",
        IrBinOp::LessThan => "<",
        IrBinOp::LessThanOrEqual => "<=",
        IrBinOp::GreaterThan => ">",
        IrBinOp::GreaterThanOrEqual => ">=",
    }
}

// Whether an expression is a number written down rather than a value with a
// type of its own. One of these takes the type of whatever it sits beside; a
// negated one counts, since `-0.6` is the same literal with a sign.
fn is_bare_number(ast: &Ast, expression: ExprId) -> bool {
    match ast.expr(expression) {
        Expression::Literal(Literal::Integer(_))
        | Expression::Literal(Literal::Float(_)) => true,
        Expression::Prefix(Operator::Negate, inner) => {
            is_bare_number(ast, *inner)
        }
        _ => false,
    }
}

fn binop_of(operator: Operator) -> Result<IrBinOp> {
    Ok(match operator {
        Operator::Add => IrBinOp::Add,
        Operator::Subtract => IrBinOp::Subtract,
        Operator::Multiply => IrBinOp::Multiply,
        Operator::Divide => IrBinOp::Divide,
        Operator::Modulo => IrBinOp::Modulo,
        Operator::BitwiseAnd => IrBinOp::BitwiseAnd,
        Operator::BitwiseOr => IrBinOp::BitwiseOr,
        Operator::ShiftLeft => IrBinOp::ShiftLeft,
        Operator::ShiftRight => IrBinOp::ShiftRight,
        Operator::Equal => IrBinOp::Equal,
        Operator::NotEqual => IrBinOp::NotEqual,
        Operator::LessThan => IrBinOp::LessThan,
        Operator::LessThanOrEqual => IrBinOp::LessThanOrEqual,
        Operator::GreaterThan => IrBinOp::GreaterThan,
        Operator::GreaterThanOrEqual => IrBinOp::GreaterThanOrEqual,
        other => bail!("unsupported binary operator: {other}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Lexer, Parser};

    fn lowered(source: &str) -> (IrModule, Vec<crate::diagnostic::Diagnostic>) {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let mut module = parser.parse().unwrap();
        let linear = parser.linear_types().clone();
        build_module_recovering(&mut module.ast, &module.roots, &linear, false)
            .unwrap()
    }

    // One failed function does not mask another, and a function that lowers
    // is in the module while a function that failed is not.
    #[test]
    fn every_failed_function_is_reported() {
        let source = "a :: fn() -> i64 { missing_one() }\n\
                      good :: fn() -> i64 { 7 }\n\
                      b :: fn() -> i64 { missing_two() }\n";
        let (module, diagnostics) = lowered(source);
        assert_eq!(diagnostics.len(), 2, "{diagnostics:?}");
        assert!(
            diagnostics[0].message.contains("missing_one"),
            "{diagnostics:?}"
        );
        assert!(
            diagnostics[1].message.contains("missing_two"),
            "{diagnostics:?}"
        );
        assert!(module.functions.iter().any(|held| held.name == "good"));
        assert!(!module.functions.iter().any(|held| held.name == "a"));
    }

    // The strict entry point refuses with everything recovery found, so the
    // command line reports every broken function in one build.
    #[test]
    fn the_strict_path_reports_every_failure_it_recovered() {
        let source = "a :: fn() -> i64 { missing_one() }\n\
                      b :: fn() -> i64 { missing_two() }\n";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let mut module = parser.parse().unwrap();
        let linear = parser.linear_types().clone();
        let error =
            build_module(&mut module.ast, &module.roots, &linear).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("missing_one"), "{text}");
        assert!(text.contains("missing_two"), "{text}");
    }
}
