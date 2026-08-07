// Which instantiations of a generic struct are resources.
//
// A struct holding a resource is a resource, and both places that work that out
// read the declarations: they ask whether `Slab`'s fields hold one. `Slab`'s
// field is `[N]T`, whose element is a parameter bound to nothing, so the answer
// is no for `Slab` and therefore no for every `Slab<T, N>` a program writes. A
// `Slab<Node, 2>` whose `Node` holds a `File` was ordinary data, so the resource
// put in a slot was dropped when the slot was reused and nothing said so.
//
// The instantiations a program mentions carry their arguments in the name, so
// binding the template's parameters to them gives the field types that
// instantiation really has, and those answer the question. Every type position
// is walked to find the names, since an instantiation is written wherever a type
// is: a binding's annotation, a parameter, a return, a field of another struct.

use crate::ast::{
    Ast, ExprId, Expression, Range32, ReturnKind, Statement, StmtId,
};
use crate::ir::build::substitute_type;
use crate::lexer::Position;
use crate::parser::type_from_string;
use crate::types::Type;
use std::collections::{HashMap, HashSet};

/// The generic structs a program declares, by name, with their parameters and
/// the field types written under them.
pub(crate) type Templates<'a> =
    HashMap<&'a str, (Vec<&'a str>, Vec<(&'a str, &'a Type)>)>;

/// Instantiation names with where each was written.
pub(crate) type Located = HashMap<String, Position>;

/// Grow `held` with every instantiation whose bound fields hold a resource.
/// Answers whether it grew, so a caller running a fixpoint over holders can run
/// this inside the same loop and let the two converge together: an instance is a
/// resource because of a field, and a struct is a resource because of an
/// instance in a field of its own.
pub(crate) fn note_linear_instances(
    templates: &Templates,
    instances: &HashSet<String>,
    held: &mut HashSet<String>,
) -> bool {
    if held.is_empty() {
        return false;
    }
    let mut grew = false;
    for instance in instances {
        if held.contains(instance.as_str()) {
            continue;
        }
        if instance_is_linear(instance, templates, held) {
            held.insert(instance.clone());
            grew = true;
        }
    }
    grew
}

/// Refuse a pool whose elements are resources.
///
/// A slot is emptied by bumping a generation and filled again by an insert that
/// overwrites what was there, so nothing consumes the element that leaves. The
/// container carries one obligation and its slots carry none, and there is no
/// consumer that can discharge the difference: releasing each element means
/// consuming `p.storage[i]` around a loop, which is a move inside a loop and
/// refused, correctly, because nothing says the indexes differ.
///
/// So the shape is one the language demands be consumed and gives no way to
/// consume. It is refused where it is written, which is the only place a reader
/// can do anything about it, and the message names what to write instead.
pub(crate) fn check_pooled_resources(
    ast: &Ast,
    roots: &[StmtId],
    instances: &Located,
    held: &HashSet<String>,
) -> Vec<String> {
    if held.is_empty() {
        return Vec::new();
    }
    let templates = declared_structs(ast, roots);
    let pools = every_pool(ast, roots, instances);
    let mut reports = Vec::new();
    for (instance, element, at) in refused(&pools, &templates, held) {
        reports.push(format!(
            "at {}: {}",
            at.describe(),
            pool_report(&instance, &element)
        ));
    }
    reports.sort();
    // One type reaches both rules by more than one road, and a reader wants the
    // complaint once.
    reports.dedup();
    reports
}

/// The pools refused, as the names their values are typed with.
///
/// Nothing consumes such a value the way the language asks, which is the
/// refusal, so the walk that counts consumptions leaves them alone rather than
/// telling a reader to do what cannot be done. Answered here rather than read
/// back out of the reports above: a check and the words it is reported in are
/// two things, and matching one against the other means the words cannot be
/// changed without the check quietly stopping.
pub fn pooled_instance_names(
    ast: &Ast,
    roots: &[StmtId],
    linear: &HashSet<String>,
) -> HashSet<String> {
    if linear.is_empty() {
        return HashSet::new();
    }
    // The types a program declares `linear` and the ones that hold one, which
    // is the set the check itself is asked about. A `Node` holding a `File` is
    // a resource without saying so, and asking with the declared names alone
    // answered that a slab of them was not a pool of resources when the report
    // beside it said it was.
    let held = crate::check::ownership::linear_closure(
        linear,
        &crate::check::ownership::collect_field_types(ast, roots),
        ast,
        roots,
    );
    let templates = declared_structs(ast, roots);
    let pools = every_pool(ast, roots, &locate_instances(ast, roots));
    refused(&pools, &templates, &held)
        .into_iter()
        .map(|(instance, _, _)| instance)
        .collect()
}

/// Every container a program has, instantiated or written out.
///
/// A concrete `Pool :: struct { storage: [4]File, generations: [4]i64 }` is the
/// same container written out, and asking only about instantiations let it
/// through.
fn every_pool(
    ast: &Ast,
    roots: &[StmtId],
    instances: &Located,
) -> Vec<(String, Position)> {
    let mut pools: Vec<(String, Position)> = instances
        .iter()
        .map(|(name, at)| (name.clone(), *at))
        .collect();
    for statement in roots {
        if let Statement::Struct(name, params, _) = ast.stmt(*statement)
            && params.is_empty()
        {
            pools.push((
                ast.name(*name).to_string(),
                ast.stmt_position(*statement),
            ));
        }
    }
    pools
}

/// The ones among them whose slots hold a resource.
fn refused(
    pools: &[(String, Position)],
    templates: &Templates,
    held: &HashSet<String>,
) -> Vec<(String, Type, Position)> {
    let mut found = Vec::new();
    for (instance, at) in pools {
        // A name with no arguments binds nothing, and its fields are already the
        // types it has, so the same substitution answers for both shapes.
        let (base, arguments) = split_instance(instance)
            .unwrap_or_else(|| (instance.clone(), Vec::new()));
        let Some(element) = pool_element(&base, &arguments, templates) else {
            continue;
        };
        if !element.is_linear_with(held) {
            continue;
        }
        found.push((instance.clone(), element, *at));
    }
    found
}

/// What to say about a pool holding a resource. The name carries the prefix an
/// import gives a private declaration, and a reader wrote neither, so it is put
/// back the way every other diagnostic puts it back.
fn pool_report(instance: &str, element: &Type) -> String {
    let named = crate::modules::imports::demangle_private_names(&format!(
        "'{instance}' is a pool of '{element}'"
    ));
    format!(
        "{named}, which is a resource. A slot is released by bumping a \
         generation and filled again by an insert that overwrites it, so \
         nothing consumes the element that leaves. Keep the resource outside \
         the pool and put a handle to it in the slot, or hold the elements \
         beside the pool: one array of offsets giving each element its range \
         into a single run that owns the whole of it."
    )
}

/// The complaint to make about a concrete type holding a pool of resources, or
/// `None` where the type is sound.
///
/// The rule itself is 10.3's. What this adds is where it is asked: of the types
/// a program *forms* rather than the ones it writes down. A generic's own field
/// names a parameter bound to nothing, so `Slab` holds no resource while
/// `Slab<Node, 2>` does, and a program that never writes that name still makes
/// one when it calls a generic that answers with it.
pub(crate) fn pooled_resource_in(
    ty: &Type,
    templates: &Templates,
    held: &HashSet<String>,
) -> Option<String> {
    if held.is_empty() {
        return None;
    }
    let mut seen = HashSet::new();
    walk_concrete(ty, templates, held, &mut seen)
}

fn walk_concrete(
    ty: &Type,
    templates: &Templates,
    held: &HashSet<String>,
    seen: &mut HashSet<String>,
) -> Option<String> {
    match ty {
        // A name binds its arguments to the template's parameters, so the
        // fields it really has are what answer. A type reached twice is left
        // alone the second time, since a struct may name itself through a
        // pointer and this walk would not end.
        Type::Struct(name) | Type::Enum(name) => {
            if !name.contains('<') || !seen.insert(name.clone()) {
                return None;
            }
            let (base, arguments) = split_instance(name)?;
            let (params, fields) = templates.get(base.as_str())?;
            if params.len() != arguments.len() {
                return None;
            }
            // A pool is the same question about the same type, so it is asked
            // here as well: an instantiation the program never writes down is
            // reached through what a call makes rather than through a name, and
            // the walk over written names cannot see one.
            if let Some(element) = pool_element(&base, &arguments, templates)
                && element.is_linear_with(held)
            {
                return Some(pool_report(name, &element));
            }
            let mut subst: HashMap<String, Type> = HashMap::new();
            for (param, argument) in params.iter().zip(arguments.iter()) {
                subst.insert((*param).to_string(), argument_type(argument));
            }
            fields.iter().find_map(|(_, field_type)| {
                let concrete = substitute_type(field_type, &subst);
                walk_concrete(&concrete, templates, held, seen)
            })
        }
        Type::Ptr(inner)
        | Type::Ref(inner)
        | Type::RefMut(inner)
        | Type::Array(inner, _)
        | Type::ArrayGeneric(inner, _)
        | Type::Slice(inner)
        | Type::Handle(inner)
        | Type::Distinct(_, inner) => {
            walk_concrete(inner, templates, held, seen)
        }
        Type::Proc(parameters, result) => parameters
            .iter()
            .chain(std::iter::once(result.as_ref()))
            .find_map(|held_type| {
                walk_concrete(held_type, templates, held, seen)
            }),
        _ => None,
    }
}

/// The element type of a pool, or `None` where this instantiation is not one.
///
/// `columns<T, N>` is the compiler's own container and names its element first.
/// A slab is an ordinary struct recognized by its shape, which is a `storage`
/// array beside a `generations` array: that is the same rule the handle deref
/// uses, so a type that can be indexed by a handle is a type this asks about.
fn pool_element(
    base: &str,
    arguments: &[String],
    templates: &Templates,
) -> Option<Type> {
    if base == "columns" {
        return arguments.first().map(|argument| argument_type(argument));
    }
    let (params, fields) = templates.get(base)?;
    if params.len() != arguments.len() {
        return None;
    }
    let field_type = |wanted: &str| {
        fields
            .iter()
            .find(|(name, _)| *name == wanted)
            .map(|(_, field_type)| *field_type)
    };
    let storage = field_type("storage")?;
    let generations = field_type("generations")?;
    if !is_run(storage) || !is_run(generations) {
        return None;
    }
    let mut subst: HashMap<String, Type> = HashMap::new();
    for (param, argument) in params.iter().zip(arguments.iter()) {
        subst.insert((*param).to_string(), argument_type(argument));
    }
    match substitute_type(storage, &subst) {
        Type::Array(inner, _) | Type::ArrayGeneric(inner, _) => Some(*inner),
        _ => None,
    }
}

/// Whether a field is a run of elements held by value, which is what a slot
/// table is. A slice looks at storage it does not own and is not one.
fn is_run(ty: &Type) -> bool {
    matches!(ty, Type::Array(..) | Type::ArrayGeneric(..))
}

/// The declared structs, gathered once. Both the closure that runs to a fixpoint
/// and the pool rule read the same table, and building it per round made a pass
/// meant to stay linear in a program's size walk it again for every type the
/// closure found.
pub(crate) fn declared_structs<'a>(
    ast: &'a Ast,
    roots: &[StmtId],
) -> Templates<'a> {
    let mut templates: Templates = HashMap::new();
    for statement in roots {
        match ast.stmt(*statement) {
            Statement::Struct(name, params, fields) => {
                let params: Vec<&str> = ast
                    .symbols_in(*params)
                    .iter()
                    .map(|param| ast.name(*param))
                    .collect();
                let fields: Vec<(&str, &Type)> = ast
                    .fields_in(*fields)
                    .iter()
                    .map(|field| (ast.name(field.name), &field.field_type))
                    .collect();
                templates.insert(ast.name(*name), (params, fields));
            }
            // An enum holds a resource when any variant's payload does, the
            // same way a struct holds one when any field does. Reading only
            // the structs left `Option<File>` ordinary data, so a resource put
            // in one lost its obligation on the way in.
            Statement::Enum(name, params, variants) => {
                let params: Vec<&str> = ast
                    .symbols_in(*params)
                    .iter()
                    .map(|param| ast.name(*param))
                    .collect();
                let payload: Vec<(&str, &Type)> = ast
                    .variants_in(*variants)
                    .iter()
                    .filter_map(|variant| variant.fields)
                    .flat_map(|fields| ast.fields_in(fields))
                    .map(|field| (ast.name(field.name), &field.field_type))
                    .collect();
                templates.insert(ast.name(*name), (params, payload));
            }
            _ => {}
        }
    }
    templates
}

/// Whether one instantiation holds a resource, by binding the template's
/// parameters to the arguments written in its name and asking what its fields
/// then are.
fn instance_is_linear(
    instance: &str,
    templates: &Templates,
    held: &HashSet<String>,
) -> bool {
    let Some((base, arguments)) = split_instance(instance) else {
        return false;
    };
    let Some((params, fields)) = templates.get(base.as_str()) else {
        return false;
    };
    if params.len() != arguments.len() {
        return false;
    }
    let mut subst: HashMap<String, Type> = HashMap::new();
    for (param, argument) in params.iter().zip(arguments.iter()) {
        subst.insert((*param).to_string(), argument_type(argument));
    }
    fields.iter().any(|(_, field_type)| {
        substitute_type(field_type, &subst).is_linear_with(held)
    })
}

/// An argument written in an instance's name, as the type it stands for. A
/// number is a length rather than a type, which is what an array of a generic
/// size is waiting for. Anything the type reader cannot parse is left as the
/// name it was written with, which names no resource and so answers no.
fn argument_type(argument: &str) -> Type {
    if let Ok(length) = argument.trim().parse::<usize>() {
        return Type::ConstUsize(length);
    }
    match type_from_string(argument) {
        Ok(ty) => ty,
        Err(_) => Type::Struct(argument.to_string()),
    }
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

/// Every generic instantiation a program writes down. Collected once, since the
/// set does not change while the linear set grows.
pub(crate) fn collect_instances(
    ast: &Ast,
    roots: &[StmtId],
) -> HashSet<String> {
    locate_instances(ast, roots).into_keys().collect()
}

/// The same, with where each was written, which is where a complaint about one
/// belongs. The position is the statement the name was found in, since that is
/// the line a reader looks at.
pub(crate) fn locate_instances(ast: &Ast, roots: &[StmtId]) -> Located {
    let mut found = Located::new();
    for statement in roots {
        walk_statement(
            ast,
            *statement,
            &mut found,
            ast.stmt_position(*statement),
        );
    }
    // An argument can itself be an instantiation, and a field of one names types
    // the outer name does not, so the names inside a name are taken too.
    let mut pending: Vec<(String, Position)> =
        found.iter().map(|(name, at)| (name.clone(), *at)).collect();
    while let Some((name, at)) = pending.pop() {
        let Some((_, arguments)) = split_instance(&name) else {
            continue;
        };
        for argument in arguments {
            if argument.contains('<') && !found.contains_key(argument.as_str())
            {
                found.insert(argument.clone(), at);
                pending.push((argument, at));
            }
        }
    }
    found
}

fn note_type(ty: &Type, found: &mut Located, at: Position) {
    match ty {
        Type::Struct(name) | Type::Enum(name) => {
            if name.contains('<') {
                found.entry(name.clone()).or_insert(at);
            }
        }
        Type::Ptr(inner)
        | Type::Slice(inner)
        | Type::Ref(inner)
        | Type::RefMut(inner)
        | Type::Array(inner, _)
        | Type::ArrayGeneric(inner, _)
        | Type::Handle(inner)
        | Type::Distinct(_, inner) => note_type(inner, found, at),
        Type::Proc(parameters, result) => {
            for parameter in parameters {
                note_type(parameter, found, at);
            }
            note_type(result, found, at);
        }
        _ => {}
    }
}

fn walk_parameters(
    ast: &Ast,
    parameters: Range32,
    found: &mut Located,
    at: Position,
) {
    for parameter in ast.params_in(parameters) {
        if let Some(ty) = &parameter.type_annotation {
            note_type(ty, found, at);
        }
        if let Some(ty) = &parameter.compile_time_signature {
            note_type(ty, found, at);
        }
    }
}

fn walk_return(
    ast: &Ast,
    kind: &ReturnKind,
    found: &mut Located,
    at: Position,
) {
    match kind {
        ReturnKind::None => {}
        ReturnKind::Single(ty) => note_type(ty, found, at),
        ReturnKind::Multiple(values) => {
            for held in ast.return_values_in(*values) {
                note_type(&held.value_type, found, at);
            }
        }
        ReturnKind::Fallible(value, error) => {
            note_type(value, found, at);
            note_type(error, found, at);
        }
    }
}

fn walk_block(ast: &Ast, block: Range32, found: &mut Located) {
    for statement in ast.stmts_in(block) {
        walk_statement(ast, *statement, found, ast.stmt_position(*statement));
    }
}

// Listed rather than caught by a wildcard, so a new statement form is a compile
// error here instead of a type position nobody walks.
fn walk_statement(
    ast: &Ast,
    statement: StmtId,
    found: &mut Located,
    at: Position,
) {
    match ast.stmt(statement) {
        Statement::TypeAlias(_, ty) | Statement::Flags(_, ty, _) => {
            note_type(ty, found, at)
        }
        Statement::Struct(_, _, fields) => {
            for field in ast.fields_in(*fields) {
                note_type(&field.field_type, found, at);
            }
        }
        Statement::Enum(_, _, variants) => {
            for variant in ast.variants_in(*variants) {
                if let Some(fields) = variant.fields {
                    for field in ast.fields_in(fields) {
                        note_type(&field.field_type, found, at);
                    }
                }
            }
        }
        Statement::Extern {
            params,
            return_type,
            ..
        } => {
            walk_parameters(ast, *params, found, at);
            if let Some(ty) = return_type {
                note_type(ty, found, at);
            }
        }
        Statement::Declared {
            params, return_sig, ..
        } => {
            walk_parameters(ast, *params, found, at);
            let signature = ast.signature(*return_sig);
            walk_return(ast, &signature.kind, found, at);
            for capability in &signature.uses {
                note_type(capability, found, at);
            }
        }
        Statement::Let {
            type_annotation,
            value,
            ..
        } => {
            if let Some(ty) = type_annotation {
                note_type(ty, found, at);
            }
            walk_expression(ast, *value, found, at);
        }
        Statement::LetMultiple(_, value)
        | Statement::Constant(_, value)
        | Statement::Return(value)
        | Statement::Expression(value) => {
            walk_expression(ast, *value, found, at)
        }
        Statement::Assignment(place, value) => {
            walk_expression(ast, *place, found, at);
            walk_expression(ast, *value, found, at);
        }
        Statement::Defer(inner) | Statement::ErrDefer(inner) => {
            walk_statement(ast, *inner, found, at)
        }
        Statement::For(_, _, sequence, body) => {
            walk_expression(ast, *sequence, found, at);
            walk_block(ast, *body, found);
        }
        Statement::While(condition, body) => {
            walk_expression(ast, *condition, found, at);
            walk_block(ast, *body, found);
        }
        Statement::With(_, body) => walk_block(ast, *body, found),
        Statement::Break | Statement::Continue | Statement::Import(..) => {}
    }
}

fn walk_expression(
    ast: &Ast,
    expression: ExprId,
    found: &mut Located,
    at: Position,
) {
    match ast.expr(expression) {
        Expression::Function(parameters, signature, body)
        | Expression::Proc(parameters, signature, body) => {
            walk_parameters(ast, *parameters, found, at);
            let signature = ast.signature(*signature);
            walk_return(ast, &signature.kind, found, at);
            for capability in &signature.uses {
                note_type(capability, found, at);
            }
            walk_block(ast, *body, found);
        }
        Expression::TypeValue(ty) => note_type(ty, found, at),
        Expression::PackMap(inner, _, _)
        | Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Dereference(inner)
        | Expression::UnsafeFn(inner)
        | Expression::FieldAccess(inner, _)
        | Expression::ArrayRepeat(inner, _)
        | Expression::Try(inner) => walk_expression(ast, *inner, found, at),
        Expression::Infix(left, _, right)
        | Expression::Index(left, right)
        | Expression::Range(left, right, _) => {
            walk_expression(ast, *left, found, at);
            walk_expression(ast, *right, found, at);
        }
        Expression::If(condition, then_block, else_block) => {
            walk_expression(ast, *condition, found, at);
            walk_block(ast, *then_block, found);
            if let Some(block) = else_block {
                walk_block(ast, *block, found);
            }
        }
        Expression::Unsafe(block) => walk_block(ast, *block, found),
        Expression::Switch(scrutinee, cases) => {
            walk_expression(ast, *scrutinee, found, at);
            for case in ast.cases_in(*cases) {
                walk_block(ast, case.body, found);
            }
        }
        Expression::Call(callee, arguments) => {
            walk_expression(ast, *callee, found, at);
            for argument in ast.exprs_in(*arguments) {
                walk_expression(ast, *argument, found, at);
            }
        }
        Expression::Tuple(values) => {
            for held in ast.exprs_in(*values) {
                walk_expression(ast, *held, found, at);
            }
        }
        // A struct literal names its type, and a generic one names the
        // instantiation it makes.
        Expression::StructInit(name, fields) => {
            let name = ast.name(*name);
            if name.contains('<') {
                found.entry(name.to_string()).or_insert(at);
            }
            for field in ast.named_in(*fields) {
                walk_expression(ast, field.value, found, at);
            }
        }
        Expression::EnumVariantInit(name, _, fields) => {
            let name = ast.name(*name);
            if name.contains('<') {
                found.entry(name.to_string()).or_insert(at);
            }
            for field in ast.named_in(*fields) {
                walk_expression(ast, field.value, found, at);
            }
        }
        Expression::Identifier(_)
        | Expression::Literal(_)
        | Expression::Boolean(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn instances(source: &str) -> HashSet<String> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let module = parser.parse().unwrap();
        collect_instances(&module.ast, &module.roots)
    }

    #[test]
    fn an_instantiation_binding_a_resource_is_one() {
        let source = "File :: linear struct { handle: i64 }\n\
                      Pool :: struct($T: Type) { slot: $T }\n\
                      run :: fn() {\n\
                          held : Pool<File> = Pool { slot = File { handle = 1 } }\n\
                      }";
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let module = parser.parse().unwrap();
        let found = collect_instances(&module.ast, &module.roots);
        let mut held = parser.linear_types().clone();
        let templates = declared_structs(&module.ast, &module.roots);
        note_linear_instances(&templates, &found, &mut held);
        assert!(held.contains("Pool<File>"), "held {held:?}");
    }

    #[test]
    fn an_annotation_names_its_instantiation() {
        let found = instances(
            "File :: linear struct { handle: i64 }\n\
             Pool :: struct($T: Type) { slot: $T }\n\
             run :: fn() {\n\
                 held : Pool<File> = Pool { slot = File { handle = 1 } }\n\
             }",
        );
        assert!(found.contains("Pool<File>"), "found {found:?}");
    }
}
