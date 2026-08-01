use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, bail};

use crate::ir::{
    BlockId, EnumLayout, EnumVariantLayout, FieldLayout, IrBinOp, IrBlock,
    IrConstant, IrExtern, IrFunction, IrLocal, IrModule, IrOperand, IrRvalue,
    IrStatement, IrTerminator, IrUnOp, LocalId, StructLayout,
};
use crate::lexer::Position;
use crate::parser::{
    Block, EnumVariant, Expression, Parameter, Pattern, ReturnKind,
    ReturnSignature, Spanned, Statement, StructField, SwitchCase,
};
use crate::types::Type;
use crate::{Literal, Operator};

pub const BUILTIN_FUNCTIONS: &[&str] = &[
    "assert",
    "cast",
    "flags_has",
    "ptr_cast",
    "ptr_to",
    "slice_from",
    "slice_len",
    "str_len",
    "wrap_add",
    "wrap_mul",
    "wrap_sub",
];

struct FunctionSignature {
    parameters: Vec<Type>,
    return_type: Type,
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
    constants: HashMap<String, Expression>,
    generic_functions: HashMap<String, GenericFunction>,
    generic_struct_defs: HashMap<String, (Vec<String>, Vec<StructField>)>,
    linear: HashSet<String>,
    // Callback registrations, by name.
    registrations: HashMap<String, crate::callbacks::CallbackShape>,
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

struct AnonRequest {
    name: String,
    parameters: Vec<Parameter>,
    return_sig: ReturnSignature,
    body: Block,
    // The module whose lowering produced this literal, carried for the same
    // reason `Specialization` carries it. A generic instantiated from inside an
    // anonymous function is work that module would have to do.
    requested_by: u32,
}

fn locate<T>(result: Result<T>, position: Position) -> Result<T> {
    result.map_err(|error| {
        let text = crate::imports::demangle_private_names(&error.to_string());
        if position == Position::default() || text.starts_with("at ") {
            anyhow::anyhow!("{text}")
        } else {
            anyhow::anyhow!("at {}: {text}", position.describe())
        }
    })
}

pub fn build_module(
    statements: &[Spanned<Statement>],
    linear: &HashSet<String>,
) -> Result<IrModule> {
    build_module_inner(statements, linear, false)
}

// The same lowering, but a specialization is emitted once per module that
// instantiates it rather than once per program. Only correct when the result is
// about to be split into one object per module, since two definitions of a name
// in a single object is a duplicate symbol. Split, they are module-local and a
// module's object is self-contained.
pub fn build_module_per_module(
    statements: &[Spanned<Statement>],
    linear: &HashSet<String>,
) -> Result<IrModule> {
    build_module_inner(statements, linear, true)
}

fn build_module_inner(
    statements: &[Spanned<Statement>],
    linear: &HashSet<String>,
    per_module: bool,
) -> Result<IrModule> {
    let synthetic_structs = expand_generic_structs(statements)?;
    let mut layout_statements: Vec<Statement> =
        statements.iter().map(|s| s.node.clone()).collect();
    layout_statements.extend(synthetic_structs);
    let (structs, enums) = compute_layouts(&layout_statements);
    let mut constants = HashMap::new();
    for statement in statements {
        if let Statement::Constant(name, value) = &statement.node
            && !matches!(value, Expression::Function(..) | Expression::Proc(..))
        {
            constants.insert(name.clone(), value.clone());
        }
    }
    check_constant_cycles(&constants)?;
    let mut generic_functions = HashMap::new();
    for statement in statements {
        if let Statement::Constant(
            name,
            Expression::Function(parameters, return_sig, body)
            | Expression::Proc(parameters, return_sig, body),
        ) = &statement.node
            && function_is_generic(parameters)
        {
            let type_params = function_type_params(parameters);
            generic_functions.insert(
                name.clone(),
                GenericFunction {
                    type_params,
                    parameters: parameters.clone(),
                    return_sig: return_sig.clone(),
                    body: body.clone(),
                },
            );
        }
    }

    let mut generic_struct_defs = HashMap::new();
    for statement in statements {
        if let Statement::Struct(name, type_params, fields) = &statement.node
            && !type_params.is_empty()
        {
            generic_struct_defs
                .insert(name.clone(), (type_params.clone(), fields.clone()));
        }
    }

    let mut flags: HashMap<String, FlagsLayout> = HashMap::new();
    for statement in statements {
        if let Statement::Flags(name, repr, bits) = &statement.node {
            flags.insert(
                name.clone(),
                FlagsLayout {
                    repr: repr.clone(),
                    bits: bits
                        .iter()
                        .map(|bit| (bit.name.clone(), bit.value))
                        .collect(),
                },
            );
        }
    }

    // The concrete types beside the written ones, so a place reached through an
    // instantiation has a type. `Vec<File>` is where `storage` is a run of
    // resources; `Vec` alone says only that it is a run of whatever `T` stands
    // for, and nothing can be told about a resource from that. A call that
    // answers with an instantiation makes one without anyone writing its name,
    // which is why this is read from what specialization forms rather than from
    // what the source spells out.
    let mut with_instances: Vec<Spanned<Statement>> = statements.to_vec();
    with_instances.extend(
        layout_statements
            .iter()
            .skip(statements.len())
            .map(|statement| {
                Spanned::new(statement.clone(), Position::default())
            }),
    );

    let mut builder = IrBuilder {
        signatures: HashMap::new(),
        structs,
        enums,
        flags,
        constants,
        generic_functions,
        generic_struct_defs,
        linear: linear_with_holders(linear, &with_instances),
        registrations: crate::callbacks::callback_registrations(statements),
        type_ids: std::cell::RefCell::new(HashMap::new()),
        anon_counter: std::cell::Cell::new(0),
    };
    builder.collect_signatures(statements);

    let ownership = crate::ownership::specializations(&with_instances, linear);

    let mut functions = Vec::new();
    let mut externs = Vec::new();
    let mut declared = Vec::new();
    let mut top_level = Vec::new();
    let mut has_main = false;
    let mut pending: Vec<Specialization> = Vec::new();
    let mut pending_anon: Vec<AnonRequest> = Vec::new();

    for statement in statements {
        let position = statement.position;
        match &statement.node {
            Statement::Constant(
                name,
                Expression::Function(parameters, return_sig, body)
                | Expression::Proc(parameters, return_sig, body),
            ) => {
                if function_is_generic(parameters) {
                    continue;
                }
                if name == "main" {
                    has_main = true;
                }
                // Expansion time runs over every body, not only a
                // specialization's: a walk over a type's fields is decided by
                // a declaration rather than by a call, so an ordinary function
                // may write one.
                let body = &expand_compile_time(
                    body.clone(),
                    None,
                    parameters,
                    ExpansionContext {
                        structs: &builder.structs,
                        subst: &HashMap::new(),
                        linear,
                    },
                )?;
                // The ownership rules again, over the types specialization
                // forms rather than only the ones the source writes down. A
                // call that answers with an instantiation makes one without
                // anyone naming it, so `held := option_some($File, ...)` left
                // `Option<File>` ordinary data and the obligation on the
                // resource inside it went in and did not come out.
                if let Some(first) = ownership.check(parameters, body).first() {
                    bail!(
                        "{}",
                        crate::imports::demangle_private_names(&format!(
                            "at {}: {first}",
                            position.describe()
                        ))
                    );
                }
                let (function, requests, anon) = locate(
                    builder.lower_function(name, parameters, return_sig, body),
                    position,
                )?;
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
                let return_type = return_type.clone().unwrap_or(Type::Void);
                let return_layout = builder.c_layout(&return_type);
                let param_layouts = params
                    .iter()
                    .map(|parameter| {
                        if parameter.mode != crate::parser::ParamMode::Value {
                            return Ok(None);
                        }
                        let Some(ty) = &parameter.type_annotation else {
                            bail!(
                                "the parameter '{}' of the extern '{name}' is written 'value' but has no type",
                                parameter.name
                            );
                        };
                        let Some(layout) = builder.c_layout(ty) else {
                            bail!(
                                "'{}' of the extern '{name}' is written 'value', but '{ty}' is not an aggregate; a scalar already goes to C by value and needs no mode",
                                parameter.name
                            );
                        };
                        Ok(Some(layout))
                    })
                    .collect::<Result<Vec<_>>>()?;
                externs.push(IrExtern {
                    name: name.clone(),
                    params: extern_parameter_types(params),
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
                declared.push(declared_function(name, params, return_sig));
            }
            Statement::Struct(..)
            | Statement::Enum(..)
            | Statement::Flags(..)
            | Statement::TypeAlias(..)
            | Statement::Import(..) => {}
            _ => top_level.push(statement.clone()),
        }
    }

    if !has_main && !top_level.is_empty() {
        let empty_params: Vec<Parameter> = Vec::new();
        let mut body = top_level.clone();
        let ends_in_expression = matches!(
            body.last().map(|statement| &statement.node),
            Some(Statement::Expression(_))
        );
        if !ends_in_expression
            && let Some(position) =
                body.last().map(|statement| statement.position)
        {
            body.push(Spanned::new(
                Statement::Expression(Expression::Literal(Literal::Integer(0))),
                position,
            ));
        }
        let (function, requests, anon) = builder.lower_function(
            "main",
            &empty_params,
            &ReturnSignature::plain(ReturnKind::Single(Type::I64)),
            &body,
        )?;
        functions.push(in_module(function, 0));
        // Synthesized `main` from loose top-level statements, which belong to
        // the entry file.
        pending.extend(requested_by(requests, 0));
        pending_anon.extend(anon_requested_by(anon, 0));
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
            let mut parameters: Vec<Parameter> = generic
                .parameters
                .iter()
                .filter(|parameter| {
                    !is_type_parameter(parameter) && !parameter.pack
                })
                .map(|parameter| Parameter {
                    name: parameter.name.clone(),
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
                    pack: false,
                })
                .collect();
            // The bound was checked at the call that asked for this
            // specialization, so the specialized signature carries none.
            let return_sig = ReturnSignature {
                bound: None,
                kind: match generic.return_sig.to_type() {
                    Some(ty) => ReturnKind::Single(substitute_type(
                        &ty,
                        &specialization.subst,
                    )),
                    None => ReturnKind::None,
                },
                uses: generic
                    .return_sig
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
                        name: name.clone(),
                        type_annotation: Some(ty.clone()),
                        mutable: false,
                        mode: crate::parser::ParamMode::Read,
                        compile_time_signature: None,
                        pack: false,
                    });
                }
            }
            let body = substitute_block(&generic.body, &specialization.subst);
            // Expansion time: a `for` over the list unrolls, `list[K]` becomes
            // the Kth element, and an `if` over a type predicate keeps the one
            // branch that survives. All three are decided here, where the types
            // are known, and none of them exists afterwards.
            let body = expand_compile_time(
                body,
                specialization.pack.as_ref(),
                &parameters,
                ExpansionContext {
                    structs: &builder.structs,
                    subst: &specialization.subst,
                    linear,
                },
            )?;
            // The ownership rules, asked of the body that really exists. The
            // template's own says nothing: its parameters are bound to nothing,
            // so no type there is a resource and a list has no elements to
            // unroll. This is the first and only point where both are true.
            let complaints = ownership.check(&parameters, &body);
            if let Some(first) = complaints.first() {
                // The prefix an import gives a private name is nothing the
                // reader wrote, so it comes back off the way it does in every
                // other diagnostic.
                bail!(
                    "{}",
                    crate::imports::demangle_private_names(&format!(
                        "at {}: instantiating '{}': {first}",
                        specialization.requested_at.describe(),
                        specialization.display
                    ))
                );
            }
            let (function, requests, anon) = locate_instantiation(
                builder.lower_function(
                    &specialization.mangled_name,
                    &parameters,
                    &return_sig,
                    &body,
                ),
                &specialization,
            )?;
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
            let (function, requests, anon) = builder.lower_function(
                &request.name,
                &request.parameters,
                &request.return_sig,
                &request.body,
            )?;
            functions.push(local_to_module(function, request.requested_by));
            pending.extend(requested_by(requests, request.requested_by));
            pending_anon.extend(anon_requested_by(anon, request.requested_by));
        } else {
            break;
        }
    }

    report_module_specializations(&instantiated_by);
    Ok(IrModule {
        functions,
        externs,
        imported: declared,
    })
}

// The shape a declared function needs to have for a backend to emit a call to
// it: parameter types, a return type, and no blocks. It rides in `imported`,
// which already means "declared here, defined in another object", and which the
// backends already declare with the same signature builder that builds a
// definition.
fn declared_function(
    name: &str,
    params: &[Parameter],
    return_sig: &ReturnSignature,
) -> IrFunction {
    let locals: Vec<crate::ir::IrLocal> = params
        .iter()
        .map(|parameter| {
            let ty = parameter_type(parameter);
            crate::ir::IrLocal {
                size: ty.size_of(),
                ty,
                name: Some(parameter.name.clone()),
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
        return_type: return_sig.to_type().unwrap_or(Type::Void),
        locals,
        blocks: Vec::new(),
        entry: 0,
        instantiated: None,
        module: 0,
        local: false,
    }
}

// What an extern's parameters are once C sees them. For a registration these
// are not what the declaration says literally. The `$handler` parameter is the
// callback pointer, and the context is passed as an address, because the library
// keeps it past the call.
fn extern_parameter_types(params: &[Parameter]) -> Vec<Type> {
    let shape = crate::callbacks::callback_shape(params);
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
    result.map_err(|error| {
        let text = crate::imports::demangle_private_names(&error.to_string());
        let display =
            crate::imports::demangle_private_names(&specialization.display);
        if specialization.requested_at == Position::default() {
            anyhow::anyhow!("instantiating '{display}': {text}")
        } else {
            anyhow::anyhow!(
                "at {}: instantiating '{display}': {text}",
                specialization.requested_at.describe()
            )
        }
    })
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
    constants: &HashMap<String, Expression>,
) -> Result<()> {
    let mut settled: HashSet<String> = HashSet::new();
    let mut path: Vec<String> = Vec::new();
    let mut names: Vec<&String> = constants.keys().collect();
    names.sort();
    for name in names {
        walk_constant(name, constants, &mut settled, &mut path)?;
    }
    Ok(())
}

fn walk_constant(
    name: &str,
    constants: &HashMap<String, Expression>,
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
        crate::interface_names::names_in_expression(value, &mut referenced);
    }
    for reference in referenced {
        if constants.contains_key(&reference) {
            walk_constant(&reference, constants, settled, path)?;
        }
    }
    path.pop();
    settled.insert(name.to_string());
    Ok(())
}

impl IrBuilder {
    fn collect_signatures(&mut self, statements: &[Spanned<Statement>]) {
        for statement in statements {
            match &statement.node {
                Statement::Constant(
                    name,
                    Expression::Function(parameters, return_sig, _)
                    | Expression::Proc(parameters, return_sig, _),
                ) => {
                    if function_is_generic(parameters) {
                        continue;
                    }
                    self.signatures.insert(
                        name.clone(),
                        FunctionSignature {
                            parameters: parameters
                                .iter()
                                .map(parameter_type)
                                .collect(),
                            return_type: return_sig
                                .to_type()
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
                        name.clone(),
                        FunctionSignature {
                            parameters: extern_parameter_types(params),
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
                        name.clone(),
                        FunctionSignature {
                            parameters: params
                                .iter()
                                .map(parameter_type)
                                .collect(),
                            return_type: return_sig
                                .to_type()
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
        name: &str,
        parameters: &[Parameter],
        return_sig: &ReturnSignature,
        body: &Block,
    ) -> Result<(IrFunction, Vec<Specialization>, Vec<AnonRequest>)> {
        let return_type = return_sig.to_type().unwrap_or(Type::Void);
        let mut function = FunctionLowering::new(self, return_type.clone());

        // A parameter is bound before any statement runs, so it would carry no
        // position and a type error about one would name a function and nothing
        // else. The body's first statement is where a reader looks.
        if let Some(first) = body.first() {
            function.current_position = first.position;
        }
        for parameter in parameters {
            let ty = parameter_type(parameter);
            let local = function.fresh_local(ty, Some(parameter.name.clone()));
            function.define_variable(&parameter.name, local);
        }

        let has_defers =
            body.iter().any(|s| matches!(s.node, Statement::Defer(_)));
        if has_defers {
            function.lower_body_with_defers(body, &return_type)?;
        } else {
            let (value, value_type) =
                function.lower_block(body, Some(&return_type))?;
            if !function.current_is_terminated() {
                if matches!(return_type, Type::Void) {
                    function.set_terminator(IrTerminator::Return(None));
                } else {
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
        let param_layouts = parameters
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
                param_count: parameters.len(),
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

    fn byte_size(&self, ty: &Type) -> usize {
        size_and_align(ty, &self.structs, &self.enums)
            .map(|(size, _)| size)
            .unwrap_or(0)
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
fn linear_with_holders(
    declared: &HashSet<String>,
    statements: &[Spanned<Statement>],
) -> HashSet<String> {
    let mut held = declared.clone();
    if held.is_empty() {
        return held;
    }
    // The instantiations, which the declarations alone cannot answer for: a
    // generic's field is a parameter bound to nothing here, so `Slab` holds no
    // resource while `Slab<Node, 2>` does.
    let instances = crate::linear_instances::collect_instances(statements);
    let templates = crate::linear_instances::declared_structs(statements);
    loop {
        let mut grew = false;
        for statement in statements {
            // A variant's payload is held by the enum exactly as a field is held
            // by a struct, so an enum carrying a resource is one. Reading only
            // the structs left an option holding a file ordinary data, and the
            // obligation went in and did not come out.
            let (name, field_types): (&String, Vec<&Type>) =
                match &statement.node {
                    Statement::Struct(name, _, fields) => (
                        name,
                        fields.iter().map(|field| &field.field_type).collect(),
                    ),
                    Statement::Enum(name, _, variants) => (
                        name,
                        variants
                            .iter()
                            .filter_map(|variant| variant.fields.as_ref())
                            .flatten()
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
            if held.contains(name.as_str())
                || held.contains(Type::template_of(name))
            {
                continue;
            }
            let holds = field_types.iter().any(|ty| ty.is_linear_with(&held));
            if holds && held.insert(name.clone()) {
                grew = true;
            }
        }
        // In the same loop as the holders, since an instance is a resource
        // because of a field and a struct is one because of an instance in a
        // field of its own.
        if crate::linear_instances::note_linear_instances(
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
    parameters: Vec<Parameter>,
    return_sig: ReturnSignature,
    body: Block,
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
    fn as_argument(&self) -> Expression {
        match self {
            PackElement::Value(name, _) => Expression::Identifier(name.clone()),
            PackElement::Type(ty) => Expression::TypeValue(ty.clone()),
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

fn function_type_params(parameters: &[Parameter]) -> Vec<String> {
    let mut names = Vec::new();
    for parameter in parameters {
        collect_type_params(&parameter_type(parameter), &mut names);
    }
    names
}

fn function_is_generic(parameters: &[Parameter]) -> bool {
    !function_type_params(parameters).is_empty()
        || parameters.iter().any(|parameter| parameter.pack)
}

// The compile-time list a function takes, if it takes one. It is the last
// parameter, since what followed it at a call would have nothing to say which
// side of the list it belonged to.
fn pack_parameter(parameters: &[Parameter]) -> Option<&Parameter> {
    let found = parameters.iter().position(|parameter| parameter.pack)?;
    Some(&parameters[found])
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
    argument: &Expression,
) -> bool {
    if let Some(ty) = probed {
        return ty.is_copy();
    }
    matches!(
        argument,
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

fn is_type_parameter(parameter: &Parameter) -> bool {
    matches!(
        &parameter.type_annotation,
        Some(Type::TypeParam(name)) if name == &parameter.name
    )
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
        Type::ArrayGeneric(inner, size_param) => {
            let inner = substitute_type(inner, subst);
            match subst.get(size_param) {
                Some(Type::ConstUsize(size)) => {
                    Type::Array(Box::new(inner), *size)
                }
                _ => Type::ArrayGeneric(Box::new(inner), size_param.clone()),
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

fn infer_subst_into(
    pattern: &Type,
    concrete: &Type,
    type_params: &[String],
    subst: &mut HashMap<String, Type>,
) {
    match pattern {
        Type::TypeParam(name) => {
            subst
                .entry(name.clone())
                .or_insert_with(|| concrete.clone());
            return;
        }
        Type::Struct(name) if type_params.contains(name) => {
            subst
                .entry(name.clone())
                .or_insert_with(|| concrete.clone());
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

// The enum an inferred `.Variant` belongs to, taken from what the context
// expects. Without one there is nothing to infer from, which is what the
// message says rather than leaving a nameless enum to fail later.
fn name_inferred_variant(
    variant: &str,
    fields: &[(String, Expression)],
    expected: Option<&Type>,
) -> Result<Expression> {
    let Some(Type::Enum(name) | Type::Struct(name)) = expected else {
        bail!(
            "`.{variant}` takes its enum from what the context expects, and here there is nothing to take it from; write `Enum::{variant}`"
        );
    };
    Ok(Expression::EnumVariantInit(
        name.clone(),
        variant.to_string(),
        fields.to_vec(),
    ))
}

// The struct a `{ x = 1 }` builds, taken from what the context expects.
fn name_inferred_literal(
    fields: &[(String, Expression)],
    expected: Option<&Type>,
) -> Result<Expression> {
    let Some(Type::Struct(name) | Type::Enum(name)) = expected else {
        bail!(
            "a `{{ ... }}` literal takes its type from what the context expects, and here there is nothing to take it from; name the struct"
        );
    };
    Ok(Expression::StructInit(name.clone(), fields.to_vec()))
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
    value: &Expression,
    from: &Type,
    to: &Type,
    flags: &HashMap<String, FlagsLayout>,
) -> bool {
    let Type::Distinct(name, _) = to else {
        return false;
    };
    let literal = matches!(
        value,
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
    value: &Expression,
    to: &Type,
    flags: &HashMap<String, FlagsLayout>,
) -> (&'static str, &'static str) {
    let flagged =
        matches!(to, Type::Distinct(name, _) if flags.contains_key(name));
    if !flagged {
        return ("", "a distinct type is not its representation");
    }
    if matches!(
        value,
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
    value: &Expression,
    value_type: &Type,
    to: &Type,
    flags: &HashMap<String, FlagsLayout>,
) -> (String, String) {
    let (described, note) = nominal_reason(value, to, flags);
    let described = if described.is_empty() {
        format!("a '{value_type}'")
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

fn evaluate_bound(
    expression: &Expression,
    subst: &HashMap<String, Type>,
    linear: &HashSet<String>,
) -> Result<bool> {
    match expression {
        Expression::Infix(left, operator, right) => {
            let left = evaluate_bound(left, subst, linear)?;
            let right = evaluate_bound(right, subst, linear)?;
            match operator {
                crate::parser::Operator::And => Ok(left && right),
                crate::parser::Operator::Or => Ok(left || right),
                other => bail!(
                    "a `where` bound combines its terms with `&&`, `||` and `!`, and '{other}' is none of those"
                ),
            }
        }
        Expression::Prefix(crate::parser::Operator::Not, inner) => {
            Ok(!evaluate_bound(inner, subst, linear)?)
        }
        Expression::Call(callee, arguments) => {
            let Expression::Identifier(predicate) = callee.as_ref() else {
                bail!(
                    "a `where` bound is a predicate applied to a compile-time parameter"
                )
            };
            if arguments.len() != 1 {
                bail!(
                    "'{predicate}' takes one compile-time parameter, and {} were given",
                    arguments.len()
                )
            }
            let Expression::Identifier(parameter) = &arguments[0] else {
                bail!("'{predicate}' takes a compile-time parameter by name")
            };
            let Some(ty) = subst.get(parameter) else {
                bail!(
                    "the bound names '{parameter}', which is not a compile-time parameter of this function"
                )
            };
            match type_predicate(predicate, ty, linear) {
                Some(answer) => Ok(answer),
                None => bail!(
                    "'{predicate}' is not one of the bounds a type can be held to, which are: {BOUND_VOCABULARY}"
                ),
            }
        }
        other => bail!(
            "a `where` bound is a predicate applied to a compile-time parameter, and '{other}' is not one"
        ),
    }
}

// The bound, and what to say when it does not hold. The binding is named, since
// the reader chose it at the call and the template is not theirs.
fn check_bound(
    bound: &Expression,
    subst: &HashMap<String, Type>,
    callee: &str,
    linear: &HashSet<String>,
) -> Result<()> {
    if evaluate_bound(bound, subst, linear)? {
        return Ok(());
    }
    let mut bindings: Vec<String> = subst
        .iter()
        .map(|(name, ty)| format!("{name} = {ty}"))
        .collect();
    bindings.sort();
    bail!(
        "'{callee}' is declared `where {bound}`, and that does not hold for {}",
        bindings.join(", ")
    )
}

// A format literal, split into the text between its holes and the holes
// themselves. `{}` is a hole, and `{{` and `}}` are one brace each. A lone `}`
// is itself, since nothing else can be meant by it.
//
// `None` is a hole and `Some(text)` is text, so the count of holes is the count
// of arguments that must follow.
fn split_format(text: &str) -> Result<Vec<Option<String>>> {
    let mut pieces = Vec::new();
    let mut current = String::new();
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        if character == '}' {
            if characters.peek() == Some(&'}') {
                characters.next();
            }
            current.push('}');
            continue;
        }
        if character != '{' {
            current.push(character);
            continue;
        }
        if characters.peek() == Some(&'{') {
            characters.next();
            current.push('{');
            continue;
        }
        match characters.next() {
            Some('}') => {
                if !current.is_empty() {
                    pieces.push(Some(std::mem::take(&mut current)));
                }
                pieces.push(None);
            }
            _ => bail!(
                "a hole in a format is written `{{}}`, and `{{` on its own is written `{{{{`"
            ),
        }
    }
    if !current.is_empty() {
        pieces.push(Some(current));
    }
    Ok(pieces)
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
    body: Block,
    pack: Option<&(String, Vec<PackElement>)>,
    parameters: &[Parameter],
    context: ExpansionContext<'_>,
) -> Result<Block> {
    let ExpansionContext {
        structs,
        subst,
        linear,
    } = context;
    let mut types = HashMap::new();
    for parameter in parameters {
        if let Some(ty) = &parameter.type_annotation {
            types.insert(parameter.name.clone(), ty.clone());
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
    expansion.block(body)
}

impl Expansion<'_> {
    fn block(&self, block: Block) -> Result<Block> {
        let mut expanded = Vec::with_capacity(block.len());
        for statement in block {
            let position = statement.position;
            // A `for` over the pack, and an `if` whose condition is answered
            // here, both stand for several statements or none, so they are
            // spliced rather than replaced.
            // A `for` over a type's fields: the body is written once and
            // compiled once per field, with the loop's name standing for that
            // field. The list is the struct's own field list, so its length is
            // fixed by a declaration rather than by anything this walks.
            if let Statement::For(variable, None, iterable, body) =
                &statement.node
                && let Some(layout) = self.fields_named(iterable)
            {
                let fields: Vec<(usize, Type)> = layout
                    .fields
                    .iter()
                    .map(|field| (field.offset, field.ty.clone()))
                    .collect();
                for field in fields {
                    let bound = self.with_field(variable, field);
                    expanded.extend(bound.block(body.clone())?);
                }
                continue;
            }
            if let Statement::For(variable, None, iterable, body) =
                &statement.node
                && let Some(elements) = self.pack_named(iterable)
            {
                for element in elements {
                    // A value element is a parameter of this specialization, so
                    // the loop's name stands for that parameter. A type element
                    // is not a value at all: the loop's name is a type, and
                    // what the body wrote it in are type positions.
                    match element {
                        PackElement::Value(name, _) => {
                            let bound = substitute_identifier(
                                Block::from(body.clone()),
                                variable,
                                name,
                            );
                            expanded.extend(self.block(bound)?);
                        }
                        PackElement::Type(ty) => {
                            let one =
                                HashMap::from([(variable.clone(), ty.clone())]);
                            let bound = substitute_block(&body.clone(), &one);
                            let inner = self.with_type(variable, ty.clone());
                            expanded.extend(inner.block(bound)?);
                        }
                    }
                }
                continue;
            }
            if let Statement::Expression(Expression::If(
                condition,
                consequence,
                alternative,
            )) = &statement.node
                && let Some(taken) = self.answer(condition)?
            {
                let kept = if taken {
                    Some(consequence.clone())
                } else {
                    alternative.clone()
                };
                if let Some(kept) = kept {
                    expanded.extend(self.block(kept)?);
                }
                continue;
            }
            let node = self.statement(statement.node)?;
            expanded.push(Spanned { node, position });
        }
        Ok(expanded)
    }

    // The struct a `fields(...)` names, when this expression is one. The
    // argument is a type: a type parameter this specialization bound, or a
    // struct named outright.
    fn fields_named(&self, expression: &Expression) -> Option<&StructLayout> {
        let Expression::Call(callee, arguments) = expression else {
            return None;
        };
        let Expression::Identifier(named) = callee.as_ref() else {
            return None;
        };
        if named != "fields" || arguments.len() != 1 {
            return None;
        }
        self.structs.get(&self.named_type(&arguments[0])?)
    }

    // The name of the type an expression names, following the type arguments
    // this specialization was made for.
    fn named_type(&self, expression: &Expression) -> Option<String> {
        let named = match expression {
            Expression::Identifier(named) => named.clone(),
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
    fn field_named(&self, expression: &Expression) -> Option<&(usize, Type)> {
        match expression {
            Expression::Identifier(named) => self.fields.get(named),
            _ => None,
        }
    }

    // This expansion with one more field in force.
    // One argument of a `g(T) for T in list`: the template with the element's
    // name standing for that element, expanded as an ordinary expression.
    fn mapped(
        &self,
        element: &PackElement,
        variable: &str,
        body: &Expression,
    ) -> Result<Expression> {
        match element {
            PackElement::Value(name, _) => {
                let bound = substitute_identifier_in_expression(
                    body.clone(),
                    variable,
                    name,
                );
                self.expression(bound)
            }
            PackElement::Type(ty) => {
                let one = HashMap::from([(variable.to_string(), ty.clone())]);
                let bound = substitute_expression(body, &one);
                let inner = self.with_type(variable, ty.clone());
                inner.expression(bound)
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
    fn constant_call(&self, expression: &Expression) -> Result<Option<i64>> {
        let Expression::Call(callee, arguments) = expression else {
            return Ok(None);
        };
        let Expression::Identifier(named) = callee.as_ref() else {
            return Ok(None);
        };
        if arguments.len() != 1 {
            return Ok(None);
        }
        if named == "field_count"
            && let Some(name) = self.named_type(&arguments[0])
            && let Some(layout) = self.structs.get(&name)
        {
            return Ok(Some(layout.fields.len() as i64));
        }
        if named == "offset_of" {
            let Some((offset, _)) = self.field_named(&arguments[0]) else {
                bail!(
                    "offset_of names a field of a type, which is what a `for` over `fields(T)` binds"
                )
            };
            return Ok(Some(*offset as i64));
        }
        Ok(None)
    }

    // The elements of the pack, when this expression names it.
    fn pack_named(&self, expression: &Expression) -> Option<&Vec<PackElement>> {
        let (name, elements) = self.pack?;
        match expression {
            Expression::Identifier(named) if named == name => Some(elements),
            _ => None,
        }
    }

    // Whether a condition is one this can answer, and what it answers. `None`
    // means it is an ordinary condition and stays one.
    fn answer(&self, condition: &Expression) -> Result<Option<bool>> {
        match condition {
            Expression::Prefix(crate::parser::Operator::Not, inner) => {
                Ok(self.answer(inner)?.map(|held| !held))
            }
            Expression::Call(callee, arguments) => {
                let Expression::Identifier(predicate) = callee.as_ref() else {
                    return Ok(None);
                };
                if arguments.len() != 1 {
                    return Ok(None);
                }
                let Expression::Identifier(subject) = &arguments[0] else {
                    return Ok(None);
                };
                // A parameter of the specialization, or a field the `for`
                // around this bound. Both are types known here.
                let ty = match self.fields.get(subject) {
                    Some((_, ty)) => ty,
                    None => match self.types.get(subject) {
                        Some(ty) => ty,
                        None => return Ok(None),
                    },
                };
                Ok(type_predicate(predicate, ty, self.linear))
            }
            _ => Ok(None),
        }
    }

    fn statement(&self, statement: Statement) -> Result<Statement> {
        let expanded = match statement {
            Statement::Let {
                name,
                type_annotation,
                value,
                mutable,
            } => Statement::Let {
                name,
                type_annotation,
                value: self.expression(value)?,
                mutable,
            },
            Statement::Constant(name, value) => {
                Statement::Constant(name, self.expression(value)?)
            }
            Statement::Return(value) => {
                Statement::Return(self.expression(value)?)
            }
            Statement::Expression(value) => {
                Statement::Expression(self.expression(value)?)
            }
            Statement::Print(value, arguments) => {
                let mut expanded = Vec::with_capacity(arguments.len());
                for argument in arguments {
                    expanded.push(self.expression(argument)?);
                }
                Statement::Print(self.expression(value)?, expanded)
            }
            Statement::Assignment(place, value) => Statement::Assignment(
                self.expression(place)?,
                self.expression(value)?,
            ),
            Statement::Defer(inner) => {
                Statement::Defer(Box::new(self.statement(*inner)?))
            }
            Statement::For(variable, second, iterable, body) => Statement::For(
                variable,
                second,
                self.expression(iterable)?,
                self.block(body)?,
            ),
            Statement::While(condition, body) => {
                Statement::While(self.expression(condition)?, self.block(body)?)
            }
            Statement::With(capability, body) => {
                Statement::With(capability, self.block(body)?)
            }
            other => other,
        };
        Ok(expanded)
    }

    fn expression(&self, expression: Expression) -> Result<Expression> {
        // `offset_of(field)` and `field_count(T)` are numbers this works out
        // here, where the layout is known and nothing has been emitted yet.
        if let Some(value) = self.constant_call(&expression)? {
            return Ok(Expression::Literal(Literal::Integer(value)));
        }
        // `sizeof(field)` is the width of what that field holds. A field reads
        // as a named type to the parser, which is what makes this the place
        // that tells the two apart.
        if let Expression::Sizeof(Type::Struct(named)) = &expression
            && let Some((_, ty)) = self.fields.get(named)
        {
            return Ok(Expression::Sizeof(ty.clone()));
        }
        // The same for `type_id`, and for a name a `for` over a list of types
        // bound, which is a type here and nowhere else.
        if let Expression::TypeId(Type::Struct(named)) = &expression {
            if let Some((_, ty)) = self.fields.get(named) {
                return Ok(Expression::TypeId(ty.clone()));
            }
            if let Some(ty) = self.types.get(named) {
                return Ok(Expression::TypeId(ty.clone()));
            }
        }
        if let Expression::Sizeof(Type::Struct(named)) = &expression
            && let Some(ty) = self.types.get(named)
        {
            return Ok(Expression::Sizeof(ty.clone()));
        }
        // A type predicate is a question this answers wherever it is asked, not
        // only in the condition of an `if`, so a table may carry the answer as
        // an ordinary field.
        if let Some(held) = self.answer(&expression)? {
            return Ok(Expression::Boolean(held));
        }
        // A field is not a value. Naming one anywhere else is a mistake worth
        // catching here rather than as an unknown variable later.
        if let Expression::Identifier(named) = &expression
            && self.fields.contains_key(named)
        {
            bail!(
                "'{named}' is a field of a type, so it is asked about with `offset_of`, `sizeof` and the type predicates, and is not a value"
            )
        }
        // `pack[K]` is the Kth element, which is a parameter of this
        // specialization. Anything else that names the pack is an error: a
        // compile-time list is not a value.
        if let Expression::Index(base, index) = &expression
            && let Some(elements) = self.pack_named(base)
        {
            let Expression::Literal(Literal::Integer(at)) = index.as_ref()
            else {
                bail!(
                    "a compile-time list is indexed by a literal, since which element it is has to be known here"
                )
            };
            let Some(element) =
                usize::try_from(*at).ok().and_then(|at| elements.get(at))
            else {
                bail!(
                    "this call gave {} element(s) to the list, so there is no element {at}",
                    elements.len()
                )
            };
            return Ok(match element {
                PackElement::Value(name, _) => {
                    Expression::Identifier(name.clone())
                }
                PackElement::Type(ty) => Expression::TypeValue(ty.clone()),
            });
        }
        if self.pack_named(&expression).is_some() {
            bail!(
                "a compile-time list is iterated with `for` or indexed by a literal, and is not a value of its own"
            )
        }
        let expanded = match expression {
            Expression::Prefix(operator, inner) => {
                Expression::Prefix(operator, Box::new(self.expression(*inner)?))
            }
            Expression::Infix(left, operator, right) => Expression::Infix(
                Box::new(self.expression(*left)?),
                operator,
                Box::new(self.expression(*right)?),
            ),
            Expression::If(condition, consequence, alternative) => {
                Expression::If(
                    Box::new(self.expression(*condition)?),
                    self.block(consequence)?,
                    match alternative {
                        Some(block) => Some(self.block(block)?),
                        None => None,
                    },
                )
            }
            Expression::Call(callee, arguments) => {
                let mut expanded = Vec::with_capacity(arguments.len());
                for argument in arguments {
                    // An argument list is the one place a compile-time list
                    // stands for several things at once. Naming it hands over
                    // its elements, which is how one list is passed on to
                    // another. `g(T) for T in list` hands over the template
                    // once per element, which is how a call gets an arity the
                    // list decides.
                    if let Some(elements) = self.pack_named(&argument) {
                        for element in elements {
                            expanded.push(element.as_argument());
                        }
                        continue;
                    }
                    if let Expression::PackMap(body, variable, list) = &argument
                    {
                        let named = Expression::Identifier(list.clone());
                        let Some(elements) = self.pack_named(&named) else {
                            bail!(
                                "`for {variable} in {list}` walks a compile-time list, and '{list}' is not one here"
                            );
                        };
                        for element in elements.clone() {
                            expanded
                                .push(self.mapped(&element, variable, body)?);
                        }
                        continue;
                    }
                    expanded.push(self.expression(argument)?);
                }
                Expression::Call(Box::new(self.expression(*callee)?), expanded)
            }
            Expression::Index(base, index) => Expression::Index(
                Box::new(self.expression(*base)?),
                Box::new(self.expression(*index)?),
            ),
            Expression::FieldAccess(base, field) => Expression::FieldAccess(
                Box::new(self.expression(*base)?),
                field,
            ),
            Expression::AddressOf(inner) => {
                Expression::AddressOf(Box::new(self.expression(*inner)?))
            }
            Expression::Borrow(inner) => {
                Expression::Borrow(Box::new(self.expression(*inner)?))
            }
            Expression::BorrowMut(inner) => {
                Expression::BorrowMut(Box::new(self.expression(*inner)?))
            }
            Expression::Dereference(inner) => {
                Expression::Dereference(Box::new(self.expression(*inner)?))
            }
            Expression::Try(inner) => {
                Expression::Try(Box::new(self.expression(*inner)?))
            }
            Expression::Unsafe(body) => Expression::Unsafe(self.block(body)?),
            Expression::UnsafeFn(inner) => {
                Expression::UnsafeFn(Box::new(self.expression(*inner)?))
            }
            Expression::StructInit(name, fields) => {
                let mut expanded = Vec::with_capacity(fields.len());
                for (field, value) in fields {
                    expanded.push((field, self.expression(value)?));
                }
                Expression::StructInit(name, expanded)
            }
            Expression::EnumVariantInit(name, variant, fields) => {
                let mut expanded = Vec::with_capacity(fields.len());
                for (field, value) in fields {
                    expanded.push((field, self.expression(value)?));
                }
                Expression::EnumVariantInit(name, variant, expanded)
            }
            Expression::Tuple(items) => {
                let mut expanded = Vec::with_capacity(items.len());
                for item in items {
                    expanded.push(self.expression(item)?);
                }
                Expression::Tuple(expanded)
            }
            Expression::Range(start, end, inclusive) => Expression::Range(
                Box::new(self.expression(*start)?),
                Box::new(self.expression(*end)?),
                inclusive,
            ),
            Expression::Switch(scrutinee, cases) => {
                let mut expanded = Vec::with_capacity(cases.len());
                for case in cases {
                    expanded.push(crate::parser::SwitchCase {
                        pattern: case.pattern,
                        body: self.block(case.body)?,
                    });
                }
                Expression::Switch(
                    Box::new(self.expression(*scrutinee)?),
                    expanded,
                )
            }
            Expression::ArrayRepeat(value, count) => Expression::ArrayRepeat(
                Box::new(self.expression(*value)?),
                count,
            ),
            Expression::Literal(Literal::Array(elements)) => {
                let mut expanded = Vec::with_capacity(elements.len());
                for element in elements {
                    expanded.push(self.expression(element)?);
                }
                Expression::Literal(Literal::Array(expanded))
            }
            other => other,
        };
        Ok(expanded)
    }
}

// One name for another, through a block. This is what binds a `for` variable to
// the element the copy is for.
fn substitute_identifier(block: Block, from: &str, to: &str) -> Block {
    let mut subst = HashMap::new();
    subst.insert(from.to_string(), to.to_string());
    rename_block(block, &subst)
}

fn substitute_identifier_in_expression(
    expression: Expression,
    from: &str,
    to: &str,
) -> Expression {
    let mut subst = HashMap::new();
    subst.insert(from.to_string(), to.to_string());
    rename_expression(expression, &subst)
}

fn rename_block(block: Block, subst: &HashMap<String, String>) -> Block {
    block
        .into_iter()
        .map(|statement| Spanned {
            node: rename_statement(statement.node, subst),
            position: statement.position,
        })
        .collect()
}

fn rename_statement(
    statement: Statement,
    subst: &HashMap<String, String>,
) -> Statement {
    match statement {
        Statement::Let {
            name,
            type_annotation,
            value,
            mutable,
        } => Statement::Let {
            name,
            type_annotation,
            value: rename_expression(value, subst),
            mutable,
        },
        Statement::Constant(name, value) => {
            Statement::Constant(name, rename_expression(value, subst))
        }
        Statement::Return(value) => {
            Statement::Return(rename_expression(value, subst))
        }
        Statement::Expression(value) => {
            Statement::Expression(rename_expression(value, subst))
        }
        Statement::Print(value, arguments) => Statement::Print(
            rename_expression(value, subst),
            arguments
                .into_iter()
                .map(|argument| rename_expression(argument, subst))
                .collect(),
        ),
        Statement::Assignment(place, value) => Statement::Assignment(
            rename_expression(place, subst),
            rename_expression(value, subst),
        ),
        Statement::Defer(inner) => {
            Statement::Defer(Box::new(rename_statement(*inner, subst)))
        }
        Statement::For(variable, second, iterable, body) => Statement::For(
            variable,
            second,
            rename_expression(iterable, subst),
            rename_block(body, subst),
        ),
        Statement::While(condition, body) => Statement::While(
            rename_expression(condition, subst),
            rename_block(body, subst),
        ),
        Statement::With(capability, body) => {
            Statement::With(capability, rename_block(body, subst))
        }
        other => other,
    }
}

fn rename_expression(
    expression: Expression,
    subst: &HashMap<String, String>,
) -> Expression {
    match expression {
        Expression::Identifier(name) => match subst.get(&name) {
            Some(renamed) => Expression::Identifier(renamed.clone()),
            None => Expression::Identifier(name),
        },
        Expression::Prefix(operator, inner) => Expression::Prefix(
            operator,
            Box::new(rename_expression(*inner, subst)),
        ),
        Expression::Infix(left, operator, right) => Expression::Infix(
            Box::new(rename_expression(*left, subst)),
            operator,
            Box::new(rename_expression(*right, subst)),
        ),
        Expression::If(condition, consequence, alternative) => Expression::If(
            Box::new(rename_expression(*condition, subst)),
            rename_block(consequence, subst),
            alternative.map(|block| rename_block(block, subst)),
        ),
        Expression::Call(callee, arguments) => Expression::Call(
            Box::new(rename_expression(*callee, subst)),
            arguments
                .into_iter()
                .map(|argument| rename_expression(argument, subst))
                .collect(),
        ),
        Expression::Index(base, index) => Expression::Index(
            Box::new(rename_expression(*base, subst)),
            Box::new(rename_expression(*index, subst)),
        ),
        Expression::FieldAccess(base, field) => Expression::FieldAccess(
            Box::new(rename_expression(*base, subst)),
            field,
        ),
        Expression::AddressOf(inner) => {
            Expression::AddressOf(Box::new(rename_expression(*inner, subst)))
        }
        Expression::Borrow(inner) => {
            Expression::Borrow(Box::new(rename_expression(*inner, subst)))
        }
        Expression::BorrowMut(inner) => {
            Expression::BorrowMut(Box::new(rename_expression(*inner, subst)))
        }
        Expression::Dereference(inner) => {
            Expression::Dereference(Box::new(rename_expression(*inner, subst)))
        }
        Expression::Try(inner) => {
            Expression::Try(Box::new(rename_expression(*inner, subst)))
        }
        Expression::Unsafe(body) => {
            Expression::Unsafe(rename_block(body, subst))
        }
        Expression::UnsafeFn(inner) => {
            Expression::UnsafeFn(Box::new(rename_expression(*inner, subst)))
        }
        Expression::StructInit(name, fields) => Expression::StructInit(
            name,
            fields
                .into_iter()
                .map(|(field, value)| (field, rename_expression(value, subst)))
                .collect(),
        ),
        Expression::EnumVariantInit(name, variant, fields) => {
            Expression::EnumVariantInit(
                name,
                variant,
                fields
                    .into_iter()
                    .map(|(field, value)| {
                        (field, rename_expression(value, subst))
                    })
                    .collect(),
            )
        }
        Expression::Tuple(items) => Expression::Tuple(
            items
                .into_iter()
                .map(|item| rename_expression(item, subst))
                .collect(),
        ),
        Expression::Range(start, end, inclusive) => Expression::Range(
            Box::new(rename_expression(*start, subst)),
            Box::new(rename_expression(*end, subst)),
            inclusive,
        ),
        Expression::Switch(scrutinee, cases) => Expression::Switch(
            Box::new(rename_expression(*scrutinee, subst)),
            cases
                .into_iter()
                .map(|case| crate::parser::SwitchCase {
                    pattern: case.pattern,
                    body: rename_block(case.body, subst),
                })
                .collect(),
        ),
        Expression::ArrayRepeat(value, count) => Expression::ArrayRepeat(
            Box::new(rename_expression(*value, subst)),
            count,
        ),
        Expression::Literal(Literal::Array(elements)) => {
            Expression::Literal(Literal::Array(
                elements
                    .into_iter()
                    .map(|element| rename_expression(element, subst))
                    .collect(),
            ))
        }
        other => other,
    }
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

fn collect_instances_in_block(block: &Block, out: &mut Vec<String>) {
    for statement in block {
        collect_instances_in_statement(statement, out);
    }
}

fn collect_instances_in_statement(
    statement: &Statement,
    out: &mut Vec<String>,
) {
    match statement {
        Statement::Let {
            type_annotation,
            value,
            ..
        } => {
            if let Some(ty) = type_annotation {
                collect_instances_in_type(ty, out);
            }
            collect_instances_in_expression(value, out);
        }
        Statement::Return(expression) | Statement::Expression(expression) => {
            collect_instances_in_expression(expression, out);
        }
        Statement::Assignment(target, value) => {
            collect_instances_in_expression(target, out);
            collect_instances_in_expression(value, out);
        }
        Statement::For(_, _, range, body) => {
            collect_instances_in_expression(range, out);
            collect_instances_in_block(body, out);
        }
        Statement::While(condition, body) => {
            collect_instances_in_expression(condition, out);
            collect_instances_in_block(body, out);
        }
        Statement::Defer(inner) => {
            collect_instances_in_statement(inner, out);
        }
        // A constant whose value is not a function is that value wherever it is
        // named, so an instance it builds is asked for here. A function
        // constant's body is walked with its parameter types in scope instead.
        Statement::Constant(_, value)
            if !matches!(
                value,
                Expression::Function(..) | Expression::Proc(..)
            ) =>
        {
            collect_instances_in_expression(value, out);
        }
        _ => {}
    }
}

fn collect_instances_in_expression(
    expression: &Expression,
    out: &mut Vec<String>,
) {
    match expression {
        Expression::Sizeof(ty)
        | Expression::TypeId(ty)
        | Expression::TypeName(ty) => collect_instances_in_type(ty, out),
        Expression::Prefix(_, operand)
        | Expression::AddressOf(operand)
        | Expression::Borrow(operand)
        | Expression::BorrowMut(operand)
        | Expression::Dereference(operand) => {
            collect_instances_in_expression(operand, out);
        }
        Expression::Infix(left, _, right) => {
            collect_instances_in_expression(left, out);
            collect_instances_in_expression(right, out);
        }
        Expression::If(condition, consequence, alternative) => {
            collect_instances_in_expression(condition, out);
            collect_instances_in_block(consequence, out);
            if let Some(block) = alternative {
                collect_instances_in_block(block, out);
            }
        }
        Expression::Call(callee, arguments) => {
            collect_instances_in_expression(callee, out);
            for argument in arguments {
                collect_instances_in_expression(argument, out);
            }
        }
        Expression::Index(base, index) => {
            collect_instances_in_expression(base, out);
            collect_instances_in_expression(index, out);
        }
        Expression::FieldAccess(base, _) => {
            collect_instances_in_expression(base, out);
        }
        // A literal that says which instance it is asks for that instance, the
        // same as a type annotation naming one. Without this the only literals
        // that reached an instance were the ones a name elsewhere had already
        // built.
        Expression::StructInit(name, fields)
            if is_generic_instance(name) && !out.contains(name) =>
        {
            out.push(name.clone());
            for (_, value) in fields {
                collect_instances_in_expression(value, out);
            }
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => {
            for (_, value) in fields {
                collect_instances_in_expression(value, out);
            }
        }
        Expression::Range(start, end, _) => {
            collect_instances_in_expression(start, out);
            collect_instances_in_expression(end, out);
        }
        Expression::Tuple(elements) => {
            for element in elements {
                collect_instances_in_expression(element, out);
            }
        }
        Expression::Switch(scrutinee, cases) => {
            collect_instances_in_expression(scrutinee, out);
            for case in cases {
                collect_instances_in_block(&case.body, out);
            }
        }
        _ => {}
    }
}

struct Discovery<'a> {
    functions: &'a HashMap<String, GenericFunction>,
    structs: &'a HashMap<String, (Vec<String>, Vec<StructField>)>,
}

fn infer_struct_instance_shallow(
    struct_name: &str,
    field_inits: &[(String, Expression)],
    env: &HashMap<String, Type>,
    discovery: &Discovery,
) -> Option<String> {
    let (type_params, fields) = discovery.structs.get(struct_name)?;
    let mut subst: HashMap<String, Type> = HashMap::new();
    for (field_name, value) in field_inits {
        if let Some(field) =
            fields.iter().find(|field| &field.name == field_name)
            && let Some(value_type) =
                infer_expr_type_shallow(value, env, discovery)
        {
            infer_subst_into(
                &field.field_type,
                &value_type,
                type_params,
                &mut subst,
            );
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
    expression: &Expression,
    env: &HashMap<String, Type>,
    discovery: &Discovery,
) -> Option<Type> {
    match expression {
        Expression::Literal(Literal::Integer(_)) => Some(Type::I64),
        Expression::Literal(Literal::Float(_)) => Some(Type::F64),
        Expression::Literal(Literal::Float32(_)) => Some(Type::F32),
        Expression::Boolean(_) | Expression::Literal(Literal::Boolean(_)) => {
            Some(Type::Bool)
        }
        Expression::Identifier(name) => env.get(name).cloned(),
        Expression::StructInit(name, fields) => {
            if discovery.structs.contains_key(name) {
                infer_struct_instance_shallow(name, fields, env, discovery)
                    .map(Type::Struct)
            } else {
                Some(Type::Struct(name.clone()))
            }
        }
        Expression::EnumVariantInit(name, _, _) => {
            Some(Type::Enum(name.clone()))
        }
        Expression::Borrow(inner) => {
            infer_expr_type_shallow(inner, env, discovery)
                .map(|inner| Type::Ref(Box::new(inner)))
        }
        Expression::BorrowMut(inner) => {
            infer_expr_type_shallow(inner, env, discovery)
                .map(|inner| Type::RefMut(Box::new(inner)))
        }
        Expression::Call(callee, arguments) => {
            let Expression::Identifier(name) = callee.as_ref() else {
                return None;
            };
            let generic = discovery.functions.get(name)?;
            let subst = infer_call_subst(generic, arguments, env, discovery);
            generic
                .return_sig
                .to_type()
                .map(|ty| substitute_type(&ty, &subst))
        }
        _ => None,
    }
}

fn infer_call_subst(
    generic: &GenericFunction,
    arguments: &[Expression],
    env: &HashMap<String, Type>,
    discovery: &Discovery,
) -> HashMap<String, Type> {
    let mut subst = HashMap::new();
    for (parameter, argument) in generic.parameters.iter().zip(arguments) {
        if is_type_parameter(parameter)
            && let Expression::TypeValue(ty) = argument
        {
            subst.insert(parameter.name.clone(), ty.clone());
            continue;
        }
        if let Some(argument_type) =
            infer_expr_type_shallow(argument, env, discovery)
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
    block: &Block,
    env: &mut HashMap<String, Type>,
    discovery: &Discovery,
    out: &mut Vec<String>,
) {
    for statement in block {
        collect_call_instances_in_statement(statement, env, discovery, out);
    }
}

fn collect_call_instances_in_statement(
    statement: &Statement,
    env: &mut HashMap<String, Type>,
    discovery: &Discovery,
    out: &mut Vec<String>,
) {
    match statement {
        Statement::Let {
            name,
            type_annotation,
            value,
            ..
        } => {
            collect_call_instances_in_expression(value, env, discovery, out);
            let inferred = type_annotation
                .clone()
                .or_else(|| infer_expr_type_shallow(value, env, discovery));
            if let Some(ty) = inferred {
                env.insert(name.clone(), ty);
            }
        }
        Statement::Return(expression) | Statement::Expression(expression) => {
            collect_call_instances_in_expression(
                expression, env, discovery, out,
            );
        }
        Statement::Assignment(target, value) => {
            collect_call_instances_in_expression(target, env, discovery, out);
            collect_call_instances_in_expression(value, env, discovery, out);
        }
        Statement::For(variable, _, range, body) => {
            collect_call_instances_in_expression(range, env, discovery, out);
            env.insert(variable.clone(), Type::I64);
            collect_call_instances_in_block(body, env, discovery, out);
        }
        Statement::While(condition, body) => {
            collect_call_instances_in_expression(
                condition, env, discovery, out,
            );
            collect_call_instances_in_block(body, env, discovery, out);
        }
        Statement::Defer(inner) => {
            collect_call_instances_in_statement(inner, env, discovery, out);
        }
        _ => {}
    }
}

fn collect_call_instances_in_expression(
    expression: &Expression,
    env: &mut HashMap<String, Type>,
    discovery: &Discovery,
    out: &mut Vec<String>,
) {
    match expression {
        Expression::Call(callee, arguments) => {
            if let Expression::Identifier(name) = callee.as_ref()
                && let Some(generic) = discovery.functions.get(name)
            {
                let subst =
                    infer_call_subst(generic, arguments, env, discovery);
                if let Some(return_type) = generic.return_sig.to_type() {
                    collect_instances_in_type(
                        &substitute_type(&return_type, &subst),
                        out,
                    );
                }
                for parameter in &generic.parameters {
                    collect_instances_in_type(
                        &substitute_type(&parameter_type(parameter), &subst),
                        out,
                    );
                }
            }
            collect_call_instances_in_expression(callee, env, discovery, out);
            for argument in arguments {
                collect_call_instances_in_expression(
                    argument, env, discovery, out,
                );
            }
        }
        Expression::StructInit(name, fields) => {
            if discovery.structs.contains_key(name)
                && let Some(instance) =
                    infer_struct_instance_shallow(name, fields, env, discovery)
            {
                out.push(instance);
            }
            for (_, value) in fields {
                collect_call_instances_in_expression(
                    value, env, discovery, out,
                );
            }
        }
        Expression::Prefix(_, operand)
        | Expression::AddressOf(operand)
        | Expression::Borrow(operand)
        | Expression::BorrowMut(operand)
        | Expression::Dereference(operand) => {
            collect_call_instances_in_expression(operand, env, discovery, out);
        }
        Expression::Infix(left, _, right) => {
            collect_call_instances_in_expression(left, env, discovery, out);
            collect_call_instances_in_expression(right, env, discovery, out);
        }
        Expression::If(condition, consequence, alternative) => {
            collect_call_instances_in_expression(
                condition, env, discovery, out,
            );
            let mut branch_env = env.clone();
            collect_call_instances_in_block(
                consequence,
                &mut branch_env,
                discovery,
                out,
            );
            if let Some(block) = alternative {
                let mut branch_env = env.clone();
                collect_call_instances_in_block(
                    block,
                    &mut branch_env,
                    discovery,
                    out,
                );
            }
        }
        Expression::Index(base, index) => {
            collect_call_instances_in_expression(base, env, discovery, out);
            collect_call_instances_in_expression(index, env, discovery, out);
        }
        Expression::FieldAccess(base, _) => {
            collect_call_instances_in_expression(base, env, discovery, out);
        }
        Expression::EnumVariantInit(_, _, fields) => {
            for (_, value) in fields {
                collect_call_instances_in_expression(
                    value, env, discovery, out,
                );
            }
        }
        Expression::Switch(scrutinee, cases) => {
            collect_call_instances_in_expression(
                scrutinee, env, discovery, out,
            );
            for case in cases {
                let mut branch_env = env.clone();
                collect_call_instances_in_block(
                    &case.body,
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
    statements: &[Spanned<Statement>],
) -> Result<Vec<Statement>> {
    let mut generic_structs: HashMap<String, (Vec<String>, Vec<StructField>)> =
        HashMap::new();
    let mut generic_enums: HashMap<String, (Vec<String>, Vec<EnumVariant>)> =
        HashMap::new();
    for statement in statements {
        let statement = &statement.node;
        if let Statement::Struct(name, type_params, fields) = statement
            && !type_params.is_empty()
        {
            generic_structs
                .insert(name.clone(), (type_params.clone(), fields.clone()));
        }
        if let Statement::Enum(name, type_params, variants) = statement
            && !type_params.is_empty()
        {
            generic_enums
                .insert(name.clone(), (type_params.clone(), variants.clone()));
        }
    }
    // No early return on empty templates. A `columns<T, N>` is synthesized here
    // too and needs no user template, so the instance walk below must still run.

    let mut generic_functions: HashMap<String, GenericFunction> =
        HashMap::new();
    for statement in statements {
        let statement = &statement.node;
        if let Statement::Constant(
            name,
            Expression::Function(parameters, return_sig, body)
            | Expression::Proc(parameters, return_sig, body),
        ) = statement
            && function_is_generic(parameters)
        {
            generic_functions.insert(
                name.clone(),
                GenericFunction {
                    type_params: function_type_params(parameters),
                    parameters: parameters.clone(),
                    return_sig: return_sig.clone(),
                    body: body.clone(),
                },
            );
        }
    }

    let discovery = Discovery {
        functions: &generic_functions,
        structs: &generic_structs,
    };
    let mut queue: Vec<String> = Vec::new();
    for statement in statements {
        let statement = &statement.node;
        if let Statement::Constant(
            _,
            Expression::Function(parameters, _, body)
            | Expression::Proc(parameters, _, body),
        ) = statement
        {
            let mut env: HashMap<String, Type> = HashMap::new();
            for parameter in parameters {
                if let Some(ty) = &parameter.type_annotation {
                    env.insert(parameter.name.clone(), ty.clone());
                }
            }
            collect_call_instances_in_block(
                body, &mut env, &discovery, &mut queue,
            );
        }
        collect_instances_in_statement(statement, &mut queue);
        if let Statement::Struct(_, _, fields) = statement {
            for field in fields {
                collect_instances_in_type(&field.field_type, &mut queue);
            }
        }
        if let Statement::Enum(_, _, variants) = statement {
            for variant in variants {
                if let Some(fields) = &variant.fields {
                    for field in fields {
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
        } = statement
        {
            for parameter in params {
                if let Some(ty) = &parameter.type_annotation {
                    collect_instances_in_type(ty, &mut queue);
                }
            }
            if let Some(ty) = return_type {
                collect_instances_in_type(ty, &mut queue);
            }
        }
        if let Statement::Constant(
            _,
            Expression::Function(parameters, return_sig, body)
            | Expression::Proc(parameters, return_sig, body),
        ) = statement
        {
            for parameter in parameters {
                if let Some(ty) = &parameter.type_annotation {
                    collect_instances_in_type(ty, &mut queue);
                }
            }
            if let Some(ty) = return_sig.to_type() {
                collect_instances_in_type(&ty, &mut queue);
            }
            collect_instances_in_block(body, &mut queue);
        }
    }

    // The non-generic struct definitions, so a `columns<T, N>` can reflect over
    // T's fields to synthesize one array per field. T is required to be a plain
    // struct, the same restriction the self-hosted compiler has.
    let concrete_structs: HashMap<String, Vec<StructField>> = statements
        .iter()
        .filter_map(|statement| match &statement.node {
            Statement::Struct(name, type_params, fields)
                if type_params.is_empty() =>
            {
                Some((name.clone(), fields.clone()))
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
            let concrete_variants: Vec<EnumVariant> = variants
                .iter()
                .map(|variant| EnumVariant {
                    name: variant.name.clone(),
                    fields: variant.fields.as_ref().map(|fields| {
                        fields
                            .iter()
                            .map(|field| StructField {
                                name: field.name.clone(),
                                field_type: substitute_type(
                                    &field.field_type,
                                    &subst,
                                ),
                            })
                            .collect()
                    }),
                })
                .collect();
            for variant in &concrete_variants {
                if let Some(fields) = &variant.fields {
                    for field in fields {
                        collect_instances_in_type(
                            &field.field_type,
                            &mut queue,
                        );
                    }
                }
            }
            synthetic.push(Statement::Enum(
                instance.clone(),
                Vec::new(),
                concrete_variants,
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
            let mut columns_fields: Vec<StructField> = element_fields
                .iter()
                .map(|field| StructField {
                    name: field.name.clone(),
                    field_type: Type::Array(
                        Box::new(field.field_type.clone()),
                        count,
                    ),
                })
                .collect();
            columns_fields.push(StructField {
                name: "generations".to_string(),
                field_type: Type::Array(Box::new(Type::I64), count),
            });
            columns_fields.push(StructField {
                name: "free_list".to_string(),
                field_type: Type::Array(Box::new(Type::I64), count),
            });
            columns_fields.push(StructField {
                name: "free_count".to_string(),
                field_type: Type::I64,
            });
            for field in &columns_fields {
                collect_instances_in_type(&field.field_type, &mut queue);
            }
            synthetic.push(Statement::Struct(
                instance.clone(),
                Vec::new(),
                columns_fields,
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
        let concrete_fields: Vec<StructField> = fields
            .iter()
            .map(|field| StructField {
                name: field.name.clone(),
                field_type: substitute_type(&field.field_type, &subst),
            })
            .collect();
        for field in &concrete_fields {
            collect_instances_in_type(&field.field_type, &mut queue);
        }
        synthetic.push(Statement::Struct(
            instance.clone(),
            Vec::new(),
            concrete_fields,
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

fn substitute_block(block: &Block, subst: &HashMap<String, Type>) -> Block {
    block
        .iter()
        .map(|statement| {
            Spanned::new(
                substitute_statement(&statement.node, subst),
                statement.position,
            )
        })
        .collect()
}

fn substitute_statement(
    statement: &Statement,
    subst: &HashMap<String, Type>,
) -> Statement {
    match statement {
        Statement::Let {
            name,
            type_annotation,
            value,
            mutable,
        } => Statement::Let {
            name: name.clone(),
            type_annotation: type_annotation
                .as_ref()
                .map(|ty| substitute_type(ty, subst)),
            value: substitute_expression(value, subst),
            mutable: *mutable,
        },
        Statement::Return(expression) => {
            Statement::Return(substitute_expression(expression, subst))
        }
        Statement::Expression(expression) => {
            Statement::Expression(substitute_expression(expression, subst))
        }
        Statement::Assignment(target, value) => Statement::Assignment(
            substitute_expression(target, subst),
            substitute_expression(value, subst),
        ),
        Statement::For(variable, second, range, body) => Statement::For(
            variable.clone(),
            second.clone(),
            substitute_expression(range, subst),
            substitute_block(body, subst),
        ),
        Statement::While(condition, body) => Statement::While(
            substitute_expression(condition, subst),
            substitute_block(body, subst),
        ),
        Statement::Defer(inner) => {
            Statement::Defer(Box::new(substitute_statement(inner, subst)))
        }
        Statement::Print(value, arguments) => Statement::Print(
            substitute_expression(value, subst),
            arguments
                .iter()
                .map(|argument| substitute_expression(argument, subst))
                .collect(),
        ),
        Statement::Constant(name, value) => Statement::Constant(
            name.clone(),
            substitute_expression(value, subst),
        ),
        Statement::LetMultiple(bindings, value) => Statement::LetMultiple(
            bindings.clone(),
            substitute_expression(value, subst),
        ),
        Statement::With(name, body) => {
            Statement::With(name.clone(), substitute_block(body, subst))
        }
        other => other.clone(),
    }
}

fn substitute_expression(
    expression: &Expression,
    subst: &HashMap<String, Type>,
) -> Expression {
    // A call through a compile-time function parameter is a call to the
    // function that parameter was given. There is nothing left to dispatch on
    // by the time the specialized body is lowered: the comparator ends up
    // inlined into the loop rather than called through a pointer.
    // A value parameter stands for its integer everywhere the body names it, not
    // only in a type. `while (i < N)` has to mean the capacity.
    if let Expression::Identifier(name) = expression
        && let Some(Type::ConstUsize(value)) = subst.get(name)
    {
        return Expression::Literal(crate::parser::Literal::Integer(
            *value as i64,
        ));
    }
    if let Expression::Identifier(name) = expression
        && let Some(Type::ConstValue(target)) = subst.get(name)
    {
        return Expression::Identifier(target.clone());
    }
    if let Expression::Call(callee, arguments) = expression
        && let Expression::Identifier(name) = callee.as_ref()
        && let Some(Type::ConstFn(target)) = subst.get(name)
    {
        return Expression::Call(
            Box::new(Expression::Identifier(target.clone())),
            arguments
                .iter()
                .map(|argument| substitute_expression(argument, subst))
                .collect(),
        );
    }
    match expression {
        Expression::Prefix(operator, operand) => Expression::Prefix(
            *operator,
            Box::new(substitute_expression(operand, subst)),
        ),
        Expression::Infix(left, operator, right) => Expression::Infix(
            Box::new(substitute_expression(left, subst)),
            *operator,
            Box::new(substitute_expression(right, subst)),
        ),
        Expression::If(condition, consequence, alternative) => Expression::If(
            Box::new(substitute_expression(condition, subst)),
            substitute_block(consequence, subst),
            alternative
                .as_ref()
                .map(|block| substitute_block(block, subst)),
        ),
        Expression::Call(callee, arguments) => Expression::Call(
            Box::new(substitute_expression(callee, subst)),
            arguments
                .iter()
                .map(|argument| substitute_expression(argument, subst))
                .collect(),
        ),
        Expression::Index(base, index) => Expression::Index(
            Box::new(substitute_expression(base, subst)),
            Box::new(substitute_expression(index, subst)),
        ),
        Expression::FieldAccess(base, field) => Expression::FieldAccess(
            Box::new(substitute_expression(base, subst)),
            field.clone(),
        ),
        Expression::AddressOf(inner) => {
            Expression::AddressOf(Box::new(substitute_expression(inner, subst)))
        }
        Expression::Borrow(inner) => {
            Expression::Borrow(Box::new(substitute_expression(inner, subst)))
        }
        Expression::BorrowMut(inner) => {
            Expression::BorrowMut(Box::new(substitute_expression(inner, subst)))
        }
        Expression::Dereference(inner) => Expression::Dereference(Box::new(
            substitute_expression(inner, subst),
        )),
        Expression::StructInit(name, fields) => Expression::StructInit(
            name.clone(),
            fields
                .iter()
                .map(|(field, value)| {
                    (field.clone(), substitute_expression(value, subst))
                })
                .collect(),
        ),
        Expression::EnumVariantInit(name, variant, fields) => {
            Expression::EnumVariantInit(
                name.clone(),
                variant.clone(),
                fields
                    .iter()
                    .map(|(field, value)| {
                        (field.clone(), substitute_expression(value, subst))
                    })
                    .collect(),
            )
        }
        Expression::Sizeof(ty) => {
            Expression::Sizeof(substitute_type(ty, subst))
        }
        Expression::TypeName(ty) => {
            Expression::TypeName(substitute_type(ty, subst))
        }
        Expression::TypeId(ty) => {
            Expression::TypeId(substitute_type(ty, subst))
        }
        // `[value; N]` becomes the array it always meant, now that N is a
        // number. A count still unbound is one the enclosing generic passes on
        // to a further instantiation, so the form is carried along.
        Expression::ArrayRepeat(value, count) => {
            let value = substitute_expression(value, subst);
            match subst.get(count) {
                Some(Type::ConstUsize(size)) => {
                    Expression::Literal(Literal::Array(vec![value; *size]))
                }
                _ => Expression::ArrayRepeat(Box::new(value), count.clone()),
            }
        }
        // A compile-time argument handed on to another generic. Without this a
        // `$T` or a `$f` forwarded from one generic to the next arrived as the
        // parameter's own name rather than what it was bound to.
        Expression::TypeValue(ty) => {
            Expression::TypeValue(substitute_type(ty, subst))
        }
        Expression::Range(start, end, inclusive) => Expression::Range(
            Box::new(substitute_expression(start, subst)),
            Box::new(substitute_expression(end, subst)),
            *inclusive,
        ),
        Expression::Tuple(elements) => Expression::Tuple(
            elements
                .iter()
                .map(|element| substitute_expression(element, subst))
                .collect(),
        ),
        Expression::Switch(scrutinee, cases) => Expression::Switch(
            Box::new(substitute_expression(scrutinee, subst)),
            cases
                .iter()
                .map(|case| SwitchCase {
                    pattern: case.pattern.clone(),
                    body: substitute_block(&case.body, subst),
                })
                .collect(),
        ),
        // The body of an `unsafe` block is ordinary code, so a type parameter
        // used inside one substitutes the same as anywhere else. Missing this
        // left `sizeof(T)` reading as zero inside `unsafe { }`.
        Expression::Unsafe(body) => {
            Expression::Unsafe(substitute_block(body, subst))
        }
        Expression::UnsafeFn(inner) => {
            Expression::UnsafeFn(Box::new(substitute_expression(inner, subst)))
        }
        other => other.clone(),
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
fn is_place_expression(expression: &Expression) -> bool {
    matches!(
        expression,
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
    annotation: Option<&Type>,
    elements: &[Expression],
    signatures: &HashMap<String, FunctionSignature>,
) -> Type {
    match annotation {
        Some(Type::Array(inner, _)) | Some(Type::Slice(inner)) => {
            return (**inner).clone();
        }
        _ => {}
    }
    match elements.first() {
        Some(Expression::Literal(Literal::Integer(_))) => Type::I64,
        Some(Expression::Literal(Literal::Float(_))) => Type::F64,
        Some(Expression::Literal(Literal::Float32(_))) => Type::F32,
        Some(Expression::Literal(Literal::Boolean(_)))
        | Some(Expression::Boolean(_)) => Type::Bool,
        Some(Expression::StructInit(name, _)) => Type::Struct(name.clone()),
        Some(Expression::EnumVariantInit(name, _, _)) => {
            Type::Enum(name.clone())
        }
        Some(Expression::Identifier(name))
            if let Some(signature) = signatures.get(name) =>
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
            parameters.iter().map(parameter_type).collect(),
            Box::new(return_sig.to_type().unwrap_or(Type::Void)),
        ),
        Some(Expression::Literal(Literal::Array(inner))) => Type::Array(
            Box::new(array_element_type(None, inner, signatures)),
            inner.len(),
        ),
        _ => Type::I64,
    }
}

// A deferred statement is lowered again at every exit and its names are resolved
// there, so a name it mentions that is bound again after the `defer` reads as
// that later binding rather than the one in scope where the `defer` was written.
// Refused where it is written, because neither reading is the one the line has,
// and one of them is a binding the path taken never reached. The self-hosted
// compiler refuses the same programs, walking the deferred statement's tokens
// against the locals the function bound.
fn check_defer_names(
    deferred: &Statement,
    rest: &[Spanned<Statement>],
) -> Result<()> {
    let mut mentioned = Vec::new();
    crate::interface_names::names_in_statement(deferred, &mut mentioned);
    let mut rebound = HashSet::new();
    for statement in rest {
        crate::import_visibility::bound_in_statement(
            &statement.node,
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

fn compute_layouts(statements: &[Statement]) -> LayoutMaps {
    let struct_defs: Vec<(&String, &Vec<StructField>)> = statements
        .iter()
        .filter_map(|statement| match statement {
            Statement::Struct(name, _, fields) => Some((name, fields)),
            _ => None,
        })
        .collect();
    let enum_defs: Vec<(&String, &Vec<EnumVariant>)> = statements
        .iter()
        .filter_map(|statement| match statement {
            Statement::Enum(name, _, variants) => Some((name, variants)),
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
            if let Some(layout) = try_struct_layout(fields, &structs, &enums) {
                structs.insert((*name).clone(), layout);
                progress = true;
            }
        }
        for (name, variants) in &enum_defs {
            if enums.contains_key(*name) {
                continue;
            }
            if let Some(layout) = try_enum_layout(variants, &structs, &enums) {
                enums.insert((*name).clone(), layout);
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
    fields: &[StructField],
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
) -> Option<StructLayout> {
    let mut offset = 0;
    let mut align = 1;
    let mut field_layouts = Vec::with_capacity(fields.len());
    for field in fields {
        let (field_size, field_align) =
            size_and_align(&field.field_type, structs, enums)?;
        offset = round_up(offset, field_align);
        field_layouts.push(FieldLayout {
            name: field.name.clone(),
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
    variants: &[EnumVariant],
    structs: &HashMap<String, StructLayout>,
    enums: &HashMap<String, EnumLayout>,
) -> Option<EnumLayout> {
    let tag_size = 4;
    let mut payload_align = 1;
    for variant in variants {
        if let Some(fields) = &variant.fields {
            for field in fields {
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
        if let Some(fields) = &variant.fields {
            for field in fields {
                let (field_size, field_align) =
                    size_and_align(&field.field_type, structs, enums)?;
                offset = round_up(offset, field_align);
                field_layouts.push(FieldLayout {
                    name: field.name.clone(),
                    ty: field.field_type.clone(),
                    offset,
                });
                offset += field_size;
            }
        }
        max_end = max_end.max(offset);
        variant_layouts.push(EnumVariantLayout {
            name: variant.name.clone(),
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
    locals: Vec<IrLocal>,
    blocks: Vec<BlockUnderConstruction>,
    scopes: Vec<HashMap<String, LocalId>>,
    loops: Vec<LoopTargets>,
    current: BlockId,
    return_type: Type,
    active_defers: Vec<Statement>,
    current_position: Position,
    specializations: Vec<Specialization>,
    anonymous: Vec<AnonRequest>,
}

impl<'a> FunctionLowering<'a> {
    fn new(builder: &'a IrBuilder, return_type: Type) -> Self {
        let entry = BlockUnderConstruction {
            statements: Vec::new(),
            terminator: None,
        };
        FunctionLowering {
            builder,
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
        }
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
        block: &Block,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        self.push_scope();
        let mut result = (unit_operand(), Type::Void);
        for (index, statement) in block.iter().enumerate() {
            let is_last = index + 1 == block.len();
            let position = statement.position;
            self.current_position = position;
            if is_last
                && let Statement::Expression(expression) = &statement.node
            {
                result = locate(
                    self.lower_expression(expression, expected),
                    position,
                )?;
            } else {
                locate(self.lower_statement(&statement.node), position)?;
            }
        }
        self.pop_scope();
        Ok(result)
    }

    fn lower_body_with_defers(
        &mut self,
        body: &Block,
        return_type: &Type,
    ) -> Result<()> {
        let outer_defers = self.active_defers.len();
        self.push_scope();
        for (index, statement) in body.iter().enumerate() {
            let is_last = index + 1 == body.len();
            let position = statement.position;
            self.current_position = position;
            match &statement.node {
                Statement::Defer(inner) => {
                    locate(
                        check_defer_names(inner, &body[index + 1..]),
                        position,
                    )?;
                    self.active_defers.push((**inner).clone());
                }
                Statement::Expression(expression) if is_last => {
                    let (value, value_type) = locate(
                        self.lower_expression(expression, Some(return_type)),
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
                other => locate(self.lower_statement(other), position)?,
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
        self.run_active_defers()?;
        self.set_terminator(IrTerminator::Return(operand));
        Ok(())
    }

    fn run_active_defers(&mut self) -> Result<()> {
        let defers = self.active_defers.clone();
        for deferred in defers.iter().rev() {
            self.lower_statement(deferred)?;
        }
        Ok(())
    }

    fn lower_statement(&mut self, statement: &Statement) -> Result<()> {
        match statement {
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                if let Expression::StructInit(struct_name, field_inits) = value
                {
                    let layout_name = match type_annotation {
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
                            .contains_key(struct_name) =>
                        {
                            let Some(instance) = self
                                .generic_instance_of(struct_name, field_inits)
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
                    self.define_variable(name, local);
                    return Ok(());
                }
                if let Expression::Literal(Literal::Array(elements)) = value {
                    let element_type = array_element_type(
                        type_annotation.as_ref(),
                        elements,
                        &self.builder.signatures,
                    );
                    let ty = Type::Array(
                        Box::new(element_type.clone()),
                        elements.len(),
                    );
                    let local = self.fresh_local(ty, Some(name.clone()));
                    self.init_array(local, &element_type, elements)?;
                    self.define_variable(name, local);
                    return Ok(());
                }
                if let Expression::EnumVariantInit(
                    enum_name,
                    variant_name,
                    field_inits,
                ) = value
                {
                    // `o : Option<i64> = Option::Some { value = 3 }`: the
                    // annotation says which instance, the literal does not
                    // carry arguments, so the annotation is what names the
                    // layout. Same rule as a generic struct literal above.
                    let layout_name = match type_annotation {
                        // `c : Color = .Red`: the annotation is the only place
                        // the enum is named, which is the whole point of the
                        // leading dot.
                        Some(
                            Type::Enum(annotated) | Type::Struct(annotated),
                        ) if enum_name.is_empty() => annotated.clone(),
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
                    if layout_name.is_empty() {
                        bail!(
                            "`.{variant_name}` takes its enum from what the context expects, and this binding has no type to take it from; annotate it or write `Enum::{variant_name}`"
                        );
                    }
                    let ty = Type::Enum(layout_name.clone());
                    let local = self.fresh_local(ty, Some(name.clone()));
                    self.init_enum(
                        local,
                        &layout_name,
                        variant_name,
                        field_inits,
                    )?;
                    self.define_variable(name, local);
                    return Ok(());
                }
                let (operand, value_type) =
                    self.lower_expression(value, type_annotation.as_ref())?;
                if let Some(annotated) = type_annotation
                    && distinct_mismatch(
                        value,
                        &value_type,
                        annotated,
                        &self.builder.flags,
                    )
                {
                    let (described, note) = nominal_words(
                        value,
                        &value_type,
                        annotated,
                        &self.builder.flags,
                    );
                    bail!(
                        "this binding is a '{annotated}' and the value is {described}; {note}"
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
                            && matches!(value, Expression::Identifier(_)) =>
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
                let declared = match (type_annotation, &borrowed_aggregate) {
                    (Some(annotated), _) => annotated.clone(),
                    (None, Some(inner)) => inner.clone(),
                    (None, None) => value_type.clone(),
                };
                let coerced = self.coerce(operand, &value_type, &declared)?;
                let local =
                    self.fresh_local(declared.clone(), Some(name.clone()));
                self.emit(IrStatement::Assign(local, IrRvalue::Use(coerced)));
                self.define_variable(name, local);
                Ok(())
            }
            Statement::Constant(name, value) => {
                let (operand, value_type) =
                    self.lower_expression(value, None)?;
                let local = self.fresh_local(value_type, Some(name.clone()));
                self.emit(IrStatement::Assign(local, IrRvalue::Use(operand)));
                self.define_variable(name, local);
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
                        expression,
                        &value_type,
                        &return_type,
                        &self.builder.flags,
                    ) {
                        let (described, note) = nominal_words(
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
                    self.emit_return(Some(coerced))?;
                }
                Ok(())
            }
            Statement::Expression(expression) => {
                self.lower_expression(expression, None)?;
                Ok(())
            }
            Statement::Print(expression, arguments) => {
                self.lower_print(expression, arguments)
            }
            Statement::While(condition, body) => {
                self.lower_while(condition, body)
            }
            Statement::For(variable, second, range, body) => {
                self.lower_for(variable, second.as_deref(), range, body)
            }
            // Only the top level of a function body collects a `defer`, so one
            // reaching here is written inside a block. Named rather than left to
            // the catch-all below, which says a statement is unsupported and
            // gives a reader nothing to do about it.
            Statement::Defer(_) => {
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
            other => bail!("unsupported statement: {other}"),
        }
    }

    fn lower_while(
        &mut self,
        condition: &Expression,
        body: &Block,
    ) -> Result<()> {
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
        iterable: &Expression,
        body: &Block,
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
        let walked = Expression::Identifier(sequence_name);

        let (length, length_type) = match &sequence_type {
            Type::Array(_, count) => (
                IrOperand::Constant(IrConstant::Integer(
                    *count as i64,
                    Type::I64,
                )),
                Type::I64,
            ),
            Type::Str => {
                self.lower_str_len(std::slice::from_ref(&walked.clone()))?
            }
            _ => self.lower_slice_len(std::slice::from_ref(&walked.clone()))?,
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
            self.bind_sequence_element(item, &walked, index, &element)?;
        } else {
            self.bind_sequence_element(variable, &walked, index, &element)?;
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
        iterable: &Expression,
        index: LocalId,
        element: &Type,
    ) -> Result<()> {
        let indexed = Expression::Index(
            Box::new(iterable.clone()),
            Box::new(Expression::Identifier(format!("__for_index_{index}"))),
        );
        // The index is a local the loop owns rather than something the reader
        // wrote, so it is named into scope only for this lookup.
        self.define_variable(&format!("__for_index_{index}"), index);
        let (address, _) = self.element_address_of(&indexed)?;
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
        indexed: &Expression,
    ) -> Result<(IrOperand, Type)> {
        let Expression::Index(base, index) = indexed else {
            bail!("expected an index expression");
        };
        self.element_address(base, index)
    }

    fn lower_for(
        &mut self,
        variable: &str,
        second: Option<&str>,
        range: &Expression,
        body: &Block,
    ) -> Result<()> {
        let Expression::Range(start, end, inclusive) = range else {
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
        let compare = if *inclusive {
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

    fn lower_expression(
        &mut self,
        expression: &Expression,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        // `.Circle { radius = 5 }` and `{ x = 1, y = 2 }` name their type
        // nowhere, so the type the context expects is what says what they are.
        // Filling it in here covers every position that carries one: an
        // argument, a field, a return, an assignment and an element.
        let named;
        let expression = match expression {
            Expression::EnumVariantInit(name, variant, fields)
                if name.is_empty() =>
            {
                named = name_inferred_variant(variant, fields, expected)?;
                &named
            }
            Expression::StructInit(name, fields) if name.is_empty() => {
                named = name_inferred_literal(fields, expected)?;
                &named
            }
            other => other,
        };
        match expression {
            Expression::Literal(literal) => {
                self.lower_literal(literal, expected)
            }
            Expression::Boolean(value) => {
                Ok((IrOperand::Constant(IrConstant::Bool(*value)), Type::Bool))
            }
            Expression::Identifier(name) => {
                if let Some(local) = self.resolve_variable(name) {
                    if self.locals[local].linear {
                        self.emit(IrStatement::Consume(local));
                    }
                    return Ok((
                        IrOperand::Local(local),
                        self.type_of_local(local),
                    ));
                }
                if let Some(signature) = self.builder.signature(name) {
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
                if let Some(value) = self.builder.constants.get(name).cloned() {
                    return self.lower_expression(&value, expected);
                }
                bail!("unknown variable '{name}'");
            }
            Expression::Function(parameters, return_sig, body)
            | Expression::Proc(parameters, return_sig, body) => {
                let id = self.builder.anon_counter.get();
                self.builder.anon_counter.set(id + 1);
                let name = format!("__anon_{id}");
                let param_types: Vec<Type> =
                    parameters.iter().map(parameter_type).collect();
                let return_type = return_sig.to_type().unwrap_or(Type::Void);
                let proc_type = Type::Proc(param_types, Box::new(return_type));
                self.anonymous.push(AnonRequest {
                    name: name.clone(),
                    parameters: parameters.clone(),
                    return_sig: return_sig.clone(),
                    body: body.clone(),
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
                    operand.as_ref(),
                    Expression::Literal(Literal::Integer(_))
                ) =>
            {
                let Expression::Literal(Literal::Integer(value)) =
                    operand.as_ref()
                else {
                    unreachable!()
                };
                self.lower_literal(&Literal::Integer(-value), expected)
            }
            Expression::Prefix(operator, operand) => {
                self.lower_prefix(*operator, operand, expected)
            }
            Expression::Infix(left, operator, right) => {
                self.lower_infix(left, *operator, right, expected)
            }
            Expression::If(condition, consequence, alternative) => self
                .lower_if(
                    condition,
                    consequence,
                    alternative.as_ref(),
                    expected,
                ),
            Expression::Call(callee, arguments) => {
                if let Expression::Identifier(name) = callee.as_ref()
                    && name == "columns_new"
                    && self.resolve_variable(name).is_none()
                    && self.builder.signature(name).is_none()
                    && !self.builder.generic_functions.contains_key(name)
                {
                    return self.lower_columns_new(expected);
                }
                self.lower_call(callee, arguments)
            }
            Expression::Sizeof(ty) => {
                let size = self.builder.byte_size(ty) as i64;
                Ok((
                    IrOperand::Constant(IrConstant::Integer(size, Type::I64)),
                    Type::I64,
                ))
            }
            Expression::TypeId(ty) => {
                let id = self.builder.type_id(ty);
                Ok((
                    IrOperand::Constant(IrConstant::Integer(id, Type::I64)),
                    Type::I64,
                ))
            }
            Expression::TypeName(ty) => {
                let name =
                    crate::imports::demangle_private_names(&ty.to_string());
                if matches!(expected, Some(Type::Ptr(_))) {
                    return Ok((
                        IrOperand::Constant(IrConstant::CString(name)),
                        Type::Ptr(Box::new(Type::I8)),
                    ));
                }
                let local = self.fresh_local(Type::Str, None);
                self.build_str_value(local, &name);
                Ok((IrOperand::Local(local), Type::Str))
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
                self.lower_field_read(base, field)
            }
            Expression::Index(base, index) => {
                let (address, element_type) =
                    self.element_address(base, index)?;
                self.load_from(address, element_type)
            }
            Expression::Switch(scrutinee, cases) => {
                self.lower_match(scrutinee, cases, expected)
            }
            Expression::StructInit(struct_name, _) => {
                let ty = match expected {
                    Some(Type::Struct(instance))
                        if is_generic_instance(instance)
                            && instance
                                .starts_with(&format!("{struct_name}<")) =>
                    {
                        Type::Struct(instance.clone())
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
                if self.builder.flags.contains_key(type_name) =>
            {
                let layout = &self.builder.flags[type_name];
                if !fields.is_empty() {
                    bail!(
                        "'{type_name}::{bit}' is a bit of a set, so it carries nothing"
                    );
                }
                let Some(value) = layout.bits.get(bit).copied() else {
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
            Expression::EnumVariantInit(enum_name, _, _) => {
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
                    "'{count}' is not a constant or a value parameter, so there is no count for this array literal"
                )
            }
            other => {
                bail!("unsupported expression: {other}")
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
                let element = array_element_type(
                    expected,
                    elements,
                    &self.builder.signatures,
                );
                let ty = Type::Array(Box::new(element.clone()), elements.len());
                let temp = self.fresh_local(ty.clone(), None);
                self.init_array(temp, &element, elements)?;
                Ok((IrOperand::Local(temp), ty))
            }
        }
    }

    // `print`, in both of its forms. A string literal is a format: its holes
    // are filled by the arguments that follow, in order, and the whole line is
    // written as pieces with one newline at the end. Anything else is one
    // value, written the same way a hole is.
    //
    // The literal is read here, by the compiler, and never exists at run time.
    // There is no parser for it anywhere else and nothing walks it as data,
    // which is what keeps this from needing an evaluator of its own.
    fn lower_print(
        &mut self,
        first: &Expression,
        arguments: &[Expression],
    ) -> Result<()> {
        if let Expression::Literal(Literal::String(text)) = first {
            let pieces = split_format(text)?;
            let holes = pieces.iter().filter(|piece| piece.is_none()).count();
            if holes != arguments.len() {
                bail!(
                    "this format has {holes} hole(s) and {} argument(s)",
                    arguments.len()
                )
            }
            let mut next = arguments.iter();
            for piece in &pieces {
                match piece {
                    Some(literal) => self.write_bytes(literal),
                    None => {
                        let argument = next.next().expect("hole counted above");
                        self.write_value(argument)?;
                    }
                }
            }
            self.write_newline();
            return Ok(());
        }
        if !arguments.is_empty() {
            bail!(
                "a `print` with arguments starts with a format string, and this one does not"
            )
        }
        self.write_value(first)?;
        self.write_newline();
        Ok(())
    }

    fn write_bytes(&mut self, text: &str) {
        let sink = self.fresh_local(Type::Void, None);
        self.emit(IrStatement::Assign(
            sink,
            IrRvalue::Call {
                function: "frost_rt_write_bytes".to_string(),
                arguments: vec![
                    IrOperand::Constant(IrConstant::CString(text.to_string())),
                    IrOperand::Constant(IrConstant::Integer(
                        text.len() as i64,
                        Type::I64,
                    )),
                ],
            },
        ));
    }

    fn write_newline(&mut self) {
        let sink = self.fresh_local(Type::Void, None);
        self.emit(IrStatement::Assign(
            sink,
            IrRvalue::Call {
                function: "frost_rt_write_newline".to_string(),
                arguments: Vec::new(),
            },
        ));
    }

    // One value, written as what it is. An integer of any width widens to i64
    // and a float to f64, since that is what the two helpers take. A `str` is
    // its bytes, which is the one thing `print` could not do before.
    fn write_value(&mut self, expression: &Expression) -> Result<()> {
        if let Expression::Literal(Literal::String(text)) = expression {
            self.write_bytes(text);
            return Ok(());
        }
        let (operand, value_type) = self.lower_expression(expression, None)?;
        if matches!(value_type, Type::Str) {
            let base = self.str_operand_address(operand, &value_type)?;
            let data = self.str_field(
                base.clone(),
                STR_PTR_OFFSET,
                str_byte_ptr_type(),
            );
            let length = self.str_field(base, STR_LEN_OFFSET, Type::Usize);
            let length = self.coerce(length, &Type::Usize, &Type::I64)?;
            let sink = self.fresh_local(Type::Void, None);
            self.emit(IrStatement::Assign(
                sink,
                IrRvalue::Call {
                    function: "frost_rt_write_bytes".to_string(),
                    arguments: vec![data, length],
                },
            ));
            return Ok(());
        }
        // A `^i8` is a C string, whose length is where its NUL is. That is
        // what a string literal is when it reaches C, and what the self-hosted
        // compiler's own string literals are.
        if matches!(&value_type, Type::Ptr(inner) if **inner == Type::I8) {
            let sink = self.fresh_local(Type::Void, None);
            self.emit(IrStatement::Assign(
                sink,
                IrRvalue::Call {
                    function: "frost_rt_write_cstr".to_string(),
                    arguments: vec![operand],
                },
            ));
            return Ok(());
        }
        let (function, target) = if value_type.is_float() {
            ("frost_rt_write_f64", Type::F64)
        } else if value_type.is_integer()
            || matches!(value_type, Type::Bool | Type::Handle(_))
        {
            ("frost_rt_write_i64", Type::I64)
        } else {
            bail!(
                "there is no way to write a '{value_type}', so print what it holds instead"
            )
        };
        let coerced = self.coerce(operand, &value_type, &target)?;
        let sink = self.fresh_local(Type::Void, None);
        self.emit(IrStatement::Assign(
            sink,
            IrRvalue::Call {
                function: function.to_string(),
                arguments: vec![coerced],
            },
        ));
        Ok(())
    }

    // The address of a `str` value that has already been lowered. A local is
    // one. Anything else is spilled to one, since a str is read through its
    // address.
    fn str_operand_address(
        &mut self,
        operand: IrOperand,
        value_type: &Type,
    ) -> Result<IrOperand> {
        if let IrOperand::Local(local) = operand {
            return Ok(self.address_of_local(local, value_type));
        }
        bail!("a str to write is not a place")
    }

    fn lower_prefix(
        &mut self,
        operator: Operator,
        operand: &Expression,
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        match operator {
            Operator::Negate => {
                let (value, ty) = self.lower_expression(operand, expected)?;
                let result = self.fresh_local(ty.clone(), None);
                self.emit(IrStatement::Assign(
                    result,
                    IrRvalue::Unary(IrUnOp::Negate, value),
                ));
                Ok((IrOperand::Local(result), ty))
            }
            Operator::Not => {
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
        left: &Expression,
        operator: Operator,
        right: &Expression,
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
                if is_bare_number(left) && !is_bare_number(right) {
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
                        crate::imports::demangle_private_names(&name);
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

        let (left_operand, left_type) =
            self.lower_expression(left, expected)?;
        let (right_operand, right_type) =
            self.lower_expression(right, Some(&left_type))?;
        self.check_flags_operator(
            binop,
            (left, &left_type),
            (right, &right_type),
        )?;
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

    // A set of bits answers to union, intersection and whether it is the same
    // set. Adding two of them, or ordering them, or shifting one along, is a
    // question about the number underneath rather than about the set, and the
    // declaration exists to say that the number is not what this is. Reading
    // one as its representation is still allowed, so a program that means the
    // arithmetic writes the conversion and gets it.
    fn check_flags_operator(
        &self,
        binop: IrBinOp,
        left: (&Expression, &Type),
        right: (&Expression, &Type),
    ) -> Result<()> {
        let (left, left_type) = left;
        let (right, right_type) = right;
        let named = self
            .flags_name_of(left_type)
            .or_else(|| self.flags_name_of(right_type));
        let Some(name) = named else {
            return Ok(());
        };
        let readable = crate::imports::demangle_private_names(name);
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
        if matches!(left, Expression::Literal(Literal::Integer(_)))
            || matches!(right, Expression::Literal(Literal::Integer(_)))
        {
            bail!(
                "'{readable}' is a set of bits, built from the names declared under it, and a number is not one of them"
            );
        }
        // Two sets combine when they are the same set. Otherwise the answer
        // would be a number wearing one of the two names.
        if left_type != right_type {
            bail!(
                "'{readable}' combines only with itself, and this is a '{left_type}' against a '{right_type}'"
            );
        }
        Ok(())
    }

    fn lower_logical(
        &mut self,
        left: &Expression,
        operator: Operator,
        right: &Expression,
    ) -> Result<(IrOperand, Type)> {
        let result = self.fresh_local(Type::Bool, None);
        let (left_operand, _) =
            self.lower_expression(left, Some(&Type::Bool))?;

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
        let (right_operand, _) =
            self.lower_expression(right, Some(&Type::Bool))?;
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
        condition: &Expression,
        consequence: &Block,
        alternative: Option<&Block>,
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
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        let shape = self
            .builder
            .registrations
            .get(name)
            .cloned()
            .expect("a registration the caller just looked up");
        let mut rewritten = arguments.to_vec();
        let Some(handler) = rewritten.get(shape.handler) else {
            bail!(
                "'{name}' registers a callback and needs one as its argument {}",
                shape.handler + 1
            );
        };
        let Expression::TypeValue(Type::Struct(handler)) = handler else {
            bail!(
                "argument {} of '{name}' is the callback and has to be written '$name'",
                shape.handler + 1
            );
        };
        rewritten[shape.handler] = Expression::Identifier(handler.clone());
        let Some(context) = rewritten.get(shape.context).cloned() else {
            bail!(
                "'{name}' registers a callback and needs its context as argument {}",
                shape.context + 1
            );
        };
        rewritten[shape.context] = Expression::Call(
            Box::new(Expression::Identifier("ptr_to".to_string())),
            vec![context],
        );
        self.lower_direct_call(name, &rewritten)
    }

    fn lower_call(
        &mut self,
        callee: &Expression,
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if let Expression::Identifier(name) = callee
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
                let mut located = arguments.to_vec();
                located.push(Expression::Literal(Literal::String(
                    self.current_position.describe(),
                )));
                return self.lower_direct_call("frost_rt_assert_at", &located);
            }
            return self.lower_direct_call("frost_rt_assert", arguments);
        }
        if let Expression::Identifier(name) = callee
            && name == "str_len"
            && self.resolve_variable(name).is_none()
            && self.builder.signature(name).is_none()
            && !self.builder.generic_functions.contains_key(name)
        {
            return self.lower_str_len(arguments);
        }
        if let Expression::Identifier(name) = callee
            && name == "slice_len"
            && self.resolve_variable(name).is_none()
            && self.builder.signature(name).is_none()
            && !self.builder.generic_functions.contains_key(name)
        {
            return self.lower_slice_len(arguments);
        }
        if let Expression::Identifier(name) = callee
            && name == "flags_has"
            && self.resolve_variable(name).is_none()
            && self.builder.signature(name).is_none()
            && !self.builder.generic_functions.contains_key(name)
        {
            return self.lower_flags_has(arguments);
        }
        if let Expression::Identifier(name) = callee
            && self.resolve_variable(name).is_none()
            && self.builder.signature(name).is_none()
            && !self.builder.generic_functions.contains_key(name)
        {
            match name.as_str() {
                "ptr_to" => return self.lower_ptr_to(arguments),
                "cast" => return self.lower_cast(arguments),
                "ptr_cast" => return self.lower_ptr_cast(arguments),
                "slice_from" => return self.lower_slice_from(arguments),
                "wrap_add" => {
                    return self
                        .lower_wrapping(IrBinOp::WrappingAdd, arguments);
                }
                "wrap_sub" => {
                    return self
                        .lower_wrapping(IrBinOp::WrappingSubtract, arguments);
                }
                "wrap_mul" => {
                    return self
                        .lower_wrapping(IrBinOp::WrappingMultiply, arguments);
                }
                _ => {}
            }
        }
        if let Expression::Identifier(name) = callee
            && self.resolve_variable(name).is_none()
        {
            if self.builder.generic_functions.contains_key(name) {
                return self.lower_generic_call(name, arguments);
            }
            if self.builder.registrations.contains_key(name) {
                return self.lower_registration_call(name, arguments);
            }
            if self.builder.signature(name).is_some() {
                return self.lower_direct_call(name, arguments);
            }
        }
        if let Some(target) = self.bundle_field_function(callee) {
            return self.lower_direct_call(&target, arguments);
        }
        self.lower_indirect_call(callee, arguments)
    }

    // The function a bundle's field names, for a bundle that is a constant.
    // `ops.less(a, b)` where `ops` is a constant whose `less` field names a
    // function is a call to that function: there is one value the field can
    // hold and it is known here, so nothing is loaded and nothing is called
    // through a pointer.
    fn bundle_field_function(&self, callee: &Expression) -> Option<String> {
        let Expression::FieldAccess(base, field) = callee else {
            return None;
        };
        let Expression::Identifier(name) = base.as_ref() else {
            return None;
        };
        if self.resolve_variable(name).is_some() {
            return None;
        }
        let Some(Expression::StructInit(_, fields)) =
            self.builder.constants.get(name)
        else {
            return None;
        };
        let (_, value) =
            fields.iter().find(|(field_name, _)| field_name == field)?;
        let Expression::Identifier(target) = value else {
            return None;
        };
        self.builder
            .signature(target)
            .is_some()
            .then(|| target.clone())
    }

    fn lower_generic_call(
        &mut self,
        name: &str,
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        let generic = self
            .builder
            .generic_functions
            .get(name)
            .expect("generic function exists")
            .clone();

        // A compile-time list takes every argument past the parameters written
        // before it, so a call may give more arguments than there are
        // parameters, and one fewer when the list is empty.
        let packed = pack_parameter(&generic.parameters).is_some();
        let fixed = generic.parameters.len() - usize::from(packed);
        if (packed && arguments.len() < fixed)
            || (!packed && arguments.len() != generic.parameters.len())
        {
            bail!(
                "generic function '{name}' expects {} argument(s) but {} were given",
                generic.parameters.len(),
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
        for (index, (parameter, argument)) in
            generic.parameters.iter().zip(arguments).enumerate()
        {
            // The list is last, and what it took is lowered below. Nothing
            // after it is a parameter of its own.
            if parameter.pack {
                break;
            }
            if is_type_parameter(parameter) {
                let Expression::TypeValue(ty) = argument else {
                    bail!(
                        "type parameter '{}' of '{name}' requires a type argument like '${}'",
                        parameter.name,
                        parameter.name
                    );
                };
                // `$f` where f is a function rather than a type is a
                // compile-time function argument. It reads as a named type
                // here, so which one it is comes from whether the name is a
                // function this program declares.
                let bound = match ty {
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
                            parameter.name
                        );
                    }
                    other => other.clone(),
                };
                match parameter.compile_time_signature.as_ref() {
                    Some(Type::Proc(..)) => {
                        let Type::ConstFn(target) = &bound else {
                            bail!(
                                "'{}' of '{name}' is declared as a function, so it needs a function as its argument, not the type '{}'",
                                parameter.name,
                                bound
                            );
                        };
                        signature_checks.push((parameter, target.clone()));
                    }
                    Some(_) => {
                        let Type::ConstValue(target) = &bound else {
                            bail!(
                                "'{}' of '{name}' is declared as a bundle, so it needs a constant of that type as its argument, not '{}'",
                                parameter.name,
                                bound
                            );
                        };
                        bundle_checks.push((parameter, target.clone()));
                    }
                    None => {}
                }
                subst.insert(parameter.name.clone(), bound);
                continue;
            }
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
                        self.probe_type(argument).as_ref(),
                        argument,
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
                    argument,
                    Expression::Borrow(_)
                        | Expression::BorrowMut(_)
                        | Expression::AddressOf(_)
                ) || matches!(
                    self.probe_type(argument),
                    Some(Type::Ref(_) | Type::RefMut(_) | Type::Ptr(_))
                );
                if already_reference {
                    let (operand, value_type) =
                        self.lower_expression(argument, None)?;
                    infer_subst_into(
                        &param_ty,
                        &value_type,
                        &generic.type_params,
                        &mut subst,
                    );
                    plans.push(ArgPlan::Value(operand, value_type));
                } else {
                    if let Some(place_type) = self.probe_type(argument) {
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
                    self.lower_expression(argument, expected.as_ref())?;
                infer_subst_into(
                    &param_ty,
                    &value_type,
                    &generic.type_params,
                    &mut subst,
                );
                plans.push(ArgPlan::Value(operand, value_type));
            }
        }

        // The bound, before the body is specialized, so a type that cannot
        // work is refused here rather than inside code the reader never wrote.
        if let Some(bound) = &generic.return_sig.bound {
            check_bound(bound, &subst, name, &self.builder.linear)?;
        }

        for (parameter, target) in bundle_checks {
            let Some(declared) = parameter.compile_time_signature.as_ref()
            else {
                continue;
            };
            let expected = substitute_type(declared, &subst);
            let Some(Expression::StructInit(actual, _)) =
                self.builder.constants.get(&target)
            else {
                bail!(
                    "'{target}' given to '{name}' as '{}' is not a struct constant, and '{}' is declared as '{expected}'",
                    parameter.name,
                    parameter.name
                );
            };
            if Type::Struct(actual.clone()) != expected {
                bail!(
                    "'{target}' given to '{name}' as '{}' is a '{actual}', but '{}' is declared as '{expected}'",
                    parameter.name,
                    parameter.name
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
                    "'{}' given to '{name}' as '{}' has the signature '{actual}', but '{}' is declared as '{expected}'",
                    target,
                    parameter.name,
                    parameter.name
                );
            }
        }

        // Every argument the list took, lowered as a value. The specialization
        // takes one ordinary parameter for each, so each is evaluated once
        // however many times the unrolled body names it.
        let mut pack_elements: Vec<PackElement> = Vec::new();
        if let Some(parameter) = pack_parameter(&generic.parameters) {
            for (index, argument) in arguments[fixed..].iter().enumerate() {
                // `$Position` in the list is a type rather than a value. It
                // takes no parameter and is evaluated nowhere: what it leaves
                // behind is a name the body writes where a type belongs.
                if let Expression::TypeValue(ty) = argument {
                    pack_elements
                        .push(PackElement::Type(substitute_type(ty, &subst)));
                    continue;
                }
                let (operand, value_type) =
                    self.lower_expression(argument, None)?;
                pack_elements.push(PackElement::Value(
                    pack_element_name(&parameter.name, index),
                    value_type.clone(),
                ));
                plans.push(ArgPlan::Value(operand, value_type));
            }
        }

        let mut value_parameter_types: Vec<Type> = generic
            .parameters
            .iter()
            .filter(|parameter| {
                !is_type_parameter(parameter) && !parameter.pack
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
        let return_type = generic
            .return_sig
            .to_type()
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
            let templates: crate::linear_instances::Templates<'_> = self
                .builder
                .generic_struct_defs
                .iter()
                .map(|(held, (params, fields))| {
                    (held.as_str(), (params.as_slice(), fields.clone()))
                })
                .collect();
            for concrete in value_parameter_types
                .iter()
                .chain(std::iter::once(&return_type))
            {
                if let Some(report) =
                    crate::linear_instances::pooled_resource_in(
                        concrete,
                        &templates,
                        &self.builder.linear,
                    )
                {
                    bail!("linearity: {report}");
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
            pack: pack_parameter(&generic.parameters).map(|parameter| {
                (parameter.name.clone(), pack_elements.clone())
            }),
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
                                "this argument is a '{value_type}' with no storage, and '{name}' borrows it here"
                            );
                        };
                        lowered.push(self.address_of_local(local, inner));
                        continue;
                    }
                    // An array reaching a `[]T` parameter becomes a slice of
                    // the whole of itself first. Without this the callee is
                    // handed the array's own address and reads its first two
                    // elements as a pointer and a length.
                    if let (Type::Slice(element), Type::Array(held, count)) =
                        (target, &value_type)
                        && held == element
                    {
                        let IrOperand::Local(local) = operand else {
                            bail!(
                                "an array argument to a generic call is not a place"
                            );
                        };
                        let base = self.address_of_local(local, &value_type);
                        let slice = self
                            .build_slice_from_address(base, element, *count);
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
                            &arguments[index],
                            Some(&pointee),
                        )?;
                        let coerced =
                            self.coerce(operand, &value_type, &pointee)?;
                        lowered.push(coerced);
                        continue;
                    }
                    let address = self.aggregate_argument_address(
                        &arguments[index],
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
        arguments: &[Expression],
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
            // Auto-borrow. A `read`/`mut` parameter is a reference, and a plain
            // value place passed to it takes its address here. An argument that
            // is already a reference (a reference-typed local passed onward) or
            // an explicit borrow is left alone, so nothing is double-referenced.
            if let Some(reference @ (Type::Ref(inner) | Type::RefMut(inner))) =
                expected
            {
                let already_reference = matches!(
                    argument,
                    Expression::Borrow(_)
                        | Expression::BorrowMut(_)
                        | Expression::AddressOf(_)
                ) || self.probe_type(argument).as_ref()
                    == Some(reference);
                if !already_reference {
                    let pointee = (**inner).clone();
                    let address = self.aggregate_argument_address(
                        argument, &pointee, false,
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
                Some(Type::TypeParam(_)) => match self.probe_type(argument) {
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
                    self.aggregate_argument_address(argument, target, true)?;
                lowered.push(address);
                continue;
            }
            let (operand, value_type) =
                self.lower_expression(argument, expected)?;
            if let Some(Type::Ref(inner) | Type::RefMut(inner)) = expected
                && needs_memory(&value_type)
                && value_type == **inner
            {
                bail!(
                    "cannot pass a '{value_type}' by value to a reference parameter '&{value_type}'; take a reference with '&' or '&mut'"
                );
            }
            if let Some(target) = expected
                && distinct_mismatch(
                    argument,
                    &value_type,
                    target,
                    &self.builder.flags,
                )
            {
                let (described, note) = nominal_words(
                    argument,
                    &value_type,
                    target,
                    &self.builder.flags,
                );
                bail!(
                    "'{name}' takes a '{target}' here and this argument is {described}; {note}"
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
        callee: &Expression,
        arguments: &[Expression],
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
            // Auto-borrow. A `read`/`mut` parameter is a reference, and a plain
            // value place passed to it takes its address here. An argument that
            // is already a reference (a reference-typed local passed onward) or
            // an explicit borrow is left alone, so nothing is double-referenced.
            if let Some(reference @ (Type::Ref(inner) | Type::RefMut(inner))) =
                expected
            {
                let already_reference = matches!(
                    argument,
                    Expression::Borrow(_)
                        | Expression::BorrowMut(_)
                        | Expression::AddressOf(_)
                ) || self.probe_type(argument).as_ref()
                    == Some(reference);
                if !already_reference {
                    let pointee = (**inner).clone();
                    let address = self.aggregate_argument_address(
                        argument, &pointee, false,
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
                Some(Type::TypeParam(_)) => match self.probe_type(argument) {
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
                    self.aggregate_argument_address(argument, target, true)?;
                lowered.push(address);
                continue;
            }
            let (operand, value_type) =
                self.lower_expression(argument, expected)?;
            if let Some(Type::Ref(inner) | Type::RefMut(inner)) = expected
                && needs_memory(&value_type)
                && value_type == **inner
            {
                bail!(
                    "cannot pass a '{value_type}' by value to a reference parameter '&{value_type}'; take a reference with '&' or '&mut'"
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
        argument: &Expression,
        target: &Type,
        consume: bool,
    ) -> Result<IrOperand> {
        // Passing a `[N]T` array where a `[]T` slice is wanted. Build the slice
        // view and hand over its address, rather than the array's.
        if let Type::Slice(element) = target
            && let Some(Type::Array(array_element, count)) =
                self.probe_type(argument)
            && array_element == *element
        {
            // Any array place, not only a bare variable: a struct field (such as
            // a columns column `c.x`), an index, a deref. `probe_type` reads the
            // place chain's type and `place_address` walks it, so the slice
            // carries the right base and length instead of collapsing to a bare
            // pointer.
            let (base, _) = self.place_address(argument)?;
            let slice = self.build_slice_from_address(base, element, count);
            let IrOperand::Local(slice_local) = slice else {
                bail!("slice construction did not yield a place");
            };
            return Ok(self.address_of_local(slice_local, target));
        }
        match argument {
            Expression::Identifier(name) => {
                if consume
                    && let Some(local) = self.resolve_variable(name)
                    && self.locals[local].linear
                {
                    self.emit(IrStatement::Consume(local));
                }
                // A name already holding the address of the aggregate is that
                // address. A read parameter of struct type is one: it arrived
                // as a borrow, so what it holds is where the value is, and
                // taking its address again would hand over the address of the
                // pointer.
                if let Some(local) = self.resolve_variable(name)
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
        expression: &Expression,
    ) -> Result<()> {
        match expression {
            Expression::StructInit(name, fields) => {
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
            Expression::EnumVariantInit(name, variant, fields) => {
                // The local's type already names the instance when the context
                // resolved one, and that is the layout to write into. It is also
                // what names the enum of a `.Variant`, which names none itself.
                let layout_name = match self.type_of_local(local) {
                    Type::Enum(instance) | Type::Struct(instance)
                        if name.is_empty()
                            || is_generic_instance(&instance) =>
                    {
                        instance
                    }
                    _ => name.clone(),
                };
                self.init_enum(local, &layout_name, variant, fields)
            }
            Expression::Literal(Literal::Array(elements)) => {
                let Type::Array(element, _) = self.type_of_local(local) else {
                    bail!("array literal has non-array type");
                };
                self.init_array(local, &element, elements)
            }
            _ => {
                bail!("cannot materialize this aggregate")
            }
        }
    }

    fn lower_assignment(
        &mut self,
        target: &Expression,
        value: &Expression,
    ) -> Result<()> {
        if let Expression::Identifier(name) = target {
            let Some(local) = self.resolve_variable(name) else {
                bail!("assignment to unknown variable '{name}'");
            };
            let target_type = self.type_of_local(local);
            let (operand, value_type) =
                self.lower_expression(value, Some(&target_type))?;
            if distinct_mismatch(
                value,
                &value_type,
                &target_type,
                &self.builder.flags,
            ) {
                let (described, note) = nominal_words(
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
        if let Expression::Index(container, index_expr) = target
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
        if distinct_mismatch(value, &value_type, &pointee, &self.builder.flags)
        {
            let (described, note) = nominal_words(
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
        inner: &Expression,
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
        pointer: &Expression,
    ) -> Result<(IrOperand, Type)> {
        let (address, pointee) = self.place_address_of_deref(pointer)?;
        self.load_from(address, pointee)
    }

    fn lower_field_read(
        &mut self,
        base: &Expression,
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
    fn probe_type(&self, expression: &Expression) -> Option<Type> {
        match expression {
            Expression::Identifier(name) => self
                .resolve_variable(name)
                .map(|local| self.type_of_local(local)),
            Expression::Dereference(inner) => {
                deref_target(&self.probe_type(inner)?).ok()
            }
            Expression::FieldAccess(base, field) => {
                let base_type = self.probe_type(base)?;
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
                    .field(field)
                    .map(|field| field.ty.clone())
            }
            _ => None,
        }
    }

    fn raw_pointer_element_address(
        &mut self,
        base: &Expression,
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

    fn str_value_address(
        &mut self,
        expression: &Expression,
    ) -> Result<IrOperand> {
        if matches!(self.probe_type(expression), Some(Type::Str)) {
            let (address, _) = self.place_address(expression)?;
            return Ok(address);
        }
        let (operand, value_type) =
            self.lower_expression(expression, Some(&Type::Str))?;
        if value_type != Type::Str {
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
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 1 {
            bail!("str_len expects one argument");
        }
        let base = self.str_value_address(&arguments[0])?;
        let length = self.str_field(base, STR_LEN_OFFSET, Type::Usize);
        Ok((length, Type::Usize))
    }

    fn str_byte_address(
        &mut self,
        base: &Expression,
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

    fn slice_value_address(
        &mut self,
        expression: &Expression,
    ) -> Result<IrOperand> {
        // A slice that lives somewhere addressable, reached by any place chain:
        // a local, a struct field holding one, or a `mut` parameter, which
        // param-mode lowering turns into a deref of a pointer to the slice.
        // `place_address` walks all three, so recognizing the slice is what was
        // missing, not addressing it.
        if matches!(self.probe_type(expression), Some(Type::Slice(_))) {
            let (address, _) = self.place_address(expression)?;
            return Ok(address);
        }
        let (operand, value_type) = self.lower_expression(expression, None)?;
        let Type::Slice(_) = value_type else {
            bail!("expected a slice value, found {value_type}");
        };
        let IrOperand::Local(local) = operand else {
            bail!("slice value is not addressable");
        };
        self.mark_in_memory(local);
        Ok(self.address_of_local(local, &value_type))
    }

    fn slice_element_of(&self, base: &Expression) -> Option<Type> {
        match self.probe_type(base) {
            Some(Type::Slice(element)) => Some(*element),
            _ => None,
        }
    }

    fn slice_element_address(
        &mut self,
        base: &Expression,
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
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 1 {
            bail!("slice_len expects one argument");
        }
        let base = self.slice_value_address(&arguments[0])?;
        let length = self.str_field(base, SLICE_LEN_OFFSET, Type::Usize);
        Ok((length, Type::Usize))
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
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 2 {
            bail!(
                "flags_has takes the set and the bits to look for, as in 'flags_has(chosen, InitFlags::Video)'"
            );
        }
        let (set, set_type) = self.lower_expression(&arguments[0], None)?;
        let (wanted, wanted_type) =
            self.lower_expression(&arguments[1], Some(&set_type))?;
        if distinct_mismatch(
            &arguments[1],
            &wanted_type,
            &set_type,
            &self.builder.flags,
        ) {
            let (described, note) = nominal_words(
                &arguments[1],
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
            let readable = crate::imports::demangle_private_names(name);
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
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 1 {
            bail!("ptr_to expects one place argument");
        }
        let (address, pointee) = self.place_address(&arguments[0])?;
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
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 3 {
            bail!(
                "slice_from expects a type, a pointer and a length, as in slice_from($T, p, n)"
            );
        }
        let Expression::TypeValue(element) = &arguments[0] else {
            bail!("slice_from's first argument must be a type, as in $Entity");
        };
        let element = element.clone();
        let (pointer, _) = self.lower_expression(&arguments[1], None)?;
        let (length, length_type) =
            self.lower_expression(&arguments[2], None)?;
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
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 2 {
            bail!("this takes two numbers, as in wrap_mul(a, b)");
        }
        let (left, left_type) = self.lower_expression(&arguments[0], None)?;
        let (right, right_type) =
            self.lower_expression(&arguments[1], Some(&left_type))?;
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
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 2 {
            bail!("cast expects a type and a value, as in cast($u8, n)");
        }
        let Expression::TypeValue(target) = &arguments[0] else {
            bail!("cast's first argument must be a type, as in $u8");
        };
        let target = target.clone();
        let (value, from) = self.lower_expression(&arguments[1], None)?;
        if !is_numeric(&from) || !is_numeric(&target) {
            bail!(
                "cast converts between numbers, and this is asked to turn a {from} into a {target}"
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
        arguments: &[Expression],
    ) -> Result<(IrOperand, Type)> {
        if arguments.len() != 2 {
            bail!(
                "ptr_cast expects a type and a pointer, as in ptr_cast($T, p)"
            );
        }
        let Expression::TypeValue(target) = &arguments[0] else {
            bail!("ptr_cast's first argument must be a type, as in $Entity");
        };
        let target = Type::Ptr(Box::new(target.clone()));
        let (pointer, _) = self.lower_expression(&arguments[1], None)?;
        let result = self.fresh_local(target.clone(), None);
        self.emit(IrStatement::Assign(result, IrRvalue::Use(pointer)));
        Ok((IrOperand::Local(result), target))
    }

    fn place_address(
        &mut self,
        place: &Expression,
    ) -> Result<(IrOperand, Type)> {
        match place {
            Expression::Identifier(name) => {
                let Some(local) = self.resolve_variable(name) else {
                    // A constant has no storage of its own, so the address of
                    // one is the address of the copy built here. This is what a
                    // bundle passed at runtime, rather than as a compile-time
                    // argument, travels as.
                    if let Some(value) =
                        self.builder.constants.get(name).cloned()
                    {
                        // A constant naming a place is that place. One naming a
                        // value has none, so the copy built here is what the
                        // address is of: `fs_read(PATH)` with `PATH :: "x"` is
                        // this, and a string constant is the common case.
                        if is_place_expression(&value) {
                            return self.place_address(&value);
                        }
                        let (operand, ty) =
                            self.lower_expression(&value, None)?;
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
                    bail!("address of unknown variable '{name}'");
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
                self.field_address(base, field)
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
            other => {
                bail!("expression is not an assignable place: {other}")
            }
        }
    }

    fn element_address(
        &mut self,
        base: &Expression,
        index: &Expression,
    ) -> Result<(IrOperand, Type)> {
        // A constant is its value wherever it is named, and every question
        // below asks what the base is before it asks where it lives: whether it
        // is a string, a slice, a raw pointer. A name none of them can resolve
        // reaches the array path and comes back as an unknown variable, so the
        // value goes in ahead of them and they see a string literal rather than
        // a name. Before the index is lowered, so it is lowered once.
        if let Expression::Identifier(name) = base
            && self.resolve_variable(name).is_none()
            && let Some(value) = self.builder.constants.get(name).cloned()
        {
            return self.element_address(&value, index);
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
            || matches!(base, Expression::Literal(Literal::String(_)))
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
        let (base_pointer, element_type, length) = if !is_place_expression(base)
        {
            let (value, value_type) = self.lower_expression(base, None)?;
            let IrOperand::Local(local) = value else {
                bail!("cannot index into: {base}");
            };
            match value_type {
                Type::Slice(element) => {
                    self.mark_in_memory(local);
                    let slice_address = self
                        .address_of_local(local, &Type::Slice(element.clone()));
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
                _ => bail!("cannot index into: {base}"),
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
    fn slab_shaped_base(&self, base: &Expression) -> Option<String> {
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
        base: &Expression,
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
    fn columns_shaped_base(&self, base: &Expression) -> Option<String> {
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
        base: &Expression,
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
        base: &Expression,
        struct_name: &str,
        handle: IrOperand,
        value: &Expression,
    ) -> Result<()> {
        let column_fields: Vec<String> = {
            let layout =
                self.builder.struct_layout(struct_name).ok_or_else(|| {
                    anyhow::anyhow!("unknown columns '{struct_name}'")
                })?;
            layout
                .fields
                .iter()
                .filter(|field| {
                    !matches!(
                        field.name.as_str(),
                        "generations" | "free_list" | "free_count"
                    )
                })
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
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        let Some(Type::Struct(name)) = expected else {
            bail!(
                "columns_new() needs a columns type from its context, e.g. `mut c : columns<T, N> = columns_new()`"
            );
        };
        if !name.starts_with("columns<") {
            bail!("columns_new() initializes a columns type, not '{name}'");
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

    fn array_base_pointer(
        &mut self,
        base: &Expression,
    ) -> Result<(IrOperand, Type, Option<usize>)> {
        match base {
            Expression::Identifier(name) => {
                let Some(local) = self.resolve_variable(name) else {
                    bail!("unknown variable '{name}'");
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
                    other => bail!("'{name}' is not an array (found {other})"),
                }
            }
            Expression::FieldAccess(inner, field) => {
                let (address, field_type) = self.field_address(inner, field)?;
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
                    pointer_type
                else {
                    bail!("cannot index into: {base}");
                };
                let Type::Array(element, count) = *inner else {
                    bail!("cannot index into: {base}");
                };
                Ok((operand, *element, Some(count)))
            }
            Expression::Index(inner, index) => {
                let (address, element_type) =
                    self.element_address(inner, index)?;
                let Type::Array(element, count) = element_type else {
                    bail!("indexed value is not an array");
                };
                Ok((address, *element, Some(count)))
            }
            other => {
                bail!("cannot index into: {other}")
            }
        }
    }

    fn init_array(
        &mut self,
        local: LocalId,
        element_type: &Type,
        elements: &[Expression],
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
                    self.aggregate_field_source(element, element_type)?;
                self.emit(IrStatement::Copy {
                    destination: IrOperand::Local(address),
                    source,
                    size: element_size,
                });
            } else {
                let (operand, value_type) =
                    self.lower_expression(element, Some(element_type))?;
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
        pointer: &Expression,
    ) -> Result<(IrOperand, Type)> {
        let (pointer_operand, pointer_type) =
            self.lower_expression(pointer, None)?;
        let pointee = deref_target(&pointer_type)?;
        Ok((pointer_operand, pointee))
    }

    fn field_address(
        &mut self,
        base: &Expression,
        field: &str,
    ) -> Result<(IrOperand, Type)> {
        // A columns element field `c[h].field`: the field selects a column, the
        // handle a validated slot in it.
        if let Expression::Index(container, index_expr) = base
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
        let layout = self
            .builder
            .struct_layout(&struct_name)
            .ok_or_else(|| anyhow::anyhow!("unknown struct '{struct_name}'"))?;
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

    fn struct_place(
        &mut self,
        base: &Expression,
    ) -> Result<(IrOperand, String)> {
        match base {
            Expression::Identifier(name) => {
                let Some(local) = self.resolve_variable(name) else {
                    // A top-level constant is its value wherever it is named,
                    // so a field of one is a field of that value.
                    if let Some(value) =
                        self.builder.constants.get(name).cloned()
                    {
                        return self.struct_place(&value);
                    }
                    bail!("unknown variable '{name}'");
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
                let (address, field_type) = self.field_address(inner, field)?;
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
            other => {
                let (operand, ty) = self.lower_expression(other, None)?;
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
                            bail!("not a struct place: {other}");
                        };
                        self.mark_in_memory(local);
                        let address = self.address_of_local(
                            local,
                            &Type::Struct(struct_name.clone()),
                        );
                        Ok((address, struct_name))
                    }
                    _ => bail!("not a struct place: {other}"),
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
        field_inits: &[(String, Expression)],
    ) -> Result<()> {
        self.mark_owned(local);
        let fields: Vec<(String, usize, Type)> = {
            let layout =
                self.builder.struct_layout(struct_name).ok_or_else(|| {
                    anyhow::anyhow!("unknown struct '{struct_name}'")
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
            .filter(|name| !field_inits.iter().any(|(given, _)| given == name))
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

        for (field_name, field_value) in field_inits {
            let Some((_, offset, field_type)) =
                fields.iter().find(|(name, _, _)| name == field_name)
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
                    self.aggregate_field_source(field_value, field_type)?;
                self.emit(IrStatement::Copy {
                    destination: IrOperand::Local(address),
                    source,
                    size: self.builder.byte_size(field_type),
                });
            } else {
                let (operand, value_type) =
                    self.lower_expression(field_value, Some(field_type))?;
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
        expression: &Expression,
        field_type: &Type,
    ) -> Result<IrOperand> {
        match expression {
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
        field_inits: &[(String, Expression)],
    ) -> Result<()> {
        self.mark_owned(local);
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
            .filter(|name| !field_inits.iter().any(|(given, _)| given == name))
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

        for (field_name, field_value) in field_inits {
            let Some((_, offset, field_type)) =
                fields.iter().find(|(name, _, _)| name == field_name)
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
                    self.aggregate_field_source(field_value, field_type)?;
                self.emit(IrStatement::Copy {
                    destination: IrOperand::Local(address),
                    source,
                    size: self.builder.byte_size(field_type),
                });
            } else {
                let (operand, value_type) =
                    self.lower_expression(field_value, Some(field_type))?;
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
            matches!(case.pattern, Pattern::Wildcard | Pattern::Identifier(_))
        });
        if catches_rest {
            return Ok(());
        }
        let Some(layout) = self.builder.enum_layout(enum_name) else {
            return Ok(());
        };
        let covered = cases
            .iter()
            .filter_map(|case| match &case.pattern {
                Pattern::EnumVariant { variant_name, .. } => {
                    Some(variant_name.as_str())
                }
                _ => None,
            })
            .collect::<HashSet<_>>();
        let missing = layout
            .variants
            .iter()
            .map(|variant| variant.name.as_str())
            .filter(|name| !covered.contains(name))
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return Ok(());
        }
        let named = missing
            .iter()
            .map(|name| format!("'.{name}'"))
            .collect::<Vec<_>>()
            .join(", ");
        let readable = crate::imports::demangle_private_names(enum_name);
        bail!(
            "match on '{readable}' does not cover {named}; add the case or a `case _` for the rest"
        )
    }

    fn lower_match(
        &mut self,
        scrutinee: &Expression,
        cases: &[SwitchCase],
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        if cases.is_empty() {
            bail!("match with no cases");
        }

        if let Expression::Tuple(elements) = scrutinee {
            return self.lower_tuple_match(elements, cases, expected);
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
            self.check_exhaustive(name, cases)?;
        }

        // Taking a linear value apart is consuming it: every arm names what it
        // held, and what the arm does with those is the arm's obligation. This
        // is what lets a fallible function hand back a resource, since the
        // result carrying one is linear too.
        let consumed = self.linear_scrutinee(scrutinee, &scalar);

        let merge = self.new_block();
        let mut result_local: Option<LocalId> = None;
        let mut result_type = Type::Void;

        for case in cases {
            let case_block = self.new_block();
            let next_block = self.new_block();

            match &case.pattern {
                Pattern::Wildcard | Pattern::Identifier(_) => {
                    self.set_terminator(IrTerminator::Jump(case_block));
                }
                Pattern::Literal(literal) => {
                    let Some((value, value_type)) = &scalar else {
                        bail!("literal pattern requires a scalar match value");
                    };
                    let (literal_operand, _) =
                        self.lower_literal(literal, Some(value_type))?;
                    let condition = self.fresh_local(Type::Bool, None);
                    self.emit(IrStatement::Assign(
                        condition,
                        IrRvalue::Binary(
                            IrBinOp::Equal,
                            value.clone(),
                            literal_operand,
                        ),
                    ));
                    self.set_terminator(IrTerminator::Branch {
                        condition: IrOperand::Local(condition),
                        then_block: case_block,
                        else_block: next_block,
                    });
                }
                Pattern::EnumVariant { variant_name, .. } => {
                    let Some(tag) = &tag_operand else {
                        bail!(
                            "enum variant pattern requires an enum match value"
                        );
                    };
                    let enum_name = enum_name.as_ref().unwrap();
                    let variant_tag = self
                        .builder
                        .enum_layout(enum_name)
                        .and_then(|layout| layout.variant(variant_name))
                        .map(|variant| variant.tag)
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "enum '{enum_name}' has no variant '{variant_name}'"
                            )
                        })?;
                    let condition = self.fresh_local(Type::Bool, None);
                    self.emit(IrStatement::Assign(
                        condition,
                        IrRvalue::Binary(
                            IrBinOp::Equal,
                            tag.clone(),
                            IrOperand::Constant(IrConstant::Integer(
                                variant_tag as i64,
                                Type::I32,
                            )),
                        ),
                    ));
                    self.set_terminator(IrTerminator::Branch {
                        condition: IrOperand::Local(condition),
                        then_block: case_block,
                        else_block: next_block,
                    });
                }
                // A tuple pattern matches a tuple, which `lower_tuple_match`
                // above has already taken. Reaching here means the pattern has
                // parts and the value being matched does not, so this is a
                // mismatch to report rather than a feature to miss.
                Pattern::Tuple(patterns) => {
                    let parts = patterns.len();
                    let described = match (&enum_name, &scalar) {
                        (Some(name), _) => {
                            crate::imports::demangle_private_names(name)
                        }
                        (None, Some((_, ty))) => ty.to_string(),
                        (None, None) => "the matched value".to_string(),
                    };
                    bail!(
                        "a `case` of {parts} parts matches a tuple, and this match is on '{described}', which has none; match on `(a, b)` to compare several values at once"
                    );
                }
            }

            self.switch_to(case_block);
            if let Some(local) = consumed {
                self.emit(IrStatement::Consume(local));
            }
            self.push_scope();
            self.bind_pattern(
                &case.pattern,
                enum_address.as_ref(),
                enum_name.as_deref(),
                scalar.as_ref(),
            )?;
            let (value, value_type) = self.lower_block(&case.body, expected)?;
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
        scrutinee: &Expression,
    ) -> Result<Option<(String, IrOperand)>> {
        if matches!(
            scrutinee,
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
        let Expression::Identifier(name) = scrutinee else {
            return Ok(None);
        };
        let Some(local) = self.resolve_variable(name) else {
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
        elements: &[Expression],
        cases: &[SwitchCase],
        expected: Option<&Type>,
    ) -> Result<(IrOperand, Type)> {
        let mut values = Vec::with_capacity(elements.len());
        for element in elements {
            values.push(self.lower_expression(element, None)?);
        }

        let merge = self.new_block();
        let mut result_local: Option<LocalId> = None;
        let mut result_type = Type::Void;

        for case in cases {
            let case_block = self.new_block();
            let next_block = self.new_block();

            let patterns: Vec<&Pattern> = match &case.pattern {
                Pattern::Tuple(patterns) => patterns.iter().collect(),
                Pattern::Wildcard | Pattern::Identifier(_) => Vec::new(),
                other => bail!("unsupported tuple match pattern: {other:?}"),
            };

            let mut condition: Option<LocalId> = None;
            for (pattern, (value, value_type)) in
                patterns.iter().zip(values.iter())
            {
                if let Pattern::Literal(literal) = pattern {
                    let (literal_operand, _) =
                        self.lower_literal(literal, Some(value_type))?;
                    let test = self.fresh_local(Type::Bool, None);
                    self.emit(IrStatement::Assign(
                        test,
                        IrRvalue::Binary(
                            IrBinOp::Equal,
                            value.clone(),
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
            for (pattern, (value, value_type)) in
                patterns.iter().zip(values.iter())
            {
                if let Pattern::Identifier(name) = pattern {
                    let bound = self
                        .fresh_local(value_type.clone(), Some(name.clone()));
                    self.emit(IrStatement::Assign(
                        bound,
                        IrRvalue::Use(value.clone()),
                    ));
                    self.define_variable(name, bound);
                }
            }
            let (value, value_type) = self.lower_block(&case.body, expected)?;
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
        scrutinee: &Expression,
        scalar: &Option<(IrOperand, Type)>,
    ) -> Option<LocalId> {
        if let Expression::Identifier(name) = scrutinee
            && let Some(local) = self.resolve_variable(name)
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
        pattern: &Pattern,
        enum_address: Option<&IrOperand>,
        enum_name: Option<&str>,
        scalar: Option<&(IrOperand, Type)>,
    ) -> Result<()> {
        match pattern {
            Pattern::EnumVariant {
                variant_name,
                bindings,
                ..
            } => {
                let (Some(address), Some(enum_name)) =
                    (enum_address, enum_name)
                else {
                    bail!("enum pattern on a non-enum match value");
                };
                let fields: Vec<(String, usize, Type)> = self
                    .builder
                    .enum_layout(enum_name)
                    .and_then(|layout| layout.variant(variant_name))
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
                for (field_name, bound_name) in bindings {
                    let Some((_, offset, field_type)) =
                        fields.iter().find(|(name, _, _)| name == field_name)
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
                    self.define_variable(bound_name, bound);
                    // A binding takes the field out of the value being
                    // matched, so it holds whatever that field held. Without
                    // this a linear field could not be consumed by the arm
                    // that named it, which is the only way to consume one.
                    self.mark_owned(bound);
                }
                Ok(())
            }
            Pattern::Identifier(name) => {
                if let Some((value, value_type)) = scalar {
                    let bound = self
                        .fresh_local(value_type.clone(), Some(name.clone()));
                    self.emit(IrStatement::Assign(
                        bound,
                        IrRvalue::Use(value.clone()),
                    ));
                    self.define_variable(name, bound);
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn type_of_local(&self, local: LocalId) -> Type {
        self.locals[local].ty.clone()
    }

    fn shallow_value_type(&self, expression: &Expression) -> Option<Type> {
        match expression {
            Expression::Literal(Literal::Integer(_)) => Some(Type::I64),
            Expression::Literal(Literal::Float(_)) => Some(Type::F64),
            Expression::Literal(Literal::Float32(_)) => Some(Type::F32),
            Expression::Boolean(_)
            | Expression::Literal(Literal::Boolean(_)) => Some(Type::Bool),
            Expression::Identifier(name) => self
                .resolve_variable(name)
                .map(|local| self.type_of_local(local)),
            Expression::Borrow(inner) => self
                .shallow_value_type(inner)
                .map(|ty| Type::Ref(Box::new(ty))),
            Expression::BorrowMut(inner) => self
                .shallow_value_type(inner)
                .map(|ty| Type::RefMut(Box::new(ty))),
            Expression::StructInit(name, fields) => {
                if self.builder.generic_struct_defs.contains_key(name) {
                    self.generic_instance_of(name, fields).map(Type::Struct)
                } else {
                    Some(Type::Struct(name.clone()))
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
        field_inits: &[(String, Expression)],
    ) -> Option<String> {
        let (type_params, fields) =
            self.builder.generic_struct_defs.get(struct_name)?.clone();
        let mut subst: HashMap<String, Type> = HashMap::new();
        for (field_name, value) in field_inits {
            if let Some(field) =
                fields.iter().find(|field| &field.name == field_name)
                && let Some(value_type) = self.shallow_value_type(value)
            {
                infer_subst_into(
                    &field.field_type,
                    &value_type,
                    &type_params,
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
        if let (Type::Array(from_element, count), Type::Slice(to_element)) =
            (from, to)
            && from_element == to_element
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
// because arithmetic on a bool is not a thing the language has, but a `print`
// of one reaches the same `%lld` helper every integer does and has to arrive at
// its width. C widened it for free, so only the native backend saw this, as a
// verifier error rather than a wrong answer.
fn is_castable_integer(ty: &Type) -> bool {
    ty.is_integer() || matches!(ty, Type::Bool)
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
fn unify(left: &Type, right: &Type) -> Type {
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
fn is_bare_number(expression: &Expression) -> bool {
    match expression {
        Expression::Literal(Literal::Integer(_))
        | Expression::Literal(Literal::Float(_)) => true,
        Expression::Prefix(Operator::Negate, inner) => is_bare_number(inner),
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
