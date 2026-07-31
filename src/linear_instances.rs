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

use crate::ir_build::substitute_type;
use crate::lexer::Position;
use crate::parser::{
    Block, Expression, Parameter, ReturnKind, Spanned, Statement, StructField,
    SwitchCase, type_from_string,
};
use crate::types::Type;
use std::collections::{HashMap, HashSet};

/// The generic structs a program declares, by name, with their parameters and
/// the field types written under them.
pub(crate) type Templates<'a> =
    HashMap<&'a str, (&'a [String], Vec<StructField>)>;

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
    statements: &[Spanned<Statement>],
    instances: &Located,
    held: &HashSet<String>,
) -> Vec<String> {
    if held.is_empty() {
        return Vec::new();
    }
    let templates = declared_structs(statements);
    // Every pool a program has, which is an instantiation of a generic one and a
    // plainly declared one alike. A concrete `Pool :: struct { storage: [4]File,
    // generations: [4]i64 }` is the same container written out, and asking only
    // about instantiations let it through.
    let mut pools: Vec<(String, Position)> = instances
        .iter()
        .map(|(name, at)| (name.clone(), *at))
        .collect();
    for statement in statements {
        if let Statement::Struct(name, params, _) = &statement.node
            && params.is_empty()
        {
            pools.push((name.clone(), statement.position));
        }
    }
    let mut reports = Vec::new();
    for (instance, at) in &pools {
        // A name with no arguments binds nothing, and its fields are already the
        // types it has, so the same substitution answers for both shapes.
        let (base, arguments) = split_instance(instance)
            .unwrap_or_else(|| (instance.clone(), Vec::new()));
        let Some(element) = pool_element(&base, &arguments, &templates) else {
            continue;
        };
        if !element.is_linear_with(held) {
            continue;
        }
        reports.push(format!(
            "at {}: linearity: {}",
            at.describe(),
            pool_report(instance, &element)
        ));
    }
    reports.sort();
    // One type reaches both rules by more than one road, and a reader wants the
    // complaint once.
    reports.dedup();
    reports
}

/// What to say about a pool holding a resource. The name carries the prefix an
/// import gives a private declaration, and a reader wrote neither, so it is put
/// back the way every other diagnostic puts it back.
fn pool_report(instance: &str, element: &Type) -> String {
    let named = crate::imports::demangle_private_names(&format!(
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
                subst.insert(param.clone(), argument_type(argument));
            }
            fields.iter().find_map(|field| {
                let concrete = substitute_type(&field.field_type, &subst);
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
            .find(|field| field.name == wanted)
            .map(|field| &field.field_type)
    };
    let storage = field_type("storage")?;
    let generations = field_type("generations")?;
    if !is_run(storage) || !is_run(generations) {
        return None;
    }
    let mut subst: HashMap<String, Type> = HashMap::new();
    for (param, argument) in params.iter().zip(arguments.iter()) {
        subst.insert(param.clone(), argument_type(argument));
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
pub(crate) fn declared_structs(
    statements: &[Spanned<Statement>],
) -> Templates<'_> {
    let mut templates: Templates = HashMap::new();
    for statement in statements {
        match &statement.node {
            Statement::Struct(name, params, fields) => {
                templates
                    .insert(name.as_str(), (params.as_slice(), fields.clone()));
            }
            // An enum holds a resource when any variant's payload does, the
            // same way a struct holds one when any field does. Reading only
            // the structs left `Option<File>` ordinary data, so a resource put
            // in one lost its obligation on the way in.
            Statement::Enum(name, params, variants) => {
                let payload: Vec<StructField> = variants
                    .iter()
                    .filter_map(|variant| variant.fields.as_ref())
                    .flatten()
                    .cloned()
                    .collect();
                templates.insert(name.as_str(), (params.as_slice(), payload));
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
        subst.insert(param.clone(), argument_type(argument));
    }
    fields.iter().any(|field| {
        substitute_type(&field.field_type, &subst).is_linear_with(held)
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
    statements: &[Spanned<Statement>],
) -> HashSet<String> {
    locate_instances(statements).into_keys().collect()
}

/// The same, with where each was written, which is where a complaint about one
/// belongs. The position is the statement the name was found in, since that is
/// the line a reader looks at.
pub(crate) fn locate_instances(statements: &[Spanned<Statement>]) -> Located {
    let mut found = Located::new();
    for statement in statements {
        walk_statement(&statement.node, &mut found, statement.position);
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
    parameters: &[Parameter],
    found: &mut Located,
    at: Position,
) {
    for parameter in parameters {
        if let Some(ty) = &parameter.type_annotation {
            note_type(ty, found, at);
        }
        if let Some(ty) = &parameter.compile_time_signature {
            note_type(ty, found, at);
        }
    }
}

fn walk_return(kind: &ReturnKind, found: &mut Located, at: Position) {
    match kind {
        ReturnKind::None => {}
        ReturnKind::Single(ty) => note_type(ty, found, at),
        ReturnKind::Multiple(values) => {
            for held in values {
                note_type(&held.value_type, found, at);
            }
        }
        ReturnKind::Fallible(value, error) => {
            note_type(value, found, at);
            note_type(error, found, at);
        }
    }
}

fn walk_block(block: &Block, found: &mut Located) {
    for statement in block {
        walk_statement(&statement.node, found, statement.position);
    }
}

// Listed rather than caught by a wildcard, so a new statement form is a compile
// error here instead of a type position nobody walks.
fn walk_statement(statement: &Statement, found: &mut Located, at: Position) {
    match statement {
        Statement::TypeAlias(_, ty) | Statement::Flags(_, ty, _) => {
            note_type(ty, found, at)
        }
        Statement::Struct(_, _, fields) => {
            for field in fields {
                note_type(&field.field_type, found, at);
            }
        }
        Statement::Enum(_, _, variants) => {
            for variant in variants {
                if let Some(fields) = &variant.fields {
                    for field in fields {
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
            walk_parameters(params, found, at);
            if let Some(ty) = return_type {
                note_type(ty, found, at);
            }
        }
        Statement::Declared {
            params, return_sig, ..
        } => {
            walk_parameters(params, found, at);
            walk_return(&return_sig.kind, found, at);
            for capability in &return_sig.uses {
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
            walk_expression(value, found, at);
        }
        Statement::LetMultiple(_, value)
        | Statement::Constant(_, value)
        | Statement::Return(value)
        | Statement::Expression(value)
        | Statement::Print(value, _) => walk_expression(value, found, at),
        Statement::Assignment(place, value) => {
            walk_expression(place, found, at);
            walk_expression(value, found, at);
        }
        Statement::Defer(inner) => walk_statement(inner, found, at),
        Statement::For(_, _, sequence, body) => {
            walk_expression(sequence, found, at);
            walk_block(body, found);
        }
        Statement::While(condition, body) => {
            walk_expression(condition, found, at);
            walk_block(body, found);
        }
        Statement::With(_, body) => walk_block(body, found),
        Statement::Break | Statement::Continue | Statement::Import(..) => {}
    }
}

fn walk_expression(expression: &Expression, found: &mut Located, at: Position) {
    match expression {
        Expression::Function(parameters, signature, body)
        | Expression::Proc(parameters, signature, body) => {
            walk_parameters(parameters, found, at);
            walk_return(&signature.kind, found, at);
            for capability in &signature.uses {
                note_type(capability, found, at);
            }
            walk_block(body, found);
        }
        Expression::Sizeof(ty)
        | Expression::TypeId(ty)
        | Expression::TypeName(ty)
        | Expression::TypeValue(ty) => note_type(ty, found, at),
        Expression::PackMap(inner, _, _)
        | Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Dereference(inner)
        | Expression::UnsafeFn(inner)
        | Expression::FieldAccess(inner, _)
        | Expression::ArrayRepeat(inner, _)
        | Expression::Try(inner) => walk_expression(inner, found, at),
        Expression::Infix(left, _, right)
        | Expression::Index(left, right)
        | Expression::Range(left, right, _) => {
            walk_expression(left, found, at);
            walk_expression(right, found, at);
        }
        Expression::If(condition, then_block, else_block) => {
            walk_expression(condition, found, at);
            walk_block(then_block, found);
            if let Some(block) = else_block {
                walk_block(block, found);
            }
        }
        Expression::Unsafe(block) => walk_block(block, found),
        Expression::Switch(scrutinee, cases) => {
            walk_expression(scrutinee, found, at);
            for SwitchCase { body, .. } in cases {
                walk_block(body, found);
            }
        }
        Expression::Call(callee, arguments) => {
            walk_expression(callee, found, at);
            for argument in arguments {
                walk_expression(argument, found, at);
            }
        }
        Expression::Tuple(values) => {
            for held in values {
                walk_expression(held, found, at);
            }
        }
        // A struct literal names its type, and a generic one names the
        // instantiation it makes.
        Expression::StructInit(name, fields) => {
            if name.contains('<') {
                found.entry(name.clone()).or_insert(at);
            }
            for (_, held) in fields {
                walk_expression(held, found, at);
            }
        }
        Expression::EnumVariantInit(name, _, fields) => {
            if name.contains('<') {
                found.entry(name.clone()).or_insert(at);
            }
            for (_, held) in fields {
                walk_expression(held, found, at);
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
        let statements = parser.parse().unwrap();
        collect_instances(&statements)
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
        let statements = parser.parse().unwrap();
        let found = collect_instances(&statements);
        let mut held = parser.linear_types().clone();
        let templates = declared_structs(&statements);
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
