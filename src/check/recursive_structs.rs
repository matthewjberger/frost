// A struct that holds one of itself.
//
// `Node :: struct { next: Node, v: i64 }` asks for storage that has no end: a
// Node holds a Node holds a Node. Both compilers took the declaration. The
// bootstrap then had no layout for it, so `sizeof(Node)` was refused where it
// was written and nothing else was; the self-hosted compiler answered with a
// width. The declaration is the thing with nothing behind it.
//
// An enum is asked the same question. One variant is live at a time, but the
// storage is wide enough for the widest, so a variant carrying the enum by
// value has no width to be wide enough for.
//
// A pointer to one is fine, and so is a slice of them: neither holds the value.
// Only what a value of the type would have to contain is followed - its fields,
// and the elements of any array among them.

use crate::ast::{Ast, Statement, StmtId};
use crate::diagnostic::Diagnostic;
use crate::types::Type;
use std::collections::HashMap;

/// The phrase a report about a type that holds itself is written with, named for
/// the driver the way [`crate::check::declared_types::UNDECLARED_TYPE`] is.
pub const RECURSIVE_STRUCT: &str = "by value, which has no end";

/// Every struct that would have to contain itself.
pub fn check_recursive_structs(ast: &Ast, roots: &[StmtId]) -> Vec<Diagnostic> {
    let mut held: HashMap<&str, Vec<&Type>> = HashMap::new();
    let mut order: Vec<(&str, StmtId)> = Vec::new();
    for statement in roots {
        let (name, fields) = match ast.stmt(*statement) {
            Statement::Struct(name, _, fields) => (
                name,
                ast.fields_in(*fields)
                    .iter()
                    .map(|field| &field.field_type)
                    .collect(),
            ),
            // A variant's payload is storage the enum takes in, so every
            // variant's fields are read as the enum's own.
            Statement::Enum(name, _, variants) => (
                name,
                ast.variants_in(*variants)
                    .iter()
                    .filter_map(|variant| variant.fields)
                    .flat_map(|fields| {
                        ast.fields_in(fields)
                            .iter()
                            .map(|field| &field.field_type)
                    })
                    .collect(),
            ),
            _ => continue,
        };
        let name = ast.name(*name);
        held.insert(name, fields);
        order.push((name, *statement));
    }
    let mut found = Vec::new();
    for (name, statement) in order {
        if let Some(inner) = holds(name, &held) {
            found.push(Diagnostic::new(
                ast.stmt_position(statement),
                format!(
                    "'{}' holds a '{}' {RECURSIVE_STRUCT}; hold a pointer to one instead",
                    crate::demangle_private_names(name),
                    crate::demangle_private_names(inner)
                ),
            ));
        }
    }
    found
}

/// The type of the first field that leads back to `wanted`, which is what the
/// report names. Saying `wanted` twice was a lie where the cycle runs through
/// something else: `A` holding a `B` holding an `A` does not hold an `A`.
fn holds<'a>(
    wanted: &str,
    held: &HashMap<&'a str, Vec<&'a Type>>,
) -> Option<&'a str> {
    for field in held.get(wanted)? {
        if by_value(field, wanted, held, &mut Vec::new()) {
            return named(field);
        }
    }
    None
}

/// The name a field's storage stands under, which an array or a distinct type
/// is written over the top of.
fn named(ty: &Type) -> Option<&str> {
    match ty {
        Type::Struct(name) => Some(name.as_str()),
        Type::Array(element, _) | Type::ArrayGeneric(element, _) => {
            named(element)
        }
        Type::Distinct(_, inner) => named(inner),
        _ => None,
    }
}

/// Whether a value of `from` has to contain a value of `wanted`.
fn reaches<'a>(
    from: &'a str,
    wanted: &str,
    held: &HashMap<&'a str, Vec<&'a Type>>,
    seen: &mut Vec<&'a str>,
) -> bool {
    if seen.contains(&from) {
        return false;
    }
    seen.push(from);
    let Some(fields) = held.get(from) else {
        return false;
    };
    fields
        .iter()
        .any(|field| by_value(field, wanted, held, seen))
}

/// The types a field's storage takes in, which is the field's own and, for an
/// array, its elements'.
fn by_value<'a>(
    ty: &'a Type,
    wanted: &str,
    held: &HashMap<&'a str, Vec<&'a Type>>,
    seen: &mut Vec<&'a str>,
) -> bool {
    match ty {
        Type::Struct(name) => {
            name == wanted || reaches(name.as_str(), wanted, held, seen)
        }
        Type::Array(element, _) | Type::ArrayGeneric(element, _) => {
            by_value(element, wanted, held, seen)
        }
        Type::Distinct(_, inner) => by_value(inner, wanted, held, seen),
        _ => false,
    }
}
