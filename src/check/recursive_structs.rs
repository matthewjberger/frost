// A struct that holds one of itself.
//
// `Node :: struct { next: Node, v: i64 }` asks for storage that has no end: a
// Node holds a Node holds a Node. Both compilers took the declaration. The
// bootstrap then had no layout for it, so `sizeof(Node)` was refused where it
// was written and nothing else was; the self-hosted compiler answered with a
// width. The declaration is the thing with nothing behind it.
//
// A pointer to one is fine, and so is a slice of them: neither holds the value.
// Only what a value of the type would have to contain is followed - its fields,
// and the elements of any array among them.

use crate::ast::{Ast, Statement, StmtId};
use crate::diagnostic::Diagnostic;
use crate::types::Type;
use std::collections::HashMap;

/// Every struct that would have to contain itself.
pub fn check_recursive_structs(
    ast: &Ast,
    roots: &[StmtId],
) -> Vec<Diagnostic> {
    let mut held: HashMap<&str, Vec<&Type>> = HashMap::new();
    let mut order: Vec<(&str, StmtId)> = Vec::new();
    for statement in roots {
        let Statement::Struct(name, _, fields) = ast.stmt(*statement) else {
            continue;
        };
        let name = ast.name(*name);
        held.insert(
            name,
            ast.fields_in(*fields)
                .iter()
                .map(|field| &field.field_type)
                .collect(),
        );
        order.push((name, *statement));
    }
    let mut found = Vec::new();
    for (name, statement) in order {
        if reaches(name, name, &held, &mut Vec::new()) {
            found.push(Diagnostic::new(
                ast.stmt_position(statement),
                format!(
                    "'{}' holds a '{}' by value, which has no end; hold a pointer to one instead",
                    crate::demangle_private_names(name),
                    crate::demangle_private_names(name)
                ),
            ));
        }
    }
    found
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
