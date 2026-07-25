// `Meters :: distinct i64` declares a nominal type: the representation of the
// inner type under a name of its own, so a Meters is not an i64 and not a Feet.
//
// The parser cannot resolve a use of the name, since a type is written before
// the declaration as often as after it and a distinct type can come from
// another module. So the declarations are collected once the imports are
// resolved, and every type in the program that names one is rewritten from the
// `Struct(name)` the parser guessed to the `Distinct(name, inner)` it is.
// Everything after this reads a Distinct, whose size, layout and arithmetic
// follow the inner type while its identity does not.

use crate::parser::{
    Block, Expression, Parameter, ReturnKind, Spanned, Statement, SwitchCase,
};
use crate::types::Type;
use std::collections::HashMap;

pub fn resolve_distinct_types(statements: &mut [Spanned<Statement>]) {
    let mut declared: HashMap<String, Type> = HashMap::new();
    for statement in statements.iter() {
        if let Statement::TypeAlias(name, ty) = &statement.node
            && matches!(ty, Type::Distinct(..))
        {
            declared.insert(name.clone(), ty.clone());
        }
    }
    if declared.is_empty() {
        return;
    }
    // A distinct type over another one resolves to what that one resolved to,
    // so the chain is flattened here rather than at every use.
    let resolved: HashMap<String, Type> = declared
        .iter()
        .map(|(name, ty)| {
            let mut ty = ty.clone();
            substitute(&mut ty, &declared);
            (name.clone(), ty)
        })
        .collect();
    for statement in statements.iter_mut() {
        walk_statement(&mut statement.node, &resolved);
    }
}

fn substitute(ty: &mut Type, declared: &HashMap<String, Type>) {
    match ty {
        Type::Struct(name) | Type::Enum(name) => {
            if let Some(found) = declared.get(name.as_str()) {
                *ty = found.clone();
            }
        }
        Type::Ptr(inner)
        | Type::Ref(inner)
        | Type::RefMut(inner)
        | Type::Array(inner, _)
        | Type::ArrayGeneric(inner, _)
        | Type::Slice(inner)
        | Type::Handle(inner)
        | Type::Distinct(_, inner) => substitute(inner, declared),
        Type::Proc(parameters, result) => {
            for parameter in parameters.iter_mut() {
                substitute(parameter, declared);
            }
            substitute(result, declared);
        }
        _ => {}
    }
}

fn walk_parameters(
    parameters: &mut [Parameter],
    declared: &HashMap<String, Type>,
) {
    for parameter in parameters.iter_mut() {
        if let Some(ty) = &mut parameter.type_annotation {
            substitute(ty, declared);
        }
        if let Some(ty) = &mut parameter.compile_time_signature {
            substitute(ty, declared);
        }
    }
}

fn walk_statement(statement: &mut Statement, declared: &HashMap<String, Type>) {
    match statement {
        Statement::TypeAlias(_, ty) => substitute(ty, declared),
        Statement::Struct(_, _, fields) => {
            for field in fields.iter_mut() {
                substitute(&mut field.field_type, declared);
            }
        }
        Statement::Enum(_, _, variants) => {
            for variant in variants.iter_mut() {
                if let Some(fields) = &mut variant.fields {
                    for field in fields.iter_mut() {
                        substitute(&mut field.field_type, declared);
                    }
                }
            }
        }
        Statement::Extern {
            params,
            return_type,
            ..
        } => {
            walk_parameters(params, declared);
            if let Some(ty) = return_type {
                substitute(ty, declared);
            }
        }
        Statement::Declared {
            params, return_sig, ..
        } => {
            walk_parameters(params, declared);
            walk_return(&mut return_sig.kind, declared);
            for capability in return_sig.uses.iter_mut() {
                substitute(capability, declared);
            }
        }
        Statement::Let {
            type_annotation,
            value,
            ..
        } => {
            if let Some(ty) = type_annotation {
                substitute(ty, declared);
            }
            walk_expression(value, declared);
        }
        Statement::LetMultiple(_, value)
        | Statement::Constant(_, value)
        | Statement::Return(value)
        | Statement::Expression(value)
        | Statement::Print(value) => walk_expression(value, declared),
        Statement::Assignment(place, value) => {
            walk_expression(place, declared);
            walk_expression(value, declared);
        }
        Statement::Defer(inner) => walk_statement(inner, declared),
        Statement::For(_, _, sequence, body) => {
            walk_expression(sequence, declared);
            walk_block(body, declared);
        }
        Statement::While(condition, body) => {
            walk_expression(condition, declared);
            walk_block(body, declared);
        }
        Statement::With(_, body) => walk_block(body, declared),
        Statement::Break | Statement::Continue | Statement::Import(_) => {}
    }
}

fn walk_return(kind: &mut ReturnKind, declared: &HashMap<String, Type>) {
    match kind {
        ReturnKind::None => {}
        ReturnKind::Single(ty) => substitute(ty, declared),
        ReturnKind::Multiple(values) => {
            for held in values.iter_mut() {
                substitute(&mut held.value_type, declared);
            }
        }
        ReturnKind::Fallible(value, error) => {
            substitute(value, declared);
            substitute(error, declared);
        }
    }
}

fn walk_block(block: &mut Block, declared: &HashMap<String, Type>) {
    for statement in block.iter_mut() {
        walk_statement(&mut statement.node, declared);
    }
}

fn walk_expression(
    expression: &mut Expression,
    declared: &HashMap<String, Type>,
) {
    match expression {
        Expression::Function(parameters, signature, body)
        | Expression::Proc(parameters, signature, body) => {
            walk_parameters(parameters, declared);
            walk_return(&mut signature.kind, declared);
            for capability in signature.uses.iter_mut() {
                substitute(capability, declared);
            }
            walk_block(body, declared);
        }
        Expression::Sizeof(ty) | Expression::TypeValue(ty) => {
            substitute(ty, declared)
        }
        Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Dereference(inner)
        | Expression::UnsafeFn(inner)
        | Expression::FieldAccess(inner, _)
        | Expression::Try(inner) => walk_expression(inner, declared),
        Expression::Infix(left, _, right)
        | Expression::Index(left, right)
        | Expression::Range(left, right, _) => {
            walk_expression(left, declared);
            walk_expression(right, declared);
        }
        Expression::If(condition, then_block, else_block) => {
            walk_expression(condition, declared);
            walk_block(then_block, declared);
            if let Some(block) = else_block {
                walk_block(block, declared);
            }
        }
        Expression::Unsafe(block) => walk_block(block, declared),
        Expression::Switch(scrutinee, cases) => {
            walk_expression(scrutinee, declared);
            for SwitchCase { body, .. } in cases.iter_mut() {
                walk_block(body, declared);
            }
        }
        Expression::Call(callee, arguments) => {
            walk_expression(callee, declared);
            for argument in arguments.iter_mut() {
                walk_expression(argument, declared);
            }
        }
        Expression::Tuple(values) => {
            for held in values.iter_mut() {
                walk_expression(held, declared);
            }
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => {
            for (_, held) in fields.iter_mut() {
                walk_expression(held, declared);
            }
        }
        Expression::Identifier(_)
        | Expression::Literal(_)
        | Expression::Boolean(_) => {}
    }
}
