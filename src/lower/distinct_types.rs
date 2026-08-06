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
//
// The walk mutates a node's own types under a borrow scoped to that node,
// then recurses on child ids copied out first, which is how every mutating
// pass walks the arena.

use crate::ast::{
    Ast, ExprId, Expression, Range32, ReturnKind, SignatureId, Statement,
    StmtId,
};
use crate::types::Type;
use std::collections::HashMap;

pub fn resolve_distinct_types(ast: &mut Ast, roots: &[StmtId]) {
    let mut declared: HashMap<String, Type> = HashMap::new();
    for statement in roots {
        match ast.stmt(*statement) {
            Statement::TypeAlias(name, ty)
                if matches!(ty, Type::Distinct(..)) =>
            {
                declared.insert(ast.name(*name).to_string(), ty.clone());
            }
            // A flags declaration is a nominal type over an integer, which is
            // what a distinct declaration is, so every use of the name resolves
            // through the same table. What a flags type has on top of that is
            // the bits, and they are named under it rather than being types.
            Statement::Flags(name, repr, _) => {
                let name = ast.name(*name).to_string();
                declared.insert(
                    name.clone(),
                    Type::Distinct(name, Box::new(repr.clone())),
                );
            }
            _ => {}
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
    for statement in roots {
        walk_statement(ast, *statement, &resolved);
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
    ast: &mut Ast,
    parameters: Range32,
    declared: &HashMap<String, Type>,
) {
    for index in parameters.indices() {
        let parameter = &mut ast.parameters[index];
        if let Some(ty) = &mut parameter.type_annotation {
            substitute(ty, declared);
        }
        if let Some(ty) = &mut parameter.compile_time_signature {
            substitute(ty, declared);
        }
    }
}

fn walk_signature(
    ast: &mut Ast,
    signature: SignatureId,
    declared: &HashMap<String, Type>,
) {
    let values = {
        let held = &mut ast.signatures[signature.0 as usize];
        for capability in held.uses.iter_mut() {
            substitute(capability, declared);
        }
        match &mut held.kind {
            ReturnKind::None => None,
            ReturnKind::Single(ty) => {
                substitute(ty, declared);
                None
            }
            ReturnKind::Multiple(values) => Some(*values),
            ReturnKind::Fallible(value, error) => {
                substitute(value, declared);
                substitute(error, declared);
                None
            }
        }
    };
    if let Some(values) = values {
        for index in values.indices() {
            substitute(&mut ast.return_values[index].value_type, declared);
        }
    }
}

fn walk_statement(
    ast: &mut Ast,
    statement: StmtId,
    declared: &HashMap<String, Type>,
) {
    match ast.stmt(statement).clone() {
        Statement::TypeAlias(..) | Statement::Flags(..) => {
            let (Statement::TypeAlias(_, ty) | Statement::Flags(_, ty, _)) =
                &mut ast.statements[statement.0 as usize]
            else {
                return;
            };
            substitute(ty, declared);
        }
        Statement::Struct(_, _, fields) => {
            for index in fields.indices() {
                substitute(&mut ast.struct_fields[index].field_type, declared);
            }
        }
        Statement::Enum(_, _, variants) => {
            for variant_index in variants.indices() {
                if let Some(fields) = ast.enum_variants[variant_index].fields {
                    for index in fields.indices() {
                        substitute(
                            &mut ast.struct_fields[index].field_type,
                            declared,
                        );
                    }
                }
            }
        }
        Statement::Extern {
            params,
            return_type,
            ..
        } => {
            walk_parameters(ast, params, declared);
            if return_type.is_some() {
                let Statement::Extern {
                    return_type: Some(ty),
                    ..
                } = &mut ast.statements[statement.0 as usize]
                else {
                    return;
                };
                substitute(ty, declared);
            }
        }
        Statement::Declared {
            params, return_sig, ..
        } => {
            walk_parameters(ast, params, declared);
            walk_signature(ast, return_sig, declared);
        }
        Statement::Let {
            type_annotation,
            value,
            ..
        } => {
            if type_annotation.is_some() {
                let Statement::Let {
                    type_annotation: Some(ty),
                    ..
                } = &mut ast.statements[statement.0 as usize]
                else {
                    return;
                };
                substitute(ty, declared);
            }
            walk_expression(ast, value, declared);
        }
        Statement::LetMultiple(_, value)
        | Statement::Constant(_, value)
        | Statement::Return(value)
        | Statement::Expression(value) => walk_expression(ast, value, declared),
        Statement::Assignment(place, value) => {
            walk_expression(ast, place, declared);
            walk_expression(ast, value, declared);
        }
        Statement::Defer(inner)
            | Statement::ErrDefer(inner) => walk_statement(ast, inner, declared),
        Statement::For(_, _, sequence, body) => {
            walk_expression(ast, sequence, declared);
            walk_block(ast, body, declared);
        }
        Statement::While(condition, body) => {
            walk_expression(ast, condition, declared);
            walk_block(ast, body, declared);
        }
        Statement::With(_, body) => walk_block(ast, body, declared),
        Statement::Break | Statement::Continue | Statement::Import(..) => {}
    }
}

fn walk_block(ast: &mut Ast, block: Range32, declared: &HashMap<String, Type>) {
    for index in block.indices() {
        let statement = ast.stmt_list[index];
        walk_statement(ast, statement, declared);
    }
}

fn walk_expression(
    ast: &mut Ast,
    expression: ExprId,
    declared: &HashMap<String, Type>,
) {
    match ast.expr(expression).clone() {
        Expression::Function(parameters, signature, body)
        | Expression::Proc(parameters, signature, body) => {
            walk_parameters(ast, parameters, declared);
            walk_signature(ast, signature, declared);
            walk_block(ast, body, declared);
        }
        Expression::TypeValue(..) => {
            let Expression::TypeValue(ty) =
                &mut ast.expressions[expression.0 as usize]
            else {
                return;
            };
            substitute(ty, declared);
        }
        Expression::PackMap(inner, _, _)
        | Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Dereference(inner)
        | Expression::UnsafeFn(inner)
        | Expression::FieldAccess(inner, _)
        | Expression::ArrayRepeat(inner, _)
        | Expression::Try(inner) => walk_expression(ast, inner, declared),
        Expression::Infix(left, _, right)
        | Expression::Index(left, right)
        | Expression::Range(left, right, _) => {
            walk_expression(ast, left, declared);
            walk_expression(ast, right, declared);
        }
        Expression::If(condition, then_block, else_block) => {
            walk_expression(ast, condition, declared);
            walk_block(ast, then_block, declared);
            if let Some(block) = else_block {
                walk_block(ast, block, declared);
            }
        }
        Expression::Unsafe(block) => walk_block(ast, block, declared),
        Expression::Switch(scrutinee, cases) => {
            walk_expression(ast, scrutinee, declared);
            for index in cases.indices() {
                let body = ast.cases[index].body;
                walk_block(ast, body, declared);
            }
        }
        Expression::Call(callee, arguments) => {
            walk_expression(ast, callee, declared);
            for index in arguments.indices() {
                let argument = ast.expr_list[index];
                walk_expression(ast, argument, declared);
            }
        }
        Expression::Tuple(values) => {
            for index in values.indices() {
                let held = ast.expr_list[index];
                walk_expression(ast, held, declared);
            }
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => {
            for index in fields.indices() {
                let held = ast.named_exprs[index].value;
                walk_expression(ast, held, declared);
            }
        }
        Expression::Identifier(_)
        | Expression::Literal(_)
        | Expression::Boolean(_) => {}
    }
}
