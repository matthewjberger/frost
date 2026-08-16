// Every type name a signature writes has to be a type something declares.
//
// A name nothing declares is the shape a wrong guess takes: `fn(v: Absent)`
// names a type that is not there, and read as an opaque name it goes through
// every pass without a word until something asks what is in it, which for an
// unused parameter is never.

use std::collections::HashSet;

use crate::ast::{Ast, Statement, StmtId};
use crate::diagnostic::Diagnostic;
use crate::types::Type;

/// Every signature and every field, checked against the types the program
/// declares.
pub fn check_declared_types(ast: &Ast, roots: &[StmtId]) -> Vec<Diagnostic> {
    let known = declared(ast, roots);
    let mut found = Vec::new();
    for statement in roots {
        let position = ast.stmt_position(*statement);
        match ast.stmt(*statement) {
            Statement::Constant(_, value) => {
                let (Expression::Function(params, signature, body)
                | Expression::Proc(params, signature, body)) = ast.expr(*value)
                else {
                    continue;
                };
                // A binding writes a type the same way a parameter does, so a
                // name nothing declares is the same fault wherever it is
                // written. Left out, `held: Widget = 3` read as a binding of a
                // type the program had, and what the reader was told was that
                // an `i64` is not a `Widget`.
                report_unknown_in_block(ast, *body, &known, &mut found);
                for parameter in ast.params_in(*params) {
                    if let Some(ty) = &parameter.type_annotation {
                        // At the type, which is what the report names. The
                        // declaration's own place put the caret on the word the
                        // reader wrote first.
                        report_unknown(ty, &known, parameter.at, &mut found);
                    }
                }
                if let crate::ast::ReturnKind::Single(ty)
                | crate::ast::ReturnKind::Fallible(ty, _) =
                    &ast.signature(*signature).kind
                {
                    report_unknown(ty, &known, position, &mut found);
                }
            }
            Statement::Struct(_, _, fields) => {
                for field in ast.fields_in(*fields) {
                    report_unknown(
                        &field.field_type,
                        &known,
                        position,
                        &mut found,
                    );
                }
            }
            Statement::Extern {
                params,
                return_type,
                ..
            } => {
                for parameter in ast.params_in(*params) {
                    if let Some(ty) = &parameter.type_annotation {
                        report_unknown(ty, &known, parameter.at, &mut found);
                    }
                }
                if let Some(ty) = return_type {
                    report_unknown(ty, &known, position, &mut found);
                }
            }
            _ => {}
        }
    }
    found
}

use crate::ast::Expression;

/// The names the program declares as types, and the names a generic binds.
fn declared<'a>(ast: &'a Ast, roots: &[StmtId]) -> HashSet<&'a str> {
    let mut held = HashSet::new();
    for statement in roots {
        match ast.stmt(*statement) {
            Statement::Struct(name, generics, _)
            | Statement::Enum(name, generics, _) => {
                held.insert(ast.name(*name));
                for parameter in ast.symbols_in(*generics) {
                    held.insert(ast.name(*parameter));
                }
            }
            Statement::Flags(name, ..) | Statement::TypeAlias(name, _) => {
                held.insert(ast.name(*name));
            }
            _ => {}
        }
    }
    // A generic's type parameters name types inside its own body, bound where
    // the generic is expanded. Read from every parameter the program holds,
    // since a generic struct binds them the same way a generic function does.
    for parameter in &ast.parameters {
        if let Some(ty) = &parameter.type_annotation {
            bound(ty, &mut held);
        }
        if parameter.compile_time_signature.is_some() {
            held.insert(ast.name(parameter.name));
        }
    }
    held
}

/// Every name a `$T` annotation binds, wherever it sits inside the type.
fn bound<'a>(ty: &'a Type, held: &mut HashSet<&'a str>) {
    match ty {
        Type::TypeParam(name) => {
            held.insert(name.as_str());
        }
        Type::Ptr(inner)
        | Type::Ref(inner)
        | Type::RefMut(inner)
        | Type::Slice(inner)
        | Type::Array(inner, _)
        | Type::ArrayGeneric(inner, _)
        | Type::Handle(inner)
        | Type::Distinct(_, inner) => bound(inner, held),
        Type::Proc(params, answer) => {
            for parameter in params {
                bound(parameter, held);
            }
            bound(answer, held);
        }
        _ => {}
    }
}

/// Every binding written inside a body, and inside the bodies inside it.
///
/// A block is reached through whatever holds it, so each statement that holds
/// one names it here. A `defer` holds a single statement rather than a block,
/// which is why it is followed the same way and not with the others.
fn report_unknown_in_block(
    ast: &Ast,
    block: crate::ast::Range32,
    known: &HashSet<&str>,
    found: &mut Vec<Diagnostic>,
) {
    for statement in ast.stmts_in(block) {
        report_unknown_in_statement(ast, *statement, known, found);
    }
}

/// One statement: the type it writes, and whatever it holds.
fn report_unknown_in_statement(
    ast: &Ast,
    statement: StmtId,
    known: &HashSet<&str>,
    found: &mut Vec<Diagnostic>,
) {
    match ast.stmt(statement) {
        Statement::Let {
            type_annotation,
            type_at,
            value,
            ..
        } => {
            if let Some(ty) = type_annotation {
                // At the type, which is what the report names, the way a
                // parameter's already is.
                report_unknown(ty, known, *type_at, found);
            }
            report_unknown_in_expression(ast, *value, known, found);
        }
        Statement::While(condition, body) => {
            report_unknown_in_expression(ast, *condition, known, found);
            report_unknown_in_block(ast, *body, known, found);
        }
        Statement::For(_, _, over, body) => {
            report_unknown_in_expression(ast, *over, known, found);
            report_unknown_in_block(ast, *body, known, found);
        }
        Statement::With(_, body) => {
            report_unknown_in_block(ast, *body, known, found);
        }
        // A `defer` holds one statement rather than a block.
        Statement::Defer(held) | Statement::ErrDefer(held) => {
            report_unknown_in_statement(ast, *held, known, found);
        }
        Statement::Expression(value) | Statement::Return(value) => {
            report_unknown_in_expression(ast, *value, known, found);
        }
        Statement::Assignment(place, value) => {
            report_unknown_in_expression(ast, *place, known, found);
            report_unknown_in_expression(ast, *value, known, found);
        }
        _ => {}
    }
}

/// The blocks an expression holds. Only the shapes that hold one are followed:
/// the rest have no statements under them and nothing to declare.
fn report_unknown_in_expression(
    ast: &Ast,
    value: crate::ast::ExprId,
    known: &HashSet<&str>,
    found: &mut Vec<Diagnostic>,
) {
    match ast.expr(value) {
        Expression::If(condition, then, otherwise) => {
            report_unknown_in_expression(ast, *condition, known, found);
            report_unknown_in_block(ast, *then, known, found);
            if let Some(held) = otherwise {
                report_unknown_in_block(ast, *held, known, found);
            }
        }
        Expression::Unsafe(held) => {
            report_unknown_in_block(ast, *held, known, found);
        }
        Expression::Switch(subject, cases) => {
            report_unknown_in_expression(ast, *subject, known, found);
            for case in ast.cases_in(*cases) {
                report_unknown_in_block(ast, case.body, known, found);
            }
        }
        _ => {}
    }
}

/// A named type nothing declares, reported once per place it is written.
///
/// Only a plain name is asked about. A generic instance carries its arguments
/// in the name it is spelled with, and a type parameter is bound where the
/// generic is expanded, so neither is a name this can answer for.
fn report_unknown(
    ty: &Type,
    known: &HashSet<&str>,
    position: crate::lexer::Position,
    found: &mut Vec<Diagnostic>,
) {
    match ty {
        Type::Struct(name) | Type::Enum(name) => {
            if name.contains('<') || name.is_empty() {
                return;
            }
            if known.contains(name.as_str()) {
                return;
            }
            found.push(Diagnostic::new(
                position,
                format!("'{name}' is not a type this program declares"),
            ));
        }
        Type::Ptr(inner)
        | Type::Ref(inner)
        | Type::RefMut(inner)
        | Type::Slice(inner)
        | Type::Array(inner, _)
        | Type::ArrayGeneric(inner, _)
        | Type::Handle(inner)
        | Type::Distinct(_, inner) => {
            report_unknown(inner, known, position, found)
        }
        Type::Proc(params, answer) => {
            for parameter in params {
                report_unknown(parameter, known, position, found);
            }
            report_unknown(answer, known, position, found);
        }
        _ => {}
    }
}
