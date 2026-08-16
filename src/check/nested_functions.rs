// A function declared inside a body.
//
// `twice :: fn(a: i64) -> i64 { a * 2 }` written inside another function parsed
// as a name bound to a function value, and every use of it was then refused: a
// call because a call names the function it goes to rather than a value holding
// one, and `$twice` because a compile-time argument names a type or a function
// and a local names neither. Nothing could be done with the declaration, so the
// declaration is the thing to say something about.

use crate::ast::{Ast, Expression, Statement, StmtId};
use crate::diagnostic::Diagnostic;

/// Every function declared under another one.
pub fn check_nested_functions(ast: &Ast, roots: &[StmtId]) -> Vec<Diagnostic> {
    let mut found = Vec::new();
    for statement in roots {
        let Statement::Constant(_, value) = ast.stmt(*statement) else {
            continue;
        };
        let (Expression::Function(_, _, body) | Expression::Proc(_, _, body)) =
            ast.expr(*value)
        else {
            continue;
        };
        in_block(ast, *body, &mut found);
    }
    found
}

fn in_block(ast: &Ast, block: crate::ast::Range32, found: &mut Vec<Diagnostic>) {
    for statement in ast.stmts_in(block) {
        in_statement(ast, *statement, found);
    }
}

fn in_statement(ast: &Ast, statement: StmtId, found: &mut Vec<Diagnostic>) {
    match ast.stmt(statement) {
        Statement::Constant(name, value) => {
            if matches!(
                ast.expr(*value),
                Expression::Function(..) | Expression::Proc(..)
            ) {
                found.push(Diagnostic::new(
                    ast.stmt_position(statement),
                    format!(
                        "a function is declared where a file's other declarations are, and '{}' is declared inside a body",
                        crate::demangle_private_names(ast.name(*name))
                    ),
                ));
            }
        }
        Statement::While(condition, body) => {
            in_expression(ast, *condition, found);
            in_block(ast, *body, found);
        }
        Statement::For(_, _, over, body) => {
            in_expression(ast, *over, found);
            in_block(ast, *body, found);
        }
        Statement::With(_, body) => in_block(ast, *body, found),
        Statement::Defer(held) | Statement::ErrDefer(held) => {
            in_statement(ast, *held, found)
        }
        Statement::Let { value, .. }
        | Statement::Expression(value)
        | Statement::Return(value) => in_expression(ast, *value, found),
        Statement::Assignment(place, value) => {
            in_expression(ast, *place, found);
            in_expression(ast, *value, found);
        }
        _ => {}
    }
}

/// The blocks an expression holds, which are the only places it can hold a
/// declaration.
fn in_expression(
    ast: &Ast,
    value: crate::ast::ExprId,
    found: &mut Vec<Diagnostic>,
) {
    match ast.expr(value) {
        Expression::If(condition, then, otherwise) => {
            in_expression(ast, *condition, found);
            in_block(ast, *then, found);
            if let Some(held) = otherwise {
                in_block(ast, *held, found);
            }
        }
        Expression::Unsafe(held) => in_block(ast, *held, found),
        Expression::Switch(subject, cases) => {
            in_expression(ast, *subject, found);
            for case in ast.cases_in(*cases) {
                in_block(ast, case.body, found);
            }
        }
        _ => {}
    }
}
