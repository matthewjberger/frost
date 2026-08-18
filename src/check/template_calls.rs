// A call written inside a generic that nothing instantiates.
//
// A body is lowered once per instance, so a generic nothing instantiates is
// lowered nowhere and the calls in it are read by no pass at all. What a reader
// gets for `fn($T: Type) { nosuchcall(v) }` is a program that builds, and the
// name that is not there is found on the day somebody first calls the generic.

use std::collections::HashSet;

use crate::ast::{Ast, ExprId, Expression, Range32, Statement, StmtId};
use crate::diagnostic::Diagnostic;
use crate::ir::build::{COMPILER_NAMES, UNDEFINED_CALL, function_is_generic};

/// The names a call may go to that no declaration in the program writes: what
/// the compiler answers itself, and the two container builtins that stand for a
/// zeroed value rather than for a function.
const ANSWERED_BY_THE_COMPILER: &[&str] = &[
    "columns_zeroed",
    "field_count",
    "fields",
    "is_array",
    "is_float",
    "is_integer",
    "is_linear",
    "is_numeric",
    "is_pointer",
    "is_slice",
    "is_struct",
    "live",
    "name_of",
    "offset_of",
    "slab_zeroed",
];

/// Every call inside a generic's body, against the names the program declares.
pub fn check_template_calls(ast: &Ast, roots: &[StmtId]) -> Vec<Diagnostic> {
    let known = callable(ast, roots);
    let mut found = Vec::new();
    for statement in roots {
        let Statement::Constant(_, value) = ast.stmt(*statement) else {
            continue;
        };
        let (Expression::Function(params, _, body)
        | Expression::Proc(params, _, body)) = ast.expr(*value)
        else {
            continue;
        };
        // A body with no compile-time parameter is lowered whether or not
        // anything calls it, and the calls in it are read there. Reading them
        // here as well would say the same thing twice.
        if !function_is_generic(ast, *params) {
            continue;
        }
        let mut scope: HashSet<&str> = known.clone();
        for parameter in ast.params_in(*params) {
            scope.insert(ast.name(parameter.name));
        }
        report_in_block(ast, *body, &scope, &mut found);
    }
    found
}

/// The names a call may name: what the program declares, and what the compiler
/// answers on its own.
fn callable<'a>(ast: &'a Ast, roots: &[StmtId]) -> HashSet<&'a str> {
    let mut held: HashSet<&str> = COMPILER_NAMES.iter().copied().collect();
    held.extend(ANSWERED_BY_THE_COMPILER.iter().copied());
    for statement in roots {
        match ast.stmt(*statement) {
            // A constant may hold a function, and one that holds a number is
            // not something a call can name either way: what a call to it is
            // gets said where the body is lowered, by the rules about calling a
            // value.
            Statement::Constant(name, _)
            | Statement::Extern { name, .. }
            | Statement::Declared { name, .. } => {
                held.insert(ast.name(*name));
            }
            _ => {}
        }
    }
    held
}

fn report_in_block(
    ast: &Ast,
    block: Range32,
    scope: &HashSet<&str>,
    found: &mut Vec<Diagnostic>,
) {
    // A name bound anywhere in the block is in scope for the whole of it, the
    // way a declaration at the top of a file is. Reading them in order would
    // make a call above the binding a call to nothing, which is a different
    // rule and one the lowering says.
    let mut inner = scope.clone();
    for statement in ast.stmts_in(block) {
        bind_statement(ast, *statement, &mut inner);
    }
    for statement in ast.stmts_in(block) {
        report_in_statement(ast, *statement, &inner, found);
    }
}

/// The names one statement binds, added to what is in scope beside it.
fn bind_statement<'a>(
    ast: &'a Ast,
    statement: StmtId,
    scope: &mut HashSet<&'a str>,
) {
    match ast.stmt(statement) {
        Statement::Let { name, .. } | Statement::Constant(name, _) => {
            scope.insert(ast.name(*name));
        }
        Statement::LetMultiple(bindings, _) => {
            for binding in ast.bindings_in(*bindings) {
                scope.insert(ast.name(binding.name));
            }
        }
        Statement::With(name, _) => {
            scope.insert(ast.name(*name));
        }
        _ => {}
    }
}

fn report_in_statement(
    ast: &Ast,
    statement: StmtId,
    scope: &HashSet<&str>,
    found: &mut Vec<Diagnostic>,
) {
    match ast.stmt(statement) {
        Statement::Let { value, .. } => {
            report_in_expression(ast, *value, scope, found);
        }
        Statement::LetMultiple(_, value) => {
            report_in_expression(ast, *value, scope, found);
        }
        Statement::Constant(_, value)
        | Statement::Return(value)
        | Statement::Expression(value) => {
            report_in_expression(ast, *value, scope, found);
        }
        Statement::Assignment(place, value) => {
            report_in_expression(ast, *place, scope, found);
            report_in_expression(ast, *value, scope, found);
        }
        Statement::While(condition, body) => {
            report_in_expression(ast, *condition, scope, found);
            report_in_block(ast, *body, scope, found);
        }
        // The loop's own bindings stand for a slot and a value inside it, so
        // they are in scope for the body and nowhere else.
        Statement::For(binding, second, over, body) => {
            report_in_expression(ast, *over, scope, found);
            let mut inner = scope.clone();
            inner.insert(ast.name(*binding));
            if let Some(held) = second {
                inner.insert(ast.name(*held));
            }
            report_in_block(ast, *body, &inner, found);
        }
        Statement::With(_, body) => {
            report_in_block(ast, *body, scope, found);
        }
        // A `defer` holds one statement rather than a block.
        Statement::Defer(held) | Statement::ErrDefer(held) => {
            report_in_statement(ast, *held, scope, found);
        }
        _ => {}
    }
}

fn report_in_expression(
    ast: &Ast,
    value: ExprId,
    scope: &HashSet<&str>,
    found: &mut Vec<Diagnostic>,
) {
    match ast.expr(value) {
        // Only a bare name is asked about. A call through a field reaches a
        // capability bundle, whose function comes from the value bound to it,
        // and a call through anything else is a call to a value, which the
        // lowering says where the body is expanded.
        Expression::Call(callee, arguments) => {
            if let Expression::Identifier(name) = ast.expr(*callee) {
                let named = ast.name(*name);
                if !scope.contains(named) {
                    found.push(Diagnostic::new(
                        ast.expr_position(*callee),
                        format!("{UNDEFINED_CALL} '{named}'"),
                    ));
                }
            } else {
                report_in_expression(ast, *callee, scope, found);
            }
            for argument in ast.exprs_in(*arguments) {
                report_in_expression(ast, *argument, scope, found);
            }
        }
        Expression::Prefix(_, held)
        | Expression::AddressOf(held)
        | Expression::Borrow(held)
        | Expression::BorrowMut(held)
        | Expression::Dereference(held)
        | Expression::FieldAccess(held, _)
        | Expression::PackMap(held, _, _)
        | Expression::Try(held)
        | Expression::UnsafeFn(held)
        | Expression::ArrayRepeat(held, _) => {
            report_in_expression(ast, *held, scope, found);
        }
        Expression::Infix(left, _, right)
        | Expression::Index(left, right)
        | Expression::Range(left, right, _) => {
            report_in_expression(ast, *left, scope, found);
            report_in_expression(ast, *right, scope, found);
        }
        Expression::If(condition, then, otherwise) => {
            report_in_expression(ast, *condition, scope, found);
            report_in_block(ast, *then, scope, found);
            if let Some(held) = otherwise {
                report_in_block(ast, *held, scope, found);
            }
        }
        Expression::Switch(subject, cases) => {
            report_in_expression(ast, *subject, scope, found);
            for case in ast.cases_in(*cases) {
                report_in_block(ast, case.body, scope, found);
            }
        }
        Expression::Unsafe(held) => {
            report_in_block(ast, *held, scope, found);
        }
        Expression::Tuple(items) => {
            for item in ast.exprs_in(*items) {
                report_in_expression(ast, *item, scope, found);
            }
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => {
            for field in ast.named_in(*fields) {
                report_in_expression(ast, field.value, scope, found);
            }
        }
        // A function written where it is used binds its own parameters, and
        // its body is read with them in scope.
        Expression::Function(params, _, body)
        | Expression::Proc(params, _, body) => {
            let mut inner = scope.clone();
            for parameter in ast.params_in(*params) {
                inner.insert(ast.name(parameter.name));
            }
            report_in_block(ast, *body, &inner, found);
        }
        _ => {}
    }
}
