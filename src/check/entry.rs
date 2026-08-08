// The program's entry takes nothing.
//
// `main` is called by the C runtime, which hands it the argument count and the
// argument vector. A Frost `main` declares no parameters and both backends emit
// it that way, so a `main` that does declare one is handed whatever the platform
// left in that register: `main :: fn(n: i64)` reads the argument count nothing
// asked for, and `main :: fn(s: str)` reads it as an address and faults on the
// first byte. Neither needs an `unsafe` block anywhere, so this is safe code
// reaching memory nobody gave it, and the only thing that can catch it is the
// declaration.

use crate::ast::{Ast, Expression, Statement, StmtId};
use crate::diagnostic::Diagnostic;

/// The entry point's signature, checked against the one shape it has.
pub fn check_entry_point(ast: &Ast, roots: &[StmtId]) -> Vec<Diagnostic> {
    let mut found = Vec::new();
    for statement in roots {
        let Statement::Constant(name, value) = ast.stmt(*statement) else {
            continue;
        };
        if ast.name(*name) != "main" {
            continue;
        }
        let (Expression::Function(params, ..) | Expression::Proc(params, ..)) =
            ast.expr(*value)
        else {
            continue;
        };
        let declared = ast.params_in(*params).len();
        if declared == 0 {
            continue;
        }
        found.push(Diagnostic::new(
            ast.stmt_position(*statement),
            format!(
                "'main' takes no parameters, and this one takes {declared}; what a call to it would supply is whatever the platform left in a register"
            ),
        ));
    }
    found
}
