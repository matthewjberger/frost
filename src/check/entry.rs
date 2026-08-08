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

use crate::ast::{Ast, Expression, ReturnKind, Statement, StmtId};
use crate::diagnostic::Diagnostic;
use crate::types::Type;

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
        let (Expression::Function(params, signature, _)
        | Expression::Proc(params, signature, _)) = ast.expr(*value)
        else {
            continue;
        };
        // The capabilities a `uses` clause draws are counted with the written
        // parameters, because that is what they become: one implicit parameter
        // each, appended to the list and taken by write borrow. `main` drawing
        // one is handed a register nobody filled and the first write through it
        // faults, which is the same fault as a written parameter and wants the
        // same sentence.
        let declared =
            ast.params_in(*params).len() + ast.signature(*signature).uses.len();
        if declared > 0 {
            found.push(Diagnostic::new(
                ast.stmt_position(*statement),
                format!(
                    "'main' takes no parameters, and this one takes {declared}; what a call to it would supply is whatever the platform left in a register"
                ),
            ));
        }
        // The one caller settles the answer as well as the arguments. A `main`
        // answering anything else was emitted into a C signature that returns
        // `int`, so a struct reached the backend as a return type it has no
        // lowering for and a failure set reached it as the tagged union the `?`
        // machinery made, which is a synthesized name the reader never wrote.
        if !matches!(
            &ast.signature(*signature).kind,
            ReturnKind::Single(Type::I64)
        ) {
            found.push(Diagnostic::new(
                ast.stmt_position(*statement),
                "'main' is called by the C runtime and its answer is the process exit code, so it answers i64".to_string(),
            ));
        }
    }
    found
}
