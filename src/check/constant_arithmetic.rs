// A constant whose value is arithmetic over whole numbers, worked out where it
// is written.
//
// `BIG :: 9223372036854775807 + 1` and `BAD :: 8 / 0` were taken: the fold that
// reads them answers nothing for either, so what reached the program was the
// arithmetic itself and the trap came when the program ran, if it ever reached
// that line. A constant made of numbers and operators has one answer and it is
// known here, so a constant with no answer is refused here.
//
// Only literals and the operators over them. A name, a call, or anything asking
// what a type measures is left to the passes that can answer those.

use crate::ast::{Ast, Expression, Literal, Statement, StmtId};
use crate::diagnostic::Diagnostic;
use crate::parser::Operator;

/// Every constant whose value is arithmetic that has no answer.
pub fn check_constant_arithmetic(
    ast: &Ast,
    roots: &[StmtId],
) -> Vec<Diagnostic> {
    let mut found = Vec::new();
    for statement in roots {
        let Statement::Constant(_, value) = ast.stmt(*statement) else {
            continue;
        };
        if let Err(message) = worked_out(ast, *value) {
            found.push(Diagnostic::new(ast.expr_position(*value), message));
        }
    }
    found
}

/// The value, or what stopped it. `None` for a shape this does not read, which
/// is not a fault: it is a question for a pass that can answer it.
fn worked_out(
    ast: &Ast,
    value: crate::ast::ExprId,
) -> Result<Option<i64>, String> {
    match ast.expr(value) {
        Expression::Literal(Literal::Integer(held)) => Ok(Some(*held)),
        // `Negate`, which is what a minus in front of a number is. Written as
        // `Subtract`, which is what a minus between two is, this arm never ran
        // and every constant holding a negative number went unread.
        Expression::Prefix(Operator::Negate, inner) => {
            let Some(held) = worked_out(ast, *inner)? else {
                return Ok(None);
            };
            Ok(Some(held.checked_neg().ok_or("negating this overflows")?))
        }
        Expression::Infix(left, operator, right) => {
            let (Some(left), Some(right)) =
                (worked_out(ast, *left)?, worked_out(ast, *right)?)
            else {
                return Ok(None);
            };
            let overflowed = || "this overflows an i64".to_string();
            let divided = || "this divides by zero".to_string();
            Ok(Some(match operator {
                Operator::Add => {
                    left.checked_add(right).ok_or_else(overflowed)?
                }
                Operator::Subtract => {
                    left.checked_sub(right).ok_or_else(overflowed)?
                }
                Operator::Multiply => {
                    left.checked_mul(right).ok_or_else(overflowed)?
                }
                // Zero is asked for before the division, because a division
                // answers nothing for two reasons and they are not the same
                // fault: `i64::MIN / -1` has an answer one wider than an i64
                // holds, and calling that dividing by zero named something the
                // program does not do.
                Operator::Divide => {
                    if right == 0 {
                        return Err(divided());
                    }
                    left.checked_div(right).ok_or_else(overflowed)?
                }
                // The remainder of the smallest by minus one is nothing, and
                // that is what the machine answers. Only the quotient leaves
                // the range.
                Operator::Modulo => {
                    if right == 0 {
                        return Err(divided());
                    }
                    if right == -1 { 0 } else { left % right }
                }
                _ => return Ok(None),
            }))
        }
        _ => Ok(None),
    }
}
