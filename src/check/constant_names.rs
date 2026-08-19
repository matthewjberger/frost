// A constant's value is built out of constants.
//
// A type's name and a function's have no value before the program runs, so a
// constant written in terms of one has none either. Left unread, `Alias ::
// Target` went through as a value constant holding a struct's name, and what
// the reader was told about it came from wherever the name was used.

use std::collections::HashSet;

use crate::ast::{Ast, ExprId, Expression, Statement, StmtId};
use crate::diagnostic::Diagnostic;
use crate::lexer::Position;

/// The phrase a report about a name a constant may not read is written with.
/// Named here so the driver, which holds back lowering on it, reads the words
/// the check writes rather than a copy that a rewording leaves behind.
pub const NOT_A_CONSTANT: &str =
    "is not a constant, so it cannot appear in a constant";

/// Every constant whose value reads a name that is not a constant.
pub fn check_constant_names(ast: &Ast, roots: &[StmtId]) -> Vec<Diagnostic> {
    let mut constants: HashSet<&str> = HashSet::new();
    for statement in roots {
        if let Statement::Constant(name, value) = ast.stmt(*statement)
            && !matches!(
                ast.expr(*value),
                Expression::Function(..) | Expression::Proc(..)
            )
        {
            constants.insert(ast.name(*name));
        }
    }
    let mut found = Vec::new();
    for statement in roots {
        let Statement::Constant(_, value) = ast.stmt(*statement) else {
            continue;
        };
        if matches!(
            ast.expr(*value),
            Expression::Function(..) | Expression::Proc(..)
        ) {
            continue;
        }
        let mut read = Vec::new();
        names_read(ast, *value, &mut read);
        for (named, at) in read {
            if constants.contains(named.as_str()) {
                continue;
            }
            let readable =
                crate::modules::imports::demangle_private_names(&named);
            found.push(Diagnostic::new(
                at,
                format!("'{readable}' {NOT_A_CONSTANT}"),
            ));
            // One per constant. Every name in a value the reader has to rewrite
            // is a consequence of the same mistake.
            break;
        }
    }
    found
}

// Every name a constant's value reads as a value, with where it is written.
//
// A call's callee is left out: `LANES :: round_up(300, 64)` is a constant and
// the name there is the function being run rather than a value being read. A
// type handed to such a call is left out for the same reason, since that is
// what `sizeof(Point)` is written with; a type standing where a value goes is
// the thing this is looking for, so the two are told apart by where the type
// sits rather than by what it names.
fn names_read(ast: &Ast, expression: ExprId, out: &mut Vec<(String, Position)>) {
    match ast.expr(expression) {
        Expression::Identifier(name) => out.push((
            ast.name(*name).to_string(),
            ast.position_of(ast.expr_span(expression)),
        )),
        Expression::TypeValue(ty) => out.push((
            ty.to_string(),
            ast.position_of(ast.expr_span(expression)),
        )),
        Expression::Call(_, arguments) => {
            for argument in ast.exprs_in(*arguments).to_vec() {
                if matches!(ast.expr(argument), Expression::TypeValue(..)) {
                    continue;
                }
                names_read(ast, argument, out);
            }
        }
        Expression::Prefix(_, inner)
        | Expression::FieldAccess(inner, _)
        | Expression::ArrayRepeat(inner, _) => names_read(ast, *inner, out),
        Expression::Infix(left, _, right) | Expression::Index(left, right) => {
            let (left, right) = (*left, *right);
            names_read(ast, left, out);
            names_read(ast, right, out);
        }
        // A run and a record are kept as written rather than worked out, so
        // what stands in one is a value the program reads where the name is
        // read. `bump_source :: Allocation<Bump> { take = bump_take }` names a
        // function there, which is what an allocation source is.
        _ => {}
    }
}
