// What `frost lint` reports.
//
// Findings, not faults: a build never refuses on one. They are read off the
// tables the compiler already fills, so a lint is a question asked of what a
// pass knows rather than a walk of its own.

use std::collections::HashSet;

use crate::ast::{Ast, Expression, Statement, StmtId};
use crate::diagnostic::Diagnostic;
use crate::lexer::Token;

/// Every finding about a program, in source order.
pub fn lint(
    ast: &Ast,
    roots: &[StmtId],
    exported: &[String],
    tokens: &[Token],
    prefix: Option<&str>,
) -> Vec<Diagnostic> {
    let mut found = Vec::new();
    found.extend(idle_unsafe_blocks(ast, roots));
    found.extend(unreachable_functions(ast, roots, exported, tokens));
    found.extend(stray_prefixes(ast, roots, exported, prefix));
    found.sort_by_key(|held| {
        (held.position.file, held.position.line, held.position.column)
    });
    found
}

/// An `unsafe` block holding nothing that needs one.
fn idle_unsafe_blocks(ast: &Ast, roots: &[StmtId]) -> Vec<Diagnostic> {
    let (_, idle) =
        crate::check::unsafety::check_unsafety_and_audit(ast, roots);
    idle
}

/// How many times each name is written in the file.
///
/// Counted over the tokens rather than over the syntax, because a name is
/// reached in more ways than a call: `$handler` hands a function to a generic,
/// a table of function pointers names one, and an export list names one. A name
/// written once is written only where it is declared.
fn mentions(tokens: &[Token]) -> std::collections::HashMap<&str, usize> {
    let mut held = std::collections::HashMap::new();
    for token in tokens {
        if let Token::Identifier(name) = token {
            *held.entry(name.as_str()).or_insert(0) += 1;
        }
    }
    held
}

/// A function nothing reaches.
///
/// `main`, an exported name, a test body and a function a C caller names are
/// roots. Every other name written anywhere but its own declaration is reached.
/// A wrong finding here sends a reader to delete working code, so the question
/// asked is the conservative one.
///
/// A function written as `extern fn` with a body keeps the name it was written
/// under, because something outside the program calls it by that name. The
/// Frost half of the runtime is written that way.
fn unreachable_functions(
    ast: &Ast,
    roots: &[StmtId],
    exported: &[String],
    tokens: &[Token],
) -> Vec<Diagnostic> {
    let written = mentions(tokens);
    let exported: HashSet<&str> = exported.iter().map(String::as_str).collect();
    let mut found = Vec::new();
    for statement in roots {
        let Statement::Constant(symbol, value) = ast.stmt(*statement) else {
            continue;
        };
        if !matches!(
            ast.expr(*value),
            Expression::Function(..) | Expression::Proc(..)
        ) {
            continue;
        }
        let name = ast.name(*symbol);
        if name == "main"
            || name.contains(crate::parser::TEST_PREFIX)
            || exported.contains(name)
            || ast.is_exported_symbol(name)
            || written.get(name).copied().unwrap_or(0) > 1
        {
            continue;
        }
        found.push(Diagnostic::new(
            ast.stmt_position(*statement),
            format!("'{name}' is never called, and nothing names it"),
        ));
    }
    found
}

/// An exported name outside the prefix its directory declares.
///
/// A finding rather than a fault: the name is a convention, and a build that
/// refuses on one blocks work that is otherwise correct. `frost lint` exits
/// nonzero, which is what lets a project hold the line where it wants to.
fn stray_prefixes(
    ast: &Ast,
    roots: &[StmtId],
    exported: &[String],
    prefix: Option<&str>,
) -> Vec<Diagnostic> {
    let Some(prefix) = prefix else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for statement in roots {
        let (Statement::Constant(name, _)
        | Statement::Struct(name, ..)
        | Statement::Enum(name, ..)) = ast.stmt(*statement)
        else {
            continue;
        };
        let name = ast.name(*name);
        if !exported.iter().any(|held| held == name) {
            continue;
        }
        if name.starts_with(prefix) {
            continue;
        }
        found.push(Diagnostic::new(
            ast.stmt_position(*statement),
            format!(
                "'{name}' is exported from a directory whose names begin with '{prefix}', and it does not"
            ),
        ));
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn findings(source: &str) -> Vec<String> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let module = parser.parse().unwrap();
        let exports: Vec<String> = parser.exports().to_vec();
        lint(&module.ast, &module.roots, &exports, &tokens, None)
            .iter()
            .map(|held| held.message.clone())
            .collect()
    }

    #[test]
    fn a_function_nothing_names_is_reported() {
        let found = findings(
            "unused :: fn() -> i64 { 1 }\nmain :: fn() -> i64 { 0 }\n",
        );
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("'unused' is never called"));
    }

    #[test]
    fn a_function_a_call_names_is_left_alone() {
        assert!(
            findings(
                "used :: fn() -> i64 { 1 }\nmain :: fn() -> i64 { used() }\n"
            )
            .is_empty()
        );
    }

    // A function held as a value is reached through whatever table holds it.
    #[test]
    fn a_function_handed_over_as_a_value_is_left_alone() {
        assert!(
            findings(
                "handler :: fn() -> i64 { 1 }\n\
                 run :: fn(f: fn() -> i64) -> i64 { f() }\n\
                 main :: fn() -> i64 { run(handler) }\n"
            )
            .is_empty()
        );
    }

    #[test]
    fn an_idle_unsafe_block_is_reported() {
        let found = findings("main :: fn() -> i64 { unsafe { 0 } }\n");
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("vouches for nothing"));
    }
}
