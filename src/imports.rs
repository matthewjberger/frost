use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::lexer::Lexer;
use crate::parser::{Parser, Spanned, Statement};

pub struct Resolved {
    pub statements: Vec<Spanned<Statement>>,
    pub linear_types: HashSet<String>,
    pub tests: Vec<(String, String)>,
}

pub fn resolve_imports(
    statements: Vec<Spanned<Statement>>,
    base_dir: &Path,
    linear_types: HashSet<String>,
    tests: Vec<(String, String)>,
) -> Result<Resolved> {
    let mut seen = HashSet::new();
    let mut resolved = Resolved {
        statements: Vec::new(),
        linear_types,
        tests,
    };
    resolve_into(statements, base_dir, &mut seen, &mut resolved)?;
    Ok(resolved)
}

fn resolve_into(
    statements: Vec<Spanned<Statement>>,
    base_dir: &Path,
    seen: &mut HashSet<PathBuf>,
    resolved: &mut Resolved,
) -> Result<()> {
    for statement in statements {
        let Statement::Import(path) = &statement.node else {
            resolved.statements.push(statement);
            continue;
        };

        let full = base_dir.join(path);
        let key = full.canonicalize().unwrap_or_else(|_| full.clone());
        if !seen.insert(key) {
            continue;
        }

        let source = fs::read_to_string(&full).with_context(|| {
            format!("failed to read imported file: {}", full.display())
        })?;
        let mut lexer = Lexer::new(&source);
        let tokens = lexer
            .tokenize()
            .with_context(|| format!("lexing {}", full.display()))?;
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        let imported = parser
            .parse()
            .with_context(|| format!("parsing {}", full.display()))?;
        resolved
            .linear_types
            .extend(parser.linear_types().iter().cloned());
        resolved.tests.extend(parser.tests().iter().cloned());

        let child_dir =
            full.parent().map(Path::to_path_buf).unwrap_or_default();
        resolve_into(imported, &child_dir, seen, resolved)?;
    }
    Ok(())
}
