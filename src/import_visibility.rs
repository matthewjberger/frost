// An import says what a file may name.
//
// The flat namespace means an exported name is one atomic token wherever it is
// written, which is what makes it greppable. What it used to also mean is that
// every exported name in the program was visible from every file, whether or
// not that file imported the module exporting it: a file could call a function
// from a module it never named, an import line could be deleted with the build
// still passing, and the list at the top of a file was not the list of what it
// depends on.
//
// So a file sees its own names and the exported names of the modules it
// imports directly, and nothing else. The names are still flat and still
// unqualified. What changed is which of them a given file can reach.
//
// The check is deliberately one-sided. A candidate is a name the file writes
// and does not bind anywhere in the same declaration, and a name bound anywhere
// in a declaration is dropped from the whole of it. That misses a use that
// shares a name with a local somewhere else in the same function, which is a
// violation reported later or not at all. Erring the other way would reject a
// valid program, which a check like this may never do.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::ast::{
    Ast, ExprId, Expression, Pattern, PatternId, Range32, Statement, StmtId,
};
use crate::interface_names::names_in_statement;

// What one file wrote: the module it is, what it declares, what it imports, and
// the names it uses. The imports are canonical paths, since the same file
// reached through two search roots has two identity strings and one path.
pub struct FileNames {
    pub module: String,
    pub declared: HashSet<String>,
    pub imports: Vec<PathBuf>,
    pub used: Vec<String>,
    // The exported names this file read under another name. One of those does
    // not arrive under its own, so it is not a name this file holds twice.
    pub renamed: HashSet<String>,
}

impl FileNames {
    pub fn of(
        module: &str,
        ast: &Ast,
        roots: &[StmtId],
        imports: &[PathBuf],
    ) -> Self {
        let mut declared = HashSet::new();
        for statement in roots {
            if let Some(name) = top_level_name(ast, *statement) {
                declared.insert(name.to_string());
            }
        }
        let mut used = Vec::new();
        for statement in roots {
            let mut candidates = Vec::new();
            names_in_statement(ast, *statement, &mut candidates);
            let mut bound = HashSet::new();
            bound_in_statement(ast, *statement, &mut bound);
            used.extend(
                candidates.into_iter().filter(|name| !bound.contains(name)),
            );
        }
        let mut renamed = HashSet::new();
        for statement in roots {
            if let Statement::Import(_, renames) = ast.stmt(*statement) {
                for rename in ast.renames_in(*renames) {
                    renamed.insert(ast.name(rename.exported).to_string());
                }
            }
        }
        FileNames {
            module: module.to_string(),
            declared,
            imports: imports.to_vec(),
            used,
            renamed,
        }
    }
}

// Every name a file declares that an import already brought in.
//
// The namespace is flat, so two declarations of one name are two things called
// the same thing and there is no qualifying one of them. Left alone the two
// compilers answered differently: the bootstrap took the file's own and said
// nothing, and the self-hosted compiler emitted both under one symbol and left
// the assembler to notice. Neither is a choice a reader made, so the collision
// is refused and the reader makes it, by renaming the import or the declaration.
pub fn shadowed_imports(
    files: &[FileNames],
    exports: &HashMap<PathBuf, (String, HashSet<String>)>,
) -> Vec<String> {
    let mut reports = Vec::new();
    for file in files {
        let visible: HashSet<&str> = file
            .imports
            .iter()
            .filter_map(|path| exports.get(path))
            .flat_map(|(_, names)| names.iter().map(String::as_str))
            .collect();
        let mut said = HashSet::new();
        for name in &file.declared {
            // A name this file renamed on the way in arrives as the new name, so
            // the old one is not visible here and is not a collision. Renaming
            // is how a reader keeps both.
            if file.renamed.contains(name.as_str())
                || !visible.contains(name.as_str())
                || !said.insert(name.as_str())
            {
                continue;
            }
            reports.push(format!(
                "{}: '{name}' is declared here and also arrives from an import; rename one of them, or read the import under another name with `import \"...\" ({name} as ...)`",
                file.module
            ));
        }
    }
    reports.sort();
    reports
}

// Every name each file used that belongs to a module it did not import, as the
// diagnostics saying so. Reported together, since one missing import usually
// accounts for several names and fixing them one build at a time is the slow
// way to find that out.
pub fn unimported_names(
    files: &[FileNames],
    exports: &HashMap<PathBuf, (String, HashSet<String>)>,
) -> Vec<String> {
    let mut owner: HashMap<&str, Vec<(&PathBuf, &str)>> = HashMap::new();
    for (path, (module, names)) in exports {
        for name in names {
            owner
                .entry(name.as_str())
                .or_default()
                .push((path, module.as_str()));
        }
    }

    let mut reports = Vec::new();
    for file in files {
        let visible: HashSet<&str> = file
            .imports
            .iter()
            .filter_map(|path| exports.get(path))
            .flat_map(|(_, names)| names.iter().map(String::as_str))
            .collect();
        let mut said = HashSet::new();
        for name in &file.used {
            if file.declared.contains(name)
                || visible.contains(name.as_str())
                || !said.insert(name.as_str())
            {
                continue;
            }
            let Some(modules) = owner.get(name.as_str()) else {
                continue;
            };
            let modules: Vec<&str> = modules
                .iter()
                .filter(|(_, module)| *module != file.module)
                .map(|(_, module)| *module)
                .collect();
            let Some(first) = modules.first() else {
                continue;
            };
            reports.push(format!(
                "{}: '{name}' is exported by {first}, which this file does not import",
                file.module
            ));
        }
    }
    reports
}

fn top_level_name(ast: &Ast, statement: StmtId) -> Option<&str> {
    match ast.stmt(statement) {
        Statement::Constant(name, _)
        | Statement::Struct(name, _, _)
        | Statement::Enum(name, _, _)
        | Statement::Flags(name, _, _)
        | Statement::TypeAlias(name, _)
        | Statement::Extern { name, .. }
        | Statement::Declared { name, .. } => Some(ast.name(*name)),
        _ => None,
    }
}

// Every name a declaration binds anywhere inside it: parameters, locals, loop
// and region names, and what a pattern binds.
pub(crate) fn bound_in_statement(
    ast: &Ast,
    statement: StmtId,
    out: &mut HashSet<String>,
) {
    match ast.stmt(statement) {
        Statement::Let { name, value, .. } => {
            out.insert(ast.name(*name).to_string());
            bound_in_expression(ast, *value, out);
        }
        Statement::LetMultiple(bindings, value) => {
            for binding in ast.bindings_in(*bindings) {
                out.insert(ast.name(binding.name).to_string());
            }
            bound_in_expression(ast, *value, out);
        }
        Statement::Constant(_, value)
        | Statement::Return(value)
        | Statement::Expression(value)
        | Statement::Print(value, _) => bound_in_expression(ast, *value, out),
        Statement::Assignment(place, value) => {
            bound_in_expression(ast, *place, out);
            bound_in_expression(ast, *value, out);
        }
        Statement::Defer(inner) => bound_in_statement(ast, *inner, out),
        Statement::For(first, second, sequence, body) => {
            out.insert(ast.name(*first).to_string());
            if let Some(second) = second {
                out.insert(ast.name(*second).to_string());
            }
            bound_in_expression(ast, *sequence, out);
            bound_in_block(ast, *body, out);
        }
        Statement::While(condition, body) => {
            bound_in_expression(ast, *condition, out);
            bound_in_block(ast, *body, out);
        }
        Statement::With(capability, body) => {
            out.insert(ast.name(*capability).to_string());
            bound_in_block(ast, *body, out);
        }
        _ => {}
    }
}

fn bound_in_block(ast: &Ast, block: Range32, out: &mut HashSet<String>) {
    for statement in ast.stmts_in(block) {
        bound_in_statement(ast, *statement, out);
    }
}

fn bound_in_pattern(ast: &Ast, pattern: PatternId, out: &mut HashSet<String>) {
    match ast.pattern(pattern) {
        Pattern::Identifier(name) => {
            out.insert(ast.name(*name).to_string());
        }
        Pattern::EnumVariant { bindings, .. } => {
            for binding in ast.pattern_bindings_in(*bindings) {
                out.insert(ast.name(binding.binding).to_string());
            }
        }
        Pattern::Tuple(patterns) => {
            for held in ast.patterns_in(*patterns) {
                bound_in_pattern(ast, *held, out);
            }
        }
        _ => {}
    }
}

fn bound_in_expression(
    ast: &Ast,
    expression: ExprId,
    out: &mut HashSet<String>,
) {
    match ast.expr(expression) {
        Expression::Function(parameters, _, body)
        | Expression::Proc(parameters, _, body) => {
            for parameter in ast.params_in(*parameters) {
                let name = ast.name(parameter.name);
                out.insert(name.to_string());
                // `$T` binds the bare name too, since that is what the body
                // writes where the type goes.
                if let Some(bare) = name.strip_prefix('$') {
                    out.insert(bare.to_string());
                }
            }
            bound_in_block(ast, *body, out);
        }
        Expression::If(condition, then_block, else_block) => {
            bound_in_expression(ast, *condition, out);
            bound_in_block(ast, *then_block, out);
            if let Some(block) = else_block {
                bound_in_block(ast, *block, out);
            }
        }
        Expression::Unsafe(block) => bound_in_block(ast, *block, out),
        Expression::Switch(scrutinee, cases) => {
            bound_in_expression(ast, *scrutinee, out);
            for case in ast.cases_in(*cases) {
                bound_in_pattern(ast, case.pattern, out);
                bound_in_block(ast, case.body, out);
            }
        }
        Expression::Call(callee, arguments) => {
            bound_in_expression(ast, *callee, out);
            for argument in ast.exprs_in(*arguments) {
                bound_in_expression(ast, *argument, out);
            }
        }
        Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Dereference(inner)
        | Expression::UnsafeFn(inner)
        | Expression::FieldAccess(inner, _)
        | Expression::Try(inner) => bound_in_expression(ast, *inner, out),
        Expression::Infix(left, _, right)
        | Expression::Index(left, right)
        | Expression::Range(left, right, _) => {
            bound_in_expression(ast, *left, out);
            bound_in_expression(ast, *right, out);
        }
        Expression::Tuple(values) => {
            for held in ast.exprs_in(*values) {
                bound_in_expression(ast, *held, out);
            }
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => {
            for held in ast.named_in(*fields) {
                bound_in_expression(ast, held.value, out);
            }
        }
        _ => {}
    }
}
