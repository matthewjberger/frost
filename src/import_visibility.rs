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

use crate::interface_names::names_in_statement;
use crate::parser::{
    Block, Expression, Pattern, Spanned, Statement, SwitchCase,
};

// What one file wrote: the module it is, what it declares, what it imports, and
// the names it uses. The imports are canonical paths, since the same file
// reached through two search roots has two identity strings and one path.
pub struct FileNames {
    pub module: String,
    pub declared: HashSet<String>,
    pub imports: Vec<PathBuf>,
    pub used: Vec<String>,
}

impl FileNames {
    pub fn of(
        module: &str,
        statements: &[Spanned<Statement>],
        imports: &[PathBuf],
    ) -> Self {
        let mut declared = HashSet::new();
        for statement in statements {
            if let Some(name) = top_level_name(&statement.node) {
                declared.insert(name.to_string());
            }
        }
        let mut used = Vec::new();
        for statement in statements {
            let mut candidates = Vec::new();
            names_in_statement(&statement.node, &mut candidates);
            let mut bound = HashSet::new();
            bound_in_statement(&statement.node, &mut bound);
            used.extend(
                candidates.into_iter().filter(|name| !bound.contains(name)),
            );
        }
        FileNames {
            module: module.to_string(),
            declared,
            imports: imports.to_vec(),
            used,
        }
    }
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

fn top_level_name(statement: &Statement) -> Option<&str> {
    match statement {
        Statement::Constant(name, _)
        | Statement::Struct(name, _, _)
        | Statement::Enum(name, _, _)
        | Statement::Flags(name, _, _)
        | Statement::TypeAlias(name, _)
        | Statement::Extern { name, .. }
        | Statement::Declared { name, .. } => Some(name.as_str()),
        _ => None,
    }
}

// Every name a declaration binds anywhere inside it: parameters, locals, loop
// and region names, and what a pattern binds.
fn bound_in_statement(statement: &Statement, out: &mut HashSet<String>) {
    match statement {
        Statement::Let { name, value, .. } => {
            out.insert(name.clone());
            bound_in_expression(value, out);
        }
        Statement::LetMultiple(bindings, value) => {
            for binding in bindings {
                out.insert(binding.name.clone());
            }
            bound_in_expression(value, out);
        }
        Statement::Constant(_, value)
        | Statement::Return(value)
        | Statement::Expression(value)
        | Statement::Print(value, _) => bound_in_expression(value, out),
        Statement::Assignment(place, value) => {
            bound_in_expression(place, out);
            bound_in_expression(value, out);
        }
        Statement::Defer(inner) => bound_in_statement(inner, out),
        Statement::For(first, second, sequence, body) => {
            out.insert(first.clone());
            if let Some(second) = second {
                out.insert(second.clone());
            }
            bound_in_expression(sequence, out);
            bound_in_block(body, out);
        }
        Statement::While(condition, body) => {
            bound_in_expression(condition, out);
            bound_in_block(body, out);
        }
        Statement::With(capability, body) => {
            out.insert(capability.clone());
            bound_in_block(body, out);
        }
        _ => {}
    }
}

fn bound_in_block(block: &Block, out: &mut HashSet<String>) {
    for statement in block {
        bound_in_statement(&statement.node, out);
    }
}

fn bound_in_pattern(pattern: &Pattern, out: &mut HashSet<String>) {
    match pattern {
        Pattern::Identifier(name) => {
            out.insert(name.clone());
        }
        Pattern::EnumVariant { bindings, .. } => {
            for (_, binding) in bindings {
                out.insert(binding.clone());
            }
        }
        Pattern::Tuple(patterns) => {
            for held in patterns {
                bound_in_pattern(held, out);
            }
        }
        _ => {}
    }
}

fn bound_in_expression(expression: &Expression, out: &mut HashSet<String>) {
    match expression {
        Expression::Function(parameters, _, body)
        | Expression::Proc(parameters, _, body) => {
            for parameter in parameters {
                out.insert(parameter.name.clone());
                // `$T` binds the bare name too, since that is what the body
                // writes where the type goes.
                if let Some(bare) = parameter.name.strip_prefix('$') {
                    out.insert(bare.to_string());
                }
            }
            bound_in_block(body, out);
        }
        Expression::If(condition, then_block, else_block) => {
            bound_in_expression(condition, out);
            bound_in_block(then_block, out);
            if let Some(block) = else_block {
                bound_in_block(block, out);
            }
        }
        Expression::Unsafe(block) => bound_in_block(block, out),
        Expression::Switch(scrutinee, cases) => {
            bound_in_expression(scrutinee, out);
            for SwitchCase { pattern, body } in cases {
                bound_in_pattern(pattern, out);
                bound_in_block(body, out);
            }
        }
        Expression::Call(callee, arguments) => {
            bound_in_expression(callee, out);
            for argument in arguments {
                bound_in_expression(argument, out);
            }
        }
        Expression::Prefix(_, inner)
        | Expression::AddressOf(inner)
        | Expression::Borrow(inner)
        | Expression::BorrowMut(inner)
        | Expression::Dereference(inner)
        | Expression::UnsafeFn(inner)
        | Expression::FieldAccess(inner, _)
        | Expression::Try(inner) => bound_in_expression(inner, out),
        Expression::Infix(left, _, right)
        | Expression::Index(left, right)
        | Expression::Range(left, right, _) => {
            bound_in_expression(left, out);
            bound_in_expression(right, out);
        }
        Expression::Tuple(values) => {
            for held in values {
                bound_in_expression(held, out);
            }
        }
        Expression::StructInit(_, fields)
        | Expression::EnumVariantInit(_, _, fields) => {
            for (_, held) in fields {
                bound_in_expression(held, out);
            }
        }
        _ => {}
    }
}
