use std::collections::HashSet;

use anyhow::{Context, Result};

use crate::ast::{Ast, Module, Splicer, StmtId};

// What a caller needs to know about a module without seeing the rest of it.
//
// Why this holds statements rather than a table of signatures: a generic's body is part
// of its interface, unavoidably, because the caller chooses the type arguments
// and so the caller is what instantiates the template. Once the body of an
// exported generic has to be here, the cheapest thing that is definitely
// complete is the declaration itself, and a signature table would be a second
// representation of the same facts that could drift from it.
//
// Non-generic bodies are not here. They are what a module can change without
// rebuilding its dependents.
//
// The declarations are a fresh arena built by copying the kept statements in
// order, symbols interned in first-use order, so the serialized form is
// deterministic and a fingerprint can hash it directly. The module's token
// position table comes with it whole, because a diagnostic raised inside a
// carried generic body while a caller instantiates it has to land on the
// statement that is wrong, not on the declaration that holds it.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct ModuleInterface {
    // The module's identity, its path relative to the project root. Also what
    // the private-name tag is derived from.
    pub module: String,
    pub exports: Vec<String>,
    pub declarations: Module,
    pub linear_types: Vec<String>,
}

// Everything an exported name's declaration can refer to has to come with it,
// so a type an exported function mentions is part of the interface whether or
// not the module chose to export the *name*. Exporting a function that returns
// an unexported struct is a program the current visibility rule allows, and the
// caller cannot type-check the call without the layout.
fn reachable_types(
    ast: &Ast,
    roots: &[StmtId],
    exports: &HashSet<String>,
) -> HashSet<String> {
    let mut wanted: HashSet<String> = HashSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for statement in roots {
            let Some(name) = declared_name(ast, *statement) else {
                continue;
            };
            if !exports.contains(name) && !wanted.contains(name) {
                continue;
            }
            let mut mentioned = Vec::new();
            crate::interface_names::names_in_statement(
                ast,
                *statement,
                &mut mentioned,
            );
            for named in mentioned {
                if !exports.contains(&named) && wanted.insert(named) {
                    changed = true;
                }
            }
        }
    }
    wanted
}

impl ModuleInterface {
    pub fn of(
        module: &str,
        ast: &Ast,
        roots: &[StmtId],
        exports: &[String],
        linear_types: &HashSet<String>,
    ) -> Self {
        let exported: HashSet<String> = exports.iter().cloned().collect();
        let carried = reachable_types(ast, roots, &exported);
        let kept: Vec<StmtId> = roots
            .iter()
            .copied()
            .filter(|statement| {
                declared_name(ast, *statement).is_some_and(|name| {
                    exported.contains(name) || carried.contains(name)
                })
            })
            .collect();
        let mut declarations = Module::default();
        let offset = crate::ast::splice_positions(&mut declarations.ast, ast);
        let splicer = Splicer::new(ast, offset);
        for statement in &kept {
            let copied = splicer.statement(
                &mut declarations.ast,
                *statement,
                &mut |name| name.to_string(),
            );
            declarations.roots.push(copied);
        }
        let mut linear: Vec<String> = linear_types.iter().cloned().collect();
        linear.sort();
        Self {
            module: module.to_string(),
            exports: exports.to_vec(),
            declarations,
            linear_types: linear,
        }
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .context("serializing a module interface")
    }

    pub fn from_json(text: &str) -> Result<Self> {
        serde_json::from_str(text).context("reading a module interface")
    }
}

// The compiler writes an interface out and reads it back, and checks that what came back says the same thing as the
// source, while still compiling from source. This is the differential oracle
// for the feature, and it exists because the class of bug separate compilation
// invites is the one that passes the test suite and links the wrong code.
//
// Off unless `FROST_CHECK_INTERFACES` is set, since a build should not pay for
// it, and on in the tests that cover imports.
pub fn interfaces_are_checked() -> bool {
    std::env::var("FROST_CHECK_INTERFACES").is_ok_and(|value| value != "0")
}

// An oracle rather than the way builds work. With this on, an imported module contributes what its interface
// says it contributes and nothing else, so a program that still compiles and
// still produces the same output is evidence that the interface is sufficient.
// That is the gate a build cache needs, and it is much cheaper to establish
// here than to debug once the compiler trusts interfaces for real.
pub fn built_from_interfaces() -> bool {
    std::env::var("FROST_BUILD_FROM_INTERFACES").is_ok_and(|value| value != "0")
}

// The interface has to carry a declaration for every name it exports, and for
// every name those declarations reach. A caller compiling against it and
// finding a name missing is the failure this is here to turn into a loud error
// at the module that caused it, rather than a confusing one at the importer.
pub fn check_interface_covers_exports(
    interface: &ModuleInterface,
) -> Result<()> {
    let held = &interface.declarations;
    let declared: HashSet<&str> = held
        .roots
        .iter()
        .filter_map(|statement| declared_name(&held.ast, *statement))
        .collect();
    for export in &interface.exports {
        if !declared.contains(export.as_str()) {
            anyhow::bail!(
                "module '{}' exports '{export}' but declares nothing by that name",
                interface.module
            );
        }
    }

    Ok(())
}

// Anything a carried declaration reaches has to be carried too, or a caller
// compiling against the interface sees a name it cannot resolve. Checked
// against the module's full declarations, since a name that this module does
// not declare at all comes from the module's own imports and is not this
// interface's to supply.
pub fn check_interface_is_closed(
    interface: &ModuleInterface,
    ast: &Ast,
    all_declarations: &[StmtId],
) -> Result<()> {
    let declared_here: HashSet<&str> = all_declarations
        .iter()
        .filter_map(|statement| declared_name(ast, *statement))
        .collect();
    let held = &interface.declarations;
    let carried: HashSet<&str> = held
        .roots
        .iter()
        .filter_map(|statement| declared_name(&held.ast, *statement))
        .collect();
    for statement in &held.roots {
        let mut mentioned = Vec::new();
        crate::interface_names::names_in_statement(
            &held.ast,
            *statement,
            &mut mentioned,
        );
        for name in mentioned {
            if declared_here.contains(name.as_str())
                && !carried.contains(name.as_str())
            {
                anyhow::bail!(
                    "the interface of '{}' reaches '{name}' but does not carry it, so a caller could not compile against it",
                    interface.module
                );
            }
        }
    }
    Ok(())
}

pub fn declared_name(ast: &Ast, statement: StmtId) -> Option<&str> {
    match ast.stmt(statement) {
        crate::ast::Statement::Constant(name, _)
        | crate::ast::Statement::Struct(name, _, _)
        | crate::ast::Statement::Enum(name, _, _)
        | crate::ast::Statement::Flags(name, _, _)
        | crate::ast::Statement::TypeAlias(name, _)
        | crate::ast::Statement::Extern { name, .. }
        | crate::ast::Statement::Declared { name, .. } => Some(ast.name(*name)),
        _ => None,
    }
}

pub fn check_interface_round_trip(interface: &ModuleInterface) -> Result<()> {
    let text = interface.to_json()?;
    let back = ModuleInterface::from_json(&text).with_context(|| {
        format!("reading back the interface of '{}'", interface.module)
    })?;
    if &back != interface {
        anyhow::bail!(
            "the interface of '{}' did not survive a round trip, so an interface does not mean the same thing written down as it does in memory: {}",
            interface.module,
            first_difference(interface, &back)
        );
    }
    Ok(())
}

/// What changed across the round trip, named closely enough to act on. A
/// message saying only that something changed leaves the reader to bisect the
/// module by hand.
fn first_difference(
    before: &ModuleInterface,
    after: &ModuleInterface,
) -> String {
    if before.module != after.module {
        return format!(
            "the module name became '{}' from '{}'",
            after.module, before.module
        );
    }
    if before.exports != after.exports {
        return format!(
            "the exports became {:?} from {:?}",
            after.exports, before.exports
        );
    }
    if before.linear_types != after.linear_types {
        return format!(
            "the linear types became {:?} from {:?}",
            after.linear_types, before.linear_types
        );
    }
    if before.declarations.roots.len() != after.declarations.roots.len() {
        return format!(
            "{} declarations came back from {}",
            after.declarations.roots.len(),
            before.declarations.roots.len()
        );
    }
    for (one, other) in before
        .declarations
        .roots
        .iter()
        .zip(&after.declarations.roots)
    {
        let rendered_before =
            crate::ast_display::display_stmt(&before.declarations.ast, *one);
        let rendered_after =
            crate::ast_display::display_stmt(&after.declarations.ast, *other);
        if rendered_before != rendered_after {
            let named = declared_name(&before.declarations.ast, *one)
                .unwrap_or("an unnamed declaration");
            return format!(
                "the declaration of '{named}' became {rendered_after:?} from {rendered_before:?}"
            );
        }
    }
    "nothing the comparison could name".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn interface_of(source: &str) -> ModuleInterface {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        let module = parser.parse().unwrap();
        ModuleInterface::of(
            "lib/test.frost",
            &module.ast,
            &module.roots,
            parser.exports(),
            &parser.linear_types().iter().cloned().collect(),
        )
    }

    // A diagnostic raised while a caller instantiates a carried generic has
    // to land on the body statement that is wrong. The interface carried one
    // position per nested statement before the arena and it still does: this
    // is the regression the stage-one review caught, where every node under a
    // declaration answered the declaration's first line.
    #[test]
    fn an_interface_keeps_the_positions_of_carried_bodies() {
        let source = "export sum\n\
                      sum :: fn($T: Type, a: $T, b: $T) -> $T {\n\
                      \x20   total := a + b\n\
                      \x20   total\n\
                      }\n";
        let interface = interface_of(source);
        let held = &interface.declarations;
        let crate::ast::Statement::Constant(_, value) =
            held.ast.stmt(held.roots[0])
        else {
            panic!("the carried declaration is the constant");
        };
        let (crate::ast::Expression::Function(_, _, body)
        | crate::ast::Expression::Proc(_, _, body)) = held.ast.expr(*value)
        else {
            panic!("the constant holds the generic function");
        };
        let first = held.ast.stmts_in(*body)[0];
        assert_eq!(held.ast.stmt_position(first).line, 3);
    }

    #[test]
    fn a_float_constant_survives_the_round_trip() {
        // Seventeen significant digits, which is where a decimal encoding stops
        // being lossless unless both ends round the same way. Written as bits,
        // there is nothing to round.
        let source = "export PITCH_LIMIT\n\
                      PITCH_LIMIT :: 1.5607963267948966\n\
                      TINY :: 0.1\n\
                      WIDE :: 1.0e300\n";
        let interface = interface_of(source);
        check_interface_round_trip(&interface).expect("a float round trips");
    }

    fn carried(interface: &ModuleInterface) -> Vec<&str> {
        let held = &interface.declarations;
        let mut names: Vec<&str> = held
            .roots
            .iter()
            .filter_map(|statement| declared_name(&held.ast, *statement))
            .collect();
        names.sort();
        names
    }

    // A private helper an exported function calls has to be in the interface,
    // or a caller instantiating that function has nothing to call.
    #[test]
    fn an_interface_carries_what_its_exports_reach() {
        let interface = interface_of(
            "export area\n\
             Shape :: struct { w: i64, h: i64 }\n\
             scale :: fn(x: i64) -> i64 { x * 2 }\n\
             unused :: fn() -> i64 { 7 }\n\
             area :: fn(s: Shape) -> i64 { scale(s.w * s.h) }\n",
        );
        assert_eq!(carried(&interface), vec!["Shape", "area", "scale"]);
    }

    // Reaching is transitive, so a helper's helper comes too.
    #[test]
    fn an_interface_closes_over_reaching() {
        let interface = interface_of(
            "export top\n\
             deep :: fn() -> i64 { 1 }\n\
             middle :: fn() -> i64 { deep() }\n\
             top :: fn() -> i64 { middle() }\n",
        );
        assert_eq!(carried(&interface), vec!["deep", "middle", "top"]);
    }

    #[test]
    fn an_interface_survives_a_round_trip() {
        let interface = interface_of(
            "export best\n\
             File :: linear struct { fd: i64 }\n\
             ascending :: fn(a: i64, b: i64) -> bool { a < b }\n\
             best :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T, move y: $T) -> $T {\n\
             \x20   mut result := x\n    if (before(y, result)) { result = y }\n    result\n}\n",
        );
        check_interface_round_trip(&interface).unwrap();
        check_interface_covers_exports(&interface).unwrap();
        assert!(interface.linear_types.contains(&"File".to_string()));
    }

    // The closure check has to fail when the closure is broken, or it
    // is not evidence of anything.
    #[test]
    fn a_broken_closure_is_reported() {
        let mut interface = interface_of(
            "export area\n\
             scale :: fn(x: i64) -> i64 { x * 2 }\n\
             area :: fn(w: i64) -> i64 { scale(w) }\n",
        );
        let source = interface.declarations.clone();
        check_interface_is_closed(&interface, &source.ast, &source.roots)
            .unwrap();
        let kept_ast = interface.declarations.ast.clone();
        interface.declarations.roots.retain(|statement| {
            declared_name(&kept_ast, *statement) != Some("scale")
        });
        assert!(
            check_interface_is_closed(&interface, &source.ast, &source.roots)
                .is_err()
        );
    }
}
