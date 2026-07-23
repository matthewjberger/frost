use std::collections::HashSet;

use anyhow::{Context, Result};

use crate::parser::{Spanned, Statement};

// What a caller needs to know about a module without seeing the rest of it.
//
// The design is in docs/separate-compilation.md. The short version of why this
// holds statements rather than a table of signatures: a generic's body is part
// of its interface, unavoidably, because the caller chooses the type arguments
// and so the caller is what instantiates the template. Once the body of an
// exported generic has to be here, the cheapest thing that is definitely
// complete is the declaration itself, and a signature table would be a second
// representation of the same facts that could drift from it.
//
// Non-generic bodies are not here. That is the whole point: they are what a
// module can change without rebuilding its dependents.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq)]
pub struct ModuleInterface {
    // The module's identity, its path relative to the project root. Also what
    // the private-name tag is derived from.
    pub module: String,
    pub exports: Vec<String>,
    pub declarations: Vec<Spanned<Statement>>,
    pub linear_types: Vec<String>,
}

// Everything an exported name's declaration can refer to has to come with it,
// so a type an exported function mentions is part of the interface whether or
// not the module chose to export the *name*. Exporting a function that returns
// an unexported struct is a program the current visibility rule allows, and the
// caller cannot type-check the call without the layout.
fn reachable_types(
    declarations: &[Spanned<Statement>],
    exports: &HashSet<String>,
) -> HashSet<String> {
    let mut wanted: HashSet<String> = HashSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for statement in declarations {
            let Some(name) = declared_name(statement) else {
                continue;
            };
            if !exports.contains(name) && !wanted.contains(name) {
                continue;
            }
            let mut mentioned = Vec::new();
            crate::interface_names::names_in_statement(
                &statement.node,
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
        declarations: &[Spanned<Statement>],
        exports: &[String],
        linear_types: &HashSet<String>,
    ) -> Self {
        let exported: HashSet<String> = exports.iter().cloned().collect();
        let carried = reachable_types(declarations, &exported);
        let kept: Vec<Spanned<Statement>> = declarations
            .iter()
            .filter(|statement| {
                declared_name(statement).is_some_and(|name| {
                    exported.contains(name) || carried.contains(name)
                })
            })
            .cloned()
            .collect();
        let mut linear: Vec<String> = linear_types.iter().cloned().collect();
        linear.sort();
        Self {
            module: module.to_string(),
            exports: exports.to_vec(),
            declarations: kept,
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

// Step 2 of docs/separate-compilation.md: the compiler writes an interface out
// and reads it back, and checks that what came back says the same thing as the
// source, while still compiling from source. This is the differential oracle
// for the feature, and it exists because the class of bug separate compilation
// invites is the one that passes the test suite and links the wrong code.
//
// Off unless `FROST_CHECK_INTERFACES` is set, since a build should not pay for
// it, and on in the tests that cover imports.
pub fn interfaces_are_checked() -> bool {
    std::env::var("FROST_CHECK_INTERFACES").is_ok_and(|value| value != "0")
}

// Step 4 of docs/separate-compilation.md, as an oracle rather than as the way
// builds work. With this on, an imported module contributes what its interface
// says it contributes and nothing else, so a program that still compiles and
// still produces the same output is evidence that the interface is sufficient.
// That is the gate step 5 needs, and it is much cheaper to establish here than
// to debug once the compiler has started trusting interfaces for real.
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
    let declared: HashSet<&str> = interface
        .declarations
        .iter()
        .filter_map(declared_name)
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
    all_declarations: &[Spanned<Statement>],
) -> Result<()> {
    let declared_here: HashSet<&str> =
        all_declarations.iter().filter_map(declared_name).collect();
    let carried: HashSet<&str> = interface
        .declarations
        .iter()
        .filter_map(declared_name)
        .collect();
    for statement in &interface.declarations {
        let mut mentioned = Vec::new();
        crate::interface_names::names_in_statement(
            &statement.node,
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

fn declared_name(statement: &Spanned<Statement>) -> Option<&str> {
    match &statement.node {
        Statement::Constant(name, _)
        | Statement::Struct(name, _, _)
        | Statement::Enum(name, _, _)
        | Statement::TypeAlias(name, _)
        | Statement::Extern { name, .. }
        | Statement::Declared { name, .. } => Some(name.as_str()),
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
            "the interface of '{}' did not survive a round trip, so an interface does not mean the same thing written down as it does in memory",
            interface.module
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn interface_of(source: &str) -> ModuleInterface {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        let statements = parser.parse().unwrap();
        ModuleInterface::of(
            "lib/test.frost",
            &statements,
            parser.exports(),
            &parser.linear_types().iter().cloned().collect(),
        )
    }

    fn carried(interface: &ModuleInterface) -> Vec<&str> {
        let mut names: Vec<&str> = interface
            .declarations
            .iter()
            .filter_map(declared_name)
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

    // The closure check has to actually fail when the closure is broken, or it
    // is not evidence of anything.
    #[test]
    fn a_broken_closure_is_reported() {
        let mut interface = interface_of(
            "export area\n\
             scale :: fn(x: i64) -> i64 { x * 2 }\n\
             area :: fn(w: i64) -> i64 { scale(w) }\n",
        );
        let source = interface.declarations.clone();
        check_interface_is_closed(&interface, &source).unwrap();
        interface
            .declarations
            .retain(|statement| declared_name(statement) != Some("scale"));
        assert!(check_interface_is_closed(&interface, &source).is_err());
    }
}
