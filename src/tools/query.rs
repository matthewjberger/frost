// What an editor asks a compiler: the symbols of a module, the type of a
// local, the fields of a struct, the declaration a name resolves to. Every
// answer here is computed when asked and never during a build, reads the
// arenas and the typed IR the passes already produce, and is keyed by the
// node ids the arena already has. The namespace is flat, so definition
// lookup is a walk over the top level rather than a resolution.

use crate::ast::{Ast, Expression, Statement, StmtId};
use crate::ir::IrModule;
use crate::lexer::Position;
use crate::types::Type;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Flags,
    TypeAlias,
    Constant,
    Extern,
    Declared,
}

#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: SymbolKind,
    pub statement: StmtId,
    pub position: Position,
}

/// Every named top-level declaration, in source order.
pub fn symbols(ast: &Ast, roots: &[StmtId]) -> Vec<SymbolInfo> {
    let mut found = Vec::new();
    for statement in roots {
        let (name, kind) = match ast.stmt(*statement) {
            Statement::Constant(name, value) => {
                let kind = match ast.expr(*value) {
                    Expression::Function(..) | Expression::Proc(..) => {
                        SymbolKind::Function
                    }
                    _ => SymbolKind::Constant,
                };
                (*name, kind)
            }
            Statement::Struct(name, _, _) => (*name, SymbolKind::Struct),
            Statement::Enum(name, _, _) => (*name, SymbolKind::Enum),
            Statement::Flags(name, _, _) => (*name, SymbolKind::Flags),
            Statement::TypeAlias(name, _) => (*name, SymbolKind::TypeAlias),
            Statement::Extern { name, .. } => (*name, SymbolKind::Extern),
            Statement::Declared { name, .. } => (*name, SymbolKind::Declared),
            _ => continue,
        };
        found.push(SymbolInfo {
            name: ast.name(name).to_string(),
            kind,
            statement: *statement,
            position: ast.stmt_position(*statement),
        });
    }
    found
}

/// The declaration a name resolves to. The namespace is flat, so there is at
/// most one, and the first declared wins the way the passes read it.
pub fn definition_of(
    ast: &Ast,
    roots: &[StmtId],
    name: &str,
) -> Option<SymbolInfo> {
    symbols(ast, roots)
        .into_iter()
        .find(|held| held.name == name)
}

/// The fields of a named struct, or of an enum variant's payload when the
/// name is written `Enum::Variant`.
pub fn fields_of(
    ast: &Ast,
    roots: &[StmtId],
    name: &str,
) -> Option<Vec<(String, Type)>> {
    for statement in roots {
        match ast.stmt(*statement) {
            Statement::Struct(held, _, fields) if ast.name(*held) == name => {
                return Some(
                    ast.fields_in(*fields)
                        .iter()
                        .map(|field| {
                            (
                                ast.name(field.name).to_string(),
                                field.field_type.clone(),
                            )
                        })
                        .collect(),
                );
            }
            Statement::Enum(held, _, variants) => {
                for variant in ast.variants_in(*variants) {
                    let written = format!(
                        "{}::{}",
                        ast.name(*held),
                        ast.name(variant.name)
                    );
                    if written != name {
                        continue;
                    }
                    let declared = variant.fields?;
                    return Some(
                        ast.fields_in(declared)
                            .iter()
                            .map(|field| {
                                (
                                    ast.name(field.name).to_string(),
                                    field.field_type.clone(),
                                )
                            })
                            .collect(),
                    );
                }
            }
            _ => {}
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct LocalInfo {
    pub name: String,
    pub ty: Type,
    pub position: Position,
}

/// The named locals of a lowered function, with the types lowering derived
/// for them. This is what a hover reads, and it comes from the same IR the
/// backends emit, so the answer is the truth rather than a second guess.
pub fn locals_of(module: &IrModule, function: &str) -> Option<Vec<LocalInfo>> {
    let held = module
        .functions
        .iter()
        .find(|candidate| candidate.name == function)?;
    Some(
        held.locals
            .iter()
            .filter_map(|local| {
                let name = local.name.clone()?;
                Some(LocalInfo {
                    name,
                    ty: local.ty.clone(),
                    position: local.position,
                })
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Lexer, Parser};

    fn parsed(
        source: &str,
    ) -> (crate::ast::Module, std::collections::HashSet<String>) {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        let module = parser.parse().unwrap();
        let linear = parser.linear_types().clone();
        (module, linear)
    }

    const SOURCE: &str = "\
        Point :: struct { x: i64, y: i64 }\n\
        Shape :: enum { Dot, Box { w: i64, h: i64 } }\n\
        LIMIT :: 32\n\
        area :: fn(w: i64, h: i64) -> i64 {\n\
            total := w * h\n\
            total\n\
        }\n";

    // Ctrl+T: the outline of a module, each entry naming its declaration.
    #[test]
    fn the_symbols_of_a_module_answer_in_source_order() {
        let (module, _) = parsed(SOURCE);
        let found = symbols(&module.ast, &module.roots);
        let names: Vec<(&str, SymbolKind)> = found
            .iter()
            .map(|held| (held.name.as_str(), held.kind))
            .collect();
        assert_eq!(
            names,
            vec![
                ("Point", SymbolKind::Struct),
                ("Shape", SymbolKind::Enum),
                ("LIMIT", SymbolKind::Constant),
                ("area", SymbolKind::Function),
            ]
        );
        assert_eq!(found[3].position.line, 4);
    }

    // Ctrl+click: a name resolves to the line that declares it.
    #[test]
    fn a_definition_lookup_answers_the_declaration_site() {
        let (module, _) = parsed(SOURCE);
        let found = definition_of(&module.ast, &module.roots, "Shape").unwrap();
        assert_eq!(found.kind, SymbolKind::Enum);
        assert_eq!(found.position.line, 2);
    }

    // Member completion: the fields a struct or a variant payload offers.
    #[test]
    fn the_fields_of_a_name_answer_with_their_types() {
        let (module, _) = parsed(SOURCE);
        let fields = fields_of(&module.ast, &module.roots, "Point").unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].0, "x");
        assert!(matches!(fields[0].1, Type::I64));
        let payload =
            fields_of(&module.ast, &module.roots, "Shape::Box").unwrap();
        assert_eq!(payload.len(), 2);
        assert_eq!(payload[1].0, "h");
    }

    // Hover: the type lowering derived for a local, off the same IR the
    // backends emit.
    #[test]
    fn the_type_of_a_local_answers_from_the_lowered_ir() {
        let (mut module, linear) = parsed(SOURCE);
        let lowered = crate::ir::build::build_module(
            &mut module.ast,
            &module.roots,
            &linear,
        )
        .unwrap();
        let locals = locals_of(&lowered, "area").unwrap();
        let total = locals
            .iter()
            .find(|held| held.name == "total")
            .expect("the body names 'total'");
        assert!(matches!(total.ty, Type::I64));
        assert_eq!(total.position.line, 5);
    }
}
