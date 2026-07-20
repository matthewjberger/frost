use std::collections::{HashMap, HashSet};

use anyhow::{Result, bail};

use crate::parser::{Block, Expression, Literal, Parameter, Statement};
use crate::types::Type;

pub fn check_ownership(statements: &[Statement]) -> Result<()> {
    for statement in statements {
        check_statement(statement)?;
    }
    Ok(())
}

fn check_statement(statement: &Statement) -> Result<()> {
    match statement {
        Statement::Struct(name, _, fields) => {
            for field in fields {
                if field.field_type.contains_reference() {
                    bail!(
                        "ownership: cannot store a reference in struct '{name}' (field '{}'); references are second-class",
                        field.name
                    );
                }
            }
        }
        Statement::Enum(name, variants) => {
            for variant in variants {
                let Some(fields) = &variant.fields else {
                    continue;
                };
                for field in fields {
                    if field.field_type.contains_reference() {
                        bail!(
                            "ownership: cannot store a reference in enum '{name}' (variant '{}', field '{}'); references are second-class",
                            variant.name,
                            field.name
                        );
                    }
                }
            }
        }
        Statement::Constant(
            name,
            Expression::Function(params, return_sig, body),
        )
        | Statement::Constant(
            name,
            Expression::Proc(params, return_sig, body),
        ) => {
            if let Some(reference) = return_sig.contains_reference() {
                bail!(
                    "ownership: function '{name}' cannot return the reference type '{reference}'; references are second-class"
                );
            }
            check_ownership(body)?;
            check_function_moves(params, body)?;
        }
        Statement::Extern {
            name, return_type, ..
        } => {
            if let Some(return_type) = return_type
                && return_type.contains_reference()
            {
                bail!(
                    "ownership: extern function '{name}' cannot return a reference"
                );
            }
        }
        _ => {}
    }
    Ok(())
}

fn check_function_moves(params: &[Parameter], body: &Block) -> Result<()> {
    let mut checker = MoveChecker {
        types: HashMap::new(),
        moved: HashSet::new(),
    };
    for parameter in params {
        if let Some(ty) = &parameter.type_annotation {
            checker.types.insert(parameter.name.clone(), ty.clone());
        }
    }
    checker.check_block(body)
}

struct MoveChecker {
    types: HashMap<String, Type>,
    moved: HashSet<String>,
}

impl MoveChecker {
    fn check_block(&mut self, block: &[Statement]) -> Result<()> {
        for statement in block {
            self.check_statement(statement)?;
        }
        Ok(())
    }

    fn check_statement(&mut self, statement: &Statement) -> Result<()> {
        match statement {
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                self.visit(value, true)?;
                let inferred =
                    infer_type(type_annotation.as_ref(), value, &self.types);
                self.moved.remove(name);
                match inferred {
                    Some(ty) => {
                        self.types.insert(name.clone(), ty);
                    }
                    None => {
                        self.types.remove(name);
                    }
                }
                Ok(())
            }
            Statement::Constant(
                _,
                Expression::Function(..) | Expression::Proc(..),
            ) => Ok(()),
            Statement::Constant(name, value) => {
                self.visit(value, true)?;
                let inferred = infer_type(None, value, &self.types);
                self.moved.remove(name);
                if let Some(ty) = inferred {
                    self.types.insert(name.clone(), ty);
                }
                Ok(())
            }
            Statement::Assignment(target, value) => {
                self.visit(value, true)?;
                if let Expression::Identifier(name) = target {
                    self.moved.remove(name);
                } else {
                    self.visit(target, false)?;
                }
                Ok(())
            }
            Statement::Return(expression) => self.visit(expression, true),
            Statement::Expression(expression) => self.visit(expression, false),
            Statement::While(condition, body) => {
                self.visit(condition, false)?;
                self.check_block(body)
            }
            Statement::For(variable, range, body) => {
                self.visit(range, false)?;
                self.types.insert(variable.clone(), Type::I64);
                self.check_block(body)
            }
            Statement::Defer(inner) => self.check_statement(inner),
            _ => Ok(()),
        }
    }

    fn visit(&mut self, expression: &Expression, moving: bool) -> Result<()> {
        match expression {
            Expression::Identifier(name) => {
                if self.moved.contains(name) {
                    bail!("ownership: use of moved value '{name}'");
                }
                if moving && self.is_move_variable(name) {
                    self.moved.insert(name.clone());
                }
                Ok(())
            }
            Expression::Borrow(inner)
            | Expression::BorrowMut(inner)
            | Expression::AddressOf(inner)
            | Expression::Dereference(inner) => self.visit(inner, false),
            Expression::FieldAccess(base, _) => self.visit(base, false),
            Expression::Index(base, index) => {
                self.visit(base, false)?;
                self.visit(index, false)
            }
            Expression::Prefix(_, operand) => self.visit(operand, false),
            Expression::Infix(left, _, right) => {
                self.visit(left, false)?;
                self.visit(right, false)
            }
            Expression::Call(callee, arguments) => {
                self.visit(callee, false)?;
                for argument in arguments {
                    self.visit(argument, true)?;
                }
                Ok(())
            }
            Expression::StructInit(_, fields) => {
                for (_, value) in fields {
                    self.visit(value, true)?;
                }
                Ok(())
            }
            Expression::EnumVariantInit(_, _, fields) => {
                for (_, value) in fields {
                    self.visit(value, true)?;
                }
                Ok(())
            }
            Expression::Literal(Literal::Array(elements)) => {
                for element in elements {
                    self.visit(element, true)?;
                }
                Ok(())
            }
            Expression::If(condition, consequence, alternative) => {
                self.visit(condition, false)?;
                self.check_block(consequence)?;
                if let Some(alternative) = alternative {
                    self.check_block(alternative)?;
                }
                Ok(())
            }
            Expression::Switch(scrutinee, cases) => {
                self.visit(scrutinee, false)?;
                for case in cases {
                    self.check_block(&case.body)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn is_move_variable(&self, name: &str) -> bool {
        self.types
            .get(name)
            .map(|ty| !ty.is_copy())
            .unwrap_or(false)
    }
}

fn infer_type(
    annotation: Option<&Type>,
    value: &Expression,
    types: &HashMap<String, Type>,
) -> Option<Type> {
    if let Some(ty) = annotation {
        return Some(ty.clone());
    }
    match value {
        Expression::StructInit(name, _) => Some(Type::Struct(name.clone())),
        Expression::EnumVariantInit(name, _, _) => {
            Some(Type::Enum(name.clone()))
        }
        Expression::Literal(Literal::String(_)) => Some(Type::Str),
        Expression::Literal(Literal::Integer(_)) => Some(Type::I64),
        Expression::Literal(Literal::Float(_)) => Some(Type::F64),
        Expression::Literal(Literal::Float32(_)) => Some(Type::F32),
        Expression::Literal(Literal::Boolean(_)) | Expression::Boolean(_) => {
            Some(Type::Bool)
        }
        Expression::Identifier(name) => types.get(name).cloned(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Lexer, Parser};

    fn check(source: &str) -> Result<()> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let statements = parser.parse()?;
        check_ownership(&statements)
    }

    #[test]
    fn reference_in_struct_is_rejected() {
        let source = "Bad :: struct { r: &i64 }";
        assert!(check(source).is_err());
    }

    #[test]
    fn reference_in_enum_is_rejected() {
        let source = "Bad :: enum { Holder { r: &mut i64 } }";
        assert!(check(source).is_err());
    }

    #[test]
    fn returning_a_reference_is_rejected() {
        let source = "bad :: fn(x: &i64) -> &i64 { x }";
        assert!(check(source).is_err());
    }

    #[test]
    fn reference_parameters_are_allowed() {
        let source = "read :: fn(x: &i64) -> i64 { x^ }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn owned_struct_is_allowed() {
        let source = "Point :: struct { x: i64, y: i64 }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn handles_can_be_stored_and_returned() {
        let source = "Store :: struct { h: Handle<i64> }\nget :: fn() -> Handle<i64> { make() }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn use_after_move_of_struct_is_rejected() {
        let source = "\
            Point :: struct { x: i64, y: i64 }\n\
            take :: fn(p: Point) -> i64 { p.x }\n\
            run :: fn() -> i64 {\n\
                p := Point { x = 1, y = 2 }\n\
                a := take(p)\n\
                b := take(p)\n\
                a + b\n\
            }";
        assert!(check(source).is_err());
    }

    #[test]
    fn copy_values_can_be_reused() {
        let source = "\
            add :: fn(a: i64, b: i64) -> i64 { a + b }\n\
            run :: fn() -> i64 {\n\
                x : i64 = 5\n\
                a := add(x, x)\n\
                b := add(x, x)\n\
                a + b\n\
            }";
        assert!(check(source).is_ok());
    }

    #[test]
    fn borrowing_does_not_move() {
        let source = "\
            Point :: struct { x: i64, y: i64 }\n\
            read :: fn(p: &Point) -> i64 { p.x }\n\
            run :: fn() -> i64 {\n\
                p := Point { x = 1, y = 2 }\n\
                a := read(&p)\n\
                b := read(&p)\n\
                a + b\n\
            }";
        assert!(check(source).is_ok());
    }
}
