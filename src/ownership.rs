use anyhow::{Result, bail};

use crate::parser::{Expression, Statement};

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
            Expression::Function(_, return_sig, body),
        )
        | Statement::Constant(name, Expression::Proc(_, return_sig, body)) => {
            if let Some(reference) = return_sig.contains_reference() {
                bail!(
                    "ownership: function '{name}' cannot return the reference type '{reference}'; references are second-class"
                );
            }
            check_ownership(body)?;
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
}
