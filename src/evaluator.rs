use crate::{Expression, Literal, Statement};
use anyhow::Result;

#[derive(Debug, PartialEq)]
pub enum Object {
    Null,
    Integer(i64),
    Boolean(bool),
}

pub fn evaluate_program(statements: &[Statement]) -> Result<Object> {
    let mut result = Object::Null;
    for statement in statements.iter() {
        result = evaluate_statement(statement)?;
    }
    Ok(result)
}

pub fn evaluate_statement(statement: &Statement) -> Result<Object> {
    Ok(match statement {
        Statement::Expression(expression) => evaluate_expression(expression)?,
        _ => Object::Null,
    })
}

pub fn evaluate_expression(expression: &Expression) -> Result<Object> {
    Ok(match expression {
        Expression::Literal(literal) => evaluate_literal(literal)?,
        Expression::Boolean(boolean) => Object::Boolean(*boolean),
        _ => Object::Null,
    })
}

pub fn evaluate_literal(literal: &Literal) -> Result<Object> {
    Ok(match literal {
        Literal::Integer(integer) => Object::Integer(*integer),
        _ => Object::Null,
    })
}

#[cfg(test)]
mod tests {
    use super::Result;
    use crate::{evaluate_program, Lexer, Object, Parser};

    #[test]
    fn test() -> Result<()> {
        let tests = [("5", 5_i64), ("10", 10_i64)];

        for (input, expected_value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            let object = evaluate_program(&program)?;

            assert_eq!(object, Object::Integer(*expected_value));
        }

        Ok(())
    }
}
