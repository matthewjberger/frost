use crate::{Expression, Literal, Operator, Statement};
use anyhow::{bail, Result};
use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Debug, PartialEq)]
pub enum Object {
    Null,
    Integer(i64),
    Boolean(bool),
}

impl Display for Object {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let statement = match self {
            Self::Null => "null".to_string(),
            Self::Integer(integer) => integer.to_string(),
            Self::Boolean(boolean) => boolean.to_string(),
        };
        write!(f, "{}", statement)
    }
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
        Expression::Prefix(operator, expression) => {
            evaluate_prefix_expression(operator, expression)?
        }
        _ => Object::Null,
    })
}

pub fn evaluate_literal(literal: &Literal) -> Result<Object> {
    Ok(match literal {
        Literal::Integer(integer) => Object::Integer(*integer),
        _ => Object::Null,
    })
}

pub fn evaluate_prefix_expression(operator: &Operator, expression: &Expression) -> Result<Object> {
    let value = evaluate_expression(expression)?;
    Ok(match operator {
        Operator::Not => apply_operator_not(&value)?,
        Operator::Negate => apply_operator_negate(&value)?,
        _ => Object::Null,
    })
}

pub fn apply_operator_not(object: &Object) -> Result<Object> {
    let boolean = match object {
        Object::Null => true,
        Object::Boolean(boolean) => !*boolean,
        _ => false,
    };
    Ok(Object::Boolean(boolean))
}

pub fn apply_operator_negate(object: &Object) -> Result<Object> {
    Ok(match object {
        Object::Integer(value) => Object::Integer(-value),
        _ => bail!("Attempted to negate a non-integer value!"),
    })
}

#[cfg(test)]
mod tests {
    use super::Result;
    use crate::{evaluate_program, Lexer, Object, Parser};

    #[test]
    fn evaluate_integer_literals() -> Result<()> {
        let tests = [
            ("5", 5_i64),
            ("10", 10_i64),
            ("-5", -5_i64),
            ("-10", -10_i64),
        ];

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

    #[test]
    fn evaluate_boolean_literals() -> Result<()> {
        let tests = [("true", true), ("false", false)];

        for (input, expected_value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            let object = evaluate_program(&program)?;

            assert_eq!(object, Object::Boolean(*expected_value));
        }

        Ok(())
    }

    #[test]
    fn not_operator() -> Result<()> {
        let tests = [
            ("!true", false),
            ("!false", true),
            ("!5", false),
            ("!!true", true),
            ("!!false", false),
            ("!!5", true),
        ];

        for (input, expected_value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            let object = evaluate_program(&program)?;

            assert_eq!(object, Object::Boolean(*expected_value));
        }

        Ok(())
    }
}
