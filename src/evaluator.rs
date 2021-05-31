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
        Expression::Infix(left_expression, operator, right_expression) => {
            evaluate_infix_expression(left_expression, operator, right_expression)?
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

pub fn evaluate_infix_expression(
    left_expression: &Expression,
    operator: &Operator,
    right_expression: &Expression,
) -> Result<Object> {
    let left_value = evaluate_expression(left_expression)?;
    let right_value = evaluate_expression(right_expression)?;

    if let Object::Integer(lhs) = left_value {
        if let Object::Integer(rhs) = right_value {
            return Ok(match operator {
                Operator::Add => Object::Integer(lhs + rhs),
                Operator::Divide => Object::Integer(lhs / rhs),
                Operator::Multiply => Object::Integer(lhs * rhs),
                Operator::Subtract => Object::Integer(lhs - rhs),
                Operator::LessThan => Object::Boolean(lhs < rhs),
                Operator::GreaterThan => Object::Boolean(lhs > rhs),
                Operator::Equal => Object::Boolean(lhs == rhs),
                Operator::NotEqual => Object::Boolean(lhs != rhs),
                _ => bail!(
                    "Operator '{}' is not valid for int<->int infix expressions",
                    operator
                ),
            });
        }
    }

    if let Object::Boolean(lhs) = left_value {
        if let Object::Boolean(rhs) = right_value {
            return Ok(match operator {
                Operator::Equal => Object::Boolean(lhs == rhs),
                Operator::NotEqual => Object::Boolean(lhs != rhs),
                _ => bail!(
                    "Operator '{}' is not valid for bool<->bool infix expressions",
                    operator
                ),
            });
        }
    }

    // TODO: Support infix on non integer values
    bail!("Could not evaluate infix expression that wasn't bool-bool or int-int")
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
            ("5 + 5 + 5 + 5 - 10", 10),
            ("2 * 2 * 2 * 2 * 2", 32),
            ("-50 + 100 + -50", 0),
            ("5 * 2 + 10", 20),
            ("5 + 2 * 10", 25),
            ("20 + 2 * -10", 0),
            ("50 / 2 * 2 + 10", 60),
            ("2 * (5 + 10)", 30),
            ("3 * 3 * 3 + 10", 37),
            ("3 * (3 * 3) + 10", 37),
            ("(5 + 10 * 2 + 15 / 3) * 2 + -10", 50),
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

    #[test]
    fn evaluate_boolean_expressions() -> Result<()> {
        let tests = [
            ("true", true),
            ("false", false),
            ("1 < 2", true),
            ("1 > 2", false),
            ("1 < 1", false),
            ("1 > 1", false),
            ("1 == 1", true),
            ("1 != 1", false),
            ("1 == 2", false),
            ("1 != 2", true),
            ("true == true", true),
            ("false == false", true),
            ("true == false", false),
            ("true != false", true),
            ("false != true", true),
            ("(1 < 2) == true", true),
            ("(1 < 2) == false", false),
            ("(1 > 2) == true", false),
            ("(1 > 2) == false", true),
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
