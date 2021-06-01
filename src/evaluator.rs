use crate::{Expression, Literal, Operator, Statement};
use anyhow::{bail, Context, Result};
use std::{
    collections::HashMap,
    fmt::{Display, Formatter, Result as FmtResult},
};

#[derive(Debug, PartialEq, Clone)]
pub enum Object {
    Null,
    Integer(i64),
    Boolean(bool),
    Return(Box<Object>),
}

impl Display for Object {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let statement = match self {
            Self::Null => "null".to_string(),
            Self::Integer(integer) => integer.to_string(),
            Self::Boolean(boolean) => boolean.to_string(),
            Self::Return(value) => value.to_string(),
        };
        write!(f, "{}", statement)
    }
}

#[derive(Debug, Default)]
pub struct Environment {
    pub bindings: HashMap<String, Object>,
}

#[derive(Debug, Default)]
pub struct Evaluator {
    environment: Environment,
}

impl Evaluator {
    pub fn evaluate_program(&mut self, statements: &[Statement]) -> Result<Object> {
        let mut result = Object::Null;
        for statement in statements.iter() {
            match self.evaluate_statement(statement)? {
                Object::Return(value) => return Ok(Object::Return(value)),
                object => result = object,
            }
        }
        Ok(result)
    }

    pub fn evaluate_statement(&mut self, statement: &Statement) -> Result<Object> {
        Ok(match statement {
            Statement::Let(identifier, expression) => {
                let value = self.evaluate_expression(expression)?;
                self.environment
                    .bindings
                    .insert(identifier.to_string(), value.clone());
                value
            }
            Statement::Expression(expression) => self.evaluate_expression(expression)?,
            Statement::Return(expression) => {
                Object::Return(Box::new(self.evaluate_expression(expression)?))
            }
        })
    }

    pub fn evaluate_expression(&mut self, expression: &Expression) -> Result<Object> {
        Ok(match expression {
            Expression::Identifier(identifier) => self
                .environment
                .bindings
                .get(identifier)
                .context(format!("Identifier '{}' not found", identifier))?
                .clone(),
            Expression::Literal(literal) => self.evaluate_literal(literal)?,
            Expression::Boolean(boolean) => Object::Boolean(*boolean),
            Expression::Prefix(operator, expression) => {
                self.evaluate_prefix_expression(operator, expression)?
            }
            Expression::Infix(left_expression, operator, right_expression) => {
                self.evaluate_infix_expression(left_expression, operator, right_expression)?
            }
            Expression::If(condition, consequence, alternative) => {
                self.evaluate_if_expression(condition, consequence, alternative)?
            }
            _ => Object::Null,
        })
    }

    pub fn evaluate_literal(&self, literal: &Literal) -> Result<Object> {
        Ok(match literal {
            Literal::Integer(integer) => Object::Integer(*integer),
            _ => Object::Null,
        })
    }

    pub fn evaluate_prefix_expression(
        &mut self,
        operator: &Operator,
        expression: &Expression,
    ) -> Result<Object> {
        let value = self.evaluate_expression(expression)?;
        Ok(match operator {
            Operator::Not => Object::Boolean(!self.object_to_bool(&value)),
            Operator::Negate => self.apply_operator_negate(&value)?,
            _ => Object::Null,
        })
    }

    pub fn evaluate_infix_expression(
        &mut self,
        left_expression: &Expression,
        operator: &Operator,
        right_expression: &Expression,
    ) -> Result<Object> {
        let left_value = self.evaluate_expression(left_expression)?;
        let right_value = self.evaluate_expression(right_expression)?;

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

        bail!("Could not evaluate infix expression that wasn't bool-bool or int-int")
    }

    pub fn evaluate_if_expression(
        &mut self,
        condition: &Expression,
        consequence: &[Statement],
        alternative: &Option<Vec<Statement>>,
    ) -> Result<Object> {
        let condition = self.evaluate_expression(condition)?;

        if self.object_to_bool(&condition) {
            self.evaluate_program(consequence)
        } else {
            match alternative.as_ref() {
                Some(alternative) => self.evaluate_program(alternative),
                None => Ok(Object::Null),
            }
        }
    }

    pub fn object_to_bool(&self, object: &Object) -> bool {
        match object {
            Object::Null => false,
            Object::Integer(_) => true,
            Object::Boolean(boolean) => *boolean,
            _ => false,
        }
    }

    pub fn apply_operator_negate(&self, object: &Object) -> Result<Object> {
        Ok(match object {
            Object::Integer(value) => Object::Integer(-value),
            _ => bail!("Attempted to negate a non-integer value!"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Result;
    use crate::{Evaluator, Lexer, Object, Parser};

    fn evaluate_tests(tests: &[(&str, Object)]) -> Result<()> {
        for (input, expected_value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            let mut evaluator = Evaluator::default();
            let object = evaluator.evaluate_program(&program)?;

            assert_eq!(object, *expected_value);
        }
        Ok(())
    }

    #[test]
    fn evaluate_integer_literals() -> Result<()> {
        let tests = [
            ("5", Object::Integer(5_i64)),
            ("10", Object::Integer(10_i64)),
            ("-5", Object::Integer(-5_i64)),
            ("-10", Object::Integer(-10_i64)),
            ("5 + 5 + 5 + 5 - 10", Object::Integer(10)),
            ("2 * 2 * 2 * 2 * 2", Object::Integer(32)),
            ("-50 + 100 + -50", Object::Integer(0)),
            ("5 * 2 + 10", Object::Integer(20)),
            ("5 + 2 * 10", Object::Integer(25)),
            ("20 + 2 * -10", Object::Integer(0)),
            ("50 / 2 * 2 + 10", Object::Integer(60)),
            ("2 * (5 + 10)", Object::Integer(30)),
            ("3 * 3 * 3 + 10", Object::Integer(37)),
            ("3 * (3 * 3) + 10", Object::Integer(37)),
            ("(5 + 10 * 2 + 15 / 3) * 2 + -10", Object::Integer(50)),
        ];

        evaluate_tests(&tests)
    }

    #[test]
    fn evaluate_boolean_literals() -> Result<()> {
        let tests = [
            ("true", Object::Boolean(true)),
            ("false", Object::Boolean(false)),
        ];
        evaluate_tests(&tests)
    }

    #[test]
    fn not_operator() -> Result<()> {
        let tests = [
            ("!true", Object::Boolean(false)),
            ("!false", Object::Boolean(true)),
            ("!5", Object::Boolean(false)),
            ("!!true", Object::Boolean(true)),
            ("!!false", Object::Boolean(false)),
            ("!!5", Object::Boolean(true)),
        ];
        evaluate_tests(&tests)
    }

    #[test]
    fn evaluate_boolean_expressions() -> Result<()> {
        let tests = [
            ("true", Object::Boolean(true)),
            ("false", Object::Boolean(false)),
            ("1 < 2", Object::Boolean(true)),
            ("1 > 2", Object::Boolean(false)),
            ("1 < 1", Object::Boolean(false)),
            ("1 > 1", Object::Boolean(false)),
            ("1 == 1", Object::Boolean(true)),
            ("1 != 1", Object::Boolean(false)),
            ("1 == 2", Object::Boolean(false)),
            ("1 != 2", Object::Boolean(true)),
            ("true == true", Object::Boolean(true)),
            ("false == false", Object::Boolean(true)),
            ("true == false", Object::Boolean(false)),
            ("true != false", Object::Boolean(true)),
            ("false != true", Object::Boolean(true)),
            ("(1 < 2) ==  true ", Object::Boolean(true)),
            ("(1 < 2) ==  false ", Object::Boolean(false)),
            ("(1 > 2) ==  true ", Object::Boolean(false)),
            ("(1 > 2) ==  false ", Object::Boolean(true)),
        ];
        evaluate_tests(&tests)
    }

    #[test]
    fn if_else_expressions() -> Result<()> {
        let tests = [
            ("if (true) { 10 }", Object::Integer(10)),
            ("if (false) { 10 }", Object::Null),
            ("if (1) { 10 }", Object::Integer(10)),
            ("if (1 < 2) { 10 }", Object::Integer(10)),
            ("if (1 > 2) { 10 }", Object::Null),
            ("if (1 > 2) { 10 } else { 20 }", Object::Integer(20)),
            ("if (1 < 2) { 10 } else { 20 }", Object::Integer(10)),
        ];
        evaluate_tests(&tests)
    }

    #[test]
    fn return_statements() -> Result<()> {
        let tests = [
            ("return 10;", Object::Return(Box::new(Object::Integer(10)))),
            (
                "return 10; 9;",
                Object::Return(Box::new(Object::Integer(10))),
            ),
            (
                "return 2 * 5; 9;",
                Object::Return(Box::new(Object::Integer(10))),
            ),
            (
                "9; return 2 * 5; 9;",
                Object::Return(Box::new(Object::Integer(10))),
            ),
            (
                "if (10 > 1) { if (10 > 1) { return 10; } return 1; }",
                Object::Return(Box::new(Object::Integer(10))),
            ),
        ];
        evaluate_tests(&tests)
    }

    #[test]
    fn let_statements() -> Result<()> {
        let tests = [
            ("let a = 5; a;", Object::Integer(5)),
            ("let a = 5 * 5; a;", Object::Integer(25)),
            ("let a = 5; let b = a; b;", Object::Integer(5)),
            (
                "let a = 5; let b = a; let c = a + b + 5; c;",
                Object::Integer(15),
            ),
        ];
        evaluate_tests(&tests)
    }
}
