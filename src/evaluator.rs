use crate::{flatten, Block, Expression, Identifier, Literal, Operator, Statement};
use anyhow::{bail, Context, Result};
use std::{
    cell::RefCell,
    collections::HashMap,
    fmt::{Display, Formatter, Result as FmtResult},
    rc::Rc,
};

#[derive(Debug, PartialEq, Clone)]
pub enum Object {
    Empty,
    Null,
    Integer(i64),
    Boolean(bool),
    Return(Box<Object>),
    Function(Vec<Identifier>, Block, Rc<RefCell<Environment>>),
}

impl Display for Object {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let statement = match self {
            Self::Empty => "".to_string(),
            Self::Null => "null".to_string(),
            Self::Integer(integer) => integer.to_string(),
            Self::Boolean(boolean) => boolean.to_string(),
            Self::Return(value) => value.to_string(),
            Self::Function(parameters, body, _environment) => {
                format!(
                    "fn({}) {{ {} }}",
                    flatten(&parameters, ", "),
                    flatten(body, "\n"),
                )
            }
        };
        write!(f, "{}", statement)
    }
}

#[derive(Debug, Default, PartialEq, Clone)]
pub struct Environment {
    pub bindings: HashMap<String, Object>,
    pub outer: Option<Rc<RefCell<Environment>>>,
}

impl Environment {
    pub fn new(outer: Option<Rc<RefCell<Environment>>>) -> Self {
        Self {
            outer,
            ..Default::default()
        }
    }

    pub fn new_rc(outer: Option<Rc<RefCell<Environment>>>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self::new(outer)))
    }

    pub fn set_binding(&mut self, binding: String, value: Object) {
        self.bindings.insert(binding, value);
    }

    pub fn get_binding(&self, binding: String) -> Result<Object> {
        if let Some(binding) = self.bindings.get(&binding) {
            return Ok(binding.clone());
        }

        if let Some(outer) = self.outer.as_ref() {
            return outer.borrow().get_binding(binding);
        }

        bail!("Binding not found: {:?}", binding)
    }
}

pub fn evaluate_statements(
    statements: &[Statement],
    environment: Rc<RefCell<Environment>>,
) -> Result<Object> {
    let mut result = Object::Null;
    for statement in statements.iter() {
        match evaluate_statement(statement, environment.clone())? {
            Object::Return(value) => return Ok(Object::Return(value)),
            object => result = object,
        }
    }
    Ok(result)
}

fn evaluate_statement(
    statement: &Statement,
    environment: Rc<RefCell<Environment>>,
) -> Result<Object> {
    Ok(match statement {
        Statement::Let(identifier, expression) => {
            let value = evaluate_expression(expression, environment.clone())?;
            environment
                .borrow_mut()
                .bindings
                .insert(identifier.to_string(), value);
            Object::Empty
        }
        Statement::Expression(expression) => evaluate_expression(expression, environment)?,
        Statement::Return(expression) => {
            Object::Return(Box::new(evaluate_expression(expression, environment)?))
        }
    })
}

fn evaluate_expressions(
    expressions: &[Expression],
    environment: Rc<RefCell<Environment>>,
) -> Result<Vec<Object>> {
    expressions
        .iter()
        .map(|expression| evaluate_expression(expression, environment.clone()))
        .collect()
}

fn evaluate_expression(
    expression: &Expression,
    environment: Rc<RefCell<Environment>>,
) -> Result<Object> {
    Ok(match expression {
        Expression::Function(parameters, body) => Object::Function(
            parameters.to_vec(),
            body.to_vec(),
            Environment::new_rc(Some(environment)),
        ),
        Expression::Call(function, arguments) => {
            let function = evaluate_expression(function, environment.clone())?;

            match function {
                Object::Function(parameters, body, inner_environment) => {
                    environment.borrow_mut().outer = Some(inner_environment);

                    let arguments = evaluate_expressions(arguments, environment.clone())?;
                    for (argument, name) in arguments.into_iter().zip(parameters.into_iter()) {
                        environment.borrow_mut().set_binding(name, argument);
                    }

                    let result = evaluate_statements(&body, environment)?;
                    match result {
                        Object::Return(value) => return Ok(*value),
                        _ => return Ok(result),
                    }
                }
                _ => bail!("'{}' is not a defined function", function),
            }
        }
        Expression::Identifier(identifier) => environment
            .borrow()
            .bindings
            .get(identifier)
            .context(format!("Identifier '{}' not found", identifier))?
            .clone(),
        Expression::Literal(literal) => evaluate_literal(literal)?,
        Expression::Boolean(boolean) => Object::Boolean(*boolean),
        Expression::Prefix(operator, expression) => {
            evaluate_prefix_expression(operator, expression, environment)?
        }
        Expression::Infix(left_expression, operator, right_expression) => {
            evaluate_infix_expression(left_expression, operator, right_expression, environment)?
        }
        Expression::If(condition, consequence, alternative) => {
            evaluate_if_expression(condition, consequence, alternative, environment)?
        }
    })
}

fn evaluate_literal(literal: &Literal) -> Result<Object> {
    Ok(match literal {
        Literal::Integer(integer) => Object::Integer(*integer),
        _ => Object::Null,
    })
}

fn evaluate_prefix_expression(
    operator: &Operator,
    expression: &Expression,
    environment: Rc<RefCell<Environment>>,
) -> Result<Object> {
    let value = evaluate_expression(expression, environment)?;
    Ok(match operator {
        Operator::Not => Object::Boolean(!object_to_bool(&value)),
        Operator::Negate => apply_operator_negate(&value)?,
        _ => Object::Null,
    })
}

fn evaluate_infix_expression(
    left_expression: &Expression,
    operator: &Operator,
    right_expression: &Expression,
    environment: Rc<RefCell<Environment>>,
) -> Result<Object> {
    let left_value = evaluate_expression(left_expression, environment.clone())?;
    let right_value = evaluate_expression(right_expression, environment)?;

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

fn evaluate_if_expression(
    condition: &Expression,
    consequence: &[Statement],
    alternative: &Option<Vec<Statement>>,
    environment: Rc<RefCell<Environment>>,
) -> Result<Object> {
    let condition = evaluate_expression(condition, environment.clone())?;

    if object_to_bool(&condition) {
        evaluate_statements(consequence, environment)
    } else {
        match alternative.as_ref() {
            Some(alternative) => evaluate_statements(alternative, environment),
            None => Ok(Object::Null),
        }
    }
}

fn object_to_bool(object: &Object) -> bool {
    match object {
        Object::Null => false,
        Object::Integer(_) => true,
        Object::Boolean(boolean) => *boolean,
        _ => false,
    }
}

fn apply_operator_negate(object: &Object) -> Result<Object> {
    Ok(match object {
        Object::Integer(value) => Object::Integer(-value),
        _ => bail!("Attempted to negate a non-integer value!"),
    })
}

#[cfg(test)]
mod tests {
    use super::Result;
    use crate::{
        evaluate_statements, Environment, Expression, Lexer, Literal, Object, Operator, Parser,
        Statement,
    };

    fn evaluate_tests(tests: &[(&str, Object)]) -> Result<()> {
        for (input, expected_value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            let environment = Environment::new_rc(None);
            let object = evaluate_statements(&program, environment)?;

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

    #[test]
    fn function_object() -> Result<()> {
        let tests = [(
            "fn(x) { x + 2; };",
            Object::Function(
                vec!["x".to_string()],
                vec![Statement::Expression(Expression::Infix(
                    Box::new(Expression::Identifier("x".to_string())),
                    Operator::Add,
                    Box::new(Expression::Literal(Literal::Integer(2))),
                ))],
                Environment::new_rc(Some(Environment::new_rc(None))),
            ),
        )];
        evaluate_tests(&tests)
    }

    #[test]
    fn function_application() -> Result<()> {
        let tests = [
            (
                "let identity = fn(x) { x; }; identity(5);",
                Object::Integer(5),
            ),
            (
                "let identity = fn(x) { return x; }; identity(5);",
                Object::Integer(5),
            ),
            (
                "let double = fn(x) { x * 2; }; double(5);",
                Object::Integer(10),
            ),
            (
                "let add = fn(x, y) { x + y; }; add(5, 5);",
                Object::Integer(10),
            ),
            (
                "let add = fn(x, y) { x + y; }; add(5 + 5, add(5, 5));",
                Object::Integer(20),
            ),
            ("fn(x) { x; }(5)", Object::Integer(5)),
        ];
        evaluate_tests(&tests)
    }

    #[test]
    fn closures() -> Result<()> {
        let tests = [(
            r"
let newAdder = fn(x) {
fn(y) { x + y };
};
let addTwo = newAdder(2);
addTwo(2);",
            Object::Integer(4),
        )];
        evaluate_tests(&tests)
    }
}
