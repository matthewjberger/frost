use crate::{flatten, Block, Expression, Identifier, Literal, Operator, Statement};
use anyhow::{bail, Context, Result};
use std::{
    cell::RefCell,
    collections::HashMap,
    fmt,
    fmt::{Display, Formatter, Result as FmtResult},
    rc::Rc,
};

#[derive(Clone)]
pub struct BuiltInFunction {
    pub name: String,
    pub action: Rc<RefCell<dyn Fn(Vec<Object>) -> Result<Object>>>,
}

impl fmt::Debug for BuiltInFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuiltInFunction")
            .field("name", &self.name)
            .finish()
    }
}

impl PartialEq for BuiltInFunction {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Object {
    Empty,
    Null,
    Integer(i64),
    Array(Vec<Object>),
    Boolean(bool),
    String(String),
    Return(Box<Object>),
    Function(Vec<Identifier>, Block, Rc<RefCell<Environment>>),
    BuiltInFunction(BuiltInFunction),
}

impl Display for Object {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let statement = match self {
            Self::Empty => "".to_string(),
            Self::Null => "null".to_string(),
            Self::Integer(integer) => integer.to_string(),
            Self::Array(objects) => {
                let objects = objects
                    .iter()
                    .map(|e| {
                        if let Object::String(_) = e {
                            format!("\"{}\"", e.to_string())
                        } else {
                            e.to_string()
                        }
                    })
                    .collect::<Vec<_>>();
                format!("[{}]", objects.join(", "))
            }
            Self::Boolean(boolean) => boolean.to_string(),
            Self::String(string) => string.to_string(),
            Self::Return(value) => value.to_string(),
            Self::Function(parameters, body, _environment) => {
                format!(
                    "fn({}) {{ {} }}",
                    flatten(&parameters, ", "),
                    flatten(body, "\n"),
                )
            }
            Self::BuiltInFunction(builtin_function) => {
                format!("BuiltIn function '{}'", builtin_function.name)
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
            evaluate_call_expression(environment, function, arguments)?
        }
        Expression::Index(left_expression, index_expression) => {
            evaluate_index_expression(environment, left_expression, index_expression)?
        }
        Expression::Identifier(identifier) => {
            let builtin_functions = builtin_functions();
            if let Some(identifier) = environment.borrow().bindings.get(identifier) {
                identifier.clone()
            } else if let Some(identifier) = builtin_functions.bindings.get(identifier) {
                identifier.clone()
            } else {
                bail!("Identifier '{}' not found", identifier)
            }
        }
        Expression::Literal(literal) => evaluate_literal(literal, environment)?,
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

fn evaluate_literal(literal: &Literal, environment: Rc<RefCell<Environment>>) -> Result<Object> {
    Ok(match literal {
        Literal::Integer(integer) => Object::Integer(*integer),
        Literal::String(string) => Object::String(string.to_string()),
        Literal::Array(elements) => {
            let elements = elements
                .iter()
                .map(|expression| evaluate_expression(expression, environment.clone()))
                .collect::<Result<Vec<_>>>()?;
            Object::Array(elements)
        }
    })
}

fn evaluate_call_expression(
    environment: Rc<RefCell<Environment>>,
    function: &Expression,
    arguments: &[Expression],
) -> Result<Object> {
    let function = evaluate_expression(function, environment.clone())?;
    match function {
        Object::Function(parameters, body, function_environment) => {
            environment.borrow_mut().outer = Some(function_environment);
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
        Object::BuiltInFunction(function) => {
            let arguments = evaluate_expressions(arguments, environment.clone())?;
            let action = function.action.borrow();
            action(arguments)
        }
        _ => bail!("'{}' is not a defined function", function),
    }
}

fn evaluate_index_expression(
    environment: Rc<RefCell<Environment>>,
    left_expression: &Expression,
    index_expression: &Expression,
) -> Result<Object> {
    let identifier = evaluate_expression(left_expression, environment.clone())?;
    let index = evaluate_expression(index_expression, environment.clone())?;

    if !matches!(identifier, Object::Array(_)) {
        bail!(
            "Identifier '{}' is not an array. Index expressions are only valid for arrays.",
            identifier
        )
    }

    if !matches!(index, Object::Integer(_)) {
        bail!("Arrays can only be indexed by integers")
    }

    if let (Object::Array(elements), Object::Integer(index)) = (identifier, index) {
        if index < 0 {
            return Ok(Object::Null);
        }
        match elements.get(index as usize) {
            Some(element) => Ok(element.clone()),
            None => Ok(Object::Null),
        }
    } else {
        bail!("Index expression is invalid!")
    }
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

    // Integer x Integer
    if let (Object::Integer(lhs), Object::Integer(rhs)) = (&left_value, &right_value) {
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
                "Operator '{}' is not valid for int-int infix expressions",
                operator
            ),
        });
    }

    // Boolean x Boolean
    if let (Object::Boolean(lhs), Object::Boolean(rhs)) = (&left_value, &right_value) {
        return Ok(match operator {
            Operator::Equal => Object::Boolean(lhs == rhs),
            Operator::NotEqual => Object::Boolean(lhs != rhs),
            _ => bail!(
                "Operator '{}' is not valid for bool-bool infix expressions",
                operator
            ),
        });
    }

    // String x String
    if let (Object::String(lhs), Object::String(rhs)) = (left_value, right_value) {
        return Ok(match operator {
            Operator::Equal => Object::Boolean(lhs == rhs),
            Operator::NotEqual => Object::Boolean(lhs != rhs),
            Operator::Add => Object::String(format!("{}{}", lhs, rhs)),
            _ => bail!(
                "Operator '{}' is not valid for string-string infix expressions",
                operator
            ),
        });
    }

    bail!("Could not evaluate infix expression that wasn't bool-bool, int-int, or string-string")
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

// TODO: Store this somewhere...
fn builtin_functions() -> Environment {
    let mut environment = Environment::new(None);
    environment.set_binding("len".to_string(), builtin_len());
    environment
}

fn builtin_len() -> Object {
    Object::BuiltInFunction(BuiltInFunction {
        name: "len".to_string(),
        action: Rc::new(RefCell::new(|args: Vec<Object>| {
            if args.len() > 1 {
                bail!("Too many arguments to 'len'")
            }

            let arg = args.first().context("No arguments were passed to 'len'!")?;

            match arg {
                Object::String(value) => Ok(Object::Integer(value.len() as _)),
                Object::Array(value) => Ok(Object::Integer(value.len() as _)),
                _ => bail!("Invalid type was provided to len function!"),
            }
        })),
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
    fn evaluate_array_literals() -> Result<()> {
        let tests = [(
            "[1, 2 * 2, 3 + 3]",
            Object::Array(vec![
                Object::Integer(1),
                Object::Integer(4),
                Object::Integer(6),
            ]),
        )];
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

    #[test]
    fn string_literals() -> Result<()> {
        let phrase = "Hello World!";
        let quoted = format!("\"{}\"", phrase);
        let tests = [(quoted.as_str(), Object::String(phrase.to_string()))];
        evaluate_tests(&tests)
    }

    #[test]
    fn string_concatenation() -> Result<()> {
        let tests = [(
            "\"Hello\" + \" \" + \"World!\"",
            Object::String("Hello World!".to_string()),
        )];
        evaluate_tests(&tests)
    }

    #[test]
    fn builtin_functions() -> Result<()> {
        let tests = [
            ("len(\"\")", Object::Integer(0)),
            ("len(\"four\")", Object::Integer(4)),
            ("len(\"hello world\")", Object::Integer(11)),
            ("len([])", Object::Integer(0)),
            ("len([1])", Object::Integer(1)),
            ("len([1, 2])", Object::Integer(2)),
            ("len([1, 2 + 18, 3 * 6, 4, \"hi\"])", Object::Integer(5)),
        ];
        evaluate_tests(&tests)
    }

    #[test]
    fn array_index_expressions() -> Result<()> {
        let tests = [
            ("[1, 2, 3][0]", Object::Integer(1)),
            ("[1, 2, 3][1]", Object::Integer(2)),
            ("[1, 2, 3][2]", Object::Integer(3)),
            ("let i = 0; [1][i];", Object::Integer(1)),
            ("[1, 2, 3][1 + 1];", Object::Integer(3)),
            ("let myArray = [1, 2, 3]; myArray[2];", Object::Integer(3)),
            (
                "let myArray = [1, 2, 3]; myArray[0] + myArray[1] + myArray[2];",
                Object::Integer(6),
            ),
            (
                "let myArray = [1, 2, 3]; let i = myArray[0]; myArray[i]",
                Object::Integer(2),
            ),
            ("[1, 2, 3][3]", Object::Null),
            ("[1, 2, 3][-1]", Object::Null),
        ];
        evaluate_tests(&tests)
    }
}
