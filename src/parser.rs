use crate::{flatten, hash, lexer::Token};
use anyhow::{bail, Result};
use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    matches,
    slice::Iter,
};

pub type Identifier = String;

pub type Block = Vec<Statement>;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Operator {
    Add,
    Divide,
    Multiply,
    Not,
    Negate,
    Subtract,
    LessThan,
    GreaterThan,
    Equal,
    NotEqual,
}

impl Operator {
    pub fn from_token(token: &Token, is_prefix: bool) -> Result<Self> {
        Ok(match token {
            Token::Plus => Self::Add,
            Token::Slash => Self::Divide,
            Token::Asterisk => Self::Multiply,
            Token::Bang => Self::Not,
            Token::Minus if is_prefix => Self::Negate,
            Token::Minus if !is_prefix => Self::Subtract,
            Token::LessThan => Self::LessThan,
            Token::GreaterThan => Self::GreaterThan,
            Token::Equal => Self::Equal,
            Token::NotEqual => Self::NotEqual,
            _ => bail!("Token is not an operator: {}", token),
        })
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let statement = match self {
            Self::Add => "+",
            Self::Subtract | Self::Negate => "-",
            Self::Divide => "/",
            Self::Multiply => "*",
            Self::Not => "!",
            Self::LessThan => "<",
            Self::GreaterThan => ">",
            Self::Equal => "==",
            Self::NotEqual => "!=",
        };
        write!(f, "{}", statement)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    Let(Identifier, Expression),
    Return(Expression),
    Expression(Expression),
}

impl Display for Statement {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let statement = match self {
            Self::Let(identifier, expression) => format!("let {} = {};", identifier, expression),
            Self::Return(expression) => format!("return {};", expression),
            Self::Expression(expression) => expression.to_string(),
        };
        write!(f, "{}", statement)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    Identifier(Identifier),
    Literal(Literal),
    Boolean(bool),
    Prefix(Operator, Box<Expression>),
    Infix(Box<Expression>, Operator, Box<Expression>),
    If(Box<Expression>, Block, Option<Block>),
    Function(Vec<Identifier>, Block),
    Call(Box<Expression>, Vec<Expression>),
    Index(Box<Expression>, Box<Expression>),
}

impl Expression {
    pub fn hash(&self) -> u64 {
        hash(&format!("{:?}", self))
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let expression = match self {
            Self::Identifier(identifier) => identifier.to_string(),
            Self::Literal(literal) => literal.to_string(),
            Self::Boolean(boolean) => boolean.to_string(),
            Self::Prefix(operator, expression) => format!("({}{})", operator, expression),
            Self::Infix(left_expression, operator, right_expression) => {
                format!("({} {} {})", left_expression, operator, right_expression)
            }
            Self::If(condition, consequence, alternative) => {
                let statement = format!("if ({}) {{ {} }}", condition, flatten(consequence, "\n"),);

                let mut result = String::new();
                result.push_str(statement.as_str());

                if let Some(alternative) = alternative {
                    let else_statement = format!("else {{ {} }}", flatten(alternative, "\n"));
                    result.push_str(&else_statement);
                }

                result
            }
            Self::Function(parameters, body) => {
                format!(
                    "fn({}) {{ {} }}",
                    flatten(parameters, ", "),
                    flatten(body, "\n"),
                )
            }
            Self::Call(expression, arguments) => {
                format!("{}({})", expression, flatten(arguments, ", "),)
            }
            Self::Index(left_expression, index_expression) => {
                format!("({}[{}])", left_expression, index_expression,)
            }
        };
        write!(f, "{}", expression)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Integer(i64),
    String(String),
    Array(Vec<Expression>),
    HashMap(Vec<(Expression, Expression)>),
}

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let literal = match self {
            Self::Integer(x) => x.to_string(),
            Self::String(x) => x.to_string(),
            Self::Array(array) => {
                let expressions = array.iter().map(|e| e.to_string()).collect::<Vec<_>>();
                format!("[{}]", expressions.join(", "))
            }
            Self::HashMap(key_value_pairs) => {
                let pairs: Vec<String> = key_value_pairs
                    .iter()
                    .map(|(key, value)| format!("{}: {}", key, value))
                    .collect();
                format!("{{ {} }}", pairs.join(", "))
            }
        };
        write!(f, "{}", literal)
    }
}

#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub enum Precedence {
    Lowest,
    Equals,
    LessThanGreaterThan,
    Sum,
    Product,
    Prefix,
    Call,
    Index,
}

impl From<&Token> for Precedence {
    fn from(token: &Token) -> Self {
        match token {
            Token::Equal => Self::Equals,
            Token::NotEqual => Self::Equals,
            Token::LessThan => Self::LessThanGreaterThan,
            Token::GreaterThan => Self::LessThanGreaterThan,
            Token::Plus => Self::Sum,
            Token::Minus => Self::Sum,
            Token::Slash => Self::Product,
            Token::Asterisk => Self::Product,
            Token::LeftParentheses => Self::Call,
            Token::LeftBracket => Self::Index,
            _ => Self::Lowest,
        }
    }
}

pub type Program = Vec<Statement>;

pub struct Parser<'a> {
    pub tokens: Iter<'a, Token>,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens: tokens.iter(),
        }
    }

    pub fn parse(&mut self) -> Result<Program> {
        let mut program = Program::new();
        while let Some(statement) = self.parse_statement()? {
            program.push(statement);
        }
        Ok(program)
    }

    pub fn parse_statement(&mut self) -> Result<Option<Statement>> {
        Ok(match self.peek_nth(0) {
            Token::EndOfFile => None,
            Token::Let => Some(self.parse_let_statement()?),
            Token::Return => Some(self.parse_return_statement()?),
            _ => Some(self.parse_expression_statement()?),
        })
    }

    fn parse_let_statement(&mut self) -> Result<Statement> {
        if !matches!(self.read_token(), Token::Let) {
            bail!("Expected 'Let' token!");
        }

        let identifier = match self.read_token() {
            Token::Identifier(identifier) => identifier.to_string(),
            token => {
                bail!("Expected 'Identifier' token! Found '{:?}'.", token);
            }
        };

        if !matches!(self.read_token(), Token::Assign) {
            bail!("Expected 'Assign' token!");
        }

        let expression = self.parse_expression(Precedence::Lowest)?;

        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }

        Ok(Statement::Let(identifier, expression))
    }

    fn parse_return_statement(&mut self) -> Result<Statement> {
        if !matches!(self.read_token(), Token::Return) {
            bail!("Expected 'Return' token!");
        }

        let expression = self.parse_expression(Precedence::Lowest)?;

        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }

        Ok(Statement::Return(expression))
    }

    fn parse_expression_statement(&mut self) -> Result<Statement> {
        let statement = Statement::Expression(self.parse_expression(Precedence::Lowest)?);
        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }
        Ok(statement)
    }

    fn parse_expression(&mut self, precedence: Precedence) -> Result<Expression> {
        let mut advance = true;
        let mut expression = match self.peek_nth(0) {
            Token::Identifier(identifier) => Expression::Identifier(identifier.to_string()),
            Token::StringLiteral(string) => {
                Expression::Literal(Literal::String(string.to_string()))
            }
            Token::Integer(value) => Expression::Literal(Literal::Integer(*value)),
            Token::Bang | Token::Minus => {
                advance = false;
                self.parse_prefix_expression()?
            }
            Token::True => Expression::Boolean(true),
            Token::False => Expression::Boolean(false),
            Token::LeftBrace => {
                advance = false;
                self.parse_hashmap_literal()?
            }
            Token::LeftBracket => {
                advance = false;
                self.parse_array_literal()?
            }
            Token::LeftParentheses => {
                advance = false;
                self.parse_grouped_expressions()?
            }
            Token::If => {
                advance = false;
                self.parse_if_expression()?
            }
            Token::Function => {
                advance = false;
                self.parse_function_literal()?
            }
            Token::EndOfFile => {
                bail!("Unexpected end of file")
            }
            token => bail!("Token not valid for an expression: {:?}", token),
        };

        if advance {
            self.read_token();
        }

        while self.peek_nth(0) != &Token::Semicolon
            && precedence < Precedence::from(self.peek_nth(0))
        {
            match self.peek_nth(0) {
                Token::Plus
                | Token::Minus
                | Token::Slash
                | Token::Asterisk
                | Token::Equal
                | Token::NotEqual
                | Token::LessThan
                | Token::GreaterThan => {
                    expression = self.parse_infix_expression(expression.clone())?;
                }
                Token::LeftBracket => {
                    expression = self.parse_index_expression(expression.clone())?;
                }
                Token::LeftParentheses => {
                    expression = self.parse_call_expression(expression.clone())?;
                }
                _ => return Ok(expression),
            };
        }

        Ok(expression)
    }

    fn parse_prefix_expression(&mut self) -> Result<Expression> {
        let operator = Operator::from_token(self.peek_nth(0), true)?;
        self.read_token();
        Ok(Expression::Prefix(
            operator,
            Box::new(self.parse_expression(Precedence::Prefix)?),
        ))
    }

    fn parse_infix_expression(&mut self, left_expression: Expression) -> Result<Expression> {
        let operator = Operator::from_token(self.peek_nth(0), false)?;
        let precedence = Precedence::from(self.peek_nth(0));
        self.read_token();
        Ok(Expression::Infix(
            Box::new(left_expression),
            operator,
            Box::new(self.parse_expression(precedence)?),
        ))
    }

    fn parse_array_literal(&mut self) -> Result<Expression> {
        let elements = self.parse_expression_list(&Token::RightBracket)?;
        Ok(Expression::Literal(Literal::Array(elements)))
    }

    fn parse_call_expression(&mut self, expression: Expression) -> Result<Expression> {
        let elements = self.parse_expression_list(&Token::RightParentheses)?;
        Ok(Expression::Call(Box::new(expression), elements))
    }

    fn parse_index_expression(&mut self, expression: Expression) -> Result<Expression> {
        self.read_token(); // [
        let index_expression = self.parse_expression(Precedence::Lowest)?;
        self.read_token(); // ]
        Ok(Expression::Index(
            Box::new(expression),
            Box::new(index_expression),
        ))
    }

    fn parse_hashmap_literal(&mut self) -> Result<Expression> {
        let mut pairs = Vec::new();
        self.read_token(); // {
        while self.peek_nth(0) != &Token::RightBrace {
            let key = self.parse_expression(Precedence::Lowest)?;
            self.read_token(); // :
            let value = self.parse_expression(Precedence::Lowest)?;
            if matches!(self.peek_nth(0), &Token::Comma) {
                self.read_token();
            }
            pairs.push((key, value));
        }
        self.read_token(); // }
        Ok(Expression::Literal(Literal::HashMap(pairs)))
    }

    fn parse_expression_list(&mut self, end_token: &Token) -> Result<Vec<Expression>> {
        self.read_token();
        let mut elements = Vec::new();
        while self.peek_nth(0) != end_token {
            elements.push(self.parse_expression(Precedence::Lowest)?);

            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }
        self.read_token();
        Ok(elements)
    }

    fn parse_grouped_expressions(&mut self) -> Result<Expression> {
        self.read_token();
        let expression = self.parse_expression(Precedence::Lowest)?;
        if matches!(self.peek_nth(0), Token::RightParentheses) {
            self.read_token();
        }
        Ok(expression)
    }

    fn parse_if_expression(&mut self) -> Result<Expression> {
        self.read_token();

        if !matches!(self.peek_nth(0), Token::LeftParentheses) {
            bail!("Expected a left parentheses in if expression!");
        }
        self.read_token();

        let condition = self.parse_expression(Precedence::Lowest)?;

        if !matches!(self.peek_nth(0), Token::RightParentheses) {
            bail!("Expected a right parentheses in if expression!");
        }
        self.read_token();

        let consequence = self.parse_block()?;

        let mut alternative = None;
        if matches!(self.peek_nth(0), Token::Else) {
            self.read_token();
            alternative = Some(self.parse_block()?);
        }

        Ok(Expression::If(
            Box::new(condition),
            consequence,
            alternative,
        ))
    }

    fn parse_function_literal(&mut self) -> Result<Expression> {
        self.read_token();
        let parameters = self.parse_function_parameters()?;
        let block = self.parse_block()?;
        Ok(Expression::Function(parameters, block))
    }

    fn parse_function_parameters(&mut self) -> Result<Vec<Identifier>> {
        if !matches!(self.peek_nth(0), Token::LeftParentheses) {
            bail!("Expected a left parentheses in parameter list!");
        }
        self.read_token();

        let mut identifiers = Vec::new();
        while self.peek_nth(0) != &Token::RightParentheses {
            if matches!(self.peek_nth(0), Token::Identifier(_)) {
                identifiers.push(self.read_token().to_string());
            }

            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }

        if !matches!(self.peek_nth(0), Token::RightParentheses) {
            bail!("Expected a right parentheses in parameter list!");
        }
        self.read_token();

        Ok(identifiers)
    }

    fn parse_block(&mut self) -> Result<Block> {
        if !matches!(self.peek_nth(0), Token::LeftBrace) {
            bail!("Expected a left brace in block!");
        }
        self.read_token();

        let mut statements = Vec::new();

        while self.peek_nth(0) != &Token::RightBrace && self.peek_nth(0) != &Token::EndOfFile {
            if let Some(statement) = self.parse_statement()? {
                statements.push(statement);
            }
        }

        if !matches!(self.peek_nth(0), Token::RightBrace) {
            bail!("Expected a right brace in block!");
        }
        self.read_token();

        Ok(statements)
    }

    fn read_token(&mut self) -> &Token {
        self.tokens.next().unwrap_or(&Token::EndOfFile)
    }

    fn peek_nth(&self, n: usize) -> &Token {
        self.tokens.clone().nth(n).unwrap_or(&Token::EndOfFile)
    }
}

#[cfg(test)]
mod tests {
    use super::{Expression, Literal, Parser, Statement};
    use crate::{lexer::Lexer, Operator};
    use anyhow::{bail, Result};

    #[test]
    fn test_let_statements() -> Result<()> {
        let tests = [
            (
                "let x = 5;",
                "x".to_string(),
                Expression::Literal(Literal::Integer(5)),
            ),
            (
                "let y = 10;",
                "y".to_string(),
                Expression::Literal(Literal::Integer(10)),
            ),
            (
                "let foobar = 838383;",
                "foobar".to_string(),
                Expression::Literal(Literal::Integer(838383)),
            ),
            (
                "let foobar = y;",
                "foobar".to_string(),
                Expression::Identifier("y".to_string()),
            ),
        ];

        for (input, expected_identifier, expected_expression) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            for statement in program.into_iter() {
                match statement {
                    Statement::Let(identifier, expression) => {
                        assert_eq!(identifier, *expected_identifier);
                        assert_eq!(expression, *expected_expression);
                    }
                    _ => bail!("Expected a let statement!"),
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_return_statements() -> Result<()> {
        let tests = [
            ("return 5;", Expression::Literal(Literal::Integer(5))),
            ("return 10;", Expression::Literal(Literal::Integer(10))),
            (
                "return 993322;",
                Expression::Literal(Literal::Integer(993322)),
            ),
            ("return y;", Expression::Identifier("y".to_string())),
        ];

        for (input, expected_expression) in tests.iter() {
            parse_statement(input, expected_expression)?;
        }

        Ok(())
    }

    #[test]
    fn ast() -> Result<()> {
        let output = "let myVar = anotherVar;";
        let ast = Statement::Let(
            "myVar".to_string(),
            Expression::Identifier("anotherVar".to_string()),
        );
        assert_eq!(ast.to_string(), output.to_string());
        Ok(())
    }

    #[test]
    fn identifier_expressions() -> Result<()> {
        parse_statement("foobar;", &Expression::Identifier("foobar".to_string()))
    }

    #[test]
    fn integer_expressions() -> Result<()> {
        parse_statement("5;", &Expression::Literal(Literal::Integer(5)))
    }

    #[test]
    fn boolean_expressions() -> Result<()> {
        let tests = [("true;", true), ("false;", false)];

        for (input, expected_value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            if let Some(Statement::Expression(Expression::Boolean(value))) =
                program.into_iter().next()
            {
                assert_eq!(value, *expected_value)
            } else {
                bail!("Expected a boolean expression statement!");
            }
        }

        Ok(())
    }

    #[test]
    fn prefix_expressions() -> Result<()> {
        let tests = [("!5;", Operator::Not, 5), ("-15;", Operator::Negate, 15)];

        for (input, operator, value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            if let Some(statement) = program.into_iter().next() {
                match statement {
                    Statement::Expression(expression) => {
                        assert_eq!(
                            expression,
                            Expression::Prefix(
                                *operator,
                                Box::new(Expression::Literal(Literal::Integer(*value))),
                            )
                        )
                    }
                    _ => bail!("Expected an expression statement!"),
                }
            }
        }

        Ok(())
    }

    #[test]
    fn prefix_boolean_expressions() -> Result<()> {
        let tests = [
            ("!true;", Operator::Not, true),
            ("!false;", Operator::Not, false),
        ];

        for (input, operator, value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            if let Some(statement) = program.into_iter().next() {
                match statement {
                    Statement::Expression(expression) => {
                        assert_eq!(
                            expression,
                            Expression::Prefix(*operator, Box::new(Expression::Boolean(*value)),)
                        )
                    }
                    _ => bail!("Expected an expression statement!"),
                }
            }
        }

        Ok(())
    }

    #[test]
    fn infix_expressions() -> Result<()> {
        let tests = [
            ("5 + 5;", 5, Operator::Add, 5),
            ("5 - 5;", 5, Operator::Subtract, 5),
            ("5 * 5;", 5, Operator::Multiply, 5),
            ("5 / 5;", 5, Operator::Divide, 5),
            ("5 > 5;", 5, Operator::GreaterThan, 5),
            ("5 < 5;", 5, Operator::LessThan, 5),
            ("5 == 5;", 5, Operator::Equal, 5),
            ("5 != 5;", 5, Operator::NotEqual, 5),
        ];

        for (input, left_value, operator, right_value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            if let Some(statement) = program.into_iter().next() {
                match statement {
                    Statement::Expression(expression) => {
                        assert_eq!(
                            expression,
                            Expression::Infix(
                                Box::new(Expression::Literal(Literal::Integer(*left_value))),
                                *operator,
                                Box::new(Expression::Literal(Literal::Integer(*right_value,))),
                            )
                        )
                    }
                    _ => bail!("Expected an expression statement!"),
                }
            }
        }

        Ok(())
    }

    #[test]
    fn infix_boolean_expressions() -> Result<()> {
        let tests = [
            ("true == true", true, Operator::Equal, true),
            ("true != false", true, Operator::NotEqual, false),
            ("false == false", false, Operator::Equal, false),
        ];

        for (input, left_value, operator, right_value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            if let Some(statement) = program.into_iter().next() {
                match statement {
                    Statement::Expression(expression) => {
                        assert_eq!(
                            expression,
                            Expression::Infix(
                                Box::new(Expression::Boolean(*left_value)),
                                *operator,
                                Box::new(Expression::Boolean(*right_value)),
                            )
                        )
                    }
                    _ => bail!("Expected an expression statement!"),
                }
            }
        }

        Ok(())
    }

    #[test]
    fn operator_precedence() -> Result<()> {
        let tests = [
            ("-a * b", "((-a) * b)"),
            ("!-a", "(!(-a))"),
            ("a + b + c", "((a + b) + c)"),
            ("a + b - c", "((a + b) - c)"),
            ("a * b * c", "((a * b) * c)"),
            ("a * b / c", "((a * b) / c)"),
            ("a + b / c", "(a + (b / c))"),
            ("a + b * c + d / e - f", "(((a + (b * c)) + (d / e)) - f)"),
            ("3 + 4; -5 * 5", "(3 + 4)((-5) * 5)"),
            ("5 > 4 == 3 < 4", "((5 > 4) == (3 < 4))"),
            ("5 < 4 != 3 > 4", "((5 < 4) != (3 > 4))"),
            (
                "3 + 4 * 5 == 3 * 1 + 4 * 5",
                "((3 + (4 * 5)) == ((3 * 1) + (4 * 5)))",
            ),
            ("true", "true"),
            ("false", "false"),
            ("3 > 5 == false", "((3 > 5) == false)"),
            ("3 < 5 == true", "((3 < 5) == true)"),
            ("1 + (2 + 3) + 4", "((1 + (2 + 3)) + 4)"),
            ("(5 + 5) * 2", "((5 + 5) * 2)"),
            ("2 / (5 + 5)", "(2 / (5 + 5))"),
            ("-(5 + 5)", "(-(5 + 5))"),
            ("!(true == true)", "(!(true == true))"),
            ("a + add(b * c) + d", "((a + add((b * c))) + d)"),
            (
                "add(a, b, 1, 2 * 3, 4 + 5, add(6, 7 * 8))",
                "add(a, b, 1, (2 * 3), (4 + 5), add(6, (7 * 8)))",
            ),
            (
                "add(a + b + c * d / f + g)",
                "add((((a + b) + ((c * d) / f)) + g))",
            ),
            (
                "a * [1, 2, 3, 4][b * c] * d",
                "((a * ([1, 2, 3, 4][(b * c)])) * d)",
            ),
            (
                "add(a * b[2], b[1], 2 * [1, 2][1])",
                "add((a * (b[2])), (b[1]), (2 * ([1, 2][1])))",
            ),
        ];

        for (input, expected) in tests.iter() {
            let mut lexer = Lexer::new(input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;
            let program_string = program
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("");

            assert_eq!(program_string, expected.to_string());
        }

        Ok(())
    }

    #[test]
    fn if_expressions() -> Result<()> {
        let expression = Expression::If(
            Box::new(Expression::Infix(
                Box::new(Expression::Identifier("x".to_string())),
                Operator::LessThan,
                Box::new(Expression::Identifier("y".to_string())),
            )),
            vec![Statement::Expression(Expression::Identifier(
                "x".to_string(),
            ))],
            None,
        );

        parse_statement("if (x < y) { x }", &expression)
    }

    #[test]
    fn if_else_expressions() -> Result<()> {
        let expression = Expression::If(
            Box::new(Expression::Infix(
                Box::new(Expression::Identifier("x".to_string())),
                Operator::LessThan,
                Box::new(Expression::Identifier("y".to_string())),
            )),
            vec![Statement::Expression(Expression::Identifier(
                "x".to_string(),
            ))],
            Some(vec![Statement::Expression(Expression::Identifier(
                "y".to_string(),
            ))]),
        );

        parse_statement("if (x < y) { x } else { y }", &expression)
    }

    #[test]
    fn function_expressions() -> Result<()> {
        let expression = Expression::Function(
            vec!["x".to_string(), "y".to_string()],
            vec![Statement::Expression(Expression::Infix(
                Box::new(Expression::Identifier("x".to_string())),
                Operator::Add,
                Box::new(Expression::Identifier("y".to_string())),
            ))],
        );
        parse_statement("fn(x, y) { x + y; }", &expression)
    }

    #[test]
    fn function_parameter_parsing() -> Result<()> {
        let tests = [
            ("fn() {};", vec![]),
            ("fn(x) {};", vec!["x".to_string()]),
            (
                "fn(x, y, z) {};",
                vec!["x".to_string(), "y".to_string(), "z".to_string()],
            ),
        ];

        for (input, expected_parameters) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            if let Some(statement) = program.into_iter().next() {
                match statement {
                    Statement::Expression(expression) => {
                        assert_eq!(
                            expression,
                            Expression::Function(expected_parameters.to_vec(), Vec::new())
                        )
                    }
                    _ => bail!("Expected an expression statement!"),
                }
            }
        }

        Ok(())
    }

    #[test]
    fn call_expressions() -> Result<()> {
        let expression = Expression::Call(
            Box::new(Expression::Identifier("add".to_string())),
            vec![
                Expression::Literal(Literal::Integer(1)),
                Expression::Infix(
                    Box::new(Expression::Literal(Literal::Integer(2))),
                    Operator::Multiply,
                    Box::new(Expression::Literal(Literal::Integer(3))),
                ),
                Expression::Infix(
                    Box::new(Expression::Literal(Literal::Integer(4))),
                    Operator::Add,
                    Box::new(Expression::Literal(Literal::Integer(5))),
                ),
            ],
        );

        parse_statement("add(1, 2 * 3, 4 + 5);", &expression)
    }

    #[test]
    fn string_literal_expression() -> Result<()> {
        parse_statement(
            "\"hello world\"",
            &Expression::Literal(Literal::String("hello world".to_string())),
        )
    }

    #[test]
    fn array_literal_expression() -> Result<()> {
        parse_statement(
            "[1, 2 * 2, 3 + 3]",
            &Expression::Literal(Literal::Array(vec![
                Expression::Literal(Literal::Integer(1)),
                Expression::Infix(
                    Box::new(Expression::Literal(Literal::Integer(2))),
                    Operator::Multiply,
                    Box::new(Expression::Literal(Literal::Integer(2))),
                ),
                Expression::Infix(
                    Box::new(Expression::Literal(Literal::Integer(3))),
                    Operator::Add,
                    Box::new(Expression::Literal(Literal::Integer(3))),
                ),
            ])),
        )
    }

    #[test]
    fn index_expression() -> Result<()> {
        parse_statement(
            "myArray[1 + 1]",
            &Expression::Index(
                Box::new(Expression::Identifier("myArray".to_string())),
                Box::new(Expression::Infix(
                    Box::new(Expression::Literal(Literal::Integer(1))),
                    Operator::Add,
                    Box::new(Expression::Literal(Literal::Integer(1))),
                )),
            ),
        )
    }

    #[test]
    fn hashmap_literal() -> Result<()> {
        parse_statement(
            r#"{"one": 1, "two": 2, "three": 3}"#,
            &Expression::Literal(Literal::HashMap(vec![
                (
                    Expression::Literal(Literal::String("one".to_string())),
                    Expression::Literal(Literal::Integer(1)),
                ),
                (
                    Expression::Literal(Literal::String("two".to_string())),
                    Expression::Literal(Literal::Integer(2)),
                ),
                (
                    Expression::Literal(Literal::String("three".to_string())),
                    Expression::Literal(Literal::Integer(3)),
                ),
            ])),
        )
    }

    #[test]
    fn empty_hashmap_literal() -> Result<()> {
        parse_statement("{}", &Expression::Literal(Literal::HashMap(vec![])))
    }

    #[test]
    fn hashmap_literal_with_expressions() -> Result<()> {
        parse_statement(
            "{\"one\": 0 + 1, \"two\": 10 - 8, \"three\": 15 / 5}",
            &Expression::Literal(Literal::HashMap(vec![
                (
                    Expression::Literal(Literal::String("one".to_string())),
                    Expression::Infix(
                        Box::new(Expression::Literal(Literal::Integer(0))),
                        Operator::Add,
                        Box::new(Expression::Literal(Literal::Integer(1))),
                    ),
                ),
                (
                    Expression::Literal(Literal::String("two".to_string())),
                    Expression::Infix(
                        Box::new(Expression::Literal(Literal::Integer(10))),
                        Operator::Subtract,
                        Box::new(Expression::Literal(Literal::Integer(8))),
                    ),
                ),
                (
                    Expression::Literal(Literal::String("three".to_string())),
                    Expression::Infix(
                        Box::new(Expression::Literal(Literal::Integer(15))),
                        Operator::Divide,
                        Box::new(Expression::Literal(Literal::Integer(5))),
                    ),
                ),
            ])),
        )
    }

    fn parse_statement(input: &str, expected_expression: &Expression) -> Result<()> {
        let mut lexer = Lexer::new(&input);
        let tokens = lexer.tokenize()?;

        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        if let Some(statement) = program.into_iter().next() {
            match statement {
                Statement::Expression(expression) | Statement::Return(expression) => {
                    assert_eq!(expression, *expected_expression)
                }
                _ => bail!("Expected an expression statement!"),
            }
        }

        Ok(())
    }
}
