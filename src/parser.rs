use crate::lexer::Token;
use anyhow::{bail, Result};
use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    matches,
    slice::Iter,
};

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
    Prefix(Prefix),
    Infix(Infix),
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let expression = match self {
            Self::Identifier(identifier) => identifier.to_string(),
            Self::Literal(literal) => literal.to_string(),
            Self::Prefix(prefix) => prefix.to_string(),
            Self::Infix(infix) => infix.to_string(),
        };
        write!(f, "{}", expression)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Prefix(pub String, pub Box<Expression>);

impl Display for Prefix {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}{}", self.0, self.1)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Infix(pub Box<Expression>, pub String, pub Box<Expression>);

impl Display for Infix {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{} {} {}", self.0, self.1, self.2)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Identifier(pub String);

impl Display for Identifier {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Integer(i64),
    Bool(bool),
    String(String),
}

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let literal = match self {
            Literal::Integer(x) => x.to_string(),
            Literal::Bool(x) => x.to_string(),
            Literal::String(x) => x.to_string(),
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
}

impl Precedence {
    pub fn of_token(token: &Token) -> Self {
        match token {
            Token::Equal => Self::Equals,
            Token::NotEqual => Self::Equals,
            Token::LessThan => Self::LessThanGreaterThan,
            Token::GreaterThan => Self::LessThanGreaterThan,
            Token::Plus => Self::Sum,
            Token::Minus => Self::Sum,
            Token::Slash => Self::Product,
            Token::Asterisk => Self::Product,
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
        loop {
            let statement = match self.peek_nth(0) {
                Token::EndOfFile => break,
                Token::Let => self.parse_let_statement()?,
                Token::Return => self.parse_return_statement()?,
                _ => self.parse_expression_statement()?,
            };
            program.push(statement);
        }
        Ok(program)
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

        // TODO: Parse expressions

        while !matches!(self.read_token(), Token::Semicolon) {}

        Ok(Statement::Let(
            Identifier(identifier),
            // TODO
            Expression::Identifier(Identifier("".to_string())),
        ))
    }

    fn parse_return_statement(&mut self) -> Result<Statement> {
        if !matches!(self.read_token(), Token::Return) {
            bail!("Expected 'Return' token!");
        }

        // TODO: Parse expressions

        while !matches!(self.read_token(), Token::Semicolon) {}

        Ok(Statement::Return(Expression::Identifier(Identifier(
            "".to_string(),
        ))))
    }

    fn parse_expression_statement(&mut self) -> Result<Statement> {
        let statement = Statement::Expression(self.parse_expression(Precedence::Lowest)?);
        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }
        Ok(statement)
    }

    fn parse_expression(&mut self, precedence: Precedence) -> Result<Expression> {
        let mut expression = match self.peek_nth(0) {
            Token::Identifier(identifier) => {
                Expression::Identifier(Identifier(identifier.to_string()))
            }
            Token::Integer(value) => Expression::Literal(Literal::Integer(*value)),
            Token::Bang | Token::Minus => self.parse_prefix_expression()?,
            token => bail!("Token not valid for an expression: {:?}", token),
        };
        self.read_token();
        while self.peek_nth(1) != &Token::Semicolon
            && precedence < Precedence::of_token(self.peek_nth(0))
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
                token => bail!("Token not valid for an infix expression: {:?}", token),
            };
            self.read_token();
        }
        Ok(expression)
    }

    fn parse_prefix_expression(&mut self) -> Result<Expression> {
        let operator = self.peek_nth(0).to_string();
        self.read_token();
        Ok(Expression::Prefix(Prefix(
            operator.to_string(),
            Box::new(self.parse_expression(Precedence::Prefix)?),
        )))
    }

    fn parse_infix_expression(&mut self, left_expression: Expression) -> Result<Expression> {
        let operator = self.peek_nth(0).to_string();
        let precedence = Precedence::of_token(self.peek_nth(0));
        self.read_token();
        Ok(Expression::Infix(Infix(
            Box::new(left_expression),
            operator.to_string(),
            Box::new(self.parse_expression(precedence)?),
        )))
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
    use super::{Expression, Identifier, Infix, Literal, Parser, Prefix, Result, Statement};
    use crate::lexer::Lexer;
    use anyhow::bail;

    #[test]
    fn test_let_statements() -> Result<()> {
        let input = r#"
        let x = 5;
        let y = 10;
        let foobar = 838383;
        "#;

        let mut lexer = Lexer::new(&input);
        let tokens = lexer.tokenize()?;

        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 3);

        let identifiers: Vec<Identifier> = ["x", "y", "foobar"]
            .iter()
            .map(|x| Identifier(x.to_string()))
            .collect();

        for (statement, expected_identifier) in program.into_iter().zip(identifiers.into_iter()) {
            println!("Found a statement: {}", statement.to_string());
            match statement {
                Statement::Let(identifier, _expression) => {
                    assert_eq!(identifier, expected_identifier);
                }
                _ => bail!("Expected a let statement!"),
            }
        }

        Ok(())
    }

    #[test]
    fn test_return_statements() -> Result<()> {
        let input = r#"
        return 5;
        return 10;
        return 993322;
        "#;

        let mut lexer = Lexer::new(&input);
        let tokens = lexer.tokenize()?;

        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 3);

        for statement in program.into_iter() {
            match statement {
                Statement::Return(_expression) => {}
                _ => bail!("Expected a return statement!"),
            }
        }

        Ok(())
    }

    #[test]
    fn test_ast() -> Result<()> {
        let output = "let myVar = anotherVar;";
        let ast = Statement::Let(
            Identifier("myVar".to_string()),
            Expression::Identifier(Identifier("anotherVar".to_string())),
        );
        assert_eq!(ast.to_string(), output.to_string());
        Ok(())
    }

    #[test]
    fn test_identifier_expression() -> Result<()> {
        let input = "foobar;";

        let mut lexer = Lexer::new(&input);
        let tokens = lexer.tokenize()?;

        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);

        let expressions = vec![Expression::Identifier(Identifier("foobar".to_string()))];

        for (statement, expected_expression) in program.into_iter().zip(expressions.into_iter()) {
            match statement {
                Statement::Expression(expression) => assert_eq!(expression, expected_expression),
                _ => bail!("Expected an expression statement!"),
            }
        }

        Ok(())
    }

    #[test]
    fn test_integer_expression() -> Result<()> {
        let input = "5;";

        let mut lexer = Lexer::new(&input);
        let tokens = lexer.tokenize()?;

        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);

        let expressions = vec![Expression::Literal(Literal::Integer(5))];

        for (statement, expected_expression) in program.into_iter().zip(expressions.into_iter()) {
            match statement {
                Statement::Expression(expression) => assert_eq!(expression, expected_expression),
                _ => bail!("Expected an expression statement!"),
            }
        }

        Ok(())
    }

    #[test]
    fn test_prefix_expressions() -> Result<()> {
        let tests = [("!5;", "!", 5), ("-15;", "-", 15)];

        for (input, operator, value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            let prefix = Prefix(
                operator.to_string(),
                Box::new(Expression::Literal(Literal::Integer(*value))),
            );

            if let Some(statement) = program.into_iter().next() {
                match statement {
                    Statement::Expression(expression) => {
                        assert_eq!(expression, Expression::Prefix(prefix))
                    }
                    _ => bail!("Expected an expression statement!"),
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_infix_expressions() -> Result<()> {
        let tests = [
            ("5 + 5;", 5, "+", 5),
            ("5 - 5;", 5, "-", 5),
            ("5 * 5;", 5, "*", 5),
            ("5 / 5;", 5, "/", 5),
            ("5 > 5;", 5, ">", 5),
            ("5 < 5;", 5, "<", 5),
            ("5 == 5;", 5, "==", 5),
            ("5 != 5;", 5, "!=", 5),
        ];

        for (input, left_value, operator, right_value) in tests.iter() {
            let mut lexer = Lexer::new(&input);
            let tokens = lexer.tokenize()?;

            let mut parser = Parser::new(&tokens);
            let program = parser.parse()?;

            assert_eq!(program.len(), 1);

            let infix = Infix(
                Box::new(Expression::Literal(Literal::Integer(*left_value))),
                operator.to_string(),
                Box::new(Expression::Literal(Literal::Integer(*right_value))),
            );

            if let Some(statement) = program.into_iter().next() {
                match statement {
                    Statement::Expression(expression) => {
                        assert_eq!(expression, Expression::Infix(infix))
                    }
                    _ => bail!("Expected an expression statement!"),
                }
            }
        }

        Ok(())
    }
}
