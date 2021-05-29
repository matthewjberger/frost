use crate::lexer::Token;
use anyhow::{bail, Result};
use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    matches,
    slice::Iter,
};

#[derive(Debug)]
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
#[derive(Debug)]
pub enum Expression {
    Identifier(Identifier),
    Literal(Literal),
}

impl Display for Expression {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let expression = match self {
            Self::Identifier(identifier) => format!("{}", identifier),
            Self::Literal(literal) => format!("{}", literal),
        };
        write!(f, "{}", expression)
    }
}

#[derive(Debug, PartialEq)]
pub struct Identifier(pub String);

impl Display for Identifier {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, PartialEq)]
pub enum Literal {
    Int(i64),
    Bool(bool),
    String(String),
}

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let literal = match self {
            Literal::Int(x) => x.to_string(),
            Literal::Bool(x) => x.to_string(),
            Literal::String(x) => x.to_string(),
        };
        write!(f, "{}", literal)
    }
}

#[derive(Debug, PartialEq, PartialOrd)]
pub enum Precedence {
    Lowest,
    Equals,
    LessThan,
    GreaterThan,
    Sum,
    Product,
    Prefix,
    Call,
}

#[derive(Debug, PartialEq)]
pub enum Prefix {
    Plus,
    Minus,
    Not,
}

pub type Program = Vec<Statement>;

// TODO: Can this just be a function instead?
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

    fn parse_expression(&mut self, _precedence: Precedence) -> Result<Expression> {
        let expression = match self.read_token() {
            Token::Identifier(identifier) => {
                Expression::Identifier(Identifier(identifier.to_string()))
            }
            token => bail!("Token not valid for an expression: {:?}", token),
        };
        Ok(expression)
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
    use super::{Expression, Identifier, Parser, Result, Statement};
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
        let tokens = lexer.exhaust()?;

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
        let tokens = lexer.exhaust()?;

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
        let tokens = lexer.exhaust()?;

        let mut parser = Parser::new(&tokens);
        let program = parser.parse()?;

        assert_eq!(program.len(), 1);

        for statement in program.into_iter() {
            match statement {
                Statement::Expression(_expression) => {}
                _ => bail!("Expected an expression statement!"),
            }
        }

        Ok(())
    }
}
