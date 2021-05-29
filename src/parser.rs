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
}

impl Display for Statement {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let statement = match self {
            Self::Let(identifier, expression) => format!("let {} = {};", identifier, expression),
            Self::Return(expression) => format!("return {};", expression),
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
            match self.peek_nth(0) {
                Token::EndOfFile => break,
                Token::Let => program.push(self.parse_let_statement()?),
                Token::Return => program.push(self.parse_return_statement()?),
                token => bail!("Unrecognized token: '{:?}'", token),
            };
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

    fn read_token(&mut self) -> &Token {
        self.tokens.next().unwrap_or(&Token::EndOfFile)
    }

    fn peek_nth(&self, n: usize) -> &Token {
        self.tokens.clone().nth(n).unwrap_or(&Token::EndOfFile)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Identifier, Parser, Result,
        Statement::{Let, Return},
    };
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

        assert!(!program.is_empty());
        assert_eq!(program.len(), 3);

        let identifiers: Vec<Identifier> = ["x", "y", "foobar"]
            .iter()
            .map(|x| Identifier(x.to_string()))
            .collect();

        for (statement, expected_identifier) in program.into_iter().zip(identifiers.into_iter()) {
            println!("Found a statement: {}", statement.to_string());
            match statement {
                Let(identifier, _expression) => {
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

        assert!(!program.is_empty());
        assert_eq!(program.len(), 3);

        for statement in program.into_iter() {
            match statement {
                Return(_expression) => {}
                _ => bail!("Expected a return statement!"),
            }
        }

        Ok(())
    }
}
