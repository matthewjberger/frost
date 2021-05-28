use crate::lexer::Token;
use anyhow::{bail, Result};
use std::{matches, slice::Iter};

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
                _ => {}
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

        loop {
            if matches!(self.read_token(), Token::Semicolon) {
                break;
            }
        }

        Ok(Statement::LetStatement(
            Identifier(identifier),
            // TODO
            Expression::IdentifierExpression(Identifier("".to_string())),
        ))
    }

    fn read_token(&mut self) -> &Token {
        self.tokens.next().unwrap_or(&Token::EndOfFile)
    }

    fn peek_nth(&self, n: usize) -> &Token {
        self.tokens.clone().nth(n).unwrap_or(&Token::EndOfFile)
    }
}

#[derive(Debug)]
pub enum Statement {
    LetStatement(Identifier, Expression),
}

#[derive(Debug)]
pub enum Expression {
    IdentifierExpression(Identifier),
    LiteralExpression(Literal),
}

#[derive(Debug, PartialEq)]
pub struct Identifier(pub String);

#[derive(Debug, PartialEq)]
pub enum Literal {
    Int(i64),
    Bool(bool),
    String(String),
}

#[cfg(test)]
mod tests {
    use super::{Identifier, Parser, Result, Statement::LetStatement};
    use crate::lexer::Lexer;

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
            match statement {
                LetStatement(identifier, _expression) => {
                    assert_eq!(identifier, expected_identifier);
                }
            }
        }

        Ok(())
    }
}
