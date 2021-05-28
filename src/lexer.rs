use anyhow::{Context, Result};

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
pub enum Token {
    Assign,
    Asterisk,
    Bang,
    Comma,
    Else,
    EndOfFile,
    Equal,
    False,
    Function,
    GreaterThan,
    Identifier(String),
    If,
    Illegal(String),
    Integer(String),
    LeftBrace,
    LeftParentheses,
    LessThan,
    Let,
    Minus,
    NotEqual,
    Plus,
    Return,
    RightBrace,
    RightParentheses,
    Semicolon,
    Slash,
    True,
}

#[derive(Default)]
pub struct Lexer {
    input: String,
    position: usize,
    read_position: usize,
    char: char,
}

impl Lexer {
    pub fn new(input: String) -> Self {
        Self {
            input,
            ..Default::default()
        }
    }

    pub fn next_token(&mut self) -> Result<Token> {
        self.read_char()?;
        let token = match self.char {
            '=' => Token::Assign,
            ';' => Token::Semicolon,
            '(' => Token::LeftParentheses,
            ')' => Token::RightParentheses,
            ',' => Token::Comma,
            '+' => Token::Plus,
            '{' => Token::LeftBrace,
            '}' => Token::RightBrace,
            '\0' => Token::EndOfFile,
            _ => Token::Equal,
        };
        Ok(token)
    }

    pub fn read_char(&mut self) -> Result<()> {
        if self.read_position >= self.input.len() {
            self.char = '\0';
        } else {
            self.char = self
                .input
                .chars()
                .nth(self.read_position)
                .context("Tried to read an out of bounds character!")?;
        }
        self.position = self.read_position;
        self.read_position += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_token() -> Result<()> {
        let input = "=+(){},;".to_string();

        let tokens = [
            Token::Assign,
            Token::Plus,
            Token::LeftParentheses,
            Token::RightParentheses,
            Token::LeftBrace,
            Token::RightBrace,
            Token::Comma,
            Token::Semicolon,
            Token::EndOfFile,
        ];

        let mut lexer = Lexer::new(input);

        for token in tokens.iter() {
            assert_eq!(lexer.next_token()?, *token);
        }

        Ok(())
    }
}
