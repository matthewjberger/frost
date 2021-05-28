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
        let input = r"
let five = 5;
let ten = 10;
let add = fn(x, y) {
x + y;
};
let result = add(five, ten);
"
        .to_string();

        let tokens = [
            // let five = 5;
            Token::Let,
            Token::Identifier("five".to_string()),
            Token::Assign,
            Token::Integer("5".to_string()),
            Token::Semicolon,
            // let ten = 10;
            Token::Let,
            Token::Identifier("ten".to_string()),
            Token::Assign,
            Token::Integer("10".to_string()),
            Token::Semicolon,
            // let add = fn(x, y) { x + y; };
            Token::Let,
            Token::Identifier("add".to_string()),
            Token::Assign,
            Token::Function,
            Token::LeftParentheses,
            Token::Identifier("x".to_string()),
            Token::Comma,
            Token::Identifier("y".to_string()),
            Token::RightParentheses,
            Token::LeftBrace,
            Token::Identifier("x".to_string()),
            Token::Plus,
            Token::Identifier("y".to_string()),
            Token::Semicolon,
            Token::RightBrace,
            Token::Semicolon,
            // let result = add(five, ten);
            Token::Let,
            Token::Identifier("result".to_string()),
            Token::Assign,
            Token::Identifier("add".to_string()),
            Token::LeftParentheses,
            Token::Identifier("five".to_string()),
            Token::Comma,
            Token::Identifier("ten".to_string()),
            Token::RightParentheses,
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
