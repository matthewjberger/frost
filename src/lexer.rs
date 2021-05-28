use self::Token::*;
use anyhow::Result;
use std::str::Chars;

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
    Integer(i32),
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

pub const EOF_CHAR: char = '\0';

pub struct Lexer<'a> {
    chars: Chars<'a>,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Lexer<'a> {
        Self {
            chars: input.chars(),
        }
    }

    pub fn next_token(&mut self) -> Result<Token> {
        self.skip_while(Self::is_whitespace);
        let first_char = self.read_char();
        let token = match first_char {
            '=' => Assign,
            ';' => Semicolon,
            '(' => LeftParentheses,
            ')' => RightParentheses,
            ',' => Comma,
            '+' => Plus,
            '{' => LeftBrace,
            '}' => RightBrace,
            c if Self::is_letter(c) => {
                let mut identifier = c.to_string();
                identifier.push_str(&self.take_while(Self::is_letter));
                Self::lookup_identifier(&identifier)
            }
            c if Self::is_digit(c) => {
                let mut number = c.to_string();
                number.push_str(&self.take_while(Self::is_digit));
                Integer(number.parse::<i32>()?)
            }
            EOF_CHAR => EndOfFile,
            illegal => Illegal(illegal.to_string()),
        };
        Ok(token)
    }

    fn read_char(&mut self) -> char {
        self.chars.next().unwrap_or(EOF_CHAR)
    }

    fn peek_nth(&self, n: usize) -> char {
        self.chars.clone().nth(n).unwrap_or(EOF_CHAR)
    }

    fn is_eof(&self) -> bool {
        self.chars.as_str().is_empty()
    }

    fn take_while(&mut self, mut predicate: impl FnMut(char) -> bool) -> String {
        let mut chars = String::new();
        while predicate(self.peek_nth(0)) && !self.is_eof() {
            chars.push(self.read_char());
        }
        chars
    }

    fn skip_while(&mut self, mut predicate: impl FnMut(char) -> bool) {
        while predicate(self.peek_nth(0)) && !self.is_eof() {
            self.read_char();
        }
    }

    fn is_letter(c: char) -> bool {
        ('a'..='z').contains(&c) || ('A'..='Z').contains(&c) || c == '_'
    }

    fn is_digit(c: char) -> bool {
        ('0'..='9').contains(&c)
    }

    fn is_whitespace(c: char) -> bool {
        c == ' ' || c == '\t' || c == '\n' || c == '\r'
    }

    fn lookup_identifier(identifier: &str) -> Token {
        match identifier {
            "fn" => Function,
            "let" => Let,
            _ => Identifier(identifier.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_token() -> Result<()> {
        let input = r"let five = 5;
let ten = 10;
let add = fn(x, y) {
x + y;
};
let result = add(five, ten);
";

        let tokens = [
            // let five = 5;
            Token::Let,
            Token::Identifier("five".to_string()),
            Token::Assign,
            Token::Integer(5),
            Token::Semicolon,
            // let ten = 10;
            Token::Let,
            Token::Identifier("ten".to_string()),
            Token::Assign,
            Token::Integer(10),
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
