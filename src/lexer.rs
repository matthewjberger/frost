use self::Token::*;
use anyhow::{bail, Result};
use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    str::Chars,
};

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
pub enum Token {
    Assign,
    Asterisk,
    Bang,
    Comma,
    Colon,
    Else,
    EndOfFile,
    Equal,
    False,
    Function,
    GreaterThan,
    Identifier(String),
    If,
    Illegal(String),
    Integer(i64),
    LeftBrace,
    LeftBracket,
    LeftParentheses,
    LessThan,
    Let,
    Minus,
    NotEqual,
    Plus,
    Return,
    RightBrace,
    RightBracket,
    RightParentheses,
    StringLiteral(String),
    Semicolon,
    Slash,
    True,
}

impl Display for Token {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let symbol = match self {
            Assign => "=".to_string(),
            Asterisk => "*".to_string(),
            Bang => "!".to_string(),
            Comma => ",".to_string(),
            Colon => ":".to_string(),
            Else => "else".to_string(),
            EndOfFile => EOF_CHAR.to_string(),
            Equal => "==".to_string(),
            False => "false".to_string(),
            Function => "function".to_string(),
            GreaterThan => ">".to_string(),
            Identifier(value) => value.to_string(),
            If => "if".to_string(),
            Illegal(value) => value.to_string(),
            Integer(number) => number.to_string(),
            LeftBrace => "{".to_string(),
            LeftBracket => "[".to_string(),
            LeftParentheses => "(".to_string(),
            LessThan => "<".to_string(),
            Let => "let".to_string(),
            Minus => "-".to_string(),
            NotEqual => "!=".to_string(),
            Plus => "+".to_string(),
            Return => "return".to_string(),
            RightBrace => "}".to_string(),
            RightBracket => "]".to_string(),
            RightParentheses => ")".to_string(),
            StringLiteral(value) => value.to_string(),
            Semicolon => ";".to_string(),
            Slash => "/".to_string(),
            True => "true".to_string(),
        };
        write!(f, "{}", symbol)
    }
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

    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        loop {
            let next_token = self.next_token()?;
            if let Token::EndOfFile = next_token {
                break;
            }
            tokens.push(next_token);
        }
        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<Token> {
        self.skip_while(Self::is_whitespace);
        let first_char = self.read_char();
        let token = match first_char {
            '=' => self.next_char_or(Assign, '=', Equal),
            ';' => Semicolon,
            '(' => LeftParentheses,
            ')' => RightParentheses,
            ',' => Comma,
            ':' => Colon,
            '+' => Plus,
            '{' => LeftBrace,
            '}' => RightBrace,
            '[' => LeftBracket,
            ']' => RightBracket,
            '!' => self.next_char_or(Bang, '=', NotEqual),
            '<' => LessThan,
            '>' => GreaterThan,
            '-' => Minus,
            '*' => Asterisk,
            '/' => Slash,
            '"' => {
                let literal = self.take_while(|x| x != '"');
                match self.peek_nth(0) {
                    EOF_CHAR => bail!(
                        "Reached end of file while scanning string. Expected closing delimiter '\"'."
                    ),
                    '"' => {
                        self.read_char();
                    }
                    _ => bail!("String literal is missing closing delimiter"),
                }
                StringLiteral(literal)
            }
            EOF_CHAR => EndOfFile,
            c if Self::is_letter(c) => {
                let mut identifier = c.to_string();
                identifier.push_str(&self.take_while(Self::is_letter));
                Self::lookup_identifier(&identifier)
            }
            c if Self::is_digit(c) => {
                let mut number = c.to_string();
                number.push_str(&self.take_while(Self::is_digit));
                Integer(number.parse::<i64>()?)
            }
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

    fn take_while(
        &mut self,
        mut predicate: impl FnMut(char) -> bool,
    ) -> String {
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
            "true" => True,
            "false" => False,
            "return" => Return,
            "if" => If,
            "else" => Else,
            _ => Identifier(identifier.to_string()),
        }
    }

    fn next_char_or(
        &mut self,
        default: Token,
        next_char: char,
        token: Token,
    ) -> Token {
        match self.peek_nth(0) {
            c if c == next_char => {
                self.read_char();
                token
            }
            _ => default,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Lexer, Result, Token};

    fn check_tokens(input: &str, expected_tokens: &[Token]) -> Result<()> {
        let mut lexer = Lexer::new(input);
        for (token, expected_token) in
            lexer.tokenize()?.into_iter().zip(expected_tokens.iter())
        {
            assert_eq!(token, *expected_token);
        }
        Ok(())
    }

    #[test]
    fn let_statement() -> Result<()> {
        check_tokens(
            "let five = 5;",
            &[
                Token::Let,
                Token::Identifier("five".to_string()),
                Token::Assign,
                Token::Integer(5),
                Token::Semicolon,
            ],
        )
    }

    #[test]
    fn function_declaration() -> Result<()> {
        check_tokens(
            "let add = fn(x, y) { x + y; };",
            &[
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
            ],
        )
    }

    #[test]
    fn function_assignment() -> Result<()> {
        check_tokens(
            "let result = add(five, ten);",
            &[
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
            ],
        )
    }

    #[test]
    fn operators() -> Result<()> {
        check_tokens(
            "!-/*5;",
            &[
                Token::Bang,
                Token::Minus,
                Token::Slash,
                Token::Asterisk,
                Token::Integer(5),
                Token::Semicolon,
            ],
        )
    }

    #[test]
    fn comparisons() -> Result<()> {
        check_tokens(
            "5 < 10 > 5;",
            &[
                Token::Integer(5),
                Token::LessThan,
                Token::Integer(10),
                Token::GreaterThan,
                Token::Integer(5),
                Token::Semicolon,
            ],
        )
    }

    #[test]
    fn if_else() -> Result<()> {
        check_tokens(
            "if (5 < 10) { return true; } else { return false; }",
            &[
                Token::If,
                Token::LeftParentheses,
                Token::Integer(5),
                Token::LessThan,
                Token::Integer(10),
                Token::RightParentheses,
                Token::LeftBrace,
                Token::Return,
                Token::True,
                Token::Semicolon,
                Token::RightBrace,
                Token::Else,
                Token::LeftBrace,
                Token::Return,
                Token::False,
                Token::Semicolon,
                Token::RightBrace,
            ],
        )
    }

    #[test]
    fn equality() -> Result<()> {
        check_tokens(
            "10 == 10;",
            &[
                Token::Integer(10),
                Token::Equal,
                Token::Integer(10),
                Token::Semicolon,
            ],
        )
    }

    #[test]
    fn inequality() -> Result<()> {
        check_tokens(
            "10 != 9;",
            &[
                Token::Integer(10),
                Token::NotEqual,
                Token::Integer(9),
                Token::Semicolon,
            ],
        )
    }

    #[test]
    fn string_literals() -> Result<()> {
        check_tokens(
            "\"foobar\";\"foo bar\"",
            &[
                Token::StringLiteral("foobar".to_string()),
                Token::Semicolon,
                Token::StringLiteral("foo bar".to_string()),
            ],
        )
    }

    #[test]
    fn arrays() -> Result<()> {
        check_tokens(
            "[1, 2];",
            &[
                Token::LeftBracket,
                Token::Integer(1),
                Token::Comma,
                Token::Integer(2),
                Token::RightBracket,
                Token::Semicolon,
            ],
        )
    }

    #[test]
    fn hash_maps() -> Result<()> {
        check_tokens(
            "{\"foo\": \"bar\"}",
            &[
                Token::LeftBrace,
                Token::StringLiteral("foo".to_string()),
                Token::Colon,
                Token::StringLiteral("bar".to_string()),
                Token::RightBrace,
                Token::Semicolon,
            ],
        )
    }
}
