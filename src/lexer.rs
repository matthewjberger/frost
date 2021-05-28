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
            '=' => self.next_char_or(Assign, '=', Equal),
            ';' => Semicolon,
            '(' => LeftParentheses,
            ')' => RightParentheses,
            ',' => Comma,
            '+' => Plus,
            '{' => LeftBrace,
            '}' => RightBrace,
            '!' => self.next_char_or(Bang, '=', NotEqual),
            '<' => LessThan,
            '>' => GreaterThan,
            '-' => Minus,
            '*' => Asterisk,
            '/' => Slash,
            EOF_CHAR => EndOfFile,
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
            illegal => Illegal(illegal.to_string()),
        };
        Ok(token)
    }

    pub fn exhaust(&mut self) -> Result<Vec<Token>> {
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
            "true" => True,
            "false" => False,
            "return" => Return,
            "if" => If,
            "else" => Else,
            _ => Identifier(identifier.to_string()),
        }
    }

    fn next_char_or(&mut self, default: Token, next_char: char, token: Token) -> Token {
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
    use super::*;

    #[test]
    fn test_next_token() -> Result<()> {
        let input = r"let five = 5;
let ten = 10;
let add = fn(x, y) {
x + y;
};
let result = add(five, ten);
!-/*5;
5 < 10 > 5;

if (5 < 10) {
    return true;
} else {
    return false;
}

10 == 10;
10 != 9;
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
            // !-/*5;
            Token::Bang,
            Token::Minus,
            Token::Slash,
            Token::Asterisk,
            Token::Integer(5),
            Token::Semicolon,
            // 5 < 10 > 5;
            Token::Integer(5),
            Token::LessThan,
            Token::Integer(10),
            Token::GreaterThan,
            Token::Integer(5),
            Token::Semicolon,
            // if (5 < 10) { return true; } else { return false; }
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
            // 10 == 10;
            Token::Integer(10),
            Token::Equal,
            Token::Integer(10),
            Token::Semicolon,
            // 10 != 9;
            Token::Integer(10),
            Token::NotEqual,
            Token::Integer(9),
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
