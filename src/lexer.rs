use self::Token::*;
use anyhow::{bail, Result};
use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    str::Chars,
};

#[allow(dead_code)]
#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Ampersand,
    And,
    Arrow,
    Assign,
    Asterisk,
    Bang,
    Break,
    Caret,
    Case,
    Colon,
    ColonAssign,
    Comma,
    Comptime,
    Continue,
    Defer,
    Distinct,
    DoubleColon,
    Dot,
    DotDot,
    Else,
    EndOfFile,
    Enum,
    Extern,
    Equal,
    False,
    For,
    Function,
    GreaterThan,
    GreaterThanOrEqual,
    Hash,
    Identifier(String),
    If,
    Illegal(String),
    Import,
    In,
    Integer(i64),
    Float(f64),
    Float32(f32),
    LeftBrace,
    LeftBracket,
    LeftParentheses,
    LessThan,
    LessThanOrEqual,
    Minus,
    Mut,
    NotEqual,
    Or,
    Percent,
    Pipe,
    Plus,
    Question,
    Return,
    RightBrace,
    RightBracket,
    RightParentheses,
    Semicolon,
    ShiftLeft,
    ShiftRight,
    Sizeof,
    Slash,
    StringLiteral(String),
    Struct,
    Match,
    True,
    Type,
    Typename,
    TypeBool,
    TypeF32,
    TypeF64,
    TypeI8,
    TypeI16,
    TypeI32,
    TypeI64,
    TypeStr,
    TypeU8,
    TypeU16,
    TypeU32,
    TypeU64,
    TypeVoid,
    Underscore,
    Unsafe,
    Using,
    While,
}

impl Display for Token {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let symbol = match self {
            Ampersand => "&".to_string(),
            And => "&&".to_string(),
            Arrow => "->".to_string(),
            Assign => "=".to_string(),
            Asterisk => "*".to_string(),
            Bang => "!".to_string(),
            Break => "break".to_string(),
            Caret => "^".to_string(),
            Case => "case".to_string(),
            Colon => ":".to_string(),
            ColonAssign => ":=".to_string(),
            Comma => ",".to_string(),
            Comptime => "comptime".to_string(),
            Continue => "continue".to_string(),
            Defer => "defer".to_string(),
            Distinct => "distinct".to_string(),
            Dot => ".".to_string(),
            DotDot => "..".to_string(),
            DoubleColon => "::".to_string(),
            Else => "else".to_string(),
            EndOfFile => EOF_CHAR.to_string(),
            Enum => "enum".to_string(),
            Extern => "extern".to_string(),
            Equal => "==".to_string(),
            False => "false".to_string(),
            For => "for".to_string(),
            Function => "fn".to_string(),
            GreaterThan => ">".to_string(),
            GreaterThanOrEqual => ">=".to_string(),
            Hash => "#".to_string(),
            Identifier(value) => value.to_string(),
            If => "if".to_string(),
            Illegal(value) => value.to_string(),
            Import => "import".to_string(),
            In => "in".to_string(),
            Integer(number) => number.to_string(),
            Float(number) => number.to_string(),
            Float32(number) => format!("{}f32", number),
            LeftBrace => "{".to_string(),
            LeftBracket => "[".to_string(),
            LeftParentheses => "(".to_string(),
            LessThan => "<".to_string(),
            LessThanOrEqual => "<=".to_string(),
            Minus => "-".to_string(),
            Mut => "mut".to_string(),
            NotEqual => "!=".to_string(),
            Or => "||".to_string(),
            Percent => "%".to_string(),
            Pipe => "|".to_string(),
            Plus => "+".to_string(),
            Question => "?".to_string(),
            Return => "return".to_string(),
            RightBrace => "}".to_string(),
            RightBracket => "]".to_string(),
            RightParentheses => ")".to_string(),
            Semicolon => ";".to_string(),
            ShiftLeft => "<<".to_string(),
            ShiftRight => ">>".to_string(),
            Sizeof => "sizeof".to_string(),
            Slash => "/".to_string(),
            StringLiteral(value) => value.to_string(),
            Struct => "struct".to_string(),
            Match => "match".to_string(),
            True => "true".to_string(),
            Type => "type".to_string(),
            Typename => "typename".to_string(),
            TypeBool => "bool".to_string(),
            TypeF32 => "f32".to_string(),
            TypeF64 => "f64".to_string(),
            TypeI8 => "i8".to_string(),
            TypeI16 => "i16".to_string(),
            TypeI32 => "i32".to_string(),
            TypeI64 => "i64".to_string(),
            TypeStr => "str".to_string(),
            TypeU8 => "u8".to_string(),
            TypeU16 => "u16".to_string(),
            TypeU32 => "u32".to_string(),
            TypeU64 => "u64".to_string(),
            TypeVoid => "void".to_string(),
            Underscore => "_".to_string(),
            Unsafe => "unsafe".to_string(),
            Using => "using".to_string(),
            While => "while".to_string(),
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
            ':' => match self.peek_nth(0) {
                ':' => {
                    self.read_char();
                    DoubleColon
                }
                '=' => {
                    self.read_char();
                    ColonAssign
                }
                _ => Colon,
            },
            '+' => Plus,
            '{' => LeftBrace,
            '}' => RightBrace,
            '[' => LeftBracket,
            ']' => RightBracket,
            '!' => self.next_char_or(Bang, '=', NotEqual),
            '<' => match self.peek_nth(0) {
                '<' => {
                    self.read_char();
                    ShiftLeft
                }
                '=' => {
                    self.read_char();
                    LessThanOrEqual
                }
                _ => LessThan,
            },
            '>' => match self.peek_nth(0) {
                '>' => {
                    self.read_char();
                    ShiftRight
                }
                '=' => {
                    self.read_char();
                    GreaterThanOrEqual
                }
                _ => GreaterThan,
            },
            '%' => Percent,
            '?' => Question,
            '-' => self.next_char_or(Minus, '>', Arrow),
            '*' => Asterisk,
            '/' => {
                if self.peek_nth(0) == '/' {
                    self.take_while(|c| c != '\n');
                    return self.next_token();
                }
                if self.peek_nth(0) == '*' {
                    self.read_char();
                    loop {
                        if self.is_eof() {
                            bail!("Unterminated block comment");
                        }
                        if self.peek_nth(0) == '*' && self.peek_nth(1) == '/' {
                            self.read_char();
                            self.read_char();
                            break;
                        }
                        self.read_char();
                    }
                    return self.next_token();
                }
                Slash
            }
            '^' => Caret,
            '&' => self.next_char_or(Ampersand, '&', And),
            '|' => {
                if self.peek_nth(0) == '|' {
                    self.read_char();
                    Or
                } else {
                    Pipe
                }
            }
            '#' => Hash,
            '.' => self.next_char_or(Dot, '.', DotDot),
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
            c if Self::is_ident_start(c) => {
                let mut identifier = c.to_string();
                identifier.push_str(&self.take_while(Self::is_ident_char));
                Self::lookup_identifier(&identifier)
            }
            c if Self::is_digit(c) => {
                let mut number = c.to_string();
                number.push_str(&self.take_while(Self::is_digit));
                if self.peek_nth(0) == '.' && self.peek_nth(1) != '.' {
                    number.push(self.read_char());
                    number.push_str(&self.take_while(Self::is_digit));
                    if self.peek_nth(0) == 'f' {
                        self.read_char();
                        if self.peek_nth(0) == '3' && self.peek_nth(1) == '2' {
                            self.read_char();
                            self.read_char();
                        }
                        Float32(number.parse::<f32>()?)
                    } else {
                        Float(number.parse::<f64>()?)
                    }
                } else {
                    Integer(number.parse::<i64>()?)
                }
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

    fn is_ident_start(c: char) -> bool {
        c.is_ascii_lowercase() || c.is_ascii_uppercase() || c == '_'
    }

    fn is_ident_char(c: char) -> bool {
        c.is_ascii_lowercase()
            || c.is_ascii_uppercase()
            || c.is_ascii_digit()
            || c == '_'
    }

    fn is_digit(c: char) -> bool {
        c.is_ascii_digit()
    }

    fn is_whitespace(c: char) -> bool {
        c == ' ' || c == '\t' || c == '\n' || c == '\r'
    }

    fn lookup_identifier(identifier: &str) -> Token {
        match identifier {
            "_" => Underscore,
            "fn" => Function,
            "mut" => Mut,
            "true" => True,
            "false" => False,
            "return" => Return,
            "if" => If,
            "import" => Import,
            "else" => Else,
            "struct" => Struct,
            "enum" => Enum,
            "extern" => Extern,
            "defer" => Defer,
            "using" => Using,
            "while" => While,
            "for" => For,
            "in" => In,
            "distinct" => Distinct,
            "sizeof" => Sizeof,
            "break" => Break,
            "continue" => Continue,
            "match" => Match,
            "type" => Type,
            "typename" => Typename,
            "case" => Case,
            "comptime" => Comptime,
            "i8" => TypeI8,
            "i16" => TypeI16,
            "i32" => TypeI32,
            "i64" => TypeI64,
            "u8" => TypeU8,
            "u16" => TypeU16,
            "u32" => TypeU32,
            "u64" => TypeU64,
            "f32" => TypeF32,
            "f64" => TypeF64,
            "bool" => TypeBool,
            "str" => TypeStr,
            "void" => TypeVoid,
            "unsafe" => Unsafe,
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
    fn declaration_statement() -> Result<()> {
        check_tokens(
            "five := 5;",
            &[
                Token::Identifier("five".to_string()),
                Token::ColonAssign,
                Token::Integer(5),
                Token::Semicolon,
            ],
        )
    }

    #[test]
    fn function_declaration() -> Result<()> {
        check_tokens(
            "add := fn(x, y) { x + y; };",
            &[
                Token::Identifier("add".to_string()),
                Token::ColonAssign,
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
    fn function_call() -> Result<()> {
        check_tokens(
            "result := add(five, ten);",
            &[
                Token::Identifier("result".to_string()),
                Token::ColonAssign,
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
    fn mut_declaration() -> Result<()> {
        check_tokens(
            "mut x := 5;",
            &[
                Token::Mut,
                Token::Identifier("x".to_string()),
                Token::ColonAssign,
                Token::Integer(5),
                Token::Semicolon,
            ],
        )
    }

    #[test]
    fn operators() -> Result<()> {
        check_tokens(
            "!- / *5;",
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

    #[test]
    fn type_keywords() -> Result<()> {
        check_tokens(
            "i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 bool str void",
            &[
                Token::TypeI8,
                Token::TypeI16,
                Token::TypeI32,
                Token::TypeI64,
                Token::TypeU8,
                Token::TypeU16,
                Token::TypeU32,
                Token::TypeU64,
                Token::TypeF32,
                Token::TypeF64,
                Token::TypeBool,
                Token::TypeStr,
                Token::TypeVoid,
            ],
        )
    }

    #[test]
    fn odin_style_tokens() -> Result<()> {
        check_tokens(
            ":: := -> ^ & .",
            &[
                Token::DoubleColon,
                Token::ColonAssign,
                Token::Arrow,
                Token::Caret,
                Token::Ampersand,
                Token::Dot,
            ],
        )
    }

    #[test]
    fn struct_and_other_keywords() -> Result<()> {
        check_tokens(
            "struct enum defer using for in distinct sizeof",
            &[
                Token::Struct,
                Token::Enum,
                Token::Defer,
                Token::Using,
                Token::For,
                Token::In,
                Token::Distinct,
                Token::Sizeof,
            ],
        )
    }

    #[test]
    fn typed_function() -> Result<()> {
        check_tokens(
            "add :: fn(a: i64, b: i64) -> i64 { return a + b; }",
            &[
                Token::Identifier("add".to_string()),
                Token::DoubleColon,
                Token::Function,
                Token::LeftParentheses,
                Token::Identifier("a".to_string()),
                Token::Colon,
                Token::TypeI64,
                Token::Comma,
                Token::Identifier("b".to_string()),
                Token::Colon,
                Token::TypeI64,
                Token::RightParentheses,
                Token::Arrow,
                Token::TypeI64,
                Token::LeftBrace,
                Token::Return,
                Token::Identifier("a".to_string()),
                Token::Plus,
                Token::Identifier("b".to_string()),
                Token::Semicolon,
                Token::RightBrace,
            ],
        )
    }

    #[test]
    fn odin_style_variable() -> Result<()> {
        check_tokens(
            "x := 5; y : i64 = 10;",
            &[
                Token::Identifier("x".to_string()),
                Token::ColonAssign,
                Token::Integer(5),
                Token::Semicolon,
                Token::Identifier("y".to_string()),
                Token::Colon,
                Token::TypeI64,
                Token::Assign,
                Token::Integer(10),
                Token::Semicolon,
            ],
        )
    }

    #[test]
    fn odin_style_struct() -> Result<()> {
        check_tokens(
            "Vec3 :: struct { x: f32, y: f32, z: f32, }",
            &[
                Token::Identifier("Vec3".to_string()),
                Token::DoubleColon,
                Token::Struct,
                Token::LeftBrace,
                Token::Identifier("x".to_string()),
                Token::Colon,
                Token::TypeF32,
                Token::Comma,
                Token::Identifier("y".to_string()),
                Token::Colon,
                Token::TypeF32,
                Token::Comma,
                Token::Identifier("z".to_string()),
                Token::Colon,
                Token::TypeF32,
                Token::Comma,
                Token::RightBrace,
            ],
        )
    }

    #[test]
    fn pointer_syntax() -> Result<()> {
        check_tokens(
            "p: ^i64 = &x; y := p^;",
            &[
                Token::Identifier("p".to_string()),
                Token::Colon,
                Token::Caret,
                Token::TypeI64,
                Token::Assign,
                Token::Ampersand,
                Token::Identifier("x".to_string()),
                Token::Semicolon,
                Token::Identifier("y".to_string()),
                Token::ColonAssign,
                Token::Identifier("p".to_string()),
                Token::Caret,
                Token::Semicolon,
            ],
        )
    }

    #[test]
    fn range_syntax() -> Result<()> {
        check_tokens(
            "for i in 0..10 { }",
            &[
                Token::For,
                Token::Identifier("i".to_string()),
                Token::In,
                Token::Integer(0),
                Token::DotDot,
                Token::Integer(10),
                Token::LeftBrace,
                Token::RightBrace,
            ],
        )
    }

    #[test]
    fn identifier_with_numbers() -> Result<()> {
        check_tokens(
            "x1 y2z abc123",
            &[
                Token::Identifier("x1".to_string()),
                Token::Identifier("y2z".to_string()),
                Token::Identifier("abc123".to_string()),
            ],
        )
    }

    #[test]
    fn comparison_operators_extended() -> Result<()> {
        check_tokens(
            "5 <= 10 >= 3;",
            &[
                Token::Integer(5),
                Token::LessThanOrEqual,
                Token::Integer(10),
                Token::GreaterThanOrEqual,
                Token::Integer(3),
                Token::Semicolon,
            ],
        )
    }

    #[test]
    fn modulo_operator() -> Result<()> {
        check_tokens(
            "10 % 3;",
            &[
                Token::Integer(10),
                Token::Percent,
                Token::Integer(3),
                Token::Semicolon,
            ],
        )
    }

    #[test]
    fn match_case_tokens() -> Result<()> {
        check_tokens(
            "match x { case 1: y case _: z }",
            &[
                Token::Match,
                Token::Identifier("x".to_string()),
                Token::LeftBrace,
                Token::Case,
                Token::Integer(1),
                Token::Colon,
                Token::Identifier("y".to_string()),
                Token::Case,
                Token::Underscore,
                Token::Colon,
                Token::Identifier("z".to_string()),
                Token::RightBrace,
            ],
        )
    }
}
