use self::Token::*;
use anyhow::{Result, bail};
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
    Continue,
    Defer,
    ErrDefer,
    Distinct,
    Dollar,
    DoubleColon,
    Dot,
    DotDot,
    DotDotEqual,
    Else,
    EndOfFile,
    Enum,
    Extern,
    Equal,
    For,
    Function,
    GreaterThan,
    GreaterThanOrEqual,
    Identifier(String),
    If,
    // A choice made on the target rather than at run time. A word rather than
    // a name, so the formatter reads `when (X) {` as the block it is instead of
    // as a call to something named `when`.
    When,
    Illegal(String),
    Import,
    In,
    Inline,
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
    Move,
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
    Safe,
    Semicolon,
    ShiftLeft,
    ShiftRight,
    Slash,
    StringLiteral(String),
    Struct,
    Linear,
    Match,
    Type,
    Underscore,
    Ref,
    Unsafe,
    Uses,
    Where,
    Ellipsis,
    While,
    With,
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
            Continue => "continue".to_string(),
            Defer => "defer".to_string(),
            ErrDefer => "errdefer".to_string(),
            Distinct => "distinct".to_string(),
            Dollar => "$".to_string(),
            Dot => ".".to_string(),
            DotDot => "..".to_string(),
            DotDotEqual => "..=".to_string(),
            DoubleColon => "::".to_string(),
            Else => "else".to_string(),
            EndOfFile => EOF_CHAR.to_string(),
            Enum => "enum".to_string(),
            Extern => "extern".to_string(),
            Equal => "==".to_string(),
            For => "for".to_string(),
            Function => "fn".to_string(),
            GreaterThan => ">".to_string(),
            GreaterThanOrEqual => ">=".to_string(),
            Identifier(value) => value.to_string(),
            If => "if".to_string(),
            When => "when".to_string(),
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
            Move => "move".to_string(),
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
            Safe => "safe".to_string(),
            Inline => "inline".to_string(),
            Semicolon => ";".to_string(),
            ShiftLeft => "<<".to_string(),
            ShiftRight => ">>".to_string(),
            Slash => "/".to_string(),
            StringLiteral(value) => value.to_string(),
            Struct => "struct".to_string(),
            Linear => "linear".to_string(),
            Match => "match".to_string(),
            Type => "type".to_string(),
            Underscore => "_".to_string(),
            Ref => "ref".to_string(),
            Unsafe => "unsafe".to_string(),
            Uses => "uses".to_string(),
            Where => "where".to_string(),
            Ellipsis => "...".to_string(),
            While => "while".to_string(),
            With => "with".to_string(),
        };
        write!(f, "{}", symbol)
    }
}

pub const EOF_CHAR: char = '\0';

macro_rules! keywords {
    ($($word:literal => $token:ident),* $(,)?) => {
        pub const KEYWORD_NAMES: &[&str] = &[$($word),*];

        fn lookup_identifier(identifier: &str) -> Token {
            match identifier {
                $($word => $token,)*
                _ => Identifier(identifier.to_string()),
            }
        }

        // The word a keyword token was written as. A field name is read in a
        // position where nothing else can appear, so a struct may carry a field
        // called `type` or `match` and the keyword is taken as the name there.
        // Reading it off the same table as the lexer is what keeps the two from
        // drifting apart.
        pub fn keyword_spelling(token: &Token) -> Option<&'static str> {
            match token {
                $($token => Some($word),)*
                _ => None,
            }
        }
    };
}

keywords! {
    "_" => Underscore,
    "fn" => Function,
    "mut" => Mut,
    "move" => Move,
    "return" => Return,
    "if" => If,
    "when" => When,
    "import" => Import,
    "else" => Else,
    "struct" => Struct,
    "linear" => Linear,
    "enum" => Enum,
    "extern" => Extern,
    "safe" => Safe,
    "inline" => Inline,
    "defer" => Defer,
    "errdefer" => ErrDefer,
    "while" => While,
    "for" => For,
    "in" => In,
    "distinct" => Distinct,
    "break" => Break,
    "continue" => Continue,
    "match" => Match,
    "type" => Type,
    "case" => Case,
    "ref" => Ref,
    "unsafe" => Unsafe,
    "uses" => Uses,
    "where" => Where,
    "with" => With,
}

#[derive(
    serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Default,
)]
pub struct Position {
    pub line: usize,
    pub column: usize,
    // Which file this came from, as an id into the source map. Imports flatten
    // every module into one statement list, so without this a diagnostic says
    // "line 12" and the reader looks up line 12 of the wrong file. The lexer
    // does not know which file it is reading, so it leaves this 0 and import
    // resolution stamps it, which is also where a module's provenance comes
    // from.
    #[serde(default)]
    pub file: u32,
}

impl Position {
    // How a diagnostic names this place. Falls back to the bare line and column
    // when the file is not known, which is the entry file and the tests.
    pub fn describe(&self) -> String {
        match crate::source_map::name_of(self.file) {
            Some(name) => {
                format!("{name}:{}:{}", self.line, self.column)
            }
            None => format!("line {}, column {}", self.line, self.column),
        }
    }

    // The file this place is in, by name alone. What a module is called
    // otherwise depends on where the build was started from, and a sentence
    // naming a file is compared word for word against the other compiler's.
    pub fn file_name(&self) -> Option<String> {
        let held = crate::source_map::name_of(self.file)?;
        Some(
            held.rsplit(['/', '\\'])
                .next()
                .unwrap_or(held.as_str())
                .to_string(),
        )
    }
}

pub struct Lexer<'a> {
    chars: Chars<'a>,
    line: usize,
    column: usize,
    token_start: Position,
    positions: Vec<Position>,
    // Where each token stops, which is where the cursor stands the moment the
    // token has been read. With the starts beside them this gives every token's
    // extent, and the gaps between those extents are exactly the whitespace and
    // the comments: what a formatter has to keep and what the token stream drops.
    // The lexer still says nothing about comments; it says where it was.
    ends: Vec<Position>,
    // What was wrong with the source, one entry per fault, each carrying the
    // token it was found in. A fault yields a placeholder token and lexing
    // continues, because a half-typed string is the normal state of a file
    // being edited and one bad character should not cost every diagnostic
    // after it. A program with any entry here is still refused: the parser
    // carries these forward and the build fails on them.
    diagnostics: Vec<crate::diagnostic::Diagnostic>,
}

impl<'a> Lexer<'a> {
    /// A byte-order mark is not part of the program. Windows editors write one
    /// on a UTF-8 save, and without this the first token of an otherwise valid
    /// file is an illegal character at line 1, column 1.
    pub fn new(input: &'a str) -> Lexer<'a> {
        Self {
            chars: input.strip_prefix('\u{feff}').unwrap_or(input).chars(),
            line: 1,
            column: 1,
            token_start: Position {
                line: 1,
                column: 1,
                file: 0,
            },
            positions: Vec::new(),
            ends: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    pub fn positions(&self) -> &[Position] {
        &self.positions
    }

    /// Where each token stops, one for one with `positions`.
    pub fn ends(&self) -> &[Position] {
        &self.ends
    }

    pub fn diagnostics(&self) -> &[crate::diagnostic::Diagnostic] {
        &self.diagnostics
    }

    /// The diagnostics with each position's file id filled in, for the caller
    /// that knows which file this source came from. The lexer itself does not.
    pub fn diagnostics_in_file(
        &self,
        file: u32,
    ) -> Vec<crate::diagnostic::Diagnostic> {
        self.diagnostics
            .iter()
            .map(|held| {
                crate::diagnostic::Diagnostic::new(
                    Position {
                        file,
                        ..held.position
                    },
                    held.message.clone(),
                )
            })
            .collect()
    }

    fn report(&mut self, message: String) {
        self.diagnostics.push(crate::diagnostic::Diagnostic {
            position: self.token_start,
            message,
            related: Vec::new(),
        });
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        loop {
            let next_token = self.next_token();
            if let Token::EndOfFile = next_token {
                break;
            }
            self.positions.push(self.token_start);
            self.ends.push(Position {
                line: self.line,
                column: self.column,
                file: 0,
            });
            tokens.push(next_token);
        }
        Ok(tokens)
    }

    fn next_token(&mut self) -> Token {
        self.skip_while(Self::is_whitespace);
        self.token_start = Position {
            line: self.line,
            column: self.column,
            file: 0,
        };
        let first_char = self.read_char();
        match first_char {
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
                            return Illegal("/*".to_string());
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
            '$' => Dollar,
            '.' => {
                if self.peek_nth(0) == '.' {
                    self.read_char();
                    if self.peek_nth(0) == '=' {
                        self.read_char();
                        DotDotEqual
                    } else if self.peek_nth(0) == '.' {
                        self.read_char();
                        Ellipsis
                    } else {
                        DotDot
                    }
                } else {
                    Dot
                }
            }
            '"' => {
                let mut literal = String::new();
                loop {
                    match self.peek_nth(0) {
                        EOF_CHAR => {
                            self.report(
                                "Reached end of file while scanning string. Expected closing delimiter '\"'."
                                    .to_string(),
                            );
                            break;
                        }
                        // A literal may span lines, and one written on a file
                        // saved with CRLF must be the same string as the same
                        // literal saved with LF. The carriage return is dropped
                        // so the program does not depend on how the file
                        // reached the compiler. `\r` still says the byte.
                        '\r' if self.peek_nth(1) == '\n' => {
                            self.read_char();
                        }
                        '"' => {
                            self.read_char();
                            break;
                        }
                        '\\' => {
                            self.read_char();
                            let escaped = self.read_char();
                            let resolved = match escaped {
                                'n' => '\n',
                                't' => '\t',
                                'r' => '\r',
                                '0' => '\0',
                                '\\' => '\\',
                                '"' => '"',
                                '\'' => '\'',
                                other => {
                                    self.report(format!(
                                        "Unknown escape sequence '\\{}' in string literal",
                                        other
                                    ));
                                    other
                                }
                            };
                            literal.push(resolved);
                        }
                        _ => literal.push(self.read_char()),
                    }
                }
                StringLiteral(literal)
            }
            EOF_CHAR => EndOfFile,
            c if Self::is_ident_start(c) => {
                let mut identifier = c.to_string();
                identifier.push_str(&self.take_while(Self::is_ident_char));
                lookup_identifier(&identifier)
            }
            // `0x` and `0b` are read before the decimal path, since both start
            // with a digit that is also a number on its own. A C header's
            // constants are written in hex, and transcribing them into decimal
            // by hand is a step where a digit goes missing quietly.
            '0' if matches!(self.peek_nth(0), 'x' | 'X') => {
                self.read_char();
                let digits = self.take_while(Self::is_hex_or_separator);
                let cleaned = digits.replace('_', "");
                if cleaned.is_empty() {
                    Illegal("0x".to_string())
                } else {
                    match Self::radix(&cleaned, 16) {
                        Ok(value) => Integer(value),
                        Err(error) => {
                            self.report(error.to_string());
                            Integer(0)
                        }
                    }
                }
            }
            '0' if matches!(self.peek_nth(0), 'b' | 'B') => {
                self.read_char();
                let digits = self.take_while(Self::is_binary_or_separator);
                let cleaned = digits.replace('_', "");
                if cleaned.is_empty() {
                    Illegal("0b".to_string())
                } else {
                    match Self::radix(&cleaned, 2) {
                        Ok(value) => Integer(value),
                        Err(error) => {
                            self.report(error.to_string());
                            Integer(0)
                        }
                    }
                }
            }
            c if Self::is_digit(c) => {
                let mut number = c.to_string();
                number.push_str(&self.take_while(Self::is_digit_or_separator));
                let mut is_float = false;
                if self.peek_nth(0) == '.' && self.peek_nth(1) != '.' {
                    is_float = true;
                    number.push(self.read_char());
                    number.push_str(
                        &self.take_while(Self::is_digit_or_separator),
                    );
                }
                // An exponent, so a graphics program can write 1e-6 rather than
                // a run of zeroes it has to count.
                if matches!(self.peek_nth(0), 'e' | 'E')
                    && (Self::is_digit(self.peek_nth(1))
                        || (matches!(self.peek_nth(1), '+' | '-')
                            && Self::is_digit(self.peek_nth(2))))
                {
                    is_float = true;
                    number.push(self.read_char());
                    if matches!(self.peek_nth(0), '+' | '-') {
                        number.push(self.read_char());
                    }
                    number.push_str(&self.take_while(Self::is_digit));
                }
                let number = number.replace('_', "");
                if is_float {
                    if self.peek_nth(0) == 'f' {
                        self.read_token_float_suffix();
                        match number.parse::<f32>() {
                            Ok(value) => Float32(value),
                            Err(_) => {
                                self.report(format!(
                                    "{number} is not a number"
                                ));
                                Float32(0.0)
                            }
                        }
                    } else {
                        match number.parse::<f64>() {
                            Ok(value) => Float(value),
                            Err(_) => {
                                self.report(format!(
                                    "{number} is not a number"
                                ));
                                Float(0.0)
                            }
                        }
                    }
                } else {
                    match number.parse::<i64>() {
                        Ok(value) => Integer(value),
                        Err(_) => {
                            self.report(format!(
                                "{number} does not fit in sixty-four bits"
                            ));
                            Integer(0)
                        }
                    }
                }
            }
            illegal => Illegal(illegal.to_string()),
        }
    }

    fn read_token_float_suffix(&mut self) {
        self.read_char();
        if self.peek_nth(0) == '3' && self.peek_nth(1) == '2' {
            self.read_char();
            self.read_char();
        }
    }

    fn read_char(&mut self) -> char {
        let character = self.chars.next().unwrap_or(EOF_CHAR);
        if character == '\n' {
            self.line += 1;
            self.column = 1;
        } else if character != EOF_CHAR {
            self.column += 1;
        }
        character
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

    // An underscore may go between digits, so a mask reads in groups. It is
    // dropped before the number is read, so it never reaches the value.
    fn is_digit_or_separator(c: char) -> bool {
        c.is_ascii_digit() || c == '_'
    }

    fn is_hex_or_separator(c: char) -> bool {
        c.is_ascii_hexdigit() || c == '_'
    }

    fn is_binary_or_separator(c: char) -> bool {
        c == '0' || c == '1' || c == '_'
    }

    // A hex or binary literal is read as unsigned and reinterpreted, so the
    // whole of a sixty-four bit mask can be written: `0xFFFFFFFFFFFFFFFF` is
    // the all-ones sentinel a C header spells that way, and it is past what an
    // i64 holds as a positive number.
    fn radix(digits: &str, base: u32) -> Result<i64> {
        match u64::from_str_radix(digits, base) {
            Ok(value) => Ok(value as i64),
            Err(_) => bail!("{digits} does not fit in sixty-four bits"),
        }
    }

    fn is_whitespace(c: char) -> bool {
        c == ' ' || c == '\t' || c == '\n' || c == '\r'
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
    fn a_bad_escape_still_lexes_the_rest_of_the_file() -> Result<()> {
        let mut lexer = Lexer::new("s := \"a\\qb\"\nx := 5");
        let tokens = lexer.tokenize()?;
        assert_eq!(lexer.diagnostics().len(), 1);
        assert!(
            lexer.diagnostics()[0]
                .message
                .contains("Unknown escape sequence '\\q'")
        );
        assert!(tokens.iter().any(|held| matches!(held, Token::Integer(5))));
        Ok(())
    }

    #[test]
    fn an_unterminated_string_reports_and_yields_what_it_has() -> Result<()> {
        let mut lexer = Lexer::new("s := \"abc");
        let tokens = lexer.tokenize()?;
        assert_eq!(lexer.diagnostics().len(), 1);
        assert!(
            lexer.diagnostics()[0]
                .message
                .contains("Reached end of file while scanning string")
        );
        assert!(matches!(
            tokens.last(),
            Some(Token::StringLiteral(text)) if text == "abc"
        ));
        Ok(())
    }

    #[test]
    fn an_overflowing_literal_reports_and_stands_in() -> Result<()> {
        let mut lexer = Lexer::new("x := 0xFFFFFFFFFFFFFFFFF");
        let tokens = lexer.tokenize()?;
        assert_eq!(lexer.diagnostics().len(), 1);
        assert!(
            lexer.diagnostics()[0]
                .message
                .contains("does not fit in sixty-four bits")
        );
        assert!(tokens.iter().any(|held| matches!(held, Token::Integer(0))));
        Ok(())
    }

    #[test]
    fn a_clean_file_lexes_with_nothing_to_say() -> Result<()> {
        let mut lexer = Lexer::new("x := \"fine\\n\"\ny := 0xFF");
        lexer.tokenize()?;
        assert!(lexer.diagnostics().is_empty());
        Ok(())
    }

    #[test]
    fn every_fault_in_a_file_is_reported() -> Result<()> {
        let mut lexer = Lexer::new(
            "a := \"one\\q\"\nb := 0b111111111111111111111111111111111111111111111111111111111111111111\nc := \"tail",
        );
        lexer.tokenize()?;
        assert_eq!(lexer.diagnostics().len(), 3);
        assert_eq!(lexer.diagnostics()[0].position.line, 1);
        assert_eq!(lexer.diagnostics()[1].position.line, 2);
        assert_eq!(lexer.diagnostics()[2].position.line, 3);
        Ok(())
    }

    #[test]
    fn a_byte_order_mark_is_not_a_token() -> Result<()> {
        check_tokens(
            "\u{feff}five := 5;",
            &[
                Token::Identifier("five".to_string()),
                Token::ColonAssign,
                Token::Integer(5),
                Token::Semicolon,
            ],
        )
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
    fn mutable_declaration() -> Result<()> {
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
                Token::Identifier("true".to_string()),
                Token::Semicolon,
                Token::RightBrace,
                Token::Else,
                Token::LeftBrace,
                Token::Return,
                Token::Identifier("false".to_string()),
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

    // The primitive type names and the boolean literals are predeclared
    // identifiers: the type parser and the expression parser are what give
    // them their meanings, and only there.
    #[test]
    fn type_names_are_identifiers() -> Result<()> {
        check_tokens(
            "i64 bool str true false",
            &[
                Token::Identifier("i64".to_string()),
                Token::Identifier("bool".to_string()),
                Token::Identifier("str".to_string()),
                Token::Identifier("true".to_string()),
                Token::Identifier("false".to_string()),
            ],
        )
    }

    #[test]
    fn declaration_tokens() -> Result<()> {
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
            "struct enum defer for in distinct sizeof",
            &[
                Token::Struct,
                Token::Enum,
                Token::Defer,
                Token::For,
                Token::In,
                Token::Distinct,
                Token::Identifier("sizeof".to_string()),
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
                Token::Identifier("i64".to_string()),
                Token::Comma,
                Token::Identifier("b".to_string()),
                Token::Colon,
                Token::Identifier("i64".to_string()),
                Token::RightParentheses,
                Token::Arrow,
                Token::Identifier("i64".to_string()),
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
    fn variable_declaration() -> Result<()> {
        check_tokens(
            "x := 5; y : i64 = 10;",
            &[
                Token::Identifier("x".to_string()),
                Token::ColonAssign,
                Token::Integer(5),
                Token::Semicolon,
                Token::Identifier("y".to_string()),
                Token::Colon,
                Token::Identifier("i64".to_string()),
                Token::Assign,
                Token::Integer(10),
                Token::Semicolon,
            ],
        )
    }

    #[test]
    fn struct_declaration() -> Result<()> {
        check_tokens(
            "Vec3 :: struct { x: f32, y: f32, z: f32, }",
            &[
                Token::Identifier("Vec3".to_string()),
                Token::DoubleColon,
                Token::Struct,
                Token::LeftBrace,
                Token::Identifier("x".to_string()),
                Token::Colon,
                Token::Identifier("f32".to_string()),
                Token::Comma,
                Token::Identifier("y".to_string()),
                Token::Colon,
                Token::Identifier("f32".to_string()),
                Token::Comma,
                Token::Identifier("z".to_string()),
                Token::Colon,
                Token::Identifier("f32".to_string()),
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
                Token::Identifier("i64".to_string()),
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
    fn inclusive_range_syntax() -> Result<()> {
        check_tokens(
            "for i in 0..=10 { }",
            &[
                Token::For,
                Token::Identifier("i".to_string()),
                Token::In,
                Token::Integer(0),
                Token::DotDotEqual,
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

    #[test]
    fn dollar_token() -> Result<()> {
        check_tokens(
            "$T $U",
            &[
                Token::Dollar,
                Token::Identifier("T".to_string()),
                Token::Dollar,
                Token::Identifier("U".to_string()),
            ],
        )
    }
}
