use crate::ast::{
    Ast, EnumVariant, ExprId, Expression, FlagBit, ImportRename, Literal,
    Module, MultiBinding, NamedExpr, Parameter, Pattern, PatternBinding,
    PatternId, Range32, ReturnKind, ReturnSignature, ReturnValue, Statement,
    StmtId, StructField, SwitchCase, Symbol, TokenSpan,
};
use crate::{lexer::Position, lexer::Token, types::Type};
use anyhow::{Result, bail};
use std::{
    collections::HashMap,
    fmt::{Display, Formatter, Result as FmtResult},
    matches,
    slice::Iter,
};

pub type Identifier = String;

// The prefix a `test` block's generated function carries, so import resolution
// can recognize one without a second registry to keep in step.
pub const TEST_PREFIX: &str = "__frost_test_";

// How a parameter takes its argument. `read` (the default) is a shared borrow,
// `mut` an exclusive borrow, `move` takes ownership. These are the surface. A
// later pass turns them into the reference types the rest of the compiler
// already handles and inserts the borrows at call sites.
#[derive(
    serde::Serialize, serde::Deserialize, Debug, PartialEq, Clone, Copy, Default,
)]
pub enum ParamMode {
    #[default]
    Read,
    Write,
    Move,
    // Only on an `extern fn`, and only for an aggregate: the bytes go to C the
    // way C passes a struct by value, split across registers or pushed on the
    // stack by the target's rule. Every other mode hands C a pointer, which is
    // what most C APIs take and what Frost does internally, but a library that
    // takes a struct by value could not be called at all. See src/c_abi.rs.
    Value,
}

// A function whose signature declares a return, a failure set, or an allocation
// source is a `proc` (fully typed), not a bare `fn`.
fn signature_is_typed(signature: &ReturnSignature) -> bool {
    !matches!(signature.kind, ReturnKind::None) || !signature.uses.is_empty()
}

// The predeclared type names. Each is an ordinary identifier to the lexer and
// means its type wherever a type is read, ahead of any declaration going by
// the same name, so the meaning cannot be redeclared. `void` is only a type
// internally: no surface program needs to write it, so only the compiler's own
// round trip through `type_from_string` reads it back.
fn primitive_type(name: &str, internal: bool) -> Option<Type> {
    Some(match name {
        "i8" => Type::I8,
        "i16" => Type::I16,
        "i32" => Type::I32,
        "i64" => Type::I64,
        "isize" => Type::Isize,
        "u8" => Type::U8,
        "u16" => Type::U16,
        "u32" => Type::U32,
        "u64" => Type::U64,
        "usize" => Type::Usize,
        "f32" => Type::F32,
        "f64" => Type::F64,
        "bool" => Type::Bool,
        "str" => Type::Str,
        "void" if internal => Type::Void,
        _ => return None,
    })
}

// The scalar names a `flags` declaration may follow its word with. A wider
// set than the integers on purpose, so a float or bool there reaches the
// declaration's own refusal, which names the type.
fn is_scalar_type_name(name: &str) -> bool {
    matches!(
        name,
        "i8" | "i16"
            | "i32"
            | "i64"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "usize"
            | "f32"
            | "f64"
            | "bool"
    )
}

// The integer operators a constant expression may combine names with. Their
// presence after `Name :: OtherName` is what marks the whole thing a constant
// declaration rather than `Enum::Variant` access at statement position.
fn is_constant_operator(token: &Token) -> bool {
    matches!(
        token,
        Token::Plus
            | Token::Minus
            | Token::Asterisk
            | Token::Slash
            | Token::Percent
            | Token::ShiftLeft
            | Token::ShiftRight
            | Token::Ampersand
            | Token::Pipe
    )
}

#[derive(
    serde::Serialize, serde::Deserialize, Debug, PartialEq, Copy, Clone,
)]
pub enum Operator {
    Add,
    And,
    BitwiseAnd,
    BitwiseOr,
    Divide,
    Multiply,
    Modulo,
    Not,
    Negate,
    Or,
    ShiftLeft,
    ShiftRight,
    Subtract,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
    NotEqual,
}

impl Operator {
    pub fn from_token(token: &Token, is_prefix: bool) -> Result<Self> {
        Ok(match token {
            Token::Plus => Self::Add,
            Token::Slash => Self::Divide,
            Token::Asterisk => Self::Multiply,
            Token::Percent => Self::Modulo,
            Token::Bang => Self::Not,
            Token::Minus if is_prefix => Self::Negate,
            Token::Minus if !is_prefix => Self::Subtract,
            Token::LessThan => Self::LessThan,
            Token::LessThanOrEqual => Self::LessThanOrEqual,
            Token::GreaterThan => Self::GreaterThan,
            Token::GreaterThanOrEqual => Self::GreaterThanOrEqual,
            Token::Equal => Self::Equal,
            Token::NotEqual => Self::NotEqual,
            Token::And => Self::And,
            Token::Or => Self::Or,
            Token::Ampersand => Self::BitwiseAnd,
            Token::Pipe => Self::BitwiseOr,
            Token::ShiftLeft => Self::ShiftLeft,
            Token::ShiftRight => Self::ShiftRight,
            _ => bail!("Token is not an operator: {}", token),
        })
    }

    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::Equal
                | Self::NotEqual
                | Self::LessThan
                | Self::LessThanOrEqual
                | Self::GreaterThan
                | Self::GreaterThanOrEqual
        )
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let statement = match self {
            Self::Add => "+",
            Self::And => "&&",
            Self::BitwiseAnd => "&",
            Self::Subtract | Self::Negate => "-",
            Self::Divide => "/",
            Self::Multiply => "*",
            Self::Modulo => "%",
            Self::Not => "!",
            Self::Or => "||",
            Self::LessThan => "<",
            Self::LessThanOrEqual => "<=",
            Self::GreaterThan => ">",
            Self::GreaterThanOrEqual => ">=",
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::BitwiseOr => "|",
            Self::ShiftLeft => "<<",
            Self::ShiftRight => ">>",
        };
        write!(f, "{}", statement)
    }
}

#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub enum Precedence {
    Lowest,
    Range,
    LogicalOr,
    LogicalAnd,
    Equals,
    LessThanGreaterThan,
    BitwiseOr,
    BitwiseAnd,
    Shift,
    Sum,
    Product,
    Prefix,
    Call,
    Index,
    FieldAccess,
}

impl From<&Token> for Precedence {
    fn from(token: &Token) -> Self {
        match token {
            Token::Or => Self::LogicalOr,
            Token::And => Self::LogicalAnd,
            Token::Pipe => Self::BitwiseOr,
            Token::Ampersand => Self::BitwiseAnd,
            Token::ShiftLeft => Self::Shift,
            Token::ShiftRight => Self::Shift,
            Token::DotDot | Token::DotDotEqual => Self::Range,
            Token::Equal => Self::Equals,
            Token::NotEqual => Self::Equals,
            Token::LessThan => Self::LessThanGreaterThan,
            Token::LessThanOrEqual => Self::LessThanGreaterThan,
            Token::GreaterThan => Self::LessThanGreaterThan,
            Token::GreaterThanOrEqual => Self::LessThanGreaterThan,
            Token::Plus => Self::Sum,
            Token::Minus => Self::Sum,
            Token::Slash => Self::Product,
            Token::Asterisk => Self::Product,
            Token::Percent => Self::Product,
            Token::LeftParentheses => Self::Call,
            Token::LeftBracket => Self::Index,
            Token::Dot => Self::FieldAccess,
            Token::Caret => Self::FieldAccess,
            Token::Question => Self::FieldAccess,
            Token::LeftBrace => Self::Range,
            Token::DoubleColon => Self::FieldAccess,
            _ => Self::Lowest,
        }
    }
}

// The whole program the front end hands on: one arena and the top-level
// statements in order. Import resolution splices every module into one of
// these, which is the flat AST the passes walk.
pub type Program = Module;

// Reads a type back from its `Display` form. This is the compiler talking to
// itself, so unlike the surface it accepts the reference types that parameter
// mode lowering synthesizes.
pub fn type_from_string(source: &str) -> Result<Type> {
    let mut lexer = crate::lexer::Lexer::new(source);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(&tokens);
    parser.internal_types = true;
    parser.parse_type()
}

pub use crate::diagnostic::Diagnostic;

pub struct Parser<'a> {
    pub tokens: Iter<'a, Token>,
    // The arena every parse method pushes into. `parse` hands it out inside
    // the finished `Module`.
    ast: Ast,
    linear_types: std::collections::HashSet<String>,
    tests: Vec<(String, String)>,
    exports: Vec<String>,
    positions: &'a [Position],
    consumed: usize,
    pending_angle_close: usize,
    diagnostics: Vec<Diagnostic>,
    // Monomorphization names a specialization by rendering its type arguments
    // and reads them back with `type_from_string`, so `Type` has to survive a
    // round trip through its own `Display`. Reference types have no surface
    // spelling and are rejected in source, but they do occur internally once
    // parameter modes are lowered, so that one entry point accepts them.
    internal_types: bool,
    // Set while reading the thing a `for` walks. `for item in items {` ends
    // with a brace that opens the body, and a name followed by a brace is
    // otherwise a struct literal, so the literal is not available there. The
    // same rule `match` needs, which it gets by looking for `case`.
    no_struct_literal: bool,
    // Top-level `N :: 8` constants, read off the token stream before the parse
    // so that an array size may name one wherever it appears, including above
    // the line that declares it.
    integer_constants: HashMap<String, usize>,
    // Every generic struct and enum declared in this file, by name. A literal
    // may say which instance it is, `Pair<i64, bool> { .. }`, and telling that
    // from the comparison `a < b` is a question of whether the name is one of
    // these.
    generic_types: std::collections::HashSet<String>,
    // How many blocks deep the parse is. `name :: Type { .. }` is a declaration
    // at the top level and `Enum::Variant { .. }` inside a body, and the two
    // read the same token for token, so where it is written is what tells them
    // apart.
    block_depth: usize,
}

// Where a top-level declaration's value ends: at the head of the next one, or
// at the end of the file. A declaration head is a name followed by `::`, and
// `import` is the one that starts with a keyword instead. Everything nested is
// inside brackets of some kind, so only depth zero is looked at.
fn declaration_value_end(tokens: &[Token], start: usize) -> usize {
    let mut depth = 0usize;
    for index in start..tokens.len() {
        match &tokens[index] {
            Token::LeftBrace | Token::LeftParentheses | Token::LeftBracket => {
                depth += 1
            }
            Token::RightBrace
            | Token::RightParentheses
            | Token::RightBracket => depth = depth.saturating_sub(1),
            Token::EndOfFile => return index,
            Token::Import if depth == 0 => return index,
            Token::Identifier(_)
                if depth == 0
                    && index > start
                    && matches!(
                        tokens.get(index + 1),
                        Some(Token::DoubleColon)
                    ) =>
            {
                return index;
            }
            _ => {}
        }
    }
    tokens.len()
}

// An integer constant expression, evaluated over the constants already read.
// The expression itself was parsed by the ordinary expression parser, so the
// operators bind here exactly as they do everywhere else rather than by a
// second precedence table written to agree with the first.
fn evaluate_constant(
    ast: &Ast,
    expression: ExprId,
    known: &HashMap<String, i64>,
) -> Option<i64> {
    match ast.expr(expression) {
        Expression::Literal(Literal::Integer(value)) => Some(*value),
        Expression::Identifier(name) => known.get(ast.name(*name)).copied(),
        Expression::Prefix(Operator::Negate, inner) => {
            evaluate_constant(ast, *inner, known)?.checked_neg()
        }
        Expression::Infix(left, operator, right) => {
            let left = evaluate_constant(ast, *left, known)?;
            let right = evaluate_constant(ast, *right, known)?;
            match operator {
                Operator::Add => left.checked_add(right),
                Operator::Subtract => left.checked_sub(right),
                Operator::Multiply => left.checked_mul(right),
                Operator::Divide => left.checked_div(right),
                Operator::Modulo => left.checked_rem(right),
                Operator::ShiftLeft => u32::try_from(right)
                    .ok()
                    .and_then(|by| left.checked_shl(by)),
                Operator::ShiftRight => u32::try_from(right)
                    .ok()
                    .and_then(|by| left.checked_shr(by)),
                Operator::BitwiseAnd => Some(left & right),
                Operator::BitwiseOr => Some(left | right),
                _ => None,
            }
        }
        _ => None,
    }
}

// Top-level constants that name an integer. The value has to be known here
// because an array size is part of a type and a repeat count is expanded into
// elements, both of which happen while parsing, so neither can wait for a later
// pass.
//
// A constant that names anything else, or that reads a constant declared below
// it, is left out rather than half-read: it then means the same thing it always
// did, and using it as a length is an error naming the length rather than an
// array of the wrong size.
// The names declared as `Name :: struct($T: Type)` or the enum equivalent. A
// generic instance written as a literal names the type in expression position,
// where `Pair<i64, bool> {` would otherwise read as two comparisons, so which
// names can start one is settled before the parse rather than guessed at during
// it.
pub fn scan_generic_types(
    tokens: &[Token],
) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    for index in 0..tokens.len() {
        let Token::Identifier(name) = &tokens[index] else {
            continue;
        };
        if !matches!(tokens.get(index + 1), Some(Token::DoubleColon)) {
            continue;
        }
        if !matches!(tokens.get(index + 2), Some(Token::Struct | Token::Enum)) {
            continue;
        }
        if !matches!(tokens.get(index + 3), Some(Token::LeftParentheses)) {
            continue;
        }
        names.insert(name.clone());
    }
    names
}

fn scan_integer_constants(tokens: &[Token]) -> HashMap<String, usize> {
    // A name written as `$N` anywhere is a compile-time parameter, and inside
    // the declaration that takes it `[N]u8` means that parameter rather than a
    // constant of the same name. Rather than track where each is in scope, such
    // a name is never folded, so the generic reading always wins and the clash
    // is reported instead of silently taking the constant's value.
    let mut compile_time = std::collections::HashSet::new();
    for (index, token) in tokens.iter().enumerate() {
        if matches!(token, Token::Dollar)
            && let Some(Token::Identifier(name)) = tokens.get(index + 1)
        {
            compile_time.insert(name.clone());
        }
    }

    // In source order, so that a constant reading an earlier one sees it. The
    // scan runs before the parse, so "earlier" is the only order there is.
    let mut values: HashMap<String, i64> = HashMap::new();
    let mut depth = 0usize;
    for index in 0..tokens.len() {
        match &tokens[index] {
            Token::LeftBrace | Token::LeftParentheses | Token::LeftBracket => {
                depth += 1;
                continue;
            }
            Token::RightBrace
            | Token::RightParentheses
            | Token::RightBracket => {
                depth = depth.saturating_sub(1);
                continue;
            }
            _ => {}
        }
        if depth != 0 {
            continue;
        }
        let Token::Identifier(name) = &tokens[index] else {
            continue;
        };
        if !matches!(tokens.get(index + 1), Some(Token::DoubleColon)) {
            continue;
        }
        // Only a value can start with one of these, so a function, struct or
        // enum body is never parsed a second time just to find out it is not a
        // number.
        let starts_a_value = matches!(
            tokens.get(index + 2),
            Some(
                Token::Integer(_)
                    | Token::LeftParentheses
                    | Token::Minus
                    | Token::Identifier(_)
            )
        );
        if !starts_a_value || compile_time.contains(name) {
            continue;
        }
        let end = declaration_value_end(tokens, index + 2);
        let mut sub = Parser {
            tokens: tokens[index + 2..end].iter(),
            ast: Ast::default(),
            linear_types: std::collections::HashSet::new(),
            tests: Vec::new(),
            exports: Vec::new(),
            positions: &[],
            consumed: 0,
            pending_angle_close: 0,
            diagnostics: Vec::new(),
            internal_types: false,
            no_struct_literal: false,
            integer_constants: HashMap::new(),
            generic_types: std::collections::HashSet::new(),
            block_depth: 0,
        };
        let Ok(expression) = sub.parse_expression(Precedence::Lowest) else {
            continue;
        };
        if let Some(value) = evaluate_constant(&sub.ast, expression, &values) {
            values.insert(name.clone(), value);
        }
    }
    values
        .into_iter()
        .filter(|(_, value)| *value >= 0)
        .map(|(name, value)| (name, value as usize))
        .collect()
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens: tokens.iter(),
            ast: Ast::default(),
            linear_types: std::collections::HashSet::new(),
            tests: Vec::new(),
            exports: Vec::new(),
            positions: &[],
            consumed: 0,
            pending_angle_close: 0,
            diagnostics: Vec::new(),
            internal_types: false,
            no_struct_literal: false,
            integer_constants: scan_integer_constants(tokens),
            generic_types: scan_generic_types(tokens),
            block_depth: 0,
        }
    }

    pub fn with_positions(
        tokens: &'a [Token],
        positions: &'a [Position],
    ) -> Self {
        let ast = Ast {
            token_positions: positions.to_vec(),
            ..Ast::default()
        };
        Self {
            tokens: tokens.iter(),
            ast,
            linear_types: std::collections::HashSet::new(),
            tests: Vec::new(),
            exports: Vec::new(),
            positions,
            consumed: 0,
            pending_angle_close: 0,
            diagnostics: Vec::new(),
            internal_types: false,
            no_struct_literal: false,
            integer_constants: scan_integer_constants(tokens),
            generic_types: scan_generic_types(tokens),
            block_depth: 0,
        }
    }

    // Generic types this file did not declare but may name, which is every one
    // declared by a file it imports. Which names can start a literal is settled
    // before the parse, and a file that imports `Ordering` writes
    // `Ordering<Point> { .. }` exactly as the file declaring it does.
    pub fn also_generic(
        &mut self,
        names: std::collections::HashSet<String>,
    ) -> &mut Self {
        self.generic_types.extend(names);
        self
    }

    pub fn tests(&self) -> &[(String, String)] {
        &self.tests
    }

    pub fn exports(&self) -> &[String] {
        &self.exports
    }

    fn parse_export_line(&mut self) -> Result<()> {
        self.read_token();
        loop {
            let token = self.read_token().clone();
            match token {
                Token::Identifier(name) => self.exports.push(name),
                other => {
                    return Err(self.at_consumed(format!(
                        "Expected an identifier in export list, found '{other}'"
                    )));
                }
            }
            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            } else {
                break;
            }
        }
        Ok(())
    }

    fn is_type_arg_close(&self) -> bool {
        self.pending_angle_close > 0
            || matches!(
                self.peek_nth(0),
                Token::GreaterThan | Token::ShiftRight
            )
    }

    fn consume_type_arg_close(&mut self) -> Result<()> {
        if self.pending_angle_close > 0 {
            self.pending_angle_close -= 1;
            return Ok(());
        }
        match self.peek_nth(0) {
            Token::GreaterThan => {
                self.read_token();
                Ok(())
            }
            Token::ShiftRight => {
                self.read_token();
                self.pending_angle_close += 1;
                Ok(())
            }
            other => {
                let written = other.to_string();
                Err(self.here(format!(
                    "Expected '>' to close type arguments, found '{written}'"
                )))
            }
        }
    }

    pub fn linear_types(&self) -> &std::collections::HashSet<String> {
        &self.linear_types
    }

    fn current_position(&self) -> Option<Position> {
        if self.positions.is_empty() {
            return None;
        }
        let index = self.consumed.min(self.positions.len() - 1);
        Some(self.positions[index])
    }

    // The token index a node's span opens at, taken before the node's first
    // token is read. `span_from` closes the span over everything consumed
    // since.
    fn mark(&self) -> u32 {
        self.consumed as u32
    }

    fn span_from(&self, start: u32) -> TokenSpan {
        let last = self.consumed.saturating_sub(1).max(start as usize) as u32;
        TokenSpan { first: start, last }
    }

    fn intern_all(&mut self, names: &[String]) -> Range32 {
        let symbols: Vec<Symbol> =
            names.iter().map(|name| self.ast.intern(name)).collect();
        self.ast.add_symbol_list(&symbols)
    }

    // Whether the token about to be read sits on the same line as the one
    // before it. A call's `(` and an index's `[` bind to what is on their left,
    // and a statement ends at the line break rather than at a semicolon, so on
    // a new line they would take the previous statement as the thing being
    // called or indexed:
    //
    //     if (n == 1) { return 1 }
    //     (n + 7) / 8
    //
    // read as calling the `if` with `n + 7`, and failed with "cannot call a
    // value that is not a function pointer" pointing at the `if`.
    fn on_the_same_line(&self) -> bool {
        if self.positions.is_empty() || self.consumed == 0 {
            return true;
        }
        let here = self.consumed.min(self.positions.len() - 1);
        if here == 0 {
            return true;
        }
        self.positions[here].line == self.positions[here - 1].line
    }

    fn parse_test_statement(&mut self) -> Result<StmtId> {
        let start = self.mark();
        self.read_token();
        let name = match self.read_token() {
            Token::StringLiteral(text) => text.clone(),
            other => {
                let written = other.to_string();
                return Err(self.at_consumed(format!(
                    "Expected a string literal after 'test', found '{written}'"
                )));
            }
        };
        let body = self.parse_block()?;
        let function_name = format!("{TEST_PREFIX}{}", self.tests.len());
        self.tests.push((name, function_name.clone()));
        let span = self.span_from(start);
        let signature = self
            .ast
            .push_signature(ReturnSignature::plain(ReturnKind::None));
        let function = self.ast.push_expr(
            Expression::Function(Range32::EMPTY, signature, body),
            span,
        );
        let symbol = self.ast.intern(&function_name);
        Ok(self
            .ast
            .push_stmt(Statement::Constant(symbol, function), span))
    }

    pub fn parse(&mut self) -> Result<Module> {
        let (program, diagnostics) = self.parse_recovering();
        if diagnostics.is_empty() {
            return Ok(program);
        }
        let combined = diagnostics
            .iter()
            .map(|diagnostic| diagnostic.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        Err(anyhow::anyhow!(combined))
    }

    /// Parse the whole token stream, recovering at statement boundaries so a
    /// single malformed statement does not discard the rest of the file. The
    /// returned module holds every statement that parsed, and the diagnostics
    /// list holds one entry per error encountered, at the top level and inside
    /// function bodies alike.
    pub fn parse_recovering(&mut self) -> (Module, Vec<Diagnostic>) {
        let mut roots = Vec::new();
        loop {
            let position = self.current_position().unwrap_or_default();
            match self.parse_statement() {
                Ok(Some(statement)) => {
                    roots.push(statement);
                }
                Ok(None) => break,
                Err(error) => {
                    self.record_error(position, &error);
                    self.synchronize();
                    if matches!(self.peek_nth(0), Token::EndOfFile) {
                        break;
                    }
                }
            }
        }
        let module = Module {
            ast: std::mem::take(&mut self.ast),
            roots,
        };
        (module, std::mem::take(&mut self.diagnostics))
    }

    /// Diagnostics found before parsing began, the lexer's. They go first so
    /// the report reads in source order, and their presence fails the parse
    /// the way a parse error does: tolerance is about reporting more, never
    /// about accepting more.
    pub fn preload_diagnostics(&mut self, diagnostics: Vec<Diagnostic>) {
        let mut held = diagnostics;
        held.append(&mut self.diagnostics);
        self.diagnostics = held;
    }

    // `true` and `false` always mean the booleans in expression position, so
    // a binding, parameter or constant by either name could never be read
    // back. Refused where the declaration is read, in the words the
    // self-hosted compiler also says.
    fn refuse_literal_name(&self, name: &str) -> Result<()> {
        if name == "true" || name == "false" {
            return Err(self.at_consumed(format!(
                "'{name}' always means the boolean, so it cannot be declared as a name"
            )));
        }
        Ok(())
    }

    // The error a site raises when the token it is looking at is the mistake:
    // the position is that token's, captured now, not wherever the cursor
    // ends up once recovery has skipped ahead.
    fn here(&self, message: String) -> anyhow::Error {
        anyhow::Error::new(crate::diagnostic::LocatedError {
            position: self.current_position().unwrap_or_default(),
            message,
        })
    }

    // The same, for a site that has already consumed the offending token.
    fn at_consumed(&self, message: String) -> anyhow::Error {
        let position = if self.positions.is_empty() || self.consumed == 0 {
            Position::default()
        } else {
            let index = (self.consumed - 1).min(self.positions.len() - 1);
            self.positions[index]
        };
        anyhow::Error::new(crate::diagnostic::LocatedError {
            position,
            message,
        })
    }

    fn record_error(&mut self, fallback: Position, error: &anyhow::Error) {
        if let Some(located) =
            error.downcast_ref::<crate::diagnostic::LocatedError>()
        {
            self.diagnostics.push(Diagnostic {
                position: located.position,
                message: located.message.clone(),
                related: Vec::new(),
            });
            return;
        }
        let position = self.current_position().unwrap_or(fallback);
        self.diagnostics.push(Diagnostic {
            position,
            message: error.to_string(),
            related: Vec::new(),
        });
    }

    /// After a top-level error, skip tokens until the next token begins a
    /// top-level item (a `name ::` declaration, or an `import`, `export`, or
    /// `test`). At least one token is always consumed so recovery cannot loop.
    fn synchronize(&mut self) {
        if !matches!(self.peek_nth(0), Token::EndOfFile) {
            self.read_token();
        }
        while !matches!(self.peek_nth(0), Token::EndOfFile) {
            if self.at_statement_boundary() {
                return;
            }
            self.read_token();
        }
    }

    fn at_statement_boundary(&self) -> bool {
        match self.peek_nth(0) {
            Token::Import => true,
            Token::Identifier(name) if name == "export" || name == "test" => {
                true
            }
            Token::Identifier(_) => {
                matches!(self.peek_nth(1), Token::DoubleColon)
            }
            _ => false,
        }
    }

    /// After an error inside a block, skip tokens until the next token begins a
    /// statement or closes the block, without crossing the block's own closing
    /// brace. At least one token is always consumed so recovery cannot loop.
    fn synchronize_in_block(&mut self) {
        if !matches!(self.peek_nth(0), Token::EndOfFile | Token::RightBrace) {
            self.read_token();
        }
        while !matches!(self.peek_nth(0), Token::EndOfFile | Token::RightBrace)
        {
            if self.at_block_statement_boundary() {
                return;
            }
            self.read_token();
        }
    }

    fn at_block_statement_boundary(&self) -> bool {
        match self.peek_nth(0) {
            Token::Mut
            | Token::Var
            | Token::Return
            | Token::Defer
            | Token::For
            | Token::While
            | Token::Break
            | Token::Continue
            | Token::Import => true,
            // An identifier begins a statement when it opens a binding, an
            // assignment, or an expression statement (a call `f(...)`, a place
            // assignment `p^ = ...` / `a[i] = ...` / `s.f = ...`).
            Token::Identifier(_) => matches!(
                self.peek_nth(1),
                Token::ColonAssign
                    | Token::Colon
                    | Token::DoubleColon
                    | Token::Assign
                    | Token::LeftParentheses
                    | Token::Caret
                    | Token::LeftBracket
                    | Token::Dot
            ),
            _ => false,
        }
    }

    pub fn parse_statement(&mut self) -> Result<Option<StmtId>> {
        if let Token::Identifier(name) = self.peek_nth(0)
            && name == "export"
            && matches!(self.peek_nth(1), Token::Identifier(_))
        {
            self.parse_export_line()?;
            return self.parse_statement();
        }
        // A statement keyword at the top level heads no declaration, and
        // letting its arm below take it turns the fault into whichever
        // complaint the statement's own parse dies with, columns away from
        // the word that is wrong.
        if self.block_depth == 0
            && matches!(
                self.peek_nth(0),
                Token::Return
                    | Token::Defer
                    | Token::For
                    | Token::While
                    | Token::With
                    | Token::Break
                    | Token::Continue
                    | Token::Ref
                    | Token::If
                    | Token::Match
                    | Token::Unsafe
            )
        {
            let written = self.peek_nth(0).to_string();
            return Err(self.here(format!(
                "expected a declaration head, `import`, `export`, or `test`, found '{written}'"
            )));
        }
        Ok(match self.peek_nth(0) {
            Token::EndOfFile => None,
            Token::Identifier(name)
                if name == "test"
                    && matches!(self.peek_nth(1), Token::StringLiteral(_))
                    && matches!(self.peek_nth(2), Token::LeftBrace) =>
            {
                Some(self.parse_test_statement()?)
            }
            Token::Return => Some(self.parse_return_statement()?),
            Token::Defer => Some(self.parse_defer_statement()?),
            Token::For => Some(self.parse_for_statement()?),
            Token::While => Some(self.parse_while_statement()?),
            Token::With => Some(self.parse_with_statement()?),
            Token::Break => {
                let start = self.mark();
                self.read_token();
                if matches!(self.peek_nth(0), Token::Semicolon) {
                    self.read_token();
                }
                Some(
                    self.ast.push_stmt(Statement::Break, self.span_from(start)),
                )
            }
            Token::Continue => {
                let start = self.mark();
                self.read_token();
                if matches!(self.peek_nth(0), Token::Semicolon) {
                    self.read_token();
                }
                Some(
                    self.ast
                        .push_stmt(Statement::Continue, self.span_from(start)),
                )
            }
            Token::Import => Some(self.parse_import_statement()?),
            Token::Ref => Some(self.parse_ref_declaration()?),
            Token::Var
                if matches!(self.peek_nth(1), Token::Identifier(_))
                    && matches!(self.peek_nth(2), Token::Comma) =>
            {
                Some(self.parse_multiple_declaration()?)
            }
            Token::Var => Some(self.parse_mutable_declaration()?),
            // `mut` is one thing, the parameter mode. It used to declare an
            // assignable local as well, and the two meanings shared nothing
            // but the spelling, so the local form is `var` and this refusal
            // is what a reader migrating old code meets.
            Token::Mut => {
                return Err(self.here(
                    "`mut` marks a parameter that writes the caller's value; a local that is reassigned is declared with `var`".to_string(),
                ));
            }
            Token::Identifier(_)
                if matches!(self.peek_nth(1), Token::Comma) =>
            {
                Some(self.parse_multiple_declaration()?)
            }
            Token::Identifier(_)
                if matches!(self.peek_nth(1), Token::ColonAssign) =>
            {
                Some(self.parse_declaration(false)?)
            }
            Token::Identifier(_)
                if matches!(self.peek_nth(1), Token::Colon)
                    && !matches!(self.peek_nth(2), Token::Colon) =>
            {
                Some(self.parse_typed_declaration(false)?)
            }
            // A constant whose value begins with another constant by name,
            // `STRIDE :: POSITION + NORMAL`. The trailing operator is what tells
            // it apart from `Enum::Variant` access, which is an expression and is
            // never followed by an arithmetic operator where it appears.
            Token::Identifier(_)
                if matches!(self.peek_nth(1), Token::DoubleColon)
                    && matches!(self.peek_nth(2), Token::Identifier(_))
                    && is_constant_operator(self.peek_nth(3)) =>
            {
                Some(self.parse_constant_or_struct_statement()?)
            }
            // `InitFlags :: flags u32 { Video = 32 }`. The word is not a
            // keyword, so the shape after it is what says this is a
            // declaration rather than an expression that starts with a name.
            Token::Identifier(_)
                if matches!(self.peek_nth(1), Token::DoubleColon)
                    && self.at_flags_declaration(2) =>
            {
                Some(self.parse_constant_or_struct_statement()?)
            }
            // A constant that is a struct value, such as
            // `i64_ordering :: Ordering<i64> { less = i64_less }`, and a
            // constant that is another name, `DEPTH :: TEXTURE_DEPTH24`.
            //
            // Both are `Name :: OtherName` and the depth is what settles them.
            // Inside a body those tokens are `Enum::Variant`, an expression;
            // at the top level a variant on its own is a statement with no
            // effect and nothing writes one, so a declaration is the only thing
            // it can be. What follows used to have to say so, which meant a
            // name standing for a name was read as a path expression and the
            // constant it declared did not exist: every use of it came back as
            // an unknown variable, from a file that named it two lines up.
            Token::Identifier(_)
                if self.block_depth == 0
                    && matches!(self.peek_nth(1), Token::DoubleColon)
                    && matches!(self.peek_nth(2), Token::Identifier(_)) =>
            {
                Some(self.parse_constant_or_struct_statement()?)
            }
            // A constant whose value is a boolean literal. The words are
            // identifiers now, so they are named here rather than sitting in
            // the token list below.
            Token::Identifier(_)
                if matches!(self.peek_nth(1), Token::DoubleColon)
                    && matches!(
                        self.peek_nth(2),
                        Token::Identifier(word)
                            if word == "true" || word == "false"
                    ) =>
            {
                Some(self.parse_constant_or_struct_statement()?)
            }
            Token::Identifier(_)
                if matches!(self.peek_nth(1), Token::DoubleColon)
                    && matches!(
                        self.peek_nth(2),
                        Token::Struct
                            | Token::Linear
                            | Token::Enum
                            | Token::Distinct
                            | Token::Extern
                            | Token::Safe
                            | Token::Integer(_)
                            | Token::Float(_)
                            | Token::StringLiteral(_)
                            | Token::Function
                            | Token::Inline
                            | Token::Unsafe
                            | Token::LeftBracket
                            | Token::LeftBrace
                            | Token::Minus
                            | Token::Bang
                            | Token::LeftParentheses
                    ) =>
            {
                Some(self.parse_constant_or_struct_statement()?)
            }
            _ => {
                // A token no statement can start gets the dispatch's own
                // answer, naming what could stand here, rather than whichever
                // message the expression parser dies with after taking the
                // token as the start of one. At the top level nothing runs, so
                // an expression is refused along with everything else: only a
                // declaration can stand there.
                if self.block_depth == 0 {
                    let written = self.peek_nth(0).to_string();
                    return Err(self.here(format!(
                        "expected a declaration head, `import`, `export`, or `test`, found '{written}'"
                    )));
                }
                if !Self::can_begin_expression(self.peek_nth(0)) {
                    let written = self.peek_nth(0).to_string();
                    return Err(self.here(format!(
                        "expected a statement, found '{written}'"
                    )));
                }
                Some(self.parse_expression_statement()?)
            }
        })
    }

    fn parse_import_statement(&mut self) -> Result<StmtId> {
        let start = self.mark();
        self.read_token();
        let path = match self.read_token() {
            Token::StringLiteral(path) => path.clone(),
            _ => bail!("Expected string literal after 'import'"),
        };
        // `(insert as list_insert)`: everything else still arrives under its
        // own name. This is the last resort for two modules you cannot edit
        // that export the same name, so it renames the few that clash rather
        // than qualifying every use.
        let mut renames = Vec::new();
        if matches!(self.peek_nth(0), Token::LeftParentheses)
            && self.on_the_same_line()
        {
            self.read_token();
            while !matches!(self.peek_nth(0), Token::RightParentheses) {
                if matches!(self.peek_nth(0), Token::EndOfFile) {
                    bail!("Unexpected end of input in an import rename list");
                }
                let exported = match self.read_token() {
                    Token::Identifier(name) => name.to_string(),
                    other => bail!(
                        "Expected an exported name in an import rename, found {other}"
                    ),
                };
                match self.read_token() {
                    Token::Identifier(word) if word == "as" => {}
                    other => bail!(
                        "Expected 'as' after '{exported}' in an import rename, found {other}"
                    ),
                }
                let local = match self.read_token() {
                    Token::Identifier(name) => name.to_string(),
                    other => bail!(
                        "Expected the name to read '{exported}' under, found {other}"
                    ),
                };
                let exported = self.ast.intern(&exported);
                let local = self.ast.intern(&local);
                renames.push(ImportRename { exported, local });
                if matches!(self.peek_nth(0), Token::Comma) {
                    self.read_token();
                }
            }
            self.read_token();
        }
        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }
        let renames = self.ast.add_renames(&renames);
        Ok(self
            .ast
            .push_stmt(Statement::Import(path, renames), self.span_from(start)))
    }

    fn parse_defer_statement(&mut self) -> Result<StmtId> {
        let start = self.mark();
        self.read_token();
        let statement = self
            .parse_statement()?
            .ok_or_else(|| anyhow::anyhow!("Expected statement after defer"))?;
        Ok(self
            .ast
            .push_stmt(Statement::Defer(statement), self.span_from(start)))
    }

    fn parse_for_statement(&mut self) -> Result<StmtId> {
        let start = self.mark();
        self.read_token();

        let iterator = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            _ => bail!("Expected identifier after 'for'"),
        };
        self.refuse_literal_name(&iterator)?;
        let iterator = self.ast.intern(&iterator);
        // `for index, item in items` names the position as well as the element,
        // for the loops that need to know where they are.
        let second = if matches!(self.peek_nth(0), Token::Comma) {
            self.read_token();
            match self.read_token() {
                Token::Identifier(name) => {
                    let name = name.to_string();
                    self.refuse_literal_name(&name)?;
                    Some(self.ast.intern(&name))
                }
                other => {
                    let written = other.to_string();
                    return Err(self.at_consumed(format!(
                        "Expected a second name after ',' in a for loop, found '{written}'"
                    )));
                }
            }
        } else {
            None
        };

        if !matches!(self.read_token(), Token::In) {
            bail!("Expected 'in' after for loop iterator");
        }

        let held = self.no_struct_literal;
        self.no_struct_literal = true;
        let range = self.parse_expression(Precedence::Lowest);
        self.no_struct_literal = held;
        let range = range?;

        let body = self.parse_block()?;

        Ok(self.ast.push_stmt(
            Statement::For(iterator, second, range, body),
            self.span_from(start),
        ))
    }

    fn parse_while_statement(&mut self) -> Result<StmtId> {
        let start = self.mark();
        self.read_token();

        if !matches!(self.peek_nth(0), Token::LeftParentheses) {
            bail!("Expected '(' after 'while'");
        }
        self.read_token();

        let condition = self.parse_expression(Precedence::Lowest)?;

        if !matches!(self.peek_nth(0), Token::RightParentheses) {
            bail!("Expected ')' after while condition");
        }
        self.read_token();

        let body = self.parse_block()?;

        Ok(self.ast.push_stmt(
            Statement::While(condition, body),
            self.span_from(start),
        ))
    }

    fn parse_with_statement(&mut self) -> Result<StmtId> {
        let start = self.mark();
        self.read_token();

        let capability = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            other => {
                bail!(
                    "Expected a capability variable after 'with', found {other}"
                )
            }
        };
        let capability = self.ast.intern(&capability);

        let body = self.parse_block()?;

        Ok(self.ast.push_stmt(
            Statement::With(capability, body),
            self.span_from(start),
        ))
    }

    // `quotient, remainder := divide(a, b)`, and `var` in front of any name
    // that the body goes on to write. The list is names, not patterns, since
    // what it takes apart is a return type list rather than a value.
    fn parse_multiple_declaration(&mut self) -> Result<StmtId> {
        let start = self.mark();
        let mut bindings = Vec::new();
        loop {
            if matches!(self.peek_nth(0), Token::Mut) {
                return Err(self.here(
                    "`mut` marks a parameter that writes the caller's value; a local that is reassigned is declared with `var`".to_string(),
                ));
            }
            let mutable = if matches!(self.peek_nth(0), Token::Var) {
                self.read_token();
                true
            } else {
                false
            };
            let name = match self.read_token() {
                Token::Identifier(name) => name.to_string(),
                other => bail!("Expected a name to bind, found {other}"),
            };
            self.refuse_literal_name(&name)?;
            let name = self.ast.intern(&name);
            bindings.push(MultiBinding { name, mutable });
            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            } else {
                break;
            }
        }

        if !matches!(self.read_token(), Token::ColonAssign) {
            bail!("Expected ':=' after a list of names");
        }

        let value = self.parse_expression(Precedence::Lowest)?;
        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }

        let bindings = self.ast.add_bindings(&bindings);
        Ok(self.ast.push_stmt(
            Statement::LetMultiple(bindings, value),
            self.span_from(start),
        ))
    }

    fn parse_mutable_declaration(&mut self) -> Result<StmtId> {
        self.read_token();

        if matches!(self.peek_nth(1), Token::ColonAssign) {
            self.parse_declaration(true)
        } else if matches!(self.peek_nth(1), Token::Colon) {
            self.parse_typed_declaration(true)
        } else {
            bail!("Expected ':=' or ': type =' after 'var identifier'")
        }
    }

    fn parse_declaration(&mut self, mutable: bool) -> Result<StmtId> {
        let start = self.mark();
        let name = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            _ => bail!("Expected identifier"),
        };
        self.refuse_literal_name(&name)?;

        if !matches!(self.read_token(), Token::ColonAssign) {
            bail!("Expected ':='");
        }

        let value = self.parse_expression(Precedence::Lowest)?;

        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }

        let name = self.ast.intern(&name);
        Ok(self.ast.push_stmt(
            Statement::Let {
                name,
                type_annotation: None,
                value,
                mutable,
            },
            self.span_from(start),
        ))
    }

    // `ref name := place`: a borrow bound to a local, not a copy. It aliases the
    // element or field it names, so writing through it writes there, which is
    // how a container element is mutated across statements without a raw
    // pointer. It is the local counterpart of a `mut` parameter, and the region
    // check holds it to its frame the same way. Lowered to a BorrowMut, which is
    // the internal mutable reference parameter modes already produce.
    fn parse_ref_declaration(&mut self) -> Result<StmtId> {
        let start = self.mark();
        self.read_token();
        let name = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            _ => bail!("Expected identifier after 'ref'"),
        };
        self.refuse_literal_name(&name)?;
        if !matches!(self.read_token(), Token::ColonAssign) {
            bail!("Expected ':=' after 'ref identifier'");
        }
        let place = self.parse_expression(Precedence::Lowest)?;
        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }
        let span = self.span_from(start);
        let value = self.ast.push_expr(Expression::BorrowMut(place), span);
        let name = self.ast.intern(&name);
        Ok(self.ast.push_stmt(
            Statement::Let {
                name,
                type_annotation: None,
                value,
                mutable: false,
            },
            span,
        ))
    }

    fn parse_typed_declaration(&mut self, mutable: bool) -> Result<StmtId> {
        let start = self.mark();
        let name = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            _ => bail!("Expected identifier"),
        };
        self.refuse_literal_name(&name)?;

        if !matches!(self.read_token(), Token::Colon) {
            bail!("Expected ':'");
        }

        let type_annotation = Some(self.parse_type()?);

        if !matches!(self.read_token(), Token::Assign) {
            bail!("Expected '=' after type annotation");
        }

        let value = self.parse_expression(Precedence::Lowest)?;

        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }

        let name = self.ast.intern(&name);
        Ok(self.ast.push_stmt(
            Statement::Let {
                name,
                type_annotation,
                value,
                mutable,
            },
            self.span_from(start),
        ))
    }

    fn parse_constant_or_struct_statement(&mut self) -> Result<StmtId> {
        let start = self.mark();
        let identifier = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            _ => bail!("Expected identifier"),
        };
        self.refuse_literal_name(&identifier)?;

        if !matches!(self.read_token(), Token::DoubleColon) {
            bail!("Expected '::'");
        }

        if matches!(self.peek_nth(0), Token::Linear) {
            self.read_token();
            self.linear_types.insert(identifier.clone());
        }

        if matches!(self.peek_nth(0), Token::Struct) {
            self.read_token();
            let type_params = self.parse_generic_params()?;
            if !matches!(self.read_token(), Token::LeftBrace) {
                bail!("Expected '{{' after struct");
            }
            let mut fields = Vec::new();
            while self.peek_nth(0) != &Token::RightBrace {
                {
                    let field_name = self.read_field_name("a field name")?;
                    if !matches!(self.read_token(), Token::Colon) {
                        bail!("Expected ':' after field name");
                    }
                    let field_type = self.parse_type()?;
                    let field_name = self.ast.intern(&field_name);
                    fields.push(StructField {
                        name: field_name,
                        field_type,
                    });
                }
                if matches!(self.peek_nth(0), Token::Comma) {
                    self.read_token();
                }
            }
            self.read_token();
            if matches!(self.peek_nth(0), Token::Semicolon) {
                self.read_token();
            }
            let name = self.ast.intern(&identifier);
            let type_params = self.intern_all(&type_params);
            let fields = self.ast.add_struct_fields(fields);
            Ok(self.ast.push_stmt(
                Statement::Struct(name, type_params, fields),
                self.span_from(start),
            ))
        } else if matches!(self.peek_nth(0), Token::Enum) {
            self.read_token();
            // An enum takes type parameters exactly as a struct does, and for
            // the same reason. Without them there is no way to write a sum type
            // over an arbitrary element, so `Option<T>` and `Result<T, E>` would
            // have to be rewritten once per element type.
            let type_params = self.parse_generic_params()?;
            if !matches!(self.read_token(), Token::LeftBrace) {
                bail!("Expected '{{' after enum");
            }
            let mut variants = Vec::new();
            while self.peek_nth(0) != &Token::RightBrace {
                let variant_name = match self.read_token() {
                    Token::Identifier(name) => name.to_string(),
                    _ => bail!("Expected variant name"),
                };
                let fields = if matches!(self.peek_nth(0), Token::LeftBrace) {
                    self.read_token();
                    let mut variant_fields = Vec::new();
                    while self.peek_nth(0) != &Token::RightBrace {
                        let field_name = self.read_field_name(
                            "a field name in an enum variant",
                        )?;
                        if !matches!(self.read_token(), Token::Colon) {
                            bail!(
                                "Expected ':' after field name in enum variant"
                            );
                        }
                        let field_type = self.parse_type()?;
                        let field_name = self.ast.intern(&field_name);
                        variant_fields.push(StructField {
                            name: field_name,
                            field_type,
                        });
                        if matches!(self.peek_nth(0), Token::Comma) {
                            self.read_token();
                        }
                    }
                    self.read_token();
                    Some(self.ast.add_struct_fields(variant_fields))
                } else {
                    None
                };
                let variant_name = self.ast.intern(&variant_name);
                variants.push(EnumVariant {
                    name: variant_name,
                    fields,
                });
                if matches!(self.peek_nth(0), Token::Comma) {
                    self.read_token();
                }
            }
            self.read_token();
            if matches!(self.peek_nth(0), Token::Semicolon) {
                self.read_token();
            }
            let name = self.ast.intern(&identifier);
            let type_params = self.intern_all(&type_params);
            let variants = self.ast.add_enum_variants(&variants);
            Ok(self.ast.push_stmt(
                Statement::Enum(name, type_params, variants),
                self.span_from(start),
            ))
        } else if self.at_flags_declaration(0) {
            self.read_token();
            let repr = self.parse_type()?;
            if !repr.is_integer() {
                bail!(
                    "'{identifier}' is a set of bits, so it is written over an integer type; '{repr}' is not one"
                );
            }
            self.read_token();
            let mut bits = Vec::new();
            while self.peek_nth(0) != &Token::RightBrace {
                let name = match self.read_token() {
                    Token::Identifier(name) => name.to_string(),
                    other => bail!(
                        "a flags declaration names its bits, and '{other}' is not a name"
                    ),
                };
                if !matches!(self.read_token(), Token::Assign) {
                    bail!(
                        "'{name}' needs the number it stands for, written as '{name} = 32'"
                    );
                }
                let value = match self.read_token() {
                    Token::Integer(value) => *value,
                    other => bail!(
                        "a bit of '{identifier}' is a number a C header wrote down, and '{other}' is not one"
                    ),
                };
                let name = self.ast.intern(&name);
                bits.push(FlagBit { name, value });
                if matches!(self.peek_nth(0), Token::Comma) {
                    self.read_token();
                }
            }
            self.read_token();
            if matches!(self.peek_nth(0), Token::Semicolon) {
                self.read_token();
            }
            let name = self.ast.intern(&identifier);
            let bits = self.ast.add_flag_bits(&bits);
            Ok(self.ast.push_stmt(
                Statement::Flags(name, repr, bits),
                self.span_from(start),
            ))
        } else if matches!(self.peek_nth(0), Token::Distinct) {
            let typ = self.parse_type()?;
            if matches!(self.peek_nth(0), Token::Semicolon) {
                self.read_token();
            }
            let typ = match typ {
                Type::Distinct(_, inner) => {
                    Type::Distinct(identifier.clone(), inner)
                }
                other => other,
            };
            let name = self.ast.intern(&identifier);
            Ok(self.ast.push_stmt(
                Statement::TypeAlias(name, typ),
                self.span_from(start),
            ))
        } else if matches!(self.peek_nth(0), Token::Extern)
            || (matches!(self.peek_nth(0), Token::Safe)
                && matches!(self.peek_nth(1), Token::Extern))
        {
            let safe = matches!(self.peek_nth(0), Token::Safe);
            if safe {
                self.read_token();
            }
            self.read_token();
            if !matches!(self.read_token(), Token::Function) {
                bail!("Expected 'fn' after 'extern'");
            }
            if !matches!(self.read_token(), Token::LeftParentheses) {
                bail!("Expected '(' after 'fn'");
            }
            let mut params = Vec::new();
            while self.peek_nth(0) != &Token::RightParentheses {
                // An extern takes modes and compile-time parameters for the
                // same reason an ordinary function does. A callback
                // registration is declared as an extern, it takes its context
                // by `move`, and the handler it wants is a `$` parameter with a
                // function bound.
                let mode = match self.peek_nth(0) {
                    Token::Mut => {
                        self.read_token();
                        ParamMode::Write
                    }
                    Token::Move => {
                        self.read_token();
                        ParamMode::Move
                    }
                    // `value` is a word rather than a keyword, so a parameter
                    // may still be called `value`. What tells them apart is
                    // that a mode is followed by the name and a name is
                    // followed by its type.
                    Token::Identifier(word)
                        if word == "value"
                            && matches!(
                                self.peek_nth(1),
                                Token::Identifier(_)
                            ) =>
                    {
                        self.read_token();
                        ParamMode::Value
                    }
                    _ => ParamMode::Read,
                };
                if matches!(self.peek_nth(0), Token::Dollar) {
                    params.push(self.parse_compile_time_parameter()?);
                    if matches!(self.peek_nth(0), Token::Comma) {
                        self.read_token();
                    }
                    continue;
                }
                let param_name = match self.read_token() {
                    Token::Identifier(name) => name.to_string(),
                    _ => bail!("Expected parameter name"),
                };
                self.refuse_literal_name(&param_name)?;
                if !matches!(self.read_token(), Token::Colon) {
                    bail!("Expected ':' after parameter name");
                }
                let param_type = self.parse_type()?;
                let param_name = self.ast.intern(&param_name);
                params.push(Parameter {
                    name: param_name,
                    type_annotation: Some(param_type),
                    mutable: false,
                    mode,
                    compile_time_signature: None,
                    pack: false,
                });
                if matches!(self.peek_nth(0), Token::Comma) {
                    self.read_token();
                }
            }
            self.read_token();
            let return_type = if matches!(self.peek_nth(0), Token::Arrow) {
                self.read_token();
                Some(self.parse_type()?)
            } else {
                None
            };
            if matches!(self.peek_nth(0), Token::Semicolon) {
                self.read_token();
            }
            let name = self.ast.intern(&identifier);
            let params = self.ast.add_parameters(params);
            Ok(self.ast.push_stmt(
                Statement::Extern {
                    name,
                    params,
                    return_type,
                    safe,
                },
                self.span_from(start),
            ))
        } else if matches!(self.peek_nth(0), Token::Inline) {
            // `name :: inline fn(...)`: the inline hint is the self-hosted C
            // backend's to honor, where it forces the fold. The bootstrap is the
            // oracle and its native path is Cranelift, which does not inline, so
            // it accepts the keyword and emits an ordinary function.
            self.read_token();
            let expression = self.parse_expression(Precedence::Lowest)?;
            if matches!(self.peek_nth(0), Token::Semicolon) {
                self.read_token();
            }
            let name = self.ast.intern(&identifier);
            Ok(self.ast.push_stmt(
                Statement::Constant(name, expression),
                self.span_from(start),
            ))
        } else {
            let expression = self.parse_expression(Precedence::Lowest)?;
            if matches!(self.peek_nth(0), Token::Semicolon) {
                self.read_token();
            }
            let name = self.ast.intern(&identifier);
            Ok(self.ast.push_stmt(
                Statement::Constant(name, expression),
                self.span_from(start),
            ))
        }
    }

    fn parse_return_statement(&mut self) -> Result<StmtId> {
        let start = self.mark();
        if !matches!(self.read_token(), Token::Return) {
            bail!("Expected 'Return' token!");
        }

        let expression =
            if matches!(self.peek_nth(0), Token::Semicolon | Token::RightBrace)
            {
                let span = self.span_from(start);
                self.ast.push_expr(Expression::Tuple(Range32::EMPTY), span)
            } else {
                let first = self.parse_expression(Precedence::Lowest)?;
                // `return quotient, remainder`, the several values a return type
                // list declares. The multi-return lowering turns the list into
                // the one struct the function returns.
                if matches!(self.peek_nth(0), Token::Comma) {
                    let mut values = vec![first];
                    while matches!(self.peek_nth(0), Token::Comma) {
                        self.read_token();
                        values.push(self.parse_expression(Precedence::Lowest)?);
                    }
                    let values = self.ast.add_expr_list(&values);
                    self.ast.push_expr(
                        Expression::Tuple(values),
                        self.span_from(start),
                    )
                } else {
                    first
                }
            };

        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }

        Ok(self
            .ast
            .push_stmt(Statement::Return(expression), self.span_from(start)))
    }

    fn parse_expression_statement(&mut self) -> Result<StmtId> {
        let start = self.mark();
        let expression = self.parse_expression(Precedence::Lowest)?;
        let statement = if matches!(self.peek_nth(0), Token::Assign) {
            self.read_token();
            let rhs = self.parse_expression(Precedence::Lowest)?;
            Statement::Assignment(expression, rhs)
        } else {
            Statement::Expression(expression)
        };
        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }
        Ok(self.ast.push_stmt(statement, self.span_from(start)))
    }

    // The tokens the prefix arms of `parse_expression` answer, one list kept
    // beside them so the statement dispatch can name what it expected.
    // Ampersand is here because its arm refuses with the message that teaches
    // the borrow model, which beats naming the token. A token added there and
    // not here turns a precise complaint back into whichever inner message
    // surfaces, which is wrong but not silent.
    fn can_begin_expression(token: &Token) -> bool {
        matches!(
            token,
            Token::Identifier(_)
                | Token::StringLiteral(_)
                | Token::Integer(_)
                | Token::Float(_)
                | Token::Float32(_)
                | Token::Bang
                | Token::Minus
                | Token::Ampersand
                | Token::Dollar
                | Token::Dot
                | Token::LeftBrace
                | Token::LeftBracket
                | Token::LeftParentheses
                | Token::If
                | Token::Function
                | Token::Match
                | Token::Unsafe
        )
    }

    fn parse_expression(&mut self, precedence: Precedence) -> Result<ExprId> {
        let start = self.mark();
        let mut expression = match self.peek_nth(0) {
            // `sizeof(T)`, `typename(T)` and `type_id(T)` read as calls and
            // take a type, so they are recognized here rather than left to the
            // ordinary call path, which would have to parse a type as an
            // expression. What comes out is the same `Call` every other builtin
            // is: the type rides along as an argument, so no pass has a node
            // form to enumerate for these.
            Token::Identifier(word)
                if matches!(
                    word.as_str(),
                    "sizeof" | "typename" | "type_id"
                ) && matches!(self.peek_nth(1), Token::LeftParentheses) =>
            {
                let word = word.clone();
                self.read_token();
                self.read_token();
                if word != "sizeof" && matches!(self.peek_nth(0), Token::Dollar)
                {
                    self.read_token();
                }
                let held = self.parse_type()?;
                if !matches!(self.read_token(), Token::RightParentheses) {
                    bail!("Expected ')' after the type in {word}");
                }
                let span = self.span_from(start);
                let argument =
                    self.ast.push_expr(Expression::TypeValue(held), span);
                let callee = self.ast.intern(&word);
                let callee =
                    self.ast.push_expr(Expression::Identifier(callee), span);
                let arguments = self.ast.add_expr_list(&[argument]);
                self.ast
                    .push_expr(Expression::Call(callee, arguments), span)
            }
            // The boolean literals. Predeclared meanings rather than reserved
            // words: the words always mean the booleans in expression
            // position, and declaring either as a name is refused where the
            // declaration is read.
            Token::Identifier(word) if word == "true" => {
                self.read_token();
                self.ast
                    .push_expr(Expression::Boolean(true), self.span_from(start))
            }
            Token::Identifier(word) if word == "false" => {
                self.read_token();
                self.ast.push_expr(
                    Expression::Boolean(false),
                    self.span_from(start),
                )
            }
            Token::Identifier(identifier) => {
                let identifier = identifier.to_string();
                // `Pair<i64, bool> { .. }`: the literal says which instance it
                // is. What comes out is the instance's name, so the literal
                // itself is read the way every other one is.
                if matches!(self.peek_nth(1), Token::LessThan)
                    && self.generic_types.contains(&identifier)
                {
                    self.read_token();
                    let instance =
                        self.parse_generic_instance_name(&identifier)?;
                    let symbol = self.ast.intern(&instance);
                    self.ast.push_expr(
                        Expression::Identifier(symbol),
                        self.span_from(start),
                    )
                } else {
                    self.read_token();
                    let symbol = self.ast.intern(&identifier);
                    self.ast.push_expr(
                        Expression::Identifier(symbol),
                        self.span_from(start),
                    )
                }
            }
            Token::StringLiteral(string) => {
                let string = string.to_string();
                self.read_token();
                self.ast.push_expr(
                    Expression::Literal(Literal::String(string)),
                    self.span_from(start),
                )
            }
            Token::Integer(value) => {
                let value = *value;
                self.read_token();
                self.ast.push_expr(
                    Expression::Literal(Literal::Integer(value)),
                    self.span_from(start),
                )
            }
            Token::Float(value) => {
                let value = *value;
                self.read_token();
                self.ast.push_expr(
                    Expression::Literal(Literal::Float(value)),
                    self.span_from(start),
                )
            }
            Token::Float32(value) => {
                let value = *value;
                self.read_token();
                self.ast.push_expr(
                    Expression::Literal(Literal::Float32(value)),
                    self.span_from(start),
                )
            }
            Token::Bang | Token::Minus => self.parse_prefix_expression()?,
            Token::Ampersand => {
                bail!(
                    "a borrow `&`/`&mut` is not surface syntax; pass a plain value and the compiler borrows for the parameter mode, or take a raw pointer with ptr_to(x)"
                );
            }
            Token::Dollar => {
                self.read_token();
                let held = self.parse_type()?;
                self.ast.push_expr(
                    Expression::TypeValue(held),
                    self.span_from(start),
                )
            }
            // `.Circle { radius = 5 }` where the type is already known, the
            // construction counterpart of the `case .Circle` a pattern writes.
            // The enum name is left empty and filled in from what the context
            // expects.
            Token::Dot => self.parse_inferred_variant()?,
            // `{ x = 1, y = 2 }`, a struct literal that leaves out a type name
            // the context already carries. The name is empty here and the
            // lowering fills it in. Every field is still named: there is no
            // positional form of this literal or of any other.
            Token::LeftBrace => {
                let empty = self.ast.intern("");
                self.parse_struct_init(empty)?
            }
            Token::LeftBracket => self.parse_array_literal()?,
            Token::LeftParentheses => self.parse_grouped_expressions()?,
            Token::If => self.parse_if_expression()?,
            Token::Function => self.parse_function_literal()?,
            Token::Match => self.parse_match_expression()?,
            Token::Unsafe => self.parse_unsafe_expression()?,
            Token::EndOfFile => {
                bail!("Unexpected end of file")
            }
            token => {
                let written = token.to_string();
                return Err(self.here(format!(
                    "Token not valid for an expression: '{written}'"
                )));
            }
        };

        while self.peek_nth(0) != &Token::Semicolon
            && precedence < Precedence::from(self.peek_nth(0))
        {
            match self.peek_nth(0) {
                // A `-` that opens a line negates what follows it rather than
                // subtracting it from the statement above, since a statement
                // ends at the line break and `-x` is a statement of its own.
                // The other operators have no prefix form, so a line that opens
                // with one can only be the continuation of the line above it.
                Token::Minus if !self.on_the_same_line() => break,
                Token::Plus
                | Token::Minus
                | Token::Slash
                | Token::Asterisk
                | Token::Percent
                | Token::Equal
                | Token::NotEqual
                | Token::LessThan
                | Token::LessThanOrEqual
                | Token::GreaterThan
                | Token::GreaterThanOrEqual
                | Token::And
                | Token::Or
                | Token::Ampersand
                | Token::Pipe
                | Token::ShiftLeft
                | Token::ShiftRight => {
                    expression = self.parse_infix_expression(expression)?;
                }
                Token::DotDot => {
                    expression =
                        self.parse_range_expression(expression, false)?;
                }
                Token::DotDotEqual => {
                    expression =
                        self.parse_range_expression(expression, true)?;
                }
                Token::LeftBracket => {
                    if !self.on_the_same_line() {
                        break;
                    }
                    expression = self.parse_index_expression(expression)?;
                }
                Token::LeftParentheses => {
                    if !self.on_the_same_line() {
                        break;
                    }
                    expression = self.parse_call_expression(expression)?;
                }
                Token::Dot => {
                    expression = self.parse_field_access(expression)?;
                }
                Token::Caret => {
                    expression = self.parse_dereference(expression)?;
                }
                Token::Question => {
                    self.read_token();
                    expression = self.ast.push_expr(
                        Expression::Try(expression),
                        self.span_covering(expression),
                    );
                }
                Token::LeftBrace => {
                    if self.peek_nth(1) == &Token::Case
                        || self.no_struct_literal
                    {
                        return Ok(expression);
                    }
                    if let Expression::Identifier(name) =
                        self.ast.expr(expression)
                    {
                        let name = *name;
                        expression = self.parse_struct_init(name)?;
                    } else {
                        return Ok(expression);
                    }
                }
                Token::DoubleColon => {
                    if let Expression::Identifier(enum_name) =
                        self.ast.expr(expression)
                    {
                        let enum_name = *enum_name;
                        self.read_token();
                        let variant_name = match self.read_token() {
                            Token::Identifier(v) => v.to_string(),
                            _ => bail!("Expected identifier after '::'"),
                        };
                        let variant_name = self.ast.intern(&variant_name);
                        if matches!(self.peek_nth(0), Token::LeftBrace) {
                            self.read_token();
                            let mut fields = Vec::new();
                            while self.peek_nth(0) != &Token::RightBrace {
                                let field_name = self.read_field_name(
                                    "a field name in an enum variant literal",
                                )?;
                                if !matches!(self.read_token(), Token::Assign) {
                                    bail!(
                                        "Expected '=' after field name in enum variant init"
                                    );
                                }
                                let value =
                                    self.parse_expression(Precedence::Lowest)?;
                                let name = self.ast.intern(&field_name);
                                fields.push(NamedExpr { name, value });
                                if matches!(self.peek_nth(0), Token::Comma) {
                                    self.read_token();
                                }
                            }
                            self.read_token();
                            let fields = self.ast.add_named_exprs(&fields);
                            expression = self.ast.push_expr(
                                Expression::EnumVariantInit(
                                    enum_name,
                                    variant_name,
                                    fields,
                                ),
                                self.span_from(start),
                            );
                        } else {
                            expression = self.ast.push_expr(
                                Expression::EnumVariantInit(
                                    enum_name,
                                    variant_name,
                                    Range32::EMPTY,
                                ),
                                self.span_from(start),
                            );
                        }
                    } else {
                        return Ok(expression);
                    }
                }
                _ => return Ok(expression),
            };
        }

        Ok(expression)
    }

    // The span an expression grown leftward carries: from the first token of
    // the node it grew out of, through everything consumed since.
    fn span_covering(&self, left: ExprId) -> TokenSpan {
        let first = self.ast.expr_span(left).first;
        let last = self.consumed.saturating_sub(1).max(first as usize) as u32;
        TokenSpan { first, last }
    }

    // `.Variant` or `.Variant { field = value }`. The enum is whatever the
    // context expects, so the name is empty here and the lowering fills it in.
    fn parse_inferred_variant(&mut self) -> Result<ExprId> {
        let start = self.mark();
        self.read_token();
        let variant = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            other => bail!("Expected a variant name after '.', found {other}"),
        };
        let mut fields = Range32::EMPTY;
        if matches!(self.peek_nth(0), Token::LeftBrace)
            && !self.no_struct_literal
        {
            let empty = self.ast.intern("");
            let literal = self.parse_struct_init(empty)?;
            let Expression::StructInit(_, parsed) = self.ast.expr(literal)
            else {
                bail!("Expected the fields of a variant literal");
            };
            fields = *parsed;
        }
        let empty = self.ast.intern("");
        let variant = self.ast.intern(&variant);
        Ok(self.ast.push_expr(
            Expression::EnumVariantInit(empty, variant, fields),
            self.span_from(start),
        ))
    }

    fn parse_prefix_expression(&mut self) -> Result<ExprId> {
        let start = self.mark();
        let operator = Operator::from_token(self.peek_nth(0), true)?;
        self.read_token();
        let inner = self.parse_expression(Precedence::Prefix)?;
        Ok(self.ast.push_expr(
            Expression::Prefix(operator, inner),
            self.span_from(start),
        ))
    }

    fn parse_infix_expression(
        &mut self,
        left_expression: ExprId,
    ) -> Result<ExprId> {
        let operator = Operator::from_token(self.peek_nth(0), false)?;
        let precedence = Precedence::from(self.peek_nth(0));
        self.read_token();
        let right = self.parse_expression(precedence)?;
        Ok(self.ast.push_expr(
            Expression::Infix(left_expression, operator, right),
            self.span_covering(left_expression),
        ))
    }

    fn parse_range_expression(
        &mut self,
        left_expression: ExprId,
        inclusive: bool,
    ) -> Result<ExprId> {
        self.read_token();
        let right_expression = self.parse_expression(Precedence::Range)?;
        Ok(self.ast.push_expr(
            Expression::Range(left_expression, right_expression, inclusive),
            self.span_covering(left_expression),
        ))
    }

    fn parse_array_literal(&mut self) -> Result<ExprId> {
        let start = self.mark();
        self.read_token();
        if matches!(self.peek_nth(0), Token::RightBracket) {
            self.read_token();
            return Ok(self.ast.push_expr(
                Expression::Literal(Literal::Array(Range32::EMPTY)),
                self.span_from(start),
            ));
        }
        let first = self.parse_expression(Precedence::Lowest)?;
        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
            let token = self.read_token().clone();
            let count = match &token {
                Token::Integer(value) => *value as usize,
                Token::Identifier(name)
                    if self.integer_constants.contains_key(name) =>
                {
                    self.integer_constants[name]
                }
                // A name that is not a constant is a generic's value
                // parameter, whose number arrives with the instantiation. The
                // literal is carried unexpanded until then.
                Token::Identifier(name) => {
                    if !matches!(self.read_token(), Token::RightBracket) {
                        bail!("Expected ']' after an array repeat count");
                    }
                    let name = self.ast.intern(name);
                    return Ok(self.ast.push_expr(
                        Expression::ArrayRepeat(first, name),
                        self.span_from(start),
                    ));
                }
                token => {
                    let written = token.to_string();
                    return Err(self.at_consumed(format!(
                        "Expected a count after ';' in an array literal, found '{written}'"
                    )));
                }
            };
            if !matches!(self.read_token(), Token::RightBracket) {
                bail!("Expected ']' after an array repeat count");
            }
            let elements = vec![first; count];
            let elements = self.ast.add_expr_list(&elements);
            return Ok(self.ast.push_expr(
                Expression::Literal(Literal::Array(elements)),
                self.span_from(start),
            ));
        }
        let mut elements = vec![first];
        if matches!(self.peek_nth(0), Token::Comma) {
            self.read_token();
        }
        while self.peek_nth(0) != &Token::RightBracket
            && self.peek_nth(0) != &Token::EndOfFile
        {
            elements.push(self.parse_expression(Precedence::Lowest)?);
            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }
        self.read_token();
        let elements = self.ast.add_expr_list(&elements);
        Ok(self.ast.push_expr(
            Expression::Literal(Literal::Array(elements)),
            self.span_from(start),
        ))
    }

    fn parse_call_expression(&mut self, expression: ExprId) -> Result<ExprId> {
        let elements = self.parse_expression_list(&Token::RightParentheses)?;
        let elements = self.ast.add_expr_list(&elements);
        Ok(self.ast.push_expr(
            Expression::Call(expression, elements),
            self.span_covering(expression),
        ))
    }

    fn parse_index_expression(&mut self, expression: ExprId) -> Result<ExprId> {
        self.read_token();
        let index_expression = self.parse_expression(Precedence::Lowest)?;
        self.read_token();
        Ok(self.ast.push_expr(
            Expression::Index(expression, index_expression),
            self.span_covering(expression),
        ))
    }

    // A field name, wherever one is read: declaring a struct or an enum
    // variant, writing a literal of either, and reaching a field with `.`.
    //
    // A keyword is taken as the name it is written as. Nothing but a field name
    // can appear at any of these positions, so there is nothing for `type` or
    // `match` to be confused with, and a C header calling a member `type` needs
    // no renaming to be bound.
    fn read_field_name(&mut self, context: &str) -> Result<String> {
        let token = self.read_token();
        if let Token::Identifier(name) = token {
            return Ok(name.to_string());
        }
        match crate::lexer::keyword_spelling(token) {
            Some(word) => Ok(word.to_string()),
            None => {
                let written = token.to_string();
                Err(self.at_consumed(format!(
                    "Expected {context}, found '{written}'"
                )))
            }
        }
    }

    fn parse_field_access(&mut self, expression: ExprId) -> Result<ExprId> {
        self.read_token();
        let field_name = self.read_field_name("a field name after '.'")?;
        let field_name = self.ast.intern(&field_name);
        Ok(self.ast.push_expr(
            Expression::FieldAccess(expression, field_name),
            self.span_covering(expression),
        ))
    }

    fn parse_dereference(&mut self, expression: ExprId) -> Result<ExprId> {
        self.read_token();
        Ok(self.ast.push_expr(
            Expression::Dereference(expression),
            self.span_covering(expression),
        ))
    }

    // The name of a generic instance written in expression position, read from
    // the `<` this is called on. It is the same spelling the type parser
    // produces, since the two have to name one type.
    fn parse_generic_instance_name(&mut self, base: &str) -> Result<String> {
        self.read_token();
        let mut arguments = Vec::new();
        while !self.is_type_arg_close() {
            if let Token::Integer(value) = self.peek_nth(0) {
                let value = *value as usize;
                self.read_token();
                arguments.push(Type::ConstUsize(value));
            } else {
                arguments.push(self.parse_type()?);
            }
            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }
        self.consume_type_arg_close()?;
        let rendered: Vec<String> = arguments
            .iter()
            .map(|argument| argument.to_string())
            .collect();
        Ok(format!("{base}<{}>", rendered.join(", ")))
    }

    fn parse_struct_init(&mut self, struct_name: Symbol) -> Result<ExprId> {
        let start = self.mark();
        self.read_token();
        let mut fields = Vec::new();
        while self.peek_nth(0) != &Token::RightBrace {
            let field_name =
                self.read_field_name("a field name in a struct literal")?;
            if !matches!(self.read_token(), Token::Assign) {
                bail!("Expected '=' after field name in struct init");
            }
            let value = self.parse_expression(Precedence::Lowest)?;
            let name = self.ast.intern(&field_name);
            fields.push(NamedExpr { name, value });
            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }
        self.read_token();
        let fields = self.ast.add_named_exprs(&fields);
        Ok(self.ast.push_expr(
            Expression::StructInit(struct_name, fields),
            self.span_from(start),
        ))
    }

    fn parse_expression_list(
        &mut self,
        end_token: &Token,
    ) -> Result<Vec<ExprId>> {
        self.read_token();
        let mut elements = Vec::new();
        while self.peek_nth(0) != end_token {
            let held = self.parse_expression(Precedence::Lowest)?;
            // `f(g(T) for T in list, n)`: one argument per element of a
            // compile-time list, written once with the element's name standing
            // for it. This is how a call gets an arity the list decides. Only
            // an argument can be written this way, since it is the argument
            // count that is being produced.
            if matches!(self.peek_nth(0), Token::For) {
                self.read_token();
                let Token::Identifier(variable) = self.read_token() else {
                    bail!(
                        "`for` in an argument list names the element, as in `f(g(T) for T in list)`"
                    );
                };
                let variable = variable.to_string();
                if !matches!(self.read_token(), Token::In) {
                    bail!(
                        "`for` in an argument list is written `for {variable} in <list>`"
                    );
                }
                let Token::Identifier(list) = self.read_token() else {
                    bail!(
                        "`for {variable} in` names the compile-time list to walk"
                    );
                };
                let list = list.to_string();
                let variable = self.ast.intern(&variable);
                let list = self.ast.intern(&list);
                let span = self.span_covering(held);
                elements.push(self.ast.push_expr(
                    Expression::PackMap(held, variable, list),
                    span,
                ));
            } else {
                elements.push(held);
            }

            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }
        self.read_token();
        Ok(elements)
    }

    fn parse_grouped_expressions(&mut self) -> Result<ExprId> {
        let start = self.mark();
        self.read_token();

        if matches!(self.peek_nth(0), Token::RightParentheses) {
            self.read_token();
            if matches!(self.peek_nth(0), Token::LeftBrace | Token::Arrow) {
                let return_sig = self.parse_return_signature()?;
                let block = self.parse_block()?;
                let signature = self.ast.push_signature(return_sig);
                return Ok(self.ast.push_expr(
                    Expression::Proc(Range32::EMPTY, signature, block),
                    self.span_from(start),
                ));
            }
            return Ok(self.ast.push_expr(
                Expression::Tuple(Range32::EMPTY),
                self.span_from(start),
            ));
        }

        let looks_like_params =
            self.looks_like_function_params() && !self.no_struct_literal;

        if looks_like_params {
            let parameters = self.parse_function_parameters_inner()?;

            if matches!(
                self.peek_nth(0),
                Token::LeftBrace | Token::Arrow | Token::Uses
            ) {
                let return_sig = self.parse_return_signature()?;
                let block = self.parse_block()?;
                let has_type_annotations =
                    parameters.iter().any(|p| p.type_annotation.is_some())
                        || signature_is_typed(&return_sig);
                let parameters = self.ast.add_parameters(parameters);
                let signature = self.ast.push_signature(return_sig);
                let node = if has_type_annotations {
                    Expression::Proc(parameters, signature, block)
                } else {
                    Expression::Function(parameters, signature, block)
                };
                return Ok(self.ast.push_expr(node, self.span_from(start)));
            }

            let span = self.span_from(start);
            let expressions: Vec<ExprId> = parameters
                .into_iter()
                .map(|p| {
                    self.ast.push_expr(Expression::Identifier(p.name), span)
                })
                .collect();

            if expressions.len() == 1 {
                return Ok(expressions[0]);
            }
            let elements = self.ast.add_expr_list(&expressions);
            return Ok(self.ast.push_expr(Expression::Tuple(elements), span));
        }

        let first_expression = self.parse_expression(Precedence::Lowest)?;
        if matches!(self.peek_nth(0), Token::Comma) {
            self.read_token();
            let mut elements = vec![first_expression];
            while self.peek_nth(0) != &Token::RightParentheses {
                elements.push(self.parse_expression(Precedence::Lowest)?);
                if matches!(self.peek_nth(0), Token::Comma) {
                    self.read_token();
                }
            }
            self.read_token();
            let elements = self.ast.add_expr_list(&elements);
            Ok(self
                .ast
                .push_expr(Expression::Tuple(elements), self.span_from(start)))
        } else {
            if matches!(self.peek_nth(0), Token::RightParentheses) {
                self.read_token();
            }
            Ok(first_expression)
        }
    }

    fn looks_like_function_params(&self) -> bool {
        let mut depth = 0;
        let mut index = 0;
        let max_lookahead = 1000;
        let mut saw_identifier = false;
        let mut has_non_param_content = false;
        while index < max_lookahead {
            match self.peek_nth(index) {
                // The boolean literals cannot be parameter names, so they say
                // this is an expression the way an integer does.
                Token::Identifier(word)
                    if word == "true" || word == "false" =>
                {
                    if depth == 0 {
                        has_non_param_content = true;
                        saw_identifier = false;
                    }
                }
                Token::Identifier(_) | Token::Mut | Token::Underscore => {
                    if depth == 0 {
                        saw_identifier = true;
                    }
                }
                Token::LeftParentheses
                | Token::LeftBracket
                | Token::LeftBrace => depth += 1,
                Token::RightParentheses => {
                    if depth == 0 {
                        if has_non_param_content {
                            return false;
                        }
                        let next = self.peek_nth(index + 1);
                        return matches!(next, Token::LeftBrace | Token::Arrow);
                    }
                    depth -= 1;
                }
                Token::RightBracket | Token::RightBrace => {
                    if depth == 0 {
                        return false;
                    }
                    depth -= 1;
                }
                Token::Colon => {
                    if depth == 0 && saw_identifier {
                        return true;
                    }
                }
                Token::Comma => {
                    if depth == 0 {
                        saw_identifier = false;
                    }
                }
                Token::Integer(_)
                | Token::Float(_)
                | Token::StringLiteral(_) => {
                    if depth == 0 {
                        has_non_param_content = true;
                        saw_identifier = false;
                    }
                }
                Token::Plus
                | Token::Minus
                | Token::Asterisk
                | Token::Slash
                | Token::Equal
                | Token::NotEqual
                | Token::LessThan
                | Token::GreaterThan
                | Token::And
                | Token::Or
                | Token::Percent
                | Token::Dot
                | Token::DotDot
                | Token::DotDotEqual => {
                    if depth == 0 {
                        return false;
                    }
                }
                Token::EndOfFile | Token::Semicolon => return false,
                _ => {}
            }
            index += 1;
        }
        false
    }

    fn parse_function_parameters_inner(&mut self) -> Result<Vec<Parameter>> {
        let mut parameters = Vec::new();
        while self.peek_nth(0) != &Token::RightParentheses {
            if matches!(self.peek_nth(0), Token::EndOfFile) {
                bail!("Unexpected end of input in parameter list");
            }

            let mode = match self.peek_nth(0) {
                Token::Mut => {
                    self.read_token();
                    ParamMode::Write
                }
                Token::Move => {
                    self.read_token();
                    ParamMode::Move
                }
                // A Frost function that C calls back receives its struct the
                // way C passes one. `value` is what says so, and it is the same
                // word an `extern` uses for the other direction. Contextual, so
                // a parameter may still be called `value`.
                Token::Identifier(word)
                    if word == "value"
                        && matches!(self.peek_nth(1), Token::Identifier(_)) =>
                {
                    self.read_token();
                    ParamMode::Value
                }
                _ => ParamMode::Read,
            };

            if matches!(self.peek_nth(0), Token::Dollar) {
                parameters.push(self.parse_compile_time_parameter()?);
            } else if let Token::Identifier(name) = self.peek_nth(0) {
                let name = name.to_string();
                self.read_token();
                self.refuse_literal_name(&name)?;

                // `args: $...` is a compile-time list of values rather than
                // a value of some type, so it has no type annotation of its
                // own: its length and its types arrive with the call.
                let mut pack = false;
                let type_annotation =
                    if matches!(self.peek_nth(0), Token::Colon) {
                        self.read_token();
                        if matches!(self.peek_nth(0), Token::Dollar)
                            && matches!(self.peek_nth(1), Token::Ellipsis)
                        {
                            self.read_token();
                            self.read_token();
                            pack = true;
                            None
                        } else {
                            Some(self.parse_type()?)
                        }
                    } else {
                        None
                    };

                let name = self.ast.intern(&name);
                parameters.push(Parameter {
                    name,
                    type_annotation,
                    mutable: false,
                    mode,
                    compile_time_signature: None,
                    pack,
                });
            } else {
                let written = self.peek_nth(0).to_string();
                return Err(self.here(format!(
                    "Expected a parameter name in parameter list, found '{written}'"
                )));
            }

            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }

        if !matches!(self.peek_nth(0), Token::RightParentheses) {
            bail!("Expected a right parentheses in parameter list!");
        }
        self.read_token();

        Ok(parameters)
    }

    fn parse_match_expression(&mut self) -> Result<ExprId> {
        let start = self.mark();
        self.read_token();
        // A `{` after the scrutinee opens the arms, so neither a struct literal
        // nor a function literal is available here. Without that, the `(a, b)`
        // of a match on several values reads as a parameter list with a body,
        // since bare names in parentheses are what a function literal starts
        // with too.
        let held = self.no_struct_literal;
        self.no_struct_literal = true;
        let scrutinee = self.parse_expression(Precedence::Lowest)?;
        self.no_struct_literal = held;
        if !matches!(self.read_token(), Token::LeftBrace) {
            bail!("Expected '{{' after match expression");
        }
        let mut cases = Vec::new();
        while self.peek_nth(0) != &Token::RightBrace {
            if !matches!(self.read_token(), Token::Case) {
                bail!("Expected 'case' in match");
            }
            let pattern = self.parse_pattern()?;
            if !matches!(self.read_token(), Token::Colon) {
                bail!("Expected ':' after pattern in match case");
            }
            let body = if matches!(self.peek_nth(0), Token::LeftBrace) {
                self.parse_block()?
            } else {
                let arm_start = self.mark();
                let expr = self.parse_expression(Precedence::Lowest)?;
                let statement = self.ast.push_stmt(
                    Statement::Expression(expr),
                    self.span_from(arm_start),
                );
                self.ast.add_stmt_list(&[statement])
            };
            cases.push(SwitchCase { pattern, body });
        }
        self.read_token();
        let cases = self.ast.add_cases(&cases);
        Ok(self.ast.push_expr(
            Expression::Switch(scrutinee, cases),
            self.span_from(start),
        ))
    }

    fn parse_pattern_bindings(&mut self) -> Result<Range32> {
        self.read_token();
        let mut bindings = Vec::new();
        while self.peek_nth(0) != &Token::RightBrace {
            let field_name = match self.read_token() {
                Token::Identifier(name) => name.to_string(),
                _ => bail!("Expected binding name in pattern"),
            };
            let symbol = self.ast.intern(&field_name);
            bindings.push(PatternBinding {
                field: symbol,
                binding: symbol,
            });
            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }
        self.read_token();
        Ok(self.ast.add_pattern_bindings(&bindings))
    }

    fn parse_pattern(&mut self) -> Result<PatternId> {
        let pattern = match self.peek_nth(0) {
            Token::Underscore => {
                self.read_token();
                Pattern::Wildcard
            }
            Token::Integer(value) => {
                let value = *value;
                self.read_token();
                Pattern::Literal(Literal::Integer(value))
            }
            Token::Float(value) => {
                let value = *value;
                self.read_token();
                Pattern::Literal(Literal::Float(value))
            }
            Token::Float32(value) => {
                let value = *value;
                self.read_token();
                Pattern::Literal(Literal::Float32(value))
            }
            Token::StringLiteral(s) => {
                let s = s.clone();
                self.read_token();
                Pattern::Literal(Literal::String(s))
            }
            Token::Identifier(word) if word == "true" => {
                self.read_token();
                Pattern::Literal(Literal::Boolean(true))
            }
            Token::Identifier(word) if word == "false" => {
                self.read_token();
                Pattern::Literal(Literal::Boolean(false))
            }
            Token::Dot => {
                self.read_token();
                let variant_name = match self.read_token() {
                    Token::Identifier(name) => name.to_string(),
                    _ => bail!("Expected variant name after '.'"),
                };
                let bindings = if matches!(self.peek_nth(0), Token::LeftBrace) {
                    self.parse_pattern_bindings()?
                } else {
                    Range32::EMPTY
                };
                let variant_name = self.ast.intern(&variant_name);
                Pattern::EnumVariant {
                    enum_name: None,
                    variant_name,
                    bindings,
                }
            }
            Token::LeftParentheses => {
                self.read_token();
                let mut patterns = Vec::new();
                while self.peek_nth(0) != &Token::RightParentheses {
                    patterns.push(self.parse_pattern()?);
                    if matches!(self.peek_nth(0), Token::Comma) {
                        self.read_token();
                    }
                }
                self.read_token();
                Pattern::Tuple(self.ast.add_pattern_list(&patterns))
            }
            Token::Identifier(name) => {
                let name = name.clone();
                self.read_token();
                if matches!(self.peek_nth(0), Token::DoubleColon) {
                    self.read_token();
                    let variant_name = match self.read_token() {
                        Token::Identifier(v) => v.to_string(),
                        _ => bail!("Expected variant name after '::'"),
                    };
                    let bindings =
                        if matches!(self.peek_nth(0), Token::LeftBrace) {
                            self.parse_pattern_bindings()?
                        } else {
                            Range32::EMPTY
                        };
                    let enum_name = self.ast.intern(&name);
                    let variant_name = self.ast.intern(&variant_name);
                    Pattern::EnumVariant {
                        enum_name: Some(enum_name),
                        variant_name,
                        bindings,
                    }
                } else {
                    Pattern::Identifier(self.ast.intern(&name))
                }
            }
            token => {
                let written = token.to_string();
                return Err(self.here(format!(
                    "Unexpected token in pattern: '{written}'"
                )));
            }
        };
        Ok(self.ast.push_pattern(pattern))
    }

    fn parse_if_expression(&mut self) -> Result<ExprId> {
        let start = self.mark();
        self.read_token();

        if !matches!(self.peek_nth(0), Token::LeftParentheses) {
            bail!("Expected a left parentheses in if expression!");
        }
        self.read_token();

        let condition = self.parse_expression(Precedence::Lowest)?;

        if !matches!(self.peek_nth(0), Token::RightParentheses) {
            bail!("Expected a right parentheses in if expression!");
        }
        self.read_token();

        let consequence = self.parse_block()?;

        let mut alternative = None;
        if matches!(self.peek_nth(0), Token::Else) {
            self.read_token();
            if matches!(self.peek_nth(0), Token::If) {
                let arm_start = self.mark();
                let else_if = self.parse_if_expression()?;
                let statement = self.ast.push_stmt(
                    Statement::Expression(else_if),
                    self.span_from(arm_start),
                );
                alternative = Some(self.ast.add_stmt_list(&[statement]));
            } else {
                alternative = Some(self.parse_block()?);
            }
        }

        Ok(self.ast.push_expr(
            Expression::If(condition, consequence, alternative),
            self.span_from(start),
        ))
    }

    fn parse_function_literal(&mut self) -> Result<ExprId> {
        let start = self.mark();
        self.read_token();
        let parameters = self.parse_function_parameters()?;
        let return_sig = self.parse_return_signature()?;
        let block = self.parse_block()?;
        let has_type_annotations =
            parameters.iter().any(|p| p.type_annotation.is_some())
                || signature_is_typed(&return_sig);
        let parameters = self.ast.add_parameters(parameters);
        let signature = self.ast.push_signature(return_sig);
        let node = if has_type_annotations {
            Expression::Proc(parameters, signature, block)
        } else {
            Expression::Function(parameters, signature, block)
        };
        Ok(self.ast.push_expr(node, self.span_from(start)))
    }

    fn parse_return_signature(&mut self) -> Result<ReturnSignature> {
        let kind = self.parse_return_kind()?;
        let mut uses = Vec::new();
        while matches!(self.peek_nth(0), Token::Uses) {
            self.read_token();
            uses.push(self.parse_type()?);
            while matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
                uses.push(self.parse_type()?);
            }
        }
        // `where is_numeric(T)`. The body's `{` follows the bound, so no
        // struct literal is read here.
        let mut bound = None;
        if matches!(self.peek_nth(0), Token::Where) {
            self.read_token();
            let held = self.no_struct_literal;
            self.no_struct_literal = true;
            let expression = self.parse_expression(Precedence::Lowest);
            self.no_struct_literal = held;
            bound = Some(expression?);
        }
        Ok(ReturnSignature { kind, uses, bound })
    }

    fn parse_return_kind(&mut self) -> Result<ReturnKind> {
        if !matches!(self.peek_nth(0), Token::Arrow) {
            return Ok(ReturnKind::None);
        }
        self.read_token();

        if matches!(self.peek_nth(0), Token::LeftParentheses) {
            let kind = self.parse_multiple_returns()?;
            if matches!(self.peek_nth(0), Token::Bang) {
                bail!(
                    "a return type list and a failure set do not combine; a fallible function returns one value, so name a struct and return that"
                );
            }
            return Ok(kind);
        }

        let typ = self.parse_type()?;
        if matches!(self.peek_nth(0), Token::Bang) {
            self.read_token();
            let error = self.parse_type()?;
            return Ok(ReturnKind::Fallible(typ, error));
        }
        Ok(ReturnKind::Single(typ))
    }

    // `-> (i64, i64)`, and `-> (quotient: i64, remainder: i64)` when the values
    // are worth naming. A name says which value is which at the definition, and
    // it is the field name a `return { quotient = .. }` writes.
    fn parse_multiple_returns(&mut self) -> Result<ReturnKind> {
        if !matches!(self.peek_nth(0), Token::LeftParentheses) {
            bail!("Expected '(' for a multiple return");
        }
        self.read_token();

        let mut values = Vec::new();
        while !matches!(self.peek_nth(0), Token::RightParentheses) {
            if matches!(self.peek_nth(0), Token::EndOfFile) {
                bail!("Unexpected end of input in a return type list");
            }
            let mut name = None;
            if let Token::Identifier(written) = self.peek_nth(0)
                && matches!(self.peek_nth(1), Token::Colon)
            {
                name = Some(written.to_string());
                self.read_token();
                self.read_token();
            }
            let value_type = self.parse_type()?;
            let name = name.map(|held| self.ast.intern(&held));
            values.push(ReturnValue { name, value_type });
            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }
        self.read_token();

        if values.len() < 2 {
            bail!(
                "a return type list holds two or more values; write `-> T` for one"
            );
        }
        let named = values.iter().filter(|held| held.name.is_some()).count();
        if named != 0 && named != values.len() {
            bail!(
                "a return type list names all of its values or none of them, so that a `return` by name can write every field"
            );
        }
        for (index, held) in values.iter().enumerate() {
            if let Some(name) = held.name
                && values[..index]
                    .iter()
                    .any(|earlier| earlier.name == Some(name))
            {
                let name = self.ast.name(name);
                bail!("this return type list names '{name}' twice");
            }
        }
        let values = self.ast.add_return_values(values);
        Ok(ReturnKind::Multiple(values))
    }

    fn parse_function_parameters(&mut self) -> Result<Vec<Parameter>> {
        if !matches!(self.peek_nth(0), Token::LeftParentheses) {
            bail!("Expected a left parentheses in parameter list!");
        }
        self.read_token();

        let mut parameters = Vec::new();
        while self.peek_nth(0) != &Token::RightParentheses {
            if matches!(self.peek_nth(0), Token::EndOfFile) {
                bail!("Unexpected end of input in parameter list");
            }

            let mode = match self.peek_nth(0) {
                Token::Mut => {
                    self.read_token();
                    ParamMode::Write
                }
                Token::Move => {
                    self.read_token();
                    ParamMode::Move
                }
                // A Frost function that C calls back receives its struct the
                // way C passes one. `value` is what says so, and it is the same
                // word an `extern` uses for the other direction. Contextual, so
                // a parameter may still be called `value`.
                Token::Identifier(word)
                    if word == "value"
                        && matches!(self.peek_nth(1), Token::Identifier(_)) =>
                {
                    self.read_token();
                    ParamMode::Value
                }
                _ => ParamMode::Read,
            };

            if matches!(self.peek_nth(0), Token::Dollar) {
                parameters.push(self.parse_compile_time_parameter()?);
            } else if let Token::Identifier(name) = self.peek_nth(0) {
                let name = name.to_string();
                self.read_token();
                self.refuse_literal_name(&name)?;

                // `args: $...` is a compile-time list of values rather than
                // a value of some type, so it has no type annotation of its
                // own: its length and its types arrive with the call.
                let mut pack = false;
                let type_annotation =
                    if matches!(self.peek_nth(0), Token::Colon) {
                        self.read_token();
                        if matches!(self.peek_nth(0), Token::Dollar)
                            && matches!(self.peek_nth(1), Token::Ellipsis)
                        {
                            self.read_token();
                            self.read_token();
                            pack = true;
                            None
                        } else {
                            Some(self.parse_type()?)
                        }
                    } else {
                        None
                    };

                let name = self.ast.intern(&name);
                parameters.push(Parameter {
                    name,
                    type_annotation,
                    mutable: false,
                    mode,
                    compile_time_signature: None,
                    pack,
                });
            } else {
                let written = self.peek_nth(0).to_string();
                return Err(self.here(format!(
                    "Expected a parameter name in parameter list, found '{written}'"
                )));
            }

            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }

        if !matches!(self.peek_nth(0), Token::RightParentheses) {
            bail!("Expected a right parentheses in parameter list!");
        }
        self.read_token();

        Ok(parameters)
    }

    // `$T: Type` names a type the caller supplies. `$f: fn(T, T) -> bool` names
    // a function the caller supplies, and says what its signature has to be, so
    // a mismatch is reported against the parameter list the caller can read
    // rather than against a line inside the specialized body.
    // The `($T: Type, $N: usize)` list a generic declaration carries. Shared by
    // struct and enum so the two cannot drift.
    //
    // `$T: Type` is a type parameter. `$N: usize` is a value parameter, resolved
    // to a concrete integer at instantiation. Both are recorded by name here,
    // and the argument kind decides which is which.
    fn parse_generic_params(&mut self) -> Result<Vec<String>> {
        let mut type_params = Vec::new();
        if !matches!(self.peek_nth(0), Token::LeftParentheses) {
            return Ok(type_params);
        }
        self.read_token();
        while self.peek_nth(0) != &Token::RightParentheses {
            if !matches!(self.peek_nth(0), Token::Dollar) {
                bail!("Expected '$' before type parameter name");
            }
            self.read_token();
            let param_name = match self.read_token() {
                Token::Identifier(name) => name.to_string(),
                _ => bail!("Expected type parameter name after '$'"),
            };
            if !matches!(self.read_token(), Token::Colon) {
                bail!("Expected ':' after type parameter name");
            }
            match self.peek_nth(0) {
                Token::Type => {
                    self.read_token();
                }
                Token::Identifier(word) if word == "Type" => {
                    self.read_token();
                }
                _ => {
                    self.parse_type()?;
                }
            }
            type_params.push(param_name);
            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }
        self.read_token();
        Ok(type_params)
    }

    fn parse_compile_time_parameter(&mut self) -> Result<Parameter> {
        self.read_token();
        let name = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            _ => bail!("Expected type parameter name after '$'"),
        };
        self.refuse_literal_name(&name)?;
        if !matches!(self.read_token(), Token::Colon) {
            bail!("Expected ':' after type parameter name");
        }
        let compile_time_signature = match self.peek_nth(0) {
            Token::Type => {
                self.read_token();
                None
            }
            Token::Identifier(word) if word == "Type" => {
                self.read_token();
                None
            }
            Token::Function => Some(self.parse_type()?),
            // `$N: usize` is a value parameter, the same kind a generic struct
            // already takes. The argument is an integer written `$4`, and the
            // name stands for that integer in the body as well as in a type, so
            // a function over `[N]T` can be written once rather than once per
            // capacity.
            Token::Identifier(word) if word == "usize" => {
                self.read_token();
                None
            }
            // `$ops: Ordering<T>` is a capability bundle: a struct whose fields
            // are functions, named at the call by a constant. The declared type
            // is what the argument has to be, the same way a signature is for a
            // function parameter.
            Token::Identifier(_) => Some(self.parse_type()?),
            _ => bail!(
                "Expected 'Type', 'usize', a struct type or a function signature after ':' in the compile-time parameter '${name}'"
            ),
        };
        let symbol = self.ast.intern(&name);
        Ok(Parameter {
            name: symbol,
            type_annotation: Some(Type::TypeParam(name)),
            mutable: false,
            mode: ParamMode::Read,
            compile_time_signature,
            pack: false,
        })
    }

    fn parse_type(&mut self) -> Result<Type> {
        let base_type = match self.peek_nth(0) {
            Token::Caret => {
                self.read_token();
                Type::Ptr(Box::new(self.parse_type()?))
            }
            // `ref T` is the surface borrow type: the address a `^T` holds, but
            // one the checker has proven names a live value, so reaching through
            // it is not gated. It is the type a `ref` binding and a `mut`
            // parameter already carry internally, now sayable so an accessor can
            // return one. Lowered to the mutable reference, since the surface has
            // one borrow and it may be written through.
            Token::Ref => {
                self.read_token();
                Type::RefMut(Box::new(self.parse_type()?))
            }
            Token::Ampersand if self.internal_types => {
                self.read_token();
                if matches!(self.peek_nth(0), Token::Mut) {
                    self.read_token();
                    Type::RefMut(Box::new(self.parse_type()?))
                } else {
                    Type::Ref(Box::new(self.parse_type()?))
                }
            }
            Token::Ampersand => {
                bail!(
                    "a reference is not a surface type; use a parameter mode (`mut` to write, `move` to take, unmarked to read), or a raw pointer `^T`"
                );
            }
            Token::LeftBracket => {
                self.read_token();
                if matches!(self.peek_nth(0), Token::RightBracket) {
                    self.read_token();
                    Type::Slice(Box::new(self.parse_type()?))
                } else if let Token::Integer(size) = self.peek_nth(0) {
                    let size = *size as usize;
                    self.read_token();
                    if !matches!(self.read_token(), Token::RightBracket) {
                        bail!("Expected ']' after array size");
                    }
                    Type::Array(Box::new(self.parse_type()?), size)
                } else if let Token::Identifier(size_param) = self.peek_nth(0)
                    && matches!(self.peek_nth(1), Token::RightBracket)
                {
                    let size_param = size_param.to_string();
                    self.read_token();
                    self.read_token();
                    // A name here is a constant when one goes by that name, and
                    // a generic size parameter otherwise. A constant is folded
                    // into the type, so `[N]u8` and `[8]u8` are the same type
                    // and nothing downstream has to know the difference.
                    let size = self.integer_constants.get(&size_param).copied();
                    match size {
                        Some(size) => {
                            Type::Array(Box::new(self.parse_type()?), size)
                        }
                        None => Type::ArrayGeneric(
                            Box::new(self.parse_type()?),
                            size_param,
                        ),
                    }
                } else {
                    let element_type = self.parse_type()?;
                    if !matches!(self.read_token(), Token::Semicolon) {
                        bail!("Expected ';' after array element type");
                    }
                    let size = match self.read_token() {
                        Token::Integer(size) => *size as usize,
                        token => {
                            let written = token.to_string();
                            return Err(self.at_consumed(format!(
                                "Expected array size, found '{written}'"
                            )));
                        }
                    };
                    if !matches!(self.read_token(), Token::RightBracket) {
                        bail!("Expected ']' after array size");
                    }
                    Type::Array(Box::new(element_type), size)
                }
            }
            Token::Function => {
                self.read_token();
                if !matches!(self.peek_nth(0), Token::LeftParentheses) {
                    bail!("Expected '(' after 'fn' in type");
                }
                self.read_token();
                let mut param_types = Vec::new();
                while self.peek_nth(0) != &Token::RightParentheses {
                    // An unmarked parameter of a function type is the type as
                    // written, which is what the spec says a function type is
                    // and what every existing bound means. `mut` is the one
                    // marker that has to be sayable, because a `mut` parameter
                    // is a reference in the signature and the surface has no
                    // way to write a reference type. `move` is the type as
                    // written too, and is allowed so a bound can be read
                    // beside the declaration it describes.
                    // `value` says C passes that parameter as the struct
                    // itself, which is what a callback taking one is declared
                    // with. The callee holds a borrow of it either way, so this
                    // is the same reference type a definition's `value`
                    // parameter has and the two match without a rule of their
                    // own. It is contextual, so a type may still be named
                    // `value`: as a mode it is followed by the type it marks.
                    let mut write = false;
                    let mut by_value = false;
                    match self.peek_nth(0) {
                        Token::Mut => {
                            self.read_token();
                            write = true;
                        }
                        Token::Move => {
                            self.read_token();
                        }
                        Token::Identifier(word)
                            if word == "value"
                                && !matches!(
                                    self.peek_nth(1),
                                    Token::Comma | Token::RightParentheses
                                ) =>
                        {
                            self.read_token();
                            by_value = true;
                        }
                        _ => {}
                    }
                    let param_type = self.parse_type()?;
                    param_types.push(if write {
                        Type::RefMut(Box::new(param_type))
                    } else if by_value {
                        Type::Ref(Box::new(param_type))
                    } else {
                        param_type
                    });
                    if matches!(self.peek_nth(0), Token::Comma) {
                        self.read_token();
                    }
                }
                self.read_token();
                let return_type = if matches!(self.peek_nth(0), Token::Arrow) {
                    self.read_token();
                    self.parse_type()?
                } else {
                    Type::Void
                };
                Type::Proc(param_types, Box::new(return_type))
            }
            Token::Distinct => {
                self.read_token();
                // A `distinct T` written inline has no name of its own, so it
                // is the name of the declaration it belongs to that names it.
                // `parse_constant_or_struct_statement` fills that in.
                Type::Distinct(String::new(), Box::new(self.parse_type()?))
            }
            Token::Integer(value) => {
                let value = *value as usize;
                self.read_token();
                Type::ConstUsize(value)
            }
            Token::Identifier(name) => {
                let name = name.to_string();
                self.read_token();
                if let Some(primitive) =
                    primitive_type(&name, self.internal_types)
                {
                    return Ok(primitive);
                }
                match name.as_str() {
                    "Handle" => {
                        if !matches!(self.peek_nth(0), Token::LessThan) {
                            bail!("Expected '<' after 'Handle'");
                        }
                        self.read_token();
                        let inner_type = self.parse_type()?;
                        self.consume_type_arg_close()?;
                        Type::Handle(Box::new(inner_type))
                    }
                    _ if matches!(self.peek_nth(0), Token::LessThan) => {
                        self.read_token();
                        let mut arguments = Vec::new();
                        while !self.is_type_arg_close() {
                            if let Token::Integer(value) = self.peek_nth(0) {
                                let value = *value as usize;
                                self.read_token();
                                arguments.push(Type::ConstUsize(value));
                            } else {
                                arguments.push(self.parse_type()?);
                            }
                            if matches!(self.peek_nth(0), Token::Comma) {
                                self.read_token();
                            }
                        }
                        self.consume_type_arg_close()?;
                        let rendered: Vec<String> = arguments
                            .iter()
                            .map(|argument| argument.to_string())
                            .collect();
                        Type::Struct(format!(
                            "{}<{}>",
                            name,
                            rendered.join(", ")
                        ))
                    }
                    _ => Type::Struct(name),
                }
            }
            Token::Dollar => {
                self.read_token();
                let param_name = match self.read_token() {
                    Token::Identifier(name) => name.to_string(),
                    _ => {
                        bail!("Expected identifier after '$' in type parameter")
                    }
                };
                Type::TypeParam(param_name)
            }
            token => {
                let written = token.to_string();
                return Err(
                    self.here(format!("Expected type, found '{written}'"))
                );
            }
        };
        Ok(base_type)
    }

    fn parse_block(&mut self) -> Result<Range32> {
        if !matches!(self.peek_nth(0), Token::LeftBrace) {
            bail!("Expected a left brace in block!");
        }
        self.read_token();
        self.block_depth += 1;

        let mut statements = Vec::new();

        while self.peek_nth(0) != &Token::RightBrace
            && self.peek_nth(0) != &Token::EndOfFile
        {
            let position = self.current_position().unwrap_or_default();
            // Whether this statement opens with a minus after another one. A
            // `-` that opens a line negates what follows rather than
            // subtracting it from the line above, so a long expression broken
            // before a minus quietly loses its remaining terms while the same
            // break before a plus keeps them. Read before parsing, since
            // parsing moves the cursor.
            let opened_with_minus =
                !statements.is_empty() && self.peek_nth(0) == &Token::Minus;
            match self.parse_statement() {
                Ok(Some(statement)) => {
                    statements.push(statement);
                }
                Ok(None) => break,
                Err(error) => {
                    self.record_error(position, &error);
                    self.synchronize_in_block();
                }
            }
            // A right brace here means that statement was the block's value,
            // and a block whose value is `-1` is ordinary. Anything else means
            // the minus opened a statement whose answer nothing reads.
            if opened_with_minus
                && self.peek_nth(0) != &Token::RightBrace
                && self.peek_nth(0) != &Token::EndOfFile
            {
                self.record_error(
                    position,
                    &anyhow::anyhow!(
                        "this line opens with '-', so it negates what \
                         follows rather than continuing the line above, and \
                         nothing reads what it works out. A statement ends at \
                         the end of a line: write the whole expression on one \
                         line, or leave the '-' at the end of the line above \
                         where it says a subtraction is meant"
                    ),
                );
            }
        }

        self.block_depth -= 1;
        if !matches!(self.peek_nth(0), Token::RightBrace) {
            bail!("Expected a right brace in block!");
        }
        self.read_token();

        Ok(self.ast.add_stmt_list(&statements))
    }

    fn read_token(&mut self) -> &Token {
        self.consumed += 1;
        self.tokens.next().unwrap_or(&Token::EndOfFile)
    }

    fn peek_nth(&self, n: usize) -> &Token {
        self.tokens.clone().nth(n).unwrap_or(&Token::EndOfFile)
    }

    // `flags` is a word rather than a keyword, so a parameter, a local and a
    // field may all still be called `flags`, and one is: `window_create` takes
    // one. What tells the declaration apart is the shape after it, which no
    // expression has: the word, a scalar type, and then a brace. A
    // representation that is not an integer is let through here so that the
    // declaration itself is what says so.
    fn at_flags_declaration(&self, offset: usize) -> bool {
        matches!(self.peek_nth(offset), Token::Identifier(word) if word == "flags")
            && matches!(self.peek_nth(offset + 2), Token::LeftBrace)
            && matches!(
                self.peek_nth(offset + 1),
                Token::Identifier(name) if is_scalar_type_name(name)
            )
    }

    fn parse_unsafe_expression(&mut self) -> Result<ExprId> {
        let start = self.mark();
        self.read_token();
        // `unsafe fn(...)` is an unsafe function, not a block. Calling it is a
        // gated operation and its body is an implicit unsafe block.
        if matches!(self.peek_nth(0), Token::Function) {
            let function = self.parse_function_literal()?;
            return Ok(self.ast.push_expr(
                Expression::UnsafeFn(function),
                self.span_from(start),
            ));
        }
        if !matches!(self.peek_nth(0), Token::LeftBrace) {
            bail!("Expected '{{' or 'fn' after 'unsafe'");
        }
        let body = self.parse_block()?;
        Ok(self
            .ast
            .push_expr(Expression::Unsafe(body), self.span_from(start)))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Ast, ExprId, Expression, Literal, Module, Operator, ParamMode,
        Parameter, Parser, Pattern, ReturnKind, ReturnSignature, ReturnValue,
        Statement, TokenSpan,
    };
    use crate::ast_display::display_stmt;
    use crate::{lexer::Lexer, types::Type};
    use anyhow::{Result, bail};

    fn parse_module(input: &str) -> Result<Module> {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        match parser.parse() {
            Ok(module) => Ok(module),
            Err(error)
                if error
                    .to_string()
                    .contains("expected a declaration head") =>
            {
                let mut statement_parser = Parser::new(&tokens);
                statement_parser.block_depth = 1;
                statement_parser.parse()
            }
            Err(error) => Err(error),
        }
    }

    fn single_expression(input: &str) -> Result<(Module, ExprId)> {
        let module = parse_module(input)?;
        let Some(first) = module.roots.first().copied() else {
            bail!("Expected a statement");
        };
        let expression = match module.ast.stmt(first) {
            Statement::Expression(expression)
            | Statement::Return(expression) => *expression,
            _ => bail!("Expected an expression statement!"),
        };
        Ok((module, expression))
    }

    fn assert_integer(ast: &Ast, id: ExprId, expected: i64) -> Result<()> {
        match ast.expr(id) {
            Expression::Literal(Literal::Integer(value))
                if *value == expected =>
            {
                Ok(())
            }
            other => bail!("Expected the integer {expected}, got {other:?}"),
        }
    }

    fn assert_identifier(ast: &Ast, id: ExprId, expected: &str) -> Result<()> {
        match ast.expr(id) {
            Expression::Identifier(name) if ast.name(*name) == expected => {
                Ok(())
            }
            other => {
                bail!("Expected the identifier {expected}, got {other:?}")
            }
        }
    }

    fn assert_boolean(ast: &Ast, id: ExprId, expected: bool) -> Result<()> {
        match ast.expr(id) {
            Expression::Boolean(value) if *value == expected => Ok(()),
            other => bail!("Expected the boolean {expected}, got {other:?}"),
        }
    }

    fn infix_parts(
        ast: &Ast,
        id: ExprId,
        expected: Operator,
    ) -> Result<(ExprId, ExprId)> {
        match ast.expr(id) {
            Expression::Infix(left, operator, right)
                if *operator == expected =>
            {
                Ok((*left, *right))
            }
            other => {
                bail!("Expected an infix {expected} expression, got {other:?}")
            }
        }
    }

    fn assert_untyped_read_parameter(
        ast: &Ast,
        parameter: &Parameter,
        name: &str,
    ) {
        assert_eq!(ast.name(parameter.name), name);
        assert_eq!(parameter.type_annotation, None);
        assert!(!parameter.mutable);
        assert_eq!(parameter.mode, ParamMode::Read);
        assert_eq!(parameter.compile_time_signature, None);
        assert!(!parameter.pack);
    }

    enum Leaf {
        Integer(i64),
        Identifier(&'static str),
    }

    fn assert_leaf(ast: &Ast, id: ExprId, expected: &Leaf) -> Result<()> {
        match expected {
            Leaf::Integer(value) => assert_integer(ast, id, *value),
            Leaf::Identifier(name) => assert_identifier(ast, id, name),
        }
    }

    #[test]
    fn test_let_statements() -> Result<()> {
        let tests = [
            ("x := 5;", "x", Leaf::Integer(5)),
            ("y := 10;", "y", Leaf::Integer(10)),
            ("foobar := 838383;", "foobar", Leaf::Integer(838383)),
            ("foobar := y;", "foobar", Leaf::Identifier("y")),
        ];

        for (input, expected_identifier, expected_expression) in tests.iter() {
            let module = parse_module(input)?;
            assert_eq!(module.roots.len(), 1);

            match module.ast.stmt(module.roots[0]) {
                Statement::Let { name, value, .. } => {
                    assert_eq!(module.ast.name(*name), *expected_identifier);
                    assert_leaf(&module.ast, *value, expected_expression)?;
                }
                _ => bail!("Expected a let statement!"),
            }
        }

        Ok(())
    }

    #[test]
    fn test_return_statements() -> Result<()> {
        let tests = [
            ("return 5;", Leaf::Integer(5)),
            ("return 10;", Leaf::Integer(10)),
            ("return 993322;", Leaf::Integer(993322)),
            ("return y;", Leaf::Identifier("y")),
        ];

        for (input, expected_expression) in tests.iter() {
            let (module, expression) = single_expression(input)?;
            assert_leaf(&module.ast, expression, expected_expression)?;
        }

        Ok(())
    }

    #[test]
    fn ast() -> Result<()> {
        let output = "myVar := anotherVar;";
        let mut ast = Ast::default();
        let another = ast.intern("anotherVar");
        let value =
            ast.push_expr(Expression::Identifier(another), TokenSpan::NONE);
        let name = ast.intern("myVar");
        let statement = ast.push_stmt(
            Statement::Let {
                name,
                type_annotation: None,
                value,
                mutable: false,
            },
            TokenSpan::NONE,
        );
        assert_eq!(display_stmt(&ast, statement), output.to_string());
        Ok(())
    }

    #[test]
    fn identifier_expressions() -> Result<()> {
        let (module, expression) = single_expression("foobar;")?;
        assert_identifier(&module.ast, expression, "foobar")
    }

    #[test]
    fn integer_expressions() -> Result<()> {
        let (module, expression) = single_expression("5;")?;
        assert_integer(&module.ast, expression, 5)
    }

    #[test]
    fn boolean_expressions() -> Result<()> {
        let tests = [("true;", true), ("false;", false)];

        for (input, expected_value) in tests.iter() {
            let module = parse_module(input)?;
            assert_eq!(module.roots.len(), 1);

            if let Statement::Expression(expression) =
                module.ast.stmt(module.roots[0])
                && let Expression::Boolean(value) = module.ast.expr(*expression)
            {
                assert_eq!(value, expected_value)
            } else {
                bail!("Expected a boolean expression statement!");
            }
        }

        Ok(())
    }

    #[test]
    fn prefix_expressions() -> Result<()> {
        let tests = [("!5;", Operator::Not, 5), ("-15;", Operator::Negate, 15)];

        for (input, operator, value) in tests.iter() {
            let module = parse_module(input)?;
            assert_eq!(module.roots.len(), 1);

            match module.ast.stmt(module.roots[0]) {
                Statement::Expression(expression) => {
                    let Expression::Prefix(found, inner) =
                        module.ast.expr(*expression)
                    else {
                        bail!("Expected a prefix expression!");
                    };
                    assert_eq!(found, operator);
                    assert_integer(&module.ast, *inner, *value)?;
                }
                _ => bail!("Expected an expression statement!"),
            }
        }

        Ok(())
    }

    #[test]
    fn prefix_boolean_expressions() -> Result<()> {
        let tests = [
            ("!true;", Operator::Not, true),
            ("!false;", Operator::Not, false),
        ];

        for (input, operator, value) in tests.iter() {
            let module = parse_module(input)?;
            assert_eq!(module.roots.len(), 1);

            match module.ast.stmt(module.roots[0]) {
                Statement::Expression(expression) => {
                    let Expression::Prefix(found, inner) =
                        module.ast.expr(*expression)
                    else {
                        bail!("Expected a prefix expression!");
                    };
                    assert_eq!(found, operator);
                    assert_boolean(&module.ast, *inner, *value)?;
                }
                _ => bail!("Expected an expression statement!"),
            }
        }

        Ok(())
    }

    #[test]
    fn infix_expressions() -> Result<()> {
        let tests = [
            ("5 + 5;", 5, Operator::Add, 5),
            ("5 - 5;", 5, Operator::Subtract, 5),
            ("5 * 5;", 5, Operator::Multiply, 5),
            ("5 / 5;", 5, Operator::Divide, 5),
            ("5 % 3;", 5, Operator::Modulo, 3),
            ("5 > 5;", 5, Operator::GreaterThan, 5),
            ("5 < 5;", 5, Operator::LessThan, 5),
            ("5 >= 5;", 5, Operator::GreaterThanOrEqual, 5),
            ("5 <= 5;", 5, Operator::LessThanOrEqual, 5),
            ("5 == 5;", 5, Operator::Equal, 5),
            ("5 != 5;", 5, Operator::NotEqual, 5),
        ];

        for (input, left_value, operator, right_value) in tests.iter() {
            let module = parse_module(input)?;
            assert_eq!(module.roots.len(), 1);

            match module.ast.stmt(module.roots[0]) {
                Statement::Expression(expression) => {
                    let (left, right) =
                        infix_parts(&module.ast, *expression, *operator)?;
                    assert_integer(&module.ast, left, *left_value)?;
                    assert_integer(&module.ast, right, *right_value)?;
                }
                _ => bail!("Expected an expression statement!"),
            }
        }

        Ok(())
    }

    #[test]
    fn infix_boolean_expressions() -> Result<()> {
        let tests = [
            ("true == true", true, Operator::Equal, true),
            ("true != false", true, Operator::NotEqual, false),
            ("false == false", false, Operator::Equal, false),
        ];

        for (input, left_value, operator, right_value) in tests.iter() {
            let module = parse_module(input)?;
            assert_eq!(module.roots.len(), 1);

            match module.ast.stmt(module.roots[0]) {
                Statement::Expression(expression) => {
                    let (left, right) =
                        infix_parts(&module.ast, *expression, *operator)?;
                    assert_boolean(&module.ast, left, *left_value)?;
                    assert_boolean(&module.ast, right, *right_value)?;
                }
                _ => bail!("Expected an expression statement!"),
            }
        }

        Ok(())
    }

    #[test]
    fn operator_precedence() -> Result<()> {
        let tests = [
            ("-a * b", "((-a) * b)"),
            ("!-a", "(!(-a))"),
            ("a + b + c", "((a + b) + c)"),
            ("a + b - c", "((a + b) - c)"),
            ("a * b * c", "((a * b) * c)"),
            ("a * b / c", "((a * b) / c)"),
            ("a + b / c", "(a + (b / c))"),
            ("a + b * c + d / e - f", "(((a + (b * c)) + (d / e)) - f)"),
            ("3 + 4; -5 * 5", "(3 + 4)((-5) * 5)"),
            ("5 > 4 == 3 < 4", "((5 > 4) == (3 < 4))"),
            ("5 < 4 != 3 > 4", "((5 < 4) != (3 > 4))"),
            (
                "3 + 4 * 5 == 3 * 1 + 4 * 5",
                "((3 + (4 * 5)) == ((3 * 1) + (4 * 5)))",
            ),
            ("true", "true"),
            ("false", "false"),
            ("3 > 5 == false", "((3 > 5) == false)"),
            ("3 < 5 == true", "((3 < 5) == true)"),
            ("1 + (2 + 3) + 4", "((1 + (2 + 3)) + 4)"),
            ("(5 + 5) * 2", "((5 + 5) * 2)"),
            ("2 / (5 + 5)", "(2 / (5 + 5))"),
            ("-(5 + 5)", "(-(5 + 5))"),
            ("!(true == true)", "(!(true == true))"),
            ("a + add(b * c) + d", "((a + add((b * c))) + d)"),
            (
                "add(a, b, 1, 2 * 3, 4 + 5, add(6, 7 * 8))",
                "add(a, b, 1, (2 * 3), (4 + 5), add(6, (7 * 8)))",
            ),
            (
                "add(a + b + c * d / f + g)",
                "add((((a + b) + ((c * d) / f)) + g))",
            ),
            (
                "a * [1, 2, 3, 4][b * c] * d",
                "((a * ([1, 2, 3, 4][(b * c)])) * d)",
            ),
            (
                "add(a * b[2], b[1], 2 * [1, 2][1])",
                "add((a * (b[2])), (b[1]), (2 * ([1, 2][1])))",
            ),
        ];

        for (input, expected) in tests.iter() {
            let module = parse_module(input)?;
            let program_string = module
                .roots
                .iter()
                .map(|statement| display_stmt(&module.ast, *statement))
                .collect::<Vec<_>>()
                .join("");

            assert_eq!(program_string, expected.to_string());
        }

        Ok(())
    }

    #[test]
    fn if_expressions() -> Result<()> {
        let (module, expression) = single_expression("if (x < y) { x }")?;
        let ast = &module.ast;
        let Expression::If(condition, consequence, alternative) =
            ast.expr(expression)
        else {
            bail!("Expected an if expression!");
        };
        let (left, right) = infix_parts(ast, *condition, Operator::LessThan)?;
        assert_identifier(ast, left, "x")?;
        assert_identifier(ast, right, "y")?;
        let consequence = ast.stmts_in(*consequence);
        assert_eq!(consequence.len(), 1);
        let Statement::Expression(held) = ast.stmt(consequence[0]) else {
            bail!("Expected an expression statement!");
        };
        assert_identifier(ast, *held, "x")?;
        assert!(alternative.is_none());
        Ok(())
    }

    #[test]
    fn if_else_expressions() -> Result<()> {
        let (module, expression) =
            single_expression("if (x < y) { x } else { y }")?;
        let ast = &module.ast;
        let Expression::If(condition, consequence, alternative) =
            ast.expr(expression)
        else {
            bail!("Expected an if expression!");
        };
        let (left, right) = infix_parts(ast, *condition, Operator::LessThan)?;
        assert_identifier(ast, left, "x")?;
        assert_identifier(ast, right, "y")?;
        let consequence = ast.stmts_in(*consequence);
        assert_eq!(consequence.len(), 1);
        let Statement::Expression(held) = ast.stmt(consequence[0]) else {
            bail!("Expected an expression statement!");
        };
        assert_identifier(ast, *held, "x")?;
        let Some(alternative) = alternative else {
            bail!("Expected an else block!");
        };
        let alternative = ast.stmts_in(*alternative);
        assert_eq!(alternative.len(), 1);
        let Statement::Expression(held) = ast.stmt(alternative[0]) else {
            bail!("Expected an expression statement!");
        };
        assert_identifier(ast, *held, "y")?;
        Ok(())
    }

    #[test]
    fn function_expressions() -> Result<()> {
        let (module, expression) = single_expression("fn(x, y) { x + y; }")?;
        let ast = &module.ast;
        let Expression::Function(parameters, signature, body) =
            ast.expr(expression)
        else {
            bail!("Expected a function expression!");
        };
        let parameters = ast.params_in(*parameters);
        assert_eq!(parameters.len(), 2);
        assert_untyped_read_parameter(ast, &parameters[0], "x");
        assert_untyped_read_parameter(ast, &parameters[1], "y");
        assert_eq!(
            ast.signature(*signature),
            &ReturnSignature::plain(ReturnKind::None)
        );
        let body = ast.stmts_in(*body);
        assert_eq!(body.len(), 1);
        let Statement::Expression(held) = ast.stmt(body[0]) else {
            bail!("Expected an expression statement!");
        };
        let (left, right) = infix_parts(ast, *held, Operator::Add)?;
        assert_identifier(ast, left, "x")?;
        assert_identifier(ast, right, "y")?;
        Ok(())
    }

    #[test]
    fn function_parameter_parsing() -> Result<()> {
        let tests: Vec<(&str, Vec<&str>)> = vec![
            ("fn() {};", vec![]),
            ("fn(x) {};", vec!["x"]),
            ("fn(x, y, z) {};", vec!["x", "y", "z"]),
        ];

        for (input, expected_parameters) in tests.iter() {
            let module = parse_module(input)?;
            assert_eq!(module.roots.len(), 1);
            let ast = &module.ast;

            match ast.stmt(module.roots[0]) {
                Statement::Expression(expression) => {
                    let Expression::Function(parameters, signature, body) =
                        ast.expr(*expression)
                    else {
                        bail!("Expected a function expression!");
                    };
                    let parameters = ast.params_in(*parameters);
                    assert_eq!(parameters.len(), expected_parameters.len());
                    for (parameter, expected) in
                        parameters.iter().zip(expected_parameters.iter())
                    {
                        assert_untyped_read_parameter(ast, parameter, expected);
                    }
                    assert_eq!(
                        ast.signature(*signature),
                        &ReturnSignature::plain(ReturnKind::None)
                    );
                    assert!(ast.stmts_in(*body).is_empty());
                }
                _ => bail!("Expected an expression statement!"),
            }
        }

        Ok(())
    }

    #[test]
    fn call_expressions() -> Result<()> {
        let (module, expression) = single_expression("add(1, 2 * 3, 4 + 5);")?;
        let ast = &module.ast;
        let Expression::Call(callee, arguments) = ast.expr(expression) else {
            bail!("Expected a call expression!");
        };
        assert_identifier(ast, *callee, "add")?;
        let arguments = ast.exprs_in(*arguments);
        assert_eq!(arguments.len(), 3);
        assert_integer(ast, arguments[0], 1)?;
        let (left, right) = infix_parts(ast, arguments[1], Operator::Multiply)?;
        assert_integer(ast, left, 2)?;
        assert_integer(ast, right, 3)?;
        let (left, right) = infix_parts(ast, arguments[2], Operator::Add)?;
        assert_integer(ast, left, 4)?;
        assert_integer(ast, right, 5)?;
        Ok(())
    }

    #[test]
    fn string_literal_expression() -> Result<()> {
        let (module, expression) = single_expression("\"hello world\"")?;
        let Expression::Literal(Literal::String(value)) =
            module.ast.expr(expression)
        else {
            bail!("Expected a string literal!");
        };
        assert_eq!(value, "hello world");
        Ok(())
    }

    #[test]
    fn array_literal_expression() -> Result<()> {
        let (module, expression) = single_expression("[1, 2 * 2, 3 + 3]")?;
        let ast = &module.ast;
        let Expression::Literal(Literal::Array(elements)) =
            ast.expr(expression)
        else {
            bail!("Expected an array literal!");
        };
        let elements = ast.exprs_in(*elements);
        assert_eq!(elements.len(), 3);
        assert_integer(ast, elements[0], 1)?;
        let (left, right) = infix_parts(ast, elements[1], Operator::Multiply)?;
        assert_integer(ast, left, 2)?;
        assert_integer(ast, right, 2)?;
        let (left, right) = infix_parts(ast, elements[2], Operator::Add)?;
        assert_integer(ast, left, 3)?;
        assert_integer(ast, right, 3)?;
        Ok(())
    }

    #[test]
    fn index_expression() -> Result<()> {
        let (module, expression) = single_expression("myArray[1 + 1]")?;
        let ast = &module.ast;
        let Expression::Index(base, index) = ast.expr(expression) else {
            bail!("Expected an index expression!");
        };
        assert_identifier(ast, *base, "myArray")?;
        let (left, right) = infix_parts(ast, *index, Operator::Add)?;
        assert_integer(ast, left, 1)?;
        assert_integer(ast, right, 1)?;
        Ok(())
    }

    #[test]
    fn let_with_type_annotation() -> Result<()> {
        let module = parse_module("x : i64 = 5;")?;
        assert_eq!(module.roots.len(), 1);
        match module.ast.stmt(module.roots[0]) {
            Statement::Let {
                name,
                type_annotation,
                value,
                ..
            } => {
                assert_eq!(module.ast.name(*name), "x");
                assert_eq!(type_annotation, &Some(Type::I64));
                assert_integer(&module.ast, *value, 5)?;
            }
            _ => bail!("Expected let statement"),
        }
        Ok(())
    }

    #[test]
    fn function_with_typed_parameters() -> Result<()> {
        let module = parse_module("fn(a: i64, b: i32) -> bool { true }")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Expression(expression) = ast.stmt(module.roots[0])
            && let Expression::Proc(params, return_type, body) =
                ast.expr(*expression)
        {
            let params = ast.params_in(*params);
            assert_eq!(params.len(), 2);
            assert_eq!(ast.name(params[0].name), "a");
            assert_eq!(params[0].type_annotation, Some(Type::I64));
            assert_eq!(ast.name(params[1].name), "b");
            assert_eq!(params[1].type_annotation, Some(Type::I32));
            assert_eq!(
                ast.signature(*return_type),
                &ReturnSignature::plain(ReturnKind::Single(Type::Bool))
            );
            assert_eq!(ast.stmts_in(*body).len(), 1);
        } else {
            bail!("Expected typed function expression");
        }
        Ok(())
    }

    #[test]
    fn typed_function_literal() -> Result<()> {
        let module = parse_module("fn(x: i64) -> i64 { x }")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Expression(expression) = ast.stmt(module.roots[0])
            && let Expression::Proc(params, return_type, body) =
                ast.expr(*expression)
        {
            let params = ast.params_in(*params);
            assert_eq!(params.len(), 1);
            assert_eq!(ast.name(params[0].name), "x");
            assert_eq!(params[0].type_annotation, Some(Type::I64));
            assert_eq!(
                ast.signature(*return_type),
                &ReturnSignature::plain(ReturnKind::Single(Type::I64))
            );
            assert_eq!(ast.stmts_in(*body).len(), 1);
        } else {
            bail!("Expected typed function expression");
        }
        Ok(())
    }

    #[test]
    fn struct_declaration() -> Result<()> {
        let module = parse_module("Vec3 :: struct { x: f32, y: f32, z: f32 }")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Struct(name, type_params, fields) =
            ast.stmt(module.roots[0])
        {
            assert_eq!(ast.name(*name), "Vec3");
            assert!(type_params.is_empty());
            let fields = ast.fields_in(*fields);
            assert_eq!(fields.len(), 3);
            assert_eq!(ast.name(fields[0].name), "x");
            assert_eq!(fields[0].field_type, Type::F32);
            assert_eq!(ast.name(fields[1].name), "y");
            assert_eq!(fields[1].field_type, Type::F32);
            assert_eq!(ast.name(fields[2].name), "z");
            assert_eq!(fields[2].field_type, Type::F32);
        } else {
            bail!("Expected struct declaration");
        }
        Ok(())
    }

    #[test]
    fn constant_declaration() -> Result<()> {
        let module = parse_module("PI :: 3;")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Constant(name, value) =
            module.ast.stmt(module.roots[0])
        {
            assert_eq!(module.ast.name(*name), "PI");
            assert_integer(&module.ast, *value, 3)?;
        } else {
            bail!("Expected constant declaration");
        }
        Ok(())
    }

    #[test]
    fn defer_statement() -> Result<()> {
        let module = parse_module("defer return 5;")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Defer(inner) = module.ast.stmt(module.roots[0]) {
            if let Statement::Return(value) = module.ast.stmt(*inner) {
                assert_integer(&module.ast, *value, 5)?;
            } else {
                bail!("Expected return inside defer");
            }
        } else {
            bail!("Expected defer statement");
        }
        Ok(())
    }

    #[test]
    fn for_statement() -> Result<()> {
        let module = parse_module("for i in 0..10 { i }")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::For(iterator, _, range, body) =
            ast.stmt(module.roots[0])
        {
            assert_eq!(ast.name(*iterator), "i");
            if let Expression::Range(start, end, inclusive) = ast.expr(*range) {
                assert_integer(ast, *start, 0)?;
                assert_integer(ast, *end, 10)?;
                assert!(!inclusive);
            } else {
                bail!("Expected range expression");
            }
            assert_eq!(ast.stmts_in(*body).len(), 1);
        } else {
            bail!("Expected for statement");
        }
        Ok(())
    }

    #[test]
    fn field_access() -> Result<()> {
        let module = parse_module("point.x")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Expression(expression) = ast.stmt(module.roots[0])
            && let Expression::FieldAccess(base, field) = ast.expr(*expression)
        {
            assert_identifier(ast, *base, "point")?;
            assert_eq!(ast.name(*field), "x");
        } else {
            bail!("Expected field access expression");
        }
        Ok(())
    }

    #[test]
    fn parse_errors_carry_a_source_location() -> Result<()> {
        let input = "main :: fn() -> i64 {\n    x := 5\n    y := )\n    0\n}";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        let error = parser.parse().unwrap_err().to_string();
        assert!(
            error.contains("line 3"),
            "expected a line 3 location, got: {error}"
        );
        assert!(
            error.contains("column"),
            "expected a column in the error, got: {error}"
        );
        Ok(())
    }

    #[test]
    fn scoped_identifier() -> Result<()> {
        // `Name :: Other` is two things and the depth is what settles them.
        // Inside a body it is variant access, an expression, which is where
        // every variant anyone writes appears. At the top level it is a
        // constant: a variant on its own is a statement with no effect there,
        // and the file that has no `main` returns its last expression as an
        // exit code, which a variant is not one of either.
        let input =
            "shown :: fn() -> i64 {\n    held := Color::Green\n    0\n}";
        let module = parse_module(input)?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        let Statement::Constant(_, value) = ast.stmt(module.roots[0]) else {
            bail!("Expected a function, got {:?}", ast.stmt(module.roots[0]));
        };
        let (Expression::Function(_, _, body) | Expression::Proc(_, _, body)) =
            ast.expr(*value)
        else {
            bail!("Expected a function, got {:?}", ast.expr(*value));
        };
        let body = ast.stmts_in(*body);
        if let Statement::Let { value, .. } = ast.stmt(body[0])
            && let Expression::EnumVariantInit(enum_name, variant_name, fields) =
                ast.expr(*value)
        {
            assert_eq!(ast.name(*enum_name), "Color");
            assert_eq!(ast.name(*variant_name), "Green");
            assert!(fields.is_empty());
        } else {
            bail!("Expected EnumVariantInit, got {:?}", ast.stmt(body[0]));
        }
        Ok(())
    }

    #[test]
    fn a_name_standing_for_a_name_is_a_constant() -> Result<()> {
        // The same two tokens at the top level. This used to parse as the
        // variant access above, so the constant it declares did not exist and
        // every use of it was an unknown variable.
        let module = parse_module("DEPTH :: TEXTURE_DEPTH24")?;
        assert_eq!(module.roots.len(), 1);
        let Statement::Constant(name, value) = module.ast.stmt(module.roots[0])
        else {
            bail!(
                "Expected a constant, got {:?}",
                module.ast.stmt(module.roots[0])
            );
        };
        assert_eq!(module.ast.name(*name), "DEPTH");
        assert_identifier(&module.ast, *value, "TEXTURE_DEPTH24")?;
        Ok(())
    }

    #[test]
    fn borrow_expression_is_rejected() {
        let mut lexer = Lexer::new("f :: fn() -> i64 { x := 1; &x }");
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        assert!(
            parser.parse().is_err(),
            "a `&` borrow is no longer surface syntax"
        );
    }

    #[test]
    fn reference_type_is_rejected() {
        let mut lexer = Lexer::new("f :: fn(x: &i64) -> i64 { 0 }");
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        assert!(
            parser.parse().is_err(),
            "a `&` reference is no longer a surface type"
        );
    }

    #[test]
    fn dereference_expression() -> Result<()> {
        let module = parse_module("p^")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Expression(expression) =
            module.ast.stmt(module.roots[0])
            && let Expression::Dereference(inner) = module.ast.expr(*expression)
        {
            assert_identifier(&module.ast, *inner, "p")?;
        } else {
            bail!("Expected dereference expression");
        }
        Ok(())
    }

    #[test]
    fn pointer_type_annotation() -> Result<()> {
        let module = parse_module("p : ^i64 = x;")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Let {
            name,
            type_annotation,
            ..
        } = module.ast.stmt(module.roots[0])
        {
            assert_eq!(module.ast.name(*name), "p");
            assert_eq!(type_annotation, &Some(Type::Ptr(Box::new(Type::I64))));
        } else {
            bail!("Expected let statement with pointer type");
        }
        Ok(())
    }

    #[test]
    fn array_type_annotation() -> Result<()> {
        let module = parse_module("arr : [10]i64 = x;")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Let {
            name,
            type_annotation,
            ..
        } = module.ast.stmt(module.roots[0])
        {
            assert_eq!(module.ast.name(*name), "arr");
            assert_eq!(
                type_annotation,
                &Some(Type::Array(Box::new(Type::I64), 10))
            );
        } else {
            bail!("Expected let statement with array type");
        }
        Ok(())
    }

    #[test]
    fn slice_type_annotation() -> Result<()> {
        let module = parse_module("slice : []f32 = x;")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Let {
            name,
            type_annotation,
            ..
        } = module.ast.stmt(module.roots[0])
        {
            assert_eq!(module.ast.name(*name), "slice");
            assert_eq!(
                type_annotation,
                &Some(Type::Slice(Box::new(Type::F32)))
            );
        } else {
            bail!("Expected let statement with slice type");
        }
        Ok(())
    }

    #[test]
    fn fn_type_annotation() -> Result<()> {
        let module = parse_module("callback : fn(i64, i64) -> i64 = x;")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Let {
            name,
            type_annotation,
            ..
        } = module.ast.stmt(module.roots[0])
        {
            assert_eq!(module.ast.name(*name), "callback");
            assert_eq!(
                type_annotation,
                &Some(Type::Proc(
                    vec![Type::I64, Type::I64],
                    Box::new(Type::I64)
                ))
            );
        } else {
            bail!("Expected let statement with fn type");
        }
        Ok(())
    }

    #[test]
    fn struct_init() -> Result<()> {
        let module = parse_module("Point { x = 1, y = 2 }")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Expression(expression) = ast.stmt(module.roots[0])
            && let Expression::StructInit(name, fields) = ast.expr(*expression)
        {
            assert_eq!(ast.name(*name), "Point");
            let fields = ast.named_in(*fields);
            assert_eq!(fields.len(), 2);
            assert_eq!(ast.name(fields[0].name), "x");
            assert_eq!(ast.name(fields[1].name), "y");
        } else {
            bail!("Expected struct init expression");
        }
        Ok(())
    }

    #[test]
    fn pointer_store() -> Result<()> {
        let module = parse_module("p^ = 42")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Assignment(place, value) = ast.stmt(module.roots[0]) {
            if let Expression::Dereference(pointer) = ast.expr(*place) {
                assert_identifier(ast, *pointer, "p")?;
            } else {
                bail!("Expected dereference on lhs");
            }
            assert_integer(ast, *value, 42)?;
        } else {
            bail!("Expected assignment statement");
        }
        Ok(())
    }

    #[test]
    fn field_assignment() -> Result<()> {
        let module = parse_module("p.x = 42")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Assignment(place, value) = ast.stmt(module.roots[0]) {
            if let Expression::FieldAccess(base, field) = ast.expr(*place) {
                assert_identifier(ast, *base, "p")?;
                assert_eq!(ast.name(*field), "x");
            } else {
                bail!("Expected field access on lhs");
            }
            assert_integer(ast, *value, 42)?;
        } else {
            bail!("Expected assignment statement");
        }
        Ok(())
    }

    // The type a builtin call carries, when the statement is one. `sizeof`,
    // `typename` and `type_id` all parse as a call to the name with the type
    // riding along as a `TypeValue` argument, so no pass has a node form to
    // enumerate for them.
    fn builtin_call_type(module: &Module, name: &str) -> Result<Type> {
        let Statement::Expression(expression) =
            module.ast.stmt(module.roots[0])
        else {
            bail!("Expected an expression statement");
        };
        let Expression::Call(callee, arguments) = module.ast.expr(*expression)
        else {
            bail!("Expected a call expression");
        };
        let Expression::Identifier(word) = module.ast.expr(*callee) else {
            bail!("Expected the builtin's name as the callee");
        };
        assert_eq!(module.ast.name(*word), name);
        let arguments = module.ast.exprs_in(*arguments);
        assert_eq!(arguments.len(), 1);
        let Expression::TypeValue(typ) = module.ast.expr(arguments[0]) else {
            bail!("Expected the type as a TypeValue argument");
        };
        Ok(typ.clone())
    }

    #[test]
    fn sizeof_expression() -> Result<()> {
        let module = parse_module("sizeof(i64)")?;
        assert_eq!(module.roots.len(), 1);
        assert_eq!(builtin_call_type(&module, "sizeof")?, Type::I64);
        Ok(())
    }

    #[test]
    fn sizeof_pointer_expression() -> Result<()> {
        let module = parse_module("sizeof(^i64)")?;
        assert_eq!(module.roots.len(), 1);
        assert_eq!(
            builtin_call_type(&module, "sizeof")?,
            Type::Ptr(Box::new(Type::I64))
        );
        Ok(())
    }

    #[test]
    fn colon_assign_declaration() -> Result<()> {
        let module = parse_module("x := 5")?;
        assert_eq!(module.roots.len(), 1);
        match module.ast.stmt(module.roots[0]) {
            Statement::Let {
                name,
                type_annotation,
                value,
                mutable,
            } => {
                assert_eq!(module.ast.name(*name), "x");
                assert_eq!(type_annotation, &None);
                assert_integer(&module.ast, *value, 5)?;
                assert!(!mutable);
            }
            other => bail!("Expected a let statement, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn typed_declaration() -> Result<()> {
        let module = parse_module("x : i64 = 42")?;
        assert_eq!(module.roots.len(), 1);
        match module.ast.stmt(module.roots[0]) {
            Statement::Let {
                name,
                type_annotation,
                value,
                mutable,
            } => {
                assert_eq!(module.ast.name(*name), "x");
                assert_eq!(type_annotation, &Some(Type::I64));
                assert_integer(&module.ast, *value, 42)?;
                assert!(!mutable);
            }
            other => bail!("Expected a let statement, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn function_declaration() -> Result<()> {
        let module = parse_module("add := fn(a, b) { a + b }")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Let {
            name,
            type_annotation: None,
            value,
            ..
        } = ast.stmt(module.roots[0])
            && let Expression::Function(params, _, _) = ast.expr(*value)
        {
            assert_eq!(ast.name(*name), "add");
            assert_eq!(ast.params_in(*params).len(), 2);
        } else {
            bail!("Expected let statement with function");
        }
        Ok(())
    }

    #[test]
    fn mutable_declaration() -> Result<()> {
        let module = parse_module("var x := 5")?;
        assert_eq!(module.roots.len(), 1);
        match module.ast.stmt(module.roots[0]) {
            Statement::Let {
                name,
                type_annotation,
                value,
                mutable,
            } => {
                assert_eq!(module.ast.name(*name), "x");
                assert_eq!(type_annotation, &None);
                assert_integer(&module.ast, *value, 5)?;
                assert!(*mutable);
            }
            other => bail!("Expected a let statement, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn mutable_typed_declaration() -> Result<()> {
        let module = parse_module("var x : i64 = 42")?;
        assert_eq!(module.roots.len(), 1);
        match module.ast.stmt(module.roots[0]) {
            Statement::Let {
                name,
                type_annotation,
                value,
                mutable,
            } => {
                assert_eq!(module.ast.name(*name), "x");
                assert_eq!(type_annotation, &Some(Type::I64));
                assert_integer(&module.ast, *value, 42)?;
                assert!(*mutable);
            }
            other => bail!("Expected a let statement, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn immutable_declaration_default() -> Result<()> {
        let module = parse_module("x := 5")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Let { mutable, .. } = module.ast.stmt(module.roots[0])
        {
            assert!(!mutable, "Declaration without mut should be immutable");
        } else {
            bail!("Expected let statement");
        }
        Ok(())
    }

    #[test]
    fn mutable_ast_display() -> Result<()> {
        let output = "var myVar := anotherVar;";
        let mut ast = Ast::default();
        let another = ast.intern("anotherVar");
        let value =
            ast.push_expr(Expression::Identifier(another), TokenSpan::NONE);
        let name = ast.intern("myVar");
        let statement = ast.push_stmt(
            Statement::Let {
                name,
                type_annotation: None,
                value,
                mutable: true,
            },
            TokenSpan::NONE,
        );
        assert_eq!(display_stmt(&ast, statement), output.to_string());
        Ok(())
    }

    #[test]
    fn mutable_typed_ast_display() -> Result<()> {
        let output = "var x : i64 = 5;";
        let mut ast = Ast::default();
        let value = ast.push_expr(
            Expression::Literal(Literal::Integer(5)),
            TokenSpan::NONE,
        );
        let name = ast.intern("x");
        let statement = ast.push_stmt(
            Statement::Let {
                name,
                type_annotation: Some(Type::I64),
                value,
                mutable: true,
            },
            TokenSpan::NONE,
        );
        assert_eq!(display_stmt(&ast, statement), output.to_string());
        Ok(())
    }

    #[test]
    fn shift_operators() -> Result<()> {
        let module = parse_module("x := 1 << 2")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Let { value, .. } = module.ast.stmt(module.roots[0]) {
            if let Expression::Infix(_, operator, _) = module.ast.expr(*value) {
                assert_eq!(*operator, Operator::ShiftLeft);
            } else {
                bail!("Expected Infix expression");
            }
        } else {
            bail!("Expected Let statement");
        }
        Ok(())
    }

    #[test]
    fn bitwise_or_operator() -> Result<()> {
        let module = parse_module("x := 1 | 2")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Let { value, .. } = module.ast.stmt(module.roots[0]) {
            if let Expression::Infix(_, operator, _) = module.ast.expr(*value) {
                assert_eq!(*operator, Operator::BitwiseOr);
            } else {
                bail!("Expected Infix expression");
            }
        } else {
            bail!("Expected Let statement");
        }
        Ok(())
    }

    #[test]
    fn handle_type_annotation() -> Result<()> {
        let module = parse_module("h : Handle<Entity> = x;")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Let {
            name,
            type_annotation,
            ..
        } = module.ast.stmt(module.roots[0])
        {
            assert_eq!(module.ast.name(*name), "h");
            assert_eq!(
                type_annotation,
                &Some(Type::Handle(Box::new(Type::Struct(
                    "Entity".to_string()
                ))))
            );
        } else {
            bail!("Expected let statement with Handle type");
        }
        Ok(())
    }

    #[test]
    fn extern_declaration() -> Result<()> {
        let module = parse_module("puts :: extern fn(s: ^i8) -> i32")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Extern {
            name,
            params,
            return_type,
            ..
        } = ast.stmt(module.roots[0])
        {
            assert_eq!(ast.name(*name), "puts");
            let params = ast.params_in(*params);
            assert_eq!(params.len(), 1);
            assert_eq!(ast.name(params[0].name), "s");
            assert_eq!(
                params[0].type_annotation,
                Some(Type::Ptr(Box::new(Type::I8)))
            );
            assert_eq!(return_type, &Some(Type::I32));
        } else {
            bail!("Expected extern declaration");
        }
        Ok(())
    }

    #[test]
    fn unsafe_block() -> Result<()> {
        let module = parse_module("x := unsafe { ptr^ }")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Let { value, .. } = module.ast.stmt(module.roots[0]) {
            if let Expression::Unsafe(body) = module.ast.expr(*value) {
                assert_eq!(module.ast.stmts_in(*body).len(), 1);
            } else {
                bail!("Expected unsafe expression");
            }
        } else {
            bail!("Expected let statement");
        }
        Ok(())
    }

    #[test]
    fn inclusive_range_expression() -> Result<()> {
        let module = parse_module("for i in 0..=10 { print(i) }")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::For(_, _, range, _) = module.ast.stmt(module.roots[0])
        {
            if let Expression::Range(_, _, inclusive) = module.ast.expr(*range)
            {
                assert!(inclusive, "Expected inclusive range");
            } else {
                bail!("Expected range expression");
            }
        } else {
            bail!("Expected for statement");
        }
        Ok(())
    }

    #[test]
    fn exclusive_range_expression() -> Result<()> {
        let module = parse_module("for i in 0..10 { print(i) }")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::For(_, _, range, _) = module.ast.stmt(module.roots[0])
        {
            if let Expression::Range(_, _, inclusive) = module.ast.expr(*range)
            {
                assert!(!inclusive, "Expected exclusive range");
            } else {
                bail!("Expected range expression");
            }
        } else {
            bail!("Expected for statement");
        }
        Ok(())
    }

    #[test]
    fn isize_type_annotation() -> Result<()> {
        let module = parse_module("x: isize = 42")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Let {
            type_annotation, ..
        } = module.ast.stmt(module.roots[0])
        {
            assert_eq!(type_annotation.as_ref(), Some(&Type::Isize));
        } else {
            bail!("Expected let statement");
        }
        Ok(())
    }

    #[test]
    fn usize_type_annotation() -> Result<()> {
        let module = parse_module("x: usize = 42")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Let {
            type_annotation, ..
        } = module.ast.stmt(module.roots[0])
        {
            assert_eq!(type_annotation.as_ref(), Some(&Type::Usize));
        } else {
            bail!("Expected let statement");
        }
        Ok(())
    }

    #[test]
    fn enum_declaration_unit_variants() -> Result<()> {
        let module = parse_module("Color :: enum { Red, Green, Blue }")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Enum(name, _, variants) = ast.stmt(module.roots[0]) {
            assert_eq!(ast.name(*name), "Color");
            let variants = ast.variants_in(*variants);
            assert_eq!(variants.len(), 3);
            assert_eq!(ast.name(variants[0].name), "Red");
            assert!(variants[0].fields.is_none());
            assert_eq!(ast.name(variants[1].name), "Green");
            assert!(variants[1].fields.is_none());
            assert_eq!(ast.name(variants[2].name), "Blue");
            assert!(variants[2].fields.is_none());
        } else {
            bail!("Expected enum declaration");
        }
        Ok(())
    }

    #[test]
    fn enum_declaration_data_variants() -> Result<()> {
        let input = "Result :: enum { Ok { value: i64 }, Err { code: i64, message: str } }";
        let module = parse_module(input)?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Enum(name, _, variants) = ast.stmt(module.roots[0]) {
            assert_eq!(ast.name(*name), "Result");
            let variants = ast.variants_in(*variants);
            assert_eq!(variants.len(), 2);
            assert_eq!(ast.name(variants[0].name), "Ok");
            let ok_fields = ast.fields_in(variants[0].fields.unwrap());
            assert_eq!(ok_fields.len(), 1);
            assert_eq!(ast.name(ok_fields[0].name), "value");
            assert_eq!(ast.name(variants[1].name), "Err");
            let err_fields = ast.fields_in(variants[1].fields.unwrap());
            assert_eq!(err_fields.len(), 2);
            assert_eq!(ast.name(err_fields[0].name), "code");
            assert_eq!(ast.name(err_fields[1].name), "message");
        } else {
            bail!("Expected enum declaration");
        }
        Ok(())
    }

    #[test]
    fn enum_variant_init_unit() -> Result<()> {
        let module = parse_module("color := Color::Red")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Let { value, .. } = ast.stmt(module.roots[0]) {
            if let Expression::EnumVariantInit(
                enum_name,
                variant_name,
                fields,
            ) = ast.expr(*value)
            {
                assert_eq!(ast.name(*enum_name), "Color");
                assert_eq!(ast.name(*variant_name), "Red");
                assert!(fields.is_empty());
            } else {
                bail!("Expected enum variant init, got {:?}", ast.expr(*value));
            }
        } else {
            bail!("Expected let statement");
        }
        Ok(())
    }

    #[test]
    fn enum_variant_init_with_fields() -> Result<()> {
        let module = parse_module("result := Result::Ok { value = 42 }")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Let { value, .. } = ast.stmt(module.roots[0]) {
            if let Expression::EnumVariantInit(
                enum_name,
                variant_name,
                fields,
            ) = ast.expr(*value)
            {
                assert_eq!(ast.name(*enum_name), "Result");
                assert_eq!(ast.name(*variant_name), "Ok");
                let fields = ast.named_in(*fields);
                assert_eq!(fields.len(), 1);
                assert_eq!(ast.name(fields[0].name), "value");
            } else {
                bail!("Expected enum variant init, got {:?}", ast.expr(*value));
            }
        } else {
            bail!("Expected let statement");
        }
        Ok(())
    }

    #[test]
    fn match_expression_integer_patterns() -> Result<()> {
        let input = r#"
            result := match x {
                case 1: "one"
                case 2: "two"
                case _: "other"
            }
        "#;
        let module = parse_module(input)?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Let { value, .. } = ast.stmt(module.roots[0]) {
            if let Expression::Switch(scrutinee, cases) = ast.expr(*value) {
                assert_identifier(ast, *scrutinee, "x")?;
                let cases = ast.cases_in(*cases);
                assert_eq!(cases.len(), 3);
                assert!(matches!(
                    ast.pattern(cases[0].pattern),
                    Pattern::Literal(Literal::Integer(1))
                ));
                assert!(matches!(
                    ast.pattern(cases[1].pattern),
                    Pattern::Literal(Literal::Integer(2))
                ));
                assert!(matches!(
                    ast.pattern(cases[2].pattern),
                    Pattern::Wildcard
                ));
            } else {
                bail!("Expected match expression");
            }
        } else {
            bail!("Expected let statement");
        }
        Ok(())
    }

    #[test]
    fn match_expression_shorthand_enum_pattern() -> Result<()> {
        let input = r#"
            result := match color {
                case .Red: 0
                case .Green: 1
                case .Blue: 2
            }
        "#;
        let module = parse_module(input)?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Let { value, .. } = ast.stmt(module.roots[0]) {
            if let Expression::Switch(_, cases) = ast.expr(*value) {
                let cases = ast.cases_in(*cases);
                assert_eq!(cases.len(), 3);
                if let Pattern::EnumVariant {
                    enum_name,
                    variant_name,
                    bindings,
                } = ast.pattern(cases[0].pattern)
                {
                    assert!(enum_name.is_none());
                    assert_eq!(ast.name(*variant_name), "Red");
                    assert!(bindings.is_empty());
                } else {
                    bail!("Expected enum variant pattern");
                }
            } else {
                bail!("Expected match expression");
            }
        } else {
            bail!("Expected let statement");
        }
        Ok(())
    }

    #[test]
    fn match_expression_enum_pattern_with_bindings() -> Result<()> {
        let input = r#"
            result := match opt {
                case .Some { value }: value
                case .None: 0
            }
        "#;
        let module = parse_module(input)?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Let { value, .. } = ast.stmt(module.roots[0]) {
            if let Expression::Switch(_, cases) = ast.expr(*value) {
                let cases = ast.cases_in(*cases);
                assert_eq!(cases.len(), 2);
                if let Pattern::EnumVariant {
                    enum_name,
                    variant_name,
                    bindings,
                } = ast.pattern(cases[0].pattern)
                {
                    assert!(enum_name.is_none());
                    assert_eq!(ast.name(*variant_name), "Some");
                    let bindings = ast.pattern_bindings_in(*bindings);
                    assert_eq!(bindings.len(), 1);
                    assert_eq!(ast.name(bindings[0].field), "value");
                } else {
                    bail!("Expected enum variant pattern with bindings");
                }
            } else {
                bail!("Expected match expression");
            }
        } else {
            bail!("Expected let statement");
        }
        Ok(())
    }

    #[test]
    fn match_expression_fully_qualified_pattern() -> Result<()> {
        let input = r#"
            result := match color {
                case Color::Red: 0
                case Color::Green: 1
            }
        "#;
        let module = parse_module(input)?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Let { value, .. } = ast.stmt(module.roots[0]) {
            if let Expression::Switch(_, cases) = ast.expr(*value) {
                let cases = ast.cases_in(*cases);
                assert_eq!(cases.len(), 2);
                if let Pattern::EnumVariant {
                    enum_name,
                    variant_name,
                    ..
                } = ast.pattern(cases[0].pattern)
                {
                    assert_eq!(ast.name((*enum_name).unwrap()), "Color");
                    assert_eq!(ast.name(*variant_name), "Red");
                } else {
                    bail!("Expected fully qualified enum variant pattern");
                }
            } else {
                bail!("Expected match expression");
            }
        } else {
            bail!("Expected let statement");
        }
        Ok(())
    }

    #[test]
    fn match_expression_tuple_pattern() -> Result<()> {
        let input = r#"
            result := match (x % 3, x % 5) {
                case (0, 0): "FizzBuzz"
                case (0, _): "Fizz"
                case (_, 0): "Buzz"
            }
        "#;
        let module = parse_module(input)?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Let { value, .. } = ast.stmt(module.roots[0]) {
            if let Expression::Switch(_, cases) = ast.expr(*value) {
                let cases = ast.cases_in(*cases);
                assert_eq!(cases.len(), 3);
                if let Pattern::Tuple(patterns) = ast.pattern(cases[0].pattern)
                {
                    let patterns = ast.patterns_in(*patterns);
                    assert_eq!(patterns.len(), 2);
                    assert!(matches!(
                        ast.pattern(patterns[0]),
                        Pattern::Literal(Literal::Integer(0))
                    ));
                    assert!(matches!(
                        ast.pattern(patterns[1]),
                        Pattern::Literal(Literal::Integer(0))
                    ));
                } else {
                    bail!("Expected tuple pattern");
                }
                if let Pattern::Tuple(patterns) = ast.pattern(cases[1].pattern)
                {
                    let patterns = ast.patterns_in(*patterns);
                    assert!(matches!(
                        ast.pattern(patterns[1]),
                        Pattern::Wildcard
                    ));
                } else {
                    bail!("Expected tuple pattern with wildcard");
                }
            } else {
                bail!("Expected match expression");
            }
        } else {
            bail!("Expected let statement");
        }
        Ok(())
    }

    #[test]
    fn match_expression_bool_pattern() -> Result<()> {
        let input = r#"
            result := match flag {
                case true: 1
                case false: 0
            }
        "#;
        let module = parse_module(input)?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Let { value, .. } = ast.stmt(module.roots[0]) {
            if let Expression::Switch(_, cases) = ast.expr(*value) {
                let cases = ast.cases_in(*cases);
                assert_eq!(cases.len(), 2);
                assert!(matches!(
                    ast.pattern(cases[0].pattern),
                    Pattern::Literal(Literal::Boolean(true))
                ));
                assert!(matches!(
                    ast.pattern(cases[1].pattern),
                    Pattern::Literal(Literal::Boolean(false))
                ));
            } else {
                bail!("Expected match expression");
            }
        } else {
            bail!("Expected let statement");
        }
        Ok(())
    }

    #[test]
    fn enum_mixed_variants() -> Result<()> {
        let module =
            parse_module("Option :: enum { None, Some { value: i64 } }")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Enum(name, _, variants) = ast.stmt(module.roots[0]) {
            assert_eq!(ast.name(*name), "Option");
            let variants = ast.variants_in(*variants);
            assert_eq!(variants.len(), 2);
            assert_eq!(ast.name(variants[0].name), "None");
            assert!(variants[0].fields.is_none());
            assert_eq!(ast.name(variants[1].name), "Some");
            assert!(variants[1].fields.is_some());
            assert_eq!(ast.fields_in(variants[1].fields.unwrap()).len(), 1);
        } else {
            bail!("Expected enum declaration");
        }
        Ok(())
    }

    #[test]
    fn multiple_returns_two_types() -> Result<()> {
        let input = "fn(a: i64, b: i64) -> (i64, i64) { return a / b, a % b }";
        let module = parse_module(input)?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Expression(expression) = ast.stmt(module.roots[0])
            && let Expression::Proc(params, return_sig, body) =
                ast.expr(*expression)
        {
            assert_eq!(ast.params_in(*params).len(), 2);
            if let ReturnKind::Multiple(values) =
                &ast.signature(*return_sig).kind
            {
                let values = ast.return_values_in(*values);
                assert_eq!(values.len(), 2);
                assert!(values.iter().all(|held| held.name.is_none()));
                assert!(values.iter().all(|held| held.value_type == Type::I64));
            } else {
                bail!("Expected a return type list");
            }
            let body = ast.stmts_in(*body);
            if let Statement::Return(value) = ast.stmt(body[0])
                && let Expression::Tuple(values) = ast.expr(*value)
            {
                assert_eq!(values.len(), 2);
            } else {
                bail!("Expected a return of two values");
            }
        } else {
            bail!("Expected function expression");
        }
        Ok(())
    }

    #[test]
    fn multiple_returns_bind_by_name() -> Result<()> {
        let module = parse_module("quotient, var remainder := divide(7, 2)")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::LetMultiple(bindings, _) = ast.stmt(module.roots[0]) {
            let bindings = ast.bindings_in(*bindings);
            assert_eq!(bindings.len(), 2);
            assert_eq!(ast.name(bindings[0].name), "quotient");
            assert!(!bindings[0].mutable);
            assert_eq!(ast.name(bindings[1].name), "remainder");
            assert!(bindings[1].mutable);
        } else {
            bail!("Expected a list binding");
        }
        Ok(())
    }

    #[test]
    fn a_return_type_list_needs_two_types() {
        let input = "fn(a: i64) -> (i64) { a }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        assert!(parser.parse().is_err());
    }

    #[test]
    fn return_signature_to_type_single() {
        let ast = Ast::default();
        let sig = ReturnSignature::plain(ReturnKind::Single(Type::I64));
        assert_eq!(ast.signature_to_type(&sig), Some(Type::I64));
    }

    #[test]
    fn return_signature_to_type_multiple() {
        let mut ast = Ast::default();
        let unnamed = |value_type| ReturnValue {
            name: None,
            value_type,
        };
        let values = ast
            .add_return_values(vec![unnamed(Type::I64), unnamed(Type::Bool)]);
        let sig = ReturnSignature::plain(ReturnKind::Multiple(values));
        assert_eq!(
            ast.signature_to_type(&sig),
            Some(Type::Struct("__multi_i64_bool".to_string()))
        );

        // Named values are part of what the struct is, so a list that names
        // them is a different struct from one that does not.
        let quotient = ast.intern("quotient");
        let remainder = ast.intern("remainder");
        let values = ast.add_return_values(vec![
            ReturnValue {
                name: Some(quotient),
                value_type: Type::I64,
            },
            ReturnValue {
                name: Some(remainder),
                value_type: Type::I64,
            },
        ]);
        let sig = ReturnSignature::plain(ReturnKind::Multiple(values));
        assert_eq!(
            ast.signature_to_type(&sig),
            Some(Type::Struct(
                "__multi_quotient__i64_remainder__i64".to_string()
            ))
        );
    }

    #[test]
    fn a_return_type_list_names_all_of_its_values_or_none() {
        let input = "fn(a: i64) -> (quotient: i64, i64) { return a, a }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(&tokens);
        assert!(parser.parse().is_err());
    }

    #[test]
    fn return_signature_to_type_none() {
        let ast = Ast::default();
        let sig = ReturnSignature::plain(ReturnKind::None);
        assert_eq!(ast.signature_to_type(&sig), None);
    }

    #[test]
    fn generic_function_parameter() -> Result<()> {
        let module = parse_module("identity :: fn(x: $T) -> T { x }")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Constant(name, value) = ast.stmt(module.roots[0])
            && let Expression::Proc(params, return_sig, _body) =
                ast.expr(*value)
        {
            assert_eq!(ast.name(*name), "identity");
            let params = ast.params_in(*params);
            assert_eq!(params.len(), 1);
            assert_eq!(ast.name(params[0].name), "x");
            assert_eq!(
                params[0].type_annotation,
                Some(Type::TypeParam("T".to_string()))
            );
            if let ReturnKind::Single(ret_type) =
                &ast.signature(*return_sig).kind
            {
                assert_eq!(*ret_type, Type::Struct("T".to_string()));
            } else {
                bail!("Expected single return type");
            }
        } else {
            bail!("Expected constant function declaration");
        }
        Ok(())
    }

    #[test]
    fn generic_function_multiple_type_params() -> Result<()> {
        let module = parse_module("pair :: fn(a: $T, b: $U) -> void { }")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Constant(name, value) = ast.stmt(module.roots[0])
            && let Expression::Proc(params, _, _) = ast.expr(*value)
        {
            assert_eq!(ast.name(*name), "pair");
            let params = ast.params_in(*params);
            assert_eq!(params.len(), 2);
            assert_eq!(
                params[0].type_annotation,
                Some(Type::TypeParam("T".to_string()))
            );
            assert_eq!(
                params[1].type_annotation,
                Some(Type::TypeParam("U".to_string()))
            );
        } else {
            bail!("Expected constant function declaration");
        }
        Ok(())
    }

    #[test]
    fn parameterized_struct() -> Result<()> {
        let input =
            "Pair :: struct($T: Type, $U: Type) { first: T, second: U }";
        let module = parse_module(input)?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Struct(name, type_params, fields) =
            ast.stmt(module.roots[0])
        {
            assert_eq!(ast.name(*name), "Pair");
            let type_params = ast.symbols_in(*type_params);
            assert_eq!(type_params.len(), 2);
            assert_eq!(ast.name(type_params[0]), "T");
            assert_eq!(ast.name(type_params[1]), "U");
            let fields = ast.fields_in(*fields);
            assert_eq!(fields.len(), 2);
            assert_eq!(ast.name(fields[0].name), "first");
            assert_eq!(fields[0].field_type, Type::Struct("T".to_string()));
            assert_eq!(ast.name(fields[1].name), "second");
            assert_eq!(fields[1].field_type, Type::Struct("U".to_string()));
        } else {
            bail!("Expected struct declaration");
        }
        Ok(())
    }

    #[test]
    fn parameterized_struct_single_param() -> Result<()> {
        let module = parse_module("Wrapper :: struct($T: Type) { value: T }")?;
        assert_eq!(module.roots.len(), 1);
        let ast = &module.ast;
        if let Statement::Struct(name, type_params, fields) =
            ast.stmt(module.roots[0])
        {
            assert_eq!(ast.name(*name), "Wrapper");
            let type_params = ast.symbols_in(*type_params);
            assert_eq!(type_params.len(), 1);
            assert_eq!(ast.name(type_params[0]), "T");
            let fields = ast.fields_in(*fields);
            assert_eq!(fields.len(), 1);
            assert_eq!(ast.name(fields[0].name), "value");
            assert_eq!(fields[0].field_type, Type::Struct("T".to_string()));
        } else {
            bail!("Expected struct declaration");
        }
        Ok(())
    }

    fn constant_names(module: &Module) -> Vec<String> {
        module
            .roots
            .iter()
            .filter_map(|statement| match module.ast.stmt(*statement) {
                Statement::Constant(name, _) => {
                    Some(module.ast.name(*name).to_string())
                }
                Statement::Struct(name, _, _) => {
                    Some(module.ast.name(*name).to_string())
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn recovery_keeps_declarations_after_a_bad_one() -> Result<()> {
        let input = "first :: fn(a: i64 { a }\nsecond :: fn() -> i64 { 7 }\n";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        let (program, diagnostics) = parser.parse_recovering();
        assert!(
            !diagnostics.is_empty(),
            "the malformed declaration should produce a diagnostic"
        );
        let names = constant_names(&program);
        assert!(
            names.iter().any(|name| name == "second"),
            "recovery should keep 'second', got {names:?}"
        );
        Ok(())
    }

    // A token in a message reads as the reader wrote it, not as the compiler
    // stores it: 'found '5'', never 'found Integer(5)'.
    #[test]
    fn a_message_spells_the_token_the_reader_wrote() -> Result<()> {
        let input = "f :: fn(5) -> i64 { 1 }\n";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        let (_, diagnostics) = parser.parse_recovering();
        assert!(!diagnostics.is_empty());
        assert!(
            diagnostics[0].message.contains("found '5'"),
            "got {diagnostics:?}"
        );
        assert!(
            !diagnostics[0].message.contains("Integer"),
            "got {diagnostics:?}"
        );
        Ok(())
    }

    // The statement dispatch answers with what could stand at the position,
    // rather than whichever inner branch's message surfaces.
    #[test]
    fn a_stray_token_names_what_a_declaration_head_could_be() -> Result<()> {
        let input = "}\ngood :: fn() -> i64 { 1 }\n";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        let (program, diagnostics) = parser.parse_recovering();
        assert!(!diagnostics.is_empty());
        assert!(
            diagnostics[0].message.contains(
                "expected a declaration head, `import`, `export`, or `test`, found '}'"
            ),
            "got {diagnostics:?}"
        );
        let names = constant_names(&program);
        assert!(names.iter().any(|name| name == "good"));
        Ok(())
    }

    // The diagnostic lands on the offending token, not wherever the cursor
    // stopped once recovery had skipped ahead.
    #[test]
    fn a_diagnostic_lands_on_the_offending_token() -> Result<()> {
        let input = "a :: fn() -> i64 { 1 }\n\
                     b :: fn(, x: i64) -> i64 { x }\n\
                     c :: fn() -> i64 { 3 }\n";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        let (_, diagnostics) = parser.parse_recovering();
        assert!(!diagnostics.is_empty());
        assert_eq!(diagnostics[0].position.line, 2, "got {diagnostics:?}");
        assert!(
            diagnostics[0]
                .message
                .contains("Expected a parameter name in parameter list"),
            "got {diagnostics:?}"
        );
        Ok(())
    }

    // A file broken in the lexer and broken in the parser reports both, the
    // lexer's first, and a program with either kind of fault is still refused.
    #[test]
    fn lexer_faults_and_parse_faults_report_together() -> Result<()> {
        let input = "text :: \"one\\q\"\nbroken :: fn(a: i64 { a }\nlast :: fn() -> i64 { 7 }\n";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        parser.preload_diagnostics(lexer.diagnostics().to_vec());
        let (program, diagnostics) = parser.parse_recovering();
        assert!(diagnostics.len() >= 2, "got {diagnostics:?}");
        assert!(
            diagnostics[0].message.contains("Unknown escape sequence"),
            "the lexer's diagnostic goes first, got {diagnostics:?}"
        );
        assert_eq!(diagnostics[0].position.line, 1);
        let names = constant_names(&program);
        assert!(
            names.iter().any(|name| name == "last"),
            "recovery should keep 'last', got {names:?}"
        );

        let mut refused = Parser::with_positions(&tokens, &positions);
        refused.preload_diagnostics(lexer.diagnostics().to_vec());
        assert!(refused.parse().is_err(), "a lexer fault still refuses");
        Ok(())
    }

    #[test]
    fn recovery_reports_every_bad_declaration() -> Result<()> {
        let input = "a :: fn(x: i64 { x }\n\
                     good :: fn() -> i64 { 1 }\n\
                     b :: fn(y: i64 { y }\n";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        let (program, diagnostics) = parser.parse_recovering();
        assert_eq!(
            diagnostics.len(),
            2,
            "each malformed declaration should be reported: {diagnostics:?}"
        );
        assert!(
            diagnostics
                .iter()
                .all(|diagnostic| diagnostic.position.line > 0),
            "diagnostics should carry a source position"
        );
        let names = constant_names(&program);
        assert!(
            names.iter().any(|name| name == "good"),
            "the valid declaration between the errors should survive"
        );
        Ok(())
    }

    #[test]
    fn clean_input_produces_no_diagnostics() -> Result<()> {
        let input = "a :: fn() -> i64 { 1 }\nb :: fn() -> i64 { 2 }\n";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let mut parser = Parser::new(&tokens);
        let (program, diagnostics) = parser.parse_recovering();
        assert!(diagnostics.is_empty(), "clean input has no diagnostics");
        assert_eq!(program.roots.len(), 2);
        Ok(())
    }

    #[test]
    fn recovery_keeps_statements_after_a_bad_one_in_a_block() -> Result<()> {
        let input = "main :: fn() -> i64 {\n\
                     x := 1\n\
                     y := )\n\
                     z := 3\n\
                     z\n\
                     }\n";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        let (program, diagnostics) = parser.parse_recovering();
        assert!(
            !diagnostics.is_empty(),
            "the malformed statement should produce a diagnostic"
        );
        let ast = &program.ast;
        let Statement::Constant(name, value) = ast.stmt(program.roots[0])
        else {
            bail!("expected a function constant");
        };
        let (Expression::Function(_, _, body) | Expression::Proc(_, _, body)) =
            ast.expr(*value)
        else {
            bail!("expected a function constant");
        };
        assert_eq!(ast.name(*name), "main");
        let bindings: Vec<&str> = ast
            .stmts_in(*body)
            .iter()
            .filter_map(|statement| match ast.stmt(*statement) {
                Statement::Let { name, .. } => Some(ast.name(*name)),
                _ => None,
            })
            .collect();
        assert!(
            bindings.contains(&"x") && bindings.contains(&"z"),
            "recovery inside the block should keep 'x' and 'z', got {bindings:?}"
        );
        Ok(())
    }

    #[test]
    fn recovery_anchors_on_a_call_statement() -> Result<()> {
        let input = "main :: fn() -> i64 {\n\
                     bad := )\n\
                     log(bad)\n\
                     0\n\
                     }\n";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        let (program, diagnostics) = parser.parse_recovering();
        assert!(!diagnostics.is_empty());
        let ast = &program.ast;
        let Statement::Constant(_, value) = ast.stmt(program.roots[0]) else {
            bail!("expected a function constant");
        };
        let (Expression::Function(_, _, body) | Expression::Proc(_, _, body)) =
            ast.expr(*value)
        else {
            bail!("expected a function constant");
        };
        let has_call = ast.stmts_in(*body).iter().any(|statement| {
            matches!(
                ast.stmt(*statement),
                Statement::Expression(expression)
                    if matches!(ast.expr(*expression), Expression::Call(..))
            )
        });
        assert!(
            has_call,
            "recovery should anchor on the call statement 'log(bad)' and keep it"
        );
        Ok(())
    }

    #[test]
    fn recovery_inside_a_block_keeps_later_declarations() -> Result<()> {
        let input = "first :: fn() -> i64 {\n\
                     bad := )\n\
                     0\n\
                     }\n\
                     second :: fn() -> i64 { 7 }\n";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize()?;
        let positions = lexer.positions().to_vec();
        let mut parser = Parser::with_positions(&tokens, &positions);
        let (program, diagnostics) = parser.parse_recovering();
        assert!(!diagnostics.is_empty());
        let names = constant_names(&program);
        assert!(
            names.iter().any(|name| name == "first")
                && names.iter().any(|name| name == "second"),
            "both functions should survive a body error, got {names:?}"
        );
        Ok(())
    }
}
