use crate::ast::{
    Ast, EnumVariant, ExprId, Expression, FlagBit, ImportRename, Literal,
    Module, MultiBinding, NamedExpr, Parameter, Pattern, PatternBinding,
    PatternId, Range32, ReturnKind, ReturnSignature, ReturnValue, Statement,
    StmtId, StructField, SwitchCase, Symbol, TokenSpan, TypeValue,
};
use crate::{
    lexer::Position,
    lexer::Token,
    types::{SizeExpr, SizeOp, Type},
};
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

// What a `case` says about its own coverage. These are the four ways a pattern
// can claim a set that the arm it sits on cannot mean, and both compilers word
// them the same because a program refused here is refused by the language
// rather than by whichever compiler read it.
// What the space around a `!` says. The two meanings are told apart by it, so a
// program is held to writing each the way it is read: both compilers word these
// the same because a program refused here is refused by the language rather
// than by whichever compiler read it.
const FAILURE_SET_SPACING: &str = "a `!` against what follows it negates, and this one marks a failure set; write `-> T ! E`, with a space on both sides";
const NEGATION_SPACING: &str = "a `!` with a space after it marks a failure set, and this one negates; write it against what it negates, as `!ready`";

const CATCH_ALL_ALTERNATIVE: &str = "this alternative covers everything on its own, so give it a case of its own";
const BINDING_ALTERNATIVE: &str = "an alternative binding payload fields holds one name to two shapes, so give it a case of its own";
const TUPLE_PART: &str = "a tuple case compares one value per part, so an alternative or a range belongs in a match on one value";
const NOT_A_BOUND: &str =
    "a case range runs between whole numbers, and this bound is not one";
const REPEATED_ALTERNATIVE: &str =
    "this alternative repeats one the same case already names";
const DECIMAL_PATTERN: &str = "a case matches whole numbers, booleans and variants, so a decimal belongs in an `if`";
const TEXT_PATTERN: &str = "a case matches whole numbers, booleans and variants, so text belongs in an `if`";
const NAME_PATTERN: &str = "a name in a case is the value it stands for, and this one names no constant; `_` is the case that covers the rest";

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

// Which bit of a set a number reaches, counting the lowest as the first. A set
// over `u32` holds thirty-two, so anything reaching past that is refused, and
// the answer is what the refusal names. Zero reaches no bit and fits anywhere,
// which is what lets `None :: 0` open a set of any width.
fn bit_ordinal(value: i64) -> u32 {
    if value <= 0 {
        return 0;
    }
    64 - (value as u64).leading_zeros()
}

// A count written the way it is spoken. Only the widths of the integer types
// reach this, so the numbers are small, but the rule is the English one so the
// two compilers have the same one to write.
fn ordinal(count: u32) -> String {
    let suffix = if count % 100 / 10 == 1 {
        "th"
    } else {
        match count % 10 {
            1 => "st",
            2 => "nd",
            3 => "rd",
            _ => "th",
        }
    };
    format!("{count}{suffix}")
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
    // The same tokens as a whole, which the constant scan reads. It runs over
    // the file rather than along it, since a constant may name a function
    // written below the line that calls it.
    all_tokens: &'a [Token],
    // The arena every parse method pushes into. `parse` hands it out inside
    // the finished `Module`.
    ast: Ast,
    linear_types: std::collections::HashSet<String>,
    tests: Vec<(String, String)>,
    exports: Vec<String>,
    positions: &'a [Position],
    consumed: usize,
    // How many `_ :=` this file has taken, so each binds a name of its own.
    discards: usize,
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
    // What the `for` loops around the statement being read have named. A write
    // to one of them is a write to a copy, which changes nothing.
    loop_names: Vec<Symbol>,
    // Top-level `N :: 8` constants, read off the token stream before the parse
    // so that an array size may name one wherever it appears, including above
    // the line that declares it.
    /// Every constant whose value is a whole number, held as one.
    ///
    /// Held as a count, a negative was not in it at all and the name read as
    /// one nothing declares: `N :: 0 - 4` written as an array length was
    /// reported as not being a constant, which it is. Every place that asks
    /// what a length is asks this, so there is one answer rather than one per
    /// site.
    integer_constants: HashMap<String, i64>,
    // The same constants with what each one worked out to, and the machinery
    // that works a call out. A length may call a function the file declares,
    // and the answer has to be in hand where the length is read.
    constant_values: HashMap<String, crate::const_eval::Value>,
    folder: crate::const_eval::Folder<'a>,
    // What the files this one imports export, as function bodies a
    // compile-time call may read. Filled in before the parse, since the
    // constants are worked out before it too.
    imported_bodies: HashMap<String, std::rc::Rc<Vec<Token>>>,
    settled: bool,
    // Every generic struct and enum declared in this file, by name. A literal
    // may say which instance it is, `Pair<i64, bool> { .. }`, and telling that
    // from the comparison `a < b` is a question of whether the name is one of
    // these.
    generic_types: GenericDefaults,
    // The lines a `when` kept, by the range each branch spanned.
    lifted_lines: Vec<(usize, usize)>,
    // How many blocks deep the parse is. `name :: Type { .. }` is a declaration
    // at the top level and `Enum::Variant { .. }` inside a body, and the two
    // read the same token for token, so where it is written is what tells them
    // apart.
    block_depth: usize,
    // How many brackets are open. A line break inside one says nothing, so an
    // expression may run over lines there and nowhere else.
    bracket_depth: i32,
    // Whether the file being parsed is the runtime, which is the one file that
    // may define a name in the runtime's own name space.
    runtime_names: bool,
    // The compile-time parameters in scope, one frame per declaration being
    // read. A length may name one, and a number arrives for it where an
    // instance is written, so a name in here is a length that is waiting rather
    // than a name the program binds to nothing.
    compile_time_names: Vec<std::collections::HashSet<String>>,
}

// Where a top-level declaration's value ends: at the head of the next one, or
// at the end of the file. A declaration head is a name followed by `::`, and
// `import` is the one that starts with a keyword instead. Everything nested is
// inside brackets of some kind, so only depth zero is looked at.
/// How many brackets a run of tokens opens and does not close.
fn left_open(tokens: Iter<'_, Token>, read: usize) -> u32 {
    let mut open: u32 = 0;
    for token in tokens.take(read) {
        open = counted(open, token);
    }
    open
}

/// The count of open brackets after one more token.
fn counted(open: u32, token: &Token) -> u32 {
    match token {
        Token::LeftBrace | Token::LeftParentheses | Token::LeftBracket => {
            open + 1
        }
        Token::RightBrace | Token::RightParentheses | Token::RightBracket => {
            open.saturating_sub(1)
        }
        _ => open,
    }
}

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
/// What each of a generic type's parameters stands for where an instance
/// leaves it out, in declaration order. `None` is a parameter every instance
/// writes.
pub type GenericDefaults =
    std::collections::HashMap<String, Vec<Option<crate::types::Type>>>;

pub fn scan_generic_types(tokens: &[Token]) -> GenericDefaults {
    let mut names = GenericDefaults::new();
    for index in 0..tokens.len() {
        let Token::Identifier(name) = &tokens[index] else {
            continue;
        };
        if !matches!(tokens.get(index + 1), Some(Token::DoubleColon)) {
            continue;
        }
        // The words a declaration may put between `::` and `struct`. Read as
        // part of the head rather than looked for after it, so a `linear`
        // generic is a generic here the way it is everywhere else: left out, a
        // literal of one read as a name and a comparison, and a reader was told
        // the name was a variable nothing declares.
        let mut head = index + 2;
        while matches!(tokens.get(head), Some(Token::Linear))
            || matches!(tokens.get(head), Some(Token::Identifier(word)) if word == "packed")
        {
            head += 1;
        }
        if !matches!(tokens.get(head), Some(Token::Struct | Token::Enum)) {
            continue;
        }
        if !matches!(tokens.get(head + 1), Some(Token::LeftParentheses)) {
            continue;
        }
        names.insert(name.clone(), scan_parameter_defaults(tokens, head + 2));
    }
    names
}

/// The defaults of one type-parameter list, read from the token after its `(`.
/// The list is `$name : kind` repeated, with `= default` after the kind where
/// there is one, so what is read is the run of tokens between that `=` and the
/// comma or paren closing the parameter.
fn scan_parameter_defaults(
    tokens: &[Token],
    from: usize,
) -> Vec<Option<crate::types::Type>> {
    let mut found = Vec::new();
    let mut index = from;
    let mut depth = 0i32;
    while index < tokens.len() {
        match &tokens[index] {
            Token::LeftParentheses | Token::LessThan => depth += 1,
            Token::RightParentheses if depth == 0 => break,
            Token::RightParentheses | Token::GreaterThan => depth -= 1,
            Token::Dollar if depth == 0 => {
                let mut at = index + 1;
                while at < tokens.len()
                    && !matches!(
                        tokens[at],
                        Token::Comma | Token::Assign | Token::RightParentheses
                    )
                {
                    at += 1;
                }
                if matches!(tokens.get(at), Some(Token::Assign)) {
                    let start = at + 1;
                    let mut end = start;
                    let mut inner = 0i32;
                    while end < tokens.len() {
                        match &tokens[end] {
                            Token::LessThan | Token::LeftBracket => inner += 1,
                            Token::GreaterThan | Token::RightBracket => {
                                inner -= 1
                            }
                            Token::Comma | Token::RightParentheses
                                if inner <= 0 =>
                            {
                                break;
                            }
                            _ => {}
                        }
                        end += 1;
                    }
                    found.push(type_from_tokens(&tokens[start..end]));
                    index = end;
                    continue;
                }
                found.push(None);
                index = at;
                continue;
            }
            _ => {}
        }
        index += 1;
    }
    found
}

/// One type read out of the tokens it was written as. A default is written
/// where the declaration is and read where an instance leaves the parameter
/// out, which are two different files.
fn type_from_tokens(tokens: &[Token]) -> Option<crate::types::Type> {
    if let [Token::Integer(value)] = tokens {
        return Some(crate::types::Type::ConstUsize(*value as usize));
    }
    let mut parser = bare_parser(tokens);
    parser.parse_type().ok()
}

// A parser over a run of tokens and nothing else, for reading one value back
// out of the file it was written in. It knows no constants and no generic
// types, so what it reads is what the tokens say.
fn bare_parser(tokens: &[Token]) -> Parser<'_> {
    Parser {
        tokens: tokens.iter(),
        all_tokens: tokens,
        ast: Ast::default(),
        linear_types: std::collections::HashSet::new(),
        tests: Vec::new(),
        exports: Vec::new(),
        positions: &[],
        consumed: 0,
        discards: 0,
        pending_angle_close: 0,
        diagnostics: Vec::new(),
        internal_types: false,
        no_struct_literal: false,
        loop_names: Vec::new(),
        integer_constants: HashMap::new(),
        constant_values: HashMap::new(),
        folder: crate::const_eval::Folder::new(
            &[],
            HashMap::new(),
            HashMap::new(),
        ),
        imported_bodies: HashMap::new(),
        settled: true,
        generic_types: GenericDefaults::new(),
        lifted_lines: Vec::new(),
        block_depth: 0,
        bracket_depth: 0,
        runtime_names: false,
        compile_time_names: Vec::new(),
    }
}

/// Every function a file exports, as the run of tokens its value occupies.
/// A file that imports this one may call one where a compile-time value is
/// read, and a private name it could not write is left out.
pub fn exported_function_bodies(
    tokens: &[Token],
) -> HashMap<String, std::rc::Rc<Vec<Token>>> {
    let exported = exported_names(tokens);
    scan_function_bodies(tokens)
        .into_iter()
        .filter(|(name, _)| exported.contains(name))
        .map(|(name, (start, end))| {
            (name, std::rc::Rc::new(tokens[start..end].to_vec()))
        })
        .collect()
}

// The names on a file's `export` lines, read off the tokens because this runs
// before the parse that would record them.
fn exported_names(tokens: &[Token]) -> std::collections::HashSet<String> {
    let mut found = std::collections::HashSet::new();
    let mut index = 0;
    while index < tokens.len() {
        let Token::Identifier(word) = &tokens[index] else {
            index += 1;
            continue;
        };
        if word != "export" {
            index += 1;
            continue;
        }
        index += 1;
        while let Some(Token::Identifier(name)) = tokens.get(index) {
            found.insert(name.clone());
            index += 1;
            if !matches!(tokens.get(index), Some(Token::Comma)) {
                break;
            }
            index += 1;
        }
    }
    found
}

/// A function value read on its own, out of the tokens it was written in.
///
/// A constant may call a function declared below it, and the call is worked out
/// before the parse that needs the number gets there, so the body has to be
/// reachable without waiting for the parse to reach it.
pub(crate) fn parse_function_value(tokens: &[Token]) -> Option<(Ast, ExprId)> {
    let mut sub = bare_parser(tokens);
    let expression = sub.parse_expression(Precedence::Lowest).ok()?;
    // A function whose parameters carry types is a `Proc` and one written
    // without them is a `Function`. Either is a body to read.
    if !matches!(
        sub.ast.expr(expression),
        Expression::Function(..) | Expression::Proc(..)
    ) {
        return None;
    }
    Some((sub.ast, expression))
}

// Every top-level `name :: fn(...)`, as the token range its value occupies.
// This is what a compile-time call reads, and it is collected off the tokens
// because a constant may name a function written below it.
fn scan_function_bodies(tokens: &[Token]) -> HashMap<String, (usize, usize)> {
    let mut found = HashMap::new();
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
        if !matches!(tokens.get(index + 2), Some(Token::Function)) {
            continue;
        }
        let end = declaration_value_end(tokens, index + 2);
        found.insert(name.clone(), (index + 2, end));
    }
    found
}

// Whether this expression asks to be worked out: it calls something, reads an
// item out of a run, or reads a field. A constant that does is one the program
// asked the compiler to settle, so failing to settle it is a fault rather than
// a constant left unfolded. In particular an index past the end is refused
// where it is written rather than left to abort while the program runs.
fn holds_a_call(ast: &Ast, expression: ExprId) -> bool {
    match ast.expr(expression) {
        Expression::Call(..)
        | Expression::Index(..)
        | Expression::FieldAccess(..) => true,
        Expression::Prefix(_, inner) => holds_a_call(ast, *inner),
        Expression::Infix(left, _, right) => {
            holds_a_call(ast, *left) || holds_a_call(ast, *right)
        }
        _ => false,
    }
}

// Every top-level constant, with what it worked out to, the machinery that
// works a call out, and whatever a constant asking for one broke on.
//
// The values are needed before the parse: an array size is part of a type and
// a repeat count is expanded into elements, both while parsing, so neither can
// wait for a later pass.
type ConstantScan<'a> = (
    HashMap<String, crate::const_eval::Value>,
    crate::const_eval::Folder<'a>,
    Vec<(usize, String)>,
);

fn scan_constants(
    tokens: &[Token],
    imported: HashMap<String, std::rc::Rc<Vec<Token>>>,
) -> ConstantScan<'_> {
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

    let mut folder = crate::const_eval::Folder::new(
        tokens,
        scan_function_bodies(tokens),
        imported,
    );
    // In source order, so that a constant reading an earlier one sees it. The
    // scan runs before the parse, so "earlier" is the only order there is.
    let mut values: HashMap<String, crate::const_eval::Value> = HashMap::new();
    let mut faults = Vec::new();
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
                    | Token::LeftBracket
                    | Token::StringLiteral(_)
                    | Token::Minus
                    | Token::Float(_)
                    | Token::Identifier(_)
            )
        );
        if !starts_a_value || compile_time.contains(name) {
            continue;
        }
        let end = declaration_value_end(tokens, index + 2);
        let mut sub = bare_parser(&tokens[index + 2..end]);
        let Ok(expression) = sub.parse_expression(Precedence::Lowest) else {
            continue;
        };
        // A constant that calls something asked for the call to be worked out,
        // so what stopped it is said here. One that does not is left out
        // rather than half-read, which is what it always was: it then means
        // what it always meant, and using it as a length is an error naming
        // the length rather than an array of the wrong size.
        let asked = holds_a_call(&sub.ast, expression);
        match folder.expression(&sub.ast, expression, &values) {
            Ok(value) => {
                values.insert(name.clone(), value);
            }
            // One that asked a type for its layout is set aside rather than
            // refused. It is worked out again once the types have been read,
            // which is where a layout has an answer, and refused there if it
            // still has none.
            Err(reason)
                if asked && !crate::const_eval::asks_a_measurement(&reason) =>
            {
                faults.push((index + 2, reason))
            }
            Err(_) => {}
        }
    }
    (values, folder, faults)
}

// What a constant asking to be worked out at compile time broke on, at the
// token it was written at. The scan runs before the parse, so these are held
// until the parse has somewhere to report them.
fn constant_faults(
    faults: &[(usize, String)],
    positions: &[Position],
) -> Vec<Diagnostic> {
    faults
        .iter()
        .map(|(at, reason)| Diagnostic {
            position: positions.get(*at).copied().unwrap_or_default(),
            message: reason.clone(),
            related: Vec::new(),
        })
        .collect()
}

/// A length that works out below zero. There is no array of that many elements,
/// and every place that reads a length says so in the same words.
fn negative_length(position: crate::lexer::Position) -> anyhow::Error {
    crate::diagnostic::LocatedError {
        position,
        message: "an array holds a number of elements that cannot be negative"
            .to_string(),
    }
    .into()
}

fn integers_among(
    values: &HashMap<String, crate::const_eval::Value>,
) -> HashMap<String, i64> {
    values
        .iter()
        .filter_map(|(name, value)| Some((name.clone(), value.integer()?)))
        .collect()
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens: tokens.iter(),
            all_tokens: tokens,
            ast: Ast::default(),
            linear_types: std::collections::HashSet::new(),
            tests: Vec::new(),
            exports: Vec::new(),
            positions: &[],
            consumed: 0,
            discards: 0,
            pending_angle_close: 0,
            diagnostics: Vec::new(),
            internal_types: false,
            no_struct_literal: false,
            loop_names: Vec::new(),
            integer_constants: HashMap::new(),
            constant_values: HashMap::new(),
            folder: crate::const_eval::Folder::new(
                &[],
                HashMap::new(),
                HashMap::new(),
            ),
            imported_bodies: HashMap::new(),
            settled: false,
            generic_types: scan_generic_types(tokens),
            lifted_lines: Vec::new(),
            block_depth: 0,
            bracket_depth: 0,
            runtime_names: false,
            compile_time_names: Vec::new(),
        }
    }

    // Work out every top-level constant, once the imported bodies a call may
    // read are in hand. Held until here rather than done where the parser is
    // made, since what a file imports is added after that and a constant may
    // call one of those.
    fn settle_constants(&mut self) {
        if self.settled {
            return;
        }
        self.settled = true;
        let imported = std::mem::take(&mut self.imported_bodies);
        let (values, folder, faults) =
            scan_constants(self.all_tokens, imported);
        self.integer_constants = integers_among(&values);
        self.constant_values = values;
        self.folder = folder;
        let mut held = constant_faults(&faults, self.positions);
        held.append(&mut self.diagnostics);
        self.diagnostics = held;
    }

    /// The functions the files this one imports export, as bodies a
    /// compile-time call may read.
    pub fn also_const_functions(
        &mut self,
        bodies: HashMap<String, std::rc::Rc<Vec<Token>>>,
    ) -> &mut Self {
        self.imported_bodies.extend(bodies);
        self
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
            all_tokens: tokens,
            ast,
            linear_types: std::collections::HashSet::new(),
            tests: Vec::new(),
            exports: Vec::new(),
            positions,
            consumed: 0,
            discards: 0,
            pending_angle_close: 0,
            diagnostics: Vec::new(),
            internal_types: false,
            no_struct_literal: false,
            loop_names: Vec::new(),
            integer_constants: HashMap::new(),
            constant_values: HashMap::new(),
            folder: crate::const_eval::Folder::new(
                &[],
                HashMap::new(),
                HashMap::new(),
            ),
            imported_bodies: HashMap::new(),
            settled: false,
            generic_types: scan_generic_types(tokens),
            lifted_lines: Vec::new(),
            block_depth: 0,
            bracket_depth: 0,
            runtime_names: false,
            compile_time_names: Vec::new(),
        }
    }

    // Generic types this file did not declare but may name, which is every one
    // declared by a file it imports. Which names can start a literal is settled
    // before the parse, and a file that imports `Ordering` writes
    // `Ordering<Point> { .. }` exactly as the file declaring it does.
    pub fn also_generic(&mut self, names: GenericDefaults) -> &mut Self {
        self.generic_types.extend(names);
        self
    }

    /// The lines a `when` kept, which now stand one level out from the braces
    /// they were written inside. Every statement of a block begins at the same
    /// column, and these begin deeper, so the rule that reads a deeper line as
    /// continuing the one above it is not asked about them.
    pub fn also_lifted_lines(
        &mut self,
        lines: Vec<(usize, usize)>,
    ) -> &mut Self {
        self.lifted_lines.extend(lines);
        self
    }

    fn was_lifted(&self, line: usize) -> bool {
        self.lifted_lines
            .iter()
            .any(|(from, to)| line >= *from && line <= *to)
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

    // Whether the token at the cursor ends where the one after it begins.
    //
    // `!` carries two meanings and the space after it is what tells them
    // apart: written against what follows it, it negates; written with a space
    // on both sides in a return type, it marks a failure set. So this is the
    // question the rule about it asks.
    // Answers nothing where there are no positions to read it off, which is a
    // parse of tokens alone, and a rule about spacing has nothing to say there.
    fn touches_next(&self) -> Option<bool> {
        let here = self.consumed;
        if here + 1 >= self.positions.len() {
            return None;
        }
        let held = self.positions[here];
        let next = self.positions[here + 1];
        Some(next.line == held.line && next.column == held.column + 1)
    }

    // The same of the token before it, whose width is what it is written as.
    // Every token that can end a return type is written the way it reads.
    fn touches_previous(&self) -> Option<bool> {
        let here = self.consumed;
        if here == 0 || here >= self.positions.len() {
            return None;
        }
        let previous = self.positions[here - 1];
        let width = self.all_tokens[here - 1].to_string().chars().count();
        Some(
            self.positions[here].line == previous.line
                && self.positions[here].column == previous.column + width,
        )
    }

    // The place of a token by its index, for a report about something a couple
    // of tokens ahead of where the read is.
    fn position_at(&self, index: u32) -> Option<Position> {
        self.positions.get(index as usize).copied()
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

    // The tokens between two marks, written back the way they were read. A join
    // takes a space either side and nothing else does, which is the spelling the
    // self-hosted compiler replays them with.
    fn tokens_written(&self, first: u32, last: u32) -> String {
        // `read_token` counts a read past the end as a read, so the mark it
        // answers with can be one past the last token there is. Nothing here
        // reaches that, and a slice taken with it would end the compile with a
        // panic rather than a report, which is not a trade worth leaving open.
        let last = (last as usize).min(self.all_tokens.len());
        let mut out = String::new();
        for token in &self.all_tokens[first as usize..last] {
            match token {
                Token::And | Token::Or => {
                    out.push(' ');
                    out.push_str(&token.to_string());
                    out.push(' ');
                }
                other => out.push_str(&other.to_string()),
            }
        }
        out
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
        self.settle_constants();
        let mut roots = Vec::new();
        loop {
            let position = self.current_position().unwrap_or_default();
            // Where the declaration began, so a fault inside it can be told
            // what it had left open.
            let began = self.tokens.clone();
            let before = self.tokens.len();
            match self.parse_statement() {
                Ok(Some(statement)) => {
                    roots.push(statement);
                }
                Ok(None) => break,
                Err(error) => {
                    self.record_error(position, &error);
                    let read = before - self.tokens.len();
                    self.synchronize(left_open(began, read));
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

    // A name-keeping definition is emitted under the name it was written under,
    // so two of them by one name are one symbol, and the runtime is linked into
    // every program. `frost_rt_` is what the runtime emits and `frost_u_` is
    // what the compiler's C backend names an ordinary function, so a program
    // defining either would replace something every program calls. Refused
    // where the declaration is read, in the words the self-hosted compiler also
    // says.
    fn refuse_reserved_name(&self, name: &str, start: u32) -> Result<()> {
        if self.runtime_names
            || !(name.starts_with("frost_rt_") || name.starts_with("frost_u_"))
        {
            return Ok(());
        }
        let position = self
            .positions
            .get(start as usize)
            .copied()
            .unwrap_or_default();
        Err(anyhow::Error::new(crate::diagnostic::LocatedError {
            position,
            message: format!(
                "'{name}' keeps the name it is written under, and 'frost_rt_' and 'frost_u_' are the runtime's and the compiler's own, so a definition here would replace what every program calls"
            ),
        }))
    }

    /// Say that this file is the runtime, so it may define names in the
    /// runtime's own name space. Set by the driver for the one file it resolved
    /// as the runtime and for nothing else.
    pub fn compiling_the_runtime(&mut self) {
        self.runtime_names = true;
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
    // Forward to just past the brace that closes the match the cursor is
    // inside. A refusal among the arms leaves it where no statement begins, and
    // recovery from there reads the match's own closing brace as the one that
    // ends the function: what the reader is then told is that the brace ending
    // the function is not a declaration, about a line with nothing wrong.
    fn skip_past_match(&mut self) {
        let mut depth = 1;
        while depth > 0 && !matches!(self.peek_nth(0), Token::EndOfFile) {
            match self.peek_nth(0) {
                Token::LeftBrace => depth += 1,
                Token::RightBrace => depth -= 1,
                _ => {}
            }
            self.read_token();
        }
    }

    // The same place, saying something else. A caller takes the position while
    // the token it is about is still ahead of the parser, and says what is
    // wrong with it once it has read it.
    fn reword(&self, held: anyhow::Error, message: &str) -> anyhow::Error {
        let position = held
            .downcast_ref::<crate::diagnostic::LocatedError>()
            .map(|located| located.position)
            .unwrap_or_default();
        anyhow::Error::new(crate::diagnostic::LocatedError {
            position,
            message: message.to_string(),
        })
    }

    fn here(&self, message: String) -> anyhow::Error {
        anyhow::Error::new(crate::diagnostic::LocatedError {
            position: self.current_position().unwrap_or_default(),
            message,
        })
    }

    // The first type in `name<T, U>(...)` written after a name that is not a
    // generic type, or nothing where the shape is something else.
    //
    // The type in `name<T>(...)` written after a name that is not a generic
    // type, or nothing where the shape is something else.
    //
    // A generic struct is written `Pair<i64>`, and the shape reads as the same
    // thing for a call, which is what other languages do with it. Frost writes a
    // call's compile-time argument among its arguments with a `$` on it, so this
    // is a comparison of a name against a type, and both compilers found that
    // out somewhere further on and said something about wherever they got to
    // rather than about what was written.
    //
    // One type and no comma. `f(a < b, c > (d))` is two arguments, each an
    // ordinary comparison, and reading the comma as a separator between type
    // arguments is what makes this shape ambiguous in the languages that do
    // take it. Without a comma there is nothing to lose: `(a < b) > (c)` weighs
    // a truth value against a number, which no program this would refuse could
    // have done anyway.
    fn angled_call_argument(&self) -> Option<String> {
        let mut ahead = 2usize;
        let mut named = 0usize;
        let mut written = String::new();
        loop {
            let held = self.peek_nth(ahead);
            match held {
                Token::Identifier(_) => {
                    if named == 1 {
                        return None;
                    }
                    named += 1;
                }
                Token::Caret
                | Token::LeftBracket
                | Token::RightBracket
                | Token::Integer(_) => {}
                Token::GreaterThan => {
                    return match self.peek_nth(ahead + 1) {
                        Token::LeftParentheses if named == 1 => Some(written),
                        _ => None,
                    };
                }
                _ => return None,
            }
            // The type as it was written, spelling by spelling. Read off the
            // tokens rather than the source so that what the two compilers say
            // does not depend on where the reader put a space.
            written.push_str(&held.to_string());
            ahead += 1;
        }
    }

    // The same, for a token the reader has looked ahead to and not consumed.
    fn at_ahead(&self, ahead: usize, message: String) -> anyhow::Error {
        let position = if self.positions.is_empty() {
            Position::default()
        } else {
            let index = (self.consumed + ahead).min(self.positions.len() - 1);
            self.positions[index]
        };
        anyhow::Error::new(crate::diagnostic::LocatedError {
            position,
            message,
        })
    }

    // The same, for a site that has already consumed the offending token.
    // A fault about something already read, shown where that thing began. The
    // block of values a type names is read entry by entry, so by the time one
    // is found wrong the cursor is past it and `here` would name whatever comes
    // next.
    fn at_mark(&self, start: u32, message: String) -> anyhow::Error {
        let position = self
            .positions
            .get(start as usize)
            .copied()
            .unwrap_or_default();
        anyhow::Error::new(crate::diagnostic::LocatedError {
            position,
            message,
        })
    }

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
    /// `open` is how many brackets the declaration had opened and not closed
    /// when it failed, which the walk has to come back out of before a name
    /// with a `::` after it means the next declaration. A bit refused inside a
    /// `flags` block left that block open, and the bit below it read as a
    /// declaration head, so one fault was reported as three.
    fn synchronize(&mut self, open: u32) {
        let mut depth = open;
        if !matches!(self.peek_nth(0), Token::EndOfFile) {
            depth = counted(depth, self.peek_nth(0));
            self.read_token();
        }
        while !matches!(self.peek_nth(0), Token::EndOfFile) {
            if self.at_statement_boundary()
                && (depth == 0 || self.at_left_margin())
            {
                return;
            }
            depth = counted(depth, self.peek_nth(0));
            self.read_token();
        }
    }

    /// Whether the token here is written flush left, which at the top level is
    /// what a declaration of its own looks like.
    ///
    /// A declaration that failed with a bracket still open is a declaration
    /// whose bracket is never closed, so counting down to nothing never
    /// happens and every declaration below it is walked over. `flags u8 { A ::
    /// 1` and `fn(a: i64 {` are the same count and different faults, and the
    /// margin is what the reader wrote to tell them apart: a bit inside a body
    /// is indented and the next declaration is not.
    fn at_left_margin(&self) -> bool {
        self.current_position().is_some_and(|held| held.column == 1)
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
    ///
    /// `open` is how many brackets the statement had opened and not closed when
    /// it failed, which the walk has to come back out of before a closing brace
    /// means the block's own. A `match` whose arm was refused failed inside its
    /// own brace, and that brace read as the end of the function the match was
    /// written in, so the line after it was reported as a declaration head that
    /// is not one.
    fn synchronize_in_block(&mut self, open: u32) {
        // A run, an argument list and a nested block are counted, so a
        // statement that failed with one of them still open is walked over
        // whole. Uncounted, a fault before `Bag<8> { data = [0; 8] }` stopped
        // recovery on that literal's own closing brace, the enclosing block
        // ended there, and everything after it read as a declaration: one fault
        // was told twice and the second time about a line that was fine.
        let mut depth: u32 = open;
        // At least one token is always consumed so recovery cannot loop, and it
        // counts like every other: a closer stepped over uncounted left the
        // walk holding a bracket that was already shut, and it ran to the end
        // of the file looking for what would close it.
        if !matches!(self.peek_nth(0), Token::EndOfFile)
            && (depth > 0 || !matches!(self.peek_nth(0), Token::RightBrace))
        {
            depth = counted(depth, self.peek_nth(0));
            self.read_token();
        }
        while !matches!(self.peek_nth(0), Token::EndOfFile) {
            if depth == 0
                && (matches!(self.peek_nth(0), Token::RightBrace)
                    || self.at_block_statement_boundary())
            {
                return;
            }
            depth = counted(depth, self.peek_nth(0));
            self.read_token();
        }
    }

    fn at_block_statement_boundary(&self) -> bool {
        match self.peek_nth(0) {
            Token::Mut
            | Token::Return
            | Token::Defer
            | Token::ErrDefer
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
                    | Token::ErrDefer
                    | Token::For
                    | Token::While
                    | Token::With
                    | Token::Break
                    | Token::Continue
                    | Token::Ref
                    | Token::Mut
                    | Token::If
                    | Token::Match
                    | Token::Unsafe
            )
        {
            self.refuse_ambient_state()?;
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
            // A `test` whose body does not open where one should says what is
            // in the way. Without this the statement was not read as a test at
            // all and the reader was told a declaration head was expected,
            // about the word `test` they had just written.
            Token::Identifier(name)
                if name == "test"
                    && matches!(self.peek_nth(1), Token::StringLiteral(_))
                    && matches!(self.peek_nth(2), Token::Uses) =>
            {
                // At the `uses`, which is what the reader takes away. The
                // word `test` is not what is wrong with the line.
                let drawn =
                    self.position_at(self.mark() + 2).unwrap_or_else(|| {
                        self.current_position().unwrap_or_default()
                    });
                return Err(anyhow::Error::new(
                    crate::diagnostic::LocatedError {
                        position: drawn,
                        message: "a `test` body is run by the test runner, which supplies nothing, so a test draws no capability".to_string(),
                    },
                ));
            }
            Token::Return => Some(self.parse_return_statement()?),
            Token::Defer | Token::ErrDefer => {
                Some(self.parse_defer_statement()?)
            }
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
            Token::Mut
                if matches!(self.peek_nth(1), Token::Identifier(_))
                    && matches!(self.peek_nth(2), Token::Comma) =>
            {
                Some(self.parse_multiple_declaration()?)
            }
            Token::Mut => Some(self.parse_mutable_declaration()?),
            // `var` is an ordinary name and opens nothing, so an older file
            // reads as a name followed by a binding. Named here rather than
            // left to whatever that parses as, which is a complaint about the
            // second word.
            Token::Identifier(name)
                if name == "var"
                    && matches!(self.peek_nth(1), Token::Identifier(_))
                    && matches!(
                        self.peek_nth(2),
                        Token::ColonAssign | Token::Colon | Token::Comma
                    ) =>
            {
                return Err(self.here("a local that is reassigned is declared with `mut`, the word a parameter that writes the caller's value carries".to_string()));
            }
            // A binding lives in a block. At the top level the same tokens
            // fall through to the arm below that names what may stand there,
            // which is what the self-hosted compiler also says.
            // A `_` opens the list the same way a name does, since a caller
            // may want the second value and none of the first.
            Token::Identifier(_) | Token::Underscore
                if self.block_depth > 0
                    && matches!(self.peek_nth(1), Token::Comma) =>
            {
                Some(self.parse_multiple_declaration()?)
            }
            Token::Identifier(_)
                if self.block_depth > 0
                    && matches!(self.peek_nth(1), Token::ColonAssign) =>
            {
                Some(self.parse_declaration(false)?)
            }
            // `_ := call()` takes an answer the caller has no use for. A list of
            // one is a list, so it reads the way the `_` in a longer one does,
            // and it is the only way to say an answer was meant to go unread.
            Token::Underscore
                if self.block_depth > 0
                    && matches!(self.peek_nth(1), Token::ColonAssign) =>
            {
                Some(self.parse_discard_declaration()?)
            }
            Token::Identifier(_)
                if self.block_depth > 0
                    && matches!(self.peek_nth(1), Token::Colon)
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
            // `InitFlags :: flags u32 { Video :: 32 }`. The word is not a
            // keyword, so the shape after it is what says this is a
            // declaration rather than an expression that starts with a name.
            Token::Identifier(_)
                if matches!(self.peek_nth(1), Token::DoubleColon)
                    && self.at_flags_declaration(2) =>
            {
                Some(self.parse_constant_or_struct_statement()?)
            }
            // `Tight :: packed struct { .. }`. The word is not a keyword
            // either, so what says this is a declaration is the `struct` that
            // has to follow it.
            Token::Identifier(_)
                if matches!(self.peek_nth(1), Token::DoubleColon)
                    && self.at_packed_struct(2) =>
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
            // A type is declared where every other declaration is. Inside a
            // body it reached the lowering as a statement nothing handles, and
            // the reader was told the statement was unsupported rather than
            // where the declaration belongs.
            Token::Identifier(name)
                if self.block_depth > 0
                    && matches!(self.peek_nth(1), Token::DoubleColon)
                    && (matches!(
                        self.peek_nth(2),
                        Token::Struct | Token::Enum | Token::Distinct
                    ) || self.at_packed_struct(2)
                        || self.at_flags_declaration(2)
                        || (matches!(self.peek_nth(2), Token::Linear)
                            && matches!(
                                self.peek_nth(3),
                                Token::Struct | Token::Enum
                            ))) =>
            {
                let name = name.clone();
                return Err(self.here(format!(
                    "a type is declared where a file's other declarations are, and '{name}' is declared inside a body"
                )));
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
                            | Token::LeftBracket
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
                    // A name with a `::` after it opens a declaration, so what
                    // is wrong is the word after the `::` rather than the name.
                    // Read as a token no declaration starts with, the report
                    // named the declaration's own name and the reader was sent
                    // to the word that is fine.
                    if matches!(self.peek_nth(0), Token::Identifier(_))
                        && matches!(self.peek_nth(1), Token::DoubleColon)
                    {
                        self.read_token();
                        self.read_token();
                        self.refuse_unknown_declaration_head()?;
                    }
                    self.refuse_ambient_state()?;
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

    /// `_ := expr`, which evaluates the expression and binds nothing anyone can
    /// name. The binding is real so what it holds is still owed a consumption
    /// where it is a resource; what the `_` says is only that no name reads it.
    fn parse_discard_declaration(&mut self) -> Result<StmtId> {
        let start = self.mark();
        self.read_token();
        self.read_token();
        let value = self.parse_expression(Precedence::Lowest)?;
        let name = format!("__discard{}", self.discards);
        self.discards += 1;
        let name = self.ast.intern(&name);
        Ok(self.ast.push_stmt(
            Statement::Let {
                name,
                type_annotation: None,
                type_at: crate::lexer::Position::default(),
                value,
                mutable: false,
            },
            self.span_from(start),
        ))
    }

    /// `align(N)` after a field's type: the field starts at a multiple of N.
    ///
    /// A struct's own alignment is the widest its fields ask for, so this is
    /// also how a struct is given one, and there is no second form saying the
    /// same thing about the whole declaration.
    // `for name in fields(T) { Type }` inside a struct body. It stands for one
    // field per field of T, so what is recorded is the loop's name, the type
    // the body wrote, and which parameter is walked. The instance is where the
    // three become fields, since only there is T a type with fields to read.
    fn parse_field_walk(&mut self, packed: bool) -> Result<StructField> {
        self.read_token();
        let at = self.current_position().unwrap_or_default();
        let name = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            _ => bail!("a `for` in a struct body names what it walks"),
        };
        if !matches!(self.read_token(), Token::In) {
            bail!("a `for` in a struct body names what it walks and then what it walks over");
        }
        let walked = match self.read_token() {
            Token::Identifier(word) if word == "fields" => {
                if !matches!(self.read_token(), Token::LeftParentheses) {
                    bail!("`fields` reads the type it walks in parentheses");
                }
                let walked = match self.read_token() {
                    Token::Identifier(name) => name.to_string(),
                    _ => bail!(
                        "a `for` in a struct body walks the fields of a type parameter"
                    ),
                };
                if !matches!(self.read_token(), Token::RightParentheses) {
                    bail!("`fields` reads one type");
                }
                walked
            }
            _ => bail!(
                "a `for` in a struct body walks `fields(T)`, which is the one list a declaration has"
            ),
        };
        if !matches!(self.read_token(), Token::LeftBrace) {
            bail!("a `for` in a struct body writes one type in braces");
        }
        let field_type = self.parse_type()?;
        let align = self.parse_field_alignment(packed)?;
        if !matches!(self.read_token(), Token::RightBrace) {
            bail!("a `for` in a struct body writes one type in braces");
        }
        let name = self.ast.intern(&name);
        Ok(StructField {
            name,
            field_type,
            align,
            at,
            walk_over: Some(walked),
        })
    }

    fn parse_field_alignment(&mut self, packed: bool) -> Result<Option<usize>> {
        if !self.at_field_alignment() {
            return Ok(None);
        }
        self.read_token();
        if !matches!(self.read_token(), Token::LeftParentheses) {
            bail!("Expected '(' after `align`");
        }
        let Token::Integer(value) = self.read_token().clone() else {
            return Err(self.at_consumed(
                "`align` takes a number, which is what a field starts at a multiple of"
                    .to_string(),
            ));
        };
        if value <= 0 || (value & (value - 1)) != 0 {
            return Err(self.at_consumed(format!(
                "`align` takes a power of two, and {value} is not one; an address is a multiple of a power of two or of nothing"
            )));
        }
        // `packed` says no field is padded and `align` says this one is, so a
        // declaration writing both says two things that cannot both hold.
        if packed {
            return Err(self.at_consumed(
                "a `packed struct` pads no field, and `align` asks for this one to be padded; drop one of the two"
                    .to_string(),
            ));
        }
        if !matches!(self.read_token(), Token::RightParentheses) {
            bail!("Expected ')' after the alignment");
        }
        Ok(Some(value as usize))
    }

    fn parse_defer_statement(&mut self) -> Result<StmtId> {
        let start = self.mark();
        let on_failure = matches!(self.peek_nth(0), Token::ErrDefer);
        let word = if on_failure { "errdefer" } else { "defer" };
        self.read_token();
        // A `defer` runs where the function leaves, so one that leaves the
        // function is read again at the exit it makes, which makes another
        // exit. Both compilers ran out of stack on it rather than saying so.
        let at_return = self.here(String::new());
        if matches!(self.peek_nth(0), Token::Return) {
            return Err(self.reword(
                at_return,
                "a `defer` runs where the function leaves, and a `return` inside one would leave it again",
            ));
        }
        let statement = self.parse_statement()?.ok_or_else(|| {
            anyhow::anyhow!("Expected statement after {word}")
        })?;
        let held = if on_failure {
            Statement::ErrDefer(statement)
        } else {
            Statement::Defer(statement)
        };
        Ok(self.ast.push_stmt(held, self.span_from(start)))
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

        // What the loop names is bound by the loop and not by the reader, so
        // it is held while the body is read and dropped after it.
        let held = self.loop_names.len();
        self.loop_names.push(iterator);
        if let Some(second) = second {
            self.loop_names.push(second);
        }
        let body = self.parse_block();
        self.loop_names.truncate(held);
        let body = body?;

        Ok(self.ast.push_stmt(
            Statement::For(iterator, second, range, body),
            self.span_from(start),
        ))
    }

    /// A write to what a `for` names.
    ///
    /// The element binds the way a parameter of its type would, so a scalar is
    /// a copy of what the container holds and writing to it changes nothing.
    /// Both compilers took the write and emitted it, and the container came out
    /// of the loop as it went in. What the reader meant is a write to the
    /// element, which is the container and the index.
    fn refuse_write_to_a_loop_name(&mut self, place: ExprId) -> Result<()> {
        let Expression::Identifier(name) = self.ast.expr(place) else {
            return Ok(());
        };
        if !self.loop_names.contains(name) {
            return Ok(());
        }
        let written = self.ast.name(*name).to_string();
        Err(anyhow::Error::new(crate::diagnostic::LocatedError {
            position: self.ast.position_of(self.ast.expr_span(place)),
            message: format!(
                "'{written}' is what the loop names, and writing to it changes nothing: the element binds the way a parameter of its type would. Write to the container at the index, or bind what you want to keep before the loop"
            ),
        }))
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

    // `quotient, remainder := divide(a, b)`, and `mut` in front of any name
    // that the body goes on to write. The list is names, not patterns, since
    // what it takes apart is a return type list rather than a value.
    fn parse_multiple_declaration(&mut self) -> Result<StmtId> {
        let start = self.mark();
        let mut bindings = Vec::new();
        loop {
            if matches!(self.peek_nth(0), Token::Identifier(name) if name == "var")
                && matches!(self.peek_nth(1), Token::Identifier(_))
            {
                return Err(self.here("a local that is reassigned is declared with `mut`, the word a parameter that writes the caller's value carries".to_string()));
            }
            // Where the `mut` is, since it is the word the reader takes away.
            let mut_at = self.current_position().unwrap_or_default();
            let mutable = if matches!(self.peek_nth(0), Token::Mut) {
                self.read_token();
                true
            } else {
                false
            };
            // `_` takes a value the caller has no use for. The list binds one
            // name per value the call answers with, so without it a caller
            // wanting only the first has to invent a name for the rest, and
            // that name is a live binding somebody can read by mistake. It
            // stands here as the wildcard token rather than as a name, so the
            // lowering below gives it storage nothing can reach and any number
            // of them may sit in one list.
            let name = match self.read_token() {
                Token::Identifier(name) => name.to_string(),
                Token::Underscore => String::from("_"),
                other => bail!("Expected a name to bind, found {other}"),
            };
            if name == "_" {
                if mutable {
                    return Err(anyhow::Error::new(
                        crate::diagnostic::LocatedError {
                            position: mut_at,
                            message: "`mut` makes a binding assignable and `_` binds nothing; write `_` on its own".to_string(),
                        },
                    ));
                }
            } else {
                self.refuse_literal_name(&name)?;
            }
            let name = self.ast.intern(&name);
            bindings.push(MultiBinding {
                name,
                mutable,
                at: mut_at,
            });
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
            bail!("Expected ':=' or ': type =' after 'mut identifier'")
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
                type_at: crate::lexer::Position::default(),
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
        // At the token that is wrong, taken before it is read. Read afterwards,
        // the position is whatever the parser reached next, which for `ref 7`
        // was the line below the one holding it.
        let at_name = self.here(String::new());
        let name = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            _ => {
                return Err(self.reword(
                    at_name,
                    "a name goes after `ref`, and this is not one",
                ));
            }
        };
        self.refuse_literal_name(&name)?;
        let at_bind = self.here(String::new());
        if !matches!(self.read_token(), Token::ColonAssign) {
            return Err(
                self.reword(at_bind, "`:=` goes after the name a `ref` binds")
            );
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
                type_at: crate::lexer::Position::default(),
                value,
                mutable: false,
            },
            span,
        ))
    }

    fn parse_typed_declaration(&mut self, mutable: bool) -> Result<StmtId> {
        let start = self.mark();
        // The binding's own place, taken before anything is read, since a
        // report about it points at the name the reader wrote.
        let at_binding = self.here(String::new());
        let name = match self.read_token() {
            Token::Identifier(name) => name.to_string(),
            _ => bail!("Expected identifier"),
        };
        self.refuse_literal_name(&name)?;

        if !matches!(self.read_token(), Token::Colon) {
            bail!("Expected ':'");
        }

        // Where the type is written, which is where a report about it points.
        let type_at = self.current_position().unwrap_or_default();
        let type_annotation = Some(self.parse_type()?);

        if !matches!(self.read_token(), Token::Assign) {
            bail!("Expected '=' after type annotation");
        }

        let value = self.parse_expression(Precedence::Lowest)?;

        if matches!(self.peek_nth(0), Token::Semicolon) {
            self.read_token();
        }

        // A literal of a different length from the one the binding declares.
        // The literal takes its type from the annotation, so by the time
        // anything compared the two they agreed, and the elements the literal
        // left out were storage nothing wrote.
        if let Some(crate::types::Type::Array(element, wanted)) =
            &type_annotation
            && let Expression::Literal(crate::ast::Literal::Array(elements)) =
                self.ast.expr(value)
        {
            let written = self.ast.exprs_in(*elements).len();
            if written != *wanted {
                return Err(self.reword(
                    at_binding,
                    &format!(
                        "this binding is a '[{wanted}]{element}' and the value is a '[{written}]{element}'"
                    ),
                ));
            }
        }

        let name = self.ast.intern(&name);
        Ok(self.ast.push_stmt(
            Statement::Let {
                name,
                type_annotation,
                type_at,
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

        // `Type::NAME :: value` written at the top level. A value under a type
        // is declared in that type's own block, and read from out here this is
        // the type's own name declared a second time, which is what the reader
        // was told before.
        if matches!(self.peek_nth(0), Token::Identifier(_))
            && matches!(self.peek_nth(1), Token::DoubleColon)
        {
            let message = "a value under a type is declared in that type's own block, as in `Name :: struct { .. } { VALUE :: .. }`".to_string();
            // At the type's own name, which is the word the whole declaration
            // moves under, rather than at the value's.
            return Err(match self.position_at(start) {
                Some(position) => {
                    anyhow::Error::new(crate::diagnostic::LocatedError {
                        position,
                        message,
                    })
                }
                None => anyhow::anyhow!("{message}"),
            });
        }

        if matches!(self.peek_nth(0), Token::Linear) {
            self.read_token();
            self.linear_types.insert(identifier.clone());
        }

        // `packed struct` reads beside `linear struct`: a word before the
        // keyword, which is what every other marker on a declaration is.
        let packed = self.at_packed_struct(0);
        if packed {
            self.read_token();
        }

        if matches!(self.peek_nth(0), Token::Struct) {
            self.read_token();
            if packed {
                let symbol = self.ast.intern(&identifier);
                self.ast.packed_structs.push(symbol);
            }
            let type_params = self.parse_generic_params()?;
            self.open_compile_time_frame(&type_params);
            if !matches!(self.read_token(), Token::LeftBrace) {
                bail!("Expected '{{' after struct");
            }
            let mut fields = Vec::new();
            while self.peek_nth(0) != &Token::RightBrace {
                // `for name in fields(T) { Type }` writes one field per field
                // of T, each keeping the name its own declaration gave it. The
                // loop's name stands for what the walked field holds, so the
                // type after it is written the way `sizeof(name)` reads one.
                if matches!(self.peek_nth(0), Token::For) {
                    fields.push(self.parse_field_walk(packed)?);
                    if matches!(self.peek_nth(0), Token::Comma) {
                        self.read_token();
                    }
                    continue;
                }
                {
                    let field_name = self.read_field_name("a field name")?;
                    if !matches!(self.read_token(), Token::Colon) {
                        bail!("Expected ':' after field name");
                    }
                    self.refuse_field_sigil(&field_name)?;
                    let field_at = self.current_position().unwrap_or_default();
                    let field_type = self.parse_type()?;
                    let field_align = self.parse_field_alignment(packed)?;
                    let field_name = self.ast.intern(&field_name);
                    fields.push(StructField {
                        name: field_name,
                        field_type,
                        align: field_align,
                        at: field_at,
                        walk_over: None,
                    });
                }
                if matches!(self.peek_nth(0), Token::Comma) {
                    self.read_token();
                }
            }
            self.read_token();
            // A field is reached by its name, so two under one name are two
            // things a reader cannot tell apart and one of them is
            // unreachable.
            let mut seen: std::collections::HashSet<crate::ast::Symbol> =
                std::collections::HashSet::new();
            for field in &fields {
                if !seen.insert(field.name) {
                    let named = self.ast.name(field.name).to_string();
                    return Err(self.at_mark(
                        start,
                        format!(
                            "'{named}' is declared twice here, and a field is reached by its name"
                        ),
                    ));
                }
            }
            self.parse_values_under(&identifier, &[], !type_params.is_empty())?;
            if matches!(self.peek_nth(0), Token::Semicolon) {
                self.read_token();
            }
            let name = self.ast.intern(&identifier);
            let type_params = self.intern_all(&type_params);
            let fields = self.ast.add_struct_fields(fields);
            self.compile_time_names.pop();
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
            self.open_compile_time_frame(&type_params);
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
                        self.refuse_field_sigil(&field_name)?;
                        let field_at =
                            self.current_position().unwrap_or_default();
                        let field_type = self.parse_type()?;
                        let field_name = self.ast.intern(&field_name);
                        variant_fields.push(StructField {
                            name: field_name,
                            field_type,
                            // A variant's payload is laid out by the enum, not
                            // by a declaration, so there is nothing to state.
                            align: None,
                            at: field_at,
                            walk_over: None,
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
            let variant_names: Vec<String> = variants
                .iter()
                .map(|variant| self.ast.name(variant.name).to_string())
                .collect();
            self.parse_values_under(
                &identifier,
                &variant_names,
                !type_params.is_empty(),
            )?;
            if matches!(self.peek_nth(0), Token::Semicolon) {
                self.read_token();
            }
            let name = self.ast.intern(&identifier);
            let type_params = self.intern_all(&type_params);
            let variants = self.ast.add_enum_variants(&variants);
            self.compile_time_names.pop();
            Ok(self.ast.push_stmt(
                Statement::Enum(name, type_params, variants),
                self.span_from(start),
            ))
        } else if self.at_flags_declaration(0) {
            self.read_token();
            let written_repr = self.mark();
            let repr = self.parse_type()?;
            if !repr.is_integer() {
                // The block is read past before the fault is raised, so what
                // follows it is where reading resumes. Left in place, the
                // recovery took the block's closing brace for a declaration
                // head and said so, and one mistake was reported as two.
                self.skip_braced_block();
                return Err(self.at_mark(
                    written_repr,
                    format!("'{identifier}' is a set of bits, so it is written over an integer type; '{repr}' is not one"),
                ));
            }
            self.read_token();
            let mut bits: Vec<FlagBit> = Vec::new();
            let mut written: Vec<(String, i64)> = Vec::new();
            // What a bit takes when it states no number. The first is 1 and
            // each bit that is one bit doubles it, so a run of bare names is
            // 1, 2, 4, 8. A number that is not one bit (`None :: 0`, an
            // `All :: 15` closing a set) leaves it where it stood, which is
            // what lets those open and close a block of bare names.
            let mut counter: i64 = 1;
            let width = (repr.size_of() * 8) as u32;
            // A bit is declared the way every value named under a type is,
            // with `::`, and separated from the next by the line it is on.
            // What a bit may hold is the difference between the two blocks:
            // a number a C header fixed, rather than an expression, and a bit
            // may hold nothing at all and take the number its place gives it.
            while self.peek_nth(0) != &Token::RightBrace {
                if matches!(self.peek_nth(0), Token::EndOfFile) {
                    return Err(self.at_mark(
                        start,
                        format!("the bits '{identifier}' names are written inside braces, and this block is not closed"),
                    ));
                }
                // A bare name has no `::` to end it, so where one bit stops
                // and the next starts is the line, which is what separates one
                // declaration from the next everywhere else in the language.
                if !bits.is_empty() && self.on_the_same_line() {
                    return Err(self.at_mark(
                        self.mark(),
                        "a set of bits names each of them on a line of its own, and this one follows another".to_string(),
                    ));
                }
                let entry = self.mark();
                let name = match self.read_token() {
                    Token::Identifier(name) => name.to_string(),
                    _ => {
                        return Err(self.at_mark(
                            entry,
                            "a set of bits names each of them with a name, and this is not one".to_string(),
                        ));
                    }
                };
                let value = if matches!(self.peek_nth(0), Token::DoubleColon)
                    && self.on_the_same_line()
                {
                    self.read_token();
                    match self.read_token() {
                        Token::Integer(value) => *value,
                        _ => {
                            return Err(self.at_mark(
                                entry,
                                format!("a bit of '{identifier}' is a number a C header wrote down, and this is not one"),
                            ));
                        }
                    }
                } else if self.on_the_same_line()
                    && !matches!(
                        self.peek_nth(0),
                        Token::RightBrace | Token::Identifier(_)
                    )
                {
                    return Err(self.at_mark(
                        entry,
                        format!("a bit of a set is declared the way every value named under a type is, so '{name}' is written as '{name} :: 32'"),
                    ));
                } else {
                    counter
                };
                if bit_ordinal(value) > width {
                    let asked = ordinal(bit_ordinal(value));
                    return Err(self.at_mark(
                        entry,
                        format!("a set over '{repr}' holds {width} bits, and '{name}' is the {asked}"),
                    ));
                }
                if let Some((held, _)) =
                    written.iter().find(|(_, held)| *held == value)
                {
                    return Err(self.at_mark(
                        entry,
                        format!("each bit of a set holds a number of its own, and '{name}' holds the same one as '{held}'"),
                    ));
                }
                if value > 0 && value.count_ones() == 1 {
                    counter = value * 2;
                }
                written.push((name.clone(), value));
                let name = self.ast.intern(&name);
                bits.push(FlagBit { name, value });
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
            self.parse_values_under(&identifier, &[], false)?;
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
                // Where the type is written, for a report about the type.
                let type_at = self.current_position().unwrap_or_default();
                let param_type = self.parse_type()?;
                let param_name = self.ast.intern(&param_name);
                params.push(Parameter {
                    at: type_at,
                    name: param_name,
                    type_annotation: Some(param_type),
                    mutable: false,
                    mode,
                    compile_time_signature: None,
                    compile_time_default: None,
                    pack: false,
                    format: false,
                    capability: false,
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
            // A body makes this a definition rather than a declaration: the
            // function is written here and keeps the name it was written under,
            // where an ordinary one is emitted under a name of the compiler's
            // choosing. That is what lets Frost supply a symbol C already calls
            // by name, which is what the runtime is.
            if matches!(self.peek_nth(0), Token::LeftBrace) {
                self.refuse_reserved_name(&identifier, start)?;
                let name = self.ast.intern(&identifier);
                let block = self.parse_block()?;
                let params = self.ast.add_parameters(params);
                let signature = self.ast.push_signature(ReturnSignature {
                    kind: match return_type {
                        Some(ty) => ReturnKind::Single(ty),
                        None => ReturnKind::None,
                    },
                    uses: Vec::new(),
                    bound: None,
                    bound_text: String::new(),
                    bound_message: None,
                    at: Default::default(),
                });
                let body = self.ast.push_expr(
                    Expression::Proc(params, signature, block),
                    self.span_from(start),
                );
                self.ast.note_exported_symbol(name);
                return Ok(self.ast.push_stmt(
                    Statement::Constant(name, body),
                    self.span_from(start),
                ));
            }
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
            // A declaration is a value or one of the heads above, and a word
            // that opens neither is neither. Read as an expression it went to
            // whatever `while` or `ref` parses as, failed somewhere inside,
            // and recovery named the declaration's own name as the token no
            // declaration starts with.
            self.refuse_unknown_declaration_head()?;
            let mut expression = self.parse_expression(Precedence::Lowest)?;
            if matches!(self.peek_nth(0), Token::Semicolon) {
                self.read_token();
            }
            // A constant that calls something was worked out before the parse,
            // and what it answered stands here in place of the call. Left as
            // written it would be a call the program makes while it runs,
            // which is a second meaning for one declaration.
            if holds_a_call(&self.ast, expression)
                && let Some(value) = self.constant_values.get(&identifier)
            {
                let value = value.clone();
                let span = self.ast.expr_span(expression);
                expression = self.written_back(&value, span);
            }
            let name = self.ast.intern(&identifier);
            Ok(self.ast.push_stmt(
                Statement::Constant(name, expression),
                self.span_from(start),
            ))
        }
    }

    /// What follows `::` where it opens neither a value nor a declaration head.
    ///
    /// Named where it is written. A declaration's value is an expression, so
    /// what may stand there is what may open one; anything else is a token the
    /// reader has to replace, and read as an expression it went to whatever
    /// `while` or `ref` parses as and failed somewhere inside.
    /// A binding written at the top level, which is a reader reaching for state
    /// every function can see. There are none and there will be none, so the
    /// three shapes that carry state instead are named where the reach happens.
    fn refuse_ambient_state(&mut self) -> Result<()> {
        if self.block_depth > 0 || !matches!(self.peek_nth(0), Token::Mut) {
            return Ok(());
        }
        Err(self.here(
            "a binding at the top level would be state every function can see, and there are none. What holds state is a `mut` parameter, a value the caller owns and passes down, or `uses` on the function and a `with` block around the call"
                .to_string(),
        ))
    }

    fn refuse_unknown_declaration_head(&mut self) -> Result<()> {
        if Self::can_begin_expression(self.peek_nth(0)) {
            return Ok(());
        }
        let written = self.peek_nth(0).to_string();
        let message = format!(
            "a declaration is a value or one of `fn`, `struct`, `enum`, `distinct`, `flags`, `extern`, and this is '{written}'"
        );
        // At the token, which is the word the reader replaces, rather than at
        // the `::` the read has just been over.
        Err(match self.current_position() {
            Some(position) => {
                anyhow::Error::new(crate::diagnostic::LocatedError {
                    position,
                    message,
                })
            }
            None => anyhow::anyhow!("{message}"),
        })
    }

    /// The name of a type's answer standing where the cursor is: one of the
    /// names the layout pass settles, with the parentheses that make it a call.
    fn at_layout_answer(&self) -> Option<String> {
        let Token::Identifier(word) = self.peek_nth(0) else {
            return None;
        };
        if crate::const_eval::LAYOUT_ANSWERS.contains(&word.as_str())
            && matches!(self.peek_nth(1), Token::LeftParentheses)
        {
            return Some(word.clone());
        }
        None
    }

    /// A field names a type parameter by writing its name. The `$` is what
    /// declares one, and the declaration stands in the parameter list, so a
    /// sigil here is a second spelling for a name already written. It was
    /// accepted and read as the parameter, which left the two compilers on
    /// opposite sides of one declaration.
    fn refuse_field_sigil(&mut self, field: &str) -> Result<()> {
        // The whole of the type, not only what opens it. `f: fn($T) -> $T`
        // carries the sigil twice inside a function type, and it is the same
        // second spelling there as it is on a bare name.
        //
        // The comma between two fields is optional, so what ends this field is
        // the start of the next one as much as it is a comma or the brace. A
        // type never holds a name followed by a colon: a function type writes
        // what it takes and not what those are called.
        let mut ahead = 0usize;
        let mut depth = 0usize;
        loop {
            match self.peek_nth(ahead) {
                Token::Dollar => {
                    return Err(self.at_ahead(
                        ahead,
                        format!(
                            "a field names a type parameter by its name, and '$' is what declares one, so '{field}' is written without it"
                        ),
                    ));
                }
                Token::LeftParentheses
                | Token::LeftBracket
                | Token::LessThan => depth += 1,
                Token::RightParentheses
                | Token::RightBracket
                | Token::GreaterThan => depth = depth.saturating_sub(1),
                // `Vec<Vec<i64>>` closes two at once: the lexer reads `>>` as
                // the shift it also spells.
                Token::ShiftRight => depth = depth.saturating_sub(2),
                Token::Identifier(_)
                    if depth == 0
                        && ahead > 0
                        && matches!(self.peek_nth(ahead + 1), Token::Colon) =>
                {
                    return Ok(());
                }
                Token::Comma | Token::RightBrace if depth == 0 => {
                    return Ok(());
                }
                Token::EndOfFile => return Ok(()),
                _ => {}
            }
            ahead += 1;
        }
    }

    /// What a worked-out value is written as where the constant naming it
    /// stands. A whole number and a yes or no are the literals they read as; a
    /// run of bytes, a run of values and a set of named ones are written the
    /// way a program writes each, so the answer reaches every use as though it
    /// had been written there.
    fn written_back(
        &mut self,
        value: &crate::const_eval::Value,
        span: TokenSpan,
    ) -> ExprId {
        let expression = match value {
            crate::const_eval::Value::Integer(held) => {
                Expression::Literal(Literal::Integer(*held))
            }
            crate::const_eval::Value::Boolean(held) => {
                Expression::Literal(Literal::Boolean(*held))
            }
            crate::const_eval::Value::Text(held) => {
                Expression::Literal(Literal::String(held.to_string()))
            }
            crate::const_eval::Value::Array(items) => {
                let elements: Vec<ExprId> = items
                    .iter()
                    .map(|item| self.written_back(item, span))
                    .collect();
                Expression::Literal(Literal::Array(
                    self.ast.add_expr_list(&elements),
                ))
            }
            crate::const_eval::Value::Record(name, fields) => {
                let initializers: Vec<NamedExpr> = fields
                    .iter()
                    .map(|(field, held)| {
                        let value = self.written_back(held, span);
                        NamedExpr {
                            name: self.ast.intern(field),
                            value,
                            at: Default::default(),
                        }
                    })
                    .collect();
                let name = self.ast.intern(name);
                Expression::StructInit(
                    name,
                    self.ast.add_named_exprs(&initializers),
                )
            }
        };
        self.ast.push_expr(expression, span)
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
            {
                self.refuse_write_to_a_loop_name(expression)?;
                Statement::Assignment(expression, rhs)
            }
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
            // `sizeof(T)`, `alignof(T)`, `typename(T)` and `type_id(T)` read as
            // calls and take a type, so they are recognized here rather than
            // left to the ordinary call path, which would have to parse a type
            // as an expression. What comes out is the same `Call` every other
            // builtin is: the type rides along as an argument, so no pass has a
            // node form to enumerate for these.
            Token::Identifier(word)
                if matches!(
                    word.as_str(),
                    "sizeof"
                        | "alignof"
                        | "typename"
                        | "type_id"
                        | "field_count"
                ) && matches!(self.peek_nth(1), Token::LeftParentheses) =>
            {
                let word = word.clone();
                self.read_token();
                self.read_token();
                // The two that measure a type keep a leading `$`, so
                // `sizeof($P)` and `alignof($P)` read the constant a call was
                // given rather than the type its name would otherwise be.
                if !matches!(word.as_str(), "sizeof" | "alignof")
                    && matches!(self.peek_nth(0), Token::Dollar)
                {
                    self.read_token();
                }
                // The type's own place, not the call's. A report about a name
                // nothing declares says it where the reader wrote it, and this
                // span reaching back to `sizeof` put the caret on the word that
                // is fine.
                // The name, not the sigil. `sizeof` and `alignof` leave the
                // `$` for the type parse to read, and a report about the type
                // belongs on the word the reader has to change.
                let at_type = match self.peek_nth(0) {
                    Token::Dollar => self.mark() + 1,
                    _ => self.mark(),
                };
                let held = self.parse_type()?;
                if !matches!(self.read_token(), Token::RightParentheses) {
                    bail!("Expected ')' after the type in {word}");
                }
                let span = self.span_from(start);
                let argument = self.ast.push_expr(
                    Expression::TypeValue(held),
                    self.span_from(at_type),
                );
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
                    && !self.generic_types.contains_key(&identifier)
                    && let Some(written) = self.angled_call_argument()
                {
                    return Err(self.at_ahead(
                        1,
                        format!(
                            "a call writes a compile-time argument among its arguments, so this is written '{identifier}(${written}, ...)'"
                        ),
                    ));
                }
                if matches!(self.peek_nth(1), Token::LessThan)
                    && self.generic_types.contains_key(&identifier)
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
            // `$if` is the branch the compiler answers, and the `$` is what
            // says so. Read before the arm below, which takes a `$` as the
            // mark on a type argument and would read `if` as a type name.
            Token::Dollar if matches!(self.peek_nth(1), Token::If) => {
                self.parse_if_expression_marked(true)?
            }
            Token::Dollar => {
                self.read_token();
                // `$fn(a: i64) -> i64 { a + 1 }` names a function the same way
                // `$double` does, and the function is written where it is
                // named. A body is what tells it from the type `fn(i64) ->
                // i64`, which carries none.
                if matches!(self.peek_nth(0), Token::Function)
                    && self.function_literal_follows()
                {
                    self.parse_function_literal()?
                } else {
                    let held = self.parse_type()?;
                    self.ast.push_expr(
                        Expression::TypeValue(held),
                        self.span_from(start),
                    )
                }
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
                let start = self.mark();
                let empty = self.ast.intern("");
                self.parse_struct_init(empty, start)?
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
                // A line break ends a statement, so an operator that opens a
                // line has nothing above it to join to. Inside a bracket a
                // break says nothing and the expression runs on.
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
                | Token::ShiftRight
                    if !self.on_the_same_line() && self.bracket_depth == 0 =>
                {
                    break;
                }
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
                        let start = self.ast.expr_span(expression).first;
                        expression = self.parse_struct_init(name, start)?;
                    } else {
                        return Ok(expression);
                    }
                }
                Token::DoubleColon => {
                    if let Expression::Identifier(enum_name) =
                        self.ast.expr(expression)
                    {
                        let enum_name = *enum_name;
                        let named = self.ast.name(enum_name).to_string();
                        self.read_token();
                        let variant_name = match self.read_token() {
                            Token::Identifier(v) => v.to_string(),
                            _ => bail!("Expected identifier after '::'"),
                        };
                        // `Held<i64>::None` writes the arguments on the value.
                        // They belong to the type the value is bound to, and
                        // the value names the enum alone. Read as a name of its
                        // own, it resolved against whatever instance a function
                        // elsewhere had already made, so the same spelling was
                        // taken in one program and refused in another.
                        if let Some(cut) = named.find('<') {
                            let position = self
                                .positions
                                .get(self.ast.expr_span(expression).first
                                    as usize)
                                .copied()
                                .unwrap_or_default();
                            return Err(anyhow::Error::new(
                                crate::diagnostic::LocatedError {
                                    position,
                                    message: format!(
                                        "a generic enum's arguments go on the type a value is bound to, so this is written '{}::{variant_name}' with the binding typed '{named}'",
                                        &named[..cut]
                                    ),
                                },
                            ));
                        }
                        let variant_name = self.ast.intern(&variant_name);
                        if matches!(self.peek_nth(0), Token::LeftBrace) {
                            self.read_token();
                            let mut fields = Vec::new();
                            while self.peek_nth(0) != &Token::RightBrace {
                                let field_name = self.read_field_name(
                                    "a field name in an enum variant literal",
                                )?;
                                let written = self.peek_nth(0).to_string();
                                if !matches!(self.read_token(), Token::Assign) {
                                    return Err(self.at_consumed(format!(
                                        "a field in a literal is given its value with '=', and this writes '{written}'"
                                    )));
                                }
                                let value =
                                    self.parse_expression(Precedence::Lowest)?;
                                let name = self.ast.intern(&field_name);
                                fields.push(NamedExpr {
                                    name,
                                    value,
                                    at: Default::default(),
                                });
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
            let start = self.mark();
            let empty = self.ast.intern("");
            let literal = self.parse_struct_init(empty, start)?;
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
        // Recorded rather than raised. The expression reads one way whichever
        // spacing it was written with, so the parse carries on and the reader
        // is told the one thing that is wrong.
        if operator == Operator::Not && self.touches_next() == Some(false) {
            let position = self.current_position().unwrap_or_default();
            self.diagnostics.push(crate::diagnostic::Diagnostic::new(
                position,
                NEGATION_SPACING.to_string(),
            ));
        }
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
                    let held = self.integer_constants[name];
                    let Ok(held) = usize::try_from(held) else {
                        return Err(negative_length(
                            self.position_at(self.mark() - 1)
                                .unwrap_or_default(),
                        ));
                    };
                    held
                }
                // A name that is not a constant is a generic's value
                // parameter, whose number arrives with the instantiation. The
                // literal is carried unexpanded until then.
                Token::Identifier(name) => {
                    if !self.names_a_compile_time_number(name) {
                        return Err(self.refuse_a_length_naming_nothing(name));
                    }
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
        let at = self.current_position().unwrap_or_default();
        let index_expression = self.parse_expression(Precedence::Lowest)?;
        // `xs[1..5]` names a run rather than an element. Carried through, it
        // reached the walk that lowers expressions and was refused there as a
        // shape that walk has no arm for, which named the range and said
        // nothing about it.
        if matches!(self.ast.expr(index_expression), Expression::Range(..)) {
            return Err(crate::diagnostic::LocatedError {
                position: at,
                message: "an index names one element, and this is a range; `slice_range` takes part of a run"
                    .to_string(),
            }
            .into());
        }
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

    /// The arguments an instance was written with, followed by what the
    /// declaration says the ones left out stand for. Both spellings name one
    /// type, so this runs wherever an instance name is formed rather than
    /// leaving two names for the same thing.
    fn with_defaults(&self, base: &str, mut arguments: Vec<Type>) -> Vec<Type> {
        let Some(declared) = self.generic_types.get(base) else {
            return arguments;
        };
        for default in declared.iter().skip(arguments.len()) {
            let Some(default) = default else {
                break;
            };
            arguments.push(default.clone());
        }
        arguments
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
        let rendered: Vec<String> = self
            .with_defaults(base, arguments)
            .iter()
            .map(|argument| argument.to_string())
            .collect();
        Ok(format!("{base}<{}>", rendered.join(", ")))
    }

    // `start` is the token the whole literal begins at, which is the type's
    // name where one is written. This function is entered at the `{`, so the
    // name is already read and only the caller still knows where it was: a
    // complaint about `Absent { a = 1 }` names `Absent`, and a span beginning
    // at the brace sent the reader past the word that is wrong.
    fn parse_struct_init(
        &mut self,
        struct_name: Symbol,
        start: u32,
    ) -> Result<ExprId> {
        self.read_token();
        let mut fields = Vec::new();
        while self.peek_nth(0) != &Token::RightBrace {
            let at = self.current_position().unwrap_or_default();
            let field_name =
                self.read_field_name("a field name in a struct literal")?;
            let written = self.peek_nth(0).to_string();
            if !matches!(self.read_token(), Token::Assign) {
                return Err(self.at_consumed(format!(
                    "a field in a literal is given its value with '=', and this writes '{written}'"
                )));
            }
            let value = self.parse_expression(Precedence::Lowest)?;
            let name = self.ast.intern(&field_name);
            fields.push(NamedExpr { name, value, at });
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
                continue;
            }
            // A comma separates one element from the next. A token that begins
            // no expression is reported as the token it is, by the walk that
            // reads the next element.
            if self.peek_nth(0) != end_token
                && !matches!(self.peek_nth(0), Token::Illegal(_))
            {
                let written = self.peek_nth(0).to_string();
                return Err(self.here(format!(
                    "expected ',' between one element and the next, found '{written}'"
                )));
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

            // `format fmt: str` says the argument written here is a string
            // literal, and that the holes in it are counted against the
            // compile-time list the parameter list ends with. Contextual, so a
            // parameter may still be called `format`.
            let format =
                if matches!(
                    self.peek_nth(0),
                    Token::Identifier(word) if word == "format"
                ) && matches!(self.peek_nth(1), Token::Identifier(_))
                {
                    self.read_token();
                    true
                } else {
                    false
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
                // Where the type is written, which is what a report about the
                // type is placed at.
                let mut type_at = self.current_position().unwrap_or_default();
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
                            type_at =
                                self.current_position().unwrap_or_default();
                            Some(self.parse_type()?)
                        }
                    } else {
                        None
                    };

                let name = self.ast.intern(&name);
                parameters.push(Parameter {
                    at: type_at,
                    name,
                    type_annotation,
                    mutable: false,
                    mode,
                    compile_time_signature: None,
                    compile_time_default: None,
                    pack,
                    format,
                    capability: false,
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
                // An arm answers with a value or runs a block. A statement
                // written bare after the colon is neither, and what it wants is
                // the braces the block form already has.
                if matches!(
                    self.peek_nth(0),
                    Token::Return | Token::Break | Token::Continue
                ) || self.at_arm_assignment()
                {
                    return Err(self.at_mark(
                        self.mark(),
                        "an arm of a `match` is an expression or a block, and this is a statement"
                            .to_string(),
                    ));
                }
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

    // Whether an assignment stands where the cursor is: a name, whatever
    // indices and fields follow it, and an `=` behind those.
    fn at_arm_assignment(&self) -> bool {
        if !matches!(self.peek_nth(0), Token::Identifier(_)) {
            return false;
        }
        let mut ahead = 1;
        let mut depth = 0;
        loop {
            match self.peek_nth(ahead) {
                Token::LeftBracket => depth += 1,
                Token::RightBracket => depth -= 1,
                Token::Assign if depth == 0 => return true,
                Token::Dot | Token::Identifier(_) => {}
                Token::EndOfFile => return false,
                _ if depth == 0 => return false,
                _ => {}
            }
            ahead += 1;
        }
    }

    fn parse_pattern_bindings(&mut self) -> Result<Range32> {
        self.read_token();
        let mut bindings = Vec::new();
        while self.peek_nth(0) != &Token::RightBrace {
            let at = self.current_position().unwrap_or_default();
            let field_name = match self.read_token() {
                Token::Identifier(name) => name.to_string(),
                _ => bail!("Expected binding name in pattern"),
            };
            let symbol = self.ast.intern(&field_name);
            bindings.push(PatternBinding {
                field: symbol,
                binding: symbol,
                at,
            });
            if matches!(self.peek_nth(0), Token::Comma) {
                self.read_token();
            }
        }
        self.read_token();
        Ok(self.ast.add_pattern_bindings(&bindings))
    }

    /// An arm's whole pattern: one alternative, or several joined by `|`.
    ///
    /// The alternatives cover a set between them and the body runs for any of
    /// them, which is why nothing in an alternative binds: two variants hold
    /// two shapes, and a name reading a field would mean one thing in one
    /// alternative and another in the next.
    fn parse_pattern(&mut self) -> Result<PatternId> {
        let start = self.mark();
        let first = self.parse_pattern_alternative()?;
        if !matches!(self.peek_nth(0), Token::Pipe) {
            return Ok(first);
        }
        let mut alternatives = vec![first];
        // Where the alternatives are joined. A report about the alternative as
        // a whole belongs at the bar that made it one, rather than past the
        // last thing written on the line.
        let joined_at = self.current_position().unwrap_or_default();
        while matches!(self.peek_nth(0), Token::Pipe) {
            self.read_token();
            alternatives.push(self.parse_pattern_alternative()?);
        }
        for (index, held) in alternatives.iter().enumerate() {
            match self.ast.pattern(*held) {
                Pattern::Wildcard => {
                    return Err(self.here(CATCH_ALL_ALTERNATIVE.to_string()));
                }
                Pattern::EnumVariant { bindings, .. }
                    if !bindings.is_empty() =>
                {
                    let held =
                        anyhow::Error::new(crate::diagnostic::LocatedError {
                            position: joined_at,
                            message: BINDING_ALTERNATIVE.to_string(),
                        });
                    self.skip_past_match();
                    return Err(held);
                }
                _ => {}
            }
            let written = crate::ast_display::display_pattern(&self.ast, *held);
            if alternatives[..index].iter().any(|earlier| {
                crate::ast_display::display_pattern(&self.ast, *earlier)
                    == written
            }) {
                return Err(self.here(REPEATED_ALTERNATIVE.to_string()));
            }
        }
        let list = self.ast.add_pattern_list(&alternatives);
        let span = self.span_from(start);
        Ok(self.ast.push_pattern(Pattern::Or(list), span))
    }

    /// One alternative of a pattern. A range is read here rather than beside
    /// `|`, so `0 | 5..10` joins a number and a span the way it reads.
    ///
    /// A name is the value it stands for, everywhere and always: `case CH_0:`
    /// compares against that constant exactly as `case CH_0..=CH_9:` does. A
    /// name used to bind whatever was matched instead, which made those two
    /// arms mean opposite things and made `case CH_0:` a comparison that
    /// silently was not one. `_` is the arm that covers the rest.
    fn parse_pattern_alternative(&mut self) -> Result<PatternId> {
        let start = self.mark();
        let pattern = match self.peek_nth(0) {
            Token::Underscore => {
                self.read_token();
                Pattern::Wildcard
            }
            // What a `case` covers is a set a reader can count. A decimal
            // covers one of the reals, which is a claim nobody can act on, and
            // text is compared rather than counted; both belong in an `if`.
            Token::Float(_) | Token::Float32(_) => {
                return Err(self.here(DECIMAL_PATTERN.to_string()));
            }
            Token::StringLiteral(_) => {
                return Err(self.here(TEXT_PATTERN.to_string()));
            }
            Token::Identifier(word) if word == "true" => {
                self.read_token();
                Pattern::Literal(Literal::Boolean(true))
            }
            Token::Identifier(word) if word == "false" => {
                self.read_token();
                Pattern::Literal(Literal::Boolean(false))
            }
            _ if self.at_pattern_number() => {
                // Where the span begins, kept because a span that covers
                // nothing is named whole in the report and belongs at the
                // number the reader wrote first, not past the one they wrote
                // last.
                let began = self.current_position().unwrap_or_default();
                let low = self.parse_pattern_bound()?;
                match self.parse_pattern_range(low, began)? {
                    Some(range) => range,
                    None => Pattern::Literal(Literal::Integer(low)),
                }
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
                    // Where this part begins, so what is said about it is said
                    // where it was written rather than past its last token.
                    let part_at = self.current_position().unwrap_or_default();
                    let part = self.parse_pattern_alternative()?;
                    if matches!(self.ast.pattern(part), Pattern::Range { .. })
                        || matches!(self.peek_nth(0), Token::Pipe)
                    {
                        return Err(anyhow::Error::new(
                            crate::diagnostic::LocatedError {
                                position: part_at,
                                message: TUPLE_PART.to_string(),
                            },
                        ));
                    }
                    patterns.push(part);
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
                    return Err(self.at_consumed(NAME_PATTERN.to_string()));
                }
            }
            token => {
                let written = token.to_string();
                return Err(self.here(format!(
                    "Unexpected token in pattern: '{written}'"
                )));
            }
        };
        let span = self.span_from(start);
        Ok(self.ast.push_pattern(pattern, span))
    }

    /// Whether what the cursor is on opens a whole number: one written out,
    /// one under a minus, or a name a `::` declaration settled on one.
    fn at_pattern_number(&self) -> bool {
        match self.peek_nth(0) {
            Token::Integer(_) => true,
            Token::Minus => matches!(self.peek_nth(1), Token::Integer(_)),
            Token::Identifier(name) => self.integer_constant(name).is_some(),
            _ => false,
        }
    }

    /// One end of a span: a whole number written out, with a `-` in front of
    /// it where it is below zero.
    fn parse_pattern_bound(&mut self) -> Result<i64> {
        let negative = matches!(self.peek_nth(0), Token::Minus);
        if negative {
            self.read_token();
        }
        match self.peek_nth(0) {
            Token::Integer(value) => {
                let value = *value;
                self.read_token();
                Ok(if negative { -value } else { value })
            }
            Token::Float(_) | Token::Float32(_) => {
                Err(self.here(DECIMAL_PATTERN.to_string()))
            }
            Token::Identifier(name) => {
                let name = name.clone();
                match self.integer_constant(&name) {
                    Some(value) => {
                        self.read_token();
                        Ok(if negative { -value } else { value })
                    }
                    None => Err(self.here(NOT_A_BOUND.to_string())),
                }
            }
            _ => Err(self.here(NOT_A_BOUND.to_string())),
        }
    }

    /// The rest of a span once its lower end has been read, or nothing where
    /// what was read stands on its own.
    fn parse_pattern_range(
        &mut self,
        low: i64,
        began: Position,
    ) -> Result<Option<Pattern>> {
        let inclusive = match self.peek_nth(0) {
            Token::DotDot => false,
            Token::DotDotEqual => true,
            _ => return Ok(None),
        };
        self.read_token();
        let high = self.parse_pattern_bound()?;
        let holds = if inclusive { low <= high } else { low < high };
        if !holds {
            let between = if inclusive { "..=" } else { ".." };
            return Err(anyhow::Error::new(crate::diagnostic::LocatedError {
                position: began,
                message: format!(
                    "the case range {low}{between}{high} covers nothing"
                ),
            }));
        }
        Ok(Some(Pattern::Range {
            low,
            high,
            inclusive,
        }))
    }

    /// The number a `::` declaration settled on for this name, for the one
    /// position a pattern reads a name as a value rather than binding it.
    fn integer_constant(&self, name: &str) -> Option<i64> {
        match self.constant_values.get(name) {
            Some(crate::const_eval::Value::Integer(held)) => Some(*held),
            _ => None,
        }
    }

    fn parse_if_expression(&mut self) -> Result<ExprId> {
        self.parse_if_expression_marked(false)
    }

    /// `if` and `$if`. The second is answered while the body is expanded and
    /// the branch that cannot run is dropped before anything checks it, so a
    /// body may write what only one of the two arguments makes sense for. The
    /// two used to be spelled the same, and which one a reader was looking at
    /// was decided by whether a name in the condition happened to be a
    /// compile-time parameter, which is not something the line says.
    fn parse_if_expression_marked(
        &mut self,
        expansion_time: bool,
    ) -> Result<ExprId> {
        let start = self.mark();
        if expansion_time {
            self.read_token();
        }
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
            let chained = matches!(self.peek_nth(0), Token::If)
                || (matches!(self.peek_nth(0), Token::Dollar)
                    && matches!(self.peek_nth(1), Token::If));
            if chained {
                let arm_start = self.mark();
                let marked = matches!(self.peek_nth(0), Token::Dollar);
                let else_if = self.parse_if_expression_marked(marked)?;
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
            Expression::If(condition, consequence, alternative, expansion_time),
            self.span_from(start),
        ))
    }

    // Whether the `fn` at the cursor opens a function written where it is
    // named, rather than the type of one. A body is what tells them apart:
    // `fn(i64) -> i64` is a type and carries none, and `fn(a: i64) -> i64 { a }`
    // is a function and does. The parameter list and the return type are read
    // past rather than parsed, since either form may be there and only what
    // follows decides which this is.
    fn function_literal_follows(&self) -> bool {
        let mut at = 1;
        if !matches!(self.peek_nth(at), Token::LeftParentheses) {
            return false;
        }
        let mut depth = 0usize;
        loop {
            match self.peek_nth(at) {
                Token::LeftParentheses => depth += 1,
                Token::RightParentheses => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        at += 1;
                        break;
                    }
                }
                Token::EndOfFile => return false,
                _ => {}
            }
            at += 1;
        }
        // Past the parameters. A function says what it answers with or opens
        // its body right here, so anything else means the brackets held an
        // expression rather than a parameter list. Asked before the search
        // below, which would otherwise run to whatever brace came next.
        if !matches!(
            self.peek_nth(at),
            Token::LeftBrace | Token::Arrow | Token::Uses
        ) {
            return false;
        }
        // Whatever the return type is, the brace that opens a body comes after
        // it. A comma or a closing bracket at the depth this started on is the
        // argument ending, which means there was no body and this is a type.
        let mut depth = 0usize;
        loop {
            match self.peek_nth(at) {
                Token::LeftBrace if depth == 0 => return true,
                Token::LeftParentheses
                | Token::LeftBracket
                | Token::LessThan => depth += 1,
                Token::RightParentheses | Token::RightBracket if depth == 0 => {
                    return false;
                }
                Token::Comma if depth == 0 => return false,
                // `Vec<Vec<i64>>` closes two of them with one token, which
                // is how the lexer reads `>>`, so it counts for two.
                Token::ShiftRight => depth = depth.saturating_sub(2),
                // Saturating, since this reads past a form it has not
                // decided the shape of yet. A `>` with no `<` in front of it is
                // not a bracket at all, and the answer here is that no body was
                // found rather than an arithmetic fault.
                Token::RightParentheses
                | Token::RightBracket
                | Token::GreaterThan => depth = depth.saturating_sub(1),
                Token::EndOfFile => return false,
                _ => {}
            }
            at += 1;
        }
    }

    fn parse_function_literal(&mut self) -> Result<ExprId> {
        let start = self.mark();
        self.read_token();
        // The frame this function's own compile-time parameters go into, open
        // from the parameter list through the body, which is everywhere one of
        // them may be written as a length.
        self.compile_time_names
            .push(std::collections::HashSet::new());
        let parsed = self.parse_function_body(&start);
        self.compile_time_names.pop();
        parsed
    }

    fn parse_function_body(&mut self, start: &u32) -> Result<ExprId> {
        let start = *start;
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
        // Where the answer's type begins, read before it, since reading it
        // moves the cursor to whatever follows. The token after the arrow is
        // the type, so its place is the one a report about that type wants.
        let at = if matches!(self.peek_nth(0), Token::Arrow) {
            self.position_at(self.mark() + 1).unwrap_or_default()
        } else {
            crate::lexer::Position::default()
        };
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
        let mut bound_text = String::new();
        let mut bound_message = None;
        if matches!(self.peek_nth(0), Token::Where) {
            self.read_token();
            let first = self.mark();
            let held = self.no_struct_literal;
            self.no_struct_literal = true;
            let expression = self.parse_expression(Precedence::Lowest);
            self.no_struct_literal = held;
            bound = Some(expression?);
            bound_text = self.tokens_written(first, self.mark());
            // `else "..."` is what the declaration says when the bound does not
            // hold. One string literal, so what a reader is told is settled
            // where the bound is written rather than worked out at the call.
            if matches!(self.peek_nth(0), Token::Else) {
                self.read_token();
                match self.read_token() {
                    Token::StringLiteral(held) => {
                        bound_message = Some(held.clone())
                    }
                    _ => {
                        return Err(self.at_consumed(
                            "`where ... else` is followed by one string, which is what the reader is told"
                                .to_string(),
                        ));
                    }
                }
            }
        }
        Ok(ReturnSignature {
            kind,
            uses,
            bound,
            bound_text,
            bound_message,
            at,
        })
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
            if self.touches_previous() == Some(true)
                || self.touches_next() == Some(true)
            {
                let position = self.current_position().unwrap_or_default();
                self.diagnostics.push(crate::diagnostic::Diagnostic::new(
                    position,
                    FAILURE_SET_SPACING.to_string(),
                ));
            }
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
        // Where the list opens, since what is wrong with it is the list rather
        // than whatever token the read of it ended on.
        let opened = self.mark();
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
        // A name says which value is which, and it is the field a `return` by
        // name writes. A list that left one out had that field called `value0`,
        // a spelling the compiler chose and the language never offered, so the
        // name is required rather than optional.
        if values.iter().any(|held| held.name.is_none()) {
            return Err(self.at_mark(
                opened,
                "a return type list names every value; write `-> (name: T, name: T)`"
                    .to_string(),
            ));
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

            // `format fmt: str` says the argument written here is a string
            // literal, and that the holes in it are counted against the
            // compile-time list the parameter list ends with. Contextual, so a
            // parameter may still be called `format`.
            let format =
                if matches!(
                    self.peek_nth(0),
                    Token::Identifier(word) if word == "format"
                ) && matches!(self.peek_nth(1), Token::Identifier(_))
                {
                    self.read_token();
                    true
                } else {
                    false
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
                // Where the type is written, which is what a report about the
                // type is placed at.
                let mut type_at = self.current_position().unwrap_or_default();
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
                            type_at =
                                self.current_position().unwrap_or_default();
                            Some(self.parse_type()?)
                        }
                    } else {
                        None
                    };

                let name = self.ast.intern(&name);
                parameters.push(Parameter {
                    at: type_at,
                    name,
                    type_annotation,
                    mutable: false,
                    mode,
                    compile_time_signature: None,
                    compile_time_default: None,
                    pack,
                    format,
                    capability: false,
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
    /// The compile-time parameters a declaration binds, opened as a frame the
    /// declaration's body is read inside. A struct's fields and an enum's
    /// variants are read where the frame is open, so `[N]T` there names the
    /// parameter rather than nothing.
    fn open_compile_time_frame(&mut self, names: &[String]) {
        self.compile_time_names
            .push(names.iter().cloned().collect());
    }

    /// Whether a name written where a compile-time number is read stands for
    /// one: a constant, or a parameter an instance supplies a number for.
    fn names_a_compile_time_number(&self, name: &str) -> bool {
        self.integer_constants.contains_key(name)
            || self.constant_values.contains_key(name)
            || self
                .compile_time_names
                .iter()
                .any(|frame| frame.contains(name))
    }

    /// A name written where a compile-time number is read that the program
    /// binds to no number. Left to stand, `[Nope]i64` was a type carrying a
    /// length nothing could ever give it, and what a reader was told came from
    /// whichever pass asked its size first.
    fn refuse_a_length_naming_nothing(&self, name: &str) -> anyhow::Error {
        anyhow::Error::new(crate::diagnostic::LocatedError {
            position: self.position_at(self.mark() - 1).unwrap_or_default(),
            message: format!(
                "'{name}' has no value at compile time, so this cannot be worked out before the program runs"
            ),
        })
    }

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
            self.refuse_literal_name(&param_name)?;
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
            // `= Heap` says what an instance leaving this parameter out means.
            // The value is read off the tokens by `scan_generic_types`, before
            // the parse, because an instance is written in a file that may not
            // be the one declaring it.
            if matches!(self.peek_nth(0), Token::Assign) {
                self.read_token();
                if matches!(self.peek_nth(0), Token::Integer(_)) {
                    self.read_token();
                } else {
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
        if let Some(frame) = self.compile_time_names.last_mut() {
            frame.insert(name.clone());
        }
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
        // `= Heap` says what the parameter stands for where a call writes
        // nothing for it. A type, a name that is a constant or a function, or a
        // number, since those are the three kinds of compile-time argument.
        let mut compile_time_default = None;
        if matches!(self.peek_nth(0), Token::Assign) {
            self.read_token();
            compile_time_default = Some(match self.peek_nth(0) {
                Token::Integer(value) => {
                    let held = *value;
                    self.read_token();
                    Type::ConstUsize(held as usize)
                }
                _ => self.parse_type()?,
            });
        }
        let symbol = self.ast.intern(&name);
        Ok(Parameter {
            at: crate::lexer::Position::default(),
            name: symbol,
            type_annotation: Some(Type::TypeParam(name)),
            mutable: false,
            mode: ParamMode::Read,
            compile_time_signature,
            compile_time_default,
            pack: false,
            format: false,
            capability: false,
        })
    }

    /// An array's length: `+` and `-` over terms, left to right.
    fn parse_size_expression(&mut self) -> Result<SizeExpr> {
        let mut left = self.parse_size_term()?;
        loop {
            let op = match self.peek_nth(0) {
                Token::Plus => SizeOp::Add,
                Token::Minus => SizeOp::Subtract,
                _ => return Ok(left),
            };
            self.read_token();
            let right = self.parse_size_term()?;
            left = SizeExpr::Binary(Box::new(left), op, Box::new(right));
        }
    }

    fn parse_size_term(&mut self) -> Result<SizeExpr> {
        let mut left = self.parse_size_atom()?;
        loop {
            let op = match self.peek_nth(0) {
                Token::Asterisk => SizeOp::Multiply,
                Token::Slash => SizeOp::Divide,
                Token::Percent => SizeOp::Modulo,
                _ => return Ok(left),
            };
            self.read_token();
            let right = self.parse_size_atom()?;
            left = SizeExpr::Binary(Box::new(left), op, Box::new(right));
        }
    }

    /// Whether a length here is worked out rather than named: a call to a
    /// function this file declares, or an item read out of a constant that
    /// holds a run of them. `[Vertex; 4]` names a type and does neither, so
    /// what follows the name is what tells them apart.
    fn at_size_call(&self) -> bool {
        let Token::Identifier(name) = self.peek_nth(0) else {
            return false;
        };
        if matches!(self.peek_nth(1), Token::LeftParentheses) {
            return self.folder.declares(name);
        }
        matches!(self.peek_nth(1), Token::LeftBracket | Token::Dot)
            && self.constant_values.contains_key(name)
    }

    /// A call written where a compile-time value is read, and the number it
    /// answers with. Every argument has to be known here, since this is where
    /// the answer is needed; a size parameter a generic has not bound yet is
    /// named rather than left half-worked-out.
    fn parse_folded_call(&mut self) -> Result<i64> {
        let at = self.mark();
        // The call and nothing after it. A `>` past one closes the generic
        // arguments it is written inside, and read as an expression it is a
        // comparison instead; the arithmetic around a length is the size
        // parser's, which is what calls this.
        let expression = self.parse_expression(Precedence::Prefix)?;
        let Parser {
            ast,
            folder,
            constant_values,
            ..
        } = self;
        let answered = folder
            .expression(ast, expression, constant_values)
            .and_then(|value| {
                value.integer().ok_or_else(|| {
                    "a length is a whole number and this answers with something else"
                        .to_string()
                })
            });
        match answered {
            Ok(value) => Ok(value),
            Err(reason) => {
                let position = self
                    .positions
                    .get(at as usize)
                    .copied()
                    .unwrap_or_default();
                Err(anyhow::Error::new(crate::diagnostic::LocatedError {
                    position,
                    message: reason,
                }))
            }
        }
    }

    /// A number, a name that stands for one, a call this file can work out, or
    /// a bracketed length. A call is run here rather than left written down,
    /// which is what keeps a length one thing rather than two.
    fn parse_size_atom(&mut self) -> Result<SizeExpr> {
        if self.at_size_call() {
            return Ok(SizeExpr::Number(self.parse_folded_call()?));
        }
        match self.read_token().clone() {
            Token::Integer(value) => Ok(SizeExpr::Number(value)),
            // A type read back from its own `Display` is the compiler talking
            // to itself, and a template's length is written there as the
            // parameter's name with no declaration around it to bind.
            Token::Identifier(name)
                if self.internal_types
                    || self.names_a_compile_time_number(&name) =>
            {
                Ok(SizeExpr::Named(name))
            }
            Token::Identifier(name) => {
                Err(self.refuse_a_length_naming_nothing(&name))
            }
            Token::LeftParentheses => {
                let inner = self.parse_size_expression()?;
                if !matches!(self.read_token(), Token::RightParentheses) {
                    bail!("Expected ')' after an array length");
                }
                Ok(inner)
            }
            token => {
                let written = token.to_string();
                Err(self.at_consumed(format!(
                    "an array's length is a number, a name standing for one, or arithmetic over those, and this is '{written}'"
                )))
            }
        }
    }

    fn parse_type(&mut self) -> Result<Type> {
        // What a layout answers is a number, and a number is not a type, so a
        // reader who wrote one here meant it as a generic's value argument or
        // as a length. Named where it stands: read on as a type it became the
        // element type of nothing, and the parentheses after it were what the
        // reader was told about.
        if let Some(named) = self.at_layout_answer() {
            bail!("{}", crate::const_eval::layout_message(&named));
        }
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
                // Where the length is written, for a report about what it works
                // out to.
                let at = self.current_position().unwrap_or_default();
                // A length is a compile-time value like a constant's, and a
                // layout is worked out later than either. Named here rather
                // than left to the size parser, which read the word as the
                // element type and asked for the `;` of a form Frost has not.
                if let Some(named) = self.at_layout_answer() {
                    bail!("{}", crate::const_eval::layout_message(&named));
                }
                if matches!(self.peek_nth(0), Token::RightBracket) {
                    self.read_token();
                    Type::Slice(Box::new(self.parse_type()?))
                } else if self.at_size_call()
                    || starts_size_expression(
                        self.peek_nth(0),
                        self.peek_nth(1),
                    )
                {
                    // A length is arithmetic over numbers, the constants a
                    // module declares, and the size parameters a generic binds.
                    // What is known is worked out here, so `[N]u8` where N is a
                    // constant and `[8]u8` are the same type and nothing
                    // downstream has to know the difference. What is not stays
                    // written until the generic that binds it is instantiated.
                    let size = self.parse_size_expression()?;
                    if !matches!(self.read_token(), Token::RightBracket) {
                        bail!("Expected ']' after array size");
                    }
                    let constants = &self.integer_constants;
                    let known =
                        size.evaluate(&|name| constants.get(name).copied());
                    // A length that works out below zero, told apart from one
                    // that works out to nothing. Read as a count the two were
                    // one answer, and a negative was carried on as a length
                    // some later instantiation would supply.
                    if known.is_some_and(|value| value < 0) {
                        return Err(negative_length(at));
                    }
                    match known.and_then(|value| usize::try_from(value).ok()) {
                        Some(known) => {
                            Type::Array(Box::new(self.parse_type()?), known)
                        }
                        None => Type::ArrayGeneric(
                            Box::new(self.parse_type()?),
                            size,
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
            // A call where a type is read is a compile-time number: the
            // argument of `columns<T, N>`, or the `$N` a generic takes at a
            // call. Both reach here, so working it out here is what makes one
            // rule out of two positions.
            Token::Identifier(_) if self.at_size_call() => {
                let value = self.parse_folded_call()?;
                let Ok(held) = usize::try_from(value) else {
                    bail!(
                        "a compile-time size is a whole number that is not negative, and this answers with {value}"
                    );
                };
                Type::ConstUsize(held)
            }
            // A call this file cannot name has no number to stand for. Read on,
            // the name was taken for a type and the parentheses after it were
            // what the reader was told about.
            Token::Identifier(name)
                if matches!(self.peek_nth(1), Token::LeftParentheses) =>
            {
                let name = name.clone();
                bail!(
                    "'{name}' is not a function this program declares, so there is nothing to work out here"
                )
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
                        let rendered: Vec<String> = self
                            .with_defaults(&name, arguments)
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
        // The line the block opens on. A statement sharing it is not indented
        // relative to anything, so it sets no column for the rest to keep.
        let opened_on = self.current_position().unwrap_or_default().line;
        self.read_token();
        self.block_depth += 1;

        let mut statements = Vec::new();
        // The column every statement of this block begins at, set by the first,
        // and the line the last one started on.
        let mut block_column: Option<usize> = None;
        let mut previous_line = opened_on;

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
            // A line outside brackets is a statement, so it cannot open with an
            // operator that joins it to the line above. An expression that runs
            // over lines is written inside the brackets it already needs, where
            // a line break says nothing at all.
            if position.line != previous_line
                && continues_an_expression(self.peek_nth(0))
            {
                let written = self.peek_nth(0).to_string();
                self.record_error(
                    position,
                    &anyhow::anyhow!(
                        "a line cannot open with '{written}': a line break ends a statement, so there is nothing above for it to join to. Put the operator at the end of the line above, or the whole expression inside brackets, where a line break says nothing"
                    ),
                );
                self.synchronize_in_block(0);
                continue;
            }
            // Every statement of a block begins at the same column. One
            // indented past its neighbours reads as a continuation of the line
            // above, and a continuation that parses as a statement of its own
            // is an expression that lost the operator joining it to that line.
            if let Some(column) = block_column
                && position.line != previous_line
                && position.column > column
                && !self.was_lifted(position.line)
            {
                self.record_error(
                    position,
                    &anyhow::anyhow!(
                        "this line is indented past the statement above it, so \
                         it reads as continuing that line, and it begins a \
                         statement of its own. An expression broken over lines \
                         carries the operator that joins them at the end of \
                         the first, or is written inside brackets"
                    ),
                );
            }
            if block_column.is_none() && position.line != previous_line {
                block_column = Some(position.column);
            }
            previous_line = position.line;
            // Where the statement began, so a fault inside it can be told what
            // it had left open.
            let began = self.tokens.clone();
            let before = self.tokens.len();
            match self.parse_statement() {
                Ok(Some(statement)) => {
                    statements.push(statement);
                }
                Ok(None) => break,
                Err(error) => {
                    self.record_error(position, &error);
                    let read = before - self.tokens.len();
                    self.synchronize_in_block(left_open(began, read));
                }
            }
            // A right brace here means that statement was the block's value,
            // and a block whose value is `-1` is ordinary. Anything else means
            // the minus opened a statement whose answer nothing reads.
            if opened_with_minus
                && self.peek_nth(0) != &Token::RightBrace
                && self.peek_nth(0) != &Token::EndOfFile
            {
                // Carried as a located fault rather than a bare one, since
                // `record_error` reads the place off the cursor for anything
                // that has none and the cursor is past this statement by now:
                // the report landed on the line after the one that opens with
                // the minus.
                self.record_error(
                    position,
                    &anyhow::Error::new(crate::diagnostic::LocatedError {
                        position,
                        message:
                        "this line opens with '-', so it negates what \
                         follows rather than continuing the line above, and \
                         nothing reads what it works out. Two readings are \
                         open here and they are different programs: this line \
                         standing on its own as a negative value, and this \
                         line joined to the one above as a subtraction. A \
                         statement ends at the end of a line, so the first is \
                         what is written. Write the whole expression on one \
                         line, or leave the '-' at the end of the line above \
                         where it says a subtraction is meant"
                            .to_string(),
                    }),
                );
            }
        }

        self.block_depth -= 1;
        if !matches!(self.peek_nth(0), Token::RightBrace) {
            bail!(
                "expected '}}' to close this block, found the end of the file"
            );
        }
        self.read_token();

        Ok(self.ast.add_stmt_list(&statements))
    }

    fn read_token(&mut self) -> &Token {
        self.consumed += 1;
        let held = self.tokens.next().unwrap_or(&Token::EndOfFile);
        match held {
            Token::LeftParentheses | Token::LeftBracket => {
                self.bracket_depth += 1
            }
            Token::RightParentheses | Token::RightBracket => {
                self.bracket_depth -= 1
            }
            _ => {}
        }
        held
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
    // `packed` is a word rather than a keyword, so a local, a field and a
    // parameter may all still be called `packed`, and one is: `slab_get` names
    // its packed handle that. The `struct` after it is what marks the
    // declaration.
    // `align` is a word too, so a field may be called `align`. What tells the
    // marker apart is the `(` after it, which the next field's `:` is not.
    fn at_field_alignment(&self) -> bool {
        matches!(self.peek_nth(0), Token::Identifier(word) if word == "align")
            && matches!(self.peek_nth(1), Token::LeftParentheses)
    }

    fn at_packed_struct(&self, offset: usize) -> bool {
        matches!(self.peek_nth(offset), Token::Identifier(word) if word == "packed")
            && matches!(self.peek_nth(offset + 1), Token::Struct)
    }

    fn at_flags_declaration(&self, offset: usize) -> bool {
        matches!(self.peek_nth(offset), Token::Identifier(word) if word == "flags")
            && matches!(self.peek_nth(offset + 2), Token::LeftBrace)
            && matches!(
                self.peek_nth(offset + 1),
                Token::Identifier(name) if is_scalar_type_name(name)
            )
    }

    // A brace after a type declaration opens the block of values it names,
    // where what follows the brace is a declaration. A struct literal written
    // with no name sets its fields with `=`, so the `::` is what tells the two
    // apart and no expression is ever read as this block.
    fn at_values_block(&self) -> bool {
        matches!(self.peek_nth(0), Token::LeftBrace)
            && matches!(self.peek_nth(1), Token::Identifier(_))
            && matches!(self.peek_nth(2), Token::DoubleColon)
    }

    // Past a `{ ... }` and everything nested in it, so a declaration refused
    // before its block is read leaves the cursor where the next one begins.
    fn skip_braced_block(&mut self) {
        let mut depth = 0i32;
        while !matches!(self.peek_nth(0), Token::EndOfFile) {
            match self.read_token() {
                Token::LeftBrace => depth += 1,
                Token::RightBrace => {
                    depth -= 1;
                    if depth == 0 {
                        return;
                    }
                }
                _ => {}
            }
        }
    }

    // `Key :: distinct i64 { Left :: 80 }`: the values a type names under
    // itself. Each is reached as `Key::Left`, which is where a variant and a
    // bit of a set are already reached, and elides to `.Left` where the
    // context names the type.
    //
    // What each value is worth is settled where it is named rather than here:
    // the types have not been resolved at this point, so a literal has nothing
    // yet to take its type from.
    //
    // Each value opens a line of its own. `Key::Left` written as a value is the
    // same two tokens an entry is, so where one entry ends and the next begins
    // is the line, which is what separates one declaration from the next
    // everywhere else in the language.
    fn parse_values_under(
        &mut self,
        type_name: &str,
        variants: &[String],
        generic: bool,
    ) -> Result<()> {
        if !self.at_values_block() {
            return Ok(());
        }
        // Where the block opens, which is what a report about the block as a
        // whole is placed at.
        let opened_block = self.mark();
        // A generic declaration is one type for each set of arguments given to
        // it, so there is no single type for a value to be a value of, and
        // nothing to write `Box::EMPTY` at.
        //
        // The block is read past before the fault is raised, so what follows it
        // is where reading resumes. Left in place, the recovery took the
        // block's closing brace for a declaration head and said so, and one
        // mistake was reported as two.
        if generic {
            let block = self.mark();
            self.skip_braced_block();
            return Err(self.at_mark(
                block,
                format!("a type names values of itself, and a generic declaration is one type for each set of arguments given to it, so '{type_name}' names none"),
            ));
        }
        self.read_token();
        let mut named: Vec<String> = Vec::new();
        while self.peek_nth(0) != &Token::RightBrace {
            if matches!(self.peek_nth(0), Token::EndOfFile) {
                // At the declaration that opened the block. Where the read
                // stopped is the end of the file, and the last value it got
                // through is not what is wrong with the program.
                return Err(self.at_mark(
                    opened_block,
                    format!(
                        "the values '{type_name}' names are written inside braces, and this block is not closed"
                    ),
                ));
            }
            if !named.is_empty() && self.on_the_same_line() {
                bail!(
                    "a type names each of its values on a line of its own, and this one follows another"
                );
            }
            let entry = self.mark();
            let name = match self.read_token() {
                Token::Identifier(name) => name.to_string(),
                _ => {
                    return Err(self.at_mark(
                        entry,
                        "a type names each of its values with a name, and this is not one".to_string(),
                    ));
                }
            };
            if !matches!(self.read_token(), Token::DoubleColon) {
                return Err(self.at_mark(
                    entry,
                    format!("'{name}' is a value named under '{type_name}', so it is written as '{name} :: <value>'"),
                ));
            }
            // The value stands on the line its `::` is on. Without this a name
            // with nothing after it read the entry below as its value, and the
            // block came out one value short.
            if matches!(self.peek_nth(0), Token::RightBrace)
                || !self.on_the_same_line()
            {
                return Err(self.at_mark(
                    entry,
                    format!("'{name}' is a value named under '{type_name}', and it is written after its '::'"),
                ));
            }
            if named.iter().any(|held| held == &name) {
                return Err(self.at_mark(
                    entry,
                    format!("a type names each of its values once, and '{type_name}' names '{name}' twice"),
                ));
            }
            if variants.iter().any(|held| held == &name) {
                return Err(self.at_mark(
                    entry,
                    format!("a type names each of its values once, and '{type_name}' names '{name}' as a variant and as a value"),
                ));
            }
            let value = self.parse_expression(Precedence::Lowest)?;
            let type_symbol = self.ast.intern(type_name);
            let name_symbol = self.ast.intern(&name);
            named.push(name);
            self.ast.type_values.push(TypeValue {
                type_name: type_symbol,
                name: name_symbol,
                value,
            });
        }
        self.read_token();
        Ok(())
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

/// Whether a token joins the line it opens to the line above it.
///
/// Every binary operator, and the `.` that reaches into a value. A `-` is not
/// one: it opens a statement by negating what follows, which is the rule
/// `opened_with_minus` is about.
/// Whether what follows a `[` is a length rather than the element type of the
/// `[T; N]` form. A number or a name is one when what comes after it closes the
/// bracket or continues the arithmetic; `[Vertex; 4]` writes a name too, and the
/// `;` after it is what tells them apart.
fn starts_size_expression(first: &Token, second: &Token) -> bool {
    match first {
        Token::LeftParentheses => true,
        Token::Integer(_) | Token::Identifier(_) => matches!(
            second,
            Token::RightBracket
                | Token::Plus
                | Token::Minus
                | Token::Asterisk
                | Token::Slash
                | Token::Percent
        ),
        _ => false,
    }
}

pub fn continues_an_expression(token: &Token) -> bool {
    matches!(
        token,
        Token::Plus
            | Token::Asterisk
            | Token::Slash
            | Token::Percent
            | Token::And
            | Token::Or
            | Token::Pipe
            | Token::Ampersand
            | Token::Equal
            | Token::NotEqual
            | Token::LessThan
            | Token::LessThanOrEqual
            | Token::GreaterThan
            | Token::GreaterThanOrEqual
            | Token::ShiftLeft
            | Token::ShiftRight
            | Token::Dot
    )
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
            // A statement handed to a parser reading declarations. Both
            // refusals say the same thing about it: what is written here
            // belongs in a block, which is where the second reading puts it.
            Err(error)
                if error.to_string().contains("expected a declaration head")
                    || error
                        .to_string()
                        .contains("a binding at the top level") =>
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
                type_at: crate::lexer::Position::default(),
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
        let Expression::If(condition, consequence, alternative, _) =
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
        let Expression::If(condition, consequence, alternative, _) =
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
        let module = parse_module("defer close(held);")?;
        assert_eq!(module.roots.len(), 1);
        if let Statement::Defer(inner) = module.ast.stmt(module.roots[0]) {
            let Statement::Expression(_) = module.ast.stmt(*inner) else {
                bail!("Expected a call inside defer");
            };
        } else {
            bail!("Expected defer statement");
        }
        Ok(())
    }

    // A `defer` runs where the function leaves, so one that leaves the function
    // is read again at the exit it makes, which makes another exit. This used
    // to parse, and both compilers ran out of stack on it.
    #[test]
    fn a_defer_may_not_leave_the_function() -> Result<()> {
        let held = parse_module("defer return 5;");
        let Err(error) = held else {
            bail!("Expected a `defer return` to be refused");
        };
        assert!(
            format!("{error}").contains("would leave it again"),
            "got: {error}"
        );
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
                ..
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
                ..
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
        let module = parse_module("mut x := 5")?;
        assert_eq!(module.roots.len(), 1);
        match module.ast.stmt(module.roots[0]) {
            Statement::Let {
                name,
                type_annotation,
                value,
                mutable,
                ..
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
        let module = parse_module("mut x : i64 = 42")?;
        assert_eq!(module.roots.len(), 1);
        match module.ast.stmt(module.roots[0]) {
            Statement::Let {
                name,
                type_annotation,
                value,
                mutable,
                ..
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
        let output = "mut myVar := anotherVar;";
        let mut ast = Ast::default();
        let another = ast.intern("anotherVar");
        let value =
            ast.push_expr(Expression::Identifier(another), TokenSpan::NONE);
        let name = ast.intern("myVar");
        let statement = ast.push_stmt(
            Statement::Let {
                name,
                type_annotation: None,
                type_at: crate::lexer::Position::default(),
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
        let output = "mut x : i64 = 5;";
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
                type_at: crate::lexer::Position::default(),
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
        let input = "fn(a: i64, b: i64) -> (quotient: i64, remainder: i64) { return a / b, a % b }";
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
                assert!(values.iter().all(|held| held.name.is_some()));
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
        let module = parse_module("quotient, mut remainder := divide(7, 2)")?;
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
        let named = |ast: &mut Ast, name: &str, value_type| ReturnValue {
            name: Some(ast.intern(name)),
            value_type,
        };
        let first = named(&mut ast, "quotient", Type::I64);
        let second = named(&mut ast, "remainder", Type::I64);
        let values = ast.add_return_values(vec![first, second]);
        let sig = ReturnSignature::plain(ReturnKind::Multiple(values));
        assert_eq!(
            ast.signature_to_type(&sig),
            Some(Type::Struct(
                "__multi_quotient__i64_remainder__i64".to_string()
            ))
        );

        // The names are part of what the struct is, so two lists holding the
        // same types under different names are different structs.
        let first = named(&mut ast, "high", Type::I64);
        let second = named(&mut ast, "low", Type::I64);
        let values = ast.add_return_values(vec![first, second]);
        let sig = ReturnSignature::plain(ReturnKind::Multiple(values));
        assert_eq!(
            ast.signature_to_type(&sig),
            Some(Type::Struct("__multi_high__i64_low__i64".to_string()))
        );
    }

    #[test]
    fn a_return_type_list_names_every_value() {
        for input in [
            "fn(a: i64) -> (quotient: i64, i64) { return a, a }",
            "fn(a: i64) -> (i64, i64) { return a, a }",
        ] {
            let mut lexer = Lexer::new(input);
            let tokens = lexer.tokenize().unwrap();
            let mut parser = Parser::new(&tokens);
            assert!(parser.parse().is_err(), "{input} was accepted");
        }
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
