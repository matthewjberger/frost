// What `frost fmt` writes.
//
// Which statement is on which line is the author's. A `+` at a line break
// continues the expression above it and a leading `-` opens a new statement, so
// moving a statement would say something nobody wrote.
//
// Inside a bracket holding a list there is no statement boundary, and where
// each element goes is a question of width. Those breaks are settled here: the
// run goes on one line when it fits in `WIDTH`, and otherwise the bracket opens
// at the end of a line, each element takes one of its own, and the bracket
// closes a line back at the indentation it opened at. So is the space inside a
// line, the indentation in front of it, how many blank lines may sit between
// two of them, the brace that opens a block, and the newline a file ends with.
//
// The token stream drops whitespace and comments, and nothing is lost by that:
// the lexer records where every token starts and stops, so the gaps between one
// token's end and the next one's start are exactly the whitespace and the
// comments. Reading the trivia is reading those gaps out of the source. The
// invariant the rest of this stands on is that the token extents and the gaps
// between them reconstruct the file byte for byte, which
// `every_source_is_its_tokens_and_gaps` holds over the whole corpus.

use crate::lexer::{Lexer, Position, Token};

/// A byte-order mark is not part of the program, and the lexer already skips
/// one. The formatter's output is the program, so the mark does not come back.
const BOM: &str = "\u{feff}";

/// What a line ends with. The formatter's output uses this whatever the input
/// used.
const LINE_BREAK: char = '\n';

/// One token's extent in the source, in bytes.
struct Extent {
    start: usize,
    end: usize,
}

/// Where every line of a source begins, in bytes, so a line and column can be
/// turned into an offset.
fn line_starts(source: &str) -> Vec<usize> {
    let mut starts = vec![0usize];
    for (at, byte) in source.bytes().enumerate() {
        if byte == b'\n' {
            starts.push(at + 1);
        }
    }
    starts
}

/// The byte a place names. A column counts characters and an edit counts bytes,
/// so the column is walked rather than added.
fn offset_of(
    source: &str,
    starts: &[usize],
    place: &Position,
) -> Option<usize> {
    let line = starts.get(place.line.checked_sub(1)?)?;
    let rest = source.get(*line..)?;
    let within = rest
        .char_indices()
        .nth(place.column.saturating_sub(1))
        .map(|(at, _)| at)
        .unwrap_or(rest.len());
    Some(line + within)
}

/// Every token's extent, in source order.
///
/// Lexed here rather than taken from a caller: this needs nothing else from the
/// front end, and a file whose parse failed still formats. A fault leaves a
/// placeholder token whose extent is where it was read from.
fn extents(source: &str) -> Option<(Vec<Token>, Vec<Extent>)> {
    let body = source.strip_prefix(BOM).unwrap_or(source);
    let shift = source.len() - body.len();
    let mut lexer = Lexer::new(body);
    let tokens = lexer.tokenize().ok()?;
    let starts = line_starts(body);
    let mut held = Vec::with_capacity(tokens.len());
    for (start, end) in lexer.positions().iter().zip(lexer.ends()) {
        let start = offset_of(body, &starts, start)? + shift;
        let end = offset_of(body, &starts, end)? + shift;
        if end < start || end > source.len() {
            return None;
        }
        held.push(Extent { start, end });
    }
    Some((tokens, held))
}

/// The source as the token texts and the gaps between them, in order, starting
/// with the gap in front of the first token and ending with the one after the
/// last. Answers `None` for a source the lexer cannot get through at all.
pub fn tokens_and_gaps(source: &str) -> Option<Vec<String>> {
    let (_, held) = extents(source)?;
    let mut pieces = Vec::new();
    let mut at = 0usize;
    for extent in &held {
        pieces.push(source.get(at..extent.start)?.to_string());
        pieces.push(source.get(extent.start..extent.end)?.to_string());
        at = extent.end;
    }
    pieces.push(source.get(at..)?.to_string());
    Some(pieces)
}

/// What sits between two tokens: the newlines, the comments, and nothing else
/// that survives.
enum Trivia {
    Break,
    Comment(String),
}

/// The gap between two tokens, read as the things a formatter keeps. Whitespace
/// is not among them: how much space to put back is this file's decision, and
/// the newlines are kept because which line a token is on is meaning.
fn trivia_in(gap: &str) -> Vec<Trivia> {
    let mut held = Vec::new();
    let mut rest = gap;
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix('\n') {
            held.push(Trivia::Break);
            rest = after;
            continue;
        }
        if rest.starts_with("//") {
            let end = rest.find('\n').unwrap_or(rest.len());
            held.push(Trivia::Comment(rest[..end].trim_end().to_string()));
            rest = &rest[end..];
            continue;
        }
        // A block comment runs to its close, over as many lines as it was
        // written across, and is kept whole. The lexer stops at the first `*/`
        // and these do not nest, so the same scan finds the same end.
        //
        // Its inside is left as it was written. Everything the formatter
        // decides is where a line starts, and the lines inside one of these are
        // the writer's: an ASCII drawing re-indented is a different drawing.
        // Without this arm the bytes fell through to the step below, which
        // walks a character at a time and keeps nothing, so formatting a file
        // that held one wrote the file back without it.
        if rest.starts_with("/*") {
            let end = rest.find("*/").map(|at| at + 2).unwrap_or(rest.len());
            held.push(Trivia::Comment(rest[..end].to_string()));
            rest = &rest[end..];
            continue;
        }
        let mut chars = rest.char_indices();
        let step = chars.next().map(|(at, held)| at + held.len_utf8());
        rest = &rest[step.unwrap_or(rest.len())..];
    }
    held
}

/// Whether a token opens or closes a nesting level, which is what the
/// indentation of a line is counted from.
fn nesting(token: &Token) -> i32 {
    match token {
        Token::LeftBrace | Token::LeftParentheses | Token::LeftBracket => 1,
        Token::RightBrace | Token::RightParentheses | Token::RightBracket => -1,
        _ => 0,
    }
}

/// What a token is doing, where the token alone does not say.
///
/// Three spellings carry two jobs each, and which job decides the spacing.
/// `Arena<64>` holds the tokens of `a < b > c`; `^` is a pointer type in front
/// of a type and a read behind a value; `[N]u8` is a type where `xs[i]` is an
/// index. Each is settled once here, over the token stream.
#[derive(Clone, Copy, PartialEq)]
enum Role {
    Plain,
    /// The `<` and `>` around a generic's arguments.
    TypeArgumentOpen,
    TypeArgumentClose,
    /// A `^` in front of the type it points to, against one behind the value it
    /// reads through.
    PointerType,
    /// A `[` that opens an array or slice type rather than an index, and the `]`
    /// that closes it.
    TypeBracketOpen,
    TypeBracketClose,
    /// The `::` that declares a name, against the one that names a variant of a
    /// type. `Point :: struct { ... }` declares and `NodeKind::Var` names, and
    /// the space is what the grammar tells them apart by.
    Declares,
}

/// Whether a token can end a value, which separates a read from a type and a
/// subtraction from a negation.
fn ends_a_value(token: Option<&Token>) -> bool {
    matches!(
        token,
        Some(
            Token::Identifier(_)
                | Token::Integer(_)
                | Token::Float(_)
                | Token::Float32(_)
                | Token::StringLiteral(_)
                | Token::RightParentheses
                | Token::RightBracket
                | Token::RightBrace
        )
    )
}

/// Whether a token can begin a type.
fn begins_a_type(token: Option<&Token>) -> bool {
    matches!(
        token,
        Some(
            Token::Identifier(_)
                | Token::Caret
                | Token::LeftBracket
                | Token::Function
                | Token::Dollar
                | Token::Ref
        )
    )
}

/// Whether the brace this token sits directly inside opens the values a type
/// names under itself, which is where a `::` declares rather than reaching into
/// a type.
///
/// The block is recognized by what opens it, the way both compilers recognize
/// it: a name and a `::`. A body reaching a name that way writes `Enum::Variant`
/// at the head of a line, which is why the enclosing brace is asked about rather
/// than the line.
fn inside_values_block(
    tokens: &[Token],
    opens_line: &[bool],
    brace_depth: &[i32],
    index: usize,
) -> bool {
    if brace_depth[index] != 1 {
        return false;
    }
    let Some(open) = (0..index).rev().find(|held| {
        matches!(tokens[*held], Token::LeftBrace) && brace_depth[*held] == 0
    }) else {
        return false;
    };
    if !matches!(tokens.get(open + 1), Some(Token::Identifier(_)))
        || !matches!(tokens.get(open + 2), Some(Token::DoubleColon))
    {
        return false;
    }
    // Which declaration the brace belongs to. Only a type declaration may name
    // values under itself, and a function body opening with a constant is the
    // same two tokens, so the word after the declaration's `::` is what tells
    // the two apart.
    let Some(head) = (0..open).rev().find(|held| {
        brace_depth[*held] == 0
            && opens_line[*held]
            && matches!(tokens[*held], Token::Identifier(_))
            && matches!(tokens.get(held + 1), Some(Token::DoubleColon))
    }) else {
        return false;
    };
    !matches!(tokens.get(head + 2), Some(Token::Function))
}

/// The job each token is doing.
fn roles(
    tokens: &[Token],
    opens_line: &[bool],
    brace_depth: &[i32],
) -> Vec<Role> {
    let mut held = vec![Role::Plain; tokens.len()];
    for index in 0..tokens.len() {
        match &tokens[index] {
            // A declaration is a name at the head of its line followed by `::`.
            // Anywhere else the `::` reaches into a type for one of the names
            // declared under it.
            // Only at the top level, where declarations live. Inside a body a
            // name at the head of its line is an expression, and
            // `TokenKind::Ident` standing alone as a function's answer is one.
            Token::DoubleColon
                if index >= 1
                    && opens_line[index - 1]
                    && matches!(tokens[index - 1], Token::Identifier(_))
                    && (brace_depth[index] == 0
                        || inside_values_block(
                            tokens,
                            opens_line,
                            brace_depth,
                            index,
                        )) =>
            {
                held[index] = Role::Declares;
            }
            // A `<` opens a generic's arguments when a name is in front of it and
            // a matching `>` closes a run holding only what a type argument can
            // be. `a < b` fails that on the first token that could not.
            Token::LessThan
                if matches!(
                    tokens.get(index.wrapping_sub(1)),
                    Some(Token::Identifier(_))
                ) =>
            {
                if let Some(close) = type_arguments_close(tokens, index) {
                    held[index] = Role::TypeArgumentOpen;
                    held[close] = Role::TypeArgumentClose;
                }
            }
            // A caret in front of something that starts a type is a pointer to
            // it. Behind a value it is a read through one.
            Token::Caret
                if !ends_a_value(tokens.get(index.wrapping_sub(1))) =>
            {
                held[index] = Role::PointerType;
            }
            // `[]T` and `[N]T` are types: a `[` with no value in front of it.
            Token::LeftBracket
                if !ends_a_value(tokens.get(index.wrapping_sub(1))) =>
            {
                if let Some(close) = matching(tokens, index)
                    && begins_a_type(tokens.get(close + 1))
                {
                    held[index] = Role::TypeBracketOpen;
                    held[close] = Role::TypeBracketClose;
                }
            }
            _ => {}
        }
    }
    held
}

/// Where the bracket opened at `open` closes.
fn matching(tokens: &[Token], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (index, token) in tokens.iter().enumerate().skip(open) {
        depth += nesting(token);
        if depth == 0 {
            return Some(index);
        }
    }
    None
}

/// Where a run of type arguments opened at `open` closes, or nothing when the
/// run holds something no type argument can be.
fn type_arguments_close(tokens: &[Token], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (index, token) in tokens.iter().enumerate().skip(open) {
        match token {
            Token::LessThan => depth += 1,
            Token::GreaterThan => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            Token::Identifier(_)
            | Token::Integer(_)
            | Token::Comma
            | Token::Caret
            | Token::LeftBracket
            | Token::RightBracket
            | Token::Dollar => {}
            // Anything else says this was a comparison after all.
            _ => return None,
        }
    }
    None
}

/// Whether these two tokens have a space between them.
///
/// The corpus is the definition: `a + b`, `f(x, y)`, `Name { field = 1 }`,
/// `x: i64`, `p^`, `^i64`, `xs[i]`, `[]u8`, `Slab<T, N>`, `-1` where the minus
/// signs a value and `a - b` where it subtracts one.
fn spaced(
    before: Option<&Token>,
    left: (&Token, Role),
    right: (&Token, Role),
) -> bool {
    let (left, left_role) = left;
    let (right, right_role) = right;
    // A name reaches into a type tight, and a declaration is spaced.
    if matches!(left, Token::DoubleColon) {
        return left_role == Role::Declares;
    }
    if matches!(right, Token::DoubleColon) {
        return right_role == Role::Declares;
    }
    // A generic's arguments are written tight, and so is the name in front of
    // them: `Slab<T, N>`, `Arena<256>`.
    if left_role == Role::TypeArgumentOpen
        || right_role == Role::TypeArgumentOpen
        || left_role == Role::TypeArgumentClose
        || right_role == Role::TypeArgumentClose
    {
        // What follows a closed argument list is spaced as if the list were a
        // name, so `Arena<256> = ...` keeps its spaces around the `=`.
        if left_role == Role::TypeArgumentClose {
            return !matches!(
                right,
                Token::LeftParentheses
                    | Token::LeftBracket
                    | Token::RightParentheses
                    | Token::Comma
                    | Token::Semicolon
                    | Token::Colon
                    | Token::LeftBrace
            ) && !matches!(right_role, Role::TypeArgumentOpen);
        }
        return false;
    }
    // A pointer type binds to the type behind it and a read binds to the value
    // in front of it, so a caret takes its space on the side facing away.
    if left_role == Role::PointerType {
        return false;
    }
    if matches!(left, Token::Caret) && right_role != Role::PointerType {
        // A read through a pointer is a value, so what follows it is spaced the
        // way it would be after a name.
        return !matches!(
            right,
            Token::RightParentheses
                | Token::RightBracket
                | Token::Comma
                | Token::Semicolon
                | Token::Colon
                | Token::LeftParentheses
                | Token::LeftBracket
                | Token::Dot
        );
    }
    if matches!(right, Token::Caret) && right_role != Role::PointerType {
        return false;
    }
    // `[]u8` and `[N]u8`: the brackets of a type and the type they are in front
    // of are written tight.
    if left_role == Role::TypeBracketOpen
        || right_role == Role::TypeBracketClose
    {
        return false;
    }
    if left_role == Role::TypeBracketClose {
        return false;
    }
    // Nothing sits inside a bracket on the open side, and nothing in front of a
    // close or a separator.
    if matches!(
        right,
        Token::RightParentheses
            | Token::RightBracket
            | Token::Comma
            | Token::Semicolon
            | Token::Colon
    ) {
        return false;
    }
    // A literal opening directly inside a bracket keeps a space, so the two
    // openers do not run together: `[ { hp = 1 }, { hp = 2 } ]`.
    if matches!(left, Token::LeftParentheses | Token::LeftBracket) {
        return matches!(right, Token::LeftBrace);
    }
    if matches!(right, Token::LeftParentheses | Token::LeftBracket) {
        // A name or a closing bracket in front of an open one is a call or an
        // index and takes no space; a keyword takes one, apart from the two that
        // read as calls: `fn()` and `struct(...)`.
        return !matches!(
            left,
            Token::Identifier(_)
                | Token::RightParentheses
                | Token::RightBracket
                | Token::Function
                | Token::Struct
        );
    }
    // A minus or a bang in front of a value signs or negates it.
    if matches!(left, Token::Minus | Token::Bang) && !ends_a_value(before) {
        return false;
    }
    // A dot binds to what is either side of it, so `a.b` and the `.Variant`
    // that takes its enum from the context are both written tight. After
    // `case` it is a pattern rather than a value, and the word and the pattern
    // are two things: `case .Circle { radius }:` is how the book writes one and
    // how a reader reads one.
    if matches!(left, Token::Case) {
        return true;
    }
    if matches!(left, Token::Dot) || matches!(left, Token::Dollar) {
        return false;
    }
    if matches!(right, Token::Dot) {
        // A dot that opens a value rather than reaching into one takes its
        // space in front, the way it does after `case`. What is on its left is
        // what tells them apart: `key == .Left` names a value and `z.x` reaches
        // into one, and an operator does not end a value.
        return !ends_a_value(Some(left));
    }
    // `{}` with nothing in it, against `{ field = 1 }` with something.
    if matches!(left, Token::LeftBrace) && matches!(right, Token::RightBrace) {
        return false;
    }
    true
}

/// Whether a line opens a declaration, which at the top level is the only thing
/// that is not a continuation of the line above. A declaration head is a name
/// followed by `::`, or one of the three words that open a line of their own.
fn opens_a_declaration(token: &Token, next: Option<&Token>) -> bool {
    // `export` and `test` are words rather than keywords, so they are read as
    // the names they are.
    if matches!(token, Token::Import) {
        return true;
    }
    if let Token::Identifier(name) = token
        && (name == "export" || name == "test")
    {
        return true;
    }
    matches!(token, Token::Identifier(_))
        && matches!(next, Some(Token::DoubleColon))
}

/// How wide a line is laid out to.
///
/// The corpus was written to it by hand before anything measured it, and the
/// Rust beside it is held to the same number by `rustfmt.toml`.
const WIDTH: usize = 80;

/// Where the bracket opened at `open` closes, when what it opens is something a
/// layout may break.
///
/// A list of things separated by commas is: the arguments of a call, the
/// elements of an array, the fields of a value. Where each one goes is a
/// question of width, and a comma says where one ends.
///
/// Anything else keeps the breaks it was written with. A bracket holding one
/// expression has no place to break that the author did not choose, and
/// `(a == b\n    || c == d)` reads the way it does because someone decided
/// where the `||` goes.
///
/// A brace has the further question of whether it holds a value or a block,
/// since joining two statements onto one line would say something the author
/// did not. That is read off what is inside it rather than off what is in front
/// of it, because `match k {` and `Point {` both have a name there.
fn reflowable(tokens: &[Token], open: usize) -> Option<usize> {
    let close = matching(tokens, open)?;
    let listed = elements_of(tokens, open, close).len() > 1;
    match tokens[open] {
        Token::LeftParentheses | Token::LeftBracket if listed => Some(close),
        Token::LeftBrace if holds_fields(tokens, open, close) => Some(close),
        _ => None,
    }
}

/// Whether every element between these braces is `name = value`, which is what
/// a value written out looks like and what no block does.
///
/// A comma at the top level is what says this is a list at all. Statements are
/// separated by their lines, so a block has none, and without asking for one a
/// `for` body opening with `at = report_run(fmt, at)` read as a field and the
/// whole body was laid out as a value.
/// What the name is spelled with is not asked. A field may be called `type`,
/// which one lexer hands back as a keyword and the other as an identifier, and
/// a rule reading the difference made the two lay `BufferBindingLayout` out
/// differently. The second token being `=` is what says this is a field, and
/// the grammar says the first is its name.
fn holds_fields(tokens: &[Token], open: usize, close: usize) -> bool {
    let elements = elements_of(tokens, open, close);
    elements.len() > 1
        && elements.iter().all(|element| {
            element.end > element.start + 1
                && matches!(tokens.get(element.start + 1), Some(Token::Assign))
        })
}

/// The elements between a bracket and the one closing it, each holding the
/// comma that ends it. The comma travels with the element so that a run written
/// out over several lines has exactly the commas the source had, including a
/// trailing one where the author wrote it.
fn elements_of(
    tokens: &[Token],
    open: usize,
    close: usize,
) -> Vec<std::ops::Range<usize>> {
    let mut held = Vec::new();
    let mut start = open + 1;
    let mut depth = 0i32;
    for (index, token) in tokens.iter().enumerate().take(close).skip(open + 1) {
        depth += nesting(token);
        if depth == 0 && matches!(token, Token::Comma) {
            held.push(start..index + 1);
            start = index + 1;
        }
    }
    if start < close {
        held.push(start..close);
    }
    held
}

/// Whether a line beginning with this token continues the line above rather than
/// beginning something new.
fn continues_a_line(token: &Token) -> bool {
    matches!(
        token,
        Token::Plus
            | Token::Asterisk
            | Token::Slash
            | Token::Percent
            | Token::And
            | Token::Or
            | Token::Dot
            | Token::Equal
            | Token::NotEqual
            | Token::LessThan
            | Token::LessThanOrEqual
            | Token::GreaterThan
            | Token::GreaterThanOrEqual
            | Token::Pipe
            | Token::Ampersand
            | Token::Arrow
            | Token::Question
    )
}

/// The one rendering of a source.
///
/// Answers the source unchanged when the lexer cannot get through it.
pub fn format(source: &str) -> String {
    let Some((tokens, held)) = extents(source) else {
        return source.to_string();
    };
    if tokens.is_empty() {
        // A file of nothing but comments and space still has its comments.
        let kept: Vec<String> = trivia_in(source)
            .iter()
            .filter_map(|held| match held {
                Trivia::Comment(text) => Some(text.clone()),
                Trivia::Break => None,
            })
            .collect();
        if kept.is_empty() {
            return String::new();
        }
        return format!("{}\n", kept.join("\n"));
    }

    let mut out = String::with_capacity(source.len());
    // What the author indented each line by, indexed by line number. A blank
    // line and a comment line hold no token, so the line a token sits on is
    // found from its offset rather than counted off as tokens are written.
    // Which tokens are the first on their line, which separates a
    // declaration's `::` from a variant's.
    let mut opens_line = Vec::with_capacity(held.len());
    let mut brace_depth = Vec::with_capacity(held.len());
    let mut previous_end = 0usize;
    let mut counted = 0i32;
    let decided = decided_here(&tokens, &held, source);
    let joined = joined_tokens(&tokens, &held, source);
    for (index, extent) in held.iter().enumerate() {
        let gap = &source[previous_end..extent.start];
        // Which tokens begin a line of the output, which is what says a `::` is
        // declaring a name rather than reaching into a type for one. Read off
        // the breaks this file decides to write rather than off the ones it was
        // handed: the two agree wherever a declaration can be written, and
        // asking the input would be a rule about the output answered by
        // something else.
        opens_line.push(
            index == 0
                || (gap.contains(LINE_BREAK)
                    && !joined[index]
                    && !decided[index]),
        );
        if matches!(tokens[index], Token::RightBrace) {
            counted -= 1;
        }
        brace_depth.push(counted);
        if matches!(tokens[index], Token::LeftBrace) {
            counted += 1;
        }
        previous_end = extent.end;
    }
    let roles = roles(&tokens, &opens_line, &brace_depth);
    let layout = Layout {
        source,
        tokens: &tokens,
        extents: &held,
        roles: &roles,
        joined: &joined,
    };
    // Braces are the statement nesting and the other brackets are an expression
    // running over more than one line. A line inside an unclosed bracket is
    // indented past the line that opened it, and a line that continues an
    // expression with no bracket open is indented one level past it.
    let mut braces: i32 = 0;
    let mut brackets: i32 = 0;
    // The run of tokens waiting to be laid out, what it is indented by, and the
    // comments that sit at the end of the last line it will be written over.
    let mut pending: Option<std::ops::Range<usize>> = None;
    let mut pending_indent = 0usize;
    let mut trailing: Vec<String> = Vec::new();

    let mut at = 0usize;
    for (index, extent) in held.iter().enumerate() {
        let gap = &source[at..extent.start];
        at = extent.end;
        let token = &tokens[index];
        // A closing bracket belongs to the level it closes, so it is counted
        // before the line it opens is indented.
        let line_depth = if nesting(token) < 0 {
            braces - 1
        } else {
            braces
        };

        let joins_the_line_above = joined[index];

        // A comment in front of the first break in the gap sits at the end of
        // the line above rather than on one of its own.
        let pieces = trivia_in(gap);
        let mut read = 0usize;
        while let Some(Trivia::Comment(text)) = pieces.get(read) {
            if pending.is_none() {
                break;
            }
            trailing.push(text.clone());
            read += 1;
        }
        let rest = &pieces[read..];
        let wrote_break =
            rest.iter().any(|piece| matches!(piece, Trivia::Break));
        let starts_line = index == 0
            || (wrote_break
                && !joins_the_line_above
                && !(decided[index] && pending.is_some()));

        if starts_line {
            if let Some(range) = pending.take() {
                flush(&layout, range, pending_indent, &mut trailing, &mut out);
            }
            let mut breaks = 0usize;
            for piece in rest {
                match piece {
                    Trivia::Break => {
                        // One blank line between two lines, never more. Nothing
                        // in the corpus separates two things by two.
                        if breaks == 1 && !out.is_empty() {
                            out.push('\n');
                        }
                        breaks += 1;
                    }
                    Trivia::Comment(text) => {
                        indent(&mut out, line_depth);
                        out.push_str(text);
                        out.push('\n');
                        breaks = 0;
                    }
                }
            }
            // One level per enclosing brace, one more for an expression still
            // inside a bracket, and one more for a line that continues the one
            // above with no bracket holding it open.
            let open_braces = if matches!(token, Token::RightBrace) {
                braces - 1
            } else {
                braces
            };
            let open_brackets = if matches!(
                token,
                Token::RightParentheses | Token::RightBracket
            ) {
                brackets - 1
            } else {
                brackets
            };
            let running_on = if open_brackets > 0 {
                open_brackets
            } else {
                // At the top level the only line that is not a continuation is
                // one opening a declaration, which is how the names of an
                // `export` list running over six lines stay indented under it.
                // Inside a block a statement opens with a name as often as not,
                // so there it is the operators that say a line runs on.
                if open_braces == 0 {
                    // A brace is a block, not a line running on. Neither is a
                    // bracket closing a list the declaration opened: it closes
                    // back at the indentation the opener sat at, which at the
                    // top level is none. Without the two on the right here, a
                    // signature broken over several lines closed its parameter
                    // list one level in, and formatting that again indented it
                    // further.
                    let closes = matches!(
                        token,
                        Token::RightBrace
                            | Token::LeftBrace
                            | Token::RightParentheses
                            | Token::RightBracket
                    );
                    i32::from(
                        !closes
                            && !opens_a_declaration(
                                token,
                                tokens.get(index + 1),
                            ),
                    )
                } else {
                    i32::from(continues_a_line(token))
                }
            };
            pending_indent =
                (open_braces.max(0) + running_on.max(0)) as usize * 4;
            pending = Some(index..index + 1);
        } else if let Some(range) = pending.as_mut() {
            range.end = index + 1;
        }
        match token {
            Token::LeftBrace => braces += 1,
            Token::RightBrace => braces -= 1,
            Token::LeftParentheses | Token::LeftBracket => brackets += 1,
            Token::RightParentheses | Token::RightBracket => brackets -= 1,
            _ => {}
        }
    }
    // Whatever followed the last token, written the way the trivia between two
    // tokens is: a comment in front of the first break sits at the end of the
    // line that token is on, and every other one keeps a line of its own.
    let tail = trivia_in(&source[at..]);
    let mut read = 0usize;
    while let Some(Trivia::Comment(text)) = tail.get(read) {
        if pending.is_none() {
            break;
        }
        trailing.push(text.clone());
        read += 1;
    }
    if let Some(range) = pending.take() {
        flush(&layout, range, pending_indent, &mut trailing, &mut out);
    }
    let mut breaks = 0usize;
    for piece in &tail[read..] {
        match piece {
            Trivia::Break => {
                // The line the last run sits on was ended by writing it, so the
                // first break here is that ending rather than a blank line.
                if breaks == 1 {
                    out.push('\n');
                }
                breaks += 1;
            }
            Trivia::Comment(text) => {
                indent(&mut out, braces);
                out.push_str(text);
                out.push('\n');
                breaks = 0;
            }
        }
    }
    while out.ends_with('\n') || out.ends_with(' ') {
        out.pop();
    }
    out.push('\n');
    out
}

fn indent(out: &mut String, depth: i32) {
    out.push_str(&" ".repeat(depth.max(0) as usize * 4));
}

/// Which tokens sit at the end of the line above whatever they were written on.
///
/// A brace that opens a block, and an `else` beside the brace that closes the
/// arm above it. A comment in between keeps its own line, so the join only
/// happens across whitespace.
fn joined_tokens(
    tokens: &[Token],
    extents: &[Extent],
    source: &str,
) -> Vec<bool> {
    let mut held = vec![false; tokens.len()];
    for index in 1..tokens.len() {
        let gap = &source[extents[index - 1].end..extents[index].start];
        if gap.contains("//") {
            continue;
        }
        held[index] = match tokens[index] {
            Token::LeftBrace => !matches!(
                tokens[index - 1],
                Token::LeftBrace
                    | Token::RightBrace
                    | Token::Semicolon
                    | Token::Comma
            ),
            Token::Else => matches!(tokens[index - 1], Token::RightBrace),
            _ => false,
        };
    }
    held
}

/// A run of tokens written out, with whatever sat at the end of its last line
/// put back there.
fn flush(
    layout: &Layout<'_>,
    range: std::ops::Range<usize>,
    indent: usize,
    trailing: &mut Vec<String>,
    out: &mut String,
) {
    let mut held = String::new();
    layout.write(range, indent, &mut held);
    if !trailing.is_empty() {
        while held.ends_with('\n') {
            held.pop();
        }
        for text in trailing.iter() {
            held.push(' ');
            held.push_str(text);
        }
        held.push('\n');
        trailing.clear();
    }
    out.push_str(&held);
}

/// Whether the break in front of each token is one a layout decides rather than
/// one the author wrote.
///
/// A break inside a round or square bracket, or inside a brace holding fields,
/// is a layout's to make: the tokens either side of it are one expression and
/// where it goes is a question of width. A break at the top level separates two
/// statements, and which statement is on which line is the author's.
///
/// A bracket holding a comment keeps every break it was written with. A comment
/// says which line it belongs to and a width does not know. So does one holding
/// a block: a list may be written inside a block and a block inside a list, and
/// joining the statements of one onto a line is what a layout may not do.
///
/// Either of those puts the breaks back for the run it sits in and for every
/// run around that one, since a group half of whose tokens keep their breaks is
/// no longer a group anything can lay out.
fn decided_here(
    tokens: &[Token],
    extents: &[Extent],
    source: &str,
) -> Vec<bool> {
    /// A bracket still open: where it opened, when what it opened is a run a
    /// layout owns, and whether something inside it has since said otherwise.
    struct Open {
        at: Option<usize>,
        blocked: bool,
    }
    let mut held = vec![false; tokens.len()];
    let mut opens: Vec<Open> = Vec::new();
    for index in 0..tokens.len() {
        let inside = opens.last().is_some_and(|open| open.at.is_some());
        held[index] = inside;
        match tokens[index] {
            Token::LeftParentheses | Token::LeftBrace | Token::LeftBracket => {
                let opens_a_run = reflowable(tokens, index).is_some();
                opens.push(Open {
                    // Inside a run already, so the whole thing travels
                    // together, and a block among it is what stops that.
                    at: if inside || opens_a_run {
                        Some(index)
                    } else {
                        None
                    },
                    blocked: inside && !opens_a_run,
                });
            }
            Token::RightParentheses
            | Token::RightBrace
            | Token::RightBracket => {
                let closed = opens.pop();
                // The bracket that closes a run belongs to that run rather than
                // to whatever holds it, so the break in front of it is the same
                // layout's to make. Read off the enclosing run instead, a
                // literal written over several lines had its closing brace on a
                // line of its own, and the run it closed was never a whole
                // group for anything to lay out.
                let inside_now =
                    opens.last().is_some_and(|open| open.at.is_some());
                held[index] =
                    closed.as_ref().is_some_and(|open| open.at.is_some())
                        || inside_now;
                let Some(open) = closed.as_ref().and_then(|open| open.at)
                else {
                    continue;
                };
                let commented = (open + 1..=index).any(|at| {
                    source[extents[at - 1].end..extents[at].start]
                        .contains("//")
                });
                if !commented && !closed.is_some_and(|open| open.blocked) {
                    continue;
                }
                for slot in held.iter_mut().take(index + 1).skip(open + 1) {
                    *slot = false;
                }
                if let Some(around) = opens.last_mut() {
                    around.blocked = true;
                }
            }
            _ => {}
        }
    }
    held
}

/// What a run of tokens needs to be written out.
struct Layout<'a> {
    source: &'a str,
    tokens: &'a [Token],
    extents: &'a [Extent],
    roles: &'a [Role],
    /// The tokens a space goes in front of whatever the spacing rules say: the
    /// brace that opens a block, and the `else` beside the brace closing the
    /// arm above it. Both sit at the end of a line they were not written on, so
    /// the space is what puts them there rather than a rule about the pair.
    joined: &'a [bool],
}

impl Layout<'_> {
    /// A run of tokens on one line, spaced the way the corpus spaces them.
    fn flat(&self, range: std::ops::Range<usize>) -> String {
        let mut held = String::new();
        for index in range.clone() {
            if index > range.start {
                // The token two back signs a minus and negates a bang, and the
                // one that opened the line has nothing behind it.
                let before = if index - 1 > range.start && index >= 2 {
                    Some(&self.tokens[index - 2])
                } else {
                    None
                };
                if self.joined[index]
                    || spaced(
                        before,
                        (&self.tokens[index - 1], self.roles[index - 1]),
                        (&self.tokens[index], self.roles[index]),
                    )
                {
                    held.push(' ');
                }
            }
            let extent = &self.extents[index];
            held.push_str(&self.source[extent.start..extent.end]);
        }
        held
    }

    /// The first bracket at the top level of this run that a layout may break,
    /// and where it closes.
    fn breakable(
        &self,
        range: std::ops::Range<usize>,
    ) -> Option<(usize, usize)> {
        let mut depth = 0i32;
        for index in range.clone() {
            if depth == 0
                && let Some(close) = reflowable(self.tokens, index)
                && close < range.end
            {
                return Some((index, close));
            }
            depth += nesting(&self.tokens[index]);
        }
        None
    }

    /// Whether one element of a run is a plain number, which is what says the
    /// run is a block of them rather than a list worth a line each. A leading
    /// minus is part of the number, and the comma that ends the element is not
    /// part of what is being asked about.
    fn is_scalar(&self, element: std::ops::Range<usize>) -> bool {
        let mut held = element;
        if matches!(self.tokens.get(held.end - 1), Some(Token::Comma)) {
            held.end -= 1;
        }
        matches!(
            &self.tokens[held.clone()],
            [Token::Integer(_) | Token::Float(_) | Token::Float32(_)]
                | [
                    Token::Minus,
                    Token::Integer(_) | Token::Float(_) | Token::Float32(_),
                ]
        )
    }

    /// A run of numbers written out at `indent`, as many to a line as fit.
    fn fill(
        &self,
        parts: &[std::ops::Range<usize>],
        indent: usize,
        out: &mut String,
    ) {
        let mut line = String::new();
        for element in parts {
            let piece = self.flat(element.clone());
            let width = piece.chars().count();
            if !line.is_empty()
                && indent + line.chars().count() + 1 + width > WIDTH
            {
                out.push_str(&" ".repeat(indent));
                out.push_str(&line);
                out.push('\n');
                line.clear();
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(&piece);
        }
        if !line.is_empty() {
            out.push_str(&" ".repeat(indent));
            out.push_str(&line);
            out.push('\n');
        }
    }

    /// This run written out at `indent`, over as many lines as it takes.
    ///
    /// One line when it fits. Otherwise the outermost bracket is opened at the
    /// end of a line, each of the elements inside it is written at one level
    /// further in, and the bracket closes a line of its own back at the
    /// indentation it opened at. An element too long for its own line is asked
    /// the same question again, so a call inside a call breaks only as far as
    /// it has to.
    fn write(
        &self,
        range: std::ops::Range<usize>,
        indent: usize,
        out: &mut String,
    ) {
        let flat = self.flat(range.clone());
        if indent + flat.chars().count() <= WIDTH {
            out.push_str(&" ".repeat(indent));
            out.push_str(&flat);
            out.push('\n');
            return;
        }
        // Nothing here opens a bracket, so there is nowhere to break that the
        // author did not write. A long run of text is left long rather than
        // broken somewhere it would read worse.
        let Some((open, close)) = self.breakable(range.clone()) else {
            out.push_str(&" ".repeat(indent));
            out.push_str(&flat);
            out.push('\n');
            return;
        };
        out.push_str(&" ".repeat(indent));
        out.push_str(&self.flat(range.start..open + 1));
        out.push('\n');
        let parts = elements_of(self.tokens, open, close);
        // A run of plain numbers is filled to the width rather than given a
        // line per element. A line per element says something about a list
        // whose elements are worth reading one at a time; sixteen floats are a
        // block, and a column of them says nothing the block does not.
        if parts.len() > 1
            && parts.iter().all(|element| self.is_scalar(element.clone()))
        {
            self.fill(&parts, indent + 4, out);
            self.write(close..range.end, indent, out);
            return;
        }
        for element in parts {
            self.write(element, indent + 4, out);
        }
        // What follows the bracket is asked the same question, so a call
        // written on the answer of another breaks rather than running long:
        // `f(a, b).g(c, d)` puts `).g(` at the head of a line of its own. The
        // run shrinks by at least the head each time, so this settles.
        self.write(close..range.end, indent, out);
    }
}

/// Whether a source is already what `format` writes.
pub fn formatted(source: &str) -> bool {
    format(source) == source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_and_gaps_reconstruct_a_source() {
        let source =
            "// a note\nmain :: fn() -> i64 {\n    x := 1  // why\n    x\n}\n";
        let pieces = tokens_and_gaps(source).expect("a lexable source");
        assert_eq!(pieces.concat(), source);
    }

    // A formatter that drops what it does not recognize is a formatter that
    // deletes source. Reading only `//` left a block comment's bytes to the
    // step that walks a character at a time and keeps nothing, so a file
    // holding one came back without it, and nothing said so.
    #[test]
    fn a_block_comment_survives_being_formatted() {
        let source = "/* At the top.
   A second line, indented on purpose. */

main :: fn() -> i64 {
    /* inside */
    x := 1 /* beside a statement */
    x
}
";
        assert_eq!(format(source), source);
    }

    #[test]
    fn formatting_is_idempotent() {
        let source = "main::fn()->i64{\n  x:=1\n      y  :=  2\n  x+y\n}\n";
        let once = format(source);
        assert_eq!(format(&once), once, "formatted twice differs:\n{once}");
    }

    // A signature broken over several lines closes its parameter list at the
    // indentation the declaration opened at. Reading that close as a line
    // running on indented it one level, and formatting the result indented it
    // again, so the tree could never be what the formatter writes.
    #[test]
    fn a_broken_parameter_list_closes_where_it_opened() {
        let source = "one :: fn(\n    a: i64,\n    held: []i64,\n    count: i64\n) -> []i64 {\n    held\n}\n";
        assert_eq!(format(source), source);
        let indented = "one :: fn(\n    a: i64,\n    held: []i64,\n    count: i64\n    ) -> []i64 {\n    held\n}\n";
        assert_eq!(format(indented), source);
    }

    #[test]
    fn indentation_follows_nesting() {
        let source =
            "main :: fn() -> i64 {\nx := 1\nif (x > 0) {\ny := 2\n}\nx\n}\n";
        assert_eq!(
            format(source),
            "main :: fn() -> i64 {\n    x := 1\n    if (x > 0) {\n        y := 2\n    }\n    x\n}\n"
        );
    }

    #[test]
    fn a_comment_keeps_its_text_and_its_line() {
        let source = "// leading\nmain :: fn() -> i64 {\n        // inside\n    0  // trailing\n}\n";
        let formatted = format(source);
        assert!(formatted.contains("// leading\n"));
        assert!(formatted.contains("    // inside\n"));
        assert!(formatted.contains("0 // trailing\n"));
    }

    #[test]
    fn a_run_of_blank_lines_becomes_one() {
        let source = "a :: 1\n\n\n\n\nb :: 2\n";
        assert_eq!(format(source), "a :: 1\n\nb :: 2\n");
    }

    #[test]
    fn trailing_space_goes_and_a_final_newline_arrives() {
        assert_eq!(format("a :: 1   \n   \n"), "a :: 1\n");
        assert_eq!(format("a :: 1"), "a :: 1\n");
    }

    #[test]
    fn a_byte_order_mark_does_not_come_back() {
        assert_eq!(format("\u{feff}a :: 1\n"), "a :: 1\n");
    }

    // Which line a token is on is meaning, so a formatter never moves one. The
    // leading `-` that opens a statement and the `+` that continues an
    // expression both survive.
    #[test]
    fn lines_never_move() {
        let source = "main :: fn() -> i64 {\n    x := 1\n        + 2\n    y := 3\n    -1\n}\n";
        let formatted = format(source);
        assert_eq!(
            formatted.lines().count(),
            source.lines().count(),
            "a line moved:\n{formatted}"
        );
        assert!(formatted.contains("\n        + 2\n"), "{formatted}");
        assert!(formatted.contains("\n    -1\n"), "{formatted}");
    }

    // `Point :: struct` declares and `NodeKind::Var` names one of the names
    // declared under a type. The space separates them.
    #[test]
    fn a_declaration_is_spaced_and_a_variant_is_tight() {
        let source = "Kind :: enum { Var }
main :: fn() -> i64 {
    k := Kind::Var
    0
}
";
        assert_eq!(format(source), source);
    }

    // A brace opening a block sits at the end of the line that says what the
    // block is for, and an `else` sits beside the brace closing the arm above.
    #[test]
    fn a_brace_joins_the_line_it_opens_a_block_for() {
        let source = "main :: fn() -> i64
{
    if (1 > 0)
    {
        1
    }
    else
    {
        0
    }
}
";
        assert_eq!(
            format(source),
            "main :: fn() -> i64 {
    if (1 > 0) {
        1
    } else {
        0
    }
}
"
        );
    }

    #[test]
    fn a_comment_keeps_a_brace_on_its_own_line() {
        let source = "main :: fn() -> i64
// why
{
    0
}
";
        assert!(format(source).contains(
            "// why
{
"
        ));
    }

    #[test]
    fn a_string_literal_is_left_alone() {
        let source = "greeting :: \"a  b   c\"\n";
        assert_eq!(format(source), source);
    }

    #[test]
    fn a_file_of_comments_keeps_them() {
        assert_eq!(format("// one\n// two\n"), "// one\n// two\n");
    }
}
