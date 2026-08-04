// What `frost fmt` writes.
//
// A conservative normalizer, not a rewrapper. Which line a token is on is
// meaning in this language: a `+` at a line break continues the expression above
// it and a leading `-` opens a new statement, so a formatter that moved a token
// to another line would change what the program says. Nothing here joins or
// splits a line. What it settles is the space inside a line, the indentation in
// front of it, how many blank lines may sit between two of them, and that the
// file ends with a newline.
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
/// index. Each is settled once here, over the token stream, so the rules below
/// ask a question with an answer instead of guessing from a pair.
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

/// Whether a token can end a value, which is what tells a read from a type and a
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
                    && brace_depth[index] == 0
                    && opens_line[index - 1]
                    && matches!(tokens[index - 1], Token::Identifier(_)) =>
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
    if matches!(left, Token::LeftParentheses | Token::LeftBracket) {
        return false;
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
    if matches!(left, Token::Dot) || matches!(right, Token::Dot) {
        return false;
    }
    if matches!(left, Token::Dollar) {
        return false;
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
/// Answers the source unchanged when the lexer cannot get through it, since a
/// formatter that cannot read a file has nothing to say about it and writing a
/// guess over it would lose the file.
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
    // Whether the last thing written was a line break, so the next thing needs
    // this line's indentation in front of it.
    let mut at_line_start = true;
    // Whether the token last written opened its line. A minus that opens a line
    // opens a statement, so it signs the value after it however the line above
    // ended.
    let mut previous_opened_line = false;
    // What the author indented each line by, indexed by line number. A blank
    // line and a comment line hold no token, so the line a token sits on is
    // found from its offset rather than counted off as tokens are written.
    // Which tokens are the first on their line, which is what tells a
    // declaration's `::` from a variant's.
    let mut opens_line = Vec::with_capacity(held.len());
    let mut brace_depth = Vec::with_capacity(held.len());
    let mut previous_end = 0usize;
    let mut counted = 0i32;
    for (index, extent) in held.iter().enumerate() {
        let gap = &source[previous_end..extent.start];
        opens_line.push(index == 0 || gap.contains(LINE_BREAK));
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
    // Braces are the statement nesting and the other brackets are an expression
    // running over more than one line. A line inside an unclosed bracket is
    // indented past the line that opened it, and a line that continues an
    // expression with no bracket open is indented one level past it.
    let mut braces: i32 = 0;
    let mut brackets: i32 = 0;

    let mut at = 0usize;
    for (index, extent) in held.iter().enumerate() {
        let gap = &source[at..extent.start];
        at = extent.end;
        let token = &tokens[index];
        // A closing bracket belongs to the level it closes, so it is counted
        // before the line it opens is indented.
        let mut opens_this_line = false;
        let line_depth = if nesting(token) < 0 {
            braces - 1
        } else {
            braces
        };

        let mut breaks = 0usize;
        for piece in trivia_in(gap) {
            match piece {
                Trivia::Break => {
                    // Trailing space is never written, so a break is a break
                    // whatever came before it on the line.
                    while out.ends_with(' ') {
                        out.pop();
                    }
                    // One blank line between two lines, never more. Nothing in
                    // the corpus separates two things by two.
                    if breaks < 2 || index == 0 {
                        out.push('\n');
                    }
                    breaks += 1;
                    at_line_start = true;
                }
                Trivia::Comment(text) => {
                    if at_line_start {
                        indent(&mut out, line_depth);
                    } else if !out.is_empty() {
                        out.push(' ');
                    }
                    out.push_str(&text);
                    at_line_start = false;
                    // A blank line is two breaks with nothing between them, so
                    // anything written resets the count and a block of comment
                    // lines keeps every one of its breaks.
                    breaks = 0;
                }
            }
        }

        if at_line_start {
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
                    // A brace closing a declaration ends it rather than running
                    // it on.
                    let closes = matches!(token, Token::RightBrace);
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
            let wanted = (open_braces.max(0) + running_on.max(0)) as usize * 4;
            out.push_str(&" ".repeat(wanted));
            opens_this_line = true;
        } else if index > 0
            && spaced(
                if previous_opened_line {
                    None
                } else {
                    index.checked_sub(2).map(|held| &tokens[held])
                },
                (&tokens[index - 1], roles[index - 1]),
                (token, roles[index]),
            )
        {
            out.push(' ');
        }
        out.push_str(&source[extent.start..extent.end]);
        at_line_start = false;
        previous_opened_line = opens_this_line;
        match token {
            Token::LeftBrace => braces += 1,
            Token::RightBrace => braces -= 1,
            Token::LeftParentheses | Token::LeftBracket => brackets += 1,
            Token::RightParentheses | Token::RightBracket => brackets -= 1,
            _ => {}
        }
    }

    // Whatever followed the last token, and then the newline every file ends
    // with. A comment after the last token keeps its own line, so the breaks
    // here are written the same way as the breaks between tokens.
    let mut breaks = 0usize;
    for piece in trivia_in(&source[at..]) {
        match piece {
            Trivia::Break => {
                while out.ends_with(' ') {
                    out.pop();
                }
                if breaks < 2 {
                    out.push('\n');
                }
                breaks += 1;
            }
            Trivia::Comment(text) => {
                if out.ends_with('\n') {
                    indent(&mut out, braces);
                } else {
                    out.push(' ');
                }
                out.push_str(&text);
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

    #[test]
    fn formatting_is_idempotent() {
        let source = "main::fn()->i64{\n  x:=1\n      y  :=  2\n  x+y\n}\n";
        let once = format(source);
        assert_eq!(format(&once), once, "formatted twice differs:\n{once}");
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
    // declared under a type. The space is what tells them apart, so a formatter
    // that spaced both would rewrite every variant in the tree.
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
