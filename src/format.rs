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

// A space inside a line is one space where the author left any and none where
// they left none.
//
// Which token pairs take a space is a question about the parse, not about the
// pair: `Arena<64>` holds the tokens of two comparisons, `-> ^i64` and `p^ = 20`
// differ only in what the caret attaches to, and `[N]u8` is a type where
// `xs[i] + 1` is an index. This runs without a parse, so that a file whose parse
// failed still formats, and collapsing a run of spaces is the whole of what can
// be said without one.

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
    let mut depth: i32 = 0;
    // Whether the last thing written was a line break, so the next thing needs
    // this line's indentation in front of it.
    let mut at_line_start = true;
    // What the author indented each line by, indexed by line number. A blank
    // line and a comment line hold no token, so the line a token sits on is
    // found from its offset rather than counted off as tokens are written.
    let written_indents = original_indentation(source);
    let starts = line_starts(source);

    let mut at = 0usize;
    for (index, extent) in held.iter().enumerate() {
        let gap = &source[at..extent.start];
        at = extent.end;
        let token = &tokens[index];
        // A closing bracket belongs to the level it closes, so it is counted
        // before the line it opens is indented.
        let closing = nesting(token) < 0;
        let line_depth = if closing { depth - 1 } else { depth };

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
            // The nesting the line sits at, or the author's indentation when
            // that is deeper. A line indented past its nesting is a continuation
            // of the line above, and how far it runs on is the author's: an
            // `export` list, a condition broken after `||` and a call whose
            // arguments run down the page each pick their own depth.
            let line = starts.partition_point(|held| *held <= extent.start) - 1;
            let written = written_indents.get(line).copied().unwrap_or(0);
            let wanted = std::cmp::max(written, line_depth.max(0) as usize * 4);
            out.push_str(&" ".repeat(wanted));
        } else if index > 0 && gap.contains(|held: char| held.is_whitespace()) {
            out.push(' ');
        }
        out.push_str(&source[extent.start..extent.end]);
        at_line_start = false;
        depth += nesting(token);
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
                    indent(&mut out, depth);
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

/// How far each line of the source is indented, in spaces, one entry per line.
fn original_indentation(source: &str) -> Vec<usize> {
    source
        .lines()
        .map(|line| line.len() - line.trim_start().len())
        .collect()
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
