// The edit a report's remedy is.
//
// A diagnostic that tells the reader what to write can hand over the writing
// instead: where the edit goes, how much of the source it stands in for, and
// the text to put there. That is what `--diagnostics=json` carries and what
// `frost fix` applies.
//
// Not every remedy is an edit. "use a parameter mode (`mut` to write, `move` to
// take, unmarked to read), or a raw pointer `^T`" names four things a reader
// might have meant and no one of them is the answer, so it carries no edit and
// says so by carrying none. The rule is that an edit is offered only where the
// remedy is a single piece of text at a single place.

use crate::diagnostic::Diagnostic;
use crate::lexer::Position;

/// What to write, and where.
#[derive(Debug, Clone, PartialEq)]
pub struct Edit {
    /// Where the replaced text begins.
    pub position: Position,
    /// How many bytes at that place the replacement stands in for. Zero is an
    /// insertion, and a replacement with empty text is a deletion.
    pub replaces: usize,
    pub replacement: String,
    /// Whether applying it unread is safe. A replacement the compiler derived
    /// from the one thing the reader can have meant is; a token a recovering
    /// parser guessed at is not, because recovery may have stopped somewhere
    /// other than where the mistake is, and `frost fix` leaves those alone.
    pub certain: bool,
}

const MUT_LOCAL: &str = "`mut` marks a parameter that writes the caller's value; a local that is reassigned is declared with `var`";

/// The edit that answers this report, when it has one.
///
/// Read off the report's own place and words. Every report that carries an edit
/// comes from the parser, which knows the token it is about, so there is no
/// case here where the place printed and the place recorded are two different
/// ones.
pub fn edit_for(diagnostic: &Diagnostic) -> Option<Edit> {
    let position = diagnostic.position;
    let message = diagnostic.message.as_str();
    if message == MUT_LOCAL {
        return Some(Edit {
            position,
            replaces: "mut".len(),
            replacement: "var".to_string(),
            certain: true,
        });
    }
    // A statement cannot begin with this token, so the token is the mistake.
    // Deleting it is what a reader does with a stray bracket, and it is a
    // guess: the same report is what a missing operator two lines up looks
    // like.
    if let Some(written) = quoted_after(message, "expected a statement, found ")
    {
        return Some(Edit {
            position,
            replaces: written.len(),
            replacement: String::new(),
            certain: false,
        });
    }
    // The parser knows the one token that belongs here. Where it belongs is
    // where the parser stopped, which is right unless recovery had already
    // walked past the mistake, so this is offered and not applied unread.
    if let Some(wanted) = expected_token(message) {
        return Some(Edit {
            position,
            replaces: 0,
            replacement: wanted,
            certain: false,
        });
    }
    None
}

/// The text between the first pair of single quotes after `prefix`.
fn quoted_after(message: &str, prefix: &str) -> Option<String> {
    let rest = message.strip_prefix(prefix)?;
    let rest = rest.strip_prefix('\'')?;
    let (written, _) = rest.split_once('\'')?;
    Some(written.to_string())
}

/// The one token an `Expected 'x'` report names, when it names exactly one.
///
/// Punctuation only. A report that wants "an identifier" or "a statement" names
/// a kind of thing rather than a thing to write, and there is no text an edit
/// could put there.
fn expected_token(message: &str) -> Option<String> {
    let rest = message.strip_prefix("Expected '")?;
    let (wanted, _) = rest.split_once('\'')?;
    let punctuation = wanted.chars().all(|held| {
        matches!(held, ')' | ']' | '}' | '>' | ',' | ';' | ':' | '=' | '(')
    });
    if wanted.is_empty() || !punctuation {
        return None;
    }
    Some(wanted.to_string())
}

/// Where a place is in a file, counted in bytes from its start.
///
/// A column is counted in characters, so a line holding anything outside ASCII
/// puts the two counts apart, and an edit is applied to bytes. Answers `None`
/// when the file has no such place, which is what a report about a program held
/// in memory looks like.
pub fn byte_offset(source: &str, line: usize, column: usize) -> Option<usize> {
    if line == 0 || column == 0 {
        return None;
    }
    let mut offset = 0usize;
    for (number, text) in source.split_inclusive('\n').enumerate() {
        if number + 1 == line {
            let within = text
                .char_indices()
                .nth(column - 1)
                .map(|(at, _)| at)
                .unwrap_or_else(|| text.trim_end_matches(['\n', '\r']).len());
            return Some(offset + within);
        }
        offset += text.len();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(message: &str) -> Diagnostic {
        Diagnostic::new(
            Position {
                line: 3,
                column: 5,
                file: 0,
            },
            message.to_string(),
        )
    }

    #[test]
    fn a_mut_local_is_a_var() {
        let edit = edit_for(&report(MUT_LOCAL)).expect("an edit");
        assert_eq!(edit.replacement, "var");
        assert_eq!(edit.replaces, 3);
        assert!(edit.certain);
    }

    #[test]
    fn a_choice_of_remedies_is_not_an_edit() {
        assert!(
            edit_for(&report(
                "a reference is not a surface type; use a parameter mode (`mut` to write, `move` to take, unmarked to read), or a raw pointer `^T`"
            ))
            .is_none()
        );
    }

    #[test]
    fn a_wanted_bracket_is_an_insertion() {
        let edit = edit_for(&report("Expected ')' after while condition"))
            .expect("an edit");
        assert_eq!(edit.replacement, ")");
        assert_eq!(edit.replaces, 0);
        assert!(!edit.certain);
    }

    #[test]
    fn a_wanted_kind_of_thing_is_not_an_edit() {
        assert!(edit_for(&report("Expected identifier after 'ref'")).is_none());
    }

    #[test]
    fn a_stray_token_is_a_deletion() {
        let edit = edit_for(&report("expected a statement, found ']'"))
            .expect("an edit");
        assert_eq!(edit.replacement, "");
        assert_eq!(edit.replaces, 1);
    }

    #[test]
    fn a_column_is_a_byte_offset_in_the_line_it_is_on() {
        let source = "one\ntwo\nthree\n";
        assert_eq!(byte_offset(source, 1, 1), Some(0));
        assert_eq!(byte_offset(source, 2, 1), Some(4));
        assert_eq!(byte_offset(source, 3, 3), Some(10));
        assert_eq!(byte_offset(source, 9, 1), None);
    }
}
