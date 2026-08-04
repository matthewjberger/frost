// How a failure reaches the reader.
//
// Every pass that knows where it is says so by starting its message with
// `at <path>:<line>:<column>: `, and that was as far as it went: one line,
// naming a place the reader then had to go and look at. The self-hosted
// compiler has printed the line and a caret under the column for a long time,
// and there is no reason for two formats, so this is that one.
//
// The source is read back at the time of the failure rather than kept, which
// costs nothing because it only happens when something has already gone wrong,
// and it means no pass has to carry text around to be able to say where it is.

use std::fmt::Write as _;

// A located failure, the unit every recovering pass answers with. The
// position names where, the message says what, and rendering happens at the
// boundary rather than in the pass that found it. A related entry is a
// second place the failure is about, "moved here" beside a use-after-move,
// and it renders as another located line in the one format both compilers
// print, so nothing downstream learns a new shape.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub position: crate::lexer::Position,
    pub message: String,
    pub related: Vec<(crate::lexer::Position, String)>,
}

impl Diagnostic {
    pub fn new(position: crate::lexer::Position, message: String) -> Self {
        Self {
            position,
            message,
            related: Vec::new(),
        }
    }

    /// The whole report as located lines: the message, then each related
    /// place on a line of its own. A message that already says where it is,
    /// or one with nowhere to point, prints as it stands, which is the rule
    /// `locate` has always applied.
    pub fn rendered(&self) -> String {
        let mut out = if self.message.starts_with("at ")
            || self.position == crate::lexer::Position::default()
        {
            self.message.clone()
        } else {
            format!("at {}: {}", self.position.describe(), self.message)
        };
        for (position, note) in &self.related {
            out.push('\n');
            out.push_str(&format!("at {}: {note}", position.describe()));
        }
        out
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.rendered())
    }
}

// An error that knows where it happened. A bail site that can see the
// offending token stamps that token's position into one of these, and the
// recovery loop reads it back out instead of stamping wherever the cursor
// stopped, which after a long declaration is lines past the mistake.
#[derive(Debug)]
pub struct LocatedError {
    pub position: crate::lexer::Position,
    pub message: String,
}

impl std::fmt::Display for LocatedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "at {}: {}", self.position.describe(), self.message)
    }
}

impl std::error::Error for LocatedError {}

/// The whole of what a failed compile prints.
///
/// The chain's innermost message is the one with something to say; the outer
/// ones are the phase it happened in, which the position already implies. A
/// message may carry several located lines, since a pass that reports every
/// function rather than stopping at the first joins them.
pub fn render(error: &anyhow::Error) -> String {
    let chain: Vec<String> =
        error.chain().map(|held| held.to_string()).collect();
    let innermost = chain.last().cloned().unwrap_or_default();
    let mut out = String::new();
    for line in innermost.lines() {
        match located(line) {
            Some((path, row, column, message)) => {
                render_located(&mut out, &path, row, column, &message)
            }
            None => {
                let _ = writeln!(out, "frost: {line}");
            }
        }
    }
    if out.is_empty() {
        let _ = writeln!(out, "frost: {innermost}");
    }
    out
}

/// Splits `at <path>:<line>:<column>: <message>` into its parts.
///
/// A path may hold colons, `C:/frost/std/ecs.frost` among them, so the split is
/// from the right: the last two colon-separated pieces before the message are
/// the line and the column, and whatever came before them is the path.
fn located(line: &str) -> Option<(String, usize, usize, String)> {
    let rest = line.strip_prefix("at ")?;
    let (head, message) = rest.split_once(": ")?;
    let (head, column) = head.rsplit_once(':')?;
    let (path, row) = head.rsplit_once(':')?;
    Some((
        path.to_string(),
        row.parse().ok()?,
        column.parse().ok()?,
        message.to_string(),
    ))
}

fn render_located(
    out: &mut String,
    path: &str,
    row: usize,
    column: usize,
    message: &str,
) {
    let _ = writeln!(out, "{path}:{row}:{column}:");
    let on_disk =
        crate::source_map::path_of(path).unwrap_or_else(|| path.to_string());
    let Some(text) = std::fs::read_to_string(&on_disk).ok().and_then(|held| {
        held.lines().nth(row.saturating_sub(1)).map(str::to_string)
    }) else {
        // The file has moved or was never on disk, which is what a test that
        // lexes a string in memory looks like. The message still stands.
        let _ = writeln!(out, "    ^ {message}");
        return;
    };
    let _ = writeln!(out, "{text}");
    // A tab in the source is one column to the compiler and eight to a
    // terminal, so it is carried into the caret line rather than counted, and
    // the caret lands under the same character either way.
    let mut caret = String::new();
    for held in text.chars().take(column.saturating_sub(1)) {
        caret.push(if held == '\t' { '\t' } else { ' ' });
    }
    let _ = writeln!(out, "{caret}^ {message}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_a_located_message_whose_path_holds_a_colon() {
        let (path, row, column, message) =
            located("at C:/frost/std/ecs.frost:12:5: something is wrong")
                .expect("a located line");
        assert_eq!(path, "C:/frost/std/ecs.frost");
        assert_eq!(row, 12);
        assert_eq!(column, 5);
        assert_eq!(message, "something is wrong");
    }

    #[test]
    fn leaves_a_message_with_no_position_alone() {
        assert!(located("something is wrong").is_none());
    }

    #[test]
    fn renders_a_caret_under_the_column() {
        let mut out = String::new();
        render_located(&mut out, "no/such/file.frost", 3, 5, "wrong");
        assert_eq!(out, "no/such/file.frost:3:5:\n    ^ wrong\n");
    }
}
