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

/// Where a report lands and what it says, as the reader is shown it.
///
/// A message that already says where it is carries its own place, which is the
/// one printed, so that is the pair two reports are the same by. The ownership
/// rules are walked twice, once over the source and once over the bodies
/// specialization expands, and for a program with no generic in it both walks
/// answer the same.
///
/// A pass that reports a whole item's failure names the item and then repeats
/// what the walk inside it said, which is already located, so the prefixes
/// nest: `at f:4:1: at f:7:5: ...`. Stripping until nothing is left to strip
/// reaches the place the reader is pointed at and the words alone.
fn shown_as(diagnostic: &Diagnostic) -> (String, &str) {
    let mut place = None;
    let mut message = diagnostic.message.as_str();
    while let Some((named, without)) = leading_place(message) {
        place = Some(named);
        message = without;
    }
    (
        place.unwrap_or_else(|| diagnostic.position.describe()),
        message,
    )
}

/// A message with any `at <place>: ` prefix taken off it.
///
/// A fault raised inside an instance carries the place in the template it came
/// from, and the report is shown at the call that asked for the instance. Two
/// places in one report renders as the first with the second read as part of
/// the claim, so a reader is handed a sentence beginning with a file name.
pub fn without_leading_place(message: &str) -> String {
    match leading_place(message) {
        Some((_, rest)) => rest.to_string(),
        None => message.to_string(),
    }
}

/// The place an `at ...: ` prefix names, and what follows it.
///
/// Both spellings `Position::describe` produces: a file that is known reads
/// `path:line:column`, and one that is not reads `line N, column M`, which is
/// what a program held in memory and the compiler's own tests look like.
fn leading_place(message: &str) -> Option<(String, &str)> {
    let rest = message.strip_prefix("at ")?;
    let (named, without) = rest.split_once(": ")?;
    let numbered = named.rsplit(':').take(2).all(|part| {
        !part.is_empty() && part.bytes().all(|held| held.is_ascii_digit())
    });
    let described = named.starts_with("line ") && named.contains(", column ");
    if !numbered && !described {
        return None;
    }
    Some((named.to_string(), without))
}

/// One fault per thing that is wrong.
///
/// Two reports of the same words about the same place are one fault, whichever
/// walks found them. The same words about different places are one fault with
/// several places: an undeclared name used in six functions is one thing to fix
/// and six places it shows, and printing it six times buries the five other
/// faults the run found. A report carrying an edit is left alone, since the
/// edit belongs to the place it was made for and a reader applying them wants
/// one per place.
pub fn grouped(diagnostics: Vec<Diagnostic>) -> Vec<Diagnostic> {
    let mut kept: Vec<Diagnostic> = Vec::new();
    let mut seen: Vec<(String, String)> = Vec::new();
    for diagnostic in diagnostics {
        let (place, _) = shown_as(&diagnostic);
        let claim = claim_of(&diagnostic).to_string();
        if seen.contains(&(place.clone(), claim.clone())) {
            continue;
        }
        seen.push((place.clone(), claim.clone()));
        let same_words = kept.iter_mut().find(|held| {
            claim_of(held) == claim
                && crate::tools::fixes::edit_for(held).is_none()
        });
        match same_words {
            Some(held) => {
                // The places a folded report named come with it. Two uses of a
                // moved value each point at the line the value went on, and
                // those are different lines, so keeping only the claim drops
                // half of what the second report had to say.
                let at = shown_position(&diagnostic);
                held.related.push((at, claim));
                held.related.extend(diagnostic.related);
            }
            None => kept.push(diagnostic),
        }
    }
    kept
}

/// What a report claims, which is the first line of what it says.
///
/// A pass that reports a whole item's failure repeats the walk's whole report,
/// the other places it named included, as one message. The claim is the line
/// that says what is wrong; the lines under it are places, and two reports of
/// one fault do not stop being one because one of them wrote its places out as
/// text.
fn claim_of(diagnostic: &Diagnostic) -> &str {
    let (_, message) = shown_as(diagnostic);
    message.split('\n').next().unwrap_or(message)
}

/// The place a report is shown at, as a position.
///
/// A message that carries its own place is shown there rather than at the
/// position recorded beside it, so that is the place it contributes when it
/// joins another report as one more place the fault shows.
fn shown_position(diagnostic: &Diagnostic) -> crate::lexer::Position {
    let (place, _) = shown_as(diagnostic);
    let Some((path, line, column)) = numbered(&place) else {
        return diagnostic.position;
    };
    let Some(file) = crate::source_map::id_of(&path) else {
        return diagnostic.position;
    };
    crate::lexer::Position { line, column, file }
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

/// A report as a program reads it.
///
/// One object per report, one report per line, so a reader of the stream needs
/// no bracket matching and a run that is still going has already said what it
/// found. The place is where the caret report puts the caret: the file, the line
/// and column, and the same place counted in bytes from the start of the file,
/// since an editor applying an edit works in bytes. `span` is that offset and
/// where the text the report is about ends; for a report that names a point
/// rather than a range the two are the same number.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Report {
    pub file: Option<String>,
    pub line: usize,
    pub column: usize,
    pub span: (usize, usize),
    pub severity: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related: Vec<Place>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fix: Option<Replacement>,
}

/// Another place one report is about.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Place {
    pub file: Option<String>,
    pub line: usize,
    pub column: usize,
    pub span: (usize, usize),
    pub message: String,
}

/// An edit that answers a report: the bytes to replace and what to put there.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Replacement {
    pub file: Option<String>,
    pub span: (usize, usize),
    pub replacement: String,
    /// Whether `frost fix` applies it without being asked twice.
    pub certain: bool,
}

/// Where a report points and what it says, resolved to a file and a place in
/// it. A message that carries its own place is read for that place, since that
/// is the one the caret report shows.
fn placed(diagnostic: &Diagnostic) -> (Option<String>, usize, usize, String) {
    let (place, message) = shown_as(diagnostic);
    let message = message.to_string();
    if let Some((path, line, column)) = numbered(&place) {
        return (Some(path), line, column, message);
    }
    (
        None,
        diagnostic.position.line,
        diagnostic.position.column,
        message,
    )
}

/// `path:line:column`, split from the right so a path holding a colon survives.
fn numbered(place: &str) -> Option<(String, usize, usize)> {
    let (head, column) = place.rsplit_once(':')?;
    let (path, line) = head.rsplit_once(':')?;
    Some((path.to_string(), line.parse().ok()?, column.parse().ok()?))
}

/// The byte the line and column of a known file name.
fn offset_in(path: &Option<String>, line: usize, column: usize) -> usize {
    let Some(path) = path else {
        return 0;
    };
    let on_disk = crate::source_map::path_of(path).unwrap_or(path.clone());
    let Ok(source) = std::fs::read_to_string(&on_disk) else {
        return 0;
    };
    crate::tools::fixes::byte_offset(&source, line, column).unwrap_or(0)
}

/// One report, as the object a program reads.
pub fn as_report(diagnostic: &Diagnostic, severity: &'static str) -> Report {
    let (file, line, column, message) = placed(diagnostic);
    let at = offset_in(&file, line, column);
    let related = diagnostic
        .related
        .iter()
        .map(|(position, note)| {
            let named = crate::source_map::name_of(position.file);
            let at = offset_in(&named, position.line, position.column);
            Place {
                file: named,
                line: position.line,
                column: position.column,
                span: (at, at),
                message: note.clone(),
            }
        })
        .collect();
    let fix = crate::tools::fixes::edit_for(diagnostic).map(|edit| {
        let named = crate::source_map::name_of(edit.position.file)
            .or_else(|| file.clone());
        let start = offset_in(&named, edit.position.line, edit.position.column);
        Replacement {
            file: named,
            span: (start, start + edit.replaces),
            replacement: edit.replacement,
            certain: edit.certain,
        }
    });
    Report {
        file,
        line,
        column,
        span: (at, at),
        severity: severity.to_string(),
        message,
        related,
        fix,
    }
}

/// Every report as one line of JSON each.
pub fn as_json(diagnostics: &[Diagnostic], severity: &'static str) -> String {
    let mut out = String::new();
    for diagnostic in diagnostics {
        let report = as_report(diagnostic, severity);
        // A report that cannot be written as JSON would be a report lost, so
        // the message goes out as itself rather than as nothing.
        match serde_json::to_string(&report) {
            Ok(line) => {
                out.push_str(&line);
                out.push('\n');
            }
            Err(_) => {
                let _ = writeln!(out, "{{\"message\":\"unprintable\"}}");
            }
        }
    }
    out
}

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

    fn at(line: usize, message: &str) -> Diagnostic {
        Diagnostic::new(
            crate::lexer::Position {
                line,
                column: 1,
                file: 0,
            },
            message.to_string(),
        )
    }

    #[test]
    fn the_same_words_about_the_same_place_are_one_fault() {
        let walked = at(7, "use of moved value 'held'");
        let again = Diagnostic::new(
            crate::lexer::Position {
                line: 4,
                column: 1,
                file: 0,
            },
            "at line 7, column 1: use of moved value 'held'".to_string(),
        );
        let kept = grouped(vec![walked, again]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].position.line, 7);
    }

    #[test]
    fn the_same_words_about_two_places_are_one_fault_with_two_places() {
        let kept = grouped(vec![
            at(2, "'Absent' is not a type this program declares"),
            at(9, "'Absent' is not a type this program declares"),
        ]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].related.len(), 1);
        assert_eq!(kept[0].related[0].0.line, 9);
    }

    #[test]
    fn two_faults_stay_two() {
        let kept = grouped(vec![
            at(2, "unknown variable 'one'"),
            at(9, "unknown variable 'other'"),
        ]);
        assert_eq!(kept.len(), 2);
    }

    // A report carrying an edit stays one per place: the edit was made for the
    // place it names, and a reader applying them wants one for each.
    #[test]
    fn a_fixable_fault_is_not_folded() {
        let message = "`mut` marks a parameter that writes the caller's value; a local that is reassigned is declared with `var`";
        let kept = grouped(vec![at(2, message), at(9, message)]);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn renders_a_caret_under_the_column() {
        let mut out = String::new();
        render_located(&mut out, "no/such/file.frost", 3, 5, "wrong");
        assert_eq!(out, "no/such/file.frost:3:5:\n    ^ wrong\n");
    }
}
