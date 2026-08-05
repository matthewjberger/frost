// The exported surface of a project, and the nearest name to one that is not
// there.
//
// A flat namespace gives up the narrowing a `.` provides in a method language:
// a wrong guess is an undefined name rather than a shortened list. `frost api
// <prefix>` is that list, asked for by the prefix a family shares, and the
// suggestion below is the narrowing applied to a name already written.

use std::path::{Path, PathBuf};

/// One exported name, as it is declared.
#[derive(serde::Serialize)]
pub struct Exported {
    pub name: String,
    pub file: String,
    pub line: usize,
    /// The declaration head as written, without the brace that opens its body.
    pub signature: String,
}

/// Every exported name in these files whose name begins with `prefix`.
///
/// The signature is read off the source rather than rebuilt from the syntax, so
/// what is printed is what the author wrote: the parameter names, the modes and
/// the `where` clause included.
pub fn exported(files: &[PathBuf], prefix: &str) -> Vec<Exported> {
    let mut found = Vec::new();
    for path in files {
        let Ok(source) = std::fs::read_to_string(path) else {
            continue;
        };
        let shown = path
            .file_name()
            .map(|held| held.to_string_lossy().to_string())
            .unwrap_or_default();
        let names = exports_of(&source);
        let lines: Vec<&str> = source.lines().collect();
        for (number, line) in lines.iter().enumerate() {
            let Some(name) = declaration_head(line) else {
                continue;
            };
            if !name.starts_with(prefix) || !names.contains(&name) {
                continue;
            }
            found.push(Exported {
                name,
                file: shown.clone(),
                line: number + 1,
                signature: signature_at(&lines, number),
            });
        }
    }
    found.sort_by(|one, other| one.name.cmp(&other.name));
    found
}

/// The names an `export` line hands to the files that import this one.
fn exports_of(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut reading = false;
    for line in source.lines() {
        let text = line.trim();
        if let Some(rest) = text.strip_prefix("export ") {
            reading = true;
            names.extend(listed(rest));
            if !rest.trim_end().ends_with(',') {
                reading = false;
            }
            continue;
        }
        if reading {
            names.extend(listed(text));
            if !text.trim_end().ends_with(',') {
                reading = false;
            }
        }
    }
    names
}

fn listed(text: &str) -> Vec<String> {
    text.split(',')
        .map(|held| held.trim().to_string())
        .filter(|held| {
            !held.is_empty()
                && held
                    .chars()
                    .all(|held| held.is_alphanumeric() || held == '_')
        })
        .collect()
}

/// The name a declaration head declares, or nothing for any other line.
fn declaration_head(line: &str) -> Option<String> {
    if line.starts_with(char::is_whitespace) {
        return None;
    }
    let (name, _) = line.split_once("::")?;
    let name = name.trim();
    if name.is_empty()
        || !name
            .chars()
            .all(|held| held.is_alphanumeric() || held == '_')
    {
        return None;
    }
    // `Type::Variant` is written tight and a declaration is spaced, which is
    // the rule the grammar tells them apart by.
    if !line[name.len()..].starts_with(' ') {
        return None;
    }
    Some(name.to_string())
}

/// A declaration head, carried over the lines its parameters run onto.
fn signature_at(lines: &[&str], number: usize) -> String {
    let mut text = lines[number].to_string();
    let mut depth = brackets_in(lines[number]);
    let mut at = number;
    while depth > 0 && at + 1 < lines.len() && at - number < 12 {
        at += 1;
        text.push('\n');
        text.push_str(lines[at]);
        depth += brackets_in(lines[at]);
    }
    text.trim_end().trim_end_matches('{').trim_end().to_string()
}

fn brackets_in(line: &str) -> i32 {
    let mut depth = 0;
    for held in line.chars() {
        match held {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            _ => {}
        }
    }
    depth
}

/// The name a reader most likely meant, out of the names that exist.
///
/// A name sharing the prefix comes first, since a prefix is what a family is
/// named by here and a wrong guess inside one is the common miss. Failing that,
/// the nearest by edit distance, and only when one name is nearer than every
/// other: an ambiguous suggestion is worse than none, because a reader who
/// takes it has to work out that it was a guess.
pub fn nearest<'a>(wanted: &str, known: &[&'a str]) -> Option<&'a str> {
    let mut ranked: Vec<(usize, &str)> = known
        .iter()
        .filter(|held| **held != wanted)
        .map(|held| (distance(wanted, held), *held))
        .filter(|(held, _)| *held * 3 <= wanted.len().max(1) * 2)
        .collect();
    ranked.sort_by(|one, other| one.0.cmp(&other.0).then(one.1.cmp(other.1)));
    let (best, name) = *ranked.first()?;
    if ranked.iter().filter(|(held, _)| *held == best).count() > 1 {
        return None;
    }
    Some(name)
}

/// How many single-character edits turn one name into the other.
fn distance(from: &str, to: &str) -> usize {
    let from: Vec<char> = from.chars().collect();
    let to: Vec<char> = to.chars().collect();
    let mut row: Vec<usize> = (0..=to.len()).collect();
    for (one, held) in from.iter().enumerate() {
        let mut previous = row[0];
        row[0] = one + 1;
        for (other, against) in to.iter().enumerate() {
            let cost = usize::from(held != against);
            let replaced = previous + cost;
            previous = row[other + 1];
            row[other + 1] =
                (row[other] + 1).min(row[other + 1] + 1).min(replaced);
        }
    }
    row[to.len()]
}

/// Every `.frost` file under a directory.
pub fn sources(directory: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(next) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&next) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let skipped = path
                    .file_name()
                    .map(|held| {
                        held == "target"
                            || held == ".git"
                            || held == ".frost-build"
                    })
                    .unwrap_or(false);
                if !skipped {
                    stack.push(path);
                }
            } else if path.extension().is_some_and(|kind| kind == "frost") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_one_edit_away_is_suggested() {
        let known = ["vec_push", "vec_pop", "map_get"];
        assert_eq!(nearest("vec_puhs", &known), Some("vec_push"));
    }

    #[test]
    fn a_name_nothing_is_near_gets_no_suggestion() {
        let known = ["vec_push", "map_get"];
        assert_eq!(nearest("frobnicate_the_widget", &known), None);
    }

    // Two names equally near is a guess, and a guess a reader takes without
    // checking is worse than no suggestion.
    #[test]
    fn an_ambiguous_suggestion_is_withheld() {
        let known = ["vec_get", "vec_set"];
        assert_eq!(nearest("vec_bet", &known), None);
    }

    #[test]
    fn an_export_list_is_read_over_its_lines() {
        let source = "export one, two,\n    three\nfour :: 1\n";
        assert_eq!(exports_of(source), vec!["one", "two", "three"]);
    }

    #[test]
    fn a_declaration_head_is_spaced_and_a_variant_is_not() {
        assert_eq!(
            declaration_head("vec_push :: fn(x: i64) {"),
            Some("vec_push".to_string())
        );
        assert_eq!(declaration_head("    held := Kind::Var"), None);
    }
}
