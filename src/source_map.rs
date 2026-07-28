use std::sync::{OnceLock, RwLock};

// Which file a `Position` came from, kept beside the positions rather than in
// them, because a `Position` is `Copy` and threaded through every pass and a
// path is neither. A file id indexes this table, and 0 means "not recorded",
// which is the entry file and anything the tests lex directly.
//
// This is process-wide, which is what a source map is. One program is compiled
// per process, and every position in it means the same thing everywhere.
fn table() -> &'static RwLock<Vec<String>> {
    static TABLE: OnceLock<RwLock<Vec<String>>> = OnceLock::new();
    TABLE.get_or_init(|| RwLock::new(Vec::new()))
}

// Records a file and returns its id, which is never 0 for a recorded file. The
// same path registered twice gets the same id, so a diamond import does not
// grow the table.
pub fn register(name: &str) -> u32 {
    let mut files = match table().write() {
        Ok(files) => files,
        Err(poisoned) => poisoned.into_inner(),
    };
    if let Some(index) = files.iter().position(|held| held == name) {
        return index as u32 + 1;
    }
    files.push(name.to_string());
    files.len() as u32
}

// Where a module's text is on disk, keyed by the name a diagnostic shows.
//
// The two are different on purpose. A diagnostic says `std/ecs.frost`, which is
// what the reader wrote and what stays the same whoever is compiling; reading
// the line back needs the absolute path, which does not. Keeping the second
// beside the first is what lets a failure show the line it is about without
// every pass carrying the text around.
fn paths() -> &'static RwLock<Vec<(String, String)>> {
    static PATHS: OnceLock<RwLock<Vec<(String, String)>>> = OnceLock::new();
    PATHS.get_or_init(|| RwLock::new(Vec::new()))
}

pub fn register_at(name: &str, path: &str) -> u32 {
    let id = register(name);
    let mut known = match paths().write() {
        Ok(known) => known,
        Err(poisoned) => poisoned.into_inner(),
    };
    if !known.iter().any(|(held, _)| held == name) {
        known.push((name.to_string(), path.to_string()));
    }
    id
}

pub fn path_of(name: &str) -> Option<String> {
    let known = match paths().read() {
        Ok(known) => known,
        Err(poisoned) => poisoned.into_inner(),
    };
    known
        .iter()
        .find(|(held, _)| held == name)
        .map(|(_, path)| path.clone())
}

pub fn name_of(file: u32) -> Option<String> {
    if file == 0 {
        return None;
    }
    let files = match table().read() {
        Ok(files) => files,
        Err(poisoned) => poisoned.into_inner(),
    };
    files.get(file as usize - 1).cloned()
}

/// Puts a position in front of a message, in the one shape everything reads.
///
/// `at <path>:<line>:<column>: <message>` is what the renderer looks for, and
/// what a pass writes when it knows where it is. A message that already carries
/// one is left alone, so wrapping twice does not stutter, and a default
/// position means the pass did not know, which is better said by leaving it off
/// than by naming line zero.
pub fn locate<T>(
    result: anyhow::Result<T>,
    position: crate::lexer::Position,
) -> anyhow::Result<T> {
    result.map_err(|error| {
        let text = crate::imports::demangle_private_names(&error.to_string());
        if position == crate::lexer::Position::default()
            || text.starts_with("at ")
        {
            anyhow::anyhow!("{text}")
        } else {
            anyhow::anyhow!("at {}: {text}", position.describe())
        }
    })
}
