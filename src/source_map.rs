use std::sync::{OnceLock, RwLock};

// Which file a `Position` came from, kept beside the positions rather than in
// them, because a `Position` is `Copy` and threaded through every pass and a
// path is neither. A file id indexes this table, and 0 means "not recorded",
// which is the entry file and anything the tests lex directly.
//
// This is process-wide, which is what a source map is: one program is compiled
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
