use std::path::{Path, PathBuf};
use std::process::Command;

// Every fenced Frost block in the book, compiled. A doc that shows code is
// making a claim about what the compiler accepts, and the only way to hold it
// to that is to hand the code to the compiler.
//
// The fence says what the block is:
//
//   ```frost           a program, or a run of top-level declarations. Compiled
//                      as written, with a `main` appended when it declares
//                      none. It carries its own imports, so it stands alone.
//   ```frost,inside    statements that belong in a function. Wrapped in a
//                      `main` and compiled.
//   ```frost,refused   what the compiler will not accept. Compiled, and the
//                      compile has to fail.
//   ```frost,sketch    a shape rather than a program: an ellipsis, a library
//                      that is not in this tree, a form being proposed. Not
//                      compiled, and saying so is what keeps the rest honest.
struct Block {
    file: PathBuf,
    line: usize,
    kind: String,
    body: String,
}

fn markdown_files() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut found = Vec::new();
    let mut stack = vec![root.join("docs").join("book").join("src")];
    while let Some(next) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&next) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|held| held == "md") {
                found.push(path);
            }
        }
    }
    found.push(root.join("README.md"));
    found.sort();
    found
}

fn blocks_in(file: &Path) -> Vec<Block> {
    let Ok(text) = std::fs::read_to_string(file) else {
        return Vec::new();
    };
    let lines: Vec<&str> = text.lines().collect();
    let mut found = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let Some(info) = lines[index].strip_prefix("```") else {
            index += 1;
            continue;
        };
        let info = info.trim().to_string();
        let start = index + 1;
        index += 1;
        while index < lines.len() && !lines[index].starts_with("```") {
            index += 1;
        }
        if info.starts_with("frost") {
            found.push(Block {
                file: file.to_path_buf(),
                line: start,
                kind: info,
                body: lines[start..index].join("\n"),
            });
        }
        index += 1;
    }
    found
}

// What is handed to the compiler for one block, or nothing for a block the
// fence says is a shape rather than a program.
fn program_of(block: &Block) -> Option<String> {
    let declares_main = block
        .body
        .lines()
        .any(|line| line.trim_start().starts_with("main ::"));
    match block.kind.as_str() {
        "frost" | "frost,refused" if declares_main => Some(block.body.clone()),
        "frost" | "frost,refused" => {
            Some(format!("{}\n\nmain :: fn() -> i64 {{ 0 }}\n", block.body))
        }
        "frost,inside" => Some(format!(
            "main :: fn() -> i64 {{\n{}\n0\n}}\n",
            block.body
        )),
        _ => None,
    }
}

fn compiles(label: &str, source: &str) -> (bool, String) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let directory = std::env::temp_dir();
    let input = directory.join(format!("frost_doc_{label}.frost"));
    std::fs::write(&input, source).unwrap();
    let object = directory.join(format!("frost_doc_{label}.o"));
    let run = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("-L")
        .arg(root.join("std"))
        .arg("-n")
        .arg("-o")
        .arg(&object)
        .arg(&input)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&object);
    (
        run.status.success(),
        String::from_utf8_lossy(&run.stderr).to_string(),
    )
}

#[test]
fn every_block_in_the_book_is_what_its_fence_says_it_is() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut faults = Vec::new();
    let mut checked = 0;
    for file in markdown_files() {
        for block in blocks_in(&file) {
            let known = matches!(
                block.kind.as_str(),
                "frost" | "frost,inside" | "frost,refused" | "frost,sketch"
            );
            let shown = block
                .file
                .strip_prefix(&root)
                .unwrap_or(&block.file)
                .display()
                .to_string()
                .replace('\\', "/");
            if !known {
                faults.push(format!(
                    "{shown}:{}  the fence says '{}', which is not one of \
                     frost, frost,inside, frost,refused, frost,sketch",
                    block.line, block.kind
                ));
                continue;
            }
            let Some(source) = program_of(&block) else {
                continue;
            };
            checked += 1;
            let label = format!(
                "{}_{}",
                shown.replace(['/', '.', '-'], "_"),
                block.line
            );
            let (built, said) = compiles(&label, &source);
            if block.kind == "frost,refused" && built {
                faults.push(format!(
                    "{shown}:{}  the fence says this is refused and it compiled",
                    block.line
                ));
            } else if block.kind != "frost,refused" && !built {
                let first =
                    said.lines().rfind(|line| !line.trim().is_empty())
                        .unwrap_or("");
                faults.push(format!(
                    "{shown}:{}  {first}",
                    block.line
                ));
            }
        }
    }
    assert!(
        faults.is_empty(),
        "{} of {checked} blocks in the book do not do what their fence says:\n{}",
        faults.len(),
        faults.join("\n")
    );
}
