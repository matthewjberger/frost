// The editor support, driven end to end.
//
// The compiler serves the editor over a pipe, and an extension asks it. Both
// halves are checked by drivers under .vscode/frost/check, which need a
// JavaScript runtime rather than a compiler. This runs them, so a break in
// either shows on the same commit as the change that caused it rather than the
// next time somebody opens a file.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support.rs"]
mod support;

use support::build_self_hosted_compiler;

const DRIVERS: &[(&str, &str)] = &[
    ("the server on a pipe", ".vscode/frost/check/server.js"),
    ("the extension", ".vscode/frost/check/run.js"),
];

#[test]
fn the_editor_support_answers_every_question_a_reader_asks() {
    let Some(runtime) = javascript_runtime() else {
        return;
    };
    let Some(compiler) = build_self_hosted_compiler("editor") else {
        return;
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for (half, driver) in DRIVERS {
        let answered = Command::new(&runtime)
            .arg(root.join(driver))
            .arg(&compiler)
            .current_dir(root)
            .output()
            .unwrap();
        assert!(
            answered.status.success(),
            "{half} did not answer:\n{}\n{}",
            String::from_utf8_lossy(&answered.stdout),
            String::from_utf8_lossy(&answered.stderr)
        );
    }
    let _ = std::fs::remove_file(&compiler);
}

fn javascript_runtime() -> Option<PathBuf> {
    let node = PathBuf::from("node");
    let found = Command::new(&node)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);
    if found {
        return Some(node);
    }
    // Without a runtime this test says nothing, which on a machine that is
    // supposed to have one is silence where a failure belongs. CI sets
    // FROST_REQUIRE_NODE to turn it into one.
    assert!(
        std::env::var("FROST_REQUIRE_NODE").is_err(),
        "FROST_REQUIRE_NODE is set and no JavaScript runtime was found"
    );
    None
}
