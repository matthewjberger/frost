// What the test binaries need to reach a compiler and run what it produced.
//
// Each of them takes the whole file and uses the part it needs, so an unused
// helper here is one the other binaries are using.
#![allow(dead_code)]

//
// It lives here rather than in each because the fuzzer and the integration
// suite were building the self-hosted compiler two different ways, and a
// difference between two copies of the same plumbing is a difference nobody
// finds until one of them is wrong.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A temp-file stem no run and no other test reuses. On Windows a just-run or
/// just-deleted executable stays briefly locked, so relinking over the same
/// name fails intermittently; a fresh name every time sidesteps it. The process
/// id separates one `cargo test` run from the next, the counter separates tests
/// within a run.
pub fn unique(base: &str) -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!(
        "{base}_{}_{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

pub fn linker_available() -> bool {
    for linker in ["cc", "gcc", "clang"] {
        let found = Command::new(linker)
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
        if found {
            return true;
        }
    }
    // Every test that links silently no-ops without a C toolchain, so a broken
    // one on a machine that is supposed to have it would hide behind a green
    // run. CI sets FROST_REQUIRE_LINKER to turn that silence into a failure.
    assert!(
        std::env::var("FROST_REQUIRE_LINKER").is_err(),
        "FROST_REQUIRE_LINKER is set and no linker was found"
    );
    false
}

pub fn c_compiler() -> Option<&'static str> {
    for compiler in ["gcc", "clang", "cc"] {
        let found = Command::new(compiler)
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
        if found {
            return Some(compiler);
        }
    }
    None
}

/// The self-hosted compiler is a set of modules that import each other, so it
/// is compiled where it sits rather than copied into a temporary directory, the
/// way the examples are.
pub fn self_hosted_source() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("selfhosted")
        .join("frost.frost")
}

pub fn runtime_source() -> String {
    format!("{}/runtime/frost_runtime.c", env!("CARGO_MANIFEST_DIR"))
}

/// Each caller gets its own copy, named after itself. The test binary runs its
/// tests in parallel, so a shared path is two tests writing one file.
pub fn build_self_hosted_compiler(name: &str) -> Option<PathBuf> {
    if c_compiler().is_none() || !linker_available() {
        return None;
    }
    let directory = std::env::temp_dir();
    let compiler = directory.join(format!(
        "{}{}",
        unique(&format!("frost_selfhosted_{name}")),
        std::env::consts::EXE_SUFFIX
    ));
    let frost = env!("CARGO_BIN_EXE_frost");
    let build = Command::new(frost)
        .arg("--link")
        .arg("-o")
        .arg(&compiler)
        .arg(self_hosted_source())
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "the self-hosted compiler failed to build:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
    Some(compiler)
}

/// Compiles `source` with the self-hosted compiler through one of its backends,
/// builds what it emitted, runs it, and answers with what it printed.
pub fn selfhosted_default_output(
    compiler: &Path,
    name: &str,
    source: &str,
    backend: &str,
    suffix: &str,
) -> String {
    let directory = std::env::temp_dir();
    let input = directory.join(format!("frost_sl_{name}.frost"));
    std::fs::write(&input, source).unwrap();
    let emitted = directory.join(format!("frost_sl_{name}.{suffix}"));
    let exe = directory.join(format!(
        "frost_sl_{name}_{suffix}{}",
        std::env::consts::EXE_SUFFIX
    ));
    let emit = Command::new(compiler)
        .arg(backend)
        .arg("-o")
        .arg(&emitted)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        emit.status.success(),
        "the self-hosted compiler refused {name} through {backend}:
{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let built = Command::new(c_compiler().unwrap())
        .arg(&emitted)
        .arg(runtime_source())
        .arg("-lm")
        .arg("-o")
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "what {backend} emitted for {name} did not build:
{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    assert!(run.status.success(), "{name} exited with failure");
    let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&emitted);
    let _ = std::fs::remove_file(&exe);
    output
}

// Build and run `source` with the bootstrap, and return what it printed.
pub fn bootstrap_output(name: &str, source: &str) -> Option<String> {
    if c_compiler().is_none() || !linker_available() {
        return None;
    }
    let directory = std::env::temp_dir();
    let input = directory.join(format!("frost_bs_{name}.frost"));
    std::fs::write(&input, source).unwrap();
    let exe = directory
        .join(format!("frost_bs_{name}{}", std::env::consts::EXE_SUFFIX));
    let build = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("-o")
        .arg(&exe)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "the bootstrap refused {name}:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    assert!(run.status.success(), "{name} exited with failure");
    let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&exe);
    Some(output)
}

// What the bootstrap said when it would not compile a program.
pub fn bootstrap_refusal(name: &str, source: &str) -> String {
    let directory = std::env::temp_dir();
    let source_path = directory.join(format!("frost_refuse_{name}.frost"));
    std::fs::write(&source_path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("-o")
        .arg(directory.join(format!("frost_refuse_{name}.o")))
        .arg(&source_path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&source_path);
    assert!(
        !output.status.success(),
        "the bootstrap accepted {name}, which it should refuse"
    );
    String::from_utf8_lossy(&output.stderr).to_string()
}
