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
use std::sync::OnceLock;
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

/// Whether a linker is on the path, asked once. Every helper here asks, and
/// asking meant spawning three processes per question: the suite spent longer
/// running `--version` than compiling some of the programs it checks.
/// One job per item, spread over as many threads as the machine has.
///
/// A test that compiles a list of programs one after another sets the floor for
/// how long the whole suite takes, however many cores are idle beside it: the
/// binary's other tests finish long before the list does. Each job here is a
/// compiler and a C toolchain in their own processes, so the work is theirs to
/// overlap and this only stops handing it to them one at a time.
///
/// The answers come back in the order the items were given, so a failure names
/// the item it came from.
pub fn in_parallel<T, R>(items: &[T], job: impl Fn(&T) -> R + Sync) -> Vec<R>
where
    T: Sync,
    R: Send,
{
    let threads = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(4)
        .min(items.len().max(1));
    let width = items.len().div_ceil(threads.max(1));
    let chunks: Vec<&[T]> = items.chunks(width.max(1)).collect();
    let job = &job;
    std::thread::scope(|scope| {
        let running: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                scope.spawn(move || chunk.iter().map(job).collect::<Vec<R>>())
            })
            .collect();
        running
            .into_iter()
            .flat_map(|handle| handle.join().unwrap_or_default())
            .collect::<Vec<R>>()
    })
}

pub fn linker_available() -> bool {
    static FOUND: OnceLock<bool> = OnceLock::new();
    *FOUND.get_or_init(find_linker)
}

fn find_linker() -> bool {
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
    static FOUND: OnceLock<Option<&'static str>> = OnceLock::new();
    *FOUND.get_or_init(find_c_compiler)
}

fn find_c_compiler() -> Option<&'static str> {
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

pub fn frost_runtime_source() -> String {
    format!("{}/runtime/runtime.frost", env!("CARGO_MANIFEST_DIR"))
}

/// The Frost half of the runtime, compiled once by the bootstrap and linked
/// beside the C half by every test that links.
///
/// The compilers each cache this for themselves on their own link paths; a test
/// that hands the toolchain an emitted unit is doing the linking itself, so it
/// needs the object itself. Built by the bootstrap for both compilers, since
/// what it holds is the same either way and one build serves the whole run.
pub fn frost_runtime_object() -> String {
    static BUILT: OnceLock<String> = OnceLock::new();
    BUILT
        .get_or_init(|| {
            let object = std::env::temp_dir().join(format!(
                "frost_test_runtime_frost_{}.o",
                std::process::id()
            ));
            let built = Command::new(env!("CARGO_BIN_EXE_frost"))
                .arg("--native")
                .arg("-o")
                .arg(&object)
                .arg(frost_runtime_source())
                .output()
                .expect("the Frost runtime failed to compile");
            assert!(
                built.status.success(),
                "the Frost runtime failed to compile:\n{}",
                String::from_utf8_lossy(&built.stderr)
            );
            object.display().to_string()
        })
        .clone()
}

/// The runtime, compiled once and linked thereafter.
///
/// Every test that runs a program links the runtime beside what the compiler
/// emitted, and handing the C compiler the source each time made that the
/// slowest thing a test run did: the link took 0.6 seconds against the source
/// and 0.12 against an object, and the suite does it hundreds of times. Both
/// compilers already cache this object on their own link paths; this is the
/// harness doing what they do. The source is unchanged by any test, so one
/// object serves the whole run.
///
/// Falls back to the source where the object cannot be built, so a machine this
/// does not suit is slow rather than broken.
pub fn runtime_object() -> String {
    static BUILT: OnceLock<String> = OnceLock::new();
    BUILT
        .get_or_init(|| {
            let Some(compiler) = c_compiler() else {
                return runtime_source();
            };
            let object = std::env::temp_dir()
                .join(format!("frost_test_runtime_{}.o", std::process::id()));
            let built = Command::new(compiler)
                .arg("-std=c11")
                .arg("-c")
                .arg(runtime_source())
                .arg("-o")
                .arg(&object)
                .output();
            match built {
                Ok(done) if done.status.success() => {
                    object.display().to_string()
                }
                _ => runtime_source(),
            }
        })
        .clone()
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
    // The stem carries this process's id, so the same test running in another
    // test binary at the same time writes its own files rather than this one's.
    let stem = unique(&format!("frost_sl_{name}"));
    let input = directory.join(format!("{stem}.frost"));
    std::fs::write(&input, source).unwrap();
    let emitted = directory.join(format!("{stem}.{suffix}"));
    let exe = directory
        .join(format!("{stem}_{suffix}{}", std::env::consts::EXE_SUFFIX));
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
        .arg(runtime_object())
        .arg(frost_runtime_object())
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
    let stem = unique(&format!("frost_bs_{name}"));
    let input = directory.join(format!("{stem}.frost"));
    std::fs::write(&input, source).unwrap();
    let exe = directory.join(format!("{stem}{}", std::env::consts::EXE_SUFFIX));
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
/// What the bootstrap says about a file already on disk, which is how a test
/// hands one file to both compilers. What a report calls a file is part of what
/// is compared, and two files named differently cannot answer that.
pub fn bootstrap_report_at(name: &str, source_path: &Path) -> (bool, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--emit-c")
        .arg("-o")
        .arg(source_path.with_extension("c"))
        .arg(source_path)
        .output()
        .unwrap_or_else(|_| panic!("the bootstrap did not run for {name}"));
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

pub fn bootstrap_refusal(name: &str, source: &str) -> String {
    let directory = std::env::temp_dir();
    let stem = unique(&format!("frost_refuse_{name}"));
    let source_path = directory.join(format!("{stem}.frost"));
    std::fs::write(&source_path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("-o")
        .arg(directory.join(format!("{stem}.o")))
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
