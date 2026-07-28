// Random programs, run through every compiler and every backend, checked for
// one answer.
//
// It used to have three oracles and all three were the bootstrap: its native
// backend, its C backend, and its IR interpreter. Three backends, one compiler,
// which is the shape docs/book/src/impl/self-hosted.md already records as the
// mistake that let an earlier bug through. It cost something real: the
// generator has always written `rng.below(20) - 10` into the source, so it has
// been emitting bare negative literals since it was written, and the
// self-hosted compiler miscompiled every one of them. Seed 0 would have caught
// it on the first run if anything had asked the compiler that ships.
//
// Five oracles now. Anything they do not all agree about is a bug in whichever
// is wrong rather than a feature with a caveat.

use std::path::Path;
use std::process::Command;

#[path = "support.rs"]
mod support;

use support::{
    build_self_hosted_compiler, linker_available, selfhosted_default_output,
};

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1),
        }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

fn gen_expr(rng: &mut Rng, depth: u32) -> String {
    if depth == 0 || rng.below(3) == 0 {
        let value = rng.below(20) as i64 - 10;
        // Written as a prefix minus rather than as the digits of a negative
        // number, since a prefix minus is the thing this is here to check.
        if value < 0 {
            return format!("-{}", -value);
        }
        return format!("{value}");
    }
    let left = gen_expr(rng, depth - 1);
    let right = gen_expr(rng, depth - 1);
    match rng.below(8) {
        0 => format!("({left} + {right})"),
        1 => format!("({left} - {right})"),
        2 => format!("({left} * {})", rng.below(5) as i64),
        3 => format!("({left} % {})", rng.below(7) as i64 + 1),
        4 => format!("({left} & {})", rng.below(255) as i64),
        5 => format!("({left} | {})", rng.below(255) as i64),
        6 => format!("(-{left})"),
        _ => format!("({left} - -{right})"),
    }
}

fn gen_cond(rng: &mut Rng, depth: u32) -> String {
    let left = gen_expr(rng, depth);
    let right = gen_expr(rng, depth);
    let op = ["<", "<=", ">", ">=", "==", "!="][rng.below(6) as usize];
    let plain = format!("({left} {op} {right})");
    // A negated comparison, which is the other prefix operator and the one that
    // used to abort the self-hosted compiler outright.
    if rng.below(3) == 0 {
        return format!("(!{plain})");
    }
    plain
}

// A float expression, kept to values every backend prints the same way. The
// point is that the sign reaches the literal, so `-0.5` is a negative literal
// rather than a subtraction, and that a float compares like one.
fn gen_float(rng: &mut Rng, depth: u32) -> String {
    if depth == 0 || rng.below(3) == 0 {
        let whole = rng.below(9) as i64 - 4;
        let half = if rng.below(2) == 0 { ".5" } else { ".25" };
        if whole < 0 {
            return format!("-{}{half}", -whole);
        }
        return format!("{whole}{half}");
    }
    let left = gen_float(rng, depth - 1);
    let right = gen_float(rng, depth - 1);
    match rng.below(4) {
        0 => format!("({left} + {right})"),
        1 => format!("({left} - {right})"),
        2 => format!("(-{left})"),
        _ => format!("({left} * 2.0)"),
    }
}

fn gen_program(rng: &mut Rng, lines: usize) -> String {
    // `print` rather than an extern `printf`. It is the language's own, so it
    // needs no `unsafe` and no declaration that a C header might already have
    // its own idea about, and every backend has to agree about what it writes.
    let mut source = String::from("main :: fn() -> i64 {\n");
    for index in 0..lines {
        match rng.below(6) {
            0 => {
                let cond = gen_cond(rng, 2);
                let then = gen_expr(rng, 3);
                let els = gen_expr(rng, 3);
                source.push_str(&format!(
                    "    print if {cond} {{ {then} }} else {{ {els} }}\n"
                ));
            }
            1 => {
                // A comparison is a truth value and prints as one, which is how
                // a bool reaches the output without printf having to take one.
                let cond = gen_cond(rng, 2);
                source.push_str(&format!("    print {cond}\n"));
            }
            2 => {
                // A narrower integer, so a value that has been through one is
                // checked rather than assumed. The mask keeps it in range,
                // since an out-of-range literal is a separate question.
                let width = ["i8", "i16", "i32"][rng.below(3) as usize];
                let bound: i64 = match width {
                    "i8" => 60,
                    "i16" => 30000,
                    _ => 1_000_000,
                };
                let value = rng.below(bound as u64 * 2) as i64 - bound;
                let written = if value < 0 {
                    format!("-{}", -value)
                } else {
                    format!("{value}")
                };
                source.push_str(&format!(
                    "    narrow{index} : {width} = {written}\n\
                     \x20   wide{index} : i64 = narrow{index}\n\
                     \x20   print wide{index}\n"
                ));
            }
            3 => {
                // Floats compared rather than printed, so every backend agrees
                // about the answer without agreeing about how many digits a
                // float prints with.
                let left = gen_float(rng, 2);
                let right = gen_float(rng, 2);
                source.push_str(&format!("    print ({left} < {right})\n"));
            }
            4 => {
                let value = gen_float(rng, 2);
                source.push_str(&format!(
                    "    f{index} := {value}\n\
                     \x20   print (f{index} == f{index})\n"
                ));
            }
            _ => {
                let expr = gen_expr(rng, 4);
                source.push_str(&format!("    print {expr}\n"));
            }
        }
    }
    source.push_str("    0\n}\n");
    source
}

fn run_backend(name: &str, source: &str, emit_c: bool) -> String {
    let directory = std::env::temp_dir();
    let source_path = directory.join(format!("frost_fuzz_{name}.frost"));
    let exe_path = directory
        .join(format!("frost_fuzz_{name}{}", std::env::consts::EXE_SUFFIX));
    std::fs::write(&source_path, source).unwrap();
    let frost = env!("CARGO_BIN_EXE_frost");
    let mut command = Command::new(frost);
    if emit_c {
        command.arg("--emit-c");
    }
    let compile = command
        .arg("--link")
        .arg("-o")
        .arg(&exe_path)
        .arg(&source_path)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "fuzz compilation failed (emit_c={emit_c}) for:\n{source}\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(&exe_path).output().unwrap();
    assert!(run.status.success(), "fuzz binary crashed for:\n{source}");
    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_file(&exe_path);
    String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n")
}

fn run_ir(name: &str, source: &str) -> String {
    let directory = std::env::temp_dir();
    let source_path = directory.join(format!("frost_fuzz_{name}.frost"));
    std::fs::write(&source_path, source).unwrap();
    let frost = env!("CARGO_BIN_EXE_frost");
    let output = Command::new(frost)
        .arg("--run-ir")
        .arg(&source_path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&source_path);
    assert!(
        output.status.success(),
        "ir interpreter declined a scalar program:\n{source}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n")
}

fn run_self_hosted(
    compiler: &Path,
    name: &str,
    source: &str,
    backend: &str,
    suffix: &str,
) -> String {
    selfhosted_default_output(compiler, name, source, backend, suffix)
}

// The generator is where coverage is claimed, and a claim about a generator is
// worth nothing until something reads what it wrote. The last widening was
// wasted for exactly this reason: it emitted negative literals from the day it
// was written and nothing that could see the bug was ever asked.
#[test]
fn the_generator_writes_the_constructs_it_claims_to() {
    let mut corpus = String::new();
    for seed in 0..60u64 {
        let mut rng = Rng::new(seed);
        corpus.push_str(&gen_program(&mut rng, 4));
    }
    for (what, needle) in [
        ("a prefix minus on a literal", " -1"),
        ("a prefix minus on an expression", "(-("),
        ("a prefix bang", "(!("),
        ("a narrower integer", " : i8 = "),
        ("a float literal", ".25"),
        ("a negative float literal", "-1.5"),
        ("a float comparison", ".5 < "),
    ] {
        assert!(
            corpus.contains(needle),
            "the generator never wrote {what} in sixty seeds"
        );
    }
}

#[test]
fn every_compiler_agrees_on_random_programs() {
    if !linker_available() {
        return;
    }
    // Built once. It is the expensive part, and building it per seed would put
    // the cost of sixty compiles between the fuzzer and anyone running it.
    let hosted = build_self_hosted_compiler("fuzz");
    for seed in 0..60u64 {
        let mut rng = Rng::new(seed);
        let source = gen_program(&mut rng, 4);
        let native = run_backend(&format!("s{seed}"), &source, false);
        let via_c = run_backend(&format!("s{seed}c"), &source, true);
        assert_eq!(
            native, via_c,
            "the bootstrap's backends disagree on seed {seed}:\n{source}"
        );
        let interpreted = run_ir(&format!("s{seed}i"), &source);
        assert_eq!(
            native, interpreted,
            "the IR interpreter disagrees on seed {seed}:\n{source}"
        );
        let Some(compiler) = hosted.as_ref() else {
            continue;
        };
        let hosted_asm = run_self_hosted(
            compiler,
            &format!("s{seed}a"),
            &source,
            "--emit-asm",
            "s",
        );
        assert_eq!(
            native, hosted_asm,
            "the self-hosted assembly backend disagrees on seed {seed}:\n{source}"
        );
        let hosted_c = run_self_hosted(
            compiler,
            &format!("s{seed}sc"),
            &source,
            "--emit-c",
            "c",
        );
        assert_eq!(
            native, hosted_c,
            "the self-hosted C backend disagrees on seed {seed}:\n{source}"
        );
    }
    if let Some(compiler) = hosted {
        let _ = std::fs::remove_file(&compiler);
    }
}
