use std::process::Command;

fn linker_available() -> bool {
    for linker in ["cc", "gcc", "clang"] {
        if Command::new(linker)
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
        {
            return true;
        }
    }
    false
}

// A tiny deterministic PRNG so the generated corpus is reproducible.
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
        // splitmix64
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

// Generate a random i64 arithmetic expression whose value is small enough that
// it never overflows i64 (so signed-overflow UB never enters), using only
// operators where both backends have identical, defined semantics. Divisors
// and shift amounts are always safe literals.
fn gen_expr(rng: &mut Rng, depth: u32) -> String {
    if depth == 0 || rng.below(3) == 0 {
        return format!("{}", rng.below(20) as i64 - 10);
    }
    let left = gen_expr(rng, depth - 1);
    let right = gen_expr(rng, depth - 1);
    match rng.below(6) {
        0 => format!("({left} + {right})"),
        1 => format!("({left} - {right})"),
        2 => format!("({left} * {})", rng.below(5) as i64),
        3 => format!("({left} % {})", rng.below(7) as i64 + 1), // non-zero
        4 => format!("({left} & {})", rng.below(255) as i64),
        _ => format!("({left} | {})", rng.below(255) as i64),
    }
}

// A boolean condition over the same safe arithmetic.
fn gen_cond(rng: &mut Rng, depth: u32) -> String {
    let left = gen_expr(rng, depth);
    let right = gen_expr(rng, depth);
    let op = ["<", "<=", ">", ">=", "==", "!="][rng.below(6) as usize];
    format!("({left} {op} {right})")
}

fn gen_program(rng: &mut Rng, lines: usize) -> String {
    let mut source = String::from(
        "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\nmain :: fn() -> i64 {\n",
    );
    for _ in 0..lines {
        if rng.below(3) == 0 {
            let cond = gen_cond(rng, 2);
            let then = gen_expr(rng, 3);
            let els = gen_expr(rng, 3);
            source.push_str(&format!(
                "    printf(\"%lld\\n\", if {cond} {{ {then} }} else {{ {els} }})\n"
            ));
        } else {
            let expr = gen_expr(rng, 4);
            source.push_str(&format!("    printf(\"%lld\\n\", {expr})\n"));
        }
    }
    source.push_str("    0\n}\n");
    source
}

fn run_backend(name: &str, source: &str, emit_c: bool) -> String {
    let directory = std::env::temp_dir();
    let source_path = directory.join(format!("frost_fuzz_{name}.frost"));
    let exe_path = directory.join(format!(
        "frost_fuzz_{name}{}",
        std::env::consts::EXE_SUFFIX
    ));
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

// Generate a reproducible corpus of random arithmetic/control-flow programs and
// assert the Cranelift and C backends produce byte-identical output for each.
#[test]
fn fuzz_backends_agree_on_random_programs() {
    if !linker_available() {
        return;
    }
    for seed in 0..60u64 {
        let mut rng = Rng::new(seed);
        let source = gen_program(&mut rng, 4);
        let native = run_backend(&format!("s{seed}"), &source, false);
        let via_c = run_backend(&format!("s{seed}c"), &source, true);
        assert_eq!(
            native, via_c,
            "backends disagree on fuzz seed {seed}:\n{source}"
        );
    }
}
