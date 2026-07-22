use std::path::PathBuf;
use std::process::Command;

fn linker_available() -> bool {
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
    false
}

fn compile_and_run(name: &str, source: &str) -> Option<String> {
    run_backend(name, source, false)
}

fn run_backend(name: &str, source: &str, emit_c: bool) -> Option<String> {
    if !linker_available() {
        return None;
    }

    let directory = std::env::temp_dir();
    let source_path = directory.join(format!("frost_native_{name}.frost"));
    let exe_path = directory.join(format!(
        "frost_native_{name}{}",
        std::env::consts::EXE_SUFFIX
    ));
    std::fs::write(&source_path, source).unwrap();

    let frost = env!("CARGO_BIN_EXE_frost");
    let mut command = Command::new(frost);
    if emit_c {
        command.arg("--emit-c");
    }
    command
        .arg("--link")
        .arg("-o")
        .arg(&exe_path)
        .arg(&source_path);
    let compile = command.output().unwrap();
    assert!(
        compile.status.success(),
        "compilation failed for {name} (emit_c={emit_c}):\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&exe_path).output().unwrap();
    assert!(
        run.status.success(),
        "native binary {name} exited with failure"
    );

    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_file(&exe_path);

    Some(String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"))
}

fn run_ir_oracle(name: &str, source: &str) -> Option<String> {
    let directory = std::env::temp_dir();
    let source_path = directory.join(format!("frost_oracle_{name}.frost"));
    std::fs::write(&source_path, source).unwrap();
    let frost = env!("CARGO_BIN_EXE_frost");
    let output = Command::new(frost)
        .arg("--run-ir")
        .arg(&source_path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&source_path);
    if output.status.code() == Some(3) {
        return None;
    }
    assert!(
        output.status.success(),
        "ir interpreter failed for {name}:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n"))
}

fn compile_error(name: &str, source: &str) -> String {
    let directory = std::env::temp_dir();
    let source_path = directory.join(format!("frost_err_{name}.frost"));
    let exe_path = directory
        .join(format!("frost_err_{name}{}", std::env::consts::EXE_SUFFIX));
    std::fs::write(&source_path, source).unwrap();
    let frost = env!("CARGO_BIN_EXE_frost");
    let output = Command::new(frost)
        .arg("--link")
        .arg("-o")
        .arg(&exe_path)
        .arg(&source_path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&source_path);
    assert!(
        !output.status.success(),
        "expected a compile error for {name} but it succeeded"
    );
    String::from_utf8_lossy(&output.stderr).to_string()
}

#[test]
fn ownership_errors_report_a_source_line() {
    let source = r#"
Buffer :: struct { size: i64 }

consume :: fn(move b: Buffer) -> i64 { b.size }

main :: fn() -> i64 {
    buf := Buffer { size = 10 }
    first := consume(buf)
    second := consume(buf)
    first + second
}
"#;
    let message = compile_error("uam", source);
    assert!(
        message.contains("at line 9,"),
        "expected the moved-value error at line 9, got:\n{message}"
    );
    assert!(message.contains("use of moved value"), "got:\n{message}");
}

#[test]
fn lowering_errors_report_a_source_line() {
    let source = r#"
main :: fn() -> i64 {
    x := no_such_function(3)
    x
}
"#;
    let message = compile_error("unknownfn", source);
    assert!(
        message.contains("at line 3,"),
        "expected the unknown-variable error at line 3, got:\n{message}"
    );
}

#[test]
fn borrow_exclusivity_errors_report_a_source_line() {
    let source = r#"
add_both :: fn(mut a: i64, mut b: i64) -> i64 { a^ + b^ }

main :: fn() -> i64 {
    mut value : i64 = 1
    total := add_both(value, value)
    total
}
"#;
    let message = compile_error("exclusivity", source);
    assert!(
        message.contains("at line 6,"),
        "expected the exclusivity error at line 6, got:\n{message}"
    );
}

#[test]
fn linear_not_consumed_errors_report_a_source_line() {
    let source = r#"
Resource :: linear struct { id: i64 }

make :: fn(id: i64) -> Resource { Resource { id = id } }

main :: fn() -> i64 {
    r := make(7)
    0
}
"#;
    let message = compile_error("linear", source);
    assert!(
        message.contains("at line"),
        "expected a located linear error, got:\n{message}"
    );
    assert!(
        message.contains("consumed"),
        "expected a linear-not-consumed error, got:\n{message}"
    );
}

#[test]
fn discarding_a_linear_value_is_a_compile_error() {
    let source = r#"
Resource :: linear struct { id: i64 }

make :: fn(id: i64) -> Resource { Resource { id = id } }
drop_it :: extern fn(r: Resource)

main :: fn() -> i64 {
    r := make(1)
    make(2)
    drop_it(r)
    0
}
"#;
    let message = compile_error("discard_linear", source);
    assert!(
        message.contains("never consumed") || message.contains("linear"),
        "expected a discarded-linear error, got:\n{message}"
    );
}

fn compile_and_run_status(name: &str, source: &str) -> Option<bool> {
    if !linker_available() {
        return None;
    }
    let directory = std::env::temp_dir();
    let source_path = directory.join(format!("frost_native_{name}.frost"));
    let exe_path = directory.join(format!(
        "frost_native_{name}{}",
        std::env::consts::EXE_SUFFIX
    ));
    std::fs::write(&source_path, source).unwrap();
    let frost = env!("CARGO_BIN_EXE_frost");
    let compile = Command::new(frost)
        .arg("--link")
        .arg("-o")
        .arg(&exe_path)
        .arg(&source_path)
        .output()
        .unwrap();
    assert!(compile.status.success(), "compilation failed for {name}");
    let run = Command::new(&exe_path).output().unwrap();
    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_file(&exe_path);
    Some(run.status.success())
}

const OUT_OF_BOUNDS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    arr := [10, 20, 30]
    mut i : i64 = 5
    printf("%lld\n", arr[i])
    0
}
"#;

#[test]
fn native_out_of_bounds_index_aborts() {
    let Some(succeeded) = compile_and_run_status("oob", OUT_OF_BOUNDS) else {
        return;
    };
    assert!(!succeeded, "out-of-bounds index should abort at runtime");
}

const ARITHMETIC: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

factorial :: fn(n: i64) -> i64 {
    if (n <= 1) { 1 } else { n * factorial(n - 1) }
}

sum_to :: fn(n: i64) -> i64 {
    mut total : i64 = 0
    mut i : i64 = 0
    while (i <= n) {
        total = total + i
        i = i + 1
    }
    total
}

count_evens :: fn(limit: i64) -> i64 {
    mut count : i64 = 0
    for i in 0..limit {
        if (i % 2 == 0) { count = count + 1 } else { count = count + 0 }
    }
    count
}

main :: fn() -> i64 {
    printf("%lld\n", factorial(10))
    printf("%lld\n", sum_to(100))
    printf("%lld\n", count_evens(10))
    printf("%lld\n", if (3 < 5 && 5 < 10) { 1 } else { 0 })
    printf("%lld\n", if (2 > 9 || 4 == 4) { 1 } else { 0 })
    printf("%lld\n", 1 << 10)
    printf("%lld\n", 100 % 7)
    0
}
"#;

#[test]
fn native_arithmetic_and_control_flow() {
    let Some(output) = compile_and_run("arith", ARITHMETIC) else {
        return;
    };
    assert_eq!(output, "3628800\n5050\n5\n1\n1\n1024\n2\n");
}

const FLOATS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    printf("%lld\n", if (7.0 / 2.0 > 3.0) { 1 } else { 0 })
    printf("%lld\n", if (1.5 + 1.5 == 3.0) { 1 } else { 0 })
    printf("%lld\n", if (2.0 * 2.0 < 3.9) { 1 } else { 0 })
    0
}
"#;

#[test]
fn native_float_operations() {
    let Some(output) = compile_and_run("floats", FLOATS) else {
        return;
    };
    assert_eq!(output, "1\n1\n0\n");
}

const F32_OPERATIONS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

scale :: fn(x: f32, k: f32) -> f32 { x * k }

main :: fn() -> i64 {
    a : f32 = 1.5
    b : f32 = 2.5
    c : f32 = a + b
    printf("%lld\n", if (c == 4.0) { 1 } else { 0 })
    widened : f64 = c
    printf("%lld\n", if (widened == 4.0) { 1 } else { 0 })
    printf("%lld\n", if (scale(3.0, 2.5) == 7.5) { 1 } else { 0 })
    arr : [3]f32 = [1.5, 2.5, 3.0]
    printf("%lld\n", if (arr[1] == 2.5) { 1 } else { 0 })
    0
}
"#;

#[test]
fn native_f32_operations() {
    let Some(output) = compile_and_run("f32ops", F32_OPERATIONS) else {
        return;
    };
    assert_eq!(output, "1\n1\n1\n1\n");
}

const WIDTHS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    small : i32 = 300
    printf("%lld\n", small)
    byte_sum : u8 = 100
    printf("%lld\n", byte_sum + 50)
    0
}
"#;

#[test]
fn native_integer_widths_and_casts() {
    let Some(output) = compile_and_run("widths", WIDTHS) else {
        return;
    };
    assert_eq!(output, "300\n150\n");
}

const WRAPPING_AND_UNARY: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    a : u8 = 200
    b : u8 = 100
    printf("%lld\n", a + b)
    d : u32 = 4000000000
    e : u32 = 1000000000
    printf("%lld\n", d + e)
    g : i64 = 42
    printf("%lld\n", -g)
    0
}
"#;

#[test]
fn native_wrapping_and_unary() {
    let Some(output) = compile_and_run("wrapping", WRAPPING_AND_UNARY) else {
        return;
    };
    assert_eq!(output, "44\n705032704\n-42\n");
}

const ANON_FUNCTIONS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

apply :: fn(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }

main :: fn() -> i64 {
    printf("%lld\n", apply(fn(a: i64) -> i64 { a + 1 }, 41))
    printf("%lld\n", apply(fn(a: i64) -> i64 { a * a }, 9))
    g := fn(a: i64) -> i64 { a - 3 }
    printf("%lld\n", g(50))
    ops := [fn(a: i64) -> i64 { a + 1 }, fn(a: i64) -> i64 { a * 2 }]
    printf("%lld\n", ops[1](10))
    0
}
"#;

const PARAM_MODES: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }

bump :: fn(mut p: Point) { p.x = p.x + 1 }

sum :: fn(p: Point) -> i64 { p.x + p.y }

main :: fn() -> i64 {
    mut pt : Point = Point { x = 5, y = 10 }
    bump(pt)
    bump(pt)
    printf("%lld\n", pt.x)
    printf("%lld\n", sum(pt))
    0
}
"#;

const FAILURE_SETS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

FileError :: enum { NotFound, Denied }

read_size :: fn(ok: i64) -> i64 ! FileError {
    if (ok == 0) { return FileError::NotFound {} }
    return 42
}

use_it :: fn(ok: i64) -> i64 ! FileError {
    n := read_size(ok)?
    return n + 1
}

report :: fn(ok: i64) -> i64 {
    match use_it(ok) {
        case .Ok { value }: value
        case .Err { error }: 0 - 1
    }
}

main :: fn() -> i64 {
    printf("%lld\n", report(1))
    printf("%lld\n", report(0))
    0
}
"#;

// A fallible function returns `-> T ! E`; `?` unwraps the Ok value and returns
// the enclosing function's Err on failure; the caller matches Ok/Err.
#[test]
fn native_failure_sets() {
    let Some(output) = compile_and_run("failure_sets", FAILURE_SETS) else {
        return;
    };
    assert_eq!(output, "43\n-1\n");
}

const ALLOCATION_SOURCES: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Arena :: struct($N: usize) {
    data: [N]u8,
    offset: i64,
}

alloc_int :: fn(mut a: Arena<256>) -> ^i64 {
    slot := ptr_to(a.data[a.offset])
    a.offset = a.offset + sizeof(i64)
    ptr_cast($i64, slot)
}

make_two :: fn() -> i64 uses Arena<256> {
    p := alloc_int(arena)
    p^ = 10
    q := alloc_int(arena)
    q^ = 32
    p^ + q^
}

forward :: fn() -> i64 uses Arena<256> {
    make_two()
}

main :: fn() -> i64 {
    mut arena : Arena<256> = Arena { data = [0; 256], offset = 0 }
    mut result : i64 = 0
    with arena {
        result = forward()
    }
    printf("%lld\n", result)
    0
}
"#;

// `uses Arena<256>` declares an allocation source; the capability is threaded
// implicitly through a `uses` call (`forward` forwards to `make_two`) and
// supplied by a `with` block at the root. No arena is passed by hand.
#[test]
fn native_allocation_sources() {
    let Some(output) = compile_and_run("alloc_sources", ALLOCATION_SOURCES)
    else {
        return;
    };
    assert_eq!(output, "42\n");
}

// A raw pointer into the arena may not escape its `with` block: storing an
// arena pointer in a binding that outlives the region is rejected.
#[test]
fn region_pointer_escape_is_rejected() {
    let source = r#"
Arena :: struct($N: usize) { data: [N]u8, offset: i64 }

alloc_int :: fn(mut a: Arena<256>) -> ^i64 {
    slot := ptr_to(a.data[a.offset])
    a.offset = a.offset + sizeof(i64)
    ptr_cast($i64, slot)
}

main :: fn() -> i64 {
    mut arena : Arena<256> = Arena { data = [0; 256], offset = 0 }
    mut escaped : ^i64 = ptr_to(arena.offset)
    with arena {
        escaped = alloc_int(arena)
    }
    escaped^
}
"#;
    let message = compile_error("region_escape", source);
    assert!(
        message.contains("region") && message.contains("escapes"),
        "expected a region-escape error, got:\n{message}"
    );
}

// A `uses` function may not leak an arena pointer into one of its parameters:
// the pointer would outlive the arena once the function returns. Caught without
// lifetimes, by flow alone.
#[test]
fn region_pointer_leak_into_parameter_is_rejected() {
    let source = r#"
Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
Reg :: struct { ptr: ^i64 }

alloc_int :: fn(mut a: Arena<256>) -> ^i64 {
    slot := ptr_to(a.data[a.offset])
    a.offset = a.offset + sizeof(i64)
    ptr_cast($i64, slot)
}

stash :: fn(mut r: Reg) -> i64 uses Arena<256> {
    r.ptr = alloc_int(arena)
    0
}

main :: fn() -> i64 { 0 }
"#;
    let message = compile_error("region_leak", source);
    assert!(
        message.contains("region") && message.contains("escapes"),
        "expected a region-escape error, got:\n{message}"
    );
}

// Calling a `uses` function with no capability in scope is rejected.
#[test]
fn allocation_source_without_capability_is_rejected() {
    let source = r#"
Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
make :: fn() -> i64 uses Arena<256> { 7 }
main :: fn() -> i64 { make() }
"#;
    let message = compile_error("alloc_no_cap", source);
    assert!(
        message.contains("allocation capability"),
        "expected a missing-capability error, got:\n{message}"
    );
}

// A `mut` parameter is written and a value parameter read, both called with a
// plain value and no `&`/`&mut` -- the compiler borrows for the mut parameter.
#[test]
fn native_parameter_modes() {
    let Some(output) = compile_and_run("param_modes", PARAM_MODES) else {
        return;
    };
    assert_eq!(output, "7\n17\n");
}

#[test]
fn native_anonymous_functions() {
    let Some(output) = compile_and_run("anon", ANON_FUNCTIONS) else {
        return;
    };
    assert_eq!(output, "42\n81\n47\n20\n");
}

const MINIFROST: &str = include_str!("../bootstrap/minifrost.frost");

fn c_compiler() -> Option<&'static str> {
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

fn compile_c_and_run(name: &str, c_source: &str) -> Option<String> {
    let compiler = c_compiler()?;
    let directory = std::env::temp_dir();
    let c_path = directory.join(format!("frost_emitted_{name}.c"));
    let exe_path = directory.join(format!(
        "frost_emitted_{name}{}",
        std::env::consts::EXE_SUFFIX
    ));
    std::fs::write(&c_path, c_source).unwrap();
    let compile = Command::new(compiler)
        .arg("-std=c11")
        .arg(&c_path)
        .arg("-o")
        .arg(&exe_path)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "emitted C failed to compile for {name}:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(&exe_path).output().unwrap();
    let _ = std::fs::remove_file(&c_path);
    let _ = std::fs::remove_file(&exe_path);
    Some(String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"))
}

#[test]
fn bootstrap_minifrost_emits_working_c() {
    let Some(c_source) = compile_and_run("minifrost", MINIFROST) else {
        return;
    };
    let Some(output) = compile_c_and_run("minifrost", &c_source) else {
        return;
    };
    assert_eq!(
        output,
        "55\n120\n55\n3\n14\n11\n4\n11\n11\n16\n100\n200\n999\n5\n42\n77\n"
    );
}

// Compile emitted C together with the runtime into an executable, returning its
// path. The caller runs it (optionally with environment variables) and removes it.
fn compile_c_with_runtime(name: &str, c_source: &str) -> Option<PathBuf> {
    let compiler = c_compiler()?;
    let directory = std::env::temp_dir();
    let c_path = directory.join(format!("frost_selfhost_{name}.c"));
    let exe_path = directory.join(format!(
        "frost_selfhost_{name}{}",
        std::env::consts::EXE_SUFFIX
    ));
    std::fs::write(&c_path, c_source).unwrap();
    let runtime =
        format!("{}/runtime/frost_runtime.c", env!("CARGO_MANIFEST_DIR"));
    let compile = Command::new(compiler)
        .arg("-std=c11")
        .arg(&c_path)
        .arg(&runtime)
        .arg("-o")
        .arg(&exe_path)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "self-hosted C failed to compile for {name}:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let _ = std::fs::remove_file(&c_path);
    Some(exe_path)
}

// The self-hosting fixpoint: minifrost compiles its own source, the resulting
// compiler compiles that source again, and the two emitted translation units are
// byte-identical (the classic three-stage bootstrap check).
#[test]
fn bootstrap_minifrost_self_hosts() {
    if c_compiler().is_none() {
        return;
    }
    let source_file =
        format!("{}/bootstrap/minifrost.frost", env!("CARGO_MANIFEST_DIR"));

    // Stage 1: the frost-hosted minifrost compiles minifrost.frost.
    let Some(gen1_c) =
        compile_and_run_with_input("selfhost1", MINIFROST, &source_file)
    else {
        return;
    };
    assert!(
        gen1_c.lines().count() > 1000,
        "self-hosted output implausibly small ({} lines)",
        gen1_c.lines().count()
    );

    // Stage 2: build a compiler from that C and have it compile the source again.
    let Some(gen1_exe) = compile_c_with_runtime("gen1", &gen1_c) else {
        return;
    };
    let gen2 = Command::new(&gen1_exe)
        .env("MINIFROST_INPUT", &source_file)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&gen1_exe);
    assert!(
        gen2.status.success(),
        "self-hosted compiler exited with failure"
    );
    let gen2_c = String::from_utf8_lossy(&gen2.stdout).replace("\r\n", "\n");

    assert_eq!(gen1_c, gen2_c, "self-hosting is not a fixpoint");
}

// Compile a Frost program to a native executable, run it with MINIFROST_INPUT
// set, and return its stdout.
fn compile_and_run_with_input(
    name: &str,
    source: &str,
    input: &str,
) -> Option<String> {
    if !linker_available() {
        return None;
    }
    let directory = std::env::temp_dir();
    let source_path = directory.join(format!("frost_native_{name}.frost"));
    let exe_path = directory.join(format!(
        "frost_native_{name}{}",
        std::env::consts::EXE_SUFFIX
    ));
    std::fs::write(&source_path, source).unwrap();
    let frost = env!("CARGO_BIN_EXE_frost");
    let compile = Command::new(frost)
        .arg("--link")
        .arg("-o")
        .arg(&exe_path)
        .arg(&source_path)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "compilation failed for {name}:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(&exe_path)
        .env("MINIFROST_INPUT", input)
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "native binary {name} exited with failure"
    );
    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_file(&exe_path);
    Some(String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"))
}

fn run_test_mode(name: &str, source: &str) -> Option<(String, bool)> {
    c_compiler()?;
    let directory = std::env::temp_dir();
    let source_path = directory.join(format!("frost_tm_{name}.frost"));
    std::fs::write(&source_path, source).unwrap();
    let frost = env!("CARGO_BIN_EXE_frost");
    let output = Command::new(frost)
        .arg("--test")
        .arg(&source_path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&source_path);
    Some((
        String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n"),
        output.status.success(),
    ))
}

#[test]
fn in_module_tests_report_pass() {
    let source = "add :: fn(a: i64, b: i64) -> i64 { a + b }\n\
                  test \"adds\" { assert(add(2, 3) == 5) }\n\
                  test \"identity\" { assert(add(7, 0) == 7) }\n";
    let Some((output, ok)) = run_test_mode("pass", source) else {
        return;
    };
    assert!(ok, "expected passing tests, got:\n{output}");
    assert!(output.contains("test adds ... ok"), "got:\n{output}");
    assert!(output.contains("test identity ... ok"), "got:\n{output}");
    assert!(output.contains("all tests passed"), "got:\n{output}");
}

#[test]
fn native_import_resolves_across_files() {
    if !linker_available() {
        return;
    }
    let directory = std::env::temp_dir().join("frost_import_test");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("helper.frost"),
        "export triple\ntriple :: fn(x: i64) -> i64 { x * 3 }\n",
    )
    .unwrap();
    let main_path = directory.join("main.frost");
    std::fs::write(
        &main_path,
        "import \"helper.frost\"\n\
         printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         main :: fn() -> i64 { printf(\"%lld\\n\", triple(14)) 0 }\n",
    )
    .unwrap();
    let exe_path =
        directory.join(format!("imp_main{}", std::env::consts::EXE_SUFFIX));
    let frost = env!("CARGO_BIN_EXE_frost");
    let compile = Command::new(frost)
        .arg("--link")
        .arg("-o")
        .arg(&exe_path)
        .arg(&main_path)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "import compile failed:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(&exe_path).output().unwrap();
    let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    assert_eq!(output, "42\n");
}

fn frost_compiles(dir: &std::path::Path, main: &str) -> (bool, String) {
    let exe = dir.join(format!("out{}", std::env::consts::EXE_SUFFIX));
    let output = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("-o")
        .arg(&exe)
        .arg(dir.join(main))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        return (false, stderr);
    }
    let run = Command::new(&exe).output().unwrap();
    (
        true,
        String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"),
    )
}

#[test]
fn module_export_hides_private_items() {
    if !linker_available() {
        return;
    }
    let dir = std::env::temp_dir().join("frost_export_test");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("lib.frost"),
        "export area\n\
         scale :: fn(x: i64) -> i64 { x * 2 }\n\
         area :: fn(w: i64, h: i64) -> i64 { scale(w) * h }\n",
    )
    .unwrap();

    std::fs::write(
        dir.join("uses_public.frost"),
        "import \"lib.frost\"\n\
         printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         main :: fn() -> i64 { printf(\"%lld\\n\", area(3, 5)) 0 }\n",
    )
    .unwrap();
    let (ok, out) = frost_compiles(&dir, "uses_public.frost");
    assert!(ok, "public import should compile: {out}");
    assert_eq!(out, "30\n");

    std::fs::write(
        dir.join("uses_private.frost"),
        "import \"lib.frost\"\n\
         printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         main :: fn() -> i64 { printf(\"%lld\\n\", scale(10)) 0 }\n",
    )
    .unwrap();
    let (ok, err) = frost_compiles(&dir, "uses_private.frost");
    assert!(!ok, "using a private item should fail to compile");
    assert!(err.contains("scale"), "error should mention scale: {err}");
}

#[test]
fn in_module_tests_report_failure() {
    let source = "test \"fails\" { assert(1 == 2) }\n";
    let Some((output, ok)) = run_test_mode("fail", source) else {
        return;
    };
    assert!(!ok, "a failing assert should exit non-zero");
    assert!(output.contains("FAILED"), "got:\n{output}");
}

const STRINGS: &str = r#"
puts :: extern fn(s: ^i8) -> i32

main :: fn() -> i64 {
    puts("line one")
    puts("line\ttwo")
    0
}
"#;

#[test]
fn native_strings_and_escapes() {
    let Some(output) = compile_and_run("strings", STRINGS) else {
        return;
    };
    assert_eq!(output, "line one\nline\ttwo\n");
}

const STR_VIEW: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

first_byte :: fn(s: str) -> i64 {
    s[0]
}

pick :: fn(flag: i64) -> str {
    if (flag == 0) {
        return "yes"
    }
    return "longer"
}

main :: fn() -> i64 {
    greeting := "Frost"
    n := str_len(greeting)
    printf("%lld\n", n)
    mut i : i64 = 0
    while (i < n) {
        printf("%lld\n", greeting[i])
        i = i + 1
    }
    printf("%lld\n", first_byte(greeting))
    chosen := pick(0)
    printf("%lld\n", str_len(chosen))
    printf("%lld\n", chosen[0])
    other := pick(1)
    printf("%lld\n", str_len(other))
    0
}
"#;

#[test]
fn native_str_is_a_length_carrying_view() {
    let Some(output) = compile_and_run("strview", STR_VIEW) else {
        return;
    };
    assert_eq!(output, "5\n70\n114\n111\n115\n116\n70\n3\n121\n6\n");
}

const DYNAMIC_ARENA: &str = r#"
malloc :: extern fn(size: i64) -> ^u8
free :: extern fn(pointer: ^u8)
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Arena :: linear struct { data: ^u8, cap: i64, offset: i64 }

arena_new :: fn(cap: i64) -> Arena {
    Arena { data = malloc(cap), cap = cap, offset = 0 }
}

arena_destroy :: fn(move a: Arena) { free(a.data) }

alloc_int :: fn(mut a: Arena) -> ^i64 {
    slot := ptr_to(a.data[a.offset])
    a.offset = a.offset + sizeof(i64)
    ptr_cast($i64, slot)
}

main :: fn() -> i64 {
    mut a := arena_new(256)
    p := alloc_int(a)
    p^ = 42
    q := alloc_int(a)
    q^ = 100
    printf("%lld\n", p^ + q^)
    printf("%lld\n", a.offset)
    arena_destroy(a)
    0
}
"#;

#[test]
fn native_dynamic_arena_over_malloc() {
    let Some(output) = compile_and_run("dynarena", DYNAMIC_ARENA) else {
        return;
    };
    assert_eq!(output, "142\n16\n");
}

const ALLOCATOR_INTERFACE: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Bump :: struct { data: ^u8, cap: i64, offset: i64 }

bump_take :: fn(state: ^u8, size: i64) -> ^u8 {
    b := ptr_cast($Bump, state)
    slot := ptr_to(b^.data[b^.offset])
    b^.offset = b^.offset + size
    slot
}

Allocator :: struct { take: fn(^u8, i64) -> ^u8, state: ^u8 }

alloc :: fn(a: Allocator, size: i64) -> ^u8 {
    a.take(a.state, size)
}

main :: fn() -> i64 {
    mut backing : [64]u8 = [0; 64]
    mut bump : Bump = Bump { data = ptr_to(backing[0]), cap = 64, offset = 0 }
    a : Allocator = Allocator { take = bump_take, state = ptr_cast($u8, ptr_to(bump)) }
    p := ptr_cast($i64, alloc(a, 8))
    p^ = 42
    q := ptr_cast($i64, alloc(a, 8))
    q^ = 7
    printf("%lld\n", p^ + q^)
    printf("%lld\n", bump.offset)
    0
}
"#;

#[test]
fn native_allocator_interface() {
    let Some(output) = compile_and_run("allociface", ALLOCATOR_INTERFACE)
    else {
        return;
    };
    assert_eq!(output, "49\n16\n");
}

const ARENA: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }

Arena :: struct($N: usize) {
    data: [N]u8,
    offset: i64,
}

alloc_point :: fn(mut a: Arena<128>) -> ^Point {
    slot := ptr_to(a.data[a.offset])
    a.offset = a.offset + sizeof(Point)
    ptr_cast($Point, slot)
}

alloc_int :: fn(mut a: Arena<128>) -> ^i64 {
    slot := ptr_to(a.data[a.offset])
    a.offset = a.offset + sizeof(i64)
    ptr_cast($i64, slot)
}

main :: fn() -> i64 {
    mut arena : Arena<128> = Arena { data = [0; 128], offset = 0 }
    p : ^Point = alloc_point(arena)
    p^.x = 3
    p^.y = 4
    q : ^i64 = alloc_int(arena)
    q^ = 99
    printf("%lld\n", p^.x)
    printf("%lld\n", q^)
    printf("%lld\n", arena.offset)
    arena.offset = 0
    r : ^i64 = alloc_int(arena)
    r^ = 7
    printf("%lld\n", r^)
    printf("%lld\n", arena.offset)
    0
}
"#;

#[test]
fn native_frost_arena_allocator() {
    let Some(output) = compile_and_run("arena", ARENA) else {
        return;
    };
    assert_eq!(output, "3\n99\n24\n7\n8\n");
}

const VALUE_GENERICS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Buffer :: struct($T: Type, $N: usize) {
    data: [N]T,
    len: i64,
}

push :: fn(mut b: Buffer<i64, 4>, value: i64) {
    b.data[b.len] = value
    b.len = b.len + 1
}

total :: fn(b: Buffer<i64, 4>) -> i64 {
    view : []i64 = b.data
    mut sum : i64 = 0
    mut i : i64 = 0
    while (i < b.len) {
        sum = sum + view[i]
        i = i + 1
    }
    sum
}

main :: fn() -> i64 {
    mut b : Buffer<i64, 4> = Buffer {
        data = [0, 0, 0, 0],
        len = 0,
    }
    push(b, 10)
    push(b, 20)
    push(b, 30)
    printf("%lld\n", b.len)
    printf("%lld\n", b.data[1])
    printf("%lld\n", total(b))
    0
}
"#;

#[test]
fn native_value_generic_struct() {
    let Some(output) = compile_and_run("valuegenerics", VALUE_GENERICS) else {
        return;
    };
    assert_eq!(output, "3\n20\n60\n");
}

const SLAB_DEREF: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Entity :: struct { hp: i64, mana: i64 }

Slab :: struct($T: Type, $N: usize) {
    storage: [N]T,
    generations: [N]i64,
    free_list: [N]i64,
    free_count: i64,
}

reset :: fn(mut s: Slab<Entity, 4>) {
    mut i : i64 = 0
    while (i < 4) { s.generations[i] = 0  s.free_list[i] = 3 - i  i = i + 1 }
    s.free_count = 4
}
insert :: fn(mut s: Slab<Entity, 4>, move value: Entity) -> i64 {
    s.free_count = s.free_count - 1
    index := s.free_list[s.free_count]
    s.storage[index] = value
    packed := (s.generations[index] << 32) | index
    packed
}
release :: fn(mut s: Slab<Entity, 4>, handle: i64) {
    index := handle & 4294967295
    s.generations[index] = s.generations[index] + 1
    s.free_list[s.free_count] = index
    s.free_count = s.free_count + 1
}

main :: fn() -> i64 {
    mut world : Slab<Entity, 4> = Slab {
        storage = [Entity{hp=0,mana=0}, Entity{hp=0,mana=0}, Entity{hp=0,mana=0}, Entity{hp=0,mana=0}],
        generations = [0,0,0,0], free_list = [0,0,0,0], free_count = 0,
    }
    reset(world)
    hero : Handle<Entity> = insert(world, Entity{hp=100, mana=30})
    foe : Handle<Entity> = insert(world, Entity{hp=40, mana=10})
    printf("%lld\n", world[hero].hp)
    world[hero].hp = world[hero].hp - 25
    printf("%lld\n", world[hero].hp)
    printf("%lld\n", world[foe].mana)
    0
}
"#;

#[test]
fn native_slab_handle_place_deref() {
    let Some(output) = compile_and_run("slabderef", SLAB_DEREF) else {
        return;
    };
    assert_eq!(output, "100\n75\n10\n");
}

const SLAB_STALE_HANDLE: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Entity :: struct { hp: i64 }

Slab :: struct($T: Type, $N: usize) {
    storage: [N]T,
    generations: [N]i64,
    free_list: [N]i64,
    free_count: i64,
}

reset :: fn(mut s: Slab<Entity, 4>) {
    mut i : i64 = 0
    while (i < 4) { s.generations[i] = 0  s.free_list[i] = 3 - i  i = i + 1 }
    s.free_count = 4
}
insert :: fn(mut s: Slab<Entity, 4>, move value: Entity) -> i64 {
    s.free_count = s.free_count - 1
    index := s.free_list[s.free_count]
    s.storage[index] = value
    packed := (s.generations[index] << 32) | index
    packed
}
release :: fn(mut s: Slab<Entity, 4>, handle: i64) {
    index := handle & 4294967295
    s.generations[index] = s.generations[index] + 1
    s.free_list[s.free_count] = index
    s.free_count = s.free_count + 1
}

main :: fn() -> i64 {
    mut w : Slab<Entity, 4> = Slab { storage=[Entity{hp=0},Entity{hp=0},Entity{hp=0},Entity{hp=0}], generations=[0,0,0,0], free_list=[0,0,0,0], free_count=0 }
    reset(w)
    old : Handle<Entity> = insert(w, Entity{hp=100})
    release(w, old)
    insert(w, Entity{hp=7})
    printf("%lld\n", w[old].hp)
    0
}
"#;

#[test]
fn native_slab_stale_handle_aborts() {
    let Some(succeeded) =
        compile_and_run_status("slabstale", SLAB_STALE_HANDLE)
    else {
        return;
    };
    assert!(!succeeded, "a stale handle into a slab should abort");
}

const SLICES: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

sum :: fn(s: []i64) -> i64 {
    mut total : i64 = 0
    mut i : i64 = 0
    n := slice_len(s)
    while (i < n) {
        total = total + s[i]
        i = i + 1
    }
    total
}

main :: fn() -> i64 {
    arr := [10, 20, 30, 40]
    view : []i64 = arr
    printf("%lld\n", slice_len(view))
    printf("%lld\n", view[2])
    printf("%lld\n", sum(view))
    printf("%lld\n", sum(arr))
    0
}
"#;

#[test]
fn native_slices() {
    let Some(output) = compile_and_run("slices", SLICES) else {
        return;
    };
    assert_eq!(output, "4\n30\n100\n100\n");
}

const SLICE_OUT_OF_BOUNDS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    arr := [1, 2, 3]
    view : []i64 = arr
    mut i : i64 = 7
    printf("%lld\n", view[i])
    0
}
"#;

#[test]
fn native_slice_index_is_bounds_checked() {
    let Some(succeeded) =
        compile_and_run_status("sliceoob", SLICE_OUT_OF_BOUNDS)
    else {
        return;
    };
    assert!(!succeeded, "an out-of-range slice index should abort");
}

const NATIVE_POOL: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Entity :: struct { hp: i64, mana: i64 }

Slab :: struct {
    storage: [4]Entity,
    generations: [4]i64,
    free_list: [4]i64,
    free_count: i64,
}

hpack :: fn(index: i64, generation: i64) -> i64 { (generation << 32) | index }
hindex :: fn(handle: i64) -> i64 { handle & 4294967295 }
hgen :: fn(handle: i64) -> i64 { handle >> 32 }

slab_reset :: fn(mut p: Slab) {
    mut i : i64 = 0
    while (i < 4) { p.generations[i] = 0 p.free_list[i] = 3 - i i = i + 1 }
    p.free_count = 4
}

slab_insert :: fn(mut p: Slab, move value: Entity) -> i64 {
    p.free_count = p.free_count - 1
    index := p.free_list[p.free_count]
    p.storage[index] = value
    hpack(index, p.generations[index])
}

slab_alive :: fn(p: Slab, handle: i64) -> bool {
    p.generations[hindex(handle)] == hgen(handle)
}

slab_read :: fn(p: Slab, handle: i64) -> Entity { p.storage[hindex(handle)] }

slab_release :: fn(mut p: Slab, handle: i64) {
    index := hindex(handle)
    p.generations[index] = p.generations[index] + 1
    p.free_list[p.free_count] = index
    p.free_count = p.free_count + 1
}

main :: fn() -> i64 {
    mut world : Slab = Slab {
        storage = [
            Entity { hp = 0, mana = 0 }, Entity { hp = 0, mana = 0 },
            Entity { hp = 0, mana = 0 }, Entity { hp = 0, mana = 0 },
        ],
        generations = [0, 0, 0, 0],
        free_list = [0, 0, 0, 0],
        free_count = 0,
    }
    slab_reset(world)
    hero := slab_insert(world, Entity { hp = 100, mana = 30 })
    foe := slab_insert(world, Entity { hp = 40, mana = 10 })
    a := slab_read(world, hero)
    printf("%lld\n", a.hp)
    if (slab_alive(world, foe)) { printf("%lld\n", 1) } else { printf("%lld\n", 0) }
    slab_release(world, foe)
    reused := slab_insert(world, Entity { hp = 7, mana = 7 })
    if (slab_alive(world, reused)) { printf("%lld\n", 1) } else { printf("%lld\n", 0) }
    if (slab_alive(world, foe)) { printf("%lld\n", 9) } else { printf("%lld\n", 0) }
    0
}
"#;

#[test]
fn native_generational_pool_written_in_frost() {
    let Some(output) = compile_and_run("nativepool", NATIVE_POOL) else {
        return;
    };
    // insert, read, live-before-free, reused-slot-live, stale-handle-dead
    assert_eq!(output, "100\n1\n1\n0\n");
}

const FREESTANDING: &str = r#"
Arena :: struct($N: usize) { data: [N]u8, offset: i64 }

alloc_int :: fn(mut a: Arena<64>) -> ^i64 {
    slot := ptr_to(a.data[a.offset])
    a.offset = a.offset + sizeof(i64)
    ptr_cast($i64, slot)
}

main :: fn() -> i64 {
    mut arena : Arena<64> = Arena { data = [0; 64], offset = 0 }
    p := alloc_int(arena)
    p^ = 20
    q := alloc_int(arena)
    q^ = 22
    p^ + q^
}
"#;

#[test]
fn native_freestanding_links_without_libc() {
    if !linker_available() {
        return;
    }
    let directory = std::env::temp_dir();
    let source_path = directory.join("frost_freestanding.frost");
    let exe_path = directory.join(format!(
        "frost_freestanding{}",
        std::env::consts::EXE_SUFFIX
    ));
    std::fs::write(&source_path, FREESTANDING).unwrap();
    let frost = env!("CARGO_BIN_EXE_frost");
    let compile = Command::new(frost)
        .arg("--link")
        .arg("--freestanding")
        .arg("-o")
        .arg(&exe_path)
        .arg(&source_path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&source_path);
    if !compile.status.success() {
        // --freestanding needs gcc or clang; skip where only MSVC is present.
        return;
    }
    let run = Command::new(&exe_path).status().unwrap();
    let _ = std::fs::remove_file(&exe_path);
    // The static arena computes 20 + 22 and returns it as the exit code.
    assert_eq!(run.code(), Some(42));
}

#[test]
fn native_binding_a_void_value_is_rejected() {
    let source = "\
noop :: fn() { }\n\
main :: fn() -> i64 {\n\
    x := noop()\n\
    0\n\
}\n";
    let message = compile_error("bind_void", source);
    assert!(
        message.contains("void"),
        "binding a void value should be rejected, got:\n{message}"
    );
}

const STR_OUT_OF_BOUNDS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    greeting := "hi"
    mut i : i64 = 5
    printf("%lld\n", greeting[i])
    0
}
"#;

#[test]
fn native_str_index_is_bounds_checked() {
    let Some(succeeded) = compile_and_run_status("stroob", STR_OUT_OF_BOUNDS)
    else {
        return;
    };
    assert!(!succeeded, "an out-of-range str index should abort");
}

const POINTERS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

swap :: fn(a: ^i64, b: ^i64) {
    temp := a^
    a^ = b^
    b^ = temp
}

increment :: fn(mut x: i64) {
    x^ = x^ + 1
}

read_sum :: fn(a: i64, b: i64) -> i64 {
    a + b
}

main :: fn() -> i64 {
    mut x : i64 = 10
    mut y : i64 = 20
    swap(ptr_to(x), ptr_to(y))
    printf("%lld\n", x)
    printf("%lld\n", y)
    increment(x)
    printf("%lld\n", x)
    printf("%lld\n", read_sum(x, y))
    0
}
"#;

#[test]
fn native_pointers_and_references() {
    let Some(output) = compile_and_run("pointers", POINTERS) else {
        return;
    };
    assert_eq!(output, "20\n10\n21\n31\n");
}

const STRUCTS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct {
    x: i64,
    y: i64,
}

read_sum :: fn(p: Point) -> i64 {
    p.x + p.y
}

scale :: fn(mut p: Point, factor: i64) {
    p.x = p.x * factor
    p.y = p.y * factor
}

Mixed :: struct {
    tag: i32,
    value: i64,
    flag: u8,
}

main :: fn() -> i64 {
    mut p := Point { x = 3, y = 4 }
    printf("%lld\n", p.x)
    printf("%lld\n", read_sum(p))
    p.x = 100
    scale(p, 2)
    printf("%lld\n", p.x)
    printf("%lld\n", p.y)

    m := Mixed { tag = 7, value = 1000, flag = 1 }
    printf("%lld\n", m.tag)
    printf("%lld\n", m.value)
    printf("%lld\n", m.flag)
    0
}
"#;

#[test]
fn native_structs_and_field_access() {
    let Some(output) = compile_and_run("structs", STRUCTS) else {
        return;
    };
    assert_eq!(output, "3\n7\n200\n8\n7\n1000\n1\n");
}

const ARRAYS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

sum_array :: fn(a: [5]i64) -> i64 {
    mut total : i64 = 0
    for i in 0..5 {
        total = total + a[i]
    }
    total
}

main :: fn() -> i64 {
    mut nums := [10, 20, 30, 40, 50]
    printf("%lld\n", nums[0])
    printf("%lld\n", nums[2])
    nums[1] = 99
    printf("%lld\n", nums[1])
    mut running : i64 = 0
    for i in 0..5 {
        running = running + nums[i]
    }
    printf("%lld\n", running)
    printf("%lld\n", sum_array(nums))
    0
}
"#;

#[test]
fn native_arrays_and_indexing() {
    let Some(output) = compile_and_run("arrays", ARRAYS) else {
        return;
    };
    assert_eq!(output, "10\n30\n99\n229\n229\n");
}

const ENUMS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Result :: enum {
    Ok { value: i64 },
    Err { code: i64 },
}

unwrap_or_neg :: fn(r: Result) -> i64 {
    match r {
        case .Ok { value }: value
        case .Err { code }: 0 - code
    }
}

grade :: fn(score: i64) -> i64 {
    match score {
        case 90: 4
        case 80: 3
        case _: 0
    }
}

main :: fn() -> i64 {
    ok := Result::Ok { value = 42 }
    err := Result::Err { code = 404 }
    printf("%lld\n", unwrap_or_neg(ok))
    printf("%lld\n", unwrap_or_neg(err))
    printf("%lld\n", grade(90))
    printf("%lld\n", grade(80))
    printf("%lld\n", grade(50))
    0
}
"#;

#[test]
fn native_enums_and_match() {
    let Some(output) = compile_and_run("enums", ENUMS) else {
        return;
    };
    assert_eq!(output, "42\n-404\n4\n3\n0\n");
}

const AGGREGATE_ASSIGNMENT: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Inner :: struct { a: i64, b: i64 }
Outer :: struct { one: Inner, two: Inner }
P :: struct { x: i64, y: i64 }

main :: fn() -> i64 {
    mut o := Outer { one = Inner { a = 1, b = 2 }, two = Inner { a = 3, b = 4 } }
    o.one = o.two
    printf("%lld\n", o.one.a)
    printf("%lld\n", o.one.b)

    mut arr := [P { x = 1, y = 2 }, P { x = 9, y = 8 }]
    arr[0] = arr[1]
    printf("%lld\n", arr[0].x)
    printf("%lld\n", arr[0].y)
    0
}
"#;

#[test]
fn native_aggregate_assignment_between_places() {
    let Some(output) = compile_and_run("agg_assign", AGGREGATE_ASSIGNMENT)
    else {
        return;
    };
    assert_eq!(output, "3\n4\n9\n8\n");
}

const AGGREGATE_BY_VALUE_READS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Inner :: struct { a: i64, b: i64 }
Outer :: struct { tag: i64, inner: Inner }

get_inner :: fn(o: Outer) -> Inner { o.inner }

main :: fn() -> i64 {
    o := Outer { tag = 1, inner = Inner { a = 5, b = 6 } }
    bound := o.inner
    printf("%lld\n", bound.a)
    printf("%lld\n", bound.b)

    returned := get_inner(o)
    printf("%lld\n", returned.a)

    arr := [Inner { a = 10, b = 20 }, Inner { a = 30, b = 40 }]
    picked := arr[1]
    printf("%lld\n", picked.a)
    printf("%lld\n", picked.b)
    0
}
"#;

#[test]
fn native_aggregate_by_value_reads() {
    let Some(output) = compile_and_run("agg_reads", AGGREGATE_BY_VALUE_READS)
    else {
        return;
    };
    assert_eq!(output, "5\n6\n5\n30\n40\n");
}

const MATCH_ENUM_PLACE: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

State :: enum { Idle, Running { pid: i64 }, Done { code: i64 } }
Task :: struct { id: i64, state: State }

describe :: fn(t: Task) -> i64 {
    match t.state {
        case .Idle: 0
        case .Running { pid }: pid
        case .Done { code }: 0 - code
    }
}

first :: fn(states: [2]State, i: i64) -> i64 {
    match states[i] {
        case .Running { pid }: pid
        case .Done { code }: code
        case .Idle: 0 - 1
    }
}

main :: fn() -> i64 {
    a := Task { id = 1, state = State::Running { pid = 42 } }
    b := Task { id = 2, state = State::Idle }
    c := Task { id = 3, state = State::Done { code = 7 } }
    printf("%lld\n", describe(a))
    printf("%lld\n", describe(b))
    printf("%lld\n", describe(c))

    arr := [State::Done { code = 9 }, State::Idle]
    printf("%lld\n", first(arr, 0))
    printf("%lld\n", first(arr, 1))
    0
}
"#;

#[test]
fn native_match_on_enum_place() {
    let Some(output) = compile_and_run("match_place", MATCH_ENUM_PLACE) else {
        return;
    };
    assert_eq!(output, "42\n0\n-7\n9\n-1\n");
}

const BY_VALUE: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct {
    x: i64,
    y: i64,
}

manhattan :: fn(p: Point) -> i64 {
    p.x + p.y
}

scaled_sum :: fn(p: Point, factor: i64) -> i64 {
    mut copy := p
    copy.x = copy.x * factor
    copy.y = copy.y * factor
    copy.x + copy.y
}

main :: fn() -> i64 {
    origin := Point { x = 3, y = 4 }
    printf("%lld\n", manhattan(origin))
    other := Point { x = 5, y = 6 }
    printf("%lld\n", scaled_sum(other, 10))
    0
}
"#;

#[test]
fn native_pass_struct_by_value() {
    let Some(output) = compile_and_run("byvalue", BY_VALUE) else {
        return;
    };
    assert_eq!(output, "7\n110\n");
}

const RETURN_AGGREGATE: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct {
    x: i64,
    y: i64,
}

make_point :: fn(a: i64, b: i64) -> Point {
    p := Point { x = a, y = b }
    p
}

add_points :: fn(p: Point, q: Point) -> Point {
    r := Point { x = p.x + q.x, y = p.y + q.y }
    r
}

main :: fn() -> i64 {
    a := make_point(3, 4)
    printf("%lld\n", a.x)
    printf("%lld\n", a.y)
    sum := add_points(make_point(1, 2), make_point(10, 20))
    printf("%lld\n", sum.x)
    printf("%lld\n", sum.y)
    0
}
"#;

#[test]
fn native_return_struct_by_value() {
    let Some(output) = compile_and_run("retagg", RETURN_AGGREGATE) else {
        return;
    };
    assert_eq!(output, "3\n4\n11\n22\n");
}

const SIZEOF: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }
Entity :: struct { hp: i64, mana: i64, name: i64 }

measure :: fn(move sample: $T) -> i64 { sizeof(T) }

main :: fn() -> i64 {
    printf("%lld\n", sizeof(i64))
    printf("%lld\n", sizeof(i32))
    printf("%lld\n", sizeof(Point))
    printf("%lld\n", sizeof([4]i64))
    p := Point { x = 1, y = 2 }
    e := Entity { hp = 1, mana = 2, name = 3 }
    printf("%lld\n", measure(p))
    printf("%lld\n", measure(e))
    printf("%lld\n", measure(42))
    0
}
"#;

#[test]
fn native_sizeof_including_generic() {
    let Some(output) = compile_and_run("sizeof", SIZEOF) else {
        return;
    };
    assert_eq!(output, "8\n4\n16\n32\n16\n24\n8\n");
}

const GENERIC_FUNCTIONS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }

identity :: fn(move x: $T) -> T { x }
max_of :: fn(move a: $T, move b: $T) -> T { if (a > b) { a } else { b } }
first_of :: fn(move a: $T, move b: $T) -> T { a }
wrap :: fn(move v: $T) -> T { identity(v) }

swap :: fn(mut a: $T, mut b: $T) {
    t := a^
    a^ = b^
    b^ = t
}

main :: fn() -> i64 {
    printf("%lld\n", identity(42))
    printf("%lld\n", max_of(3, 9))

    small : i32 = 7
    widened : i64 = identity(small)
    printf("%lld\n", widened)

    p := first_of(Point { x = 5, y = 6 }, Point { x = 1, y = 2 })
    printf("%lld\n", p.x)

    w := wrap(Point { x = 8, y = 9 })
    printf("%lld\n", w.y)

    mut a : i64 = 100
    mut b : i64 = 200
    swap(a, b)
    printf("%lld\n", a)
    printf("%lld\n", b)
    0
}
"#;

#[test]
fn native_generic_functions_monomorphize() {
    let Some(output) = compile_and_run("generics", GENERIC_FUNCTIONS) else {
        return;
    };
    assert_eq!(output, "42\n9\n7\n5\n9\n200\n100\n");
}

const GENERIC_STRUCTS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }
Pair :: struct($T: Type) { first: T, second: T }
Both :: struct($T: Type, $U: Type) { left: T, right: U }
Buffer :: struct($T: Type) { data: [3]T, count: i64 }
Wrapper :: struct { pair: Pair<i64>, tag: i64 }

sum_pair :: fn(p: Pair<i64>) -> i64 { p.first + p.second }

main :: fn() -> i64 {
    p : Pair<i64> = Pair { first = 3, second = 4 }
    printf("%lld\n", p.first + p.second)
    printf("%lld\n", sum_pair(p))

    pts : Pair<Point> = Pair { first = Point { x = 1, y = 2 }, second = Point { x = 3, y = 4 } }
    printf("%lld\n", pts.first.x + pts.second.y)

    mixed : Both<i64, i32> = Both { left = 100, right = 5 }
    printf("%lld\n", mixed.left)
    printf("%lld\n", mixed.right)

    b : Buffer<i64> = Buffer { data = [7, 8, 9], count = 3 }
    printf("%lld\n", b.data[2])

    mut w := Wrapper { pair = p, tag = 99 }
    w.pair.second = 40
    printf("%lld\n", w.pair.first + w.pair.second)
    printf("%lld\n", w.tag)
    0
}
"#;

#[test]
fn native_generic_structs_monomorphize() {
    let Some(output) = compile_and_run("generic_structs", GENERIC_STRUCTS)
    else {
        return;
    };
    assert_eq!(output, "7\n7\n5\n100\n5\n9\n43\n99\n");
}

const BORROW_STRUCT_LITERAL: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }

sum :: fn(p: Point) -> i64 { p.x + p.y }
scaled :: fn(mut p: Point, k: i64) -> i64 { p.x = p.x * k  p.x + p.y }

main :: fn() -> i64 {
    printf("%lld\n", sum(Point { x = 8, y = 9 }))
    mut q := Point { x = 3, y = 4 }
    printf("%lld\n", scaled(q, 10))
    0
}
"#;

#[test]
fn native_borrow_struct_literal_at_call() {
    let Some(output) = compile_and_run("borrow_struct", BORROW_STRUCT_LITERAL)
    else {
        return;
    };
    assert_eq!(output, "17\n34\n");
}

const BORROW_AGGREGATE_LITERAL: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

State :: enum { Running { pid: i64 }, Idle }

pid_of :: fn(s: State) -> i64 {
    match s {
        case .Running { pid }: match pid {
            case 0: 0 - 1
            case _: pid
        }
        case .Idle: 0
    }
}

main :: fn() -> i64 {
    printf("%lld\n", pid_of(State::Running { pid = 42 }))
    printf("%lld\n", pid_of(State::Running { pid = 0 }))
    printf("%lld\n", pid_of(State::Idle))
    0
}
"#;

#[test]
fn native_borrow_aggregate_literal() {
    let Some(output) = compile_and_run("borrow_lit", BORROW_AGGREGATE_LITERAL)
    else {
        return;
    };
    assert_eq!(output, "42\n-1\n0\n");
}

const EXPLICIT_TYPE_ARGUMENTS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32
pool_new :: extern fn(capacity: i64, elem_size: i64) -> ^u8
pool_alloc :: extern fn(pool: ^u8, value: ^u8) -> i64
pool_get :: extern fn(pool: ^u8, handle: i64) -> ^u8

Entity :: struct { hp: i64, mana: i64 }

size_of :: fn($T: Type) -> i64 { sizeof(T) }
make_pool :: fn($T: Type, capacity: i64) -> ^u8 { pool_new(capacity, sizeof(T)) }
insert :: fn(pool: ^u8, move value: $T) -> Handle<T> { pool_alloc(pool, ptr_to(value)) }

main :: fn() -> i64 {
    printf("%lld\n", size_of($i64))
    printf("%lld\n", size_of($Entity))

    world := make_pool($Entity, 16)
    h : Handle<Entity> = insert(world, Entity { hp = 100, mana = 30 })
    printf("%lld\n", world[h].hp + world[h].mana)
    0
}
"#;

#[test]
fn native_explicit_type_arguments() {
    let Some(output) =
        compile_and_run("explicit_types", EXPLICIT_TYPE_ARGUMENTS)
    else {
        return;
    };
    assert_eq!(output, "8\n16\n130\n");
}

const GENERIC_CONSTRUCTION_INFERENCE: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32
pool_new :: extern fn(capacity: i64, elem_size: i64) -> ^u8
pool_alloc :: extern fn(pool: ^u8, value: ^u8) -> i64
pool_get :: extern fn(pool: ^u8, handle: i64) -> ^u8

Pair :: struct($T: Type) { first: T, second: T }

main :: fn() -> i64 {
    inferred := Pair { first = 30, second = 12 }
    printf("%lld\n", inferred.first + inferred.second)

    pool := pool_new(4, 16)
    mut v := Pair { first = 3, second = 4 }
    h : Handle<Pair<i64>> = pool_alloc(pool, ptr_to(v))
    printf("%lld\n", pool[h].first + pool[h].second)
    0
}
"#;

#[test]
fn native_generic_construction_inference() {
    let Some(output) =
        compile_and_run("gen_construct", GENERIC_CONSTRUCTION_INFERENCE)
    else {
        return;
    };
    assert_eq!(output, "42\n7\n");
}

const LINEAR_RESOURCE_NATIVE: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32
frost_read_i64 :: extern fn(data: File) -> i64

File :: linear struct { fd: i64 }

open :: fn(n: i64) -> File { File { fd = n } }

main :: fn() -> i64 {
    f := open(42)
    printf("%lld\n", frost_read_i64(f))
    0
}
"#;

#[test]
fn native_linear_resource_consumed_by_extern() {
    let Some(output) = compile_and_run("linear", LINEAR_RESOURCE_NATIVE) else {
        return;
    };
    assert_eq!(output, "42\n");
}

const GENERIC_INSTANCE_COMBINATIONS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Pair :: struct($T: Type) { first: T, second: T }
Op :: struct($T: Type) { f: fn($T) -> $T, seed: $T }

inc :: fn(x: i64) -> i64 { x + 1 }
swap :: fn(mut a: $T, mut b: $T) {
    t := a^
    a^ = b^
    b^ = t
}

main :: fn() -> i64 {
    arr : [3]Pair<i64> = [
        Pair { first = 1, second = 2 },
        Pair { first = 3, second = 4 },
        Pair { first = 5, second = 6 }
    ]
    mut total : i64 = 0
    for i in 0..3 {
        total = total + arr[i].first + arr[i].second
    }
    printf("%lld\n", total)

    o : Op<i64> = Op { f = inc, seed = 41 }
    g := o.f
    printf("%lld\n", g(o.seed))

    mut x : Pair<i64> = Pair { first = 1, second = 2 }
    mut y : Pair<i64> = Pair { first = 9, second = 8 }
    swap(x, y)
    printf("%lld\n", x.first + y.second)
    0
}
"#;

#[test]
fn native_generic_instance_combinations() {
    let Some(output) =
        compile_and_run("gen_instance", GENERIC_INSTANCE_COMBINATIONS)
    else {
        return;
    };
    assert_eq!(output, "21\n42\n11\n");
}

const GENERIC_FACTORIES: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Pair :: struct($T: Type) { first: T, second: T }
Box :: struct($T: Type) { value: T }
Tagged :: enum { Some { p: Pair<i64> }, None }

make_pair :: fn(move a: $T, move b: $T) -> Pair<T> { Pair { first = a, second = b } }
wrap :: fn(move x: $T) -> Box<T> { Box { value = x } }
unwrap :: fn(move b: Box<$T>) -> $T { b.value }

main :: fn() -> i64 {
    p := make_pair(3, 4)
    printf("%lld\n", p.first + p.second)

    b := wrap(99)
    printf("%lld\n", unwrap(b))

    w := Tagged::Some { p = Pair { first = 5, second = 6 } }
    r := match w {
        case .Some { p }: p.first + p.second
        case .None: 0
    }
    printf("%lld\n", r)
    0
}
"#;

#[test]
fn native_generic_factories_and_payloads() {
    let Some(output) = compile_and_run("generic_factories", GENERIC_FACTORIES)
    else {
        return;
    };
    assert_eq!(output, "7\n99\n11\n");
}

const NESTED_GENERIC_STRUCTS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Pair :: struct($T: Type) { first: T, second: T }

main :: fn() -> i64 {
    p : Pair<Pair<i64>> = Pair {
        first = Pair { first = 1, second = 2 },
        second = Pair { first = 3, second = 4 }
    }
    printf("%lld\n", p.first.second)
    printf("%lld\n", p.second.first)

    q : Pair<Pair<Pair<i64>>> = Pair {
        first = Pair { first = Pair { first = 5, second = 6 }, second = Pair { first = 7, second = 8 } },
        second = Pair { first = Pair { first = 9, second = 10 }, second = Pair { first = 11, second = 12 } }
    }
    printf("%lld\n", q.first.first.second)
    printf("%lld\n", q.second.second.first)
    0
}
"#;

#[test]
fn native_nested_generic_structs() {
    let Some(output) =
        compile_and_run("nested_generics", NESTED_GENERIC_STRUCTS)
    else {
        return;
    };
    assert_eq!(output, "2\n3\n6\n11\n");
}

const GENERIC_MULTI_PARAM: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Pair :: struct { a: i64, b: i64 }

dup :: fn(move x: $T) -> T { x }
pick_first :: fn(move a: $T, move b: $U) -> T { a }
second :: fn(move a: $T, move b: $U) -> U { b }

main :: fn() -> i64 {
    p := dup(Pair { a = 3, b = 4 })
    printf("%lld\n", p.a + p.b)

    printf("%lld\n", pick_first(42, 99))
    printf("%lld\n", second(1, 7))

    q := pick_first(Pair { a = 10, b = 20 }, 5)
    printf("%lld\n", q.b)
    0
}
"#;

#[test]
fn native_generic_multiple_type_parameters() {
    let Some(output) = compile_and_run("generics_multi", GENERIC_MULTI_PARAM)
    else {
        return;
    };
    assert_eq!(output, "7\n42\n7\n20\n");
}

const TUPLE_MATCH: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

classify :: fn(i: i64) -> i64 {
    match (i % 3, i % 5) {
        case (0, 0): 15
        case (0, _): 3
        case (_, 0): 5
        case (_, _): i
    }
}

main :: fn() -> i64 {
    for i in 1..16 {
        printf("%lld\n", classify(i))
    }
    0
}
"#;

#[test]
fn native_tuple_pattern_match() {
    let Some(output) = compile_and_run("tuple", TUPLE_MATCH) else {
        return;
    };
    assert_eq!(output, "1\n2\n3\n4\n5\n3\n7\n8\n3\n5\n11\n3\n13\n14\n15\n");
}

const POOL_HANDLE_DEREF: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32
pool_new :: extern fn(capacity: i64, elem_size: i64) -> ^u8
pool_alloc :: extern fn(pool: ^u8, value: ^u8) -> i64
pool_get :: extern fn(pool: ^u8, handle: i64) -> ^u8

Entity :: struct { hp: i64, mana: i64 }

heal :: fn(mut e: Entity, amount: i64) {
    e.hp = e.hp + amount
}
total :: fn(e: Entity) -> i64 {
    e.hp + e.mana
}

main :: fn() -> i64 {
    world := pool_new(8, 16)

    mut a := Entity { hp = 50, mana = 10 }
    ha : Handle<Entity> = pool_alloc(world, ptr_to(a))
    mut b := Entity { hp = 20, mana = 5 }
    hb : Handle<Entity> = pool_alloc(world, ptr_to(b))

    printf("%lld\n", world[ha].hp)
    world[ha].hp = 60
    printf("%lld\n", world[ha].hp)

    heal(world[ha], 15)
    printf("%lld\n", world[ha].hp)
    printf("%lld\n", total(world[ha]))

    copy := world[hb]
    printf("%lld\n", copy.mana)
    printf("%lld\n", total(world[hb]))
    0
}
"#;

#[test]
fn native_pool_handle_deref_as_place() {
    let Some(output) = compile_and_run("pool_deref", POOL_HANDLE_DEREF) else {
        return;
    };
    assert_eq!(output, "50\n60\n75\n85\n5\n25\n");
}

const FUNCTION_POINTERS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

double :: fn(x: i64) -> i64 { x * 2 }
square :: fn(x: i64) -> i64 { x * x }
increment :: fn(x: i64) -> i64 { x + 1 }

apply :: fn(f: fn(i64) -> i64, value: i64) -> i64 {
    f(value)
}

apply_twice :: fn(f: fn(i64) -> i64, value: i64) -> i64 {
    f(f(value))
}

main :: fn() -> i64 {
    printf("%lld\n", apply(double, 21))
    printf("%lld\n", apply(square, 9))
    printf("%lld\n", apply_twice(increment, 40))
    g := double
    printf("%lld\n", g(50))
    0
}
"#;

const FUNCTION_POINTER_ARRAY: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

add1 :: fn(x: i64) -> i64 { x + 1 }
mul2 :: fn(x: i64) -> i64 { x * 2 }
sub3 :: fn(x: i64) -> i64 { x - 3 }

main :: fn() -> i64 {
    ops := [add1, mul2, sub3]
    mut v : i64 = 10
    for i in 0..3 {
        f := ops[i]
        v = f(v)
    }
    printf("%lld\n", v)
    printf("%lld\n", ops[1](21))
    0
}
"#;

#[test]
fn native_function_pointer_array() {
    let Some(output) = compile_and_run("fnptr_array", FUNCTION_POINTER_ARRAY)
    else {
        return;
    };
    assert_eq!(output, "19\n42\n");
}

const TOP_LEVEL_CONSTANTS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

LIMIT :: 100
STEP :: 5
OFFSET :: 0 - 3
COMPUTED :: 2 * 4 + 1

main :: fn() -> i64 {
    printf("%lld\n", LIMIT)
    printf("%lld\n", STEP)
    printf("%lld\n", OFFSET)
    printf("%lld\n", COMPUTED)
    mut total : i64 = 0
    for i in 0..LIMIT {
        if (i % STEP == 0) { total = total + 1 }
    }
    printf("%lld\n", total)
    0
}
"#;

#[test]
fn native_top_level_constants() {
    let Some(output) = compile_and_run("constants", TOP_LEVEL_CONSTANTS) else {
        return;
    };
    assert_eq!(output, "100\n5\n-3\n9\n20\n");
}

const FORWARD_REFERENCES: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    printf("%lld\n", is_even(10))
    printf("%lld\n", is_odd(7))
    printf("%lld\n", double_it(21))
    0
}

is_even :: fn(n: i64) -> i64 {
    if (n == 0) { 1 } else { is_odd(n - 1) }
}

is_odd :: fn(n: i64) -> i64 {
    if (n == 0) { 0 } else { is_even(n - 1) }
}

double_it :: fn(x: i64) -> i64 { x * 2 }
"#;

#[test]
fn native_forward_references_and_mutual_recursion() {
    let Some(output) = compile_and_run("forward", FORWARD_REFERENCES) else {
        return;
    };
    assert_eq!(output, "1\n1\n42\n");
}

#[test]
fn native_function_pointers() {
    let Some(output) = compile_and_run("funcptr", FUNCTION_POINTERS) else {
        return;
    };
    assert_eq!(output, "42\n81\n42\n100\n");
}

const KITCHEN_SINK: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Vec3 :: struct { x: i64, y: i64, z: i64 }

Shape :: enum {
    Circle { radius: i64 },
    Box { side: i64 },
}

area :: fn(s: Shape) -> i64 {
    match s {
        case .Circle { radius }: 3 * radius * radius
        case .Box { side }: side * side
    }
}

dot :: fn(a: Vec3, b: Vec3) -> i64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fib :: fn(n: i64) -> i64 {
    if (n < 2) { n } else { fib(n - 1) + fib(n - 2) }
}

triple :: fn(x: i64) -> i64 { x * 3 }

apply_to_array :: fn(f: fn(i64) -> i64, values: [4]i64) -> i64 {
    mut total : i64 = 0
    for i in 0..4 {
        total = total + f(values[i])
    }
    total
}

main :: fn() -> i64 {
    a := Vec3 { x = 1, y = 2, z = 3 }
    b := Vec3 { x = 4, y = 5, z = 6 }
    printf("%lld\n", dot(a, b))

    c := Shape::Circle { radius = 10 }
    sq := Shape::Box { side = 7 }
    printf("%lld\n", area(c))
    printf("%lld\n", area(sq))

    printf("%lld\n", fib(15))

    nums := [1, 2, 3, 4]
    printf("%lld\n", apply_to_array(triple, nums))
    0
}
"#;

#[test]
fn native_combined_features() {
    let Some(output) = compile_and_run("kitchen", KITCHEN_SINK) else {
        return;
    };
    assert_eq!(output, "32\n300\n49\n610\n30\n");
}

const DEFER: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

work :: fn() -> i64 {
    printf("%lld\n", 1)
    defer printf("%lld\n", 2)
    defer printf("%lld\n", 3)
    printf("%lld\n", 4)
    99
}

main :: fn() -> i64 {
    r := work()
    printf("%lld\n", r)
    0
}
"#;

#[test]
fn native_defer_runs_lifo_at_return() {
    let Some(output) = compile_and_run("defer", DEFER) else {
        return;
    };
    assert_eq!(output, "1\n4\n3\n2\n99\n");
}

const DEFER_NESTED_RETURN: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

work :: fn(which: i64) -> i64 {
    defer printf("%lld\n", 8)
    defer printf("%lld\n", 9)
    if (which == 0) {
        printf("%lld\n", 1)
        return 100
    }
    printf("%lld\n", 2)
    200
}

main :: fn() -> i64 {
    printf("%lld\n", work(0))
    printf("%lld\n", work(1))
    0
}
"#;

#[test]
fn native_defer_runs_on_a_nested_early_return() {
    let Some(output) = compile_and_run("defer_nested", DEFER_NESTED_RETURN)
    else {
        return;
    };
    assert_eq!(output, "1\n9\n8\n100\n2\n9\n8\n200\n");
}

const NESTED_STRUCTS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Inner :: struct { a: i64, b: i64 }
Outer :: struct { tag: i64, inner: Inner }

sum_inner :: fn(o: Outer) -> i64 {
    o.inner.a + o.inner.b
}

main :: fn() -> i64 {
    mut o := Outer { tag = 5, inner = Inner { a = 10, b = 20 } }
    printf("%lld\n", o.tag)
    printf("%lld\n", o.inner.a)
    printf("%lld\n", sum_inner(o))
    o.inner.a = 99
    printf("%lld\n", o.inner.a)
    printf("%lld\n", sum_inner(o))
    0
}
"#;

#[test]
fn native_nested_structs() {
    let Some(output) = compile_and_run("nested", NESTED_STRUCTS) else {
        return;
    };
    assert_eq!(output, "5\n10\n30\n99\n119\n");
}

const DATA_LAYOUTS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Particle :: struct { x: i64, y: i64 }
Grid :: struct { cells: [4]i64, count: i64 }

main :: fn() -> i64 {
    mut ps := [Particle { x = 1, y = 2 }, Particle { x = 3, y = 4 }, Particle { x = 5, y = 6 }]
    printf("%lld\n", ps[0].x)
    printf("%lld\n", ps[1].y)
    ps[2].x = 99
    printf("%lld\n", ps[2].x)

    mut total : i64 = 0
    for i in 0..3 {
        total = total + ps[i].x
    }
    printf("%lld\n", total)

    mut g := Grid { cells = [10, 20, 30, 40], count = 4 }
    printf("%lld\n", g.cells[1])
    g.cells[2] = 77
    printf("%lld\n", g.cells[2])
    printf("%lld\n", g.count)
    0
}
"#;

#[test]
fn native_array_of_structs_and_struct_of_arrays() {
    let Some(output) = compile_and_run("data_layouts", DATA_LAYOUTS) else {
        return;
    };
    assert_eq!(output, "1\n4\n99\n103\n20\n77\n4\n");
}

const AGGREGATE_PAYLOADS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }

Node :: enum {
    Leaf { value: i64 },
    Pair { location: Point, weight: i64 },
}

describe :: fn(n: Node) -> i64 {
    match n {
        case .Leaf { value }: value
        case .Pair { location, weight }: location.x + location.y + weight
    }
}

main :: fn() -> i64 {
    leaf := Node::Leaf { value = 7 }
    pair := Node::Pair { location = Point { x = 3, y = 4 }, weight = 100 }
    printf("%lld\n", describe(leaf))
    printf("%lld\n", describe(pair))

    mut grid := [[1, 2, 3], [4, 5, 6]]
    printf("%lld\n", grid[0][2])
    printf("%lld\n", grid[1][1])
    grid[1][2] = 99
    printf("%lld\n", grid[1][2])
    0
}
"#;

#[test]
fn native_aggregate_enum_payloads_and_2d_arrays() {
    let Some(output) = compile_and_run("agg_payloads", AGGREGATE_PAYLOADS)
    else {
        return;
    };
    assert_eq!(output, "7\n107\n3\n5\n99\n");
}

const ENUM_BY_VALUE: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Option :: enum {
    Some { value: i64 },
    None,
}

find_first_even :: fn(a: [6]i64) -> Option {
    for i in 0..6 {
        if (a[i] % 2 == 0) {
            return Option::Some { value = a[i] }
        }
    }
    Option::None
}

unwrap_or :: fn(o: Option, fallback: i64) -> i64 {
    match o {
        case .Some { value }: value
        case .None: fallback
    }
}

main :: fn() -> i64 {
    data := [1, 3, 5, 8, 9, 10]
    r := find_first_even(data)
    printf("%lld\n", unwrap_or(r, 0 - 1))

    odds := [1, 3, 5, 7, 9, 11]
    r2 := find_first_even(odds)
    printf("%lld\n", unwrap_or(r2, 0 - 1))
    0
}
"#;

#[test]
fn native_enum_returned_by_value() {
    let Some(output) = compile_and_run("enum_byval", ENUM_BY_VALUE) else {
        return;
    };
    assert_eq!(output, "8\n-1\n");
}

const FIELD_BORROW: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }

bump :: fn(mut field: i64) {
    field^ = field^ + 100
}

origin :: fn() -> Point {
    Point { x = 7, y = 9 }
}

main :: fn() -> i64 {
    mut p := Point { x = 1, y = 2 }
    bump(p.x)
    printf("%lld\n", p.x)
    printf("%lld\n", p.y)

    q := origin()
    printf("%lld\n", q.x)
    printf("%lld\n", q.y)
    0
}
"#;

#[test]
fn native_field_borrow_and_returned_struct() {
    let Some(output) = compile_and_run("field_borrow", FIELD_BORROW) else {
        return;
    };
    assert_eq!(output, "101\n2\n7\n9\n");
}

const INTEGER_SEMANTICS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    a : i32 = 0 - 5
    b : i32 = 3
    printf("%lld\n", a / b)
    printf("%lld\n", a % b)

    big : i64 = 1000000000
    printf("%lld\n", big * 3)

    neg : i64 = 0 - 100
    shifted : i64 = neg >> 2
    printf("%lld\n", shifted)

    wide : i16 = 30000
    printf("%lld\n", wide + 100)

    mask : i64 = 255
    printf("%lld\n", mask & 15)
    printf("%lld\n", mask | 256)

    small : u8 = 200
    printf("%lld\n", small + 100)
    0
}
"#;

#[test]
fn native_integer_semantics_match() {
    let Some(output) = compile_and_run("int_semantics", INTEGER_SEMANTICS)
    else {
        return;
    };
    assert_eq!(output, "-1\n-2\n3000000000\n-25\n30100\n15\n511\n44\n");
}

const GENERATIONAL_POOL: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

pool_new :: extern fn(capacity: i64, elem_size: i64) -> ^u8
pool_alloc :: extern fn(pool: ^u8, value: ^u8) -> i64
pool_get :: extern fn(pool: ^u8, handle: i64) -> ^u8
pool_free :: extern fn(pool: ^u8, handle: i64) -> i64
pool_contains :: extern fn(pool: ^u8, handle: i64) -> i64
handle_index :: extern fn(handle: i64) -> i64
handle_generation :: extern fn(handle: i64) -> i64

Entity :: struct { hp: i64, mana: i64 }

main :: fn() -> i64 {
    p := pool_new(8, 16)

    mut a := Entity { hp = 100, mana = 30 }
    mut b := Entity { hp = 50, mana = 10 }
    ha := pool_alloc(p, ptr_to(a))
    hb := pool_alloc(p, ptr_to(b))

    printf("%lld\n", handle_index(ha))
    printf("%lld\n", handle_index(hb))
    printf("%lld\n", handle_generation(ha))

    ea : ^Entity = pool_get(p, ha)
    printf("%lld\n", ea^.hp)
    ea^.hp = 999
    ea2 : ^Entity = pool_get(p, ha)
    printf("%lld\n", ea2^.hp)

    printf("%lld\n", pool_contains(p, ha))
    printf("%lld\n", pool_free(p, ha))
    printf("%lld\n", pool_contains(p, ha))

    mut c := Entity { hp = 7, mana = 7 }
    hc := pool_alloc(p, ptr_to(c))
    printf("%lld\n", handle_index(hc))
    printf("%lld\n", handle_generation(hc))
    printf("%lld\n", pool_contains(p, ha))
    0
}
"#;

const WIDENING_BINDINGS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

widen :: fn(x: i8) -> i64 { x }

main :: fn() -> i64 {
    a : i8 = 0 - 5
    printf("%lld\n", widen(a))
    b : i16 = 0 - 1000
    c : i64 = b
    printf("%lld\n", c)
    small : i32 = 42
    wide : i64 = small
    printf("%lld\n", wide)
    0
}
"#;

#[test]
fn native_widening_in_let_bindings() {
    let Some(output) = compile_and_run("widening", WIDENING_BINDINGS) else {
        return;
    };
    assert_eq!(output, "-5\n-1000\n42\n");
}

const MATCH_RETURNS_AGGREGATE: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

P :: struct { x: i64, y: i64 }
Opt :: enum { Some { v: i64 }, None }

pick :: fn(t: i64) -> P {
    match t {
        case 0: P { x = 1, y = 2 }
        case _: P { x = 9, y = 8 }
    }
}

choose :: fn(t: i64) -> Opt {
    match t {
        case 0: Opt::None
        case _: Opt::Some { v = t * 10 }
    }
}

unwrap :: fn(o: Opt) -> i64 {
    match o {
        case .Some { v }: v
        case .None: 0 - 1
    }
}

classify :: fn(a: i64, b: i64) -> P {
    match (a % 2, b % 2) {
        case (0, 0): P { x = 1, y = 1 }
        case (_, 0): P { x = 0, y = 1 }
        case (_, _): P { x = 0, y = 0 }
    }
}

main :: fn() -> i64 {
    a := pick(0)
    b := pick(5)
    printf("%lld\n", a.x)
    printf("%lld\n", b.y)

    none := choose(0)
    some := choose(7)
    printf("%lld\n", unwrap(none))
    printf("%lld\n", unwrap(some))

    p := classify(4, 6)
    q := classify(3, 6)
    printf("%lld\n", p.x)
    printf("%lld\n", q.y)
    0
}
"#;

#[test]
fn native_match_returns_aggregate_by_value() {
    let Some(output) = compile_and_run("match_agg", MATCH_RETURNS_AGGREGATE)
    else {
        return;
    };
    assert_eq!(output, "1\n8\n-1\n70\n1\n1\n");
}

#[test]
fn native_generational_pool_and_handles() {
    let Some(output) = compile_and_run("gen_pool", GENERATIONAL_POOL) else {
        return;
    };
    assert_eq!(output, "0\n1\n0\n100\n999\n1\n1\n0\n0\n1\n0\n");
}

#[test]
fn native_showcase_examples_build_and_agree() {
    if !linker_available() {
        return;
    }
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("native");
    let mut checked = 0;
    for entry in std::fs::read_dir(&directory).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("frost") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let source = std::fs::read_to_string(&path).unwrap();
        let native = run_backend(&format!("ex_{stem}"), &source, false);
        let via_c = run_backend(&format!("ex_{stem}_c"), &source, true);
        assert_eq!(native, via_c, "backends disagree on example {stem}");
        checked += 1;
    }
    assert!(checked > 0, "no native examples were found");
}

#[test]
fn cranelift_and_c_backends_agree() {
    let programs = [
        ("diff_arith", ARITHMETIC),
        ("diff_floats", FLOATS),
        ("diff_widths", WIDTHS),
        ("diff_wrapping", WRAPPING_AND_UNARY),
        ("diff_anon", ANON_FUNCTIONS),
        ("diff_minifrost", MINIFROST),
        ("diff_strings", STRINGS),
        ("diff_strview", STR_VIEW),
        ("diff_pointers", POINTERS),
        ("diff_structs", STRUCTS),
        ("diff_arrays", ARRAYS),
        ("diff_enums", ENUMS),
        ("diff_byvalue", BY_VALUE),
        ("diff_retagg", RETURN_AGGREGATE),
        ("diff_tuple", TUPLE_MATCH),
        ("diff_funcptr", FUNCTION_POINTERS),
        ("diff_kitchen", KITCHEN_SINK),
        ("diff_defer", DEFER),
        ("diff_defernested", DEFER_NESTED_RETURN),
        ("diff_nested", NESTED_STRUCTS),
        ("diff_layouts", DATA_LAYOUTS),
        ("diff_payloads", AGGREGATE_PAYLOADS),
        ("diff_enumval", ENUM_BY_VALUE),
        ("diff_fieldborrow", FIELD_BORROW),
        ("diff_intsem", INTEGER_SEMANTICS),
        ("diff_genpool", GENERATIONAL_POOL),
        ("diff_nativepool", NATIVE_POOL),
        ("diff_slices", SLICES),
        ("diff_slabderef", SLAB_DEREF),
        ("diff_valuegenerics", VALUE_GENERICS),
        ("diff_arena", ARENA),
        ("diff_allociface", ALLOCATOR_INTERFACE),
        ("diff_dynarena", DYNAMIC_ARENA),
        ("diff_widening", WIDENING_BINDINGS),
        ("diff_matchagg", MATCH_RETURNS_AGGREGATE),
        ("diff_f32", F32_OPERATIONS),
        ("diff_forward", FORWARD_REFERENCES),
        ("diff_constants", TOP_LEVEL_CONSTANTS),
        ("diff_fnptrarr", FUNCTION_POINTER_ARRAY),
        ("diff_matchplace", MATCH_ENUM_PLACE),
        ("diff_failuresets", FAILURE_SETS),
        ("diff_allocsources", ALLOCATION_SOURCES),
        ("diff_aggreads", AGGREGATE_BY_VALUE_READS),
        ("diff_aggassign", AGGREGATE_ASSIGNMENT),
        ("diff_generics", GENERIC_FUNCTIONS),
        ("diff_sizeof", SIZEOF),
        ("diff_genmulti", GENERIC_MULTI_PARAM),
        ("diff_genstructs", GENERIC_STRUCTS),
        ("diff_poolderef", POOL_HANDLE_DEREF),
        ("diff_nestedgen", NESTED_GENERIC_STRUCTS),
        ("diff_genfactory", GENERIC_FACTORIES),
        ("diff_geninstance", GENERIC_INSTANCE_COMBINATIONS),
        ("diff_linear", LINEAR_RESOURCE_NATIVE),
        ("diff_genconstruct", GENERIC_CONSTRUCTION_INFERENCE),
        ("diff_borrowlit", BORROW_AGGREGATE_LITERAL),
        ("diff_borrowstruct", BORROW_STRUCT_LITERAL),
        ("diff_explicittypes", EXPLICIT_TYPE_ARGUMENTS),
    ];
    for (name, source) in programs {
        let native = run_backend(name, source, false);
        let via_c = run_backend(name, source, true);
        if native.is_none() {
            return;
        }
        assert_eq!(
            native, via_c,
            "Cranelift and C backends disagree on {name}"
        );
        if let Some(interpreted) = run_ir_oracle(name, source) {
            assert_eq!(
                native.as_deref(),
                Some(interpreted.as_str()),
                "IR interpreter disagrees with the native backend on {name}"
            );
        }
    }
}
