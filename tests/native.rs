use std::path::{Path, PathBuf};
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
    // The interface oracle from docs/separate-compilation.md runs on every test
    // compilation, so a module whose interface would not describe it is caught
    // here rather than when something tries to compile against one.
    command
        .env("FROST_CHECK_INTERFACES", "1")
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
        message.contains(":9:"),
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
        message.contains(":3:"),
        "expected the unknown-variable error at line 3, got:\n{message}"
    );
}

#[test]
fn borrow_exclusivity_errors_report_a_source_line() {
    let source = r#"
add_both :: fn(mut a: i64, mut b: i64) -> i64 { a + b }

main :: fn() -> i64 {
    mut value : i64 = 1
    total := add_both(value, value)
    total
}
"#;
    let message = compile_error("exclusivity", source);
    assert!(
        message.contains(":6:"),
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
        message.contains(".frost:"),
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

// A `ref` binding is a borrow of a container element, not a copy. Writing
// through it writes to the element, which is the reusable handle a container
// needs without a raw pointer. Both backends must agree, and the frame check
// must refuse letting it escape.
const REF_BINDING: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32
Node :: struct { a: i64, b: i64 }
Arena :: struct { data: []Node, count: i64 }

grow :: fn(mut ar: Arena, i: i64) {
    ref n := ar.data[i]
    n.a = i * 10
    n.b = n.a + 1
}
read_it :: fn(mut ar: Arena, i: i64) -> i64 {
    ref n := ar.data[i]
    n.a + n.b
}
main :: fn() -> i64 {
    mut backing : [4]Node = [Node{a=0,b=0}, Node{a=0,b=0}, Node{a=0,b=0}, Node{a=0,b=0}]
    mut ar := Arena { data = backing, count = 0 }
    grow(ar, 2)
    grow(ar, 3)
    printf("%lld\n", read_it(ar, 2))
    printf("%lld\n", read_it(ar, 3))
    0
}
"#;

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

// The self-hosted compiler is a set of modules that import each other, so it is
// compiled where it sits rather than copied into a temporary directory, the way
// the examples are.
fn self_hosted_source() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("selfhosted")
        .join("frost.frost")
}

// Build the self-hosted compiler, run it over `input`, and return what it wrote
// to standard output.
fn self_hosted_emits(
    name: &str,
    input: &Path,
    backend: Option<&str>,
) -> Option<String> {
    let compiler = build_self_hosted_compiler(name)?;
    let mut command = Command::new(&compiler);
    command.env("FROST_INPUT", input);
    if let Some(backend) = backend {
        command.env("FROST_BACKEND", backend);
    }
    let run = command.output().unwrap();
    let _ = std::fs::remove_file(&compiler);
    assert!(
        run.status.success(),
        "the self-hosted compiler failed on {name}:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    Some(String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"))
}

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
fn self_hosted_compiler_emits_working_c() {
    let Some(compiler) = build_self_hosted_compiler("emitsc") else {
        return;
    };
    // With no FROST_INPUT it compiles the demonstration program it carries.
    let run = Command::new(&compiler).output().unwrap();
    let _ = std::fs::remove_file(&compiler);
    assert!(run.status.success(), "the self-hosted compiler failed");
    let c_source = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    let Some(output) = compile_c_and_run("selfhosted", &c_source) else {
        return;
    };
    assert_eq!(
        output,
        "55\n120\n55\n3\n14\n11\n4\n11\n11\n16\n100\n200\n999\n5\n42\n77\n"
    );
}

// Every integer width, through both backends, including a struct whose fields
// have different alignments. The two compute layout separately (C works it out
// for itself, the assembly backend has to), so sizeof and the field reads are
// what says they agree.
const SELF_HOSTED_WIDTHS: &str = "Mixed :: struct { a: i32, b: i16, c: u8, d: i64 }\n\
     main :: fn() -> i64 {\n\
     \x20   mut small : i32 = 0 - 5\n\
     \x20   mut tiny : i16 = 300\n\
     \x20   mut byte : u8 = 200\n\
     \x20   mut big : u32 = 4000000000\n\
     \x20   mut wide : usize = 9000000000\n\
     \x20   print small\n    print tiny\n    print byte\n    print big\n\
     \x20   print wide\n    print sizeof(Mixed)\n\
     \x20   m := Mixed { a = 0 - 7, b = 9, c = 250, d = 123456789 }\n\
     \x20   print m.a\n    print m.b\n    print m.c\n    print m.d\n\
     \x20   ptr := ptr_to(m)\n    ptr^.a = 0 - 1\n\
     \x20   print m.a\n    print m.d\n    0\n}\n";

const WIDTHS_EXPECTED: &str = "-5\n300\n200\n4000000000\n9000000000\n16\n-7\n9\n250\n123456789\n-1\n123456789\n";

#[test]
fn self_hosted_integer_widths_natively() {
    let Some(output) = selfhosted_native_output("widths", SELF_HOSTED_WIDTHS)
    else {
        return;
    };
    assert_eq!(output, WIDTHS_EXPECTED);
}

#[test]
fn self_hosted_integer_widths_through_c() {
    let directory = std::env::temp_dir();
    let input = directory.join("frost_selfwidths_input.frost");
    std::fs::write(&input, SELF_HOSTED_WIDTHS).unwrap();
    let Some(c_source) = self_hosted_emits("selfwidths", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(output) = compile_c_and_run("selfwidths", &c_source) else {
        return;
    };
    assert_eq!(output, WIDTHS_EXPECTED);
}

// Floats, which the assembly backend had no registers for at all: arithmetic,
// comparisons, a call taking and returning one, an f32 field beside an f64 one,
// and an integer mixed into a float expression.
const SELF_HOSTED_FLOATS: &str = "scale :: fn(v: f64, by: f64) -> f64 {\n    v * by\n}\n\
     narrow :: fn(v: f32) -> f32 {\n    v + 0.5\n}\n\
     Pair :: struct { x: f64, y: f32 }\n\
     main :: fn() -> i64 {\n\
     \x20   a := 3.5\n    b := 2.0\n\
     \x20   print a + b\n    print a - b\n    print a * b\n    print a / b\n\
     \x20   print scale(a, 4.0)\n\
     \x20   mut small : f32 = 1.25\n    print small\n    print narrow(small)\n\
     \x20   print sizeof(Pair)\n\
     \x20   pr := Pair { x = 9.75, y = 0.5 }\n\
     \x20   print pr.x\n    print pr.y\n\
     \x20   if (a > b) { print 1 } else { print 0 }\n\
     \x20   if (a < b) { print 1 } else { print 0 }\n\
     \x20   mut n : i64 = 7\n    d := n * 1.0\n    print d / 2.0\n    0\n}\n";

const FLOATS_EXPECTED: &str =
    "5.5\n1.5\n7\n1.75\n14\n1.25\n1.75\n16\n9.75\n0.5\n1\n0\n3.5\n";

#[test]
fn self_hosted_floats_natively() {
    let Some(output) = selfhosted_native_output("floats", SELF_HOSTED_FLOATS)
    else {
        return;
    };
    assert_eq!(output, FLOATS_EXPECTED);
}

#[test]
fn self_hosted_floats_through_c() {
    let directory = std::env::temp_dir();
    let input = directory.join("frost_selffloats_input.frost");
    std::fs::write(&input, SELF_HOSTED_FLOATS).unwrap();
    let Some(c_source) = self_hosted_emits("selffloats", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(output) = compile_c_and_run("selffloats", &c_source) else {
        return;
    };
    assert_eq!(output, FLOATS_EXPECTED);
}

// Arrays and slices: a literal, indexing, assignment through an index, an array
// coercing to a slice both by binding and at a call, and `slice_len` reading a
// constant from an array's type and a field from a slice.
const SELF_HOSTED_ARRAYS: &str = "sum :: fn(numbers: []i64) -> i64 {\n\
     \x20   mut total : i64 = 0\n    mut i : i64 = 0\n\
     \x20   while (i < slice_len(numbers)) {\n\
     \x20       total = total + numbers[i]\n        i = i + 1\n    }\n\
     \x20   total\n}\n\
     main :: fn() -> i64 {\n\
     \x20   scores := [40, 10, 90, 30, 70]\n\
     \x20   view : []i64 = scores\n\
     \x20   print slice_len(scores)\n    print slice_len(view)\n\
     \x20   print scores[2]\n    print view[3]\n\
     \x20   print sum(view)\n    print sum(scores)\n\
     \x20   print sizeof([5]i64)\n\
     \x20   mut grid : [3]i64 = [7, 8, 9]\n\
     \x20   grid[1] = 99\n    print grid[1]\n    print sum(grid)\n    0\n}\n";

const ARRAYS_EXPECTED: &str = "5\n5\n90\n30\n240\n240\n40\n99\n115\n";

#[test]
fn self_hosted_arrays_natively() {
    let Some(output) = selfhosted_native_output("arrays", SELF_HOSTED_ARRAYS)
    else {
        return;
    };
    assert_eq!(output, ARRAYS_EXPECTED);
}

#[test]
fn self_hosted_arrays_through_c() {
    let directory = std::env::temp_dir();
    let input = directory.join("frost_selfarrays_input.frost");
    std::fs::write(&input, SELF_HOSTED_ARRAYS).unwrap();
    let Some(c_source) = self_hosted_emits("selfarrays", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(output) = compile_c_and_run("selfarrays", &c_source) else {
        return;
    };
    assert_eq!(output, ARRAYS_EXPECTED);
}

// `&&` used to compute both sides in the assembly backend, so a guard that
// checks a thing is safe to read before reading it read it anyway.
#[test]
fn self_hosted_boolean_operators_short_circuit() {
    let source = "trap :: fn() -> i64 {\n\
         \x20   mut ok : [1]i64 = [0]\n    ok[5] = 1\n    1\n}\n\
         main :: fn() -> i64 {\n\
         \x20   mut n : i64 = 0\n\
         \x20   if (n == 1 && trap() == 1) { print 9 } else { print 1 }\n\
         \x20   if (n == 0 || trap() == 1) { print 2 } else { print 8 }\n    0\n}\n";
    let Some(output) = selfhosted_native_output("shortcircuit", source) else {
        return;
    };
    assert_eq!(output, "1\n2\n");
}

// `test` blocks, through both backends. A failing assertion has to end its own
// test and let the run carry on, and it has to say where it was written.
#[test]
fn self_hosted_runs_test_blocks() {
    let Some(compiler) = build_self_hosted_compiler("tests") else {
        return;
    };
    let directory = std::env::temp_dir();
    let input = directory.join("frost_selftests_input.frost");
    std::fs::write(
        &input,
        "double :: fn(n: i64) -> i64 { n * 2 }\n\
         test \"doubling\" {\n\
         \x20   assert(double(2) == 4)\n    assert(double(0) == 0)\n}\n\
         test \"a failing one\" {\n    assert(double(2) == 5)\n}\n\
         main :: fn() -> i64 { print double(21)  0 }\n",
    )
    .unwrap();
    let runtime =
        format!("{}/runtime/frost_runtime.c", env!("CARGO_MANIFEST_DIR"));

    for (label, backend) in [("tc", "--emit-c"), ("tasm", "--emit-asm")] {
        let exe = directory
            .join(format!("frost_{label}{}", std::env::consts::EXE_SUFFIX));
        let run = Command::new(&compiler)
            .arg(backend)
            .arg("--test")
            .arg("-o")
            .arg(&exe)
            .arg(&input)
            .env("FROST_RUNTIME", &runtime)
            .output()
            .unwrap();
        let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
        let failure =
            String::from_utf8_lossy(&run.stderr).replace("\r\n", "\n");
        assert!(
            output.contains("test doubling ... ok"),
            "{label} did not run the passing test:\n{output}{failure}"
        );
        assert!(
            output.contains("1 passed, 1 failed"),
            "{label} did not summarize:\n{output}{failure}"
        );
        assert!(
            failure.contains("frost_selftests_input.frost:7:5"),
            "{label} did not say where the assertion was:\n{failure}"
        );
        assert!(
            !run.status.success(),
            "{label} exited zero with a failing test"
        );
        let _ = std::fs::remove_file(&exe);
    }

    // Without --test the blocks are left out entirely, so the program links
    // with no test runtime at all.
    let plain =
        directory.join(format!("frost_tplain{}", std::env::consts::EXE_SUFFIX));
    let build = Command::new(&compiler)
        .arg("--link")
        .arg("-o")
        .arg(&plain)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "a file carrying tests did not build without --test:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let ran = Command::new(&plain).output().unwrap();
    assert_eq!(
        String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n"),
        "42\n"
    );

    let _ = std::fs::remove_file(&plain);
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&compiler);
}

// The standard library, compiled and run by the Frost compiler through both
// backends. It is the largest program written in the language that the compiler
// did not write itself, and its `test` blocks are what say the answers are right
// rather than merely that the two backends agree on them.
#[test]
fn self_hosted_runs_the_standard_library_tests() {
    let Some(compiler) = build_self_hosted_compiler("stdlib") else {
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let runtime = format!("{}/runtime/frost_runtime.c", root.display());
    let directory = std::env::temp_dir();

    for (label, backend) in [("stdc", "--emit-c"), ("stdasm", "--emit-asm")] {
        let exe = directory
            .join(format!("frost_{label}{}", std::env::consts::EXE_SUFFIX));
        let run = Command::new(&compiler)
            .arg(backend)
            .arg("--test")
            .arg("-o")
            .arg(&exe)
            .arg(root.join("std").join("strings.frost"))
            .env("FROST_RUNTIME", &runtime)
            .output()
            .unwrap();
        let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
        assert!(
            output.contains("6 passed, 0 failed"),
            "{label}:\n{output}{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let _ = std::fs::remove_file(&exe);
    }
    let _ = std::fs::remove_file(&compiler);
}

// `str` is a slice of bytes, an `if` is an expression, and a body ending in one
// answers with whichever branch ran.
#[test]
fn self_hosted_strings_and_if_expressions() {
    let source = "read :: fn(s: str) -> i64 {\n\
         \x20   mut i : i64 = 0\n    mut negative := false\n\
         \x20   if (str_len(s) > 0 && s[0] == 45) { negative = true  i = 1 }\n\
         \x20   mut value : i64 = 0\n\
         \x20   while (i < str_len(s)) {\n\
         \x20       value = value * 10 + (s[i] - 48)\n        i = i + 1\n    }\n\
         \x20   if (negative) { 0 - value } else { value }\n}\n\
         main :: fn() -> i64 {\n\
         \x20   print read(\"1234567\")\n    print read(\"-7\")\n\
         \x20   print read(\"0\")\n    print str_len(\"abc\")\n    0\n}\n";
    let Some(output) = selfhosted_native_output("strif", source) else {
        return;
    };
    assert_eq!(output, "1234567\n-7\n0\n3\n");
}

const CLI_PROGRAM: &str = "fib :: fn(n: i64) -> i64 {\n\
     \x20   if (n < 2) { return n }\n\
     \x20   return fib(n - 1) + fib(n - 2)\n}\n\
     main :: fn() -> i64 {\n    print fib(10)\n    print 6 * 7\n    0\n}\n";

// The compiler names its input on the command line rather than in the
// environment, writes where -o says, and finishes the build itself.
#[test]
fn self_hosted_compiler_takes_a_command_line() {
    let Some(compiler) = build_self_hosted_compiler("cli") else {
        return;
    };
    let directory = std::env::temp_dir();
    let input = directory.join("frost_cli_input.frost");
    std::fs::write(&input, CLI_PROGRAM).unwrap();

    for (label, backend, expected) in [
        ("cli_c", "--emit-c", "int main(void)"),
        ("cli_asm", "--emit-asm", ".text"),
    ] {
        let output = directory.join(format!("frost_{label}.out"));
        let emit = Command::new(&compiler)
            .arg(backend)
            .arg("-o")
            .arg(&output)
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            emit.status.success(),
            "{label} failed:\n{}",
            String::from_utf8_lossy(&emit.stderr)
        );
        assert!(
            emit.stdout.is_empty(),
            "{label} wrote to standard output as well as to -o"
        );
        let written = std::fs::read_to_string(&output).unwrap();
        assert!(
            written.contains(expected),
            "{label} did not write {expected}:\n{written}"
        );
        let _ = std::fs::remove_file(&output);
    }

    // --link finishes the build, through either backend.
    for (label, backend) in [("link_c", "--emit-c"), ("link_asm", "--emit-asm")]
    {
        let exe = directory
            .join(format!("frost_{label}{}", std::env::consts::EXE_SUFFIX));
        let build = Command::new(&compiler)
            .arg(backend)
            .arg("--link")
            .arg("-o")
            .arg(&exe)
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{label} failed to link:\n{}",
            String::from_utf8_lossy(&build.stderr)
        );
        let run = Command::new(&exe).output().unwrap();
        assert_eq!(
            String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"),
            "55\n42\n",
            "{label} produced the wrong program"
        );
        let _ = std::fs::remove_file(&exe);
    }

    let unknown = Command::new(&compiler).arg("--nonsense").output().unwrap();
    assert!(!unknown.status.success(), "an unknown option was accepted");

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&compiler);
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

// The self-hosting fixpoint: the self-hosted compiler compiles its own source, the resulting
// compiler compiles that source again, and the two emitted translation units are
// byte-identical (the classic three-stage bootstrap check).
#[test]
fn self_hosting_is_a_fixpoint() {
    if c_compiler().is_none() {
        return;
    }
    let source_file = self_hosted_source();

    // Stage 1: the compiler the bootstrap built compiles its own source.
    let Some(gen1_c) = self_hosted_emits("selfhost1", &source_file, None)
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
        .env("FROST_INPUT", &source_file)
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

// Self-hosting with no C compiler in the loop: the self-hosted compiler emits
// assembly for its own source, that assembly is assembled into a compiler, and
// that compiler emits the same assembly for the same source. The fixpoint is
// the proof, since a compiler built by a different route agreeing byte for byte
// leaves nowhere for a codegen mistake to hide.
#[test]
fn native_self_hosting_is_a_fixpoint() {
    let Some(compiler) = build_self_hosted_compiler("nativefix") else {
        return;
    };
    let directory = std::env::temp_dir();
    let source = self_hosted_source();

    let emit_self = |exe: &PathBuf, stage: &str| -> String {
        let emit = Command::new(exe)
            .env("FROST_BACKEND", "asm")
            .env("FROST_INPUT", &source)
            .output()
            .unwrap();
        assert!(
            emit.status.success(),
            "the {stage} compiler failed to emit assembly for its own source ({}, {} bytes out):\n{}",
            emit.status,
            emit.stdout.len(),
            String::from_utf8_lossy(&emit.stderr)
        );
        String::from_utf8_lossy(&emit.stdout).replace("\r\n", "\n")
    };

    let stage1 = emit_self(&compiler, "frost-built");
    assert!(
        stage1.lines().count() > 10000,
        "assembly for the compiler implausibly small ({} lines)",
        stage1.lines().count()
    );

    let asm_path = directory.join("frost_nativefix.s");
    let stage1_exe = directory
        .join(format!("frost_nativefix1{}", std::env::consts::EXE_SUFFIX));
    std::fs::write(&asm_path, &stage1).unwrap();
    let runtime =
        format!("{}/runtime/frost_runtime.c", env!("CARGO_MANIFEST_DIR"));
    let assembled = Command::new(c_compiler().unwrap())
        .arg(&asm_path)
        .arg(&runtime)
        .arg("-o")
        .arg(&stage1_exe)
        .output()
        .unwrap();
    assert!(
        assembled.status.success(),
        "the compiler's own assembly did not assemble:\n{}",
        String::from_utf8_lossy(&assembled.stderr)
    );

    let stage2 = emit_self(&stage1_exe, "assembly-built");

    let _ = std::fs::remove_file(&asm_path);
    let _ = std::fs::remove_file(&stage1_exe);
    let _ = std::fs::remove_file(&compiler);

    assert_eq!(stage1, stage2, "native self-hosting is not a fixpoint");
}

// the self-hosted compiler's native backend: it emits x64 assembly rather than C, so a build
// pays an assembler rather than a C compiler. Emit it, assemble it, run it.
#[test]
fn self_hosted_native_backend_emits_working_assembly() {
    if c_compiler().is_none() || !linker_available() {
        return;
    }
    let directory = std::env::temp_dir();
    let Some(compiler) = build_self_hosted_compiler("mfasm") else {
        return;
    };

    let program = "fib :: fn(n: i64) -> i64 {\n\
                   \x20   if (n < 2) { return n }\n\
                   \x20   return fib(n - 1) + fib(n - 2)\n}\n\
                   main :: fn() -> i64 {\n\
                   \x20   mut i : i64 = 0\n\
                   \x20   while (i < 10) {\n        print fib(i)\n        i = i + 1\n    }\n\
                   \x20   print 6 * 7\n    0\n}\n";
    let input = directory.join("frost_mfasm_input.frost");
    std::fs::write(&input, program).unwrap();

    let emit = Command::new(&compiler)
        .env("FROST_BACKEND", "asm")
        .env("FROST_INPUT", &input)
        .output()
        .unwrap();
    assert!(
        emit.status.success(),
        "native backend refused the program:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let assembly = String::from_utf8_lossy(&emit.stdout).to_string();
    assert!(assembly.contains(".text"), "got:\n{assembly}");

    let asm_path = directory.join("frost_mfasm_out.s");
    let exe_path = directory
        .join(format!("frost_mfasm_out{}", std::env::consts::EXE_SUFFIX));
    std::fs::write(&asm_path, &assembly).unwrap();
    let assembled = Command::new(c_compiler().unwrap())
        .arg(&asm_path)
        .arg("-o")
        .arg(&exe_path)
        .output()
        .unwrap();
    assert!(
        assembled.status.success(),
        "emitted assembly did not assemble:\n{}",
        String::from_utf8_lossy(&assembled.stderr)
    );

    let run = Command::new(&exe_path).output().unwrap();
    let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");

    let _ = std::fs::remove_file(&compiler);
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&asm_path);
    let _ = std::fs::remove_file(&exe_path);

    assert_eq!(output, "0\n1\n1\n2\n3\n5\n8\n13\n21\n34\n42\n");
}

// Put a program through the self-hosted compiler's native backend, assemble the
// result and run it, returning what it printed. Nothing here goes through a C
// compiler except the assembler and linker.
fn selfhosted_native_output(name: &str, source: &str) -> Option<String> {
    let compiler = build_self_hosted_compiler(name)?;
    let directory = std::env::temp_dir();
    let input = directory.join(format!("frost_nb_{name}.frost"));
    std::fs::write(&input, source).unwrap();

    let emit = Command::new(&compiler)
        .env("FROST_BACKEND", "asm")
        .env("FROST_INPUT", &input)
        .output()
        .unwrap();
    assert!(
        emit.status.success(),
        "native backend refused {name}:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );

    let asm_path = directory.join(format!("frost_nb_{name}.s"));
    let exe_path = directory
        .join(format!("frost_nb_{name}{}", std::env::consts::EXE_SUFFIX));
    std::fs::write(&asm_path, String::from_utf8_lossy(&emit.stdout).as_ref())
        .unwrap();
    let assembled = Command::new(c_compiler().unwrap())
        .arg(&asm_path)
        .arg("-o")
        .arg(&exe_path)
        .output()
        .unwrap();
    assert!(
        assembled.status.success(),
        "emitted assembly for {name} did not assemble:\n{}",
        String::from_utf8_lossy(&assembled.stderr)
    );

    let run = Command::new(&exe_path).output().unwrap();
    assert!(run.status.success(), "{name} exited with failure");
    let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&asm_path);
    let _ = std::fs::remove_file(&exe_path);
    Some(output)
}

// Each caller gets its own copy, named after itself. The test binary runs its
// tests in parallel, so a shared path is two tests writing one file.
fn build_self_hosted_compiler(name: &str) -> Option<PathBuf> {
    if c_compiler().is_none() || !linker_available() {
        return None;
    }
    let directory = std::env::temp_dir();
    let compiler = directory.join(format!(
        "frost_selfhosted_{name}{}",
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

// Each language feature the native backend generates code for, checked by
// running the program it produces.
#[test]
fn native_backend_covers_the_language() {
    let cases: &[(&str, &str, &str)] = &[
        (
            "arith",
            "main :: fn() -> i64 {\n    print 2 + 3 * 4\n    print 20 / 6\n    print 20 % 6\n    print 0 - 7\n    0\n}\n",
            "14\n3\n2\n-7\n",
        ),
        (
            "compare",
            "main :: fn() -> i64 {\n    print 1 < 2\n    print 2 <= 2\n    print 3 > 4\n    print 3 >= 4\n    print 5 == 5\n    print 5 != 5\n    0\n}\n",
            "1\n1\n0\n0\n1\n0\n",
        ),
        (
            "recursion",
            "fib :: fn(n: i64) -> i64 {\n    if (n < 2) { return n }\n    return fib(n - 1) + fib(n - 2)\n}\nmain :: fn() -> i64 {\n    print fib(12)\n    0\n}\n",
            "144\n",
        ),
        (
            "loops",
            "main :: fn() -> i64 {\n    mut total : i64 = 0\n    mut i : i64 = 1\n    while (i <= 100) {\n        total = total + i\n        i = i + 1\n    }\n    print total\n    0\n}\n",
            "5050\n",
        ),
        (
            "structs",
            "P :: struct { x: i64, y: i64 }\nsum :: fn(q: P) -> i64 { return q.x + q.y }\nbump :: fn(mut q: P) { q.x = q.x + 100 }\nmain :: fn() -> i64 {\n    mut a : P = P { x = 3, y = 4 }\n    print sum(a)\n    bump(a)\n    print a.x\n    b := a\n    print b.x\n    print sizeof(P)\n    0\n}\n",
            "7\n103\n103\n16\n",
        ),
        (
            "nested_structs",
            "Inner :: struct { a: i64, b: i64, c: i64 }\n\
             Outer :: struct { first: i64, mid: Inner, last: i64 }\n\
             main :: fn() -> i64 {\n\
             \x20   mut o : Outer = Outer { first = 1, mid = Inner { a = 2, b = 3, c = 4 }, last = 5 }\n\
             \x20   print o.first\n    print o.mid.a\n    print o.mid.c\n    print o.last\n\
             \x20   print sizeof(Outer)\n    o.mid.b = 99\n    print o.mid.b\n    print o.last\n    0\n}\n",
            "1\n2\n4\n5\n40\n99\n5\n",
        ),
        (
            "pointers",
            "P :: struct { x: i64, y: i64 }\nmain :: fn() -> i64 {\n    mut a : P = P { x = 3, y = 4 }\n    r : ^P = ptr_to(a)\n    print r^.y\n    r^.y = 55\n    print a.y\n    0\n}\n",
            "4\n55\n",
        ),
        (
            "match",
            "classify :: fn(n: i64) -> i64 {\n    mut r : i64 = 0\n    match n {\n        case 0: r = 100\n        case 1: r = 200\n        case _: r = 300\n    }\n    return r\n}\nmain :: fn() -> i64 {\n    print classify(0)\n    print classify(1)\n    print classify(9)\n    0\n}\n",
            "100\n200\n300\n",
        ),
        (
            "manyargs",
            "six :: fn(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64) -> i64 {\n    return a + b + c + d + e + f\n}\nmain :: fn() -> i64 {\n    print six(1, 2, 3, 4, 5, 6)\n    print six(10, 20, 30, 40, 50, 60)\n    0\n}\n",
            "21\n210\n",
        ),
        (
            "nested_calls",
            "add :: fn(a: i64, b: i64) -> i64 { return a + b }\nmain :: fn() -> i64 {\n    print add(add(1, 2), add(3, 4))\n    0\n}\n",
            "10\n",
        ),
    ];

    for (name, source, expected) in cases {
        let Some(output) = selfhosted_native_output(name, source) else {
            return;
        };
        assert_eq!(&output, expected, "native backend output for {name}");
    }
}

// Build the self-hosted compiler, feed it a program, and return what it wrote to stderr after
// rejecting it. The self-hosted compiler answers for its own errors rather than deferring them
// to whatever compiles its output.
fn self_hosted_rejects(name: &str, source: &str) -> Option<String> {
    if !linker_available() {
        return None;
    }
    let directory = std::env::temp_dir();
    let compiler = build_self_hosted_compiler(&format!("ck_{name}"))?;

    let input = directory.join(format!("frost_mfck_input_{name}.frost"));
    std::fs::write(&input, source).unwrap();
    let run = Command::new(&compiler)
        .env("FROST_INPUT", &input)
        .output()
        .unwrap();

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&compiler);

    assert!(
        !run.status.success(),
        "expected the self-hosted compiler to reject the program"
    );
    Some(String::from_utf8_lossy(&run.stderr).to_string())
}

// A diagnostic names the file, the line and the column it is about, not just
// what went wrong. Every file's text is laid into one buffer, so the line has
// to be counted from where that file's own text begins.
#[test]
fn self_hosted_errors_name_a_position() {
    let source = "Point :: struct { x: i64, y: i64 }\n\
         main :: fn() -> i64 {\n\
         \x20   p := Point { x = 1, y = 2 }\n\
         \x20   print p.z\n    0\n}\n";
    let Some(message) = self_hosted_rejects("position", source) else {
        return;
    };
    assert!(
        message.contains(":4:13: struct 'Point' has no field 'z'"),
        "expected a file, line and column, got:\n{message}"
    );
    assert!(
        message.contains("frost_mfck_input_position.frost:"),
        "expected the file it came from, got:\n{message}"
    );
}

#[test]
fn self_hosted_rejects_a_call_to_an_undefined_function() {
    let source = "main :: fn() -> i64 {\n    return no_such_fn(1)\n}\n";
    let Some(message) = self_hosted_rejects("undef", source) else {
        return;
    };
    assert!(
        message.contains("undefined function"),
        "expected an undefined-function error, got:\n{message}"
    );
}

#[test]
fn self_hosted_rejects_an_undefined_variable() {
    let source = "main :: fn() -> i64 {\n    x := 1\n    return x + zzz\n}\n";
    let Some(message) = self_hosted_rejects("undefvar", source) else {
        return;
    };
    assert!(
        message.contains("undefined variable"),
        "expected an undefined-variable error, got:\n{message}"
    );
}

#[test]
fn self_hosted_rejects_a_field_the_struct_does_not_have() {
    let source = "P :: struct { x: i64, y: i64 }\n\
                  main :: fn() -> i64 {\n\
                  \x20   p := P { x = 1, y = 2 }\n    return p.zzz\n}\n";
    let Some(message) = self_hosted_rejects("badfield", source) else {
        return;
    };
    assert!(
        message.contains("has no field"),
        "expected an unknown-field error, got:\n{message}"
    );
}

#[test]
fn self_hosted_rejects_returning_the_wrong_type() {
    let source = "P :: struct { x: i64 }\n\
                  bad :: fn() -> i64 {\n\
                  \x20   p := P { x = 1 }\n    return p\n}\n\
                  main :: fn() -> i64 { return bad() }\n";
    let Some(message) = self_hosted_rejects("badreturn", source) else {
        return;
    };
    assert!(
        message.contains("wrong type"),
        "expected a return-type error, got:\n{message}"
    );
}

#[test]
fn self_hosted_rejects_an_argument_of_the_wrong_type() {
    let source = "P :: struct { x: i64 }\n\
                  take :: fn(n: i64) -> i64 { n }\n\
                  main :: fn() -> i64 {\n\
                  \x20   p := P { x = 1 }\n    return take(p)\n}\n";
    let Some(message) = self_hosted_rejects("badarg", source) else {
        return;
    };
    assert!(
        message.contains("wrong type"),
        "expected an argument-type error, got:\n{message}"
    );
}

#[test]
fn self_hosted_rejects_assigning_the_wrong_type() {
    let source = "P :: struct { x: i64 }\n\
                  main :: fn() -> i64 {\n\
                  \x20   p := P { x = 1 }\n\
                  \x20   mut n : i64 = 0\n    n = p\n    return n\n}\n";
    let Some(message) = self_hosted_rejects("badassign", source) else {
        return;
    };
    assert!(
        message.contains("wrong type"),
        "expected an assignment-type error, got:\n{message}"
    );
}

#[test]
fn self_hosted_rejects_a_use_after_move() {
    let source = "P :: struct { x: i64 }\n\
                  take :: fn(move q: P) -> i64 { q.x }\n\
                  main :: fn() -> i64 {\n\
                  \x20   p := P { x = 1 }\n\
                  \x20   a := take(p)\n    b := take(p)\n    return a + b\n}\n";
    let Some(message) = self_hosted_rejects("useafmove", source) else {
        return;
    };
    assert!(
        message.contains("moved value"),
        "expected a use-after-move error, got:\n{message}"
    );
}

// `uses A` and `with a { }` in the self-hosted compiler: the capability is an
// implicit trailing parameter, forwarded from one `uses` function to the next
// and supplied by the region at the top.
const SELF_HOSTED_ALLOCATION_SOURCES: &str = "Arena :: struct { offset: i64 }\n\
     bump :: fn(mut a: Arena, amount: i64) -> i64 {\n\
     \x20   a.offset = a.offset + amount\n    a.offset\n}\n\
     take :: fn(amount: i64) -> i64 uses Arena { bump(arena, amount) }\n\
     nested :: fn() -> i64 uses Arena { take(10) + take(32) }\n\
     main :: fn() -> i64 {\n\
     \x20   mut arena : Arena = Arena { offset = 0 }\n\
     \x20   mut result : i64 = 0\n\
     \x20   with arena { result = nested() }\n\
     \x20   print result\n    print arena.offset\n    0\n}\n";

#[test]
fn self_hosted_allocation_sources_through_c() {
    let directory = std::env::temp_dir();
    let input = directory.join("frost_selfalloc_input.frost");
    std::fs::write(&input, SELF_HOSTED_ALLOCATION_SOURCES).unwrap();
    let Some(c_source) = self_hosted_emits("selfalloc", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    assert!(
        c_source.contains("struct Arena* arena"),
        "the capability did not become a parameter:\n{c_source}"
    );
    let Some(output) = compile_c_and_run("selfalloc", &c_source) else {
        return;
    };
    assert_eq!(output, "52\n42\n");
}

#[test]
fn self_hosted_allocation_sources_natively() {
    let Some(output) =
        selfhosted_native_output("alloc", SELF_HOSTED_ALLOCATION_SOURCES)
    else {
        return;
    };
    assert_eq!(output, "52\n42\n");
}

// A `uses` call with no capability in reach is rejected rather than allocating
// from somewhere unnamed.
#[test]
fn self_hosted_rejects_a_uses_call_with_no_capability() {
    let source = "Arena :: struct { offset: i64 }\n\
                  grab :: fn() -> i64 uses Arena { 1 }\n\
                  main :: fn() -> i64 { grab() }\n";
    let Some(message) = self_hosted_rejects("nocapability", source) else {
        return;
    };
    assert!(
        message.contains("needs an allocation capability"),
        "expected a missing-capability error, got:\n{message}"
    );
}

// Regions in the self-hosted compiler: a raw pointer into the arena may not
// outlive the `with` block it was taken in.
#[test]
fn self_hosted_rejects_a_region_pointer_stored_outside() {
    let source = "Arena :: struct { offset: i64 }\n\
                  alloc :: fn() -> ^i64 uses Arena { ptr_to(arena^.offset) }\n\
                  main :: fn() -> i64 {\n\
                  \x20   mut arena : Arena = Arena { offset = 0 }\n\
                  \x20   mut escaped : ^i64 = ptr_to(arena.offset)\n\
                  \x20   with arena { escaped = alloc() }\n\
                  \x20   escaped^\n}\n";
    let Some(message) = self_hosted_rejects("regionstore", source) else {
        return;
    };
    assert!(
        message.contains("escapes its region"),
        "expected a region escape error, got:\n{message}"
    );
}

#[test]
fn self_hosted_rejects_a_returned_region_pointer() {
    let source = "Arena :: struct { offset: i64 }\n\
                  alloc :: fn() -> ^i64 uses Arena { ptr_to(arena^.offset) }\n\
                  grab :: fn() -> ^i64 {\n\
                  \x20   mut arena : Arena = Arena { offset = 0 }\n\
                  \x20   with arena { return alloc() }\n\
                  \x20   ptr_to(arena.offset)\n}\n\
                  main :: fn() -> i64 { 0 }\n";
    let Some(message) = self_hosted_rejects("regionreturn", source) else {
        return;
    };
    assert!(
        message.contains("being returned"),
        "expected a returned-pointer region error, got:\n{message}"
    );
}

// A binding declared inside the region may hold a region pointer, and reading
// through it is what the region is for, so this must be accepted.
#[test]
fn self_hosted_accepts_a_region_pointer_held_inside() {
    let source = "Arena :: struct { offset: i64 }\n\
                  alloc :: fn() -> ^i64 uses Arena { ptr_to(arena^.offset) }\n\
                  main :: fn() -> i64 {\n\
                  \x20   mut arena : Arena = Arena { offset = 7 }\n\
                  \x20   mut result : i64 = 0\n\
                  \x20   with arena {\n        held := alloc()\n\
                  \x20       result = held^\n    }\n\
                  \x20   print result\n    0\n}\n";
    let Some(output) = selfhosted_native_output("regionheld", source) else {
        return;
    };
    assert_eq!(output, "7\n");
}

// The interprocedural half: a `uses` function may hand an arena pointer back to
// its caller, whose region owns the arena, but may not store one into a
// parameter, which outlives the call.
#[test]
fn self_hosted_rejects_a_region_pointer_stored_into_a_parameter() {
    let source = "Arena :: struct { offset: i64 }\n\
                  Holder :: struct { slot: ^i64 }\n\
                  leak :: fn(mut h: Holder) -> i64 uses Arena {\n\
                  \x20   h.slot = ptr_to(arena^.offset)\n    0\n}\n\
                  main :: fn() -> i64 { 0 }\n";
    let Some(message) = self_hosted_rejects("regionparam", source) else {
        return;
    };
    assert!(
        message.contains("stored into a parameter"),
        "expected a parameter-leak region error, got:\n{message}"
    );
}

#[test]
fn self_hosted_accepts_a_region_pointer_handed_to_the_caller() {
    let source = "Arena :: struct { offset: i64 }\n\
                  alloc :: fn() -> ^i64 uses Arena {\n\
                  \x20   slot := ptr_to(arena^.offset)\n    return slot\n}\n\
                  main :: fn() -> i64 {\n\
                  \x20   mut arena : Arena = Arena { offset = 5 }\n\
                  \x20   mut result : i64 = 0\n\
                  \x20   with arena {\n        held := alloc()\n\
                  \x20       result = held^\n    }\n\
                  \x20   print result\n    0\n}\n";
    let Some(output) = selfhosted_native_output("regionhandback", source)
    else {
        return;
    };
    assert_eq!(output, "5\n");
}

// Imports in the self-hosted compiler: a nested import, a diamond (both the
// root and its dependency name the same file) and a struct declared in one file
// used in another.
#[test]
fn self_hosted_resolves_imports() {
    let Some(compiler) = build_self_hosted_compiler("imports") else {
        return;
    };
    let directory = std::env::temp_dir().join("frost_import_test");
    let library = directory.join("lib");
    std::fs::create_dir_all(&library).unwrap();
    std::fs::write(
        library.join("math.frost"),
        "export square, Pair\n\
         Pair :: struct { a: i64, b: i64 }\n\
         square :: fn(n: i64) -> i64 { n * n }\n",
    )
    .unwrap();
    std::fs::write(
        library.join("extra.frost"),
        "export cube\n\
         import \"math.frost\"\n\
         cube :: fn(n: i64) -> i64 { n * square(n) }\n",
    )
    .unwrap();
    let root = directory.join("app.frost");
    std::fs::write(
        &root,
        "import \"lib/extra.frost\"\n\
         import \"lib/math.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   print square(7)\n    print cube(3)\n\
         \x20   p := Pair { a = 4, b = 5 }\n    print p.a + p.b\n    0\n}\n",
    )
    .unwrap();

    let emit = Command::new(&compiler)
        .env("FROST_INPUT", &root)
        .output()
        .unwrap();
    assert!(
        emit.status.success(),
        "imports were not resolved:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let c_source = String::from_utf8_lossy(&emit.stdout).replace("\r\n", "\n");
    let Some(output) = compile_c_and_run("imports", &c_source) else {
        return;
    };

    let _ = std::fs::remove_dir_all(&directory);
    let _ = std::fs::remove_file(&compiler);
    assert_eq!(output, "49\n27\n9\n");
}

// A file names another that names it back. Each file lands in the buffer once,
// so this settles rather than running forever.
#[test]
fn self_hosted_survives_an_import_cycle() {
    let Some(compiler) = build_self_hosted_compiler("importcycle") else {
        return;
    };
    let directory = std::env::temp_dir().join("frost_import_cycle");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("second.frost"),
        "export beta\n\
         import \"first.frost\"\n\
         beta :: fn() -> i64 { 4 }\n",
    )
    .unwrap();
    let root = directory.join("first.frost");
    std::fs::write(
        &root,
        "import \"second.frost\"\n\
         alpha :: fn() -> i64 { 3 }\n\
         main :: fn() -> i64 { print alpha() + beta()\n    0 }\n",
    )
    .unwrap();

    let emit = Command::new(&compiler)
        .env("FROST_INPUT", &root)
        .output()
        .unwrap();
    assert!(
        emit.status.success(),
        "an import cycle was not resolved:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let c_source = String::from_utf8_lossy(&emit.stdout).replace("\r\n", "\n");
    let Some(output) = compile_c_and_run("importcycle", &c_source) else {
        return;
    };

    let _ = std::fs::remove_dir_all(&directory);
    let _ = std::fs::remove_file(&compiler);
    assert_eq!(output, "7\n");
}

// A module may export a function that returns a type it does not export, which
// the visibility rule allows and which used to not compile: the renamer walked
// a function's parameters and body but skipped its return signature, so the
// private type kept its un-renamed name and nothing could resolve it.
#[test]
fn an_exported_function_may_return_an_unexported_type() {
    let directory = std::env::temp_dir().join("frost_private_return");
    let library = directory.join("lib");
    std::fs::create_dir_all(&library).unwrap();
    std::fs::write(
        library.join("hidden.frost"),
        "export make\n\
         Hidden :: struct { v: i64 }\n\
         make :: fn(x: i64) -> Hidden { Hidden { v = x } }\n",
    )
    .unwrap();
    let root = directory.join("private_return_app.frost");
    std::fs::write(
        &root,
        "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         import \"lib/hidden.frost\"\n\
         main :: fn() -> i64 { h := make(7)  printf(\"%lld\\n\", h.v)  0 }\n",
    )
    .unwrap();

    if !linker_available() {
        let _ = std::fs::remove_dir_all(&directory);
        return;
    }
    let exe = directory.join(format!("app{}", std::env::consts::EXE_SUFFIX));
    let built = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("-o")
        .arg(&exe)
        .arg(&root)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "a private return type did not resolve:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let ran = Command::new(&exe).output().unwrap();
    let output = String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n");
    let _ = std::fs::remove_dir_all(&directory);
    assert_eq!(output, "7\n");
}

// Step 4 of docs/separate-compilation.md, as an oracle. With
// FROST_BUILD_FROM_INTERFACES an imported module contributes what its interface
// says and nothing else, so producing the same program either way is the
// evidence that an interface is sufficient. The module here uses the things
// most likely to be missing from one: a private helper reached only through an
// export, a generic whose body the caller has to instantiate, an enum, a struct
// returned by an exported function without being exported itself, and a private
// name that nothing reaches and which the interface therefore drops.
#[test]
fn a_program_built_from_interfaces_is_the_same_program() {
    let directory = std::env::temp_dir().join("frost_from_interfaces");
    let library = directory.join("lib");
    std::fs::create_dir_all(&library).unwrap();
    std::fs::write(
        library.join("shapes.frost"),
        "export area, describe, biggest, Shape\n\
         Shape :: enum { Circle { r: i64 }, Rect { w: i64, h: i64 } }\n\
         Report :: struct { value: i64, kind: i64 }\n\
         scale :: fn(x: i64) -> i64 { x * 2 }\n\
         never_used :: fn() -> i64 { 999 }\n\
         area :: fn(s: Shape) -> i64 {\n\
         \x20   match s {\n\
         \x20       case .Circle { r }: scale(3 * r * r)\n\
         \x20       case .Rect { w, h }: w * h\n\
         \x20   }\n\
         }\n\
         describe :: fn(s: Shape) -> Report { Report { value = area(s), kind = 1 } }\n\
         biggest :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T, move y: $T) -> $T {\n\
         \x20   mut best := x\n    if (before(y, best)) { best = y }\n    best\n\
         }\n",
    )
    .unwrap();
    let root = directory.join("from_interfaces_app.frost");
    std::fs::write(
        &root,
        "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         import \"lib/shapes.frost\"\n\
         wider :: fn(a: i64, b: i64) -> bool { a > b }\n\
         main :: fn() -> i64 {\n\
         \x20   printf(\"%lld\\n\", area(Shape::Rect { w = 4, h = 5 }))\n\
         \x20   report := describe(Shape::Circle { r = 2 })\n\
         \x20   printf(\"%lld\\n\", report.value)\n\
         \x20   printf(\"%lld\\n\", biggest($i64, $wider, 7, 3))\n\
         \x20   0\n\
         }\n",
    )
    .unwrap();

    let frost = env!("CARGO_BIN_EXE_frost");
    let emit = |from_interfaces: bool, name: &str| {
        let c_path = directory.join(format!("{name}.c"));
        let output = Command::new(frost)
            .env("FROST_CHECK_INTERFACES", "1")
            .env(
                "FROST_BUILD_FROM_INTERFACES",
                if from_interfaces { "1" } else { "0" },
            )
            .arg("--emit-c")
            .arg("-o")
            .arg(&c_path)
            .arg(&root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "building {name} failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::read_to_string(&c_path).unwrap()
    };

    let from_source = emit(false, "source");
    let from_interfaces = emit(true, "interfaces");

    // The private name nothing reaches is dropped by the interface, which is
    // the point of an interface, so the two texts are not expected to match.
    assert!(
        from_source.contains("never_used"),
        "the source build lost a private function it should have kept"
    );
    assert!(
        !from_interfaces.contains("never_used"),
        "the interface carried a private name nothing reaches"
    );
    for reachable in ["_area", "_scale", "_describe", "biggest"] {
        assert!(
            from_interfaces.contains(reachable),
            "the interface build lost '{reachable}'"
        );
    }

    // What has to match is the program, so run both and compare. That is the
    // claim worth checking anyway: an interface is sufficient if a program
    // built from it behaves identically, not if the emitted text is equal.
    let run = |from_interfaces: bool, name: &str| -> Option<String> {
        if !linker_available() {
            return None;
        }
        let exe =
            directory.join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
        let built = Command::new(frost)
            .env("FROST_CHECK_INTERFACES", "1")
            .env(
                "FROST_BUILD_FROM_INTERFACES",
                if from_interfaces { "1" } else { "0" },
            )
            .arg("--link")
            .arg("-o")
            .arg(&exe)
            .arg(&root)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "linking {name} failed:\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let ran = Command::new(&exe).output().unwrap();
        Some(String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n"))
    };
    let source_output = run(false, "source");
    let interface_output = run(true, "interfaces");
    let _ = std::fs::remove_dir_all(&directory);

    let (Some(source_output), Some(interface_output)) =
        (source_output, interface_output)
    else {
        return;
    };
    assert_eq!(source_output, "20\n24\n7\n");
    assert_eq!(
        source_output, interface_output,
        "building from interfaces changed what the program does"
    );
}

// Each module is its own compilation unit on the link path: one object file
// per module, cross-module calls resolved by the linker, and a specialization
// two modules both instantiate emitted privately into each of their objects
// rather than once into a shared one. That last part is what a single object
// cannot do, and getting it wrong shows up as either a duplicate symbol or an
// unresolved one, so this links a program with both shapes in it.
#[test]
fn each_module_becomes_its_own_object() {
    let directory = std::env::temp_dir().join("frost_per_module_objects");
    let library = directory.join("lib");
    std::fs::create_dir_all(&library).unwrap();
    std::fs::write(
        library.join("boxed.frost"),
        "export wrap\n\
         Boxed :: struct($T: Type) { value: T }\n\
         wrap :: fn(move v: $T) -> Boxed<T> { Boxed { value = v } }\n",
    )
    .unwrap();
    // Both modules instantiate wrap<i64>, so both objects must carry their own
    // private copy. The second also instantiates wrap<bool>, which only it has.
    std::fs::write(
        library.join("one.frost"),
        "export use_one\n\
         import \"boxed.frost\"\n\
         use_one :: fn() -> i64 { b := wrap(10)  b.value }\n",
    )
    .unwrap();
    std::fs::write(
        library.join("two.frost"),
        "export use_two\n\
         import \"boxed.frost\"\n\
         use_two :: fn() -> i64 { b := wrap(20)  c := wrap(true)  b.value }\n",
    )
    .unwrap();
    let root = directory.join("per_module_app.frost");
    std::fs::write(
        &root,
        "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         import \"lib/one.frost\"\n\
         import \"lib/two.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   printf(\"%lld\\n\", use_one() + use_two())\n\
         \x20   0\n\
         }\n",
    )
    .unwrap();

    if !linker_available() {
        let _ = std::fs::remove_dir_all(&directory);
        return;
    }
    let exe = directory
        .join(format!("per_module_app{}", std::env::consts::EXE_SUFFIX));
    let built = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("-o")
        .arg(&exe)
        .arg(&root)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "linking per-module objects failed:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let ran = Command::new(&exe).output().unwrap();
    let output = String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n");
    let _ = std::fs::remove_dir_all(&directory);
    assert_eq!(output, "30\n");
}

// Separate compilation gives each module its own copy of every specialization
// it instantiates, because cranelift has no weak or COMDAT linkage to fold
// duplicates with. Whether that duplication matters is a measurement, and this
// is the instrument: `wrap<i64>` is instantiated by both modules and `wrap<bool>`
// by one, so a single object emits two and separate objects would emit three.
#[test]
fn the_module_report_counts_what_separate_compilation_would_duplicate() {
    let directory = std::env::temp_dir().join("frost_module_report");
    let library = directory.join("lib");
    std::fs::create_dir_all(&library).unwrap();
    std::fs::write(
        library.join("box.frost"),
        "export Boxed, wrap\n\
         Boxed :: struct($T: Type) { value: T }\n\
         wrap :: fn(move v: $T) -> Boxed<T> { Boxed { value = v } }\n",
    )
    .unwrap();
    std::fs::write(
        library.join("one.frost"),
        "export use_one\n\
         import \"box.frost\"\n\
         use_one :: fn() -> i64 { b := wrap(1)  b.value }\n",
    )
    .unwrap();
    std::fs::write(
        library.join("two.frost"),
        "export use_two\n\
         import \"box.frost\"\n\
         use_two :: fn() -> i64 { b := wrap(2)  c := wrap(true)  b.value }\n",
    )
    .unwrap();
    let root = directory.join("app.frost");
    std::fs::write(
        &root,
        "import \"lib/one.frost\"\n\
         import \"lib/two.frost\"\n\
         main :: fn() -> i64 { use_one() + use_two() }\n",
    )
    .unwrap();

    let frost = env!("CARGO_BIN_EXE_frost");
    let output = Command::new(frost)
        .env("FROST_MODULE_REPORT", "1")
        .arg("--emit-c")
        .arg("-o")
        .arg(directory.join("out.c"))
        .arg(&root)
        .output()
        .unwrap();
    let report = String::from_utf8_lossy(&output.stderr).to_string();
    let _ = std::fs::remove_dir_all(&directory);

    assert!(output.status.success(), "compilation failed:\n{report}");
    assert!(
        report.contains(
            "2 specialization(s) emitted, 3 would be emitted per-module (1 instantiated by more than one module)"
        ),
        "unexpected module report:\n{report}"
    );
    assert!(
        report.contains("lib/one.frost instantiates 1")
            && report.contains("lib/two.frost instantiates 2"),
        "the report did not attribute specializations to modules:\n{report}"
    );
}

// An error inside an imported module names that module. Imports flatten every
// file into one statement list, so a bare "line 5" sent the reader to line 5 of
// whichever file they happened to be looking at. The mangled private name is
// undone too, since the reader never wrote `__m<tag>_Dot`.
#[test]
fn a_diagnostic_from_an_imported_module_names_the_file() {
    let directory = std::env::temp_dir().join("frost_import_diagnostic");
    let library = directory.join("lib");
    std::fs::create_dir_all(&library).unwrap();
    std::fs::write(
        library.join("broken.frost"),
        "export oops\n\
         Dot :: struct { x: i64 }\n\
         oops :: fn() -> i64 {\n\
         \x20   d := Dot { x = 1 }\n\
         \x20   d.missing_field\n\
         }\n",
    )
    .unwrap();
    let root = directory.join("app.frost");
    std::fs::write(
        &root,
        "import \"lib/broken.frost\"\nmain :: fn() -> i64 { oops() }\n",
    )
    .unwrap();

    let frost = env!("CARGO_BIN_EXE_frost");
    let output = Command::new(frost)
        .arg("--emit-c")
        .arg("-o")
        .arg(directory.join("out.c"))
        .arg(&root)
        .output()
        .unwrap();
    let message = String::from_utf8_lossy(&output.stderr).to_string();
    let _ = std::fs::remove_dir_all(&directory);

    assert!(!output.status.success(), "the broken module compiled");
    assert!(
        message.contains("lib/broken.frost:5:"),
        "the diagnostic did not name the imported file:\n{message}"
    );
    assert!(
        !message.contains("__m"),
        "the diagnostic leaked a mangled private name:\n{message}"
    );
    assert!(
        message.contains("'Dot'"),
        "the diagnostic did not name the struct the reader wrote:\n{message}"
    );
}

// A module's private symbols are a property of the module, not of the order it
// happened to be reached in. This is step 1 of docs/separate-compilation.md and
// the thing the rest of it cannot be built without: a module compiled once has
// to produce the symbols every other module will link against. The tag used to
// be a counter handed out in import traversal order, so the same file's private
// `secret` was `__m0_secret` reached first and `__m1_secret` reached second, and
// adding an unrelated import silently renamed everything after it.
#[test]
fn a_modules_private_symbols_do_not_depend_on_import_order() {
    let directory = std::env::temp_dir().join("frost_module_identity");
    let library = directory.join("lib");
    std::fs::create_dir_all(&library).unwrap();
    // Distinct private names, so a symbol can be traced back to the module that
    // kept it. Same-named privates would only show that two symbols exist, not
    // which module each belongs to.
    std::fs::write(
        library.join("shared.frost"),
        "export shared\n\
         secret_shared :: fn() -> i64 { 11 }\n\
         shared :: fn() -> i64 { secret_shared() }\n",
    )
    .unwrap();
    std::fs::write(
        library.join("other.frost"),
        "export other\n\
         secret_other :: fn() -> i64 { 22 }\n\
         other :: fn() -> i64 { secret_other() }\n",
    )
    .unwrap();

    // The same module, reached first in one program and second in the other.
    let alone = directory.join("alone.frost");
    std::fs::write(
        &alone,
        "import \"lib/shared.frost\"\nmain :: fn() -> i64 { shared() }\n",
    )
    .unwrap();
    let after = directory.join("after.frost");
    std::fs::write(
        &after,
        "import \"lib/other.frost\"\n\
         import \"lib/shared.frost\"\n\
         main :: fn() -> i64 { other() + shared() }\n",
    )
    .unwrap();

    // The tag a named private got, from the `__m<tag>_<name>` in the emitted C.
    let tag_of = |source_path: &std::path::Path, label: &str, private: &str| {
        let c_path = directory.join(format!("{label}.c"));
        let frost = env!("CARGO_BIN_EXE_frost");
        let emitted = Command::new(frost)
            .arg("--emit-c")
            .arg("-o")
            .arg(&c_path)
            .arg(source_path)
            .output()
            .unwrap();
        assert!(
            emitted.status.success(),
            "compiling {label} failed:\n{}",
            String::from_utf8_lossy(&emitted.stderr)
        );
        let c_source = std::fs::read_to_string(&c_path).unwrap();
        let suffix = format!("_{private}");
        c_source
            .match_indices("__m")
            .map(|(start, _)| {
                let rest = &c_source[start..];
                let end = rest
                    .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                    .unwrap_or(rest.len());
                rest[..end].to_string()
            })
            .find(|name| name.ends_with(&suffix))
            .map(|name| name[3..name.len() - suffix.len()].to_string())
            .unwrap_or_else(|| panic!("no mangled '{private}' in {label}"))
    };

    // Reached first in one program, second in the other.
    let alone_shared = tag_of(&alone, "alone", "secret_shared");
    let after_shared = tag_of(&after, "after", "secret_shared");
    let after_other = tag_of(&after, "after2", "secret_other");
    let _ = std::fs::remove_dir_all(&directory);

    assert_eq!(
        alone_shared, after_shared,
        "the same module got a different tag depending on when it was reached"
    );
    // And two different modules do not share a tag, which a constant would.
    assert_ne!(
        after_shared, after_other,
        "two different modules got the same tag"
    );
}

// A module offers what it exports and keeps the rest. Two files each keep a
// private `secret` and a private-or-exported `Thing`, and neither sees the
// other's, so the names do not collide and the root reaches only the exports.
#[test]
fn self_hosted_keeps_unexported_names_private() {
    let Some(compiler) = build_self_hosted_compiler("visibility") else {
        return;
    };
    let directory = std::env::temp_dir().join("frost_visibility_test");
    let library = directory.join("lib");
    std::fs::create_dir_all(&library).unwrap();
    std::fs::write(
        library.join("a.frost"),
        "export helper, Thing\n\
         Thing :: struct { value: i64 }\n\
         secret :: fn() -> i64 { 11 }\n\
         helper :: fn() -> i64 { secret() }\n",
    )
    .unwrap();
    std::fs::write(
        library.join("b.frost"),
        "export other\n\
         Thing :: struct { value: i64, extra: i64 }\n\
         secret :: fn() -> i64 { 22 }\n\
         other :: fn() -> i64 { secret() }\n",
    )
    .unwrap();
    let root = directory.join("app.frost");
    let program = "import \"lib/a.frost\"\n\
                   import \"lib/b.frost\"\n\
                   main :: fn() -> i64 {\n\
                   \x20   print helper()\n    print other()\n\
                   \x20   t := Thing { value = 5 }\n    print t.value\n    0\n}\n";
    std::fs::write(&root, program).unwrap();

    let emit = Command::new(&compiler)
        .env("FROST_INPUT", &root)
        .output()
        .unwrap();
    assert!(
        emit.status.success(),
        "exports were not honoured:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let c_source = String::from_utf8_lossy(&emit.stdout).replace("\r\n", "\n");
    let Some(output) = compile_c_and_run("visibility", &c_source) else {
        return;
    };
    assert_eq!(output, "11\n22\n5\n");

    // Naming what a module kept to itself is naming nothing.
    std::fs::write(
        &root,
        "import \"lib/a.frost\"\nmain :: fn() -> i64 { secret() }\n",
    )
    .unwrap();
    let refused = Command::new(&compiler)
        .env("FROST_INPUT", &root)
        .output()
        .unwrap();
    let message = String::from_utf8_lossy(&refused.stderr).to_string();

    let _ = std::fs::remove_dir_all(&directory);
    let _ = std::fs::remove_file(&compiler);
    assert!(
        !refused.status.success() && message.contains("undefined function"),
        "a private name was reachable from another file: {message}"
    );
}

#[test]
fn self_hosted_reports_an_unreadable_import() {
    let source = "import \"nowhere_at_all.frost\"\n\
                  main :: fn() -> i64 { 0 }\n";
    let Some(message) = self_hosted_rejects("importmissing", source) else {
        return;
    };
    assert!(
        message.contains("cannot read"),
        "expected an unreadable-import error, got:\n{message}"
    );
}

// Failure sets in the self-hosted compiler: `-> T ! E` answers with a value or
// a failure, `?` hands a failure on, and both sides come back out at the top.
const SELF_HOSTED_FAILURE_SETS: &str = "OpenError :: struct { code: i64 }\n\
     halve :: fn(n: i64) -> i64 ! OpenError {\n\
     \x20   if (n % 2 != 0) { return OpenError { code = 7 } }\n\
     \x20   n / 2\n}\n\
     twice :: fn(n: i64) -> i64 ! OpenError {\n\
     \x20   a := halve(n)?\n    b := halve(a)?\n    a + b\n}\n\
     main :: fn() -> i64 {\n\
     \x20   good := twice(8)\n    print good.tag\n    print good.value\n\
     \x20   bad := twice(6)\n    print bad.tag\n    print bad.error.code\n\
     \x20   0\n}\n";

#[test]
fn self_hosted_failure_sets_through_c() {
    let directory = std::env::temp_dir();
    let input = directory.join("frost_selffail_input.frost");
    std::fs::write(&input, SELF_HOSTED_FAILURE_SETS).unwrap();
    let Some(c_source) = self_hosted_emits("selffail", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(output) = compile_c_and_run("selffail", &c_source) else {
        return;
    };
    assert_eq!(output, "0\n6\n1\n7\n");
}

#[test]
fn self_hosted_failure_sets_natively() {
    let Some(output) =
        selfhosted_native_output("failsets", SELF_HOSTED_FAILURE_SETS)
    else {
        return;
    };
    assert_eq!(output, "0\n6\n1\n7\n");
}

// `?` only means something where there is a failure to hand on.
#[test]
fn self_hosted_rejects_a_try_outside_a_fallible_function() {
    let source = "E :: struct { c: i64 }\n\
                  f :: fn() -> i64 ! E { 1 }\n\
                  main :: fn() -> i64 { f()? }\n";
    let Some(message) = self_hosted_rejects("tryoutside", source) else {
        return;
    };
    assert!(
        message.contains("declares a failure set"),
        "expected a misplaced-'?' error, got:\n{message}"
    );
}

// Enums with payloads in the self-hosted compiler: variants with fields, a
// variant with none, construction, a match that binds a variant's fields, and
// the match standing for a value.
const SELF_HOSTED_ENUMS: &str = "Shape :: enum {\n\
     \x20   Circle { radius: i64 },\n\
     \x20   Rectangle { width: i64, height: i64 },\n\
     \x20   Point,\n}\n\
     area :: fn(s: Shape) -> i64 {\n\
     \x20   match s {\n\
     \x20       case .Circle { radius }: 3 * radius * radius\n\
     \x20       case .Rectangle { width, height }: width * height\n\
     \x20       case .Point: 0\n    }\n}\n\
     main :: fn() -> i64 {\n\
     \x20   c := Shape::Circle { radius = 5 }\n\
     \x20   r := Shape::Rectangle { width = 4, height = 6 }\n\
     \x20   pt := Shape::Point\n\
     \x20   print area(c)\n    print area(r)\n    print area(pt)\n\
     \x20   print sizeof(Shape)\n    0\n}\n";

#[test]
fn self_hosted_enums_through_c() {
    let directory = std::env::temp_dir();
    let input = directory.join("frost_selfenum_input.frost");
    std::fs::write(&input, SELF_HOSTED_ENUMS).unwrap();
    let Some(c_source) = self_hosted_emits("selfenum", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(output) = compile_c_and_run("selfenum", &c_source) else {
        return;
    };
    assert_eq!(output, "75\n24\n0\n32\n");
}

#[test]
fn self_hosted_enums_natively() {
    let Some(output) = selfhosted_native_output("enums", SELF_HOSTED_ENUMS)
    else {
        return;
    };
    assert_eq!(output, "75\n24\n0\n32\n");
}

#[test]
fn self_hosted_rejects_an_unknown_variant() {
    let source = "Kind :: enum { Player, Enemy { damage: i64 } }\n\
                  main :: fn() -> i64 {\n\
                  \x20   k := Kind::Wizard\n    0\n}\n";
    let Some(message) = self_hosted_rejects("badvariant", source) else {
        return;
    };
    assert!(
        message.contains("has no variant"),
        "expected an unknown-variant error, got:\n{message}"
    );
}

// A byte pointer strides one byte at a time, and a byte-wide type is one byte
// wide. The native backend used to read eight bytes for each.
#[test]
fn self_hosted_native_indexes_bytes() {
    let source = "main :: fn() -> i64 {\n\
                  \x20   s : ^i8 = \"hello\"\n\
                  \x20   print s[0]\n    print s[1]\n    print s[4]\n\
                  \x20   print sizeof(i8)\n    0\n}\n";
    let Some(output) = selfhosted_native_output("bytes", source) else {
        return;
    };
    assert_eq!(output, "104\n101\n111\n1\n");
}

// A `?` in a loop's condition is asked again every time round. It used to be
// lifted out of the loop and evaluated once, which read the same answer for
// ever.
#[test]
fn self_hosted_reevaluates_a_try_in_a_loop_condition() {
    let source = "E :: struct { c: i64 }\n\
                  step :: fn(n: i64) -> i64 ! E {\n\
                  \x20   if (n > 3) { return E { c = 9 } }\n\
                  \x20   n + 1\n}\n\
                  run :: fn() -> i64 ! E {\n\
                  \x20   mut n : i64 = 0\n\
                  \x20   while (step(n)? < 3) { n = n + 1 }\n\
                  \x20   n\n}\n\
                  main :: fn() -> i64 {\n\
                  \x20   r := run()\n    print r.value\n    0\n}\n";
    let Some(output) = selfhosted_native_output("trywhile", source) else {
        return;
    };
    assert_eq!(output, "2\n");
}

// A generic function used with a type that no generic struct was written with.
// Instantiation used to be driven by the struct instances alone, so this was
// called and never emitted, and the program failed to link.
#[test]
fn self_hosted_emits_a_generic_function_with_no_struct_instance() {
    let source = "Box :: struct($T: Type) { value: $T }\n\
                  wrap :: fn($T: Type, v: $T) -> Box<T> { Box { value = v } }\n\
                  unwrap :: fn(b: Box<$T>) -> $T { b^.value }\n\
                  main :: fn() -> i64 {\n\
                  \x20   b := wrap($i64, 41)\n\
                  \x20   print unwrap(b) + 1\n    0\n}\n";
    let Some(output) = selfhosted_native_output("genericonly", source) else {
        return;
    };
    assert_eq!(output, "42\n");
}

// Everything the two self-hosted backends can express, run through both. They
// answer the same thing or one of them is wrong.
#[test]
fn self_hosted_backends_agree() {
    let source = "malloc :: extern fn(size: i64) -> ^i8\n\
         Inner :: struct { a: i64, b: i64 }\n\
         Outer :: struct { first: i64, mid: Inner, last: i64 }\n\
         Bytes :: struct { flag: i8, count: i64, mark: i8 }\n\
         Kind :: enum { None, One { x: i64 }, Two { x: i64, y: i64 } }\n\
         Box :: struct($T: Type) { value: $T }\n\
         wrap :: fn($T: Type, v: $T) -> Box<T> { Box { value = v } }\n\
         unwrap :: fn(b: Box<$T>) -> $T { b^.value }\n\
         sum_kind :: fn(k: Kind) -> i64 {\n\
         \x20   match k {\n        case .None: 0\n\
         \x20       case .One { x }: x\n        case .Two { x, y }: x + y\n    }\n}\n\
         bump :: fn(mut o: Outer) -> i64 {\n\
         \x20   o.mid.b = o.mid.b + 1\n    o.mid.b\n}\n\
         main :: fn() -> i64 {\n\
         \x20   mut o : Outer = Outer { first = 1, mid = Inner { a = 2, b = 3 }, last = 4 }\n\
         \x20   print bump(o)\n    print o.mid.b\n    print sizeof(Outer)\n\
         \x20   mut bs : Bytes = Bytes { flag = 1, count = 77, mark = 2 }\n\
         \x20   print bs.flag\n    print bs.count\n    bs.mark = 5\n\
         \x20   print bs.mark\n    print sizeof(Bytes)\n\
         \x20   print sum_kind(Kind::None)\n\
         \x20   print sum_kind(Kind::One { x = 6 })\n\
         \x20   print sum_kind(Kind::Two { x = 6, y = 7 })\n\
         \x20   b := wrap($i64, 41)\n    print unwrap(b) + 1\n\
         \x20   buf := malloc(8)\n    buf[0] = 65\n    buf[1] = 66\n\
         \x20   print buf[0]\n    print buf[1]\n\
         \x20   mut acc : i64 = 0\n    mut i : i64 = 0\n\
         \x20   while (i < 5) {\n\
         \x20       if (i % 2 == 0) { acc = acc + i } else { acc = acc - i }\n\
         \x20       i = i + 1\n    }\n\
         \x20   print acc\n    print (3 < 4) && (5 >= 5)\n\
         \x20   print 17 / 5\n    print 0 - 17 % 5\n    0\n}\n";
    let expected =
        "4\n4\n32\n1\n77\n5\n24\n0\n6\n13\n42\n65\n66\n2\n1\n3\n-2\n";

    let Some(native) = selfhosted_native_output("agree", source) else {
        return;
    };
    assert_eq!(native, expected, "the native backend disagrees");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_agree_input.frost");
    std::fs::write(&input, source).unwrap();
    let Some(c_source) = self_hosted_emits("agree", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("agree", &c_source) else {
        return;
    };
    assert_eq!(via_c, expected, "the C backend disagrees");
}

#[test]
fn self_hosted_rejects_a_linear_value_never_consumed() {
    let source = "File :: linear struct { h: i64 }\n\
                  close :: extern fn(move f: File)\n\
                  main :: fn() -> i64 {\n\
                  \x20   r := File { h = 1 }\n    return 0\n}\n";
    let Some(message) = self_hosted_rejects("linearleak", source) else {
        return;
    };
    assert!(
        message.contains("never consumed"),
        "expected a linear-not-consumed error, got:\n{message}"
    );
}

#[test]
fn self_hosted_rejects_consuming_a_linear_value_twice() {
    let source = "File :: linear struct { h: i64 }\n\
                  close :: extern fn(move f: File)\n\
                  main :: fn() -> i64 {\n\
                  \x20   r := File { h = 1 }\n\
                  \x20   close(r)\n    close(r)\n    return 0\n}\n";
    let Some(message) = self_hosted_rejects("lineartwice", source) else {
        return;
    };
    assert!(
        message.contains("moved value"),
        "expected a double-consume error, got:\n{message}"
    );
}

#[test]
fn self_hosted_rejects_a_call_with_the_wrong_argument_count() {
    let source = "add :: fn(a: i64, b: i64) -> i64 { a + b }\n\
                  main :: fn() -> i64 {\n    return add(1)\n}\n";
    let Some(message) = self_hosted_rejects("arity", source) else {
        return;
    };
    assert!(
        message.contains("expects 2"),
        "expected an argument-count error, got:\n{message}"
    );
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
    assert!(output.contains("2 passed, 0 failed"), "got:\n{output}");
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
    x = x + 1
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
    t := a
    a = b
    b = t
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

Entity :: struct { hp: i64, mana: i64 }

Slab :: struct($T: Type, $N: usize) {
    storage: [N]T,
    free_list: [N]i64,
    generations: [N]i64,
    free_count: i64,
}

size_of :: fn($T: Type) -> i64 { sizeof(T) }

insert :: fn($T: Type, $N: usize, mut s: Slab<T, N>, move value: $T) -> Handle<T> {
    index := s.free_count
    s.free_count = s.free_count + 1
    s.storage[index] = value
    packed := (s.generations[index] << 32) | index
    packed
}

main :: fn() -> i64 {
    printf("%lld\n", size_of($i64))
    printf("%lld\n", size_of($Entity))

    mut world : Slab<Entity, 16> = Slab {
        storage = [Entity { hp = 0, mana = 0 }; 16],
        free_list = [0; 16],
        generations = [0; 16],
        free_count = 0,
    }
    h := insert($Entity, $16, world, Entity { hp = 100, mana = 30 })
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

Pair :: struct($T: Type) { first: T, second: T }

Slab :: struct($T: Type, $N: usize) {
    storage: [N]T,
    generations: [N]i64,
    free_count: i64,
}

zero_pair :: fn() -> Pair<i64> { Pair { first = 0, second = 0 } }

insert :: fn($T: Type, $N: usize, mut s: Slab<T, N>, move value: $T) -> Handle<T> {
    index := s.free_count
    s.free_count = s.free_count + 1
    s.storage[index] = value
    packed := (s.generations[index] << 32) | index
    packed
}

main :: fn() -> i64 {
    inferred := Pair { first = 30, second = 12 }
    printf("%lld\n", inferred.first + inferred.second)

    mut pool : Slab<Pair<i64>, 4> = Slab {
        storage = [zero_pair(); 4],
        generations = [0; 4],
        free_count = 0,
    }
    h := insert($Pair<i64>, $4, pool, Pair { first = 3, second = 4 })
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
    t := a
    a = b
    b = t
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

Entity :: struct { hp: i64, mana: i64 }

Slab :: struct($T: Type, $N: usize) {
    storage: [N]T,
    generations: [N]i64,
    free_list: [N]i64,
    free_count: i64,
}

reset :: fn($T: Type, $N: usize, mut s: Slab<T, N>) {
    mut i : i64 = 0
    while (i < N) { s.generations[i] = 0  s.free_list[i] = N - 1 - i  i = i + 1 }
    s.free_count = N
}

insert :: fn($T: Type, $N: usize, mut s: Slab<T, N>, move value: $T) -> Handle<T> {
    s.free_count = s.free_count - 1
    index := s.free_list[s.free_count]
    s.storage[index] = value
    packed := (s.generations[index] << 32) | index
    packed
}

alive :: fn($T: Type, $N: usize, s: Slab<T, N>, handle: Handle<T>) -> i64 {
    raw : i64 = handle
    if (s.generations[raw & 4294967295] == (raw >> 32)) { 1 } else { 0 }
}

release :: fn($T: Type, $N: usize, mut s: Slab<T, N>, handle: Handle<T>) -> i64 {
    raw : i64 = handle
    index := raw & 4294967295
    if (s.generations[index] != (raw >> 32)) { return 0 }
    s.generations[index] = s.generations[index] + 1
    s.free_list[s.free_count] = index
    s.free_count = s.free_count + 1
    1
}

heal :: fn(mut e: Entity, amount: i64) {
    e.hp = e.hp + amount
}
total :: fn(e: Entity) -> i64 {
    e.hp + e.mana
}

main :: fn() -> i64 {
    mut world : Slab<Entity, 8> = Slab {
        storage = [Entity { hp = 0, mana = 0 }; 8],
        generations = [0; 8],
        free_list = [0; 8],
        free_count = 0,
    }
    reset($Entity, $8, world)

    ha := insert($Entity, $8, world, Entity { hp = 50, mana = 10 })
    hb := insert($Entity, $8, world, Entity { hp = 20, mana = 5 })

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
    field = field + 100
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

Entity :: struct { hp: i64, mana: i64 }

Slab :: struct($T: Type, $N: usize) {
    storage: [N]T,
    generations: [N]i64,
    free_list: [N]i64,
    free_count: i64,
}

reset :: fn($T: Type, $N: usize, mut s: Slab<T, N>) {
    mut i : i64 = 0
    while (i < N) { s.generations[i] = 0  s.free_list[i] = N - 1 - i  i = i + 1 }
    s.free_count = N
}

insert :: fn($T: Type, $N: usize, mut s: Slab<T, N>, move value: $T) -> Handle<T> {
    s.free_count = s.free_count - 1
    index := s.free_list[s.free_count]
    s.storage[index] = value
    packed := (s.generations[index] << 32) | index
    packed
}

alive :: fn($T: Type, $N: usize, s: Slab<T, N>, handle: Handle<T>) -> i64 {
    raw : i64 = handle
    if (s.generations[raw & 4294967295] == (raw >> 32)) { 1 } else { 0 }
}

release :: fn($T: Type, $N: usize, mut s: Slab<T, N>, handle: Handle<T>) -> i64 {
    raw : i64 = handle
    index := raw & 4294967295
    if (s.generations[index] != (raw >> 32)) { return 0 }
    s.generations[index] = s.generations[index] + 1
    s.free_list[s.free_count] = index
    s.free_count = s.free_count + 1
    1
}

index_of :: fn(handle: Handle<Entity>) -> i64 { raw : i64 = handle  raw & 4294967295 }
generation_of :: fn(handle: Handle<Entity>) -> i64 { raw : i64 = handle  raw >> 32 }

main :: fn() -> i64 {
    mut p : Slab<Entity, 8> = Slab {
        storage = [Entity { hp = 0, mana = 0 }; 8],
        generations = [0; 8],
        free_list = [0; 8],
        free_count = 0,
    }
    reset($Entity, $8, p)

    ha := insert($Entity, $8, p, Entity { hp = 100, mana = 30 })
    hb := insert($Entity, $8, p, Entity { hp = 50, mana = 10 })

    printf("%lld\n", index_of(ha))
    printf("%lld\n", index_of(hb))
    printf("%lld\n", generation_of(ha))

    printf("%lld\n", p[ha].hp)
    p[ha].hp = 999
    printf("%lld\n", p[ha].hp)

    printf("%lld\n", alive($Entity, $8, p, ha))
    printf("%lld\n", release($Entity, $8, p, ha))
    printf("%lld\n", alive($Entity, $8, p, ha))

    hc := insert($Entity, $8, p, Entity { hp = 7, mana = 7 })
    printf("%lld\n", index_of(hc))
    printf("%lld\n", generation_of(hc))
    printf("%lld\n", alive($Entity, $8, p, ha))
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

// Build and run an example where it lives rather than copying its text to a
// temp directory, because an example may import a sibling and an import is
// resolved relative to the file that wrote it.
fn run_example(
    name: &str,
    source: &std::path::Path,
    emit_c: bool,
) -> Option<String> {
    if !linker_available() {
        return None;
    }
    let exe_path = std::env::temp_dir().join(format!(
        "frost_example_{name}{}",
        std::env::consts::EXE_SUFFIX
    ));
    let mut command = Command::new(env!("CARGO_BIN_EXE_frost"));
    if emit_c {
        command.arg("--emit-c");
    }
    command
        .env("FROST_CHECK_INTERFACES", "1")
        .arg("--link")
        .arg("-o")
        .arg(&exe_path)
        .arg(source);
    let compile = command.output().unwrap();
    assert!(
        compile.status.success(),
        "compilation failed for {name} (emit_c={emit_c}):\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(&exe_path).output().unwrap();
    assert!(run.status.success(), "example {name} exited with failure");
    let output = normalize_newlines(&run.stdout);
    let _ = std::fs::remove_file(&exe_path);
    Some(output)
}

// The self-hosted compiler is the largest program in the repository, so putting
// it through both backends is the widest single differential check there is. It
// used to ride in the list above as a source string; it is a set of modules
// now, so it is compiled where it sits.
#[test]
fn cranelift_and_c_backends_agree_on_the_self_hosted_compiler() {
    let source = self_hosted_source();
    let Some(native) = run_example("diff_self_hosted", &source, false) else {
        return;
    };
    let via_c = run_example("diff_self_hosted_c", &source, true);
    assert_eq!(
        Some(native),
        via_c,
        "Cranelift and C backends disagree on the self-hosted compiler"
    );
}

fn normalize_newlines(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .split("\r\n")
        .collect::<Vec<_>>()
        .join("\n")
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
        let native = run_example(&format!("ex_{stem}"), &path, false);
        let via_c = run_example(&format!("ex_{stem}_c"), &path, true);
        assert_eq!(native, via_c, "backends disagree on example {stem}");
        checked += 1;
    }
    assert!(checked > 0, "no native examples were found");
}

#[test]
fn cranelift_and_c_backends_agree() {
    let programs = [
        ("diff_arith", ARITHMETIC),
        ("diff_refbind", REF_BINDING),
        ("diff_floats", FLOATS),
        ("diff_widths", WIDTHS),
        ("diff_wrapping", WRAPPING_AND_UNARY),
        ("diff_anon", ANON_FUNCTIONS),
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

// A `mut` parameter of a scalar type is a reference the body never asked for,
// so the body reads through it. This used to fail with an internal IR error.
#[test]
fn mut_parameter_on_a_scalar_writes_through() {
    let source = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

bump :: fn(mut n: i64) { n = n + 1 }

twice :: fn(mut n: i64) {
    bump(n)
    bump(n)
}

main :: fn() -> i64 {
    mut x : i64 = 5
    bump(x)
    printf("%lld\n", x)
    twice(x)
    printf("%lld\n", x)
    0
}
"#;
    let Some(output) = compile_and_run("mutscalar", source) else {
        return;
    };
    assert_eq!(output, "6\n8\n");
}

// A function's locals die when it returns, so a pointer or a slice into one of
// them may not be the thing it answers with.
#[test]
fn a_pointer_into_the_frame_may_not_be_returned() {
    let source = "leak :: fn() -> ^i64 {\n\
                  \x20   mut local : i64 = 42\n\
                  \x20   ptr_to(local)\n}\n\
                  main :: fn() -> i64 { 0 }\n";
    let message = compile_error("frameptr", source);
    assert!(
        message.contains("pointer into the frame of"),
        "expected a frame escape error, got:\n{message}"
    );
}

#[test]
fn a_slice_over_a_local_may_not_be_returned() {
    let source = "leak :: fn() -> []i64 {\n\
                  \x20   arr := [11, 22, 33]\n\
                  \x20   view : []i64 = arr\n\
                  \x20   view\n}\n\
                  main :: fn() -> i64 { 0 }\n";
    let message = compile_error("frameslice", source);
    assert!(
        message.contains("pointer into the frame of"),
        "expected a frame escape error, got:\n{message}"
    );
}

// A pointer a function was handed is not its frame's, so passing it back out is
// fine and must not be caught by the frame check.
#[test]
fn a_pointer_handed_in_may_be_returned() {
    let source = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

pass_through :: fn(p: ^i64) -> ^i64 {
    held := p
    held
}

main :: fn() -> i64 {
    mut n : i64 = 7
    q := pass_through(ptr_to(n))
    printf("%lld\n", q^)
    0
}
"#;
    let Some(output) = compile_and_run("framepass", source) else {
        return;
    };
    assert_eq!(output, "7\n");
}

// A compile-time function argument. `$f` names a function at the call, the
// generic specializes once per function it is given, and the body calls it
// directly. This is what closes the inner-loop gap left by having no traits,
// no closures and no operator overloading: the comparator is in the loop
// rather than reached through a pointer.
#[test]
fn a_function_may_be_a_compile_time_argument() {
    let source = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

ascending :: fn(a: i64, b: i64) -> bool { a < b }
descending :: fn(a: i64, b: i64) -> bool { a > b }

best3 :: fn($T: Type, $before: Type, move x: $T, move y: $T, move z: $T) -> $T {
    mut result := x
    if (before(y, result)) { result = y }
    if (before(z, result)) { result = z }
    result
}

main :: fn() -> i64 {
    printf("%lld\n", best3($i64, $ascending, 7, 3, 9))
    printf("%lld\n", best3($i64, $descending, 7, 3, 9))
    0
}
"#;
    let Some(output) = compile_and_run("constfn", source) else {
        return;
    };
    assert_eq!(output, "3\n9\n");
}

// One specialization per function given, and the call inside it is direct.
#[test]
fn a_compile_time_function_argument_specializes_and_calls_directly() {
    let source = "cmp :: fn(a: i64, b: i64) -> bool { a < b }\n\
                  pick :: fn($T: Type, $f: Type, move a: $T, move b: $T) -> $T {\n\
                  \x20   mut best := a\n    if (f(b, best)) { best = b }\n    best\n}\n\
                  main :: fn() -> i64 { pick($i64, $cmp, 2, 1) }\n";
    let Some(c_source) = emit_c_source("constfndirect", source) else {
        return;
    };
    assert!(
        c_source.contains("pick__i64__cmp"),
        "expected a specialization named for the function:\n{c_source}"
    );
    assert!(
        c_source.contains("= frost_cmp("),
        "expected a direct call to the comparator:\n{c_source}"
    );
}

// Naming a `mut` parameter means the caller's value, whatever the type and
// whether or not the type came from a type parameter. This used to hold only
// for concrete scalars: a `mut x: $T` bound to a scalar, and a `mut x: Struct`,
// both assigned to the body's own reference instead of through it, so the
// caller saw nothing and no error was reported.
#[test]
fn a_mut_parameter_writes_back_through_every_shape() {
    let source = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }

swap_scalar :: fn(mut a: i64, mut b: i64) { t := a  a = b  b = t }
swap_generic :: fn(mut a: $T, mut b: $T) { t := a  a = b  b = t }
replace :: fn(mut p: Point, move q: Point) { p = q }

main :: fn() -> i64 {
    mut x : i64 = 1
    mut y : i64 = 2
    swap_scalar(x, y)
    printf("%lld\n", x)

    mut m : i64 = 3
    mut n : i64 = 4
    swap_generic(m, n)
    printf("%lld\n", m)

    mut a := Point { x = 1, y = 2 }
    mut b := Point { x = 9, y = 8 }
    swap_generic(a, b)
    printf("%lld\n", a.x)

    mut c := Point { x = 5, y = 6 }
    replace(c, Point { x = 7, y = 0 })
    printf("%lld\n", c.x)
    0
}
"#;
    let Some(output) = compile_and_run("mutwriteback", source) else {
        return;
    };
    assert_eq!(output, "2\n4\n9\n7\n");
}

// A read-mode `$T` parameter bound to a copy type is passed by value, exactly
// as a concrete copy-typed parameter is. It used to stay a reference, which
// failed the moment the body stored it anywhere.
#[test]
fn a_read_mode_type_parameter_bound_to_a_scalar_is_a_value() {
    let source = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Pair :: struct($T: Type) { first: T, second: T }
make_pair :: fn(a: $T, b: $T) -> Pair<T> { Pair { first = a, second = b } }

main :: fn() -> i64 {
    p := make_pair(3, 4)
    printf("%lld\n", p.first + p.second)
    m : i64 = 10
    n : i64 = 11
    q := make_pair(m, n)
    printf("%lld\n", q.first + q.second)
    0
}
"#;
    let Some(output) = compile_and_run("readmodevalue", source) else {
        return;
    };
    assert_eq!(output, "7\n21\n");
}

// A compile-time function parameter may say what signature it needs, and then
// the mismatch is reported against the parameter list rather than against a
// line inside the specialized body that the reader never wrote.
#[test]
fn a_compile_time_function_argument_may_declare_its_signature() {
    let source = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

ascending :: fn(a: i64, b: i64) -> bool { a < b }

best :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T, move y: $T) -> $T {
    mut result := x
    if (before(y, result)) { result = y }
    result
}

main :: fn() -> i64 {
    printf("%lld\n", best($i64, $ascending, 7, 3))
    0
}
"#;
    let Some(output) = compile_and_run("constfnbound", source) else {
        return;
    };
    assert_eq!(output, "3\n");
}

#[test]
fn a_compile_time_function_argument_is_checked_against_its_signature() {
    let source = "\
wrong :: fn(a: i64) -> i64 { a }
best :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T) -> $T { x }
main :: fn() -> i64 { best($i64, $wrong, 1) }
";
    let message = compile_error("constfnbadsig", source);
    assert!(
        message.contains("'wrong'")
            && message.contains("proc(i64, i64) -> bool"),
        "expected the signature mismatch to name both signatures:\n{message}"
    );
}

#[test]
fn a_type_given_where_a_function_is_declared_is_rejected() {
    let source = "\
Point :: struct { x: i64 }
best :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T) -> $T { x }
main :: fn() -> i64 { best($i64, $Point, 1) }
";
    let message = compile_error("constfnnotafn", source);
    assert!(
        message.contains("needs a function as its argument"),
        "expected a function to be required:\n{message}"
    );
}

// The C the compiler emits for a program, for tests that need to look at the
// shape of the lowering rather than only at what it prints.
fn emit_c_source(name: &str, source: &str) -> Option<String> {
    let directory = std::env::temp_dir();
    let source_path = directory.join(format!("frost_cemit_{name}.frost"));
    let c_path = directory.join(format!("frost_cemit_{name}.c"));
    std::fs::write(&source_path, source).unwrap();
    let frost = env!("CARGO_BIN_EXE_frost");
    let emitted = Command::new(frost)
        .arg("--emit-c")
        .arg("-o")
        .arg(&c_path)
        .arg(&source_path)
        .output()
        .unwrap();
    assert!(
        emitted.status.success(),
        "C emission failed for {name}:\n{}",
        String::from_utf8_lossy(&emitted.stderr)
    );
    let text = std::fs::read_to_string(&c_path).ok();
    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_file(&c_path);
    text
}

// The other ways a pointer into the frame could leave it: written into a
// parameter, answered from a branch, or carried out inside a struct.
#[test]
fn a_frame_pointer_may_not_leave_by_any_road() {
    let cases: &[(&str, &str)] = &[
        (
            "param",
            "stash :: fn(mut slot: ^i64) {\n\
             \x20   mut local : i64 = 5\n    slot = ptr_to(local)\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "branch",
            "pick :: fn(c: bool) -> ^i64 {\n\
             \x20   mut a : i64 = 1\n\
             \x20   if (c) { ptr_to(a) } else { ptr_to(a) }\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "struct",
            "Holder :: struct { p: ^i64 }\n\
             wrap :: fn() -> Holder {\n\
             \x20   mut a : i64 = 1\n    Holder { p = ptr_to(a) }\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
    ];
    for (name, source) in cases {
        let message = compile_error(&format!("frame_{name}"), source);
        assert!(
            message.contains("pointer into the frame of"),
            "{name}: expected a frame escape error, got:\n{message}"
        );
    }
}

// Step 5 of docs/separate-compilation.md. A module is rebuilt only when its own
// source or an imported interface changes, and the distinction that decides it
// is that a generic's body is part of its interface while an ordinary body is
// not. This builds a three module chain and edits the leaf twice, once in each
// kind of body, which is the only way to tell a cache that works from one that
// rebuilds everything or nothing.
#[test]
fn only_the_modules_an_edit_reaches_are_rebuilt() {
    if !linker_available() {
        return;
    }
    let directory = std::env::temp_dir().join("frost_incremental");
    let library = directory.join("lib");
    let build = directory.join("build");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&library).unwrap();

    let leaf = library.join("leaf.frost");
    let write_leaf = |bump: &str, twice: &str| {
        std::fs::write(
            &leaf,
            format!(
                "export bump, twice, boxed, Boxed\n\
                 Boxed :: struct {{ value: i64 }}\n\
                 secret :: fn(x: i64) -> i64 {{ x + 100 }}\n\
                 bump :: fn(x: i64) -> i64 {{ {bump} }}\n\
                 boxed :: fn(x: i64) -> Boxed {{ Boxed {{ value = secret(x) }} }}\n\
                 twice :: fn($T: Type, move v: $T) -> $T {{ {twice} }}\n"
            ),
        )
        .unwrap();
    };
    write_leaf("x + 1", "v + v");
    std::fs::write(
        library.join("mid.frost"),
        "export combine\n\
         import \"leaf.frost\"\n\
         combine :: fn(x: i64) -> i64 { bump(twice($i64, x)) }\n",
    )
    .unwrap();
    let root = directory.join("incremental_app.frost");
    std::fs::write(
        &root,
        "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         import \"lib/mid.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   b := boxed(1)\n\
         \x20   printf(\"%lld\n\", combine(5) + b.value + twice($i64, 2))\n\
         \x20   0\n\
         }\n",
    )
    .unwrap();

    let exe = directory.join(format!("app{}", std::env::consts::EXE_SUFFIX));
    let build_once = || -> (Vec<String>, String) {
        let built = Command::new(env!("CARGO_BIN_EXE_frost"))
            .env("FROST_CHECK_INTERFACES", "1")
            .arg("--link")
            .arg("--incremental")
            .arg("--build-dir")
            .arg(&build)
            .arg("-o")
            .arg(&exe)
            .arg(&root)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "an incremental build failed:\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let reused: Vec<String> = String::from_utf8_lossy(&built.stdout)
            .lines()
            .filter_map(|line| line.strip_prefix("Reused ").map(str::to_string))
            .collect();
        let ran = Command::new(&exe).output().unwrap();
        (
            reused,
            String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n"),
        )
    };

    // Nothing is cached yet, so nothing is reused and the program still runs.
    let (reused, output) = build_once();
    assert!(reused.is_empty(), "a first build reused {reused:?}");
    assert_eq!(
        output,
        "116
"
    );

    // Nothing changed, so neither imported module is read or built again.
    let (reused, output) = build_once();
    assert_eq!(reused, vec!["lib/leaf.frost", "lib/mid.frost"]);
    assert_eq!(
        output,
        "116
"
    );

    // An ordinary body is the module's own business. The leaf is rebuilt and
    // the module that calls it is not, because the call is resolved by the
    // linker and nothing about it changed.
    write_leaf("x + 3", "v + v");
    let (reused, output) = build_once();
    assert_eq!(reused, vec!["lib/mid.frost"]);
    assert_eq!(
        output,
        "118
"
    );

    // A generic body is its callers' business too, since the caller is what
    // stamps out the template, so this reaches the module above.
    write_leaf("x + 3", "v + v + v");
    let (reused, output) = build_once();
    assert!(
        reused.is_empty(),
        "a generic body changed and {reused:?} was reused anyway"
    );
    assert_eq!(
        output,
        "125
"
    );

    // And back to a steady state, which is what proves the previous build
    // wrote its records rather than merely rebuilding.
    let (reused, output) = build_once();
    assert_eq!(reused, vec!["lib/leaf.frost", "lib/mid.frost"]);
    assert_eq!(
        output,
        "125
"
    );

    // A record answers for a module only while the object it describes is still
    // there, so throwing the objects away has to mean a rebuild rather than a
    // link against nothing.
    for entry in std::fs::read_dir(&build).unwrap().flatten() {
        if entry.path().extension().is_some_and(|ext| ext == "o") {
            std::fs::remove_file(entry.path()).unwrap();
        }
    }
    let (reused, output) = build_once();
    assert!(
        reused.is_empty(),
        "the objects were gone and {reused:?} was reused anyway"
    );
    assert_eq!(
        output,
        "125
"
    );

    let _ = std::fs::remove_dir_all(&directory);
}

// docs/callbacks.md, step 1. A callback registration is an `extern fn` with a
// `$handler` parameter bound to a function signature, and the whole ownership
// argument is that the context moves in. What this checks is that the argument
// costs no new machinery: `check_ownership` already stops a caller touching a
// moved value, so a program that registers a context and then reads it is
// rejected by the pass that was already there.
#[test]
fn a_registered_context_may_not_still_be_read() {
    let message = compile_error(
        "callback_moved",
        "Ctx :: struct { hits: i64 }\n\
         on_event :: fn(mut ctx: Ctx, code: i64) { ctx.hits = ctx.hits + code }\n\
         register :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64\n\
         main :: fn() -> i64 {\n\
         \x20   c := Ctx { hits = 0 }\n\
         \x20   t := register($on_event, c)\n\
         \x20   c.hits\n\
         }\n",
    );
    assert!(
        message.contains("use of moved value"),
        "expected the context to be moved into the registration, got:\n{message}"
    );
}

// The declaration checks reach the driver, not just the unit tests, and they
// name the thing the reader wrote rather than something downstream of it.
#[test]
fn a_registration_declaration_is_checked() {
    let message = compile_error(
        "callback_borrowed_context",
        "Ctx :: struct { hits: i64 }\n\
         on_event :: fn(mut ctx: Ctx, code: i64) { ctx.hits = ctx.hits + code }\n\
         register :: extern fn($handler: fn(mut Ctx, i64), ctx: Ctx) -> i64\n\
         main :: fn() -> i64 { 0 }\n",
    );
    assert!(
        message.contains("'move'"),
        "expected the context to have to be taken by move, got:\n{message}"
    );
}

// docs/callbacks.md, step 2, and the whole safety argument. A registration
// holds its context for as long as it is registered, so the value it answers
// with names storage in the frame that holds the context. A context in that
// frame is the ordinary case and is safe, because `check_linearity` forces the
// registration to be consumed in the function that made it; what has to be
// stopped is the registration leaving that function by another road.
#[test]
fn a_registration_may_not_outlive_its_context() {
    let message = compile_error(
        "callback_escape",
        "Ctx :: struct { hits: i64 }\n\
         Registration :: linear struct { token: i64 }\n\
         on_event :: fn(mut ctx: Ctx, code: i64) { ctx.hits = ctx.hits + code }\n\
         register :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> Registration\n\
         leak :: fn() -> Registration {\n\
         \x20   c := Ctx { hits = 0 }\n\
         \x20   register($on_event, c)\n\
         }\n\
         main :: fn() -> i64 { 0 }\n",
    );
    assert!(
        message.contains("pointer into the frame of 'leak'"),
        "expected the registration to be held to its context's frame, got:\n{message}"
    );
}

// The other half of that check, which matters more: registering a context in
// this frame and unregistering it here is the shape the design is for, and it
// has to get past every check the language has. Only lowering is missing, which
// is steps 3 and 4, so this fails there and nowhere earlier. When those land
// this stops being a compile error and becomes a program to run, which is the
// point at which it should be rewritten rather than deleted.
#[test]
fn registering_and_unregistering_in_one_frame_is_allowed() {
    let message = compile_error(
        "callback_roundtrip",
        "Ctx :: struct { hits: i64 }\n\
         Registration :: linear struct { token: i64 }\n\
         on_event :: fn(mut ctx: Ctx, code: i64) { ctx.hits = ctx.hits + code }\n\
         register :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> Registration\n\
         unregister :: fn(move r: Registration) -> i64 { r.token }\n\
         main :: fn() -> i64 {\n\
         \x20   c := Ctx { hits = 0 }\n\
         \x20   r := register($on_event, c)\n\
         \x20   unregister(r)\n\
         }\n",
    );
    for premature in ["Region error", "Ownership error", "Linearity error"] {
        assert!(
            !message.contains(premature),
            "the safe shape was rejected by {premature}:\n{message}"
        );
    }
}

// docs/callbacks.md, step 5: bind a real C callback API and register against
// it, because every earlier step is checkable without a library and none of
// them proves the ABI. The library here is the smallest one that is still the
// real shape, `(callback, userdata)` stored and called back later, compiled by
// the C compiler and linked in.
//
// What this settles is the finding that made step 3 disappear. A `mut`
// parameter is already a pointer in the signature and Frost and C share a
// calling convention, so the handler compiled for Frost *is* the
// `void (*)(void*, int64_t)` the library wants, and there is no trampoline and
// no cast anywhere. If that were wrong, this is where it would crash.
//
// The context goes in by `move` and comes back out through unregistration,
// which is an ordinary extern returning a struct by value. That needed no
// callback machinery at all, only the C return classification in
// src/c_abi.rs, which is what closed the last open question in
// docs/callbacks.md.
#[test]
fn a_callback_registered_with_a_c_library_runs() {
    let Some(compiler) = c_compiler() else {
        return;
    };
    let directory = std::env::temp_dir().join("frost_callback_abi");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();

    let library_source = directory.join("events.c");
    std::fs::write(
        &library_source,
        "#include <stdint.h>\n\
         static void (*held)(void*, int64_t);\n\
         static void* held_context;\n\
         int64_t register_handler(void (*handler)(void*, int64_t), void* context) {\n\
         \x20   held = handler;\n\
         \x20   held_context = context;\n\
         \x20   return 77;\n\
         }\n\
         void pump(int64_t code) { held(held_context, code); }\n\
         int64_t peek(void) { return *(int64_t*)held_context; }\n",
    )
    .unwrap();
    let library = directory.join("events.o");
    let built = Command::new(compiler)
        .arg("-c")
        .arg(&library_source)
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "the C library did not compile:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );

    let root = directory.join("events.frost");
    std::fs::write(
        &root,
        "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         pump :: extern fn(code: i64)\n\
         peek :: extern fn() -> i64\n\
         Ctx :: struct { hits: i64 }\n\
         Registration :: linear struct { token: i64 }\n\
         on_event :: fn(mut ctx: Ctx, code: i64) { ctx.hits = ctx.hits + code }\n\
         register_handler :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64\n\
         unregister :: fn(move r: Registration) -> i64 { r.token }\n\
         main :: fn() -> i64 {\n\
         \x20   c := Ctx { hits = 0 }\n\
         \x20   r := Registration { token = register_handler($on_event, c) }\n\
         \x20   pump(4)\n\
         \x20   pump(5)\n\
         \x20   printf(\"%lld\n\", peek())\n\
         \x20   printf(\"%lld\n\", unregister(r))\n\
         \x20   0\n\
         }\n",
    )
    .unwrap();

    let exe = directory.join(format!("events{}", std::env::consts::EXE_SUFFIX));
    let built = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("--libs")
        .arg(&library)
        .arg("-o")
        .arg(&exe)
        .arg(&root)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "the callback program did not build:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );

    let ran = Command::new(&exe).output().unwrap();
    let output = String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n");
    let _ = std::fs::remove_dir_all(&directory);
    // 4 then 5 through the callback, read back by the library out of the Frost
    // struct it was handed, then the token the library returned. The context is
    // read by the library rather than by Frost because it was moved in, and
    // getting it back is the open question at the end of docs/callbacks.md.
    assert_eq!(output, "9\n77\n");
}

// Item 4 of docs/roadmap.md. C returns a struct by a rule of its own, and the
// rule differs by target and, on some targets, by whether the fields are
// floating point. Every shape here was chosen because it lands on a different
// side of some boundary: 3 bytes is not a power of two, 4 bytes of float is the
// case where Windows and System V disagree, 16 bytes is the last size System V
// returns in registers and the first size Windows does not.
//
// The library is compiled by the C compiler, so its side of the call is the
// real convention rather than Frost's opinion of it. If the classification in
// src/c_abi.rs were wrong, the values would come back scrambled rather than
// missing, which is why this is a run rather than a compile.
#[test]
fn a_struct_returned_from_c_comes_back_correctly() {
    let Some(compiler) = c_compiler() else {
        return;
    };
    if !linker_available() {
        return;
    }
    let directory = std::env::temp_dir().join("frost_c_struct_returns");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();

    let library_source = directory.join("shapes.c");
    std::fs::write(
        &library_source,
        "#include <stdint.h>\n\
         #include <stdio.h>\n\
         typedef struct { uint8_t a; } S1;\n\
         typedef struct { int16_t a; } S2;\n\
         typedef struct { uint8_t a, b, c; } S3;\n\
         typedef struct { int32_t a; } S4;\n\
         typedef struct { int64_t a; } S8;\n\
         typedef struct { float a; } SF;\n\
         typedef struct { float a, b; } SFF;\n\
         typedef struct { int32_t a; float b; } SMix;\n\
         typedef struct { int64_t a, b; } S16;\n\
         typedef struct { double a, b; } SDD;\n\
         typedef struct { int64_t a, b, c; } S24;\n\
         S1 m1(void){ S1 v={7}; return v; }\n\
         S2 m2(void){ S2 v={-300}; return v; }\n\
         S3 m3(void){ S3 v={1,2,3}; return v; }\n\
         S4 m4(void){ S4 v={-70000}; return v; }\n\
         S8 m8(void){ S8 v={1234567890123}; return v; }\n\
         SF mf(void){ SF v={2.5f}; return v; }\n\
         SFF mff(void){ SFF v={1.5f,-3.25f}; return v; }\n\
         SMix mmix(void){ SMix v={-9, 6.75f}; return v; }\n\
         S16 m16(void){ S16 v={11,22}; return v; }\n\
         SDD mdd(void){ SDD v={1.25,-2.5}; return v; }\n\
         S24 m24(void){ S24 v={5,6,7}; return v; }\n\
         int printf_d(const char* fmt, double v) { return printf(fmt, v); }\n",
    )
    .unwrap();
    let library = directory.join("shapes.o");
    let built = Command::new(compiler)
        .arg("-c")
        .arg(&library_source)
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "the C library did not compile:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );

    let root = directory.join("shapes.frost");
    std::fs::write(
        &root,
        "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         printf_d :: extern fn(fmt: ^i8, value: f64) -> i32\n\
         S1 :: struct { a: u8 }\n\
         S2 :: struct { a: i16 }\n\
         S3 :: struct { a: u8, b: u8, c: u8 }\n\
         S4 :: struct { a: i32 }\n\
         S8 :: struct { a: i64 }\n\
         SF :: struct { a: f32 }\n\
         SFF :: struct { a: f32, b: f32 }\n\
         SMix :: struct { a: i32, b: f32 }\n\
         S16 :: struct { a: i64, b: i64 }\n\
         SDD :: struct { a: f64, b: f64 }\n\
         S24 :: struct { a: i64, b: i64, c: i64 }\n\
         m1 :: extern fn() -> S1\n\
         m2 :: extern fn() -> S2\n\
         m3 :: extern fn() -> S3\n\
         m4 :: extern fn() -> S4\n\
         m8 :: extern fn() -> S8\n\
         mf :: extern fn() -> SF\n\
         mff :: extern fn() -> SFF\n\
         mmix :: extern fn() -> SMix\n\
         m16 :: extern fn() -> S16\n\
         mdd :: extern fn() -> SDD\n\
         m24 :: extern fn() -> S24\n\
         show :: fn(v: i64) { printf(\"%lld\n\", v) }\n\
         showd :: fn(v: f64) { printf_d(\"%.4f\n\", v) }\n\
         main :: fn() -> i64 {\n\
         \x20   v1 := m1()   a1 : i64 = v1.a  show(a1)\n\
         \x20   v2 := m2()   a2 : i64 = v2.a  show(a2)\n\
         \x20   v3 := m3()   a3 : i64 = v3.a  b3 : i64 = v3.b  c3 : i64 = v3.c\n\
         \x20   show(a3 * 100 + b3 * 10 + c3)\n\
         \x20   v4 := m4()   a4 : i64 = v4.a  show(a4)\n\
         \x20   v8 := m8()   show(v8.a)\n\
         \x20   vf := mf()   af : f64 = vf.a  showd(af)\n\
         \x20   vff := mff() aff : f64 = vff.a  bff : f64 = vff.b  showd(aff)  showd(bff)\n\
         \x20   vm := mmix() am : i64 = vm.a  bm : f64 = vm.b  show(am)  showd(bm)\n\
         \x20   v16 := m16() show(v16.a)  show(v16.b)\n\
         \x20   vdd := mdd() showd(vdd.a)  showd(vdd.b)\n\
         \x20   v24 := m24() show(v24.a)  show(v24.b)  show(v24.c)\n\
         \x20   0\n\
         }\n",
    )
    .unwrap();

    let expected = "7\n-300\n123\n-70000\n1234567890123\n2.5000\n1.5000\n\
                    -3.2500\n-9\n6.7500\n11\n22\n1.2500\n-2.5000\n5\n6\n7\n";
    for emit_c in [false, true] {
        let exe = directory
            .join(format!("shapes_{emit_c}{}", std::env::consts::EXE_SUFFIX));
        let mut command = Command::new(env!("CARGO_BIN_EXE_frost"));
        if emit_c {
            command.arg("--emit-c");
        }
        let built = command
            .arg("--link")
            .arg("--libs")
            .arg(&library)
            .arg("-o")
            .arg(&exe)
            .arg(&root)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "the shapes program did not build (emit_c={emit_c}):\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let ran = Command::new(&exe).output().unwrap();
        let output = String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n");
        assert_eq!(output, expected, "emit_c={emit_c}");
    }

    let _ = std::fs::remove_dir_all(&directory);
}

// An enum is the one union-like shape Frost has, and a C ABI classifies a union
// by combining what every member could put in each byte. The flattening in
// `c_layout` carries every variant's fields for exactly that reason, so this is
// the case that would break if it carried only one.
#[test]
fn an_enum_returned_from_c_comes_back_correctly() {
    let Some(compiler) = c_compiler() else {
        return;
    };
    if !linker_available() {
        return;
    }
    let directory = std::env::temp_dir().join("frost_c_enum_return");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();

    let library_source = directory.join("shape.c");
    std::fs::write(
        &library_source,
        "#include <stdint.h>\n\
         typedef struct { uint32_t tag; int64_t v; } Shape;\n\
         Shape mk(void) { Shape s; s.tag = 1; s.v = 42; return s; }\n",
    )
    .unwrap();
    let library = directory.join("shape.o");
    assert!(
        Command::new(compiler)
            .arg("-c")
            .arg(&library_source)
            .arg("-o")
            .arg(&library)
            .output()
            .unwrap()
            .status
            .success()
    );

    let root = directory.join("shape.frost");
    std::fs::write(
        &root,
        "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         Shape :: enum { Empty, Full { v: i64 } }\n\
         mk :: extern fn() -> Shape\n\
         main :: fn() -> i64 {\n\
         \x20   s := mk()\n\
         \x20   match s {\n\
         \x20       case .Empty: printf(\"%lld\n\", 0)\n\
         \x20       case .Full { v }: printf(\"%lld\n\", v)\n\
         \x20   }\n\
         \x20   0\n\
         }\n",
    )
    .unwrap();

    for emit_c in [false, true] {
        let exe = directory
            .join(format!("shape_{emit_c}{}", std::env::consts::EXE_SUFFIX));
        let mut command = Command::new(env!("CARGO_BIN_EXE_frost"));
        if emit_c {
            command.arg("--emit-c");
        }
        let built = command
            .arg("--link")
            .arg("--libs")
            .arg(&library)
            .arg("-o")
            .arg(&exe)
            .arg(&root)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "the enum program did not build (emit_c={emit_c}):\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let ran = Command::new(&exe).output().unwrap();
        let output = String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n");
        assert_eq!(output, "42\n", "emit_c={emit_c}");
    }

    let _ = std::fs::remove_dir_all(&directory);
}

// A type error inside a specialization used to name a line in the generic's
// body, which is code the reader never wrote and often in a file they do not
// own. The call that asked for the specialization is the line they did write,
// so it comes first, and the specialization is named the way they wrote it
// rather than by its mangled symbol.
#[test]
fn an_error_inside_a_specialization_names_the_call() {
    let directory = std::env::temp_dir().join("frost_generic_diagnostic");
    let library = directory.join("lib");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&library).unwrap();
    std::fs::write(
        library.join("g.frost"),
        "export add\n\
         add :: fn($T: Type, move a: $T) -> $T { a + a }\n",
    )
    .unwrap();
    let root = directory.join("generic_diagnostic_app.frost");
    std::fs::write(
        &root,
        "import \"lib/g.frost\"\n\
         Point :: struct { x: i64 }\n\
         main :: fn() -> i64 {\n\
         \x20   p := Point { x = 1 }\n\
         \x20   q := add($Point, p)\n\
         \x20   0\n\
         }\n",
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--emit-c")
        .arg("-o")
        .arg(directory.join("out.c"))
        .arg(&root)
        .output()
        .unwrap();
    assert!(!output.status.success(), "the bad instantiation compiled");
    let message = String::from_utf8_lossy(&output.stderr).to_string();
    let _ = std::fs::remove_dir_all(&directory);

    // The call, first.
    assert!(
        message.contains("generic_diagnostic_app.frost:5:5"),
        "the error did not name the call site:\n{message}"
    );
    // Named the way it was written, not as `add__Point`.
    assert!(
        message.contains("instantiating 'add<Point>'"),
        "the error did not name the instantiation:\n{message}"
    );
    assert!(
        !message.contains("add__Point"),
        "a mangled name reached the reader:\n{message}"
    );
    // And the template position is still there for whoever owns the generic.
    assert!(
        message.contains("lib/g.frost:2:"),
        "the error lost the template position:\n{message}"
    );
}

// A field left out of an aggregate literal used to compile, and the storage it
// named was never written, so reading it read whatever was on the stack. That
// is the shape goal 2 says should be unrepresentable, and nothing downstream
// could have caught it: the value has a type, an address, and a plausible bit
// pattern.
#[test]
fn an_aggregate_literal_must_write_every_field() {
    let message = compile_error(
        "partial_struct",
        "E :: struct { hp: i64, mana: i64 }\n\
         main :: fn() -> i64 { e := E { hp = 5 }  e.mana }\n",
    );
    assert!(
        message.contains("is missing field 'mana'"),
        "expected the missing field to be named, got:\n{message}"
    );

    let message = compile_error(
        "empty_struct",
        "E :: struct { hp: i64, mana: i64 }\n\
         main :: fn() -> i64 { e := E {}  e.hp }\n",
    );
    assert!(
        message.contains("missing fields 'hp', 'mana'"),
        "expected both fields to be named, got:\n{message}"
    );

    // An enum payload is the same storage with a tag in front of it.
    let message = compile_error(
        "partial_variant",
        "Shape :: enum { Rect { w: i64, h: i64 } }\n\
         main :: fn() -> i64 {\n\
         \x20   s := Shape::Rect { w = 3 }\n\
         \x20   match s { case .Rect { w, h }: h }\n\
         }\n",
    );
    assert!(
        message.contains("is missing field 'h'"),
        "expected the missing payload field to be named, got:\n{message}"
    );
}

// A generic struct literal passed straight to a generic function used to fail
// with "unknown struct 'Pair'": the argument was lowered with no expected type,
// so the literal had nothing to tell it which instance it was and fell back to
// the bare template, which has no layout. The parameter's type is known at the
// call, so the fix is to substitute what is bound so far and hand that down.
#[test]
fn a_generic_literal_can_be_a_generic_argument() {
    let source = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Pair :: struct($T: Type) { first: T, second: T }
Slab :: struct($T: Type, $N: usize) { storage: [N]T, count: i64 }

sum :: fn($T: Type, move v: $T) -> i64 { v.first + v.second }

insert :: fn($T: Type, $N: usize, mut s: Slab<T, N>, move value: $T) -> i64 {
    index := s.count
    s.count = s.count + 1
    s.storage[index] = value
    index
}

zero :: fn() -> Pair<i64> { Pair { first = 0, second = 0 } }

main :: fn() -> i64 {
    printf("%lld\n", sum($Pair<i64>, Pair { first = 3, second = 4 }))

    mut pool : Slab<Pair<i64>, 4> = Slab { storage = [zero(); 4], count = 0 }
    h := insert($Pair<i64>, $4, pool, Pair { first = 10, second = 20 })
    printf("%lld\n", pool.storage[h].first + pool.storage[h].second)
    0
}
"#;
    let Some(output) = compile_and_run("generic_literal_arg", source) else {
        return;
    };
    assert_eq!(output, "7\n30\n");
}

// A struct could take type parameters and an enum could not, so there was no
// way to write a sum type over an arbitrary element: no `Maybe<T>`, no
// `Result<T, E>`, and no way for a library to offer one. This covers the shapes
// that would break separately: two instances of one enum, two type parameters,
// an aggregate payload, and an instance nested inside another.
#[test]
fn an_enum_can_be_generic() {
    let source = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }
Maybe :: enum($T: Type) { Nothing, Just { value: T } }
Either :: enum($L: Type, $R: Type) { Left { value: L }, Right { value: R } }

unwrap_or :: fn($T: Type, m: Maybe<T>, fallback: $T) -> $T {
    match m {
        case .Nothing: fallback
        case .Just { value }: value
    }
}

main :: fn() -> i64 {
    a : Maybe<i64> = Maybe::Just { value = 42 }
    b : Maybe<i64> = Maybe::Nothing
    printf("%lld\n", unwrap_or($i64, a, 0))
    printf("%lld\n", unwrap_or($i64, b, 7))

    p : Maybe<Point> = Maybe::Just { value = Point { x = 3, y = 4 } }
    match p {
        case .Nothing: printf("%lld\n", 0)
        case .Just { value }: printf("%lld\n", value.x + value.y)
    }

    e : Either<i64, Point> = Either::Right { value = Point { x = 5, y = 6 } }
    match e {
        case .Left { value }: printf("%lld\n", value)
        case .Right { value }: printf("%lld\n", value.y)
    }

    nested : Maybe<Maybe<i64>> = Maybe::Just { value = Maybe::Just { value = 8 } }
    match nested {
        case .Nothing: printf("%lld\n", 0)
        case .Just { value }: match value {
            case .Nothing: printf("%lld\n", 0)
            case .Just { value }: printf("%lld\n", value)
        }
    }
    0
}
"#;
    let Some(output) = compile_and_run("generic_enum", source) else {
        return;
    };
    assert_eq!(output, "42\n7\n7\n6\n8\n");
}

// Import resolution has four ways to find a module beyond the importing file's
// own directory, and this exercises each one on the same program so the only
// thing that differs is how the library was reached. Getting any of them wrong
// shows up as a compile failure rather than a wrong answer, which is why they
// share one library and one expected output.
#[test]
fn an_import_resolves_through_every_search_root() {
    if !linker_available() {
        return;
    }
    let directory = std::env::temp_dir().join("frost_search_roots");
    let elsewhere = directory.join("elsewhere");
    let declared = directory.join("declared");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::fs::create_dir_all(&declared).unwrap();

    let library = "export twice\ntwice :: fn(x: i64) -> i64 { x * 2 }\n";
    std::fs::write(elsewhere.join("helper.frost"), library).unwrap();
    std::fs::write(declared.join("helper.frost"), library).unwrap();
    // A neighbour of the entry file, which is the case that needs no roots.
    std::fs::write(directory.join("beside.frost"), library).unwrap();

    let program = |import: &str| {
        format!(
            "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
             import \"{import}\"\n\
             main :: fn() -> i64 {{ printf(\"%lld\n\", twice(21))  0 }}\n"
        )
    };

    let build =
        |name: &str, source: &str, args: &[&str], env: &[(&str, &str)]| {
            let root = directory.join(format!("{name}.frost"));
            std::fs::write(&root, source).unwrap();
            let exe = directory
                .join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
            let mut command = Command::new(env!("CARGO_BIN_EXE_frost"));
            for (key, value) in env {
                command.env(key, value);
            }
            let built = command
                .args(args)
                .arg("--link")
                .arg("-o")
                .arg(&exe)
                .arg(&root)
                .output()
                .unwrap();
            assert!(
                built.status.success(),
                "{name} did not build:\n{}",
                String::from_utf8_lossy(&built.stderr)
            );
            let ran = Command::new(&exe).output().unwrap();
            normalize_newlines(&ran.stdout)
        };

    // 1. Beside the importing file. No search root involved.
    assert_eq!(
        build("neighbour", &program("beside.frost"), &[], &[]),
        "42\n"
    );

    // 2. A directory named on the command line.
    assert_eq!(
        build(
            "flagged",
            &program("helper.frost"),
            &["-L", elsewhere.to_str().unwrap()],
            &[]
        ),
        "42\n"
    );

    // 3. A directory named by the environment.
    assert_eq!(
        build(
            "environment",
            &program("helper.frost"),
            &[],
            &[("FROST_PATH", elsewhere.to_str().unwrap())]
        ),
        "42\n"
    );

    // 4. A directory the project's manifest declares.
    std::fs::write(
        directory.join("frost.json"),
        r#"{ "name": "demo", "paths": ["declared"] }"#,
    )
    .unwrap();
    assert_eq!(
        build("manifest", &program("helper.frost"), &[], &[]),
        "42\n"
    );
    std::fs::remove_file(directory.join("frost.json")).unwrap();

    // 5. The standard library, which needs nothing declared at all.
    let uses_std = "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         import \"maybe.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   m := maybe_some($i64, 42)\n\
         \x20   printf(\"%lld\n\", maybe_or($i64, m, 0))\n\
         \x20   0\n\
         }\n";
    assert_eq!(build("standard", uses_std, &[], &[]), "42\n");

    let _ = std::fs::remove_dir_all(&directory);
}

// A failing test used to end the run, so one bad test hid every test after it,
// and the only thing it said was "assertion failed". Now the failure ends that
// test, names where it was written, and the run continues to a summary.
#[test]
fn a_failing_test_does_not_hide_the_ones_after_it() {
    let source = "add :: fn(a: i64, b: i64) -> i64 { a + b }\n\
                  test \"wrong\" { assert(add(2, 2) == 5) }\n\
                  test \"right\" { assert(add(1, 1) == 2) }\n";
    let Some((output, ok)) = run_test_mode("mixed", source) else {
        return;
    };
    assert!(!ok, "a failing test must fail the run:\n{output}");
    assert!(output.contains("test wrong ... FAILED"), "got:\n{output}");
    // The test after the failure still ran, which is the whole point.
    assert!(output.contains("test right ... ok"), "got:\n{output}");
    assert!(output.contains("1 passed, 1 failed"), "got:\n{output}");
}

// Mixed-width integer arithmetic widens to the wider operand, which the spec
// has always said and the compiler used to do backwards: an `i64` mixed with a
// narrower integer took the *narrower* type, so an accumulator fed by string
// bytes computed at eight bits and silently answered the wrong number. This is
// the shape that found it, reading a decimal integer out of a `str`.
#[test]
fn mixed_width_arithmetic_widens_to_the_wider_operand() {
    let source = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

to_i64 :: fn(s: str) -> i64 {
    mut value : i64 = 0
    mut i : i64 = 0
    while (i < str_len(s)) {
        value = value * 10 + (s[i] - 48)
        i = i + 1
    }
    value
}

main :: fn() -> i64 {
    printf("%lld\n", to_i64("1234567"))

    text := "7"
    byte := text[0]
    mut accumulator : i64 = 1234
    printf("%lld\n", accumulator * 10 + (byte - 48))

    // A literal still takes the width of what it is combined with, which is
    // what the backwards rule was there to protect and which still holds.
    mut small : u8 = 250
    small = small + 10
    wide : i64 = small
    printf("%lld\n", wide)
    0
}
"#;
    let Some(output) = compile_and_run("mixed_widths", source) else {
        return;
    };
    // 1234567 read a byte at a time, 12347 from the same shape by hand, and
    // 250 + 10 wrapping at eight bits because both sides really are u8.
    assert_eq!(output, "1234567\n12347\n4\n");
}
