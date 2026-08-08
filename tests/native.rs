use std::path::{Path, PathBuf};
use std::process::Command;

use frost::{CLayout, CReturn, CScalar, CTarget, Type, classify_return};

#[path = "support.rs"]
mod support;

use support::{
    bootstrap_output, bootstrap_refusal, build_self_hosted_compiler,
    c_compiler, frost_runtime_object, frost_runtime_source, in_parallel,
    linker_available, runtime_object, runtime_source, self_hosted_source,
    selfhosted_default_output, unique,
};

// A temp-file stem no run and no other test reuses. On Windows a just-run or
// just-deleted executable stays briefly locked, so relinking over the same name
// fails intermittently; a fresh name every time sidesteps it. The process id
// separates one `cargo test` run from the next, the counter separates tests
// within a run.
fn compile_and_run_unaudited(name: &str, source: &str) -> Option<String> {
    run_backend(name, source, false)
}

// The same, for a program expected to end on a runtime check. Answers what it
// printed where it ran to the end, and nothing where a check stopped it, which
// is the shape `run_ir_oracle` answers in so the two can be compared.
fn compile_and_run_checked(name: &str, source: &str) -> Option<Option<String>> {
    if !linker_available() {
        return None;
    }
    let directory = std::env::temp_dir();
    let source_path = directory.join(format!("frost_check_{name}.frost"));
    let exe_path = directory.join(format!(
        "frost_check_{name}{}",
        std::env::consts::EXE_SUFFIX
    ));
    std::fs::write(&source_path, source).unwrap();
    let built = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("-o")
        .arg(&exe_path)
        .arg(&source_path)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "the bootstrap refused {name}:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let run = Command::new(&exe_path).output().unwrap();
    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_file(&exe_path);
    if !run.status.success() {
        return Some(None);
    }
    Some(Some(
        String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"),
    ))
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
    // The interface oracle runs on every test compilation, so a module whose interface would not describe it is caught
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
    compile_error_gated(name, source)
}

/// A compile error from a source built with the unsafety gate on, which is the
/// default and the only way a real program is written. `compile_error` turns
/// the gate off, so anything it checks may say `ptr_to` bare, and a check that
/// only ever sees the bare form is not being tested against what programs
/// actually contain.
fn compile_error_checked(name: &str, source: &str) -> String {
    compile_error_gated(name, source)
}

fn compile_error_gated(name: &str, source: &str) -> String {
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

// A binding taken from a call answering with several values, then indexed. The
// multiple-return lowering runs after the unsafe gate, so the gate sees the
// destructure and bound each name to no type at all; the index rule then met a
// base it could not name and refused, which is the answer it owes a raw pointer
// rather than one it owes ordinary code.
#[test]
fn a_destructured_multi_return_can_be_indexed() {
    let source = "import \"io.frost\"\n\
split :: fn(mut source: [4]i64) -> (view: []i64, count: i64) {\n\
\x20   return { view = source, count = 4 }\n\
}\n\
main :: fn() -> i64 {\n\
\x20   var data : [4]i64 = [11, 22, 33, 44]\n\
\x20   view, count := split(data)\n\
\x20   print(\"{}\\n\", view[0] + count)\n\
\x20   0\n\
}\n";
    let Some(output) = compile_and_run_unaudited("multiindex", source) else {
        return;
    };
    assert_eq!(output, "15\n");
}

// A `uses` function indexing the arena it draws from. The capability is threaded
// in as a parameter by a lowering that runs after the unsafe gate, so the body
// names something nothing has declared, and the index rule refuses a base whose
// type it cannot read. That left no way to write a bump allocator over its own
// arena without an `unsafe` block around an operation that is checked.
#[test]
fn a_uses_function_indexes_its_own_arena() {
    let source = "import \"io.frost\"\n\
Arena :: struct($N: usize) { data: [N]u8, offset: i64 }\n\
bump :: fn() -> i64 uses Arena<256> {\n\
\x20   slot := ptr_to(arena.data[arena.offset])\n\
\x20   arena.offset = arena.offset + 8\n\
\x20   arena.offset\n\
}\n\
main :: fn() -> i64 {\n\
\x20   var arena : Arena<256> = Arena { data = [0; 256], offset = 0 }\n\
\x20   var total : i64 = 0\n\
\x20   with arena {\n\
\x20       total = bump()\n\
\x20   }\n\
\x20   print(\"{}\\n\", total)\n\
\x20   0\n\
}\n";
    let Some(output) = compile_and_run_unaudited("usesindex", source) else {
        return;
    };
    assert_eq!(output, "8\n");
}

// Growing one run while a caller holds a view of another is what a container
// with more than one run does all the time, and it is the case that decides how
// fine the run summaries have to be. The ECS is the program that does it: a
// group grows its slots while a `ref` into its members is live, and a summary
// naming only the parameter cannot tell the two apart.
#[test]
fn growing_one_run_leaves_a_view_of_another_alone() {
    let source = "import \"io.frost\"\n\
import \"vec.frost\"\n\
Pair :: linear struct { left: Vec<i64>, right: Vec<i64> }\n\
pair_right :: fn(p: Pair) -> []i64 { vec_slice($i64, p.right) }\n\
pair_grow :: fn(mut p: Pair, value: i64) { vec_push($i64, p.left, value) }\n\
pair_free :: fn(move p: Pair) { vec_free($i64, p.left)  vec_free($i64, p.right) }\n\
main :: fn() -> i64 {\n\
\x20   var pair : Pair = { left = vec_new($i64, 1), right = vec_new($i64, 1) }\n\
\x20   vec_push($i64, pair.right, 7)\n\
\x20   view := pair_right(pair)\n\
\x20   pair_grow(pair, 1)\n\
\x20   pair_grow(pair, 2)\n\
\x20   print(\"{}\\n\", view[0])\n\
\x20   pair_free(pair)\n\
\x20   0\n\
}\n";
    let Some(output) = compile_and_run_unaudited("rungrain", source) else {
        return;
    };
    assert_eq!(output, "7\n");
}

// The view taken again after the growth, which is what the refusal tells a
// caller to do. Rebinding has to clear the staleness or the advice is no advice.
#[test]
fn a_view_taken_again_after_a_growth_reads_the_new_block() {
    let source = "import \"io.frost\"\n\
import \"vec.frost\"\n\
main :: fn() -> i64 {\n\
\x20   var v := vec_new($i64, 1)\n\
\x20   vec_push($i64, v, 111)\n\
\x20   var view := vec_slice($i64, v)\n\
\x20   var count : i64 = 0\n\
\x20   while (count < 8) {\n\
\x20       vec_push($i64, v, count)\n\
\x20       view = vec_slice($i64, v)\n\
\x20       count = count + 1\n\
\x20   }\n\
\x20   print(\"{}\\n\", view[0])\n\
\x20   vec_free($i64, v)\n\
\x20   0\n\
}\n";
    let Some(output) = compile_and_run_unaudited("runretake", source) else {
        return;
    };
    assert_eq!(output, "111\n");
}

// A number copied out of a container is a value of its own from the moment it is
// made, so a push afterwards has nothing to say about it. Reading the copy as a
// view of the block it came out of refused the ECS, which reads a generation out
// of its slots and pushes to the same container two lines later.
#[test]
fn a_number_copied_out_of_a_container_survives_a_growth() {
    let source = "import \"io.frost\"\n\
import \"vec.frost\"\n\
main :: fn() -> i64 {\n\
\x20   var v := vec_new($i64, 1)\n\
\x20   vec_push($i64, v, 111)\n\
\x20   held := vec_slice($i64, v)[0]\n\
\x20   var count : i64 = 0\n\
\x20   while (count < 8) {\n\
\x20       vec_push($i64, v, count)\n\
\x20       count = count + 1\n\
\x20   }\n\
\x20   print(\"{}\\n\", held)\n\
\x20   vec_free($i64, v)\n\
\x20   0\n\
}\n";
    let Some(output) = compile_and_run_unaudited("runcopy", source) else {
        return;
    };
    assert_eq!(output, "111\n");
}

// A view forwarded through a wrapper and read before anything moves the run.
// The rule that refuses the stale form has to leave this one alone, or every
// accessor written over a container becomes unusable.
#[test]
fn a_view_through_a_wrapper_reads_before_the_growth() {
    let source = "import \"io.frost\"\n\
import \"vec.frost\"\n\
passthrough :: fn(s: []i64) -> []i64 { s }\n\
main :: fn() -> i64 {\n\
\x20   var v := vec_new($i64, 1)\n\
\x20   vec_push($i64, v, 111)\n\
\x20   view := passthrough(vec_slice($i64, v))\n\
\x20   print(\"{}\\n\", view[0])\n\
\x20   vec_push($i64, v, 222)\n\
\x20   again := passthrough(vec_slice($i64, v))\n\
\x20   print(\"{}\\n\", again[1])\n\
\x20   vec_free($i64, v)\n\
\x20   0\n\
}\n";
    let Some(output) = compile_and_run_unaudited("wrapview", source) else {
        return;
    };
    assert_eq!(output, "111\n222\n");
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
    var value : i64 = 1
    total := add_both(value, value)
    total
}
"#;
    let message = compile_error("exclusivity", source);
    assert!(
        message.contains(":6:"),
        "expected the exclusivity error at line 6, got:\n{message}"
    );
    assert!(
        message.contains("exclusive"),
        "expected the exclusivity reason, got:\n{message}"
    );
}

// Exclusivity is over the place a borrow names, not just its root variable, so
// two mutable borrows of the same field of the same struct conflict.
#[test]
fn borrow_exclusivity_catches_an_overlapping_field_path() {
    let source = r#"
Pair :: struct { x: i64, y: i64 }
mix :: fn(mut a: i64, mut b: i64) -> i64 { a + b }

main :: fn() -> i64 {
    var p : Pair = Pair { x = 1, y = 2 }
    total := mix(p.x, p.x)
    total
}
"#;
    let message = compile_error("field_exclusivity", source);
    assert!(
        message.contains("p.x") && message.contains("exclusive"),
        "expected a place-path exclusivity error naming p.x, got:\n{message}"
    );
}

// Every step in front of a raw dereference says where the pointer was read from
// and none of them says where it points, so two places that each reach through
// one may be one place however different their roots read. They used to be read
// as apart, and `f(p^, q^)` with `p` and `q` holding one address handed the same
// storage to two `mut` parameters.
#[test]
fn two_places_reached_through_raw_pointers_may_be_one() {
    let source = "bump_both :: fn(mut a: i64, mut b: i64) {\n\
         \x20   a = a + 1\n\
         \x20   b = b + 10\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   var x : i64 = 0\n\
         \x20   p := unsafe { ptr_to(x) }\n\
         \x20   q := unsafe { ptr_to(x) }\n\
         \x20   unsafe { bump_both(p^, q^) }\n\
         \x20   0\n}\n";
    let message = compile_error_checked("rawalias", source);
    assert!(
        message.contains("exclusive"),
        "two raw dereferences should not be read as apart, got:\n{message}"
    );
}

// The same question with only one side raw: `p` may hold `x`'s address, and the
// names settle that no better than they settle two pointers against each other.
#[test]
fn a_raw_place_and_an_ordinary_one_may_be_one() {
    let source = "bump_both :: fn(mut a: i64, mut b: i64) {\n\
         \x20   a = a + 1\n\
         \x20   b = b + 10\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   var x : i64 = 0\n\
         \x20   p := unsafe { ptr_to(x) }\n\
         \x20   unsafe { bump_both(p^, x) }\n\
         \x20   0\n}\n";
    let message = compile_error_checked("rawalias2", source);
    assert!(
        message.contains("exclusive"),
        "a raw place and the local it may name should not be apart, got:\n{message}"
    );
}

// The rule has to tell a raw pointer from a borrow, because the parameter-mode
// lowering rewrites every `mut` scalar parameter to `name^`. Two distinct `mut`
// parameters passed on to a second call are two dereferences and are genuinely
// apart, which is what the caller's own exclusivity check says.
#[test]
fn two_borrowed_parameters_passed_on_are_still_apart() {
    let source = "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         bump_both :: fn(mut a: i64, mut b: i64) {\n\
         \x20   a = a + 1\n\
         \x20   b = b + 10\n\
         }\n\
         outer :: fn(mut m: i64, mut n: i64) { bump_both(m, n) }\n\
         main :: fn() -> i64 {\n\
         \x20   var x : i64 = 0\n\
         \x20   var y : i64 = 0\n\
         \x20   outer(x, y)\n\
         \x20   unsafe { printf(\"%lld\\n\", x + y) }\n\
         \x20   0\n}\n";
    let Some(output) = compile_and_run_unaudited("borrowedparams", source)
    else {
        return;
    };
    assert_eq!(output, "11\n");
}

// Two mutable borrows of different fields of one struct name disjoint storage,
// so they do not conflict and the program compiles and runs.
#[test]
fn borrow_exclusivity_allows_disjoint_fields() {
    let source = "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
        Pair :: struct { x: i64, y: i64 }\n\
        mix :: fn(mut a: i64, mut b: i64) -> i64 { a + b }\n\
        main :: fn() -> i64 {\n\
        \x20   var p : Pair = Pair { x = 3, y = 4 }\n\
        \x20   unsafe { printf(\"%lld\\n\", mix(p.x, p.y)) }\n    0\n}\n";
    assert_eq!(
        compile_and_run_unaudited("disjoint_fields", source),
        Some("7\n".to_string())
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

// Compiles and runs a program, answering whether it exited cleanly and what it
// wrote to stderr. An abort test wants both: the nonzero exit and the message
// the runtime composed, so a crash for the wrong reason does not read as the
// right one.
fn compile_and_run_status(name: &str, source: &str) -> Option<(bool, String)> {
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
    let stderr = String::from_utf8_lossy(&run.stderr).replace("\r\n", "\n");
    Some((run.status.success(), stderr))
}

const OUT_OF_BOUNDS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    arr := [10, 20, 30]
    var i : i64 = 5
    unsafe { printf("%lld\n", arr[i]) }
    0
}
"#;

#[test]
fn native_out_of_bounds_index_aborts() {
    let Some((succeeded, stderr)) =
        compile_and_run_status("oob", OUT_OF_BOUNDS)
    else {
        return;
    };
    assert!(!succeeded, "out-of-bounds index should abort at runtime");
    assert!(
        stderr.contains("out of bounds"),
        "expected the bounds-check message, got:\n{stderr}"
    );
}

// A `ref` binding is a borrow of a container element, not a copy. Writing
// through it writes to the element, which is the reusable handle a container
// needs without a raw pointer. Both backends must agree, and the frame check
// must refuse letting it escape.
// The hash map from std: many puts forcing a rehash, lookups with a default,
// removal, and presence. A dozen generic functions over one value type, several
// sharing a lookup helper, which is the shape that exercises the specialization
// worklist hardest. Both backends must agree.
const MAP_LIBRARY: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32
frost_rt_heap_alloc :: extern fn(size: i64) -> ^u8
frost_rt_heap_free  :: extern fn(block: ^u8)

Map :: struct($V: Type) { keys: ^i64, values: ^V, state: ^u8, cap: i64, count: i64 }

map_new :: fn($V: Type, capacity: i64) -> Map<V> {
    var cap : i64 = 8
    while (cap < capacity) { cap = cap * 2 }
    keys := unsafe { ptr_cast($i64, frost_rt_heap_alloc(cap * 8)) }
    values := unsafe { ptr_cast($V, frost_rt_heap_alloc(cap * sizeof(V))) }
    state := unsafe { ptr_cast($u8, frost_rt_heap_alloc(cap)) }
    var i : i64 = 0
    while (i < cap) { unsafe { state[i] = 0 }  i = i + 1 }
    Map { keys = keys, values = values, state = state, cap = cap, count = 0 }
}
map_find :: fn($V: Type, m: Map<V>, key: i64) -> i64 {
    var h := key * 2654435761
    var i := (h + h / 4096) % m.cap
    while (true) {
        s := unsafe { m.state[i] }
        if (s == 0) { return i }
        if (s == 1 && unsafe { m.keys[i] } == key) { return i }
        i = i + 1
        if (i >= m.cap) { i = 0 }
    }
    0 - 1
}
map_insert :: fn($V: Type, mut m: Map<V>, key: i64, move value: $V) {
    slot := map_find($V, m, key)
    if (unsafe { m.state[slot] } != 1) { m.count = m.count + 1 }
    unsafe { m.keys[slot] = key  m.values[slot] = value  m.state[slot] = 1 }
}
map_grow :: fn($V: Type, mut m: Map<V>) {
    ok := m.keys  ov := m.values  os := m.state  oc := m.cap
    fresh := map_new($V, m.cap * 2)
    m.keys = fresh.keys  m.values = fresh.values  m.state = fresh.state
    m.cap = fresh.cap  m.count = 0
    var i : i64 = 0
    while (i < oc) {
        if (unsafe { os[i] } == 1) { map_insert($V, m, unsafe { ok[i] }, unsafe { ov[i] }) }
        i = i + 1
    }
    unsafe { frost_rt_heap_free(ptr_cast($u8, ok))  frost_rt_heap_free(ptr_cast($u8, ov))  frost_rt_heap_free(os) }
}
map_put :: fn($V: Type, mut m: Map<V>, key: i64, move value: $V) {
    if (m.count * 2 >= m.cap) { map_grow($V, m) }
    map_insert($V, m, key, value)
}
map_get :: fn($V: Type, m: Map<V>, key: i64, move fallback: $V) -> $V {
    slot := map_find($V, m, key)
    if (unsafe { m.state[slot] } == 1) { return unsafe { m.values[slot] } }
    fallback
}
map_has :: fn($V: Type, m: Map<V>, key: i64) -> bool {
    slot := map_find($V, m, key)
    unsafe { m.state[slot] } == 1
}
map_remove :: fn($V: Type, mut m: Map<V>, key: i64) -> bool {
    slot := map_find($V, m, key)
    found := unsafe { m.state[slot] } == 1
    if (found) { unsafe { m.state[slot] = 2 }  m.count = m.count - 1 }
    found
}
main :: fn() -> i64 {
    var m := map_new($i64, 4)
    var i : i64 = 0
    while (i < 50) { map_put($i64, m, i, i * i)  i = i + 1 }
    unsafe { printf("%lld\n", map_len_i(m)) }
    unsafe { printf("%lld\n", map_get($i64, m, 7, -1)) }
    unsafe { printf("%lld\n", map_get($i64, m, 999, -1)) }
    if (map_remove($i64, m, 7)) { unsafe { printf("%lld\n", 1) } }
    if (map_has($i64, m, 7)) { unsafe { printf("%lld\n", 1) } } else { unsafe { printf("%lld\n", 0) } }
    unsafe { frost_rt_heap_free(ptr_cast($u8, m.keys)) }
    0
}
map_len_i :: fn(m: Map<i64>) -> i64 { m.count }
"#;

// The growable vector from std, exercised end to end: pushing past capacity so
// it reallocates, reading, setting, and summing. It leans on slice_from over
// heap storage, sizeof inside unsafe, and a generic move value, each of which
// was a compiler fix. Both backends must agree.
const VEC_LIBRARY: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32
frost_rt_heap_alloc   :: extern fn(size: i64) -> ^u8
frost_rt_heap_realloc :: extern fn(block: ^u8, size: i64) -> ^u8
frost_rt_heap_free    :: extern fn(block: ^u8)

Vec :: struct($T: Type) { data: ^T, len: i64, cap: i64 }

vec_new :: fn($T: Type, capacity: i64) -> Vec<T> {
    var room := capacity
    if (room < 1) { room = 1 }
    block := unsafe { frost_rt_heap_alloc(room * sizeof(T)) }
    Vec { data = unsafe { ptr_cast($T, block) }, len = 0, cap = room }
}
vec_push :: fn($T: Type, mut v: Vec<T>, move value: $T) {
    if (v.len >= v.cap) {
        var room := v.cap * 2
        if (room < 1) { room = 1 }
        v.data = unsafe { ptr_cast($T, frost_rt_heap_realloc(ptr_cast($u8, v.data), room * sizeof(T))) }
        v.cap = room
    }
    unsafe { v.data[v.len] = value }
    v.len = v.len + 1
}
vec_get :: fn($T: Type, v: Vec<T>, index: i64) -> $T { unsafe { v.data[index] } }
vec_set :: fn($T: Type, mut v: Vec<T>, index: i64, move value: $T) { unsafe { v.data[index] = value } }
vec_len :: fn($T: Type, v: Vec<T>) -> i64 { v.len }

main :: fn() -> i64 {
    var v := vec_new($i64, 2)
    var i : i64 = 0
    while (i < 10) { vec_push($i64, v, i * i)  i = i + 1 }
    unsafe { printf("%lld\n", vec_len($i64, v)) }
    unsafe { printf("%lld\n", vec_get($i64, v, 9)) }
    vec_set($i64, v, 3, 999)
    unsafe { printf("%lld\n", vec_get($i64, v, 3)) }
    var sum : i64 = 0
    var j : i64 = 0
    while (j < vec_len($i64, v)) { sum = sum + vec_get($i64, v, j)  j = j + 1 }
    unsafe { printf("%lld\n", sum) }
    unsafe { frost_rt_heap_free(ptr_cast($u8, v.data)) }
    0
}
"#;

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
    var backing : [4]Node = [Node{a=0,b=0}, Node{a=0,b=0}, Node{a=0,b=0}, Node{a=0,b=0}]
    var ar := Arena { data = backing, count = 0 }
    grow(ar, 2)
    grow(ar, 3)
    unsafe { printf("%lld\n", read_it(ar, 2)) }
    unsafe { printf("%lld\n", read_it(ar, 3)) }
    0
}
"#;

const ARITHMETIC: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

factorial :: fn(n: i64) -> i64 {
    if (n <= 1) { 1 } else { n * factorial(n - 1) }
}

sum_to :: fn(n: i64) -> i64 {
    var total : i64 = 0
    var i : i64 = 0
    while (i <= n) {
        total = total + i
        i = i + 1
    }
    total
}

count_evens :: fn(limit: i64) -> i64 {
    var count : i64 = 0
    for i in 0..limit {
        if (i % 2 == 0) { count = count + 1 } else { count = count + 0 }
    }
    count
}

main :: fn() -> i64 {
    unsafe { printf("%lld\n", factorial(10)) }
    unsafe { printf("%lld\n", sum_to(100)) }
    unsafe { printf("%lld\n", count_evens(10)) }
    unsafe { printf("%lld\n", if (3 < 5 && 5 < 10) { 1 } else { 0 }) }
    unsafe { printf("%lld\n", if (2 > 9 || 4 == 4) { 1 } else { 0 }) }
    unsafe { printf("%lld\n", 1 << 10) }
    unsafe { printf("%lld\n", 100 % 7) }
    0
}
"#;

#[test]
fn native_arithmetic_and_control_flow() {
    let Some(output) = compile_and_run_unaudited("arith", ARITHMETIC) else {
        return;
    };
    assert_eq!(output, "3628800\n5050\n5\n1\n1\n1024\n2\n");
}

const FLOATS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    unsafe { printf("%lld\n", if (7.0 / 2.0 > 3.0) { 1 } else { 0 }) }
    unsafe { printf("%lld\n", if (1.5 + 1.5 == 3.0) { 1 } else { 0 }) }
    unsafe { printf("%lld\n", if (2.0 * 2.0 < 3.9) { 1 } else { 0 }) }
    0
}
"#;

#[test]
fn native_float_operations() {
    let Some(output) = compile_and_run_unaudited("floats", FLOATS) else {
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
    unsafe { printf("%lld\n", if (c == 4.0) { 1 } else { 0 }) }
    widened : f64 = c
    unsafe { printf("%lld\n", if (widened == 4.0) { 1 } else { 0 }) }
    unsafe { printf("%lld\n", if (scale(3.0, 2.5) == 7.5) { 1 } else { 0 }) }
    arr : [3]f32 = [1.5, 2.5, 3.0]
    unsafe { printf("%lld\n", if (arr[1] == 2.5) { 1 } else { 0 }) }
    0
}
"#;

#[test]
fn native_f32_operations() {
    let Some(output) = compile_and_run_unaudited("f32ops", F32_OPERATIONS)
    else {
        return;
    };
    assert_eq!(output, "1\n1\n1\n1\n");
}

const WIDTHS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    small : i32 = 300
    unsafe { printf("%lld\n", small) }
    byte_sum : u8 = 100
    unsafe { printf("%lld\n", byte_sum + 50) }
    0
}
"#;

#[test]
fn native_integer_widths_and_casts() {
    let Some(output) = compile_and_run_unaudited("widths", WIDTHS) else {
        return;
    };
    assert_eq!(output, "300\n150\n");
}

const WRAPPING_AND_UNARY: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    a : u8 = 200
    b : u8 = 100
    unsafe { printf("%lld\n", wrap_add(a, b)) }
    d : u32 = 4000000000
    e : u32 = 1000000000
    unsafe { printf("%lld\n", wrap_add(d, e)) }
    g : i64 = 42
    unsafe { printf("%lld\n", -g) }
    0
}
"#;

#[test]
fn native_wrapping_and_unary() {
    let Some(output) =
        compile_and_run_unaudited("wrapping", WRAPPING_AND_UNARY)
    else {
        return;
    };
    assert_eq!(output, "44\n705032704\n-42\n");
}

const ANON_FUNCTIONS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

apply :: fn(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }

main :: fn() -> i64 {
    unsafe { printf("%lld\n", apply(fn(a: i64) -> i64 { a + 1 }, 41)) }
    unsafe { printf("%lld\n", apply(fn(a: i64) -> i64 { a * a }, 9)) }
    g := fn(a: i64) -> i64 { a - 3 }
    unsafe { printf("%lld\n", g(50)) }
    ops := [fn(a: i64) -> i64 { a + 1 }, fn(a: i64) -> i64 { a * 2 }]
    unsafe { printf("%lld\n", ops[1](10)) }
    0
}
"#;

const PARAM_MODES: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }

bump :: fn(mut p: Point) { p.x = p.x + 1 }

sum :: fn(p: Point) -> i64 { p.x + p.y }

main :: fn() -> i64 {
    var pt : Point = Point { x = 5, y = 10 }
    bump(pt)
    bump(pt)
    unsafe { printf("%lld\n", pt.x) }
    unsafe { printf("%lld\n", sum(pt)) }
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
        case .Err { error }: -1
    }
}

main :: fn() -> i64 {
    unsafe { printf("%lld\n", report(1)) }
    unsafe { printf("%lld\n", report(0)) }
    0
}
"#;

// `-> (a: i64, b: i64)` and the binding that takes it apart. There is no tuple
// type:
// each distinct list of types becomes one struct, the `return` becomes a struct
// literal, and the binding becomes a temporary and a field read per name. Two
// functions returning the same list share the struct, which is what
// `divide` and `split_bytes` check here.
const MULTIPLE_RETURN_VALUES: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }

divide :: fn(a: i64, b: i64) -> (quotient: i64, remainder: i64) {
    return a / b, a % b
}

// The same values under the same names, so both functions share one struct.
halve :: fn(a: i64) -> (quotient: i64, remainder: i64) {
    return a / 2, a % 2
}

// Named values can be returned by name rather than in order, which reads at
// the `return` the way the signature reads at the definition.
split_bytes :: fn(value: i64) -> (high: i64, low: i64) {
    return { high = value / 256, low = value % 256 }
}

// A return from inside a nested block, and a value that is not an integer.
classify :: fn(value: i64) -> (size: i64, negative: bool) {
    if (value < 0) {
        return 0 - value, true
    }
    return value, false
}

// An aggregate is one of the values like any other.
corners :: fn(size: i64) -> (origin: Point, far: Point) {
    return Point { x = 0, y = 0 }, Point { x = size, y = size }
}

main :: fn() -> i64 {
    quotient, remainder := divide(17, 5)
    unsafe { printf("%lld\n", quotient) }
    unsafe { printf("%lld\n", remainder) }

    high, low := split_bytes(700)
    unsafe { printf("%lld\n", high) }
    unsafe { printf("%lld\n", low) }

    half, odd := halve(9)
    unsafe { printf("%lld\n", half * 10 + odd) }

    magnitude, var negative := classify(-9)
    unsafe { printf("%lld\n", magnitude) }
    if (negative) { unsafe { printf("%lld\n", 1) } } else { unsafe { printf("%lld\n", 0) } }
    negative = false
    if (negative) { unsafe { printf("%lld\n", 1) } } else { unsafe { printf("%lld\n", 0) } }

    origin, far := corners(12)
    unsafe { printf("%lld\n", origin.x + far.x) }
    unsafe { printf("%lld\n", far.y) }

    // The values feed straight into another call.
    a, b := divide(9, 4)
    c, d := divide(a, b)
    unsafe { printf("%lld\n", c * 10 + d) }
    0
}
"#;

#[test]
fn a_function_returns_several_values() {
    let Some(output) =
        compile_and_run_unaudited("multiret", MULTIPLE_RETURN_VALUES)
    else {
        return;
    };
    assert_eq!(output, "3\n2\n2\n188\n41\n9\n1\n0\n12\n12\n20\n");
}

// The things a return type list does not do, each with the diagnostic that says
// so. They are compile errors rather than surprises at run time.
#[test]
fn a_return_type_list_is_held_to_its_shape() {
    let cases = [
        ("return a / b\n", "so its `return` lists them"),
        (
            "return a, b, a\n",
            "lists 3 values and the function returns 2",
        ),
    ];
    for (body, expected) in cases {
        let source = format!(
            "divide :: fn(a: i64, b: i64) -> (quotient: i64, remainder: i64) {{ {body} }}\nmain :: fn() -> i64 {{ 0 }}\n"
        );
        let message = compile_error("multiretbad", &source);
        assert!(
            message.contains(expected),
            "expected {expected:?} in:\n{message}"
        );
    }

    let bound_wrong = "divide :: fn(a: i64, b: i64) -> (q: i64, r: i64) { return a / b, a % b }\n\
         main :: fn() -> i64 {\n\
         \x20   only := divide(7, 2)\n\
         \x20   0\n\
         }\n";
    let message = compile_error("multiretbind", bound_wrong);
    assert!(
        message.contains("bound by a list of names"),
        "expected the binding diagnostic in:\n{message}"
    );

    // A list names every value, so a `return` by name can write every field,
    // and no name is used twice. Leaving one out and leaving all of them out
    // are the same fault, since the field a `return` would write is the
    // compiler's own `value0` either way.
    for unnamed in [
        "divide :: fn(a: i64) -> (quotient: i64, i64) { return a, a }\n\
         main :: fn() -> i64 { 0 }\n",
        "divide :: fn(a: i64) -> (i64, i64) { return a, a }\n\
         main :: fn() -> i64 { 0 }\n",
    ] {
        let message = compile_error("multiretunnamed", unnamed);
        assert!(
            message.contains("names every value"),
            "expected the every-value diagnostic in:\n{message}"
        );
    }

    let twice = "divide :: fn(a: i64) -> (n: i64, n: i64) { return a, a }\n\
         main :: fn() -> i64 { 0 }\n";
    let message = compile_error("multirettwice", twice);
    assert!(
        message.contains("names 'n' twice"),
        "expected the duplicate-name diagnostic in:\n{message}"
    );
}

// `.Circle { radius = 5 }` where the type is already known, the construction
// counterpart of the `case .Circle` a pattern writes. The enum comes from what
// the context expects: an annotation, a call's parameter, a struct field, a
// return, an assignment, or an element of an array.
const INFERRED_VARIANTS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Shape :: enum { Circle { radius: i64 }, Square { side: i64 } }
Color :: enum { Red, Green, Blue }
Theme :: struct { primary: Color, accent: Color }
Fault :: enum { Missing, Denied }

area :: fn(s: Shape) -> i64 {
    match s {
        case .Circle { radius }: radius * radius * 3
        case .Square { side }: side * side
    }
}

paint :: fn(c: Color) -> i64 {
    match c {
        case .Red: 1
        case .Green: 2
        case .Blue: 3
    }
}

// The return type is what the dot takes its enum from.
round :: fn(r: i64) -> Shape {
    return .Circle { radius = r }
}

// A failure set: the dot names a variant of the error, and the compiler tells
// it apart from the value the function answers with.
pick :: fn(want: i64) -> i64 ! Fault {
    if (want == 0) {
        return .Missing
    }
    want * 2
}

main :: fn() -> i64 {
    s : Shape = .Circle { radius = 4 }
    unsafe { printf("%lld\n", area(s)) }

    // A call's parameter, including one whose function is written later.
    unsafe { printf("%lld\n", area(.Square { side = 5 })) }
    unsafe { printf("%lld\n", paint(.Green)) }
    unsafe { printf("%lld\n", later(.Blue)) }
    unsafe { printf("%lld\n", area(round(2))) }

    // A struct field.
    t := Theme { primary = .Red, accent = .Blue }
    unsafe { printf("%lld\n", paint(t.primary)) }
    unsafe { printf("%lld\n", paint(t.accent)) }

    // An assignment to a place whose type is known.
    var c : Color = .Red
    c = .Blue
    unsafe { printf("%lld\n", paint(c)) }

    // An element of an array, whose type the annotation gives.
    var wheel : [3]Color = [.Red, .Green, .Blue]
    var sum : i64 = 0
    for held in wheel {
        sum = sum + paint(held)
    }
    unsafe { printf("%lld\n", sum) }

    good := match pick(5) {
        case .Ok { value }: value
        case .Err { error }: 0
    }
    unsafe { printf("%lld\n", good) }
    bad := match pick(0) {
        case .Ok { value }: value
        case .Err { error }: -1
    }
    unsafe { printf("%lld\n", bad) }
    0
}

later :: fn(c: Color) -> i64 { paint(c) * 10 }
"#;

#[test]
fn a_variant_takes_its_enum_from_the_context() {
    let Some(output) =
        compile_and_run_unaudited("dotvariant", INFERRED_VARIANTS)
    else {
        return;
    };
    assert_eq!(output, "48\n25\n2\n30\n12\n1\n3\n3\n6\n10\n-1\n");
}

// A dot with nothing to take its enum from says so, rather than failing later
// as a nameless enum.
#[test]
fn a_variant_without_a_context_is_rejected() {
    let source = "Color :: enum { Red, Green }\n\
         main :: fn() -> i64 {\n\
         \x20   c := .Red\n\
         \x20   0\n\
         }\n";
    let message = compile_error("dotvariantbad", source);
    assert!(
        message.contains("takes its enum from what the context expects"),
        "expected the inference diagnostic in:\n{message}"
    );
}

// `Meters :: distinct i64` is a nominal type: the representation of the inner
// type under a name of its own. Arithmetic and layout follow the inner type,
// its identity does not, and a literal takes the type the context wants, which
// is what makes `m : Meters = 3` read the way it should.
const DISTINCT_TYPES: &str = r#"import "io.frost"

printf :: extern fn(fmt: ^i8, value: i64) -> i32

Meters :: distinct i64
Feet :: distinct i64
// The representation decides the arithmetic, so a distinct float subtracts as
// a float rather than lowering to an integer subtract on float registers.
Seconds :: distinct f64

add_meters :: fn(a: Meters, b: Meters) -> Meters { a + b }
elapsed :: fn(from: Seconds, to: Seconds) -> Seconds { to - from }

// A distinct type crosses a signature and lands in a struct like any other.
Trip :: struct { there: Meters, back: Meters }
Leg :: struct { distance: Meters, took: Seconds }

total :: fn(t: Trip) -> Meters { add_meters(t.there, t.back) }

main :: fn() -> i64 {
    a : Meters = 3
    b : Meters = 4
    unsafe { printf("%lld\n", add_meters(a, b)) }

    f : Feet = 10
    unsafe { printf("%lld\n", f) }

    trip := Trip { there = a, back = b }
    unsafe { printf("%lld\n", total(trip)) }

    leg := Leg { distance = 12, took = 1.5 }
    print("{}\n", elapsed(leg.took, 4.25))
    unsafe { printf("%lld\n", leg.distance) }

    var marks : [3]Meters = [1, 2, 3]
    unsafe { printf("%lld\n", marks[2]) }
    0
}
"#;

#[test]
fn a_distinct_type_carries_its_own_name() {
    let Some(output) = compile_and_run_unaudited("distinct", DISTINCT_TYPES)
    else {
        return;
    };
    assert_eq!(output, "7\n10\n7\n2.75\n12\n3\n");
}

// The identity is the point, so the places a value crosses are the places the
// mismatch is caught.
#[test]
fn a_distinct_type_is_not_its_representation() {
    let prelude = "Meters :: distinct i64\n\
         Feet :: distinct i64\n\
         add_meters :: fn(a: Meters, b: Meters) -> Meters { a + b }\n";
    let cases = [
        // Another distinct type over the same representation, at a call.
        "main :: fn() -> i64 {\n\
         \x20   m : Meters = 3\n\
         \x20   f : Feet = 4\n\
         \x20   add_meters(m, f)\n\
         \x20   0\n\
         }\n",
        // The representation itself, where the distinct type is wanted.
        "main :: fn() -> i64 {\n\
         \x20   n : i64 = 3\n\
         \x20   m : Meters = n\n\
         \x20   0\n\
         }\n",
        // And at an assignment.
        "main :: fn() -> i64 {\n\
         \x20   var m : Meters = 3\n\
         \x20   f : Feet = 4\n\
         \x20   m = f\n\
         \x20   0\n\
         }\n",
        // Arithmetic on the representation is still the representation. Only a
        // run of literals is exempt, so two i64 locals added together do not
        // become a Meters by being added.
        "main :: fn() -> i64 {\n\
         \x20   n : i64 = 3\n\
         \x20   o : i64 = 4\n\
         \x20   add_meters(n + o, 1)\n\
         \x20   0\n\
         }\n",
    ];
    for (index, body) in cases.iter().enumerate() {
        let source = format!("{prelude}{body}");
        let message = compile_error(&format!("distinctbad{index}"), &source);
        assert!(
            message.contains("a distinct type is not its representation"),
            "expected the nominal diagnostic in:\n{message}"
        );
        // The identity is a rule of the language, so the self-hosted compiler
        // refuses the same three programs for the same reason.
        let Some(compiler) = build_self_hosted_compiler("distinctbad") else {
            continue;
        };
        let directory = std::env::temp_dir();
        let input = directory.join(format!("frost_distinctbad{index}.frost"));
        std::fs::write(&input, &source).unwrap();
        let refused = Command::new(&compiler)
            .env("FROST_INPUT", &input)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            !refused.status.success(),
            "the self-hosted compiler accepted case {index}"
        );
        let said = String::from_utf8_lossy(&refused.stderr);
        assert!(
            said.contains("a distinct type is not its representation"),
            "expected the nominal diagnostic from the self-hosted compiler in:\n{said}"
        );
    }
}

// The same program through the self-hosted compiler. Both compilers give a
// distinct type a code of its own, carrying the representation it computes as
// beside the identity that separates it from every other type.
const SELF_HOSTED_DISTINCT: &str = "import \"io.frost\"\nMeters :: distinct i64\n\
     Seconds :: distinct f64\n\
     add_meters :: fn(a: Meters, b: Meters) -> Meters {\n\
     \x20   a + b\n\
     }\n\
     elapsed :: fn(from: Seconds, to: Seconds) -> Seconds {\n\
     \x20   to - from\n\
     }\n\
     Leg :: struct { distance: Meters, took: Seconds }\n\
     main :: fn() -> i64 {\n\
     \x20   a : Meters = 3\n\
     \x20   b : Meters = 4\n\
     \x20   print(\"{}\\n\", add_meters(a, b))\n\
     \x20   leg := Leg { distance = 12, took = 1.5 }\n\
     \x20   print(\"{}\\n\", elapsed(leg.took, 4.25))\n\
     \x20   print(\"{}\\n\", leg.distance)\n\
     \x20   0\n\
     }\n";

#[test]
fn self_hosted_compiles_a_distinct_type() {
    let Some(output) =
        selfhosted_unaudited_output("shdistinct", SELF_HOSTED_DISTINCT)
    else {
        return;
    };
    assert_eq!(output, "7\n2.75\n12\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shdistinct_input.frost");
    std::fs::write(&input, SELF_HOSTED_DISTINCT).unwrap();
    let Some(c_source) = self_hosted_emits("shdistinct", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shdistinct", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// Arithmetic on a distinct type answers with that type, so a combination goes
// back where either operand came from with nothing written down to say what it
// is. The bitwise operators are the ones a bit set is combined with, and they
// were the ones the self-hosted compiler dropped the type on, because its
// operator codes put them above the comparisons and it read the boundary off
// the numbers.
const DISTINCT_ARITHMETIC: &str = "import \"io.frost\"\nMask :: distinct u32\n\
     Count :: distinct i64\n\
     bits :: fn(m: Mask) -> u32 { m }\n\
     total :: fn(c: Count) -> i64 { c }\n\
     main :: fn() -> i64 {\n\
     \x20   a : Mask = 16\n\
     \x20   b : Mask = 32\n\
     \x20   print(\"{}\\n\", bits(a | b))\n\
     \x20   print(\"{}\\n\", bits(a & a))\n\
     \x20   print(\"{}\\n\", bits(a + b))\n\
     \x20   n : Count = 3\n\
     \x20   print(\"{}\\n\", total(n << 2))\n\
     \x20   print(\"{}\\n\", total(n >> 1))\n\
     \x20   print(\"{}\\n\", total(n * n))\n\
     \x20   both : Mask = a | b\n\
     \x20   print(\"{}\\n\", bits(both))\n\
     \x20   if (a < b) {\n\
     \x20       print(\"{}\\n\", 1)\n\
     \x20   }\n\
     \x20   0\n\
     }\n";

#[test]
fn arithmetic_on_a_distinct_type_answers_with_it() {
    let Some(output) =
        compile_and_run_unaudited("distinctops", DISTINCT_ARITHMETIC)
    else {
        return;
    };
    assert_eq!(output, "48\n16\n48\n12\n1\n9\n48\n1\n");
}

#[test]
fn self_hosted_arithmetic_on_a_distinct_type_answers_with_it() {
    let Some(output) =
        selfhosted_unaudited_output("shdistinctops", DISTINCT_ARITHMETIC)
    else {
        return;
    };
    assert_eq!(output, "48\n16\n48\n12\n1\n9\n48\n1\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shdistinctops_input.frost");
    std::fs::write(&input, DISTINCT_ARITHMETIC).unwrap();
    let Some(c_source) = self_hosted_emits("shdistinctops", &input, None)
    else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shdistinctops", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// A named set of bits with a type of its own. The numbers are a C header's, the
// names are the type's, and a combination of two of them is still that type, so
// it goes into a call with nothing written down to say what it is.
const FLAGS: &str = "import \"io.frost\"\nInitFlags :: flags u32 {
         Audio   = 16,
         Video   = 32,
         Events  = 16384,
     }
     WindowFlags :: flags u64 {
         Fullscreen = 1,
         Resizable  = 32,
     }
     started :: fn(f: InitFlags) -> u32 { f }
     opened :: fn(f: WindowFlags) -> u64 { f }
     main :: fn() -> i64 {
         print(\"{}\\n\", started(InitFlags::Video))
         print(\"{}\\n\", started(InitFlags::Video | InitFlags::Audio))
         chosen := InitFlags::Video | InitFlags::Events
         print(\"{}\\n\", started(chosen & InitFlags::Events))
         if (flags_has(chosen, InitFlags::Video)) { print(\"{}\\n\", 1) }
         if (flags_has(chosen, InitFlags::Audio) == false) { print(\"{}\\n\", 2) }
         if (chosen == InitFlags::Video | InitFlags::Events) { print(\"{}\\n\", 3) }
         if (chosen != InitFlags::Audio) { print(\"{}\\n\", 4) }
         print(\"{}\\n\", opened(WindowFlags::Resizable | WindowFlags::Fullscreen))
         0
     }
";

const FLAGS_OUTPUT: &str = "32
48
16384
1
2
3
4
33
";

#[test]
fn a_flags_type_names_its_bits() {
    let Some(output) = compile_and_run_unaudited("flagsbits", FLAGS) else {
        return;
    };
    assert_eq!(output, FLAGS_OUTPUT);
}

#[test]
fn self_hosted_compiles_a_flags_type() {
    let Some(output) = selfhosted_unaudited_output("shflagsbits", FLAGS) else {
        return;
    };
    assert_eq!(output, FLAGS_OUTPUT);

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shflagsbits_input.frost");
    std::fs::write(&input, FLAGS).unwrap();
    let Some(c_source) = self_hosted_emits("shflagsbits", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shflagsbits", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// `flags` is a word rather than a keyword, so a program that uses it as a name
// still compiles. `window_create` takes a parameter called `flags`.
#[test]
fn flags_is_still_a_name() {
    let source = "import \"io.frost\"\nMask :: flags u32 { One = 1, Two = 2 }
         take :: fn(flags: Mask) -> u32 { flags }
         Holder :: struct { flags: i64 }
         main :: fn() -> i64 {
             flags := Holder { flags = 7 }
             print(\"{}\\n\", flags.flags)
             print(\"{}\\n\", take(Mask::One | Mask::Two))
             0
         }
";
    let Some(output) = compile_and_run_unaudited("flagsname", source) else {
        return;
    };
    assert_eq!(
        output,
        "7
3
"
    );
}

// What the declaration is for is what it refuses. Both compilers refuse the
// same programs, because the rules are the language's rather than one
// compiler's.
#[test]
fn a_flags_type_refuses_what_is_not_one_of_its_bits() {
    let prelude = "InitFlags :: flags u32 { Audio = 16, Video = 32 }
         WindowFlags :: flags u64 { Resizable = 32 }
         started :: fn(f: InitFlags) -> u32 { f }
";
    let cases = [
        // Another flags type.
        ("import \"io.frost\"\nmain :: fn() -> i64 { print(\"{}\\n\", started(WindowFlags::Resizable))  0 }
",
         "built only from the names declared under it"),
        // A number, which a distinct type would have taken from the context.
        ("import \"io.frost\"\nmain :: fn() -> i64 { print(\"{}\\n\", started(48))  0 }
",
         "a number is not one of them"),
        ("import \"io.frost\"\nmain :: fn() -> i64 { f : InitFlags = 5  print(\"{}\\n\", started(f))  0 }
",
         "a number is not one of them"),
        // The representation.
        ("import \"io.frost\"\nmain :: fn() -> i64 { n : u32 = 48  print(\"{}\\n\", started(n))  0 }
",
         "built only from the names declared under it"),
        // Operators a set of bits does not answer.
        ("import \"io.frost\"\nmain :: fn() -> i64 { print(\"{}\\n\", started(InitFlags::Video + InitFlags::Audio))  0 }
",
         "is not something two sets answer"),
        ("import \"io.frost\"\nmain :: fn() -> i64 { print(\"{}\\n\", started(InitFlags::Video << 1))  0 }
",
         "is not something two sets answer"),
        ("import \"io.frost\"\nmain :: fn() -> i64 { if (InitFlags::Video < InitFlags::Audio) { print(\"{}\\n\", 1) }  0 }
",
         "is not something two sets answer"),
        // Two different sets combined.
        ("import \"io.frost\"\nmain :: fn() -> i64 { print(\"{}\\n\", started(InitFlags::Video | WindowFlags::Resizable))  0 }
",
         "combines only with itself"),
        // A bit the type does not name.
        ("import \"io.frost\"\nmain :: fn() -> i64 { print(\"{}\\n\", started(InitFlags::Gamepad))  0 }
",
         "no bit called"),
        // And the same through flags_has.
        ("import \"io.frost\"\nmain :: fn() -> i64 { if (flags_has(InitFlags::Video, 32)) { print(\"{}\\n\", 1) }  0 }
",
         "a number is not one of them"),
        // And a number written straight into a combination.
        ("import \"io.frost\"\nmain :: fn() -> i64 { print(\"{}\\n\", started(InitFlags::Video | 4))  0 }
",
         "a number is not one of them"),
    ];
    let compiler = build_self_hosted_compiler("flagsbad");
    for (index, (body, expected)) in cases.iter().enumerate() {
        let source = format!("{prelude}{body}");
        let message = compile_error(&format!("flagsbad{index}"), &source);
        assert!(
            message.contains(expected),
            "case {index} wanted '{expected}' in:
{message}"
        );
        let Some(compiler) = &compiler else {
            continue;
        };
        let directory = std::env::temp_dir();
        let input = directory.join(format!("frost_flagsbad{index}.frost"));
        std::fs::write(&input, &source).unwrap();
        let refused = Command::new(compiler)
            .env("FROST_INPUT", &input)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            !refused.status.success(),
            "the self-hosted compiler accepted case {index}"
        );
        let said = String::from_utf8_lossy(&refused.stderr);
        assert!(
            said.contains(expected),
            "case {index} wanted '{expected}' from the self-hosted compiler in:
{said}"
        );
    }
}

// Naming a borrowed aggregate binds a copy of the caller's value, not a second
// name for the caller's storage, so writing through the binding does not reach
// back. `ref` is how a second name is asked for, and a call answering with a
// `ref T` handed one out on purpose, so both keep the borrow they were given.
const BINDING_COPIES: &str =
    "import \"io.frost\"\nPair :: struct { a: i64, b: i64 }
     Held :: struct { items: [4]i64 }
     bump :: fn(p: Pair) -> Pair {
         var out := p
         out.a = 7
         out
     }
     first :: fn(h: Held) -> Held {
         var out := h
         out.items[0] = 9
         out
     }
     main :: fn() -> i64 {
         q := Pair { a = 1, b = 2 }
         r := bump(q)
         print(\"{}\\n\", r.a)
         print(\"{}\\n\", q.a)
         h := Held { items = [0, 0, 0, 0] }
         k := first(h)
         print(\"{}\\n\", k.items[0])
         print(\"{}\\n\", h.items[0])
         0
     }
";

#[test]
fn naming_a_borrowed_aggregate_binds_a_copy() {
    let Some(output) = compile_and_run_unaudited("bindcopy", BINDING_COPIES)
    else {
        return;
    };
    assert_eq!(
        output,
        "7
1
9
0
"
    );
}

#[test]
fn self_hosted_naming_a_borrowed_aggregate_binds_a_copy() {
    let Some(output) =
        selfhosted_unaudited_output("shbindcopy", BINDING_COPIES)
    else {
        return;
    };
    assert_eq!(
        output,
        "7
1
9
0
"
    );

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shbindcopy_input.frost");
    std::fs::write(&input, BINDING_COPIES).unwrap();
    let Some(c_source) = self_hosted_emits("shbindcopy", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shbindcopy", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// A frame wider than the page the native backend used to give every function.
// Two kilobytes of locals in one function used to run past the frame and write
// over whatever was below it, silently: the answer came back wrong rather than
// the program crashing. The frame is now sized per function, so this is an
// ordinary program.
const WIDE_FRAME: &str =
    "import \"io.frost\"\nTable :: struct { rows: [900]i64 }
     fill :: fn(seed: i64) -> Table {
         var t := Table { rows = [0; 900] }
         var i : i64 = 0
         while (i < 900) {
             t.rows[i] = seed + i
             i = i + 1
         }
         t
     }
     main :: fn() -> i64 {
         var total : i64 = 0
         held := fill(1)
         other := fill(1000)
         var i : i64 = 0
         while (i < 900) {
             total = total + held.rows[i] + other.rows[i]
             i = i + 1
         }
         print(\"{}\\n\", total)
         print(\"{}\\n\", held.rows[899])
         print(\"{}\\n\", other.rows[0])
         0
     }
";

#[test]
fn a_function_wider_than_a_page_of_frame() {
    let Some(output) = compile_and_run_unaudited("wideframe", WIDE_FRAME)
    else {
        return;
    };
    assert_eq!(
        output,
        "1710000
900
1000
"
    );
}

#[test]
fn self_hosted_a_function_wider_than_a_page_of_frame() {
    let Some(output) = selfhosted_unaudited_output("shwideframe", WIDE_FRAME)
    else {
        return;
    };
    assert_eq!(
        output,
        "1710000
900
1000
"
    );
}

// A compile-time list holds types as well as values, is handed on to another
// list by naming it, and expands into a call's argument list once per element.
// The three together are what lets one function serve a query over any number
// of components: the list decides the arity, so there is no `for_each3`.
const PACK_FEATURES: &str =
    "import \"io.frost\"\nwidths :: fn($body: fn(i64, i64), types: $...) {
         body(sizeof(T) for T in types)
     }
     Big :: struct { a: i64, b: i64, c: i64 }
     show2 :: fn(a: i64, b: i64) {
         print(\"{}\\n\", a)
         print(\"{}\\n\", b)
     }
     doubled :: fn(v: i64) -> i64 { v * 2 }
     show3 :: fn(a: i64, b: i64, c: i64) {
         print(\"{}\\n\", a)
         print(\"{}\\n\", b)
         print(\"{}\\n\", c)
     }
     apply :: fn($body: fn(i64, i64, i64), values: $...) {
         body(doubled(v) for v in values)
     }
     total :: fn(values: $...) -> i64 {
         var sum : i64 = 0
         for v in values {
             sum = sum + v
         }
         sum
     }
     passed_on :: fn(values: $...) -> i64 {
         total(values)
     }
     main :: fn() -> i64 {
         widths($show2, $i64, $Big)
         apply($show3, 1, 2, 3)
         print(\"{}\\n\", passed_on(4, 5, 6))
         0
     }
";

const PACK_OUTPUT: &str = "8
24
2
4
6
15
";

// A number per type, the same wherever the type is written and different for
// every other type, so a table decided at run time can be keyed by one.
const TYPE_IDS: &str = "import \"io.frost\"\nA :: struct { x: i64 }
     B :: struct { y: i64 }
     ident :: fn($T: Type) -> i64 { type_id(T) }
     main :: fn() -> i64 {
         print(\"{}\\n\", type_id($A) == type_id($A))
         print(\"{}\\n\", type_id($A) == type_id($B))
         print(\"{}\\n\", ident($A) == type_id($A))
         print(\"{}\\n\", ident($B) == type_id($A))
         print(\"{}\\n\", type_id($i64) == type_id($u8))
         0
     }
";

#[test]
fn a_type_has_a_number_of_its_own() {
    let Some(output) = compile_and_run_unaudited("typeids", TYPE_IDS) else {
        return;
    };
    assert_eq!(
        output,
        "1
0
1
0
0
"
    );
}

#[test]
fn self_hosted_a_type_has_a_number_of_its_own() {
    let Some(output) = selfhosted_unaudited_output("shtypeids", TYPE_IDS)
    else {
        return;
    };
    assert_eq!(
        output,
        "1
0
1
0
0
"
    );
}

#[test]
fn self_hosted_a_compile_time_list_holds_types_and_expands_into_a_call() {
    let Some(output) = selfhosted_unaudited_output("shpackfeat", PACK_FEATURES)
    else {
        return;
    };
    assert_eq!(output, PACK_OUTPUT);

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shpackfeat_input.frost");
    std::fs::write(&input, PACK_FEATURES).unwrap();
    let Some(c_source) = self_hosted_emits("shpackfeat", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shpackfeat", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

#[test]
fn a_compile_time_list_holds_types_and_expands_into_a_call() {
    let Some(output) = compile_and_run_unaudited("packfeat", PACK_FEATURES)
    else {
        return;
    };
    assert_eq!(output, PACK_OUTPUT);
}

// `break` leaves the innermost loop and `continue` goes round it again. The
// self-hosted compiler had neither, so a loop written with one compiled under
// the bootstrap and not under it.
const LOOP_CONTROL: &str = "import \"io.frost\"\nmain :: fn() -> i64 {
         var sum : i64 = 0
         var i : i64 = 0
         while (i < 10) {
             i = i + 1
             if (i == 3) { continue }
             if (i == 7) { break }
             sum = sum + i
         }
         print(\"{}\\n\", sum)
         var outer : i64 = 0
         var total : i64 = 0
         while (outer < 3) {
             outer = outer + 1
             var inner : i64 = 0
             while (inner < 5) {
                 inner = inner + 1
                 if (inner == 2) { continue }
                 if (inner == 4) { break }
                 total = total + 1
             }
         }
         print(\"{}\\n\", total)
         0
     }
";

#[test]
fn break_and_continue_answer_to_the_innermost_loop() {
    let Some(output) = compile_and_run_unaudited("loopctl", LOOP_CONTROL)
    else {
        return;
    };
    assert_eq!(
        output,
        "18
6
"
    );
}

#[test]
fn self_hosted_break_and_continue_answer_to_the_innermost_loop() {
    let Some(output) = selfhosted_unaudited_output("shloopctl", LOOP_CONTROL)
    else {
        return;
    };
    assert_eq!(
        output,
        "18
6
"
    );

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shloopctl_input.frost");
    std::fs::write(&input, LOOP_CONTROL).unwrap();
    let Some(c_source) = self_hosted_emits("shloopctl", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shloopctl", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// The gate is the complete list of places to look when memory is corrupted, so
// a shape that slips past it is worth a test of its own. A generic's fields are
// declared once on the template, and the gate looked them up under the instance
// name, so `b.data[i]` where `data: ^T` was not seen as a raw pointer at all.
// Every container in the standard library is a generic holding a pointer, so
// the hole was as wide as the library.
#[test]
fn indexing_a_raw_pointer_held_by_a_generic_is_gated() {
    let source = "Box :: struct($T: Type) { data: ^T, count: i64 }
         put :: fn(mut b: Box<$T>, value: $T) {
             b.data[b.count] = value
         }
         main :: fn() -> i64 {
             var b := Box<i64> { data = unsafe { ptr_cast($i64, 0) },
                 count = 0 }
             put(b, 7)
             0
         }
";
    let message = compile_error_checked("gategeneric", source);
    assert!(
        message.contains("indexing a raw pointer"),
        "a raw pointer reached through a generic's field escaped the gate:
{message}"
    );
}

// A function's trailing expression is its return value, so a call in that
// position hands its answer to the caller. A call and a struct literal are both
// valid there, and a linear answer is consumed either way.
const LINEAR_TRAILING_CALL: &str =
    "import \"io.frost\"\nHolder :: linear struct { count: i64 }

     make_holder :: fn(count: i64) -> Holder {
         Holder { count = count }
     }

     forward :: fn(count: i64) -> Holder {
         make_holder(count * 2)
     }

     literal :: fn(count: i64) -> Holder {
         Holder { count = count }
     }

     release :: fn(move held: Holder) -> i64 {
         held.count
     }

     main :: fn() -> i64 {
         through_call := forward(21)
         print(\"{}\\n\", release(through_call))
         through_literal := literal(7)
         print(\"{}\\n\", release(through_literal))
         0
     }
";

#[test]
fn a_trailing_call_answering_with_a_linear_value_is_the_return() {
    let Some(output) =
        compile_and_run_unaudited("lintrail", LINEAR_TRAILING_CALL)
    else {
        return;
    };
    assert_eq!(
        output,
        "42
7
"
    );
}

#[test]
fn self_hosted_a_trailing_call_answering_with_a_linear_value_is_the_return() {
    let Some(output) =
        selfhosted_unaudited_output("shlintrail", LINEAR_TRAILING_CALL)
    else {
        return;
    };
    assert_eq!(
        output,
        "42
7
"
    );

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shlintrail_input.frost");
    std::fs::write(&input, LINEAR_TRAILING_CALL).unwrap();
    let Some(c_source) = self_hosted_emits("shlintrail", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shlintrail", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// A call answering with a resource, handed to a `move` parameter from inside
// another call's argument list. The value is consumed however deep the call
// holding it sits, so nesting does not change who owes the consumption.
const LINEAR_NESTED_ARGUMENT: &str =
    "import \"io.frost\"\nHolder :: linear struct { count: i64 }

     make_holder :: fn(count: i64) -> Holder {
         Holder { count = count }
     }

     release :: fn(move held: Holder) -> i64 {
         held.count
     }

     twice :: fn(value: i64) -> i64 {
         value * 2
     }

     main :: fn() -> i64 {
         print(\"{}\\n\", twice(release(make_holder(21))))
         print(\"{}\\n\", release(make_holder(3)))
         0
     }
";

#[test]
fn a_linear_value_is_consumed_inside_a_nested_call() {
    let Some(output) =
        compile_and_run_unaudited("linnest", LINEAR_NESTED_ARGUMENT)
    else {
        return;
    };
    assert_eq!(
        output,
        "42
3
"
    );
}

#[test]
fn self_hosted_a_linear_value_is_consumed_inside_a_nested_call() {
    let Some(output) =
        selfhosted_unaudited_output("shlinnest", LINEAR_NESTED_ARGUMENT)
    else {
        return;
    };
    assert_eq!(
        output,
        "42
3
"
    );

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shlinnest_input.frost");
    std::fs::write(&input, LINEAR_NESTED_ARGUMENT).unwrap();
    let Some(c_source) = self_hosted_emits("shlinnest", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shlinnest", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// A call whose answer holds a resource, made as a statement, drops it. Nothing
// consumes what `make_holder` answers with here.
const LINEAR_DROPPED_CALL: &str = "Holder :: linear struct { count: i64 }

     make_holder :: fn(count: i64) -> Holder {
         Holder { count = count }
     }

     leaks :: fn() -> i64 {
         make_holder(1)
         0
     }

     main :: fn() -> i64 {
         leaks()
     }
";

#[test]
fn a_linear_value_a_statement_drops_is_still_refused() {
    let message = compile_error("lindrop", LINEAR_DROPPED_CALL);
    assert!(
        message.contains("never consumed"),
        "the bootstrap let a dropped linear value through:
{message}"
    );
}

// A constant whose value is text, which is how a binding names a string the
// program uses in several places.
const STRING_CONSTANT: &str = "import \"io.frost\"\nGREETING :: \"hello\"
     PROP :: \"SDL.window.win32.hwnd\"
     shout :: fn(text: str) -> i64 { str_len(text) }
     main :: fn() -> i64 {
         print(\"{}\\n\", GREETING)
         print(\"{}\\n\", shout(GREETING))
         print(\"{}\\n\", shout(PROP))
         0
     }
";

#[test]
fn a_constant_can_be_text() {
    let Some(output) = compile_and_run_unaudited("strconst", STRING_CONSTANT)
    else {
        return;
    };
    assert_eq!(
        output,
        "hello
5
21
"
    );
}

#[test]
fn self_hosted_a_constant_can_be_text() {
    let Some(output) =
        selfhosted_unaudited_output("shstrconst", STRING_CONSTANT)
    else {
        return;
    };
    assert_eq!(
        output,
        "hello
5
21
"
    );

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shstrconst_input.frost");
    std::fs::write(&input, STRING_CONSTANT).unwrap();
    let Some(c_source) = self_hosted_emits("shstrconst", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shstrconst", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// A call through a function pointer that answers with a struct. A Frost
// function hands an aggregate back through a trailing out-pointer, and a call
// through a pointer is the same call, so the signature the call site builds is
// the one the callee was compiled with.
const INDIRECT_AGGREGATE_RETURN: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }

origin :: fn(scale: i64) -> Point {
    Point { x = scale, y = scale * 2 }
}

shifted :: fn(scale: i64) -> Point {
    Point { x = scale + 100, y = scale }
}

// The pointer as a parameter, which is how a caller varies what a loop calls.
apply :: fn(make: fn(i64) -> Point, scale: i64) -> i64 {
    p := make(scale)
    p.x * 1000 + p.y
}

main :: fn() -> i64 {
    unsafe { printf("%lld\n", apply(origin, 3)) }
    unsafe { printf("%lld\n", apply(shifted, 3)) }

    // And as a local that is reassigned, so the callee is not known at the
    // call site at all.
    var chosen : fn(i64) -> Point = origin
    q := chosen(5)
    unsafe { printf("%lld\n", q.x + q.y) }
    chosen = shifted
    r := chosen(5)
    unsafe { printf("%lld\n", r.x + r.y) }
    0
}
"#;

#[test]
fn an_indirect_call_returns_an_aggregate() {
    let Some(output) =
        compile_and_run_unaudited("indagg", INDIRECT_AGGREGATE_RETURN)
    else {
        return;
    };
    assert_eq!(output, "3006\n103003\n15\n110\n");
}

// `{ x = 1, y = 2 }` where the type is already stated, the struct counterpart of
// the leading-dot variant. Every field is still named: there is no positional
// literal, here or anywhere else, since a field's name is what says where the
// value lands.
const INFERRED_LITERALS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }
Line :: struct { from: Point, to: Point }
Color :: enum { Red, Green, Blue }
Marked :: struct { at: Point, colour: Color }

sum :: fn(p: Point) -> i64 { p.x + p.y }

length_sq :: fn(l: Line) -> i64 {
    dx := l.to.x - l.from.x
    dy := l.to.y - l.from.y
    dx * dx + dy * dy
}

origin :: fn() -> Point { return { x = 0, y = 0 } }

paint :: fn(m: Marked) -> i64 {
    base := match m.colour {
        case .Red: 1
        case .Green: 2
        case .Blue: 3
    }
    base * 100 + m.at.x
}

main :: fn() -> i64 {
    p : Point = { x = 3, y = 4 }
    unsafe { printf("%lld\n", sum(p)) }

    // A parameter, including one whose function is written later.
    unsafe { printf("%lld\n", sum({ x = 10, y = 20 })) }
    unsafe { printf("%lld\n", later({ x = 2, y = 3 })) }

    // Nested, each inner literal taking its type from the field it fills.
    unsafe { printf("%lld\n", length_sq({ from = { x = 0, y = 0 }, to = { x = 3, y = 4 } })) }

    // A return.
    unsafe { printf("%lld\n", sum(origin())) }

    // A variant inside an inferred literal, taking its enum from the field.
    unsafe { printf("%lld\n", paint({ at = { x = 7, y = 0 }, colour = .Green })) }

    // An assignment to a place whose type is known.
    var q : Point = { x = 1, y = 1 }
    q = { x = 5, y = 6 }
    unsafe { printf("%lld\n", sum(q)) }

    // Elements of an array, from the annotation's element type.
    grid : [2]Point = [{ x = 1, y = 2 }, { x = 3, y = 4 }]
    var total : i64 = 0
    for held in grid {
        total = total + sum(held)
    }
    unsafe { printf("%lld\n", total) }
    0
}

later :: fn(p: Point) -> i64 { sum(p) * 10 }
"#;

#[test]
fn a_literal_takes_its_type_from_the_context() {
    let Some(output) = compile_and_run_unaudited("inflit", INFERRED_LITERALS)
    else {
        return;
    };
    assert_eq!(output, "7\n30\n50\n25\n0\n207\n11\n10\n");
}

// A literal with nothing to take its type from says so.
#[test]
fn a_literal_without_a_context_is_rejected() {
    let source = "Point :: struct { x: i64, y: i64 }\n\
         main :: fn() -> i64 {\n\
         \x20   p := { x = 1, y = 2 }\n\
         \x20   0\n\
         }\n";
    let message = compile_error("inflitbad", source);
    assert!(
        message.contains("takes its type from what the context expects"),
        "expected the inference diagnostic in:\n{message}"
    );
}

// `for item in items` over a slice, an array and a `str`. It is the
// index-and-bound loop written out rather than an iterator: nothing is called
// per element, and what the backend sees is what the same loop written by hand
// produces. The manual triple appeared dozens of times across std/ before this.
const FOR_OVER_A_SEQUENCE: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Point :: struct { x: i64, y: i64 }

sum_slice :: fn(xs: []i64) -> i64 {
    var total : i64 = 0
    for value in xs {
        total = total + value
    }
    total
}

// The sequence is evaluated once, so a call in that position happens once
// however many elements it answers with.
counted :: fn(mut calls: i64) -> []i64 {
    calls = calls + 1
    unsafe { slice_from($i64, ptr_cast($i64, ptr_to(calls)), 1) }
}

main :: fn() -> i64 {
    var numbers : [4]i64 = [10, 20, 30, 40]
    var total : i64 = 0
    for value in numbers {
        total = total + value
    }
    unsafe { printf("%lld\n", total) }
    unsafe { printf("%lld\n", sum_slice(numbers)) }

    // The position as well as the element.
    var weighted : i64 = 0
    for index, value in numbers {
        weighted = weighted + index * value
    }
    unsafe { printf("%lld\n", weighted) }

    // An aggregate element binds as a borrow, so nothing is copied per step.
    var points : [3]Point = [
        Point { x = 1, y = 2 },
        Point { x = 3, y = 4 },
        Point { x = 5, y = 6 },
    ]
    var sum : i64 = 0
    for p in points {
        sum = sum + p.x * p.y
    }
    unsafe { printf("%lld\n", sum) }

    // A `str` yields its bytes.
    var bytes : i64 = 0
    for byte in "abc" {
        bytes = bytes + byte
    }
    unsafe { printf("%lld\n", bytes) }

    // `break` and `continue` reach the loop the same as in a range.
    var first : i64 = 0
    for value in numbers {
        if (value == 10) { continue }
        first = value
        break
    }
    unsafe { printf("%lld\n", first) }

    var empty : [0]i64 = []
    var never : i64 = 7
    for value in empty {
        never = value
    }
    unsafe { printf("%lld\n", never) }

    var calls : i64 = 0
    var seen : i64 = 0
    for value in counted(calls) {
        seen = seen + value
    }
    unsafe { printf("%lld\n", calls) }
    0
}
"#;

#[test]
fn a_for_walks_a_sequence() {
    let Some(output) = compile_and_run_unaudited("forseq", FOR_OVER_A_SEQUENCE)
    else {
        return;
    };
    assert_eq!(output, "100\n100\n200\n44\n294\n20\n7\n1\n");
}

#[test]
fn a_for_over_something_that_is_not_a_sequence_says_so() {
    let source = "import \"io.frost\"\nmain :: fn() -> i64 {\n\
                  \x20   n := 7\n\
                  \x20   for x in n { print(\"{}\\n\", x) }\n\
                  \x20   0\n\
                  }\n";
    let message = compile_error("fornotseq", source);
    assert!(
        message.contains("walks a range") && message.contains("i64"),
        "expected the type to be named, got:\n{message}"
    );
}

// Two enum values compare by their tags, which for an enum whose variants carry
// nothing is the whole value. Asking which variant something is used to need a
// `match` with a case per variant, which is a lot of lines to answer one
// question, and the node kinds in the self-hosted compiler are integer
// constants partly because of it.
const ENUM_EQUALITY: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Kind :: enum { Num, Var, Bin }
Node :: struct { kind: Kind, weight: i64 }

// Through a borrowed parameter, where the field's type is still a name the
// parser has not resolved to an enum.
is_var :: fn(node: Node) -> i64 {
    if (node.kind == Kind::Var) { return 1 }
    0
}

main :: fn() -> i64 {
    a := Kind::Var
    b := Kind::Var
    c := Kind::Bin
    unsafe { printf("%lld\n", a == b) }
    unsafe { printf("%lld\n", a == c) }
    unsafe { printf("%lld\n", a != c) }

    node := Node { kind = Kind::Var, weight = 7 }
    unsafe { printf("%lld\n", is_var(node)) }
    unsafe { printf("%lld\n", node.kind == Kind::Bin) }

    held := node.kind
    unsafe { printf("%lld\n", held == Kind::Var) }
    0
}
"#;

#[test]
fn enum_values_compare_by_variant() {
    let Some(output) = compile_and_run_unaudited("enumeq", ENUM_EQUALITY)
    else {
        return;
    };
    assert_eq!(output, "1\n0\n1\n1\n0\n1\n");
}

// A variant carrying fields makes the question ambiguous: two values can be the
// same variant and different values. Rather than pick one reading, it says so
// and points at `match`.
#[test]
fn an_enum_carrying_fields_is_not_compared_with_equals() {
    let source = "Shape :: enum { Round, Sized { at: i64 } }\n\
                  main :: fn() -> i64 {\n\
                  \x20   a := Shape::Round {}\n\
                  \x20   if (a == Shape::Round {}) { return 1 }\n\
                  \x20   0\n\
                  }\n";
    let message = compile_error("enumeqfields", source);
    assert!(
        message.contains("carries fields") && message.contains("match"),
        "expected the ambiguity to be named, got:\n{message}"
    );
}

// Matching several values at once against a tuple of patterns. This has always
// worked; what did not was the message when a tuple pattern met a value that is
// not one, which said the feature was missing rather than that the case had
// parts the value does not.
const TUPLE_PATTERNS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

label :: fn(n: i64) -> i64 {
    match (n % 3, n % 5) {
        case (0, 0): 15
        case (0, _): 3
        case (_, 0): 5
        case _: 0
    }
}

shape :: fn(a: i64, b: i64, c: i64) -> i64 {
    match (a, b, c) {
        case (1, 2, 3): 123
        case (1, _, 3): 103
        case _: 0
    }
}

main :: fn() -> i64 {
    unsafe { printf("%lld\n", label(15)) }
    unsafe { printf("%lld\n", label(9)) }
    unsafe { printf("%lld\n", label(10)) }
    unsafe { printf("%lld\n", label(7)) }
    unsafe { printf("%lld\n", shape(1, 2, 3)) }
    unsafe { printf("%lld\n", shape(1, 9, 3)) }
    unsafe { printf("%lld\n", shape(4, 5, 6)) }
    0
}
"#;

#[test]
fn a_match_compares_several_values_at_once() {
    let Some(output) = compile_and_run_unaudited("tuplepat", TUPLE_PATTERNS)
    else {
        return;
    };
    assert_eq!(output, "15\n3\n5\n0\n123\n103\n0\n");
}

#[test]
fn a_tuple_case_on_a_scalar_says_what_is_wrong() {
    let source = "main :: fn() -> i64 {\n\
                  \x20   x := 4\n\
                  \x20   match x { case (0, 0): 1  case _: 2 }\n\
                  }\n";
    let message = compile_error("tuplescalar", source);
    assert!(
        message.contains("matches a tuple") && message.contains("'i64'"),
        "expected the mismatch to name the type, got:\n{message}"
    );
}

// A statement ends at the line break, so a `(` starting the next line begins a
// new statement rather than calling what came before it. It used to bind to the
// left, so a function whose body ended in a parenthesised expression after an
// `if` was read as calling the `if`, and failed with "cannot call a value that
// is not a function pointer" pointing at a line with no call on it.
const PARENTHESISED_STATEMENT: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

eightbytes :: fn(size: i64) -> i64 {
    if (size == 0) { return 0 }
    (size + 7) / 8
}

doubled :: fn(n: i64) -> i64 {
    var total := n
    (total)
}

// A call still binds when the parenthesis is where the call is written, across
// as many lines as the arguments take.
spread :: fn(a: i64, b: i64) -> i64 { a * 10 + b }

main :: fn() -> i64 {
    unsafe { printf("%lld\n", eightbytes(0)) }
    unsafe { printf("%lld\n", eightbytes(24)) }
    unsafe { printf("%lld\n", eightbytes(1)) }
    unsafe { printf("%lld\n", doubled(5)) }
    unsafe { printf("%lld\n", spread(
        3,
        4)) }
    0
}
"#;

#[test]
fn a_parenthesis_on_a_new_line_starts_a_statement() {
    let Some(output) =
        compile_and_run_unaudited("parenstmt", PARENTHESISED_STATEMENT)
    else {
        return;
    };
    assert_eq!(output, "0\n3\n1\n5\n34\n");
}

// A narrow value widens to the width of the writer that takes it. A `bool` is
// one byte and was not counted as an integer for that, so the native backend
// built a call handing an i8 where an i64 belonged and failed in the Cranelift
// verifier. C widened it silently, so only one backend saw it.
const PRINT_NARROW_VALUES: &str = r#"import "io.frost"

answered :: fn(n: i64) -> bool { n > 0 }
narrow :: fn() -> i32 { 7 }
byte :: fn() -> u8 { 200 }

main :: fn() -> i64 {
    print("{}\n", answered(1))
    print("{}\n", answered(-1))
    print("{}\n", narrow())
    print("{}\n", byte())
    print("{}\n", true)
    print("{}\n", 3 < 4)
    0
}
"#;

#[test]
fn print_widens_a_narrow_value() {
    let Some(output) =
        compile_and_run_unaudited("printnarrow", PRINT_NARROW_VALUES)
    else {
        return;
    };
    assert_eq!(output, "1\n0\n7\n200\n1\n1\n");
}

// The failure type as a struct rather than an enum, which is the other half of
// what a failure set accepts and the half nothing covered. `return Blocked {
// at = hp }` is a struct literal, and only an enum-variant literal counted as
// building the error, so it was wrapped as the Ok value and reached the backend
// as a struct where the value type belonged. examples/selfhosted/failures.frost
// had been failing on the bootstrap the whole time for this reason.
const STRUCT_FAILURE_TYPE: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Blocked :: struct { at: i64, why: i64 }

strike :: fn(hp: i64) -> i64 ! Blocked {
    if (hp <= 0) {
        return Blocked { at = hp, why = 7 }
    }
    hp * 2
}

twice :: fn(hp: i64) -> i64 ! Blocked {
    once := strike(hp)?
    once + 1
}

// Both sides read, including a field of the error, which is the part that only
// works if the error really is the error.
report :: fn(hp: i64) -> i64 {
    match twice(hp) {
        case .Ok { value }: value
        case .Err { error }: error.at * 100 + error.why
    }
}

main :: fn() -> i64 {
    unsafe { printf("%lld\n", report(5)) }
    unsafe { printf("%lld\n", report(0)) }
    unsafe { printf("%lld\n", report(-3)) }
    0
}
"#;

#[test]
fn a_failure_type_may_be_a_struct() {
    let Some(output) =
        compile_and_run_unaudited("structfail", STRUCT_FAILURE_TYPE)
    else {
        return;
    };
    assert_eq!(output, "11\n7\n-293\n");
}

// A fallible function returns `-> T ! E`; `?` unwraps the Ok value and returns
// the enclosing function's Err on failure; the caller matches Ok/Err.
#[test]
fn native_failure_sets() {
    let Some(output) = compile_and_run_unaudited("failure_sets", FAILURE_SETS)
    else {
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
    slot := unsafe { ptr_to(a.data[a.offset]) }
    a.offset = a.offset + sizeof(i64)
    unsafe { ptr_cast($i64, slot) }
}

make_two :: fn() -> i64 uses Arena<256> {
    p := alloc_int(arena)
    unsafe { p^ = 10 }
    q := alloc_int(arena)
    unsafe { q^ = 32 }
    unsafe { p^ + q^ }
}

forward :: fn() -> i64 uses Arena<256> {
    make_two()
}

main :: fn() -> i64 {
    var arena : Arena<256> = Arena { data = [0; 256], offset = 0 }
    var result : i64 = 0
    with arena {
        result = forward()
    }
    unsafe { printf("%lld\n", result) }
    0
}
"#;

// A function may draw more than one allocation source. Each is an implicit
// parameter its body reaches by the type's own name lowercased, and a call
// supplies one argument per source, chosen by that name, so a function drawing
// two can tell them apart. A function drawing one takes whatever is innermost
// whatever it is called, which is what lets a `with scratch` block supply a
// `uses Arena`.
const TWO_ALLOCATION_SOURCES: &str = r#"import "io.frost"

Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
Scratch :: struct($N: usize) { data: [N]u8, offset: i64 }

take_arena :: fn(mut a: Arena<256>) -> i64 {
    a.offset = a.offset + 8
    a.offset
}

take_scratch :: fn(mut s: Scratch<64>) -> i64 {
    s.offset = s.offset + 1
    s.offset
}

both :: fn() -> i64 uses Arena<256>, Scratch<64> {
    take_arena(arena) + take_scratch(scratch)
}

forwards :: fn() -> i64 uses Arena<256>, Scratch<64> {
    both()
}

only_one :: fn() -> i64 uses Scratch<64> {
    take_scratch(scratch)
}

main :: fn() -> i64 {
    var arena : Arena<256> = Arena { data = [0; 256], offset = 0 }
    var scratch : Scratch<64> = Scratch { data = [0; 64], offset = 0 }
    var result : i64 = 0
    with arena {
        with scratch {
            result = forwards()
            result = result + only_one()
        }
    }
    print("{}\n", result)
    0
}
"#;

#[test]
fn a_function_may_draw_two_allocation_sources() {
    let Some(output) =
        compile_and_run_unaudited("twosources", TWO_ALLOCATION_SOURCES)
    else {
        return;
    };
    assert_eq!(output, "11\n");
}

#[test]
fn self_hosted_draws_two_allocation_sources() {
    let Some(output) =
        selfhosted_unaudited_output("shtwosources", TWO_ALLOCATION_SOURCES)
    else {
        return;
    };
    assert_eq!(output, "11\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shtwosources_input.frost");
    std::fs::write(&input, TWO_ALLOCATION_SOURCES).unwrap();
    let Some(c_source) = self_hosted_emits("shtwosources", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shtwosources", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// `uses Arena<256>` declares an allocation source; the capability is threaded
// implicitly through a `uses` call (`forward` forwards to `make_two`) and
// supplied by a `with` block at the root. No arena is passed by hand.
#[test]
fn native_allocation_sources() {
    let Some(output) =
        compile_and_run_unaudited("alloc_sources", ALLOCATION_SOURCES)
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
    slot := unsafe { ptr_to(a.data[a.offset]) }
    a.offset = a.offset + sizeof(i64)
    unsafe { ptr_cast($i64, slot) }
}

main :: fn() -> i64 {
    var arena : Arena<256> = Arena { data = [0; 256], offset = 0 }
    var escaped : ^i64 = ptr_to(arena.offset)
    with arena {
        escaped = alloc_int(arena)
    }
    unsafe { escaped^ }
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
    slot := unsafe { ptr_to(a.data[a.offset]) }
    a.offset = a.offset + sizeof(i64)
    unsafe { ptr_cast($i64, slot) }
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

// A `$T` written as an argument is a type. Nothing runs it and there is no
// storage behind it, so an answer built with one names no parameter of whoever
// wrote it.
//
// The walk that decides which parameters an answer can name read an unfamiliar
// expression as naming every one of them, which is the right default and the
// wrong answer here: `heap_slice($i64, room)` made `store_new`'s answer name its
// `view` parameter too, and a caller holding a short-lived `view` was then
// refused for handing the store back. `render_world_new(device, room)` is the
// shape that found it, and an `App` holding a `RenderWorld` could not be built.
const TYPE_ARGUMENT_STORAGE: &str = "import \"io.frost\"\nimport \"mem.frost\"\n\
     Owner :: struct { items: []i64, tag: i64 }\n\
     Store :: struct { room: []i64 }\n\
     owner_items :: fn(o: Owner) -> []i64 { o.items }\n\
     store_new :: fn(view: []i64, room: i64) -> Store {\n\
     \x20   Store { room = heap_slice($i64, room) }\n\
     }\n\
     make :: fn() -> Store {\n\
     \x20   var cells : [2]i64 = [0; 2]\n\
     \x20   o := Owner { items = cells, tag = 7 }\n\
     \x20   held := owner_items(o)\n\
     \x20   store_new(held, 4)\n\
     }\n\
     main :: fn() -> i64 {\n\
     \x20   var s := make()\n\
     \x20   s.room[0] = 5\n\
     \x20   print(\"{}\\n\", s.room[0])\n\
     \x20   heap_release_slice($i64, s.room)\n\
     \x20   0\n\
     }\n";

#[test]
fn a_type_argument_names_no_caller_storage() {
    let Some(output) =
        compile_and_run_unaudited("typearg_storage", TYPE_ARGUMENT_STORAGE)
    else {
        return;
    };
    assert_eq!(output, "5\n");
}

#[test]
fn self_hosted_takes_a_type_argument_as_naming_no_storage() {
    let Some(output) =
        selfhosted_unaudited_output("shtypearg", TYPE_ARGUMENT_STORAGE)
    else {
        return;
    };
    assert_eq!(output, "5\n");
}

// A resource named in a struct literal is moved into it. The struct holds it
// from there on, and whoever holds the struct owes the consumption.
//
// The self-hosted compiler read a struct literal's initializers with the walk it
// reads call arguments with, where naming a value lends it rather than handing
// it over, so `Pair { one = h, tag = 1 }` left `h` looking unconsumed and every
// function that assembles a resource out of resources was refused.
const RESOURCE_INTO_A_STRUCT: &str = "import \"io.frost\"\nHeld :: linear struct { id: i64 }\n\
     Pair :: linear struct { one: Held, tag: i64 }\n\
     open :: fn(id: i64) -> Held { Held { id = id } }\n\
     close :: fn(move h: Held) -> i64 { h.id }\n\
     pair_new :: fn() -> Pair {\n\
     \x20   var h := open(3)\n\
     \x20   Pair { one = h, tag = 1 }\n\
     }\n\
     pair_close :: fn(move p: Pair) -> i64 { close(p.one) + p.tag }\n\
     main :: fn() -> i64 {\n\
     \x20   var p := pair_new()\n\
     \x20   print(\"{}\\n\", pair_close(p))\n\
     \x20   0\n\
     }\n";

#[test]
fn a_resource_is_consumed_by_the_struct_that_holds_it() {
    let Some(output) =
        compile_and_run_unaudited("res_into_struct", RESOURCE_INTO_A_STRUCT)
    else {
        return;
    };
    assert_eq!(output, "4\n");
}

#[test]
fn self_hosted_takes_a_struct_literal_as_consuming_what_it_names() {
    let Some(output) =
        selfhosted_unaudited_output("shresstruct", RESOURCE_INTO_A_STRUCT)
    else {
        return;
    };
    assert_eq!(output, "4\n");
}

// The same rule for an array literal, which is the other half of the arm that
// reads a literal's items. An element written into an array is handed to the
// array exactly as a field written into a struct is handed to the struct, so
// `pair` owes nothing for the two it opened once the array holds them.
//
// The array is handed straight to what consumes it. A resource can be taken out
// of an array the call was given by `move`, and cannot be taken out of one a
// frame is still holding, which is why `both` is a parameter here.
const RESOURCE_INTO_AN_ARRAY: &str = "import \"io.frost\"\nHeld :: linear struct { id: i64 }\n\
     open :: fn(id: i64) -> Held { Held { id = id } }\n\
     close :: fn(move h: Held) -> i64 { h.id }\n\
     both :: fn(move held: [2]Held) -> i64 {\n\
     \x20   close(held[0]) + close(held[1])\n\
     }\n\
     pair :: fn() -> [2]Held {\n\
     \x20   var one := open(3)\n\
     \x20   var two := open(4)\n\
     \x20   [one, two]\n\
     }\n\
     main :: fn() -> i64 {\n\
     \x20   print(\"{}\\n\", both(pair()))\n\
     \x20   0\n\
     }\n";

#[test]
fn a_resource_is_consumed_by_the_array_that_holds_it() {
    let Some(output) =
        compile_and_run_unaudited("res_into_array", RESOURCE_INTO_AN_ARRAY)
    else {
        return;
    };
    assert_eq!(output, "7\n");
}

#[test]
fn self_hosted_takes_an_array_literal_as_consuming_what_it_names() {
    let Some(output) =
        selfhosted_unaudited_output("shresarray", RESOURCE_INTO_AN_ARRAY)
    else {
        return;
    };
    assert_eq!(output, "7\n");
}

// A column is found by counting the mask bits below it, so a component the
// table does not hold still answers a number: another column's. Reading through
// it is one component's bytes read as another and writing through it is one
// written over another, both silent and both in safe code.
//
// `ecs_add` is what gives an entity a component it does not have. `ecs_get` and
// `ecs_set` are for one it does, and a table that does not hold it stops.
const COMPONENT_AN_ENTITY_LACKS: &str = "import \"io.frost\"\nimport \"ecs.frost\"\n\
     Held :: struct { a: i64, b: i64 }\n\
     Other :: struct { n: i64 }\n\
     main :: fn() -> i64 {\n\
     \x20   var world := ecs_new()\n\
     \x20   held := ecs_register($Held, world)\n\
     \x20   other := ecs_register($Other, world)\n\
     \x20   entity := ecs_spawn_with(world, mask_with(mask_empty(), held))\n\
     \x20   ecs_set($Other, world, entity, other, Other { n = 5 })\n\
     \x20   print(\"{}\\n\", 0)\n\
     \x20   ecs_free(world)\n\
     \x20   0\n\
     }\n";

#[test]
fn a_component_an_entity_lacks_is_refused_rather_than_guessed() {
    let Some((succeeded, stderr)) =
        compile_and_run_status("lackscomponent", COMPONENT_AN_ENTITY_LACKS)
    else {
        return;
    };
    assert!(
        !succeeded,
        "writing a component the entity lacks should stop"
    );
    assert!(
        stderr.contains("a component it does not have"),
        "expected the missing-component stop, got:\n{stderr}"
    );
}

// Two enums may name the same variant. A bare `.Render` in a match arm is a
// variant of whichever enum is being matched, and the exhaustiveness check has
// to weigh the arms against that enum rather than against whichever declared
// the name first.
const SHARED_VARIANT_NAME: &str = "import \"io.frost\"\nStage :: enum { First, Render, Last }\n\
     Pass :: enum { Render, Compute }\n\
     stage_at :: fn(s: Stage) -> i64 {\n\
     \x20   match s {\n\
     \x20       case .First: 0\n\
     \x20       case .Render: 1\n\
     \x20       case .Last: 2\n\
     \x20   }\n\
     }\n\
     pass_at :: fn(k: Pass) -> i64 {\n\
     \x20   match k {\n\
     \x20       case .Compute: 10\n\
     \x20       case .Render: 11\n\
     \x20   }\n\
     }\n\
     main :: fn() -> i64 {\n\
     \x20   print(\"{}\\n\", stage_at(Stage::Render))\n\
     \x20   print(\"{}\\n\", pass_at(Pass::Render))\n\
     \x20   0\n\
     }\n";

#[test]
fn two_enums_may_share_a_variant_name() {
    let Some(output) =
        compile_and_run_unaudited("sharedvariant", SHARED_VARIANT_NAME)
    else {
        return;
    };
    assert_eq!(output, "1\n11\n");
}

#[test]
fn self_hosted_reads_a_shared_variant_name_by_its_enum() {
    let Some(output) =
        selfhosted_unaudited_output("shsharedvariant", SHARED_VARIANT_NAME)
    else {
        return;
    };
    assert_eq!(output, "1\n11\n");
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
// plain value and no `&`/`&mut`. The compiler borrows for the mut parameter.
#[test]
fn native_parameter_modes() {
    let Some(output) = compile_and_run_unaudited("param_modes", PARAM_MODES)
    else {
        return;
    };
    assert_eq!(output, "7\n17\n");
}

#[test]
fn native_anonymous_functions() {
    let Some(output) = compile_and_run_unaudited("anon", ANON_FUNCTIONS) else {
        return;
    };
    assert_eq!(output, "42\n81\n47\n20\n");
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
    assert!(
        run.status.success(),
        "the self-hosted compiler failed on {name}:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    Some(String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"))
}

// The interpreter answers the runtime's checks itself, and it has to answer
// them the way `runtime/runtime.frost` does. It cannot share that code: the
// `frost_rt_` prefix is the runtime's own, so a program carrying those
// definitions is refused, which is what keeps a program from replacing what
// every other program calls. Two readings of one rule is the price, and this is
// what holds them together: each program either faults under both or runs under
// both, and where it runs the answers match.
#[test]
fn the_interpreter_faults_where_the_runtime_faults() {
    let cases: &[(&str, &str, bool)] = &[
        (
            "in_bounds",
            "import \"io.frost\"\nmain :: fn() -> i64 {\n\
             \x20   var xs : [3]i64 = [7, 8, 9]\n\
             \x20   print(\"{}\\n\", xs[2])\n    0\n}\n",
            false,
        ),
        (
            "past_the_end",
            "import \"io.frost\"\nmain :: fn() -> i64 {\n\
             \x20   var xs : [3]i64 = [7, 8, 9]\n\
             \x20   var at : i64 = 5\n\
             \x20   print(\"{}\\n\", xs[at])\n    0\n}\n",
            true,
        ),
        (
            "a_span_inside_the_run",
            "import \"io.frost\"\nimport \"mem.frost\"\nmain :: fn() -> i64 {\n\
             \x20   var xs : [4]i64 = [1, 2, 3, 4]\n\
             \x20   view := slice_range($i64, xs, 1, 2)\n\
             \x20   print(\"{}\\n\", view[1])\n    0\n}\n",
            false,
        ),
        (
            "a_span_past_the_run",
            "import \"io.frost\"\nimport \"mem.frost\"\nmain :: fn() -> i64 {\n\
             \x20   var xs : [4]i64 = [1, 2, 3, 4]\n\
             \x20   var count : i64 = 9\n\
             \x20   view := slice_range($i64, xs, 1, count)\n\
             \x20   print(\"{}\\n\", view[0])\n    0\n}\n",
            true,
        ),
    ];
    for (name, source, faults) in cases {
        let interpreted = run_ir_oracle(&format!("chk_{name}"), source);
        let Some(linked) =
            compile_and_run_checked(&format!("chk_{name}"), source)
        else {
            continue;
        };
        assert_eq!(
            interpreted.is_none(),
            *faults,
            "the interpreter disagrees about whether {name} faults"
        );
        assert_eq!(
            linked.is_none(),
            *faults,
            "the linked runtime disagrees about whether {name} faults"
        );
        if let (Some(interpreted), Some(linked)) = (interpreted, linked) {
            assert_eq!(
                interpreted, linked,
                "the interpreter and the linked runtime disagree about {name}"
            );
        }
    }
}

// The interpreter spells a float the way the linked runtime does. It is one of
// the three backends a program is checked against, so a spelling of its own
// would make every program that prints a float unanswerable rather than
// compared. The values are the shapes `%g` chooses between: the plain form, the
// exponent form at each end, a fraction that ends in zeros, and one that does
// not end at all.
#[test]
fn the_interpreter_spells_a_float_the_way_the_runtime_does() {
    let source = "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", 2.5)\n\
         \x20   print(\"{}\\n\", 0.1)\n\
         \x20   print(\"{}\\n\", 1.0)\n\
         \x20   print(\"{}\\n\", 100.0)\n\
         \x20   print(\"{}\\n\", 1234567.0)\n\
         \x20   print(\"{}\\n\", 0.000012345)\n\
         \x20   print(\"{}\\n\", -0.5)\n\
         \x20   print(\"{}\\n\", 1.0 / 3.0)\n\
         \x20   0\n\
         }\n";
    let Some(interpreted) = run_ir_oracle("floatg", source) else {
        panic!("the interpreter declined a program that prints floats");
    };
    let Some(linked) = compile_and_run_unaudited("floatg", source) else {
        return;
    };
    assert_eq!(
        interpreted, linked,
        "the interpreter and the linked runtime spell a float differently"
    );
}

// A call through a function-pointer field has to agree with the definition it
// reaches about how an aggregate travels. Every definition takes one by
// address, so the cast the call goes through says a pointer too.
//
// Behaviour cannot answer this. A sixteen-byte struct passed by value goes as a
// hidden pointer under the Windows convention, which is the same thing the
// callee reads, so a mismatch runs correctly there and reads the data pointer
// as the struct under the System V one. The emitted text is the same on both,
// so that is what this reads.
#[test]
fn a_call_through_a_field_passes_an_aggregate_by_address() {
    let source = "import \"io.frost\"\n\
         loud :: fn(text: str) { write(to_stdout, \"[{}]\", text) }\n\
         main :: fn() -> i64 {\n\
         \x20   write(loud, \"a {} b\\n\", 3)\n\
         \x20   0\n\
         }\n";
    let directory = std::env::temp_dir();
    let input = directory.join("frost_fieldabi.frost");
    std::fs::write(&input, source).unwrap();
    let Some(emitted) = self_hosted_emits("fieldabi", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);

    // `(*)(struct __arr0)` is the by-value spelling and `(*)(struct __arr0*)`
    // the one that matches. Finding the first is the fault.
    let mut faults = Vec::new();
    for (index, _) in emitted.match_indices("(*)(") {
        let rest = &emitted[index + 4..];
        let Some(close) = rest.find(')') else {
            continue;
        };
        for parameter in rest[..close].split(',') {
            let parameter = parameter.trim();
            if parameter.starts_with("struct ") && !parameter.ends_with('*') {
                faults.push(parameter.to_string());
            }
        }
    }
    assert!(
        faults.is_empty(),
        "a function-pointer cast takes an aggregate by value, which the \
         definition does not: {faults:?}"
    );
}

fn compile_c_and_run(name: &str, c_source: &str) -> Option<String> {
    let compiler = c_compiler()?;
    let directory = std::env::temp_dir();
    let stem = unique(&format!("frost_emitted_{name}"));
    let c_path = directory.join(format!("{stem}.c"));
    let exe_path =
        directory.join(format!("{stem}{}", std::env::consts::EXE_SUFFIX));
    std::fs::write(&c_path, c_source).unwrap();
    // The emitted code calls into the runtime for the bounds check an index
    // compiles to and for assertions, so the runtime is linked alongside it,
    // as the object every test in this binary shares.
    let runtime = runtime_object();
    // The math functions `std/math.frost` reaches live in libm where the
    // platform keeps them out of the C runtime, which is Linux and the BSDs.
    // Both compilers pass this on their own link paths; a test that links what
    // one of them emitted has to pass it too, and only a platform that keeps
    // them apart says so.
    let compile = Command::new(compiler)
        .arg("-std=c11")
        .arg(&c_path)
        .arg(&runtime)
        .arg(frost_runtime_object())
        .arg("-o")
        .arg(&exe_path)
        .arg("-lm")
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
const SELF_HOSTED_WIDTHS: &str = "import \"io.frost\"\nMixed :: struct { a: i32, b: i16, c: u8, d: i64 }\n\
     main :: fn() -> i64 {\n\
     \x20   var small : i32 = -5\n\
     \x20   var tiny : i16 = 300\n\
     \x20   var byte : u8 = 200\n\
     \x20   var big : u32 = 4000000000\n\
     \x20   var wide : usize = 9000000000\n\
     \x20   print(\"{}\\n\", small)\n    print(\"{}\\n\", tiny)\n    print(\"{}\\n\", byte)\n    print(\"{}\\n\", big)\n\
     \x20   print(\"{}\\n\", wide)\n    print(\"{}\\n\", sizeof(Mixed))\n\
     \x20   m := Mixed { a = -7, b = 9, c = 250, d = 123456789 }\n\
     \x20   print(\"{}\\n\", m.a)\n    print(\"{}\\n\", m.b)\n    print(\"{}\\n\", m.c)\n    print(\"{}\\n\", m.d)\n\
     \x20   ptr := ptr_to(m)\n    unsafe { ptr^.a = -1 }\n\
     \x20   print(\"{}\\n\", m.a)\n    print(\"{}\\n\", m.d)\n    0\n}\n";

const WIDTHS_EXPECTED: &str = "-5\n300\n200\n4000000000\n9000000000\n16\n-7\n9\n250\n123456789\n-1\n123456789\n";

#[test]
fn self_hosted_integer_widths_natively() {
    let Some(output) =
        selfhosted_unaudited_output("widths", SELF_HOSTED_WIDTHS)
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
const SELF_HOSTED_FLOATS: &str = "import \"io.frost\"\nscale :: fn(v: f64, by: f64) -> f64 {\n    v * by\n}\n\
     narrow :: fn(v: f32) -> f32 {\n    v + 0.5\n}\n\
     Pair :: struct { x: f64, y: f32 }\n\
     main :: fn() -> i64 {\n\
     \x20   a := 3.5\n    b := 2.0\n\
     \x20   print(\"{}\\n\", a + b)\n    print(\"{}\\n\", a - b)\n    print(\"{}\\n\", a * b)\n    print(\"{}\\n\", a / b)\n\
     \x20   print(\"{}\\n\", scale(a, 4.0))\n\
     \x20   var small : f32 = 1.25\n    print(\"{}\\n\", small)\n    print(\"{}\\n\", narrow(small))\n\
     \x20   print(\"{}\\n\", sizeof(Pair))\n\
     \x20   pr := Pair { x = 9.75, y = 0.5 }\n\
     \x20   print(\"{}\\n\", pr.x)\n    print(\"{}\\n\", pr.y)\n\
     \x20   if (a > b) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
     \x20   if (a < b) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
     \x20   var n : i64 = 7\n    d := n * 1.0\n    print(\"{}\\n\", d / 2.0)\n    0\n}\n";

const FLOATS_EXPECTED: &str =
    "5.5\n1.5\n7\n1.75\n14\n1.25\n1.75\n16\n9.75\n0.5\n1\n0\n3.5\n";

#[test]
fn self_hosted_floats_natively() {
    let Some(output) =
        selfhosted_unaudited_output("floats", SELF_HOSTED_FLOATS)
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
const SELF_HOSTED_ARRAYS: &str = "import \"io.frost\"\nsum :: fn(numbers: []i64) -> i64 {\n\
     \x20   var total : i64 = 0\n    var i : i64 = 0\n\
     \x20   while (i < slice_len(numbers)) {\n\
     \x20       total = total + numbers[i]\n        i = i + 1\n    }\n\
     \x20   total\n}\n\
     main :: fn() -> i64 {\n\
     \x20   scores := [40, 10, 90, 30, 70]\n\
     \x20   view : []i64 = scores\n\
     \x20   print(\"{}\\n\", slice_len(scores))\n    print(\"{}\\n\", slice_len(view))\n\
     \x20   print(\"{}\\n\", scores[2])\n    print(\"{}\\n\", view[3])\n\
     \x20   print(\"{}\\n\", sum(view))\n    print(\"{}\\n\", sum(scores))\n\
     \x20   print(\"{}\\n\", sizeof([5]i64))\n\
     \x20   var grid : [3]i64 = [7, 8, 9]\n\
     \x20   grid[1] = 99\n    print(\"{}\\n\", grid[1])\n    print(\"{}\\n\", sum(grid))\n    0\n}\n";

const ARRAYS_EXPECTED: &str = "5\n5\n90\n30\n240\n240\n40\n99\n115\n";

#[test]
fn self_hosted_arrays_natively() {
    let Some(output) =
        selfhosted_unaudited_output("arrays", SELF_HOSTED_ARRAYS)
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
    let source = "import \"io.frost\"\ntrap :: fn() -> i64 {\n\
         \x20   var ok : [1]i64 = [0]\n    ok[5] = 1\n    1\n}\n\
         main :: fn() -> i64 {\n\
         \x20   var n : i64 = 0\n\
         \x20   if (n == 1 && trap() == 1) { print(\"{}\\n\", 9) } else { print(\"{}\\n\", 1) }\n\
         \x20   if (n == 0 || trap() == 1) { print(\"{}\\n\", 2) } else { print(\"{}\\n\", 8) }\n    0\n}\n";
    let Some(output) = selfhosted_unaudited_output("shortcircuit", source)
    else {
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
        "import \"io.frost\"\ndouble :: fn(n: i64) -> i64 { n * 2 }\n\
         test \"doubling\" {\n\
         \x20   assert(double(2) == 4)\n    assert(double(0) == 0)\n}\n\
         test \"a failing one\" {\n    assert(double(2) == 5)\n}\n\
         main :: fn() -> i64 { print(\"{}\\n\", double(21))  0 }\n",
    )
    .unwrap();
    let runtime =
        format!("{}/runtime/frost_runtime.c", env!("CARGO_MANIFEST_DIR"));

    for (label, backend) in [("tc", "--emit-c"), ("tasm", "--emit-asm")] {
        let exe = directory.join(format!(
            "{}{}",
            unique(&format!("frost_{label}")),
            std::env::consts::EXE_SUFFIX
        ));
        let run = Command::new(&compiler)
            .arg(backend)
            .arg("--test")
            .arg("-o")
            .arg(&exe)
            .arg(&input)
            .env("FROST_RUNTIME", &runtime)
            .env("FROST_RUNTIME_FROST", frost_runtime_source())
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
            failure.contains("frost_selftests_input.frost:8:5"),
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
    let plain = directory.join(format!(
        "{}{}",
        unique("frost_tplain"),
        std::env::consts::EXE_SUFFIX
    ));
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
}

// The self-hosted compiler, the most pointer-heavy program in the tree, compiles
// clean under the unsafety gate. Every function that touches raw memory marks
// its body `unsafe`, so the gate has nothing to refuse. This is what says the
// gate is livable for real code and why it can be on by default.
#[test]
fn the_self_hosted_compiler_is_clean_under_the_unsafe_gate() {
    let build = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--emit-c")
        .arg("-o")
        .arg(std::env::temp_dir().join("frost_gate_selfhosted.c"))
        .arg(self_hosted_source())
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "the self-hosted compiler was rejected by its own gate:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
}

// OS threads from std: two threads accumulate a range into a shared word
// through an atomic add, and the total is exact every time, which is what says
// the atomic holds and the join waits.
#[test]
fn self_hosted_threads_share_a_counter() {
    if !linker_available() || c_compiler().is_none() {
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let directory = std::env::temp_dir().join("frost_threads");
    let _ = std::fs::create_dir_all(&directory);
    let source = directory.join("threads.frost");
    std::fs::write(
        &source,
        "import \"io.frost\"\n\
         import \"thread.frost\"\n\
         Work :: struct { start: i64, count: i64, total: ^i64 }\n\
         worker :: fn(raw: ^u8) {\n\
         \x20   w := unsafe { ptr_cast($Work, raw) }\n\
         \x20   var i : i64 = 0\n\
         \x20   count := unsafe { w^.count }\n\
         \x20   cell := unsafe { w^.total }\n\
         \x20   start := unsafe { w^.start }\n\
         \x20   while (i < count) { atomic_add(cell, start + i)  i = i + 1 }\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   var total : i64 = 0\n\
         \x20   var w1 := Work { start = 0, count = 500, total = ptr_to(total) }\n\
         \x20   var w2 := Work { start = 500, count = 500, total = ptr_to(total) }\n\
         \x20   t1 := unsafe { spawn(worker, ptr_cast($u8, ptr_to(w1))) }\n\
         \x20   t2 := unsafe { spawn(worker, ptr_cast($u8, ptr_to(w2))) }\n\
         \x20   join(t1)  join(t2)\n\
         \x20   print(\"{}\\n\", total)\n    0\n}\n",
    )
    .unwrap();
    let c_path = directory.join("threads.c");
    let build = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("-L")
        .arg(root.join("std"))
        .arg("--emit-c")
        .arg("-o")
        .arg(&c_path)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "the threads program did not compile:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let exe = directory.join(format!(
        "{}{}",
        unique("threads"),
        std::env::consts::EXE_SUFFIX
    ));
    let runtime = format!("{}/runtime/frost_runtime.c", root.display());
    let compile = Command::new(c_compiler().unwrap())
        .arg(&c_path)
        .arg(&runtime)
        .arg(frost_runtime_object())
        .arg("-o")
        .arg(&exe)
        .output()
        .unwrap();
    if !compile.status.success() {
        return;
    }
    let run = Command::new(&exe).output().unwrap();
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"),
        "499500\n"
    );
    let _ = std::fs::remove_dir_all(&directory);
}

// `safe extern fn` says a C function was audited and cannot corrupt memory, so
// calling it needs no block. The assertion belongs on the declaration, once,
// rather than at every call site, which is what keeps `unsafe` to what can
// actually go wrong. A plain `extern fn` is still gated.
#[test]
fn a_safe_extern_needs_no_unsafe_block() {
    let audited = "printf :: safe extern fn(fmt: ^i8, v: i64) -> i32\n\
         main :: fn() -> i64 { printf(\"%lld\\n\", 7)  0 }\n";
    let Some(output) = compile_and_run_unaudited("safe_extern", audited) else {
        return;
    };
    assert_eq!(output, "7\n");

    let plain = "printf :: extern fn(fmt: ^i8, v: i64) -> i32\n\
         main :: fn() -> i64 { printf(\"%lld\\n\", 7)  0 }\n";
    let directory = std::env::temp_dir();
    let source_path = directory.join("frost_plain_extern.frost");
    let exe_path = directory.join(format!(
        "frost_plain_extern{}",
        std::env::consts::EXE_SUFFIX
    ));
    std::fs::write(&source_path, plain).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("-o")
        .arg(&exe_path)
        .arg(&source_path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&source_path);
    assert!(
        !output.status.success(),
        "a plain extern should still be gated but it compiled"
    );
    let message = String::from_utf8_lossy(&output.stderr);
    assert!(
        message.contains("unchecked, so it belongs in an `unsafe` block"),
        "a plain extern should still be gated, got:\n{message}"
    );
}

// Indexing a raw pointer is unchecked, so it belongs in an `unsafe` block. The
// gate used to miss it when the pointer was bound from `ptr_to` without an
// annotation, because the unsafety pass could not name the binding's type; the
// index then compiled to raw pointer arithmetic with no bounds check, reading
// out of bounds from ordinary safe code.
#[test]
fn indexing_a_ptr_to_binding_is_gated() {
    let source = "main :: fn() -> i64 {\n\
         arr := [1, 2, 3]\n\
         p := ptr_to(arr)\n\
         x := p[9]\n\
         0\n\
         }\n";
    let directory = std::env::temp_dir();
    let source_path = directory.join("frost_ptr_to_index_gate.frost");
    let exe_path = directory.join(format!(
        "frost_ptr_to_index_gate{}",
        std::env::consts::EXE_SUFFIX
    ));
    std::fs::write(&source_path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("-o")
        .arg(&exe_path)
        .arg(&source_path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&source_path);
    assert!(
        !output.status.success(),
        "indexing a ptr_to binding should be gated but it compiled"
    );
    let message = String::from_utf8_lossy(&output.stderr);
    assert!(
        message.contains("indexing a raw pointer")
            && message.contains("belongs in an `unsafe` block"),
        "expected the raw-pointer index gate, got:\n{message}"
    );
}

// A call argument holds expressions like any other position, and the gate has
// to walk into it. The `print` statement was the hole here once: the gate
// walked past it, and a program with no `unsafe` block read out of bounds
// through a raw pointer and died on the access.
#[test]
fn a_call_argument_does_not_hide_a_gated_operation() {
    let cases = [
        (
            "printderef",
            "import \"io.frost\"\nmain :: fn() -> i64 {\n\
             \x20   var x : i64 = 42\n\
             \x20   p := ptr_to(x)\n\
             \x20   print(\"{}\\n\", p^)\n\
             \x20   0\n}\n",
            "reading through a raw pointer",
        ),
        (
            "printindex",
            "import \"io.frost\"\nmain :: fn() -> i64 {\n\
             \x20   var xs : [3]i64 = [7, 8, 9]\n\
             \x20   p := ptr_to(xs[0])\n\
             \x20   print(\"{}\\n\", p[100000])\n\
             \x20   0\n}\n",
            "indexing a raw pointer",
        ),
        (
            "printextern",
            "import \"io.frost\"\nabs :: extern fn(n: i32) -> i32\n\
             main :: fn() -> i64 {\n\
             \x20   print(\"{}\\n\", abs(0 - 5))\n\
             \x20   0\n}\n",
            "calling the C function 'abs'",
        ),
    ];
    for (name, source, wanted) in cases {
        let message = compile_error_checked(name, source);
        assert!(
            message.contains(wanted)
                && message.contains("belongs in an `unsafe` block"),
            "{name} should be gated, got:\n{message}"
        );
    }
}

// The multiple-return lowering runs after the gate, so a call bound to several
// names is still written as one statement when the gate walks it.
#[test]
fn a_multiple_return_binding_does_not_hide_a_gated_call() {
    let source = "pair :: extern fn() -> i64\n\
                  main :: fn() -> i64 {\n\
                  \x20   a := pair()\n\
                  \x20   a\n}\n";
    let message = compile_error_checked("multiretgate", source);
    assert!(
        message.contains("calling the C function 'pair'"),
        "expected the extern gate, got:\n{message}"
    );
}

// A base whose type the gate cannot name may be a raw pointer, so it is refused
// rather than allowed. What keeps that from refusing ordinary code is that the
// pass reads a call's return type off the declaration, so a binding produced by
// a call is named rather than unknown.
#[test]
fn indexing_a_base_of_unknown_type_is_gated() {
    let source = "opaque :: fn(n: i64) -> ^i64 { unsafe { ptr_cast($i64, ptr_to(n)) } }\n\
                  main :: fn() -> i64 {\n\
                  \x20   held := opaque(3)\n\
                  \x20   x := held[1]\n\
                  \x20   x\n}\n";
    let message = compile_error_checked("unknownindex", source);
    assert!(
        message.contains("indexing a raw pointer"),
        "expected the raw-pointer index gate through the call's return type, got:\n{message}"
    );
}

// A `ref` local holds the type of the place it names, so a field or an element
// reached through one is named rather than unknown. The gate used to have no
// type for a `ref` at all, and refused every read through one; the standard
// library's snapshot module could not be compiled by the bootstrap for that
// reason while the self-hosted compiler took it.
#[test]
fn a_ref_local_carries_the_type_of_what_it_names() {
    let source = "import \"io.frost\"\nInner :: struct { cells: [4]i64 }\n\
                  Outer :: struct { inner: Inner }\n\
                  main :: fn() -> i64 {\n\
                  \x20   var items := [Outer { inner = Inner { cells = [1, 2, 3, 4] } }]\n\
                  \x20   ref held := items[0]\n\
                  \x20   print(\"{}\\n\", held.inner.cells[2])\n\
                  \x20   0\n}\n";
    let Some(output) = compile_and_run_unaudited("reftype", source) else {
        return;
    };
    assert_eq!(output, "3\n");
}

// And a borrow of a raw pointer is that pointer under another name, so indexing
// one is gated exactly as indexing the pointer is. Giving a `ref` a type is only
// safe if the gate reads through the borrow to what is behind it.
#[test]
fn a_ref_to_a_raw_pointer_indexes_as_a_raw_pointer() {
    let source = "import \"io.frost\"\nmain :: fn() -> i64 {\n\
                  \x20   var cells := [1, 2, 3, 4]\n\
                  \x20   var here := unsafe { ptr_to(cells[0]) }\n\
                  \x20   ref aliased := here\n\
                  \x20   print(\"{}\\n\", aliased[1])\n\
                  \x20   0\n}\n";
    let message = compile_error_checked("refptrindex", source);
    assert!(
        message.contains("indexing a raw pointer"),
        "expected the raw-pointer index gate through the borrow, got:\n{message}"
    );
}

// The whole point of the standard library absorbing the unsafe floor: a program
// that uses vec, sort, format, strings and io compiles under the unsafety gate
// with no `unsafe` of its own. The containers' raw pointers and FFI are wrapped
// in their modules, so nothing above them is unchecked.
#[test]
fn a_program_using_std_is_clean_under_the_unsafe_gate() {
    if !linker_available() {
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let directory = std::env::temp_dir().join("frost_gate_std");
    let _ = std::fs::create_dir_all(&directory);
    let source = directory.join("uses_std.frost");
    std::fs::write(
        &source,
        "import \"io.frost\"\n\
         import \"vec.frost\"\n\
         import \"sort.frost\"\n\
         import \"format.frost\"\n\
         import \"strings.frost\"\n\
         import \"ordering.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   var v := vec_new($i64, 4)\n\
         \x20   seed := [5, 2, 9, 1, 7]\n\
         \x20   var i : i64 = 0\n\
         \x20   while (i < 5) { vec_push($i64, v, seed[i])  i = i + 1 }\n\
         \x20   sort_vec($i64, $i64_ascending, v)\n\
         \x20   var b := builder_new(16)\n\
         \x20   var j : i64 = 0\n\
         \x20   while (j < vec_len($i64, v)) {\n\
         \x20       builder_int(b, vec_get($i64, v, j))  builder_byte(b, 32)  j = j + 1\n\
         \x20   }\n\
         \x20   print(\"{}\\n\", builder_str(b))\n\
         \x20   if (str_eq(\"frost\", \"frost\")) { print(\"ok\\n\") }\n\
         \x20   builder_free(b)  vec_free($i64, v)  0\n}\n",
    )
    .unwrap();
    let exe = directory.join(format!(
        "{}{}",
        unique("uses_std"),
        std::env::consts::EXE_SUFFIX
    ));
    let build = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("-L")
        .arg(root.join("std"))
        .arg("--link")
        .arg("-o")
        .arg(&exe)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "a std program was rejected by the unsafe gate:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"),
        "1 2 5 7 9 \nok\n"
    );
    let _ = std::fs::remove_dir_all(&directory);
}

// Every standard library module that carries tests, run through both of the
// bootstrap's backends. Its counterpart below runs the same blocks under the
// self-hosted compiler, and until this covered more than math the two compilers
// were not being asked the same question: `snapshot.frost` was rejected by the
// bootstrap's unsafe gate and compiled by the self-hosted one, and any program
// importing `mem.frost` failed to build through the C backend on a declaration
// the emitted prelude had already made.
//
// The math modules are here for a second reason. Every exported function is
// covered, because a differential test says the two backends agree and these say
// the answers are right. A rotation that turns the wrong way, a projection with
// the depth range inverted and a quaternion that is its own inverse all agree
// across backends and are all wrong. Both precisions run: the f64 library is the
// f32 one with its element type changed, so a formula that survived the copy
// wrong fails here.
#[test]
fn the_standard_library_passes_its_own_tests() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let directory = std::env::temp_dir();

    // The C half needs a C compiler, and until `--test` honoured `--emit-c` it
    // did not: both halves of this loop built through Cranelift and the C
    // backend ran none of these bodies.
    let with_c = c_compiler().is_some();

    let jobs: Vec<(&str, &str, &str, bool)> = STD_MODULES
        .iter()
        .flat_map(|(module, expected)| {
            [("stdnative", false), ("stdc", true)]
                .into_iter()
                .map(move |(label, emit_c)| (label, *module, *expected, emit_c))
        })
        .filter(|(_, _, _, emit_c)| with_c || !emit_c)
        .collect();
    let faults = in_parallel(&jobs, |(label, module, expected, emit_c)| {
        let exe = directory.join(format!(
            "{}{}",
            unique(&format!("frost_{label}")),
            std::env::consts::EXE_SUFFIX
        ));
        let mut command = Command::new(env!("CARGO_BIN_EXE_frost"));
        if *emit_c {
            command.arg("--emit-c");
        }
        let run = command
            .arg("--test")
            .arg("-o")
            .arg(&exe)
            .arg(root.join("std").join(module))
            .output()
            .unwrap();
        let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
        let held = if output.contains(expected) && output.contains("0 failed") {
            None
        } else {
            Some(format!(
                "{module} {label}:\n{output}{}",
                String::from_utf8_lossy(&run.stderr)
            ))
        };
        let _ = std::fs::remove_file(&exe);
        held
    });
    let faults: Vec<String> = faults.into_iter().flatten().collect();
    assert!(faults.is_empty(), "{}", faults.join("\n"));
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

    // Every module that carries `test` blocks, through both of the self-hosted
    // compiler's backends. math.frost is the one that exercises floats, fixed
    // arrays and a struct returned by value, none of which strings.frost
    // reaches, and sort.frost is the one that carries a capability bundle
    // through two levels of generic.
    //
    // ecs.frost is the largest of them and was not here, which is how the
    // assembler went without the two single-precision conversions for as long
    // as it did: it is the only module whose tests write an integer into an
    // `f32` field, and every other one arrives at its floats as float literals.
    // A list naming the standard library and holding four of its modules is the
    // shape this suite exists to catch.
    //
    // `STD_MODULES` is the one list, which the bootstrap suite reads too. They
    // used to be two literals and drifted by three modules.
    let jobs: Vec<(&str, &str, &str, &str)> =
        [("stdc", "--emit-c"), ("stdasm", "--emit-asm")]
            .iter()
            .flat_map(|(label, backend)| {
                STD_MODULES.iter().map(move |(module, expected)| {
                    (*label, *backend, *module, *expected)
                })
            })
            .collect();
    let faults = in_parallel(&jobs, |(label, backend, module, expected)| {
        let exe = directory.join(format!(
            "{}{}",
            unique(&format!("frost_{label}")),
            std::env::consts::EXE_SUFFIX
        ));
        let run = Command::new(&compiler)
            .arg(backend)
            .arg("--test")
            .arg("-o")
            .arg(&exe)
            .arg(root.join("std").join(module))
            .env("FROST_RUNTIME", &runtime)
            .env("FROST_RUNTIME_FROST", frost_runtime_source())
            .output()
            .unwrap();
        let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
        let held = if output.contains(expected) && output.contains("0 failed") {
            None
        } else {
            Some(format!(
                "{label} on {module}:\n{output}{}",
                String::from_utf8_lossy(&run.stderr)
            ))
        };
        let _ = std::fs::remove_file(&exe);
        held
    });
    let faults: Vec<String> = faults.into_iter().flatten().collect();
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

// A generic in one module whose body calls a helper in that same module, reached
// from a program that imported only the generic. `for_each_row` is the one: it
// hands its columns to a row loop that `ecs.frost` does not export, and an
// instance of it is stamped out wherever it is called. Nothing else in the tree
// calls `for_each_row` from another module, so this is what says the helper
// resolves from where the instance is made rather than from where it is
// written.
#[test]
fn a_generic_reaches_its_own_module_from_a_program_that_imported_it() {
    let Some(compiler) = build_self_hosted_compiler("ecsimport") else {
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let directory = std::env::temp_dir();
    let source = directory
        .join(unique("frost_ecs_import"))
        .with_extension("frost");
    std::fs::write(
        &source,
        "import \"io.frost\"\nimport \"ecs.frost\"\n\
         Position :: struct { x: f32, y: f32 }\n\
         nudge :: fn(p: []Position, row: i64) {\n\
         \x20   p[row].x = p[row].x + 1.0\n}\n\
         main :: fn() -> i64 {\n\
         \x20   var world := ecs_new()\n\
         \x20   position := ecs_register($Position, world)\n\
         \x20   a := ecs_spawn(world)\n\
         \x20   ecs_add($Position, world, a, position,\n\
         \x20       Position { x = 1.0, y = 2.0 })\n\
         \x20   for_each_row($nudge, world, no_filters(), $Position)\n\
         \x20   held := ecs_get($Position, world, a, position)\n\
         \x20   print(\"{}\\n\", held.x)\n    ecs_free(world)\n    0\n}\n",
    )
    .unwrap();

    let run_with = |compiler: &Path, label: &str, extra: &[&str]| {
        let exe = directory.join(format!(
            "{}{}",
            unique("frost_ecs_import"),
            std::env::consts::EXE_SUFFIX
        ));
        let built = Command::new(compiler)
            .args(extra)
            .arg("-L")
            .arg(root.join("std"))
            .arg("--link")
            .arg("-o")
            .arg(&exe)
            .arg(&source)
            .env(
                "FROST_RUNTIME",
                format!("{}/runtime/frost_runtime.c", root.display()),
            )
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "{label} refused a program importing ecs.frost:\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let ran = Command::new(&exe).output().unwrap();
        let output = String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n");
        let _ = std::fs::remove_file(&exe);
        assert_eq!(
            output, "2\n",
            "{label} ran the body the wrong number of times"
        );
    };

    run_with(Path::new(env!("CARGO_BIN_EXE_frost")), "the bootstrap", &[]);
    run_with(&compiler, "the self-hosted C backend", &["--emit-c"]);
    run_with(
        &compiler,
        "the self-hosted assembly backend",
        &["--emit-asm"],
    );
    let _ = std::fs::remove_file(&source);
}

// `str` is a slice of bytes, an `if` is an expression, and a body ending in one
// answers with whichever branch ran.
#[test]
fn self_hosted_strings_and_if_expressions() {
    let source = "import \"io.frost\"\nread :: fn(s: str) -> i64 {\n\
         \x20   var i : i64 = 0\n    var negative := false\n\
         \x20   if (str_len(s) > 0 && s[0] == 45) { negative = true  i = 1 }\n\
         \x20   var value : i64 = 0\n\
         \x20   while (i < str_len(s)) {\n\
         \x20       value = value * 10 + (s[i] - 48)\n        i = i + 1\n    }\n\
         \x20   if (negative) { 0 - value } else { value }\n}\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", read(\"1234567\"))\n    print(\"{}\\n\", read(\"-7\"))\n\
         \x20   print(\"{}\\n\", read(\"0\"))\n    print(\"{}\\n\", str_len(\"abc\"))\n    0\n}\n";
    let Some(output) = selfhosted_unaudited_output("strif", source) else {
        return;
    };
    assert_eq!(output, "1234567\n-7\n0\n3\n");
}

// A `str` kept in a struct, read through the field. The C backend used to store
// the literal as a bare pointer, so the length beside it stayed zero and the
// first index aborted; the field now takes the value the way a parameter of the
// same type would.
const SELF_HOSTED_STR_FIELD: &str = "import \"io.frost\"\nDocument :: struct { source: str, at: i64 }\n\
     read_at :: fn(document: Document, index: i64) -> i64 {\n\
     \x20   document.source[index]\n}\n\
     main :: fn() -> i64 {\n\
     \x20   document := Document { source = \"hello\", at = 1 }\n\
     \x20   print(\"{}\\n\", document.source[0])\n\
     \x20   print(\"{}\\n\", document.source[document.at])\n\
     \x20   print(\"{}\\n\", read_at(document, 4))\n\
     \x20   print(\"{}\\n\", str_len(document.source))\n    0\n}\n";

#[test]
fn self_hosted_str_held_in_a_struct_is_indexable() {
    let Some(output) =
        selfhosted_unaudited_output("shstrfield", SELF_HOSTED_STR_FIELD)
    else {
        return;
    };
    assert_eq!(output, "104\n101\n111\n5\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shstrfield_input.frost");
    std::fs::write(&input, SELF_HOSTED_STR_FIELD).unwrap();
    let Some(c_source) = self_hosted_emits("shstrfield", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shstrfield", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// The self-hosted lexer dropped a semicolon with the whitespace, so `[7; 4]`
// lexed as the two elements 7 and 4 and built an array of two. The repeat form
// is read here, and an array size may name a constant the way the declared
// length already could.
const SELF_HOSTED_CONSTANT_ARRAYS: &str = "import \"io.frost\"\nCAPACITY :: 8\n\
     Buffer :: struct { bytes: [CAPACITY]u8, used: i64 }\n\
     main :: fn() -> i64 {\n\
     \x20   var a : [4]i64 = [7; 4]\n\
     \x20   print(\"{}\\n\", a[0])\n    print(\"{}\\n\", a[3])\n\
     \x20   var b : Buffer = Buffer { bytes = [0; CAPACITY], used = 0 }\n\
     \x20   b.bytes[7] = 65\n\
     \x20   print(\"{}\\n\", b.bytes[7])\n    print(\"{}\\n\", sizeof(Buffer))\n\
     \x20   print(\"{}\\n\", 1)\n    0\n}\n";

#[test]
fn self_hosted_repeats_an_array_and_sizes_it_by_a_constant() {
    let Some(output) =
        selfhosted_unaudited_output("shconstarr", SELF_HOSTED_CONSTANT_ARRAYS)
    else {
        return;
    };
    assert_eq!(output, "7\n7\n65\n16\n1\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shconstarr_input.frost");
    std::fs::write(&input, SELF_HOSTED_CONSTANT_ARRAYS).unwrap();
    let Some(c_source) = self_hosted_emits("shconstarr", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shconstarr", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// The self-hosted parser already read a field name as a source range and never
// classified it, so a keyword there always worked. Its C backend wrote the name
// through, which for a C keyword is a syntax error rather than a field.
const SELF_HOSTED_KEYWORD_FIELDS: &str = "import \"io.frost\"\nNode :: struct { struct: i64, return: i64, case: i64, int: i64 }\n\
     main :: fn() -> i64 {\n\
     \x20   var n : Node = Node { struct = 1, return = 2, case = 3, int = 4 }\n\
     \x20   n.int = 9\n\
     \x20   print(\"{}\\n\", n.struct)\n    print(\"{}\\n\", n.return)\n\
     \x20   print(\"{}\\n\", n.case)\n    print(\"{}\\n\", n.int)\n    0\n}\n";

#[test]
fn self_hosted_emits_a_field_named_for_a_c_keyword() {
    let Some(output) =
        selfhosted_unaudited_output("shkwfield", SELF_HOSTED_KEYWORD_FIELDS)
    else {
        return;
    };
    assert_eq!(output, "1\n2\n3\n9\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shkwfield_input.frost");
    std::fs::write(&input, SELF_HOSTED_KEYWORD_FIELDS).unwrap();
    let Some(c_source) = self_hosted_emits("shkwfield", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shkwfield", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

const CLI_PROGRAM: &str = "import \"io.frost\"\nfib :: fn(n: i64) -> i64 {\n\
     \x20   if (n < 2) { return n }\n\
     \x20   return fib(n - 1) + fib(n - 2)\n}\n\
     main :: fn() -> i64 {\n    print(\"{}\\n\", fib(10))\n    print(\"{}\\n\", 6 * 7)\n    0\n}\n";

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
        let exe = directory.join(format!(
            "{}{}",
            unique(&format!("frost_{label}")),
            std::env::consts::EXE_SUFFIX
        ));
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
    // The math functions `std/math.frost` reaches live in libm where the
    // platform keeps them out of the C runtime, which is Linux and the BSDs.
    // Both compilers pass this on their own link paths; a test that links what
    // one of them emitted has to pass it too, and only a platform that keeps
    // them apart says so.
    let compile = Command::new(compiler)
        .arg("-std=c11")
        .arg(&c_path)
        .arg(&runtime)
        .arg(frost_runtime_object())
        .arg("-o")
        .arg(&exe_path)
        .arg("-lm")
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

// The compiler that ships is built through the bootstrap's C backend, which is
// what `just selfhost-build` asks for, because the compiler that comes out is
// two and a half times faster than the same source through Cranelift. Both
// fixpoints build their first stage through Cranelift, and every other helper
// here does too, so without this nothing in the suite would build a compiler
// the way the one people run is built. Two routes to the same compiler have to
// answer the same, and a difference is a miscompilation in whichever of them is
// wrong.
#[test]
fn both_routes_build_the_same_compiler() {
    if c_compiler().is_none() || !linker_available() {
        return;
    }
    let directory = std::env::temp_dir();
    let source = self_hosted_source();
    let frost = env!("CARGO_BIN_EXE_frost");
    let mut emitted: Vec<String> = Vec::new();
    for (route, through_c) in [("cranelift", false), ("throughc", true)] {
        let compiler = directory.join(format!(
            "{}{}",
            unique(&format!("frost_route_{route}")),
            std::env::consts::EXE_SUFFIX
        ));
        let mut build = Command::new(frost);
        build.arg("--link");
        if through_c {
            build.arg("--emit-c");
        }
        let built = build
            .arg("-o")
            .arg(&compiler)
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "the {route} route did not build a compiler:\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let run = Command::new(&compiler)
            .env("FROST_BACKEND", "asm")
            .env("FROST_INPUT", &source)
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "the compiler the {route} route built failed on its own source:\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        emitted
            .push(String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"));
    }
    assert!(
        emitted[0].lines().count() > 10000,
        "assembly for the compiler implausibly small ({} lines)",
        emitted[0].lines().count()
    );
    assert_eq!(
        emitted[0], emitted[1],
        "the two routes to a compiler disagree about what its own source compiles to"
    );
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
    let assembled = Command::new(c_compiler().unwrap())
        .arg(&asm_path)
        .arg(runtime_object())
        .arg(frost_runtime_object())
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

    let program = "import \"io.frost\"\nfib :: fn(n: i64) -> i64 {\n\
                   \x20   if (n < 2) { return n }\n\
                   \x20   return fib(n - 1) + fib(n - 2)\n}\n\
                   main :: fn() -> i64 {\n\
                   \x20   var i : i64 = 0\n\
                   \x20   while (i < 10) {\n        print(\"{}\\n\", fib(i))\n        i = i + 1\n    }\n\
                   \x20   print(\"{}\\n\", 6 * 7)\n    0\n}\n";
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
    // With the runtime, since a program that does arithmetic names the check
    // the way one that indexes names the bounds check.
    let assembled = Command::new(c_compiler().unwrap())
        .arg(&asm_path)
        .arg(runtime_object())
        .arg(frost_runtime_object())
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

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&asm_path);
    let _ = std::fs::remove_file(&exe_path);

    assert_eq!(output, "0\n1\n1\n2\n3\n5\n8\n13\n21\n34\n42\n");
}

// Put a program through the self-hosted compiler's native backend, assemble the
// result and run it, returning what it printed. Nothing here goes through a C
// compiler except the assembler and linker.
fn selfhosted_unaudited_output(name: &str, source: &str) -> Option<String> {
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
    // The emitted code calls into the runtime for the bounds check an index
    // compiles to and for assertions, so the runtime is linked alongside it,
    // as the object every test in this binary shares.
    let runtime = runtime_object();
    let assembled = Command::new(c_compiler().unwrap())
        .arg(&asm_path)
        .arg(&runtime)
        .arg(frost_runtime_object())
        // libm, for a program that reaches std/math.frost's sqrtf and the rest.
        // Needed on Linux, a harmless no-op on macOS and mingw.
        .arg("-lm")
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

// Like selfhosted_native_output, but for a program meant to abort: the run is
// allowed to fail and its exit status and standard error are returned so a
// caller can assert on the runtime's abort message.
fn selfhosted_native_status(
    name: &str,
    source: &str,
) -> Option<(bool, String)> {
    let compiler = build_self_hosted_compiler(name)?;
    let directory = std::env::temp_dir();
    let input = directory.join(format!("frost_ns_{name}.frost"));
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

    let asm_path = directory.join(format!("frost_ns_{name}.s"));
    let exe_path = directory
        .join(format!("frost_ns_{name}{}", std::env::consts::EXE_SUFFIX));
    std::fs::write(&asm_path, String::from_utf8_lossy(&emit.stdout).as_ref())
        .unwrap();
    let runtime =
        format!("{}/runtime/frost_runtime.c", env!("CARGO_MANIFEST_DIR"));
    let assembled = Command::new(c_compiler().unwrap())
        .arg(&asm_path)
        .arg(&runtime)
        .arg(frost_runtime_object())
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
    let succeeded = run.status.success();
    let stderr = String::from_utf8_lossy(&run.stderr).replace("\r\n", "\n");

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&asm_path);
    let _ = std::fs::remove_file(&exe_path);
    Some((succeeded, stderr))
}

// Each language feature the native backend generates code for, checked by
// running the program it produces.
#[test]
fn native_backend_covers_the_language() {
    let cases: &[(&str, &str, &str)] = &[
        (
            "arith",
            "import \"io.frost\"\nmain :: fn() -> i64 {\n    print(\"{}\\n\", 2 + 3 * 4)\n    print(\"{}\\n\", 20 / 6)\n    print(\"{}\\n\", 20 % 6)\n    print(\"{}\\n\", -7)\n    0\n}\n",
            "14\n3\n2\n-7\n",
        ),
        (
            "compare",
            "import \"io.frost\"\nmain :: fn() -> i64 {\n    print(\"{}\\n\", 1 < 2)\n    print(\"{}\\n\", 2 <= 2)\n    print(\"{}\\n\", 3 > 4)\n    print(\"{}\\n\", 3 >= 4)\n    print(\"{}\\n\", 5 == 5)\n    print(\"{}\\n\", 5 != 5)\n    0\n}\n",
            "1\n1\n0\n0\n1\n0\n",
        ),
        (
            "recursion",
            "import \"io.frost\"\nfib :: fn(n: i64) -> i64 {\n    if (n < 2) { return n }\n    return fib(n - 1) + fib(n - 2)\n}\nmain :: fn() -> i64 {\n    print(\"{}\\n\", fib(12))\n    0\n}\n",
            "144\n",
        ),
        (
            "loops",
            "import \"io.frost\"\nmain :: fn() -> i64 {\n    var total : i64 = 0\n    var i : i64 = 1\n    while (i <= 100) {\n        total = total + i\n        i = i + 1\n    }\n    print(\"{}\\n\", total)\n    0\n}\n",
            "5050\n",
        ),
        (
            "structs",
            "import \"io.frost\"\nP :: struct { x: i64, y: i64 }\nsum :: fn(q: P) -> i64 { return q.x + q.y }\nbump :: fn(mut q: P) { q.x = q.x + 100 }\nmain :: fn() -> i64 {\n    var a : P = P { x = 3, y = 4 }\n    print(\"{}\\n\", sum(a))\n    bump(a)\n    print(\"{}\\n\", a.x)\n    b := a\n    print(\"{}\\n\", b.x)\n    print(\"{}\\n\", sizeof(P))\n    0\n}\n",
            "7\n103\n103\n16\n",
        ),
        (
            "nested_structs",
            "import \"io.frost\"\nInner :: struct { a: i64, b: i64, c: i64 }\n\
             Outer :: struct { first: i64, mid: Inner, last: i64 }\n\
             main :: fn() -> i64 {\n\
             \x20   var o : Outer = Outer { first = 1, mid = Inner { a = 2, b = 3, c = 4 }, last = 5 }\n\
             \x20   print(\"{}\\n\", o.first)\n    print(\"{}\\n\", o.mid.a)\n    print(\"{}\\n\", o.mid.c)\n    print(\"{}\\n\", o.last)\n\
             \x20   print(\"{}\\n\", sizeof(Outer))\n    o.mid.b = 99\n    print(\"{}\\n\", o.mid.b)\n    print(\"{}\\n\", o.last)\n    0\n}\n",
            "1\n2\n4\n5\n40\n99\n5\n",
        ),
        (
            "pointers",
            "import \"io.frost\"\nP :: struct { x: i64, y: i64 }\nmain :: fn() -> i64 {\n    var a : P = P { x = 3, y = 4 }\n    r : ^P = ptr_to(a)\n    unsafe { print(\"{}\\n\", r^.y) }\n    unsafe { r^.y = 55 }\n    print(\"{}\\n\", a.y)\n    0\n}\n",
            "4\n55\n",
        ),
        (
            "match",
            "import \"io.frost\"\nclassify :: fn(n: i64) -> i64 {\n    var r : i64 = 0\n    match n {\n        case 0: r = 100\n        case 1: r = 200\n        case _: r = 300\n    }\n    return r\n}\nmain :: fn() -> i64 {\n    print(\"{}\\n\", classify(0))\n    print(\"{}\\n\", classify(1))\n    print(\"{}\\n\", classify(9))\n    0\n}\n",
            "100\n200\n300\n",
        ),
        (
            "manyargs",
            "import \"io.frost\"\nsix :: fn(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64) -> i64 {\n    return a + b + c + d + e + f\n}\nmain :: fn() -> i64 {\n    print(\"{}\\n\", six(1, 2, 3, 4, 5, 6))\n    print(\"{}\\n\", six(10, 20, 30, 40, 50, 60))\n    0\n}\n",
            "21\n210\n",
        ),
        (
            "nested_calls",
            "import \"io.frost\"\nadd :: fn(a: i64, b: i64) -> i64 { return a + b }\nmain :: fn() -> i64 {\n    print(\"{}\\n\", add(add(1, 2), add(3, 4)))\n    0\n}\n",
            "10\n",
        ),
    ];

    for (name, source, expected) in cases {
        let Some(output) = selfhosted_unaudited_output(name, source) else {
            return;
        };
        assert_eq!(&output, expected, "native backend output for {name}");
    }
}

// Build the self-hosted compiler, feed it a program, and return what it wrote to stderr after
// rejecting it. The self-hosted compiler answers for its own errors rather than deferring them
// to whatever compiles its output.
// A program the compiler refuses has to be refused whichever backend was asked
// for. The type rules belong to the language, not to the C emitter that used to
// hold them, so this runs each rejected program through both backends and
// insists on the same answer from each.
#[test]
fn both_self_hosted_backends_refuse_the_same_programs() {
    let cases = [
        // An argument of the wrong type.
        (
            "P :: struct { x: i64 }\n\
          take :: fn(n: i64) -> i64 { n }\n\
          main :: fn() -> i64 { p := P { x = 1 }  return take(p) }\n",
            "is what is wanted here",
        ),
        // A value of the wrong type assigned to a place.
        (
            "P :: struct { x: i64 }\n\
          main :: fn() -> i64 { var n : i64 = 0  p := P { x = 1 }  n = p  0 }\n",
            "this place is a",
        ),
        // A returned value of the wrong type.
        (
            "P :: struct { x: i64 }\n\
          bad :: fn() -> i64 { p := P { x = 1 }  return p }\n\
          main :: fn() -> i64 { bad() }\n",
            "wrong type",
        ),
        // A binding that said what it is.
        (
            "Meters :: distinct i64\n\
          Feet :: distinct i64\n\
          main :: fn() -> i64 { f : Feet = 4  m : Meters = f  0 }\n",
            "a distinct type is not its representation",
        ),
    ];
    let Some(compiler) = build_self_hosted_compiler("bothrefuse") else {
        return;
    };
    let directory = std::env::temp_dir();
    for (index, (source, expected)) in cases.iter().enumerate() {
        let input = directory.join(format!("frost_bothrefuse{index}.frost"));
        std::fs::write(&input, source).unwrap();
        for backend in ["", "asm"] {
            let run = Command::new(&compiler)
                .env("FROST_BACKEND", backend)
                .env("FROST_INPUT", &input)
                .output()
                .unwrap();
            assert!(
                !run.status.success(),
                "case {index} was accepted with FROST_BACKEND='{backend}'"
            );
            let said = String::from_utf8_lossy(&run.stderr);
            assert!(
                said.contains(expected),
                "case {index} with FROST_BACKEND='{backend}' said:\n{said}"
            );
        }
        let _ = std::fs::remove_file(&input);
    }
}

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

    assert!(
        !run.status.success(),
        "expected the self-hosted compiler to reject the program"
    );

    // The bootstrap refuses it too, since a rule belongs to the language rather
    // than to one compiler. The wording stays each compiler's own; the refusal
    // is what both answer for.
    let object = directory.join(format!("frost_mfck_boot_{name}.o"));
    let bootstrap = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("-o")
        .arg(&object)
        .arg(&input)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&object);
    assert!(
        !bootstrap.status.success(),
        "the self-hosted compiler refused {name} and the bootstrap built it:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );

    Some(String::from_utf8_lossy(&run.stderr).to_string())
}

// The self-hosted compiler enforces the unsafety gate rather than only parsing
// the markers, so a program it accepts has had the same thing checked of it that
// the bootstrap checks. The helpers above turn the gate off, because most of
// those programs predate it and are about something else; this one leaves it on.
#[test]
fn self_hosted_enforces_the_unsafe_gate() {
    let Some(compiler) = build_self_hosted_compiler("gate") else {
        return;
    };
    let directory = std::env::temp_dir();
    let check = |name: &str, source: &str| -> (bool, String) {
        let input = directory.join(format!("frost_gate_{name}.frost"));
        std::fs::write(&input, source).unwrap();
        let run = Command::new(&compiler)
            .env("FROST_INPUT", &input)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        (
            run.status.success(),
            String::from_utf8_lossy(&run.stderr).to_string(),
        )
    };

    let (ok, message) = check(
        "extern",
        "puts :: extern fn(s: ^i8) -> i64\nmain :: fn() -> i64 { puts(\"hi\")  0 }\n",
    );
    assert!(!ok, "a bare C call should be gated");
    assert!(
        message.contains("calling the C function 'puts' is unchecked"),
        "expected the gate to name the call, got:\n{message}"
    );

    let (ok, message) = check(
        "deref",
        "main :: fn() -> i64 { var x : i64 = 1  p := ptr_to(x)  p^ }\n",
    );
    assert!(!ok, "a bare raw-pointer read should be gated");
    assert!(
        message.contains("reading through a raw pointer is unchecked"),
        "expected the gate to name the read, got:\n{message}"
    );

    // `safe extern fn` is the audit, so this one stands without a block. An
    // array knows its own length, so indexing it was never unchecked.
    let (ok, _) = check(
        "safe",
        "puts :: safe extern fn(s: ^i8) -> i64\n\
         main :: fn() -> i64 { unsafe { puts(\"hi\") }  var xs : [3]i64 = [1,2,3]  xs[1] }\n",
    );
    assert!(ok, "a safe extern and an array index need no block");

    let (ok, _) = check(
        "wrapped",
        "main :: fn() -> i64 { var x : i64 = 1  p := ptr_to(x)  unsafe { p^ } }\n",
    );
    assert!(ok, "the same read inside a block is allowed");
}

// A diagnostic names the file, the line and the column it is about, not just
// what went wrong. Every file's text is laid into one buffer, so the line has
// to be counted from where that file's own text begins.
#[test]
fn self_hosted_errors_name_a_position() {
    let source = "import \"io.frost\"\nPoint :: struct { x: i64, y: i64 }\n\
         main :: fn() -> i64 {\n\
         \x20   p := Point { x = 1, y = 2 }\n\
         \x20   print(\"{}\\n\", p.z)\n    0\n}\n";
    let Some(message) = self_hosted_rejects("position", source) else {
        return;
    };
    assert!(
        message.contains(":5:21:")
            && message.contains("print(\"{}\\n\", p.z)")
            && message.contains("^ struct 'Point' has no field 'z'"),
        "expected a located error with the source line and a caret, got:\n{message}"
    );
    assert!(
        message.contains("frost_mfck_input_position.frost:"),
        "expected the file it came from, got:\n{message}"
    );
}

// An import says what a file may name. A file used to see every exported name
// in the program, so it could call a function from a module it never imported,
// an import line could be deleted with the build still passing, and the list at
// the top of a file was not the list of what it depends on. Both compilers hold
// the line now, which is checked here through both, since they enforce it
// differently: the bootstrap compares each file's uses against what it
// imported, and the self-hosted compiler makes the import an edge that name
// resolution has to cross.
#[test]
fn a_file_may_only_name_what_it_imported() {
    let directory = std::env::temp_dir().join(unique("frost_visibility"));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();

    std::fs::write(
        directory.join("deep.frost"),
        "export deep\n\
         deep :: fn() -> i64 { 42 }\n",
    )
    .unwrap();
    // The middle module imports deep and re-exports nothing of it, which is
    // exactly the shape that used to leak.
    std::fs::write(
        directory.join("middle.frost"),
        "import \"deep.frost\"\n\
         export middle\n\
         middle :: fn() -> i64 { deep() }\n",
    )
    .unwrap();
    let entry = directory.join("app.frost");
    std::fs::write(
        &entry,
        "import \"io.frost\"\nimport \"middle.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", deep())\n\
         \x20   0\n\
         }\n",
    )
    .unwrap();

    let built = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--native")
        .arg("-o")
        .arg(directory.join("app.o"))
        .arg(&entry)
        .output()
        .unwrap();
    let message = String::from_utf8_lossy(&built.stderr).to_string();
    assert!(
        !built.status.success(),
        "expected a rejection:
{message}"
    );
    assert!(
        message.contains("does not import"),
        "the bootstrap let a file name what it did not import:\n{message}"
    );

    // The same program through the self-hosted compiler, which rejects it by
    // never finding the name rather than by comparing lists.
    if let Some(compiler) = build_self_hosted_compiler("visibility") {
        let run = Command::new(&compiler)
            .env("FROST_INPUT", &entry)
            .output()
            .unwrap();
        let said = format!(
            "{}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
        assert!(
            said.contains("undefined function 'deep'"),
            "the self-hosted compiler let a file name what it did not import:\n{said}"
        );
    }

    // Adding the import is the whole fix, and then both compile it.
    std::fs::write(
        &entry,
        "import \"io.frost\"\nimport \"middle.frost\"\n\
         import \"deep.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", deep())\n\
         \x20   0\n\
         }\n",
    )
    .unwrap();
    let exe = directory.join(format!("app{}", std::env::consts::EXE_SUFFIX));
    let built = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("-o")
        .arg(&exe)
        .arg(&entry)
        .output()
        .unwrap();
    if built.status.success() {
        let run = Command::new(&exe).output().unwrap();
        assert_eq!(
            String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"),
            "42\n"
        );
    }
    let _ = std::fs::remove_dir_all(&directory);
}

// A top-level form the self-hosted compiler does not implement used to fall
// through to the function parser and die inside the arena, far from the text
// that caused it. It names the declaration instead.
#[test]
fn self_hosted_rejects_an_unsupported_declaration() {
    let source = "Id :: opaque i64\nmain :: fn() -> i64 { 0 }\n";
    let Some(message) = self_hosted_rejects("unsupported_decl", source) else {
        return;
    };
    assert!(
        message.contains(":1:1:")
            && message.contains("^ this declaration is not supported yet"),
        "expected a located unsupported-declaration error, got:\n{message}"
    );
    assert!(
        !message.contains("arena was indexed out of range"),
        "an unsupported declaration should not crash the compiler:\n{message}"
    );
}

// The primitive type names are predeclared identifiers, so a local may go by
// one and the type parser still means the type. The bootstrap used to reserve
// them while the self-hosted compiler read them as names, which was a program
// one compiler accepted and the other refused; this holds the two to the same
// answer.
const PREDECLARED_NAMES: &str = "import \"io.frost\"\n\
     main :: fn() -> i64 {\n\
     \x20   i64 := 5\n\
     \x20   str := 2\n\
     \x20   usize := i64 + str\n\
     \x20   held : i64 = usize\n\
     \x20   print(\"{}\\n\", held)\n\
     \x20   0\n}\n";

#[test]
fn both_compilers_accept_a_local_named_after_a_primitive() {
    let Some(bootstrap) = bootstrap_output("predeclared", PREDECLARED_NAMES)
    else {
        return;
    };
    assert_eq!(bootstrap, "7\n");
    let Some(hosted) =
        selfhosted_unaudited_output("shpredeclared", PREDECLARED_NAMES)
    else {
        return;
    };
    assert_eq!(hosted, "7\n");
}

// The tour's comments claim numbers, so it is compiled and run and the
// numbers are checked.
#[test]
fn the_tour_prints_what_its_comments_claim() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let program = std::fs::read_to_string(root.join("examples/tour.frost"))
        .unwrap()
        .replace("\r\n", "\n");
    let Some(output) = compile_and_run_unaudited("tour", &program) else {
        return;
    };
    // 90 healed by 10, a Hero's 10 damage, that doubled by `round`, the party
    // walked by `for` and answered for in two values, the strongest again with
    // the total discarded, the columns container walked over the slots that
    // still hold something, a handle from another container refused, and the
    // session's id handed back by the `move` that consumed it, with the two
    // stated layouts read off between them.
    assert_eq!(
        output,
        "100\n10\n20\n52\n30\n30\n26\n0\n9\n32\n320\n32\n72\n7\n"
    );
}

// The self-hosted compiler passes a struct to C by value too, through both of
// its backends. Its C backend hands the C compiler a real struct and lets it
// apply the rule; its assembly backend has to know the rule, so it carries the
// same classification src/c_abi.rs does and splits one argument into the slots
// the target wants.
//
// Every shape here lands somewhere different: 8 bytes is one register, 4 bytes
// of float is where the two targets disagree about which register file, 12 is
// neither a register size on Windows nor two eightbytes, 16 is the last size
// System V puts in registers, and 24 and 32 are the ones it pushes onto the
// stack. `clobber` is what says the callee got a copy.
const SELF_HOSTED_BY_VALUE: &str = "import \"io.frost\"\nView :: struct { data: ^i8, len: i64 }\n\
     Pair :: struct { x: i32, y: i32 }\n\
     Single :: struct { a: f32 }\n\
     Triple :: struct { a: i32, b: i32, c: i32 }\n\
     Wide :: struct { a: i64, b: i64, c: i64, d: i64 }\n\
     Wider :: struct { a: f64, b: f64, c: f64 }\n\
     view_len :: extern fn(value v: View) -> i64\n\
     pair_sum :: extern fn(value p: Pair) -> i64\n\
     single_ten :: extern fn(value s: Single) -> i64\n\
     triple_sum :: extern fn(value t: Triple) -> i64\n\
     clobber :: extern fn(value t: Triple) -> i64\n\
     wide_sum :: extern fn(value w: Wide) -> i64\n\
     wider_sum :: extern fn(value w: Wider) -> i64\n\
     wide_after :: extern fn(before: i64, value w: Wide, after: i64) -> i64\n\
     mixed :: extern fn(before: i64, value p: Pair, after: i64) -> i64\n\
     main :: fn() -> i64 {\n\
     \x20   v := View { data = \"hello\", len = 5 }\n\
     \x20   print(\"{}\\n\", unsafe { view_len(v) })\n\
     \x20   p := Pair { x = 3, y = 4 }\n\
     \x20   print(\"{}\\n\", unsafe { pair_sum(p) })\n\
     \x20   print(\"{}\\n\", unsafe { single_ten(Single { a = 2.5 }) })\n\
     \x20   t := Triple { a = 1, b = 2, c = 3 }\n\
     \x20   print(\"{}\\n\", unsafe { triple_sum(t) })\n\
     \x20   print(\"{}\\n\", unsafe { clobber(t) })\n\
     \x20   print(\"{}\\n\", unsafe { triple_sum(t) })\n\
     \x20   print(\"{}\\n\", unsafe { mixed(7, p, 9) })\n\
     \x20   w := Wide { a = 1, b = 2, c = 3, d = 4 }\n\
     \x20   print(\"{}\\n\", unsafe { wide_sum(w) })\n\
     \x20   print(\"{}\\n\", unsafe { wider_sum(Wider { a = 1.5, b = 2.5, c = 3.0 }) })\n\
     \x20   print(\"{}\\n\", unsafe { wide_after(5, w, 6) })\n\
     \x20   0\n\
     }\n";

const BY_VALUE_LIBRARY: &str = "#include <stdint.h>\n\
     typedef struct { const char* data; int64_t len; } View;\n\
     typedef struct { int32_t x; int32_t y; } Pair;\n\
     typedef struct { float a; } Single;\n\
     typedef struct { int32_t a; int32_t b; int32_t c; } Triple;\n\
     typedef struct { int64_t a, b, c, d; } Wide;\n\
     typedef struct { double a, b, c; } Wider;\n\
     int64_t view_len(View v) { return v.len; }\n\
     int64_t pair_sum(Pair p) { return p.x + p.y; }\n\
     int64_t single_ten(Single s) { return (int64_t)(s.a * 10.0f); }\n\
     int64_t triple_sum(Triple t) { return t.a + t.b + t.c; }\n\
     int64_t clobber(Triple t) { t.a = 999; return t.a; }\n\
     int64_t wide_sum(Wide w) { return w.a + w.b + w.c + w.d; }\n\
     int64_t wider_sum(Wider w) { return (int64_t)(w.a + w.b + w.c); }\n\
     int64_t wide_after(int64_t before, Wide w, int64_t after) {\n\
     \x20   return before * 1000 + w.a + w.d + after;\n\
     }\n\
     int64_t mixed(int64_t before, Pair p, int64_t after) {\n\
     \x20   return before * 100 + p.x * 10 + p.y + after;\n\
     }\n";

// The self-hosted compiler walks a sequence too. It has no `for` of its own to
// extend, so this is the whole statement, written out at parse time as the
// index-and-bound loop it stands for. No node kind, no pass and no backend
// learns anything new, which is why one desugaring covers both backends.
const SELF_HOSTED_FOR: &str = "import \"io.frost\"\nsum_slice :: fn(xs: []i64) -> i64 {\n\
     \x20   var total : i64 = 0\n\
     \x20   for value in xs {\n\
     \x20       total = total + value\n\
     \x20   }\n\
     \x20   total\n\
     }\n\
     main :: fn() -> i64 {\n\
     \x20   var numbers : [4]i64 = [10, 20, 30, 40]\n\
     \x20   var total : i64 = 0\n\
     \x20   for value in numbers {\n\
     \x20       total = total + value\n\
     \x20   }\n\
     \x20   print(\"{}\\n\", total)\n\
     \x20   print(\"{}\\n\", sum_slice(numbers))\n\
     \x20   var weighted : i64 = 0\n\
     \x20   for index, value in numbers {\n\
     \x20       weighted = weighted + index * value\n\
     \x20   }\n\
     \x20   print(\"{}\\n\", weighted)\n\
     \x20   var bytes : i64 = 0\n\
     \x20   for byte in \"abc\" {\n\
     \x20       bytes = bytes + byte\n\
     \x20   }\n\
     \x20   print(\"{}\\n\", bytes)\n\
     \x20   0\n\
     }\n";

#[test]
fn self_hosted_for_walks_a_sequence() {
    let Some(output) = selfhosted_unaudited_output("shfor", SELF_HOSTED_FOR)
    else {
        return;
    };
    assert_eq!(output, "100\n100\n200\n294\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shfor_input.frost");
    std::fs::write(&input, SELF_HOSTED_FOR).unwrap();
    let Some(c_source) = self_hosted_emits("shfor", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shfor", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// The self-hosted compiler returns several values too. A return type list
// becomes
// a struct made fresh for that function, `return a, b` becomes a literal of it,
// and the binding becomes a temporary and a field read per name. That is the
// whole feature, written out at parse time, so both backends see the struct
// return they already handle.
const SELF_HOSTED_MULTIPLE_RETURNS: &str =
    "import \"io.frost\"\ndivide :: fn(a: i64, b: i64) -> (quotient: i64, remainder: i64) {
         return a / b, a % b
     }
     split_bytes :: fn(value: i64) -> (high: i64, low: i64) {
         return { high = value / 256, low = value % 256 }
     }
     classify :: fn(value: i64) -> (size: i64, negative: bool) {
         if (value < 0) {
             return 0 - value, true
         }
         return value, false
     }
     main :: fn() -> i64 {
         quotient, remainder := divide(17, 5)
         print(\"{}\\n\", quotient)
         print(\"{}\\n\", remainder)
         high, low := split_bytes(700)
         print(\"{}\\n\", high)
         print(\"{}\\n\", low)
         magnitude, var negative := classify(-9)
         print(\"{}\\n\", magnitude)
         if (negative) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }
         negative = false
         if (negative) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }
         0
     }
";

#[test]
fn self_hosted_returns_several_values() {
    let Some(output) =
        selfhosted_unaudited_output("shmulti", SELF_HOSTED_MULTIPLE_RETURNS)
    else {
        return;
    };
    assert_eq!(output, "3\n2\n2\n188\n9\n1\n0\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shmulti_input.frost");
    std::fs::write(&input, SELF_HOSTED_MULTIPLE_RETURNS).unwrap();
    let Some(c_source) = self_hosted_emits("shmulti", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shmulti", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// A function chosen at the call site, in the self-hosted compiler. `$compare`
// binds to a function the way `$T` binds to a type, so the body's call to it is
// direct and specialized once per function given. The last two show why this
// matters: `order` calls a `swap` that takes only the element type, so the
// program uses two argument tuples of different shapes at once.
const SELF_HOSTED_COMPILE_TIME_FUNCTIONS: &str =
    "import \"io.frost\"\napply :: fn($f: fn(i64) -> i64, x: i64) -> i64 { f(x) }
     double :: fn(x: i64) -> i64 { x * 2 }
     negate :: fn(x: i64) -> i64 { 0 - x }
     ascending :: fn(a: i64, b: i64) -> bool { a < b }
     descending :: fn(a: i64, b: i64) -> bool { a > b }
     swap :: fn($T: Type, mut items: []T, i: i64, j: i64) {
         hold := items[i]
         items[i] = items[j]
         items[j] = hold
     }
     order :: fn($T: Type, $less: fn(T, T) -> bool, mut items: []T, count: i64) {
         var i := 1
         while (i < count) {
             var j := i
             while (j > 0 && less(items[j], items[j - 1])) {
                 swap($T, items, j, j - 1)
                 j = j - 1
             }
             i = i + 1
         }
     }
     show :: fn(items: []i64, count: i64) {
         var i := 0
         while (i < count) {
             print(\"{}\\n\", items[i])
             i = i + 1
         }
     }
     main :: fn() -> i64 {
         print(\"{}\\n\", apply($double, 21))
         print(\"{}\\n\", apply($negate, 9))
         var numbers : [5]i64 = [5, 3, 9, 1, 7]
         order($i64, $ascending, numbers, 5)
         show(numbers, 5)
         order($i64, $descending, numbers, 5)
         show(numbers, 5)
         0
     }
";

#[test]
fn self_hosted_takes_a_function_as_a_compile_time_argument() {
    let expected = "42\n-9\n1\n3\n5\n7\n9\n9\n7\n5\n3\n1\n";
    // The bootstrap first, since the point of the feature is that both
    // compilers take it.
    let Some(bootstrap) = compile_and_run_unaudited(
        "constfnparity",
        SELF_HOSTED_COMPILE_TIME_FUNCTIONS,
    ) else {
        return;
    };
    assert_eq!(bootstrap, expected);

    let Some(output) = selfhosted_unaudited_output(
        "shconstfn",
        SELF_HOSTED_COMPILE_TIME_FUNCTIONS,
    ) else {
        return;
    };
    assert_eq!(output, expected);

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shconstfn_input.frost");
    std::fs::write(&input, SELF_HOSTED_COMPILE_TIME_FUNCTIONS).unwrap();
    let Some(c_source) = self_hosted_emits("shconstfn", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shconstfn", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// The self-hosted compiler infers a variant's enum too. It resolves the tag at
// parse time, so an argument whose function is written later is left unresolved
// and patched once every signature is known.
const SELF_HOSTED_INFERRED_VARIANTS: &str =
    "import \"io.frost\"\nShape :: enum { Circle { radius: i64 }, Square { side: i64 } }
     Color :: enum { Red, Green, Blue }
     Theme :: struct { primary: Color, accent: Color }
     area :: fn(s: Shape) -> i64 {
         match s {
             case .Circle { radius }: radius * radius * 3
             case .Square { side }: side * side
         }
     }
     paint :: fn(c: Color) -> i64 {
         match c {
             case .Red: 1
             case .Green: 2
             case .Blue: 3
         }
     }
     round :: fn(r: i64) -> Shape {
         return .Circle { radius = r }
     }
     main :: fn() -> i64 {
         s : Shape = .Circle { radius = 4 }
         print(\"{}\\n\", area(s))
         print(\"{}\\n\", area(.Square { side = 5 }))
         print(\"{}\\n\", paint(.Green))
         print(\"{}\\n\", later(.Blue))
         print(\"{}\\n\", area(round(2)))
         t := Theme { primary = .Red, accent = .Blue }
         print(\"{}\\n\", paint(t.primary))
         print(\"{}\\n\", paint(t.accent))
         0
     }
     later :: fn(c: Color) -> i64 { paint(c) * 10 }
";

#[test]
fn self_hosted_infers_a_variant_enum() {
    let Some(output) =
        selfhosted_unaudited_output("shdot", SELF_HOSTED_INFERRED_VARIANTS)
    else {
        return;
    };
    assert_eq!(
        output,
        "48
25
2
30
12
1
3
"
    );

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shdot_input.frost");
    std::fs::write(&input, SELF_HOSTED_INFERRED_VARIANTS).unwrap();
    let Some(c_source) = self_hosted_emits("shdot", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shdot", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// The self-hosted compiler infers a literal's type too, through the same fixup
// an argument's variant goes through when the callee is written later.
const SELF_HOSTED_INFERRED_LITERALS: &str = "import \"io.frost\"\nPoint :: struct { x: i64, y: i64 }\n\
     Line :: struct { from: Point, to: Point }\n\
     Color :: enum { Red, Green, Blue }\n\
     Marked :: struct { at: Point, colour: Color }\n\
     sum :: fn(p: Point) -> i64 { p.x + p.y }\n\
     length_sq :: fn(l: Line) -> i64 {\n\
     \x20   dx := l.to.x - l.from.x\n\
     \x20   dy := l.to.y - l.from.y\n\
     \x20   dx * dx + dy * dy\n\
     }\n\
     origin :: fn() -> Point { return { x = 0, y = 0 } }\n\
     paint :: fn(m: Marked) -> i64 {\n\
     \x20   base := match m.colour {\n\
     \x20       case .Red: 1\n\
     \x20       case .Green: 2\n\
     \x20       case .Blue: 3\n\
     \x20   }\n\
     \x20   base * 100 + m.at.x\n\
     }\n\
     main :: fn() -> i64 {\n\
     \x20   p : Point = { x = 3, y = 4 }\n\
     \x20   print(\"{}\\n\", sum(p))\n\
     \x20   print(\"{}\\n\", sum({ x = 10, y = 20 }))\n\
     \x20   print(\"{}\\n\", length_sq({ from = { x = 0, y = 0 }, to = { x = 3, y = 4 } }))\n\
     \x20   print(\"{}\\n\", sum(origin()))\n\
     \x20   print(\"{}\\n\", paint({ at = { x = 7, y = 0 }, colour = .Green }))\n\
     \x20   print(\"{}\\n\", later({ x = 2, y = 3 }))\n\
     \x20   0\n\
     }\n\
     later :: fn(p: Point) -> i64 { sum(p) * 10 }\n";

#[test]
fn self_hosted_infers_a_literal_type() {
    let Some(output) =
        selfhosted_unaudited_output("shlit", SELF_HOSTED_INFERRED_LITERALS)
    else {
        return;
    };
    assert_eq!(output, "7\n30\n25\n0\n207\n50\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shlit_input.frost");
    std::fs::write(&input, SELF_HOSTED_INFERRED_LITERALS).unwrap();
    let Some(c_source) = self_hosted_emits("shlit", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shlit", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// The self-hosted compiler matches on several values too. It has no tuple type
// and no tuple pattern of its own: each value is bound once and each arm is the
// `and` of the tests its pattern names, so what runs is the chain of `if`s the
// patterns stand for.
const SELF_HOSTED_TUPLE_PATTERNS: &str = "import \"io.frost\"\nlabel :: fn(n: i64) -> i64 {\n\
     \x20   match (n % 3, n % 5) {\n\
     \x20       case (0, 0): 15\n\
     \x20       case (0, _): 3\n\
     \x20       case (_, 0): 5\n\
     \x20       case _: 0\n\
     \x20   }\n\
     }\n\
     // An arm that names nothing runs when the value falls past every test.\n\
     shape :: fn(a: i64, b: i64, c: i64) -> i64 {\n\
     \x20   match (a, b, c) {\n\
     \x20       case (1, 2, 3): 123\n\
     \x20       case (1, _, 3): 103\n\
     \x20       case _: 0\n\
     \x20   }\n\
     }\n\
     main :: fn() -> i64 {\n\
     \x20   print(\"{}\\n\", label(15))\n\
     \x20   print(\"{}\\n\", label(9))\n\
     \x20   print(\"{}\\n\", label(10))\n\
     \x20   print(\"{}\\n\", label(7))\n\
     \x20   print(\"{}\\n\", shape(1, 2, 3))\n\
     \x20   print(\"{}\\n\", shape(1, 9, 3))\n\
     \x20   print(\"{}\\n\", shape(4, 5, 6))\n\
     \x20   0\n\
     }\n";

#[test]
fn self_hosted_matches_several_values_at_once() {
    let Some(output) =
        selfhosted_unaudited_output("shtuple", SELF_HOSTED_TUPLE_PATTERNS)
    else {
        return;
    };
    assert_eq!(output, "15\n3\n5\n0\n123\n103\n0\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shtuple_input.frost");
    std::fs::write(&input, SELF_HOSTED_TUPLE_PATTERNS).unwrap();
    let Some(c_source) = self_hosted_emits("shtuple", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shtuple", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// The self-hosted compiler compares enums by tag too, through both backends.
// It lays an enum out as a struct with a `tag` beside every variant's fields,
// so the rewrite reads that field. A variant value is a struct literal with the
// tag written into it, and a literal is not a place either backend can address,
// so that side uses the value it was given.
const SELF_HOSTED_ENUM_EQUALITY: &str = "import \"io.frost\"\nKind :: enum { Num, Var, Bin }\n\
     Node :: struct { kind: Kind, weight: i64 }\n\
     is_var :: fn(node: Node) -> i64 {\n\
     \x20   if (node.kind == Kind::Var) { return 1 }\n\
     \x20   0\n\
     }\n\
     main :: fn() -> i64 {\n\
     \x20   a := Kind::Var\n\
     \x20   b := Kind::Var\n\
     \x20   c := Kind::Bin\n\
     \x20   if (a == b) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
     \x20   if (a == c) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
     \x20   if (a != c) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
     \x20   node := Node { kind = Kind::Var, weight = 7 }\n\
     \x20   print(\"{}\\n\", is_var(node))\n\
     \x20   held := node.kind\n\
     \x20   if (held == Kind::Var) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
     \x20   0\n\
     }\n";

#[test]
fn self_hosted_compares_enums_by_variant() {
    let Some(output) =
        selfhosted_unaudited_output("shenumeq", SELF_HOSTED_ENUM_EQUALITY)
    else {
        return;
    };
    assert_eq!(output, "1\n0\n1\n1\n1\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shenumeq_input.frost");
    std::fs::write(&input, SELF_HOSTED_ENUM_EQUALITY).unwrap();
    let Some(c_source) = self_hosted_emits("shenumeq", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shenumeq", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// The receiving direction, through both self-hosted backends: C calls a Frost
// function and hands it a struct by value. Its assembly backend has to put the
// pieces back together, which is the mirror of taking them apart at a call.
const SELF_HOSTED_CALLBACK_BY_VALUE: &str = "import \"io.frost\"\nView :: struct { data: ^i8, len: i64 }\n\
     Wide :: struct { a: i64, b: i64, c: i64, d: i64 }\n\
     install :: extern fn(f: fn(i32, value View, i64))\n\
     install_wide :: extern fn(f: fn(i32, value Wide, i64))\n\
     fire :: extern fn()\n\
     fire_wide :: extern fn()\n\
     handler :: fn(status: i32, value message: View, tail: i64) {\n\
     \x20   print(\"{}\\n\", status)\n\
     \x20   print(\"{}\\n\", message.len)\n\
     \x20   print(\"{}\\n\", tail)\n\
     }\n\
     wide_handler :: fn(status: i32, value w: Wide, tail: i64) {\n\
     \x20   print(\"{}\\n\", status)\n\
     \x20   print(\"{}\\n\", w.a + w.d)\n\
     \x20   print(\"{}\\n\", tail)\n\
     }\n\
     main :: fn() -> i64 {\n\
     \x20   unsafe { install(handler) }\n\
     \x20   unsafe { fire() }\n\
     \x20   unsafe { install_wide(wide_handler) }\n\
     \x20   unsafe { fire_wide() }\n\
     \x20   0\n\
     }\n";

const CALLBACK_BY_VALUE_LIBRARY: &str = "#include <stdint.h>\n\
     typedef struct { const char* data; int64_t len; } View;\n\
     typedef struct { int64_t a, b, c, d; } Wide;\n\
     static void (*held)(int32_t, View, int64_t);\n\
     static void (*held_wide)(int32_t, Wide, int64_t);\n\
     void install(void (*f)(int32_t, View, int64_t)) { held = f; }\n\
     void install_wide(void (*f)(int32_t, Wide, int64_t)) { held_wide = f; }\n\
     void fire(void) {\n\
     \x20   View v; v.data = \"hello\"; v.len = 5;\n\
     \x20   held(7, v, 99);\n\
     }\n\
     void fire_wide(void) {\n\
     \x20   Wide w; w.a = 1; w.b = 2; w.c = 3; w.d = 4;\n\
     \x20   held_wide(8, w, 77);\n\
     }\n";

#[test]
fn self_hosted_receives_a_struct_from_c_by_value() {
    run_self_hosted_against_c(
        "shcb",
        SELF_HOSTED_CALLBACK_BY_VALUE,
        CALLBACK_BY_VALUE_LIBRARY,
        "7\n5\n99\n8\n5\n77\n",
    );
}

// Build a Frost program with the self-hosted compiler through both of its
// backends, link it against a C library, and check what it prints.
fn run_self_hosted_against_c(
    name: &str,
    source: &str,
    library_source: &str,
    expected: &str,
) {
    let Some(cc) = c_compiler() else {
        return;
    };
    let Some(compiler) = build_self_hosted_compiler(name) else {
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let runtime = format!("{}/runtime/frost_runtime.c", root.display());
    let directory = std::env::temp_dir();

    let library =
        directory.join(format!("{}.c", unique(&format!("{name}lib"))));
    std::fs::write(&library, library_source).unwrap();
    let input = directory.join(format!("{}.frost", unique(name)));
    std::fs::write(&input, source).unwrap();

    for (backend, suffix) in [("--emit-c", "c"), ("--emit-asm", "s")] {
        let emitted = directory.join(format!("{}.{suffix}", unique(name)));
        let built = Command::new(&compiler)
            .arg(backend)
            .arg("-o")
            .arg(&emitted)
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "{name} {backend} did not emit:\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let exe = directory.join(format!(
            "{}{}",
            unique(name),
            std::env::consts::EXE_SUFFIX
        ));
        let linked = Command::new(cc)
            .arg(&emitted)
            .arg(&library)
            .arg(&runtime)
            .arg(frost_runtime_object())
            .arg("-o")
            .arg(&exe)
            .output()
            .unwrap();
        assert!(
            linked.status.success(),
            "{name} {backend} did not link:\n{}",
            String::from_utf8_lossy(&linked.stderr)
        );
        let ran = Command::new(&exe).output().unwrap();
        let output = String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n");
        assert_eq!(output, expected, "{name} {backend} disagrees");
        let _ = std::fs::remove_file(&emitted);
        let _ = std::fs::remove_file(&exe);
    }
    let _ = std::fs::remove_file(&library);
    let _ = std::fs::remove_file(&input);
}

#[test]
fn self_hosted_passes_a_struct_to_c_by_value() {
    let Some(cc) = c_compiler() else {
        return;
    };
    let Some(compiler) = build_self_hosted_compiler("byvalue") else {
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let runtime = format!("{}/runtime/frost_runtime.c", root.display());
    let directory = std::env::temp_dir();

    let library = directory.join(format!("{}.c", unique("frost_byvalue_lib")));
    std::fs::write(&library, BY_VALUE_LIBRARY).unwrap();
    let input = directory.join(format!("{}.frost", unique("frost_byvalue")));
    std::fs::write(&input, SELF_HOSTED_BY_VALUE).unwrap();

    for (label, backend, suffix) in
        [("shbvc", "--emit-c", "c"), ("shbvasm", "--emit-asm", "s")]
    {
        let emitted = directory.join(format!("{}.{suffix}", unique(label)));
        let built = Command::new(&compiler)
            .arg(backend)
            .arg("-o")
            .arg(&emitted)
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            built.status.success(),
            "{label} did not emit:\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let exe = directory.join(format!(
            "{}{}",
            unique(label),
            std::env::consts::EXE_SUFFIX
        ));
        let linked = Command::new(cc)
            .arg(&emitted)
            .arg(&library)
            .arg(&runtime)
            .arg(frost_runtime_object())
            .arg("-o")
            .arg(&exe)
            .output()
            .unwrap();
        assert!(
            linked.status.success(),
            "{label} did not link:\n{}",
            String::from_utf8_lossy(&linked.stderr)
        );
        let ran = Command::new(&exe).output().unwrap();
        let output = String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n");
        assert_eq!(
            output, "5\n7\n25\n6\n999\n6\n743\n10\n7\n5011\n",
            "{label} disagrees"
        );
        let _ = std::fs::remove_file(&emitted);
        let _ = std::fs::remove_file(&exe);
    }
    let _ = std::fs::remove_file(&library);
    let _ = std::fs::remove_file(&input);
}

// A byte that is no operator used to keep the "nothing matched" value, which is
// the end-of-file token, so the parser stopped there and the compiler wrote an
// empty program and reported success. One stray byte truncated a file in
// silence. It is a token now, refused where it stands, so the report names
// the character and the line it is on.
#[test]
fn self_hosted_rejects_a_stray_byte() {
    let source = "main :: fn() -> i64 {\n    x := 7\u{a3}\n    0\n}\n";
    let Some(message) = self_hosted_rejects("straybyte", source) else {
        return;
    };
    assert!(
        message.contains("expected a statement, found '\u{a3}'"),
        "expected the stray byte to be named, got:\n{message}"
    );
}

// Some editors write a byte-order mark at the head of a UTF-8 file. It is not
// an operator, so it used to take the same silent path as any stray byte.
#[test]
fn self_hosted_skips_a_byte_order_mark() {
    let source = "\u{feff}import \"io.frost\"\nmain :: fn() -> i64 {\n    print(\"{}\\n\", 7)\n    0\n}\n";
    let Some(output) = selfhosted_unaudited_output("bom", source) else {
        return;
    };
    assert_eq!(output, "7\n");
}

// The top-level loop stops at the first token that cannot begin a declaration.
// Stopping quietly emitted whatever had been read, so a file whose declaration
// is named after a keyword compiled to an empty program and the mistake showed
// up as a missing symbol at link time, or as nothing at all.
#[test]
fn self_hosted_rejects_a_stray_top_level_token() {
    let source = "add :: fn(a: i64) -> i64 { a }\nreturn :: fn(v: i64) -> i64 { v }\nmain :: fn() -> i64 { 0 }\n";
    let Some(message) = self_hosted_rejects("stray_top", source) else {
        return;
    };
    assert!(
        message.contains("expected a declaration head"),
        "expected the parse to stop loudly, got:\n{message}"
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
        message.contains("unknown variable"),
        "expected an unknown-variable error, got:\n{message}"
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
        message.contains("is what is wanted here"),
        "expected an argument-type error, got:\n{message}"
    );
}

#[test]
fn self_hosted_rejects_assigning_the_wrong_type() {
    let source = "P :: struct { x: i64 }\n\
                  main :: fn() -> i64 {\n\
                  \x20   p := P { x = 1 }\n\
                  \x20   var n : i64 = 0\n    n = p\n    return n\n}\n";
    let Some(message) = self_hosted_rejects("badassign", source) else {
        return;
    };
    assert!(
        message.contains("this place is a"),
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
const SELF_HOSTED_ALLOCATION_SOURCES: &str = "import \"io.frost\"\nArena :: struct { offset: i64 }\n\
     bump :: fn(mut a: Arena, amount: i64) -> i64 {\n\
     \x20   a.offset = a.offset + amount\n    a.offset\n}\n\
     take :: fn(amount: i64) -> i64 uses Arena { bump(arena, amount) }\n\
     nested :: fn() -> i64 uses Arena { take(10) + take(32) }\n\
     main :: fn() -> i64 {\n\
     \x20   var arena : Arena = Arena { offset = 0 }\n\
     \x20   var result : i64 = 0\n\
     \x20   with arena { result = nested() }\n\
     \x20   print(\"{}\\n\", result)\n    print(\"{}\\n\", arena.offset)\n    0\n}\n";

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
        selfhosted_unaudited_output("alloc", SELF_HOSTED_ALLOCATION_SOURCES)
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
                  alloc :: fn() -> ^i64 uses Arena { unsafe { ptr_to(arena^.offset) } }\n\
                  main :: fn() -> i64 {\n\
                  \x20   var arena : Arena = Arena { offset = 0 }\n\
                  \x20   var escaped : ^i64 = ptr_to(arena.offset)\n\
                  \x20   with arena { escaped = alloc() }\n\
                  \x20   unsafe { escaped^ }\n}\n";
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
                  alloc :: fn() -> ^i64 uses Arena { unsafe { ptr_to(arena^.offset) } }\n\
                  grab :: fn() -> ^i64 {\n\
                  \x20   var arena : Arena = Arena { offset = 0 }\n\
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

// The frame check reaches the self-hosted compiler too: a pointer to a local
// may not be returned, since the local's storage dies when the call returns.
#[test]
fn self_hosted_rejects_a_returned_frame_pointer() {
    let source = "grab :: fn() -> ^i64 {\n\
                  \x20   var x : i64 = 5\n\
                  \x20   ptr_to(x)\n}\n\
                  main :: fn() -> i64 { 0 }\n";
    let Some(message) = self_hosted_rejects("framereturn", source) else {
        return;
    };
    assert!(
        message.contains("into the frame of 'grab'"),
        "expected a frame-escape error, got:\n{message}"
    );
}

// A frame pointer wrapped in a struct (or array) literal escapes the same as a
// bare one, so the frame check must look inside an aggregate value. An enum
// variant is a struct literal too, so the same case covers it.
#[test]
fn self_hosted_rejects_a_frame_pointer_inside_a_struct() {
    let source = "Box :: struct { p: ^i64 }\n\
                  grab :: fn() -> Box {\n\
                  \x20   var x : i64 = 5\n\
                  \x20   return Box { p = ptr_to(x) }\n}\n\
                  main :: fn() -> i64 { 0 }\n";
    let Some(message) = self_hosted_rejects("framestruct", source) else {
        return;
    };
    assert!(
        message.contains("into the frame of 'grab'"),
        "expected a frame-escape error, got:\n{message}"
    );
}

// A borrow may be returned but not stored, so a `ref`-typed struct field is
// refused. A raw pointer field stays allowed, which is what the arena needs.
#[test]
fn self_hosted_rejects_a_reference_struct_field() {
    let source = "Bad :: struct { r: ref i64 }\n\
                  main :: fn() -> i64 { 0 }\n";
    let Some(message) = self_hosted_rejects("reffield", source) else {
        return;
    };
    assert!(
        message.contains("cannot store a reference"),
        "expected a reference-field error, got:\n{message}"
    );
}

// Exclusivity reaches the self-hosted compiler too: two mutable borrows of the
// same place in one call conflict, over the place path, not just the variable.
#[test]
fn self_hosted_rejects_overlapping_mutable_borrows() {
    let source = "Pair :: struct { x: i64, y: i64 }\n\
                  mix :: fn(mut a: i64, mut b: i64) -> i64 { a + b }\n\
                  main :: fn() -> i64 {\n\
                  \x20   var p : Pair = Pair { x = 1, y = 2 }\n\
                  \x20   mix(p.x, p.x)\n    0\n}\n";
    let Some(message) = self_hosted_rejects("exclusivity", source) else {
        return;
    };
    assert!(
        message.contains("exclusive"),
        "expected an exclusivity error, got:\n{message}"
    );
}

// A binding declared inside the region may hold a region pointer, and reading
// through it is what the region is for, so this must be accepted.
#[test]
fn self_hosted_accepts_a_region_pointer_held_inside() {
    let source = "import \"io.frost\"\nArena :: struct { offset: i64 }\n\
                  alloc :: fn() -> ^i64 uses Arena { unsafe { ptr_to(arena^.offset) } }\n\
                  main :: fn() -> i64 {\n\
                  \x20   var arena : Arena = Arena { offset = 7 }\n\
                  \x20   var result : i64 = 0\n\
                  \x20   with arena {\n        held := alloc()\n\
                  \x20       result = unsafe { held^ }\n    }\n\
                  \x20   print(\"{}\\n\", result)\n    0\n}\n";
    let Some(output) = selfhosted_unaudited_output("regionheld", source) else {
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
    let source = "import \"io.frost\"\nArena :: struct { offset: i64 }\n\
                  alloc :: fn() -> ^i64 uses Arena {\n\
                  \x20   slot := unsafe { ptr_to(arena^.offset) }\n    return slot\n}\n\
                  main :: fn() -> i64 {\n\
                  \x20   var arena : Arena = Arena { offset = 5 }\n\
                  \x20   var result : i64 = 0\n\
                  \x20   with arena {\n        held := alloc()\n\
                  \x20       result = unsafe { held^ }\n    }\n\
                  \x20   print(\"{}\\n\", result)\n    0\n}\n";
    let Some(output) = selfhosted_unaudited_output("regionhandback", source)
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
        "import \"io.frost\"\nimport \"lib/extra.frost\"\n\
         import \"lib/math.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", square(7))\n    print(\"{}\\n\", cube(3))\n\
         \x20   p := Pair { a = 4, b = 5 }\n    print(\"{}\\n\", p.a + p.b)\n    0\n}\n",
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
    assert_eq!(output, "49\n27\n9\n");
}

// Two modules exporting the same name is no longer a collision on its own: a
// file that imports one of them sees one name. It is a collision when a file
// imports both and writes the name, and the answer is to read one of them
// under another name.
#[test]
fn export_name_collision_is_rejected_by_both_compilers() {
    let directory = std::env::temp_dir().join("frost_export_collision");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("coll_a.frost"),
        "export helper\nhelper :: fn() -> i64 { 1 }\n",
    )
    .unwrap();
    std::fs::write(
        directory.join("coll_b.frost"),
        "export helper\nhelper :: fn() -> i64 { 2 }\n",
    )
    .unwrap();
    let root = directory.join("coll_main.frost");
    std::fs::write(
        &root,
        "import \"coll_a.frost\"\n\
         import \"coll_b.frost\"\n\
         main :: fn() -> i64 { helper() }\n",
    )
    .unwrap();

    // The bootstrap rejects it during import resolution.
    let bootstrap = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--emit-c")
        .arg(&root)
        .output()
        .unwrap();
    assert!(
        !bootstrap.status.success(),
        "the bootstrap accepted two modules exporting the same name"
    );
    let bootstrap_error = String::from_utf8_lossy(&bootstrap.stderr);
    assert!(
        bootstrap_error.contains("exported by two modules")
            && bootstrap_error.contains("as ..."),
        "unexpected bootstrap collision message:\n{bootstrap_error}"
    );

    // The self-hosted compiler rejects it the same way.
    let Some(compiler) = build_self_hosted_compiler("exportcollision") else {
        return;
    };
    let selfhosted = Command::new(&compiler)
        .env("FROST_INPUT", &root)
        .output()
        .unwrap();
    assert!(
        !selfhosted.status.success(),
        "the self-hosted compiler accepted two modules exporting the same name"
    );
    let selfhosted_error = String::from_utf8_lossy(&selfhosted.stderr);
    assert!(
        selfhosted_error.contains("exported by two modules")
            && selfhosted_error.contains("as ..."),
        "unexpected self-hosted collision message:\n{selfhosted_error}"
    );
}

// The escape hatch the collision message points at, in both compilers. This is
// the case the flat namespace cannot answer any other way: two libraries you
// cannot edit that export the same name.
#[test]
fn a_name_can_be_read_under_another_on_import() {
    let directory = std::env::temp_dir().join(unique("frost_rename"));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("list.frost"),
        "export insert\ninsert :: fn(value: i64) -> i64 { value * 10 }\n",
    )
    .unwrap();
    std::fs::write(
        directory.join("tree.frost"),
        "export insert\ninsert :: fn(value: i64) -> i64 { value + 1 }\n",
    )
    .unwrap();
    // A third module imports one of them and never renames anything, which is
    // what says a rename belongs to the file that wrote it.
    std::fs::write(
        directory.join("plain.frost"),
        "import \"list.frost\"\n\
         export doubled\n\
         doubled :: fn(x: i64) -> i64 { insert(x) * 2 }\n",
    )
    .unwrap();
    let root = directory.join("app.frost");
    std::fs::write(
        &root,
        "import \"io.frost\"\nimport \"list.frost\" (insert as list_insert)\n\
         import \"tree.frost\" (insert as tree_insert)\n\
         import \"plain.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", list_insert(4))\n\
         \x20   print(\"{}\\n\", tree_insert(4))\n\
         \x20   print(\"{}\\n\", doubled(4))\n\
         \x20   0\n\
         }\n",
    )
    .unwrap();

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
        "the bootstrap rejected a rename on import:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"),
        "40\n5\n80\n"
    );

    // Renaming a name the module does not export is an error that says so.
    let wrong = directory.join("wrong.frost");
    std::fs::write(
        &wrong,
        "import \"list.frost\" (missing as gone)\n\
         main :: fn() -> i64 { 0 }\n",
    )
    .unwrap();
    let rejected = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--emit-c")
        .arg(&wrong)
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr)
            .contains("does not export 'missing'"),
        "expected the missing-export diagnostic"
    );

    // The same program through the self-hosted compiler.
    if let Some(compiler) = build_self_hosted_compiler("rename") {
        let emitted = Command::new(&compiler)
            .env("FROST_INPUT", &root)
            .output()
            .unwrap();
        assert!(
            emitted.status.success(),
            "the self-hosted compiler rejected a rename on import:\n{}",
            String::from_utf8_lossy(&emitted.stderr)
        );
        let c_source = String::from_utf8_lossy(&emitted.stdout).to_string();
        if let Some(output) = compile_c_and_run("rename", &c_source) {
            assert_eq!(output, "40\n5\n80\n");
        }
    }
    let _ = std::fs::remove_dir_all(&directory);
}

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
        "import \"io.frost\"\nimport \"second.frost\"\n\
         alpha :: fn() -> i64 { 3 }\n\
         main :: fn() -> i64 { print(\"{}\\n\", alpha() + beta())\n    0 }\n",
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
         main :: fn() -> i64 { h := make(7)  unsafe { printf(\"%lld\\n\", h.v) }  0 }\n",
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

// A constant standing for a constant another module exports, which is what
// declaring one is for: a name for a number a C header wrote down, given once
// where the binding is and used by everything that imports it.
//
// Worth its own case because the substitution and the import are separate
// mechanisms and a name that resolves inside one file says nothing about a name
// that arrives through an export list.
#[test]
fn a_constant_may_stand_for_one_from_another_module() {
    let directory = std::env::temp_dir().join("frost_const_alias");
    let library = directory.join("lib");
    std::fs::create_dir_all(&library).unwrap();
    std::fs::write(
        library.join("formats.frost"),
        "export TEXTURE_DEPTH24\n\
         TEXTURE_DEPTH24 :: 46\n",
    )
    .unwrap();
    let root = directory.join("const_alias_app.frost");
    std::fs::write(
        &root,
        "import \"io.frost\"\nimport \"lib/formats.frost\"\n\
         DEPTH :: TEXTURE_DEPTH24\n\
         main :: fn() -> i64 {\n    print(\"{}\\n\", DEPTH)\n    0\n}\n",
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
        "a constant naming an imported constant did not compile:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let ran = Command::new(&exe).output().unwrap();
    let output = String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n");

    // The self-hosted compiler answers the same, since a constant is a constant
    // in one language rather than in one compiler.
    let hosted = build_self_hosted_compiler("constalias").map(|compiler| {
        let hosted_exe =
            directory.join(format!("hosted{}", std::env::consts::EXE_SUFFIX));
        let emitted = Command::new(&compiler)
            .arg("--link")
            .arg("-o")
            .arg(&hosted_exe)
            .arg(&root)
            .output()
            .unwrap();
        assert!(
            emitted.status.success(),
            "the self-hosted compiler refused the imported constant:\n{}",
            String::from_utf8_lossy(&emitted.stderr)
        );
        let ran = Command::new(&hosted_exe).output().unwrap();
        String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n")
    });

    let _ = std::fs::remove_dir_all(&directory);
    assert_eq!(output, "46\n");
    if let Some(hosted) = hosted {
        assert_eq!(hosted, "46\n", "the two compilers disagreed");
    }
}

// A stated layout crosses a module boundary. Every top-level name is mangled
// when a module is spliced, and packing is recorded against the name, so the
// record has to be renamed with it: left behind, the `packed struct` below was
// laid out as an ordinary one and answered twelve where the file declaring it
// answered nine, with nothing saying the two disagreed. A stated `align`
// travels on the field and never had the fault, which is what makes the pair
// worth printing together.
#[test]
fn a_stated_layout_crosses_a_module() {
    let directory = std::env::temp_dir().join("frost_layout_module");
    let library = directory.join("lib");
    std::fs::create_dir_all(&library).unwrap();
    std::fs::write(
        library.join("shape.frost"),
        "export Header, Block\n\
         Header :: packed struct { magic: u32, kind: u8, length: u32 }\n\
         Block :: struct { flag: u8, weight: i64 align(16) }\n",
    )
    .unwrap();
    let root = directory.join("layout_app.frost");
    std::fs::write(
        &root,
        "import \"io.frost\"\nimport \"lib/shape.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", sizeof(Header))\n\
         \x20   print(\"{}\\n\", sizeof(Block))\n\
         \x20   for field in fields(Header) { print(\"{}\\n\", offset_of(field)) }\n\
         \x20   for field in fields(Block) { print(\"{}\\n\", offset_of(field)) }\n\
         \x20   0\n}\n",
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
        "a stated layout in an imported module did not compile:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let ran = Command::new(&exe).output().unwrap();
    let output = String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n");

    let hosted = build_self_hosted_compiler("layoutmodule").map(|compiler| {
        let hosted_exe =
            directory.join(format!("hosted{}", std::env::consts::EXE_SUFFIX));
        let emitted = Command::new(&compiler)
            .arg("--link")
            .arg("-o")
            .arg(&hosted_exe)
            .arg(&root)
            .output()
            .unwrap();
        assert!(
            emitted.status.success(),
            "the self-hosted compiler refused the imported layout:\n{}",
            String::from_utf8_lossy(&emitted.stderr)
        );
        let ran = Command::new(&hosted_exe).output().unwrap();
        String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n")
    });

    let _ = std::fs::remove_dir_all(&directory);
    assert_eq!(output, "9\n32\n0\n4\n5\n0\n16\n");
    if let Some(hosted) = hosted {
        assert_eq!(
            hosted, "9\n32\n0\n4\n5\n0\n16\n",
            "the two compilers disagreed"
        );
    }
}

// A compile-time call names a function the file can name, which is what it
// declares and what the files it imports export. The two compilers arrive at
// that rule from opposite ends: the self-hosted compiler holds every module in
// one token stream and asks the same visibility question every other name asks,
// while the bootstrap lexes each imported file before the parse and keeps the
// exported bodies. A private name in an imported module is in neither.
#[test]
fn a_compile_time_call_crosses_a_module() {
    let directory = std::env::temp_dir().join("frost_const_module");
    let library = directory.join("lib");
    std::fs::create_dir_all(&library).unwrap();
    std::fs::write(
        library.join("sizes.frost"),
        "export round_up, next_power_of_two\n\
         round_up :: fn(value: i64, to: i64) -> i64 {\n\
         \x20   (value + to - 1) / to * to\n}\n\
         next_power_of_two :: fn(n: i64) -> i64 {\n\
         \x20   var held : i64 = 1\n\
         \x20   while (held < n) { held = held * 2 }\n\
         \x20   held\n}\n",
    )
    .unwrap();
    let root = directory.join("const_app.frost");
    std::fs::write(
        &root,
        "import \"io.frost\"\nimport \"lib/sizes.frost\"\n\
         LANES :: round_up(300, 64)\n\
         Buffer :: struct { bytes: [next_power_of_two(300)]u8 }\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", LANES)\n\
         \x20   print(\"{}\\n\", sizeof(Buffer))\n\
         \x20   0\n}\n",
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
        "a compile-time call into an imported module did not compile:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let ran = Command::new(&exe).output().unwrap();
    let output = String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n");

    let hosted = build_self_hosted_compiler("constmodule").map(|compiler| {
        let hosted_exe =
            directory.join(format!("hosted{}", std::env::consts::EXE_SUFFIX));
        let emitted = Command::new(&compiler)
            .arg("--link")
            .arg("-o")
            .arg(&hosted_exe)
            .arg(&root)
            .output()
            .unwrap();
        assert!(
            emitted.status.success(),
            "the self-hosted compiler refused the imported call:\n{}",
            String::from_utf8_lossy(&emitted.stderr)
        );
        let ran = Command::new(&hosted_exe).output().unwrap();
        String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n")
    });

    let _ = std::fs::remove_dir_all(&directory);
    assert_eq!(output, "320\n512\n");
    if let Some(hosted) = hosted {
        assert_eq!(hosted, "320\n512\n", "the two compilers disagreed");
    }
}

// A constant that reaches itself has no value, and substituting one never
// finishes: the compiler recursed until the stack ran out and reported nothing.
// It is refused now, by name, before anything is lowered.
#[test]
fn a_constant_defined_in_terms_of_itself_is_refused() {
    let directory = std::env::temp_dir();
    let input = directory
        .join(unique("frost_const_cycle"))
        .with_extension("frost");
    std::fs::write(
        &input,
        "import \"io.frost\"\nFIRST :: SECOND + 1\n\
         SECOND :: FIRST + 1\n\
         main :: fn() -> i64 {\n    print(\"{}\\n\", FIRST)\n    0\n}\n",
    )
    .unwrap();
    let exe = directory
        .join(unique("frost_const_cycle"))
        .with_extension(std::env::consts::EXE_EXTENSION);

    let built = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("-o")
        .arg(&exe)
        .arg(&input)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&exe);
    assert!(
        !built.status.success(),
        "a constant defined in terms of itself compiled"
    );
    let complaint = String::from_utf8_lossy(&built.stderr);
    assert!(
        complaint.contains("defined in terms of itself"),
        "the refusal did not say what was wrong:\n{complaint}"
    );
    assert!(
        complaint.contains("FIRST") && complaint.contains("SECOND"),
        "the refusal did not name the constants in the cycle:\n{complaint}"
    );
}

// A value moved inside a call argument is moved. The ownership pass once
// walked past the `print` statement entirely, so that was the one place a
// use-after-move went unnoticed, and the compiler that ships caught it while
// the bootstrap did not.
#[test]
fn a_value_moved_inside_a_call_argument_is_moved() {
    let directory = std::env::temp_dir();
    let input = directory.join(unique("frost_move")).with_extension("frost");
    std::fs::write(
        &input,
        "import \"io.frost\"\nHeld :: struct { value: i64 }\n\
         take :: fn(move h: Held) -> i64 { h.value }\n\
         main :: fn() -> i64 {\n\
         \x20   a := Held { value = 1 }\n\
         \x20   print(\"{}\\n\", take(a))\n\
         \x20   print(\"{}\\n\", a.value)\n    0\n}\n",
    )
    .unwrap();
    let exe = directory
        .join(unique("frost_move"))
        .with_extension(std::env::consts::EXE_EXTENSION);
    let built = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("-o")
        .arg(&exe)
        .arg(&input)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&exe);
    assert!(!built.status.success(), "a use after move compiled");
    let complaint = String::from_utf8_lossy(&built.stderr);
    assert!(
        complaint.contains("use of moved value 'a'"),
        "the refusal did not say what was wrong:\n{complaint}"
    );

    // And the compiler that ships says the same thing, which is the point:
    // what a language accepts is what both of them accept.
    if let Some(compiler) = build_self_hosted_compiler("moveprint") {
        let emitted = directory.join(unique("frost_move")).with_extension("c");
        let hosted = Command::new(&compiler)
            .arg("--emit-c")
            .arg("-o")
            .arg(&emitted)
            .arg(&input)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&emitted);
        assert!(!hosted.status.success());
        assert!(
            String::from_utf8_lossy(&hosted.stderr)
                .contains("use of moved value 'a'")
        );
    }
    let _ = std::fs::remove_file(&input);
}

// A line table, so a debugger can turn an address back into the line it came
// from. `-g` rather than always, because the paths in it are absolute and a
// build that carries them is not the same bytes on another machine, which is
// what the self-hosting fixpoint is about.
#[test]
fn the_assembly_backend_writes_a_line_table_when_asked() {
    let Some(compiler) = build_self_hosted_compiler("debuglines") else {
        return;
    };
    let directory = std::env::temp_dir();
    let input = directory
        .join(unique("frost_lines"))
        .with_extension("frost");
    std::fs::write(
        &input,
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   a := 1\n\
         \x20   b := a + 2\n\
         \x20   print(\"{}\\n\", b)\n    0\n}\n",
    )
    .unwrap();

    let plain = directory.join(unique("frost_lines")).with_extension("s");
    let built = Command::new(&compiler)
        .arg("--emit-asm")
        .arg("-o")
        .arg(&plain)
        .arg(&input)
        .output()
        .unwrap();
    assert!(built.status.success());
    let text = std::fs::read_to_string(&plain).unwrap();
    assert!(
        !text.contains(".loc "),
        "a line table was written without being asked for"
    );

    let debugged = directory.join(unique("frost_lines")).with_extension("s");
    let built = Command::new(&compiler)
        .arg("-g")
        .arg("--emit-asm")
        .arg("-o")
        .arg(&debugged)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "-g was refused:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let text = std::fs::read_to_string(&debugged).unwrap();
    assert!(text.contains(".file 1 "), "no file was declared:\n{text}");
    // The three statements of the body, each naming the line it was written on.
    // The trailing `0` is the answer rather than a statement of its own, so it
    // adds no entry.
    for line in [3, 4, 5] {
        assert!(
            text.contains(&format!(".loc 1 {line} ")),
            "line {line} is not in the table"
        );
    }

    // And it assembles: a directive the assembler refuses is worse than none.
    if let Some(cc) = c_compiler() {
        let object = directory.join(unique("frost_lines")).with_extension("o");
        let assembled = Command::new(cc)
            .arg("-c")
            .arg(&debugged)
            .arg("-o")
            .arg(&object)
            .output()
            .unwrap();
        assert!(
            assembled.status.success(),
            "the line table did not assemble:\n{}",
            String::from_utf8_lossy(&assembled.stderr)
        );
        let _ = std::fs::remove_file(&object);
    }

    // The compiler's own assembler reads past them rather than refusing, so
    // there is one kind of assembly text rather than two.
    let exe = directory
        .join(unique("frost_lines"))
        .with_extension(std::env::consts::EXE_EXTENSION);
    let linked = Command::new(&compiler)
        .arg("-g")
        .arg("--link")
        .arg("-o")
        .arg(&exe)
        .arg(&input)
        .env("FROST_RUNTIME", runtime_source())
        .env("FROST_RUNTIME_FROST", frost_runtime_source())
        .output()
        .unwrap();
    assert!(
        linked.status.success(),
        "the in-process assembler refused a line table:\n{}",
        String::from_utf8_lossy(&linked.stderr)
    );
    let ran = Command::new(&exe).output().unwrap();
    assert_eq!(
        String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n"),
        "3\n"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&plain);
    let _ = std::fs::remove_file(&debugged);
    let _ = std::fs::remove_file(&exe);
}

// What the bootstrap prints when it refuses a program. It used to be one line,
// `at file:line:col: message`, naming a place the reader then had to go and
// look at. The self-hosted compiler has shown the line and a caret under the
// column for a long time and there is no reason for two formats.
#[test]
fn the_bootstrap_shows_the_line_a_failure_is_about() {
    let directory = std::env::temp_dir();
    let input = directory
        .join(unique("frost_render"))
        .with_extension("frost");
    std::fs::write(
        &input,
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   a : u8 = 300\n\
         \x20   print(\"{}\\n\", a)\n    0\n}\n",
    )
    .unwrap();
    let exe = directory
        .join(unique("frost_render"))
        .with_extension(std::env::consts::EXE_EXTENSION);
    let built = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("-o")
        .arg(&exe)
        .arg(&input)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&exe);
    assert!(!built.status.success());
    let complaint = String::from_utf8_lossy(&built.stderr);
    assert!(complaint.contains(":3:5:"), "no position in:\n{complaint}");
    assert!(
        complaint.contains("    a : u8 = 300"),
        "the line at fault was not shown:\n{complaint}"
    );
    assert!(
        complaint.contains("    ^ 300 does not fit"),
        "no caret under the column:\n{complaint}"
    );
    // The phase label and anyhow's chain are gone: the position says which
    // phase it was better than the word does.
    assert!(
        !complaint.contains("Caused by"),
        "the error still prints as a chain:\n{complaint}"
    );
}

// Where a diagnostic points. Every one of them used to land one construct late,
// because a node was stamped with the cursor when it was built, which is after
// its children had been read. An error about a binding on line five was
// reported at whatever statement came next.
//
// The blank lines matter: they are what makes a late position land somewhere
// obviously wrong rather than a line or two off.
#[test]
fn a_self_hosted_diagnostic_points_at_the_construct_it_is_about() {
    let Some(compiler) = build_self_hosted_compiler("positions") else {
        return;
    };
    let directory = std::env::temp_dir();
    for (source, line, column, carets) in [
        (
            "import \"io.frost\"\nPoint :: struct { x: i64, y: i64 }\n\
             \n\
             main :: fn() -> i64 {\n\
             \x20   p := Point { x = 1, y = 2 }\n\
             \x20   n : i64 = p\n\
             \n\
             \n\
             \x20   print(\"{}\\n\", 1)\n\
             \x20   print(\"{}\\n\", 2)\n    0\n}\n",
            6,
            5,
            "n : i64 = p",
        ),
        (
            "import \"io.frost\"\nmain :: fn() -> i64 {\n\
             \x20   a : u8 = 300\n\
             \n\
             \n\
             \x20   print(\"{}\\n\", a)\n    0\n}\n",
            3,
            14,
            "a : u8 = 300",
        ),
    ] {
        let input = directory.join(unique("frost_pos")).with_extension("frost");
        std::fs::write(&input, source).unwrap();
        let emitted = directory.join(unique("frost_pos")).with_extension("c");
        let built = Command::new(&compiler)
            .arg("--emit-c")
            .arg("-o")
            .arg(&emitted)
            .arg(&input)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&emitted);
        assert!(!built.status.success(), "this was supposed to be refused");
        let complaint = String::from_utf8_lossy(&built.stderr);
        assert!(
            complaint.contains(&format!(":{line}:{column}:")),
            "expected {line}:{column}, got:\n{complaint}"
        );
        assert!(
            complaint.contains(carets),
            "the line shown was not the one at fault:\n{complaint}"
        );
    }
}

// A literal that does not fit the type it is written at. Both compilers used to
// truncate it in silence: `a : u8 = 300` was 44 and `b : i8 = 200` was -56.
// Both agreeing about it is exactly why running them against each other could
// not see it, and the type is in hand at the point of coercion, so nothing was
// missing except the look.
#[test]
fn a_literal_that_does_not_fit_its_type_is_refused() {
    let directory = std::env::temp_dir();
    for (source, wanted) in [
        (
            "import \"io.frost\"\nmain :: fn() -> i64 {\n    a : u8 = 300\n    print(\"{}\\n\", a)\n    0\n}\n",
            "u8",
        ),
        (
            "import \"io.frost\"\nmain :: fn() -> i64 {\n    b : i8 = 200\n    print(\"{}\\n\", b)\n    0\n}\n",
            "i8",
        ),
        (
            "import \"io.frost\"\nmain :: fn() -> i64 {\n    c : i16 = 40000\n    print(\"{}\\n\", c)\n    0\n}\n",
            "i16",
        ),
        (
            "import \"io.frost\"\nmain :: fn() -> i64 {\n    d : u8 = -1\n    print(\"{}\\n\", d)\n    0\n}\n",
            "u8",
        ),
    ] {
        let input = directory
            .join(unique("frost_range"))
            .with_extension("frost");
        std::fs::write(&input, source).unwrap();
        let exe = directory
            .join(unique("frost_range"))
            .with_extension(std::env::consts::EXE_EXTENSION);
        let built = Command::new(env!("CARGO_BIN_EXE_frost"))
            .arg("--link")
            .arg("-o")
            .arg(&exe)
            .arg(&input)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&exe);
        assert!(
            !built.status.success(),
            "a literal outside {wanted} compiled:\n{source}"
        );
        let complaint = String::from_utf8_lossy(&built.stderr);
        assert!(
            complaint.contains("does not fit in a")
                && complaint.contains(wanted),
            "the refusal did not say what was wrong:\n{complaint}"
        );
    }
}

// And the same from the compiler that ships, since a language is what both of
// them accept.
#[test]
fn the_self_hosted_compiler_refuses_a_literal_that_does_not_fit() {
    let Some(compiler) = build_self_hosted_compiler("range") else {
        return;
    };
    let directory = std::env::temp_dir();
    let input = directory
        .join(unique("frost_shrange"))
        .with_extension("frost");
    std::fs::write(
        &input,
        "import \"io.frost\"\nmain :: fn() -> i64 {\n    a : u8 = 300\n    print(\"{}\\n\", a)\n    0\n}\n",
    )
    .unwrap();
    let emitted = directory.join(unique("frost_shrange")).with_extension("c");
    let built = Command::new(&compiler)
        .arg("--emit-c")
        .arg("-o")
        .arg(&emitted)
        .arg(&input)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&emitted);
    assert!(
        !built.status.success(),
        "the self-hosted compiler took a literal outside a u8"
    );
    let complaint = String::from_utf8_lossy(&built.stderr);
    assert!(
        complaint.contains("300 does not fit in a 'u8'"),
        "the refusal did not say what was wrong:\n{complaint}"
    );
}

// A conversion that cannot hold what it is given has to be written. Every one
// of these used to happen in silence at an assignment, an argument, or a
// return, so `count : i32 = total` quietly kept the low half and a float
// handed to an integer parameter quietly lost its fraction.
const NARROWING_REFUSALS: &[(&str, &str, &str)] = &[
    (
        "assign",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n    total : i64 = 5000000000\n    \
         count : i32 = total\n    print(\"{}\\n\", count)\n    0\n}\n",
        "i32",
    ),
    (
        "argument",
        "import \"io.frost\"\ntake :: fn(b: u8) -> i64 { b }\n\
         main :: fn() -> i64 {\n    n : i64 = 7\n    print(\"{}\\n\", take(n))\n    0\n}\n",
        "u8",
    ),
    (
        "return",
        "import \"io.frost\"\nshrink :: fn(n: i64) -> u16 { n }\n\
         main :: fn() -> i64 { print(\"{}\\n\", shrink(9))  0 }\n",
        "u16",
    ),
    (
        "float",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n    tall : f64 = 3.9\n    short : f32 = tall\n \
         \x20  print(\"{}\\n\", short)\n    0\n}\n",
        "f32",
    ),
    (
        "truncate",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n    tall : f64 = 3.9\n    whole : i64 = tall\n \
         \x20  print(\"{}\\n\", whole)\n    0\n}\n",
        "i64",
    ),
];

#[test]
fn a_narrowing_conversion_has_to_be_written() {
    for (name, source, wanted) in NARROWING_REFUSALS {
        let complaint = compile_error(&format!("narrow_{name}"), source);
        assert!(
            complaint.contains("which cannot hold all of one")
                && complaint.contains(&format!("write cast(${wanted}")),
            "the {name} narrowing was not refused the way it should be:\n\
             {complaint}"
        );
    }
}

// And from the compiler that ships, since a language is what both of them
// accept. One program is enough here: the refusal is one check in one place,
// and the bootstrap covers the shapes it fires on.
#[test]
fn the_self_hosted_compiler_refuses_a_narrowing_conversion() {
    let Some(compiler) = build_self_hosted_compiler("narrow") else {
        return;
    };
    let directory = std::env::temp_dir();
    let input = directory
        .join(unique("frost_shnarrow"))
        .with_extension("frost");
    std::fs::write(&input, NARROWING_REFUSALS[0].1).unwrap();
    let emitted = directory.join(unique("frost_shnarrow")).with_extension("c");
    let built = Command::new(&compiler)
        .arg("--emit-c")
        .arg("-o")
        .arg(&emitted)
        .arg(&input)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&emitted);
    assert!(
        !built.status.success(),
        "the self-hosted compiler narrowed an i64 into an i32 in silence"
    );
    let complaint = String::from_utf8_lossy(&built.stderr);
    assert!(
        complaint.contains("this is a 'i64' and a 'i32' is wanted")
            && complaint.contains("write cast($i32"),
        "the refusal did not say what was wrong:\n{complaint}"
    );
}

// `cast` is the conversion written out loud, and what it answers is what the
// hardware does: the low bits of an integer, a float truncated toward zero, a
// negative read as the unsigned pattern it already is. It never checks and
// never traps, which is the reason it has to be asked for.
const CASTS: &str = "import \"io.frost\"\nmain :: fn() -> i64 {\n\
     \x20   big : i64 = 300\n\
     \x20   small : u8 = cast($u8, big)\n\
     \x20   wide : i64 = small\n\
     \x20   print(\"{}\\n\", wide)\n\
     \x20   over : i64 = 200\n\
     \x20   signed : i8 = cast($i8, over)\n\
     \x20   print(\"{}\\n\", cast($i64, signed))\n\
     \x20   tall : f64 = 3.9\n\
     \x20   print(\"{}\\n\", cast($i64, tall))\n\
     \x20   print(\"{}\\n\", cast($i64, -tall))\n\
     \x20   span : i64 = -1\n\
     \x20   unsigned : u32 = cast($u32, span)\n\
     \x20   print(\"{}\\n\", cast($i64, unsigned))\n\
     \x20   narrow : f32 = cast($f32, tall)\n\
     \x20   print(\"{}\\n\", cast($i64, narrow * 10.0))\n\
     \x20   0\n\
     }\n";

const CAST_RESULTS: &str = "44\n-56\n3\n-3\n4294967295\n39\n";

#[test]
fn a_cast_converts_and_says_so() {
    let Some(output) = compile_and_run_unaudited("casts", CASTS) else {
        return;
    };
    assert_eq!(output, CAST_RESULTS);
    if let Some(interpreted) = run_ir_oracle("casts", CASTS) {
        assert_eq!(interpreted, CAST_RESULTS, "the ir interpreter disagrees");
    }
}

#[test]
fn the_self_hosted_compiler_casts_the_same_way() {
    let Some(output) = selfhosted_unaudited_output("shcasts", CASTS) else {
        return;
    };
    assert_eq!(output, CAST_RESULTS);

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shcasts_input.frost");
    std::fs::write(&input, CASTS).unwrap();
    let Some(c_source) = self_hosted_emits("shcasts", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shcasts", &c_source) else {
        return;
    };
    assert_eq!(via_c, CAST_RESULTS, "the self-hosted C backend disagrees");
}

// A named integer constant is a name for a number, inlined at every use, so it
// is typed where it is written exactly as the number would be. The self-hosted
// compiler used to give one an i64 of its own, which made `into[at] = CH_LF`
// into a narrowing and made every character constant unusable at a `u8`. The
// same rule types `byte + 32`: the literal has no width, so it takes the one
// beside it, and the sum stays a u8 rather than widening to an i64.
const CONSTANTS_ARE_LITERALS: &str = "import \"io.frost\"\nLIMIT :: 200\n\
     SHIFT :: 32\n\
     main :: fn() -> i64 {\n\
     \x20   top : u8 = LIMIT\n\
     \x20   print(\"{}\\n\", cast($i64, top))\n\
     \x20   var byte : u8 = 65\n\
     \x20   byte = byte + SHIFT\n\
     \x20   print(\"{}\\n\", cast($i64, byte))\n\
     \x20   room : i16 = LIMIT * 2\n\
     \x20   print(\"{}\\n\", cast($i64, room))\n\
     \x20   0\n\
     }\n";

const CONSTANT_RESULTS: &str = "200\n97\n400\n";

#[test]
fn a_named_constant_is_typed_where_it_is_written() {
    let Some(output) =
        compile_and_run_unaudited("constlit", CONSTANTS_ARE_LITERALS)
    else {
        return;
    };
    assert_eq!(output, CONSTANT_RESULTS);
}

#[test]
fn the_self_hosted_compiler_types_a_constant_where_it_is_written() {
    let Some(output) =
        selfhosted_unaudited_output("shconstlit", CONSTANTS_ARE_LITERALS)
    else {
        return;
    };
    assert_eq!(output, CONSTANT_RESULTS);

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shconstlit_input.frost");
    std::fs::write(&input, CONSTANTS_ARE_LITERALS).unwrap();
    let Some(c_source) = self_hosted_emits("shconstlit", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shconstlit", &c_source) else {
        return;
    };
    assert_eq!(
        via_c, CONSTANT_RESULTS,
        "the self-hosted C backend disagrees"
    );
}

// Being a literal cuts both ways: a constant that does not fit where it is
// written is refused there, the same as the number would be.
#[test]
fn a_named_constant_too_large_for_its_use_is_refused() {
    let source = "import \"io.frost\"\nLIMIT :: 200\n\
                  main :: fn() -> i64 {\n\
                  \x20   over : u8 = LIMIT * 2\n\
                  \x20   print(\"{}\\n\", cast($i64, over))\n\
                  \x20   0\n\
                  }\n";
    let complaint = compile_error("constrange", source);
    assert!(
        complaint.contains("400 does not fit in a") && complaint.contains("u8"),
        "the refusal did not say what was wrong:\n{complaint}"
    );
}

// A named float constant is a name for a number, so it is typed where it is
// written the way a written one is. A whole family of these: the integer form,
// the text form, and this one, all asking what the name stands for rather than
// what the name is.
//
// The mixed case belongs to it too. `0.3 + count` has no float width of its
// own, since the only float in it is the literal and the integer converts to
// whatever the context asks for, so it reads at an f32 without being narrowed.
const FLOAT_CONSTANTS: &str = "import \"io.frost\"\nLIMIT :: 1.5607963267948966
     STEP :: 0.5
     narrow :: fn(value: f32) -> i64 { cast($i64, value * 1000.0) }
     main :: fn() -> i64 {
         held : f32 = LIMIT
         print(\"{}\\n\", narrow(held))
         print(\"{}\\n\", narrow(STEP))
         print(\"{}\\n\", narrow(-STEP))
         var count : i64 = 2
         mixed : f32 = 0.25 + count
         print(\"{}\\n\", narrow(mixed))
         also : f32 = STEP * count
         print(\"{}\\n\", narrow(also))
         0
     }
";

const FLOAT_CONSTANT_RESULTS: &str = "1560
500
-500
2250
1000
";

#[test]
fn a_named_float_constant_is_typed_where_it_is_written() {
    let Some(output) = compile_and_run_unaudited("floatconst", FLOAT_CONSTANTS)
    else {
        return;
    };
    assert_eq!(output, FLOAT_CONSTANT_RESULTS);
}

#[test]
fn the_self_hosted_compiler_types_a_float_constant_the_same_way() {
    let Some(output) =
        selfhosted_unaudited_output("shfloatconst", FLOAT_CONSTANTS)
    else {
        return;
    };
    assert_eq!(output, FLOAT_CONSTANT_RESULTS);

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shfloatconst_input.frost");
    std::fs::write(&input, FLOAT_CONSTANTS).unwrap();
    let Some(c_source) = self_hosted_emits("shfloatconst", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shfloatconst", &c_source) else {
        return;
    };
    assert_eq!(
        via_c, FLOAT_CONSTANT_RESULTS,
        "the self-hosted C backend disagrees"
    );
}

// A type's name as text. The compiler knows it well enough to put in a
// diagnostic, and a registry keyed by type that wants to be readable in a file
// or a debugger wants the same string. Folded at the point the type is known,
// so a generic gets the name of what it was instantiated with rather than the
// name of its own parameter.
const TYPE_NAMES: &str =
    "import \"io.frost\"\nPosition :: struct { x: f32, y: f32 }
     Health :: struct { points: i64 }
     Meters :: distinct f32
     name_of :: fn($T: Type) -> str { typename($T) }
     width :: fn($T: Type) -> i64 { str_len(typename($T)) }
     main :: fn() -> i64 {
         print(\"{}\\n\", typename($Position))
         print(\"{}\\n\", typename($i64))
         print(\"{}\\n\", typename($Meters))
         print(\"{}\\n\", name_of($Health))
         print(\"{}\\n\", name_of($Position))
         print(\"{}\\n\", width($Health))
         held := typename($Position)
         print(\"{}\\n\", str_len(held))
         print(\"{}\\n\", held[0])
         0
     }
";

// The two struct names, a scalar, the distinct type by its own name rather
// than its representation, the same two through a generic, the length of
// "Health", the length of "Position" through a binding, and its first byte.
const TYPE_NAME_RESULTS: &str = "Position
i64
Meters
Health
Position
6
8
80
";

#[test]
fn a_type_can_be_asked_for_its_name() {
    let Some(output) = compile_and_run_unaudited("typename", TYPE_NAMES) else {
        return;
    };
    assert_eq!(output, TYPE_NAME_RESULTS);
    if let Some(interpreted) = run_ir_oracle("typename", TYPE_NAMES) {
        assert_eq!(
            interpreted, TYPE_NAME_RESULTS,
            "the ir interpreter disagrees"
        );
    }
}

#[test]
fn the_self_hosted_compiler_names_a_type_the_same_way() {
    let Some(output) = selfhosted_unaudited_output("shtypename", TYPE_NAMES)
    else {
        return;
    };
    assert_eq!(output, TYPE_NAME_RESULTS);

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shtypename_input.frost");
    std::fs::write(&input, TYPE_NAMES).unwrap();
    let Some(c_source) = self_hosted_emits("shtypename", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shtypename", &c_source) else {
        return;
    };
    assert_eq!(
        via_c, TYPE_NAME_RESULTS,
        "the self-hosted C backend disagrees"
    );
}

// A constant whose value is text is written out where it is named, so a name
// standing for one is that literal and everything a literal can do it can do.
// `str_len` was the one place that did not look through the name: it tested the
// node in hand for a literal, found a name, and fell through to reading a
// length field off a bare pointer, which the C backend spelled
// `(".debug_line")->len` and would not compile.
const NAMED_TEXT: &str = "import \"io.frost\"\nSECTION :: \".debug_line\"\n\
     EMPTY :: \"\"\n\
     width :: fn(text: str) -> i64 { str_len(text) }\n\
     main :: fn() -> i64 {\n\
     \x20   print(\"{}\\n\", str_len(SECTION))\n\
     \x20   print(\"{}\\n\", str_len(EMPTY))\n\
     \x20   print(\"{}\\n\", width(SECTION))\n\
     \x20   unsafe { print(\"{}\\n\", SECTION[1]) }\n\
     \x20   held := SECTION\n\
     \x20   print(\"{}\\n\", str_len(held))\n\
     \x20   print(\"{}\\n\", str_len(SECTION) + 1)\n\
     \x20   0\n\
     }\n";

// The length of \".debug_line\", nothing, the same length through a parameter,
// the 'd' after the dot, the length again through a binding, and one more.
const NAMED_TEXT_RESULTS: &str = "11\n0\n11\n100\n11\n12\n";

#[test]
fn a_named_text_constant_is_the_literal_it_stands_for() {
    let Some(output) = compile_and_run_unaudited("namedtext", NAMED_TEXT)
    else {
        return;
    };
    assert_eq!(output, NAMED_TEXT_RESULTS);
    if let Some(interpreted) = run_ir_oracle("namedtext", NAMED_TEXT) {
        assert_eq!(
            interpreted, NAMED_TEXT_RESULTS,
            "the ir interpreter disagrees"
        );
    }
}

#[test]
fn the_self_hosted_compiler_reads_a_named_text_constant_the_same_way() {
    let Some(output) = selfhosted_unaudited_output("shnamedtext", NAMED_TEXT)
    else {
        return;
    };
    assert_eq!(output, NAMED_TEXT_RESULTS);

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shnamedtext_input.frost");
    std::fs::write(&input, NAMED_TEXT).unwrap();
    let Some(c_source) = self_hosted_emits("shnamedtext", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shnamedtext", &c_source) else {
        return;
    };
    assert_eq!(
        via_c, NAMED_TEXT_RESULTS,
        "the self-hosted C backend disagrees"
    );
}

// A statement ends at the line break, so a `-` that opens a line negates what
// follows it rather than subtracting it from the line above. Written the other
// way round, `count = 4` followed by `-total` was one statement, `count = 4 -
// total`, and the assignment silently held a different number than the one
// written beside it.
//
// The parse is unchanged. What changed is that a `-` opening a statement which
// is not the block's own value is refused, since the only two things it can be
// are a value nobody reads and the rest of the line above. So this keeps the
// trailing `-`, which is how a subtraction spanning a line break is written,
// and `MINUS_THAT_GOES_NOWHERE` holds the shape that is refused.
const MINUS_OPENS_A_STATEMENT: &str = "import \"io.frost\"\nmain :: fn() -> i64 {\n\
     \x20   var total : i64 = 10\n\
     \x20   var count : i64 = 0\n\
     \x20   count = 4\n\
     \x20   print(\"{}\\n\", total)\n\
     \x20   print(\"{}\\n\", count)\n\
     \x20   held := count -\n\
     \x20       total\n\
     \x20   print(\"{}\\n\", held)\n\
     \x20   0\n\
     }\n";

// The same break landing before the minus, which is what a long expression
// split across lines looks like. It parses, the second line negates its own
// operand, and what it works out goes nowhere.
const MINUS_THAT_GOES_NOWHERE: &str = "import \"io.frost\"\nmain :: fn() -> i64 {\n\
     \x20   var total : i64 = 10\n\
     \x20   var count : i64 = 4\n\
     \x20   held := count\n\
     \x20       - total\n\
     \x20   print(\"{}\\n\", held)\n\
     \x20   0\n\
     }\n";

// 10 and 4, so the assignment took only its own line; then -6, so an expression
// still spans a line break when the `-` is written at the end of the first one,
// which is where it says a subtraction is meant.
const MINUS_RESULTS: &str = "10\n4\n-6\n";

#[test]
fn a_minus_that_opens_a_line_starts_a_statement() {
    let Some(output) =
        compile_and_run_unaudited("minusline", MINUS_OPENS_A_STATEMENT)
    else {
        return;
    };
    assert_eq!(output, MINUS_RESULTS);
}

// The hazard the rule above leaves, refused by both compilers. `parse_add`
// carries an expression across a line break for `+` and not for `-`, so a long
// expression broken before a minus keeps half its terms: a matrix inverse
// written that way came out right for the sparse matrices and wrong for the
// rest, and nothing pointed at the line. A block whose value is `-1` is
// untouched, which is why the rule asks whether the statement is the last one.
#[test]
fn both_compilers_refuse_a_minus_that_opens_a_line_mid_block() {
    let bootstrap = compile_error("minusdropped", MINUS_THAT_GOES_NOWHERE);
    assert!(
        bootstrap.contains("opens with '-'"),
        "the bootstrap took a minus that opens a line mid-block:\n{bootstrap}"
    );
    let Some(hosted) =
        self_hosted_rejects("minusdropped", MINUS_THAT_GOES_NOWHERE)
    else {
        return;
    };
    assert!(
        hosted.contains("opens with '-'"),
        "the self-hosted compiler took one:\n{hosted}"
    );
}

#[test]
fn the_self_hosted_compiler_reads_a_leading_minus_the_same_way() {
    let Some(output) =
        selfhosted_unaudited_output("shminusline", MINUS_OPENS_A_STATEMENT)
    else {
        return;
    };
    assert_eq!(output, MINUS_RESULTS);

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shminusline_input.frost");
    std::fs::write(&input, MINUS_OPENS_A_STATEMENT).unwrap();
    let Some(c_source) = self_hosted_emits("shminusline", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shminusline", &c_source) else {
        return;
    };
    assert_eq!(via_c, MINUS_RESULTS, "the self-hosted C backend disagrees");
}

// A `bool` is one byte. What that costs is that every read and write of one
// has to be byte wide, so three of them packed into a struct sit at offsets 0,
// 1 and 2, and writing the middle one must leave its neighbours alone.
const BOOL_IS_A_BYTE: &str = "import \"io.frost\"\nPacked :: struct { a: bool, b: bool, c: bool }\n\
     Flagged :: struct { on: bool, count: i64 }\n\
     main :: fn() -> i64 {\n\
     \x20   print(\"{}\\n\", sizeof(bool))\n\
     \x20   print(\"{}\\n\", sizeof(Packed))\n\
     \x20   print(\"{}\\n\", sizeof(Flagged))\n\
     \x20   var packed := Packed { a = true, b = false, c = true }\n\
     \x20   if (packed.a) { print(\"{}\\n\", 1) }\n\
     \x20   if (packed.b) { print(\"{}\\n\", 2) }\n\
     \x20   if (packed.c) { print(\"{}\\n\", 3) }\n\
     \x20   packed.b = true\n\
     \x20   if (packed.a) { print(\"{}\\n\", 4) }\n\
     \x20   if (packed.b) { print(\"{}\\n\", 5) }\n\
     \x20   if (packed.c) { print(\"{}\\n\", 6) }\n\
     \x20   flagged := Flagged { on = false, count = 77 }\n\
     \x20   if (flagged.on) { print(\"{}\\n\", 7) }\n\
     \x20   print(\"{}\\n\", flagged.count)\n\
     \x20   0\n\
     }\n";

const BOOL_SIZES: &str = "1\n3\n16\n1\n3\n4\n5\n6\n77\n";

#[test]
fn a_bool_is_one_byte() {
    let Some(output) = compile_and_run_unaudited("boolbyte", BOOL_IS_A_BYTE)
    else {
        return;
    };
    assert_eq!(output, BOOL_SIZES);
    if let Some(interpreted) = run_ir_oracle("boolbyte", BOOL_IS_A_BYTE) {
        assert_eq!(interpreted, BOOL_SIZES, "the ir interpreter disagrees");
    }
}

#[test]
fn the_self_hosted_compiler_agrees_a_bool_is_one_byte() {
    let Some(output) =
        selfhosted_unaudited_output("shboolbyte", BOOL_IS_A_BYTE)
    else {
        return;
    };
    assert_eq!(output, BOOL_SIZES);

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shboolbyte_input.frost");
    std::fs::write(&input, BOOL_IS_A_BYTE).unwrap();
    let Some(c_source) = self_hosted_emits("shboolbyte", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shboolbyte", &c_source) else {
        return;
    };
    assert_eq!(via_c, BOOL_SIZES, "the self-hosted C backend disagrees");
}

// With FROST_BUILD_FROM_INTERFACES an imported module contributes what its
// interface says and nothing else, so producing the same program either way is the
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
         \x20   var best := x\n    if (before(y, best)) { best = y }\n    best\n\
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
         \x20   unsafe { printf(\"%lld\\n\", area(Shape::Rect { w = 4, h = 5 })) }\n\
         \x20   report := describe(Shape::Circle { r = 2 })\n\
         \x20   unsafe { printf(\"%lld\\n\", report.value) }\n\
         \x20   unsafe { printf(\"%lld\\n\", biggest($i64, $wider, 7, 3)) }\n\
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
         \x20   unsafe { printf(\"%lld\\n\", use_one() + use_two()) }\n\
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
// happened to be reached in, and separate compilation cannot be built without
// it: a module compiled once has
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
    let program = "import \"io.frost\"\nimport \"lib/a.frost\"\n\
                   import \"lib/b.frost\"\n\
                   main :: fn() -> i64 {\n\
                   \x20   print(\"{}\\n\", helper())\n    print(\"{}\\n\", other())\n\
                   \x20   t := Thing { value = 5 }\n    print(\"{}\\n\", t.value)\n    0\n}\n";
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

// A name a file declares that one of its imports already offers. The namespace
// is flat, so there is no qualifying one of them: the bootstrap silently took
// the file's own, and the self-hosted compiler emitted both under one symbol and
// left whichever assembler read the output to notice.
#[test]
fn both_compilers_refuse_a_name_an_import_already_offers() {
    let directory = std::env::temp_dir().join(unique("frost_collide"));
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("lib.frost"),
        "export shared\n\nshared :: fn(n: i64) -> i64 { n + 1 }\n",
    )
    .unwrap();
    let main = directory.join("main.frost");
    std::fs::write(
        &main,
        "import \"io.frost\"\nimport \"lib.frost\"\n\
         shared :: fn(n: i64) -> i64 { n * 100 }\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", shared(2))\n\
         \x20   0\n}\n",
    )
    .unwrap();

    let bootstrap = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("-o")
        .arg(directory.join("m.o"))
        .arg(&main)
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&bootstrap.stderr);
    assert!(
        !bootstrap.status.success()
            && said.contains("also arrives from an import"),
        "the bootstrap took a name declared twice:\n{said}"
    );

    let Some(compiler) = build_self_hosted_compiler("collide") else {
        return;
    };
    let hosted = Command::new(&compiler)
        .env("FROST_INPUT", &main)
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&hosted.stderr);
    assert!(
        !hosted.status.success()
            && said.contains("also arrives from an import"),
        "the self-hosted compiler took it:\n{said}"
    );
    let _ = std::fs::remove_dir_all(&directory);
}

// Failure sets in the self-hosted compiler: `-> T ! E` answers with a value or
// a failure, `?` hands a failure on, and both sides come back out at the top.
const SELF_HOSTED_FAILURE_SETS: &str = "import \"io.frost\"\nOpenError :: struct { code: i64 }\n\
     halve :: fn(n: i64) -> i64 ! OpenError {\n\
     \x20   if (n % 2 != 0) { return OpenError { code = 7 } }\n\
     \x20   n / 2\n}\n\
     twice :: fn(n: i64) -> i64 ! OpenError {\n\
     \x20   a := halve(n)?\n    b := halve(a)?\n    a + b\n}\n\
     side :: fn(n: i64) -> i64 {\n\
     \x20   match twice(n) { case .Ok { value }: 0 case .Err { error }: 1 }\n}\n\
     payload :: fn(n: i64) -> i64 {\n\
     \x20   match twice(n) { case .Ok { value }: value case .Err { error }: error.code }\n}\n\
     main :: fn() -> i64 {\n\
     \x20   print(\"{}\\n\", side(8))\n    print(\"{}\\n\", payload(8))\n\
     \x20   print(\"{}\\n\", side(6))\n    print(\"{}\\n\", payload(6))\n\
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
        selfhosted_unaudited_output("failsets", SELF_HOSTED_FAILURE_SETS)
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
// the match standing for a value. The size is the tag plus the widest variant,
// not the tag plus every variant: two variants are never both live, so they sit
// at the same offset. `an_enum_is_the_same_width_under_both_compilers` is what
// holds the two compilers to that answer together.
const SELF_HOSTED_ENUMS: &str = "import \"io.frost\"\nShape :: enum {\n\
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
     \x20   print(\"{}\\n\", area(c))\n    print(\"{}\\n\", area(r))\n    print(\"{}\\n\", area(pt))\n\
     \x20   print(\"{}\\n\", sizeof(Shape))\n    0\n}\n";

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
    assert_eq!(output, "75\n24\n0\n24\n");
}

#[test]
fn self_hosted_enums_natively() {
    let Some(output) = selfhosted_unaudited_output("enums", SELF_HOSTED_ENUMS)
    else {
        return;
    };
    assert_eq!(output, "75\n24\n0\n24\n");
}

// A layout is a number both compilers have to reach on their own: the bootstrap
// computes it and hands the IR explicit offsets, the self-hosted C backend
// writes a C type and lets the C compiler place the fields, and the assembly
// backend places them itself. The self-hosted one used to lay every variant's
// fields end to end, so this program printed 32 there and 24 under the
// bootstrap. Narrow payloads are here because they are what the tag's own width
// shows up in: a four-byte tag ahead of a four-byte payload is eight bytes, and
// an eight-byte tag would be twelve rounded to sixteen.
#[test]
fn an_enum_is_the_same_width_under_both_compilers() {
    let source = "import \"io.frost\"\nShape :: enum {\n\
         \x20   Circle { radius: i64 },\n\
         \x20   Rectangle { width: i64, height: i64 },\n\
         \x20   Point,\n}\n\
         Small :: enum { One { a: i32 }, Two { b: i32, c: i32 } }\n\
         Bare :: enum { Yes, No }\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", sizeof(Shape))\n\
         \x20   print(\"{}\\n\", sizeof(Small))\n\
         \x20   print(\"{}\\n\", sizeof(Bare))\n\
         \x20   s := Shape::Rectangle { width = 4, height = 6 }\n\
         \x20   print(\"{}\\n\",
    match s {\n\
         \x20       case .Circle { radius }: radius\n\
         \x20       case .Rectangle { width, height }: width * height\n\
         \x20       case .Point: 0\n    })\n\
         \x20   t := Small::Two { b = 7, c = 9 }\n\
         \x20   print(\"{}\\n\",
    match t {\n\
         \x20       case .One { a }: a\n\
         \x20       case .Two { b, c }: b + c\n    })\n\
         \x20   0\n}\n";
    let Some(bootstrap) = bootstrap_output("enumwidth", source) else {
        return;
    };
    let Some(compiler) = build_self_hosted_compiler("enumwidth") else {
        return;
    };
    for (backend, suffix) in [("--emit-asm", "s"), ("--emit-c", "c")] {
        let hosted = selfhosted_default_output(
            &compiler,
            "enumwidth",
            source,
            backend,
            suffix,
        );
        assert_eq!(
            hosted, bootstrap,
            "the self-hosted compiler's {backend} laid an enum out differently"
        );
    }
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
    let source = "import \"io.frost\"\nmain :: fn() -> i64 {\n\
                  \x20   s : ^i8 = \"hello\"\n\
                  \x20   unsafe { print(\"{}\\n\", s[0]) }\n    unsafe { print(\"{}\\n\", s[1]) }\n\
                  \x20   unsafe { print(\"{}\\n\", s[4]) }\n\
                  \x20   print(\"{}\\n\", sizeof(i8))\n    0\n}\n";
    let Some(output) = selfhosted_unaudited_output("bytes", source) else {
        return;
    };
    assert_eq!(output, "104\n101\n111\n1\n");
}

// A `?` in a loop's condition is asked again every time round. It used to be
// lifted out of the loop and evaluated once, which read the same answer for
// ever.
#[test]
fn self_hosted_reevaluates_a_try_in_a_loop_condition() {
    let source = "import \"io.frost\"\nE :: struct { c: i64 }\n\
                  step :: fn(n: i64) -> i64 ! E {\n\
                  \x20   if (n > 3) { return E { c = 9 } }\n\
                  \x20   n + 1\n}\n\
                  run :: fn() -> i64 ! E {\n\
                  \x20   var n : i64 = 0\n\
                  \x20   while (step(n)? < 3) { n = n + 1 }\n\
                  \x20   n\n}\n\
                  got :: fn() -> i64 {\n\
                  \x20   match run() { case .Ok { value }: value case .Err { error }: -1 }\n}\n\
                  main :: fn() -> i64 {\n\
                  \x20   print(\"{}\\n\", got())\n    0\n}\n";
    let Some(output) = selfhosted_unaudited_output("trywhile", source) else {
        return;
    };
    assert_eq!(output, "2\n");
}

// A generic function used with a type that no generic struct was written with.
// Instantiation used to be driven by the struct instances alone, so this was
// called and never emitted, and the program failed to link.
#[test]
fn self_hosted_emits_a_generic_function_with_no_struct_instance() {
    let source = "import \"io.frost\"\nBox :: struct($T: Type) { value: $T }\n\
                  wrap :: fn($T: Type, v: $T) -> Box<T> { Box { value = v } }\n\
                  unwrap :: fn(b: Box<$T>) -> $T { unsafe { b^.value } }\n\
                  main :: fn() -> i64 {\n\
                  \x20   b := wrap($i64, 41)\n\
                  \x20   print(\"{}\\n\", unwrap(b) + 1)\n    0\n}\n";
    let Some(output) = selfhosted_unaudited_output("genericonly", source)
    else {
        return;
    };
    assert_eq!(output, "42\n");
}

// A `defer` says it runs where the function leaves, so written inside a block
// it means something other than it reads: it would run past the end of that
// block, and one in a loop would run once afterwards with whatever the loop left
// behind rather than once a turn. Both compilers refuse it rather than pick one
// of those. The self-hosted one used to take it and run it at function exit,
// which is the shape that made this worth a test: agreeing on what a program
// means includes agreeing on which programs there are.
#[test]
fn both_compilers_refuse_a_defer_inside_a_block() {
    let cases = [
        (
            "deferif",
            "import \"io.frost\"\ntrace :: fn(n: i64) { print(\"{}\\n\", n) }\n\
             f :: fn(x: i64) -> i64 {\n\
             \x20   if (x > 0) {\n        defer trace(1)\n    }\n\
             \x20   0\n}\n\
             main :: fn() -> i64 { print(\"{}\\n\", f(1))  0 }\n",
        ),
        (
            "deferloop",
            "import \"io.frost\"\ntrace :: fn(n: i64) { print(\"{}\\n\", n) }\n\
             f :: fn() -> i64 {\n\
             \x20   var i : i64 = 0\n\
             \x20   while (i < 2) {\n        defer trace(i)\n        i = i + 1\n    }\n\
             \x20   0\n}\n\
             main :: fn() -> i64 { print(\"{}\\n\", f())  0 }\n",
        ),
    ];
    // Both halves assert the same sentence, so neither compiler can refuse it
    // with something a reader cannot act on and still pass.
    for (name, source) in cases {
        let bootstrap = compile_error(name, source);
        assert!(
            bootstrap.contains("`defer` belongs at the top level"),
            "the bootstrap took a nested `defer` in {name}, or refused it \
             without naming it:\n{bootstrap}"
        );
        let Some(hosted) = self_hosted_rejects(name, source) else {
            return;
        };
        assert!(
            hosted.contains("`defer` belongs at the top level"),
            "the self-hosted compiler did not refuse the nested `defer` in \
             {name}:\n{hosted}"
        );
    }
}

// The other half of a loop variable. Its value carries nothing, but its address
// still names storage this frame owns, so handing that address out is refused by
// both compilers. This is the tightening that goes with the loosening in
// `a_loop_variable_read_carries_no_storage`: without it, giving the loop
// variable a value worth something would have opened the road that rule exists
// to close.
#[test]
fn both_compilers_refuse_an_address_of_a_loop_variable() {
    let cases = [
        (
            "loopaddr",
            "leak :: fn(v: []i64) -> ^i64 {\n\
             \x20   for x in v {\n        return unsafe { ptr_to(x) }\n    }\n\
             \x20   unsafe { ptr_to(v[0]) }\n}\n\
             main :: fn() -> i64 { xs := [1, 2]  p := leak(xs)  0 }\n",
        ),
        (
            "loopref",
            "pick :: fn(v: []i64) -> ref i64 {\n\
             \x20   for x in v {\n        return x\n    }\n\
             \x20   v[0]\n}\n\
             main :: fn() -> i64 { xs := [1, 2]  held := pick(xs)  0 }\n",
        ),
    ];
    // The sentence both compilers end on, rather than the word only one of them
    // opens with.
    for (name, source) in cases {
        let bootstrap = compile_error(name, source);
        assert!(
            bootstrap.contains("dies when the call returns"),
            "the bootstrap let the address of a loop variable out in \
             {name}:\n{bootstrap}"
        );
        let Some(hosted) = self_hosted_rejects(name, source) else {
            return;
        };
        assert!(
            hosted.contains("dies when the call returns"),
            "the self-hosted compiler let the address of a loop variable out \
             in {name}:\n{hosted}"
        );
    }
}

// A registration hands a callee something to hold, so what it holds has to
// outlive it. `graph_pass(g, ..., ptr_to(scene))` puts the scene in the graph,
// and a scene that dies when the registering call returns leaves the graph
// naming storage that is gone: every frame after reads whatever the machine put
// there. Neither compiler saw it, because both judged a pointer only where a
// call answered with one and never where a call was handed one.
//
// Handing it to something that dies with this frame is the ordinary case and
// both still accept it, which is what every one of the graphics examples does.
#[test]
fn both_compilers_refuse_a_pointer_kept_by_something_that_outlives_it() {
    let kept = "Slot :: struct { at: ^i64 }
         Table :: struct { held: Slot }
         put :: fn(mut t: Table, value: Slot) {
    t.held = value
}
         register :: fn(mut t: Table) {
             var n : i64 = 7
             put(t, Slot { at = ptr_to(n) })
         }
         main :: fn() -> i64 {
             var seed : i64 = 0
             var t := Table { held = Slot { at = ptr_to(seed) } }
             register(t)
             0
         }
";
    let bootstrap = compile_error("keptescape", kept);
    assert!(
        bootstrap.contains("keeps it in something that outlives this frame"),
        "the bootstrap let a kept frame pointer out:
{bootstrap}"
    );
    let Some(hosted) = self_hosted_rejects("keptescape", kept) else {
        return;
    };
    assert!(
        hosted.contains("keeps it in something that outlives this frame"),
        "the self-hosted compiler let a kept frame pointer out:
{hosted}"
    );

    // The same shape where what keeps it dies with the pointer. Both take it,
    // because the two are gone at the same moment and nothing reads either
    // after that.
    let together = "import \"io.frost\"\nSlot :: struct { at: ^i64 }
         Table :: struct { held: Slot }
         put :: fn(mut t: Table, value: Slot) {
    t.held = value
}
         main :: fn() -> i64 {
             var n : i64 = 7
             var seed : i64 = 0
             var t := Table { held = Slot { at = ptr_to(seed) } }
             put(t, Slot { at = ptr_to(n) })
             print(\"{}\\n\", n)
             0
         }
";
    assert!(
        bootstrap_output("kepttogether", together).is_some(),
        "the bootstrap refused a state that lives as long as what holds it"
    );
}

// A deferred statement is written out again at every exit and its names are
// resolved there, so a name it mentions that is bound again below it reads as
// that later binding. Both compilers took this and both got it wrong, by
// different routes and to different answers: the bootstrap and the self-hosted
// C backend printed the shadowing value, and the self-hosted assembly backend
// printed a slot the path taken never wrote, because its locals are flat and
// the last binding of a name wins. Three answers to one program, none of them
// the one the line has.
#[test]
fn both_compilers_refuse_a_defer_over_a_rebound_name() {
    let source = "import \"io.frost\"\ntrace :: fn(n: i64) { print(\"{}\\n\", n) }\n\
         shadowed :: fn(c: i64) -> i64 {\n\
         \x20   x := 1\n\
         \x20   defer trace(x)\n\
         \x20   if (c > 0) {\n        x := 99\n        trace(x)\n\
         \x20       return 0\n    }\n\
         \x20   7\n}\n\
         main :: fn() -> i64 { print(\"{}\\n\", shadowed(1))  0 }\n";
    let bootstrap = compile_error("defershadow", source);
    assert!(
        bootstrap.contains("bound again below this `defer`"),
        "the bootstrap took a `defer` over a rebound name:\n{bootstrap}"
    );
    let Some(hosted) = self_hosted_rejects("defershadow", source) else {
        return;
    };
    assert!(
        hosted.contains("bound again below the `defer`"),
        "the self-hosted compiler took a `defer` over a rebound name:\n{hosted}"
    );
}

// A `test` body is a function body and defers like one. It gets its own mark,
// and without one a deferred statement written in a test never ran, because
// nothing put the copies where that body falls off the end, and the entry stayed
// on the list for whatever was parsed next to run.
#[test]
fn a_defer_in_a_test_body_runs() {
    let source = "import \"io.frost\"\ntrace :: fn(n: i64) { print(\"{}\\n\", n) }\n\
         test \"a defer runs where the test body ends\" {\n\
         \x20   defer trace(7)\n\
         \x20   trace(1)\n}\n\
         test \"and belongs to the test that wrote it\" {\n\
         \x20   trace(2)\n}\n";
    let directory = std::env::temp_dir();
    let input = directory.join(format!("{}.frost", unique("frost_defertest")));
    std::fs::write(&input, source).unwrap();
    let exe = directory.join(format!(
        "{}{}",
        unique("frost_defertest"),
        std::env::consts::EXE_SUFFIX
    ));
    let run = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--test")
        .arg("-o")
        .arg(&exe)
        .arg(&input)
        .output()
        .unwrap();
    let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&exe);
    // 1 then 7 says the deferred statement ran where that body ended, and the
    // single 2 says it did not follow the parse into the next test.
    assert!(
        output.contains("1\n7\n") && output.contains("2\n"),
        "a `defer` in a test body did not run where the body ended:\n{output}"
    );
    assert!(
        output.matches('7').count() == 1,
        "a `defer` in one test body ran in another:\n{output}"
    );
}

// The standard library modules that carry `test` blocks, and how many each
// carries. One list rather than two that currently match: the bootstrap suite
// and the self-hosted one drifted apart by three modules, so map, slab and vec
// were compiled by one compiler and never the other.
const STD_MODULES: &[(&str, &str)] = &[
    ("arena.frost", "4 passed"),
    ("ecs.frost", "116 passed"),
    ("fixed.frost", "4 passed"),
    ("fs.frost", "2 passed"),
    ("map.frost", "13 passed"),
    ("math.frost", "33 passed"),
    ("math64.frost", "23 passed"),
    ("mem.frost", "13 passed"),
    ("slab.frost", "2 passed"),
    ("snapshot.frost", "6 passed"),
    ("sort.frost", "3 passed"),
    ("strings.frost", "12 passed"),
    ("thread.frost", "3 passed"),
    ("vec.frost", "5 passed"),
];

// The example programs, run under both compilers and compared. They are the
// longest programs in the tree that neither compiler wrote, and holding the two
// to the same answer on them is what found a `for` over a range crashing the
// self-hosted parser, two range loops in one function colliding in the emitted
// C, and an enum laid out two different ways.
//
// The list is every example, with nothing left out.
const SHARED_EXAMPLES: &[&str] = &[
    "native/allocator.frost",
    "native/arena.frost",
    "native/custom_allocator.frost",
    "native/dynamic_arena.frost",
    "native/entity_system.frost",
    "native/game_world.frost",
    "native/generic_algorithms.frost",
    "native/generic_pool_library.frost",
    "native/generic_slab.frost",
    "native/generic_stack.frost",
    "native/math_transform.frost",
    "native/native_pool.frost",
    "native/pipeline.frost",
    "native/pool_entities.frost",
    "native/pool_linked_list.frost",
    "native/pool_stress.frost",
    "native/shapes.frost",
    "native/slices.frost",
    "native/vectors.frost",
    "scratch_frame.frost",
    "selfhosted/diamond.frost",
    "tour.frost",
];

#[test]
fn both_compilers_agree_on_the_examples() {
    if c_compiler().is_none() || !linker_available() {
        return;
    }
    let Some(compiler) = build_self_hosted_compiler("sharedex") else {
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let directory = std::env::temp_dir();
    let faults = in_parallel(SHARED_EXAMPLES, |example| {
        let source = root.join("examples").join(example);
        let label = unique("frost_shared");
        let exe =
            directory.join(format!("{label}{}", std::env::consts::EXE_SUFFIX));
        let built = Command::new(env!("CARGO_BIN_EXE_frost"))
            .arg("--link")
            .arg("-o")
            .arg(&exe)
            .arg(&source)
            .output()
            .unwrap();
        if !built.status.success() {
            return Some(format!(
                "the bootstrap refused {example}:\n{}",
                String::from_utf8_lossy(&built.stderr)
            ));
        }
        let want = String::from_utf8_lossy(
            &Command::new(&exe).output().unwrap().stdout,
        )
        .replace("\r\n", "\n");
        let _ = std::fs::remove_file(&exe);

        let emitted = directory.join(format!("{label}.c"));
        let run = Command::new(&compiler)
            .arg(&source)
            .arg("-o")
            .arg(&emitted)
            .output()
            .unwrap();
        if !run.status.success() {
            return Some(format!(
                "the self-hosted compiler refused {example}:\n{}",
                String::from_utf8_lossy(&run.stderr)
            ));
        }
        let c_source = std::fs::read_to_string(&emitted).unwrap();
        let _ = std::fs::remove_file(&emitted);
        let got = compile_c_and_run(&label, &c_source)?;
        if got != want {
            return Some(format!(
                "the two compilers disagree about {example}:\n{got}{want}"
            ));
        }

        // The same program built the way a build builds one: each module to its
        // own object, then linked. A whole-program unit emits every body it
        // names, so a specialization the module that names it does not emit is
        // a name only this path is missing. That is how a `[]i64` keyed one way
        // at the call and another in the body went unseen.
        let linked = directory
            .join(format!("{label}_l{}", std::env::consts::EXE_SUFFIX));
        let built = Command::new(&compiler)
            .arg("--link")
            .arg("-o")
            .arg(&linked)
            .arg(&source)
            .output()
            .unwrap();
        if !built.status.success() {
            return Some(format!(
                "the self-hosted compiler could not link {example}:\n{}",
                String::from_utf8_lossy(&built.stderr)
            ));
        }
        let ran = String::from_utf8_lossy(
            &Command::new(&linked).output().unwrap().stdout,
        )
        .replace("\r\n", "\n");
        let _ = std::fs::remove_file(&linked);
        (ran != want).then(|| {
            format!(
                "the linked self-hosted build disagrees about {example}:\n{ran}{want}"
            )
        })
    });
    let faults: Vec<String> = faults.into_iter().flatten().collect();
    assert!(faults.is_empty(), "{}", faults.join("\n"));
}

// Everything the two self-hosted backends can express, run through both. They
// answer the same thing or one of them is wrong.
#[test]
fn self_hosted_backends_agree() {
    let source = "import \"io.frost\"\nmalloc :: extern fn(size: i64) -> ^i8\n\
         Inner :: struct { a: i64, b: i64 }\n\
         Outer :: struct { first: i64, mid: Inner, last: i64 }\n\
         Bytes :: struct { flag: i8, count: i64, mark: i8 }\n\
         Kind :: enum { None, One { x: i64 }, Two { x: i64, y: i64 } }\n\
         Box :: struct($T: Type) { value: $T }\n\
         wrap :: fn($T: Type, v: $T) -> Box<T> { Box { value = v } }\n\
         unwrap :: fn(b: Box<$T>) -> $T { unsafe { b^.value } }\n\
         sum_kind :: fn(k: Kind) -> i64 {\n\
         \x20   match k {\n        case .None: 0\n\
         \x20       case .One { x }: x\n        case .Two { x, y }: x + y\n    }\n}\n\
         bump :: fn(mut o: Outer) -> i64 {\n\
         \x20   o.mid.b = o.mid.b + 1\n    o.mid.b\n}\n\
         main :: fn() -> i64 {\n\
         \x20   var o : Outer = Outer { first = 1, mid = Inner { a = 2, b = 3 }, last = 4 }\n\
         \x20   print(\"{}\\n\", bump(o))\n    print(\"{}\\n\", o.mid.b)\n    print(\"{}\\n\", sizeof(Outer))\n\
         \x20   var bs : Bytes = Bytes { flag = 1, count = 77, mark = 2 }\n\
         \x20   print(\"{}\\n\", bs.flag)\n    print(\"{}\\n\", bs.count)\n    bs.mark = 5\n\
         \x20   print(\"{}\\n\", bs.mark)\n    print(\"{}\\n\", sizeof(Bytes))\n\
         \x20   print(\"{}\\n\", sum_kind(Kind::None))\n\
         \x20   print(\"{}\\n\", sum_kind(Kind::One { x = 6 }))\n\
         \x20   print(\"{}\\n\", sum_kind(Kind::Two { x = 6, y = 7 }))\n\
         \x20   b := wrap($i64, 41)\n    print(\"{}\\n\", unwrap(b) + 1)\n\
         \x20   buf := unsafe { malloc(8) }\n\
         \x20   unsafe { buf[0] = 65 }\n    unsafe { buf[1] = 66 }\n\
         \x20   unsafe { print(\"{}\\n\", buf[0]) }\n    unsafe { print(\"{}\\n\", buf[1]) }\n\
         \x20   var acc : i64 = 0\n    var i : i64 = 0\n\
         \x20   while (i < 5) {\n\
         \x20       if (i % 2 == 0) { acc = acc + i } else { acc = acc - i }\n\
         \x20       i = i + 1\n    }\n\
         \x20   print(\"{}\\n\", acc)\n    print(\"{}\\n\", (3 < 4) && (5 >= 5))\n\
         \x20   print(\"{}\\n\", 17 / 5)\n    print(\"{}\\n\", -17 % 5)\n    0\n}\n";
    let expected =
        "4\n4\n32\n1\n77\n5\n24\n0\n6\n13\n42\n65\n66\n2\n1\n3\n-2\n";

    let Some(native) = selfhosted_unaudited_output("agree", source) else {
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
        message.contains("not consumed"),
        "expected a linear-not-consumed error, got:\n{message}"
    );
}

// Flow-aware: a linear value consumed in only one arm of an `if` is leaked on
// the other path. The old accumulate-only check saw the one consume and read it
// as consumed; the flow-aware one asks whether every path consumes it.
#[test]
fn self_hosted_rejects_a_linear_consumed_on_one_branch() {
    let source = "File :: linear struct { h: i64 }\n\
                  close :: extern fn(move f: File)\n\
                  main :: fn() -> i64 {\n\
                  \x20   var c : i64 = 0\n\
                  \x20   r := File { h = 1 }\n\
                  \x20   if (c == 1) { close(r) } else { }\n    0\n}\n";
    let Some(message) = self_hosted_rejects("linearonebranch", source) else {
        return;
    };
    assert!(
        message.contains("not consumed"),
        "expected a linear-leak error for the untaken branch, got:\n{message}"
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
         main :: fn() -> i64 { unsafe { printf(\"%lld\\n\", triple(14)) } 0 }\n",
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
         main :: fn() -> i64 { unsafe { printf(\"%lld\\n\", area(3, 5)) } 0 }\n",
    )
    .unwrap();
    let (ok, out) = frost_compiles(&dir, "uses_public.frost");
    assert!(ok, "public import should compile: {out}");
    assert_eq!(out, "30\n");

    std::fs::write(
        dir.join("uses_private.frost"),
        "import \"lib.frost\"\n\
         printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         main :: fn() -> i64 { unsafe { printf(\"%lld\\n\", scale(10)) } 0 }\n",
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
    unsafe { puts("line one") }
    unsafe { puts("line\ttwo") }
    0
}
"#;

#[test]
fn native_strings_and_escapes() {
    let Some(output) = compile_and_run_unaudited("strings", STRINGS) else {
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
    unsafe { printf("%lld\n", n) }
    var i : i64 = 0
    while (i < n) {
        unsafe { printf("%lld\n", greeting[i]) }
        i = i + 1
    }
    unsafe { printf("%lld\n", first_byte(greeting)) }
    chosen := pick(0)
    unsafe { printf("%lld\n", str_len(chosen)) }
    unsafe { printf("%lld\n", chosen[0]) }
    other := pick(1)
    unsafe { printf("%lld\n", str_len(other)) }
    0
}
"#;

#[test]
fn native_str_is_a_length_carrying_view() {
    let Some(output) = compile_and_run_unaudited("strview", STR_VIEW) else {
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
    unsafe { Arena { data = malloc(cap), cap = cap, offset = 0 } }
}

arena_destroy :: fn(move a: Arena) { unsafe { free(a.data) } }

alloc_int :: fn(mut a: Arena) -> ^i64 {
    slot := unsafe { ptr_to(a.data[a.offset]) }
    a.offset = a.offset + sizeof(i64)
    unsafe { ptr_cast($i64, slot) }
}

main :: fn() -> i64 {
    var a := arena_new(256)
    p := alloc_int(a)
    unsafe { p^ = 42 }
    q := alloc_int(a)
    unsafe { q^ = 100 }
    unsafe { printf("%lld\n", unsafe { p^ + q^ }) }
    unsafe { printf("%lld\n", a.offset) }
    arena_destroy(a)
    0
}
"#;

#[test]
fn native_dynamic_arena_over_malloc() {
    let Some(output) = compile_and_run_unaudited("dynarena", DYNAMIC_ARENA)
    else {
        return;
    };
    assert_eq!(output, "142\n16\n");
}

const ALLOCATOR_INTERFACE: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Bump :: struct { data: ^u8, cap: i64, offset: i64 }

bump_take :: fn(state: ^u8, size: i64) -> ^u8 {
    b := unsafe { ptr_cast($Bump, state) }
    slot := unsafe { ptr_to(b^.data[b^.offset]) }
    unsafe { b^.offset = b^.offset + size }
    slot
}

Allocator :: struct { take: fn(^u8, i64) -> ^u8, state: ^u8 }

alloc :: fn(a: Allocator, size: i64) -> ^u8 {
    a.take(a.state, size)
}

main :: fn() -> i64 {
    var backing : [64]u8 = [0; 64]
    var bump : Bump = Bump { data = ptr_to(backing[0]), cap = 64, offset = 0 }
    a : Allocator = unsafe { Allocator { take = bump_take, state = ptr_cast($u8, ptr_to(bump)) } }
    p := unsafe { ptr_cast($i64, alloc(a, 8)) }
    unsafe { p^ = 42 }
    q := unsafe { ptr_cast($i64, alloc(a, 8)) }
    unsafe { q^ = 7 }
    unsafe { printf("%lld\n", unsafe { p^ + q^ }) }
    unsafe { printf("%lld\n", bump.offset) }
    0
}
"#;

#[test]
fn native_allocator_interface() {
    let Some(output) =
        compile_and_run_unaudited("allociface", ALLOCATOR_INTERFACE)
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
    slot := unsafe { ptr_to(a.data[a.offset]) }
    a.offset = a.offset + sizeof(Point)
    unsafe { ptr_cast($Point, slot) }
}

alloc_int :: fn(mut a: Arena<128>) -> ^i64 {
    slot := unsafe { ptr_to(a.data[a.offset]) }
    a.offset = a.offset + sizeof(i64)
    unsafe { ptr_cast($i64, slot) }
}

main :: fn() -> i64 {
    var arena : Arena<128> = Arena { data = [0; 128], offset = 0 }
    p : ^Point = alloc_point(arena)
    unsafe { p^.x = 3 }
    unsafe { p^.y = 4 }
    q : ^i64 = alloc_int(arena)
    unsafe { q^ = 99 }
    unsafe { printf("%lld\n", p^.x) }
    unsafe { printf("%lld\n", q^) }
    unsafe { printf("%lld\n", arena.offset) }
    arena.offset = 0
    r : ^i64 = alloc_int(arena)
    unsafe { r^ = 7 }
    unsafe { printf("%lld\n", r^) }
    unsafe { printf("%lld\n", arena.offset) }
    0
}
"#;

#[test]
fn native_frost_arena_allocator() {
    let Some(output) = compile_and_run_unaudited("arena", ARENA) else {
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
    var sum : i64 = 0
    var i : i64 = 0
    while (i < b.len) {
        sum = sum + view[i]
        i = i + 1
    }
    sum
}

main :: fn() -> i64 {
    var b : Buffer<i64, 4> = Buffer {
        data = [0, 0, 0, 0],
        len = 0,
    }
    push(b, 10)
    push(b, 20)
    push(b, 30)
    unsafe { printf("%lld\n", b.len) }
    unsafe { printf("%lld\n", b.data[1]) }
    unsafe { printf("%lld\n", total(b)) }
    0
}
"#;

#[test]
fn native_value_generic_struct() {
    let Some(output) =
        compile_and_run_unaudited("valuegenerics", VALUE_GENERICS)
    else {
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
    var i : i64 = 0
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
    var world : Slab<Entity, 4> = slab_new()
    reset(world)
    hero : Handle<Entity> = insert(world, Entity{hp=100, mana=30})
    foe : Handle<Entity> = insert(world, Entity{hp=40, mana=10})
    unsafe { printf("%lld\n", world[hero].hp) }
    world[hero].hp = world[hero].hp - 25
    unsafe { printf("%lld\n", world[hero].hp) }
    unsafe { printf("%lld\n", world[foe].mana) }
    0
}
"#;

#[test]
fn native_slab_handle_place_deref() {
    let Some(output) = compile_and_run_unaudited("slabderef", SLAB_DEREF)
    else {
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
    var i : i64 = 0
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
    var w : Slab<Entity, 4> = slab_new()
    reset(w)
    old : Handle<Entity> = insert(w, Entity{hp=100})
    release(w, old)
    insert(w, Entity{hp=7})
    unsafe { printf("%lld\n", w[old].hp) }
    0
}
"#;

#[test]
fn native_slab_stale_handle_aborts() {
    let Some((succeeded, stderr)) =
        compile_and_run_status("slabstale", SLAB_STALE_HANDLE)
    else {
        return;
    };
    assert!(!succeeded, "a stale handle into a slab should abort");
    assert!(
        stderr.contains("stale handle"),
        "expected the generation-check message, got:\n{stderr}"
    );
}

// The self-hosted compiler owes the same slab place-deref as the bootstrap. It
// has no value generics, so the slab is a fixed-capacity struct rather than
// Slab<T, N>, but the shape the compiler keys on, a `storage` run beside a
// `generations` run, is the same, as is `Handle<T>` and its packing.
const SELFHOSTED_SLAB: &str = concat!(
    "import \"io.frost\"\n",
    "Entity :: struct { hp: i64, mana: i64 }\n",
    "Slab :: struct {\n",
    "    storage: [4]Entity,\n",
    "    generations: [4]i64,\n",
    "    free_list: [4]i64,\n",
    "    free_count: i64,\n",
    "}\n",
    "reset :: fn(mut s: Slab) {\n",
    "    var i : i64 = 0\n",
    "    while (i < 4) { s.generations[i] = 0  s.free_list[i] = 3 - i  i = i + 1 }\n",
    "    s.free_count = 4\n",
    "}\n",
    "insert :: fn(mut s: Slab, move value: Entity) -> Handle<Entity> {\n",
    "    s.free_count = s.free_count - 1\n",
    "    index := s.free_list[s.free_count]\n",
    "    s.storage[index] = value\n",
    "    packed := (s.generations[index] << 32) | index\n",
    "    packed\n",
    "}\n",
    "main :: fn() -> i64 {\n",
    "    var world : Slab = slab_new()\n",
    "    reset(world)\n",
    "    hero : Handle<Entity> = insert(world, Entity{hp=100, mana=30})\n",
    "    foe : Handle<Entity> = insert(world, Entity{hp=40, mana=10})\n",
    "    print(\"{}\\n\", world[hero].hp)\n",
    "    world[hero].hp = world[hero].hp - 25\n",
    "    print(\"{}\\n\", world[hero].hp)\n",
    "    print(\"{}\\n\", world[foe].mana)\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_slab_handle_place_deref() {
    let Some(output) = selfhosted_unaudited_output("slab", SELFHOSTED_SLAB)
    else {
        return;
    };
    assert_eq!(output, "100\n75\n10\n");
}

const SELFHOSTED_SLAB_STALE: &str = concat!(
    "import \"io.frost\"\n",
    "Entity :: struct { hp: i64, mana: i64 }\n",
    "Slab :: struct {\n",
    "    storage: [4]Entity,\n",
    "    generations: [4]i64,\n",
    "    free_list: [4]i64,\n",
    "    free_count: i64,\n",
    "}\n",
    "reset :: fn(mut s: Slab) {\n",
    "    var i : i64 = 0\n",
    "    while (i < 4) { s.generations[i] = 0  s.free_list[i] = 3 - i  i = i + 1 }\n",
    "    s.free_count = 4\n",
    "}\n",
    "insert :: fn(mut s: Slab, move value: Entity) -> Handle<Entity> {\n",
    "    s.free_count = s.free_count - 1\n",
    "    index := s.free_list[s.free_count]\n",
    "    s.storage[index] = value\n",
    "    packed := (s.generations[index] << 32) | index\n",
    "    packed\n",
    "}\n",
    "release :: fn(mut s: Slab, handle: Handle<Entity>) {\n",
    "    raw : i64 = handle\n",
    "    index := raw & 4294967295\n",
    "    s.generations[index] = s.generations[index] + 1\n",
    "    s.free_list[s.free_count] = index\n",
    "    s.free_count = s.free_count + 1\n",
    "}\n",
    "main :: fn() -> i64 {\n",
    "    var world : Slab = slab_new()\n",
    "    reset(world)\n",
    "    old : Handle<Entity> = insert(world, Entity{hp=100, mana=0})\n",
    "    release(world, old)\n",
    "    insert(world, Entity{hp=7, mana=0})\n",
    "    print(\"{}\\n\", world[old].hp)\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_slab_stale_handle_aborts() {
    let Some((succeeded, stderr)) =
        selfhosted_native_status("slabstale", SELFHOSTED_SLAB_STALE)
    else {
        return;
    };
    assert!(!succeeded, "a stale handle into a slab should abort");
    assert!(
        stderr.contains("stale handle"),
        "expected the generation-check message, got:\n{stderr}"
    );
}

// A value generic: the capacity is a compile-time number the field type is sized
// by. The self-hosted compiler had only type parameters before; a value one is a
// parameter that binds to a number rather than a type.
const SELFHOSTED_VALUE_GENERIC: &str = concat!(
    "import \"io.frost\"\n",
    "Buf :: struct($N: usize) {\n",
    "    data: [N]i64,\n",
    "    count: i64,\n",
    "}\n",
    "sum :: fn($N: usize, b: Buf<N>) -> i64 {\n",
    "    var total : i64 = 0\n",
    "    var i : i64 = 0\n",
    "    while (i < N) {\n",
    "        total = total + b.data[i]\n",
    "        i = i + 1\n",
    "    }\n",
    "    total\n",
    "}\n",
    "main :: fn() -> i64 {\n",
    "    b : Buf<4> = Buf { data = [10, 20, 30, 40], count = 4 }\n",
    "    print(\"{}\\n\", b.data[0])\n",
    "    print(\"{}\\n\", b.data[3])\n",
    "    print(\"{}\\n\", sum($4, b))\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_value_generic_struct() {
    let Some(output) =
        selfhosted_unaudited_output("valuegen", SELFHOSTED_VALUE_GENERIC)
    else {
        return;
    };
    assert_eq!(output, "10\n40\n100\n");
}

// The generic slab: two parameters, a type and a value, and generic functions
// taking both as `$Type` and `$value` arguments, monomorphized on the whole
// tuple. This is the shape std/slab.frost uses; the place-deref is the same one
// the fixed-capacity slab test exercises, now over a generic instance.
const SELFHOSTED_GENERIC_SLAB: &str = concat!(
    "import \"io.frost\"\n",
    "Entity :: struct { hp: i64, mana: i64 }\n",
    "Slab :: struct($T: Type, $N: usize) {\n",
    "    storage: [N]T,\n",
    "    generations: [N]i64,\n",
    "    free_list: [N]i64,\n",
    "    free_count: i64,\n",
    "}\n",
    "slab_reset :: fn($T: Type, $N: usize, mut s: Slab<T, N>) {\n",
    "    var i : i64 = 0\n",
    "    while (i < N) {\n",
    "        s.generations[i] = 0\n",
    "        s.free_list[i] = N - 1 - i\n",
    "        i = i + 1\n",
    "    }\n",
    "    s.free_count = N\n",
    "}\n",
    "slab_insert :: fn($T: Type, $N: usize, mut s: Slab<T, N>, move value: T) -> Handle<T> {\n",
    "    s.free_count = s.free_count - 1\n",
    "    index := s.free_list[s.free_count]\n",
    "    s.storage[index] = value\n",
    "    packed := (s.generations[index] << 32) | index\n",
    "    packed\n",
    "}\n",
    "main :: fn() -> i64 {\n",
    "    var world : Slab<Entity, 4> = slab_new()\n",
    "    slab_reset($Entity, $4, world)\n",
    "    hero := slab_insert($Entity, $4, world, Entity{hp=100, mana=30})\n",
    "    foe := slab_insert($Entity, $4, world, Entity{hp=40, mana=10})\n",
    "    print(\"{}\\n\", world[hero].hp)\n",
    "    world[hero].hp = world[hero].hp - 25\n",
    "    print(\"{}\\n\", world[hero].hp)\n",
    "    print(\"{}\\n\", world[foe].mana)\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_generic_slab_place_deref() {
    let Some(output) =
        selfhosted_unaudited_output("genslab", SELFHOSTED_GENERIC_SLAB)
    else {
        return;
    };
    assert_eq!(output, "100\n75\n10\n");
}

// A generic enum with a payload, matched and constructed inside a generic
// function. Each instance re-parses the body with its type bound, so the
// variant resolves per instance rather than at the template.
const SELFHOSTED_GENERIC_ENUM: &str = concat!(
    "import \"io.frost\"\n",
    "Option :: enum($T: Type) { None, Some { value: T } }\n",
    "unwrap_or :: fn($T: Type, m: Option<T>, fallback: $T) -> $T {\n",
    "    match m {\n",
    "        case .Some { value }: value\n",
    "        case .None: fallback\n",
    "    }\n",
    "}\n",
    "main :: fn() -> i64 {\n",
    "    a : Option<i64> = Option::Some { value = 42 }\n",
    "    b : Option<i64> = Option::None\n",
    "    print(\"{}\\n\", unwrap_or($i64, a, 0))\n",
    "    print(\"{}\\n\", unwrap_or($i64, b, 99))\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_generic_enum() {
    let Some(output) =
        selfhosted_unaudited_output("genenum", SELFHOSTED_GENERIC_ENUM)
    else {
        return;
    };
    assert_eq!(output, "42\n99\n");
}

// A generic instance nested inside another as its last argument, so the lexer
// runs the two closing angle brackets into one `>>` token that has to be split,
// and a nested literal that has to resolve to the inner instance, not the outer.
const SELFHOSTED_NESTED_GENERIC: &str = concat!(
    "import \"io.frost\"\n",
    "Pair :: struct($A: Type, $B: Type) { first: A, second: B }\n",
    "main :: fn() -> i64 {\n",
    "    p : Pair<i64, Pair<i64, i64>> = Pair { first = 1, second = Pair { first = 2, second = 3 } }\n",
    "    print(\"{}\\n\", p.first)\n",
    "    print(\"{}\\n\", p.second.first)\n",
    "    print(\"{}\\n\", p.second.second)\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_nested_generic() {
    let Some(output) =
        selfhosted_unaudited_output("nestgen", SELFHOSTED_NESTED_GENERIC)
    else {
        return;
    };
    assert_eq!(output, "1\n2\n3\n");
}

// A growable heap-backed vector, the shape of std/vec.frost, compiled by the
// self-hosted compiler. It leans on generics, an `unsafe { ... }` used as a
// value (the allocation and the pointer cast), and the heap runtime, all of
// which the value-generics work and the fixes around it brought in.
const SELFHOSTED_VEC: &str = concat!(
    "import \"io.frost\"\n",
    "frost_rt_heap_alloc :: extern fn(size: i64) -> ^u8\n",
    "frost_rt_heap_realloc :: extern fn(block: ^u8, size: i64) -> ^u8\n",
    "frost_rt_heap_free :: extern fn(block: ^u8)\n",
    "Vec :: struct($T: Type) { data: ^T, len: i64, cap: i64 }\n",
    "vec_new :: fn($T: Type, capacity: i64) -> Vec<T> {\n",
    "    var room := capacity\n",
    "    if (room < 1) { room = 1 }\n",
    "    block := unsafe { frost_rt_heap_alloc(room * sizeof(T)) }\n",
    "    Vec { data = unsafe { ptr_cast($T, block) }, len = 0, cap = room }\n",
    "}\n",
    "vec_push :: fn($T: Type, mut v: Vec<T>, move value: $T) {\n",
    "    if (v.len >= v.cap) {\n",
    "        var room := v.cap * 2\n",
    "        if (room < 1) { room = 1 }\n",
    "        v.data = unsafe { ptr_cast($T, frost_rt_heap_realloc(ptr_cast($u8, v.data), room * sizeof(T))) }\n",
    "        v.cap = room\n",
    "    }\n",
    "    unsafe { v.data[v.len] = value }\n",
    "    v.len = v.len + 1\n",
    "}\n",
    "vec_get :: fn($T: Type, v: Vec<T>, index: i64) -> $T { unsafe { v.data[index] } }\n",
    "vec_len :: fn($T: Type, v: Vec<T>) -> i64 { v.len }\n",
    "vec_free :: fn($T: Type, move v: Vec<T>) { unsafe { frost_rt_heap_free(ptr_cast($u8, v.data)) } }\n",
    "main :: fn() -> i64 {\n",
    "    var v : Vec<i64> = vec_new($i64, 2)\n",
    "    vec_push($i64, v, 10)\n",
    "    vec_push($i64, v, 20)\n",
    "    vec_push($i64, v, 30)\n",
    "    print(\"{}\\n\", vec_len($i64, v))\n",
    "    print(\"{}\\n\", vec_get($i64, v, 0))\n",
    "    print(\"{}\\n\", vec_get($i64, v, 2))\n",
    "    vec_free($i64, v)\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_growable_vector() {
    let Some(output) = selfhosted_unaudited_output("vec", SELFHOSTED_VEC)
    else {
        return;
    };
    assert_eq!(output, "3\n10\n30\n");
}

// The actual standard-library hash map, imported rather than inlined, compiled
// by the self-hosted compiler. `std` is one of its import roots, so the harness
// running from the crate root reaches std/map.frost. It exercises generics over
// the value type, the heap runtime, and `unsafe { ... }` both as a statement
// block and as a value in a comparison.
const SELFHOSTED_STD_MAP: &str = concat!(
    "import \"io.frost\"\n",
    "import \"map.frost\"\n",
    "main :: fn() -> i64 {\n",
    "    var m : Map<i64, i64> = map_new($i64, $i64, 8)\n",
    "    map_put($i64, $i64, $i64_keys, m, 100, 42)\n",
    "    map_put($i64, $i64, $i64_keys, m, 200, 99)\n",
    "    map_put($i64, $i64, $i64_keys, m, 100, 7)\n",
    "    print(\"{}\\n\", map_len($i64, $i64, m))\n",
    "    print(\"{}\\n\", map_get($i64, $i64, $i64_keys, m, 100, 0))\n",
    "    print(\"{}\\n\", map_get($i64, $i64, $i64_keys, m, 200, 0))\n",
    "    if (map_has($i64, $i64, $i64_keys, m, 300)) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n",
    "    map_free($i64, $i64, m)\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_standard_library_map() {
    let Some(output) =
        selfhosted_unaudited_output("stdmap", SELFHOSTED_STD_MAP)
    else {
        return;
    };
    assert_eq!(output, "2\n7\n99\n0\n");
}

// The standard-library output helpers, imported and compiled by the self-hosted
// compiler. io.frost writes through the runtime's stdout helpers, and its bare
// writers are named for what they write.
const SELFHOSTED_STD_IO: &str = concat!(
    "import \"io.frost\"\n",
    "main :: fn() -> i64 {\n",
    "    print(\"hi \")\n",
    "    print(\"{}\\n\", 42)\n",
    "    print(\"done\\n\")\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_standard_library_io() {
    let Some(output) = selfhosted_unaudited_output("stdio", SELFHOSTED_STD_IO)
    else {
        return;
    };
    assert_eq!(output, "hi 42\ndone\n");
}

// The standard-library string builder, which formats an integer into a heap
// buffer through a local `[20]u8` scratch. That local fixed array takes its
// element type from its declaration, so the byte literals are bytes, not i64.
const SELFHOSTED_STD_FORMAT: &str = concat!(
    "import \"format.frost\"\n",
    "import \"io.frost\"\n",
    "main :: fn() -> i64 {\n",
    "    var b : Builder = builder_new(16)\n",
    "    builder_str_value(b, \"count = \")\n",
    "    builder_int(b, 12345)\n",
    "    builder_int(b, -99)\n",
    "    print(\"{}\\n\", builder_str(b))\n",
    "    builder_free(b)\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_standard_library_format() {
    let Some(output) =
        selfhosted_unaudited_output("stdfmt", SELFHOSTED_STD_FORMAT)
    else {
        return;
    };
    assert_eq!(output, "count = 12345-99\n");
}

// The single-precision math library, imported and compiled by the self-hosted
// compiler: vectors, a matrix transform, a matrix product, and a quaternion
// rotation, plus the trig-heavy projection and view builders. Cases are chosen
// to land on clean values. Exercises floats/SSE end to end on both backends.
const SELFHOSTED_STD_MATH: &str = concat!(
    "import \"io.frost\"\n",
    "import \"math.frost\"\n",
    "main :: fn() -> i64 {\n",
    "    print(\"{}\\n\", vec3_dot(vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0)))\n",
    "    c := vec3_cross(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0))\n",
    "    print(\"{}\\n\", c.z)\n",
    "    print(\"{}\\n\", vec3_length(vec3(3.0, 4.0, 0.0)))\n",
    "    n := vec3_normalize(vec3(3.0, 4.0, 0.0))\n",
    "    print(\"{}\\n\", n.x)\n",
    "    p := mat4_transform_point(mat4_translation(vec3(1.0, 2.0, 3.0)), vec3(10.0, 20.0, 30.0))\n",
    "    print(\"{}\\n\", p.x)\n",
    "    id := mat4_mul(mat4_identity(), mat4_identity())\n",
    "    print(\"{}\\n\", id.m[0])\n",
    "    rotated := mat4_transform_dir(mat4_rotation_z(radians(90.0)), vec3(3.0, 4.0, 0.0))\n",
    "    print(\"{}\\n\", vec3_length(rotated))\n",
    "    persp := mat4_perspective(radians(90.0), 1.0, 1.0, 10.0)\n",
    "    print(\"{}\\n\", persp.m[0])\n",
    "    view := mat4_look_at(vec3(0.0, 0.0, 5.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0))\n",
    "    vp := mat4_transform_point(view, vec3(0.0, 0.0, 0.0))\n",
    "    print(\"{}\\n\", vp.z)\n",
    "    q := quat_from_axis_angle(vec3(0.0, 0.0, 1.0), radians(90.0))\n",
    "    qr := quat_rotate_vec3(q, vec3(2.0, 0.0, 0.0))\n",
    "    print(\"{}\\n\", vec3_length(qr))\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_standard_library_math() {
    let Some(output) =
        selfhosted_unaudited_output("stdmath", SELFHOSTED_STD_MATH)
    else {
        return;
    };
    assert_eq!(output, "32\n1\n5\n0.6\n11\n1\n5\n1\n-5\n2\n");
}

// The SoA `columns<T, N>` container, compiled by the self-hosted compiler: the
// synthesized per-field-array layout, `columns_new()` construction, the
// generational library (reset/insert/alive/release) imported from
// std/columns.frost, the handle-checked place-deref `c[h].field` for read and
// write, a column sliced into a hot loop (`c.x` is an ordinary array), and the
// stale-handle check going false after a release. Exercises the field-
// reflection builtin and the scatter/deref lowering on the native backend.
const SELFHOSTED_COLUMNS: &str = concat!(
    "import \"io.frost\"\n",
    "import \"columns.frost\"\n",
    "Particle :: struct { x: i64, y: i64 }\n",
    "sum_col :: fn(xs: []i64) -> i64 {\n",
    "    var total : i64 = 0\n",
    "    var i : i64 = 0\n",
    "    n := slice_len(xs)\n",
    "    while (i < n) { total = total + xs[i]  i = i + 1 }\n",
    "    total\n",
    "}\n",
    "main :: fn() -> i64 {\n",
    "    var c : columns<Particle, 8> = columns_new()\n",
    "    columns_reset($Particle, $8, c)\n",
    "    a := columns_insert($Particle, $8, c, Particle { x = 10, y = 1 })\n",
    "    columns_insert($Particle, $8, c, Particle { x = 20, y = 2 })\n",
    "    print(\"{}\\n\", c[a].x + c[a].y)\n",
    "    c[a].x = 100\n",
    "    print(\"{}\\n\", c[a].x)\n",
    "    print(\"{}\\n\", sum_col(c.x))\n",
    "    if (columns_alive($Particle, $8, c, a)) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n",
    "    columns_release($Particle, $8, c, a)\n",
    "    if (columns_alive($Particle, $8, c, a)) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_columns_container() {
    let Some(output) =
        selfhosted_unaudited_output("columns", SELFHOSTED_COLUMNS)
    else {
        return;
    };
    // 10+1; then x set to 100; column sum 100+20; alive true; then false.
    assert_eq!(output, "11\n100\n120\n1\n0\n");
}

// The same SoA `columns<T, N>` container, compiled by the BOOTSTRAP compiler on
// both of its backends. Parity with the self-hosted compiler: the container is
// synthesized by field reflection in ir::build, `columns_new()` zero-inits it,
// `c[h].field` and `c[h] = value` lower to the generational check, and a column
// (`c.x`) slices into a hot loop. Generic functions over `columns<T, N>` are
// monomorphized like any other. Kept self-contained (its own `printf` extern,
// no std import) so it runs under the bootstrap.
const BOOTSTRAP_COLUMNS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32
Particle :: struct { x: i64, y: i64 }
col_reset :: fn($T: Type, $N: usize, mut c: columns<T, N>) {
    var i : i64 = 0
    while (i < N) { c.generations[i] = 0  c.free_list[i] = N - 1 - i  i = i + 1 }
    c.free_count = N
}
col_insert :: fn($T: Type, $N: usize, mut c: columns<T, N>, move value: $T) -> Handle<T> {
    c.free_count = c.free_count - 1
    index := c.free_list[c.free_count]
    handle : Handle<T> = (c.generations[index] << 32) | index
    c[handle] = value
    handle
}
col_alive :: fn($T: Type, $N: usize, c: columns<T, N>, handle: Handle<T>) -> bool {
    raw : i64 = handle
    c.generations[raw & 4294967295] == (raw >> 32)
}
sum_col :: fn(xs: []i64) -> i64 {
    var total : i64 = 0
    var i : i64 = 0
    n := slice_len(xs)
    while (i < n) { total = total + xs[i]  i = i + 1 }
    total
}
main :: fn() -> i64 {
    var c : columns<Particle, 8> = columns_new()
    col_reset($Particle, $8, c)
    a := col_insert($Particle, $8, c, Particle { x = 10, y = 1 })
    col_insert($Particle, $8, c, Particle { x = 20, y = 2 })
    unsafe { printf("%lld\n", c[a].x + c[a].y) }
    c[a].x = 100
    unsafe { printf("%lld\n", c[a].x) }
    unsafe { printf("%lld\n", sum_col(c.x)) }
    if (col_alive($Particle, $8, c, a)) { unsafe { printf("%lld\n", 1) } } else { unsafe { printf("%lld\n", 0) } }
    0
}
"#;

#[test]
fn bootstrap_columns_container_both_backends() {
    let Some(native) =
        run_backend("columns_boot_native", BOOTSTRAP_COLUMNS, false)
    else {
        return;
    };
    let Some(c) = run_backend("columns_boot_c", BOOTSTRAP_COLUMNS, true) else {
        return;
    };
    // 10+1; x set to 100; column sum 100+20; alive true.
    assert_eq!(native, "11\n100\n120\n1\n");
    assert_eq!(c, "11\n100\n120\n1\n");
}

// The std/io writers, compiled by the BOOTSTRAP on both backends. An integer
// writes as %lld and a float as %g through the runtime's write helpers, and an
// f32 widens to the f64 the helper takes, so both backends have to agree on
// the widening as well as the digits.
const IO_WRITERS: &str = concat!(
    "import \"io.frost\"\n",
    "main :: fn() -> i64 {\n",
    "    print(\"{}\\n\", 42)\n",
    "    print(\"{}\\n\", 7 * 6)\n",
    "    x : f32 = 0.5\n",
    "    print(\"{}\\n\", x)\n",
    "    0\n",
    "}\n",
);

#[test]
fn bootstrap_io_writers_both_backends() {
    let Some(native) = run_backend("print_boot_native", IO_WRITERS, false)
    else {
        return;
    };
    let Some(c) = run_backend("print_boot_c", IO_WRITERS, true) else {
        return;
    };
    assert_eq!(native, "42\n42\n0.5\n");
    assert_eq!(c, "42\n42\n0.5\n");
}

// Constant expressions: a constant whose value is an integer expression over
// literals and earlier constants, folded at compile time, including as an array
// size (`[STRIDE]i64`), the vertex-layout case. Exercised on both compilers and
// all backends, since they must agree on the folded values.
const CONST_EXPRESSIONS: &str = concat!(
    "import \"io.frost\"\n",
    "POS :: 3\n",
    "NORMAL :: 3\n",
    "UV :: 2\n",
    "STRIDE :: POS + NORMAL + UV\n",
    "FLAGS :: 1 << 3\n",
    "NEG :: -5\n",
    "main :: fn() -> i64 {\n",
    "    data : [STRIDE]i64 = [1, 2, 3, 4, 5, 6, 7, 8]\n",
    "    print(\"{}\\n\", STRIDE)\n",
    "    print(\"{}\\n\", FLAGS)\n",
    "    print(\"{}\\n\", NEG)\n",
    "    print(\"{}\\n\", data[STRIDE - 1])\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_const_expressions() {
    let Some(output) =
        selfhosted_unaudited_output("constexpr", CONST_EXPRESSIONS)
    else {
        return;
    };
    assert_eq!(output, "8\n8\n-5\n8\n");
}

#[test]
fn bootstrap_const_expressions_both_backends() {
    let Some(native) =
        run_backend("constexpr_native", CONST_EXPRESSIONS, false)
    else {
        return;
    };
    let Some(c) = run_backend("constexpr_c", CONST_EXPRESSIONS, true) else {
        return;
    };
    assert_eq!(native, "8\n8\n-5\n8\n");
    assert_eq!(c, "8\n8\n-5\n8\n");
}

// `inline fn`, compiled by the self-hosted compiler. The marker is a no-op on
// the native (asm) backend, which does not inline, so the program runs and
// answers exactly as an ordinary function would.
const SELFHOSTED_INLINE: &str = concat!(
    "square :: inline fn(x: i64) -> i64 { x * x }\n",
    "import \"io.frost\"\nmain :: fn() -> i64 { print(\"{}\\n\", square(7))  0 }\n",
);

#[test]
fn self_hosted_inline_marker_runs_on_native() {
    let Some(output) = selfhosted_unaudited_output("inline", SELFHOSTED_INLINE)
    else {
        return;
    };
    assert_eq!(output, "49\n");
}

// The C backend turns `inline fn` into a forced-inline definition. The
// self-hosted compiler's default backend is C, so its emitted translation unit
// carries the always_inline qualifier, on both the prototype and the
// definition, where an ordinary function carries neither.
#[test]
fn self_hosted_inline_emits_c_qualifier() {
    let Some(compiler) = build_self_hosted_compiler("inline_c") else {
        return;
    };
    let directory = std::env::temp_dir();
    let input = directory.join("frost_inline_c.frost");
    std::fs::write(&input, SELFHOSTED_INLINE).unwrap();
    let emit = Command::new(&compiler)
        .env("FROST_INPUT", &input)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    assert!(
        emit.status.success(),
        "self-hosted C backend refused the inline program:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let emitted = String::from_utf8_lossy(&emit.stdout);
    let inline_definitions = emitted
        .matches("static inline __attribute__((always_inline))")
        .count();
    assert!(
        inline_definitions >= 2,
        "emitted C did not force the inline function on both its prototype and \
         definition:\n{emitted}"
    );
}

// A runtime function pointer: a higher-order function taking a `fn(i64) -> i64`
// and calling through it, with a function's name passed as its address. A
// single function pointer is a closed call target, not a vtable.
const SELFHOSTED_FUNCTION_POINTER: &str = concat!(
    "import \"io.frost\"\n",
    "apply :: fn(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }\n",
    "double :: fn(x: i64) -> i64 { x * 2 }\n",
    "inc :: fn(x: i64) -> i64 { x + 1 }\n",
    "main :: fn() -> i64 {\n",
    "    print(\"{}\\n\", apply(double, 21))\n",
    "    print(\"{}\\n\", apply(inc, 41))\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_function_pointer() {
    let Some(output) =
        selfhosted_unaudited_output("funptr", SELFHOSTED_FUNCTION_POINTER)
    else {
        return;
    };
    assert_eq!(output, "42\n42\n");
}

// A closure the data-oriented way: an explicit captured-data struct passed
// alongside a function pointer, called through together. No hidden capture, no
// heap, no vtable, no collector; the context is a value copied in.
const SELFHOSTED_CLOSURE: &str = concat!(
    "import \"io.frost\"\n",
    "Adder :: struct { amount: i64 }\n",
    "add_by :: fn(ctx: Adder, x: i64) -> i64 { x + ctx.amount }\n",
    "apply_each :: fn(f: fn(Adder, i64) -> i64, ctx: Adder, a: i64, b: i64) -> i64 {\n",
    "    f(ctx, a) + f(ctx, b)\n",
    "}\n",
    "main :: fn() -> i64 {\n",
    "    plus10 : Adder = Adder { amount = 10 }\n",
    "    print(\"{}\\n\", apply_each(add_by, plus10, 1, 2))\n",
    "    0\n",
    "}\n",
);

#[test]
fn self_hosted_closure_as_context_and_function() {
    let Some(output) =
        selfhosted_unaudited_output("closure", SELFHOSTED_CLOSURE)
    else {
        return;
    };
    assert_eq!(output, "23\n");
}

const SLICES: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

sum :: fn(s: []i64) -> i64 {
    var total : i64 = 0
    var i : i64 = 0
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
    unsafe { printf("%lld\n", slice_len(view)) }
    unsafe { printf("%lld\n", view[2]) }
    unsafe { printf("%lld\n", sum(view)) }
    unsafe { printf("%lld\n", sum(arr)) }
    0
}
"#;

#[test]
fn native_slices() {
    let Some(output) = compile_and_run_unaudited("slices", SLICES) else {
        return;
    };
    assert_eq!(output, "4\n30\n100\n100\n");
}

const SLICE_OUT_OF_BOUNDS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    arr := [1, 2, 3]
    view : []i64 = arr
    var i : i64 = 7
    unsafe { printf("%lld\n", view[i]) }
    0
}
"#;

#[test]
fn native_slice_index_is_bounds_checked() {
    let Some((succeeded, stderr)) =
        compile_and_run_status("sliceoob", SLICE_OUT_OF_BOUNDS)
    else {
        return;
    };
    assert!(!succeeded, "an out-of-range slice index should abort");
    assert!(
        stderr.contains("out of bounds"),
        "expected the bounds-check message, got:\n{stderr}"
    );
}

// The bounds check compares unsigned, which is what makes one comparison answer
// for a negative index as well as for one past the end. The same cast read a
// negative *length* as enormous, so every index through such a slice passed and
// the slice was unchecked. `slice_prefix($T, xs, -1)` reached this from ordinary
// safe code, with no `unsafe` block anywhere in the program.
#[test]
fn a_slice_may_not_be_built_with_a_negative_length() {
    let source = "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   arr := [1, 2, 3, 4]\n\
         \x20   bad := unsafe { slice_from($i64, ptr_to(arr[0]), 0 - 1) }\n\
         \x20   print(\"{}\\n\", slice_len(bad))\n\
         \x20   0\n}\n";
    let Some((succeeded, stderr)) = compile_and_run_status("neglen", source)
    else {
        return;
    };
    assert!(!succeeded, "a negative slice length should abort");
    assert!(
        stderr.contains("cannot be -1 elements long"),
        "expected the length check, got:\n{stderr}"
    );
}

// Sub-slicing knows the run it is cutting from, so a view longer than what is
// left is a false claim rather than an unverifiable one. It used to be taken on
// trust, and the view that came back carried a length its storage did not have,
// so every access through it was bounds-checked against that length and passed.
#[test]
fn a_sub_slice_may_not_reach_past_the_run_it_came_from() {
    let cases = [
        ("prefixwiden", "slice_prefix($i64, xs, 1000000)"),
        ("rangewiden", "slice_range($i64, xs, 2, 1000000)"),
        ("rangepast", "slice_range($i64, xs, 9, 1)"),
    ];
    for (name, view) in cases {
        let source = format!(
            "import \"io.frost\"\nimport \"mem.frost\"\n\
             main :: fn() -> i64 {{\n\
             \x20   xs := heap_slice($i64, 4)\n\
             \x20   wide := {view}\n\
             \x20   print(\"{{}}\\n\", slice_len(wide))\n\
             \x20   0\n}}\n"
        );
        let Some((succeeded, stderr)) = compile_and_run_status(name, &source)
        else {
            return;
        };
        assert!(!succeeded, "{name} should abort");
        assert!(
            stderr.contains("reaches past a run")
                || stderr.contains("cannot start"),
            "expected the span check for {name}, got:\n{stderr}"
        );
    }
}

// Arithmetic whose result does not fit the type it is computed at stops there.
// An index used to be the case that showed why: one computed with arithmetic
// that wrapped landed on the wrong element, in range, and nothing said so.
#[test]
fn arithmetic_that_leaves_its_type_stops_there() {
    let cases = [
        (
            "ovadd",
            "a : i64 = 9223372036854775807\n    print(\"{}\\n\", a + 1)\n",
            "this addition overflowed",
        ),
        (
            "ovmul",
            "huge : i64 = 4611686018427387904\n    print(\"{}\\n\", huge * 4)\n",
            "multiplication overflowed",
        ),
        (
            "ovnarrow",
            "a : u8 = 200\n    b : u8 = 100\n    print(\"{}\\n\", a + b)\n",
            "this addition overflowed",
        ),
        (
            "ovdiv",
            "a : i64 = 1\n    b : i64 = 0\n    print(\"{}\\n\", a / b)\n",
            "division by zero",
        ),
    ];
    for (name, body, wanted) in cases {
        let source = format!(
            "import \"io.frost\"\nmain :: fn() -> i64 {{\n    {body}    0\n}}\n"
        );
        let Some((succeeded, stderr)) = compile_and_run_status(name, &source)
        else {
            return;
        };
        assert!(!succeeded, "{name} should stop");
        assert!(
            stderr.contains(wanted),
            "expected {wanted} for {name}, got:\n{stderr}"
        );
    }
}

// A literal may span lines, and one written in a file saved with CRLF has to be
// the same string as the same literal saved with LF. The two compilers read the
// carriage return differently, so the same program measured four bytes under one
// and three under the other.
#[test]
fn a_literal_spanning_lines_does_not_carry_the_carriage_return() {
    let source = "import \"io.frost\"\nmain :: fn() -> i64 {\n    s := \"a\r\nb\"\n    print(\"{}\\n\", str_len(s))\n    0\n}\n";
    let Some(output) = compile_and_run_unaudited("crlfliteral", source) else {
        return;
    };
    assert_eq!(output, "3\n");
}

// The other half: leaving the range is the point of a hash, so it is spelled.
#[test]
fn wrapping_arithmetic_is_asked_for_by_name() {
    let source = "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         main :: fn() -> i64 {\n\
         \x20   key : i64 = 123456789\n\
         \x20   var h := wrap_mul(key, 2654435761)\n\
         \x20   h = wrap_add(h, 7)\n\
         \x20   unsafe { printf(\"%lld\\n\", h & 4611686018427387903) }\n\
         \x20   0\n}\n";
    let Some(output) = compile_and_run_unaudited("wraphash", source) else {
        return;
    };
    assert_eq!(output, "327708115659831436\n");
}

// A frame wider than a page has to touch each page on the way down, or the
// stack pointer moves past the guard in one step and the first write below it
// lands in whatever is mapped there. Both backends probe; this is the check that
// the probe is right rather than merely present, since a wrong one faults on
// every call.
#[test]
fn a_frame_wider_than_a_page_touches_each_page() {
    let source = "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         wide :: fn(seed: i64) -> i64 {\n\
         \x20   var buffer : [8192]i64 = [0; 8192]\n\
         \x20   buffer[0] = seed\n\
         \x20   buffer[8191] = seed\n\
         \x20   buffer[0] + buffer[8191]\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   unsafe { printf(\"%lld\\n\", wide(21)) }\n\
         \x20   0\n}\n";
    let Some(output) = compile_and_run_unaudited("pageprobe", source) else {
        return;
    };
    assert_eq!(output, "42\n");
}

// Running the stack out is safe either way, because every frame reaches the
// guard rather than stepping over it. What this checks is that it says so: the
// process used to die with a fault address and nothing naming which of the many
// ways to fault it was.
#[test]
fn an_exhausted_stack_says_what_happened() {
    let source = "import \"io.frost\"\ndown :: fn(n: i64) -> i64 {\n\
         \x20   if (n <= 0) { return 0 }\n\
         \x20   down(n - 1) + 1\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", down(100000000))\n\
         \x20   0\n}\n";
    let Some((succeeded, stderr)) = compile_and_run_status("stackout", source)
    else {
        return;
    };
    assert!(!succeeded, "unbounded recursion should not succeed");
    assert!(
        stderr.contains("the stack ran out"),
        "expected the stack-exhaustion message, got:\n{stderr}"
    );
}

// The other end of the same rule: an empty slice is not a negative one, and a
// container that hands out a zero-length prefix has to keep working.
#[test]
fn a_slice_of_no_elements_is_still_allowed() {
    let source = "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   arr := [1, 2, 3, 4]\n\
         \x20   none := unsafe { slice_from($i64, ptr_to(arr[0]), 0) }\n\
         \x20   print(\"{}\\n\", slice_len(none))\n\
         \x20   0\n}\n";
    let Some((succeeded, stderr)) = compile_and_run_status("zerolen", source)
    else {
        return;
    };
    assert!(succeeded, "an empty slice should be fine, got:\n{stderr}");
}

// `count * sizeof(T)` wrapped, so the allocator was asked for fewer bytes than
// the caller believed and the slice built over the block carried the count that
// was asked for. Every read past the block's real end was then checked against
// the wrong number and passed.
#[test]
fn an_allocation_size_that_wraps_is_refused() {
    let source = "import \"io.frost\"\nfrost_rt_check_size :: safe extern fn(count: i64, width: i64) -> i64\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", frost_rt_check_size(2305843009213693952, 8))\n\
         \x20   0\n}\n";
    let Some((succeeded, stderr)) = compile_and_run_status("sizewrap", source)
    else {
        return;
    };
    assert!(!succeeded, "an allocation size that wraps should abort");
    assert!(
        stderr.contains("more memory than can be addressed"),
        "expected the size check, got:\n{stderr}"
    );
}

// An allocation that failed answered with nothing, and every caller in std/
// wrapped what came back in a slice without looking. A null wrapped in a slice
// reads as a run of `count` elements at address zero, each access checked
// against a length that has nothing to do with what was allocated.
#[test]
fn an_allocation_that_fails_aborts_rather_than_answering_null() {
    let source = "import \"io.frost\"\nfrost_rt_heap_alloc :: extern fn(size: i64) -> ^u8\n\
         main :: fn() -> i64 {\n\
         \x20   held := unsafe { frost_rt_heap_alloc(9000000000000000) }\n\
         \x20   print(\"{}\\n\", 1)\n\
         \x20   0\n}\n";
    let Some((succeeded, stderr)) = compile_and_run_status("allocfail", source)
    else {
        return;
    };
    assert!(!succeeded, "a failed allocation should abort");
    assert!(
        stderr.contains("out of memory"),
        "expected the allocation-failure message, got:\n{stderr}"
    );
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
    var i : i64 = 0
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
    var world : Slab = slab_new()
    slab_reset(world)
    hero := slab_insert(world, Entity { hp = 100, mana = 30 })
    foe := slab_insert(world, Entity { hp = 40, mana = 10 })
    a := slab_read(world, hero)
    unsafe { printf("%lld\n", a.hp) }
    if (slab_alive(world, foe)) { unsafe { printf("%lld\n", 1) } } else { unsafe { printf("%lld\n", 0) } }
    slab_release(world, foe)
    reused := slab_insert(world, Entity { hp = 7, mana = 7 })
    if (slab_alive(world, reused)) { unsafe { printf("%lld\n", 1) } } else { unsafe { printf("%lld\n", 0) } }
    if (slab_alive(world, foe)) { unsafe { printf("%lld\n", 9) } } else { unsafe { printf("%lld\n", 0) } }
    0
}
"#;

#[test]
fn native_generational_pool_written_in_frost() {
    let Some(output) = compile_and_run_unaudited("nativepool", NATIVE_POOL)
    else {
        return;
    };
    // insert, read, live-before-free, reused-slot-live, stale-handle-dead
    assert_eq!(output, "100\n1\n1\n0\n");
}

// The example itself rather than a copy of it, so the two cannot drift.
#[test]
fn native_freestanding_links_without_libc() {
    if !linker_available() {
        return;
    }
    let source_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("freestanding.frost");
    let directory = std::env::temp_dir();
    let exe_path = directory.join(format!(
        "frost_heap_freestanding{}",
        std::env::consts::EXE_SUFFIX
    ));
    let frost = env!("CARGO_BIN_EXE_frost");
    let compile = Command::new(frost)
        .arg("--link")
        .arg("--freestanding")
        .arg("-o")
        .arg(&exe_path)
        .arg(&source_path)
        .output()
        .unwrap();
    if !compile.status.success() {
        let message = String::from_utf8_lossy(&compile.stderr);
        // --freestanding needs gcc or clang; skip where only MSVC is present.
        // A refusal by the compiler itself is a broken example, not a missing
        // toolchain, and is what this exists to catch.
        assert!(
            !message.contains("unsafe"),
            "examples/freestanding.frost does not compile:\n{message}"
        );
        return;
    }
    let run = Command::new(&exe_path).status().unwrap();
    let _ = std::fs::remove_file(&exe_path);
    // The static arena computes 20 + 22 and returns it as the exit code.
    assert_eq!(run.code(), Some(42));
}

// The bounds check reaches even a program with no libc: the freestanding
// runtime traps rather than printing, so an out-of-range index still cannot
// read past the array. Status only, since a trap composes no message.
const FREESTANDING_OUT_OF_BOUNDS: &str = r#"
main :: fn() -> i64 {
    var a : [3]i64 = [10, 20, 30]
    var i : i64 = 5
    a[i]
}
"#;

#[test]
fn native_freestanding_out_of_bounds_traps() {
    if !linker_available() {
        return;
    }
    let directory = std::env::temp_dir();
    let source_path = directory.join("frost_freestanding_oob.frost");
    let exe_path = directory.join(format!(
        "frost_freestanding_oob{}",
        std::env::consts::EXE_SUFFIX
    ));
    std::fs::write(&source_path, FREESTANDING_OUT_OF_BOUNDS).unwrap();
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
    assert!(
        !run.success(),
        "an out-of-range index should trap in a freestanding build, not return"
    );
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

// A `str` held in a struct is indexed where it sits. Before this, only a `str`
// bound to a local could be indexed, so every reader over a string kept in a
// struct had to copy the field into a local first.
const STR_IN_A_FIELD: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Document :: struct { source: str, at: i64 }

Outer :: struct { inner: Document }

byte_at :: fn(document: Document, index: i64) -> i64 {
    document.source[index]
}

main :: fn() -> i64 {
    document := Document { source = "hello", at = 1 }
    unsafe { printf("%lld\n", document.source[0]) }
    unsafe { printf("%lld\n", document.source[document.at]) }
    unsafe { printf("%lld\n", byte_at(document, 4)) }
    unsafe { printf("%lld\n", str_len(document.source)) }

    outer := Outer { inner = document }
    unsafe { printf("%lld\n", outer.inner.source[2]) }
    0
}
"#;

#[test]
fn native_str_held_in_a_struct_is_indexable() {
    let Some(output) = compile_and_run_unaudited("strfield", STR_IN_A_FIELD)
    else {
        return;
    };
    assert_eq!(output, "104\n101\n111\n5\n108\n");
}

// Every name here belongs to a runtime symbol. A Frost function is emitted as
// `frost_u_<name>` and the runtime owns `frost_rt_<name>`, so the two sets are
// disjoint at a fixed position and no name is reserved. `byte_at` used to fail
// to link, and there was no list saying which other names would.
const RUNTIME_SYMBOL_NAMES: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

byte_at :: fn(text: str, index: i64) -> i64 { text[index] }
// `byte_set` rather than `str_len`, which the runtime also has: what this is
// about is a runtime symbol, and a name the compiler owns cannot be declared at
// all.
byte_set :: fn(n: i64) -> i64 { n * 2 }
die :: fn(n: i64) -> i64 { n + 1 }
error :: fn(n: i64) -> i64 { n + 2 }
slot :: fn(n: i64) -> i64 { n + 3 }
getenv :: fn(n: i64) -> i64 { n + 4 }
mem_set :: fn(n: i64) -> i64 { n + 5 }
bounds_check :: fn(n: i64) -> i64 { n + 6 }
check_index :: fn(n: i64) -> i64 { n + 7 }
assert_at :: fn(n: i64) -> i64 { n + 8 }
print_i64 :: fn(n: i64) -> i64 { n + 9 }
heap_alloc :: fn(n: i64) -> i64 { n + 10 }
// A name in the runtime's own prefix space, which a Frost function reaches
// only by being called this and so must not land on the runtime's `slot`.
rt_slot :: fn(n: i64) -> i64 { n + 11 }

main :: fn() -> i64 {
    unsafe { printf("%lld\n", byte_at("hello", 1)) }
    unsafe { printf("%lld\n", byte_set(1)) }
    unsafe { printf("%lld\n", die(1) + error(1) + slot(1) + getenv(1)) }
    unsafe { printf("%lld\n", mem_set(1) + bounds_check(1) + check_index(1)) }
    unsafe { printf("%lld\n", assert_at(1) + print_i64(1) + heap_alloc(1)) }
    unsafe { printf("%lld\n", rt_slot(1)) }
    0
}
"#;

#[test]
fn a_frost_function_may_carry_a_runtime_symbols_name() {
    let Some(output) =
        compile_and_run_unaudited("rtnames", RUNTIME_SYMBOL_NAMES)
    else {
        return;
    };
    assert_eq!(output, "101\n2\n14\n21\n30\n12\n");
}

const STR_OUT_OF_BOUNDS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    greeting := "hi"
    var i : i64 = 5
    unsafe { printf("%lld\n", greeting[i]) }
    0
}
"#;

#[test]
fn native_str_index_is_bounds_checked() {
    let Some((succeeded, stderr)) =
        compile_and_run_status("stroob", STR_OUT_OF_BOUNDS)
    else {
        return;
    };
    assert!(!succeeded, "an out-of-range str index should abort");
    assert!(
        stderr.contains("out of bounds"),
        "expected the bounds-check message, got:\n{stderr}"
    );
}

const POINTERS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

swap :: fn(a: ^i64, b: ^i64) {
    temp := unsafe { a^ }
    unsafe { a^ = b^ }
    unsafe { b^ = temp }
}

increment :: fn(mut x: i64) {
    x = x + 1
}

read_sum :: fn(a: i64, b: i64) -> i64 {
    a + b
}

main :: fn() -> i64 {
    var x : i64 = 10
    var y : i64 = 20
    swap(ptr_to(x), ptr_to(y))
    unsafe { printf("%lld\n", x) }
    unsafe { printf("%lld\n", y) }
    increment(x)
    unsafe { printf("%lld\n", x) }
    unsafe { printf("%lld\n", read_sum(x, y)) }
    0
}
"#;

#[test]
fn native_pointers_and_references() {
    let Some(output) = compile_and_run_unaudited("pointers", POINTERS) else {
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
    var p := Point { x = 3, y = 4 }
    unsafe { printf("%lld\n", p.x) }
    unsafe { printf("%lld\n", read_sum(p)) }
    p.x = 100
    scale(p, 2)
    unsafe { printf("%lld\n", p.x) }
    unsafe { printf("%lld\n", p.y) }

    m := Mixed { tag = 7, value = 1000, flag = 1 }
    unsafe { printf("%lld\n", m.tag) }
    unsafe { printf("%lld\n", m.value) }
    unsafe { printf("%lld\n", m.flag) }
    0
}
"#;

#[test]
fn native_structs_and_field_access() {
    let Some(output) = compile_and_run_unaudited("structs", STRUCTS) else {
        return;
    };
    assert_eq!(output, "3\n7\n200\n8\n7\n1000\n1\n");
}

const ARRAYS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

sum_array :: fn(a: [5]i64) -> i64 {
    var total : i64 = 0
    for i in 0..5 {
        total = total + a[i]
    }
    total
}

main :: fn() -> i64 {
    var nums := [10, 20, 30, 40, 50]
    unsafe { printf("%lld\n", nums[0]) }
    unsafe { printf("%lld\n", nums[2]) }
    nums[1] = 99
    unsafe { printf("%lld\n", nums[1]) }
    var running : i64 = 0
    for i in 0..5 {
        running = running + nums[i]
    }
    unsafe { printf("%lld\n", running) }
    unsafe { printf("%lld\n", sum_array(nums)) }
    0
}
"#;

#[test]
fn native_arrays_and_indexing() {
    let Some(output) = compile_and_run_unaudited("arrays", ARRAYS) else {
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
    unsafe { printf("%lld\n", unwrap_or_neg(ok)) }
    unsafe { printf("%lld\n", unwrap_or_neg(err)) }
    unsafe { printf("%lld\n", grade(90)) }
    unsafe { printf("%lld\n", grade(80)) }
    unsafe { printf("%lld\n", grade(50)) }
    0
}
"#;

#[test]
fn native_enums_and_match() {
    let Some(output) = compile_and_run_unaudited("enums", ENUMS) else {
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
    var o := Outer { one = Inner { a = 1, b = 2 }, two = Inner { a = 3, b = 4 } }
    o.one = o.two
    unsafe { printf("%lld\n", o.one.a) }
    unsafe { printf("%lld\n", o.one.b) }

    var arr := [P { x = 1, y = 2 }, P { x = 9, y = 8 }]
    arr[0] = arr[1]
    unsafe { printf("%lld\n", arr[0].x) }
    unsafe { printf("%lld\n", arr[0].y) }
    0
}
"#;

#[test]
fn native_aggregate_assignment_between_places() {
    let Some(output) =
        compile_and_run_unaudited("agg_assign", AGGREGATE_ASSIGNMENT)
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
    unsafe { printf("%lld\n", bound.a) }
    unsafe { printf("%lld\n", bound.b) }

    returned := get_inner(o)
    unsafe { printf("%lld\n", returned.a) }

    arr := [Inner { a = 10, b = 20 }, Inner { a = 30, b = 40 }]
    picked := arr[1]
    unsafe { printf("%lld\n", picked.a) }
    unsafe { printf("%lld\n", picked.b) }
    0
}
"#;

#[test]
fn native_aggregate_by_value_reads() {
    let Some(output) =
        compile_and_run_unaudited("agg_reads", AGGREGATE_BY_VALUE_READS)
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
        case .Idle: -1
    }
}

main :: fn() -> i64 {
    a := Task { id = 1, state = State::Running { pid = 42 } }
    b := Task { id = 2, state = State::Idle }
    c := Task { id = 3, state = State::Done { code = 7 } }
    unsafe { printf("%lld\n", describe(a)) }
    unsafe { printf("%lld\n", describe(b)) }
    unsafe { printf("%lld\n", describe(c)) }

    arr := [State::Done { code = 9 }, State::Idle]
    unsafe { printf("%lld\n", first(arr, 0)) }
    unsafe { printf("%lld\n", first(arr, 1)) }
    0
}
"#;

#[test]
fn native_match_on_enum_place() {
    let Some(output) =
        compile_and_run_unaudited("match_place", MATCH_ENUM_PLACE)
    else {
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
    var copy := p
    copy.x = copy.x * factor
    copy.y = copy.y * factor
    copy.x + copy.y
}

main :: fn() -> i64 {
    origin := Point { x = 3, y = 4 }
    unsafe { printf("%lld\n", manhattan(origin)) }
    other := Point { x = 5, y = 6 }
    unsafe { printf("%lld\n", scaled_sum(other, 10)) }
    0
}
"#;

#[test]
fn native_pass_struct_by_value() {
    let Some(output) = compile_and_run_unaudited("byvalue", BY_VALUE) else {
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
    unsafe { printf("%lld\n", a.x) }
    unsafe { printf("%lld\n", a.y) }
    sum := add_points(make_point(1, 2), make_point(10, 20))
    unsafe { printf("%lld\n", sum.x) }
    unsafe { printf("%lld\n", sum.y) }
    0
}
"#;

#[test]
fn native_return_struct_by_value() {
    let Some(output) = compile_and_run_unaudited("retagg", RETURN_AGGREGATE)
    else {
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
    unsafe { printf("%lld\n", sizeof(i64)) }
    unsafe { printf("%lld\n", sizeof(i32)) }
    unsafe { printf("%lld\n", sizeof(Point)) }
    unsafe { printf("%lld\n", sizeof([4]i64)) }
    p := Point { x = 1, y = 2 }
    e := Entity { hp = 1, mana = 2, name = 3 }
    unsafe { printf("%lld\n", measure(p)) }
    unsafe { printf("%lld\n", measure(e)) }
    unsafe { printf("%lld\n", measure(42)) }
    0
}
"#;

#[test]
fn native_sizeof_including_generic() {
    let Some(output) = compile_and_run_unaudited("sizeof", SIZEOF) else {
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
    unsafe { printf("%lld\n", identity(42)) }
    unsafe { printf("%lld\n", max_of(3, 9)) }

    small : i32 = 7
    widened : i64 = identity(small)
    unsafe { printf("%lld\n", widened) }

    p := first_of(Point { x = 5, y = 6 }, Point { x = 1, y = 2 })
    unsafe { printf("%lld\n", p.x) }

    w := wrap(Point { x = 8, y = 9 })
    unsafe { printf("%lld\n", w.y) }

    var a : i64 = 100
    var b : i64 = 200
    swap(a, b)
    unsafe { printf("%lld\n", a) }
    unsafe { printf("%lld\n", b) }
    0
}
"#;

#[test]
fn native_generic_functions_monomorphize() {
    let Some(output) = compile_and_run_unaudited("generics", GENERIC_FUNCTIONS)
    else {
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
    unsafe { printf("%lld\n", p.first + p.second) }
    unsafe { printf("%lld\n", sum_pair(p)) }

    pts : Pair<Point> = Pair { first = Point { x = 1, y = 2 }, second = Point { x = 3, y = 4 } }
    unsafe { printf("%lld\n", pts.first.x + pts.second.y) }

    mixed : Both<i64, i32> = Both { left = 100, right = 5 }
    unsafe { printf("%lld\n", mixed.left) }
    unsafe { printf("%lld\n", mixed.right) }

    b : Buffer<i64> = Buffer { data = [7, 8, 9], count = 3 }
    unsafe { printf("%lld\n", b.data[2]) }

    var w := Wrapper { pair = p, tag = 99 }
    w.pair.second = 40
    unsafe { printf("%lld\n", w.pair.first + w.pair.second) }
    unsafe { printf("%lld\n", w.tag) }
    0
}
"#;

#[test]
fn native_generic_structs_monomorphize() {
    let Some(output) =
        compile_and_run_unaudited("generic_structs", GENERIC_STRUCTS)
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
    unsafe { printf("%lld\n", sum(Point { x = 8, y = 9 })) }
    var q := Point { x = 3, y = 4 }
    unsafe { printf("%lld\n", scaled(q, 10)) }
    0
}
"#;

#[test]
fn native_borrow_struct_literal_at_call() {
    let Some(output) =
        compile_and_run_unaudited("borrow_struct", BORROW_STRUCT_LITERAL)
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
            case 0: -1
            case _: pid
        }
        case .Idle: 0
    }
}

main :: fn() -> i64 {
    unsafe { printf("%lld\n", pid_of(State::Running { pid = 42 })) }
    unsafe { printf("%lld\n", pid_of(State::Running { pid = 0 })) }
    unsafe { printf("%lld\n", pid_of(State::Idle)) }
    0
}
"#;

#[test]
fn native_borrow_aggregate_literal() {
    let Some(output) =
        compile_and_run_unaudited("borrow_lit", BORROW_AGGREGATE_LITERAL)
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
    unsafe { printf("%lld\n", size_of($i64)) }
    unsafe { printf("%lld\n", size_of($Entity)) }

    var world : Slab<Entity, 16> = slab_new()
    h := insert($Entity, $16, world, Entity { hp = 100, mana = 30 })
    unsafe { printf("%lld\n", world[h].hp + world[h].mana) }
    0
}
"#;

#[test]
fn native_explicit_type_arguments() {
    let Some(output) =
        compile_and_run_unaudited("explicit_types", EXPLICIT_TYPE_ARGUMENTS)
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
    unsafe { printf("%lld\n", inferred.first + inferred.second) }

    var pool : Slab<Pair<i64>, 4> = slab_new()
    h := insert($Pair<i64>, $4, pool, Pair { first = 3, second = 4 })
    unsafe { printf("%lld\n", pool[h].first + pool[h].second) }
    0
}
"#;

#[test]
fn native_generic_construction_inference() {
    let Some(output) = compile_and_run_unaudited(
        "gen_construct",
        GENERIC_CONSTRUCTION_INFERENCE,
    ) else {
        return;
    };
    assert_eq!(output, "42\n7\n");
}

const LINEAR_RESOURCE_NATIVE: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32
frost_rt_read_i64 :: extern fn(data: File) -> i64

File :: linear struct { fd: i64 }

open :: fn(n: i64) -> File { File { fd = n } }

main :: fn() -> i64 {
    f := open(42)
    unsafe { printf("%lld\n", frost_rt_read_i64(f)) }
    0
}
"#;

#[test]
fn native_linear_resource_consumed_by_extern() {
    let Some(output) =
        compile_and_run_unaudited("linear", LINEAR_RESOURCE_NATIVE)
    else {
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
    var total : i64 = 0
    for i in 0..3 {
        total = total + arr[i].first + arr[i].second
    }
    unsafe { printf("%lld\n", total) }

    o : Op<i64> = Op { f = inc, seed = 41 }
    g := o.f
    unsafe { printf("%lld\n", g(o.seed)) }

    var x : Pair<i64> = Pair { first = 1, second = 2 }
    var y : Pair<i64> = Pair { first = 9, second = 8 }
    swap(x, y)
    unsafe { printf("%lld\n", x.first + y.second) }
    0
}
"#;

#[test]
fn native_generic_instance_combinations() {
    let Some(output) = compile_and_run_unaudited(
        "gen_instance",
        GENERIC_INSTANCE_COMBINATIONS,
    ) else {
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
    unsafe { printf("%lld\n", p.first + p.second) }

    b := wrap(99)
    unsafe { printf("%lld\n", unwrap(b)) }

    w := Tagged::Some { p = Pair { first = 5, second = 6 } }
    r := match w {
        case .Some { p }: p.first + p.second
        case .None: 0
    }
    unsafe { printf("%lld\n", r) }
    0
}
"#;

#[test]
fn native_generic_factories_and_payloads() {
    let Some(output) =
        compile_and_run_unaudited("generic_factories", GENERIC_FACTORIES)
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
    unsafe { printf("%lld\n", p.first.second) }
    unsafe { printf("%lld\n", p.second.first) }

    q : Pair<Pair<Pair<i64>>> = Pair {
        first = Pair { first = Pair { first = 5, second = 6 }, second = Pair { first = 7, second = 8 } },
        second = Pair { first = Pair { first = 9, second = 10 }, second = Pair { first = 11, second = 12 } }
    }
    unsafe { printf("%lld\n", q.first.first.second) }
    unsafe { printf("%lld\n", q.second.second.first) }
    0
}
"#;

#[test]
fn native_nested_generic_structs() {
    let Some(output) =
        compile_and_run_unaudited("nested_generics", NESTED_GENERIC_STRUCTS)
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
    unsafe { printf("%lld\n", p.a + p.b) }

    unsafe { printf("%lld\n", pick_first(42, 99)) }
    unsafe { printf("%lld\n", second(1, 7)) }

    q := pick_first(Pair { a = 10, b = 20 }, 5)
    unsafe { printf("%lld\n", q.b) }
    0
}
"#;

#[test]
fn native_generic_multiple_type_parameters() {
    let Some(output) =
        compile_and_run_unaudited("generics_multi", GENERIC_MULTI_PARAM)
    else {
        return;
    };
    assert_eq!(output, "7\n42\n7\n20\n");
}

// A type argument written out, with a `bool` among them. `true` and `false` are
// their own kind of expression rather than literals, so a read-mode `$T` bound
// to `bool` used to stay a reference and the call tried to pass a constant by
// address. Both compilers take this, which is what the two of them accepting
// the same language means.
const GENERIC_BOOL_ARGUMENT: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Pair :: struct($A: Type, $B: Type) { first: A, second: B }

make :: fn($A: Type, $B: Type, a: $A, b: $B) -> Pair<A, B> {
    Pair { first = a, second = b }
}

count :: fn($A: Type, a: $A) -> i64 { 1 }

main :: fn() -> i64 {
    q := make($i64, $bool, 9, true)
    unsafe { printf("%lld\n", q.first) }
    if (q.second) { unsafe { printf("%lld\n", 1) } } else { unsafe { printf("%lld\n", 0) } }
    unsafe { printf("%lld\n", count($bool, false)) }
    0
}
"#;

// A literal that says which instance it is, and a repeat count that is a value
// parameter. Both are written inside a generic and both are only knowable once
// the generic is instantiated: the literal names the instance rather than
// waiting for the context to say, and the count arrives with the argument.
const GENERIC_WRITTEN_OUT: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Pair :: struct($A: Type, $B: Type) { first: A, second: B }
Buffer :: struct($T: Type, $N: usize) { items: [N]T, count: i64 }

filled :: fn($T: Type, $N: usize, value: $T) -> Buffer<T, N> {
    Buffer { items = [value; N], count = N }
}

main :: fn() -> i64 {
    p := Pair<i64, bool> { first = 7, second = true }
    unsafe { printf("%lld\n", p.first) }
    if (p.second) { unsafe { printf("%lld\n", 1) } } else { unsafe { printf("%lld\n", 0) } }

    var b := filled($i64, $4, 3)
    b.items[1] = 9
    unsafe { printf("%lld\n", b.items[0] + b.items[1] + b.count) }
    0
}
"#;

#[test]
fn a_generic_literal_may_name_its_arguments() {
    let Some(output) =
        compile_and_run_unaudited("genwritten", GENERIC_WRITTEN_OUT)
    else {
        return;
    };
    assert_eq!(output, "7\n1\n16\n");
}

// The same program through the self-hosted compiler, on both of its backends.
const SELF_HOSTED_GENERIC_WRITTEN: &str = "import \"io.frost\"\nPair :: struct($A: Type, $B: Type) { first: A, second: B }\n\
     Buffer :: struct($T: Type, $N: usize) { items: [N]T, count: i64 }\n\
     filled :: fn($T: Type, $N: usize, value: $T) -> Buffer<T, N> {\n\
     \x20   Buffer { items = [value; N], count = N }\n\
     }\n\
     main :: fn() -> i64 {\n\
     \x20   p := Pair<i64, bool> { first = 7, second = true }\n\
     \x20   print(\"{}\\n\", p.first)\n\
     \x20   if (p.second) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
     \x20   var b := filled($i64, $4, 3)\n\
     \x20   b.items[1] = 9\n\
     \x20   print(\"{}\\n\", b.items[0] + b.items[1] + b.count)\n\
     \x20   0\n\
     }\n";

#[test]
fn self_hosted_takes_a_generic_literal_written_out() {
    let Some(output) = selfhosted_unaudited_output(
        "shgenwritten",
        SELF_HOSTED_GENERIC_WRITTEN,
    ) else {
        return;
    };
    assert_eq!(output, "7\n1\n16\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shgenwritten_input.frost");
    std::fs::write(&input, SELF_HOSTED_GENERIC_WRITTEN).unwrap();
    let Some(c_source) = self_hosted_emits("shgenwritten", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shgenwritten", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

#[test]
fn a_boolean_is_a_value_when_it_is_a_generic_argument() {
    let Some(output) =
        compile_and_run_unaudited("genericbool", GENERIC_BOOL_ARGUMENT)
    else {
        return;
    };
    assert_eq!(output, "9\n1\n1\n");
}

// The same program through the self-hosted compiler, on both of its backends.
const SELF_HOSTED_GENERIC_BOOL: &str = "import \"io.frost\"\nPair :: struct($A: Type, $B: Type) { first: A, second: B }\n\
     make :: fn($A: Type, $B: Type, a: $A, b: $B) -> Pair<A, B> {\n\
     \x20   Pair { first = a, second = b }\n\
     }\n\
     count :: fn($A: Type, a: $A) -> i64 { 1 }\n\
     main :: fn() -> i64 {\n\
     \x20   q := make($i64, $bool, 9, true)\n\
     \x20   print(\"{}\\n\", q.first)\n\
     \x20   if (q.second) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
     \x20   print(\"{}\\n\", count($bool, false))\n\
     \x20   0\n\
     }\n";

#[test]
fn self_hosted_takes_a_boolean_as_a_generic_argument() {
    let Some(output) =
        selfhosted_unaudited_output("shgenbool", SELF_HOSTED_GENERIC_BOOL)
    else {
        return;
    };
    assert_eq!(output, "9\n1\n1\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shgenbool_input.frost");
    std::fs::write(&input, SELF_HOSTED_GENERIC_BOOL).unwrap();
    let Some(c_source) = self_hosted_emits("shgenbool", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shgenbool", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
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
        unsafe { printf("%lld\n", classify(i)) }
    }
    0
}
"#;

#[test]
fn native_tuple_pattern_match() {
    let Some(output) = compile_and_run_unaudited("tuple", TUPLE_MATCH) else {
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
    var i : i64 = 0
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
    var world : Slab<Entity, 8> = slab_new()
    reset($Entity, $8, world)

    ha := insert($Entity, $8, world, Entity { hp = 50, mana = 10 })
    hb := insert($Entity, $8, world, Entity { hp = 20, mana = 5 })

    unsafe { printf("%lld\n", world[ha].hp) }
    world[ha].hp = 60
    unsafe { printf("%lld\n", world[ha].hp) }

    heal(world[ha], 15)
    unsafe { printf("%lld\n", world[ha].hp) }
    unsafe { printf("%lld\n", total(world[ha])) }

    copy := world[hb]
    unsafe { printf("%lld\n", copy.mana) }
    unsafe { printf("%lld\n", total(world[hb])) }
    0
}
"#;

#[test]
fn native_pool_handle_deref_as_place() {
    let Some(output) =
        compile_and_run_unaudited("pool_deref", POOL_HANDLE_DEREF)
    else {
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
    unsafe { printf("%lld\n", apply(double, 21)) }
    unsafe { printf("%lld\n", apply(square, 9)) }
    unsafe { printf("%lld\n", apply_twice(increment, 40)) }
    g := double
    unsafe { printf("%lld\n", g(50)) }
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
    var v : i64 = 10
    for i in 0..3 {
        f := ops[i]
        v = f(v)
    }
    unsafe { printf("%lld\n", v) }
    unsafe { printf("%lld\n", ops[1](21)) }
    0
}
"#;

#[test]
fn native_function_pointer_array() {
    let Some(output) =
        compile_and_run_unaudited("fnptr_array", FUNCTION_POINTER_ARRAY)
    else {
        return;
    };
    assert_eq!(output, "19\n42\n");
}

// A field name is read where nothing else can appear, so a keyword is taken as
// the name it is written as. webgpu.json calls a member `type`, and every
// generated binding had to rename it. `struct` and `return` are also C
// keywords, so the C backend has to prefix them or write a syntax error.
const KEYWORD_FIELD_NAMES: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Descriptor :: struct { type: i64, match: i64, struct: i64, return: i64 }

// A variant field may be named for a keyword too. A pattern binding may not,
// since that name becomes a local and `type` in expression position is the
// keyword, so a variant's keyword field is bound alongside a named one here.
Shape :: enum { Round { type: i64, id: i64 }, Flat { id: i64 } }

kind_of :: fn(shape: Shape) -> i64 {
    match shape {
        case .Round { id }: id
        case .Flat { id }: id
    }
}

main :: fn() -> i64 {
    var d : Descriptor = Descriptor {
        type = 3, match = 4, struct = 5, return = 6,
    }
    d.type = 9
    unsafe { printf("%lld\n", d.type) }
    unsafe { printf("%lld\n", d.match) }
    unsafe { printf("%lld\n", d.struct) }
    unsafe { printf("%lld\n", d.return) }
    unsafe { printf("%lld\n", kind_of(Shape::Round { type = 1, id = 7 })) }
    unsafe { printf("%lld\n", kind_of(Shape::Flat { id = 8 })) }
    0
}
"#;

#[test]
fn a_field_may_be_named_for_a_keyword() {
    let Some(output) =
        compile_and_run_unaudited("kwfield", KEYWORD_FIELD_NAMES)
    else {
        return;
    };
    assert_eq!(output, "9\n4\n5\n6\n7\n8\n");
}

// An array size may name a constant, in a field, in a local's type, and as a
// repeat count. Both are part of a type or expanded into elements while
// parsing, so the value has to be known there; it is read off the token stream
// before the parse, which is why CAPACITY works above the line declaring it.
const CONSTANT_ARRAY_SIZES: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Buffer :: struct { bytes: [CAPACITY]u8, used: i64 }

CAPACITY :: 8
// A constant expression, and one that reads an earlier constant, both of which
// the spec already allowed where a compile-time integer is required.
DOUBLE :: CAPACITY * 2
STRIDE :: 1 << 4 | 0

fill :: fn(mut buffer: Buffer) {
    var index : i64 = 0
    while (index < CAPACITY) {
        buffer.bytes[index] = cast($u8, 65 + index)
        index = index + 1
    }
    buffer.used = CAPACITY
}

main :: fn() -> i64 {
    var buffer : Buffer = Buffer { bytes = [0; CAPACITY], used = 0 }
    fill(buffer)
    unsafe { printf("%lld\n", buffer.bytes[0]) }
    unsafe { printf("%lld\n", buffer.bytes[7]) }
    unsafe { printf("%lld\n", buffer.used) }
    unsafe { printf("%lld\n", sizeof(Buffer)) }

    var wide : [DOUBLE]i64 = [3; DOUBLE]
    wide[15] = 9
    unsafe { printf("%lld\n", wide[0]) }
    unsafe { printf("%lld\n", wide[15]) }
    unsafe { printf("%lld\n", sizeof([DOUBLE]i64)) }
    unsafe { printf("%lld\n", sizeof([STRIDE]u8)) }
    0
}
"#;

#[test]
fn a_constant_sizes_an_array() {
    let Some(output) =
        compile_and_run_unaudited("constarray", CONSTANT_ARRAY_SIZES)
    else {
        return;
    };
    assert_eq!(output, "65\n72\n8\n16\n3\n9\n128\n16\n");
}

const TOP_LEVEL_CONSTANTS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

LIMIT :: 100
STEP :: 5
OFFSET :: -3
COMPUTED :: 2 * 4 + 1

main :: fn() -> i64 {
    unsafe { printf("%lld\n", LIMIT) }
    unsafe { printf("%lld\n", STEP) }
    unsafe { printf("%lld\n", OFFSET) }
    unsafe { printf("%lld\n", COMPUTED) }
    var total : i64 = 0
    for i in 0..LIMIT {
        if (i % STEP == 0) { total = total + 1 }
    }
    unsafe { printf("%lld\n", total) }
    0
}
"#;

#[test]
fn native_top_level_constants() {
    let Some(output) =
        compile_and_run_unaudited("constants", TOP_LEVEL_CONSTANTS)
    else {
        return;
    };
    assert_eq!(output, "100\n5\n-3\n9\n20\n");
}

const FORWARD_REFERENCES: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    unsafe { printf("%lld\n", is_even(10)) }
    unsafe { printf("%lld\n", is_odd(7)) }
    unsafe { printf("%lld\n", double_it(21)) }
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
    let Some(output) = compile_and_run_unaudited("forward", FORWARD_REFERENCES)
    else {
        return;
    };
    assert_eq!(output, "1\n1\n42\n");
}

#[test]
fn native_function_pointers() {
    let Some(output) = compile_and_run_unaudited("funcptr", FUNCTION_POINTERS)
    else {
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
    var total : i64 = 0
    for i in 0..4 {
        total = total + f(values[i])
    }
    total
}

main :: fn() -> i64 {
    a := Vec3 { x = 1, y = 2, z = 3 }
    b := Vec3 { x = 4, y = 5, z = 6 }
    unsafe { printf("%lld\n", dot(a, b)) }

    c := Shape::Circle { radius = 10 }
    sq := Shape::Box { side = 7 }
    unsafe { printf("%lld\n", area(c)) }
    unsafe { printf("%lld\n", area(sq)) }

    unsafe { printf("%lld\n", fib(15)) }

    nums := [1, 2, 3, 4]
    unsafe { printf("%lld\n", apply_to_array(triple, nums)) }
    0
}
"#;

#[test]
fn native_combined_features() {
    let Some(output) = compile_and_run_unaudited("kitchen", KITCHEN_SINK)
    else {
        return;
    };
    assert_eq!(output, "32\n300\n49\n610\n30\n");
}

const DEFER: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

work :: fn() -> i64 {
    unsafe { printf("%lld\n", 1) }
    defer unsafe { printf("%lld\n", 2) }
    defer unsafe { printf("%lld\n", 3) }
    unsafe { printf("%lld\n", 4) }
    99
}

main :: fn() -> i64 {
    r := work()
    unsafe { printf("%lld\n", r) }
    0
}
"#;

#[test]
fn native_defer_runs_lifo_at_return() {
    let Some(output) = compile_and_run_unaudited("defer", DEFER) else {
        return;
    };
    assert_eq!(output, "1\n4\n3\n2\n99\n");
}

const DEFER_NESTED_RETURN: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

work :: fn(which: i64) -> i64 {
    defer unsafe { printf("%lld\n", 8) }
    defer unsafe { printf("%lld\n", 9) }
    if (which == 0) {
        unsafe { printf("%lld\n", 1) }
        return 100
    }
    unsafe { printf("%lld\n", 2) }
    200
}

main :: fn() -> i64 {
    unsafe { printf("%lld\n", work(0)) }
    unsafe { printf("%lld\n", work(1)) }
    0
}
"#;

#[test]
fn native_defer_runs_on_a_nested_early_return() {
    let Some(output) =
        compile_and_run_unaudited("defer_nested", DEFER_NESTED_RETURN)
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
    var o := Outer { tag = 5, inner = Inner { a = 10, b = 20 } }
    unsafe { printf("%lld\n", o.tag) }
    unsafe { printf("%lld\n", o.inner.a) }
    unsafe { printf("%lld\n", sum_inner(o)) }
    o.inner.a = 99
    unsafe { printf("%lld\n", o.inner.a) }
    unsafe { printf("%lld\n", sum_inner(o)) }
    0
}
"#;

#[test]
fn native_nested_structs() {
    let Some(output) = compile_and_run_unaudited("nested", NESTED_STRUCTS)
    else {
        return;
    };
    assert_eq!(output, "5\n10\n30\n99\n119\n");
}

const DATA_LAYOUTS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Particle :: struct { x: i64, y: i64 }
Grid :: struct { cells: [4]i64, count: i64 }

main :: fn() -> i64 {
    var ps := [Particle { x = 1, y = 2 }, Particle { x = 3, y = 4 }, Particle { x = 5, y = 6 }]
    unsafe { printf("%lld\n", ps[0].x) }
    unsafe { printf("%lld\n", ps[1].y) }
    ps[2].x = 99
    unsafe { printf("%lld\n", ps[2].x) }

    var total : i64 = 0
    for i in 0..3 {
        total = total + ps[i].x
    }
    unsafe { printf("%lld\n", total) }

    var g := Grid { cells = [10, 20, 30, 40], count = 4 }
    unsafe { printf("%lld\n", g.cells[1]) }
    g.cells[2] = 77
    unsafe { printf("%lld\n", g.cells[2]) }
    unsafe { printf("%lld\n", g.count) }
    0
}
"#;

#[test]
fn native_array_of_structs_and_struct_of_arrays() {
    let Some(output) = compile_and_run_unaudited("data_layouts", DATA_LAYOUTS)
    else {
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
    unsafe { printf("%lld\n", describe(leaf)) }
    unsafe { printf("%lld\n", describe(pair)) }

    var grid := [[1, 2, 3], [4, 5, 6]]
    unsafe { printf("%lld\n", grid[0][2]) }
    unsafe { printf("%lld\n", grid[1][1]) }
    grid[1][2] = 99
    unsafe { printf("%lld\n", grid[1][2]) }
    0
}
"#;

#[test]
fn native_aggregate_enum_payloads_and_2d_arrays() {
    let Some(output) =
        compile_and_run_unaudited("agg_payloads", AGGREGATE_PAYLOADS)
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
    unsafe { printf("%lld\n", unwrap_or(r, -1)) }

    odds := [1, 3, 5, 7, 9, 11]
    r2 := find_first_even(odds)
    unsafe { printf("%lld\n", unwrap_or(r2, -1)) }
    0
}
"#;

#[test]
fn native_enum_returned_by_value() {
    let Some(output) = compile_and_run_unaudited("enum_byval", ENUM_BY_VALUE)
    else {
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
    var p := Point { x = 1, y = 2 }
    bump(p.x)
    unsafe { printf("%lld\n", p.x) }
    unsafe { printf("%lld\n", p.y) }

    q := origin()
    unsafe { printf("%lld\n", q.x) }
    unsafe { printf("%lld\n", q.y) }
    0
}
"#;

#[test]
fn native_field_borrow_and_returned_struct() {
    let Some(output) = compile_and_run_unaudited("field_borrow", FIELD_BORROW)
    else {
        return;
    };
    assert_eq!(output, "101\n2\n7\n9\n");
}

const INTEGER_SEMANTICS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    a : i32 = -5
    b : i32 = 3
    unsafe { printf("%lld\n", a / b) }
    unsafe { printf("%lld\n", a % b) }

    big : i64 = 1000000000
    unsafe { printf("%lld\n", big * 3) }

    neg : i64 = -100
    shifted : i64 = neg >> 2
    unsafe { printf("%lld\n", shifted) }

    wide : i16 = 30000
    unsafe { printf("%lld\n", wide + 100) }

    mask : i64 = 255
    unsafe { printf("%lld\n", mask & 15) }
    unsafe { printf("%lld\n", mask | 256) }

    small : u8 = 200
    unsafe { printf("%lld\n", wrap_add(small, 100)) }
    0
}
"#;

#[test]
fn native_integer_semantics_match() {
    let Some(output) =
        compile_and_run_unaudited("int_semantics", INTEGER_SEMANTICS)
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
    var i : i64 = 0
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
    var p : Slab<Entity, 8> = slab_new()
    reset($Entity, $8, p)

    ha := insert($Entity, $8, p, Entity { hp = 100, mana = 30 })
    hb := insert($Entity, $8, p, Entity { hp = 50, mana = 10 })

    unsafe { printf("%lld\n", index_of(ha)) }
    unsafe { printf("%lld\n", index_of(hb)) }
    unsafe { printf("%lld\n", generation_of(ha)) }

    unsafe { printf("%lld\n", p[ha].hp) }
    p[ha].hp = 999
    unsafe { printf("%lld\n", p[ha].hp) }

    unsafe { printf("%lld\n", alive($Entity, $8, p, ha)) }
    unsafe { printf("%lld\n", release($Entity, $8, p, ha)) }
    unsafe { printf("%lld\n", alive($Entity, $8, p, ha)) }

    hc := insert($Entity, $8, p, Entity { hp = 7, mana = 7 })
    unsafe { printf("%lld\n", index_of(hc)) }
    unsafe { printf("%lld\n", generation_of(hc)) }
    unsafe { printf("%lld\n", alive($Entity, $8, p, ha)) }
    0
}
"#;

const WIDENING_BINDINGS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

widen :: fn(x: i8) -> i64 { x }

main :: fn() -> i64 {
    a : i8 = -5
    unsafe { printf("%lld\n", widen(a)) }
    b : i16 = -1000
    c : i64 = b
    unsafe { printf("%lld\n", c) }
    small : i32 = 42
    wide : i64 = small
    unsafe { printf("%lld\n", wide) }
    0
}
"#;

#[test]
fn native_widening_in_let_bindings() {
    let Some(output) = compile_and_run_unaudited("widening", WIDENING_BINDINGS)
    else {
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
        case .None: -1
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
    unsafe { printf("%lld\n", a.x) }
    unsafe { printf("%lld\n", b.y) }

    none := choose(0)
    some := choose(7)
    unsafe { printf("%lld\n", unwrap(none)) }
    unsafe { printf("%lld\n", unwrap(some)) }

    p := classify(4, 6)
    q := classify(3, 6)
    unsafe { printf("%lld\n", p.x) }
    unsafe { printf("%lld\n", q.y) }
    0
}
"#;

#[test]
fn native_match_returns_aggregate_by_value() {
    let Some(output) =
        compile_and_run_unaudited("match_agg", MATCH_RETURNS_AGGREGATE)
    else {
        return;
    };
    assert_eq!(output, "1\n8\n-1\n70\n1\n1\n");
}

#[test]
fn native_generational_pool_and_handles() {
    let Some(output) = compile_and_run_unaudited("gen_pool", GENERATIONAL_POOL)
    else {
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
        ("diff_vec", VEC_LIBRARY),
        ("diff_map", MAP_LIBRARY),
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
        ("diff_strfield", STR_IN_A_FIELD),
        ("diff_rtnames", RUNTIME_SYMBOL_NAMES),
        ("diff_constarray", CONSTANT_ARRAY_SIZES),
        ("diff_kwfield", KEYWORD_FIELD_NAMES),
        ("diff_structfail", STRUCT_FAILURE_TYPE),
        ("diff_printnarrow", PRINT_NARROW_VALUES),
        ("diff_parenstmt", PARENTHESISED_STATEMENT),
        ("diff_tuplepat", TUPLE_PATTERNS),
        ("diff_enumeq", ENUM_EQUALITY),
        ("diff_forseq", FOR_OVER_A_SEQUENCE),
        ("diff_multiret", MULTIPLE_RETURN_VALUES),
        ("diff_dotvariant", INFERRED_VARIANTS),
        ("diff_inflit", INFERRED_LITERALS),
        ("diff_indagg", INDIRECT_AGGREGATE_RETURN),
        ("diff_distinct", DISTINCT_TYPES),
        ("diff_genbool", GENERIC_BOOL_ARGUMENT),
        ("diff_genwritten", GENERIC_WRITTEN_OUT),
        ("diff_wherebound", WHERE_BOUNDS),
        ("diff_format", IO_PRINTS),
        ("diff_fieldcall", FIELD_CALLS),
        ("diff_enumvalues", ENUM_VALUES),
        ("diff_bundle", CAPABILITY_BUNDLE),
        ("diff_composed", COMPOSED_BUNDLES),
        ("diff_linenum", LINEAR_ENUM),
        ("diff_packlist", COMPILE_TIME_LIST),
        ("diff_packempty", EMPTY_LIST),
        ("diff_mutscalar", MUT_SCALAR_PARAMETER),
        ("diff_fieldwalk", FIELD_WALK),
        ("diff_fieldgeneric", FIELD_WALK_GENERIC),
        ("diff_failure", FAILURE_SET_PARSE),
        ("diff_bracedarm", BRACED_ARMS),
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
    var x : i64 = 5
    bump(x)
    unsafe { printf("%lld\n", x) }
    twice(x)
    unsafe { printf("%lld\n", x) }
    0
}
"#;
    let Some(output) = compile_and_run_unaudited("mutscalar", source) else {
        return;
    };
    assert_eq!(output, "6\n8\n");
}

// One compile should name every mistake it can, so a fix-recompile-repeat loop
// costs one pass rather than one pass per mistake.
#[test]
fn two_move_errors_in_one_function_are_both_reported() {
    let source = "Thing :: struct { n: i64 }\n\
                  eat :: fn(move t: Thing) -> i64 { t.n }\n\
                  one :: fn() -> i64 {\n\
                  \x20   t := Thing { n = 1 }\n\
                  \x20   u := Thing { n = 2 }\n\
                  \x20   a := eat(t)\n\
                  \x20   b := eat(t)\n\
                  \x20   c := eat(u)\n\
                  \x20   d := eat(u)\n\
                  \x20   a + b + c + d\n}\n\
                  main :: fn() -> i64 { one() }\n";
    let message = compile_error("twomoves", source);
    assert!(
        message.contains("use of moved value 't'"),
        "expected the first move error, got:\n{message}"
    );
    assert!(
        message.contains("use of moved value 'u'"),
        "expected the second move error, got:\n{message}"
    );
}

// Past a move the binding stays moved, so every later mention fails the same
// way. Those are echoes of one mistake, and saying it once is the useful answer.
#[test]
fn a_repeated_use_of_one_moved_value_is_reported_once() {
    let source = "Thing :: struct { n: i64 }\n\
                  eat :: fn(move t: Thing) -> i64 { t.n }\n\
                  one :: fn() -> i64 {\n\
                  \x20   t := Thing { n = 1 }\n\
                  \x20   a := eat(t)\n\
                  \x20   b := eat(t)\n\
                  \x20   c := eat(t)\n\
                  \x20   a + b + c\n}\n\
                  main :: fn() -> i64 { one() }\n";
    let message = compile_error("echomoves", source);
    assert_eq!(
        message.matches("use of moved value 't'").count(),
        1,
        "expected the echo to be suppressed, got:\n{message}"
    );
}

#[test]
fn a_move_error_in_each_of_two_functions_is_reported() {
    let source = "Thing :: struct { n: i64 }\n\
                  eat :: fn(move t: Thing) -> i64 { t.n }\n\
                  first :: fn() -> i64 {\n\
                  \x20   t := Thing { n = 1 }\n\
                  \x20   eat(t) + eat(t)\n}\n\
                  second :: fn() -> i64 {\n\
                  \x20   u := Thing { n = 2 }\n\
                  \x20   eat(u) + eat(u)\n}\n\
                  main :: fn() -> i64 { first() + second() }\n";
    let message = compile_error("twofnmoves", source);
    assert!(
        message.contains("use of moved value 't'")
            && message.contains("use of moved value 'u'"),
        "expected a report from each function, got:\n{message}"
    );
}

// A function's locals die when it returns, so a pointer or a slice into one of
// them may not be the thing it answers with.
#[test]
fn a_pointer_into_the_frame_may_not_be_returned() {
    let source = "leak :: fn() -> ^i64 {\n\
                  \x20   var local : i64 = 42\n\
                  \x20   ptr_to(local)\n}\n\
                  main :: fn() -> i64 { 0 }\n";
    let message = compile_error("frameptr", source);
    assert!(
        message.contains("pointer into the frame of"),
        "expected a frame escape error, got:\n{message}"
    );
}

// `ptr_to` is refused outside an `unsafe` block, so the wrapped form is the
// only one a real program can contain. A frame check that does not look through
// the block does not fire on any pointer a program can actually write.
#[test]
fn a_frame_pointer_may_not_escape_wrapped_in_an_unsafe_block() {
    let source = "leak :: fn() -> ^i64 {\n\
                  \x20   var local : i64 = 42\n\
                  \x20   unsafe { ptr_to(local) }\n}\n\
                  main :: fn() -> i64 { 0 }\n";
    let message = compile_error_checked("frameunsafe", source);
    assert!(
        message.contains("pointer into the frame of"),
        "expected a frame escape error, got:\n{message}"
    );
}

#[test]
fn a_frame_pointer_may_not_be_returned_from_an_unsafe_block() {
    let source = "leak :: fn() -> ^i64 {\n\
                  \x20   var local : i64 = 42\n\
                  \x20   return unsafe { ptr_to(local) }\n}\n\
                  main :: fn() -> i64 { 0 }\n";
    let message = compile_error_checked("frameunsafereturn", source);
    assert!(
        message.contains("pointer into the frame of"),
        "expected a frame escape error, got:\n{message}"
    );
}

#[test]
fn a_frame_pointer_bound_inside_an_unsafe_block_may_not_be_returned() {
    let source = "leak :: fn() -> ^i64 {\n\
                  \x20   var local : i64 = 42\n\
                  \x20   held := unsafe { ptr_to(local) }\n\
                  \x20   held\n}\n\
                  main :: fn() -> i64 { 0 }\n";
    let message = compile_error_checked("frameunsafebound", source);
    assert!(
        message.contains("pointer into the frame of"),
        "expected a frame escape error, got:\n{message}"
    );
}

// Every road out of an `unsafe` block, since that block is where a frame
// pointer is necessarily formed. Reading the value the block answers with is
// not enough on its own: the pointer can be returned from inside it, bound
// inside it, or handed back by a branch of a block used as a value.
#[test]
fn a_frame_pointer_may_not_leave_an_unsafe_block_by_any_road() {
    let cases = [
        (
            "returninside",
            "leak :: fn() -> ^i64 {\n\
             \x20   var x : i64 = 42\n\
             \x20   unsafe { return ptr_to(x) }\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "boundinside",
            "leak :: fn() -> ^i64 {\n\
             \x20   var x : i64 = 42\n\
             \x20   unsafe { p := ptr_to(x)\n\
             \x20   p }\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "ifasvalue",
            "leak :: fn(c: bool) -> ^i64 {\n\
             \x20   var x : i64 = 42\n\
             \x20   var y : i64 = 7\n\
             \x20   unsafe { if (c) { ptr_to(x) } else { ptr_to(y) } }\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "matchasvalue",
            "Pick :: enum { One, Two }\n\
             leak :: fn(p: Pick) -> ^i64 {\n\
             \x20   var x : i64 = 42\n\
             \x20   unsafe { match p { case .One: ptr_to(x) case .Two: ptr_to(x) } }\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "nestedunsafe",
            "leak :: fn() -> ^i64 {\n\
             \x20   var x : i64 = 42\n\
             \x20   unsafe { ptr_to(x) }\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "structwrap",
            "Box :: struct { p: ^i64 }\n\
             leak :: fn() -> Box {\n\
             \x20   var x : i64 = 42\n\
             \x20   unsafe { Box { p = ptr_to(x) } }\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "sliceinside",
            "leak :: fn() -> []i64 {\n\
             \x20   arr := [11, 22, 33]\n\
             \x20   unsafe { slice_from($i64, ptr_to(arr[0]), 3) }\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "slicebound",
            "leak :: fn() -> []i64 {\n\
             \x20   arr := [11, 22, 33]\n\
             \x20   unsafe { v := slice_from($i64, ptr_to(arr[0]), 3)\n\
             \x20   v }\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
    ];
    for (name, source) in cases {
        let message = compile_error_checked(name, source);
        assert!(
            message.contains("pointer into the frame of"),
            "{name} should not compile, got:\n{message}"
        );
    }
}

// Provenance the walk has to establish rather than assume. Each of these handed
// back a view of a dead frame, and each did it by taking a road the walk had not
// been taught: an ordinary call, a call through a function pointer, an
// assignment into a local, a `return` inside a match arm, the address of a
// `move` parameter. Answering "this does not point into the frame" for a shape
// nobody enumerated is what they all had in common.
#[test]
fn a_frame_view_may_not_leave_by_a_road_the_walk_cannot_follow() {
    let cases = [
        (
            "launderedcall",
            "launder :: fn(p: ^i64) -> ^i64 { p }\n\
             leak :: fn() -> ^i64 {\n\
             \x20   var local : i64 = 42\n\
             \x20   launder(ptr_to(local))\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "launderedref",
            "Holder :: struct { a: [4]i64 }\n\
             pick :: fn(mut h: Holder, i: i64) -> ref i64 { h.a[i] }\n\
             leak :: fn() -> ref i64 {\n\
             \x20   var local : Holder = Holder { a = [11, 22, 33, 44] }\n\
             \x20   pick(local, 0)\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "launderedfnptr",
            "Ops :: struct { pass: fn(^i64) -> ^i64 }\n\
             identity :: fn(p: ^i64) -> ^i64 { p }\n\
             leak :: fn() -> ^i64 {\n\
             \x20   var local : i64 = 42\n\
             \x20   ops := Ops { pass = identity }\n\
             \x20   ops.pass(ptr_to(local))\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "assignedthenreturned",
            "start :: fn(seed: ^i64) -> ^i64 { seed }\n\
             leak :: fn(seed: ^i64) -> ^i64 {\n\
             \x20   var local : i64 = 42\n\
             \x20   var p : ^i64 = start(seed)\n\
             \x20   p = ptr_to(local)\n\
             \x20   p\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "returninmatcharm",
            "Pick :: enum { One, Two }\n\
             leak :: fn(p: Pick, fallback: ^i64) -> ^i64 {\n\
             \x20   var x : i64 = 42\n\
             \x20   match p {\n\
             \x20       case .One: { return ptr_to(x) }\n\
             \x20       case .Two: { x = x + 1 }\n\
             \x20   }\n\
             \x20   fallback\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "moveparameteraddress",
            "Point :: struct { x: i64, y: i64 }\n\
             leak :: fn(move p: Point) -> ^i64 { ptr_to(p.x) }\n\
             main :: fn() -> i64 { 0 }\n",
        ),
    ];
    for (name, source) in cases {
        let message = compile_error_checked(name, source);
        assert!(
            message.contains("region:"),
            "{name} should not compile, got:\n{message}"
        );
    }
}

// The other half of the same rule: what the walk can trace to a parameter, an
// allocation capability or the heap still compiles. Without these the inversion
// would be a check that refuses everything.
#[test]
fn a_view_traced_to_storage_that_outlives_the_call_is_allowed() {
    let source = "import \"io.frost\"\nHolder :: struct { a: [4]i64 }\n\
                  Ops :: struct { pass: fn(^i64) -> ^i64 }\n\
                  identity :: fn(p: ^i64) -> ^i64 { p }\n\
                  pick :: fn(mut h: Holder, i: i64) -> ref i64 { h.a[i] }\n\
                  through :: fn(mut h: Holder) -> ref i64 { pick(h, 0) }\n\
                  handed :: fn(p: ^i64) -> ^i64 { identity(p) }\n\
                  indirect :: fn(p: ^i64, ops: Ops) -> ^i64 { ops.pass(p) }\n\
                  span :: fn(held: []i64, from: i64) -> []i64 {\n\
                  \x20   count := slice_len(held) - from\n\
                  \x20   unsafe { slice_from($i64, ptr_to(held[from]), count) }\n}\n\
                  bump :: fn(mut v: i64) -> i64 { v }\n\
                  main :: fn() -> i64 {\n\
                  \x20   var h : Holder = Holder { a = [1, 2, 3, 4] }\n\
                  \x20   print(\"{}\\n\", bump(through(h)))\n\
                  \x20   0\n}\n";
    let directory = std::env::temp_dir();
    let source_path = directory.join("frost_traced_view.frost");
    let exe_path = directory
        .join(format!("frost_traced_view{}", std::env::consts::EXE_SUFFIX));
    std::fs::write(&source_path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("-o")
        .arg(&exe_path)
        .arg(&source_path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&source_path);
    assert!(
        output.status.success(),
        "a view traced to a parameter should compile, got:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// The self-hosted compiler answers the same way, on the same programs. Two
// compilers that disagree about which programs are safe is one guarantee written
// twice and kept in one place, so the same roads and the same traced view are
// run through both. The compiler is built once here rather than per case, since
// building it is most of what the test costs.
#[test]
fn the_self_hosted_compiler_traces_a_frame_view_the_same_way() {
    let Some(compiler) = build_self_hosted_compiler("frametrace") else {
        return;
    };
    let refused = [
        (
            "launderedcall",
            "launder :: fn(p: ^i64) -> ^i64 { p }\n\
             leak :: fn() -> ^i64 {\n\
             \x20   var local : i64 = 42\n\
             \x20   launder(ptr_to(local))\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "launderedref",
            "Holder :: struct { a: [4]i64 }\n\
             pick :: fn(mut h: Holder, i: i64) -> ref i64 { h.a[i] }\n\
             leak :: fn() -> ref i64 {\n\
             \x20   var local : Holder = Holder { a = [11, 22, 33, 44] }\n\
             \x20   pick(local, 0)\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "launderedfnptr",
            "Ops :: struct { pass: fn(^i64) -> ^i64 }\n\
             identity :: fn(p: ^i64) -> ^i64 { p }\n\
             leak :: fn() -> ^i64 {\n\
             \x20   var local : i64 = 42\n\
             \x20   ops := Ops { pass = identity }\n\
             \x20   ops.pass(ptr_to(local))\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "assignedthenreturned",
            "start :: fn(seed: ^i64) -> ^i64 { seed }\n\
             leak :: fn(seed: ^i64) -> ^i64 {\n\
             \x20   var local : i64 = 42\n\
             \x20   var p : ^i64 = start(seed)\n\
             \x20   p = ptr_to(local)\n\
             \x20   p\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "moveparameteraddress",
            "Point :: struct { x: i64, y: i64 }\n\
             leak :: fn(move p: Point) -> ^i64 { ptr_to(p.x) }\n\
             main :: fn() -> i64 { 0 }\n",
        ),
    ];
    let directory = std::env::temp_dir();
    let emitted = directory.join("frost_shframe_out.c");
    for (name, source) in refused {
        let input = directory.join(format!("frost_shframe_{name}.frost"));
        std::fs::write(&input, source).unwrap();
        let run = Command::new(&compiler)
            .arg(&input)
            .arg("-o")
            .arg(&emitted)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        let message = String::from_utf8_lossy(&run.stderr).to_string();
        assert!(
            !run.status.success(),
            "{name} should not compile under the self-hosted compiler"
        );
        assert!(
            message.contains("pointer into the frame of")
                || message.contains("cannot be traced"),
            "{name} should be refused by the frame check, got:\n{message}"
        );
    }

    // And the other half: what the walk can trace still compiles, so the
    // inversion is not a check that refuses everything.
    let allowed = "import \"io.frost\"\nHolder :: struct { a: [4]i64 }\n\
                   Ops :: struct { pass: fn(^i64) -> ^i64 }\n\
                   identity :: fn(p: ^i64) -> ^i64 { p }\n\
                   pick :: fn(mut h: Holder, i: i64) -> ref i64 { h.a[i] }\n\
                   through :: fn(mut h: Holder) -> ref i64 { pick(h, 0) }\n\
                   handed :: fn(p: ^i64) -> ^i64 { identity(p) }\n\
                   indirect :: fn(p: ^i64, ops: Ops) -> ^i64 { ops.pass(p) }\n\
                   bump :: fn(mut v: i64) -> i64 { v }\n\
                   main :: fn() -> i64 {\n\
                   \x20   var h : Holder = Holder { a = [1, 2, 3, 4] }\n\
                   \x20   print(\"{}\\n\", bump(through(h)))\n\
                   \x20   0\n}\n";
    let input = directory.join("frost_shframe_allowed.frost");
    std::fs::write(&input, allowed).unwrap();
    let run = Command::new(&compiler)
        .arg(&input)
        .arg("-o")
        .arg(&emitted)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&emitted);
    assert!(
        run.status.success(),
        "a view traced to a parameter should compile, got:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
}

// Reading back through a pointer at a local that holds a frame pointer hands
// the frame pointer out again.
#[test]
fn a_frame_pointer_read_back_through_a_pointer_may_not_be_returned() {
    let source = "leak :: fn() -> ^i64 {\n\
                  \x20   var x : i64 = 42\n\
                  \x20   var p : ^i64 = unsafe { ptr_to(x) }\n\
                  \x20   unsafe { pp := ptr_to(p)\n\
                  \x20   pp^ }\n}\n\
                  main :: fn() -> i64 { 0 }\n";
    let message = compile_error_checked("framederef", source);
    assert!(
        message.contains("pointer into the frame of"),
        "expected a frame escape error, got:\n{message}"
    );
    assert_eq!(
        message.matches("pointer into the frame of").count(),
        1,
        "one escape should be named once, got:\n{message}"
    );
}

// `p^` is the ordinary way to read a local through a pointer. A function
// answering with a scalar is not handing out a view of anything.
#[test]
fn reading_a_local_through_a_pointer_is_still_allowed() {
    let source = "import \"io.frost\"\nread :: fn() -> i64 {\n\
                  \x20   var x : i64 = 42\n\
                  \x20   p := unsafe { ptr_to(x) }\n\
                  \x20   unsafe { p^ }\n}\n\
                  main :: fn() -> i64 { print(\"{}\\n\", read())\n0 }\n";
    let Some(output) = compile_and_run_unaudited("okderef", source) else {
        return;
    };
    assert_eq!(output, "42\n");
}

const EXHAUSTIVE_SHAPE: &str =
    "Shape :: enum { Circle { r: i64 }, Rect { w: i64, h: i64 }, Point }\n";

// A match on an enum has to cover every variant or say what the rest do.
// Without it, adding a variant silently changes the meaning of every match on
// that enum instead of pointing at the places that now need a case.
#[test]
fn a_match_missing_a_variant_is_refused() {
    let source = format!(
        "{EXHAUSTIVE_SHAPE}\
         area :: fn(s: Shape) -> i64 {{\n\
         \x20   match s {{\n\
         \x20       case .Circle {{ r }}: r * r * 3\n\
         \x20       case .Rect {{ w, h }}: w * h\n\
         \x20   }}\n}}\n\
         main :: fn() -> i64 {{ area(Shape::Point {{}}) }}\n"
    );
    let message = compile_error("exhaustive", &source);
    assert!(
        message.contains("does not cover") && message.contains(".Point"),
        "expected the uncovered variant to be named, got:\n{message}"
    );
}

#[test]
fn a_match_naming_every_variant_is_allowed() {
    let source = format!(
        "import \"io.frost\"\n{EXHAUSTIVE_SHAPE}\
         area :: fn(s: Shape) -> i64 {{\n\
         \x20   match s {{\n\
         \x20       case .Circle {{ r }}: r * r * 3\n\
         \x20       case .Rect {{ w, h }}: w * h\n\
         \x20       case .Point: 0\n\
         \x20   }}\n}}\n\
         main :: fn() -> i64 {{ print(\"{{}}\\n\", area(Shape::Rect {{ w = 3, h = 4 }}))\n0 }}\n"
    );
    let Some(output) = compile_and_run_unaudited("exhaustiveall", &source)
    else {
        return;
    };
    assert_eq!(output, "12\n");
}

#[test]
fn a_match_with_a_wildcard_need_not_name_every_variant() {
    let source = format!(
        "import \"io.frost\"\n{EXHAUSTIVE_SHAPE}\
         area :: fn(s: Shape) -> i64 {{\n\
         \x20   match s {{\n\
         \x20       case .Circle {{ r }}: r * r * 3\n\
         \x20       case _: 0\n\
         \x20   }}\n}}\n\
         main :: fn() -> i64 {{ print(\"{{}}\\n\", area(Shape::Point {{}}))\n0 }}\n"
    );
    let Some(output) = compile_and_run_unaudited("exhaustivewild", &source)
    else {
        return;
    };
    assert_eq!(output, "0\n");
}

// The same rule in the self-hosted compiler, which is held to the bootstrap's
// feature set by test rather than by hope.
#[test]
fn self_hosted_refuses_a_match_missing_a_variant() {
    let source = format!(
        "{EXHAUSTIVE_SHAPE}\
         area :: fn(s: Shape) -> i64 {{\n\
         \x20   match s {{\n\
         \x20       case .Circle {{ r }}: r * r * 3\n\
         \x20       case .Rect {{ w, h }}: w * h\n\
         \x20   }}\n}}\n\
         main :: fn() -> i64 {{ area(Shape::Point {{}}) }}\n"
    );
    let Some(message) = self_hosted_rejects("shexhaustive", &source) else {
        return;
    };
    assert!(
        message.contains("does not cover") && message.contains(".Point"),
        "expected the uncovered variant to be named, got:\n{message}"
    );
}

const TRY_HEAD: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
FileError :: enum { NotFound, Denied }

read_size :: fn(ok: i64) -> i64 ! FileError {
    if (ok == 0) { return FileError::NotFound {} }
    return 42
}
"#;

const TRY_TAIL: &str = r#"
report :: fn(ok: i64) -> i64 {
    match use_it(ok) {
        case .Ok { value }: value
        case .Err { error }: -1
    }
}

main :: fn() -> i64 {
    unsafe { printf("%lld\n", report(1)) }
    unsafe { printf("%lld\n", report(0)) }
    0
}
"#;

// `?` inside an `unsafe` block or a `with` block is still `?`. The failure-set
// lowering looked for one in neither, so the desugaring never ran and the
// operator survived into the backend as an unsupported expression. Each answers
// the same as the plain form, which is what says the desugaring is equivalent
// and not merely present.
#[test]
fn a_try_inside_an_unsafe_block_lowers_like_any_other() {
    let source = format!(
        "{TRY_HEAD}\n\
         use_it :: fn(ok: i64) -> i64 ! FileError {{\n\
         \x20   n := unsafe {{ read_size(ok)? }}\n\
         \x20   return n + 1\n}}\n{TRY_TAIL}"
    );
    let Some(output) = compile_and_run_unaudited("tryinunsafe", &source) else {
        return;
    };
    assert_eq!(output, "43\n-1\n");
}

#[test]
fn a_try_inside_a_with_block_lowers_like_any_other() {
    let source = format!(
        "{TRY_HEAD}\n\
         use_it :: fn(ok: i64) -> i64 ! FileError {{\n\
         \x20   var arena : Arena<64> = Arena {{ data = [0; 64], offset = 0 }}\n\
         \x20   var n : i64 = 0\n\
         \x20   with arena {{\n\
         \x20       n = read_size(ok)?\n\
         \x20   }}\n\
         \x20   return n + 1\n}}\n{TRY_TAIL}"
    );
    let Some(output) = compile_and_run_unaudited("tryinwith", &source) else {
        return;
    };
    assert_eq!(output, "43\n-1\n");
}

// A move made inside an `unsafe` block is a move. Found by asking which checks
// were only ever tested with the unsafety gate off, which is what let the frame
// escapes through for as long as it did.
#[test]
fn a_value_moved_inside_an_unsafe_block_may_not_be_used_again() {
    let source = "Thing :: struct { n: i64 }\n\
                  eat :: fn(move t: Thing) -> i64 { t.n }\n\
                  main :: fn() -> i64 {\n\
                  \x20   t := Thing { n = 1 }\n\
                  \x20   a := unsafe { eat(t) }\n\
                  \x20   b := unsafe { eat(t) }\n\
                  \x20   a + b\n}\n";
    let message = compile_error_checked("moveinunsafe", source);
    assert!(
        message.contains("use of moved value 't'"),
        "expected a use-after-move, got:\n{message}"
    );
}

// The other side of it: consuming a linear value inside an unsafe block counts
// as consuming it, so walking in must not turn that into a double report.
#[test]
fn a_linear_value_consumed_inside_an_unsafe_block_is_consumed() {
    let source = "import \"io.frost\"\nFile :: linear struct { fd: i64 }\n\
                  close :: fn(move f: File) -> i64 { f.fd }\n\
                  main :: fn() -> i64 {\n\
                  \x20   f := File { fd = 1 }\n\
                  \x20   print(\"{}\\n\", unsafe { close(f) })\n\
                  \x20   0\n}\n";
    let Some(output) = compile_and_run_unaudited("linearinunsafe", source)
    else {
        return;
    };
    assert_eq!(output, "1\n");
}

const ARENA_PRELUDE: &str = "Arena :: struct($N: usize) { data: [N]u8, offset: i64 }\n\
     alloc_int :: fn(mut a: Arena<256>) -> ^i64 {\n\
     \x20   slot := unsafe { ptr_to(a.data[a.offset]) }\n\
     \x20   a.offset = a.offset + sizeof(i64)\n\
     \x20   unsafe { ptr_cast($i64, slot) }\n}\n";

// An arena pointer read back through a pointer at it still belongs to the
// region. Telling that from reading the value the arena pointer names is what
// makes this narrow enough to be worth having.
#[test]
fn an_arena_pointer_read_back_through_a_pointer_may_not_outlive_its_region() {
    let source = format!(
        "{ARENA_PRELUDE}\
         grab :: fn() -> i64 {{\n\
         \x20   var arena : Arena<256> = Arena {{ data = [0; 256], offset = 0 }}\n\
         \x20   var out : ^i64 = unsafe {{ ptr_to(arena.offset) }}\n\
         \x20   with arena {{\n\
         \x20       p := alloc_int(arena)\n\
         \x20       pp := unsafe {{ ptr_to(p) }}\n\
         \x20       out = unsafe {{ pp^ }}\n\
         \x20   }}\n\
         \x20   unsafe {{ out^ }}\n}}\n\
         main :: fn() -> i64 {{ grab() }}\n"
    );
    let message = compile_error_checked("arenaderef", &source);
    assert!(
        message.contains("escapes its region"),
        "expected a region escape error, got:\n{message}"
    );
}

// Reading the value an arena pointer names is an ordinary read, not an escape.
// This is the side the rule above has to leave alone.
#[test]
fn reading_the_value_an_arena_pointer_names_is_allowed() {
    let source = format!(
        "import \"io.frost\"\n{ARENA_PRELUDE}\
         grab :: fn() -> i64 {{\n\
         \x20   var arena : Arena<256> = Arena {{ data = [0; 256], offset = 0 }}\n\
         \x20   var total : i64 = 0\n\
         \x20   with arena {{\n\
         \x20       p := alloc_int(arena)\n\
         \x20       unsafe {{ p^ = 7 }}\n\
         \x20       total = unsafe {{ p^ }}\n\
         \x20   }}\n\
         \x20   print(\"{{}}\\n\", total)\n\
         \x20   0\n}}\n\
         main :: fn() -> i64 {{ grab() }}\n"
    );
    let Some(output) = compile_and_run_unaudited("arenaread", &source) else {
        return;
    };
    assert_eq!(output, "7\n");
}

// A callback registration names storage in this frame on purpose, and is safe
// because linearity forces it to be consumed in the same function. Wrapping it
// in an `unsafe` block must not turn it into an escape: only the block a
// function actually answers with can carry one out.
#[test]
fn a_registration_inside_a_non_final_unsafe_block_is_not_an_escape() {
    let source = "Ctx :: struct { got: i64 }
                  reg :: extern fn($cb: fn(mut Ctx, ^u8), move ctx: Ctx)
                  on :: fn(mut c: Ctx, a: ^u8) { c.got = 1 }
                  f :: fn() -> i64 {
                      var ctx : Ctx = Ctx { got = 0 }
                      unsafe { reg($on, ctx) }
                      0
}
                  main :: fn() -> i64 { f() }
";
    let directory = std::env::temp_dir();
    let source_path = directory.join("frost_ok_registration.frost");
    let object = directory.join("frost_ok_registration.o");
    std::fs::write(&source_path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--native")
        .arg("-o")
        .arg(&object)
        .arg(&source_path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&source_path);
    assert!(
        output.status.success(),
        "a registration in a non-final unsafe block should compile, got:
{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// The tightened check must not start refusing a pointer the function was handed,
// which is not its frame's to begin with.
#[test]
fn a_pointer_handed_in_still_passes_back_out() {
    let source = "pass :: fn(p: ^i64) -> ^i64 { p }\n\
                  main :: fn() -> i64 { 0 }\n";
    let directory = std::env::temp_dir();
    let source_path = directory.join("frost_ok_handedin.frost");
    let object = directory.join("frost_ok_handedin.o");
    std::fs::write(&source_path, source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--native")
        .arg("-o")
        .arg(&object)
        .arg(&source_path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&source_path);
    assert!(
        output.status.success(),
        "a pointer handed in should still be returnable, got:\n{}",
        String::from_utf8_lossy(&output.stderr)
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
    var n : i64 = 7
    q := pass_through(ptr_to(n))
    unsafe { printf("%lld\n", q^) }
    0
}
"#;
    let Some(output) = compile_and_run_unaudited("framepass", source) else {
        return;
    };
    assert_eq!(output, "7\n");
}

// std/io writes every kind of value a program prints to look at: a `str` by
// its bytes, an integer as its digits, a float the way C writes `%g`. A line
// built from several values is several calls, and the pieces land on one line
// because only `print_*_line` writes the newline.
const IO_PRINTS: &str = r#"import "io.frost"

Point :: struct { x: i64, y: i64 }

main :: fn() -> i64 {
    print("hello\n")
    name := "world"
    print("{}\n", name)
    print("hp ")
    print("{}", 7)
    print(" of ")
    print("{}\n", 20)
    print("a then ")
    print("{}", 2)
    print(" then ")
    print("{}\n", 3.5)
    print("braces {{ and }} stay\n")
    p := Point { x = 3, y = 4 }
    print("point ")
    print("{}", p.x)
    print(" ")
    print("{}\n", p.y)
    0
}
"#;

const IO_PRINTS_EXPECTED: &str = "hello\nworld\nhp 7 of 20\na then 2 then 3.5\nbraces { and } stay\npoint 3 4\n";

#[test]
fn io_writes_every_kind_of_value() {
    let Some(output) = compile_and_run_unaudited("ioprints", IO_PRINTS) else {
        return;
    };
    assert_eq!(output, IO_PRINTS_EXPECTED);
}

// The same program through the self-hosted compiler, on both of its backends.
#[test]
fn self_hosted_io_writes_every_kind_of_value() {
    let Some(output) = selfhosted_unaudited_output("shioprints", IO_PRINTS)
    else {
        return;
    };
    assert_eq!(output, IO_PRINTS_EXPECTED);

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shioprints_input.frost");
    std::fs::write(&input, IO_PRINTS).unwrap();
    let Some(c_source) = self_hosted_emits("shioprints", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shioprints", &c_source) else {
        return;
    };
    assert_eq!(
        via_c, IO_PRINTS_EXPECTED,
        "the self-hosted C backend disagrees"
    );
}

// A `where` bound holds a generic to what its body needs, over a fixed
// vocabulary of questions the compiler already answers about a type. It is a
// precondition rather than a set of operations a type registers into: nothing
// implements it, nothing is named, and there is nothing to resolve. The bound
// is read at the call, so a type that cannot work is refused against the line
// the reader wrote.
const WHERE_BOUNDS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

twice :: fn($T: Type, v: $T) -> T where is_numeric(T) {
    v + v
}

first :: fn($T: Type, xs: []T) -> T where is_numeric(T) && !is_pointer(T) {
    xs[0]
}

widest :: fn($T: Type, a: $T, b: $T) -> T where is_integer(T) || is_float(T) {
    if (a > b) { a } else { b }
}

main :: fn() -> i64 {
    unsafe { printf("%lld\n", twice($i64, 21)) }
    var numbers : [3]i64 = [7, 8, 9]
    unsafe { printf("%lld\n", first($i64, numbers)) }
    unsafe { printf("%lld\n", widest($i64, 4, 9)) }
    0
}
"#;

#[test]
fn a_where_bound_holds_a_generic_to_what_it_needs() {
    let Some(output) = compile_and_run_unaudited("wherebound", WHERE_BOUNDS)
    else {
        return;
    };
    assert_eq!(output, "42\n7\n9\n");
}

// The bound is the point, so what it refuses is the test. Both compilers refuse
// the same two programs: a type the bound does not hold for, and a predicate
// that is not one of the bounds a type can be held to.
#[test]
fn a_where_bound_is_checked_at_the_call() {
    let cases = [
        (
            "Point :: struct { x: i64 }\n\
             twice :: fn($T: Type, v: $T) -> T where is_numeric(T) { v }\n\
             main :: fn() -> i64 {\n\
             \x20   p := Point { x = 1 }\n\
             \x20   q := twice($Point, p)\n\
             \x20   q.x\n\
             }\n",
            "does not hold",
        ),
        (
            "twice :: fn($T: Type, v: $T) -> T where is_sortable(T) { v }\n\
             main :: fn() -> i64 { twice($i64, 1) }\n",
            "not one of the bounds",
        ),
    ];
    for (index, (source, expected)) in cases.iter().enumerate() {
        let message = compile_error(&format!("boundbad{index}"), source);
        assert!(message.contains(expected), "the bootstrap said:\n{message}");
        let Some(compiler) = build_self_hosted_compiler("boundbad") else {
            continue;
        };
        let directory = std::env::temp_dir();
        let input = directory.join(format!("frost_boundbad{index}.frost"));
        std::fs::write(&input, source).unwrap();
        let refused = Command::new(&compiler)
            .env("FROST_INPUT", &input)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            !refused.status.success(),
            "the self-hosted compiler accepted case {index}"
        );
        let said = String::from_utf8_lossy(&refused.stderr);
        assert!(
            said.contains(expected),
            "the self-hosted compiler said:\n{said}"
        );
    }
}

// The same program through the self-hosted compiler, on both of its backends.
// The float case is here because a generic instantiated at `f64` used to pass
// its argument in an integer register while the specialized body read it from
// an SSE one.
const SELF_HOSTED_WHERE: &str = "import \"io.frost\"\ntwice :: fn($T: Type, v: $T) -> T where is_numeric(T) {\n\
     \x20   v + v\n\
     }\n\
     first :: fn($T: Type, xs: []T) -> T where !is_pointer(T) {\n\
     \x20   xs[0]\n\
     }\n\
     main :: fn() -> i64 {\n\
     \x20   print(\"{}\\n\", twice($i64, 21))\n\
     \x20   print(\"{}\\n\", twice($f64, 1.5))\n\
     \x20   var numbers : [3]i64 = [7, 8, 9]\n\
     \x20   print(\"{}\\n\", first($i64, numbers))\n\
     \x20   0\n\
     }\n";

#[test]
fn self_hosted_holds_a_generic_to_its_bound() {
    let Some(output) =
        selfhosted_unaudited_output("shwhere", SELF_HOSTED_WHERE)
    else {
        return;
    };
    assert_eq!(output, "42\n3\n7\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shwhere_input.frost");
    std::fs::write(&input, SELF_HOSTED_WHERE).unwrap();
    let Some(c_source) = self_hosted_emits("shwhere", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shwhere", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// An enum used as a value, in the three places that were holes: a parameter,
// compared against a variant; a variant written at a call, where the argument
// wants an address; and the answer of a call, whose tag is read out of what came
// back.
const ENUM_VALUES: &str = r#"import "io.frost"

Kind :: enum { One, Two }

is_one :: fn(k: Kind) -> bool { k == Kind::One }

pick :: fn(n: i64) -> Kind {
    if (n > 0) { return Kind::One }
    Kind::Two
}

main :: fn() -> i64 {
    if (is_one(Kind::One)) { print("{}\n", 1) } else { print("{}\n", 0) }
    if (is_one(Kind::Two)) { print("{}\n", 1) } else { print("{}\n", 0) }
    if (pick(5) == Kind::One) { print("{}\n", 1) } else { print("{}\n", 0) }
    if (pick(0) == Kind::One) { print("{}\n", 1) } else { print("{}\n", 0) }
    k := pick(1)
    if (is_one(k)) { print("{}\n", 1) } else { print("{}\n", 0) }
    0
}
"#;

#[test]
fn an_enum_is_a_value_like_any_other() {
    let Some(output) = compile_and_run_unaudited("enumvalues", ENUM_VALUES)
    else {
        return;
    };
    assert_eq!(output, "1\n0\n1\n0\n1\n");
}

#[test]
fn self_hosted_uses_an_enum_as_a_value() {
    let Some(output) = selfhosted_unaudited_output("shenumvalues", ENUM_VALUES)
    else {
        return;
    };
    assert_eq!(output, "1\n0\n1\n0\n1\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shenumvalues_input.frost");
    std::fs::write(&input, ENUM_VALUES).unwrap();
    let Some(c_source) = self_hosted_emits("shenumvalues", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shenumvalues", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// A struct whose fields are function pointers, called through. This is what an
// interface is here: an ordinary value with an ordinary type, so the field is
// reached the way any field is and the call goes through what it holds.
const FIELD_CALLS: &str = r#"import "io.frost"

Point :: struct { x: i64, y: i64 }

Ops :: struct {
    less: fn(i64, i64) -> bool,
    combine: fn(i64, i64) -> i64,
    scale: fn(f64, i64) -> f64,
    origin: fn(i64) -> Point,
}

i64_less :: fn(a: i64, b: i64) -> bool { a < b }
i64_add :: fn(a: i64, b: i64) -> i64 { a + b }
f64_double :: fn(v: f64, by: i64) -> f64 { v * 2.0 }
make_point :: fn(n: i64) -> Point { Point { x = n, y = n * 2 } }

main :: fn() -> i64 {
    ops := Ops { less = i64_less, combine = i64_add, scale = f64_double,
        origin = make_point }
    if (ops.less(1, 2)) { print("{}\n", 1) } else { print("{}\n", 0) }
    print("{}\n", ops.combine(20, 22))
    print("{}\n", ops.scale(1.5, 3))
    p := ops.origin(7)
    print("{}\n", p.x + p.y)
    0
}
"#;

#[test]
fn a_function_pointer_field_is_called_through() {
    let Some(output) = compile_and_run_unaudited("fieldcall", FIELD_CALLS)
    else {
        return;
    };
    assert_eq!(output, "1\n42\n3\n21\n");
}

#[test]
fn self_hosted_calls_through_a_function_pointer_field() {
    let Some(output) = selfhosted_unaudited_output("shfieldcall", FIELD_CALLS)
    else {
        return;
    };
    assert_eq!(output, "1\n42\n3\n21\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shfieldcall_input.frost");
    std::fs::write(&input, FIELD_CALLS).unwrap();
    let Some(c_source) = self_hosted_emits("shfieldcall", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shfieldcall", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// A bundle declared in one file, an ordering for a type declared in another,
// and a sort over both. Writing `Ordering<Point> { .. }` in a file that imports
// `Ordering` is the whole point of a bundle being an ordinary type: which names
// can begin a literal comes from what a file imports as well as what it
// declares.
#[test]
fn a_program_declares_its_own_ordering_for_its_own_type() {
    if !linker_available() {
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let directory = std::env::temp_dir().join("frost_own_ordering");
    let _ = std::fs::create_dir_all(&directory);
    let source = directory.join("own_ordering.frost");
    std::fs::write(
        &source,
        "import \"io.frost\"\nimport \"ordering.frost\"\n\
         import \"sort.frost\"\n\
         Point :: struct { x: i64, y: i64 }\n\
         point_less :: fn(a: Point, b: Point) -> bool { a.x < b.x }\n\
         point_equal :: fn(a: Point, b: Point) -> bool { a.x == b.x }\n\
         point_order :: Ordering<Point> { less = point_less, equal = point_equal }\n\
         main :: fn() -> i64 {\n\
         \x20   var points := [Point { x = 3, y = 0 }, Point { x = 1, y = 0 }]\n\
         \x20   sort($Point, $point_order, points)\n\
         \x20   print(\"{}\\n\", points[0].x)\n\
         \x20   print(\"{}\\n\", points[1].x)\n    0\n}\n",
    )
    .unwrap();
    let exe = directory.join(format!(
        "{}{}",
        unique("own_ordering"),
        std::env::consts::EXE_SUFFIX
    ));
    let build = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("-L")
        .arg(root.join("std"))
        .arg("--link")
        .arg("-o")
        .arg(&exe)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "the program did not compile:\n{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    assert_eq!(output, "1\n3\n");
    let _ = std::fs::remove_file(&exe);
}

// A failure set end to end: a function that answers with a value or a failure,
// `?` handing one up, and a caller reading which it got. The failure type is a
// type the program declares, so `error.at` is a field of it, and `error` is the
// name the `.Err` case binds.
const FAILURE_SET_PARSE: &str = r#"import "io.frost"

Parse :: struct { at: i64, code: i64 }

digit :: fn(text: str, index: i64) -> i64 ! Parse {
    byte := text[index]
    if (byte < 48 || byte > 57) {
        return { at = index, code = byte }
    }
    byte - 48
}

number :: fn(text: str) -> i64 ! Parse {
    var total : i64 = 0
    var index : i64 = 0
    while (index < str_len(text)) {
        d := digit(text, index)?
        total = total * 10 + d
        index = index + 1
    }
    total
}

report :: fn(text: str) {
    match number(text) {
        case .Ok { value }: { print("{}\n", value) }
        case .Err { error }: { print("{}\n", 0 - error.at) }
    }
}

main :: fn() -> i64 {
    report("407")
    report("4x7")
    report("40x")
    0
}
"#;

#[test]
fn a_failure_set_carries_a_value_or_a_failure() {
    let Some(output) =
        compile_and_run_unaudited("failureset", FAILURE_SET_PARSE)
    else {
        return;
    };
    assert_eq!(output, "407\n-1\n-2\n");
}

#[test]
fn self_hosted_carries_a_value_or_a_failure() {
    let Some(output) =
        selfhosted_unaudited_output("shfailureset", FAILURE_SET_PARSE)
    else {
        return;
    };
    assert_eq!(output, "407\n-1\n-2\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shfailureset_input.frost");
    std::fs::write(&input, FAILURE_SET_PARSE).unwrap();
    let Some(c_source) = self_hosted_emits("shfailureset", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shfailureset", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// A resource handed back through a failure set is still a resource: the result
// carrying one has to be consumed, and matching it is what consumes it.
const LINEAR_THROUGH_FAILURE: &str = r#"import "io.frost"

Denied :: struct { code: i64 }

File :: linear struct { fd: i64 }

open :: fn(n: i64) -> File ! Denied {
    if (n < 0) { return Denied { code = n } }
    File { fd = n }
}

close :: fn(move f: File) -> i64 { f.fd }

use_it :: fn(n: i64) -> i64 {
    match open(n) {
        case .Ok { value }: close(value)
        case .Err { error }: error.code
    }
}

held :: fn(n: i64) -> i64 {
    result := open(n)
    match result {
        case .Ok { value }: close(value)
        case .Err { error }: error.code
    }
}

main :: fn() -> i64 {
    print("{}\n", use_it(3))
    print("{}\n", use_it(-2))
    print("{}\n", held(5))
    0
}
"#;

#[test]
fn a_linear_value_survives_a_failure_set() {
    let Some(output) =
        compile_and_run_unaudited("linfail", LINEAR_THROUGH_FAILURE)
    else {
        return;
    };
    assert_eq!(output, "3\n-2\n5\n");
}

#[test]
fn self_hosted_carries_a_linear_value_through_a_failure_set() {
    let Some(output) =
        selfhosted_unaudited_output("shlinfail", LINEAR_THROUGH_FAILURE)
    else {
        return;
    };
    assert_eq!(output, "3\n-2\n5\n");
}

// And the reason it is linear: ignoring the call would drop the resource, so
// the call that answers with one has to be answered for.
#[test]
fn an_ignored_fallible_call_that_holds_a_resource_is_refused() {
    let source = "Denied :: struct { code: i64 }\n\
                  File :: linear struct { fd: i64 }\n\
                  open :: fn(n: i64) -> File ! Denied {\n\
                  \x20   if (n < 0) { return Denied { code = n } }\n\
                  \x20   File { fd = n }\n}\n\
                  close :: fn(move f: File) -> i64 { f.fd }\n\
                  main :: fn() -> i64 { _ := open(3)  0 }\n";
    let message = compile_error("linfaildrop", source);
    assert!(
        message.contains("linear"),
        "expected the dropped resource to be named, got: {message}"
    );
}

// A `{` after a case opens a block. An arm runs statements far more often than
// it answers with an unnamed value, and both compilers have to read it the same
// way or a program means two things.
const BRACED_ARMS: &str = r#"import "io.frost"

Kind :: enum { One, Two { n: i64 } }

main :: fn() -> i64 {
    held := Kind::Two { n = 7 }
    match held {
        case .One: { print("{}\n", 1) }
        case .Two { n }: { print("{}\n", n)  print("{}\n", n + 1) }
    }
    0
}
"#;

#[test]
fn a_braced_match_arm_is_a_block() {
    let Some(output) = compile_and_run_unaudited("bracedarm", BRACED_ARMS)
    else {
        return;
    };
    assert_eq!(output, "7\n8\n");
}

#[test]
fn self_hosted_reads_a_braced_match_arm_as_a_block() {
    let Some(output) = selfhosted_unaudited_output("shbracedarm", BRACED_ARMS)
    else {
        return;
    };
    assert_eq!(output, "7\n8\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shbracedarm_input.frost");
    std::fs::write(&input, BRACED_ARMS).unwrap();
    let Some(c_source) = self_hosted_emits("shbracedarm", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shbracedarm", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// Composing bundles is a struct with struct fields rather than a list of
// bounds, and the body reads through both names. The outer bundle is a
// compile-time argument, so the inner one is known too.
const COMPOSED_BUNDLES: &str = r#"import "io.frost"

Ordering :: struct($T: Type) { less: fn(T, T) -> bool }
Hashing :: struct($T: Type) { hash: fn(T) -> i64 }

Element :: struct($T: Type) {
    ordering: Ordering<T>,
    hashing: Hashing<T>,
}

i64_less :: fn(a: i64, b: i64) -> bool { a < b }
i64_hash :: fn(a: i64) -> i64 { a * 31 }

i64_ordering :: Ordering<i64> { less = i64_less }
i64_hashing :: Hashing<i64> { hash = i64_hash }

i64_element :: Element<i64> { ordering = i64_ordering, hashing = i64_hashing }

pick :: fn($T: Type, $ops: Element<T>, a: $T, b: $T) -> $T {
    if (ops.ordering.less(a, b)) { return a }
    b
}

main :: fn() -> i64 {
    print("{}\n", pick($i64, $i64_element, 7, 3))
    print("{}\n", i64_element.hashing.hash(2))
    0
}
"#;

#[test]
fn a_bundle_may_hold_other_bundles() {
    let Some(output) = compile_and_run_unaudited("composed", COMPOSED_BUNDLES)
    else {
        return;
    };
    assert_eq!(output, "3\n62\n");
}

#[test]
fn self_hosted_composes_bundles() {
    let Some(output) =
        selfhosted_unaudited_output("shcomposed", COMPOSED_BUNDLES)
    else {
        return;
    };
    assert_eq!(output, "3\n62\n");
}

// A `linear enum` is a resource with alternatives: it has to be consumed
// exactly once, and matching it is what consumes it. The arm that names a
// linear field takes that field out, so consuming it is the arm's own
// obligation.
const LINEAR_ENUM: &str = r#"import "io.frost"

File :: linear struct { fd: i64 }

Answer :: linear enum { None, Some { file: File } }

close :: fn(move f: File) -> i64 { f.fd }

take :: fn(a: Answer) -> i64 {
    match a {
        case .None: 0
        case .Some { file }: close(file)
    }
}

main :: fn() -> i64 {
    print("{}\n", take(Answer::Some { file = File { fd = 4 } }))
    print("{}\n", take(Answer::None))
    0
}
"#;

#[test]
fn a_linear_enum_is_consumed_by_matching_it() {
    let Some(output) = compile_and_run_unaudited("linenum", LINEAR_ENUM) else {
        return;
    };
    assert_eq!(output, "4\n0\n");
}

#[test]
fn self_hosted_consumes_a_linear_enum_by_matching_it() {
    let Some(output) = selfhosted_unaudited_output("shlinenum", LINEAR_ENUM)
    else {
        return;
    };
    assert_eq!(output, "4\n0\n");
}

#[test]
fn a_dropped_linear_enum_is_refused() {
    let source = "File :: linear struct { fd: i64 }\n\
                  Answer :: linear enum { None, Some { file: File } }\n\
                  close :: fn(move f: File) -> i64 { f.fd }\n\
                  main :: fn() -> i64 {\n\
                  \x20   held := Answer::Some { file = File { fd = 9 } }\n\
                  \x20   0\n}\n";
    let message = compile_error("linenumdrop", source);
    assert!(
        message.contains("linear"),
        "expected the dropped resource to be named, got: {message}"
    );
}

// A `mut` parameter is the caller's value whatever its type: the signature
// holds a pointer, the body reads and writes through it, and the call site
// takes the address. An aggregate already travels as its address, so the rule
// is only visible on a scalar, which is where it used to be missing.
const MUT_SCALAR_PARAMETER: &str = r#"import "io.frost"

bump :: fn(mut n: i64) {
    n = n + 1
}

double :: fn(mut n: i64) -> i64 {
    n = n * 2
    n
}

take :: fn(v: i64) { print("{}\n", v) }

main :: fn() -> i64 {
    var counter : i64 = 0
    bump(counter)
    print("{}\n", counter)
    bump(counter)
    print("{}\n", counter)
    print("{}\n", double(counter))
    print("{}\n", counter)
    take(double(counter))
    print("{}\n", counter)
    0
}
"#;

#[test]
fn a_mut_scalar_parameter_writes_through_to_the_caller() {
    let Some(output) =
        compile_and_run_unaudited("mutscalarparam", MUT_SCALAR_PARAMETER)
    else {
        return;
    };
    assert_eq!(output, "1\n2\n4\n4\n8\n8\n");
}

#[test]
fn self_hosted_writes_through_a_mut_scalar_parameter() {
    let Some(output) =
        selfhosted_unaudited_output("shmutscalarparam", MUT_SCALAR_PARAMETER)
    else {
        return;
    };
    assert_eq!(output, "1\n2\n4\n4\n8\n8\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shmutscalar_input.frost");
    std::fs::write(&input, MUT_SCALAR_PARAMETER).unwrap();
    let Some(c_source) = self_hosted_emits("shmutscalar", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shmutscalar", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// The self-hosted compiler's incremental build: one object per module, and a
// module whose emitted assembly is byte for byte the last build's is not
// assembled again. What comes out has to be the same program as the
// whole-program build, or the cache is not a cache but a second compiler.
#[test]
fn the_self_hosted_incremental_build_is_the_same_program() {
    let Some(compiler) = build_self_hosted_compiler("incremental") else {
        return;
    };
    if !linker_available() {
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let runtime = format!("{}/runtime/frost_runtime.c", root.display());
    let directory = std::env::temp_dir().join(unique("frost_incremental"));
    let _ = std::fs::create_dir_all(&directory);
    let source = directory.join("program.frost");
    std::fs::write(
        &source,
        "import \"io.frost\"\nhelper :: fn(n: i64) -> i64 { n * 3 }\n\
         main :: fn() -> i64 { print(\"{}\\n\", helper(7))  0 }\n",
    )
    .unwrap();
    let build = directory.join("build");
    let exe = directory.join(format!("out{}", std::env::consts::EXE_SUFFIX));

    let run = |exe: &std::path::Path, build: &std::path::Path| {
        Command::new(&compiler)
            .arg("--incremental")
            .arg("--build-dir")
            .arg(build)
            .arg("-o")
            .arg(exe)
            .arg(&source)
            .env("FROST_RUNTIME", &runtime)
            .env("FROST_RUNTIME_FROST", frost_runtime_source())
            .output()
            .unwrap()
    };
    // The same through the C backend, which splits into units the same way and
    // carries the declarations in each.
    let run_c = |exe: &std::path::Path, build: &std::path::Path| {
        Command::new(&compiler)
            .arg("--emit-c")
            .arg("--incremental")
            .arg("--build-dir")
            .arg(build)
            .arg("-o")
            .arg(exe)
            .arg(&source)
            .env("FROST_RUNTIME", &runtime)
            .env("FROST_RUNTIME_FROST", frost_runtime_source())
            .output()
            .unwrap()
    };

    let first = run(&exe, &build);
    assert!(
        first.status.success(),
        "the incremental build failed:\n{}",
        String::from_utf8_lossy(&first.stderr)
    );
    let output = Command::new(&exe).output().unwrap();
    let printed = String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");
    assert_eq!(printed, "21\n");

    // Again with nothing changed: every object is reused, and the program it
    // links is the same one.
    let again = run(&exe, &build);
    assert!(
        again.status.success(),
        "the second incremental build failed:\n{}",
        String::from_utf8_lossy(&again.stderr)
    );
    let output = Command::new(&exe).output().unwrap();
    let printed = String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");
    assert_eq!(printed, "21\n");

    // And after an edit, which has to reach the executable.
    std::fs::write(
        &source,
        "import \"io.frost\"\nhelper :: fn(n: i64) -> i64 { n * 4 }\n\
         main :: fn() -> i64 { print(\"{}\\n\", helper(7))  0 }\n",
    )
    .unwrap();
    let edited = run(&exe, &build);
    assert!(
        edited.status.success(),
        "the incremental rebuild failed:\n{}",
        String::from_utf8_lossy(&edited.stderr)
    );
    let output = Command::new(&exe).output().unwrap();
    let printed = String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");
    assert_eq!(printed, "28\n");

    // Through the C backend, from nothing, twice, and after an edit. It splits
    // into units the same way, and each carries the declarations a C unit needs
    // to call what the others define.
    let c_build = directory.join("build_c");
    let c_exe =
        directory.join(format!("out_c{}", std::env::consts::EXE_SUFFIX));
    for (expected, body) in [("28\n", "n * 4"), ("35\n", "n * 5")] {
        std::fs::write(
            &source,
            format!(
                "import \"io.frost\"\nhelper :: fn(n: i64) -> i64 {{ {body} }}\n\
                 main :: fn() -> i64 {{ print(\"{{}}\\n\", helper(7))  0 }}\n"
            ),
        )
        .unwrap();
        let built = run_c(&c_exe, &c_build);
        assert!(
            built.status.success(),
            "the C incremental build failed:\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let output = Command::new(&c_exe).output().unwrap();
        let printed =
            String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");
        assert_eq!(printed, expected);
    }

    // A `--test` build splits too. Its bodies are given names the compiler made
    // up, which sit past every module, so what says which unit one belongs in
    // is where the test was written.
    let test_build = directory.join("build_test");
    let test_exe =
        directory.join(format!("tests{}", std::env::consts::EXE_SUFFIX));
    std::fs::write(
        &source,
        "double :: fn(n: i64) -> i64 { n * 2 }\n\
         test \"doubling\" { assert(double(4) == 8) }\n",
    )
    .unwrap();
    let built = Command::new(&compiler)
        .arg("--test")
        .arg("--incremental")
        .arg("--build-dir")
        .arg(&test_build)
        .arg("-o")
        .arg(&test_exe)
        .arg(&source)
        .env("FROST_RUNTIME", &runtime)
        .env("FROST_RUNTIME_FROST", frost_runtime_source())
        .output()
        .unwrap();
    let printed = String::from_utf8_lossy(&built.stdout).replace("\r\n", "\n");
    assert!(
        printed.contains("1 passed") && printed.contains("0 failed"),
        "the incremental test build did not run the tests:\n{printed}{}",
        String::from_utf8_lossy(&built.stderr)
    );

    let _ = std::fs::remove_dir_all(&directory);
}

// A walk over a type's fields, at expansion time. A vertex format, a uniform
// layout and a descriptor table are all a table of offsets and sizes over a
// struct the program declared, and the compiler worked those numbers out to lay
// the struct out. `for field in fields(T)` writes the table once, over whatever
// fields the struct has, and `offset_of`, `sizeof` and the type predicates are
// what may be asked of one.
const FIELD_WALK: &str = r#"import "io.frost"

Vec3 :: struct { x: f32, y: f32, z: f32 }
Vec2 :: struct { u: f32, v: f32 }

Vertex :: struct {
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    id: i64,
}

main :: fn() -> i64 {
    print("{}\n", field_count(Vertex))
    for field in fields(Vertex) {
        print("{}\n", offset_of(field))
        print("{}\n", sizeof(field))
        if (is_struct(field)) { print("{}\n", 1) } else { print("{}\n", 0) }
    }
    0
}
"#;

#[test]
fn a_walk_over_a_types_fields_is_a_layout_table() {
    let Some(output) = compile_and_run_unaudited("fieldwalk", FIELD_WALK)
    else {
        return;
    };
    assert_eq!(output, "4\n0\n12\n1\n12\n12\n1\n24\n8\n1\n32\n8\n0\n");
}

#[test]
fn self_hosted_walks_a_types_fields() {
    let Some(output) = selfhosted_unaudited_output("shfieldwalk", FIELD_WALK)
    else {
        return;
    };
    assert_eq!(output, "4\n0\n12\n1\n12\n12\n1\n24\n8\n1\n32\n8\n0\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shfieldwalk_input.frost");
    std::fs::write(&input, FIELD_WALK).unwrap();
    let Some(c_source) = self_hosted_emits("shfieldwalk", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shfieldwalk", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// The same walk inside a generic, which is where it earns its keep: one
// description written once, and a table per type the call names.
const FIELD_WALK_GENERIC: &str = r#"import "io.frost"

Vec3 :: struct { x: f32, y: f32, z: f32 }
Vec2 :: struct { u: f32, v: f32 }

Vertex :: struct { position: Vec3, uv: Vec2 }
Particle :: struct { position: Vec3, life: f32 }

Attribute :: struct { offset: i64, size: i64, floating: bool }

describe :: fn($T: Type, mut out: []Attribute) -> i64 {
    var index : i64 = 0
    for field in fields(T) {
        out[index] = Attribute {
            offset = offset_of(field),
            size = sizeof(field),
            floating = is_float(field),
        }
        index = index + 1
    }
    index
}

show :: fn($T: Type) {
    var table := [Attribute { offset = 0, size = 0, floating = false }; 8]
    count := describe($T, table)
    var i : i64 = 0
    while (i < count) {
        print("{}\n", table[i].offset)
        print("{}\n", table[i].size)
        if (table[i].floating) { print("{}\n", 1) } else { print("{}\n", 0) }
        i = i + 1
    }
}

main :: fn() -> i64 {
    show($Vertex)
    print("{}\n", -1)
    show($Particle)
    0
}
"#;

#[test]
fn a_field_walk_in_a_generic_describes_the_type_it_is_given() {
    let Some(output) =
        compile_and_run_unaudited("fieldgeneric", FIELD_WALK_GENERIC)
    else {
        return;
    };
    assert_eq!(output, "0\n12\n0\n12\n8\n0\n-1\n0\n12\n0\n12\n4\n1\n");
}

#[test]
fn self_hosted_walks_the_fields_of_a_type_argument() {
    let Some(output) =
        selfhosted_unaudited_output("shfieldgeneric", FIELD_WALK_GENERIC)
    else {
        return;
    };
    assert_eq!(output, "0\n12\n0\n12\n8\n0\n-1\n0\n12\n0\n12\n4\n1\n");
}

// A field is not a value: it is asked about, and that is all. Naming one where a
// value belongs is caught where it is written.
#[test]
fn a_field_used_as_a_value_is_refused() {
    let source = "import \"io.frost\"\nPoint :: struct { x: i64, y: i64 }\n\
                  main :: fn() -> i64 {\n\
                  \x20   for field in fields(Point) { print(\"{}\\n\", field) }\n\
                  \x20   0\n}\n";
    let message = compile_error("fieldvalue", source);
    assert!(
        message.contains("not a value"),
        "expected the message to say a field is not a value, got: {message}"
    );
}

// A compile-time argument list. `args: $...` takes as many arguments as the
// call gives it, of whatever types they are, and the specialization takes one
// ordinary parameter per element. A `for` over the list unrolls into one copy
// of its body per element, `list[K]` names the Kth, and an `if` over a type
// predicate keeps the branch that survives for that element and drops the
// other before anything checks it.
const COMPILE_TIME_LIST: &str = r#"import "io.frost"

printall :: fn(args: $...) {
    for value in args {
        print("{}\n", value)
    }
}

show :: fn(args: $...) {
    for value in args {
        if (is_float(value)) {
            print("{}\n", 1)
            print("{}\n", value)
        } else {
            print("{}\n", 0)
            print("{}\n", value)
        }
    }
}

first :: fn(args: $...) -> i64 {
    unsafe { args[0] }
}

count :: fn(label: i64, args: $...) -> i64 {
    var total := label
    for value in args { total = total + 1 }
    total
}

main :: fn() -> i64 {
    printall(1, 2, 3)
    printall(7)
    show(4, 2.5, 9)
    print("{}\n", first(11, 22))
    print("{}\n", count(100, 1, 2, 3, 4))
    print("{}\n", count(100))
    0
}
"#;

#[test]
fn a_compile_time_list_unrolls_indexes_and_prunes() {
    let Some(output) = compile_and_run_unaudited("packlist", COMPILE_TIME_LIST)
    else {
        return;
    };
    assert_eq!(output, "1\n2\n3\n7\n0\n4\n1\n2.5\n0\n9\n11\n104\n100\n");
}

#[test]
fn self_hosted_unrolls_a_compile_time_list() {
    let Some(output) =
        selfhosted_unaudited_output("shpacklist", COMPILE_TIME_LIST)
    else {
        return;
    };
    assert_eq!(output, "1\n2\n3\n7\n0\n4\n1\n2.5\n0\n9\n11\n104\n100\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shpacklist_input.frost");
    std::fs::write(&input, COMPILE_TIME_LIST).unwrap();
    let Some(c_source) = self_hosted_emits("shpacklist", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shpacklist", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
}

// Two calls giving different types are two specializations, and each element is
// evaluated once however many times the unrolled body names it.
const LIST_SPECIALIZES: &str = r#"import "io.frost"

bump :: fn(mut n: i64) -> i64 {
    n = n + 1
    n
}

twice :: fn(args: $...) {
    for value in args {
        if (is_float(value)) {
            print("{}\n", value)
            print("{}\n", value)
        } else {
            print("{}\n", value)
            print("{}\n", value)
        }
    }
}

main :: fn() -> i64 {
    var counter : i64 = 0
    twice(bump(counter))
    print("{}\n", counter)
    twice(1.5)
    twice(3, 4)
    0
}
"#;

#[test]
fn a_list_element_is_evaluated_once() {
    let Some(output) = compile_and_run_unaudited("packonce", LIST_SPECIALIZES)
    else {
        return;
    };
    assert_eq!(output, "1\n1\n1\n1.5\n1.5\n3\n3\n4\n4\n");
}

#[test]
fn self_hosted_evaluates_a_list_element_once() {
    let Some(output) =
        selfhosted_unaudited_output("shpackonce", LIST_SPECIALIZES)
    else {
        return;
    };
    assert_eq!(output, "1\n1\n1\n1.5\n1.5\n3\n3\n4\n4\n");
}

// An empty list is a list: the `for` over it keeps nothing, and the call gives
// the parameters before it and stops.
const EMPTY_LIST: &str = r#"import "io.frost"

tally :: fn(base: i64, args: $...) -> i64 {
    var total := base
    for value in args { total = total + value }
    total
}

main :: fn() -> i64 {
    print("{}\n", tally(10))
    print("{}\n", tally(10, 1, 2))
    0
}
"#;

#[test]
fn an_empty_compile_time_list_keeps_nothing() {
    let Some(output) = compile_and_run_unaudited("packempty", EMPTY_LIST)
    else {
        return;
    };
    assert_eq!(output, "10\n13\n");
}

#[test]
fn self_hosted_keeps_nothing_for_an_empty_list() {
    let Some(output) = selfhosted_unaudited_output("shpackempty", EMPTY_LIST)
    else {
        return;
    };
    assert_eq!(output, "10\n13\n");
}

// Which element `list[K]` is has to be known where it is written, so an index
// that is not a literal is refused rather than read at run time.
#[test]
fn a_list_indexed_by_a_variable_is_refused() {
    let source = "pick :: fn(args: $...) -> i64 {\n\
                  \x20   var at : i64 = 0\n\
                  \x20   unsafe { args[at] }\n}\n\
                  main :: fn() -> i64 { pick(1, 2) }\n";
    let message = compile_error("packindexvar", source);
    assert!(
        message.contains("literal"),
        "expected the message to say the index has to be a literal, got: {message}"
    );
}

// A capability bundle: a generic struct whose fields are functions, a constant
// of it, and a generic that takes that constant as a compile-time argument. The
// bundle says what can be done with a type, the constant says how it is done for
// one type, and the specialization calls the functions it names directly. This
// is what stands in for a trait: an ordinary value with an ordinary type,
// chosen at the call rather than resolved by a search.
const CAPABILITY_BUNDLE: &str = r#"import "io.frost"

Ordering :: struct($T: Type) {
    less: fn(T, T) -> bool,
}

i64_less :: fn(a: i64, b: i64) -> bool { a < b }
i64_greater :: fn(a: i64, b: i64) -> bool { a > b }

ascending :: Ordering<i64> { less = i64_less }
descending :: Ordering<i64> { less = i64_greater }

smaller :: fn($T: Type, $ops: Ordering<T>, a: $T, b: $T) -> $T {
    if (ops.less(a, b)) { return a }
    b
}

chosen :: fn(ops: Ordering<i64>, a: i64, b: i64) -> i64 {
    if (ops.less(a, b)) { return a }
    b
}

main :: fn() -> i64 {
    print("{}\n", smaller($i64, $ascending, 7, 3))
    print("{}\n", smaller($i64, $descending, 7, 3))
    print("{}\n", ascending.less(1, 2))
    print("{}\n", chosen(ascending, 2, 9))
    print("{}\n", chosen(descending, 2, 9))
    held := descending
    print("{}\n", chosen(held, 4, 5))
    0
}
"#;

#[test]
fn a_capability_bundle_is_a_constant_of_function_fields() {
    let Some(output) = compile_and_run_unaudited("bundle", CAPABILITY_BUNDLE)
    else {
        return;
    };
    assert_eq!(output, "3\n7\n1\n2\n9\n5\n");
}

// The compile-time form leaves no function pointer behind: the specialization
// calls the function the bundle's field names, so there is nothing to load and
// nothing to dispatch on.
#[test]
fn a_compile_time_bundle_folds_to_a_direct_call() {
    let source = "Ordering :: struct($T: Type) { less: fn(T, T) -> bool }\n\
                  i64_less :: fn(a: i64, b: i64) -> bool { a < b }\n\
                  ascending :: Ordering<i64> { less = i64_less }\n\
                  smaller :: fn($T: Type, $ops: Ordering<T>, a: $T, b: $T) -> $T {\n\
                  \x20   if (ops.less(a, b)) { return a }\n    b\n}\n\
                  main :: fn() -> i64 { smaller($i64, $ascending, 7, 3) }\n";
    let Some(c_source) = emit_c_source("bundledirect", source) else {
        return;
    };
    assert!(
        !c_source.contains("(*)("),
        "expected no call through a pointer:\n{c_source}"
    );
    assert!(
        c_source.contains("frost_u_i64_less("),
        "expected a direct call to the function the field names:\n{c_source}"
    );
}

// A bundle a generic is not given at all is an error the call site names, since
// the argument is what says which functions the body ends up calling.
#[test]
fn a_bundle_argument_of_the_wrong_type_is_refused() {
    let source = "Ordering :: struct($T: Type) { less: fn(T, T) -> bool }\n\
                  Pair :: struct($T: Type) { first: T, second: T }\n\
                  pair :: Pair<i64> { first = 1, second = 2 }\n\
                  smaller :: fn($T: Type, $ops: Ordering<T>, a: $T, b: $T) -> $T {\n\
                  \x20   if (ops.less(a, b)) { return a }\n    b\n}\n\
                  main :: fn() -> i64 { smaller($i64, $pair, 7, 3) }\n";
    let message = compile_error("bundlewrong", source);
    assert!(
        message.contains("Pair<i64>") && message.contains("Ordering<i64>"),
        "expected the message to name both types, got: {message}"
    );
}

#[test]
fn self_hosted_takes_a_capability_bundle() {
    let Some(output) =
        selfhosted_unaudited_output("shbundle", CAPABILITY_BUNDLE)
    else {
        return;
    };
    assert_eq!(output, "3\n7\n1\n2\n9\n5\n");

    let directory = std::env::temp_dir();
    let input = directory.join("frost_shbundle_input.frost");
    std::fs::write(&input, CAPABILITY_BUNDLE).unwrap();
    let Some(c_source) = self_hosted_emits("shbundle", &input, None) else {
        return;
    };
    let _ = std::fs::remove_file(&input);
    let Some(via_c) = compile_c_and_run("shbundle", &c_source) else {
        return;
    };
    assert_eq!(via_c, output, "the self-hosted C backend disagrees");
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
    var result := x
    if (before(y, result)) { result = y }
    if (before(z, result)) { result = z }
    result
}

main :: fn() -> i64 {
    unsafe { printf("%lld\n", best3($i64, $ascending, 7, 3, 9)) }
    unsafe { printf("%lld\n", best3($i64, $descending, 7, 3, 9)) }
    0
}
"#;
    let Some(output) = compile_and_run_unaudited("constfn", source) else {
        return;
    };
    assert_eq!(output, "3\n9\n");
}

// One specialization per function given, and the call inside it is direct.
#[test]
fn a_compile_time_function_argument_specializes_and_calls_directly() {
    let source = "cmp :: fn(a: i64, b: i64) -> bool { a < b }\n\
                  pick :: fn($T: Type, $f: Type, move a: $T, move b: $T) -> $T {\n\
                  \x20   var best := a\n    if (f(b, best)) { best = b }\n    best\n}\n\
                  main :: fn() -> i64 { pick($i64, $cmp, 2, 1) }\n";
    let Some(c_source) = emit_c_source("constfndirect", source) else {
        return;
    };
    assert!(
        c_source.contains("pick__i64__cmp"),
        "expected a specialization named for the function:\n{c_source}"
    );
    assert!(
        c_source.contains("= frost_u_cmp("),
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
    var x : i64 = 1
    var y : i64 = 2
    swap_scalar(x, y)
    unsafe { printf("%lld\n", x) }

    var m : i64 = 3
    var n : i64 = 4
    swap_generic(m, n)
    unsafe { printf("%lld\n", m) }

    var a := Point { x = 1, y = 2 }
    var b := Point { x = 9, y = 8 }
    swap_generic(a, b)
    unsafe { printf("%lld\n", a.x) }

    var c := Point { x = 5, y = 6 }
    replace(c, Point { x = 7, y = 0 })
    unsafe { printf("%lld\n", c.x) }
    0
}
"#;
    let Some(output) = compile_and_run_unaudited("mutwriteback", source) else {
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
    unsafe { printf("%lld\n", p.first + p.second) }
    m : i64 = 10
    n : i64 = 11
    q := make_pair(m, n)
    unsafe { printf("%lld\n", q.first + q.second) }
    0
}
"#;
    let Some(output) = compile_and_run_unaudited("readmodevalue", source)
    else {
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
    var result := x
    if (before(y, result)) { result = y }
    result
}

main :: fn() -> i64 {
    unsafe { printf("%lld\n", best($i64, $ascending, 7, 3)) }
    0
}
"#;
    let Some(output) = compile_and_run_unaudited("constfnbound", source) else {
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
    // Spelled the way a reader writes a function type. This used to pin
    // `proc(..)`, which is the name the type table files one under and not
    // syntax the surface has.
    assert!(
        message.contains("'wrong'") && message.contains("fn(i64, i64) -> bool"),
        "expected the signature mismatch to name both signatures:\n{message}"
    );
    assert!(
        !message.contains("proc("),
        "the signature mismatch named a function type as `proc`:\n{message}"
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
             \x20   var local : i64 = 5\n    slot = ptr_to(local)\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "branch",
            "pick :: fn(c: bool) -> ^i64 {\n\
             \x20   var a : i64 = 1\n\
             \x20   if (c) { ptr_to(a) } else { ptr_to(a) }\n}\n\
             main :: fn() -> i64 { 0 }\n",
        ),
        (
            "struct",
            "Holder :: struct { p: ^i64 }\n\
             wrap :: fn() -> Holder {\n\
             \x20   var a : i64 = 1\n    Holder { p = ptr_to(a) }\n}\n\
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

// A module is rebuilt only when its own source or an imported interface
// changes, and the distinction that decides it
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
         import \"lib/leaf.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   b := boxed(1)\n\
         \x20   unsafe { printf(\"%lld\n\", combine(5) + b.value + twice($i64, 2)) }\n\
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

// A destructure of a call into an imported module, on the build that reads
// that module back from the cache.
//
// The unsafe gate reads what each function answers with off the top level, and
// a cached module arrives as a declaration built from its interface rather than
// as the constant its source spells. The list of values was read off the
// constant alone, so on the second build every name in the destructure was
// bound to no type, and the index rule met a base it could not name and refused
// `view[0]`. The first build took the same program, which is what makes this
// worth a test of its own: a suite that compiles each program once cannot see
// it.
#[test]
fn a_destructure_of_a_cached_module_is_taken_on_every_build() {
    if !linker_available() {
        return;
    }
    let directory = std::env::temp_dir().join(unique("frost_multi_cached"));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(
        directory.join("split.frost"),
        "export split\n\
         split :: fn(source: []i64) -> (view: []i64, count: i64) {\n\
         \x20   return { view = source, count = 4 }\n\
         }\n",
    )
    .unwrap();
    let root = directory.join("multi_cached.frost");
    std::fs::write(
        &root,
        "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         import \"split.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   var data : [4]i64 = [10, 20, 30, 40]\n\
         \x20   view, count := split(data)\n\
         \x20   unsafe { printf(\"%lld\n\", view[0] + count) }\n\
         \x20   0\n\
         }\n",
    )
    .unwrap();

    let exe = directory.join(format!("app{}", std::env::consts::EXE_SUFFIX));
    let build = directory.join("build");
    for pass in 1..=3 {
        let built = Command::new(env!("CARGO_BIN_EXE_frost"))
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
            "build {pass} refused the program:\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let ran = Command::new(&exe).output().unwrap();
        assert_eq!(
            String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n"),
            "14\n",
            "build {pass} produced the wrong program"
        );
    }
    let _ = std::fs::remove_dir_all(&directory);
}

// An address read as something other than a whole number.
//
// A pointer is an address and an address is a number, so a pointer and an
// integer reach each other: that is what a call into C hands over and what
// address arithmetic reads back, and both compilers take it. A float holds a
// different encoding of the same bits and a `bool` holds one of two values, so
// reading an address as either is a mistake. The bootstrap caught both in the
// typed IR. The self-hosted compiler's `types_compatible` ran out of rules and
// answered yes, so `f : f64 = p` and `b : bool = p` compiled there and were
// refused here.
//
// The wording stays each compiler's own, since the bootstrap names the IR local
// and the self-hosted one names the binding. The refusal is what both answer
// for, and the two shapes that stay legal are checked beside them so this
// cannot be satisfied by refusing every pointer conversion.
#[test]
fn both_compilers_refuse_an_address_read_as_a_float_or_a_bool() {
    const HELD: &str = "import \"io.frost\"
         main :: fn() -> i64 {
             var data : [4]i64 = [10, 20, 30, 40]
             p := unsafe { ptr_cast($i64, ptr_to(data[0])) }
             ";
    for (name, tail) in [
        (
            "ptrfloat",
            "f : f64 = p
             print(\"{}\\n\", 1)",
        ),
        (
            "ptrbool",
            "b : bool = p
             if (b) { print(\"{}\\n\", 1) }",
        ),
    ] {
        let source = format!(
            "{HELD}{tail}
             0
         }}
"
        );
        let bootstrap = compile_error(name, &source);
        assert!(
            bootstrap.contains("^i64"),
            "the bootstrap took an address read as a {name}:
{bootstrap}"
        );
        let Some(hosted) = self_hosted_rejects(name, &source) else {
            return;
        };
        assert!(
            hosted.contains("^i64"),
            "the self-hosted compiler refused {name} without naming the              pointer:
{hosted}"
        );
    }

    // The integer directions stay legal, both ways, in both compilers.
    let legal = "import \"io.frost\"
         main :: fn() -> i64 {
             var data : [4]i64 = [10, 20, 30, 40]
             p := unsafe { ptr_cast($i64, ptr_to(data[0])) }
             q : i64 = p
             back : ^i64 = q
             print(\"{}\\n\", unsafe { back[0] })
             0
         }
";
    let Some(output) = bootstrap_output("ptrint", legal) else {
        return;
    };
    assert_eq!(
        output,
        "10
"
    );
}

// A callback registration is an `extern fn` with a `$handler` parameter bound
// to a function signature, and the whole ownership
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
         \x20   t := unsafe { register($on_event, c) }\n\
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

// The whole safety argument for a callback. A registration holds its context
// for as long as it is registered, so the value it answers
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
         \x20   unsafe { register($on_event, c) }\n\
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
         \x20   r := unsafe { register($on_event, c) }\n\
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

// Bind a real C callback API and register against it, because every other
// check here works without a library and none of them proves the ABI. The library here is the smallest one that is still the
// real shape, `(callback, userdata)` stored and called back later, compiled by
// the C compiler and linked in.
//
// What this settles is that no trampoline is needed at all. A `mut`
// parameter is already a pointer in the signature and Frost and C share a
// calling convention, so the handler compiled for Frost *is* the
// `void (*)(void*, int64_t)` the library wants, and there is no trampoline and
// no cast anywhere. If that were wrong, this is where it would crash.
//
// The context goes in by `move` and comes back out through unregistration,
// which is an ordinary extern returning a struct by value. That needed no
// callback machinery at all, only the C return classification.
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
         \x20   r := unsafe { Registration { token = register_handler($on_event, c) } }\n\
         \x20   unsafe { pump(4) }\n\
         \x20   unsafe { pump(5) }\n\
         \x20   unsafe { printf(\"%lld\n\", peek()) }\n\
         \x20   unsafe { printf(\"%lld\n\", unregister(r)) }\n\
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
    // getting it back is what unregistration is for.
    assert_eq!(output, "9\n77\n");
}

// Passing a struct to C by value had no spelling, and `value` is it. Every shape here lands on a different side of
// some rule: 16 bytes is what wgpu's WGPUStringView is and is two eightbytes on
// System V, 8 bytes is one integer register on Windows, 4 bytes of float is
// where Windows and System V disagree, and 12 bytes is not a power of two so
// Windows takes the address of a copy.
//
// The library is compiled by the C compiler from a header written by hand, so
// what this checks is Frost against the real ABI rather than against its own
// idea of it. Every call is by value, so a callee that writes to its parameter
// must not be writing to the caller's, which the last line checks.
#[test]
fn a_struct_is_passed_to_c_by_value() {
    let Some(compiler) = c_compiler() else {
        return;
    };
    let directory = std::env::temp_dir().join("frost_byvalue");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();

    let library_source = directory.join("shapes.c");
    std::fs::write(
        &library_source,
        "#include <stdint.h>\n\
         typedef struct { const char* data; int64_t len; } View;\n\
         typedef struct { int32_t x; int32_t y; } Pair;\n\
         typedef struct { float a; } Single;\n\
         typedef struct { int32_t a; int32_t b; int32_t c; } Triple;\n\
         typedef struct { int64_t a, b, c, d; } Wide;\n\
         typedef struct { double a, b, c; } Wider;\n\
         int64_t view_len(View v) { return v.len; }\n\
         int64_t view_first(View v) { return (int64_t)(unsigned char)v.data[0]; }\n\
         int64_t pair_sum(Pair p) { return p.x + p.y; }\n\
         int64_t single_ten(Single s) { return (int64_t)(s.a * 10.0f); }\n\
         int64_t triple_sum(Triple t) { return t.a + t.b + t.c; }\n\
         int64_t clobber(Triple t) { t.a = 999; return t.a; }\n\
         int64_t wide_sum(Wide w) { return w.a + w.b + w.c + w.d; }\n\
         int64_t wider_sum(Wider w) { return (int64_t)(w.a + w.b + w.c); }\n\
         int64_t wide_after(int64_t before, Wide w, int64_t after) {\n\
         \x20   return before * 1000 + w.a + w.d + after;\n\
         }\n\
         int64_t mixed(int64_t before, Pair p, int64_t after) {\n\
         \x20   return before * 100 + p.x * 10 + p.y + after;\n\
         }\n",
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

    let source = "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         View :: struct { data: ^i8, len: i64 }\n\
         Pair :: struct { x: i32, y: i32 }\n\
         Single :: struct { a: f32 }\n\
         Triple :: struct { a: i32, b: i32, c: i32 }\n\
         Wide :: struct { a: i64, b: i64, c: i64, d: i64 }\n\
         Wider :: struct { a: f64, b: f64, c: f64 }\n\
         wide_sum :: extern fn(value w: Wide) -> i64\n\
         wider_sum :: extern fn(value w: Wider) -> i64\n\
         wide_after :: extern fn(before: i64, value w: Wide, after: i64) -> i64\n\
         view_len :: extern fn(value v: View) -> i64\n\
         view_first :: extern fn(value v: View) -> i64\n\
         pair_sum :: extern fn(value p: Pair) -> i64\n\
         single_ten :: extern fn(value s: Single) -> i64\n\
         triple_sum :: extern fn(value t: Triple) -> i64\n\
         clobber :: extern fn(value t: Triple) -> i64\n\
         mixed :: extern fn(before: i64, value p: Pair, after: i64) -> i64\n\
         main :: fn() -> i64 {\n\
         \x20   v := View { data = \"hello\", len = 5 }\n\
         \x20   unsafe { printf(\"%lld\n\", view_len(v)) }\n\
         \x20   unsafe { printf(\"%lld\n\", view_first(v)) }\n\
         \x20   p := Pair { x = 3, y = 4 }\n\
         \x20   unsafe { printf(\"%lld\n\", pair_sum(p)) }\n\
         \x20   unsafe { printf(\"%lld\n\", single_ten(Single { a = 2.5 })) }\n\
         \x20   t := Triple { a = 1, b = 2, c = 3 }\n\
         \x20   unsafe { printf(\"%lld\n\", triple_sum(t)) }\n\
         \x20   unsafe { printf(\"%lld\n\", clobber(t)) }\n\
         \x20   unsafe { printf(\"%lld\n\", triple_sum(t)) }\n\
         \x20   unsafe { printf(\"%lld\n\", mixed(7, p, 9)) }\n\
         \x20   w := Wide { a = 1, b = 2, c = 3, d = 4 }\n\
         \x20   unsafe { printf(\"%lld\n\", wide_sum(w)) }\n\
         \x20   unsafe { printf(\"%lld\n\", wider_sum(Wider { a = 1.5, b = 2.5, c = 3.0 })) }\n\
         \x20   unsafe { printf(\"%lld\n\", wide_after(5, w, 6)) }\n\
         \x20   0\n\
         }\n";
    let root = directory.join("shapes.frost");
    std::fs::write(&root, source).unwrap();

    // 5, 'h', 3+4, 2.5*10, 1+2+3, the callee's own copy, the caller's value
    // still 6 because the copy was the callee's, then 7*100 + 3*10 + 4 + 9.
    // Then the two structs too large for registers, which System V pushes onto
    // the stack rather than passing as an address, and one of those with an
    // ordinary argument on either side of it: 5*1000 + 1 + 4 + 6.
    let expected = "5\n104\n7\n25\n6\n999\n6\n743\n10\n7\n5011\n";

    for emit_c in [false, true] {
        let exe = directory.join(format!(
            "shapes{}{}",
            u8::from(emit_c),
            std::env::consts::EXE_SUFFIX
        ));
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
            "the by-value program did not build (emit_c={emit_c}):\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let ran = Command::new(&exe).output().unwrap();
        let output = String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n");
        assert_eq!(output, expected, "backend disagrees (emit_c={emit_c})");
    }
    let _ = std::fs::remove_dir_all(&directory);
}

// The other direction: C calls a Frost function and passes a struct by value.
// A callback's signature is a function-pointer type, and until it could say
// `value` the struct had to be declared as one pointer. That is what Windows
// hands a sixteen-byte struct to a callee as, so it worked there and was wrong
// on System V, where the struct takes two registers and every argument after it
// comes out of the wrong one. wgpu's callbacks are exactly this shape.
//
// There is still no trampoline. The Frost function is compiled to receive what
// C sends, which is the same claim as for the simple case, extended to the one
// shape that needed the compiler to know the rule.
#[test]
fn c_calls_back_with_a_struct_by_value() {
    let Some(compiler) = c_compiler() else {
        return;
    };
    let directory = std::env::temp_dir().join("frost_callback_byvalue");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();

    let library_source = directory.join("shapes.c");
    std::fs::write(
        &library_source,
        "#include <stdint.h>\n\
         typedef struct { const char* data; int64_t len; } View;\n\
         typedef struct { int64_t a, b, c, d; } Wide;\n\
         static void (*held)(int32_t, View, int64_t);\n\
         static void (*held_wide)(int32_t, Wide, int64_t);\n\
         void install(void (*f)(int32_t, View, int64_t)) { held = f; }\n\
         void install_wide(void (*f)(int32_t, Wide, int64_t)) {\n\
         \x20   held_wide = f;\n\
         }\n\
         void fire(void) {\n\
         \x20   View v; v.data = \"hello\"; v.len = 5;\n\
         \x20   held(7, v, 99);\n\
         }\n\
         void fire_wide(void) {\n\
         \x20   Wide w; w.a = 1; w.b = 2; w.c = 3; w.d = 4;\n\
         \x20   held_wide(8, w, 77);\n\
         }\n",
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

    let root = directory.join("callback.frost");
    std::fs::write(
        &root,
        "import \"io.frost\"\nView :: struct { data: ^i8, len: i64 }\n\
         Wide :: struct { a: i64, b: i64, c: i64, d: i64 }\n\
         install :: extern fn(f: fn(i32, value View, i64))\n\
         install_wide :: extern fn(f: fn(i32, value Wide, i64))\n\
         fire :: extern fn()\n\
         fire_wide :: extern fn()\n\
         handler :: fn(status: i32, value message: View, tail: i64) {\n\
         \x20   print(\"{}\\n\", status)\n\
         \x20   print(\"{}\\n\", message.len)\n\
         \x20   unsafe { print(\"{}\\n\", message.data[0]) }\n\
         \x20   print(\"{}\\n\", tail)\n\
         }\n\
         wide_handler :: fn(status: i32, value w: Wide, tail: i64) {\n\
         \x20   print(\"{}\\n\", status)\n\
         \x20   print(\"{}\\n\", w.a + w.d)\n\
         \x20   print(\"{}\\n\", tail)\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   unsafe { install(handler) }\n\
         \x20   unsafe { fire() }\n\
         \x20   unsafe { install_wide(wide_handler) }\n\
         \x20   unsafe { fire_wide() }\n\
         \x20   0\n\
         }\n",
    )
    .unwrap();

    // 7 and 99 are the arguments either side of the struct, so reading them
    // back is what says the struct took the register count it should have.
    let expected = "7\n5\n104\n99\n8\n5\n77\n";

    for emit_c in [false, true] {
        let exe = directory.join(format!(
            "callback{}{}",
            u8::from(emit_c),
            std::env::consts::EXE_SUFFIX
        ));
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
            "the callback program did not build (emit_c={emit_c}):\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let ran = Command::new(&exe).output().unwrap();
        let output = String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n");
        assert_eq!(output, expected, "backend disagrees (emit_c={emit_c})");
    }
    let _ = std::fs::remove_dir_all(&directory);
}

// The same thing with the userdata last, which is the order wgpu-native and
// most modern C APIs take. Nothing about the ABI changes: the handler compiled
// for Frost is still exactly the function pointer the library holds, only the
// pointer sits in a different argument slot. This shape used to be
// undeclarable, so the alternative was a function pointer in a struct field,
// where none of the callback checks apply at all.
#[test]
fn a_callback_whose_context_comes_last_runs() {
    let Some(compiler) = c_compiler() else {
        return;
    };
    let directory = std::env::temp_dir().join("frost_callback_last");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();

    let library_source = directory.join("late.c");
    std::fs::write(
        &library_source,
        "#include <stdint.h>\n\
         static void (*held)(int32_t, int64_t, void*);\n\
         static void* held_context;\n\
         int64_t request(void (*handler)(int32_t, int64_t, void*), void* context) {\n\
         \x20   held = handler;\n\
         \x20   held_context = context;\n\
         \x20   return 31;\n\
         }\n\
         void deliver(int32_t status, int64_t code) {\n\
         \x20   held(status, code, held_context);\n\
         }\n\
         int64_t peek(void) { return *(int64_t*)held_context; }\n",
    )
    .unwrap();
    let library = directory.join("late.o");
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

    let root = directory.join("late.frost");
    std::fs::write(
        &root,
        "printf :: extern fn(fmt: ^i8, value: i64) -> i32\n\
         deliver :: extern fn(status: i32, code: i64)\n\
         peek :: extern fn() -> i64\n\
         Ctx :: struct { hits: i64 }\n\
         Registration :: linear struct { token: i64 }\n\
         on_ready :: fn(status: i32, code: i64, mut ctx: Ctx) {\n\
         \x20   ctx.hits = ctx.hits + code\n\
         }\n\
         request :: extern fn($handler: fn(i32, i64, mut Ctx), move ctx: Ctx) -> i64\n\
         unregister :: fn(move r: Registration) -> i64 { r.token }\n\
         main :: fn() -> i64 {\n\
         \x20   c := Ctx { hits = 0 }\n\
         \x20   r := unsafe { Registration { token = request($on_ready, c) } }\n\
         \x20   unsafe { deliver(1, 6) }\n\
         \x20   unsafe { deliver(1, 7) }\n\
         \x20   unsafe { printf(\"%lld\n\", peek()) }\n\
         \x20   unsafe { printf(\"%lld\n\", unregister(r)) }\n\
         \x20   0\n\
         }\n",
    )
    .unwrap();

    let exe = directory.join(format!("late{}", std::env::consts::EXE_SUFFIX));
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
        "the context-last callback did not build:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );

    let ran = Command::new(&exe).output().unwrap();
    let output = String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n");
    let _ = std::fs::remove_dir_all(&directory);
    assert_eq!(output, "13\n31\n");
}

// C returns a struct by a rule of its own, and the rule differs by target and, on some targets, by whether the fields are
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
         show :: fn(v: i64) { unsafe { printf(\"%lld\n\", v) } }\n\
         showd :: fn(v: f64) { unsafe { printf_d(\"%.4f\n\", v) } }\n\
         main :: fn() -> i64 {\n\
         \x20   v1 := unsafe { m1() }   a1 : i64 = v1.a  show(a1)\n\
         \x20   v2 := unsafe { m2() }   a2 : i64 = v2.a  show(a2)\n\
         \x20   v3 := unsafe { m3() }   a3 : i64 = v3.a  b3 : i64 = v3.b  c3 : i64 = v3.c\n\
         \x20   show(a3 * 100 + b3 * 10 + c3)\n\
         \x20   v4 := unsafe { m4() }   a4 : i64 = v4.a  show(a4)\n\
         \x20   v8 := unsafe { m8() }   show(v8.a)\n\
         \x20   vf := unsafe { mf() }   af : f64 = vf.a  showd(af)\n\
         \x20   vff := unsafe { mff() } aff : f64 = vff.a  bff : f64 = vff.b  showd(aff)  showd(bff)\n\
         \x20   vm := unsafe { mmix() } am : i64 = vm.a  bm : f64 = vm.b  show(am)  showd(bm)\n\
         \x20   v16 := unsafe { m16() } show(v16.a)  show(v16.b)\n\
         \x20   vdd := unsafe { mdd() } showd(vdd.a)  showd(vdd.b)\n\
         \x20   v24 := unsafe { m24() } show(v24.a)  show(v24.b)  show(v24.c)\n\
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
         \x20   s := unsafe { mk() }\n\
         \x20   match s {\n\
         \x20       case .Empty: unsafe { printf(\"%lld\n\", 0) }\n\
         \x20       case .Full { v }: unsafe { printf(\"%lld\n\", v) }\n\
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

    // The line to change, which is the one in the template. The instance it
    // went wrong for is named in the claim, and a generic is one function to
    // look at, so the call that asked for it is a search away while the line
    // holding the fault is not.
    assert!(
        message.contains("lib/g.frost:2:"),
        "the error did not name the line at fault:
{message}"
    );
    // Named the way it was written, not as `add__Point`, and in the words the
    // fault itself uses. The instance is not announced by a phrase of the
    // compiler's in front of the claim: for a fault that carried its own place
    // that left two places in one report, so the sentence a reader had to act
    // on began with a file name.
    assert!(
        message.contains("'add<Point>'"),
        "the error did not name the instantiation:\n{message}"
    );
    assert!(
        !message.contains("add__Point"),
        "a mangled name reached the reader:\n{message}"
    );
    // And one place, which is the call. The template's own position used to
    // ride along inside the words, so the sentence a reader had to act on began
    // `at lib/g.frost:2:41:` and the renderer showed the call above it: two
    // places in one report, one of them a file name in the middle of a
    // sentence. The generic is named in the claim, which is the hop to its
    // body; where a fault has a second line worth pointing at it says so as a
    // placed line of its own, the way "was moved here" does.
    assert!(
        !message.contains("generic_diagnostic_app.frost:5:"),
        "the claim carries a second place:\n{message}"
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
// with "'Pair' is not a type this program declares": the argument was lowered
// with no expected type, so the literal had nothing to tell it which instance
// it was and fell back to
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
    unsafe { printf("%lld\n", sum($Pair<i64>, Pair { first = 3, second = 4 })) }

    var pool : Slab<Pair<i64>, 4> = slab_new()
    h := insert($Pair<i64>, $4, pool, Pair { first = 10, second = 20 })
    unsafe { printf("%lld\n", pool.storage[h].first + pool.storage[h].second) }
    0
}
"#;
    let Some(output) = compile_and_run_unaudited("generic_literal_arg", source)
    else {
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
Option :: enum($T: Type) { None, Some { value: T } }
Either :: enum($L: Type, $R: Type) { Left { value: L }, Right { value: R } }

unwrap_or :: fn($T: Type, m: Option<T>, fallback: $T) -> $T {
    match m {
        case .None: fallback
        case .Some { value }: value
    }
}

main :: fn() -> i64 {
    a : Option<i64> = Option::Some { value = 42 }
    b : Option<i64> = Option::None
    unsafe { printf("%lld\n", unwrap_or($i64, a, 0)) }
    unsafe { printf("%lld\n", unwrap_or($i64, b, 7)) }

    p : Option<Point> = Option::Some { value = Point { x = 3, y = 4 } }
    match p {
        case .None: unsafe { printf("%lld\n", 0) }
        case .Some { value }: unsafe { printf("%lld\n", value.x + value.y) }
    }

    e : Either<i64, Point> = Either::Right { value = Point { x = 5, y = 6 } }
    match e {
        case .Left { value }: unsafe { printf("%lld\n", value) }
        case .Right { value }: unsafe { printf("%lld\n", value.y) }
    }

    nested : Option<Option<i64>> = Option::Some { value = Option::Some { value = 8 } }
    match nested {
        case .None: unsafe { printf("%lld\n", 0) }
        case .Some { value }: match value {
            case .None: unsafe { printf("%lld\n", 0) }
            case .Some { value }: unsafe { printf("%lld\n", value) }
        }
    }
    0
}
"#;
    let Some(output) = compile_and_run_unaudited("generic_enum", source) else {
        return;
    };
    assert_eq!(output, "42\n7\n7\n6\n8\n");
}

// The same five, through the other compiler. A program one compiler builds and
// the other cannot find the pieces of is two languages, and the manifest was
// exactly that until this was written: the bootstrap read `frost.json` and the
// self-hosted compiler had never heard of it, so a project that declared where
// its libraries live compiled with one and not the other.
#[test]
fn the_self_hosted_compiler_finds_the_same_search_roots() {
    if !linker_available() {
        return;
    }
    let Some(compiler) = build_self_hosted_compiler("shroots") else {
        return;
    };
    let directory = std::env::temp_dir().join("frost_self_search_roots");
    let elsewhere = directory.join("elsewhere");
    let declared = directory.join("declared");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::fs::create_dir_all(&declared).unwrap();

    let library = "export twice
twice :: fn(x: i64) -> i64 { x * 2 }
";
    std::fs::write(elsewhere.join("helper.frost"), library).unwrap();
    std::fs::write(declared.join("helper.frost"), library).unwrap();
    std::fs::write(directory.join("beside.frost"), library).unwrap();

    let program = |import: &str| {
        format!(
            "printf :: extern fn(fmt: ^i8, value: i64) -> i32
             import \"{import}\"
             main :: fn() -> i64 {{ unsafe {{ printf(\"%lld
\", twice(21)) }}  0 }}
"
        )
    };

    let build =
        |name: &str, source: &str, args: &[&str], env: &[(&str, &str)]| {
            let root = directory.join(format!("{name}.frost"));
            std::fs::write(&root, source).unwrap();
            let exe = directory
                .join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
            let mut command = Command::new(&compiler);
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
                "the self-hosted compiler could not build {name}:
{}",
                String::from_utf8_lossy(&built.stderr)
            );
            let ran = Command::new(&exe).output().unwrap();
            normalize_newlines(&ran.stdout)
        };

    assert_eq!(
        build("neighbour", &program("beside.frost"), &[], &[]),
        "42
"
    );
    assert_eq!(
        build(
            "flagged",
            &program("helper.frost"),
            &["-L", elsewhere.to_str().unwrap()],
            &[]
        ),
        "42
"
    );
    assert_eq!(
        build(
            "environment",
            &program("helper.frost"),
            &[],
            &[("FROST_PATH", elsewhere.to_str().unwrap())]
        ),
        "42
"
    );
    std::fs::write(
        directory.join("frost.json"),
        r#"{ "name": "demo", "paths": ["declared"] }"#,
    )
    .unwrap();
    assert_eq!(
        build("manifest", &program("helper.frost"), &[], &[]),
        "42
"
    );
    std::fs::remove_file(directory.join("frost.json")).unwrap();

    let uses_std = "printf :: extern fn(fmt: ^i8, value: i64) -> i32
         import \"option.frost\"
         main :: fn() -> i64 {
             m := option_some($i64, 42)
             unsafe { printf(\"%lld
\", option_unwrap_or($i64, m, 0)) }
             0
         }
";
    assert_eq!(
        build("standard", uses_std, &[], &[]),
        "42
"
    );

    let _ = std::fs::remove_dir_all(&directory);
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
             main :: fn() -> i64 {{ unsafe {{ printf(\"%lld\n\", twice(21)) }}  0 }}\n"
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
         import \"option.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   m := option_some($i64, 42)\n\
         \x20   unsafe { printf(\"%lld\n\", option_unwrap_or($i64, m, 0)) }\n\
         \x20   0\n\
         }\n";
    assert_eq!(build("standard", uses_std, &[], &[]), "42\n");

    let _ = std::fs::remove_dir_all(&directory);
}

// The JSON reader parses into one flat array of nodes addressed by index, so
// these exercise a nested document rather than a scalar: object, array, member
// lookup by name, element by position, and a number read back out.
#[test]
fn the_standard_json_reader_walks_a_nested_document() {
    let source = "import \"io.frost\"\nimport \"json.frost\"\n\
                  main :: fn() -> i64 {\n\
                  \x20   text := \"{\\\"name\\\":\\\"color\\\",\\\"members\\\":[{\\\"n\\\":1},{\\\"n\\\":22}],\\\"ok\\\":true}\"\n\
                  \x20   var document := json_parse(text)\n\
                  \x20   root := json_root(document)\n\
                  \x20   if (json_kind(document, root) == JsonKind::Object) { print(\"{}\\n\", 6) } else { print(\"{}\\n\", 0) }\n\
                  \x20   if (json_text_eq(document, json_member(document, root, \"name\"), \"color\")) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
                  \x20   members := json_member(document, root, \"members\")\n\
                  \x20   print(\"{}\\n\", json_count(document, members))\n\
                  \x20   print(\"{}\\n\", json_number(document, json_member(document, json_at(document, members, 1), \"n\")))\n\
                  \x20   if (json_kind(document, json_member(document, root, \"ok\")) == JsonKind::True) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
                  \x20   json_free(document)\n\
                  \x20   0\n}\n";
    let Some(output) = compile_and_run_unaudited("stdjson", source) else {
        return;
    };
    assert_eq!(output, "6\n1\n2\n22\n1\n");
}

// A number is read twice over: `json_number` answers with the integer part and
// `json_real` with the whole of it. What is worth checking is that the second
// one keeps the fraction and the exponent the first one steps over, since a
// reader that dropped them would still answer plausibly on every round value.
#[test]
fn the_standard_json_reader_answers_with_whole_numbers() {
    let source = "import \"io.frost\"\nimport \"json.frost\"\n\
                  main :: fn() -> i64 {\n\
                  \x20   text := \"{\\\"a\\\":2.75,\\\"b\\\":-0.25,\\\"c\\\":1.5e2,\\\"d\\\":7}\"\n\
                  \x20   var document := json_parse(text)\n\
                  \x20   root := json_root(document)\n\
                  \x20   if (json_real(document, json_member(document, root, \"a\")) == 2.75) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
                  \x20   if (json_real(document, json_member(document, root, \"b\")) == -0.25) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
                  \x20   if (json_real(document, json_member(document, root, \"c\")) == 150.0) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
                  \x20   if (json_real(document, json_member(document, root, \"d\")) == 7.0) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
                  \x20   print(\"{}\\n\", json_number(document, json_member(document, root, \"a\")))\n\
                  \x20   if (json_real(document, json_member(document, root, \"missing\")) == 0.0) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
                  \x20   json_free(document)\n\
                  \x20   0\n}\n";
    let Some(output) = compile_and_run_unaudited("stdjsonreal", source) else {
        return;
    };
    assert_eq!(output, "1\n1\n1\n1\n2\n1\n");
}

// The optional type is an ordinary generic enum in the standard library. Both
// variants and every function it exports.
#[test]
fn the_standard_option_covers_both_variants() {
    let source = "import \"io.frost\"\nimport \"option.frost\"\n\
                  main :: fn() -> i64 {\n\
                  \x20   a := option_some($i64, 42)\n\
                  \x20   b := option_none($i64)\n\
                  \x20   print(\"{}\\n\", option_unwrap_or($i64, a, 0))\n\
                  \x20   print(\"{}\\n\", option_unwrap_or($i64, b, 7))\n\
                  \x20   if (option_is_some($i64, a)) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
                  \x20   if (option_is_some($i64, b)) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
                  \x20   0\n}\n";
    let Some(output) = compile_and_run_unaudited("stdoption", source) else {
        return;
    };
    assert_eq!(output, "42\n7\n1\n0\n");
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
    var value : i64 = 0
    var i : i64 = 0
    while (i < str_len(s)) {
        value = value * 10 + (s[i] - 48)
        i = i + 1
    }
    value
}

main :: fn() -> i64 {
    unsafe { printf("%lld\n", to_i64("1234567")) }

    text := "7"
    byte := text[0]
    var accumulator : i64 = 1234
    unsafe { printf("%lld\n", accumulator * 10 + (byte - 48)) }

    // A literal still takes the width of what it is combined with, which is
    // what the backwards rule was there to protect and which still holds. The
    // sum leaves eight bits, so it says which it means.
    var small : u8 = 250
    small = wrap_add(small, 10)
    wide : i64 = small
    unsafe { printf("%lld\n", wide) }
    0
}
"#;
    let Some(output) = compile_and_run_unaudited("mixed_widths", source) else {
        return;
    };
    // 1234567 read a byte at a time, 12347 from the same shape by hand, and
    // 250 + 10 wrapping at eight bits because both sides really are u8.
    assert_eq!(output, "1234567\n12347\n4\n");
}

// A `linear` container is a resource whatever it holds, and so is a struct
// holding one. Both were silently ordinary until the word was read off the
// template rather than off the instance's name, which is why the standard
// library's containers could leak without a word from the compiler.
#[test]
fn a_leaked_generic_linear_is_refused() {
    let source = "Box :: linear struct($T: Type) { value: T }\n\
                  make :: fn($T: Type, value: $T) -> Box<T> { Box { value = value } }\n\
                  take :: fn($T: Type, move b: Box<T>) -> i64 { 1 }\n\
                  main :: fn() -> i64 {\n\
                  \x20   held := make($i64, 5)\n\
                  \x20   0\n}\n";
    let message = compile_error("genericlinear", source);
    assert!(
        message.contains("not consumed"),
        "expected a linearity error, got:\n{message}"
    );
}

#[test]
fn a_consumed_generic_linear_is_accepted() {
    let source = "import \"io.frost\"\nBox :: linear struct($T: Type) { value: T }\n\
                  make :: fn($T: Type, value: $T) -> Box<T> { Box { value = value } }\n\
                  take :: fn($T: Type, move b: Box<T>) -> i64 { b.value }\n\
                  main :: fn() -> i64 {\n\
                  \x20   held := make($i64, 5)\n\
                  \x20   print(\"{}\\n\", take($i64, held))\n\
                  \x20   0\n}\n";
    let Some(output) = compile_and_run_unaudited("genericlinearok", source)
    else {
        return;
    };
    assert_eq!(output, "5\n");
}

#[test]
fn a_struct_holding_a_resource_is_one() {
    let source = "import \"io.frost\"\nResource :: linear struct { id: i64 }\n\
                  Holder :: struct { held: Resource }\n\
                  main :: fn() -> i64 {\n\
                  \x20   h := Holder { held = Resource { id = 1 } }\n\
                  \x20   print(\"{}\\n\", h.held.id)\n\
                  \x20   0\n}\n";
    let message = compile_error("linearheld", source);
    assert!(
        message.contains("not consumed"),
        "expected a linearity error, got:\n{message}"
    );
}

#[test]
fn the_self_hosted_compiler_refuses_a_leaked_generic_linear() {
    let source = "Box :: linear struct($T: Type) { value: T }\n\
                  make :: fn($T: Type, value: $T) -> Box<T> { Box { value = value } }\n\
                  main :: fn() -> i64 {\n\
                  \x20   held := make($i64, 5)\n\
                  \x20   0\n}\n";
    let Some(message) = self_hosted_rejects("shlinear", source) else {
        return;
    };
    assert!(
        message.contains("not consumed"),
        "expected a linearity error, got:\n{message}"
    );
}

// A run of resources is a resource. The bootstrap read a fixed array of a
// linear type that way and the self-hosted compiler did not, so a struct
// holding one was an obligation in one compiler and ordinary data in the other:
// the leak below was refused by one and compiled by the other. A slice is not
// one of these, since it does not own what it looks at.
// A generic struct's declared field names a parameter bound to nothing, so the
// declarations alone say `Slab` holds no resource and therefore that no
// `Slab<T, N>` does. A pool of a type holding a resource was ordinary data, so
// the resource in a slot was dropped when the slot was reused and neither
// compiler said anything. The instantiation the program writes is what binds the
// parameter, and that is what is asked now.
#[test]
fn both_compilers_refuse_a_pool_of_resources_nobody_releases() {
    let source = "Slab :: struct($T: Type, $N: usize) {\n\
                  \x20   storage: [N]T,\n\
                  \x20   generations: [N]i64,\n\
                  \x20   free_list: [N]i64,\n\
                  \x20   free_count: i64,\n}\n\
                  File :: linear struct { fd: i64 }\n\
                  Node :: struct { file: File, hp: i64 }\n\
                  main :: fn() -> i64 {\n\
                  \x20   var pool : Slab<Node, 2> = slab_new()\n\
                  \x20   0\n}\n";
    // The self-hosted half is in REFUSED_BY_BOTH.
    let bootstrap = bootstrap_refusal("poolboot", source);
    assert!(
        bootstrap.contains("is a pool of"),
        "the bootstrap took a pool of resources:\n{bootstrap}"
    );
}

// A pool written out rather than instantiated from a generic one. Both rules ran
// over the instantiations a program names, so a concrete container of the same
// shape was not asked about at all.
#[test]
fn both_compilers_refuse_a_concrete_pool_of_resources() {
    let source = "File :: linear struct { fd: i64 }\n\
                  Pool :: struct { storage: [2]File, generations: [2]i64 }\n\
                  drop_pool :: fn(move p: Pool) -> i64 { 0 }\n\
                  main :: fn() -> i64 {\n\
                  \x20   p := Pool { storage = [File { fd = 1 }; 2], generations = [0; 2] }\n\
                  \x20   drop_pool(p)\n}\n";
    let bootstrap = bootstrap_refusal("cpoolboot", source);
    assert!(
        bootstrap.contains("is a pool of"),
        "the bootstrap took a concrete pool of resources:\n{bootstrap}"
    );
}

// Consuming a value, writing into part of it, and consuming it again. The first
// consumption hands the storage to someone else, so the write is not the caller's
// to make and the second consumption is the same value twice.
#[test]
fn both_compilers_refuse_writing_into_what_was_consumed() {
    let source = "File :: linear struct { fd: i64 }\n\
                  Holder :: struct { file: File, name: i64 }\n\
                  close :: fn(move f: File) -> i64 { f.fd }\n\
                  open :: fn(n: i64) -> File { File { fd = n } }\n\
                  drop_holder :: fn(move h: Holder) -> i64 { close(h.file) }\n\
                  main :: fn() -> i64 {\n\
                  \x20   var h := Holder { file = open(7), name = 1 }\n\
                  \x20   drop_holder(h)\n\
                  \x20   h.file = open(9)\n\
                  \x20   drop_holder(h)\n}\n";
    let bootstrap = bootstrap_refusal("reviveboot", source);
    assert!(
        bootstrap.contains("moved"),
        "the bootstrap took a write into consumed storage:\n{bootstrap}"
    );
}

// A slab-shaped struct carrying a run of resources that is not its slot table.
// The rule is about the elements a handle addresses, so `storage` is the question
// and another field is not: a field holding resources makes the struct a resource
// by the ordinary rule, and that is where it is answered for. The two compilers
// read this differently at first, one asking `storage` and the other every array
// it could find, which is two languages rather than one.
#[test]
fn both_compilers_take_a_pool_beside_a_run_of_resources() {
    let source = "File :: linear struct { fd: i64 }\n\
                  Thing :: struct {\n\
                  \x20   storage: [2]i64,\n\
                  \x20   generations: [2]i64,\n\
                  \x20   extras: [2]File,\n}\n\
                  drop_thing :: fn(move t: Thing) -> i64 { 0 }\n\
                  main :: fn() -> i64 {\n\
                  \x20   t := Thing {\n\
                  \x20       storage = [0; 2],\n\
                  \x20       generations = [0; 2],\n\
                  \x20       extras = [File { fd = 1 }; 2],\n\
                  \x20   }\n\
                  \x20   drop_thing(t)\n}\n";
    let Some((ok, stderr)) = compile_and_run_status("pooladj", source) else {
        return;
    };
    assert!(
        ok,
        "the bootstrap refused a pool beside resources:\n{stderr}"
    );
    let Some(compiler) = build_self_hosted_compiler("pooladj") else {
        return;
    };
    let hosted = selfhosted_default_output(
        &compiler, "pooladj", source, "--emit-c", "c",
    );
    assert_eq!(hosted, "", "the self-hosted compiler disagreed");
}

// The questions an editor asks, answered by the self-hosted compiler from the
// tables its passes already build. The expectations are the ones the
// bootstrap's query layer answers in its own unit tests, so the two compilers
// are held to one oracle: the outline in source order, a definition's line, a
// struct's fields with their types, and a local's type off the checked walk.
#[test]
fn self_hosted_answers_editor_queries() {
    let Some(compiler) = build_self_hosted_compiler("query") else {
        return;
    };
    let directory = std::env::temp_dir();
    let input = directory.join("frost_query_sample.frost");
    let source = "Point :: struct { x: i64, y: i64 }\n\
                  Shape :: enum { Dot, Box { w: i64, h: i64 } }\n\
                  LIMIT :: 32\n\
                  area :: fn(w: i64, h: i64) -> i64 {\n\
                  \x20   total := w * h\n\
                  \x20   total\n\
                  }\n\
                  main :: fn() -> i64 { area(2, 3) }\n";
    std::fs::write(&input, source).unwrap();
    let asked = [
        (
            "symbols",
            "Point struct 1\nShape enum 2\nLIMIT const 3\narea fn 4\nmain fn 8\n",
        ),
        ("definition Shape", "2\n"),
        ("fields Point", "x i64\ny i64\n"),
        ("local area total", "i64 5\n"),
    ];
    for (question, wanted) in asked {
        let run = Command::new(&compiler)
            .env("FROST_INPUT", &input)
            .env("FROST_QUERY", question)
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "the query '{question}' failed:\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        let answer = String::from_utf8_lossy(&run.stderr).replace("\r\n", "\n");
        assert_eq!(
            answer, wanted,
            "the query '{question}' answered differently"
        );
        assert!(
            run.stdout.is_empty(),
            "the query '{question}' emitted a program"
        );
    }
    let _ = std::fs::remove_file(&input);
}

// A resource reached through a field is a place of its own, so consuming it
// twice consumes it twice. Both compilers tracked moves by name, so the field
// was never recorded and neither said the second consumption was one: three
// calls to a consumer over one resource, in safe code, with no `unsafe`
// anywhere. With a consumer that frees, that is a double free.
//
// Held here against both compilers, since a refusal only one of them makes is a
// divergence rather than a rule.
// Programs whose faults both compilers keep reading past, so one run reports
// every one of them. Each is compared as the (line, message) pairs read off
// the caret reports, in order: the same faults, on the same lines, in the
// same words. Column and path spelling stay each compiler's own.
const SAME_FAULTS: &[(&str, &str)] = &[
    (
        "twostmt",
        "good :: fn() -> i64 {\n    held := 3\n    ]\n    other := 4\n    )\n    held + other\n}\nmain :: fn() -> i64 { good() }\n",
    ),
    (
        "topandstmt",
        "42\ngood :: fn() -> i64 {\n    held := 3\n    ]\n    held\n}\nmain :: fn() -> i64 { good() }\n",
    ),
    (
        "strayinbody",
        "main :: fn() -> i64 {\n    x := 7\u{a3}\n    0\n}\n",
    ),
    (
        "strayincall",
        "take :: fn(v: i64) -> i64 { v }\nmain :: fn() -> i64 {\n    take(7\u{a3})\n    0\n}\n",
    ),
    ("topbinding", "x := 1\nmain :: fn() -> i64 {\n    0\n}\n"),
    (
        "booltparam",
        "Pair :: struct($true: Type) { first: i64 }\nmain :: fn() -> i64 { 0 }\n",
    ),
    (
        "mutlocal",
        "main :: fn() -> i64 {\n    mut x := 1\n    x = 2\n    x\n}\n",
    ),
    (
        "mutinlist",
        "divide :: fn(a: i64, b: i64) -> (q: i64, r: i64) {\n    return a / b, a % b\n}\nmain :: fn() -> i64 {\n    a, mut b := divide(7, 2)\n    b = b + 1\n    a + b\n}\n",
    ),
    (
        "boolname",
        "main :: fn() -> i64 {\n    true := 1\n    0\n}\n",
    ),
    (
        "twounknown",
        "one :: fn() -> i64 {\n    bogus\n}\ntwo :: fn() -> i64 {\n    other\n}\nmain :: fn() -> i64 { one() + two() }\n",
    ),
    (
        "moveagain",
        "Buffer :: linear struct { size: i64 }\nbuffer_make :: fn() -> Buffer { Buffer { size = 4 } }\nbuffer_free :: fn(move b: Buffer) { }\nmain :: fn() -> i64 {\n    held := buffer_make()\n    buffer_free(held)\n    buffer_free(held)\n    0\n}\n",
    ),
];

// A program whose faults are found by four different walks: an escape from a
// frame, a resource consumed twice, a call given too many arguments and a name
// nothing declares. One run names all four, which is the whole point of the
// checks reporting rather than stopping: an agent editing this file learns
// everything wrong with it from one compile instead of four.
const FOUR_WALKS: &str = "Buffer :: linear struct { size: i64 }\n\
     buffer_make :: fn() -> Buffer { Buffer { size = 4 } }\n\
     buffer_free :: fn(move b: Buffer) { }\n\
     takes_one :: fn(v: i64) -> i64 { v }\n\
     moved_twice :: fn() -> i64 {\n\
     \x20   held := buffer_make()\n\
     \x20   buffer_free(held)\n\
     \x20   buffer_free(held)\n\
     \x20   0\n}\n\
     escapes :: fn() -> ^i64 {\n\
     \x20   var local : i64 = 3\n\
     \x20   unsafe { ptr_to(local) }\n}\n\
     too_many :: fn() -> i64 { takes_one(1, 2) }\n\
     unknown_name :: fn() -> i64 { nowhere }\n\
     main :: fn() -> i64 { moved_twice() + too_many() + unknown_name() }\n";

#[test]
fn one_run_names_what_every_walk_found() {
    let report = bootstrap_refusal("fourwalks", FOUR_WALKS);
    for wanted in [
        "region: a pointer into the frame of 'escapes'",
        "use of moved value 'held'",
        "expects 1 argument(s) but 2 were given",
        "unknown variable 'nowhere'",
    ] {
        assert!(
            report.contains(wanted),
            "one run did not say '{wanted}':\n{report}"
        );
    }
}

// A fault reported by two walks is one fault. The ownership rules are walked
// once over the source and once over the bodies specialization expands, and for
// a program with no generic in it both walks say the same thing; the run used
// to say it twice.
#[test]
fn one_fault_is_reported_once() {
    let report = bootstrap_refusal(
        "saidonce",
        "Buffer :: linear struct { size: i64 }\n\
         buffer_make :: fn() -> Buffer { Buffer { size = 4 } }\n\
         buffer_free :: fn(move b: Buffer) { }\n\
         main :: fn() -> i64 {\n\
         \x20   held := buffer_make()\n\
         \x20   buffer_free(held)\n\
         \x20   buffer_free(held)\n\
         \x20   0\n}\n",
    );
    let said = report.matches("use of moved value 'held'").count();
    assert_eq!(said, 1, "the one fault was named {said} times:\n{report}");
}

// An undeclared name used in two places is one thing to fix, so it is one
// report carrying both places rather than two reports.
#[test]
fn one_root_cause_is_one_report_with_every_place() {
    let report = bootstrap_refusal(
        "oneroot",
        "one :: fn() -> i64 { Absent { a = 1 }.a }\n\
         two :: fn() -> i64 { Absent { a = 2 }.a }\n\
         three :: fn() -> i64 { Absent { a = 3 }.a }\n\
         main :: fn() -> i64 { one() + two() + three() }\n",
    );
    let said = report
        .matches("'Absent' is not a type this program declares")
        .count();
    assert_eq!(
        said, 3,
        "one root cause was named {said} times, and it shows in three places:\n{report}"
    );
    // Three places, and the caret report puts each on the line it is about.
    for line in ["1:22", "2:22", "3:24"] {
        assert!(
            report.contains(line),
            "the report does not point at {line}:\n{report}"
        );
    }
}

// What `--diagnostics=json` writes, read back and applied. The file is broken
// only in ways an edit answers, so a reader that applies every certain edit
// gets a program that builds, which is what makes the channel worth having.
#[test]
fn the_json_reports_round_trip_through_frost_fix() {
    if !linker_available() {
        return;
    }
    let directory = std::env::temp_dir().join(unique("frost_fixup"));
    std::fs::create_dir_all(&directory).unwrap();
    let file = directory.join("fixme.frost");
    std::fs::write(
        &file,
        "counted :: fn() -> i64 {\n\
         \x20   mut total := 0\n\
         \x20   total = total + 1\n\
         \x20   total\n}\n\
         doubled :: fn(n: i64) -> i64 {\n\
         \x20   mut held := n\n\
         \x20   held = held * 2\n\
         \x20   held\n}\n\
         main :: fn() -> i64 { counted() + doubled(4) }\n",
    )
    .unwrap();

    let object = directory.join("fixme.o");
    let asked = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--diagnostics=json")
        .arg("--native")
        .arg("-o")
        .arg(&object)
        .arg(&file)
        .output()
        .unwrap();
    let reports = String::from_utf8_lossy(&asked.stderr);
    let lines: Vec<&str> =
        reports.lines().filter(|line| !line.is_empty()).collect();
    assert_eq!(lines.len(), 2, "one report per fault:\n{reports}");
    for line in &lines {
        let held: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|_| panic!("not a report: {line}"));
        assert_eq!(held["severity"], "error");
        assert_eq!(held["fix"]["replacement"], "var");
        assert_eq!(held["fix"]["certain"], true);
        // The span is where the edit goes, counted in bytes, and it is as long
        // as the text it stands in for.
        let span = &held["fix"]["span"];
        assert_eq!(
            span[1].as_u64().unwrap() - span[0].as_u64().unwrap(),
            3,
            "the edit replaces `mut`"
        );
    }

    let fixed = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("fix")
        .arg(&file)
        .output()
        .unwrap();
    assert!(
        fixed.status.success(),
        "frost fix failed:\n{}",
        String::from_utf8_lossy(&fixed.stderr)
    );
    let text = std::fs::read_to_string(&file).unwrap();
    assert!(
        !text.contains("mut "),
        "an edit was left unapplied:\n{text}"
    );

    let built = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--native")
        .arg("-o")
        .arg(&object)
        .arg(&file)
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "the fixed file did not build:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let _ = std::fs::remove_dir_all(&directory);
}

// The (line, message) pairs of a diagnostic dump: each `path:line:col:` header
// sets the line, and each caret line under it contributes its message.
fn located_faults(report: &str) -> Vec<(usize, String)> {
    let mut faults = Vec::new();
    let mut at = 0usize;
    for line in report.lines() {
        let held = line.trim_end();
        if let Some(rest) = held.strip_suffix(':') {
            let mut parts = rest.rsplitn(3, ':');
            let column = parts.next().unwrap_or("");
            let row = parts.next().unwrap_or("");
            if !column.is_empty()
                && !row.is_empty()
                && column.bytes().all(|b| b.is_ascii_digit())
                && row.bytes().all(|b| b.is_ascii_digit())
            {
                at = row.parse().unwrap();
            }
        }
        if let Some(mark) = held.find("^ ") {
            faults.push((at, held[mark + 2..].to_string()));
        }
    }
    faults
}

// Programs whose faults are found by three different walks, so one run names
// all of them and the two compilers name the same ones.
//
// Compared as a set rather than in order, which is the one thing here the two
// compilers do not share: which fault comes first follows which walk found it,
// and each runs its walks in its own order. The bootstrap checks what a program
// owns before lowering it and finds a call of the wrong length while lowering;
// the self-hosted compiler counts arguments first and walks moves after. The
// faults, their lines and their words are the same, and that is the part worth
// holding.
const SAME_FAULTS_ANY_ORDER: &[(&str, &str)] = &[(
    "threewalks",
    "Buffer :: linear struct { size: i64 }\n\
     buffer_make :: fn() -> Buffer { Buffer { size = 4 } }\n\
     buffer_free :: fn(move b: Buffer) { }\n\
     takes_one :: fn(v: i64) -> i64 { v }\n\
     moved_twice :: fn() -> i64 {\n\
     \x20   held := buffer_make()\n\
     \x20   buffer_free(held)\n\
     \x20   buffer_free(held)\n\
     \x20   0\n}\n\
     too_many :: fn() -> i64 { takes_one(1, 2) }\n\
     unknown_name :: fn() -> i64 { nowhere }\n\
     main :: fn() -> i64 { moved_twice() + too_many() + unknown_name() }\n",
)];

#[test]
fn both_compilers_name_the_same_faults_whatever_found_them() {
    let Some(compiler) = build_self_hosted_compiler("anyorder") else {
        return;
    };
    let directory = std::env::temp_dir();
    for (name, source) in SAME_FAULTS_ANY_ORDER {
        let bootstrap = bootstrap_refusal(name, source);
        let input = directory.join(format!("frost_order_{name}.frost"));
        std::fs::write(&input, source).unwrap();
        let run = Command::new(&compiler)
            .env("FROST_INPUT", &input)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            !run.status.success(),
            "the self-hosted compiler built {name}, which the bootstrap refuses"
        );
        let hosted = String::from_utf8_lossy(&run.stderr).to_string();
        let mut wanted = located_faults(&bootstrap);
        let mut got = located_faults(&hosted);
        wanted.sort();
        got.sort();
        // Three faults, and the move points back at where the value went, so
        // four located lines.
        assert_eq!(
            wanted.len(),
            4,
            "one run should name every fault of {name}:\n{bootstrap}"
        );
        assert_eq!(
            wanted, got,
            "the two compilers name different faults in {name}:\nbootstrap:\n{bootstrap}\nself-hosted:\n{hosted}"
        );
    }
}

#[test]
fn both_compilers_report_the_same_fault_lines() {
    let Some(compiler) = build_self_hosted_compiler("faultlines") else {
        return;
    };
    let directory = std::env::temp_dir();
    for (name, source) in SAME_FAULTS {
        let bootstrap = bootstrap_refusal(name, source);
        let input = directory.join(format!("frost_faults_{name}.frost"));
        std::fs::write(&input, source).unwrap();
        let run = Command::new(&compiler)
            .env("FROST_INPUT", &input)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        assert!(
            !run.status.success(),
            "the self-hosted compiler built {name}, which the bootstrap refuses"
        );
        let hosted = String::from_utf8_lossy(&run.stderr).to_string();
        let wanted = located_faults(&bootstrap);
        let got = located_faults(&hosted);
        assert!(
            !wanted.is_empty(),
            "no located fault in the bootstrap's report for {name}:\n{bootstrap}"
        );
        assert_eq!(
            wanted, got,
            "the two compilers describe {name} differently:\nbootstrap:\n{bootstrap}\nself-hosted:\n{hosted}"
        );
    }
}

#[test]
fn both_compilers_refuse_consuming_a_field_twice() {
    let source = "File :: linear struct { fd: i64 }\n\
                  Holder :: struct { file: File, name: i64 }\n\
                  close :: fn(move f: File) -> i64 { f.fd }\n\
                  drop_holder :: fn(move h: Holder) -> i64 { close(h.file) }\n\
                  main :: fn() -> i64 {\n\
                  \x20   h := Holder { file = File { fd = 7 }, name = 1 }\n\
                  \x20   close(h.file)\n\
                  \x20   close(h.file)\n\
                  \x20   drop_holder(h)\n}\n";
    let bootstrap = bootstrap_refusal("dfboot", source);
    assert!(
        bootstrap.contains("moved"),
        "the bootstrap took a resource consumed twice:\n{bootstrap}"
    );
}

// The same through an element, and through a `mut` borrow, which reaches the
// place by a different road: the mode lowering has already turned the parameter
// into a borrow by the time the check runs, so what the callee declared it
// takes is what says a resource was handed over.
#[test]
fn both_compilers_refuse_consuming_an_element_or_a_borrowed_field_twice() {
    let element = "File :: linear struct { fd: i64 }\n\
                   close :: fn(move f: File) -> i64 { f.fd }\n\
                   drop_run :: fn(move xs: [2]File) -> i64 { 0 }\n\
                   main :: fn() -> i64 {\n\
                   \x20   var run : [2]File = [File { fd = 9 }; 2]\n\
                   \x20   close(run[0])\n\
                   \x20   close(run[0])\n\
                   \x20   drop_run(run)\n}\n";
    let borrowed = "File :: linear struct { fd: i64 }\n\
                    Holder :: struct { file: File, name: i64 }\n\
                    close :: fn(move f: File) -> i64 { f.fd }\n\
                    twice :: fn(mut h: Holder) -> i64 {\n\
                    \x20   close(h.file)\n\
                    \x20   close(h.file)\n}\n\
                    drop_holder :: fn(move h: Holder) -> i64 { close(h.file) }\n\
                    main :: fn() -> i64 {\n\
                    \x20   var h := Holder { file = File { fd = 5 }, name = 1 }\n\
                    \x20   twice(h)\n\
                    \x20   drop_holder(h)\n}\n";
    for (name, source) in [("dfelem", element), ("dfborrow", borrowed)] {
        let bootstrap = bootstrap_refusal(name, source);
        assert!(
            bootstrap.contains("moved"),
            "the bootstrap took {name}:\n{bootstrap}"
        );
    }
}

// A block vouching for nothing is reported on every build. `--audit-unsafe`
// turns the warning into a failure, which is what holds a tree to zero of them.
fn audit_unsafe(name: &str, source: &str) -> String {
    compile_reporting_unsafe(name, source, true).0
}

/// The compiler's stderr and whether it succeeded, with the audit optionally
/// promoted to a failure.
fn compile_reporting_unsafe(
    name: &str,
    source: &str,
    fatal: bool,
) -> (String, bool) {
    let directory = std::env::temp_dir();
    let source_path = directory.join(format!("frost_audit_{name}.frost"));
    std::fs::write(&source_path, source).unwrap();
    let frost = env!("CARGO_BIN_EXE_frost");
    let mut command = Command::new(frost);
    if fatal {
        command.arg("--audit-unsafe");
    }
    let output = command
        .arg("-o")
        .arg(directory.join(format!("frost_audit_{name}.o")))
        .arg(&source_path)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&source_path);
    (
        String::from_utf8_lossy(&output.stderr).to_string(),
        output.status.success(),
    )
}

#[test]
fn an_idle_block_warns_on_an_ordinary_build() {
    let source = "import \"io.frost\"\nmain :: fn() -> i64 {\n\
                  \x20   x := unsafe { 1 + 1 }\n\
                  \x20   print(\"{}\\n\", x)\n\
                  \x20   0\n}\n";
    let (message, built) = compile_reporting_unsafe("warn", source, false);
    assert!(
        message.contains("vouches for nothing"),
        "expected a warning without the flag, got:\n{message}"
    );
    // A warning, so the program still builds. The flag is what refuses it.
    assert!(built, "the warning failed the build:\n{message}");
    let (_, refused) = compile_reporting_unsafe("warnfatal", source, true);
    assert!(
        !refused,
        "`--audit-unsafe` took a block that vouches for nothing"
    );
}

#[test]
fn the_audit_reports_a_block_that_vouches_for_nothing() {
    let source = "import \"io.frost\"\nmain :: fn() -> i64 {\n\
                  \x20   x := unsafe { 1 + 1 }\n\
                  \x20   print(\"{}\\n\", x)\n\
                  \x20   0\n}\n";
    let message = audit_unsafe("idle", source);
    assert!(
        message.contains("vouches for nothing"),
        "expected an audit finding, got:\n{message}"
    );
}

#[test]
fn the_audit_reports_a_block_inside_another() {
    let source = "import \"io.frost\"\nheld :: extern fn(x: i64) -> i64\n\
                  main :: fn() -> i64 {\n\
                  \x20   x := unsafe { unsafe { held(1) } }\n\
                  \x20   print(\"{}\\n\", x)\n\
                  \x20   0\n}\n";
    let message = audit_unsafe("nested", source);
    assert!(
        message.contains("inside another one"),
        "expected an audit finding, got:\n{message}"
    );
}

#[test]
fn the_audit_is_quiet_when_every_block_earns_itself() {
    let source = "import \"io.frost\"\nheld :: extern fn(x: i64) -> i64\n\
                  main :: fn() -> i64 {\n\
                  \x20   x := unsafe { held(1) }\n\
                  \x20   print(\"{}\\n\", x)\n\
                  \x20   0\n}\n";
    let message = audit_unsafe("clean", source);
    assert!(
        !message.contains("vouches") && !message.contains("inside another"),
        "expected no audit findings, got:\n{message}"
    );
}

// Indexing a raw pointer is unchecked wherever it is written, including inside
// `ptr_to`. The self-hosted compiler asked the question where no local's type
// was known and let it through.
#[test]
fn the_self_hosted_compiler_gates_an_index_through_a_raw_pointer() {
    let source = "import \"io.frost\"\nhold :: extern fn(size: i64) -> ^u8\n\
                  at :: fn(block: ^u8, offset: i64) -> ^u8 { ptr_to(block[offset]) }\n\
                  main :: fn() -> i64 {\n\
                  \x20   print(\"{}\\n\", 0)\n\
                  \x20   0\n}\n";
    let Some(compiler) = build_self_hosted_compiler("ckrawindex") else {
        return;
    };
    let directory = std::env::temp_dir();
    let input = directory.join("frost_rawindex.frost");
    std::fs::write(&input, source).unwrap();
    let run = Command::new(&compiler)
        .env("FROST_INPUT", &input)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    let message = String::from_utf8_lossy(&run.stderr).to_string();
    assert!(
        !run.status.success() && message.contains("indexing a raw pointer"),
        "expected the gate to refuse it, got:\n{message}"
    );
}

// A name written inside an array literal is a name the module may have
// imported. The import rewrite treated every literal as a leaf, so a call to an
// imported function inside one kept the name it had in the other module and
// reached the backend as an unknown variable.
#[test]
fn an_imported_call_inside_an_array_literal_resolves() {
    let directory = std::env::temp_dir();
    let helper = directory.join("frost_arraylit_helper.frost");
    std::fs::write(
        &helper,
        "export Point, point\n\
         Point :: struct { x: i64 }\n\
         point :: fn(x: i64) -> Point { Point { x = x } }\n",
    )
    .unwrap();
    let source = "import \"io.frost\"\nimport \"frost_arraylit_helper.frost\"\n\
                  Held :: struct { p: Point }\n\
                  main :: fn() -> i64 {\n\
                  \x20   listed : [2]Held = [ Held { p = point(1) }, Held { p = point(2) } ]\n\
                  \x20   repeated : [2]Held = [ Held { p = point(3) }; 2 ]\n\
                  \x20   print(\"{}\\n\", listed[1].p.x)\n\
                  \x20   print(\"{}\\n\", repeated[0].p.x)\n\
                  \x20   0\n}\n";
    let Some(output) = compile_and_run_unaudited("arraylit", source) else {
        return;
    };
    let _ = std::fs::remove_file(&helper);
    assert_eq!(output, "2\n3\n");
}

// The graphics examples cannot be linked without SDL3 and wgpu-native, but they
// can be compiled, which is what says the bindings still typecheck: the handle
// types, the checked constructors, and the wrappers that stand between a
// program and the C calls.
#[test]
fn the_graphics_examples_compile_against_their_bindings() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let frost = env!("CARGO_BIN_EXE_frost");
    // The tree carries the generated binding, so this is normally there. A
    // checkout that has had it removed still checks the hand-written one.
    let generated = graphics_source(&root, "wgpu.frost");
    // Every demo, so one that imports `renderer.frost` is among them. The two
    // that import nothing were the whole list, and a renderer the region check
    // refused compiled here for as long as nobody named it.
    let examples: &[&str] = if generated.exists() {
        &[
            "window.frost",
            "triangle.frost",
            "scene.frost",
            "spinning.frost",
            "textured.frost",
            "shadowed.frost",
            "graph.frost",
            "scene_sync.frost",
            "geometry.frost",
            "uniform.frost",
            "world.frost",
            "app.frost",
            "gltf.frost",
            "gltf_model.frost",
            "lit.frost",
            "swarm.frost",
            "template.frost",
        ]
    } else {
        &["window.frost"]
    };
    for example in examples {
        let source = graphics_source(&root, example);
        let object =
            std::env::temp_dir().join(format!("frost_gfx_{example}.o"));
        let output = Command::new(frost)
            .arg("--audit-unsafe")
            .arg("--native")
            .arg("-o")
            .arg(&object)
            .arg(&source)
            .current_dir(&root)
            .output()
            .unwrap();
        let message = String::from_utf8_lossy(&output.stderr).to_string();
        assert!(
            output.status.success(),
            "{example} did not compile:\n{message}"
        );
        assert!(
            !message.contains("vouches") && !message.contains("inside another"),
            "{example} has an idle unsafe block:\n{message}"
        );
        let _ = std::fs::remove_file(&object);
    }
}

// What a render graph orders, and what it refuses. Ordering, resource lifetimes
// and pool assignment are all arithmetic over tables, so every one of these
// runs with `no_device()` in place of a GPU: the order a pass runs in, which
// transients end up sharing one texture, the load op each attachment gets, the
// phase and enabled state a pass carries, and the five graphs that cannot run
// at all.
//
// Linking is what needs the libraries, since the graph imports the wgpu binding
// and that names symbols whether or not a test calls them. Where they are not
// beside the examples this says nothing rather than failing, which is a
// checkout that has not run `just deps`.
//
// A `--test` build runs from the temp directory, and the loader looks for a
// shared library beside the executable rather than beside the source. The two
// directories holding them go on the search path, or the test binary dies
// before `main` with nothing on either stream and the failure reads as the
// graph answering wrong.
#[test]
fn the_render_graph_orders_its_passes() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let Some(libraries) = graphics_libraries(&root) else {
        return;
    };
    let source = graphics_source(&root, "graph.frost");
    let search = library_search_path(&root);
    // Both backends, because a difference between them is a difference in what
    // the graph decides.
    for emit_c in [false, true] {
        if emit_c && c_compiler().is_none() {
            continue;
        }
        let mut command = Command::new(env!("CARGO_BIN_EXE_frost"));
        if emit_c {
            command.arg("--emit-c");
        }
        for library in &libraries {
            command.arg("--libs").arg(library);
        }
        let run = command
            .arg("--test")
            .arg(&source)
            .current_dir(&root)
            .env(
                if cfg!(windows) {
                    "PATH"
                } else {
                    "LD_LIBRARY_PATH"
                },
                &search,
            )
            .output()
            .unwrap();
        let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
        assert!(
            output.contains("24 passed, 0 failed"),
            "the render graph's own tests did not pass (emit_c: {emit_c}):\n{output}{}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

// How the shared buffers grow when a mesh does not fit and shrink when most of
// what was in them has gone. Unlike the graph, this cannot be answered on
// tables: what a grow does is allocate a second buffer and copy on the device,
// so the tests open a device with no window behind it and read the buffer back
// afterwards.
//
// A machine with no adapter at all says so and passes, which is what a
// container without a GPU is. The alternative reports the machine rather than
// the code.
#[test]
fn the_mesh_cache_grows_and_compacts_its_buffers() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let Some(libraries) = graphics_libraries(&root) else {
        return;
    };
    let source = graphics_source(&root, "mesh.frost");
    let search = library_search_path(&root);
    for emit_c in [false, true] {
        if emit_c && c_compiler().is_none() {
            continue;
        }
        let mut command = Command::new(env!("CARGO_BIN_EXE_frost"));
        if emit_c {
            command.arg("--emit-c");
        }
        for library in &libraries {
            command.arg("--libs").arg(library);
        }
        let run = command
            .arg("--test")
            .arg(&source)
            .current_dir(&root)
            .env(
                if cfg!(windows) {
                    "PATH"
                } else {
                    "LD_LIBRARY_PATH"
                },
                &search,
            )
            .output()
            .unwrap();
        let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
        assert!(
            output.contains("7 passed, 0 failed"),
            "the mesh cache's own tests did not pass (emit_c: {emit_c}):\n{output}{}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

// How a program composes what the engine offers: a group whose members are
// replaceable and removable before any of them is installed. Answered without a
// device, because which plugins are in a group is arithmetic over a list.
#[test]
fn the_app_composes_its_plugins() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let Some(libraries) = graphics_libraries(&root) else {
        return;
    };
    let source = graphics_source(&root, "app.frost");
    let search = library_search_path(&root);
    for emit_c in [false, true] {
        if emit_c && c_compiler().is_none() {
            continue;
        }
        let mut command = Command::new(env!("CARGO_BIN_EXE_frost"));
        if emit_c {
            command.arg("--emit-c");
        }
        for library in &libraries {
            command.arg("--libs").arg(library);
        }
        let run = command
            .arg("--test")
            .arg(&source)
            .current_dir(&root)
            .env(
                if cfg!(windows) {
                    "PATH"
                } else {
                    "LD_LIBRARY_PATH"
                },
                &search,
            )
            .output()
            .unwrap();
        let output = String::from_utf8_lossy(&run.stdout);
        assert!(
            output.contains("5 passed, 0 failed"),
            "the App's own tests did not pass (emit_c: {emit_c}):
{output}{}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

// What a frame does to a world, with no device anywhere near it: the camera
// moved by what the world was told was held, a frame in which nothing was held
// moving nothing, and a thing three deep in a tree placed against the whole
// chain above it rather than against its own parent alone.
//
// The libraries and the search path are needed for the same reason the graph's
// tests need them: the file reaches the wgpu binding through the renderer side
// it writes into, and that names symbols whether or not a test calls them.
#[test]
fn the_frame_moves_the_world_it_was_given() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let Some(libraries) = graphics_libraries(&root) else {
        return;
    };
    let source = graphics_source(&root, "world.frost");
    let search = library_search_path(&root);
    for emit_c in [false, true] {
        if emit_c && c_compiler().is_none() {
            continue;
        }
        let mut command = Command::new(env!("CARGO_BIN_EXE_frost"));
        if emit_c {
            command.arg("--emit-c");
        }
        for library in &libraries {
            command.arg("--libs").arg(library);
        }
        let run = command
            .arg("--test")
            .arg(&source)
            .current_dir(&root)
            .env(
                if cfg!(windows) {
                    "PATH"
                } else {
                    "LD_LIBRARY_PATH"
                },
                &search,
            )
            .output()
            .unwrap();
        let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
        assert!(
            output.contains("7 passed, 0 failed"),
            "the frame's own tests did not pass (emit_c: {emit_c}):\n{output}{}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

// Giving back a renderer that never opened. Opening is a run of acquisitions
// and any of them can fail partway, so a renderer that got some of the way
// holds real handles for the steps that worked and handles naming nothing for
// the rest. Releasing one of those has to be a no-op: wgpu's release functions
// abort the process when handed a null, so without the guard the close after a
// failed open is what kills the program, on the machine that was already having
// trouble opening a device.
//
// The guard is written by `tools/wgpu_bindgen.frost` into every one of the
// twenty-three release wrappers, so this covers the shape rather than one of
// them.
#[test]
fn a_renderer_that_never_opened_is_closed_safely() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let Some(libraries) = graphics_libraries(&root) else {
        return;
    };
    let source = graphics_source(&root, "renderer.frost");
    let search = library_search_path(&root);
    for emit_c in [false, true] {
        if emit_c && c_compiler().is_none() {
            continue;
        }
        let mut command = Command::new(env!("CARGO_BIN_EXE_frost"));
        if emit_c {
            command.arg("--emit-c");
        }
        for library in &libraries {
            command.arg("--libs").arg(library);
        }
        let run = command
            .arg("--test")
            .arg(&source)
            .current_dir(&root)
            .env(
                if cfg!(windows) {
                    "PATH"
                } else {
                    "LD_LIBRARY_PATH"
                },
                &search,
            )
            .output()
            .unwrap();
        let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
        assert!(
            output.contains("1 passed, 0 failed"),
            "the renderer's own tests did not pass (emit_c: {emit_c}):\n{output}{}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

// A binary glTF file read back into geometry, materials and a tree of nodes,
// against `lib/engine/assets/shapes.glb`. That file is written by
// `assets/generate.py` and is deliberately awkward in the ways real exporters
// are: two index widths and a primitive with none, an explicit byte stride and
// views without one, a primitive missing its normal, a node giving three parts
// and a node giving a whole matrix, and two roots.
//
// One of these counts blocks rather than values. A reader that took a run twice
// and let the second stand would answer every other test correctly and leak a
// block per primitive, which is what that one is there to see.
#[test]
fn a_binary_gltf_file_reads_back_as_geometry() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let Some(libraries) = graphics_libraries(&root) else {
        return;
    };
    let source = graphics_source(&root, "gltf.frost");
    let search = library_search_path(&root);
    for emit_c in [false, true] {
        if emit_c && c_compiler().is_none() {
            continue;
        }
        let mut command = Command::new(env!("CARGO_BIN_EXE_frost"));
        if emit_c {
            command.arg("--emit-c");
        }
        for library in &libraries {
            command.arg("--libs").arg(library);
        }
        let run = command
            .arg("--test")
            .arg(&source)
            .current_dir(&root)
            .env(
                if cfg!(windows) {
                    "PATH"
                } else {
                    "LD_LIBRARY_PATH"
                },
                &search,
            )
            .output()
            .unwrap();
        let output = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
        assert!(
            output.contains("11 passed, 0 failed"),
            "the glTF reader's own tests did not pass (emit_c: {emit_c}):\n{output}{}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
}
// The graphics code is four layers and they are directories: `lib/platform`
// binds the window and the input, `lib/renderer` binds the device and owns what
// it needs of an entity, `lib/engine` works out where everything ended up, and
// the programs under `examples/graphics` use all three. Each may reach the ones
// below it and none may reach the ones above.
//
// The order is declared in `frost.json` and both compilers refuse a crossing
// when they resolve the import. This checks that they both do, on the same
// program, because a rule one compiler enforces and the other does not is a
// difference in what they accept.
//
// The upward import is written into a real layer directory rather than a
// temporary one, since the check is about which layer a file sits in and a file
// somewhere else is under no layer at all.
// The same rule where the import is found through a search root written
// absolutely rather than beside the importing file. A relative path and an
// absolute one name the same file and share no prefix, so a check comparing
// them as written answers no and lets the crossing through. Both compilers make
// the two paths absolute before comparing, and this is what says both do.
#[test]
fn both_compilers_refuse_a_layer_reached_through_an_absolute_root() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root
        .join("lib")
        .join("renderer")
        .join("absolute_probe.frost");
    std::fs::write(
        &source,
        "import \"world.frost\"\n\nmain :: fn() -> i64 { 0 }\n",
    )
    .unwrap();
    let engine = root.join("lib").join("engine");
    // Named the way a command line names it, relative to where the build runs,
    // while the search root is absolute. That is the pair the check has to make
    // comparable; two absolute paths compare without any of it.
    let named = Path::new("lib")
        .join("renderer")
        .join("absolute_probe.frost");

    let object = std::env::temp_dir().join("frost_absolute_probe.o");
    let bootstrap = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("-L")
        .arg(&engine)
        .arg("--native")
        .arg("-o")
        .arg(&object)
        .arg(&named)
        .current_dir(&root)
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&bootstrap.stderr).to_string();

    let hosted = build_self_hosted_compiler("absoluteprobe").map(|compiler| {
        let emitted = std::env::temp_dir().join("frost_absolute_probe.c");
        let run = Command::new(&compiler)
            .env("FROST_PATH", &engine)
            .arg("-o")
            .arg(&emitted)
            .arg(&named)
            .current_dir(&root)
            .output()
            .unwrap();
        (
            run.status.success(),
            String::from_utf8_lossy(&run.stderr).to_string(),
        )
    });

    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&object);

    let wanted = "layer: 'lib/renderer' may not reach 'lib/engine'";
    assert!(
        !bootstrap.status.success() && said.contains(wanted),
        "the bootstrap took a layer reached through an absolute root:\n{said}"
    );
    let Some((succeeded, hosted_said)) = hosted else {
        return;
    };
    assert!(
        !succeeded && hosted_said.contains(wanted),
        "the self-hosted compiler took it:\n{hosted_said}"
    );
}

#[test]
fn both_compilers_refuse_a_layer_reaching_upward() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("lib").join("renderer").join("upward_probe.frost");
    std::fs::write(
        &source,
        "import \"../engine/world.frost\"\n\nmain :: fn() -> i64 { 0 }\n",
    )
    .unwrap();

    let object = std::env::temp_dir().join("frost_layer_probe.o");
    let bootstrap = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--native")
        .arg("-o")
        .arg(&object)
        .arg(&source)
        .current_dir(&root)
        .output()
        .unwrap();
    let said = String::from_utf8_lossy(&bootstrap.stderr).to_string();

    let hosted = build_self_hosted_compiler("layerprobe").map(|compiler| {
        let emitted = std::env::temp_dir().join("frost_layer_probe.c");
        let run = Command::new(&compiler)
            .arg("-o")
            .arg(&emitted)
            .arg(&source)
            .current_dir(&root)
            .output()
            .unwrap();
        (
            run.status.success(),
            String::from_utf8_lossy(&run.stderr).to_string(),
        )
    });

    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&object);

    assert!(
        !bootstrap.status.success(),
        "the bootstrap compiled a layer reaching upward"
    );
    let wanted = "layer: 'lib/renderer' may not reach 'lib/engine'";
    assert!(
        said.contains(wanted),
        "the bootstrap said something else:\n{said}"
    );
    let Some((succeeded, hosted_said)) = hosted else {
        return;
    };
    assert!(
        !succeeded,
        "the self-hosted compiler took a layer reaching upward"
    );
    // The same sentence from both, since one message means one rule.
    assert!(
        hosted_said.contains(wanted),
        "the self-hosted compiler said something else:\n{hosted_said}"
    );
}

// Where a graphics module lives. The layers are directories, so a test names a
// file and this says which one holds it; a module moving between layers is one
// line here rather than an edit in every test that compiles it.
fn graphics_source(root: &Path, name: &str) -> PathBuf {
    let layer = match name {
        "sdl.frost" | "platform.frost" => "lib/platform",
        "wgpu.frost" | "renderer.frost" | "graph.frost" | "mesh.frost"
        | "material.frost" | "texture.frost" | "render_world.frost"
        | "geometry.frost" | "uniform.frost" | "gpu.frost"
        | "cluster.frost" => "lib/renderer",
        "template.frost" => "examples",
        "world.frost" | "camera.frost" | "scene_sync.frost" | "gltf.frost"
        | "app.frost" => "lib/engine",
        _ => "examples/graphics",
    };
    root.join(layer).join(name)
}

// Where the loader has to look for the graphics libraries, in front of
// whatever it already searched. On Windows this is `PATH` and elsewhere it is
// `LD_LIBRARY_PATH`; both are read at load time by the process being started,
// which is why it is set on the compiler that starts it.
fn library_search_path(root: &Path) -> std::ffi::OsString {
    let variable = if cfg!(windows) {
        "PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    let existing = std::env::var_os(variable).unwrap_or_default();
    let mut directories = vec![
        root.join("lib").join("platform"),
        root.join("lib").join("renderer").join("wgpu"),
    ];
    directories.extend(std::env::split_paths(&existing));
    std::env::join_paths(directories).unwrap_or(existing)
}

// What to link the graphics examples against, or nothing where they are not
// here. wgpu-native is downloaded beside the examples by `just deps` and is not
// in the repository, so its absence is what says a checkout has not run it.
// SDL3 is beside the examples on Windows and the system's package elsewhere.
fn graphics_libraries(root: &Path) -> Option<Vec<String>> {
    let platform = root.join("lib").join("platform");
    let renderer = root.join("lib").join("renderer");
    if !renderer.join("wgpu.frost").exists() {
        return None;
    }
    if cfg!(windows) {
        let sdl = platform.join("SDL3.dll");
        let wgpu = renderer.join("wgpu").join("wgpu_native.dll");
        if !sdl.exists() || !wgpu.exists() {
            return None;
        }
        return Some(vec![
            sdl.to_string_lossy().into_owned(),
            wgpu.to_string_lossy().into_owned(),
        ]);
    }
    for name in ["libwgpu_native.so", "libwgpu_native.dylib"] {
        let wgpu = renderer.join("wgpu").join(name);
        if wgpu.exists() {
            return Some(vec![
                "-lSDL3".to_string(),
                wgpu.to_string_lossy().into_owned(),
            ]);
        }
    }
    None
}

// The same binding through the other compiler. It could not read the SDL
// binding at all until a constant was allowed to be text, which is four lines
// of `sdl.frost` naming the window properties, so the whole graphics surface
// had never been past this compiler's type checker.
#[test]
fn self_hosted_compiles_the_sdl_binding() {
    let Some(compiler) = build_self_hosted_compiler("gfxsh") else {
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // `window.frost` reaches SDL and nothing else. `textured.frost` pulls in the
    // renderer, the camera, the mesh cache and the wgpu binding, which is the
    // whole graphics surface and the part a region rule reaches. `shadowed.frost`
    // adds the render graph, which is where a `match` answers with a texture and
    // where this compiler used to read one at the width of the zero its binding
    // was seeded with.
    //
    // The last two read `wgpu.frost`. The tree carries it now, so this is the
    // whole list on any checkout; the branch below is what a tree with the
    // binding deleted still does.
    let generated = graphics_source(&root, "wgpu.frost");
    let examples: &[&str] = if generated.exists() {
        &[
            "window.frost",
            "textured.frost",
            "shadowed.frost",
            "gltf_model.frost",
        ]
    } else {
        &["window.frost"]
    };
    for example in examples {
        let source = graphics_source(&root, example);
        let emitted =
            std::env::temp_dir().join(format!("frost_gfxsh_{example}.c"));
        let run = Command::new(&compiler)
            .arg("-o")
            .arg(&emitted)
            .arg(&source)
            .current_dir(&root)
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "the self-hosted compiler refused {example}:
{}",
            String::from_utf8_lossy(&run.stderr)
        );
        if *example == "window.frost" {
            let emitted_c =
                std::fs::read_to_string(&emitted).unwrap_or_default();
            assert!(
                emitted_c.contains("SDL_CreateWindow"),
                "the emitted C does not reach SDL"
            );
        }
        let _ = std::fs::remove_file(&emitted);
    }
}

// Both compilers, run from a working directory that is not the checkout, on a
// program that imports the standard library. This is what separates a compiler
// that is installed from one that only works where it was built: the standard
// library, the runtime and the toolchain all have to be found from the binary's
// own location rather than from wherever the caller happens to be standing.
fn installed_layout(name: &str, compiler: &Path) -> Option<(PathBuf, PathBuf)> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let home = std::env::temp_dir().join(unique(&format!("frost_away_{name}")));
    let bin = home.join("bin");
    std::fs::create_dir_all(bin.join("std")).ok()?;
    std::fs::create_dir_all(bin.join("runtime")).ok()?;
    std::fs::create_dir_all(home.join("work")).ok()?;
    for entry in std::fs::read_dir(root.join("std")).ok()? {
        let entry = entry.ok()?;
        if entry.path().extension().is_some_and(|it| it == "frost") {
            std::fs::copy(
                entry.path(),
                bin.join("std").join(entry.file_name()),
            )
            .ok()?;
        }
    }
    // Both halves of the runtime: the C stub and the Frost file holding the
    // checks. An install that carries one of them is an install where a program
    // that indexes anything does not link.
    std::fs::copy(
        root.join("runtime").join("frost_runtime.c"),
        bin.join("runtime").join("frost_runtime.c"),
    )
    .ok()?;
    std::fs::copy(
        root.join("runtime").join("runtime.frost"),
        bin.join("runtime").join("runtime.frost"),
    )
    .ok()?;
    let installed = bin.join(format!("frostc{}", std::env::consts::EXE_SUFFIX));
    std::fs::copy(compiler, &installed).ok()?;
    Some((installed, home.join("work")))
}

const AWAY_FROM_THE_CHECKOUT: &str = "import \"io.frost\"\nimport \"vec.frost\"

     main :: fn() -> i64 {
         var numbers := vec_new($i64, 4)
         vec_push($i64, numbers, 20)
         vec_push($i64, numbers, 22)
         print(\"{}\\n\", vec_get($i64, numbers, 0) + vec_get($i64, numbers, 1))
         vec_free($i64, numbers)
         0
     }
";

/// Run something that was written to disk a moment ago, waiting out the window
/// where the kernel still counts it as open for writing.
///
/// The copy closes its own handle before it returns. What keeps the file busy
/// is another test starting a process at that instant: a spawn forks first and
/// execs after, and in between the child holds every descriptor this process
/// had, the one the copy wrote through among them. Until that child reaches its
/// own exec the file has a writer, and Linux refuses to execute a file that
/// has one. Nothing here can close that window, so this waits for it rather
/// than failing a run over it.
fn run_when_no_longer_busy(command: &mut Command) -> std::process::Output {
    // ETXTBSY. Matched by number because the `ErrorKind` that names it is not
    // stable yet.
    const TEXT_FILE_BUSY: i32 = 26;
    for _ in 0..100 {
        match command.output() {
            Ok(output) => return output,
            Err(error) if error.raw_os_error() == Some(TEXT_FILE_BUSY) => {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(error) => {
                panic!("could not run what was just written: {error}")
            }
        }
    }
    panic!("what was just written stayed busy");
}

#[test]
fn both_compilers_build_and_run_from_outside_the_checkout() {
    if c_compiler().is_none() || !linker_available() {
        return;
    }
    let bootstrap = PathBuf::from(env!("CARGO_BIN_EXE_frost"));
    let self_hosted = build_self_hosted_compiler("away")
        .expect("the self-hosted compiler is required for this test");

    for (label, compiler) in
        [("bootstrap", &bootstrap), ("self-hosted", &self_hosted)]
    {
        let (installed, work) = installed_layout(label, compiler)
            .expect("could not lay out an installed compiler");
        std::fs::write(work.join("program.frost"), AWAY_FROM_THE_CHECKOUT)
            .unwrap();
        let exe = work.join(format!("program{}", std::env::consts::EXE_SUFFIX));
        let build = run_when_no_longer_busy(
            Command::new(&installed)
                .current_dir(&work)
                .arg("--link")
                .arg("-o")
                .arg(&exe)
                .arg("program.frost"),
        );
        assert!(
            build.status.success() && exe.exists(),
            "{label} could not build from {}:\n{}{}",
            work.display(),
            String::from_utf8_lossy(&build.stdout),
            String::from_utf8_lossy(&build.stderr)
        );
        let run =
            run_when_no_longer_busy(Command::new(&exe).current_dir(&work));
        assert!(
            run.status.success(),
            "{label} built a program that did not run"
        );
        assert_eq!(
            String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"),
            "42\n",
            "{label} built a program that answered wrongly"
        );
    }
}

// The binding to a C library, built by both compilers from outside the
// checkout. This is the case `--libs` exists for, and it is the one that says
// whether a compiler can build a program that talks to anything outside itself.
// The program opens a window and waits for the user to close it, so it is built
// and not run.
#[test]
fn both_compilers_link_a_c_library_from_outside_the_checkout() {
    if c_compiler().is_none() || !linker_available() {
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let library = root.join("lib").join("platform").join("SDL3.dll");
    if !library.exists() {
        return;
    }
    let bootstrap = PathBuf::from(env!("CARGO_BIN_EXE_frost"));
    let self_hosted = build_self_hosted_compiler("libs")
        .expect("the self-hosted compiler is required for this test");

    let mut outcomes = Vec::new();
    for (label, compiler) in
        [("bootstrap", &bootstrap), ("self-hosted", &self_hosted)]
    {
        let (installed, work) =
            installed_layout(&format!("libs_{label}"), compiler)
                .expect("could not lay out an installed compiler");
        // Copied under the shape the program was written against, since it
        // names the binding by where that sits relative to it. Copying the two
        // into one directory would be a different program.
        let program = work.join("examples").join("graphics");
        let binding = work.join("lib").join("platform");
        std::fs::create_dir_all(&program).unwrap();
        std::fs::create_dir_all(&binding).unwrap();
        std::fs::copy(
            graphics_source(&root, "window.frost"),
            program.join("window.frost"),
        )
        .unwrap();
        std::fs::copy(
            graphics_source(&root, "sdl.frost"),
            binding.join("sdl.frost"),
        )
        .unwrap();
        // The library has to sit beside the program it was linked against, or
        // neither build starts and the comparison below proves nothing.
        std::fs::copy(&library, work.join("SDL3.dll")).unwrap();
        let exe = work.join(format!("window{}", std::env::consts::EXE_SUFFIX));
        let build = run_when_no_longer_busy(
            Command::new(&installed)
                .current_dir(&work)
                .arg("--link")
                .arg("--libs")
                .arg(&library)
                .arg("-o")
                .arg(&exe)
                .arg("examples/graphics/window.frost"),
        );
        assert!(
            build.status.success() && exe.exists(),
            "{label} could not link against a C library from {}:\n{}{}",
            work.display(),
            String::from_utf8_lossy(&build.stdout),
            String::from_utf8_lossy(&build.stderr)
        );
        outcomes.push((label, survives_briefly(&exe, &work)));
    }

    // The program opens a window and waits, so what is asked is whether it is
    // still running a moment later rather than what it printed. A machine with
    // no display fails the same way under both compilers, and that is not this
    // test's business, so the two are compared rather than either being
    // required to succeed on its own.
    assert_eq!(
        outcomes[0].1, outcomes[1].1,
        "the two compilers disagree on whether the program survives startup"
    );
}

// Whether a program is still running a moment after it starts, and the exit
// code when it is not. A binding miscompiled at the call boundary shows up
// here as an immediate exit rather than as wrong output.
fn survives_briefly(exe: &Path, work: &Path) -> Option<i32> {
    let mut child = Command::new(exe)
        .current_dir(work)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("the built program could not start");
    for _ in 0..15 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if let Ok(Some(status)) = child.try_wait() {
            return Some(status.code().unwrap_or(-1));
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    None
}

// A C library whose functions answer with structs of the sizes the ABI rules
// turn on. Microsoft x64 answers 1, 2, 4 and 8 bytes in a register whatever the
// fields are, and takes a hidden pointer for every other size. System V answers
// up to two eightbytes in registers. Both compilers write this classification
// out separately, so the only thing that holds them together is a program that
// calls one and is run.
const C_STRUCT_RETURNS: &str = r#"#include <stdint.h>
typedef struct { uint32_t id; uint32_t generation; } Pair32;
typedef struct { uint8_t a, b, c; } Three;
typedef struct { float only; } OneFloat;
typedef struct { double x, y; } TwoDoubles;
typedef struct { int64_t a, b, c; } Big;

Pair32 make_pair(int64_t seed) {
    Pair32 made; made.id = (uint32_t)seed; made.generation = (uint32_t)(seed * 2); return made;
}
Three make_three(int64_t seed) {
    Three made; made.a = (uint8_t)seed; made.b = (uint8_t)(seed + 1); made.c = (uint8_t)(seed + 2); return made;
}
OneFloat make_float(float seed) { OneFloat made; made.only = seed * 2.0f; return made; }
TwoDoubles make_doubles(double seed) { TwoDoubles made; made.x = seed; made.y = seed * 3.0; return made; }
Big make_big(int64_t seed) { Big made; made.a = seed; made.b = seed * 2; made.c = seed * 3; return made; }
uint8_t small_u8(void) { return 7; }
int32_t neg_i32(void) { return -3; }
uint32_t big_u32(void) { return 4000000000u; }
"#;

// A callee written by hand, because what matters is the bits a C compiler is
// free to leave above a narrow answer and no C source can be made to leave them
// reliably. Both of these answer in %al with the rest of the register holding
// something else, which the ABI permits and a caller reading the whole register
// reads as true.
const DIRTY_NARROW_RETURNS: &str = "    .text
    .globl dirty_false
dirty_false:
    movabsq $0xDEADBEEF00, %rax
    ret
    .globl dirty_true
dirty_true:
    movabsq $0xDEADBEEF01, %rax
    ret
";

const CALLS_C_STRUCT_RETURNS: &str =
    "import \"io.frost\"\nPair32 :: struct { id: u32, generation: u32 }
     Three :: struct { a: u8, b: u8, c: u8 }
     OneFloat :: struct { only: f32 }
     TwoDoubles :: struct { x: f64, y: f64 }
     Big :: struct { a: i64, b: i64, c: i64 }

     make_pair    :: extern fn(seed: i64) -> Pair32
     make_three   :: extern fn(seed: i64) -> Three
     make_float   :: extern fn(seed: f32) -> OneFloat
     make_doubles :: extern fn(seed: f64) -> TwoDoubles
     make_big     :: extern fn(seed: i64) -> Big
     small_u8     :: extern fn() -> u8
     neg_i32      :: extern fn() -> i32
     big_u32      :: extern fn() -> u32
     dirty_false  :: extern fn() -> bool
     dirty_true   :: extern fn() -> bool

     main :: fn() -> i64 {
         pair := unsafe { make_pair(21) }
         print(\"{}\\n\", pair.id)
         print(\"{}\\n\", pair.generation)
         three := unsafe { make_three(10) }
         print(\"{}\\n\", three.a)
         print(\"{}\\n\", three.c)
         single := unsafe { make_float(1.5) }
         print(\"{}\\n\", single.only)
         doubles := unsafe { make_doubles(2.0) }
         print(\"{}\\n\", doubles.x)
         print(\"{}\\n\", doubles.y)
         big := unsafe { make_big(5) }
         print(\"{}\\n\", big.a)
         print(\"{}\\n\", big.c)
         print(\"{}\\n\", unsafe { small_u8() })
         print(\"{}\\n\", unsafe { neg_i32() })
         print(\"{}\\n\", unsafe { big_u32() })
         if (unsafe { dirty_false() }) { print(\"{}\\n\", 999) } else { print(\"{}\\n\", 0) }
         if (unsafe { dirty_true() }) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 999) }
         0
     }
";

#[test]
fn both_compilers_call_a_c_function_answering_with_a_struct() {
    let Some(compiler) = c_compiler() else {
        return;
    };
    if !linker_available() {
        return;
    }
    let directory = std::env::temp_dir().join(unique("frost_cret"));
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("shapes.c");
    let object = directory.join("shapes.o");
    std::fs::write(&source, C_STRUCT_RETURNS).unwrap();
    let built = Command::new(compiler)
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .output()
        .unwrap();
    assert!(built.status.success(), "the C library did not compile");

    let dirty_source = directory.join("dirty.s");
    let dirty_object = directory.join("dirty.o");
    std::fs::write(&dirty_source, DIRTY_NARROW_RETURNS).unwrap();
    let assembled = Command::new(compiler)
        .arg("-c")
        .arg(&dirty_source)
        .arg("-o")
        .arg(&dirty_object)
        .output()
        .unwrap();
    if !assembled.status.success() {
        return;
    }

    let program = directory.join("program.frost");
    std::fs::write(&program, CALLS_C_STRUCT_RETURNS).unwrap();

    let bootstrap = PathBuf::from(env!("CARGO_BIN_EXE_frost"));
    let self_hosted = build_self_hosted_compiler("cret")
        .expect("the self-hosted compiler is required for this test");

    let mut answers = Vec::new();
    for (label, frost) in
        [("bootstrap", &bootstrap), ("self-hosted", &self_hosted)]
    {
        // Through an installed layout and from a working directory that is
        // not the checkout, so nothing here resolves by accident of where the
        // test runner happens to stand.
        let (installed, work) =
            installed_layout(&format!("cret_{label}"), frost)
                .expect("could not lay out an installed compiler");
        let exe = work.join(format!("{label}{}", std::env::consts::EXE_SUFFIX));
        let build = run_when_no_longer_busy(
            Command::new(&installed)
                .current_dir(&work)
                .arg("--link")
                .arg("--libs")
                .arg(&object)
                .arg("--libs")
                .arg(&dirty_object)
                .arg("-o")
                .arg(&exe)
                .arg(&program),
        );
        assert!(
            build.status.success() && exe.exists(),
            "{label} could not build the program from {}:\n{}{}",
            work.display(),
            String::from_utf8_lossy(&build.stdout),
            String::from_utf8_lossy(&build.stderr)
        );
        let run =
            run_when_no_longer_busy(Command::new(&exe).current_dir(&work));
        assert!(
            run.status.success(),
            "{label} built a program that did not run: {:?}",
            run.status.code()
        );
        answers.push((
            label,
            String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"),
        ));
    }

    assert_eq!(
        answers[0].1,
        "21\n42\n10\n12\n3\n2\n6\n5\n15\n7\n-3\n4000000000\n0\n1\n",
        "the bootstrap answered wrongly"
    );
    assert_eq!(
        answers[0].1, answers[1].1,
        "the two compilers disagree on what a C function answered with"
    );
}

// ---------------------------------------------------------------------------
// The C return rule, written down once.
//
// Two backends classify aggregate returns separately: `src/c_abi.rs` for the
// bootstrap and `selfhosted/emit_asm.frost` for the assembly backend. They have
// drifted before, and a classification that is wrong compiles cleanly and
// produces a program that reads the wrong register, so nothing complains until
// something built against a real C library behaves oddly.
//
// This table is the rule. `classify_return` is checked against it directly
// below, and the self-hosted backend is checked against the same shapes by
// `both_compilers_call_a_c_function_answering_with_a_struct`, which builds a C
// library of these shapes and runs the result under both compilers. Adding a
// row here is the place to start when either backend learns a new shape.
//
// `None` means the answer travels through a hidden pointer. `Some(v)` means it
// comes back in `v.len()` registers, where each entry says whether that
// eightbyte is floating point.
struct ReturnCase {
    what: &'static str,
    size: usize,
    align: usize,
    scalars: &'static [(usize, Type)],
    windows: Option<&'static [bool]>,
    sysv: Option<&'static [bool]>,
}

const RETURN_RULE: &[ReturnCase] = &[
    ReturnCase {
        what: "struct { u32, u32 }, one eightbyte of integers",
        size: 8,
        align: 4,
        scalars: &[(0, Type::U32), (4, Type::U32)],
        windows: Some(&[false]),
        sysv: Some(&[false]),
    },
    ReturnCase {
        what: "struct { u32 }, half an eightbyte",
        size: 4,
        align: 4,
        scalars: &[(0, Type::U32)],
        windows: Some(&[false]),
        sysv: Some(&[false]),
    },
    ReturnCase {
        what: "struct { u8, u8, u8 }, three bytes, not a register size",
        size: 3,
        align: 1,
        scalars: &[(0, Type::U8), (1, Type::U8), (2, Type::U8)],
        // Microsoft x64 takes a pointer for any size that is not 1, 2, 4 or 8,
        // even one smaller than a register.
        windows: None,
        sysv: Some(&[false]),
    },
    ReturnCase {
        what: "struct { f32 }, where the two targets disagree",
        size: 4,
        align: 4,
        scalars: &[(0, Type::F32)],
        // Microsoft x64 answers in %rax whatever the field is. System V looks
        // at the field and answers in %xmm0. Reading this one off the wrong
        // target's rule is the classic mistake.
        windows: Some(&[false]),
        sysv: Some(&[true]),
    },
    ReturnCase {
        what: "struct { f64, f64 }, two floating eightbytes",
        size: 16,
        align: 8,
        scalars: &[(0, Type::F64), (8, Type::F64)],
        windows: None,
        sysv: Some(&[true, true]),
    },
    ReturnCase {
        what: "struct { f64, i64 }, one of each",
        size: 16,
        align: 8,
        scalars: &[(0, Type::F64), (8, Type::I64)],
        windows: None,
        sysv: Some(&[true, false]),
    },
    ReturnCase {
        what: "struct { f32, i32 }, sharing one eightbyte",
        size: 8,
        align: 4,
        scalars: &[(0, Type::F32), (4, Type::I32)],
        // An eightbyte anything integral reaches is an integer eightbyte, so
        // the float beside it travels in %rax rather than %xmm0.
        windows: Some(&[false]),
        sysv: Some(&[false]),
    },
    ReturnCase {
        what: "struct { i64, i64, i64 }, over two eightbytes",
        size: 24,
        align: 8,
        scalars: &[(0, Type::I64), (8, Type::I64), (16, Type::I64)],
        windows: None,
        sysv: None,
    },
];

fn describe(answer: &CReturn) -> Option<Vec<bool>> {
    match answer {
        CReturn::Indirect => None,
        CReturn::Registers(registers) => {
            Some(registers.iter().map(|register| register.float).collect())
        }
    }
}

#[test]
fn the_c_return_rule_is_what_the_bootstrap_implements() {
    for case in RETURN_RULE {
        let layout = CLayout {
            name: case.what.to_string(),
            size: case.size,
            align: case.align,
            scalars: case
                .scalars
                .iter()
                .map(|(offset, ty)| CScalar {
                    offset: *offset,
                    ty: ty.clone(),
                })
                .collect(),
        };
        for (target, expected) in
            [(CTarget::Windows, case.windows), (CTarget::SysV, case.sysv)]
        {
            let got = describe(&classify_return(&layout, target));
            let want = expected.map(|floats| floats.to_vec());
            assert_eq!(
                got, want,
                "{target:?} disagrees with the written rule for {}",
                case.what
            );
        }
    }
}

fn binutils_available() -> bool {
    ["as", "objcopy", "objdump"]
        .iter()
        .all(|tool| Command::new(tool).arg("--version").output().is_ok())
}

fn section_bytes(object: &Path, section: &str, into: &Path) -> Vec<u8> {
    let extract = Command::new("objcopy")
        .arg("-O")
        .arg("binary")
        .arg(format!("--only-section={section}"))
        .arg(object)
        .arg(into)
        .output()
        .unwrap();
    assert!(
        extract.status.success(),
        "objcopy could not read {section} out of {}:\n{}",
        object.display(),
        String::from_utf8_lossy(&extract.stderr)
    );
    std::fs::read(into).unwrap_or_default()
}

fn relocation_lines(object: &Path) -> Vec<String> {
    let listing = Command::new("objdump")
        .arg("-r")
        .arg(object)
        .output()
        .unwrap();
    assert!(listing.status.success());
    let mut lines: Vec<String> = String::from_utf8_lossy(&listing.stdout)
        .lines()
        .filter(|line| line.contains("IMAGE_REL") || line.contains("R_X86_64"))
        .map(|line| line.trim().to_string())
        .collect();
    lines.sort();
    lines
}

// The rows of a line table, as `line address`, read back out of an object with
// whatever wrote it. Only the decoded table is compared: a line program can be
// spelled several ways and two spellings that decode alike are both right.
fn decoded_line_rows(object: &Path) -> Vec<String> {
    let listing = Command::new("objdump")
        .arg("--dwarf=decodedline")
        .arg(object)
        .output()
        .unwrap();
    assert!(
        listing.status.success(),
        "objdump could not read the line table:\n{}",
        String::from_utf8_lossy(&listing.stderr)
    );
    String::from_utf8_lossy(&listing.stdout)
        .lines()
        .filter_map(|line| {
            // A row reads `<file> <line> <address>` and may carry a view
            // number and a statement mark after it, so the address is found
            // by its own shape rather than by counting from either end.
            let fields: Vec<&str> = line.split_whitespace().collect();
            let at = fields.iter().position(|f| f.starts_with("0x"))?;
            if at == 0 {
                return None;
            }
            let number = fields[at - 1];
            // The last row of a sequence has no line, only the address it
            // stops at, and that address differs by format: a COFF section is
            // padded to its alignment and an ELF one is not.
            if number == "-" || number.parse::<u32>().is_err() {
                return None;
            }
            Some(format!("{number} {}", fields[at]))
        })
        .collect()
}

// The line table this compiler writes into an object itself, against the one
// the system assembler writes from the same text. The encoder here has to
// agree with `as` about which address every source line begins at, and that is
// the whole of what a debugger reads.
#[test]
fn the_assembler_writes_the_line_table_the_system_assembler_writes() {
    if Command::new("objdump").arg("--version").output().is_err()
        || Command::new("as").arg("--version").output().is_err()
    {
        return;
    }
    let Some(compiler) = build_self_hosted_compiler("dwarf") else {
        return;
    };
    let directory = std::env::temp_dir();
    let source = directory
        .join(unique("frost_dwarf"))
        .with_extension("frost");
    std::fs::write(
        &source,
        "import \"io.frost\"\nadd :: fn(a: i64, b: i64) -> i64 {\n\
         \x20   total := a + b\n\
         \x20   total\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   var sum : i64 = 0\n\
         \x20   sum = add(sum, 3)\n\
         \x20   sum = add(sum, 4)\n\
         \x20   while (sum < 40) {\n\
         \x20       sum = add(sum, 5)\n\
         \x20   }\n\
         \x20   print(\"{}\\n\", sum)\n\
         \x20   0\n\
         }\n",
    )
    .unwrap();

    let assembly = directory.join(unique("frost_dwarf")).with_extension("s");
    let emitted = Command::new(&compiler)
        .arg("--emit-asm")
        .arg("-g")
        .arg("-o")
        .arg(&assembly)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        emitted.status.success(),
        "the self-hosted compiler could not emit assembly with -g:\n{}",
        String::from_utf8_lossy(&emitted.stderr)
    );

    let reference =
        directory.join(unique("frost_dwarf_as")).with_extension("o");
    let system = Command::new("as")
        .arg("-o")
        .arg(&reference)
        .arg(&assembly)
        .output()
        .unwrap();
    assert!(
        system.status.success(),
        "`as` rejected the emitted text:\n{}",
        String::from_utf8_lossy(&system.stderr)
    );

    let ours = directory
        .join(unique("frost_dwarf_ours"))
        .with_extension("o");
    let encoded = Command::new(&compiler)
        .arg("--native")
        .arg("-g")
        .arg("-o")
        .arg(&ours)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        encoded.status.success(),
        "this compiler's assembler failed with -g:\n{}",
        String::from_utf8_lossy(&encoded.stderr)
    );

    let want = decoded_line_rows(&reference);
    let got = decoded_line_rows(&ours);
    assert!(
        !want.is_empty(),
        "`as` wrote no line table, so nothing was compared"
    );
    assert_eq!(
        got, want,
        "the line table disagrees with the one `as` wrote"
    );

    let _ = std::fs::remove_file(&source);
    let _ = std::fs::remove_file(&assembly);
    let _ = std::fs::remove_file(&reference);
    let _ = std::fs::remove_file(&ours);
}

fn std_source(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("std")
        .join(name)
}

// A section's bytes out of an ELF object. `objcopy` from binutils reads COFF
// here, so the LLVM one is what reads the other format.
fn elf_section_bytes(object: &Path, section: &str, into: &Path) -> Vec<u8> {
    let extract = Command::new("llvm-objcopy")
        .arg("-O")
        .arg("binary")
        .arg(format!("--only-section={section}"))
        .arg(object)
        .arg(into)
        .output()
        .unwrap();
    assert!(
        extract.status.success(),
        "could not read {section}:\n{}",
        String::from_utf8_lossy(&extract.stderr)
    );
    let bytes = std::fs::read(into).unwrap_or_default();
    let _ = std::fs::remove_file(into);
    bytes
}

fn clang_available() -> bool {
    Command::new("clang").arg("--version").output().is_ok()
        && Command::new("llvm-objcopy")
            .arg("--version")
            .output()
            .is_ok()
        && Command::new("readelf").arg("--version").output().is_ok()
}

// The relocations an object carries, as name, kind and addend against the place
// each fixes up, which is the whole of what a linker acts on.
fn elf_relocations(object: &Path) -> Vec<String> {
    let listing = Command::new("readelf")
        .arg("-r")
        .arg(object)
        .output()
        .unwrap();
    assert!(listing.status.success());
    let mut lines: Vec<String> = String::from_utf8_lossy(&listing.stdout)
        .lines()
        .filter(|line| line.contains("R_X86_64"))
        .map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            format!("{} {} {}", fields[0], fields[2], fields[4..].join(" "))
        })
        .collect();
    lines.sort();
    lines
}

// The assembler's ELF half against clang, byte for byte, the way its COFF half
// is held to `as`. Both read the same text, so a disagreement is this
// compiler's object being wrong.
//
// One substitution is needed first. The backend writes a frame size as a symbol
// a `.set` gives a value to further down the file, because the size is not
// known until the body has been emitted. `as` sizes that immediate without the
// value and takes the wide form, which is what this assembler does; clang makes
// a further pass and takes the short one. Both are correct and the two texts
// are then different lengths, so the test resolves those symbols itself and
// hands both assemblers a file with no forward reference left in it. What that
// costs is coverage of one instruction form, and
// `the_assembler_encodes_what_the_system_assembler_does` covers it against `as`.
#[test]
fn the_assembler_writes_the_elf_object_clang_writes() {
    if !clang_available() {
        return;
    }
    let Some(compiler) = build_self_hosted_compiler("elf") else {
        return;
    };
    let directory = std::env::temp_dir();
    let sources: &[(&str, PathBuf)] = &[
        ("the compiler itself", self_hosted_source()),
        ("the entity store", std_source("ecs.frost")),
        ("the sort library", std_source("sort.frost")),
        ("the maths library", std_source("math.frost")),
    ];

    for (what, source) in sources {
        let assembly = directory.join(unique("frost_elf")).with_extension("s");
        let emitted = Command::new(&compiler)
            .arg("--emit-asm")
            .arg("-o")
            .arg(&assembly)
            .arg(source)
            .output()
            .unwrap();
        assert!(
            emitted.status.success(),
            "the self-hosted compiler could not emit assembly for {what}:\n{}",
            String::from_utf8_lossy(&emitted.stderr)
        );

        let text = std::fs::read_to_string(&assembly).unwrap();
        let mut sizes: Vec<(String, String)> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix(".set ")
                && let Some((name, value)) = rest.split_once(", ")
            {
                sizes.push((format!("${name}"), format!("${value}")));
            }
        }
        let mut flattened = text.clone();
        for (name, value) in &sizes {
            flattened =
                flattened.replace(&format!("{name},"), &format!("{value},"));
        }
        let flat = directory.join(unique("frost_elf_flat")).with_extension("s");
        std::fs::write(&flat, &flattened).unwrap();

        let reference =
            directory.join(unique("frost_elf_gold")).with_extension("o");
        let system = Command::new("clang")
            .arg("-target")
            .arg("x86_64-unknown-linux-gnu")
            .arg("-c")
            .arg(&flat)
            .arg("-o")
            .arg(&reference)
            .output()
            .unwrap();
        assert!(
            system.status.success(),
            "clang rejected the emitted text for {what}:\n{}",
            String::from_utf8_lossy(&system.stderr)
        );

        let ours = directory.join(unique("frost_elf_ours")).with_extension("o");
        let encoded = Command::new(&compiler)
            .env("FROST_OBJECT", "elf")
            .arg("--assemble")
            .arg("-o")
            .arg(&ours)
            .arg(&flat)
            .output()
            .unwrap();
        assert!(
            encoded.status.success(),
            "this compiler's assembler failed on {what}:\n{}",
            String::from_utf8_lossy(&encoded.stderr)
        );

        for section in [".text", ".data"] {
            let want = elf_section_bytes(
                &reference,
                section,
                &directory.join(unique("elf_gold_section")),
            );
            let got = elf_section_bytes(
                &ours,
                section,
                &directory.join(unique("elf_ours_section")),
            );
            assert_eq!(
                want.len(),
                got.len(),
                "{section} is a different length for {what}"
            );
            if let Some(at) = want
                .iter()
                .zip(&got)
                .position(|(left, right)| left != right)
            {
                let from = at.saturating_sub(8);
                let to = (at + 8).min(want.len());
                panic!(
                    "{section} differs for {what} at byte {at}\n  clang: {:02x?}\n  ours:  {:02x?}",
                    &want[from..to],
                    &got[from..to]
                );
            }
        }

        assert_eq!(
            elf_relocations(&reference),
            elf_relocations(&ours),
            "the fixups differ for {what}"
        );

        let _ = std::fs::remove_file(&assembly);
        let _ = std::fs::remove_file(&flat);
        let _ = std::fs::remove_file(&reference);
        let _ = std::fs::remove_file(&ours);
    }
}

// The two formats against each other on the same input, which is what says the
// ELF encoding is the COFF one with the places a linker fills in left for it.
// COFF is held to `as` byte for byte, so this carries that over rather than
// checking the same bytes twice.
#[test]
fn the_two_object_formats_differ_only_where_a_linker_fills_in() {
    if !clang_available() || !binutils_available() {
        return;
    }
    let Some(compiler) = build_self_hosted_compiler("formats") else {
        return;
    };
    let directory = std::env::temp_dir();
    let assembly = directory.join(unique("frost_formats")).with_extension("s");
    assert!(
        Command::new(&compiler)
            .arg("--emit-asm")
            .arg("-o")
            .arg(&assembly)
            .arg(std_source("ecs.frost"))
            .output()
            .unwrap()
            .status
            .success()
    );

    let mut objects = Vec::new();
    for format in ["coff", "elf"] {
        let object = directory
            .join(unique(&format!("frost_formats_{format}")))
            .with_extension("o");
        assert!(
            Command::new(&compiler)
                .env("FROST_OBJECT", format)
                .arg("--assemble")
                .arg("-o")
                .arg(&object)
                .arg(&assembly)
                .output()
                .unwrap()
                .status
                .success(),
            "the assembler could not write a {format} object"
        );
        objects.push(object);
    }

    let coff = section_bytes(
        &objects[0],
        ".text",
        &directory.join(unique("coff_text")),
    );
    let elf = elf_section_bytes(
        &objects[1],
        ".text",
        &directory.join(unique("elf_text")),
    );
    // COFF pads a section up to its alignment and ELF states the alignment in
    // its header instead, so the ELF one is the shorter of the two.
    assert!(
        coff.len() >= elf.len(),
        "the ELF text is longer than the COFF one"
    );

    let mut filled = vec![false; elf.len()];
    for line in elf_relocations(&objects[1]) {
        let at =
            usize::from_str_radix(line.split_whitespace().next().unwrap(), 16)
                .unwrap();
        filled[at..at + 4].fill(true);
    }

    let mut fixups = 0;
    for at in 0..elf.len() {
        if filled[at] {
            fixups += 1;
            assert_eq!(
                elf[at], 0,
                "a place the linker fills in was not left empty, at byte {at}"
            );
        } else {
            assert_eq!(
                coff[at], elf[at],
                "the two formats encoded byte {at} differently"
            );
        }
    }
    assert!(
        fixups > 1000,
        "only {fixups} bytes were left for the linker"
    );

    let _ = std::fs::remove_file(&assembly);
    for object in &objects {
        let _ = std::fs::remove_file(object);
    }
}

// Decimal literals covering the whole range a double reaches: the ties that sit
// exactly between two of them, the exact decimal of a double, the numbers too
// small to be normal, and a spread of ordinary ones.
fn float_literal_cases() -> Vec<String> {
    let mut literals: Vec<String> = Vec::new();
    // Exactly between two doubles, where the answer is whichever of them has a
    // zero last bit. The last two go opposite ways.
    for tie in [
        "9007199254740993",
        "9007199254740995.0",
        "1.00000000000000011102230246251565404236316680908203125",
        "1.00000000000000033306690738754696212708950042724609375",
        "0.100000000000000012490009027033011079765856266021728515625",
    ] {
        literals.push(tie.to_string());
    }
    // The exact decimal of a double, which has to come back as the double it
    // was written from. The smallest one is 751 digits past the point.
    for value in [
        0.0f64,
        1.0,
        0.5,
        0.1,
        std::f64::consts::PI,
        0.017453292519943295,
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::from_bits(1),
        f64::from_bits(2),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::from_bits(0x0010_0000_0000_0001),
        123_456_789.123_456_79,
    ] {
        literals.push(format!("{value:.1100}"));
    }
    // And a spread of ordinary ones, as digits with the point moved through
    // them, which is the shape a program actually holds.
    let mut seed: u64 = 0x2026_0727_0000_0001;
    for _ in 0..600 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let length = 1 + (seed >> 33) % 25;
        let mut digits = String::new();
        for _ in 0..length {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            digits.push((b'0' + ((seed >> 33) % 10) as u8) as char);
        }
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let point = ((seed >> 33) % (length + 6)) as usize;
        literals.push(if point >= digits.len() {
            format!("0.{}{digits}", "0".repeat(point - digits.len()))
        } else {
            let at = digits.len() - point;
            format!("{}.{}", &digits[..at], &digits[at..])
        });
    }
    literals
}

// A decimal literal names an exact number and a double holds only some numbers,
// so writing one as the other is a rounding, and there is exactly one right
// answer: the nearest double, with a tie going to the one whose last bit is
// zero. This holds the assembler to it against a reader known to round
// correctly, rather than against `as`, which rounds a tie away from zero and so
// answers 9007199254740994 where the right answer is 9007199254740992.
//
// Nothing else in the suite reaches this. The backend used to hand every float
// literal to the system assembler as text, so a build only met these numbers
// once this compiler started encoding its own output.
#[test]
fn the_assembler_rounds_a_float_literal_to_the_nearest_double() {
    if !cfg!(windows) || !binutils_available() {
        return;
    }
    let Some(compiler) = build_self_hosted_compiler("float_literals") else {
        return;
    };
    let literals = float_literal_cases();
    let directory = std::env::temp_dir();
    let assembly = directory.join(unique("frost_floats")).with_extension("s");
    let mut text = String::from("    .data\n");
    for literal in &literals {
        text.push_str("    .double ");
        text.push_str(literal);
        text.push('\n');
    }
    std::fs::write(&assembly, &text).unwrap();

    let object = directory.join(unique("frost_floats")).with_extension("o");
    let encoded = Command::new(&compiler)
        .arg("--assemble")
        .arg("-o")
        .arg(&object)
        .arg(&assembly)
        .output()
        .unwrap();
    assert!(
        encoded.status.success(),
        "the assembler refused the float literals:\n{}",
        String::from_utf8_lossy(&encoded.stderr)
    );

    let got = section_bytes(
        &object,
        ".data",
        &directory.join(unique("floats_section")),
    );
    for (index, literal) in literals.iter().enumerate() {
        let want = literal.parse::<f64>().unwrap().to_bits();
        let mut eight = [0u8; 8];
        eight.copy_from_slice(&got[index * 8..index * 8 + 8]);
        let mine = u64::from_le_bytes(eight);
        assert_eq!(
            want,
            mine,
            "{} rounded to {mine:#018x} rather than {want:#018x}",
            if literal.len() > 60 {
                format!("{}...", &literal[..57])
            } else {
                literal.clone()
            }
        );
    }

    let _ = std::fs::remove_file(&assembly);
    let _ = std::fs::remove_file(&object);
}

// The assembler in `selfhosted/assemble.frost` against the system assembler,
// byte for byte. Both read the same text, so any disagreement is this
// compiler's encoding being wrong, and the object it would have written would
// have been wrong in a way no test of the language's behaviour would catch.
//
// The compiler's own source is one of the inputs because it is the widest
// coverage available: 134,000 instructions over every form the backend emits
// for integer code. The rest reach the forms it does not use itself.
//
// One disagreement is sanctioned and is not a bug here. `as` rounds a float
// literal that sits exactly between two doubles away from zero, where the
// answer is the one whose last bit is zero, so it writes 9007199254740994 for
// 9007199254740993. If `.data` ever differs by eight bytes and the two readings
// are neighbouring doubles, this is why, and
// `the_assembler_rounds_a_float_literal_to_the_nearest_double` is the test that
// says which of them is right.
//
// Windows only, and what that leaves uncovered is worth knowing: the backend
// emits a different instruction set per calling convention, so an encoding used
// only by System V is never weighed here. `movb $imm, %reg`, which says how
// many vector registers a variadic call uses, was wrong for exactly that
// reason. `the_assembler_writes_the_elf_object_clang_writes` is the other half
// and needs clang.
#[test]
fn the_assembler_encodes_what_the_system_assembler_does() {
    if !cfg!(windows) || !binutils_available() {
        return;
    }
    let Some(compiler) = build_self_hosted_compiler("oracle") else {
        return;
    };
    let directory = std::env::temp_dir();
    let sources: &[(&str, PathBuf)] = &[
        ("the compiler itself", self_hosted_source()),
        (
            "the entity store",
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("std")
                .join("ecs.frost"),
        ),
        (
            "the sort library",
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("std")
                .join("sort.frost"),
        ),
    ];

    let mut floats = directory.join(unique("frost_oracle_floats"));
    floats.set_extension("frost");
    std::fs::write(
        &floats,
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   var x : f64 = 1.5\n\
         \x20   var y : f64 = 0.25\n\
         \x20   print(\"{}\\n\", x + y)\n    print(\"{}\\n\", x - y)\n    print(\"{}\\n\", x * y)\n    print(\"{}\\n\", x / y)\n\
         \x20   if (x > y) { print(\"{}\\n\", 1) }\n\
         \x20   var n : i64 = 3\n    z := x * 2.0\n    print(\"{}\\n\", z)\n    print(\"{}\\n\", n)\n    0\n}\n",
    )
    .unwrap();

    let mut cases: Vec<(String, PathBuf)> = sources
        .iter()
        .map(|(what, path)| ((*what).to_string(), path.clone()))
        .collect();
    cases.push(("double-precision arithmetic".to_string(), floats));

    for (what, source) in &cases {
        let assembly =
            directory.join(unique("frost_oracle")).with_extension("s");
        let emitted = Command::new(&compiler)
            .arg("--emit-asm")
            .arg("-o")
            .arg(&assembly)
            .arg(source)
            .output()
            .unwrap();
        assert!(
            emitted.status.success(),
            "the self-hosted compiler could not emit assembly for {what}:\n{}",
            String::from_utf8_lossy(&emitted.stderr)
        );

        let reference = directory
            .join(unique("frost_oracle_gold"))
            .with_extension("o");
        let system = Command::new("as")
            .arg("-o")
            .arg(&reference)
            .arg(&assembly)
            .output()
            .unwrap();
        assert!(
            system.status.success(),
            "the system assembler rejected the emitted text for {what}:\n{}",
            String::from_utf8_lossy(&system.stderr)
        );

        let ours = directory
            .join(unique("frost_oracle_ours"))
            .with_extension("o");
        let encoded = Command::new(&compiler)
            .arg("--assemble")
            .arg("-o")
            .arg(&ours)
            .arg(&assembly)
            .output()
            .unwrap();
        assert!(
            encoded.status.success(),
            "this compiler's assembler failed on {what}:\n{}",
            String::from_utf8_lossy(&encoded.stderr)
        );

        // The same source again through `--native`, which is the path a build
        // takes: the assembly is never written down, so the encoder reads it
        // out of memory and the room it works in is grown rather than read off
        // a file's length. Checking only `--assemble` would leave the path
        // everyone actually uses untested, and the two differ in exactly the
        // place a size is decided.
        let straight = directory
            .join(unique("frost_oracle_native"))
            .with_extension("o");
        let native = Command::new(&compiler)
            .arg("--native")
            .arg("-o")
            .arg(&straight)
            .arg(source)
            .output()
            .unwrap();
        assert!(
            native.status.success(),
            "this compiler could not compile {what} straight to an object:\n{}",
            String::from_utf8_lossy(&native.stderr)
        );

        for (theirs, how) in [
            (&ours, "reading the text back"),
            (&straight, "straight from memory"),
        ] {
            for section in [".text", ".data"] {
                let want = section_bytes(
                    &reference,
                    section,
                    &directory.join(unique("gold_section")),
                );
                let got = section_bytes(
                    theirs,
                    section,
                    &directory.join(unique("ours_section")),
                );
                assert_eq!(
                    want.len(),
                    got.len(),
                    "{section} is a different length for {what}, {how}"
                );
                if let Some(at) = want
                    .iter()
                    .zip(&got)
                    .position(|(left, right)| left != right)
                {
                    let from = at.saturating_sub(8);
                    let to = (at + 8).min(want.len());
                    panic!(
                        "{section} differs for {what} at byte {at}, {how}\n  system: {:02x?}\n  ours:   {:02x?}",
                        &want[from..to],
                        &got[from..to]
                    );
                }
            }

            assert_eq!(
                relocation_lines(&reference),
                relocation_lines(theirs),
                "the fixups differ for {what}, {how}"
            );
        }

        let _ = std::fs::remove_file(&assembly);
        let _ = std::fs::remove_file(&reference);
        let _ = std::fs::remove_file(&ours);
        let _ = std::fs::remove_file(&straight);
    }
}

// A file's bytes through `include_str`, spliced by both compilers and read
// back through every backend. The data file carries a quote, a backslash and
// CRLF line endings, which are the three things the two compilers store
// differently: the bootstrap holds a literal resolved and the self-hosted
// holds it in source form, so a byte mishandled on either side shows up as
// the two disagreeing.
#[test]
fn include_str_reads_the_same_bytes_through_both_compilers() {
    let directory = std::env::temp_dir();
    let data = directory.join("frost_include_data.txt");
    std::fs::write(
        &data,
        "line one\r\nsays \"quoted\" and back\\slash\r\nend\r\n",
    )
    .unwrap();
    let source = "import \"io.frost\"\nDATA :: include_str(\"frost_include_data.txt\")\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", DATA)\n\
         \x20   print(\"{}\\n\", str_len(DATA))\n\
         \x20   0\n}\n";
    let want = "line one\nsays \"quoted\" and back\\slash\nend\n\n42\n";
    let Some(bootstrap) = bootstrap_output("includestr", source) else {
        return;
    };
    assert_eq!(bootstrap, want, "the bootstrap read the bytes differently");
    let Some(compiler) = build_self_hosted_compiler("includestr") else {
        return;
    };
    for (backend, suffix) in [("--emit-asm", "s"), ("--emit-c", "c")] {
        let hosted = selfhosted_default_output(
            &compiler,
            "includestr",
            source,
            backend,
            suffix,
        );
        assert_eq!(
            hosted, want,
            "the self-hosted compiler's {backend} read the bytes differently"
        );
    }
    let _ = std::fs::remove_file(&data);
}

// The gate runs on every compilation and nothing turns it off, so a program
// with an unguarded extern call is refused. This is what holds that: the
// compiler is invoked the way anyone invokes it, with no environment set.
#[test]
fn the_default_configuration_audits_unsafe_operations() {
    if c_compiler().is_none() || !linker_available() {
        return;
    }
    let directory = std::env::temp_dir();
    let input = directory
        .join(unique("frost_audit_default"))
        .with_extension("frost");
    std::fs::write(
        &input,
        "printf :: extern fn(format: ^i8, value: i64) -> i32\n\
         main :: fn() -> i64 {\n    printf(\"%lld\n\", 7)\n    0\n}\n",
    )
    .unwrap();
    let exe = directory
        .join(unique("frost_audit_default"))
        .with_extension(std::env::consts::EXE_EXTENSION);

    let audited = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("-o")
        .arg(&exe)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        !audited.status.success(),
        "an unguarded extern call compiled with nothing turning the audit off"
    );
    let complaint = String::from_utf8_lossy(&audited.stderr);
    assert!(
        complaint.contains("unsafe"),
        "the refusal did not name the reason:\n{complaint}"
    );

    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&exe);
}

// Both compilers render a file the same way, over the whole corpus.
//
// Run in one process against one build of each compiler rather than as a shell
// loop, so a rule changed in one and not the other is named in seconds and the
// file it differs on is printed with the two lines beside each other.
#[test]
fn both_compilers_format_the_corpus_the_same_way() {
    let Some(compiler) = build_self_hosted_compiler("fmtparity") else {
        return;
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for directory in ["std", "lib", "selfhosted", "examples", "tools"] {
        files.extend(frost_sources(&root.join(directory)));
    }
    assert!(files.len() > 100, "the corpus should be the whole tree");

    let work = std::env::temp_dir().join(unique("frost_fmt_parity"));
    std::fs::create_dir_all(&work).unwrap();
    let mine = work.join("bootstrap.frost");
    let theirs = work.join("selfhosted.frost");
    let mut differ = Vec::new();
    for file in &files {
        let Ok(source) = std::fs::read_to_string(file) else {
            // A file the walk saw and something else removed before it was
            // read. Nothing to compare.
            continue;
        };
        std::fs::write(&mine, &source).unwrap();
        std::fs::write(&theirs, &source).unwrap();
        let ran = Command::new(env!("CARGO_BIN_EXE_frost"))
            .arg("fmt")
            .arg(&mine)
            .output()
            .unwrap();
        assert!(
            ran.status.success(),
            "the bootstrap could not format {}",
            file.display()
        );
        let ran = Command::new(&compiler)
            .arg("fmt")
            .arg(&theirs)
            .output()
            .unwrap();
        assert!(
            ran.status.success(),
            "the self-hosted compiler could not format {}",
            file.display()
        );
        let left = std::fs::read_to_string(&mine).unwrap();
        let right = std::fs::read_to_string(&theirs).unwrap();
        if left != right {
            let at = left
                .lines()
                .zip(right.lines())
                .position(|(one, other)| one != other);
            let shown = match at {
                Some(at) => format!(
                    "line {}:\n  bootstrap:   {:?}\n  self-hosted: {:?}",
                    at + 1,
                    left.lines().nth(at).unwrap_or(""),
                    right.lines().nth(at).unwrap_or("")
                ),
                None => "one is longer than the other".to_string(),
            };
            differ.push(format!("{}\n{shown}", file.display()));
        }
    }
    let _ = std::fs::remove_dir_all(&work);
    assert!(
        differ.is_empty(),
        "{} of {} files render differently:\n{}",
        differ.len(),
        files.len(),
        differ.join("\n\n")
    );
}

// Every `.frost` file under a directory.
fn frost_sources(directory: &std::path::Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(next) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&next) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|kind| kind == "frost") {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

// Both compilers write the same reports as JSON: the same schema, the same
// places, the same words.
#[test]
fn both_compilers_write_the_same_json_reports() {
    let Some(compiler) = build_self_hosted_compiler("jsonparity") else {
        return;
    };
    let directory = std::env::temp_dir().join(unique("frost_json_parity"));
    std::fs::create_dir_all(&directory).unwrap();
    let file = directory.join("names.frost");
    std::fs::write(
        &file,
        "one :: fn() -> i64 {
    bogus
}
         three :: fn() -> i64 { 3 }
         two :: fn() -> i64 {
    third
}
         main :: fn() -> i64 { one() + two() }
",
    )
    .unwrap();

    let records = |text: &str| -> Vec<(u64, u64, String)> {
        let mut found: Vec<(u64, u64, String)> = text
            .lines()
            .filter(|line| line.starts_with('{'))
            .filter_map(|line| {
                serde_json::from_str::<serde_json::Value>(line).ok()
            })
            .map(|held| {
                (
                    held["line"].as_u64().unwrap_or(0),
                    held["column"].as_u64().unwrap_or(0),
                    held["message"].as_str().unwrap_or("").to_string(),
                )
            })
            .collect();
        found.sort();
        found
    };

    let ran = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--diagnostics=json")
        .arg("--native")
        .arg("-o")
        .arg(directory.join("out.o"))
        .arg(&file)
        .output()
        .unwrap();
    let mine = records(&String::from_utf8_lossy(&ran.stderr));
    let ran = Command::new(&compiler)
        .env("FROST_INPUT", &file)
        .arg("--diagnostics=json")
        .arg("-o")
        .arg(directory.join("out.c"))
        .output()
        .unwrap();
    let theirs = records(&String::from_utf8_lossy(&ran.stderr));

    assert_eq!(mine.len(), 2, "two names are not declared: {mine:?}");
    assert!(
        mine.iter()
            .any(|(_, _, message)| message.contains("did you mean 'three'")),
        "the nearest name is offered: {mine:?}"
    );
    assert_eq!(mine, theirs, "the two compilers write different reports");
    let _ = std::fs::remove_dir_all(&directory);
}

// Both compilers find the same things worth a look, in the same words. The
// order follows which walk found each, which each compiler runs in its own
// order, so the findings are compared as a set the way the fault lines are.
#[test]
fn both_compilers_lint_the_same_way() {
    let Some(compiler) = build_self_hosted_compiler("lintparity") else {
        return;
    };
    let directory = std::env::temp_dir().join(unique("frost_lint_parity"));
    std::fs::create_dir_all(&directory).unwrap();
    let file = directory.join("findings.frost");
    std::fs::write(
        &file,
        "unused :: fn() -> i64 { 1 }\n\
         reached :: fn() -> i64 { 2 }\n\
         main :: fn() -> i64 { unsafe { reached() } }\n",
    )
    .unwrap();

    let said = |output: std::process::Output| -> Vec<String> {
        let text = String::from_utf8_lossy(&output.stdout).to_string()
            + &String::from_utf8_lossy(&output.stderr);
        let mut found: Vec<String> = text
            .lines()
            .filter_map(|line| line.split_once("^ "))
            .map(|(_, message)| message.trim().to_string())
            .collect();
        found.sort();
        found
    };

    let mine = said(
        Command::new(env!("CARGO_BIN_EXE_frost"))
            .arg("lint")
            .arg(&file)
            .output()
            .unwrap(),
    );
    let theirs = said(
        Command::new(&compiler)
            .arg("lint")
            .arg(&file)
            .output()
            .unwrap(),
    );
    assert_eq!(
        mine.len(),
        2,
        "the file has an unreached function and an idle block: {mine:?}"
    );
    assert_eq!(mine, theirs, "the two compilers report different findings");
    let _ = std::fs::remove_dir_all(&directory);
}

// The line-boundary grammar, held against every operator that could carry an
// expression across a break.
//
// The property: the parse of a line never depends on the token that opens the
// next one. Outside brackets a line is a statement and nothing after it can
// change that; inside brackets a line break says nothing at all. So no single
// token added, dropped or altered at a boundary can silently change how many
// statements the surrounding text parses as. It preserves meaning or it fails.
const JOINING_OPERATORS: &[&str] = &[
    "+", "*", "/", "%", "&&", "||", "|", "&", "==", "!=", "<", "<=", ">", ">=",
    "<<", ">>",
];

#[test]
fn no_operator_carries_an_expression_across_a_line_outside_brackets() {
    for operator in JOINING_OPERATORS {
        // Added leading token: refused, and in the same words by both.
        let source = format!(
            "main :: fn() -> i64 {{\n    x := 10\n        {operator} 3\n    x\n}}\n"
        );
        let said =
            bootstrap_refusal(&format!("join{}", operator.len()), &source);
        assert!(
            said.contains("a line cannot open with"),
            "'{operator}' opening a line was accepted:\n{said}"
        );
    }
}

// Inside brackets the same expression is one expression, and dropping the
// operator that joins it fails to parse rather than answering differently.
#[test]
fn a_dropped_operator_inside_brackets_does_not_parse() {
    let joined = "import \"io.frost\"\nmain :: fn() -> i64 {\n\
                  \x20   x := (10\n        + 3)\n\
                  \x20   print(\"{}\\n\", x)\n    0\n}\n";
    let dropped = "import \"io.frost\"\nmain :: fn() -> i64 {\n\
                   \x20   x := (10\n        3)\n\
                   \x20   print(\"{}\\n\", x)\n    0\n}\n";
    let directory = std::env::temp_dir().join(unique("frost_join"));
    std::fs::create_dir_all(&directory).unwrap();
    let file = directory.join("joined.frost");
    std::fs::write(&file, joined).unwrap();
    let built = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--native")
        .arg("-o")
        .arg(directory.join("joined.o"))
        .arg(&file)
        .output()
        .unwrap();
    assert!(built.status.success(), "the joined form should compile");
    let said = bootstrap_refusal("joindropped", dropped);
    assert!(
        said.contains("expected") || said.contains("cannot"),
        "dropping the operator inside brackets should not parse:\n{said}"
    );
    let _ = std::fs::remove_dir_all(&directory);
}

// The same programs through the other two backends, so the rule is the
// language's and not one backend's reading of it.
#[test]
fn the_line_rule_is_the_same_through_every_backend() {
    let source = "import \"io.frost\"\nmain :: fn() -> i64 {\n\
                  \x20   x := (10\n        + 3\n        + 7)\n\
                  \x20   print(\"{}\\n\", x)\n    0\n}\n";
    let directory = std::env::temp_dir().join(unique("frost_backends"));
    std::fs::create_dir_all(&directory).unwrap();
    let file = directory.join("wrapped.frost");
    std::fs::write(&file, source).unwrap();

    let ran = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--run-ir")
        .arg(&file)
        .output()
        .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n"),
        "20\n",
        "the IR interpreter"
    );
    if linker_available() {
        for backend in [vec![], vec!["--emit-c"]] {
            let exe = directory.join(format!(
                "wrapped{}{}",
                backend.len(),
                std::env::consts::EXE_SUFFIX
            ));
            let built = Command::new(env!("CARGO_BIN_EXE_frost"))
                .args(&backend)
                .arg("--link")
                .arg("-o")
                .arg(&exe)
                .arg(&file)
                .output()
                .unwrap();
            assert!(built.status.success(), "{backend:?} did not build");
            let ran = Command::new(&exe).output().unwrap();
            assert_eq!(
                String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n"),
                "20\n",
                "{backend:?}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&directory);
}

// A report names a type the way a reader writes one. `proc(..)` and `&mut T`
// are the names the type table files a function type and a borrow under, and
// they round-trip through `type_from_string` for monomorphization, so the table
// keeps them. A reader writes `fn(..)` and `mut T`, and a report that says
// otherwise is describing syntax the surface dropped.
//
// The bootstrap alone, because the self-hosted compiler does not reach this
// message: two function values of different signatures are compatible there,
// which is its own defect and a wider one than the spelling.
#[test]
fn a_report_spells_a_function_type_the_way_it_is_written() {
    let source = "Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
                  bump :: fn($N: usize, mut a: Arena<N>) -> i64 {
                      a.offset = a.offset + 8
                      a.offset
}
                  plain :: fn(n: i64, mut a: Arena<256>) -> i64 {
                      bump($256, a) + n
}
                  call_it :: fn(f: fn(i64) -> i64, v: i64) -> i64 { f(v) }
                  main :: fn() -> i64 { call_it(plain, 3) }
";
    let report = bootstrap_refusal("spelledfn", source);
    assert!(
        report.contains("fn(i64, mut Arena<256>) -> i64"),
        "the report did not spell the signature the way it is written:
{report}"
    );
    assert!(
        report.contains("fn(i64) -> i64"),
        "the report did not spell the wanted signature:
{report}"
    );
    assert!(
        !report.contains("proc("),
        "the report named a function type as `proc`:
{report}"
    );
    assert!(
        !report.contains("&mut "),
        "the report named a borrow as `&mut`:
{report}"
    );
}
