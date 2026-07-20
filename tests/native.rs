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

    let run = Command::new(&exe_path).output().unwrap();
    assert!(
        run.status.success(),
        "native binary {name} exited with failure"
    );

    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_file(&exe_path);

    Some(String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"))
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

const POINTERS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

swap :: fn(a: ^i64, b: ^i64) {
    temp := a^
    a^ = b^
    b^ = temp
}

increment :: fn(x: &mut i64) {
    x^ = x^ + 1
}

read_sum :: fn(a: &i64, b: &i64) -> i64 {
    a^ + b^
}

main :: fn() -> i64 {
    mut x : i64 = 10
    mut y : i64 = 20
    swap(&x, &y)
    printf("%lld\n", x)
    printf("%lld\n", y)
    increment(&mut x)
    printf("%lld\n", x)
    printf("%lld\n", read_sum(&x, &y))
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

read_sum :: fn(p: &Point) -> i64 {
    p.x + p.y
}

scale :: fn(p: &mut Point, factor: i64) {
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
    printf("%lld\n", read_sum(&p))
    p.x = 100
    scale(&mut p, 2)
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

sum_array :: fn(a: &[5]i64) -> i64 {
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
    printf("%lld\n", sum_array(&nums))
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
