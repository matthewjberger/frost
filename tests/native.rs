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

const ENUMS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Result :: enum {
    Ok { value: i64 },
    Err { code: i64 },
}

unwrap_or_neg :: fn(r: &Result) -> i64 {
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
    printf("%lld\n", unwrap_or_neg(&ok))
    printf("%lld\n", unwrap_or_neg(&err))
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

area :: fn(s: &Shape) -> i64 {
    match s {
        case .Circle { radius }: 3 * radius * radius
        case .Box { side }: side * side
    }
}

dot :: fn(a: &Vec3, b: &Vec3) -> i64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fib :: fn(n: i64) -> i64 {
    if (n < 2) { n } else { fib(n - 1) + fib(n - 2) }
}

triple :: fn(x: i64) -> i64 { x * 3 }

apply_to_array :: fn(f: fn(i64) -> i64, values: &[4]i64) -> i64 {
    mut total : i64 = 0
    for i in 0..4 {
        total = total + f(values[i])
    }
    total
}

main :: fn() -> i64 {
    a := Vec3 { x = 1, y = 2, z = 3 }
    b := Vec3 { x = 4, y = 5, z = 6 }
    printf("%lld\n", dot(&a, &b))

    c := Shape::Circle { radius = 10 }
    sq := Shape::Box { side = 7 }
    printf("%lld\n", area(&c))
    printf("%lld\n", area(&sq))

    printf("%lld\n", fib(15))

    nums := [1, 2, 3, 4]
    printf("%lld\n", apply_to_array(triple, &nums))
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

#[test]
fn cranelift_and_c_backends_agree() {
    let programs = [
        ("diff_arith", ARITHMETIC),
        ("diff_floats", FLOATS),
        ("diff_widths", WIDTHS),
        ("diff_strings", STRINGS),
        ("diff_pointers", POINTERS),
        ("diff_structs", STRUCTS),
        ("diff_arrays", ARRAYS),
        ("diff_enums", ENUMS),
        ("diff_byvalue", BY_VALUE),
        ("diff_retagg", RETURN_AGGREGATE),
        ("diff_tuple", TUPLE_MATCH),
        ("diff_funcptr", FUNCTION_POINTERS),
        ("diff_kitchen", KITCHEN_SINK),
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
    }
}
