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

const NESTED_STRUCTS: &str = r#"
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Inner :: struct { a: i64, b: i64 }
Outer :: struct { tag: i64, inner: Inner }

sum_inner :: fn(o: &Outer) -> i64 {
    o.inner.a + o.inner.b
}

main :: fn() -> i64 {
    mut o := Outer { tag = 5, inner = Inner { a = 10, b = 20 } }
    printf("%lld\n", o.tag)
    printf("%lld\n", o.inner.a)
    printf("%lld\n", sum_inner(&o))
    o.inner.a = 99
    printf("%lld\n", o.inner.a)
    printf("%lld\n", sum_inner(&o))
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

describe :: fn(n: &Node) -> i64 {
    match n {
        case .Leaf { value }: value
        case .Pair { location, weight }: location.x + location.y + weight
    }
}

main :: fn() -> i64 {
    leaf := Node::Leaf { value = 7 }
    pair := Node::Pair { location = Point { x = 3, y = 4 }, weight = 100 }
    printf("%lld\n", describe(&leaf))
    printf("%lld\n", describe(&pair))

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

find_first_even :: fn(a: &[6]i64) -> Option {
    for i in 0..6 {
        if (a[i] % 2 == 0) {
            return Option::Some { value = a[i] }
        }
    }
    Option::None
}

unwrap_or :: fn(o: &Option, fallback: i64) -> i64 {
    match o {
        case .Some { value }: value
        case .None: fallback
    }
}

main :: fn() -> i64 {
    data := [1, 3, 5, 8, 9, 10]
    r := find_first_even(&data)
    printf("%lld\n", unwrap_or(&r, 0 - 1))

    odds := [1, 3, 5, 7, 9, 11]
    r2 := find_first_even(&odds)
    printf("%lld\n", unwrap_or(&r2, 0 - 1))
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

bump :: fn(field: &mut i64) {
    field^ = field^ + 100
}

origin :: fn() -> Point {
    Point { x = 7, y = 9 }
}

main :: fn() -> i64 {
    mut p := Point { x = 1, y = 2 }
    bump(&mut p.x)
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
    ha := pool_alloc(p, &a)
    hb := pool_alloc(p, &b)

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
    hc := pool_alloc(p, &c)
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

unwrap :: fn(o: &Opt) -> i64 {
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
    printf("%lld\n", unwrap(&none))
    printf("%lld\n", unwrap(&some))

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
        ("diff_defer", DEFER),
        ("diff_nested", NESTED_STRUCTS),
        ("diff_layouts", DATA_LAYOUTS),
        ("diff_payloads", AGGREGATE_PAYLOADS),
        ("diff_enumval", ENUM_BY_VALUE),
        ("diff_fieldborrow", FIELD_BORROW),
        ("diff_intsem", INTEGER_SEMANTICS),
        ("diff_genpool", GENERATIONAL_POOL),
        ("diff_widening", WIDENING_BINDINGS),
        ("diff_matchagg", MATCH_RETURNS_AGGREGATE),
        ("diff_f32", F32_OPERATIONS),
        ("diff_forward", FORWARD_REFERENCES),
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
