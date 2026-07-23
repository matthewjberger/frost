use frost::{Lexer, Parser};

fn parses(source: &str) -> bool {
    let mut lexer = Lexer::new(source);
    let Ok(tokens) = lexer.tokenize() else {
        return false;
    };
    let mut parser = Parser::new(&tokens);
    parser.parse().is_ok()
}

#[test]
fn grammar_accepts_specified_constructs() {
    let valid = [
        "MAX :: 10",
        "PI :: 3.14",
        "GREETING :: \"hello\"",
        "add :: fn(a: i64, b: i64) -> i64 { a + b }",
        "noop :: fn() { }",
        "Point :: struct { x: i64, y: i64 }",
        "Pair :: struct($T: Type) { first: T, second: T }",
        "Shape :: enum { Circle { r: i64 }, Rect { w: i64, h: i64 } }",
        "Kind :: enum { A, B { n: i64 } }",
        "File :: linear struct { fd: i64 }",
        "Meters :: distinct i64",
        "printf :: extern fn(fmt: ^i8, value: i64) -> i32",
        "close :: extern fn(f: i64)",
        // An extern takes parameter modes and compile-time parameters, which
        // is what a callback registration is written with (spec 12.1).
        "consume :: extern fn(move f: i64)",
        "reg :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64",
        "make :: extern fn(v: i64) -> Ctx",
        // A function type may say `mut`, because a `mut` parameter is a
        // reference in the signature and the surface has no reference type.
        "hold :: fn(f: fn(mut Ctx, i64)) { }",
        "held :: fn(f: fn(move Ctx) -> i64) { }",
        // A value parameter is a compile-time integer, on a function as well
        // as on a struct, and stands for its value in the body.
        "Slab :: struct($T: Type, $N: usize) { storage: [N]T }",
        "reset :: fn($T: Type, $N: usize, mut s: Slab<T, N>) { }",
        "size :: fn($N: usize) -> i64 { N }",
        // An enum takes type parameters the same way a struct does.
        "Maybe :: enum($T: Type) { Nothing, Just { value: T } }",
        "Either :: enum($L: Type, $R: Type) { Left { v: L }, Right { v: R } }",
        "main :: fn() -> i64 {\n x := 5\n mut y : i64 = 0\n y = y + 1\n 0\n }",
        "cond :: fn() -> i64 { if (1 < 2) { 1 } else { 0 } }",
        "loop :: fn() -> i64 {\n mut i : i64 = 0\n while (i < 3) { i = i + 1 }\n i\n }",
        "counted :: fn() { for i in 0..10 { } }",
        "area :: fn(s: i64) -> i64 {\n match s {\n case 0: 1\n case _: 0\n }\n }",
        "shape :: fn(s: i64) -> i64 {\n match s {\n case .Circle { r }: r\n case .Rect { w, h }: w\n case _: 0\n }\n }",
        "tup :: fn(x: i64) -> i64 {\n match (x % 3, x % 5) {\n case (0, 0): 1\n case (_, _): 0\n }\n }",
        "ptrs :: fn(a: ^i8, b: i64, mut c: i64, d: []i64, e: [4]i64, h: Handle<i64>, o: ?i64) -> i64 { 0 }",
        "nested :: fn(p: Pair<Pair<i64>>) -> i64 { 0 }",
        "make :: fn($T: Type, n: i64) -> i64 { sizeof(T) }",
        "callit :: fn() -> i64 { make($i64, 8) }",
        "deref :: fn(p: ^i8) -> i8 { p^ }",
        "field :: fn() -> i64 {\n pt := Point { x = 1, y = 2 }\n pt.x\n }",
        "variant :: fn() -> i64 { unwrap(Shape::Circle { radius = 5 }) }",
        "unit :: fn() -> i64 { pick(Kind::A) }",
        "deferred :: fn() -> i64 {\n defer noop()\n 0\n }",
        "arr :: fn() -> i64 {\n xs := [1, 2, 3]\n xs[0]\n }",
        "fnptr :: fn(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }",
        "ranges :: fn() -> i64 {\n for i in 0..=5 { }\n 0\n }",
    ];
    for source in valid {
        assert!(parses(source), "grammar should accept:\n{source}");
    }
}

#[test]
fn grammar_rejects_malformed_input() {
    let invalid = [
        "cond :: fn() -> i64 { if 1 < 2 { 1 } else { 0 } }",
        "loop :: fn() { while i < 3 { } }",
        "Point :: struct { x i64 }",
        "bad :: fn() -> i64 { 1 + }",
        // Truncated inputs must reject, not hang (regression: the parameter
        // loop used to spin forever at end of input).
        "trunc :: fn(a: i64",
        "trunc2 :: fn(a: i64, b",
        "truncstruct :: struct { x: i64",
        "truncmatch :: fn() -> i64 { match x { case 0: 1",
        "nofields :: struct {",
    ];
    for source in invalid {
        assert!(!parses(source), "grammar should reject:\n{source}");
    }
}
