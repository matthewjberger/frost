// Random programs, run through every compiler and every backend, checked for
// one answer.
//
// Three oracles that are all the same compiler are three backends and one
// front end, and a front-end bug passes all three. It cost something real: the
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

// What every generated program declares above `main`. Fixed rather than
// generated, because the shapes that have gone wrong are not the declarations
// but how a value reaches one: a literal read at the width a binding asks for,
// a number read back through a name it was given, a pointer to a type whose
// code the compiler has to look up rather than compute.
const PRELUDE: &str = "\
frost_rt_write_i64 :: safe extern fn(value: i64)\n\
frost_rt_write_char :: safe extern fn(byte: i64)\n\
print_int_line :: fn(value: i64) {\n\
\x20   frost_rt_write_i64(value)\n\
\x20   frost_rt_write_char(10)\n\
}\n\
print_bool_line :: fn(value: bool) {\n\
\x20   if (value) { frost_rt_write_i64(1) } else { frost_rt_write_i64(0) }\n\
\x20   frost_rt_write_char(10)\n\
}\n\
Meters :: distinct i64\n\
Grip :: distinct ^u8\n\
Pair :: struct { first: i64, second: f32 }\n\
LOW :: -2.1\n\
HIGH :: 1.1\n\
WIDE :: 0.5\n\
COUNT :: 7\n\
no_grip :: fn() -> Grip {\n\
\x20   zero := 0\n\
\x20   unsafe { ptr_cast($u8, zero) }\n\
}\n\
through :: fn(p: ^Meters) -> i64 {\n\
\x20   held : Meters = unsafe { p^ }\n\
\x20   count : i64 = held\n\
\x20   count\n\
}\n\
grips :: fn(p: ^Grip) -> i64 {\n\
\x20   held : Grip = unsafe { p^ }\n\
\x20   if (held == no_grip()) {\n\
\x20       return 1\n\
\x20   }\n\
\x20   0\n\
}\n";

// A float expression written where a binding of a stated width takes it, which
// is the one place a literal's own width is decided by something other than
// itself. Printed as a scaled integer, so the four backends are compared on the
// number rather than on how many digits each writes.
fn gen_typed_float(rng: &mut Rng, index: usize, width: &str) -> String {
    let value = match rng.below(4) {
        // Two constants, so the name is read at the width as well as the
        // digits. These are the values that are not exact in either width, so
        // adding them at one and at the other answers differently.
        0 => "LOW + HIGH".to_string(),
        1 => "LOW + WIDE".to_string(),
        _ => format!("{} + {}", gen_float(rng, 1), gen_float(rng, 1)),
    };
    format!(
        "    w{index} : {width} = {value}\n\
         \x20   print_int_line(cast($i64, w{index} * 100.0))\n"
    )
}

fn gen_program(rng: &mut Rng, lines: usize) -> String {
    // The writers are declared in the prelude over the runtime's own stdout
    // helpers rather than an extern `printf`, so they need no `unsafe` and no
    // declaration a C header might already have its own idea about, and every
    // backend has to agree about what they write.
    let mut source = String::from(PRELUDE);
    source.push_str("main :: fn() -> i64 {\n");
    for index in 0..lines {
        match rng.below(12) {
            0 => {
                let cond = gen_cond(rng, 2);
                let then = gen_expr(rng, 3);
                let els = gen_expr(rng, 3);
                source.push_str(&format!(
                    "    print_int_line(if {cond} {{ {then} }} else {{ {els} }})\n"
                ));
            }
            1 => {
                // A comparison is a truth value and prints as one, which is how
                // a bool reaches the output without printf having to take one.
                let cond = gen_cond(rng, 2);
                source.push_str(&format!("    print_bool_line({cond})\n"));
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
                     \x20   print_int_line(wide{index})\n"
                ));
            }
            3 => {
                // Floats compared rather than printed, so every backend agrees
                // about the answer without agreeing about how many digits a
                // float prints with.
                let left = gen_float(rng, 2);
                let right = gen_float(rng, 2);
                source.push_str(&format!(
                    "    print_bool_line({left} < {right})\n"
                ));
            }
            4 => {
                let value = gen_float(rng, 2);
                source.push_str(&format!(
                    "    f{index} := {value}\n\
                     \x20   print_bool_line(f{index} == f{index})\n"
                ));
            }
            5 => {
                // A literal read at single precision, and the same shape read
                // at double beside it. The two answer differently on purpose:
                // `-2.1 + 1.1` is one number added at one width and another at
                // the other, and a compiler that reads every float literal at
                // double answers the wide one twice.
                let width = if rng.below(2) == 0 { "f32" } else { "f64" };
                source.push_str(&gen_typed_float(rng, index, width));
            }
            6 => {
                // A number through a distinct type and back. The name is the
                // whole of the difference, so what comes out has to be what
                // went in. Half of them read a named constant at the distinct
                // type, which is the integer counterpart of a float constant
                // read at a width: the name has to reach the type the same way
                // the digits would.
                let expr = if rng.below(2) == 0 {
                    format!("COUNT + {}", gen_expr(rng, 1))
                } else {
                    gen_expr(rng, 2)
                };
                source.push_str(&format!(
                    "    m{index} : Meters = {expr}\n\
                     \x20   back{index} : i64 = m{index}\n\
                     \x20   print_int_line(back{index})\n"
                ));
            }
            7 => {
                // A pointer to a distinct type, which is the one type this
                // compiler looks up rather than computes. Reading it back at
                // the distinct type is what says the lookup answered with the
                // type rather than with what it is represented by.
                let expr = gen_expr(rng, 2);
                source.push_str(&format!(
                    "    var p{index} : Meters = {expr}\n\
                     \x20   print_int_line(through(ptr_to(p{index})))\n"
                ));
            }
            8 => {
                // The same, over a pointer rather than an integer. The two go
                // wrong differently: a distinct over a pointer moves the type
                // code by a whole stride and one over an integer moves it
                // inside the first.
                source.push_str(&format!(
                    "    var g{index} := no_grip()\n\
                     \x20   print_int_line(grips(ptr_to(g{index})))\n"
                ));
            }
            9 => {
                // A `match` where a value is wanted. The binding its arms write
                // into is made before an arm has been read, so what type it
                // ends up with is decided by the arms and by nothing else.
                let expr = gen_expr(rng, 2);
                let (a, b, c) =
                    (gen_expr(rng, 1), gen_expr(rng, 1), gen_expr(rng, 1));
                source.push_str(&format!(
                    "    v{index} := match (({expr}) % 3) {{\n\
                     \x20       case 0: {a}\n\
                     \x20       case 1: {b}\n\
                     \x20       case _: {c}\n\
                     \x20   }}\n\
                     \x20   print_int_line(v{index})\n"
                ));
            }
            10 => {
                // The same construct answering with something wider than a
                // register, which is the half of it that was refused outright.
                let expr = gen_expr(rng, 2);
                let (a, b) = (gen_expr(rng, 1), gen_expr(rng, 1));
                let (x, y) = (gen_float(rng, 1), gen_float(rng, 1));
                source.push_str(&format!(
                    "    s{index} := match (({expr}) % 2) {{\n\
                     \x20       case 0: Pair {{ first = {a}, second = {x} }}\n\
                     \x20       case _: Pair {{ first = {b}, second = {y} }}\n\
                     \x20   }}\n\
                     \x20   print_int_line(s{index}.first)\n\
                     \x20   print_int_line(cast($i64, s{index}.second * 100.0))\n"
                ));
            }
            _ => {
                let expr = gen_expr(rng, 4);
                source.push_str(&format!("    print_int_line({expr})\n"));
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

// The interpreter answers for the programs it can hold and says so for the
// rest: a local wider than a register is one it declines, and the generator
// writes those on purpose now. A refusal it names is not a disagreement, so it
// steps aside; anything else it does is.
fn run_ir(name: &str, source: &str) -> Option<String> {
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
    let complaint = String::from_utf8_lossy(&output.stderr).to_string();
    if output.status.code() == Some(3) && complaint.contains("declined") {
        return None;
    }
    assert!(
        output.status.success(),
        "the ir interpreter failed rather than declining:\n{source}\n{complaint}"
    );
    Some(String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n"))
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

// The other half of what a generator is for. The run above asks what a program
// prints, so it only ever sees programs that compile; a rule that lets a view
// of a dead frame through is invisible to it, because the program it wrongly
// accepted runs fine on the machine and prints whatever was left in memory.
//
// This one generates the shapes that must not compile, and the shapes that must,
// and asks both compilers. What it is really watching for is disagreement: the
// bugs this session found were all one compiler refusing what the other built,
// and a divergence is a bug in whichever is behind rather than a difference to
// document.
const VIEW_KINDS: usize = 7;
const ESCAPE_POSITIONS: usize = 7;

/// The kind whose view is a `ref`, which reaches only one position.
const REF_KIND: usize = 6;

/// How many positions a kind can be written in.
///
/// A `ref T` is the one view a program may not store anywhere, so the only
/// position it reaches is the answer itself. A struct field holding one is
/// refused for being a stored borrow before the escape is even asked about, and
/// a cell refused on both halves says nothing about which rule answered.
fn positions_for(kind: usize) -> usize {
    match kind {
        REF_KIND => 1,
        _ => ESCAPE_POSITIONS,
    }
}

/// What one view kind is written as: the module its storage needs, the type the
/// view has, the local it names in the violating form, the expression that view
/// is, and the parameter and expression the honest form uses instead.
struct ViewKind {
    import: &'static str,
    held_type: &'static str,
    storage: &'static str,
    view: &'static str,
    parameter: &'static str,
    honest: &'static str,
}

fn view_kind(kind: usize) -> ViewKind {
    match kind {
        0 => ViewKind {
            import: "mem.frost",
            held_type: "[]i64",
            storage: "data : [4]i64 = [11, 22, 33, 44]",
            view: "data",
            parameter: "mut source: [4]i64",
            honest: "source",
        },
        1 => ViewKind {
            import: "mem.frost",
            held_type: "^i64",
            storage: "var cell : i64 = 5",
            view: "ptr_to(cell)",
            parameter: "mut source: i64",
            honest: "ptr_to(source)",
        },
        2 => ViewKind {
            import: "mem.frost",
            held_type: "[]i64",
            storage: "data : [4]i64 = [11, 22, 33, 44]",
            view: "slice_range($i64, data, 0, 2)",
            parameter: "mut source: [4]i64",
            honest: "slice_range($i64, source, 0, 2)",
        },
        // A `str` is a `[]u8`, so it names storage the way a slice does and
        // leaves a frame the same way. The bootstrap would not build the honest
        // half of this until an array of bytes was allowed to become one.
        3 => ViewKind {
            import: "mem.frost",
            held_type: "str",
            storage: "data : [4]u8 = [104, 105, 33, 0]",
            view: "data",
            parameter: "mut source: [4]u8",
            honest: "source",
        },
        // The two container shapes in `std` that reach storage by a route the
        // plain array cells never take: a slab's fixed run, and one column of a
        // struct of arrays.
        4 => ViewKind {
            import: "slab.frost",
            held_type: "[]i64",
            storage: "var bag : Slab<i64, 4> = { storage = [0; 4], generations = [0; 4], free_list = [0; 4], free_count = 0 }",
            view: "bag.storage",
            parameter: "mut source: Slab<i64, 4>",
            honest: "source.storage",
        },
        5 => ViewKind {
            import: "columns.frost",
            held_type: "[]i64",
            storage: "var bag : columns<Pt, 4> = columns_new()",
            view: "bag.x",
            parameter: "mut source: columns<Pt, 4>",
            honest: "source.x",
        },
        _ => ViewKind {
            import: "mem.frost",
            held_type: "ref i64",
            storage: "var data : [4]i64 = [11, 22, 33, 44]",
            view: "data[0]",
            parameter: "mut source: [4]i64",
            honest: "source[0]",
        },
    }
}

/// The declarations a case needs before its `escape`.
///
/// The wrappers are left out for a `ref`, since a struct field may not hold one
/// at all and the declaration alone would be refused before the escape it is
/// asking about. That kind only reaches the answer position, which uses none of
/// them.
fn safety_head(kind: usize, held_type: &str) -> String {
    let import = view_kind(kind).import;
    let mut head = format!("import \"{import}\"\n");
    if kind == 5 {
        head.push_str("Pt :: struct { x: i64, y: i64 }\n");
    }
    if kind == REF_KIND {
        return head;
    }
    head.push_str(&format!(
        "Holder :: struct {{ view: {held_type} }}\n\
         Outer :: struct {{ inner: Holder }}\n\
         Payload :: enum {{ Empty, Full {{ view: {held_type} }} }}\n\
         keep :: fn(mut h: Holder, held: {held_type}) {{ h.view = held }}\n"
    ));
    head
}

/// One program from the grid. `honest` swaps the storage the view names from a
/// local, which dies at the return, to a `mut` parameter, which is the caller's
/// and outlives the call. Everything else about the program is the same, so a
/// compiler that refuses the honest one is refusing the position rather than
/// the escape.
fn safety_case(
    rng: &mut Rng,
    kind: usize,
    position: usize,
    honest: bool,
) -> String {
    let held = view_kind(kind);
    let held_type = held.held_type;
    let parameter = held.parameter;
    // The honest form names the parameter instead, so nothing the call answers
    // with points into the frame that is about to go.
    let (declare, source) = if honest {
        (String::new(), held.honest.to_string())
    } else {
        (format!("    {}\n", held.storage), held.view.to_string())
    };
    // Noise around the escape, so the walk meets it at a depth it did not pick.
    let noise = match rng.below(3) {
        0 => String::new(),
        1 => format!(
            "    var spare : i64 = {}\n    spare = spare + 1\n",
            rng.below(9)
        ),
        _ => format!(
            "    var spare : i64 = 0\n    while (spare < {}) {{ spare = spare + 1 }}\n",
            rng.below(4) + 1
        ),
    };
    let head = safety_head(kind, held_type);
    let body = match position {
        0 => format!(
            "escape :: fn({parameter}) -> {held_type} {{\n{declare}{noise}    {source}\n}}\n"
        ),
        1 => format!(
            "escape :: fn({parameter}) -> Holder {{\n{declare}{noise}    Holder {{ view = {source} }}\n}}\n"
        ),
        2 => format!(
            "escape :: fn({parameter}) -> Outer {{\n{declare}{noise}    Outer {{ inner = Holder {{ view = {source} }} }}\n}}\n"
        ),
        3 => format!(
            "escape :: fn({parameter}, mut sink: Holder) {{\n{declare}{noise}    sink.view = {source}\n}}\n"
        ),
        4 => format!(
            "escape :: fn({parameter}, mut sink: Holder) {{\n{declare}{noise}    keep(sink, {source})\n}}\n"
        ),
        5 => format!(
            "escape :: fn({parameter}) -> (held: {held_type}, count: i64) {{\n{declare}{noise}    return {{ held = {source}, count = 1 }}\n}}\n"
        ),
        // A variant's payload is held by the enum exactly as a field is held by
        // a struct, so a view put into one leaves the call the same way.
        _ => format!(
            "escape :: fn({parameter}) -> Payload {{\n{declare}{noise}    Payload::Full {{ view = {source} }}\n}}\n"
        ),
    };
    format!("{head}{body}main :: fn() -> i64 {{\n    0\n}}\n")
}

/// Whether a compiler built it, ignoring what it said.
fn builds(compiler: &Path, name: &str, source: &str, hosted: bool) -> bool {
    let directory = std::env::temp_dir();
    let input = directory.join(format!("frost_safety_{name}.frost"));
    std::fs::write(&input, source).unwrap();
    let object = directory.join(format!("frost_safety_{name}.o"));
    let mut command = Command::new(compiler);
    if hosted {
        command.env("FROST_INPUT", &input);
    }
    let output = command
        .arg("-L")
        .arg("std")
        .arg("--native")
        .arg("-o")
        .arg(&object)
        .arg(&input)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&object);
    output.status.success()
}

#[test]
fn both_compilers_agree_about_what_escapes() {
    let Some(hosted) = build_self_hosted_compiler("safety") else {
        return;
    };
    // Fewer seeds than the oracle above, because the grid itself is what covers
    // the shapes and a seed only varies the noise around them. Every cell builds
    // two programs through two compilers, so the count is what keeps the suite
    // usable; `FROST_FUZZ_SEEDS` turns it up for hunting.
    let seeds: u64 = std::env::var("FROST_FUZZ_SEEDS")
        .ok()
        .and_then(|held| held.parse().ok())
        .unwrap_or(2);
    for seed in 0..seeds {
        for kind in 0..VIEW_KINDS {
            for position in 0..positions_for(kind) {
                for honest in [false, true] {
                    let mut rng =
                        Rng::new(seed * 1_000 + (kind * 10 + position) as u64);
                    let source = safety_case(&mut rng, kind, position, honest);
                    let name = format!("{seed}_{kind}_{position}_{honest}");
                    let boot = builds(
                        Path::new(env!("CARGO_BIN_EXE_frost")),
                        &format!("b{name}"),
                        &source,
                        false,
                    );
                    let self_hosted =
                        builds(&hosted, &format!("h{name}"), &source, true);
                    assert_eq!(
                        boot, self_hosted,
                        "the two compilers disagree about this program \
                         (bootstrap built it: {boot}):\n{source}"
                    );
                    assert_eq!(
                        boot,
                        honest,
                        "a view of {} was {} (expected the opposite):\n{source}",
                        if honest { "a parameter" } else { "a local" },
                        if boot { "built" } else { "refused" }
                    );
                }
            }
        }
    }
    let _ = std::fs::remove_file(&hosted);
}

// The same question asked of growth rather than of escape. A container that
// fills asks the allocator for a wider block and gives the old one back, so a
// view taken before that names storage the allocator has taken.
//
// The honest half grows a container nobody is looking at. Everything else about
// the program is the same, so a compiler that refuses it is reading the
// container rather than the run, and that granularity is the whole difficulty:
// the ECS grows one run of a group while a borrow into another is live.
const GROWTH_KINDS: usize = 4;
const GROWTH_POSITIONS: usize = 4;

fn growth_case(
    rng: &mut Rng,
    kind: usize,
    position: usize,
    honest: bool,
) -> String {
    // The last kind views the run as a `str`, which is a `[]u8`, so the
    // container it grows holds bytes and everything written against it follows.
    let element = if kind == 3 { "u8" } else { "i64" };
    let head = format!(
        "import \"vec.frost\"\n\
         import \"io.frost\"\n\
         wrap :: fn(w: Vec<{element}>) -> []{element} {{ vec_slice(${element}, w) }}\n\
         grow :: fn(mut w: Vec<{element}>, value: {element}) {{ vec_push(${element}, w, value) }}\n"
    );
    let grown = if honest { "other" } else { "v" };
    // How the view is taken, and what reads it afterwards. The `ref` writes
    // through the borrow, which lands in the freed block rather than reading it.
    let (take, read) = match kind {
        0 => (
            "view := vec_slice($i64, v)".to_string(),
            "print_int_line(view[0])",
        ),
        1 => ("ref held := vec_slice($i64, v)[0]".to_string(), "held = 99"),
        2 => ("view := wrap(v)".to_string(), "print_int_line(view[0])"),
        _ => (
            "held : str = vec_slice($u8, v)".to_string(),
            "print_int_line(str_len(held))",
        ),
    };
    let push = match position {
        0 => format!("    vec_push(${element}, {grown}, 7)\n"),
        1 => format!(
            "    var step : {element} = 0\n\
             \x20   while (step < {}) {{ vec_push(${element}, {grown}, step)  step = step + 1 }}\n",
            rng.below(3) + 1
        ),
        2 => format!(
            "    if ({} > 0) {{ vec_push(${element}, {grown}, 7) }}\n",
            rng.below(2)
        ),
        _ => format!("    grow({grown}, 7)\n"),
    };
    format!(
        "{head}main :: fn() -> i64 {{\n\
         \x20   var v := vec_new(${element}, 1)\n\
         \x20   var other := vec_new(${element}, 1)\n\
         \x20   vec_push(${element}, v, 11)\n\
         \x20   vec_push(${element}, other, 22)\n\
         \x20   {take}\n\
         {push}\
         \x20   {read}\n\
         \x20   vec_free(${element}, v)\n\
         \x20   vec_free(${element}, other)\n\
         \x20   0\n\
         }}\n"
    )
}

#[test]
fn both_compilers_agree_about_what_growth_invalidates() {
    let Some(hosted) = build_self_hosted_compiler("growth") else {
        return;
    };
    let seeds: u64 = std::env::var("FROST_FUZZ_SEEDS")
        .ok()
        .and_then(|held| held.parse().ok())
        .unwrap_or(2);
    for seed in 0..seeds {
        for kind in 0..GROWTH_KINDS {
            for position in 0..GROWTH_POSITIONS {
                for honest in [false, true] {
                    let mut rng =
                        Rng::new(seed * 1_000 + (kind * 10 + position) as u64);
                    let source = growth_case(&mut rng, kind, position, honest);
                    let name = format!("g{seed}_{kind}_{position}_{honest}");
                    let boot = builds(
                        Path::new(env!("CARGO_BIN_EXE_frost")),
                        &format!("b{name}"),
                        &source,
                        false,
                    );
                    let self_hosted =
                        builds(&hosted, &format!("h{name}"), &source, true);
                    assert_eq!(
                        boot, self_hosted,
                        "the two compilers disagree about this program \
                         (bootstrap built it: {boot}):\n{source}"
                    );
                    assert_eq!(
                        boot,
                        honest,
                        "a view read after {} grew was {} (expected the \
                         opposite):\n{source}",
                        if honest {
                            "another container"
                        } else {
                            "the container behind it"
                        },
                        if boot { "built" } else { "refused" }
                    );
                }
            }
        }
    }
    let _ = std::fs::remove_file(&hosted);
}

// The generator is where coverage is claimed, and a claim about a generator is
// worth nothing until something reads what it wrote. The last widening was
// wasted for exactly this reason: it emitted negative literals from the day it
// was written and nothing that could see the bug was ever asked.
#[test]
fn the_generator_writes_the_constructs_it_claims_to() {
    // More seeds than the run below, because this one compiles nothing and a
    // statement kind that shows up once in a dozen needs the room to. Adding
    // six kinds without widening this is what made it claim a construct had
    // stopped being written when it had only become rarer.
    let mut corpus = String::new();
    for seed in 0..300u64 {
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
        ("a literal read at single precision", " : f32 = "),
        ("the same literal read at double", " : f64 = "),
        ("a constant in a typed float context", "LOW + HIGH"),
        ("a value through a distinct type", " : Meters = "),
        ("a pointer to a distinct integer", "through(ptr_to("),
        ("a pointer to a distinct pointer", "grips(ptr_to("),
        ("a match where a value is wanted", ":= match (("),
        ("a match answering with a struct", "case 0: Pair {"),
        ("a named integer constant at a distinct type", "= COUNT + "),
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
    // Sixty by default, which is what a gate can afford. `FROST_FUZZ_SEEDS`
    // raises it for a hunt: the shapes this writes are the ones five oracles
    // have disagreed about before, and a disagreement that takes a thousand
    // seeds to reach is still a disagreement.
    let seeds: u64 = std::env::var("FROST_FUZZ_SEEDS")
        .ok()
        .and_then(|held| held.parse().ok())
        .unwrap_or(60);
    let first: u64 = std::env::var("FROST_FUZZ_FROM")
        .ok()
        .and_then(|held| held.parse().ok())
        .unwrap_or(0);
    for seed in first..first + seeds {
        let mut rng = Rng::new(seed);
        let source = gen_program(&mut rng, 4);
        let native = run_backend(&format!("s{seed}"), &source, false);
        let via_c = run_backend(&format!("s{seed}c"), &source, true);
        assert_eq!(
            native, via_c,
            "the bootstrap's backends disagree on seed {seed}:\n{source}"
        );
        if let Some(interpreted) = run_ir(&format!("s{seed}i"), &source) {
            assert_eq!(
                native, interpreted,
                "the IR interpreter disagrees on seed {seed}:\n{source}"
            );
        }
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
}
