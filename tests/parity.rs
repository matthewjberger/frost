use std::process::Command;

// The two tables that say what one language is, and the one that says where the
// range ends.
//
// A parity table is a claim about both compilers at once: the first holds
// programs they have to agree about, the second programs they both have to
// refuse in the same words, and the third expressions the fold and the machine
// have to put on the same side of the line. They live together because a
// divergence found anywhere else ends up as a row in one of them.
#[path = "support.rs"]
mod support;

use support::{
    bootstrap_output, bootstrap_report_at, build_self_hosted_compiler,
    linker_available, selfhosted_default_output,
};

// Programs the language refuses, put through both compilers. The third field is
// what each has to say, since two compilers refusing one program for two
// different reasons is a divergence that a refusal alone would not show.
// Programs both compilers accept and both say something about. A warning is a
// report the build does not refuse on, and it is held to what a refusal is held
// to: one rendering, and the same words. The two used to render one the same
// way only by coincidence, since nothing compared them: the bootstrap wrote a
// bare `warning: at f:3:5: ...` line while the self-hosted compiler wrote a
// caret block with no word saying which kind of report it was.
const WARNED_BY_BOTH: &[(&str, &str, &str)] = &[
    // An `unsafe` block around nothing that needs one.
    (
        "an_unsafe_block_that_vouches_for_nothing",
        "main :: fn() -> i64 {
             var n : i64 = 3
             unsafe {
                 n = n + 1
             }
             n
}
",
        "this `unsafe` block holds no unchecked operation, so it vouches for nothing",
    ),
    // One inside another, where the outer already vouches for what is in it.
    (
        "an_unsafe_block_inside_another",
        "main :: fn() -> i64 {
             var cells : [2]i64 = [1, 2]
             unsafe {
                 p := ptr_to(cells[0])
                 unsafe {
                     p^
                 }
             }
}
",
        "already vouches for what is in it",
    ),
];

const REFUSED_BY_BOTH: &[(&str, &str, &str)] = &[
    // A call writing a compile-time argument the signature settles says twice
    // what the argument says once. Which of them a call writes is a property of
    // the signature, so taking this as well would be two spellings for every
    // call to a generic over a container.
    (
        "a_settled_compile_time_argument_may_not_be_written_at_the_call",
        "import \"io.frost\"\n\
         Box :: struct($T: Type) { held: $T }\n\
         unwrap :: fn($T: Type, b: Box<T>) -> $T { b.held }\n\
         main :: fn() -> i64 {\n\
         \x20   var b := Box<i64> { held = 41 }\n\
         \x20   print(\"{}\n\", unwrap($i64, b))\n\
         \x20   0\n\
         }\n",
        "is settled by the type of",
    ),
    // A struct written into a binding declared as another struct. Both travel
    // by address, so nothing downstream can tell the two apart and the
    // comparison has to happen while both are still spelled the way the reader
    // wrote them. The bootstrap took the address and named the lowered local
    // instead, which read as a compiler talking about itself.
    (
        "a_binding_takes_the_struct_its_declaration_names",
        "import \"io.frost\"\n\
         Left :: struct { a: i64 }\n\
         Right :: struct { a: i64 }\n\
         main :: fn() -> i64 {\n\
         \x20   var kept := Left { a = 1 }\n\
         \x20   var swapped: Right = kept\n\
         \x20   print(\"{}\n\", swapped.a)\n\
         \x20   0\n\
         }\n",
        "this binding is a 'Right' and the value is a 'Left'",
    ),
    // A bundle a container carries is part of what the container is, so two
    // containers hashed different ways are two types and neither passes where
    // the other is wanted. Pinned on the type as each compiler spells it: the
    // bundle has to reach the reader by name, since a report that wrote both
    // sides the same way would say they differ and show nothing that does.
    (
        "a_container_carrying_one_bundle_is_not_one_carrying_another",
        "import \"io.frost\"\n\
         Hashing :: struct($K: Type) { hash: fn(K) -> i64 }\n\
         Bag :: struct($K: Type, $ops: Hashing<K>) { first: $K }\n\
         one :: fn(k: i64) -> i64 { k }\n\
         two :: fn(k: i64) -> i64 { k * 2 }\n\
         plain :: Hashing<i64> { hash = one }\n\
         doubled :: Hashing<i64> { hash = two }\n\
         bag_new :: fn($K: Type, $ops: Hashing<K>, first: $K) -> Bag<K, ops> {\n\
         \x20   Bag { first = first }\n\
         }\n\
         bag_hash :: fn($K: Type, $ops: Hashing<K>, b: Bag<K, ops>) -> i64 {\n\
         \x20   ops.hash(b.first)\n\
         }\n\
         doubled_only :: fn(b: Bag<i64, doubled>) -> i64 {\n\
         \x20   bag_hash(b)\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   var kept := bag_new($plain, 21)\n\
         \x20   print(\"{}\n\", doubled_only(kept))\n\
         \x20   0\n\
         }\n",
        "this argument is a 'Bag<i64, plain>' and a 'Bag<i64, doubled>' is what \
         is wanted here",
    ),
    // A function supplied at the call site is one whose body neither compiler
    // can see, so what it answers with is worth the shortest-lived argument
    // that could have reached it. An allocator built in this frame is one of
    // those, and a run carved out of it dies with the call however the carve
    // was written.
    (
        "a_bundle_hands_back_a_view_of_the_callers_frame",
        "import \"io.frost\"
         import \"mem.frost\"
         Bump :: struct { data: []u8, offset: i64 }
         Allocation :: struct($A: Type) { take: fn(mut A, i64) -> []u8 }
         bump_take :: fn(mut b: Bump, size: i64) -> []u8 {
             run := slice_range(b.data, b.offset, size)
             b.offset = b.offset + size
             run
         }
         bump_source :: Allocation<Bump> { take = bump_take }
         leak :: fn($source: Allocation<Bump>, n: i64) -> []u8 {
             var backing: [64]u8 = [0; 64]
             var here := Bump { data = backing, offset = 0 }
             source.take(here, n)
         }
         main :: fn() -> i64 {
             got := leak($bump_source, 8)
             print(\"{}\\n\", slice_len(got))
             0
         }
",
        "region: a pointer into the frame of 'leak' is the call's answer; the \
         storage it names dies when the call returns",
    ),
    // The same, through a plain compile-time function argument rather than a
    // bundle, since the bundle is not what the rule is about.
    (
        "a_compile_time_function_hands_back_a_view_of_the_callers_frame",
        "import \"io.frost\"
         import \"mem.frost\"
         Bump :: struct { data: []u8, offset: i64 }
         bump_take :: fn(mut b: Bump, size: i64) -> []u8 {
             run := slice_range(b.data, b.offset, size)
             b.offset = b.offset + size
             run
         }
         leak :: fn($take: fn(mut Bump, i64) -> []u8, n: i64) -> []u8 {
             var backing: [64]u8 = [0; 64]
             var here := Bump { data = backing, offset = 0 }
             take(here, n)
         }
         main :: fn() -> i64 {
             got := leak($bump_take, 8)
             print(\"{}\\n\", slice_len(got))
             0
         }
",
        "region: a pointer into the frame of 'leak' is the call's answer; the \
         storage it names dies when the call returns",
    ),
    // `main` is called by the C runtime, which hands it the argument count and
    // the argument vector, and a Frost `main` declares neither. One that
    // declares a parameter is handed whatever the platform left in that
    // register: an `i64` reads the argument count nothing asked for, and a
    // `str` reads it as an address and faults on the first byte. Both of those
    // built and ran with no `unsafe` written anywhere.
    (
        "an_entry_point_that_declares_a_parameter",
        "import \"io.frost\"
         main :: fn(s: str) -> i64 {
             print(\"{}\\n\", str_len(s))
             0
         }
",
        "'main' takes no parameters, and this one takes 1; what a call to it \
         would supply is whatever the platform left in a register",
    ),
    // A capability is an implicit parameter appended to the list, so a `main`
    // drawing one is handed a register nobody filled and the first write
    // through it faults. The self-hosted compiler counted it and the bootstrap
    // did not, which is how the same program was refused by one and ran by the
    // other.
    (
        "an_entry_point_that_draws_a_capability",
        "import \"io.frost\"
         Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
         grab :: fn($N: usize, mut a: Arena<N>) -> i64 {
             a.offset = a.offset + 8
             a.offset
         }
         main :: fn() -> i64 uses Arena<256> {
             print(\"{}\\n\", grab(arena))
             0
         }
",
        "'main' takes no parameters, and this one takes 1; what a call to it \
         would supply is whatever the platform left in a register",
    ),
    // A capability is filled in at each call, so a function that draws one and
    // is taken as a value has no call to fill it at. The address goes somewhere
    // that calls it through a type saying nothing about the capability, and the
    // callee reads the register nobody wrote. The bootstrap caught it as a
    // signature that did not match and the self-hosted compiler built it, and
    // what it built faulted with no `unsafe` written anywhere.
    (
        "a_capability_drawing_function_taken_as_a_value",
        "import \"io.frost\"
         Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
         bump :: fn($N: usize, mut a: Arena<N>) -> i64 {
             a.offset = a.offset + 8
             a.offset
         }
         worker :: fn(n: i64) -> i64 uses Arena<256> {
             bump(arena) + n
         }
         call_it :: fn(f: fn(i64) -> i64, v: i64) -> i64 { f(v) }
         main :: fn() -> i64 {
             var scratch: Arena<256> = Arena { data = [0; 256], offset = 0 }
             var out: i64 = 0
             with scratch {
                 out = call_it(worker, 3)
             }
             print(\"{}\\n\", out)
             0
         }
",
        "'worker' draws a capability, which is one more parameter, so it \
         cannot be taken as a value: a call through a function value supplies \
         what its type says and nothing else",
    ),
    // `$f` names a function as a compile-time argument, which reaches both
    // compilers as a type rather than as a name and so slipped past the rule
    // above. One reported a signature the reader never wrote, in spellings the
    // surface does not have, and the other said the call needed a capability
    // while standing inside the `with` block that supplies one.
    (
        "a_capability_drawing_function_as_a_compile_time_argument",
        "import \"io.frost\"
         Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
         bump :: fn($N: usize, mut a: Arena<N>) -> i64 {
             a.offset = a.offset + 8
             a.offset
         }
         worker :: fn(n: i64) -> i64 uses Arena<256> {
             bump(arena) + n
         }
         apply :: fn($f: fn(i64) -> i64, v: i64) -> i64 { f(v) }
         main :: fn() -> i64 {
             var scratch: Arena<256> = Arena { data = [0; 256], offset = 0 }
             var out: i64 = 0
             with scratch {
                 out = apply($worker, 3)
             }
             print(\"{}\\n\", out)
             0
         }
",
        "'worker' draws a capability, which is one more parameter, so it \
         cannot be taken as a value: a call through a function value supplies \
         what its type says and nothing else",
    ),
    // A function value is its signature. The self-hosted compiler let every
    // signature reach every other, because its compatibility chain ended in an
    // answer of yes and a function type fell to it, so a name taking two
    // parameters was handed to a place wanting one: the call passed one and the
    // callee read whatever sat in the register for the other. It built and
    // faulted with no `unsafe` written anywhere.
    //
    // Both name the signature the way a reader writes one, arguments and all.
    // The sentences around it differ: the bootstrap answers here from its IR
    // check and the self-hosted from its walk over the syntax.
    (
        "a_function_value_whose_signature_does_not_match",
        "Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
         bump :: fn($N: usize, mut a: Arena<N>) -> i64 {
             a.offset = a.offset + 8
             a.offset
         }
         plain :: fn(n: i64, mut a: Arena<256>) -> i64 { bump(a) + n }
         call_it :: fn(f: fn(i64) -> i64, v: i64) -> i64 { f(v) }
         main :: fn() -> i64 { call_it(plain, 3) }
",
        "fn(i64, mut Arena<256>) -> i64",
    ),
    // A test body is run by the runner as a function taking nothing, so it has
    // nowhere to draw a capability from. Neither compiler parsed `uses` there,
    // so neither said so: one reported that a declaration head was expected,
    // about the word `test` just written, and the other read the body as an
    // expression and reported a struct with no field named after the first
    // thing inside it.
    (
        "a_test_that_draws_a_capability",
        "Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
         bump :: fn($N: usize, mut a: Arena<N>) -> i64 {
             a.offset = a.offset + 8
             a.offset
         }
         test \"draws a capability\" uses Arena<256> {
             assert(bump(arena) == 8)
         }
",
        "a `test` body is run by the test runner, which supplies nothing, so a \
         test draws no capability",
    ),
    // The one caller settles the answer as well as the arguments. A `main`
    // that can fail answers the tagged union the `?` machinery made, which the
    // bootstrap's backend then named in a message about a type the reader never
    // wrote while the self-hosted compiler emitted it into a C `int`.
    (
        "an_entry_point_that_can_fail",
        "import \"io.frost\"
         Broken :: struct { at: i64 }
         step :: fn(n: i64) -> i64 ! Broken {
             if (n < 0) { return { at = n } }
             n * 2
         }
         main :: fn() -> i64 ! Broken {
             print(\"{}\\n\", step(3)?)
             0
         }
",
        "'main' is called by the C runtime and its answer is the process exit \
         code, so it answers i64",
    ),
    (
        "an_entry_point_that_answers_an_aggregate",
        "Pair :: struct { a: i64, b: i64 }
         main :: fn() -> Pair { Pair { a = 1, b = 2 } }
",
        "'main' is called by the C runtime and its answer is the process exit \
         code, so it answers i64",
    ),
    // A format string is read where the call is written, so the count it names
    // and the count the call gives have to agree there. Both compilers read the
    // literal, so both have to say the same thing about it.
    (
        "a_format_string_with_too_few_values",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   print(\"{} of {}\n\", 1)\n\
         \x20   0\n\
         }\n",
        "this format string opens 2 hole(s) and the call gives 1 value(s)",
    ),
    (
        "a_format_string_with_too_many_values",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   print(\"{}\n\", 1, 2)\n\
         \x20   0\n\
         }\n",
        "this format string opens 1 hole(s) and the call gives 2 value(s)",
    ),
    // A value whose type is what a call to a generic answers with. The parse
    // reads it as the default until every instance has been walked, so judging
    // it there accepted a `[]i64` for reading as an `i64`; the self-hosted
    // compiler then refused the program a step later, as an ordinary type error
    // inside the library. Both spellings are here because a name bound to the
    // call and the call written out reach the answer by different routes.
    (
        "a_format_string_given_a_generic_call",
        "import \"io.frost\"\nimport \"mem.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   var xs: [3]i64 = [1, 2, 3]\n\
         \x20   print(\"{}\n\", slice_range(xs, 0, 3))\n\
         \x20   0\n\
         }\n",
        "a format string writes a number, a yes or no, or a str, and this is a []i64",
    ),
    (
        "a_format_string_given_a_name_bound_to_a_generic_call",
        "import \"io.frost\"\nimport \"mem.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   var xs: [3]i64 = [1, 2, 3]\n\
         \x20   view := slice_range(xs, 0, 3)\n\
         \x20   print(\"{}\n\", view)\n\
         \x20   0\n\
         }\n",
        "a format string writes a number, a yes or no, or a str, and this is a []i64",
    ),
    // The same question asked through `write`, which is the door a program's
    // own destination goes through and the one that used to have no check at
    // all behind it.
    (
        "a_write_to_a_named_destination_is_checked",
        "import \"io.frost\"\nloud :: fn(text: str) { write(to_stdout, \"[{}]\", text) }\n\
         main :: fn() -> i64 {\n\
         \x20   write(loud, \"{} of {}\n\", 1)\n\
         \x20   0\n\
         }\n",
        "this format string opens 2 hole(s) and the call gives 1 value(s)",
    ),
    // A lone brace is a typo rather than a decision, so it is refused rather
    // than written out. `{{` is how one is meant.
    (
        "a_lone_brace_in_a_format_string",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   print(\"a { b\n\")\n\
         \x20   0\n\
         }\n",
        "opens a hole or stands for one brace",
    ),
    // How many values follow is settled where the call is written, so what says
    // how many the line names has to be written there too.
    (
        "a_format_string_that_is_not_a_literal",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   var held := \"x{}y\"\n\
         \x20   print(held, 1)\n\
         \x20   0\n\
         }\n",
        "a format string is written as a literal",
    ),
    // A struct reached the backend before this check existed, where it became
    // a codegen fault naming a type rather than a refusal naming the line.
    (
        "a_format_string_given_a_struct",
        "import \"io.frost\"\nPoint :: struct { x: i64, y: i64 }\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\n\", Point { x = 1, y = 2 })\n\
         \x20   0\n\
         }\n",
        "a format string writes a number, a yes or no, or a str, and this is a Point",
    ),
    // A call that can fail answers with which of the two happened, and a
    // statement reads neither. The rule used to reach only a call holding a
    // resource, where linearity caught it; a failure nobody reads is the same
    // fault whether or not there is anything to leak.
    (
        "a_fallible_answer_nobody_reads",
        "import \"io.frost\"
         Blocked :: struct { at: i64 }
         step :: fn(n: i64) -> i64 ! Blocked {
             if (n < 0) { return { at = n } }
             n * 2
         }
         main :: fn() -> i64 {
             step(3)
             0
         }
",
        "this can fail and nothing reads whether it did",
    ),
    // An `errdefer` in a function that cannot fail names an exit that function
    // does not have.
    (
        "an_errdefer_without_a_failure_set",
        "import \"io.frost\"
         work :: fn() -> i64 {
             errdefer print(\"{}\\n\", 9)
             7
         }
         main :: fn() -> i64 { print(\"{}\\n\", work())  0 }
",
        "runs where a function leaves through its failure set, and this one has none",
    ),
    // What an `errdefer` covers is the failure path alone, so the value is
    // still the straight-line path's to consume and a body that never does
    // leaks on the way out with an answer.
    (
        "an_errdefer_does_not_answer_for_the_straight_line",
        "import \"io.frost\"
         FileError :: enum { Missing }
         Held :: linear struct { id: i64 }
         close :: fn(move h: Held) { print(\"{}\\n\", h.id) }
         opened :: fn(id: i64) -> Held { Held { id = id } }
         work :: fn() -> i64 ! FileError {
             h := opened(1)
             errdefer close(h)
             7
         }
         main :: fn() -> i64 { print(\"{}\\n\", 1)  0 }
",
        // The two word the leak differently, which is a divergence of its own
        // and older than this: the bootstrap says which paths, the self-hosted
        // compiler says it plainly. What both say is the same fault about the
        // same name.
        "linear value 'h' is",
    ),
    // A name the compiler owns, declared by a program. It used to be taken: the
    // bootstrap let the declaration win and said nothing, and the self-hosted
    // compiler kept reading the name as its own, so `slice_len(4)` was a call in
    // one and a length in the other. One spelling cannot mean two things, so the
    // declaration is refused where it is written rather than at a call that
    // behaves oddly.
    (
        "a_declaration_of_a_compiler_name",
        "import \"io.frost\"
         slice_len :: fn(x: i64) -> i64 { x + 1 }
         main :: fn() -> i64 {
             print(\"{}\\n\", slice_len(4))
             0
         }
",
        "is the compiler's own, and a name means one thing wherever it is written",
    ),
    // `live_slots(c)` written where a value goes. The slots it names are walked and
    // never held, so there is nothing for a binding to be given.
    (
        "a_live_walk_read_as_a_value",
        "import \"io.frost\"
         import \"columns.frost\"
         Cell :: struct { v: i64 }
         main :: fn() -> i64 {
             var c : columns<Cell, 8> = columns_new()
             columns_reset(c)
             held := live_slots(c)
             print(\"{}\\n\", held)
             0
         }
",
        "is the subject of a `for` and nothing else",
    ),
    // A subject the walk would have to work out. The container is read where it
    // stands rather than bound, so a call there would run once a word.
    (
        "a_live_walk_over_a_computed_container",
        "import \"io.frost\"
         import \"columns.frost\"
         Cell :: struct { v: i64 }
         made :: fn() -> columns<Cell, 8> { columns_new() }
         main :: fn() -> i64 {
             for slot in live_slots(made()) { print(\"{}\\n\", slot) }
             0
         }
",
        "walks a container that is named",
    ),
    // A struct a program declared whose name happens to start the way the
    // multiple-return lowering names the ones it synthesizes. The bootstrap
    // told them apart by that prefix, so this one was taken out of the linear
    // closure, stopped being a resource, and the `File` in it leaked with no
    // diagnostic. The lowering records the structs it made now, so a name is a
    // name.
    (
        "a_declared_struct_named_like_a_lowered_one",
        "import \"io.frost\"
         File :: linear struct { handle: i64 }
         __multiHolder :: struct { f: File }
         open :: fn(n: i64) -> File { File { handle = n } }
         main :: fn() -> i64 {
             held := __multiHolder { f = open(3) }
             print(\"{}\\n\", held.f.handle)
             0
         }
",
        "linear value 'held'",
    ),
    // `var` on a discard. One makes a binding assignable and the other binds
    // nothing, so the pair says two things that cannot both be true.
    (
        "a_var_on_a_discard",
        "import \"io.frost\"
         split :: fn(v: i64) -> (high: i64, low: i64) { return v / 256, v % 256 }
         main :: fn() -> i64 {
             high, var _ := split(4096)
             print(\"{}\\n\", high)
             0
         }
",
        "`var` makes a binding assignable and `_` binds nothing",
    ),
    // A `_` taking a value that has to be consumed. The list binds one name per
    // value, so a resource has to land on one and be consumed there. Refused
    // where the `_` is written rather than by the linearity walk, which runs on
    // what the lowering left behind and would name the storage it gave the
    // discard, a name nothing in the program spells.
    (
        "an_underscore_dropping_a_resource",
        "import \"io.frost\"
         File :: linear struct { handle: i64 }
         pair :: fn(n: i64) -> (opened: File, count: i64) {
             return { opened = File { handle = n }, count = 1 }
         }
         close :: fn(move f: File) { print(\"{}\\n\", f.handle) }
         main :: fn() -> i64 {
             _, count := pair(3)
             print(\"{}\\n\", count)
             0
         }
",
        "this `_` drops a 'File', which is consumed exactly once",
    ),
    // `_` as an ordinary binding. It is the wildcard token of 2.3 and never a
    // name, so this has nowhere to parse. The self-hosted lexer had no token
    // for it at all: `_` fell into the identifier rule, so `_ := 5` bound a
    // readable local called `_` that a second one silently shadowed.
    // `_` read back. Same cause, and the reason the one above matters: under
    // the self-hosted compiler this printed 5.
    (
        "an_underscore_read_as_a_value",
        "import \"io.frost\"
         split :: fn(v: i64) -> (high: i64, low: i64) { return v / 256, v % 256 }
         main :: fn() -> i64 {
             high, _ := split(4096)
             print(\"{}\\n\", high + _)
             0
         }
",
        "Token not valid for an expression: '_'",
    ),
    // A return type list that leaves a value unnamed. The fields were then
    // called `value0` and `value1`, spellings each compiler picked for itself
    // and no program was allowed to write, which took a refusal apiece to
    // enforce. No signature in the corpus wrote one, so the list names every
    // value and both the synthesis and the refusal guarding it are gone.
    (
        "a_return_type_list_that_leaves_a_value_unnamed",
        "import \"io.frost\"
         split :: fn(value: i64) -> (i64, i64) {
             return value / 256, value % 256
         }
         main :: fn() -> i64 {
             high, low := split(4096)
             print(\"{}\\n\", high + low)
             0
         }
",
        "a return type list names every value",
    ),
    // A call answering with a return type list, bound to one name. The struct
    // behind the list carries a name each compiler chose, so a program holding
    // one holds a value of a type it has no way to write, and reading a field
    // off it reaches a field name the compiler picked. The self-hosted compiler
    // bound it, read `held.high`, and ran.
    (
        "a_multi_return_bound_to_one_name",
        "import \"io.frost\"\n\
         split :: fn(value: i64) -> (high: i64, low: i64) {\n\
         \x20   return { high = value / 256, low = value % 256 }\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   held := split(4096)\n\
         \x20   print(\"{}\\n\", held.high)\n\
         \x20   0\n\
         }\n",
        "'split' returns 2 values, so its call is bound by a list of names",
    ),
    // An aggregate travels by address, so once a call has taken one there is
    // nothing left to tell a pointer to one struct from a pointer to another:
    // every check after that point is looking at a machine word. The bootstrap
    // took the address without comparing the two types and a twelve-byte value
    // was read as sixty-four bytes, with the bounds checks agreeing throughout.
    // The self-hosted compiler had always refused it.
    (
        "an_aggregate_of_the_wrong_type_is_refused",
        "import \"io.frost\"\nSmall :: struct { x: f32, y: f32, z: f32 }\n\
         Large :: struct { m: [16]f32 }\n\
         takes_large :: fn(held: Large) -> f32 { held.m[0] }\n\
         main :: fn() -> i64 {\n\
         \x20   small := Small { x = 1.0, y = 2.0, z = 3.0 }\n\
         \x20   answer := takes_large(small)\n\
         \x20   if (answer > 0.0) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
         \x20   0\n\
         }\n",
        "Large",
    ),
    // A view of a parameter handed back from inside a branch. The walk that
    // works out which parameters a function's answer can name read `return`
    // inside a loop and inside a `with`, and read the block's trailing value,
    // and never read one inside an `if` or a `match`: a branch is an
    // expression, so a `return` written in one arrives as an expression
    // statement rather than as a statement of its own.
    //
    // So a function answering with a view of a parameter from a branch was
    // recorded as naming nothing, and a caller was free to store what it
    // answered with somewhere that outlives what it points into. Adding the
    // branch made the check *more* permissive than the same function without
    // it, which is a fixpoint losing a source rather than growing one.
    //
    // The caller has to take the container as a parameter for this to bite: an
    // answer naming a parameter outlives the call, so the check only turns on
    // the source it was missing. With the container a local of the caller, the
    // answer names that frame either way and the same program is refused with
    // the hole open, which is what made the first attempt at this case pass
    // whether or not the fix was in.
    //
    // `tag` is load-bearing and not padding. A `Grip` views a byte, so the byte
    // is what makes it possible for the answer to point into a `Resource` at
    // all; without it the answer is three pointers copied out of a struct and
    // nothing here names the caller's frame. It was added when `view_lands_in`
    // stopped giving up on every aggregate answer, which turned this case from
    // a refusal into an honest program.
    (
        "a_view_of_a_parameter_returned_from_a_branch",
        "import \"io.frost\"\nimport \"mem.frost\"\n\
         Grip :: distinct ^u8\n\
         Trio :: struct { one: Grip, two: Grip, three: Grip }\n\
         Resource :: struct { held: Trio, transient: bool, slot: i64, tag: u8 }\n\
         Slot :: struct { held: Trio, made: bool }\n\
         Box :: struct { pool: []Slot, into: []Resource }\n\
         no_trio :: fn() -> Trio {\n\
         \x20   zero := 0\n\
         \x20   Trio { one = unsafe { ptr_cast($u8, zero) },\n\
         \x20       two = unsafe { ptr_cast($u8, zero) },\n\
         \x20       three = unsafe { ptr_cast($u8, zero) } }\n}\n\
         backing_of :: fn(b: Box, one: Resource) -> Trio {\n\
         \x20   if (one.transient == false) {\n\
         \x20       return one.held\n\
         \x20   }\n\
         \x20   pool := b.pool\n\
         \x20   pool[one.slot].held\n}\n\
         put :: fn(mut b: Box, at: i64) {\n\
         \x20   source := Resource { held = no_trio(), transient = false, tag = 0,\n\
         \x20       slot = 0 }\n\
         \x20   given := backing_of(b, source)\n\
         \x20   var into := b.into\n\
         \x20   into[at].held = given\n}\n\
         main :: fn() -> i64 {\n\
         \x20   var b := Box { pool = heap_slice($Slot, 2),\n\
         \x20       into = heap_slice($Resource, 2) }\n\
         \x20   put(b, 0)\n\
         \x20   print(\"{}\\n\", 1)\n\
         \x20   heap_release_slice(b.into)\n\
         \x20   heap_release_slice(b.pool)\n\
         \x20   0\n}\n",
        "frame",
    ),
    // A distinct type taken from its representation through a field. The
    // bootstrap held a named local, an argument and a binding to the rule and
    // let a field through, because the check sat in the branch that assigns to
    // an identifier and the branch that assigns to a place did not have it. A
    // `flags` type is a distinct one with names under it, so the same hole put
    // a bare number into a set of bits, which is how it was found: the wgpu
    // binding declares `usage` as one.
    (
        "distinct_through_a_field",
        "import \"io.frost\"\nMeters :: distinct i64\n\
         Holder :: struct { m: Meters }\n\
         main :: fn() -> i64 {\n\
         \x20   plain : i64 = 2\n\
         \x20   var h := Holder { m = cast($Meters, 0) }\n\
         \x20   h.m = plain\n\
         \x20   print(\"{}\\n\", 1)\n\
         \x20   0\n}\n",
        "representation",
    ),
    (
        "flags_through_a_field",
        "import \"io.frost\"\nUsage :: flags u64 { None = 0, Read = 1, Write = 2 }\n\
         Holder :: struct { usage: Usage }\n\
         main :: fn() -> i64 {\n\
         \x20   plain : u32 = 2\n\
         \x20   var h := Holder { usage = Usage::None }\n\
         \x20   h.usage = plain\n\
         \x20   print(\"{}\\n\", 1)\n\
         \x20   0\n}\n",
        "names declared under it",
    ),
    // A resource reached through a field is a place of its own, so consuming it
    // twice consumes it twice. Tracked by name, the field was never recorded and
    // neither compiler said the second consumption was one: with a consumer that
    // frees, a double free in safe code with no `unsafe` anywhere.
    (
        "field_twice",
        "File :: linear struct { fd: i64 }\n\
         Holder :: struct { file: File, name: i64 }\n\
         close :: fn(move f: File) -> i64 { f.fd }\n\
         drop_holder :: fn(move h: Holder) -> i64 { close(h.file) }\n\
         main :: fn() -> i64 {\n\
         \x20   h := Holder { file = File { fd = 7 }, name = 1 }\n\
         \x20   close(h.file)\n\
         \x20   close(h.file)\n\
         \x20   drop_holder(h)\n}\n",
        "moved",
    ),
    // An element answers the same way.
    (
        "element_twice",
        "File :: linear struct { fd: i64 }\n\
         close :: fn(move f: File) -> i64 { f.fd }\n\
         drop_run :: fn(move xs: [2]File) -> i64 { 0 }\n\
         main :: fn() -> i64 {\n\
         \x20   var run : [2]File = [File { fd = 9 }; 2]\n\
         \x20   close(run[0])\n\
         \x20   close(run[0])\n\
         \x20   drop_run(run)\n}\n",
        "moved",
    ),
    // And so does a place behind a `mut` borrow, which reaches it by a different
    // road: the mode lowering has already made the parameter a borrow by the time
    // the check runs, so what the callee declared it takes is what says a
    // resource was handed over.
    (
        "borrowed_field_twice",
        "File :: linear struct { fd: i64 }\n\
         Holder :: struct { file: File, name: i64 }\n\
         close :: fn(move f: File) -> i64 { f.fd }\n\
         twice :: fn(mut h: Holder) -> i64 {\n\
         \x20   close(h.file)\n\
         \x20   close(h.file)\n}\n\
         drop_holder :: fn(move h: Holder) -> i64 { close(h.file) }\n\
         main :: fn() -> i64 {\n\
         \x20   var h := Holder { file = File { fd = 5 }, name = 1 }\n\
         \x20   twice(h)\n\
         \x20   drop_holder(h)\n}\n",
        "moved",
    ),
    // Writing into storage that was handed away is not taking it back, and
    // reviving the container from a write to one field let a value be consumed,
    // written into, and consumed again.
    (
        "write_into_consumed",
        "File :: linear struct { fd: i64 }\n\
         Holder :: struct { file: File, name: i64 }\n\
         close :: fn(move f: File) -> i64 { f.fd }\n\
         open :: fn(n: i64) -> File { File { fd = n } }\n\
         drop_holder :: fn(move h: Holder) -> i64 { close(h.file) }\n\
         main :: fn() -> i64 {\n\
         \x20   var h := Holder { file = open(7), name = 1 }\n\
         \x20   drop_holder(h)\n\
         \x20   h.file = open(9)\n\
         \x20   drop_holder(h)\n}\n",
        "moved",
    ),
    // A run of resources is a resource, since freeing the run is not freeing what
    // is in it and a fixed array holds its elements by value.
    (
        "leaked_run",
        "File :: linear struct { fd: i64 }\n\
         Holder :: struct { items: [2]File }\n\
         main :: fn() -> i64 {\n\
         \x20   h := Holder { items = [File { fd = 1 }; 2] }\n\
         \x20   0\n}\n",
        "consumed",
    ),
    // A pool of resources, from a generic container. A slot is emptied by bumping
    // a generation and filled again by an insert that overwrites it, so nothing
    // consumes the element that leaves and no consumer can be written that would.
    (
        "generic_pool",
        "Slab :: struct($T: Type, $N: usize) {\n\
         \x20   storage: [N]T,\n\
         \x20   generations: [N]i64,\n\
         \x20   free_list: [N]i64,\n\
         \x20   free_count: i64,\n}\n\
         File :: linear struct { fd: i64 }\n\
         Node :: struct { file: File, hp: i64 }\n\
         main :: fn() -> i64 {\n\
         \x20   var pool : Slab<Node, 2> = slab_new()\n\
         \x20   0\n}\n",
        "is a pool of",
    ),
    // The same pool, reached without writing its name. Both rules read the
    // instantiations a program writes down, and a call that answers with one
    // makes it without anyone writing it, so a pool of resources compiled as
    // long as nobody annotated the binding that held it.
    (
        "pool_nobody_named",
        "Slab :: struct($T: Type, $N: usize) {\n\
         \x20   storage: [N]T,\n    generations: [N]i64,\n}\n\
         File :: linear struct { fd: i64 }\n\
         Node :: struct { file: File, hp: i64 }\n\
         fresh :: fn($T: Type, $N: usize, seed: $T) -> Slab<T, N> {\n\
         \x20   slab_new()\n}\n\
         main :: fn() -> i64 {\n\
         \x20   pool := fresh($2, Node { file = File { fd = 1 }, hp = 0 })\n\
         \x20   0\n}\n",
        "is a pool of",
    ),
    // The same container written out rather than instantiated. Both rules ran
    // over the instantiations a program names, so a concrete one slipped.
    (
        "concrete_pool",
        "File :: linear struct { fd: i64 }\n\
         Pool :: struct { storage: [2]File, generations: [2]i64 }\n\
         drop_pool :: fn(move p: Pool) -> i64 { 0 }\n\
         main :: fn() -> i64 {\n\
         \x20   p := Pool { storage = [File { fd = 1 }; 2], generations = [0; 2] }\n\
         \x20   drop_pool(p)\n}\n",
        "is a pool of",
    ),
    // A resource consumed through a borrowed parameter, twice. The count is
    // kept per place and a place lived in one function, so `once` refused a
    // second `close(h.file)` inside itself and said nothing to its caller. Both
    // compilers took two calls to it: a double free in safe code with no
    // `unsafe` anywhere, and the largest hole the guarantees had.
    (
        "consumed_through_a_borrow_twice",
        "import \"io.frost\"\nFile :: linear struct { fd: i64 }\n\
         Holder :: struct { file: File, name: i64 }\n\
         close :: fn(move f: File) -> i64 { f.fd }\n\
         once :: fn(mut h: Holder) -> i64 { close(h.file) }\n\
         drop_holder :: fn(move h: Holder) -> i64 { close(h.file) }\n\
         main :: fn() -> i64 {\n\
         \x20   var h := Holder { file = File { fd = 5 }, name = 1 }\n\
         \x20   print(\"{}\\n\", once(h))\n    print(\"{}\\n\", once(h))\n    drop_holder(h)\n}\n",
        "moved",
    ),
    // The same by the other road: handing the resource out rather than
    // consuming it. A function answering with what it read out of a borrow can
    // be called as many times as you like, and each answer is the same storage.
    (
        "handed_out_of_a_borrow_twice",
        "import \"io.frost\"\nFile :: linear struct { fd: i64 }\n\
         Holder :: struct { file: File, name: i64 }\n\
         close :: fn(move f: File) -> i64 { f.fd }\n\
         lift :: fn(h: Holder) -> File { h.file }\n\
         drop_holder :: fn(move h: Holder) -> i64 { close(h.file) }\n\
         main :: fn() -> i64 {\n\
         \x20   h := Holder { file = File { fd = 5 }, name = 1 }\n\
         \x20   print(\"{}\\n\", close(lift(h)))\n    print(\"{}\\n\", close(lift(h)))\n\
         \x20   drop_holder(h)\n}\n",
        "moved",
    ),
    // A resource still held on a path that leaves before the line that hands it
    // on. The self-hosted check read forward for a consuming statement and took
    // the first one it found for the whole answer, so a `return` written above
    // that line read as though the path through it went on to consume: a
    // resource silently dropped, in safe code, with no `unsafe` anywhere.
    (
        "leaked_on_early_return",
        "import \"io.frost\"\nFile :: linear struct { fd: i64 }\n\
         close :: fn(move f: File) -> i64 { f.fd }\n\
         run :: fn(early: i64) -> i64 {\n\
         \x20   f := File { fd = 3 }\n\
         \x20   if (early > 0) {\n        return 1\n    }\n\
         \x20   close(f)\n}\n\
         main :: fn() -> i64 {\n    print(\"{}\\n\", run(1))\n    0\n}\n",
        "consumed",
    ),
    // `break` is the same path out of the block a loop body is, and it binds to
    // the nearest loop, which is what tells it from a `break` written inside a
    // loop further in.
    (
        "leaked_on_break",
        "import \"io.frost\"\nFile :: linear struct { fd: i64 }\n\
         close :: fn(move f: File) -> i64 { f.fd }\n\
         run :: fn() -> i64 {\n\
         \x20   var i : i64 = 0\n\
         \x20   while (i < 4) {\n\
         \x20       f := File { fd = i }\n\
         \x20       if (i == 2) {\n            break\n        }\n\
         \x20       close(f)\n        i = i + 1\n    }\n    0\n}\n\
         main :: fn() -> i64 {\n    print(\"{}\\n\", run())\n    0\n}\n",
        "consumed",
    ),
    // The rules that were already shared, here so the table is the whole list
    // rather than only what this round added.
    (
        "use_after_move",
        "P :: struct { x: i64 }\n\
         take :: fn(move p: P) -> i64 { p.x }\n\
         main :: fn() -> i64 {\n\
         \x20   p := P { x = 1 }\n\
         \x20   take(p)\n\
         \x20   take(p)\n}\n",
        "moved",
    ),
    (
        "leaked_resource",
        "File :: linear struct { fd: i64 }\n\
         main :: fn() -> i64 {\n\
         \x20   f := File { fd = 1 }\n\
         \x20   0\n}\n",
        "consumed",
    ),
    (
        "reference_in_a_field",
        "Bad :: struct { r: ref i64 }\n\
         main :: fn() -> i64 { 0 }\n",
        "second-class",
    ),
    // A slice is an address and a length. A bare pointer is the address alone,
    // so a callee handed one reads whatever sat beside it for the length: this
    // answered a length of two trillion for a five-byte string, and indexing a
    // hundred thousand past the end of a two-byte one passed the bounds check.
    // The one pointer that does reach a slice of bytes is a string literal,
    // where the compiler wrote the bytes and knows how many there are.
    (
        "pointer_for_text",
        "import \"io.frost\"\nwidth :: fn(s: str) -> i64 { str_len(s) }\n\
         main :: fn() -> i64 {\n\
         \x20   p : ^i8 = \"hello\"\n\
         \x20   print(\"{}\\n\", width(p))\n\
         \x20   0\n}\n",
        "argument",
    ),
    // A resource handed out of a borrowed container by an element. A summary
    // tells a caller which field of a borrowed parameter went, and an element is
    // reached by a number worked out while the program runs, so there is no
    // place a caller could be told about: `peek` answers with the same storage
    // however many times it is asked, and each answer is consumed. Refused
    // rather than approximated, since a summary saying "some element went" is
    // the whole container as far as a caller can act on it and would refuse a
    // container releasing its elements one at a time.
    (
        "handed_out_by_element",
        "import \"io.frost\"\nBox :: struct($T: Type) { storage: [2]T, len: i64 }\n\
         File :: linear struct { fd: i64 }\n\
         close :: fn(move f: File) -> i64 { f.fd }\n\
         peek :: fn($T: Type, b: Box<T>, index: i64) -> $T { b.storage[index] }\n\
         drop_box :: fn(move b: Box<File>) -> i64 { 0 }\n\
         main :: fn() -> i64 {\n\
         \x20   b : Box<File> = Box { storage = [File { fd = 1 }; 2], len = 2 }\n\
         \x20   print(\"{}\\n\", close(peek(b, 0)))\n\
         \x20   print(\"{}\\n\", close(peek(b, 0)))\n\
         \x20   drop_box(b)\n}\n",
        "by an element",
    ),
    // A value moved twice through a compile-time list. A generic's own body
    // names parameters bound to nothing and a list there has no elements to
    // unroll, so the template says nothing about the moves its instances make.
    // The bootstrap checked the template alone and took this; the check now runs
    // on the substituted body, which is the first point where both the element
    // type and the unrolled list exist.
    (
        "moved_through_a_compile_time_list",
        "P :: struct { x: i64 }\n\
         take :: fn(move p: P) -> i64 { p.x }\n\
         each :: fn($body: Type, items: $...) {\n\
         \x20   body(c for c in items)\n\
         \x20   body(c for c in items)\n}\n\
         main :: fn() -> i64 {\n\
         \x20   p := P { x = 1 }\n\
         \x20   each($take, p)\n\
         \x20   0\n}\n",
        "moved",
    ),
    // The same bound, where it does not hold. The complaint lands on the call
    // the reader wrote rather than inside a library body they never saw, which
    // is the whole point of asking at the call.
    (
        "a_resource_against_a_bound_that_refuses_one",
        "import \"io.frost\"\nFile :: linear struct { fd: i64 }\n\
         close :: fn(move f: File) -> i64 { f.fd }\n\
         only_plain :: fn($T: Type, v: $T) -> i64 where !is_linear(T) { 1 }\n\
         main :: fn() -> i64 {\n\
         \x20   f := File { fd = 1 }\n\
         \x20   print(\"{}\\n\", only_plain(f))\n\
         \x20   print(\"{}\\n\", close(f))\n\
         \x20   0\n}\n",
        "is_linear",
    ),
    // The same, written with brackets the reader put there and a join. A parse
    // does not keep grouping, so rendering the shape it built quoted a line
    // nobody wrote, while the compiler that replays the tokens quoted the one
    // they did. The declaration is what is quoted now, by both.
    (
        "a_bound_written_with_brackets",
        "import \"io.frost\"
File :: linear struct { fd: i64 }
         close :: fn(move f: File) -> i64 { f.fd }
         either :: fn($T: Type, v: $T) -> i64
             where (is_float(T) || is_struct(T)) && !is_linear(T) { 2 }
         main :: fn() -> i64 {
             f := File { fd = 1 }
             print(\"{}\n\", either(f))
             print(\"{}\n\", close(f))
             0
}
",
        "is declared `where (is_float(T) || is_struct(T)) && !is_linear(T)`",
    ),
    // A borrow bound by `ref` handed to a format string. A borrow of a struct is
    // what a value of it already is here, since an aggregate travels by address
    // either way, so what is refused is writing a struct rather than holding
    // one. The type it names had to be read off the parse's own table: asking
    // `type_of` while the body is still being read asks the table the emitters
    // fill, which is empty, so the borrow came back one of an i64 and the call
    // went out to a specialization nothing defines.
    (
        "a_borrow_handed_to_a_format_string",
        "import \"io.frost\"
Holder :: struct { a: i64, b: i64 }
         main :: fn() -> i64 {
             var h : Holder = Holder { a = 1, b = 2 }
             ref r := h
             print(\"{}\n\", r)
             0
}
",
        "a format string writes a number, a yes or no, or a str, and this is a Holder",
    ),
    // A borrow named in a report, which is spelled the way a reader writes one.
    // The bootstrap files a type under a name that round-trips through the
    // reader monomorphization reads its arguments back with, and there a borrow
    // takes `&T` and `&mut T`, two forms the surface dropped; the self-hosted
    // compiler wrote `^T`, which is a raw pointer and a different thing.
    (
        "a_borrow_named_in_a_report",
        "Holder :: struct { a: i64, b: i64 }
         Other :: struct { x: i64 }
         take :: fn(o: Other) -> i64 { o.x }
         main :: fn() -> i64 {
             var h : Holder = Holder { a = 1, b = 2 }
             ref r := h
             take(r)
}
",
        "this argument is a 'ref Holder' and a 'Other' is what is wanted here",
    ),
    // `!` answers the opposite of a yes or no and takes one. A number is not
    // one: reading `!count` as `count == 0` is a conversion nothing wrote, and
    // a corpus full of `started == 0` over an i64 flag means what it says. So
    // the two spellings coexist and neither reaches into the other.
    (
        "a_negation_of_a_number",
        "main :: fn() -> i64 {
             n := 3
             if (!n) {
                 return 1
             }
             0
}
",
        "'!' answers the opposite of a yes or no, and this is a 'i64'",
    ),
    (
        "a_negation_of_a_decimal",
        "main :: fn() -> i64 {
             x := 1.5
             if (!x) {
                 return 1
             }
             0
}
",
        "'!' answers the opposite of a yes or no, and this is a 'f64'",
    ),
    (
        "a_negation_of_a_struct",
        "Pair :: struct { a: i64 }
         main :: fn() -> i64 {
             p := Pair { a = 1 }
             if (!p) {
                 return 1
             }
             0
}
",
        "'!' answers the opposite of a yes or no, and this is a 'Pair'",
    ),
    // `!` carries two meanings and the space around it says which. A failure
    // set is written `-> T ! E` and a negation against what it negates, and
    // each written the other way says the other thing. The compiler is never
    // confused, since either has one parse; a reader skimming is, and a corpus
    // where `!` beside a space is a failure set and `!` against a name is a
    // negation is what the rule buys.
    (
        "a_failure_set_written_without_its_spaces",
        "Bad :: enum { Nope }
         tight :: fn(n: i64) -> i64 !Bad {
             n
}
         main :: fn() -> i64 { 0 }
",
        "a `!` against what follows it negates, and this one marks a failure set; write `-> T ! E`, with a space on both sides",
    ),
    (
        "a_negation_written_with_a_space",
        "ready :: fn() -> bool { true }
         main :: fn() -> i64 {
             if (! ready()) {
                 return 1
             }
             0
}
",
        "a `!` with a space after it marks a failure set, and this one negates; write it against what it negates, as `!ready`",
    ),
    // A generic instantiated with a resource where the program never writes the
    // instantiation's name. What a type holds was read off the names the source
    // spells out, so `held := wrap(...)` left `Opt<File>` ordinary data
    // and the obligation on the resource inside it went in and did not come
    // out. The types a call forms answer for it now, and a variant's payload is
    // held by its enum the way a field is held by its struct.
    (
        "a_resource_in_an_instance_nobody_named",
        "File :: linear struct { fd: i64 }\n\
         Opt :: enum($T: Type) { None, Some { value: T } }\n\
         wrap :: fn($T: Type, move value: $T) -> Opt<T> {\n\
         \x20   Opt::Some { value = value }\n}\n\
         main :: fn() -> i64 {\n\
         \x20   held := wrap(File { fd = 1 })\n\
         \x20   0\n}\n",
        "consumed",
    ),
    // A generic literal no field of which is written as the bare parameter, so
    // there is nothing to read the argument off and nothing in the context
    // naming it. The self-hosted compiler stood a placeholder in for the type
    // and the emitters read a struct index out of something that is not a
    // struct, dying with an arena index of -16 rather than saying anything. The
    // bootstrap rendered the parameter it could not bind as its own name and
    // called `Box<T>` a type nothing declares, which points at neither the cause
    // nor the fix.
    (
        "a_generic_literal_with_no_argument_to_read",
        "import \"io.frost\"\nBox :: struct($T: Type) { storage: [2]T, len: i64 }\n\
         main :: fn() -> i64 {\n\
         \x20   b := Box { storage = [7; 2], len = 2 }\n\
         \x20   print(\"{}\\n\", b.storage[0] + b.len)\n\
         \x20   0\n}\n",
        "is generic",
    ),
    // A struct answer that really does name the argument. A parameter handed by
    // address can be pointed at by whatever comes back, and `view_lands_in` is
    // what decides whether it could be: the answer holds a `^Inner` and the
    // parameter is an `Inner`, so it could, and this stays refused after that
    // rule stopped giving up on every aggregate answer.
    (
        "a_struct_answer_that_points_at_its_argument",
        "Inner :: struct { p: ^u8 }
         Held :: struct { at: ^Inner }
         Outer :: struct { kept: Held }
         made :: fn() -> Inner {
             zero := 0
             Inner { p = unsafe { ptr_cast($u8, zero) } }
         }
         point_at :: fn(a: Inner) -> Held { Held { at = ptr_to(a) } }
         from_local :: fn(mut o: Outer) {
             a := made()
             o.kept = point_at(a)
         }
         main :: fn() -> i64 {
             zero := 0
             var o := Outer { kept = Held { at = unsafe { ptr_cast($Inner, zero) } } }
             from_local(o)
             0
         }
",
        "stored where the call cannot see",
    ),
    // `include_str` opens the file while the program is compiled, so a path
    // that is not a string literal has nothing to open, and a file that is
    // not there is found out here rather than at run time. The wording is
    // shared by both compilers on purpose: the refusal is part of the
    // language.
    (
        "an_include_of_a_file_that_is_not_there",
        "import \"io.frost\"\nMISSING :: include_str(\"frost_no_such_file_anywhere.wgsl\")\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", str_len(MISSING))\n\
         \x20   0\n}\n",
        "include_str: cannot read",
    ),
    (
        "an_include_whose_path_is_not_a_literal",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   held := \"lit.wgsl\"\n\
         \x20   print(\"{}\\n\", str_len(include_str(held)))\n\
         \x20   0\n}\n",
        "include_str takes one string literal",
    ),
    // A view of a local leaving the call by every road it can take. An array
    // becoming a `[]T` is a view being *formed*, and nothing about the
    // expression says so: `data` reads the same in all of these, and only the
    // type on the other side says an address is being taken. So the check
    // asked what the array *held*, which for a run of numbers is nothing, and
    // every one of these compiled and handed back storage that had died while
    // the same view written as the return expression was refused.
    //
    // One entry per position rather than one for the family, because each is a
    // separate place the question has to be asked and a check that asks it in
    // eight places out of nine is the check that was here before.
    (
        "a_slice_of_a_local_in_a_returned_struct",
        "Holder :: struct { view: []i64 }\n\
         escape :: fn() -> Holder {\n\
         \x20   data : [4]i64 = [11, 22, 33, 44]\n\
         \x20   Holder { view = data }\n}\n\
         main :: fn() -> i64 {\n\
         \x20   h := escape()\n\
         \x20   h.view[0]\n}\n",
        "the storage it names dies when the call returns",
    ),
    (
        "a_slice_of_a_local_in_a_nested_struct",
        "Inner :: struct { view: []i64 }\n\
         Outer :: struct { inner: Inner }\n\
         escape :: fn() -> Outer {\n\
         \x20   data : [4]i64 = [11, 22, 33, 44]\n\
         \x20   Outer { inner = Inner { view = data } }\n}\n\
         main :: fn() -> i64 {\n\
         \x20   o := escape()\n\
         \x20   o.inner.view[0]\n}\n",
        "the storage it names dies when the call returns",
    ),
    (
        "a_slice_of_a_local_written_into_a_parameter",
        "Holder :: struct { view: []i64 }\n\
         stash :: fn(mut h: Holder) {\n\
         \x20   data : [4]i64 = [11, 22, 33, 44]\n\
         \x20   h.view = data\n}\n\
         main :: fn() -> i64 {\n\
         \x20   outer : [1]i64 = [0]\n\
         \x20   var h : Holder = { view = outer }\n\
         \x20   stash(h)\n\
         \x20   h.view[0]\n}\n",
        "the storage it names dies when the call returns",
    ),
    (
        "a_slice_of_a_local_handed_to_a_call_that_keeps_it",
        "Holder :: struct { view: []i64 }\n\
         keep :: fn(mut h: Holder, view: []i64) {\n\
         \x20   h.view = view\n}\n\
         escape :: fn(mut h: Holder) {\n\
         \x20   data : [4]i64 = [11, 22, 33, 44]\n\
         \x20   keep(h, data)\n}\n\
         main :: fn() -> i64 {\n\
         \x20   outer : [1]i64 = [0]\n\
         \x20   var h : Holder = { view = outer }\n\
         \x20   escape(h)\n\
         \x20   h.view[0]\n}\n",
        "the storage it names dies when the call returns",
    ),
    // The one that says the rule is about forming a view rather than about
    // struct fields. `slice_range` answers with a view of its own parameter,
    // which outlives it; what this call handed that parameter was a local.
    (
        "a_slice_built_by_a_call_over_a_local",
        "import \"mem.frost\"\n\
         escape :: fn() -> []i64 {\n\
         \x20   data : [4]i64 = [11, 22, 33, 44]\n\
         \x20   slice_range(data, 0, 2)\n}\n\
         main :: fn() -> i64 {\n\
         \x20   view := escape()\n\
         \x20   view[0]\n}\n",
        "the storage it names dies when the call returns",
    ),
    (
        "a_slice_of_a_local_stored_in_a_struct_then_returned",
        "Holder :: struct { view: []i64 }\n\
         escape :: fn(seed: []i64) -> Holder {\n\
         \x20   data : [4]i64 = [11, 22, 33, 44]\n\
         \x20   var h : Holder = { view = seed }\n\
         \x20   h.view = data\n\
         \x20   h\n}\n\
         main :: fn() -> i64 {\n\
         \x20   outer : [1]i64 = [0]\n\
         \x20   h := escape(outer)\n\
         \x20   h.view[0]\n}\n",
        "the storage it names dies when the call returns",
    ),
    (
        "a_slice_of_a_local_in_a_multi_return",
        "escape :: fn() -> (view: []i64, count: i64) {\n\
         \x20   data : [4]i64 = [11, 22, 33, 44]\n\
         \x20   return { view = data, count = 4 }\n}\n\
         main :: fn() -> i64 {\n\
         \x20   view, count := escape()\n\
         \x20   held : []i64 = view\n\
         \x20   held[0] + count\n}\n",
        "the storage it names dies when the call returns",
    ),
    // A container takes its element as a `$T`, so what the call binds that to
    // is what says whether handing it an array forms a view. The self-hosted
    // compiler cannot read the parameter's type off the declaration at all
    // here: what the node records is whatever instantiation was made last, so
    // an imported `vec_push` carries the element type of vec.frost's own tests.
    (
        "a_slice_of_a_local_pushed_into_a_container",
        "import \"vec.frost\"\n\
         fill :: fn(mut sink: Vec<[]i64>) {\n\
         \x20   data : [4]i64 = [11, 22, 33, 44]\n\
         \x20   vec_push(sink, data)\n}\n\
         main :: fn() -> i64 {\n\
         \x20   0\n}\n",
        "the storage it names dies when the call returns",
    ),
    // One per guarantee, so the list of programs the language refuses is
    // readable in one place rather than inferred from which check happens to
    // own each. Every one is a complete program that is wrong in exactly one
    // way, and both compilers have to refuse it saying the same thing.
    (
        "a_value_used_after_it_moved",
        "Point :: struct { x: i64, y: i64 }\n\
         eat :: fn(move p: Point) -> i64 { p.x }\n\
         main :: fn() -> i64 {\n\
         \x20   p : Point = { x = 1, y = 2 }\n\
         \x20   a := eat(p)\n\
         \x20   b := eat(p)\n\
         \x20   a + b\n}\n",
        "moved",
    ),
    (
        "a_resource_left_unconsumed",
        "Session :: linear struct { id: i64 }\n\
         open :: fn() -> Session { Session { id = 7 } }\n\
         close :: fn(move s: Session) -> i64 { s.id }\n\
         main :: fn() -> i64 {\n\
         \x20   s := open()\n\
         \x20   0\n}\n",
        "consumed",
    ),
    (
        "one_value_passed_to_two_mut_parameters",
        "Point :: struct { x: i64, y: i64 }\n\
         both :: fn(mut a: Point, mut b: Point) { a.x = 1  b.x = 2 }\n\
         main :: fn() -> i64 {\n\
         \x20   var p : Point = { x = 0, y = 0 }\n\
         \x20   both(p, p)\n\
         \x20   p.x\n}\n",
        "borrow",
    ),
    (
        "a_raw_pointer_read_outside_an_unsafe_block",
        "main :: fn() -> i64 {\n\
         \x20   var n : i64 = 5\n\
         \x20   p := ptr_to(n)\n\
         \x20   p^\n}\n",
        "unsafe",
    ),
    (
        "a_pointer_cast_outside_an_unsafe_block",
        "main :: fn() -> i64 {\n\
         \x20   var n : i64 = 5\n\
         \x20   p := ptr_to(n)\n\
         \x20   q := ptr_cast($i64, p)\n\
         \x20   0\n}\n",
        "unsafe",
    ),
    (
        "a_c_function_called_outside_an_unsafe_block",
        "frost_rt_thread_join :: extern fn(handle: i64)\n\
         main :: fn() -> i64 {\n\
         \x20   frost_rt_thread_join(0)\n\
         \x20   0\n}\n",
        "unsafe",
    ),
    (
        "a_borrow_stored_in_a_struct_field",
        "Holder :: struct { held: ref i64 }\n\
         main :: fn() -> i64 {\n\
         \x20   0\n}\n",
        "ref",
    ),
    // A view of a container, read after the container gave its block back. The
    // frame check traces the view to `s` and stops there, since `s` is alive,
    // and what it cannot ask is whether the block is still the one it was.
    (
        "a_view_read_after_its_container_was_freed",
        "Sink :: linear struct { room: []i64, len: i64 }\n\
         sink_slice :: fn(s: Sink) -> []i64 { s.room }\n\
         sink_free :: fn(move s: Sink) -> i64 { s.len }\n\
         main :: fn() -> i64 {\n\
         \x20   room : [4]i64 = [11, 22, 33, 44]\n\
         \x20   s : Sink = { room = room, len = 4 }\n\
         \x20   view := sink_slice(s)\n\
         \x20   held := sink_free(s)\n\
         \x20   view[0] + held\n}\n",
        "views storage held by",
    ),
    // The same, where the container is generic. The self-hosted compiler read a
    // generic's parameter as the type argument whenever the declared type was
    // an aggregate, so `Sink<T>` became `i64`, nothing about the parameter
    // looked like something a move takes, and the move went unrecorded: this
    // program built there while the bootstrap refused it, and so did a plain
    // use of `s` after the free.
    (
        "a_generic_container_is_consumed_by_a_move",
        "Sink :: linear struct($T: Type) { room: []T, len: i64 }\n\
         sink_free :: fn($T: Type, move s: Sink<T>) -> i64 { s.len }\n\
         main :: fn() -> i64 {\n\
         \x20   room : [4]i64 = [11, 22, 33, 44]\n\
         \x20   s : Sink<i64> = { room = room, len = 4 }\n\
         \x20   held := sink_free(s)\n\
         \x20   s.len + held\n}\n",
        "moved",
    ),
    // A view of a container, read after the container grew. `vec_push` asks the
    // allocator for a wider block and gives the old one back, so the view names
    // storage the allocator has taken, and every read through it is
    // bounds-checked against a length describing what used to be there.
    (
        "a_view_read_after_its_container_grew",
        "import \"io.frost\"\nimport \"vec.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   var v := vec_new($i64, 1)\n\
         \x20   vec_push(v, 111)\n\
         \x20   view := vec_slice(v)\n\
         \x20   vec_push(v, 222)\n\
         \x20   print(\"{}\\n\", view[0])\n\
         \x20   vec_free(v)\n\
         \x20   0\n}\n",
        "has since replaced",
    ),
    // The growth inside a loop, which is where a container actually fills. The
    // read is above the push, so one walk of the body sees nothing wrong: what
    // is stale is what the turn before left behind.
    (
        "a_view_read_at_the_top_of_a_growing_loop",
        "import \"io.frost\"\nimport \"vec.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   var v := vec_new($i64, 1)\n\
         \x20   vec_push(v, 111)\n\
         \x20   view := vec_slice(v)\n\
         \x20   var count : i64 = 0\n\
         \x20   while (count < 8) {\n\
         \x20       print(\"{}\\n\", view[0])\n\
         \x20       vec_push(v, count)\n\
         \x20       count = count + 1\n\
         \x20   }\n\
         \x20   vec_free(v)\n\
         \x20   0\n}\n",
        "has since replaced",
    ),
    // A `ref` into a container is a view of the same run, so the write through
    // it after a growth lands in the block that was given back.
    (
        "a_ref_element_written_after_its_container_grew",
        "import \"vec.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   var v := vec_new($i64, 1)\n\
         \x20   vec_push(v, 111)\n\
         \x20   ref held := vec_slice(v)[0]\n\
         \x20   vec_push(v, 222)\n\
         \x20   held = 999\n\
         \x20   vec_free(v)\n\
         \x20   0\n}\n",
        "has since replaced",
    ),
    // The same view, reached through a wrapper. Nothing here holds the
    // container: `passthrough` is handed a view and answers with it, so no
    // parameter of that call names the run, and reading the argument as a place
    // of its own loses `v` entirely. The lowering hoists the inner call into a
    // temporary, so what the walk meets is a name rather than the call that
    // made it, and a binding that views a run has to stand for that run.
    (
        "a_view_through_a_wrapper_read_after_a_growth",
        "import \"io.frost\"\nimport \"vec.frost\"\n\
         passthrough :: fn(s: []i64) -> []i64 { s }\n\
         main :: fn() -> i64 {\n\
         \x20   var v := vec_new($i64, 1)\n\
         \x20   vec_push(v, 111)\n\
         \x20   view := passthrough(vec_slice(v))\n\
         \x20   vec_push(v, 222)\n\
         \x20   print(\"{}\\n\", view[0])\n\
         \x20   vec_free(v)\n\
         \x20   0\n}\n",
        "has since replaced",
    ),
    (
        "a_view_through_a_wrapper_read_after_a_release",
        "import \"io.frost\"\nimport \"vec.frost\"\n\
         passthrough :: fn(s: []i64) -> []i64 { s }\n\
         main :: fn() -> i64 {\n\
         \x20   var v := vec_new($i64, 1)\n\
         \x20   vec_push(v, 111)\n\
         \x20   view := passthrough(vec_slice(v))\n\
         \x20   vec_free(v)\n\
         \x20   print(\"{}\\n\", view[0])\n\
         \x20   0\n}\n",
        "which has been given away",
    ),
    // A body that grows a run of its own parameter and reads a view of it. The
    // write is direct rather than a call, which is how a container's own growth
    // is always written: `vec_push` writes `v.storage` because handing the
    // borrow to a helper would copy the header and lose the new block. No
    // summary carries this one, so the write itself has to be read.
    (
        "a_view_read_after_the_same_body_replaced_its_run",
        "Bag :: struct { room: []i64, len: i64 }\n\
         bag_slice :: fn(b: Bag) -> []i64 { b.room }\n\
         grow_and_read :: fn(mut b: Bag, fresh: []i64) -> i64 {\n\
         \x20   view := bag_slice(b)\n\
         \x20   b.room = fresh\n\
         \x20   view[0]\n}\n\
         main :: fn() -> i64 { 0 }\n",
        "has since replaced",
    ),
    // The same question asked of an arena rather than of a frame. A `[]T`
    // carved out of one names the arena's storage exactly as a `^T` does, and
    // reading only the pointer let the slice beside it leave the block.
    // A literal of a name nothing declares. The self-hosted compiler answered
    // `i64` for the name, since that is what it answers for a type name it
    // cannot find, and `i64` is not a struct: every pass after it read a struct
    // index out of the type and indexed the struct table at 0 - STRUCT_BASE, so
    // the compile ended with an arena complaining about -16 rather than with a
    // word about the program. A crash on a program that is wrong says nothing
    // about what is wrong with it.
    // A comma separates one argument from the next. Without the rule
    // `add(1 2)` compiled and answered what `add(1, 2)` answers.
    // A leading operator continues the line above, so dropping one turns one
    // statement into two and the terms after the break stop being read.
    // `x := 10` over `+ 20` answers 30, and without the `+` it answers 10.
    (
        "a_continuation_that_lost_its_operator",
        "main :: fn() -> i64 {
             x := 10
                 20
             x
}
",
        "indented past the statement above it",
    ),
    // A type name nothing declares. Read as an opaque name it went through
    // every pass without a word, and for a parameter nothing reads, forever.
    (
        "an_undeclared_type_in_a_signature",
        "takes :: fn(v: Absent) -> i64 { 0 }
         main :: fn() -> i64 { 0 }
",
        "is not a type this program declares",
    ),
    (
        "a_dropped_comma_between_arguments",
        "add :: fn(a: i64, b: i64) -> i64 { a + b }
         main :: fn() -> i64 { add(1 2) }
",
        "expected ',' between one element and the next",
    ),
    (
        "a_literal_of_an_undeclared_type",
        "main :: fn() -> i64 {\n\
         \x20   held := Absent { a = 2 }\n\
         \x20   0\n}\n",
        "'Absent' is not a type this program declares",
    ),
    (
        "a_slice_of_an_arena_leaving_its_with_block",
        "Arena :: struct($N: usize) { data: [N]u8, offset: i64 }\n\
         carve_from :: fn(mut a: Arena<256>) -> []u8 {\n\
         \x20   a.data\n}\n\
         carve :: fn() -> []u8 uses Arena<256> {\n\
         \x20   carve_from(arena)\n}\n\
         main :: fn() -> i64 {\n\
         \x20   var arena : Arena<256> = Arena { data = [0; 256], offset = 0 }\n\
         \x20   other : [1]u8 = [0]\n\
         \x20   var escaped : []u8 = other\n\
         \x20   with arena {\n\
         \x20       escaped = carve()\n\
         \x20   }\n\
         \x20   0\n}\n",
        "may not outlive the arena",
    ),
    // An address is a multiple of a power of two or of nothing, so a field
    // asking to start at a multiple of three is asking for a layout no machine
    // has.
    (
        "an_alignment_that_is_not_a_power_of_two",
        "Odd :: struct { a: u8, b: i64 align(3) }\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", sizeof(Odd))\n\
         \x20   0\n}\n",
        "`align` takes a power of two, and 3 is not one",
    ),
    // `packed` says no field is padded and `align` says this one is. A
    // declaration writing both says two things that cannot both hold, so it is
    // refused rather than one of them quietly winning.
    (
        "an_alignment_inside_a_packed_struct",
        "Both :: packed struct { a: u8, b: i64 align(8) }\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", sizeof(Both))\n\
         \x20   0\n}\n",
        "a `packed struct` pads no field",
    ),
    // A compile-time call is worked out by running it, so one that reaches
    // itself has no answer to work out.
    (
        "a_compile_time_call_that_reaches_itself",
        "import \"io.frost\"\n\
         loops :: fn(n: i64) -> i64 { if (n <= 0) { return 0 } loops(n - 1) + 1 }\n\
         DEEP :: loops(4)\n\
         main :: fn() -> i64 { print(\"{}\\n\", DEEP) 0 }\n",
        "reaches itself",
    ),
    // What a compile-time call may do is arithmetic over its arguments. A call
    // into the world has nothing to answer with before the program runs, and
    // the name it stops at is the one that reads it.
    (
        "a_compile_time_call_that_reads_the_world",
        "import \"io.frost\"\n\
         noisy :: fn(n: i64) -> i64 { print(\"{}\\n\", n) n }\n\
         SAID :: noisy(4)\n\
         main :: fn() -> i64 { print(\"{}\\n\", SAID) 0 }\n",
        "has no value at compile time",
    ),
    // A body may loop, so how long one takes is not read off the text. The
    // bound is what says a compile finishes.
    (
        "a_compile_time_call_that_never_ends",
        "import \"io.frost\"\n\
         spin :: fn(n: i64) -> i64 {\n\
         \x20   var i : i64 = 0\n\
         \x20   while (i >= 0) { i = i + 1 }\n\
         \x20   i\n}\n\
         FOREVER :: spin(1)\n\
         main :: fn() -> i64 { print(\"{}\\n\", FOREVER) 0 }\n",
        "took more than 1000000 steps",
    ),
    // A call is run where it is written, so every argument has to be known
    // there. A generic binds its size parameter at the instantiation, which is
    // later, so a call over one is refused and the parameter is named.
    (
        "a_compile_time_call_over_a_size_parameter",
        "import \"io.frost\"\n\
         twice :: fn(n: i64) -> i64 { n * 2 }\n\
         Holder :: struct($N: usize) { cells: [twice(N)]i64 }\n\
         main :: fn() -> i64 {\n\
         \x20   var h : Holder<4> = Holder<4> { cells = [0; 8] }\n\
         \x20   print(\"{}\\n\", slice_len(h.cells))\n\
         \x20   0\n}\n",
        "has no value at compile time",
    ),
    // An index is checked where it is written. Reading past the end of a run
    // the compiler worked out is a compile error naming the index and the
    // length, not an abort the program was going to reach.
    (
        "a_compile_time_index_past_the_end",
        "import \"io.frost\"\n\
         TABLE :: [1, 2, 4, 8]\n\
         OUT :: TABLE[9]\n\
         main :: fn() -> i64 { print(\"{}\\n\", OUT) 0 }\n",
        "this reads item 9 of a run of 4, whose items are numbered 0 to 3",
    ),
    // A field that is not there. Every field is named at the literal, so what
    // a field reads is decided without a layout, and a name nothing there
    // carries is said where it is written.
    (
        "a_compile_time_field_that_is_not_there",
        "import \"io.frost\"\n\
         Point :: struct { x: i64, y: i64 }\n\
         ORIGIN :: Point { x = 3, y = 4 }\n\
         DEPTH :: ORIGIN.z\n\
         main :: fn() -> i64 { print(\"{}\\n\", DEPTH) 0 }\n",
        "this has no field called 'z'",
    ),
    // A vector's lanes are a register's worth, so the length is a power of two.
    (
        "a_vector_whose_length_is_not_a_power_of_two",
        "import \"io.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   a : [3]f32 = [1.0, 2.0, 3.0]\n\
         \x20   b : [3]f32 = [1.0, 2.0, 3.0]\n\
         \x20   c := a + b\n\
         \x20   print(\"{}\\n\", cast($f64, c[0]))\n\
         \x20   0\n}\n",
        "elementwise arithmetic is over a vector whose length is a power of two",
    ),
    // Past a register's width the operation is a loop, and a loop the reader
    // does not see written down is what an operator may not be.
    (
        "a_vector_wider_than_a_register",
        "import \"io.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   a : [32]f64 = [1.0; 32]\n\
         \x20   b : [32]f64 = [2.0; 32]\n\
         \x20   c := a + b\n\
         \x20   print(\"{}\\n\", c[0])\n\
         \x20   0\n}\n",
        "elementwise arithmetic is over a vector of at most 64 bytes",
    ),
    // Two vectors of different types have no lane-for-lane meaning, and the
    // diagnostic names both by their length and element rather than by
    // whichever spelling each compiler's type renderer reaches for.
    (
        "two_vectors_of_different_types",
        "import \"io.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   a : [4]f32 = [1.0, 2.0, 3.0, 4.0]\n\
         \x20   b : [4]f64 = [1.0, 2.0, 3.0, 4.0]\n\
         \x20   c := a + b\n\
         \x20   print(\"{}\\n\", cast($f64, c[0]))\n\
         \x20   0\n}\n",
        "a vector of 4 f32 and a vector of 4 f64 do not go together",
    ),
    // A vector of floats takes the four arithmetic operators. The bitwise ones
    // are a question about bits, which a float does not answer.
    (
        "a_bitwise_operator_over_a_vector_of_floats",
        "import \"io.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   a : [4]f32 = [1.0, 2.0, 3.0, 4.0]\n\
         \x20   b : [4]f32 = [1.0, 2.0, 3.0, 4.0]\n\
         \x20   c := a & b\n\
         \x20   print(\"{}\\n\", cast($f64, c[0]))\n\
         \x20   0\n}\n",
        "'&' is not something two vectors answer",
    ),
    // What a comparison answers is one yes or no. A vector of them is a mask,
    // which is a type this language does not have, so two vectors are compared
    // nowhere.
    (
        "two_vectors_compared",
        "import \"io.frost\"\n\
         main :: fn() -> i64 {\n\
         \x20   a : [4]f32 = [1.0, 2.0, 3.0, 4.0]\n\
         \x20   b : [4]f32 = [1.0, 2.0, 3.0, 4.0]\n\
         \x20   if (a == b) { print(\"{}\\n\", 1) }\n\
         \x20   0\n}\n",
        "'==' is not something two vectors answer",
    ),
    // Two variants hold two shapes, so a name reading a field out of an
    // alternative would mean one thing in one alternative and another in the
    // next. The arm that binds gets an arm of its own.
    (
        "an_alternative_that_binds_a_payload",
        "import \"io.frost\"
         Side :: enum { Left { a: i64 }, Right { b: i64 } }
         main :: fn() -> i64 {
             s := Side::Left { a = 1 }
             match s {
                 case .Left { a } | .Right: a
                 case _: 0
             }
         }
",
        "an alternative binding payload fields holds one name to two shapes",
    ),
    // An alternative that covers everything leaves the others saying nothing,
    // which is an arm misstating what it covers rather than an arm that works.
    (
        "a_catch_all_among_alternatives",
        "import \"io.frost\"
         main :: fn() -> i64 {
             match 3 {
                 case 1 | _: 0
                 case _: 1
             }
         }
",
        "this alternative covers everything on its own",
    ),
    // One arm naming one pattern twice.
    (
        "an_alternative_written_twice",
        "import \"io.frost\"
         Side :: enum { Left, Right }
         main :: fn() -> i64 {
             s := Side::Left
             match s {
                 case .Left | .Left: 0
                 case _: 1
             }
         }
",
        "this alternative repeats one the same case already names",
    ),
    // A span that runs backwards, and one whose two ends meet under the
    // half-open spelling. Both cover nothing, and an arm covering nothing is a
    // mistake to name where it was written.
    (
        "a_case_range_that_runs_backwards",
        "import \"io.frost\"
         main :: fn() -> i64 {
             match 3 {
                 case 10..3: 0
                 case _: 1
             }
         }
",
        "the case range 10..3 covers nothing",
    ),
    (
        "a_case_range_whose_ends_meet",
        "import \"io.frost\"
         main :: fn() -> i64 {
             match 3 {
                 case 4..4: 0
                 case _: 1
             }
         }
",
        "the case range 4..4 covers nothing",
    ),
    // What a span covers is counted, and a count over the reals is not one.
    // What a `case` covers is a set a reader can count. A decimal covers one of
    // the reals, which is a claim nobody can act on, and text is compared
    // rather than counted; both belong in an `if`. Both compilers used to
    // disagree here in the worst way: the bootstrap matched a decimal and the
    // self-hosted one read the arm as the one covering the rest, so
    // `case 1.5:` ran for every value there and for one value here.
    (
        "a_decimal_in_a_case",
        "import \"io.frost\"
         f :: fn(x: f64) -> i64 {
             match x {
                 case 1.5: 10
                 case _: 20
             }
         }
         main :: fn() -> i64 { print(\"{}\\n\", f(1.5)) 0 }
",
        "a case matches whole numbers, booleans and variants, so a decimal belongs in an `if`",
    ),
    (
        "a_case_range_with_a_decimal_end",
        "import \"io.frost\"
         main :: fn() -> i64 {
             match 3 {
                 case 1.0..2.0: 0
                 case _: 1
             }
         }
",
        "a case matches whole numbers, booleans and variants, so a decimal belongs in an `if`",
    ),
    (
        "text_in_a_case",
        "import \"io.frost\"
         f :: fn(x: str) -> i64 {
             match x {
                 case \"hi\": 10
                 case _: 20
             }
         }
         main :: fn() -> i64 { print(\"{}\\n\", f(\"hi\")) 0 }
",
        "a case matches whole numbers, booleans and variants, so text belongs in an `if`",
    ),
    // A name in a case is the value it stands for. It used to bind whatever
    // was matched, which made `case CH_0:` and `case CH_0..=CH_9:` mean
    // opposite things: the second compared and the first quietly did not.
    (
        "a_name_in_a_case_that_names_no_constant",
        "import \"io.frost\"
         f :: fn(x: i64) -> i64 {
             match x {
                 case 1: 10
                 case n: n * 100
             }
         }
         main :: fn() -> i64 { print(\"{}\\n\", f(7)) 0 }
",
        "a name in a case is the value it stands for, and this one names no constant",
    ),
    (
        "a_name_in_a_tuple_case_that_names_no_constant",
        "import \"io.frost\"
         f :: fn(a: i64, b: i64) -> i64 {
             match (a, b) {
                 case (0, n): n
                 case (_, _): 99
             }
         }
         main :: fn() -> i64 { print(\"{}\\n\", f(0, 3)) 0 }
",
        "a name in a case is the value it stands for, and this one names no constant",
    ),
    // `_` covers everything, so an arm below one is refused by the rule about
    // coverage rather than by a rule of its own. A tuple arm naming `_` in
    // every part is the same arm.
    (
        "a_case_after_one_that_covers_everything",
        "import \"io.frost\"
         f :: fn(x: i64) -> i64 {
             match x {
                 case _: 10
                 case 1: 20
             }
         }
         main :: fn() -> i64 { print(\"{}\\n\", f(1)) 0 }
",
        "this case is covered by an earlier one, so nothing reaches it",
    ),
    (
        "a_tuple_case_after_one_that_covers_everything",
        "import \"io.frost\"
         f :: fn(a: i64, b: i64) -> i64 {
             match (a, b) {
                 case (_, _): 10
                 case (1, 1): 20
             }
         }
         main :: fn() -> i64 { print(\"{}\\n\", f(1, 1)) 0 }
",
        "this case is covered by an earlier one, so nothing reaches it",
    ),
    // Two earlier arms cover this one between them, and neither on its own.
    // That is the question a reader asks looking down the arms, so it is the
    // one both compilers answer.
    (
        "a_case_two_earlier_spans_cover_between_them",
        "import \"io.frost\"
         f :: fn(x: i64) -> i64 {
             match x {
                 case 1..5: 1
                 case 5..10: 2
                 case 3..7: 3
                 case _: 0
             }
         }
         main :: fn() -> i64 { print(\"{}\\n\", f(1)) 0 }
",
        "this case is covered by an earlier one, so nothing reaches it",
    ),
    (
        "a_case_range_ending_at_a_name_that_is_not_a_number",
        "import \"io.frost\"
         main :: fn() -> i64 {
             match 3 {
                 case 1..zzz: 0
                 case _: 1
             }
         }
",
        "a case range runs between whole numbers, and this bound is not one",
    ),
    // A tuple case compares one value per part, so a part naming a set rather
    // than a value has nothing to compare against.
    (
        "an_alternative_inside_a_tuple_case",
        "import \"io.frost\"
         main :: fn() -> i64 {
             match (1, 2) {
                 case (1 | 2, 3): 0
                 case _: 1
             }
         }
",
        "a tuple case compares one value per part",
    ),
    (
        "a_range_inside_a_tuple_case",
        "import \"io.frost\"
         main :: fn() -> i64 {
             match (1, 2) {
                 case (1..5, 3): 0
                 case _: 1
             }
         }
",
        "a tuple case compares one value per part",
    ),
    // An arm every value of which an earlier arm already takes. Read one span
    // against one span: an arm two earlier spans cover between them, and
    // neither on its own, goes on standing.
    (
        "a_case_an_earlier_span_covers",
        "import \"io.frost\"
         main :: fn() -> i64 {
             match 3 {
                 case 1..10: 0
                 case 5: 2
                 case _: 1
             }
         }
",
        "this case is covered by an earlier one, so nothing reaches it",
    ),
    (
        "a_case_an_earlier_alternative_covers",
        "import \"io.frost\"
         Side :: enum { Left, Right }
         main :: fn() -> i64 {
             s := Side::Left
             match s {
                 case .Left | .Right: 0
                 case .Left: 2
             }
         }
",
        "this case is covered by an earlier one, so nothing reaches it",
    ),
    // A span never removes the need for a `case _`. What the spans leave out is
    // not something either compiler counts, so the arm naming the rest is what
    // says the match is finished.
    (
        "a_match_over_spans_still_needs_the_rest",
        "import \"io.frost\"
         Side :: enum { Left, Right, Up }
         main :: fn() -> i64 {
             s := Side::Left
             match s {
                 case .Left | .Right: 0
             }
         }
",
        "does not cover '.Up'",
    ),
    // The region check, pinned here rather than only as "some refusal": the two
    // compilers wording an escape differently is how a walk that had stopped
    // looking inside a struct went unnoticed on one side.
    //
    // A pointer carved out of the arena, written into a binding the `with`
    // block outlives.
    (
        "an_arena_pointer_stored_outside_its_block",
        ARENA_PRELUDE_ESCAPE,
        "region: a pointer into arena 'arena' escapes its region by being \
         stored outside it; it may not outlive the arena",
    ),
    // The same pointer, one field down. A struct built in the region carries
    // the arena's storage wherever it goes, so the literal is the escape.
    (
        "an_arena_pointer_inside_a_struct_literal",
        ARENA_PRELUDE_STRUCT,
        "region: a pointer into arena 'arena' escapes its region by being \
         stored outside it; it may not outlive the arena",
    ),
    // And read back out of one. The struct stays in the region; what leaves is
    // the field, and the field is the storage.
    (
        "an_arena_pointer_read_out_of_a_struct",
        ARENA_PRELUDE_FIELD,
        "region: a pointer into arena 'arena' escapes its region by being \
         stored outside it; it may not outlive the arena",
    ),
    // A `uses` function may hand a pointer back to its caller, whose own `with`
    // block is checked, but not write one into a parameter: that frame outlives
    // the call.
    (
        "an_arena_pointer_stored_into_a_parameter",
        "Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
         Holder :: struct { p: ^i64, count: i64 }
         alloc_int :: fn(mut a: Arena<256>) -> ^i64 {
             slot := ptr_to(a.data[a.offset])
             a.offset = a.offset + 8
             unsafe { ptr_cast($i64, slot) }
         }
         stash :: fn(mut h: Holder) -> i64 uses Arena<256> {
             h.p = alloc_int(arena)
             0
         }
         main :: fn() -> i64 {
             var arena : Arena<256> = Arena { data = [0; 256], offset = 0 }
             var sink : i64 = 0
             var held : Holder = Holder { p = ptr_to(sink), count = 0 }
             with arena { sink = stash(held) }
             sink
         }
",
        "region: a pointer into arena 'arena' escapes its region by being \
         stored into a parameter; it may not outlive the arena",
    ),
    // The value a `with` block ends with flows to the enclosing scope, which is
    // the third way out beside a store and a return.
    (
        "an_arena_pointer_as_the_blocks_value",
        "Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
         alloc_int :: fn(mut a: Arena<256>) -> ^i64 {
             slot := ptr_to(a.data[a.offset])
             a.offset = a.offset + 8
             unsafe { ptr_cast($i64, slot) }
         }
         main :: fn() -> i64 {
             var arena : Arena<256> = Arena { data = [0; 256], offset = 0 }
             var sink : i64 = 0
             with arena {
                 alloc_int(arena)
             }
             sink
         }
",
        "region: a pointer into arena 'arena' escapes its region by being the \
         block's value; it may not outlive the arena",
    ),
    // The container the arena feeds. `Fixed<T>` holds a view of the run it was
    // given, so one built inside the block is the arena's storage travelling
    // under a struct's name, and it leaves the block the same way the bare
    // pointer does.
    (
        "a_container_over_an_arena_run_leaving_its_block",
        "import \"arena.frost\"
         import \"fixed.frost\"
         Sprite :: struct { x: i64 }
         main :: fn() -> i64 {
             var bytes : [256]u8 = [0; 256]
             var scratch := arena_over(bytes)
             var backing : [1]Sprite = [Sprite { x = 0 }]
             var escaped := fixed_over(backing)
             with scratch {
                 run := arena_carve($Sprite, scratch, 4)
                 escaped = fixed_over(run)
             }
             fixed_len(escaped)
         }
",
        "region: a pointer into arena 'scratch' escapes its region by being \
         stored outside it; it may not outlive the arena",
    ),
    // An accessor answering with a view of the run its container holds is the
    // shape every container is built on, and calling one on a local is honest:
    // the run belongs to whoever carved it. One that hands back a view of the
    // container's own bytes is not, and the difference is the field's type.
    (
        "an_accessor_answering_with_the_containers_own_bytes",
        "Holder :: struct($T: Type) { storage: []T, len: i64 }
         hold :: fn($T: Type, storage: []T) -> Holder<T> {
             Holder { storage = storage, len = 3 }
         }
         bad :: fn($T: Type, h: Holder<T>) -> []i64 {
             unsafe { slice_from($i64, ptr_to(h.len), 1) }
         }
         gather :: fn(run: []i64) -> []i64 {
             var kept := hold(run)
             bad(kept)
         }
         main :: fn() -> i64 {
             var backing : [4]i64 = [1, 2, 3, 4]
             gather(backing)[0]
         }
",
        "region: a pointer into the frame of 'gather' is the call's answer; \
         the storage it names dies when the call returns",
    ),
    // A compile-time parameter is written at the call, one `$` argument each.
    // A compile-time parameter nothing else names is written at the call, and
    // leaving it out lines every value argument up against the parameter beside
    // the one it was written for. That is a mistake worth naming as the count
    // it is rather than leaving to whatever the shift runs into.
    (
        "a_call_leaving_out_a_compile_time_argument",
        "Box :: struct($N: usize) { room: [N]i64, offset: i64 }
         fresh :: fn($N: usize, offset: i64) -> Box<N> {
             Box { room = [0; N], offset = offset }
         }
         main :: fn() -> i64 {
             var b := fresh(8)
             b.offset
         }
",
        "generic function 'fresh' expects 2 argument(s) but 1 were given",
    ),
    // The frame half of the same file, for the same reason: what a call answers
    // with may not name storage the call owns.
    (
        "a_frame_pointer_as_the_calls_answer",
        "grab :: fn() -> ^i64 {
             var x : i64 = 5
             ptr_to(x)
         }
         main :: fn() -> i64 { 0 }
",
        "region: a pointer into the frame of 'grab' is the call's answer; the \
         storage it names dies when the call returns",
    ),
    // The runtime is linked into every program and keeps the names it was
    // written under, so a program defining one of them would replace what every
    // program calls. The runtime is the one file that may, and it is the one
    // file the compiler resolved as the runtime.
    (
        "a_definition_in_the_runtimes_name_space",
        "frost_rt_check_index :: extern fn(index: i64, length: i64) -> i64 {
             index
         }
         main :: fn() -> i64 { 0 }
",
        "'frost_rt_check_index' keeps the name it is written under, and \
         'frost_rt_' and 'frost_u_' are the runtime's and the compiler's own, \
         so a definition here would replace what every program calls",
    ),
    // A length is read off a run, so what it is asked about has to be one. A
    // struct has none beside it, and reading one off it takes its first word
    // for the count and then bounds-checks every access through the view
    // against whatever happened to be there.
    (
        "a_length_asked_of_a_struct",
        "import \"io.frost\"
         Held :: struct { text: str, count: i64 }
         main :: fn() -> i64 {
             held := Held { text = \"hello\", count = 3 }
             print(\"{}\\n\", str_len(held))
             0
         }
",
        "expected a str value, found Held",
    ),
    // `str_len` asks for a run of bytes. A run of something else has a length,
    // which is what `slice_len` is for, but it is not text.
    (
        "a_text_length_asked_of_a_run_of_numbers",
        "import \"io.frost\"
         look :: fn(view: []i64) -> i64 { str_len(view) }
         main :: fn() -> i64 {
             var run: [4]i64 = [1; 4]
             print(\"{}\\n\", look(run))
             0
         }
",
        "expected a str value, found []i64",
    ),
    // A pointer is a run's address with nothing beside it saying how long the
    // run is, so no length can be read off one.
    (
        "a_text_length_asked_of_a_pointer",
        "import \"io.frost\"
         look :: fn(raw: ^i8) -> i64 { str_len(raw) }
         main :: fn() -> i64 { print(\"{}\\n\", unsafe { look(\"hi\") })  0 }
",
        "expected a str value, found ^i8",
    ),
    (
        "a_slice_length_asked_of_a_struct",
        "import \"io.frost\"
         Held :: struct { text: str, count: i64 }
         main :: fn() -> i64 {
             held := Held { text = \"hello\", count = 3 }
             print(\"{}\\n\", slice_len(held))
             0
         }
",
        "expected a slice value, found Held",
    ),
    // The compiler's own name space is refused by the same rule and in the same
    // words, since a C backend names an ordinary function into it.
    (
        "a_definition_in_the_compilers_name_space",
        "frost_u_helper :: extern fn(value: i64) -> i64 {
             value + 1
         }
         main :: fn() -> i64 { 0 }
",
        "'frost_u_helper' keeps the name it is written under, and \
         'frost_rt_' and 'frost_u_' are the runtime's and the compiler's own, \
         so a definition here would replace what every program calls",
    ),
];

// One arena, one carve, and the three ways the pointer leaves the block. Held
// apart from the rows so the three read as the one program they are.
const ARENA_PRELUDE_ESCAPE: &str =
    "Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
         alloc_int :: fn(mut a: Arena<256>) -> ^i64 {
             slot := ptr_to(a.data[a.offset])
             a.offset = a.offset + 8
             unsafe { ptr_cast($i64, slot) }
         }
         main :: fn() -> i64 {
             var arena : Arena<256> = Arena { data = [0; 256], offset = 0 }
             var sink : i64 = 0
             var escaped : ^i64 = ptr_to(sink)
             with arena {
                 escaped = alloc_int(arena)
             }
             unsafe { escaped^ }
         }
";

const ARENA_PRELUDE_STRUCT: &str =
    "Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
         Holder :: struct { p: ^i64, count: i64 }
         alloc_int :: fn(mut a: Arena<256>) -> ^i64 {
             slot := ptr_to(a.data[a.offset])
             a.offset = a.offset + 8
             unsafe { ptr_cast($i64, slot) }
         }
         main :: fn() -> i64 {
             var arena : Arena<256> = Arena { data = [0; 256], offset = 0 }
             var sink : i64 = 0
             var held : Holder = Holder { p = ptr_to(sink), count = 0 }
             with arena {
                 held = Holder { p = alloc_int(arena), count = 1 }
             }
             unsafe { held.p^ }
         }
";

const ARENA_PRELUDE_FIELD: &str =
    "Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
         Holder :: struct { p: ^i64, count: i64 }
         alloc_int :: fn(mut a: Arena<256>) -> ^i64 {
             slot := ptr_to(a.data[a.offset])
             a.offset = a.offset + 8
             unsafe { ptr_cast($i64, slot) }
         }
         main :: fn() -> i64 {
             var arena : Arena<256> = Arena { data = [0; 256], offset = 0 }
             var sink : i64 = 0
             var escaped : ^i64 = ptr_to(sink)
             with arena {
                 inner := Holder { p = alloc_int(arena), count = 1 }
                 escaped = inner.p
             }
             unsafe { escaped^ }
         }
";

#[test]
fn both_compilers_warn_about_the_same_programs() {
    let Some(compiler) = build_self_hosted_compiler("warnboth") else {
        return;
    };
    let directory = std::env::temp_dir();
    let mut drifted = Vec::new();
    for (name, source, wanted) in WARNED_BY_BOTH {
        // One file, handed to both. What a report calls a file is compared
        // below, and two files named differently cannot answer that.
        let input = directory.join(format!("frost_warn_{name}.frost"));
        std::fs::write(&input, source).unwrap();
        let (built, bootstrap) = bootstrap_report_at(name, &input);
        assert!(
            built,
            "the bootstrap refused {name}, which warns:
{bootstrap}"
        );
        assert!(
            bootstrap.contains(wanted),
            "the bootstrap did not warn '{wanted}' about {name}:
{bootstrap}"
        );
        let run = Command::new(&compiler)
            .env("FROST_INPUT", &input)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        let hosted = String::from_utf8_lossy(&run.stderr);
        assert!(
            run.status.success(),
            "the self-hosted compiler refused {name}, which warns:
{hosted}"
        );
        assert!(
            hosted.contains(wanted),
            "the self-hosted compiler did not warn '{wanted}' about {name}:
{hosted}"
        );
        let said = spoken(&bootstrap);
        let hosted_said = spoken(&hosted);
        if said != hosted_said {
            drifted.push(format!(
                "{name}
  bootstrap: {}
  self-hosted: {}",
                said.join(" | "),
                hosted_said.join(" | ")
            ));
        }
        assert_eq!(
            named_files(&bootstrap),
            named_files(&hosted),
            "the two compilers call the file different things in {name}"
        );
    }
    assert!(
        drifted.is_empty(),
        "the two compilers warn about these differently ({} of {}):
{}",
        drifted.len(),
        WARNED_BY_BOTH.len(),
        drifted.join(
            "
"
        )
    );
}

#[test]
fn both_compilers_refuse_the_same_programs() {
    let Some(compiler) = build_self_hosted_compiler("refuseboth") else {
        return;
    };
    let directory = std::env::temp_dir();
    let mut drifted = Vec::new();
    for (name, source, wanted) in REFUSED_BY_BOTH {
        // One file, handed to both. What a report calls a file is compared
        // below, and two files named differently cannot answer that.
        let input = directory.join(format!("frost_both_{name}.frost"));
        std::fs::write(&input, source).unwrap();
        let (built, bootstrap) = bootstrap_report_at(name, &input);
        assert!(
            !built,
            "the bootstrap accepted {name}, which it should refuse"
        );
        assert!(
            bootstrap.contains(wanted),
            "the bootstrap did not say '{wanted}' about {name}:
{bootstrap}"
        );
        let run = Command::new(&compiler)
            .env("FROST_INPUT", &input)
            .output()
            .unwrap();
        let _ = std::fs::remove_file(&input);
        let hosted = String::from_utf8_lossy(&run.stderr);
        assert!(
            !run.status.success(),
            "the self-hosted compiler built {name}, which the bootstrap refuses"
        );
        assert!(
            hosted.contains(wanted),
            "the self-hosted compiler did not say '{wanted}' about {name}:\n{hosted}"
        );
        // Containing the phrase is not saying the same thing. Two compilers can
        // both carry it and differ in every word around it, which is how one
        // came to quote a name the other left bare and to explain a leak in
        // another sentence entirely. What each said is what is compared.
        let said = spoken(&bootstrap);
        let hosted_said = spoken(&hosted);
        if said != hosted_said {
            drifted.push(format!(
                "{name}\n  bootstrap: {}\n  self-hosted: {}",
                said.join(" | "),
                hosted_said.join(" | ")
            ));
        }
        assert_eq!(
            named_files(&bootstrap),
            named_files(&hosted),
            "the two compilers call the file different things in {name}"
        );
    }
    let unexpected: Vec<&String> = drifted
        .iter()
        .filter(|report| {
            let name = report.split('\n').next().unwrap_or("");
            !WORDED_DIFFERENTLY.contains(&name)
        })
        .collect();
    assert!(
        unexpected.is_empty(),
        "the two compilers refuse these for different words, and nothing said \
         they would ({} of {}):\n{}",
        unexpected.len(),
        REFUSED_BY_BOTH.len(),
        unexpected
            .iter()
            .map(|held| held.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
    let mended: Vec<&str> = WORDED_DIFFERENTLY
        .iter()
        .filter(|name| {
            !drifted
                .iter()
                .any(|report| report.split('\n').next() == Some(name))
        })
        .copied()
        .collect();
    assert!(
        mended.is_empty(),
        "these are worded the same now, so take them off the list:\n{}",
        mended.join("\n")
    );
}

// The refusals the two compilers word differently. Empty, and it is the test
// above that keeps it so: a pair that drifts fails on the way in, and one that
// is mended fails until its name comes off. Written down after the harness
// learned to compare what was said rather than to look for a phrase inside it,
// which is what let forty of these drift unseen.
const WORDED_DIFFERENTLY: &[&str] = &[];

// The file each header names, which is what a report calls a file rather than
// where the file is.
//
// A name stays what it is whoever compiles. One compiler wrote the path it
// resolved instead, so the same fault about the same file read `std/mem.frost`
// from one and an absolute path from the other, and nothing compared them
// because `spoken` keeps the claim and drops the header the name sits in. The
// line and the column are left out: the two count a fault's place from what
// they were reading when they found it, and where each points is a question of
// its own.
fn named_files(report: &str) -> Vec<String> {
    report
        .lines()
        .filter_map(|line| line.strip_suffix(':'))
        .filter_map(|line| line.rsplit_once(':'))
        .filter_map(|(head, _)| head.rsplit_once(':'))
        .map(|(path, _)| path.to_string())
        .collect()
}

// What a compiler said, apart from where it said it. A diagnostic is a header
// naming the position, the line it is about, and a caret with the words after
// it. The words are the claim; the rest is the place, which the two count from
// their own file tables and spell their own way.
fn spoken(report: &str) -> Vec<String> {
    report
        .lines()
        .filter_map(|line| line.split_once("^ "))
        .map(|(_, said)| said.trim_end().to_string())
        .collect()
}

// Where the range ends is one answer, and two things have to give it: the fold
// that runs a call before the program does, and the arithmetic the program runs.
// Each row is an operation over two written numbers, with what it answers where
// it stays inside the range and nothing where it leaves. Each is compiled twice:
// once as a constant, which the fold settles, and once over values the compiler
// cannot see through, which the machine settles. A row where the two disagree is
// a program that builds and then aborts, or one refused for arithmetic that
// would have held.
//
// The pairs straddle the line on both sides on purpose: every row that leaves
// the range has a neighbour one step inside it.
const RANGE_EDGE: &[(&str, &str, &str, &str, Option<i64>)] = &[
    (
        "add_at_the_top",
        "a + b",
        "9223372036854775807",
        "0",
        Some(i64::MAX),
    ),
    (
        "add_past_the_top",
        "a + b",
        "9223372036854775807",
        "1",
        None,
    ),
    (
        "add_at_the_bottom",
        "a + b",
        "-9223372036854775807 - 1",
        "0",
        Some(i64::MIN),
    ),
    (
        "add_past_the_bottom",
        "a + b",
        "-9223372036854775807 - 1",
        "-1",
        None,
    ),
    (
        "subtract_at_the_bottom",
        "a - b",
        "-9223372036854775807",
        "1",
        Some(i64::MIN),
    ),
    (
        "subtract_past_the_bottom",
        "a - b",
        "-9223372036854775807 - 1",
        "1",
        None,
    ),
    (
        "multiply_inside",
        "a * b",
        "4611686018427387903",
        "2",
        Some(9223372036854775806),
    ),
    (
        "multiply_past_the_top",
        "a * b",
        "4611686018427387904",
        "2",
        None,
    ),
    (
        "multiply_the_bottom_by_one",
        "a * b",
        "-9223372036854775807 - 1",
        "1",
        Some(i64::MIN),
    ),
    (
        "multiply_the_bottom_by_minus_one",
        "a * b",
        "-9223372036854775807 - 1",
        "-1",
        None,
    ),
    (
        "divide_the_bottom_by_one",
        "a / b",
        "-9223372036854775807 - 1",
        "1",
        Some(i64::MIN),
    ),
    (
        "divide_the_bottom_by_minus_one",
        "a / b",
        "-9223372036854775807 - 1",
        "-1",
        None,
    ),
    ("divide_by_a_number", "a / b", "7", "2", Some(3)),
    ("divide_by_nothing", "a / b", "7", "0", None),
    ("remainder_by_a_number", "a % b", "7", "2", Some(1)),
    ("remainder_by_nothing", "a % b", "7", "0", None),
    (
        "remainder_of_the_bottom_by_minus_one",
        "a % b",
        "-9223372036854775807 - 1",
        "-1",
        Some(0),
    ),
    ("shift_to_the_sign_bit", "a << b", "1", "63", Some(i64::MIN)),
    ("shift_past_the_width", "a << b", "1", "64", None),
    ("shift_right_inside", "a >> b", "1024", "3", Some(128)),
    ("shift_right_past_the_width", "a >> b", "1024", "64", None),
];

fn range_edge_folded(operation: &str, left: &str, right: &str) -> String {
    format!(
        "import \"io.frost\"\n\
         step :: fn(a: i64, b: i64) -> i64 {{ {operation} }}\n\
         EDGE :: step({left}, {right})\n\
         main :: fn() -> i64 {{ print(\"{{}}\\n\", EDGE) 0 }}\n"
    )
}

fn range_edge_run(operation: &str, left: &str, right: &str) -> String {
    format!(
        "import \"io.frost\"\n\
         step :: fn(a: i64, b: i64) -> i64 {{ {operation} }}\n\
         main :: fn() -> i64 {{\n\
         \x20   var a : i64 = {left}\n\
         \x20   var b : i64 = {right}\n\
         \x20   print(\"{{}}\\n\", step(a, b))\n\
         \x20   0\n}}\n"
    )
}

#[test]
fn the_fold_and_the_machine_agree_about_where_the_range_ends() {
    if !linker_available() {
        return;
    }
    let hosted = build_self_hosted_compiler("rangeedge");
    let directory = std::env::temp_dir();
    for (name, operation, left, right, answer) in RANGE_EDGE {
        let folded = range_edge_folded(operation, left, right);
        let run = range_edge_run(operation, left, right);

        // What the fold says, on both compilers.
        let source = directory.join(format!("frost_edge_{name}.frost"));
        std::fs::write(&source, &folded).unwrap();
        let bootstrap = Command::new(env!("CARGO_BIN_EXE_frost"))
            .arg("-o")
            .arg(directory.join(format!("frost_edge_{name}.o")))
            .arg(&source)
            .output()
            .unwrap();
        assert_eq!(
            bootstrap.status.success(),
            answer.is_some(),
            "the bootstrap's fold put {name} on the wrong side of the line:\n{}",
            String::from_utf8_lossy(&bootstrap.stderr)
        );
        if let Some(compiler) = &hosted {
            let held = Command::new(compiler)
                .env("FROST_INPUT", &source)
                .output()
                .unwrap();
            assert_eq!(
                held.status.success(),
                answer.is_some(),
                "the self-hosted fold put {name} on the wrong side of the line:\n{}",
                String::from_utf8_lossy(&held.stderr)
            );
        }
        let _ = std::fs::remove_file(&source);

        // What the machine says, running the same arithmetic over values the
        // compiler cannot see through.
        let Some(output) = compile_and_run_unaudited_allowing_failure(
            &format!("edge{name}"),
            &run,
        ) else {
            continue;
        };
        match answer {
            Some(wanted) => assert_eq!(
                output.trim(),
                wanted.to_string(),
                "running {name} answered differently from the fold"
            ),
            None => assert!(
                output.is_empty(),
                "running {name} answered '{output}' where the fold refused it"
            ),
        }
    }
    if let Some(compiler) = hosted {
        let _ = std::fs::remove_file(compiler);
    }
}

// Build and run a program that is expected to abort, so the abort is the
// answer rather than a failure of the harness. Nothing on stdout is what an
// abort before the first print looks like.
fn compile_and_run_unaudited_allowing_failure(
    label: &str,
    source: &str,
) -> Option<String> {
    let directory = std::env::temp_dir();
    let input = directory.join(format!("frost_{label}.frost"));
    std::fs::write(&input, source).unwrap();
    let exe = directory
        .join(format!("frost_{label}{}", std::env::consts::EXE_SUFFIX));
    let built = Command::new(env!("CARGO_BIN_EXE_frost"))
        .arg("--link")
        .arg("-o")
        .arg(&exe)
        .arg(&input)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    assert!(
        built.status.success(),
        "{label} did not compile:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let ran = Command::new(&exe).output().unwrap();
    let _ = std::fs::remove_file(&exe);
    Some(String::from_utf8_lossy(&ran.stdout).replace("\r\n", "\n"))
}

// Programs whose meaning the two compilers used to disagree about. Each ran
// correctly under the bootstrap and was miscompiled or refused by the
// self-hosted one, which is the drift that matters: the language is whatever
// both compilers do, so a construct only one of them handles is a bug in
// whichever is wrong rather than a feature with a caveat.
const SAME_LANGUAGE_CASES: &[(&str, &str, &str)] = &[
    // A call's arguments happen in the order they are written. C sequences
    // neither a call's arguments nor an operator's operands, so both backends
    // that go through C read whatever runs something into a slot ahead of the
    // call, and the one that emits instructions works the arguments out
    // forwards and pushes them backwards.
    (
        "call_arguments_happen_in_the_order_they_are_written",
        "import \"io.frost\"\n\
         Counter :: struct { n: i64 }\n\
         bump :: fn(mut c: Counter) -> i64 { c.n = c.n + 10\n c.n }\n\
         two :: fn(a: i64, b: i64) -> i64 { a * 100 + b }\n\
         three :: fn(a: i64, b: i64, c: i64) -> i64 {\n\
         \x20   a * 10000 + b * 100 + c\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   var c := Counter { n = 0 }\n\
         \x20   print(\"{}\n\", two(bump(c), c.n))\n\
         \x20   var d := Counter { n = 0 }\n\
         \x20   print(\"{}\n\", two(d.n, bump(d)))\n\
         \x20   var e := Counter { n = 0 }\n\
         \x20   print(\"{}\n\", three(bump(e), bump(e), e.n))\n\
         \x20   var f := Counter { n = 0 }\n\
         \x20   print(\"{} {}\n\", bump(f), f.n)\n\
         \x20   0\n\
         }\n",
        "1010\n10\n102020\n10 10\n",
    ),
    // A carve through a capability bundle: the function that does the taking is
    // named at the call, so neither check can walk it, and what it answers with
    // is worth the arguments it was handed. The allocator here is the caller's,
    // so the run outlives the call and the program is one both compilers run.
    (
        "a_carve_through_a_bundle_answers_with_the_callers_storage",
        "import \"io.frost\"\nimport \"mem.frost\"\n\
         Bump :: struct { data: []u8, offset: i64 }\n\
         Allocation :: struct($A: Type) {\n\
         \x20   take: fn(mut A, i64, i64) -> []u8\n\
         }\n\
         bump_take :: fn(mut b: Bump, size: i64, align: i64) -> []u8 {\n\
         \x20   start := (b.offset + align - 1) / align * align\n\
         \x20   run := slice_range(b.data, start, size)\n\
         \x20   b.offset = start + size\n\
         \x20   run\n\
         }\n\
         bump_source :: Allocation<Bump> { take = bump_take }\n\
         carve :: fn(\n\
         \x20   $T: Type,\n\
         \x20   $A: Type,\n\
         \x20   $source: Allocation<A>,\n\
         \x20   mut a: A,\n\
         \x20   count: i64\n\
         ) -> []T {\n\
         \x20   run := source.take(a, count * sizeof(T), alignof(T))\n\
         \x20   bytes_as($T, ptr_to(run[0]), count)\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   var backing: [256]u8 = [0; 256]\n\
         \x20   var b := Bump { data = backing, offset = 0 }\n\
         \x20   got := carve($i64, $bump_source, b, 3)\n\
         \x20   got[0] = 7\n\
         \x20   got[2] = 9\n\
         \x20   print(\"{} {} {}\n\", slice_len(got), got[0], got[2])\n\
         \x20   print(\"{}\n\", b.offset)\n\
         \x20   0\n\
         }\n",
        "3 7 9\n24\n",
    ),
    // A generic type's parameters take defaults too, so an instance names the
    // arguments the writer cares about and the declaration says the rest. Both
    // spellings are one type, which the assignment between them is what proves.
    (
        "a_generic_type_fills_the_arguments_an_instance_leaves_out",
        "import \"io.frost\"\n\
         Heap :: struct { }\n\
         Bump :: struct { room: i64 }\n\
         Holder :: struct($T: Type, $A: Type = Heap) {\n\
         \x20   value: T,\n\
         \x20   where_from: A\n\
         }\n\
         widen :: fn($T: Type, $A: Type, h: Holder<T, A>) -> i64 {\n\
         \x20   sizeof(A) * 100 + h.value\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   var plain := Holder<i64> { value = 7, where_from = Heap { } }\n\
         \x20   var full: Holder<i64, Heap> = plain\n\
         \x20   var wide := Holder<i64, Bump> {\n\
         \x20       value = 9,\n\
         \x20       where_from = Bump { room = 1 }\n\
         \x20   }\n\
         \x20   print(\"{}\n\", widen(full))\n\
         \x20   print(\"{}\n\", widen(wide))\n\
         \x20   0\n\
         }\n",
        "7\n809\n",
    ),
    // A compile-time parameter with a default is written or left out, and the
    // call is aligned by the `$` rather than by counting: an argument carrying
    // one binds a compile-time parameter and every other binds a value
    // parameter. All three kinds of default in one program, since each rides in
    // the same slot and a reader has to see that they do.
    (
        "a_compile_time_parameter_may_stand_for_a_default",
        "import \"io.frost\"\n\
         Heap :: struct { }\n\
         Bump :: struct { room: i64, mark: i64 }\n\
         Sizing :: struct($A: Type) { room: fn(i64) -> i64 }\n\
         heap_room :: fn(count: i64) -> i64 { count * 8 }\n\
         bump_room :: fn(count: i64) -> i64 { count * 4 }\n\
         heap_sizing :: Sizing<Heap> { room = heap_room }\n\
         bump_sizing :: Sizing<Bump> { room = bump_room }\n\
         room_of :: fn(\n\
         \x20   $A: Type = Heap,\n\
         \x20   $sizing: Sizing<A> = heap_sizing,\n\
         \x20   $slack: usize = 2,\n\
         \x20   count: i64\n\
         ) -> i64 {\n\
         \x20   sizing.room(count) + slack + sizeof(A)\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\n\", room_of(3))\n\
         \x20   print(\"{}\n\", room_of($Bump, $bump_sizing, 3))\n\
         \x20   print(\"{}\n\", room_of($Bump, $bump_sizing, $10, 3))\n\
         \x20   0\n\
         }\n",
        "26\n30\n38\n",
    ),
    // A compile-time parameter a value parameter settles is read off that
    // argument, and one nothing else names is written. Both spellings in one
    // program, since what decides which is the signature and a reader has to be
    // able to see the two beside each other.
    (
        "a_settled_compile_time_parameter_is_read_off_the_argument",
        "import \"io.frost\"\n\
         Box :: struct($T: Type) { held: $T }\n\
         unwrap :: fn($T: Type, b: Box<T>) -> $T { b.held }\n\
         twice :: fn($T: Type, value: $T) -> $T { value }\n\
         first :: fn($T: Type, run: []$T) -> $T { run[0] }\n\
         widths :: fn($T: Type, count: i64) -> i64 { count * sizeof(T) }\n\
         main :: fn() -> i64 {\n\
         \x20   var b := Box<i64> { held = 41 }\n\
         \x20   var run: [3]i64 = [4, 5, 6]\n\
         \x20   var view: []i64 = run\n\
         \x20   print(\"{}\n\", unwrap(b))\n\
         \x20   print(\"{}\n\", twice(7))\n\
         \x20   print(\"{}\n\", first(view))\n\
         \x20   print(\"{}\n\", widths($i64, 3))\n\
         \x20   0\n\
         }\n",
        "41\n7\n4\n24\n",
    ),
    // A bundle in a container's type. `ops` is written where the container is
    // made and settled off it after, so the two containers below run different
    // hashes from one body and neither call names a bundle. What each compiler
    // has to agree on is that the constant reaches the specialization through
    // the instance rather than through an argument.
    (
        "a_bundle_travels_in_the_containers_type",
        "import \"io.frost\"\n\
         Hashing :: struct($K: Type) { hash: fn(K) -> i64 }\n\
         Bag :: struct($K: Type, $ops: Hashing<K>) { first: $K }\n\
         one :: fn(k: i64) -> i64 { k }\n\
         two :: fn(k: i64) -> i64 { k * 2 }\n\
         plain :: Hashing<i64> { hash = one }\n\
         doubled :: Hashing<i64> { hash = two }\n\
         bag_new :: fn($K: Type, $ops: Hashing<K>, first: $K) -> Bag<K, ops> {\n\
         \x20   Bag { first = first }\n\
         }\n\
         bag_hash :: fn($K: Type, $ops: Hashing<K>, b: Bag<K, ops>) -> i64 {\n\
         \x20   ops.hash(b.first)\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   var kept := bag_new($plain, 21)\n\
         \x20   var wide := bag_new($doubled, 21)\n\
         \x20   print(\"{}\n\", bag_hash(kept))\n\
         \x20   print(\"{}\n\", bag_hash(wide))\n\
         \x20   0\n\
         }\n",
        "21\n42\n",
    ),
    // `alignof` reads the layout each compiler worked out rather than asking
    // the backend, so a stated alignment reaches it and the two agree. The C
    // backend writes the number out for exactly this reason: the emitted struct
    // carries no attribute saying what Frost aligned it to, so `_Alignof` on it
    // would answer for a type C laid out its own way.
    (
        "alignof_reads_the_layout_the_compiler_made",
        "import \"io.frost\"\n\
         Wide :: struct { a: i64, b: i8 }\n\
         Narrow :: struct { a: i8, b: i8 }\n\
         Stated :: struct { x: f32, y: f32, z: f32, w: f32 align(16) }\n\
         through :: fn($T: Type) -> i64 { alignof(T) }\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{} {} {}\n\", alignof(i8), alignof(i32), alignof(i64))\n\
         \x20   print(\"{} {}\n\", alignof(Wide), alignof(Narrow))\n\
         \x20   print(\"{} {}\n\", alignof(Stated), sizeof(Stated))\n\
         \x20   print(\"{}\n\", through($Wide))\n\
         \x20   0\n\
         }\n",
        "1 4 8\n8 1\n16 32\n8\n",
    ),
    // An arena is a view of bytes somebody else owns, so a carve starts on what
    // the element is aligned to and the offset lands in the caller's arena.
    (
        "an_arena_carves_onto_the_element_alignment",
        "import \"io.frost\"\nimport \"arena.frost\"\n\
         Wide :: struct { v: f32 align(16) }\n\
         main :: fn() -> i64 {\n\
         \x20   var backing: [256]u8 = [0; 256]\n\
         \x20   var a := arena_over(backing)\n\
         \x20   var one := arena_carve($u8, a, 1)\n\
         \x20   one[0] = 3\n\
         \x20   var wide := arena_carve($Wide, a, 2)\n\
         \x20   print(\"{} {}\n\", arena_used(a), slice_len(wide))\n\
         \x20   print(\"{} {}\n\", one[0], arena_left(a))\n\
         \x20   arena_reset(a, 0)\n\
         \x20   print(\"{}\n\", arena_used(a))\n\
         \x20   0\n\
         }\n",
        "48 2\n3 208\n0\n",
    ),
    // One `print` writes every kind of value, with the writer for each chosen
    // while the body is expanded. The chain of predicates is four arms long,
    // which the self-hosted compiler refused until `parse_expansion_if` learned
    // to read `else if`: it called `parse_block` on whatever followed `else`,
    // so a chain longer than two arms died on the second one.
    (
        "one_print_writes_every_kind_of_value",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   print(\"none\n\")\n\
         \x20   print(\"{} {}\n\", 1, 2)\n\
         \x20   print(\"{}\n\", 2.5)\n\
         \x20   print(\"{}\n\", \"text\")\n\
         \x20   print(\"{} {}\n\", true, false)\n\
         \x20   print(\"{}\", 7)\n\
         \x20   print(\"\n\")\n\
         \x20   0\n\
         }\n",
        "none\n1 2\n2.5\ntext\n1 0\n7\n",
    ),
    // The edges of the rule, where a run has no bytes on one side of a hole or
    // none at all. Nothing is written for an empty run, and a call that names
    // no hole and gives no value is a call that writes its literal and stops.
    (
        "a_format_string_at_its_edges",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   print(\"\")\n\
         \x20   print(\"{}{}{}\", 1, 2, 3)\n\
         \x20   print(\"\n\")\n\
         \x20   print(\"{}\", \"\")\n\
         \x20   print(\"[{}]\n\", \"\")\n\
         \x20   print(\"{}{}\n\", \"\", 4)\n\
         \x20   0\n\
         }\n",
        "123\n[]\n4\n",
    ),
    // A doubled brace against a hole, which is where the walk has to decide
    // twice in a row and the counters have to agree with it.
    (
        "doubled_braces_meet_a_hole",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   print(\"{{{}}}\n\", 1)\n\
         \x20   print(\"{{{}\n\", 2)\n\
         \x20   print(\"{}}}\n\", 3)\n\
         \x20   print(\"}}{}{{\n\", 4)\n\
         \x20   print(\"{{}}{}\n\", 5)\n\
         \x20   0\n\
         }\n",
        "{1}\n{2\n3}\n}4{\n{}5\n",
    ),
    // Every arm of the chain the body expands, each asked for on its own so a
    // failure names which one. A `bool` answers no to every predicate and is
    // what the last two arms are for.
    (
        "each_arm_of_the_writer_chain",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   var f: f32 = 1.5\n\
         \x20   var d: f64 = 2.5\n\
         \x20   var s: i8 = -3\n\
         \x20   var u: u32 = 4\n\
         \x20   text := \"five\"\n\
         \x20   print(\"{}\n\", f)\n\
         \x20   print(\"{}\n\", d)\n\
         \x20   print(\"{}\n\", s)\n\
         \x20   print(\"{}\n\", u)\n\
         \x20   print(\"{}\n\", text)\n\
         \x20   print(\"{}\n\", true)\n\
         \x20   print(\"{}\n\", false)\n\
         \x20   0\n\
         }\n",
        "1.5\n2.5\n-3\n4\nfive\n1\n0\n",
    ),
    // A doubled brace stands for one, and a `}` on its own writes itself. The
    // compilers count the holes and the run-time walk writes the text, so all
    // three readings of the same rule have to agree.
    (
        "a_format_string_holds_braces",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   print(\"braces {{ and }} stay\n\")\n\
         \x20   print(\"}} alone } fine {{ too\n\")\n\
         \x20   print(\"{{{}}}\n\", 5)\n\
         \x20   0\n\
         }\n",
        "braces { and } stay\n} alone } fine { too\n{5}\n",
    ),
    // Holes at the ends and side by side, where a run between two of them is
    // empty and the walk has nothing to write.
    (
        "format_holes_at_the_edges",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   print(\"{}{}\n\", 1, 2)\n\
         \x20   print(\"{} trailing\n\", 3)\n\
         \x20   print(\"leading {}\n\", 4)\n\
         \x20   print(\"{}\", 5)\n\
         \x20   print(\"\n\")\n\
         \x20   0\n\
         }\n",
        "12\n3 trailing\nleading 4\n5\n",
    ),
    // Every integer width and both float widths reach the same two writers,
    // through the cast the expanded branch carries.
    (
        "a_format_string_writes_every_width",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   var a: u8 = 200\n\
         \x20   var b: i32 = -7\n\
         \x20   var c: u32 = 9\n\
         \x20   var d: f32 = 1.5\n\
         \x20   print(\"{} {} {} {}\n\", a, b, c, d)\n\
         \x20   0\n\
         }\n",
        "200 -7 9 1.5\n",
    ),
    // The destination is a value the caller names. `print` is `write` with
    // standard output supplied, and a second `Sink` proves nothing about the
    // destination is baked into the formatter. A `fn(str)` field is what the
    // self-hosted C backend rendered as `char*` while a slice travels by value.
    (
        "a_program_names_its_own_sink",
        "import \"io.frost\"\nloud :: fn(text: str) {\n\
         \x20   write(to_stdout, \"[{}]\", text)\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   write(loud, \"a {} b\n\", 3)\n\
         \x20   0\n\
         }\n",
        "[a 3 b\n]",
    ),
    // A string literal handed to a compile-time list is a run of bytes, the way
    // it is everywhere else a `str` is read. The self-hosted compiler answered
    // `^i8` for one, which made the element a pointer and the branch that
    // writes text refuse it.
    (
        "a_string_literal_is_a_list_element",
        "import \"io.frost\"\nlast :: fn(args: $...) -> i64 {\n\
         \x20   var n: i64 = 0\n\
         \x20   for v in args { n = n + 1 }\n\
         \x20   n\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{} {}\n\", \"one\", last(\"a\", \"b\", 3))\n\
         \x20   0\n\
         }\n",
        "one 3\n",
    ),
    // `extern fn` with a body is written here and keeps the name it was
    // written under, where an ordinary function is emitted under one the
    // compiler chose. Both compilers have to agree, because what the form is
    // for is a symbol something outside the program calls by name: the answer
    // is the same, and so is the name in the object.
    (
        "an_extern_with_a_body_keeps_its_name",
        "import \"io.frost\"
         frost_demo_double :: extern fn(value: i64) -> i64 {
             value * 2
         }
         ordinary :: fn(value: i64) -> i64 { value + 1 }
         main :: fn() -> i64 {
             print(\"{}\\n\", frost_demo_double(20))
             print(\"{}\\n\", ordinary(6))
             0
         }
",
        "40\n7\n",
    ),
    // A `str` is a `[]u8` (3.2), so `str_len` reads either spelling and
    // `slice_len` reads a run of anything, a `str` included. Both compilers had
    // sites that only half believed it: one refused `slice_len` on text and the
    // other refused `str_len` on bytes.
    (
        "a_length_reads_a_str_and_a_byte_run_alike",
        "import \"io.frost\"
         text_length :: fn(view: []u8) -> i64 { str_len(view) }
         run_length :: fn(view: str) -> i64 { slice_len(view) }
         main :: fn() -> i64 {
             print(\"{}\\n\", text_length(\"hello\"))
             print(\"{}\\n\", run_length(\"hi\"))
             print(\"{}\\n\", str_len(\"four\"))
             0
         }
",
        "5\n2\n4\n",
    ),
    // `$P` on a type the program declared reads as a type parameter, since a
    // `$` argument is one everywhere else. `sizeof` measured that as nothing
    // and answered zero, which a program cannot tell from a real zero, so
    // `sizeof($P)` said 0 and `sizeof(P)` said 16 about one struct.
    (
        "a_dollar_on_a_declared_type_is_that_type",
        "import \"io.frost\"
         P :: struct { a: i64, b: i64 }
         Side :: enum { Left, Right }
         main :: fn() -> i64 {
             print(\"{}\\n\", sizeof($P))
             print(\"{}\\n\", sizeof(P))
             print(\"{}\\n\", sizeof($Side))
             0
         }
",
        "16
16
4
",
    ),
    // `_ :=` says an answer was meant to go unread, which is the one way past
    // the rule above. A list of one is a list, so it reads the way the `_` in a
    // longer binding list does, and what it binds is storage under a name no
    // source can spell: a resource taken this way is still owed a consumer.
    (
        "a_discard_takes_an_answer_nobody_wants",
        "import \"io.frost\"
         Blocked :: struct { at: i64 }
         step :: fn(n: i64) -> i64 ! Blocked {
             if (n < 0) { return { at = n } }
             n * 2
         }
         main :: fn() -> i64 {
             _ := step(3)
             print(\"{}\\n\", 1)
             0
         }
",
        "1
",
    ),
    // `errdefer` runs where the function leaves through its failure set and
    // nowhere else, and it shares one list with `defer`, so the two run in the
    // order they were written, last first. On the way out with an answer the
    // body's own `close` runs and the `errdefer` does not; on the way out with
    // a failure the `errdefer` runs and the body's `close` is never reached.
    (
        "an_errdefer_runs_only_where_a_failure_leaves",
        "import \"io.frost\"
         FileError :: enum { Missing }
         Held :: linear struct { id: i64 }
         close :: fn(move h: Held) { print(\"{}\\n\", h.id) }
         opened :: fn(id: i64) -> Held { Held { id = id } }
         step :: fn(fail: bool) -> i64 ! FileError {
             if (fail) { return FileError::Missing }
             7
         }
         work :: fn(fail: bool) -> i64 ! FileError {
             h := opened(1)
             defer print(\"{}\\n\", 100)
             errdefer close(h)
             value := step(fail)?
             close(h)
             value
         }
         main :: fn() -> i64 {
             match (work(false)) {
                 case .Ok { value }: print(\"{}\\n\", value)
                 case .Err { error }: print(\"{}\\n\", -1)
             }
             match (work(true)) {
                 case .Ok { value }: print(\"{}\\n\", value)
                 case .Err { error }: print(\"{}\\n\", -2)
             }
             0
         }
",
        "1
100
7
1
100
-2
",
    ),
    // An array's length is arithmetic over numbers, module constants, and the
    // size parameters a generic binds. `Slab<T, N>` needs it: one liveness word
    // per sixty-four slots is `[(N + 63) / 64]i64`, and a length that could only
    // be a number or a name could not say it. The bootstrap carries a length
    // whose names are still unbound as written and works it out where the
    // generic is instantiated; the self-hosted compiler parses a body once per
    // instantiation with the parameters bound, so it works it out on the spot.
    (
        "an_array_length_is_arithmetic",
        "import \"io.frost\"
         SIDE :: 6
         Grid :: struct($N: usize) { cells: [N * N]i64, rows: [(N + 1) / 2]i64 }
         filled :: fn($N: usize, mut g: Grid<N>) -> i64 {
             var i : i64 = 0
             while (i < N * N) { g.cells[i] = i  i = i + 1 }
             var total : i64 = 0
             for value in g.cells { total = total + value }
             total
         }
         main :: fn() -> i64 {
             var board : [SIDE * 2]i64 = [0; 12]
             print(\"{}\\n\", slice_len(board))
             var g : Grid<4> = Grid<4> { cells = [0; 16], rows = [0; 2] }
             print(\"{}\\n\", filled(g))
             print(\"{}\\n\", slice_len(g.rows))
             0
         }
",
        "12
120
2
",
    ),
    // A slab carries the same record of which slots are filled, so the same
    // walk reads it. `slab_new()` is what a literal was: a slab has arrays
    // whose lengths are worked out from `N`, and enumerating them at every
    // construction was already the worst part of writing one.
    (
        "a_live_walk_reaches_a_slab_too",
        "import \"io.frost\"
         import \"slab.frost\"
         Entity :: struct { hp: i64 }
         main :: fn() -> i64 {
             var world : Slab<Entity, 130> = slab_new()
             slab_reset(world)
             var made : [130]Handle<Entity> = [0; 130]
             var i : i64 = 0
             while (i < 130) {
                 made[i] = slab_insert(world,
                     Entity { hp = i })
                 i = i + 1
             }
             var d : i64 = 0
             while (d < 130) {
                 if (d % 3 == 0) {
                     assert(slab_release(world, made[d]))
                 }
                 d = d + 1
             }
             assert(slab_release(world, made[64]))
             var total : i64 = 0
             var seen : i64 = 0
             var last : i64 = -1
             for rank, slot in live_slots(world) {
                 assert(slot > last)
                 assert(rank == seen)
                 last = slot
                 total = total + world.storage[slot].hp
                 seen = seen + 1
             }
             print(\"{}\\n\", seen)
             print(\"{}\\n\", world.live_count)
             print(\"{}\\n\", total)
             0
         }
",
        "85
85
5483
",
    ),
    // A handle carries which container it came from, not only which slot. Two
    // pools of the same element type and capacity accept each other's handles
    // otherwise: the slot is in range on both and the generations match, and
    // right after a reset every generation is zero, so the pair below was the
    // case with no protection at all rather than the unlucky one.
    (
        "a_handle_names_the_container_it_came_from",
        "import \"io.frost\"
         import \"slab.frost\"
         import \"columns.frost\"
         Entity :: struct { hp: i64 }
         main :: fn() -> i64 {
             var active : Slab<Entity, 4> = slab_new()
             var pending : Slab<Entity, 4> = slab_new()
             slab_reset(active)
             slab_reset(pending)
             a := slab_insert(active, Entity { hp = 11 })
             b := slab_insert(pending, Entity { hp = 22 })
             assert(slab_alive(active, a))
             assert(slab_alive(pending, b))
             assert(slab_alive(active, b) == false)
             assert(slab_alive(pending, a) == false)
             assert(slab_slot(active, b) == (-1))
             assert(slab_release(active, a))
             assert(slab_alive(active, a) == false)
             var one : columns<Entity, 4> = columns_new()
             var two : columns<Entity, 4> = columns_new()
             columns_reset(one)
             columns_reset(two)
             p := columns_insert(one, Entity { hp = 33 })
             q := columns_insert(two, Entity { hp = 44 })
             assert(columns_alive(one, p))
             assert(columns_alive(two, q))
             assert(columns_alive(one, q) == false)
             assert(columns_alive(two, p) == false)
             print(\"{}\\n\", one[p].hp + two[q].hp)
             0
         }
",
        "77
",
    ),
    // `for slot in live_slots(c)` over a fragmented container. Every third slot is
    // released, which takes out slot 63, the sign bit of the first liveness
    // word, and slot 64 is released on its own so the boundary between two
    // words is covered as well. The walk reaches the live slots in slot order
    // and no others, and `for rank, slot in live_slots(c)` counts them as it goes.
    (
        "a_live_walk_reaches_the_slots_that_hold_an_element",
        "import \"io.frost\"
         import \"columns.frost\"
         Particle :: struct { x: i64, y: i64 }
         main :: fn() -> i64 {
             var c : columns<Particle, 130> = columns_new()
             columns_reset(c)
             var made : [130]Handle<Particle> = [0; 130]
             var i : i64 = 0
             while (i < 130) {
                 made[i] = columns_insert(c,
                     Particle { x = i, y = 0 })
                 i = i + 1
             }
             var d : i64 = 0
             while (d < 130) {
                 if (d % 3 == 0) {
                     assert(columns_release(c, made[d]))
                 }
                 d = d + 1
             }
             assert(columns_release(c, made[64]))
             var total : i64 = 0
             var seen : i64 = 0
             var last : i64 = -1
             for rank, slot in live_slots(c) {
                 assert(slot > last)
                 assert(rank == seen)
                 last = slot
                 total = total + c.x[slot]
                 seen = seen + 1
             }
             print(\"{}\\n\", seen)
             print(\"{}\\n\", c.live_count)
             print(\"{}\\n\", total)
             0
         }
",
        "85
85
5483
",
    ),
    // `break` leaves the walk and `continue` takes the next live slot, which is
    // what they mean in any other loop. The bootstrap builds the walk out of
    // blocks and the self-hosted compiler writes it out as one loop, so this is
    // where those two have to agree.
    (
        "a_live_walk_breaks_and_continues_like_any_loop",
        "import \"io.frost\"
         import \"columns.frost\"
         Cell :: struct { v: i64 }
         main :: fn() -> i64 {
             var c : columns<Cell, 96> = columns_new()
             columns_reset(c)
             var made : [96]Handle<Cell> = [0; 96]
             var i : i64 = 0
             while (i < 96) {
                 made[i] = columns_insert(c, Cell { v = i })
                 i = i + 1
             }
             var d : i64 = 0
             while (d < 96) {
                 if (d % 5 == 0) {
                     assert(columns_release(c, made[d]))
                 }
                 d = d + 1
             }
             var counted : i64 = 0
             for slot in live_slots(c) {
                 if (c.v[slot] % 2 == 0) { continue }
                 if (c.v[slot] > 70) { break }
                 counted = counted + 1
             }
             print(\"{}\\n\", counted)
             var empty : columns<Cell, 8> = columns_new()
             columns_reset(empty)
             var none : i64 = 0
             for slot in live_slots(empty) { none = none + 1 }
             print(\"{}\\n\", none)
             0
         }
",
        "28
0
",
    ),
    // A `str` is a `[]u8` (3.2), so `slice_len` reads its length like any other
    // slice, and an array of bytes reaching a `str` is the same coercion an
    // array of anything else makes. The bootstrap asked for `Type::Slice` alone
    // and refused both; the self-hosted compiler holds the two as one type and
    // took them.
    (
        "a_str_is_a_slice_of_bytes",
        "import \"io.frost\"
         main :: fn() -> i64 {
             var bytes : [3]u8 = [104, 105, 33]
             text : str = bytes
             print(\"{}\\n\", text)
             print(\"{}\\n\", slice_len(text))
             written : str = \"ada\"
             print(\"{}\\n\", slice_len(written))
             0
         }
",
        "hi!
3
3
",
    ),
    // A return type list carrying a resource, consumed by the name it landed
    // on. The struct the list becomes holds a `linear` field, so the closure
    // that makes a struct holding a resource a resource made the temporary the
    // lowering builds one too, and nothing consumes a temporary. Both compilers
    // refused the correct program and named `__multi_result0` doing it. That
    // struct is the one aggregate a program cannot hold: it is built at the
    // `return`, taken apart at the binding, and every field is read exactly
    // once, so its obligation is the sum of its fields' and each of those lands
    // on a name that is tracked.
    (
        "a_return_type_list_carries_a_resource",
        "import \"io.frost\"
         File :: linear struct { handle: i64 }
         pair :: fn(n: i64) -> (opened: File, count: i64) {
             return { opened = File { handle = n }, count = 1 }
         }
         close :: fn(move f: File) { print(\"{}\\n\", f.handle) }
         main :: fn() -> i64 {
             held, count := pair(3)
             close(held)
             print(\"{}\\n\", count)
             0
         }
",
        "3
1
",
    ),
    // `_` in a binding list, for a value the caller has no use for. Any number
    // of them, in any position, including the first, which is the one the
    // statement dispatch had to learn: a list used to be recognized by a
    // leading identifier. The value is still read into storage the compiler
    // names, so a linear one taken by a `_` is still owed a consumer.
    (
        "a_binding_list_discards_with_an_underscore",
        "import \"io.frost\"
         split :: fn(v: i64) -> (high: i64, low: i64) { return v / 256, v % 256 }
         main :: fn() -> i64 {
             high, _ := split(4096)
             a, _ := split(512)
             _, low := split(770)
             print(\"{}\\n\", high + a + low)
             0
         }
",
        "20
",
    ),
    // Names bound by taking a call's several values apart, then used the way
    // the types they were given allow: a slice indexed, a `str` indexed, a
    // struct's own view reached through it, an array indexed. The bootstrap
    // lowers the destructure after the unsafe gate walks it, so the gate meets
    // the binding as it was written and has to read the types off the
    // signature; the self-hosted compiler lowers it as it parses. One shape per
    // type the gate can name, since a fix covering `[]T` alone covers the case
    // that was found rather than the rule.
    (
        "names_taken_from_a_call_carry_the_types_they_were_given",
        "import \"io.frost\"\nHolder :: struct { row: []i64 }
         split :: fn(source: []i64) -> (view: []i64, count: i64) {
             return { view = source, count = 4 }
         }
         label :: fn() -> (text: str, length: i64) {
             return { text = \"hello\", length = 5 }
         }
         wrap :: fn(source: []i64) -> (held: Holder, count: i64) {
             return { held = Holder { row = source }, count = 3 }
         }
         rows :: fn() -> (row: [3]i64, count: i64) {
             return { row = [4, 5, 6], count = 3 }
         }
         main :: fn() -> i64 {
             var data : [4]i64 = [10, 20, 30, 40]
             view, count := split(data)
             print(\"{}\\n\", view[0] + count)
             text, length := label()
             print(\"{}\\n\", text[0] + length)
             held, held_count := wrap(data)
             print(\"{}\\n\", held.row[1] + held_count)
             row, row_count := rows()
             print(\"{}\\n\", row[2] + row_count)
             0
         }
",
        "14
109
23
9
",
    ),
    // A parameter of array type coerced to a slice, in every position that
    // takes one. A parameter is a borrow, so the name holds where the caller's
    // array sits, and the slice is built from that address beside the length
    // the array's type carries. The self-hosted compiler refused three of these
    // positions and wrote the bare address into the other three, so the length
    // came out as whatever sat beside it; the bootstrap copied the array into
    // the callee's frame first, so a write through the slice reached the copy
    // and a slice handed back pointed into a frame that was gone. `bump` is
    // what tells the two apart: the write lands in the caller's array.
    (
        "an_array_parameter_is_sliced_where_the_caller_holds_it",
        "import \"io.frost\"\nHolder :: struct { view: []i64 }
         head :: fn(view: []i64) -> i64 { view[0] }
         by_let :: fn(mut source: [4]i64) -> i64 {
             view : []i64 = source
             view[0]
         }
         by_literal :: fn(mut source: [4]i64) -> i64 {
             sink := Holder { view = source }
             sink.view[1]
         }
         by_argument :: fn(mut source: [4]i64) -> i64 { head(source) }
         by_return :: fn(mut source: [4]i64) -> []i64 { return source }
         by_read_borrow :: fn(source: [4]i64) -> i64 {
             view : []i64 = source
             view[2]
         }
         bump :: fn(mut source: [4]i64) {
             view : []i64 = source
             view[0] = 99
         }
         main :: fn() -> i64 {
             var data : [4]i64 = [10, 20, 30, 40]
             print(\"{}\\n\", by_let(data))
             print(\"{}\\n\", by_literal(data))
             print(\"{}\\n\", by_argument(data))
             print(\"{}\\n\", by_return(data)[3])
             print(\"{}\\n\", by_read_borrow(data))
             bump(data)
             print(\"{}\\n\", data[0])
             0
         }
",
        "10
20
10
40
30
99
",
    ),
    // A struct handed to a call by value, where what comes back is a struct that
    // holds one. Nothing points at the argument: it was copied in. The region
    // walk gave up on any aggregate answer and read the argument's own storage
    // instead, which made this a leak in one compiler for a local and in the
    // other for a temporary. All three shapes are the same question.
    (
        "a_struct_copied_into_a_struct_answer",
        "import \"io.frost\"\nInner :: struct { p: ^u8 }
         Answer :: struct { held: Inner }
         Outer :: struct { answer: Answer, source: Inner }
         made :: fn() -> Inner {
             zero := 0
             Inner { p = unsafe { ptr_cast($u8, zero) } }
         }
         build :: fn(a: Inner) -> Answer { Answer { held = a } }
         from_field :: fn(mut o: Outer) { o.answer = build(o.source) }
         from_temporary :: fn(mut o: Outer) { o.answer = build(made()) }
         from_local :: fn(mut o: Outer) {
             a := made()
             o.answer = build(a)
         }
         main :: fn() -> i64 {
             var o := Outer { answer = Answer { held = made() }, source = made() }
             from_field(o)
             from_temporary(o)
             from_local(o)
             print(\"{}\\n\", 1)
             0
         }
",
        "1
",
    ),
    // A `match` written as a statement, with one side doing nothing. The
    // self-hosted compiler rewrites each arm's last statement into the binding
    // the match answers with, and an arm with no last statement was read anyway,
    // which indexed the node arena at -1 and took the compiler with it. Writing
    // a log plugin is what reached it: one rotation names the file after the
    // clock and the other leaves it alone.
    (
        "a_match_arm_may_hold_nothing",
        "import \"io.frost\"\nRotation :: enum { PerSession, Never }
         name :: fn(held: Rotation) {
             match held {
                 case .PerSession: { print(\"{}\\n\", 1) }
                 case .Never: { }
             }
         }
         main :: fn() -> i64 {
             name(Rotation::PerSession)
             name(Rotation::Never)
             print(\"{}\\n\", 2)
             0
         }
",
        "1
2
",
    ),
    // A function type is already an address, so naming one in a `ptr_cast` asks
    // for that function type rather than for a pointer to it. Both compilers
    // used to wrap it, so a `^proc` came back where a `proc` was wanted and a
    // table of callbacks could not hold one registration written against the
    // state it belongs to. Nothing is generated: a `mut` parameter is an address
    // in the signature and Frost shares C's convention, so the typed function
    // and the erased one are the same function.
    (
        "a_pointer_cast_to_a_function_type_answers_with_that_type",
        "import \"io.frost\"\nHeld :: struct { value: i64 }
         Table :: struct { call: fn(^u8, i64) }
         add :: fn(mut held: Held, more: i64) {
             held.value = held.value + more
         }
         main :: fn() -> i64 {
             var held := Held { value = 1 }
             var table := Table {
                 call = unsafe { ptr_cast($fn(^u8, i64), add) } }
             table.call(unsafe { ptr_cast($u8, ptr_to(held)) }, 41)
             print(\"{}\\n\", held.value)
             0
         }
",
        "42
",
    ),
    // A bare number takes its type from what it is compared against, whichever
    // side it is written on. Neither compiler did: the bootstrap typed the left
    // operand with nothing to go on and the self-hosted one typed a float literal
    // as the widest float there is, so `0.6 == x` widened an `f32` to compare it
    // against a number no `f32` holds. That is true for the values a float
    // represents exactly and false for the rest, so a test written with halves
    // and quarters passes and a glTF file full of measurements does not.
    (
        "a_float_literal_takes_the_width_it_is_compared_against",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   narrow : f32 = 0.6\n\
         \x20   if (narrow == 0.6) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }\n\
         \x20   if (0.6 == narrow) { print(\"{}\\n\", 2) } else { print(\"{}\\n\", 0) }\n\
         \x20   if (narrow != 0.6) { print(\"{}\\n\", 0) } else { print(\"{}\\n\", 3) }\n\
         \x20   if (0.7 > narrow) { print(\"{}\\n\", 4) } else { print(\"{}\\n\", 0) }\n\
         \x20   if (-0.6 == -narrow) { print(\"{}\\n\", 5) } else { print(\"{}\\n\", 0) }\n\
         \x20   wide : f64 = 0.6\n\
         \x20   if (wide == 0.6) { print(\"{}\\n\", 6) } else { print(\"{}\\n\", 0) }\n\
         \x20   if (0.6 == wide) { print(\"{}\\n\", 7) } else { print(\"{}\\n\", 0) }\n\
         \x20   exact : f32 = 0.25\n\
         \x20   if (0.25 == exact) { print(\"{}\\n\", 8) } else { print(\"{}\\n\", 0) }\n\
         \x20   0\n\
         }\n",
        "1\n2\n3\n4\n5\n6\n7\n8\n",
    ),
    // An element of a compile-time list whose type is what a call to a generic
    // answers with. The element is written down while the call is parsed, and a
    // generic's concrete return type is worked out after that, so the
    // self-hosted compiler recorded the element as `i64` and refused the body
    // it reaches for taking a `Pair`. The emitters run once the types have
    // settled and named the other specialization.
    //
    // The return type has to be an aggregate for this to bite: a call
    // answering with one is bound to a name where it is written, so the
    // element reads a name whose type is not recorded until the checks run,
    // rather than the call itself.
    (
        "a_list_element_typed_by_a_call_to_a_generic",
        "import \"io.frost\"\nPair :: struct($T: Type) { first: T, second: T }\n\
         hold :: fn($T: Type, value: i64) -> Pair<T> {\n\
         \x20   Pair { first = value, second = value + 1 }\n}\n\
         each :: fn($body: Type, cols: $...) {\n    body(c for c in cols)\n}\n\
         show :: fn(p: Pair<i64>) {\n    print(\"{}\\n\", p.first + p.second)\n}\n\
         main :: fn() -> i64 {\n    each($show, hold($i64, 20))\n    0\n}\n",
        "41\n",
    ),
    // The same element written as a name given the call. It reaches the answer
    // by a different route, since a name a program writes is bound where the
    // program says and the one above is bound where the compiler puts it, so a
    // fix for either one on its own leaves the other wrong.
    (
        "a_list_element_named_by_a_local_holding_a_generic_call",
        "import \"io.frost\"\nPair :: struct($T: Type) { first: T, second: T }\n\
         hold :: fn($T: Type, value: i64) -> Pair<T> {\n\
         \x20   Pair { first = value, second = value + 1 }\n}\n\
         each :: fn($body: Type, cols: $...) {\n    body(c for c in cols)\n}\n\
         show :: fn(p: Pair<i64>) {\n    print(\"{}\\n\", p.first + p.second)\n}\n\
         main :: fn() -> i64 {\n    made := hold($i64, 20)\n\
         \x20   each($show, made)\n    0\n}\n",
        "41\n",
    ),
    // `assert` outside a test. It is a builtin, so it belongs to every program,
    // and what it lowers to used to be declared only by the test harness. So it
    // read as an unknown variable in an ordinary build under the bootstrap and
    // compiled under the self-hosted compiler, which is two languages.
    (
        "an_assertion_outside_a_test",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n    assert(1 == 1)\n    print(\"{}\\n\", 7)\n    0\n}\n",
        "7\n",
    ),
    // A resource handed on from inside an expression rather than from a
    // statement of its own. The self-hosted check read the root of the
    // statement's expression and nothing below it, so a consuming call written
    // as an operand read as no consumption at all and honest code was refused
    // for leaking what it had just handed away.
    (
        "a_resource_consumed_inside_an_expression",
        "import \"io.frost\"\nFile :: linear struct { fd: i64 }\n\
         close :: fn(move f: File) -> i64 { f.fd }\n\
         run :: fn() -> i64 {\n\
         \x20   var i : i64 = 0\n\
         \x20   var total : i64 = 0\n\
         \x20   while (i < 4) {\n\
         \x20       f := File { fd = i }\n\
         \x20       total = total + close(f)\n\
         \x20       i = i + 1\n    }\n    total\n}\n\
         main :: fn() -> i64 {\n    print(\"{}\\n\", run())\n    0\n}\n",
        "6\n",
    ),
    // A constant standing for another constant. Both compilers parsed
    // `ALIAS :: BASE` as an expression rather than a declaration, because those
    // are also the tokens of `Enum::Variant` and what followed had to say which
    // it was. So the constant did not exist, and every use of it came back as
    // an unknown variable from a file that named it two lines up.
    //
    // At the top level a variant on its own is a statement with no effect and
    // nothing writes one, so the depth settles it and nothing has to follow.
    (
        "a_constant_standing_for_another_constant",
        "import \"io.frost\"\nBASE :: 46\n\
         ALIAS :: BASE\n\
         main :: fn() -> i64 {\n    print(\"{}\\n\", ALIAS)\n    0\n}\n",
        "46\n",
    ),
    // The same through three, since one link working says nothing about a
    // chain: each is substituted into the next until a literal is reached.
    (
        "a_chain_of_three_constants",
        "import \"io.frost\"\nFIRST :: 7\n\
         SECOND :: FIRST\n\
         THIRD :: SECOND\n\
         main :: fn() -> i64 {\n    print(\"{}\\n\", THIRD + 1)\n    0\n}\n",
        "8\n",
    ),
    // Indexing a constant that is a string. The name is the literal wherever it
    // is written, and every question the index path asks was asked of the name
    // instead: the bootstrap reached the array path and answered unknown
    // variable, the self-hosted compiler typed the literal as a raw pointer and
    // demanded an `unsafe` block for a read whose length it had counted itself.
    (
        "indexing_a_constant_that_is_a_string",
        "import \"io.frost\"\nSUFFIX :: \"xyzw\"\n\
         main :: fn() -> i64 {\n\
         \x20   unsafe { print(\"{}\\n\", SUFFIX[0]) }\n\
         \x20   unsafe { print(\"{}\\n\", SUFFIX[3]) }\n    0\n}\n",
        "120\n119\n",
    ),
    // And one that is an array, which is the same bug: an aggregate constant
    // has to have an address before it can be indexed, and neither compiler
    // gave one a place to be.
    (
        "indexing_a_constant_that_is_an_array",
        "import \"io.frost\"\nROW :: [10, 20, 30]\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", ROW[1])\n    0\n}\n",
        "20\n",
    ),
    // Hex, binary, digit separators and an exponent. A `flags u32` declaration
    // transcribes a C header written in hex, and doing that by hand in decimal
    // is a step where a digit goes missing quietly. The exponent is what lets a
    // graphics program write 1e-6 rather than a run of zeroes it has to count.
    (
        "hex_binary_and_separator_literals",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", 0xFF)\n\
         \x20   print(\"{}\\n\", 0xff)\n\
         \x20   print(\"{}\\n\", 0b1010)\n\
         \x20   print(\"{}\\n\", 0x_1_0)\n\
         \x20   print(\"{}\\n\", 1_000_000)\n\
         \x20   print(\"{}\\n\", 0x7FFFFFFF)\n\
         \x20   mask : u64 = 0xFFFFFFFFFFFFFFFF\n\
         \x20   print(\"{}\\n\", mask == -1)\n    0\n}\n",
        "255\n255\n10\n16\n1000000\n2147483647\n1\n",
    ),
    (
        "exponent_literals",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   a := 1e3\n\
         \x20   print(\"{}\\n\", a)\n\
         \x20   b := 1.5e-3\n\
         \x20   print(\"{}\\n\", b * 1000.0)\n\
         \x20   c := 2.5e2\n\
         \x20   print(\"{}\\n\", c)\n\
         \x20   d := 1E2\n\
         \x20   print(\"{}\\n\", d)\n    0\n}\n",
        "1000\n1.5\n250\n100\n",
    ),
    // A prefix minus. The self-hosted parser had no prefix layer at all, so
    // `-1` came out right by accident and `-z` was read as a subtraction with
    // nothing on its left: it printed whatever was lying there. It was not
    // refused, it was miscompiled, which is why `-1` appears five hundred
    // times across the standard library and this compiler and why there is not
    // one bare negative literal in either.
    (
        "a_prefix_minus",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   x := -1\n\
         \x20   print(\"{}\\n\", x)\n\
         \x20   y := -1\n\
         \x20   print(\"{}\\n\", y)\n\
         \x20   z := 5\n\
         \x20   print(\"{}\\n\", -z)\n\
         \x20   print(\"{}\\n\", -z * 2)\n\
         \x20   print(\"{}\\n\", -(z - 8))\n\
         \x20   print(\"{}\\n\", 3 - -z)\n    0\n}\n",
        "-1\n-1\n-5\n-10\n3\n8\n",
    ),
    // The same for a float, where the sign is carried on the literal rather
    // than turned into a subtraction, so that -0.0 stays what it was written
    // as and a negative literal is still a literal taking a type.
    (
        "a_prefix_minus_on_a_float",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   a := -1.5\n\
         \x20   b := 2.5\n\
         \x20   print(\"{}\\n\", a + b)\n\
         \x20   print(\"{}\\n\", -b)\n\
         \x20   c : f32 = -0.75\n\
         \x20   print(\"{}\\n\", c)\n    0\n}\n",
        "1\n-2.5\n-0.75\n",
    ),
    // Prefix `!`, which aborted the self-hosted compiler with an internal
    // arena error rather than a diagnostic. It is `x == false`, which is what
    // it means and what the rest of the tree writes out by hand.
    (
        "a_prefix_bang",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   flag := false\n\
         \x20   if (!flag) { print(\"{}\\n\", 111) } else { print(\"{}\\n\", 222) }\n\
         \x20   n := 7\n\
         \x20   print(\"{}\\n\", !(n == 7))\n\
         \x20   print(\"{}\\n\", !!(n == 7))\n\
         \x20   print(\"{}\\n\", !(n == 8))\n    0\n}\n",
        "111\n0\n1\n1\n",
    ),
    // A call's result handed straight to something that borrows it. The
    // self-hosted C backend wrote `&f()`, which C refuses because a call has no
    // address, and the bootstrap never noticed because its IR spills every call
    // result to a local first.
    (
        "passing_a_call_result_where_a_borrow_is_wanted",
        "import \"io.frost\"\nHeld :: struct { a: i64, b: i64 }\n\
         make :: fn() -> Held { Held { a = 1, b = 2 } }\n\
         use :: fn(h: Held) -> i64 { h.a + h.b }\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", use(make()))\n\
         \x20   print(\"{}\\n\", use(Held { a = 10, b = 20 }))\n    0\n}\n",
        "3\n30\n",
    ),
    // Copying a struct whose size is not a multiple of eight. The self-hosted
    // assembly backend moved the tail in whole words whatever was left, so a
    // four-byte struct was copied as eight and wrote four bytes past wherever
    // it was copied into.
    //
    // It took a long time to find because of where it lands. A struct that size
    // in the last element of a heap block writes off the end of the block, and
    // whether anything notices depends on what the allocator put there, so the
    // same program crashed in a different test each run and sometimes not at
    // all. The guard here is the pattern that makes it deterministic: a run
    // sized exactly for the struct, with something whose value is known written
    // after it.
    (
        "copying_a_struct_that_is_not_a_multiple_of_eight_bytes",
        "import \"io.frost\"\nSmall :: struct { value: f32 }\n\
         Trio :: struct { a: Small, b: Small, tail: i64 }\n\
         take :: fn(held: Small) -> Small { held }\n\
         main :: fn() -> i64 {\n\
         \x20   var t := Trio { a = Small { value = 0.0 },\n\
         \x20       b = Small { value = 2.5 }, tail = 4242 }\n\
         \x20   t.a = take(Small { value = 1.5 })\n\
         \x20   print(\"{}\\n\", t.b.value == 2.5)\n\
         \x20   print(\"{}\\n\", t.tail)\n\
         \x20   var held : [3]Small = [Small { value = 0.0 }; 3]\n\
         \x20   var guard : i64 = 777\n\
         \x20   held[1] = take(Small { value = 3.5 })\n\
         \x20   print(\"{}\\n\", held[2].value == 0.0)\n\
         \x20   print(\"{}\\n\", guard)\n    0\n}\n",
        "1\n4242\n1\n777\n",
    ),
    // Indexing a fixed-size array taken as a `mut` parameter. It arrives as a
    // pointer to the array, and the bootstrap's index path had no case for one:
    // it type-checked and died at lowering. The self-hosted compiler compiled
    // it correctly the whole time, which is what said what the answer was
    // rather than whether there should be one.
    //
    // Nothing in the tree writes it. Every array parameter anywhere is a slice,
    // which is why a shape the language accepts went years without a backend.
    (
        "indexing_an_array_taken_as_a_mut_parameter",
        "import \"io.frost\"\nfill :: fn(mut out: [4]i64) {\n\
         \x20   out[0] = 7\n\
         \x20   out[3] = out[0] + 1\n}\n\
         main :: fn() -> i64 {\n\
         \x20   var held : [4]i64 = [0; 4]\n\
         \x20   fill(held)\n\
         \x20   print(\"{}\\n\", held[0])\n\
         \x20   print(\"{}\\n\", held[3])\n    0\n}\n",
        "7\n8\n",
    ),
    // A string literal bound to a name with no annotation. The literal is a
    // pointer where a `^i8` is asked for and a `str` where a `str` is, and a
    // binding asks for neither, so the self-hosted compiler kept the pointer
    // and the length was gone by the time anything wanted it. Passing the name
    // to anything taking a `str` then handed over a pointer: the C backend
    // refused it and the assembly backend read a length out of whatever
    // followed.
    //
    // Nothing in the tree had ever written it. Every literal was either an
    // argument, where the coercion sees the target type, or assigned to a name
    // already declared.
    (
        "a_string_literal_bound_to_a_name",
        "import \"io.frost\"\nshown :: fn(s: str) -> i64 { str_len(s) }\n\
         main :: fn() -> i64 {\n\
         \x20   held := \"hello\"\n\
         \x20   print(\"{}\\n\", shown(held))\n\
         \x20   print(\"{}\\n\", str_len(held))\n    0\n}\n",
        "5\n5\n",
    ),
    // A type named before it is declared. The name resolved to nothing and was
    // taken for an i64, so the assembly backend read the wrong bytes and the C
    // backend died looking the type up. `sizeof` agreed by coincidence, which
    // is what kept it invisible.
    (
        "type_named_before_it_is_declared",
        "import \"io.frost\"\nmake :: fn() -> i64 {\n\
         \x20   var held := Later { value = 3 }\n\
         \x20   held.value\n}\n\
         Later :: struct { value: i64 }\n\
         main :: fn() -> i64 {\n    print(\"{}\\n\", make())\n    0\n}\n",
        "3\n",
    ),
    // The same, as the type of a field rather than of a literal.
    (
        "field_of_a_type_declared_below",
        "import \"io.frost\"\nHolder :: struct { inner: Later, tag: i64 }\n\
         Later :: struct { value: i64 }\n\
         main :: fn() -> i64 {\n\
         \x20   var h := Holder { inner = Later { value = 7 }, tag = 1 }\n\
         \x20   print(\"{}\\n\", h.inner.value)\n    print(\"{}\\n\", sizeof(Holder))\n    0\n}\n",
        "7\n16\n",
    ),
    // A field read straight off a generic call's answer, where the caller has a
    // parameter named the same as one of the template's. The unsafety gate runs
    // with whatever local table the pass before it left behind, so it looked
    // the caller's name up against the last instance's locals. Sharing a name
    // made that a diagnostic; not sharing one made it a wrong answer nobody
    // saw.
    (
        "a_field_of_a_generic_call_answer",
        "import \"io.frost\"\nCell :: struct { value: i64 }
         Bag :: struct($T: Type) { one: T }
         only :: fn(a: Bag<$T>) -> ref T {
             ref held := a.one
    held
}
         Store :: struct { bag: Bag<Cell>, count: i64 }
         reach :: fn(mut a: Store) -> i64 { only(a.bag).value }
         main :: fn() -> i64 {
             var s := Store { bag = Bag { one = Cell { value = 42 } }, count = 1 }
             print(\"{}\\n\", reach(s))
    0
}
",
        "42
",
    ),
    // A string literal answered as a `str`. The literal is a pointer and the
    // place it reaches carries a length beside it, and the C backend emitted
    // the return without the conversion that adds one, so a function shaped
    // like this compiled through the assembly backend and not through C.
    (
        "a_string_literal_answered_as_a_str",
        "import \"io.frost\"\npick :: fn(n: i64) -> str {
    if (n == 0) { return \"zero\" }
    \"many\"
}
main :: fn() -> i64 {
    print(\"{}\\n\", pick(0))
    print(\"{}\\n\", pick(7))
    0
}
",
        "zero
many
",
    ),
    // Arithmetic on the widest unsigned types. The assembly backend divided,
    // compared and shifted every integer as a signed one, so a `u64` or a
    // `usize` past 2^63 came out negative there and right everywhere else. The
    // bootstrap and the C backend both follow the type. Nothing caught it
    // because the compiler's own `usize` values are array lengths, which never
    // reach the top bit.
    (
        "the_widest_unsigned_types_are_unsigned",
        "import \"io.frost\"\nmain :: fn() -> i64 {
    var big : u64 = 9223372036854775807
    big = big + 1
    var two : u64 = 2
    var one : u64 = 1
    print(\"{}\\n\", big / two)
    print(\"{}\\n\", big % 3)
    if (big > one) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }
    if (big < one) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }
    print(\"{}\\n\", big >> 1)
    var span : usize = 9223372036854775807
    span = span + 2
    if (span > one) { print(\"{}\\n\", 1) } else { print(\"{}\\n\", 0) }
    print(\"{}\\n\", span / two)
    var n : i64 = -100
    print(\"{}\\n\", n / 7)
    print(\"{}\\n\", n >> 2)
    0
}
",
        "4611686018427387904
2
1
0
4611686018427387904
1
4611686018427387904
-14
-25
",
    ),
    // A literal shifted past the thirty-second bit. Every integer in Frost is
    // sixty-four bits wide, and the self-hosted C backend wrote a literal
    // without the suffix that says so, so a C compiler shifted a thirty-two bit
    // one and answered with zero. It surfaced as the assembler losing the sign
    // of a negative float literal, since the sign bit is written `1 << 63`, and
    // only when the compiler doing the assembling had itself been built through
    // that backend.
    (
        "a_literal_shifted_past_the_word",
        "import \"io.frost\"\nmain :: fn() -> i64 {
    print(\"{}\\n\", 1 << 63)
    print(\"{}\\n\", 1 << 40)
    var bits : i64 = 0
    bits = bits | (1 << 63)
    print(\"{}\\n\", bits)
    0
}
",
        "-9223372036854775808
1099511627776
-9223372036854775808
",
    ),
    // A `for` over a range. The self-hosted lexer had no `..` at all, so
    // `for i in 0..6` read as a field access on a number and the parse came
    // apart somewhere further down the file: two examples died indexing the
    // node arena and a third reported the next function's name as an unknown
    // enum variant. Two loops in one function are here because the first
    // desugar declared the counter where the loop stood, and a function body is
    // one C scope, so the second declaration collided with the first.
    (
        "a_for_over_a_range",
        "import \"io.frost\"\nmain :: fn() -> i64 {
    var total : i64 = 0
    for i in 0..6 {
        total = total + i
    }
    print(\"{}\\n\", total)
    var closed : i64 = 0
    for i in 1..=4 {
        closed = closed + i
    }
    print(\"{}\\n\", closed)
    n := 3
    var counted : i64 = 0
    for i in 0..n {
        counted = counted + 1
    }
    print(\"{}\\n\", counted)
    0
}
",
        "15
10
3
",
    ),
    // `defer`, which the self-hosted compiler lexed as an identifier and so did
    // not have at all. Every road out of a function is here, because each runs
    // the deferred statements somewhere different: a `return` in the middle, a
    // body ending in the value it answers with, and a body answering with
    // nothing. Two in one function say the order is last deferred, first run.
    // The answer is bound to a name before they run, so what they do cannot
    // reach it.
    (
        "defer_runs_at_every_exit",
        "import \"io.frost\"\ntrace :: fn(n: i64) { print(\"{}\\n\", n) }

early :: fn(x: i64) -> i64 {
    defer trace(1)
    defer trace(2)
    if (x > 0) {
        return x * 10
    }
    trace(3)
    0
}

quiet :: fn() {
    defer trace(4)
    trace(5)
}

main :: fn() -> i64 {
    print(\"{}\\n\", early(2))
    print(\"{}\\n\", early(0 - 1))
    quiet()
    0
}
",
        "2
1
20
3
2
1
0
5
4
",
    ),
    // A generic over a plain struct, and a `mut` parameter holding one. Three
    // separate questions meet here. The tuple a call names is read off its
    // arguments, and a plain struct carries none of its own, so `swap(u, v)`
    // has to be typed from the local. A tuple belongs to the generic whose call
    // made it, so the one `bytes($Vec3)` names is not offered to `swap`. And an
    // aggregate `mut` parameter is an address, so assigning the whole value
    // writes through the name rather than over it. The same `swap` is called
    // with a struct and a scalar, which is what says both instances exist.
    (
        "a_generic_over_a_plain_struct",
        "import \"io.frost\"\nVec3 :: struct { x: i64, y: i64, z: i64 }

swap :: fn(mut a: $T, mut b: $T) {
    t := a
    a = b
    b = t
}

bytes :: fn($T: Type) -> i64 { sizeof(T) }

main :: fn() -> i64 {
    var u := Vec3 { x = 1, y = 2, z = 3 }
    var v := Vec3 { x = 4, y = 5, z = 6 }
    swap(u, v)
    print(\"{}\\n\", u.x)
    print(\"{}\\n\", v.x)
    var a : i64 = 100
    var b : i64 = 200
    swap(a, b)
    print(\"{}\\n\", a)
    print(\"{}\\n\", bytes($Vec3))
    0
}
",
        "4
1
200
24
",
    ),
    // A generic whose type parameter is not its first parameter. What a call
    // binds the parameter to was read off the first argument whose type was
    // known, so `scale(10, u)` was compiled for `i64` and the body reached for a
    // field of one. The declaration says which argument stands for the
    // parameter, and that is the one read now.
    //
    // The second half is the tuple that cannot be named. Once the argument is
    // the right one, its type is still unknown out where the tuples are
    // gathered, because no local table lives there and a name nothing knows
    // reads as `i64`. Answering `i64` there named a second instance nothing had
    // asked for, which the assembly backend emitted happily and the C backend
    // refused, since C is what type-checks a field read.
    (
        "a_generic_whose_parameter_is_not_first",
        "import \"io.frost\"\nVec3 :: struct { x: i64, y: i64, z: i64 }

scale :: fn(by: i64, mut v: $T) {
    v.x = v.x * by
}

widest :: fn(mut a: $T, by: i64) -> i64 {
    a.y = a.y + by
    a.y
}

main :: fn() -> i64 {
    var u := Vec3 { x = 3, y = 4, z = 5 }
    scale(10, u)
    print(\"{}\\n\", u.x)
    print(\"{}\\n\", widest(u, 6))
    print(\"{}\\n\", u.y)
    0
}
",
        "30
10
10
",
    ),
    // A generic over `[]$T`. The parameter holds the type variable rather than
    // being it, so what a call binds it to is the argument's element, and the
    // two calls here name one instance over `i64` rather than one per array
    // length. Reading the argument's own type instead keyed an instance by
    // `[3]i64` and another by `[2]i64`, each with a parameter of the wrong
    // width, and the emitted C would not compile.
    (
        "a_generic_over_a_slice",
        "import \"io.frost\"\nfirst :: fn(v: []$T) -> i64 {
    slice_len(v)
}

counted :: fn(v: []$T) -> i64 {
    var n : i64 = 0
    for x in v {
        n = n + 1
    }
    n
}

main :: fn() -> i64 {
    xs := [1, 2, 3]
    ys := [9, 4]
    print(\"{}\\n\", first(xs))
    print(\"{}\\n\", first(ys))
    print(\"{}\\n\", counted(xs))
    print(\"{}\\n\", counted(ys))
    0
}
",
        "3
2
3
2
",
    ),
    // Walking a slice and keeping an element in a local. A loop variable is this
    // frame's storage, so an address of one dies with the call, but what it
    // holds is an element worth whatever the sequence was worth. The bootstrap
    // recorded only the first half, so reading a loop variable answered with
    // frame storage: `best = x` carried that into `best` and returning `best`
    // was refused, over a slice of plain integers with nothing to escape. The
    // self-hosted compiler took it, so the two disagreed about a shape as
    // ordinary as a maximum.
    (
        "a_loop_variable_read_carries_no_storage",
        "import \"io.frost\"\nwidest :: fn(v: []i64) -> i64 {
    var best : i64 = 0
    for x in v {
        if (x > best) { best = x }
    }
    best
}

counted :: fn(v: []$T) -> i64 {
    var n : i64 = 0
    for index, x in v {
        n = index
    }
    n
}

main :: fn() -> i64 {
    xs := [3, 9, 4]
    print(\"{}\\n\", widest(xs))
    print(\"{}\\n\", counted(xs))
    0
}
",
        "9
2
",
    ),
    // A `defer` and a `?`. Handing a failure on is the function leaving, so
    // whatever it deferred runs there the way it runs at a `return` written out.
    // The self-hosted compiler built the `?` return node directly rather than
    // through the path a `return` takes, so the deferred statement ran on the
    // path that succeeded and not on the path that failed, which is a resource
    // left behind on exactly the exit that was meant to be the tidy one.
    (
        "a_defer_runs_where_a_question_mark_hands_on",
        "import \"io.frost\"\nParseError :: struct { at: i64 }

digit_of :: fn(c: i64) -> i64 ! ParseError {
    if (c < 48 || c > 57) {
        return ParseError { at = c }
    }
    c - 48
}

two_digits :: fn(high: i64, low: i64) -> i64 ! ParseError {
    defer print(\"{}\\n\", 99)
    tens := digit_of(high)?
    ones := digit_of(low)?
    tens * 10 + ones
}

side :: fn(high: i64, low: i64) -> i64 {
    match two_digits(high, low) {
        case .Ok { value }: 0
        case .Err { error }: 1
    }
}

main :: fn() -> i64 {
    print(\"{}\\n\", side(52, 55))
    print(\"{}\\n\", side(52, 90))
    0
}
",
        "99
0
99
1
",
    ),
    // A body whose answer is a trailing `if`, and one whose answer is a
    // trailing `match`. Each branch of an answering `if` becomes a return, so
    // each has to carry what was deferred the way a written `return` does. The
    // self-hosted compiler put the copies after the `if` instead, which left
    // them where nothing reached and took the answer with them: both branches
    // ran their deferred statement and then the function answered 0.
    (
        "a_defer_under_a_trailing_branch",
        "import \"io.frost\"\ntrace :: fn(n: i64) { print(\"{}\\n\", n) }
Kind :: enum { One, Two }

branchy :: fn(c: i64) -> i64 {
    defer trace(1)
    if (c > 0) { 10 } else { 20 }
}

matchy :: fn(k: Kind) -> i64 {
    defer trace(2)
    match k {
        case .One: 30
        case .Two: 40
    }
}

pair :: fn() -> (a: i64, b: i64) {
    defer trace(3)
    return 4, 5
}

main :: fn() -> i64 {
    print(\"{}\\n\", branchy(1))
    print(\"{}\\n\", branchy(0 - 1))
    print(\"{}\\n\", matchy(Kind::Two))
    x, y := pair()
    print(\"{}\\n\", x)
    print(\"{}\\n\", y)
    0
}
",
        "1
10
1
20
2
40
3
4
5
",
    ),
    // A bound on whether a type has to be consumed. It is the one predicate in
    // the vocabulary that asks about an obligation rather than about a
    // representation, and it is what lets a container hold itself to elements
    // it can account for: a slot is reached by a number worked out while the
    // program runs, so a write into one drops what was there with no place a
    // check could name.
    (
        "a_bound_on_whether_a_type_is_a_resource",
        "import \"io.frost\"\nP :: struct { x: i64 }\n\
         only_plain :: fn($T: Type, v: $T) -> i64 where !is_linear(T) { 1 }\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", only_plain(7))\n\
         \x20   print(\"{}\\n\", only_plain(P { x = 2 }))\n\
         \x20   0\n}\n",
        "1\n1\n",
    ),
    // The two ways a generic literal says which instance it is: the arguments
    // written on the literal, and a declared type that names them. Both are
    // context the reader supplied rather than something worked back out of the
    // field values.
    (
        "a_generic_literal_saying_its_instance",
        "import \"io.frost\"\nPair :: struct($T: Type) { first: T, second: T }\n\
         main :: fn() -> i64 {\n\
         \x20   a := Pair<i64> { first = 1, second = 2 }\n\
         \x20   b : Pair<i64> = Pair { first = 10, second = 20 }\n\
         \x20   print(\"{}\\n\", a.first + a.second)\n\
         \x20   print(\"{}\\n\", b.first + b.second)\n\
         \x20   0\n}\n",
        "3\n30\n",
    ),
    // The same, for a generic taking more than one parameter. Each is read off
    // the field that declares it, so the instance resolves as a whole rather
    // than only where a single parameter made the answer obvious.
    (
        "a_generic_literal_inferring_two_arguments",
        "import \"io.frost\"\nDuo :: struct($A: Type, $B: Type) { first: A, second: B }\n\
         main :: fn() -> i64 {\n\
         \x20   d := Duo { first = 7, second = true }\n\
         \x20   print(\"{}\\n\", d.first)\n\
         \x20   0\n}\n",
        "7\n",
    ),
    // `is_linear` asked of a list element while the body is expanded, which is
    // the same question a `where` bound asks and the same table answers it. The
    // branch that cannot compile for this element is dropped rather than
    // skipped, so one body consumes the resource and prints the plain value.
    (
        "is_linear_decides_a_branch_at_expansion_time",
        "import \"io.frost\"\nFile :: linear struct { fd: i64 }\n\
         close :: fn(move f: File) -> i64 { f.fd }\n\
         plain :: fn(v: i64) -> i64 { v }\n\
         each :: fn(args: $...) {\n\
         \x20   for v in args {\n\
         \x20       if (is_linear(v)) { print(\"{}\\n\", close(v)) } else { print(\"{}\\n\", plain(v)) }\n\
         \x20   }\n}\n\
         main :: fn() -> i64 {\n\
         \x20   each(File { fd = 7 }, 5)\n\
         \x20   0\n}\n",
        "7\n5\n",
    ),
    // A float converted to an integer narrower than 32 bits. x64 converts a
    // float to a 32 or 64 bit register and to nothing narrower, so the
    // conversion has to land in 32 and have its width taken off after. Asking
    // for the narrow width directly answered 72 for 200 and 32767 for 65535 in
    // a release build, and tripped an assertion inside the register allocator
    // in a debug one.
    (
        "a_float_converted_to_a_narrow_integer",
        "import \"io.frost\"\nmain :: fn() -> i64 {\n\
         \x20   x : f32 = 200.7\n\
         \x20   print(\"{}\\n\", cast($i64, cast($u8, x)))\n\
         \x20   y : f64 = 65535.4\n\
         \x20   print(\"{}\\n\", cast($i64, cast($u16, y)))\n\
         \x20   s : f32 = -40.9\n\
         \x20   print(\"{}\\n\", cast($i64, cast($i8, s)))\n\
         \x20   0\n}\n",
        "200\n65535\n-40\n",
    ),
    // A `match` used where a value is wanted. The self-hosted compiler binds a
    // name, lets each arm assign to it and stands for the name, and that
    // binding is declared before a single arm has been read: it was seeded with
    // a zero and took the type of one. An arm answering with a struct was
    // refused for handing a struct to an i64, and an arm answering with a float
    // was accepted and truncated, which is the worse half.
    (
        "a_match_answers_with_a_float",
        "import \"io.frost\"\npick :: fn(k: i64) -> f64 {\n\
         \x20   match k {\n\
         \x20       case 0: 1.5\n\
         \x20       case _: 2.25\n\
         \x20   }\n}\n\
         main :: fn() -> i64 {\n\
         \x20   print(\"{}\\n\", cast($i64, pick(0) * 100.0))\n\
         \x20   print(\"{}\\n\", cast($i64, pick(1) * 100.0))\n\
         \x20   0\n}\n",
        "150\n225\n",
    ),
    (
        "a_match_answers_with_a_struct",
        "import \"io.frost\"\nHeld :: struct { x: i64, y: i64 }\n\
         one :: fn() -> Held { Held { x = 1, y = 2 } }\n\
         two :: fn() -> Held { Held { x = 3, y = 4 } }\n\
         pick :: fn(k: i64) -> Held {\n\
         \x20   match k {\n\
         \x20       case 0: one()\n\
         \x20       case _: two()\n\
         \x20   }\n}\n\
         main :: fn() -> i64 {\n\
         \x20   got := pick(1)\n\
         \x20   print(\"{}\\n\", got.x)\n\
         \x20   print(\"{}\\n\", got.y)\n\
         \x20   0\n}\n",
        "3\n4\n",
    ),
    (
        "a_match_answers_with_an_array",
        "import \"io.frost\"\npick :: fn(k: i64) -> [3]i64 {\n\
         \x20   match k {\n\
         \x20       case 0: [1, 2, 3]\n\
         \x20       case _: [4, 5, 6]\n\
         \x20   }\n}\n\
         main :: fn() -> i64 {\n\
         \x20   got := pick(1)\n\
         \x20   print(\"{}\\n\", got[2])\n\
         \x20   0\n}\n",
        "6\n",
    ),
    // The same binding, reached by the other road: an `if` used where a value
    // is wanted takes it too, and a fix for one arm shape leaves the other
    // wrong.
    // A constant whose value is a negative float. The self-hosted compiler
    // sorts a constant by the token after `::`, and a sign in front of a number
    // was only considered for the integer constants it folds arithmetic over.
    // A float carries its literal node instead, so `-2.2` matched neither and
    // was refused as a declaration the compiler does not have. A positive float
    // and a negative integer both worked, which is what kept it hidden.
    // A pointer to a distinct type. Every type in the self-hosted compiler is
    // one i64, built by adding a base to what it contains, which works because
    // each base sits above everything it can hold. A distinct type sits above
    // all of them, since its code carries the representation it is laid out as,
    // so `POINTER_BASE + Meters` landed back inside the distinct range and read
    // as a different distinct type: `pointee` answered 247 and the struct table
    // was asked for entry 231. The compiler died with an arena index rather
    // than saying anything.
    //
    // Both representations, because the arithmetic goes wrong differently for
    // each: a distinct over a pointer moved the code by a whole stride, and one
    // over an integer moved it inside the first.
    (
        "a_pointer_to_a_distinct_integer",
        "import \"io.frost\"\nMeters :: distinct i64\n\
         far :: fn(p: ^Meters) -> i64 {\n\
         \x20   held : Meters = unsafe { p^ }\n\
         \x20   count : i64 = held\n\
         \x20   count + 1\n}\n\
         main :: fn() -> i64 {\n\
         \x20   var m : Meters = 41\n\
         \x20   print(\"{}\\n\", far(ptr_to(m)))\n\
         \x20   0\n}\n",
        "42\n",
    ),
    (
        "a_pointer_to_a_distinct_pointer",
        "import \"io.frost\"\nGrip :: distinct ^u8\n\
         no_grip :: fn() -> Grip {\n\
         \x20   zero := 0\n\
         \x20   unsafe { ptr_cast($u8, zero) }\n}\n\
         same :: fn(p: ^Grip) -> i64 {\n\
         \x20   held : Grip = unsafe { p^ }\n\
         \x20   if (held == no_grip()) {\n\
         \x20       return 1\n\
         \x20   }\n\
         \x20   0\n}\n\
         main :: fn() -> i64 {\n\
         \x20   var g := no_grip()\n\
         \x20   print(\"{}\\n\", same(ptr_to(g)))\n\
         \x20   0\n}\n",
        "1\n",
    ),
    // A float literal takes its width from the context, the way an integer
    // literal does. The self-hosted compiler read every one at double, so
    // `-2.1 + 1.1` at an `f32` added two doubles and rounded once at the end
    // where the bootstrap added two singles. It contradicted itself as well as
    // the bootstrap: the same two numbers held in `f32` locals answered the
    // other way.
    (
        "a_float_literal_takes_the_width_of_its_context",
        "import \"io.frost\"\nFLOOR :: -2.1\n\
         RISE :: 1.1\n\
         main :: fn() -> i64 {\n\
         \x20   named : f32 = FLOOR + RISE\n\
         \x20   written : f32 = -2.1 + 1.1\n\
         \x20   var a : f32 = -2.1\n\
         \x20   var b : f32 = 1.1\n\
         \x20   computed := a + b\n\
         \x20   wide : f64 = -2.1 + 1.1\n\
         \x20   print(\"{}\\n\", cast($i64, named * 100.0))\n\
         \x20   print(\"{}\\n\", cast($i64, written * 100.0))\n\
         \x20   print(\"{}\\n\", cast($i64, computed * 100.0))\n\
         \x20   print(\"{}\\n\", cast($i64, wide * 100.0))\n\
         \x20   0\n}\n",
        "-99\n-99\n-99\n-100\n",
    ),
    // The numbers are exact in binary on purpose: what this pins is that the
    // declaration is read at all, and a value that needs rounding would pin
    // where the arithmetic happens instead, which is its own question.
    (
        "a_constant_holding_a_negative_float",
        "import \"io.frost\"\nFLOOR :: -2.5\n\
         RISE :: 1.25\n\
         DEEP :: -3\n\
         main :: fn() -> i64 {\n\
         \x20   lowest : f32 = FLOOR + RISE\n\
         \x20   print(\"{}\\n\", cast($i64, lowest * 100.0))\n\
         \x20   print(\"{}\\n\", DEEP)\n\
         \x20   0\n}\n",
        "-125\n-3\n",
    ),
    (
        "an_if_answers_with_a_struct",
        "import \"io.frost\"\nHeld :: struct { x: i64, y: i64 }\n\
         pick :: fn(k: i64) -> Held {\n\
         \x20   if (k == 0) { Held { x = 1, y = 2 } } else { Held { x = 3, y = 4 } }\n\
         }\n\
         main :: fn() -> i64 {\n\
         \x20   got := pick(1)\n\
         \x20   print(\"{}\\n\", got.x + got.y)\n\
         \x20   0\n}\n",
        "7\n",
    ),
    // An index whose base is itself an index. The bootstrap read the type of a
    // place through names, dereferences and fields but not through elements, so
    // `pair[0]` typed as nothing, fell past the slice path to the array one, and
    // was refused for not naming an array. The self-hosted compiler built the
    // same program and printed the right number, which is what said what the
    // answer was.
    //
    // Every base a chain can stand on, since the fix is one arm and each of
    // these reaches it differently: an array of slices, an array of strings, an
    // array reached through a field, a slice a call answered with, and an array
    // of arrays, which is the one shape that already worked.
    (
        "an_index_whose_base_is_an_index",
        "import \"io.frost\"\nimport \"vec.frost\"\n\
         Holder :: struct { rows: [2][]i64 }\n\
         main :: fn() -> i64 {\n\
         \x20   a : [2]i64 = [1, 2]\n\
         \x20   b : [2]i64 = [3, 4]\n\
         \x20   pair : [2][]i64 = [a, b]\n\
         \x20   print(\"{}\\n\", pair[1][0])\n\
         \x20   texts : [2]str = [\"ab\", \"cd\"]\n\
         \x20   print(\"{}\\n\", texts[1][0])\n\
         \x20   held : Holder = { rows = pair }\n\
         \x20   print(\"{}\\n\", held.rows[0][1])\n\
         \x20   var v := vec_new($i64, 2)\n\
         \x20   vec_push(v, 55)\n\
         \x20   print(\"{}\\n\", vec_slice(v)[0])\n\
         \x20   vec_free(v)\n\
         \x20   var grid : [2][2]i64 = [[7, 8], [9, 10]]\n\
         \x20   print(\"{}\\n\", grid[1][1])\n\
         \x20   0\n}\n",
        "3\n99\n2\n55\n10\n",
    ),
    // A stated layout. `packed struct` puts every field at the next byte and
    // gives the type an alignment of one; `align(N)` after a field's type is
    // what that field starts at a multiple of, and the widest one is the
    // struct's own. The grid is here because four backends compute these: the
    // bootstrap lays every struct out itself and its C emitter writes the
    // offsets it worked out, while the self-hosted compiler lays them out for
    // its assembly and hands C the declaration with the attributes that say
    // the same thing. A number they disagree on is a program that reads the
    // wrong bytes on one of them and nothing else would say so.
    (
        "a_stated_layout_is_the_layout",
        "import \"io.frost\"
         Plain :: struct { a: u8, b: i64, c: u8 }
         Tight :: packed struct { a: u8, b: i64, c: u8 }
         Wide :: struct { a: u8, b: i64 align(16), c: u8 }
         Mixed :: packed struct { a: u16, b: u32, c: u8, d: f64 }
         Holder :: struct { one: Tight, two: u8 }
         Spread :: struct { a: u8 align(4), b: u8 align(8), c: u8 align(32) }
         walk :: fn($T: Type) {
             print(\"{}\\n\", sizeof($T))
             for field in fields(T) { print(\"{}\\n\", offset_of(field)) }
         }
         main :: fn() -> i64 {
             walk($Plain)
             walk($Tight)
             walk($Wide)
             walk($Mixed)
             walk($Holder)
             walk($Spread)
             t := Tight { a = 1, b = 300, c = 5 }
             print(\"{}\\n\", t.a + t.b + t.c)
             w := Wide { a = 2, b = 400, c = 6 }
             print(\"{}\\n\", w.a + w.b + w.c)
             var row : [4]Tight = [Tight { a = 0, b = 0, c = 0 }; 4]
             row[3] = t
             print(\"{}\\n\", row[3].b)
             0
         }
",
        "24\n0\n8\n16\n\
         10\n0\n1\n9\n\
         32\n0\n16\n24\n\
         15\n0\n2\n6\n7\n\
         11\n0\n10\n\
         64\n0\n8\n32\n\
         306\n408\n300\n",
    ),
    // `packed` and `align` are words rather than keywords, so a local, a field
    // and a parameter may still be called either. What marks the declaration is
    // the `struct` after `packed`, and what marks the field form is the `(`
    // after `align`, neither of which a name is ever followed by. Both had been
    // keywords for one commit and `std/slab.frost` stopped compiling, which is
    // what says a keyword is the wrong shape for these two.
    (
        "packed_and_align_are_still_names",
        "import \"io.frost\"
         Held :: struct { packed: i64, align: i64 }
         squeeze :: fn(packed: i64, align: i64) -> i64 { packed + align }
         main :: fn() -> i64 {
             packed := 7
             align := 3
             h := Held { packed = 10, align = 20 }
             print(\"{}\\n\", squeeze(packed, align))
             print(\"{}\\n\", h.packed + h.align)
             0
         }
",
        "10\n30\n",
    ),
    // A call written where a compile-time value is read is worked out before
    // the program runs. What the vocabulary is, is the whole-number half of the
    // language: parameters and locals, arithmetic and comparison, `if`,
    // `while`, `return`, and calls to other functions written the same way. The
    // two compilers reach it from opposite ends, which is why this is here: the
    // bootstrap reads the callee's tokens into a parse of its own before the
    // real one starts, and the self-hosted compiler walks the tokens directly,
    // because a constant is read before any body is.
    (
        "a_compile_time_call_is_worked_out",
        "import \"io.frost\"
         round_up :: fn(value: i64, to: i64) -> i64 { (value + to - 1) / to * to }
         next_power_of_two :: fn(n: i64) -> i64 {
             var held : i64 = 1
             while (held < n) { held = held * 2 }
             held
         }
         pick :: fn(n: i64) -> i64 {
             if (n > 10) { return n * 2 }
             n + 1
         }
         digits :: fn(n: i64) -> i64 {
             var left := n
             var seen : i64 = 0
             while (true) {
                 seen = seen + 1
                 left = left / 10
                 if (left == 0) { break }
             }
             seen
         }
         LANES :: round_up(300, 64)
         SLOTS :: next_power_of_two(300)
         SMALL :: pick(3)
         BIG :: pick(30)
         NESTED :: round_up(next_power_of_two(100), 64)
         AROUND :: round_up(100, 64) * 2 + 1
         WIDE :: digits(40325)
         Buffer :: struct { bytes: [round_up(300, 64)]u8 }
         main :: fn() -> i64 {
             print(\"{}\\n\", LANES)
             print(\"{}\\n\", SLOTS)
             print(\"{}\\n\", SMALL)
             print(\"{}\\n\", BIG)
             print(\"{}\\n\", NESTED)
             print(\"{}\\n\", AROUND)
             print(\"{}\\n\", WIDE)
             print(\"{}\\n\", sizeof(Buffer))
             var held : [next_power_of_two(100)]i64 = [0; 128]
             print(\"{}\\n\", slice_len(held))
             0
         }
",
        "320\n512\n4\n60\n128\n257\n5\n320\n128\n",
    ),
    // The smallest i64 has no literal of its own: its magnitude is one past the
    // largest, so a program writes it as one more than it plus one less and the
    // fold answers with the number. Every backend then has to carry a value
    // that has no spelling. The self-hosted assembler read the digits as a
    // positive number and negated afterwards, which overflowed before the sign
    // was applied; both C emitters wrote a literal C reads as unsigned and
    // warns about. Found by the compiler compiling itself, after a constant of
    // this shape went into its own source.
    (
        "the_smallest_whole_number_reaches_every_backend",
        "import \"io.frost\"
         SMALLEST :: -9223372036854775807 - 1
         LARGEST :: 9223372036854775807
         main :: fn() -> i64 {
             print(\"{}\\n\", SMALLEST)
             print(\"{}\\n\", LARGEST)
             print(\"{}\\n\", SMALLEST + 1)
             0
         }
",
        "-9223372036854775808\n9223372036854775807\n-9223372036854775807\n",
    ),
    // What a compile-time call may hold, past a number: a run of values, a set
    // of named ones, and a run of bytes. Each is held by the evaluator rather
    // than read back out of the tokens, since an element may itself be a call
    // and a value has to outlive the names that built it. `[TABLE[3]]u8` is the
    // point of the whole thing: a lookup table decided before the program runs
    // and a length read out of it.
    (
        "a_compile_time_value_may_be_a_run_of_values",
        "import \"io.frost\"
         Point :: struct { x: i64, y: i64 }
         TABLE :: [1, 2, 4, 8]
         ORIGIN :: Point { x = 3, y = 4 }
         NAME :: \"hello\"
         SLOTS :: TABLE[2]
         DOWN :: ORIGIN.y
         LETTER :: NAME[1]
         WIDTH :: str_len(NAME)
         pick :: fn(i: i64) -> i64 {
             held := [3, 5, 7, 11, 13]
             held[i]
         }
         total :: fn() -> i64 {
             held := [1, 2, 3, 4]
             var sum : i64 = 0
             var i : i64 = 0
             while (i < 4) { sum = sum + held[i]  i = i + 1 }
             sum
         }
         corner :: fn() -> i64 {
             p := Point { x = 10, y = 20 }
             p.x + p.y
         }
         repeated :: fn() -> i64 {
             held := [7; 5]
             held[4] + slice_len(held)
         }
         CHOSEN :: pick(3)
         SUM :: total()
         CORNER :: corner()
         REPEATED :: repeated()
         Sized :: struct { bytes: [TABLE[3]]u8 }
         main :: fn() -> i64 {
             print(\"{}\\n\", SLOTS)
             print(\"{}\\n\", DOWN)
             print(\"{}\\n\", LETTER)
             print(\"{}\\n\", WIDTH)
             print(\"{}\\n\", CHOSEN)
             print(\"{}\\n\", SUM)
             print(\"{}\\n\", CORNER)
             print(\"{}\\n\", REPEATED)
             print(\"{}\\n\", sizeof(Sized))
             print(\"{}\\n\", TABLE[1])
             print(\"{}\\n\", ORIGIN.x)
             0
         }
",
        "4\n4\n101\n5\n11\n10\n30\n12\n8\n2\n3\n",
    ),
    // The three that keep the low bits are worked out before the program runs
    // too. They exist so a value may leave its range on purpose, so a hash
    // folded here has to come out the same as one computed while the program
    // runs.
    (
        "the_wrapping_operations_fold",
        "import \"io.frost\"
         mixed :: fn(a: i64) -> i64 { wrap_mul(a, 2654435761) }
         rolled :: fn(a: i64) -> i64 { wrap_add(a, 1) }
         backed :: fn(a: i64) -> i64 { wrap_sub(a, 1) }
         MIX :: mixed(9223372036854775807)
         ROLL :: rolled(9223372036854775807)
         BACK :: backed(-9223372036854775807 - 1)
         main :: fn() -> i64 {
             print(\"{}\\n\", MIX)
             print(\"{}\\n\", ROLL)
             print(\"{}\\n\", BACK)
             var a : i64 = 9223372036854775807
             print(\"{}\\n\", wrap_mul(a, 2654435761))
             0
         }
",
        "9223372034200340047\n-9223372036854775808\n9223372036854775807\n9223372034200340047\n",
    ),
    // A compile-time number is read in three places, and a call may stand in
    // all of them: a constant's value, an array's length, and the value
    // argument a generic takes, whether written in a type or at a call.
    (
        "a_call_stands_where_a_compile_time_number_is_read",
        "import \"io.frost\"
         import \"columns.frost\"
         pow2 :: fn(n: i64) -> i64 {
             var held : i64 = 1
             while (held < n) { held = held * 2 }
             held
         }
         Particle :: struct { x: i64 }
         Buffer :: struct { bytes: [pow2(5)]u8 }
         main :: fn() -> i64 {
             var c : columns<Particle, pow2(5)> = columns_new()
             columns_reset(c)
             h := columns_insert(c, Particle { x = 7 })
             print(\"{}\\n\", c[h].x)
             print(\"{}\\n\", sizeof(Buffer))
             0
         }
",
        "7\n8\n",
    ),
    // Elementwise arithmetic over a fixed array of numbers, which is what a
    // vector register holds. `a + b` is one operation per lane, so what a lane
    // does is what a number does, and a number written beside a vector is that
    // number in every lane. The grid covers both element widths, both float
    // and whole-number lanes, a vector that is a parameter (which arrives as a
    // borrow), one that a call answered with, and nesting, because the two
    // compilers get here differently: the bootstrap writes the lanes out in its
    // IR, and the self-hosted compiler leaves a float vector as one operation
    // and emits packed instructions for it.
    (
        "arithmetic_over_a_vector_is_done_once_per_lane",
        "import \"io.frost\"
         blend :: fn(a: [4]f32, b: [4]f32) -> [4]f32 { a * b + a }
         ramp :: fn(k: f32) -> [4]f32 { [k, k * 2.0, k * 3.0, k * 4.0] }
         show :: fn(v: [4]f32) {
             var i : i64 = 0
             while (i < 4) {
                 print(\"{}\\n\", cast($f64, v[i]))
                 i = i + 1
             }
         }
         main :: fn() -> i64 {
             a : [4]f32 = [1.0, 2.0, 3.0, 4.0]
             b : [4]f32 = [5.0, 6.0, 7.0, 8.0]
             show(a + b)
             show(b - a)
             show(a * 2.0)
             show(2.0 * a)
             show(-a)
             show(blend(a, b))
             show(ramp(1.0) + ramp(2.0))
             show((a + b) * 2.0 - a)
             d : [2]f64 = [1.5, 2.5]
             e : [2]f64 = [0.5, 4.0]
             sum := d + e
             quotient := d / e
             print(\"{}\\n\", sum[0])
             print(\"{}\\n\", quotient[1])
             wide : [8]f32 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
             doubled := wide * 2.0
             print(\"{}\\n\", cast($f64, doubled[7]))
             var w : [4]i32 = [1, 2, 3, 4]
             var x : [4]i32 = [10, 20, 30, 40]
             y := w + x
             z := x * 2
             q := x & 12
             print(\"{}\\n\", cast($i64, y[0]) + cast($i64, y[3]))
             print(\"{}\\n\", cast($i64, z[2]))
             print(\"{}\\n\", cast($i64, q[1]))
             0
         }
",
        "6\n8\n10\n12\n\
         4\n4\n4\n4\n\
         2\n4\n6\n8\n\
         2\n4\n6\n8\n\
         -1\n-2\n-3\n-4\n\
         6\n14\n24\n36\n\
         3\n6\n9\n12\n\
         11\n14\n17\n20\n\
         2\n0.625\n16\n55\n60\n4\n",
    ),
    // `Enum::Variant` and `.Variant` name one variant. The self-hosted parser
    // read only the second, so the first left the enum name standing as an
    // expression and the arm was refused for a reason that had nothing to do
    // with the program: '4' is not a type this program declares.
    (
        "a_qualified_variant_pattern_names_the_same_variant",
        "import \"io.frost\"
         Step :: enum { Left, Right, Up }
         which :: fn(k: Step) -> i64 {
             match k {
                 case Step::Left: 4
                 case Step::Up: 6
                 case _: 5
             }
         }
         main :: fn() -> i64 {
             print(\"{}\\n\", which(Step::Left))
             print(\"{}\\n\", which(Step::Right))
             print(\"{}\\n\", which(Step::Up))
             0
         }
",
        "4
5
6
",
    ),
    // A `break` in a match arm leaves the loop around the match. The
    // self-hosted C backend wrote the match as a `switch`, and C's `break`
    // leaves one of those, so the loop went round again: the bootstrap and the
    // assembly backend stopped at 2 and this one printed 3 and 4 as well.
    (
        "a_break_in_a_match_arm_leaves_the_loop",
        "import \"io.frost\"
         main :: fn() -> i64 {
             var i : i64 = 0
             while (i < 5) {
                 match i {
                     case 2: { break }
                     case _: { print(\"{}\\n\", i) }
                 }
                 i = i + 1
             }
             print(\"{}\\n\", 99)
             0
         }
",
        "0
1
99
",
    ),
    // A name in a case compares against the constant it stands for, a boolean
    // arm covers one of the two values, and an arm still stands where two
    // earlier spans only partly cover it.
    (
        "a_case_reads_a_name_as_its_value",
        "import \"io.frost\"
         CH_0 :: 48
         digit :: fn(x: i64) -> i64 {
             match x {
                 case CH_0: 10
                 case _: 20
             }
         }
         either :: fn(x: bool) -> i64 {
             match x {
                 case true: 1
                 case false: 2
             }
         }
         overlap :: fn(x: i64) -> i64 {
             match x {
                 case 1..10: 1
                 case 5..20: 2
                 case _: 0
             }
         }
         pair :: fn(a: i64, b: i64) -> i64 {
             match (a, b) {
                 case (0, CH_0): 7
                 case (_, _): 99
             }
         }
         main :: fn() -> i64 {
             print(\"{}\\n\", digit(48))
             print(\"{}\\n\", digit(1))
             print(\"{}\\n\", either(true))
             print(\"{}\\n\", either(false))
             print(\"{}\\n\", overlap(1))
             print(\"{}\\n\", overlap(15))
             print(\"{}\\n\", pair(0, 48))
             print(\"{}\\n\", pair(0, 1))
             0
         }
",
        "10
20
1
2
1
2
7
99
",
    ),
    // A case bound past what a compare carries in its four bytes. The assembly
    // backend wrote `cmpq $imm, %rax` whatever the number was: `as` refused the
    // instruction outright, and the compiler's own assembler wrote the low half
    // and produced a program that compared against a different number, so this
    // answered 0 and 1 the wrong way round there and correctly everywhere else.
    // The bounds straddle the edge on both sides on purpose.
    (
        "a_case_bound_wider_than_an_immediate",
        "import \"io.frost\"
         LOW :: 4000000000
         HIGH :: 6000000000
         f :: fn(x: i64) -> i64 {
             match x {
                 case LOW..HIGH: 1
                 case 2147483647: 2
                 case 2147483648: 3
                 case -2147483648: 4
                 case -2147483649: 5
                 case 9223372036854775807: 6
                 case _: 0
             }
         }
         main :: fn() -> i64 {
             print(\"{}\\n\", f(4000000000))
             print(\"{}\\n\", f(5999999999))
             print(\"{}\\n\", f(6000000000))
             print(\"{}\\n\", f(2147483647))
             print(\"{}\\n\", f(2147483648))
             print(\"{}\\n\", f(-2147483648))
             print(\"{}\\n\", f(-2147483649))
             print(\"{}\\n\", f(9223372036854775807))
             0
         }
",
        "1
1
0
2
3
4
5
6
",
    ),
    // An arm may name several patterns, and an arm over whole numbers may name
    // a span. The two compose because both are covered sets: `0 | 5..10` is one
    // number and one span, and the body runs for any value either takes.
    (
        "a_case_names_several_patterns_and_spans",
        "import \"io.frost\"
         Step :: enum { Left, Right, Up, Down }
         CH_0 :: 48
         CH_9 :: 57
         sideways :: fn(k: Step) -> i64 {
             match k {
                 case .Left | .Right: 1
                 case .Up: 2
                 case _: 3
             }
         }
         qualified :: fn(k: Step) -> i64 {
             match k {
                 case Step::Left | Step::Down: 4
                 case _: 5
             }
         }
         classify :: fn(c: i64) -> i64 {
             match c {
                 case 97..=122: 1
                 case CH_0..=CH_9: 2
                 case 0 | 5..10: 3
                 case -4..-1: 7
                 case 20..=20: 8
                 case _: 0
             }
         }
         main :: fn() -> i64 {
             print(\"{}\\n\", sideways(Step::Left))
             print(\"{}\\n\", sideways(Step::Right))
             print(\"{}\\n\", sideways(Step::Up))
             print(\"{}\\n\", sideways(Step::Down))
             print(\"{}\\n\", qualified(Step::Left))
             print(\"{}\\n\", qualified(Step::Down))
             print(\"{}\\n\", qualified(Step::Up))
             print(\"{}\\n\", classify(97))
             print(\"{}\\n\", classify(122))
             print(\"{}\\n\", classify(48))
             print(\"{}\\n\", classify(57))
             print(\"{}\\n\", classify(0))
             print(\"{}\\n\", classify(5))
             print(\"{}\\n\", classify(9))
             print(\"{}\\n\", classify(10))
             print(\"{}\\n\", classify(-4))
             print(\"{}\\n\", classify(-1))
             print(\"{}\\n\", classify(20))
             print(\"{}\\n\", classify(21))
             0
         }
",
        "1
1
2
3
4
4
5
1
1
2
2
3
3
3
0
7
0
8
0
",
    ),
    // A literal on the right of an `=` takes its type from the place on the
    // left, the way one beside an annotation takes it from there. The
    // self-hosted compiler resolves an inferred literal while it parses, out of
    // the type the context expects, and the assignment path set no expectation
    // at all: every one of these was refused there and taken by the bootstrap.
    // A `mut` parameter is in because it travels as an address, so the place is
    // what it points at rather than the pointer.
    (
        "a_literal_takes_its_type_from_the_place_it_lands_in",
        "import \"io.frost\"
         Phase :: enum { Opening, Streaming { sent: i64 }, Draining }
         Holder :: struct { phase: Phase, mark: i64 }
         Point :: struct { x: i64, y: i64 }
         begin :: fn(mut h: Holder) { h.phase = .Streaming { sent = 4 } }
         reading :: fn(p: Phase) -> i64 {
             match p {
                 case .Opening: 0
                 case .Streaming { sent }: sent
                 case .Draining: 9
             }
         }
         main :: fn() -> i64 {
             var loose: Phase = .Opening
             loose = .Draining
             print(\"{}\\n\", reading(loose))
             var h: Holder = { phase = .Opening, mark = 1 }
             h.phase = .Draining
             print(\"{}\\n\", reading(h.phase))
             begin(h)
             print(\"{}\\n\", reading(h.phase))
             var p: Point = { x = 1, y = 2 }
             p = { x = 5, y = 6 }
             print(\"{}\\n\", p.x + p.y)
             var row: [2]Point = [{ x = 1, y = 1 }, { x = 2, y = 2 }]
             row[1] = { x = 7, y = 8 }
             print(\"{}\\n\", row[1].x + row[1].y)
             0
         }
",
        "9
9
4
11
15
",
    ),
    // A borrow of a scalar, read as the value it borrows.
    //
    // Reading a borrow reads what it borrows, which for an aggregate is what a
    // field access does and for a scalar there is nothing to read through. So
    // all four of these compared the address the borrow holds against the
    // number beside it and answered no every time: `arena_at` on an arena of
    // numbers is the shape that found it, and it answers a `ref i64` the same
    // way it answers a `ref` of any struct. The name reads through where it is
    // written and a call answering one reads through where it is used, since
    // what a call answers with is known only once every signature has been
    // read. A `ref T` handed back by a function answering one stays the borrow.
    //
    // A list element is a value like any other, and it is the one place a value
    // arrives with nothing having asked for a type, so it is the one place a
    // borrow travelled as itself: both compilers told a format string to write
    // out a `^i64`.
    (
        "a_borrow_of_a_number_is_the_number",
        "import \"io.frost\"\nBag :: struct { data: ^i64, count: i64 }\n\
         at :: fn(b: Bag, i: i64) -> ref i64 {\n\
         \x20   unsafe {\n\
         \x20       ref r := b.data[i]\n\
         \x20       r\n\
         \x20   }\n}\n\
         main :: fn() -> i64 {\n\
         \x20   var cells : [4]i64 = [7, 8, 9, 10]\n\
         \x20   b := Bag { data = unsafe { ptr_to(cells[0]) }, count = 4 }\n\
         \x20   ref bound := cells[2]\n\
         \x20   print(\"{}\n\", bound == 9)\n\
         \x20   print(\"{}\n\", at(b, 2) == 9)\n\
         \x20   print(\"{}\n\", at(b, 1) + 1)\n\
         \x20   print(\"{}\n\", at(b, 3))\n\
         \x20   bound = 12\n\
         \x20   print(\"{}\n\", cells[2])\n\
         \x20   0\n}\n",
        "1
1
9
10
12
",
    ),
    // The same rule, at every operator that reads an operand rather than only
    // at the two a comparison has. A borrow of a scalar is read as the value it
    // borrows by the coercion each backend already puts every value through, so
    // a truth value under `&&`, a decimal under `+`, a byte under checked
    // arithmetic, a shift amount and an operand inside a generic's body all
    // read the same. Each of those is a path of its own in at least one
    // backend: the short-circuit that jumps, the one done at f64 and narrowed,
    // the one done at 64 bits and held to its width, the amount handed to
    // `frost_shift`, and the body parsed again per instance.
    (
        "a_borrow_of_a_number_beside_every_operator",
        "import \"io.frost\"
         Truths :: struct { data: ^bool, count: i64 }
         Reals :: struct { data: ^f64, count: i64 }
         Bytes :: struct { data: ^u8, count: i64 }
         truth :: fn(t: Truths, i: i64) -> ref bool {
             unsafe {
                 ref v := t.data[i]
                 v
             }
}
         real :: fn(r: Reals, i: i64) -> ref f64 {
             unsafe {
                 ref v := r.data[i]
                 v
             }
}
         byte :: fn(b: Bytes, i: i64) -> ref u8 {
             unsafe {
                 ref v := b.data[i]
                 v
             }
}
         widened :: fn(b: Bytes, v: $T) -> i64 { byte(b, 0) + sizeof(T) }
         main :: fn() -> i64 {
             var flags : [2]bool = [true, false]
             var reals : [2]f64 = [1.5, 2.5]
             var bytes : [2]u8 = [200, 3]
             var six : i64 = 6
             t := Truths { data = ptr_to(flags[0]), count = 2 }
             r := Reals { data = ptr_to(reals[0]), count = 2 }
             b := Bytes { data = ptr_to(bytes[0]), count = 2 }
             print(\"{}\\n\", truth(t, 0) && truth(t, 1) == false)
             print(\"{}\\n\", truth(t, 1) || truth(t, 0))
             print(\"{}\\n\", real(r, 0) + real(r, 1) > 3.9)
             print(\"{}\\n\", real(r, 1) > real(r, 0))
             print(\"{}\\n\", byte(b, 0) + byte(b, 1))
             print(\"{}\\n\", six << byte(b, 1))
             print(\"{}\\n\", widened(b, 0))
    0
}
",
        "1
1
1
1
203
48
208
",
    ),
    // A generic whose parameter is written `Bag<$T>` calling another one
    // written the same way. Which argument stands for the type parameter is
    // read off the declaration, and a parameter naming it inside a generic
    // struct's argument list read as naming it nowhere, so the tuple fell
    // through to the first argument whose type was known: the struct itself.
    // `outer` was then compiled for `Bag<i64>`, whose own parameter is a
    // `Bag<Bag<i64>>` nothing wrote, and each round of the instantiation
    // fixpoint went one level deeper until the arena holding them was full.
    // A caller that is not itself generic never reached that, because the
    // argument's type is known there and the instance it names carries the
    // tuple.
    (
        "a_generic_hands_its_own_argument_to_another",
        "import \"io.frost\"\nBag :: struct($T: Type) { one: T }
         only :: fn(a: Bag<$T>) -> ref T {
             ref held := a.one
    held
}
         plus :: fn(a: Bag<$T>, n: i64) -> i64 { only(a) + n }
         width :: fn(a: Bag<$T>) -> i64 { sizeof(T) }
         main :: fn() -> i64 {
             var carrier : Bag<i64> = Bag { one = 7 }
             print(\"{}\\n\", plus(carrier, 5))
             print(\"{}\\n\", width(carrier))
             0
}
",
        "12
8
",
    ),
    // `!` wherever a boolean expression is written: over a comparison, over a
    // call answering one, doubled, beside `&&` and `||`, and with brackets
    // deciding what it applies to. `!=` is a token of its own and `-> T ! E`
    // marks a failure set, and a program writing all three reads the same in
    // both.
    (
        "negation_beside_what_else_reads_bang",
        "import \"io.frost\"
Fault :: enum { Bad }
         risky :: fn(n: i64) -> i64 ! Fault {
             if (n < 0) {
                 return .Bad
             }
             n
}
         ok :: fn(n: i64) -> bool { n > 0 }
         main :: fn() -> i64 {
             a := 3
             print(\"{}\n\", a != 4)
             print(\"{}\n\", !ok(a))
             print(\"{}\n\", !ok(a) && a > 2)
             print(\"{}\n\", !ok(a) || a > 2)
             print(\"{}\n\", !(ok(a) && a > 5))
             print(\"{}\n\", !!ok(a))
             print(\"{}\n\", !(a > 2) == false)
             0
}
",
        "1
0
0
1
1
1
1
",
    ),
    // The three roles `!` reads in, side by side: a failure set with a space on
    // both sides, a negation against what it negates, and `!=`, which is one
    // token and untouched by either rule.
    (
        "the_three_roles_a_bang_reads_in",
        "import \"io.frost\"
Bad :: enum { Nope }
         risky :: fn(n: i64) -> i64 ! Bad {
             if (n < 0) {
                 return .Nope
             }
             n
}
         ready :: fn() -> bool { true }
         main :: fn() -> i64 {
             print(\"{}\n\", !ready())
             print(\"{}\n\", 1 != 2)
             0
}
",
        "0
1
",
    ),
    // A block comment. The bootstrap lexed one and the self-hosted lexer had no
    // case for it at all, so the same file compiled through one compiler and
    // was refused by the other with "this is not the start of a declaration".
    // Neither nests, so the first `*/` closes what the `/*` opened.
    (
        "a_block_comment_is_dropped_by_both_lexers",
        "/* the head of the file */
         import \"io.frost\"
         /* several
            lines, and a // inside one */
         main :: fn() -> i64 {
             count := 2   /* trailing */
             print(\"{}\n\", count) /* and one holding a \"string\" */
             0
}
",
        "2
",
    ),
    // A length is an `i64`. The bootstrap answered `usize` from `slice_len` and
    // `str_len` while the self-hosted compiler answered `i64`, and both are
    // eight bytes that convert silently, so nothing was refused: comparison,
    // division, remainder and right shift follow the type, so subtracting past
    // zero aborted under one compiler and answered a negative number under the
    // other.
    (
        "a_length_is_signed_in_both_compilers",
        "import \"io.frost\"
         main :: fn() -> i64 {
             text := \"abcd\"
             data : [4]i64 = [1, 2, 3, 4]
             view : []i64 = data
             print(\"{}\n\", (str_len(text) - 8) / 2)
             print(\"{}\n\", (slice_len(view) - 8) / 2)
             print(\"{}\n\", (slice_len(data) - 8) / 2)
             print(\"{}\n\", 0 - str_len(text))
             print(\"{}\n\", str_len(text) / 3)
             0
}
",
        "-2
-2
-2
-4
1
",
    ),
];

// Build and run `source` with the self-hosted compiler through one of its
// backends, under the settings a build gets by default. The shared helper turns
// the unsafe audit off, and that pass is part of what decides these programs'
// meaning: one of the cases below compiles either way with it off and is
// refused with it on.
// Both self-hosted backends run every case, because a difference between them
// is a difference in the language. The C one wrote an integer literal without
// the suffix that makes it sixty-four bits wide, so every shift past the
// thirty-second bit answered with zero there and correctly everywhere else.
// `frost fmt` is part of the language the same way a refusal is: a tree is held
// to one rendering, and two compilers writing two renderings would make which
// one formatted it decide what the file says. Nothing compared them until now,
// and the moment something did it found five files the two laid out differently.
//
// The corpus is already what the bootstrap writes, so formatting it again asks
// only whether the self-hosted compiler leaves a settled file alone. That is
// worth asking and it is not the question: every one of those five differed on
// input that was *not* settled, and a check at the fixed point would have found
// none of them. So each file is put out of shape first, by breaking every line
// at the places a layout owns, and the two are compared on that.
// A formatter that drops what it does not recognize deletes source. Both read
// only `//` and left a block comment's bytes to the step that keeps nothing, so
// a file holding one came back without it and nothing said so. The corpus holds
// none, which is why the corpus test above never saw it.
#[test]
fn both_compilers_keep_a_block_comment() {
    let Some(compiler) = build_self_hosted_compiler("blockcomment") else {
        return;
    };
    let source = concat!(
        "/* At the top.
",
        "   A second line, indented on purpose. */
",
        "
",
        "import \"io.frost\"
",
        "
",
        "/* Between declarations. */
",
        "double :: fn(n: i64) -> i64 { n * 2 } /* trailing */
",
        "
",
        "main :: fn() -> i64 {
",
        "    /* inside a body */
",
        "    x := double(3) /* beside a statement */
",
        "    print(\"{}\n\", x)
",
        "    // a line comment still works
",
        "    0
",
        "}
"
    );
    assert_eq!(
        frost::format_source(source),
        source,
        "the bootstrap did not write a block comment back"
    );
    let Some(written) = self_hosted_format(&compiler, source) else {
        return;
    };
    assert_eq!(
        written, source,
        "the self-hosted compiler did not write a block comment back"
    );
}

#[test]
fn both_compilers_format_the_corpus_the_same_way() {
    let Some(compiler) = build_self_hosted_compiler("formatparity") else {
        return;
    };
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut settled = Vec::new();
    let mut differing = Vec::new();
    for file in corpus() {
        let Ok(source) = std::fs::read_to_string(&file) else {
            continue;
        };
        let shown = file
            .strip_prefix(&root)
            .unwrap_or(&file)
            .display()
            .to_string();
        for (source, shape) in [
            (source.clone(), "as committed"),
            (out_of_shape(&source), "broken at every bracket and comma"),
        ] {
            let wanted = frost::format_source(&source);
            let Some(written) = self_hosted_format(&compiler, &source) else {
                continue;
            };
            if written != wanted {
                // Where they part, since a parity failure naming only the file
                // costs whoever reads it the bisect that found this one.
                let at = wanted
                    .lines()
                    .zip(written.lines())
                    .position(|(left, right)| left != right)
                    .unwrap_or(
                        wanted.lines().count().min(written.lines().count()),
                    );
                let window = |text: &str| -> String {
                    text.lines()
                        .skip(at.saturating_sub(3))
                        .take(7)
                        .map(|line| format!("      {line}"))
                        .collect::<Vec<String>>()
                        .join("\n")
                };
                differing.push(format!(
                    "{shown}  ({shape})  parting at line {}\n    bootstrap:\n{}\n    self-hosted:\n{}",
                    at + 1,
                    window(&wanted),
                    window(&written)
                ));
            }
            // Only the file as committed is a claim about the tree, and it is
            // the bootstrap that made it. Said the other way round, a file
            // somebody edited without formatting reads as the two compilers
            // disagreeing.
            if shape == "as committed" && wanted != source {
                settled.push(shown.clone());
            }
        }
    }
    assert!(
        settled.is_empty(),
        "these are not what `frost fmt` writes, so run it over them before \
         reading anything else here:\n{}",
        settled.join("\n")
    );
    assert!(
        differing.is_empty(),
        "the two compilers lay these out differently:\n{}",
        differing.join("\n")
    );
}

// The same tokens with a line break after every bracket that opens and every
// comma, which is every place a layout decides for itself. What comes out is
// the same program said as badly as it can be said, and both compilers have to
// answer it with the same thing.
fn out_of_shape(source: &str) -> String {
    let Some(pieces) = frost::tokens_and_gaps(source) else {
        return source.to_string();
    };
    let mut held = String::with_capacity(source.len() * 2);
    for (index, piece) in pieces.iter().enumerate() {
        held.push_str(piece);
        // The gaps sit at the even positions and the tokens between them.
        if index % 2 == 1 && matches!(piece.as_str(), "(" | "[" | "{" | ",") {
            held.push('\n');
        }
    }
    held
}

// What the self-hosted compiler writes for a source, or nothing when it could
// not be asked.
fn self_hosted_format(
    compiler: &std::path::Path,
    source: &str,
) -> Option<String> {
    let scratch = std::env::temp_dir()
        .join(support::unique("frost_fmt"))
        .with_extension("frost");
    std::fs::write(&scratch, source).ok()?;
    let ran = Command::new(compiler).arg("fmt").arg(&scratch).output();
    let written = std::fs::read_to_string(&scratch).ok();
    let _ = std::fs::remove_file(&scratch);
    ran.ok()?;
    written
}

// Every `.frost` file the repository tracks, which is what both formatters are
// held to.
fn corpus() -> Vec<std::path::PathBuf> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut found = Vec::new();
    let mut stack = vec![
        root.join("std"),
        root.join("lib"),
        root.join("selfhosted"),
        root.join("examples"),
        root.join("tools"),
    ];
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

#[test]
fn both_compilers_agree_on_these_programs() {
    let Some(compiler) = build_self_hosted_compiler("samelanguage") else {
        return;
    };
    for (name, source, want) in SAME_LANGUAGE_CASES {
        let Some(bootstrap) = bootstrap_output(name, source) else {
            return;
        };
        assert_eq!(bootstrap, *want, "the bootstrap disagreed about {name}");
        for (backend, suffix) in [("--emit-asm", "s"), ("--emit-c", "c")] {
            let hosted = selfhosted_default_output(
                &compiler, name, source, backend, suffix,
            );
            assert_eq!(
                hosted, *want,
                "the self-hosted compiler's {backend} disagreed about {name}"
            );
        }
    }
}
