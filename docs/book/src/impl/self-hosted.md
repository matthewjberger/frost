# The self-hosted compiler

`selfhosted/frost.frost` is a compiler for Frost, written in Frost. It lexes,
parses, type-checks, and emits either a C translation unit or x86-64 assembly.
Which of the two compilers is which, and why the Rust bootstrap in `src/` exists
at all, is in [build-modes.md](build-modes.md).

It uses Frost-native arenas for the tokens, the AST and the symbol tables,
integer indices between nodes, and free functions over that data. It has no
closures and no dynamic collections, every reference in it is second-class, and
its memory comes from the language's own allocator.

## What it implements

It checks its own programs. Behind the assembly backend nothing downstream
catches a mistake.

- Types and layout. The whole scalar set (`i8` through `i64` and `isize`, their
  unsigned twins, `f32`, `f64`, `bool`), `str`, pointers `^T`, fixed
  arrays `[N]T`, slices `[]T`, `Handle<T>`, structs passed and returned by
  value, and `e^`, `e[i]`, `ptr_to`, `ptr_cast`, `sizeof`.
- Ownership and linearity. Use after move, and `linear` values that must be
  consumed exactly once, with a read parameter borrowing and `move` consuming.
- Generics by monomorphization over `$T`, on structs and functions, instantiated
  once per concrete type argument.
- Enums with payloads, and `match` over them.
- Loop control. `break` leaves the innermost loop and `continue` goes round it
  again, both refused outside one.
- `for`, over a sequence (`for item in items`, `for index, item in items`) and
  over a range (`for i in 0..n`, `for i in 0..=n`). Each is written out as the
  counted loop it stands for, so the node kinds, the passes and the backends
  stay as they are.
- Compile-time lists. `args: $...`, walked with `for`, indexed by a literal,
  holding values or types, handed on to another list by naming it, and expanded
  into a call's argument list once per element with `g(T) for T in list`.
- Allocation sources. `uses A` on a function, `with a { }` around a call.
- Regions. A `with` block is a region and an arena pointer may not outlive it.
- Failure sets. `-> T ! E` and `?`.
- Imports. `import "path"` joins another file's declarations to this one.
- `extern fn` for C linkage, which is how it does its own IO, and
  `safe extern fn` for one audited not to need an `unsafe` block at its calls.
  See [c-compatibility.md](c-compatibility.md).
- `ref T`, a returnable borrow of `T`: a raw pointer's layout but one the frame
  and region checks hold to what outlives the call, so `arena_at` returns one
  and arena access needs no `unsafe`.
- The unsafety gate, enforced at every site it covers. Reading through a raw
  pointer, `ptr_cast`, and calling an `extern fn` that is not `safe` are refused
  outside an `unsafe` block, so `unsafe` is the complete list of places to look
  when memory is corrupted. It passes its own gate on its own source, and
  nothing turns it off.
- `test` blocks, run by `--test`, which reports each test and summarises.
- Diagnostics carrying a file, line and column, so a refusal names the position
  it refused.

Both backends emit from the same checked program: C through the `emit_*`
helpers in `emit.frost`, or x86-64 assembly directly. A form neither compiler
supports is refused with the position it was written at.

## The modules

Each file states what it is about at the top and lists what it offers on one
`export` line.

`core` holds the externs, the constant tables, the records and the arena.
`lexer` and `cursor` turn source into tokens and read them. `imports` lays every
file's text into one buffer. `names` interns and resolves. `types` does the
typing and the checks that ride on it. `parser` is recursive descent. `layout`
works out sizes and offsets, both backends write through `emit`, and
`emit_c` and `emit_asm` are the backends. `regions` is the region check,
`query` answers what an editor asks of a checked program, and `frost` is the
driver.

The import order is acyclic. Both backends write through `emit`, and `emit_c`
reads the argument and unit-membership walks out of `emit_asm` rather than
writing them a second time.

## How the two compilers differ

Both compilers accept the same language, so Frost can be built from a Rust
toolchain: the bootstrap compiles this compiler, this compiler compiles
itself, and no seed binary enters the build. A divergence is a hole in that,
whichever side it is on.

Parallelism. The bootstrap generates code on every core from a shared work
queue. The self-hosted compiler emits one unit at a time, and threads only the
assembler runs it hands to the toolchain, since each of those is a process it
waits on.

Output. The bootstrap emits an object through Cranelift, portable C, or runs the
IR directly. This one emits C or x86-64 assembly of its own, and encodes that
assembly to an object itself, COFF or ELF as the target asks.

## Faults are reported, then refused

The compile continues past the point a fault is found. The runtime carries a
recovery mark (`frost_rt_recover_run` arms one and runs a callback,
`frost_rt_recover_escape` returns to the nearest mark): the parser arms one
around each top-level declaration and each statement of a block, and the type
and move walks arm one around each function, so one run reports every fault a
file holds and the driver refuses the program once the whole file has had its
say. With no mark armed an escape dies exactly as `frost_rt_die` does, so a
report site outside every mark still ends the run.

The wording is the bootstrap's, byte for byte.
`both_compilers_report_the_same_fault_lines` holds the two to the same
faults, on the same lines, in the same words, and
`self_hosted_answers_editor_queries` holds `FROST_QUERY` (symbols, a
definition's line, a struct's fields, a local's type) to the answers the
bootstrap's `src/tools/query.rs` gives in its own tests.

## A property every stage shares cancels out

Three stages agreeing byte for byte has one blind spot. The stages are compared
against each other, so any property all of them share cancels out.

`core.autocrlf` is on for most Windows checkouts, so a file git writes has CRLF
and a file an editor writes may not, and every string literal holding a raw
newline captures whichever it was. One checkout emitted 4,170 stray carriage
returns and another none, with all three stages reading the same source and the
fixpoint holding both times. Comparing the output of two compilers built at
different times finds a difference of that kind.

The same reasoning covers everything a build reads besides the program: the
environment, the standard library on disk, and the runtime beside the compiler.
A build resolves its C runtime by walking up from the binary, and the object it
compiles is keyed on what that file holds, so "the runtime changed" and "a
different runtime is being read" both present as a cache key that does not
match. A failing link names the runtime path it resolved, since the linker only
names the symbol it could not find.

## Compile speed

Both compilers clear the 100,000 lines per second target. Measure it with
`just bench-selfhost`, which runs this compiler over its own source through both
backends. The assembly backend is the faster of the two, since it writes machine
code where the C backend hands text to a C compiler. The recipes are the record,
since a rate written into a page is a rate about one machine on one day.

The language compiles fast by what the spec leaves out. There are no traits or
typeclasses, so no constraint solving. No lifetimes, so the region check is a
cheap flow pass. No global type inference, only local. No macros and no
declaration-generating comptime, only specialization. No textual includes, only
modules parsed once.

That leaves the tools a build shells out to. On the C path an external C
compiler costs several times the whole Frost front end and scales worse, and the
assembly backend keeps it out of the loop.

On the native path the C compiler is out of the per-build loop. The runtime is
compiled once into an object cached in the temp directory, keyed by a hash of
its source and the tool that built it, and linked thereafter. The assembler is
in the compiler as well: `selfhosted/assemble.frost` encodes the emitted text
and `coff.frost` writes the object, so on Windows a build reaches an object
without running `as` and without the text ever becoming a file. That roughly
halves `--native` on the programs measured, since spawning a process per module
is most of what the other path costs.

The emitted text is the last cost on the direct path. Most of a build goes to
the front end and to formatting the assembly, with encoding it and writing the
object a minority. Handing the backend's instructions to the encoder as records
reaches the rest.

The linker invocation stays outside the compiler. On the bootstrap it is about
two thirds of a small build and is mostly fixed process and driver overhead, so
it barely grows with program size.

Passing `-fuse-ld=lld` to the driver on the bootstrap makes no difference:
0.113 s against 0.106 s with lld present on this machine, inside noise. The cost
is driver overhead and the link itself is a small part of it.

The only way to remove that last cost is to stop invoking an external tool at
all, which means emitting the executable directly, PE on Windows and ELF on
Linux. That is a mini-linker (symbol resolution, relocations, imports) and is
what Jai does. The other half of going C-free is the runtime: the pool lives in
`std/slab.frost` as ordinary Frost, `--freestanding` links with no libc, and
what is left in C is the aborts, the assertions and the IO.

The encoder writes ELF as well as COFF, so every platform reaches an object the
same way, at one speed and with one set of failure modes.

Two ELF differences reach back into the encoding. A reference to a name the file
both defines and offers is left for the linker, since another object may take
that name over at load time, where COFF has no such rule and `as` settles it in
the assembler. And a fixup carries its own addend instead of reading one out of
the bytes it fills in, so those are left empty. `assemble.frost` is told which
format it is encoding for, and both follow from the answer.

Checked the same way: clang assembles the same text and the two objects are
compared byte for byte, over the compiler's own 660 KB of code and 8,252 fixups.
`FROST_OBJECT` names the format from either host, the way `FROST_ABI` names the
calling convention, so the ELF half can be checked from Windows.

## Scaling past one file

Whole-program monomorphization specializes generics per type, the
specializations are a cross product, and no incremental or separate compilation
bounds the work. `just bench-scaling` spans 1,020 to 64,608 generated lines and
640 to 10,240 specializations, and four times the input costs roughly four times
the time on both the front-end and the full-native curves. The pipeline is close
to linear, with a mild superlinear term that grows with function count. Read
ratios there, since process startup is a fixed cost inside every figure and it
dominates the small end.

The front end is where the time goes. Parse, parameter modes, regions,
ownership, IR lowering, type checking, monomorphization to fixpoint and C
emission together stay near-linear because every one of them is a local pass: no
traits to solve, no lifetimes to infer, no global inference, and the
specialization worklist dedups through a hash set.

Every program in that benchmark is a single file, so a change to one line
rebuilds everything however little it reaches. `just bench-incremental` measures
the other shape, on a bootstrap program spread across many files with one of
them changed. See [separate-compilation.md](separate-compilation.md).

The Frost compiler answers that shape from its own output, since it emits
assembly. `--incremental` emits one assembly unit per module and encodes each to
its own object, and a module whose emitted assembly is byte for byte last
build's is not encoded again. The cache key is that assembly: the compiler has
just produced what the module compiles to, so nothing else has to be hashed or
walked. Where the encoder runs, the unit stays in memory and no assembly file is
written at all. Where the toolchain does the encoding, which is the C backend,
the unit is written to `<build>/m<n>.c` for the C compiler to read.

`just bench-selfhost-incremental` puts a whole-program build, a first
incremental build and one where nothing changed side by side on its own source.
The compiler that comes out is byte for byte the one the whole-program build
produces, which a test checks.

What keeps the backend off the curve, on the bootstrap:

1. Functions compile in parallel. The type system is local and
   signature-based, so once signatures are collected the functions are
   independent. Code generation runs on every core and is a minority of a full
   build, with the front end holding the rest.
2. Modules compile separately. Each module is its own object on the link path,
   monomorphization is seeded per module, `--incremental` skips the modules an
   edit cannot reach, and a skipped module contributes signatures in place of
   bodies, so the front end never walks code it will not emit. See
   [separate-compilation.md](separate-compilation.md).
3. Specializations carry across builds, since a module's object holds the ones
   that module asked for and reusing the object reuses them.

Two generator mistakes here read as compiler results. A generated program that
names a function `f32` times a parse error, and a program whose `main` holds
thousands of call sites makes parallel code generation look idle, since one
function is one thread however many cores there are.

`parse_generic_instance` memoizes `(template, argument)`, because three passes
ask for the same instance: once to record its concrete return type, and once
each for its prototype and its body. Parsing each template once and substituting
types into the AST per instantiation would go further, and it needs a
substitution pass this compiler does not have, since binding happens during the
parse.

Emission runs on one thread. It could run on every core, since the type system
is local and signature-based and a unit is buffered in memory. Emitting is most
of what a build spends, and it formats text that is immediately read back, so
handing the encoder records removes that work instead of dividing it.
