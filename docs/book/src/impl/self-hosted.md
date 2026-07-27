# The self-hosted compiler

`selfhosted/frost.frost` is a compiler for Frost, written in Frost. It lexes,
parses, type-checks, and emits either a C translation unit or x86-64 assembly.
Which of the two compilers is which, and why the Rust bootstrap in `src/` exists
at all, is in [build-modes.md](build-modes.md).

It is written the way the language wants a compiler written: Frost-native arenas
for the tokens, the AST and the symbol tables, integer indices instead of heap
pointers between nodes, and free functions over that data. No closures, no
dynamic collections, every reference second-class, and its memory comes from the
language's own allocator rather than a runtime pool.

## What it implements

Not a toy subset. It checks its own programs rather than deferring to whatever
compiles its output, which it has to, because through the assembly backend
nothing downstream would catch a mistake.

- Types and layout. `i64`, `i8`, `u8`, `bool`, `str`, pointers `^T`, fixed
  arrays `[N]T`, slices `[]T`, `Handle<T>`, structs passed and returned by
  value, and `e^`, `e[i]`, `ptr_to`, `ptr_cast`, `sizeof`.
- Ownership and linearity. Use after move, and `linear` values that must be
  consumed exactly once, with a read parameter borrowing and `move` consuming.
- Generics by monomorphization over `$T`, on structs and functions, instantiated
  once per concrete type argument.
- Enums with payloads, and `match` over them.
- Loop control. `break` leaves the innermost loop and `continue` goes round it
  again, both refused outside one.
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
- The unsafety gate, enforced rather than merely parsed. Reading through a raw
  pointer, `ptr_cast`, and calling an `extern fn` that is not `safe` are refused
  outside an `unsafe` block, so `unsafe` is the complete list of places to look
  when memory is corrupted. It passes its own gate on its own source.
  `FROST_CHECK_UNSAFE=0` turns it off, as in the bootstrap.
- `test` blocks, run by `--test`, which reports each test and summarises.
- Diagnostics carrying a file, line and column, since a compiler that refuses a
  program owes you the position.

Both backends emit from the same checked program: C through `frost_emit_*`
helpers, or x86-64 assembly directly. A form neither compiler supports is
refused with the position it was written at, rather than misparsed into a crash
somewhere else.

## The modules

Each file states what it is about at the top and lists what it offers on one
`export` line, so the shape of the compiler is readable from the imports.

`core` holds the externs, the constant tables, the records and the arena.
`lexer` and `cursor` turn source into tokens and read them. `imports` lays every
file's text into one buffer. `names` interns and resolves. `types` does the
typing and the checks that ride on it. `parser` is recursive descent. `layout`
works out sizes and offsets, `emit` is what both backends write through, and
`emit_c` and `emit_asm` are the backends. `regions` is the region check, and
`frost` is the driver.

The import order is acyclic, and the assembly backend does not depend on the C
one, which is what `emit` is for.

## Where the two compilers still differ

The two compilers accepting the same language is a requirement of this project.
It is what lets Frost be built from a Rust toolchain and nothing else: the
bootstrap compiles this compiler, this compiler compiles itself, and no seed
binary is needed anywhere. A divergence is a hole in that, whichever side it is
on. What is left is not a difference in what a program may say.

Parallelism. The bootstrap generates code on every core from a shared work
queue. The self-hosted compiler emits and assembles one unit at a time.

Output. The bootstrap emits an object through Cranelift, portable C, or runs the
IR directly. This one emits C or x86-64 assembly of its own, and encodes that
assembly to an object itself, COFF or ELF as the target asks.

## What the fixpoint cannot see

Three stages agreeing byte for byte is the strongest check here, and it has one
blind spot worth knowing: **any property every stage shares is invisible to it.**
The stages are compared against each other, so anything common to all of them
cancels.

That is not a hypothetical. `core.autocrlf` is on for most Windows checkouts, so
a file git writes has CRLF and a file an editor writes may not, and every string
literal holding a *raw* newline captured whichever it was. The compiler emitted
4,170 stray carriage returns on one checkout and none on another. All three
stages read the same source, so all three did it, so the fixpoint held. It was
found by comparing the output of two compilers built at different times, not by
any test in the suite.

The same reasoning covers everything a build reads that is not the program: the
environment, the standard library on disk, and the runtime beside the compiler.
A build resolves its C runtime by walking up from the binary, and the object it
compiles is keyed on what that file holds, which cannot tell "the runtime
changed" apart from "a different runtime is being read". Both present as a cache
key that does not match, and only one of them is a stale cache. A failing link
therefore names the runtime path it resolved, because the linker only names the
symbol it could not find.

So: a check that compares two things is blind to whatever they have in common.
Reach outside the loop for anything that has to hold absolutely, which for the
emitted assembly means a byte comparison against a build from a different tree.

## Compile speed

Both compilers clear the 100,000 lines per second target, and
[roadmap.md](../roadmap.md) is where every measurement lives, each beside the
`just` recipe that produces it. The two that matter here, from
`just bench-selfhost` on the self-hosted compiler's own 14,273 lines: about
145,000 lines per second through its C backend, and about 130,000 through its
assembly backend.

The language is built to compile fast. There are no traits or typeclasses, so no
constraint solving. No lifetimes, so the region check is a cheap flow pass. No
global type inference, only local. No macros and no declaration-generating
comptime, only specialization. No textual includes, only modules parsed once.
Those are the features that make other compilers slow, and the spec omits them
on purpose.

Which leaves the tools a build shells out to. On the C path an external C
compiler costs several times the whole Frost front end and scales worse, and
removing it from the loop is what the assembly backend is for. That is also why
self type-checking had to come first: once there is no C compiler behind the
compiler, nothing downstream catches a mistake.

On the native path the C compiler is out of the per-build loop. The runtime is
compiled once into an object cached in the temp directory, keyed by a hash of
its source and the tool that built it, and linked thereafter. The assembler went
the same way: `selfhosted/assemble.frost` encodes the emitted text and
`coff.frost` writes the object, so on Windows a build reaches an object without
running `as` and without the text ever becoming a file. That took `--native` on
`std/ecs.frost` from 155 ms to 86 and on the compiler's own source from 327 to
222, and a cold incremental build of the compiler from 675 ms to 366.

What remains of the emitted text is what the direct path has still to remove: of
the 86 ms, 64 is the front end and formatting the assembly, and 22 is encoding
it and writing the object. Handing the backend's instructions to the encoder as
records rather than as text is what reaches the rest.

What remains outside the compiler is the linker invocation, which on the
bootstrap is about two thirds of a small build and is mostly fixed process and
driver overhead rather than linking work, so it barely grows with program size.

Measured and rejected on the bootstrap: passing `-fuse-ld=lld` to the driver.
lld is present on this machine and made no difference (0.113 s against 0.106 s,
inside noise), confirming the cost is driver overhead, not the link itself. Do
not re-try this expecting a win.

The only way to remove that last cost is to stop invoking an external tool at
all, which means emitting the executable directly, PE on Windows and ELF on
Linux. That is a mini-linker (symbol resolution, relocations, imports) and is
what Jai does. The other half of going C-free is the runtime, and most of it is
already gone: the pool lives in `std/slab.frost` as ordinary Frost and
`--freestanding` already links with no libc, so what is left in C is the aborts,
the assertions and the IO.

Going C-free was not the reason to write the ELF half of the object writer. An
encoder does not remove the toolchain while a build still links through it, and
the pieces between here and that are a linker and a libc-free runtime. The
reason was narrower and better: the encoder wrote COFF, so a Windows build took
one path and every other platform took another, with different speed and
different failure modes, and only one of them was under anyone's fingers. The
ELF half makes the fast path the only path.

It is not COFF with a different header. Two differences reach back into the
encoding rather than staying in the writer. A reference to a name the file both
defines and offers is left for the linker there, because another object may take
that name over at load time, where COFF has no such rule and `as` settles it in
the assembler. And a fixup carries its own addend rather than reading one out of
the bytes it fills in, so those are left empty. `assemble.frost` is told which
format it is encoding for and both follow from the answer.

Checked the same way: clang assembles the same text and the two objects are
compared byte for byte, over the compiler's own 660 KB of code and 8,252 fixups.
`FROST_OBJECT` names the format from either host, the way `FROST_ABI` names the
calling convention, which is what lets the ELF half be checked from Windows.

## Scaling past one file

The worry with whole-program monomorphization is that it is a compile-time bomb:
generics specialize per type, specializations are a cross product, and there is
no incremental or separate compilation to bound the work. Measured rather than
argued, with `just bench-scaling`, which spans 917 to 58,107 generated lines and
640 to 10,240 specializations: four times the input costs roughly four times the
time on both the front-end and the full-native curves, so the pipeline is close
to linear with a mild superlinear term that grows with function count. Read
ratios rather than absolutes there, since about 15 ms of every figure is process
startup.

So the front end is what the curve is made of. Parse, parameter modes, regions,
ownership, IR lowering, type checking, monomorphization to fixpoint and C
emission together stay near-linear because every one of them is a local pass: no
traits to solve, no lifetimes to infer, no global inference, and the
specialization worklist dedups through a hash set rather than a scan.

What these numbers do not show is the shape problem, because every program there
is a single file, so a change to one line rebuilds everything no matter how
little it reaches. `just bench-incremental` is the measurement that does show
it: on the bootstrap, 9,484 lines across 65 files with one changed, about 580 ms
full against about 200 ms with `--incremental`. See
[separate-compilation.md](separate-compilation.md).

The Frost compiler has its own answer to the same shape problem, and a smaller
one, because it emits assembly rather than an IR. `--incremental` emits one
assembly unit per module and encodes each to its own object, and a module whose
emitted assembly is byte for byte last build's is not encoded again. The cache
key is that assembly: the compiler has just produced what the module compiles
to, so nothing else has to be hashed or walked. Where the encoder runs, the unit
stays in memory and no assembly file is written at all; where the toolchain does
the encoding it is written to `<build>/m<n>.s` for that program to read.

On its own source that is about 366 ms for a first incremental build and about
271 ms once the objects are there, and the compiler that comes out is byte for
byte the one the whole-program build produces.

What keeps the backend off the curve, on the bootstrap:

1. Functions compile in parallel. The type system is local and
   signature-based, so once signatures are collected functions are independent,
   which is a large part of why the language was designed the way it is. Code
   generation runs on every core and is 64 ms of a 349 ms build at 58k lines.
2. Modules compile separately. Each module is its own object on the link path,
   monomorphization is seeded per module, `--incremental` skips the modules an
   edit cannot reach, and a skipped module contributes signatures rather than
   bodies, so the front end never walks code it will not emit. See
   [separate-compilation.md](separate-compilation.md).
3. Specializations carry across builds, since a module's object holds the ones
   that module asked for and reusing the object reuses them.

The shape is measured rather than assumed, and the measurement is a command
rather than a memory. A benchmark is easy to get wrong in ways that look like a
compiler result: generated programs that name a function `f32` time a parse
error, and programs whose `main` holds thousands of call sites make parallel
code generation look like it does nothing, because one function is one thread
however many cores there are. Re-run the benchmark before trusting any of it,
and look at the shape of what it generates too.

Two smaller things, one taken and one not. Parsing a generic template once per
instantiation was worth taking: three passes ask for the same instance, once to
record its concrete return type and once each for its prototype and its body,
all three producing the same AST from the same template and the same argument.
`parse_generic_instance` remembers `(template, argument)` and hands back the
node and the return type it worked out. Both fixpoints stayed byte-identical,
which is what says the memo is a memo and not a change of meaning. Parsing each
template *once* and substituting types into the AST per instantiation is a
further step, and it needs a substitution pass this compiler does not have,
where binding happens during the parse.

Parallel emission is the one not taken. Emitting could run on every core, since
the type system is local and signature-based, and now that a unit is buffered in
memory rather than written straight out, the thing that stopped it is gone. What
has replaced that reason is that emitting is most of what a build now spends,
so the honest next step is to make it cost less rather than to spread it: it
formats text that is immediately read back, and handing the encoder records
instead removes the work rather than dividing it.
