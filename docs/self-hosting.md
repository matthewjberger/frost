# Self-hosting and compile speed

Self-hosted, native and freestanding are three different axes, and this document
is about the first. [build-modes.md](build-modes.md) separates them.

Frost has two compilers. `src/*.rs` is the reference compiler, written in Rust,
implementing the full language. `selfhosted/frost.frost` is a compiler written in
Frost, and it self-hosts twice over: through its C backend and through its own
x64 backend, each compiling its own source to a byte-identical fixpoint across
three stages, checked by `self_hosting_is_a_fixpoint` and
`native_self_hosting_is_a_fixpoint`. It checks the programs it compiles rather
than leaving that to whatever compiles its output.

The checklist below is finished. Every item is done: self type-checking,
ownership and linearity, the native backend, allocation sources, regions,
failure sets, imports and enums. What each one cost and what it turned up is
recorded under it, since the same shapes come up again.

## Compile speed

Measured on the self-hosted compiler compiling its own 2579 lines (2324 lines of C emitted):

| stage | time |
| --- | --- |
| frost: parse, monomorphize, emit C | 0.047 s |
| gcc compiling the emitted C | 0.249 s |
| gcc -O2 compiling the emitted C | 0.838 s |

The Frost-written compiler's own work is about 55,000 lines per second, already
in the range Jai and Odin operate in. It is not the bottleneck. The external C
compiler costs 5x to 18x more than the entire Frost compiler, and it scales
worse: extrapolated to 100k lines the front end is a couple of seconds while the
C compiler is tens of seconds.

The language is built to compile fast. There are no traits or typeclasses, so no
constraint solving. No lifetimes, so the region check is a cheap flow pass. No
global type inference, only local. No macros and no declaration-generating
comptime, only specialization. No textual includes, only modules parsed once.
Those are the features that make other compilers slow, and the spec omits them
on purpose.

So there is one lever that matters:

1. **Emit native code directly instead of C.** This removes the C compiler from
   the loop entirely, which is exactly why Jai and Odin are fast. The reference
   compiler already proves the model with its Cranelift backend, emitting a
   native object straight from the typed IR; the Frost-written compiler only has
   the C path. Note the dependency: a native backend requires the compiler to
   type-check for itself first, because there is no longer a C compiler behind it
   to catch mistakes.

On the native path the C compiler is already out of the per-build loop. The
runtime is compiled once into an object cached in the temp directory, keyed by a
hash of its source and the tool that built it, and linked thereafter.

Where a build's time goes now:

| stage | small program | the self-hosted compiler, 2579 lines |
| --- | --- | --- |
| frost compile only (`--native`) | 0.033 s | 0.045 s |
| full build including link | 0.099 s | 0.126 s |

So the compiler's own work is a third of the build and the rest is the linker
invocation. That remainder is mostly fixed process and driver overhead rather
than linking work, which means it barely grows with program size and is already
a small fraction of a large build.

Measured and rejected: passing `-fuse-ld=lld` to the driver. lld is present on
this machine and made no difference (0.113 s against 0.106 s, inside noise),
confirming the cost is driver overhead, not the link itself. Do not re-try this
expecting a win.

The only way to remove that last cost is to stop invoking an external tool at
all, which means emitting the executable directly, PE on Windows and ELF on
Linux. That is a mini-linker (symbol resolution, relocations, imports) and is
what Jai does. Worth doing for self-containment more than for speed. Porting the
runtime itself to Frost is the other half of going C-free; the pool model already
exists in Frost in `examples/native/native_pool.frost`, and `--freestanding`
already links with no libc, but it needs a prelude mechanism so a Frost-written
runtime is compiled into every program.

## Scaling past one file

The worry with whole-program monomorphization is that it is a compile-time bomb:
generics specialize per type, specializations are a cross product, and there is
no incremental or separate compilation to bound the work. Measured rather than
argued, with `just bench-scaling`:

| program | front end (`--emit-c`) | with Cranelift (`--native`) |
| --- | --- | --- |
| 917 lines | 24 ms | 25 ms |
| 3,642 lines | 26 ms | 29 ms |
| 14,532 lines | 84 ms | 94 ms |
| 58,107 lines | 338 ms | 353 ms |
| 640 specializations | 15 ms | 20 ms |
| 2,560 specializations | 38 ms | 53 ms |
| 10,240 specializations | 138 ms | 213 ms |

Read the ratios rather than the absolutes, since about 15 ms of each figure is
process startup. Four times the input costs roughly four times the time on both
curves, so the pipeline is close to linear with a mild superlinear term that
grows with function count.

The gap between the two columns used to be the story and is now nearly nothing:
58k lines is 338 ms through the front end and 353 ms for the whole native build,
about 165,000 lines per second. Code generation is 64 ms of that, because it
runs on every core (see item 1 in [roadmap.md](roadmap.md)). Before that landed
the same program took 1.11 s and the backend was two thirds of it.

So the front end is now what the curve is made of. Parse, parameter modes,
regions, ownership, IR lowering, type checking, monomorphization to fixpoint and
C emission together stay near-linear because every one of them is a local pass:
no traits to solve, no lifetimes to infer, no global inference, and the
specialization worklist dedups through a hash set rather than a scan.

What these numbers do **not** show is the shape problem, because every program
here is a single file, so a change to one line rebuilds everything no matter how
little it reaches. `just bench-incremental` is the measurement that does show it:
9,484 lines across 65 files, one changed, about 580 ms full against about 200 ms
with `--incremental`. See [separate-compilation.md](separate-compilation.md).

The backend used to be where the superlinear term lived, and is not any more:

1. **Compile functions in parallel.** *Done.* The type system is local and
   signature-based, so once signatures are collected functions are independent,
   which is a large part of why the language was designed the way it is. Code
   generation now runs on every core and is 64 ms of a 353 ms build at 58k
   lines.
2. **Separate compilation per module.** *Done.* Each module is its own object on
   the link path, monomorphization is seeded per module, `--incremental` skips
   the modules an edit cannot reach, and a skipped module contributes signatures
   rather than bodies, so the front end no longer walks code it will not emit.
   See [separate-compilation.md](separate-compilation.md).
3. **Cache specializations across builds.** Now subsumed: a module's object
   holds the specializations that module asked for, and reusing the object
   reuses them.

## Does the self-hosted compiler grow separate compilation too?

**No, and this is the decision rather than an open question.** The two compilers
are under different promises. The reference compiler is under a speed promise
that has to hold as programs grow, which is what goal 8 in
[philosophy.md](philosophy.md) commits to and what separate compilation is the
answer to. The self-hosted compiler is under a self-hosting promise: it exists
to show the language can express a real compiler, and it discharges that by
reproducing itself byte for byte.

Concretely, `selfhosted/frost.frost` is one file of about 5,400 lines with no
`import` in it, and it compiles itself in about 35 ms. Separate compilation
bounds work per module, and there is one module. Building interfaces, a build
cache and per-module objects into it would be machinery with nothing to bound,
and it would grow the surface that has to keep reproducing itself exactly.

What would reopen it: `selfhosted/frost.frost` gaining imports, or its self-build
time getting far enough into the hundreds of milliseconds that the edit-compile
loop on the compiler itself starts to bite. Neither is true, and the second is
what to measure rather than guess at.

What matters is that the shape is measured rather than assumed, and that the
measurement is a command rather than a memory. Two versions of this table have
now been wrong for benchmark reasons rather than compiler reasons: the first
because the generated programs named a function `f32`, so the timings were of a
parse error, and the second because every program had a `main` with thousands of
call sites, which is one function no amount of threading can split and which
made parallel code generation look like it did not work. Re-run the benchmark
before trusting any of it, and look at the shape of what it generates too.

Second-order levers, worth doing but small next to the above:

2. ~~Parse each generic template once per instantiation.~~ *Done.* Every
   instantiation used to be re-parsed three times over, once to record its
   concrete return type and once each for its prototype and its body, all three
   producing the same AST from the same template and the same argument.
   `parse_generic_instance` now remembers `(template, argument)` and hands back
   the node and the return type it worked out. Compiling the self-hosted
   compiler with itself went from a median of 87 ms to 35 ms, and both
   fixpoints stayed byte-identical, which is what says the memo is a memo and
   not a change of meaning.

   The lever as originally written was to parse each template *once* and
   substitute types into the AST per instantiation. That is a further step and
   it needs a substitution pass this compiler does not have, where binding
   happens during the parse. One parse per instantiation removes the repetition
   that was actually there.
3. Parallelize per-function type checking and codegen. The type system is local
   and signature-based, so functions are independent once signatures are
   collected.
4. Keep the arena allocation and the single in-memory pass per file. The self-hosted compiler
   already does this, which is why it compiles itself in about 35 ms.
5. Keep monomorphization cached and bounded. The self-hosted compiler already dedups
   instantiations, and specialization-only comptime means code generation cannot
   run away.

## The checklist

The self-hosted compiler checks its own programs now rather than deferring to whatever compiles
its output. In dependency order:

1. **Self type-checking.** Required before (3), because once the self-hosted compiler stops
   emitting C there is no C compiler behind it to catch anything.

   Done, and free. Every check was measured by running the build before and after
   it back to back: the self-compile time is identical either way. The absolute
   number moves with machine load (23 ms on a quiet machine, about 69 ms on a
   busy one), so compare builds against each other rather than against a number
   written down here.

   - Undefined calls and argument count. A name becomes defined when its `fn` or
     `extern fn` is parsed, so a name only ever called does not exist. Every node
     lives in one arena, so a linear scan reaches every call.
   - Undefined variables. `lookup_local` answers with a type and defaults an
     unknown name to `TY_I64`, which cannot tell "an i64" from "not declared",
     so `local_exists` checks the scope instead.
   - Unknown struct fields, naming the struct and the field.
   - Return types, against the function's declared return type.
   - Argument types, against the parameter list.
   - Assignment types, against the place.

   Compatibility is deliberately lenient where the emitted C is lenient. Scalars
   and pointers convert freely and a type parameter matches anything, because it
   is a placeholder. A struct only matches the same struct, which is where real
   mistakes show up. Two false positives had to be designed around, since
   frost.frost must keep passing its own checks: a generic template's
   parameter types are placeholders bound per instantiation and carry nothing to
   check against, and an auto-borrowed argument is a value whose address is taken
   at the call, so it answers to the pointee rather than the pointer.

   Left: the scope tracking is emit-time rather than a separate semantic pass,
   which is enough for these checks but would need reworking for flow-sensitive
   ones.

2. **Ownership and linearity.** Done, and also free.

   - Use after move. A struct handed to a parameter that does not borrow is
     moved out of the caller, so reading it afterwards reads a value that was
     given away.
   - Linear types. `linear struct` marks a resource; `is_linear` is recorded on
     the definition, and at the end of each body every linear value must have
     been handed on, by being returned or passed to a parameter that takes
     ownership. Together with the move check this is linearity proper, consumed
     exactly once: never consumed is a leak, consumed twice is a use after move.

   Note the shape of a real consumer. A read parameter of struct type borrows,
   so it does not consume; consuming takes `move`, as in
   `close :: extern fn(move f: File)`. A function that takes a linear value by
   `move` and only reads a field out of it is correctly rejected, because the
   resource dies there.

3. **Native backend.** Done, and self-hosting with no C compiler in the loop.

   The compiler emits assembly for its own 3500 lines, that assembly assembles
   into a compiler, and that compiler emits the same 29,545 lines of assembly
   byte for byte. `native_self_hosting_is_a_fixpoint` checks it. Compiling
   itself this way takes 0.08 s.

   Two bugs had to be found to get there, both invisible to the C backend
   because C needs neither struct offsets nor a scope discipline of its own:

   - `type_size` counted a struct's fields instead of measuring them, so every
     type holding a struct field was sized wrong. `Parser` carries ten arenas
     and an arena is three words, not one.
   - `lookup_local` and `local_slot` answered with the first binding of a name
     rather than the most recent. `emit_program` binds `node` to an i64 in one
     loop and to a `^Node` in the next, so `node^.next` typed as a scalar and
     read offset zero, and the walk over the top-level list never reached its
     end.
   - The prologue read the hidden struct-result pointer out of `%rcx`, which is
     where Windows puts the first argument but not System V. That self-hosted on
     Windows and segfaulted on Linux, and only in the second stage, since the
     first-stage compiler is built by the reference compiler and never runs this
     assembly. When a stage fails, check which one before assuming the emitter
     crashed.
4. **Allocation sources.** Done. `uses A` on a function and `with a { }` around
   a call, mirroring `src/allocation_sources.rs`.

   The capability is an implicit trailing parameter that borrows its source, so
   after parsing the function is ordinary and only the call sites know to supply
   one. A `uses` function forwards what it was handed, a `with` block supplies
   the arena it names by address, and a call with neither in reach is rejected.

   This needed a name the compiler makes up rather than reads, since the binding
   is the capability type's name lowercased and no source spells it. A name here
   is an offset and a length into the source everywhere, so rather than add a
   second kind of name, the source is copied into a buffer with room past its
   terminator and synthesized names are written there. The lexer stops at the
   terminator, so nothing in that room is ever read as code.

   Fixing this also removed a limitation worth naming: a plain function used to
   lower to `int64_t` whatever it returned, so it could not return a pointer.
   It now emits its declared return type.

5. **Regions.** Done, mirroring `src/regions.rs`. A `with` block is a region and
   a raw pointer derived from its arena may not be stored outside the block or
   returned. A binding declared inside may hold one, since it dies with the
   block, and reading through it is the point.

   No lifetimes and no region types on pointers, just a walk over the block
   tracking which names hold arena pointers. A dereference deliberately does not
   propagate: it reads the value there, and reading out of the region is what
   the region is for.

   It is interprocedural: the body of every `uses` function is a region too,
   whose arena belongs to whoever supplied it. Such a function may hand a
   pointer back to its caller, where the caller's region checks it, but may not
   store one into a parameter, which outlives the call.

6. **Failure sets.** Done. `-> T ! E` says a function answers with a T or fails
   with an E, and `e?` hands the failure on.

   Both lower to what the compiler already had. A failure set is a struct
   carrying which of the two it holds beside both payloads, rather than the
   reference compiler's Result enum, so no enum-with-payload machinery was
   needed: the two are never both live, but reading either one is then a plain
   field of a plain struct, which every backend already does. A `return` that
   builds the error type is the failure side and anything else is the value
   side, matching how the reference reads it.

   `e?` becomes a binding, a test and a return, queued for the block being
   parsed and emitted ahead of the statement the `?` was written in. That is
   what lets a `?` sit anywhere an expression can rather than only where a
   statement can.

   This needed names the compiler makes up, which the room past the source
   already provides: `__Result<n>` per failure set, `__try<n>` per `?`, and the
   three field names.

8. **Enums with payloads.** Done. `Kind :: enum { Player, Enemy { damage: i64 } }`,
   `Kind::Enemy { damage = 15 }`, and `case .Enemy { damage }:` in a match.

   An enum is a struct carrying a tag beside every variant's fields, each field
   named for the variant it came from, so two variants may each carry a
   `damage` and they stay apart. A variant is then a tag value and a set of
   field names rather than a type of its own, which is why construction, field
   access and matching all reduce to what the compiler already did with structs
   and integers. The variants sit side by side rather than overlapping: a union
   is what would save the space, and the C backend cannot express one as a plain
   struct.

   A variant pattern also made `match` an expression rather than a statement. It
   binds a name, each arm assigns to it, and the match is queued for the block
   the same way a `?` is. An arm ending in something with no value is left alone,
   since it was not producing one.

   Making `match` an expression turned up a bug in the road both it and `?`
   take. A statement queued while a condition was being parsed was placed
   before the loop, so a `?` in a `while` condition was asked once and answered
   the same for ever. The loop now carries what its condition needs and runs it
   every time round, with the test moved inside.

   Two more bugs came out of running everything through both backends and
   comparing, which is now a test of its own:

   - A generic function used with a type that no generic struct was written
     with was called and never emitted, so the program failed to link.
     Instantiation was driven by the struct instances, and a generic function
     had to ride on one. It is driven by the distinct types the generics are
     used with now, gathered from the instances and from the calls. That also
     closes a second hole: two generic structs written with the same type would
     have had every generic function emitted twice.

   The other thing this turned up: the native
   backend treated every scalar as a word, so `^i8` indexing strode eight bytes
   at a time and `sizeof(i8)` answered 8. A byte-wide type is one byte now,
   struct fields sit on their own alignment, and a byte is loaded and stored
   with byte instructions. The C backend was always right here, which is how
   this survived so long.
7. **Imports.** Done. `import "path"` names a file whose declarations join this
   one's.

   Every file's text reaches one buffer, so a name stays what it is everywhere
   else in this compiler, an offset and a length into a single source, and
   nothing downstream learns that more than one file was involved. A file lands
   there once however many times it is named, which is what separates this from
   a textual include and what makes a diamond one copy and a cycle terminate.
   Imports are found by token rather than by scanning text, so the word
   `import` inside a string or a comment is not one, and the placement is
   post-order: a file follows everything it depends on, because a struct's name
   is resolved where it is written rather than looked up later.

   Fixing this turned up a real gap: a local bound by `:=` to a call of a
   generic function took the template's return type, which still mentions the
   type parameter, so two arenas over different elements came out as the same
   type. The concrete return type per instantiation was already computed for the
   native backend; it now runs before either backend, so a call answers with it
   everywhere.

   `export a, b` lists what a file offers. A top-level name not listed is the
   file's own, so two modules may each keep one of the same name and neither
   sees the other's.

   Visibility is settled where a name is interned rather than by renaming what
   a module keeps private. A declaration's own offset says which file wrote it,
   so nothing has to carry an owner, and a lookup matches a name only when the
   two are in the same file or the declaring file exported it. A program built
   from one file has one module, so the first comparison answers yes and this
   costs nothing.

Param modes are already done: the self-hosted compiler lowers `mut`/`move`/read to pointers and
inserts the borrow at call sites.

## How to do each port

The pattern that worked for param modes:

1. Add the capability to the self-hosted compiler while leaving frost.frost unchanged, so
   the fixpoint stays byte-identical and the change is inert.
2. Migrate frost.frost to use it.
3. Verify against the reference compiler with the differential tests, and keep
   `self_hosting_is_a_fixpoint` green.
4. Commit each stage separately.

The reference compiler stays the oracle throughout.
