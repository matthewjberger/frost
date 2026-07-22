# Self-hosting and compile speed

Frost has two compilers. `src/*.rs` is the reference compiler, written in Rust,
implementing the full language. `bootstrap/frost.frost` is a compiler written
in Frost that compiles a subset of Frost, and it self-hosts: it compiles its own
source to a byte-identical fixpoint across three stages, checked by
`self_hosting_is_a_fixpoint`. It checks the programs it compiles rather than
leaving that to whatever compiles its output.

Finishing the self-hosted compiler means growing the Frost-written one until it
implements the whole language and can compile a full Frost-written compiler.
Because the spec is deliberately small and closed, this is a finite checklist.

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

Second-order levers, worth doing but small next to the above:

2. Parse each generic template to AST once and substitute types per
   instantiation. the self-hosted compiler re-lexes and re-parses the template for every
   instance, which is wasted work that grows with instantiation count.
3. Parallelize per-function type checking and codegen. The type system is local
   and signature-based, so functions are independent once signatures are
   collected.
4. Keep the arena allocation and the single in-memory pass per file. the self-hosted compiler
   already does this, which is why it reaches 47 ms.
5. Keep monomorphization cached and bounded. the self-hosted compiler already dedups
   instantiations, and specialization-only comptime means code generation cannot
   run away.

## What is left in the Frost-written compiler

the self-hosted compiler checks its own programs now rather than deferring to whatever compiles
its output. In dependency order, what is done and what remains:

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

   Left: the interprocedural half, where a `uses` function stores an arena
   pointer into one of its own parameters.

6. **Failure sets.** `-> T ! E` with `?`, mirroring `src/failure_sets.rs`. This
   one needs enums with payloads first, which the bootstrap subset skipped,
   since the desugar synthesizes a Result enum and matches on it.
7. **Imports and modules.** the self-hosted compiler is single-file today; a multi-file
   compiler needs this to compile itself.

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
