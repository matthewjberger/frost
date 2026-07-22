# Self-hosting and compile speed

Frost has two compilers. `src/*.rs` is the reference compiler, written in Rust,
implementing the full language. `bootstrap/minifrost.frost` is a compiler written
in Frost that compiles a subset of Frost, and it self-hosts: it compiles its own
2579 lines to a byte-identical fixpoint across three stages, checked by
`bootstrap_minifrost_self_hosts`.

Finishing the self-hosted compiler means growing the Frost-written one until it
implements the whole language and can compile a full Frost-written compiler.
Because the spec is deliberately small and closed, this is a finite checklist.

## Compile speed

Measured on minifrost compiling its own 2579 lines (2324 lines of C emitted):

| stage | time |
| --- | --- |
| minifrost: parse, monomorphize, emit C | 0.047 s |
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

| stage | small program | minifrost, 2579 lines |
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
   instantiation. minifrost re-lexes and re-parses the template for every
   instance, which is wasted work that grows with instantiation count.
3. Parallelize per-function type checking and codegen. The type system is local
   and signature-based, so functions are independent once signatures are
   collected.
4. Keep the arena allocation and the single in-memory pass per file. minifrost
   already does this, which is why it reaches 47 ms.
5. Keep monomorphization cached and bounded. minifrost already dedups
   instantiations, and specialization-only comptime means code generation cannot
   run away.

## What is left in the Frost-written compiler

minifrost today is a translator, not a checker: it emits C and lets the C
compiler catch type errors. In dependency order:

1. **Self type-checking.** Required before (3), because once minifrost stops
   emitting C there is no C compiler behind it to catch anything.

   Done, and free: minifrost still self-compiles its own 2600 lines in 23 ms with
   all of these running.

   - Undefined calls and argument count. A name becomes defined when its `fn` or
     `extern fn` is parsed, so a name only ever called does not exist. Every node
     lives in one arena, so a linear scan reaches every call.
   - Undefined variables. `lookup_local` answers with a type and defaults an
     unknown name to `TY_I64`, which cannot tell "an i64" from "not declared",
     so `local_exists` checks the scope instead.
   - Unknown struct fields, naming the struct and the field.
   - Return types, against the function's declared return type.
   - Argument types, against the parameter list.

   Compatibility is deliberately lenient where the emitted C is lenient. Scalars
   and pointers convert freely and a type parameter matches anything, because it
   is a placeholder. A struct only matches the same struct, which is where real
   mistakes show up. Two false positives had to be designed around, since
   minifrost.frost must keep passing its own checks: a generic template's
   parameter types are placeholders bound per instantiation and carry nothing to
   check against, and an auto-borrowed argument is a value whose address is taken
   at the call, so it answers to the pointee rather than the pointer.

   Left: statement-level checks (assignment types), and the scope tracking is
   emit-time rather than a separate semantic pass.
2. **Ownership and linearity.** Linear resources consumed exactly once, the move
   and borrow rules. The design's safety core, entirely absent from minifrost.
3. **Native backend.** The speed payoff above, unblocked by (1).
4. **Failure sets, allocation sources, regions.** `-> T ! E` with `?`,
   `uses`/`with`, and the arena escape check. These reuse minifrost's existing
   enum and match machinery and mirror the desugars in `src/failure_sets.rs`,
   `src/allocation_sources.rs`, and `src/regions.rs`.
5. **Imports and modules.** minifrost is single-file today; a multi-file
   compiler needs this to compile itself.

Param modes are already done: minifrost lowers `mut`/`move`/read to pointers and
inserts the borrow at call sites.

## How to do each port

The pattern that worked for param modes:

1. Add the capability to minifrost while leaving minifrost.frost unchanged, so
   the fixpoint stays byte-identical and the change is inert.
2. Migrate minifrost.frost to use it.
3. Verify against the reference compiler with the differential tests, and keep
   `bootstrap_minifrost_self_hosts` green.
4. Commit each stage separately.

The reference compiler stays the oracle throughout.
