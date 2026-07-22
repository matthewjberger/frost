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
hash of its source and the tool that built it, and linked thereafter. A small
program builds in 0.307 s cold and 0.104 s warm. What remains external is the
linker, which the build invokes to turn the object into an executable.

To remove that too, in increasing order of effort: port the runtime itself to
Frost (the pool memory model spike showed this works, and `--freestanding`
already links with no libc), then emit the executable directly, PE on Windows and
ELF on Linux, instead of calling a linker. That is the last external dependency
and it is what Jai does.

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

1. **Self type-checking.** Types, call arguments and arity, returns, field
   access. minifrost already has a type model (`TY_` codes, `STRUCT_BASE`,
   `POINTER_BASE`) and a `type_of`, so this extends existing machinery rather
   than starting from zero. Required before (3).
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
