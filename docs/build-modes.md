# Native, freestanding, self-hosted

Three separate axes that get conflated because all three sound like "compiles to
machine code". They are independent questions, and a build can sit anywhere on
each of them.

| axis | the question it answers |
| --- | --- |
| **native** | which backend produced the machine code |
| **freestanding** | what the produced executable depends on at run time |
| **self-hosted** | what language the compiler itself is written in |

## Native: which backend produced the code

Frost has three execution paths off one typed IR, and the flag picks which.

- `--native` / `--link` lower the IR through Cranelift to a relocatable object,
  which the system C toolchain links. This is the default: bare
  `frost program.frost` compiles, links, and runs.
- `--emit-c` lowers the same IR to portable C, which the system C compiler
  builds. That buys portability to anywhere with a C compiler, and it is the
  second half of the differential oracle.
- `--run-ir` interprets the IR directly, as a reference oracle for scalar
  programs.

"Native" means the Cranelift path: no C compiler in the middle for *your* code.
The differential test runs programs through all three and asserts they agree,
which is what catches a miscompilation that a single backend would hide.

**This is orthogonal to freestanding.** A `--native` build still links the C
runtime and libc by default. Choosing Cranelift says nothing about what the
executable needs once it is running.

## Freestanding: what the executable needs at run time

`frost --link --freestanding` links no C standard library at all: a minimal
runtime, a custom entry point, and the single OS call for process exit.

The entry point is per target, the same shape Rust's targets use. Windows exits
through kernel32 with `mainCRTStartup` as the entry. Linux uses `_start` and a
raw syscall. macOS uses `_start` too but routes through libSystem, because macOS
always does.

`examples/freestanding.frost` computes a result with a static arena and returns
it as the exit code, and the executable imports only the platform's exit
function.

This is the axis that made moving the pool out of C matter. A fixed-capacity
slab lives inside a struct rather than behind `malloc`, so generational storage
now works with no libc at all. See [native-pools.md](native-pools.md).

## Self-hosted: what the compiler is written in

`selfhosted/` is the Frost compiler, written in Frost. `src/` is the bootstrap
that compiles its stage 0.

The claim it discharges is a **three-stage fixpoint**: it compiles its own
source, a compiler built from that output compiles the source again, and the two
outputs are byte-identical. That holds through both of its backends, the C one
and its own assembly emitter (`FROST_BACKEND=asm`), and both fixpoints are
checked on every build by `self_hosting_is_a_fixpoint` and
`native_self_hosting_is_a_fixpoint`.

It implements ownership and linearity (use after move, and linear values
consumed exactly once), monomorphized generics, structs, enums with `match`, and
`extern` FFI. What it does not implement yet is listed in
[../selfhosted/README.md](../selfhosted/README.md). See
[self-hosting.md](self-hosting.md) for the measurements.

## How the axes interact

`src/` (Rust) is the **bootstrap** and `selfhosted/frost.frost` is the compiler
people will use. The bootstrap compiles stage 0 and serves as the differential
oracle, which is why every feature lands there first. That ordering is the only
reason it is ahead.

Both are under the same promises: the full language, and goal 8's speed. Where
the self-hosted compiler is behind, that is a port waiting rather than a
divergence, and that includes parallel code generation, separate compilation and
`--incremental`.

The axes compose freely. The self-hosted compiler emits assembly, so it is
native without being freestanding. A `--emit-c --link` build is neither native
nor freestanding. `--native --freestanding` is both. Nothing about being
self-hosted makes a program freestanding, and nothing about being freestanding
requires the native backend.
