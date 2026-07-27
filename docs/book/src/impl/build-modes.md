# Native, freestanding, self-hosted

Three separate axes that get conflated because all three sound like "compiles to
machine code". They are independent questions, and a build can sit anywhere on
each of them.

| axis | the question it answers |
| --- | --- |
| **native** | which backend produced the machine code |
| **freestanding** | what the produced executable depends on at run time |
| **self-hosted** | what language the compiler itself is written in |

## Native, which backend produced the code

Frost has three execution paths off one typed IR, and the flag picks which.

- `--native` / `--link` lower the IR through Cranelift to a relocatable object,
  which the system C toolchain links. This is the default. Bare
  `frost program.frost` compiles, links, and runs.
- `--emit-c` lowers the same IR to portable C, which the system C compiler
  builds. That buys portability to anywhere with a C compiler, and it is the
  second half of the differential oracle.
- `--run-ir` interprets the IR directly, as a reference oracle for scalar
  programs.

"Native" means the Cranelift path, no C compiler in the middle for *your* code.
The differential test runs programs through all three and asserts they agree,
which is what catches a miscompilation that a single backend would hide.

This is orthogonal to freestanding. A `--native` build still links the C
runtime and libc by default. Choosing Cranelift says nothing about what the
executable needs once it is running.

## Freestanding, what the executable needs at run time

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
now works with no libc at all. See
[pools-and-columns.md](../design/pools-and-columns.md).

## Self-hosted, what the compiler is written in

`selfhosted/frost.frost` is the Frost compiler, written in Frost, and it is the
one people will use. That is the destination. Frost is meant to be written in
Frost, and the compiler someone downloads is meant to be the Frost one.

`src/*.rs` is the bootstrap, and its job is to make writing that compiler
possible. It compiles stage 0, so every feature `frost.frost` uses has to exist
in Rust before a line of it can be written in Frost, and it is the oracle the
differential tests compare against, so a miscompilation in the Frost compiler
has something to be caught by. Both roles are scaffolding. The bootstrap being
ahead on a feature is a stage of the work rather than a division of labour.

The claim the Frost compiler discharges is a three-stage fixpoint. It compiles
its own source, a compiler built from that output compiles the source again, and
the two outputs are byte-identical. That holds through both of its backends, the
C one and its own assembly emitter (`FROST_BACKEND=asm`), and both fixpoints are
checked on every build by `self_hosting_is_a_fixpoint` and
`native_self_hosting_is_a_fixpoint`. The fixpoint is how the compiler is
checked, not what it is for.

Both compilers are held to the same two promises. They accept the same
language, goal 8 in [philosophy.md](../design/philosophy.md), and they are under
the same compilation-speed promise, goal 9, which matters more for the Frost one
because it is what a user's edit-compile loop runs. The self-hosted compiler
carries the speed work: it emits one unit per module, keys a build cache on the
bytes it just emitted, and rebuilds only what an edit reaches under
`--incremental`. What it does not do yet is generate code on every core.
[The self-hosted compiler](self-hosted.md) has what it implements, and
[roadmap.md](../roadmap.md) has the measurements.

## How the axes interact

The axes compose freely. The self-hosted compiler emits assembly, so it is
native without being freestanding. A `--emit-c --link` build is neither native
nor freestanding. `--native --freestanding` is both. Nothing about being
self-hosted makes a program freestanding, and nothing about being freestanding
requires the native backend.
