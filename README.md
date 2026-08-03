<p align="center">
  <a href="https://github.com/matthewjberger/frost"><img alt="github" src="https://img.shields.io/badge/github-matthewjberger/frost-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20"></a>
  <img alt="license" src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-fc8d62?style=for-the-badge&labelColor=555555" height="20">
</p>

# Frost

**A data-oriented systems language that is memory-safe with no garbage collector and no lifetimes, and compiles itself.**

A Frost program is plain data and free functions that transform it. There are no classes and nothing is allocated behind your back. It compiles to native code through Cranelift or to portable C, and the compiler is written in Frost.

Frost is under construction. It compiles itself and everything shown below runs,
but the surface still changes between commits and there is no tagged release, so
anything written against it today may need editing tomorrow.

## Borrows are parameter modes

A borrow is what a parameter mode means, so there is no reference type to write and no lifetime to describe.

```frost
wound :: fn(mut e: Entity, amount: i64) { e.hp = e.hp - amount }
```

`mut` borrows for the call and mutates in place. An unmarked parameter borrows to read. `move` takes ownership. There is no `&` anywhere in the language, so a borrow has nothing to be stored in and nowhere to escape to. That one rule is why Frost needs no lifetime annotations, and the rest of the design grows from it.

| in place of | Frost has |
| --- | --- |
| lifetimes on references | borrows that are parameter modes and cannot escape |
| a garbage collector | arenas and pools you can see |
| a long-lived pointer into a collection | a generational handle, a copy value that goes stale rather than dangling |
| destructors | linear resources, consumed exactly once, checked at compile time |
| exceptions and `Result` plumbing | failure sets, `-> T ! E` and `?` |
| dynamic dispatch | monomorphized generics, so the inner-loop call is direct |
| classes and methods | plain structs and free functions |

## A short tour

The tour covers borrows that are parameter modes, a resource the compiler counts,
a failure that travels in the signature, a `for` that needs no iterator, a
function that answers with two values, and literals that leave out a type the
context already carries while every field keeps its name.

<p align="center">
  <img src="docs/book/src/tour.svg" alt="A tour of Frost: parameter modes, a linear resource, a failure set, and the inference in a literal" width="700">
</p>

```bash
frost examples/tour.frost          # compile, link, and run
```

That is [`examples/tour.frost`](examples/tour.frost). A test compiles it, checks
what it prints, and checks it still matches what is written here.

For long-lived data, a value lives in a pool and is named by a `Handle`, a small copy value. Freeing a slot raises its generation, so a handle to a reused slot reads as stale rather than reading whatever took its place. The archetype entity-component system in [`std/ecs.frost`](std/ecs.frost) is built on that, and [`just app scene`](docs/book/src/std/ecs.md) runs it: entities in an ECS, two passes, depth deciding what is in front.

## Documentation

The documentation is a book, in [`docs/book`](docs/book).

```bash
cargo install mdbook
cd docs/book && just serve
```

[A tour of Frost](docs/book/src/tour.md) is the language by example and the
place to start. [Coming from Rust](docs/book/src/coming-from-rust.md) is the
same ground for someone who already thinks in ownership and borrows.
[Patterns](docs/book/src/patterns.md) is what to write instead, once the syntax
is familiar. The [language reference](docs/book/src/reference/conformance.md) is
normative, and [the roadmap](docs/book/src/roadmap.md) is what is left.

## What it does

The language has structs and tagged enums, `match` with payload and tuple patterns that has to cover every variant or say what the rest do, and generics that monomorphize over types, values, and functions, so a call in an inner loop stays direct rather than going through a pointer. A `for` walks a range or a sequence as the index-and-bound loop it stands for, a function answers with several values through a return type list rather than a tuple type, and a literal leaves out a type the context already carries, while every field keeps its name because the name is what says where the value lands. A resource that must be released is marked `linear` and the compiler counts it, consumed once on every path out or the program does not build. Long-lived data lives in pools addressed by generational handles, a region check keeps a pointer from outliving the block or the stack frame it points into, and a value constant can be a folded integer expression. There are no visibility modifiers and no methods.

It calls C without a binding layer. An `extern fn` links against a C library with the natural ABI, including one that returns a struct by value and one that takes a Frost function as a callback with a typed context.

One typed intermediate representation feeds three backends, a Cranelift native path, a portable C path, and a small interpreter. A differential test runs every program through all three and checks the answers match, so a lowering bug shows up as a disagreement rather than as a wrong binary.

The compiler is written in Frost. `selfhosted/frost.frost` reproduces itself byte for byte through its own C backend and its own x86-64 assembly backend, so a build can go from source to a running compiler with no C compiler in the loop. A full native build of 58k lines runs at about 166,000 lines per second with code generation spread across cores, and `--incremental` rebuilds only the modules an edit can reach.

The standard library is ordinary Frost. It has length-carrying strings, a growable `Vec` and a hash map, file and formatted output, a sort, the slab and structure-of-arrays `columns` containers, an archetype entity-component system, and vector, matrix, and quaternion math at both single and double precision. See [`std/`](std), [docs/book/src/std/ecs.md](docs/book/src/std/ecs.md) and [docs/book/src/std/math.md](docs/book/src/std/math.md).

The engine is ordinary Frost too. [`lib/`](lib) is a window and input layer over SDL3, a renderer over wgpu with a render graph that orders passes by the resources between them, and an `App` a program composes plugins into. A plugin is a `fn(mut App)`, systems run in stages, a pass carries its own typed state, and no example in the tree says `unsafe`.

## Getting started

A Rust toolchain and a C compiler (gcc or clang) for linking are the whole of
what has to be there first. [`just`](https://github.com/casey/just) runs
everything below and is worth having: `cargo install just`.

```bash
git clone https://github.com/matthewjberger/frost
cd frost
cargo build --release    # the bootstrap compiler, at target/release/frost
just install             # put it on PATH as `frost`, with std/ beside it
```

That is the whole setup on Windows. On Linux and macOS, two more lines get the
graphics examples going, which the [graphics demos](#the-graphics-demos) section
covers. Either way:

```bash
just app spinning        # build and run one of the examples
cargo test               # everything, including both self-hosting fixpoints
```

### Building the compiler with itself

There are two compilers and they accept the same language. `src/` is the
bootstrap, in Rust; `selfhosted/` is the same compiler written in Frost. Going
from nothing to a Frost compiler built by Frost is three steps:

```bash
cargo build --release    # 1. the bootstrap compiler, from Rust
just selfhost-build      # 2. it compiles selfhosted/frost.frost -> selfhosted/frost.exe
just install-self        # 3. put that on PATH as `frostc`
```

Step 2 is stage 0: the Rust compiler reading Frost source and emitting a Frost
compiler. From there `frostc` compiles anything the bootstrap does, including
its own source, and what it produces is byte for byte what stage 0 produced:

```bash
just selfhost-check          # compile itself with itself, twice, and diff (three-stage fixpoint)
just selfhost-native-check   # the same through its own x86-64 assembly backend, no C compiler
just selfhost-test           # run the compiler's own test blocks
```

The second one is the one worth understanding. `--emit-asm` makes the
self-hosted compiler write x86-64 assembly, which its own in-process assembler
encodes to an object file. So a build can go from Frost source to a running
compiler with no C compiler anywhere in the loop, and the binary that comes out
reproduces itself exactly.

Both fixpoints run in `cargo test`, so a change that makes the compiler stop
reproducing itself fails the suite rather than being noticed later.

```bash
frost program.frost                              # compile, link, and run
frost --link -o program program.frost            # link to a named executable
frost --native -o program.o program.frost        # object file only
frost --emit-c -o program.c program.frost        # portable C instead of Cranelift
frost --run-ir program.frost                     # interpret the typed IR
frost --test program.frost                       # run the file's `test` blocks

frost --link --incremental -o program program.frost   # rebuild only what changed
frost --link --freestanding -o program program.frost  # link no C standard library
frost --link -L vendor -o program program.frost       # add an import search path
```

An import is looked for beside the importing file, then on `-L` and `FROST_PATH`, then in the project's `frost.json`, then in the bundled [`std/`](std).

### The graphics demos

Ten programs, in the order they were built, each one the smallest step past the
last. All but the first and the last draw through wgpu:

```bash
just app window      # a window that opens, resizes, and closes
just app triangle    # the first thing drawn: one triangle, one pipeline
just app scene       # entities in an ECS, two passes, depth deciding what is in front
just app spinning    # lit surfaces: a mesh cache, a material registry, two bind groups
just app textured    # the same field with its surfaces read off an image
just app shadowed    # an ECS schedule, compute, shadows, bloom, and a second view
just app gltf_model  # a model read out of a file and spawned into the world
just app lit         # the same world with no shader in the program at all
just app swarm       # five hundred things in one batch, ranged lights, a world that changes while it runs
just app input       # what the platform layer saw, for when a key misbehaves
```

**On Windows these run straight from a clone.** The wgpu binding, the schema it
is generated from, and the two runtime libraries a Windows build loads are all
in the tree, so nothing is fetched and nothing is generated first.

**On Linux and macOS, install SDL3 and run `just deps` once.** SDL3 comes from
the package manager and wgpu-native is downloaded into the tree, which the demos
name at link time and carry as an rpath, so a built one runs without
`LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`. The binding itself is already there.

```bash
brew install sdl3            # macOS
sudo apt install libsdl3-dev # Debian and Ubuntu
sudo pacman -S sdl3          # Arch

just deps                    # wgpu-native, into lib/renderer/wgpu
just app spinning
```

`just deps` is also how either platform moves to a newer wgpu: the versions are
pinned at the top of the [justfile](justfile), and what it writes is committed
so a regenerated binding shows up as a diff.

`swarm` is the one that puts the renderer under load: five hundred and twelve
things sharing one mesh and one material, so they are one batch; four lamps with
a range, so the light grid has boxes to reject them from; and three keys that
spawn, despawn, and take a mesh out of the cache while it runs.

`lit` is the one to read first if the question is what the engine does for a
program rather than what a pass can be. It composes `render_plugin`, spawns
meshes and lights, and runs: no shader, no pipeline, no bind group, no pass.
The others that draw each write their own, because each of them is about what a
pass can be.

From `scene` on, what runs and in what order is a render graph
([`lib/renderer/graph.frost`](lib/renderer/graph.frost)). A pass
declares the targets it reads and writes; the graph works out the order from
those declarations, makes every target the window does not own, and decides each
load op. `shadowed` is what that buys: eight passes written in whatever order reads
well, ordered by the resources between them, with the three targets in the
middle sharing textures because the graph knows when each is last read.

`W`/`A`/`S`/`D` move, `Q` and `E` drop and rise, the arrow keys look, and escape
closes. Resizing works in all of them. Holding `B` in `shadowed` turns its second
view off, which is a system writing a resource and a hook turning two passes off
without the graph being scheduled again.

Set `SDL3_DIR` to use an SDL already on the machine instead of the one in the
tree.

The binding in [`lib/renderer/wgpu.frost`](lib/renderer/wgpu.frost) is generated
from `webgpu.json` by [`tools/wgpu_bindgen.frost`](tools/wgpu_bindgen.frost)
rather than written, which is why there are three thousand lines of it and no
hand-maintained header. `just bindgen` regenerates it.

### Editor support

```bash
just install-editor    # link the VS Code extension, then reload the window
```

A `.frost` file gets syntax highlighting, snippets for the declaration forms, and validation for `frost.json`. A fenced block tagged `frost` in a markdown file is highlighted the same way. `Ctrl+Shift+B` runs every compiler check over the open file and puts what it finds in the Problems panel, on the line that caused it.

The extension is in [`.vscode/frost`](.vscode) rather than on the marketplace, so it is linked rather than installed. `just editor-dir` prints where this VS Code reads its extensions, which is not `~/.vscode/extensions` when VS Code is a portable install. The grammar's keywords and builtins are held to the compiler's own lists by a test, so the two cannot drift apart.

## Status

Everything above works today and is checked by the test suite on every commit, including both self-hosting fixpoints and the three-backend differential run. The compiler has compiled itself.

The two compilers accept the same language. They are held to it by running the same programs through both and comparing what each accepts, and every form the language has is a test that both of them compile.

## Project layout

```
frost/
├── selfhosted/   # the Frost compiler, written in Frost
├── src/          # the bootstrap compiler, in Rust: builds stage 0, and is the oracle
├── std/          # the standard library, in Frost
├── runtime/      # a small C runtime (bounds check, assert, IO helpers)
├── lib/          # the engine, in Frost: platform, renderer, engine
├── tools/        # the wgpu binding generator, and the highlighter
├── examples/     # runnable programs
├── bench/        # the benchmark generator
├── .vscode/      # editor settings, and the VS Code grammar for .frost
└── docs/book/   # the documentation, as an mdBook
```

`lib/` is four layers and each reaches one way: `platform` is the window and the
input, `renderer` is wgpu and the render graph, `engine` is the ECS seam and the
`App`, and `examples/graphics` sits on top. Which directory may import which is
declared in [`frost.json`](frost.json) and enforced by both compilers, so a
layer reaching upward is a compile error rather than a review comment.

## Tests

```bash
cargo test              # everything, including both self-hosting fixpoints
just check              # cargo check and the format check
just ci-linux           # every gate the Linux runner runs, in a container
just test-interfaces    # the whole suite again, built from module interfaces
just bench-scaling      # how the pipeline scales
just bench-incremental  # what --incremental saves
```

The graphics tests skip themselves where the libraries a demo links are not
there, so `cargo test` is green on a machine that has never run `just deps`.

## Contributing

External contributions are not being accepted yet. The language surface is still settling. Guidelines will land here once it stops moving.

## License

Dual-licensed under either of:

- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in `frost` by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
