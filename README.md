<p align="center">
  <a href="https://github.com/matthewjberger/frost"><img alt="github" src="https://img.shields.io/badge/github-matthewjberger/frost-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20"></a>
  <img alt="license" src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-fc8d62?style=for-the-badge&labelColor=555555" height="20">
</p>

# Frost

**A data-oriented systems language that is memory-safe with no garbage collector and no lifetimes, and compiles itself.**

A Frost program is plain data and free functions that transform it. Every allocation is one the program asked for. It compiles to native code through Cranelift or to portable C, and the compiler is written in Frost.

Frost is under construction. It compiles itself and everything shown below runs,
but the surface still changes between commits and there is no tagged release, so
anything written against it today may need editing tomorrow.

## The language in one program

A dungeon crawl: a turn loop, a torch that has to be put out, and most of the
language doing one job.

```frost
// A dungeon crawl. Every top-level declaration is `name :: value`, and a file's
// exports are one `export` line.
import "io.frost"

ROOMS :: 3

// A tagged enum. A variant carries a payload where it has one to carry.
Move :: enum { Look, Go { to: i64 }, Swing, Rest }

Monster :: struct { name: str, hp: i64, bite: i64 }

Hero :: struct { hp: i64, blade: i64, room: i64 }

// A `linear` value is consumed exactly once on every path out, counted when the
// program is built, so a torch nobody puts out is a compile error.
Torch :: linear struct { turns: i64 }

light :: fn(turns: i64) -> Torch { { turns = turns } }

// `move` is what consumes it, and the name cannot be read again after the call.
douse :: fn(move t: Torch) -> i64 { t.turns }

// How walking fails. The `-> T ! E` below is where that is declared.
NoExit :: struct { from: i64 }

exit_to :: fn(from: i64, to: i64) -> i64 ! NoExit {
    if (to < 0 || to >= ROOMS) {
        return { from = from }
    }
    to
}

// `mut` borrows for the call and writes the caller's value. `?` hands a failure
// to the caller rather than reading it here.
walk :: fn(mut hero: Hero, to: i64) -> i64 ! NoExit {
    room := exit_to(hero.room, to) ?
    hero.room = room
    room
}

// A function answers with several values through a return type list. There is
// no tuple type behind it: the names say which value is which.
trade :: fn(mut hero: Hero, mut foe: Monster) -> (dealt: i64, taken: i64) {
    foe.hp = foe.hp - hero.blade
    if (foe.hp < 0) { foe.hp = 0 }
    mut taken: i64 = 0
    if (foe.hp > 0) {
        taken = foe.bite
        hero.hp = hero.hp - taken
    }
    return { dealt = hero.blade, taken = taken }
}

// An unmarked parameter borrows to read.
report :: fn(foe: Monster, dealt: i64, taken: i64) {
    print("  you hit {} for {}", foe.name, dealt)
    if (foe.hp == 0) {
        print(" and it falls\n")
        return
    }
    print(" and take {} back\n", taken)
}

main :: fn() -> i64 {
    torch := light(6)

    mut hero: Hero = { hp = 30, blade = 6, room = 0 }
    mut here: [ROOMS]Monster = [
        { name = "the dark", hp = 0, bite = 0 },
        { name = "a rat", hp = 8, bite = 3 },
        { name = "a wight", hp = 20, bite = 7 },
    ]

    // A literal leaves out a type the context already carries, and a variant
    // leaves out its enum.
    plan: [8]Move = [ .Look, .Go { to = 1 }, .Swing, .Swing,
        .Go { to = 9 }, .Go { to = 2 }, .Swing, .Rest ]

    // `for` walks an array or a slice with no iterator to implement, and
    // `match` covers every variant and binds the payload of the one it took.
    for step in plan {
        match step {
            case .Look: {
                print("you are in room {} with {}\n", hero.room, here[hero.room].name)
            }
            case .Go { to }: {
                match walk(hero, to) {
                    case .Ok { value }: {
                        print("you walk into room {}\n", value)
                    }
                    case .Err { error }: {
                        print("no way out of room {}\n", error.from)
                    }
                }
            }
            case .Swing: {
                dealt, taken := trade(hero, here[hero.room])
                report(here[hero.room], dealt, taken)
            }
            case .Rest: {
                hero.hp = hero.hp + 4
                print("you rest, and are back to {}\n", hero.hp)
            }
        }
    }

    print("you leave with {} hit points and a torch good for {} more turns\n",
        hero.hp, douse(torch))
    0
}
```

It prints a transcript:

```
you are in room 0 with the dark
you walk into room 1
  you hit a rat for 6 and take 3 back
  you hit a rat for 6 and it falls
no way out of room 1
you walk into room 2
  you hit a wight for 6 and take 7 back
you rest, and are back to 24
you leave with 24 hit points and a torch good for 6 more turns
```

Past this screen: a monster that outlives the turn it was made in lives in a
pool and travels as a generational `Handle<Monster>`, a copy value that goes
stale rather than dangling; generics monomorphize over types, values, and
functions, so `sort($i64, $ascending, xs)` calls directly; a `with` block is an
arena whose pointers cannot outlive it; `columns<T, N>` stores a struct as one
array per field; `distinct i64` is the same bits under a type of its own;
`[4]f32` takes the arithmetic operators once per lane; a call written where a
constant is read runs before the program does; `defer` runs where the function
leaves; and `test` blocks run under `--test`.
[`examples/tour.frost`](examples/tour.frost) is the official tour, a runnable
program the suite compiles and checks:

```bash
frost examples/tour.frost          # compile, link, and run
```

The archetype entity-component system in [`std/ecs.frost`](std/ecs.frost) is
built on those pools, and [`just app scene`](docs/book/src/std/ecs.md) runs it:
entities in an ECS, two passes, depth deciding what is in front.

## Borrows are parameter modes

A parameter mode is the whole of what a borrow is. `mut` borrows for the call
and writes the caller's value, an unmarked parameter borrows to read, and
`move` takes ownership. A borrow lasts for the call and has no spelling of its
own, so there is nothing to store one in and nothing to annotate.

Paying for that rule is most of what shapes a Frost program. Anything that
outlives a call needs another way to be named, so a program keeping its monsters
past the turn loop puts them in a pool and passes a `Handle<Monster>` around.
The one borrow you write down is `ref T`. It can be returned, the frame and
region checks hold it to storage that outlives the call, and an accessor uses it
to hand back a place instead of a copy.

| in place of | Frost has |
| --- | --- |
| lifetimes on references | borrows that are parameter modes and cannot escape |
| a garbage collector | arenas and pools you can see |
| a long-lived pointer into a collection | a generational handle, a copy value that goes stale rather than dangling |
| destructors | linear resources, consumed exactly once, checked at compile time |
| exceptions and `Result` plumbing | failure sets, `-> T ! E` and `?` |
| dynamic dispatch | monomorphized generics, so the inner-loop call is direct |
| classes and methods | plain structs and free functions |
| a `const fn` promise | any function, worked out where a constant reads it |
| a vector type and its intrinsics | `[4]f32`, with the operators once per lane |

## What it does

Generics monomorphize over types, values, and functions, so a call in an inner
loop is direct rather than going through a pointer. A `for` walks a range or a
sequence as the index-and-bound loop it stands for. A literal leaves out a type
the context already carries, and every field keeps its name, since the name says
where the value lands. A region check keeps a pointer from
outliving the block or the stack frame it points into, and a constant can be a
folded expression or a call worked out before the program runs. There are no
visibility modifiers and no methods.

It calls C without a binding layer. An `extern fn` links against a C library with the natural ABI, including one that returns a struct by value and one that takes a Frost function as a callback with a typed context.

One typed intermediate representation feeds three backends, a Cranelift native path, a portable C path, and a small interpreter. A differential test runs every program through all three and checks the answers match, so a lowering bug shows up as a disagreement rather than as a wrong binary.

The compiler is written in Frost. `selfhosted/frost.frost` reproduces itself byte for byte through its own C backend and its own x86-64 assembly backend, so a build can go from source to a running compiler with no C compiler in the loop. A full native build of 58k lines runs at about 166,000 lines per second with code generation spread across cores, and `--incremental` rebuilds only the modules an edit can reach.

The standard library is ordinary Frost. It has length-carrying strings, a growable `Vec` and a hash map, a sort, the slab and structure-of-arrays `columns` containers, an archetype entity-component system, and vector, matrix, and quaternion math at both single and double precision. Output is one `print("{} of {}\n", a, b)` over a compile-time argument list: the holes are counted against the values where the call is written, and each value's type picks its writer while the call is compiled, so what runs is a direct write per value. See [`std/`](std), [docs/book/src/std/ecs.md](docs/book/src/std/ecs.md) and [docs/book/src/std/math.md](docs/book/src/std/math.md).

The engine is ordinary Frost too. [`lib/`](lib) is a window and input layer over SDL3, a renderer over wgpu with a render graph that orders passes by the resources between them, and an `App` a program composes plugins into. A plugin is a `fn(mut App)`, systems run in stages, a pass carries its own typed state, and no example in the tree says `unsafe`.

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
normative.

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

The second one covers the C-free path. `--emit-asm` makes the self-hosted
compiler write x86-64 assembly, which its own in-process assembler encodes to an
object file, so a build goes from Frost source to a running compiler with no C
compiler anywhere in the loop. The binary that comes out reproduces itself
exactly.

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

Ten programs, ordered so each one is the smallest step past the last. All but
the first and the last draw through wgpu:

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

Read `lit` first to see what the engine does for a program. It composes
`render_plugin`, spawns meshes and lights, and runs: no shader, no pipeline, no
bind group, no pass. The others that draw each write their own, since each is
about what a pass can be.

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

The extension lives in [`.vscode/frost`](.vscode) and is linked into your VS Code, since it is not on the marketplace. `just editor-dir` prints where this VS Code reads its extensions, which for a portable install is somewhere other than `~/.vscode/extensions`. A test holds the grammar's keywords and builtins to the compiler's own lists, so the two cannot drift apart.

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
├── tools/        # the wgpu binding generator
├── examples/     # runnable programs
├── bench/        # the benchmark generator
├── .vscode/      # editor settings, and the VS Code grammar for .frost
└── docs/book/   # the documentation, as an mdBook
```

`lib/` is four layers and each reaches one way: `platform` is the window and the
input, `renderer` is wgpu and the render graph, `engine` is the ECS seam and the
`App`, and `examples/graphics` sits on top. Which directory may import which is
declared in [`frost.json`](frost.json) and enforced by both compilers, so a
layer reaching upward fails to compile.

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
