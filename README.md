<p align="center">
  <a href="https://github.com/matthewjberger/frost"><img alt="github" src="https://img.shields.io/badge/github-matthewjberger/frost-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20"></a>
  <img alt="license" src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-fc8d62?style=for-the-badge&labelColor=555555" height="20">
</p>

# Frost

**A data-oriented systems language that is memory-safe with no garbage collector and no lifetimes, and compiles itself.**

A Frost program is plain data and free functions that transform it. There are no classes and nothing is allocated behind your back. It compiles to native code through Cranelift or to portable C, and the compiler is written in Frost.

> Pre-1.0. It is usable today and complete enough to have been written in itself, but the surface is still moving and nothing is promised stable yet.

## Borrows are parameter modes

Most safe systems languages make you describe how long a reference lives. Frost deletes the question. A borrow is not a type you write, it is what a parameter mode means.

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

## Hello, Frost

```frost
Kind :: enum { Hero, Monster { damage: i64 } }
Entity :: struct { hp: i64, kind: Kind }

// `mut` borrows for the call and changes the caller's value. There is no `&`.
heal :: fn(mut e: Entity, amount: i64) {
    e.hp = e.hp + amount
}

// An unmarked parameter borrows to read. `match` reads the enum and binds the
// payload of the variant it took.
attack :: fn(e: Entity) -> i64 {
    match e.kind {
        case .Hero: 10
        case .Monster { damage }: damage
    }
}

main :: fn() -> i64 {
    mut hero := Entity { hp = 90, kind = Kind::Hero }
    heal(hero, 10)                 // hero is borrowed and changed
    print hero.hp                  // 100
    print attack(hero)             // 10
    0
}
```

```bash
frost hero.frost                   # compile, link, and run
```

For long-lived data, an `Entity` lives in a pool and is named by a `Handle`, a small copy value rather than a pointer. Freeing a slot raises its generation, so a handle to a reused slot reads as stale rather than reading whatever took its place. The pool is ordinary Frost code, not a runtime, and is generic over element type and capacity. See [`examples/native/game_world.frost`](examples/native/game_world.frost).

## What it does

The language has structs and tagged enums, `match` with payload and tuple patterns that has to cover every variant or say what the rest do, and generics that monomorphize over types, values, and functions, so a call in an inner loop stays direct rather than going through a pointer. A resource that must be released is marked `linear` and the compiler counts it, consumed once on every path out or the program does not build. Long-lived data lives in pools addressed by generational handles, a region check keeps a pointer from outliving the block or the stack frame it points into, and a value constant can be a folded integer expression. There are no visibility modifiers and no methods.

It calls C without a binding layer. An `extern fn` links against a C library with the natural ABI, including one that returns a struct by value and one that takes a Frost function as a callback with a typed context.

One typed intermediate representation feeds three backends, a Cranelift native path, a portable C path, and a small interpreter. A differential test runs every program through all three and checks the answers match, so a lowering bug shows up as a disagreement rather than as a wrong binary.

The compiler is written in Frost. `selfhosted/frost.frost` reproduces itself byte for byte through its own C backend and its own x86-64 assembly backend, so a build can go from source to a running compiler with no C compiler in the loop. A full native build of 58k lines runs at about 166,000 lines per second with code generation spread across cores, and `--incremental` rebuilds only the modules an edit can reach.

The standard library is ordinary Frost. It has length-carrying strings, a growable `Vec` and a hash map, file and formatted output, a sort, the slab and structure-of-arrays `columns` containers, and single-precision vector, matrix, and quaternion math. See [`std/`](std) and [docs/math.md](docs/math.md).

## Getting started

```bash
cargo build --release          # build the compiler

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

Requires a Rust toolchain and a C compiler (gcc or clang) for linking. An import is looked for beside the importing file, then on `-L` and `FROST_PATH`, then in the project's `frost.json`, then in the bundled [`std/`](std).

### Editor support

```bash
just install-editor    # link the VS Code extension, then reload the window
```

A `.frost` file gets syntax highlighting, snippets for the declaration forms, and validation for `frost.json`. A fenced block tagged `frost` in a markdown file is highlighted the same way. `Ctrl+Shift+B` runs every compiler check over the open file and puts what it finds in the Problems panel, on the line that caused it.

The extension is in [`.vscode/frost`](.vscode) rather than on the marketplace, so it is linked rather than installed. `just editor-dir` prints where this VS Code reads its extensions, which is not `~/.vscode/extensions` when VS Code is a portable install. The grammar's keywords and builtins are held to the compiler's own lists by a test, so the two cannot drift apart.

## Status

Everything above works today and is checked by the test suite on every commit, including both self-hosting fixpoints and the three-backend differential run. The compiler has compiled itself.

The two compilers have been audited against each other by running the same programs through both and comparing what each accepts. That was expected to be a confirmation and was not. It found four places where they disagreed, one of them a case where the same source meant different things rather than a compiler missing a check, and all four are closed. The probes that found them are tests now.

What is left, roughly in order:

- Renaming a name on import. Two modules that export the same name is already a compile error. The rename was designed and then held back, since the only thing it resolves is a collision between two third-party libraries you cannot edit, and there is no third-party ecosystem yet.
- An `f64` or scalar-generic version of the math library.

## Documentation

[authoring](docs/authoring.md) is the practical guide to writing correct Frost quickly, and [tour](docs/tour.md) walks the language by example. [spec](docs/spec.md) is the reference and grammar, [syntax-design](docs/syntax-design.md) explains why the syntax reads the way it does, [coming-from-rust](docs/coming-from-rust.md) maps each Rust reflex across, and [math](docs/math.md) covers the standard-library math.

The reasoning behind the design is in [philosophy](docs/philosophy.md), [memory-safety](docs/memory-safety.md), [allocators](docs/allocators.md), and [native-pools](docs/native-pools.md). The compiler itself is covered by [architecture](docs/architecture.md), [build-modes](docs/build-modes.md), [modules](docs/modules.md), [separate-compilation](docs/separate-compilation.md), [c-compatibility](docs/c-compatibility.md), [callbacks](docs/callbacks.md), [self-hosting](docs/self-hosting.md), and [roadmap](docs/roadmap.md).

## Project layout

```
frost/
├── selfhosted/   # the Frost compiler, written in Frost
├── src/          # the bootstrap compiler, in Rust: builds stage 0, and is the oracle
├── std/          # the standard library, in Frost
├── runtime/      # a small C runtime (bounds check, assert, IO helpers)
├── examples/     # runnable programs
├── bench/        # the benchmark generator
├── .vscode/      # editor settings, and the VS Code grammar for .frost
└── docs/
```

## Tests

```bash
cargo test              # everything, including both self-hosting fixpoints
just test-interfaces    # the whole suite again, built from module interfaces
just bench-scaling      # how the pipeline scales
just bench-incremental  # what --incremental saves
```

## Contributing

External contributions are not being accepted yet. The language surface is still settling. Guidelines will land here once it stops moving.

## License

Dual-licensed under either of:

- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in `frost` by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
