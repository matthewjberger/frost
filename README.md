<p align="center">
  <a href="https://github.com/matthewjberger/frost"><img alt="github" src="https://img.shields.io/badge/github-matthewjberger/frost-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20"></a>
  <img alt="license" src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-fc8d62?style=for-the-badge&labelColor=555555" height="20">
</p>

`Frost` is a small, statically typed, data-oriented systems language. A program
is plain data and free functions that transform it. It is memory-safe without a
garbage collector and without lifetime annotations, it compiles to native code
through Cranelift or through portable C, and it compiles itself.

> Pre-1.0. The language is usable and the compiler is complete enough to have
> been written in itself, but the surface is still moving and there is no
> stability promise yet.

## Why it looks like this

Most safe systems languages ask you to describe how long a reference lives.
Frost removes the question instead. **A borrow is what a parameter mode means**,
so there is no reference type to store, return, or annotate:

```
damage :: fn(mut e: Entity, amount: i64) { e.hp = e.hp - amount }
```

`mut` borrows exclusively for the call. Unmarked reads. `move` takes ownership.
There is no `&` in the surface, so a borrow has nowhere to escape to, and that
one decision is why there are no lifetimes anywhere in the language.

What replaces the things references are normally used for:

| instead of | Frost has |
| --- | --- |
| a long-lived reference into a collection | a **generational handle**, a copy value that cannot read a reused slot |
| `Drop` and destructors | **linear resources**, consumed exactly once, checked |
| exceptions or `Result` plumbing | **failure sets**, `-> T ! E` and `?` |
| a garbage collector | **arenas and pools**, allocation you can see |
| dynamic dispatch | **monomorphized generics**, so the inner-loop call is direct |

## A first program

```
printf :: extern fn(fmt: ^i8, value: i64) -> i32

square :: fn(x: i64) -> i64 { x * x }

main :: fn() -> i64 {
    mut total : i64 = 0
    for i in 0..6 {
        total = total + square(i)
    }
    printf("%lld\n", total)   // 55
    0
}
```

```bash
frost program.frost          # compile, link, and run
```

## Highlights

- **Data-oriented, not object-oriented.** Plain structs and free functions. No
  methods, no inheritance, no vtables. Behaviour is not attached to data.
- **Ownership without lifetimes.** A borrow is a parameter mode and the compiler
  inserts it at the call.
- **Regions, still without lifetimes.** A `with arena { }` block owns an arena
  and a pointer into it may not outlive the block. A function's frame is checked
  the same way, so a pointer or slice naming a local cannot be returned.
- **Linear resources instead of `Drop`.** A `linear` value must be consumed
  exactly once, and forgetting is a compile error.
- **Generational handles and pools.** A stale handle can never read a reused
  slot, and the pool is ordinary Frost code rather than a runtime.
- **Generics by monomorphization.** `$T` types, `$N` values and `$f` functions,
  on structs, enums and functions alike, so a generic algorithm calls its
  comparator directly rather than through a pointer.
- **Failure sets.** `-> T ! E` says how a function fails, `?` hands a failure on,
  and both lower to an ordinary enum and a match.
- **Calls C directly** with `extern fn` and no glue, including functions that
  return a struct by value and callbacks that take a typed context.
- **Three backends that must agree.** One typed IR feeds Cranelift, portable C
  and a direct interpreter, and a differential test puts every program through
  all three.
- **Self-hosting.** `bootstrap/frost.frost` is a Frost compiler written in Frost
  that reproduces itself byte for byte, through both of its backends.
- **Separate compilation.** Each module is its own object, and `--incremental`
  rebuilds only what an edit can reach.

## Using it

```bash
cargo build --release

frost program.frost                              # compile, link, and run
frost --link -o program program.frost            # link to a named executable
frost --native -o program.o program.frost        # object file only
frost --emit-c -o program.c program.frost        # portable C instead of Cranelift
frost --run-ir program.frost                     # interpret the typed IR
frost --test program.frost                       # run the file's `test` blocks
frost --test tests/                              # run every test under a directory

frost --link --incremental -o program program.frost   # rebuild only what changed
frost --link --freestanding -o program program.frost  # link no C standard library
frost --link -L vendor -o program program.frost       # add an import search path
```

Requires a Rust toolchain and a C compiler (gcc or clang) for linking.

An import is looked for beside the importing file, then on `-L` and
`FROST_PATH`, then in the project's `frost.json`, then in the standard library
in [`std/`](std). See [docs/modules.md](docs/modules.md).

## Examples

Runnable programs in [`examples/native/`](examples/native). Start with
[`game_world.frost`](examples/native/game_world.frost) for handles and enums, and
[`generic_pool_library.frost`](examples/native/generic_pool_library.frost) for
one generic slab used at two element types.

## Documentation

Start with **[docs/authoring.md](docs/authoring.md)**, the practical guide to
writing correct Frost quickly.

**The language**

- [docs/tour.md](docs/tour.md) - a hands-on tour by example
- [docs/spec.md](docs/spec.md) - the reference and the grammar
- [docs/coming-from-rust.md](docs/coming-from-rust.md) - every Rust reflex mapped
  to its Frost equivalent
- [docs/syntax-design.md](docs/syntax-design.md) - why the syntax is shaped this
  way, difference by difference

**The ideas**

- [docs/philosophy.md](docs/philosophy.md) - goals, non-goals, and why
  data-oriented rather than object-oriented
- [docs/memory-safety.md](docs/memory-safety.md) - how safety works without a GC
  or lifetimes, and what it does not cover
- [docs/allocators.md](docs/allocators.md) - arenas, pools, and the one platform
  call
- [docs/native-pools.md](docs/native-pools.md) - moving the memory model out of C
  and into the language

**The compiler**

- [docs/architecture.md](docs/architecture.md) - the pipeline, and what each
  backend supports
- [docs/build-modes.md](docs/build-modes.md) - native, freestanding and
  self-hosted are three separate axes
- [docs/modules.md](docs/modules.md) - where an import is looked for, the
  manifest, and the standard library
- [docs/separate-compilation.md](docs/separate-compilation.md) - the module
  boundary and what `--incremental` rebuilds
- [docs/c-compatibility.md](docs/c-compatibility.md) - calling C, and the C
  backend
- [docs/callbacks.md](docs/callbacks.md) - callbacks with a typed context
- [docs/self-hosting.md](docs/self-hosting.md) - the fixpoint, and compile speed
- [docs/roadmap.md](docs/roadmap.md) - what was built, in the order it was built

## Project layout

```
frost/
├── src/          # the reference compiler, in Rust
├── std/          # the standard library, in Frost
├── bootstrap/    # a Frost compiler written in Frost
├── runtime/      # a small C runtime (bounds check, assert, IO helpers)
├── examples/     # runnable programs
├── bench/        # the benchmark generator
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

External contributions are not being accepted yet. The language surface is still
settling. Guidelines will land here once it stops moving.

## License

Dual-licensed under either of:

- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `frost` by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
