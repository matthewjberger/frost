# Frost

Frost is a small, statically typed, data-oriented systems language. A program is
plain data (structs and enums) plus free functions that transform it. It is
memory-safe without a garbage collector and without lifetime annotations, and it
compiles to native code through Cranelift or through portable C.

## Highlights

- **Data-oriented, not object-oriented** - plain structs and free functions, no
  methods, inheritance, or vtables.
- **Ownership without lifetimes** - a borrow is what a parameter mode means
  (unmarked reads, `mut` writes, `move` takes ownership) and the compiler
  inserts it at the call. There is no `&` in the surface, so a borrow has
  nowhere to be stored and nothing to annotate.
- **Regions, still without lifetimes** - a `with arena { }` block owns an arena
  and a pointer into it may not outlive the block. A function's frame is checked
  the same way, so a pointer or slice naming a local cannot be returned.
- **Allocation is declared** - `uses Arena` draws an allocation capability,
  threaded implicitly and supplied by the `with` block that provides it.
- **Failure sets** - `-> T ! E` says how a function fails, `?` hands a failure
  on, and both lower to an ordinary enum and match.
- **Linear resources instead of `Drop`** - a `linear` value must be consumed
  exactly once, and forgetting is a compile error.
- **Generational handles and pools** - the safe replacement for reference-heavy
  object graphs. A stale handle can never read a reused slot.
- **Generics by monomorphization** - `$T` types, `$N` values, and `$f`
  functions, so a generic algorithm calls its comparator directly rather than
  through a pointer. No runtime dispatch anywhere.
- **Function pointers and non-capturing function literals**, no closures.
- **Modules with a one-line export surface** - items are private by default and
  a file lists what it offers with `export`. No `pub` anywhere, all struct
  fields public.
- **One typed IR, three backends that must agree** - Cranelift, portable C, and
  a direct IR interpreter, cross-checked by a differential test.
- **Calls C directly** with `extern fn`, no glue, including C functions that
  return a struct by value, classified per target rather than assumed.
- **Callbacks with a typed context** - register a Frost function with a C
  library by moving a typed context in; no `^u8`, no cast, and forgetting to
  unregister is a compile error.
- **Separate compilation** - each module is its own object, and
  `--incremental` rebuilds only the modules an edit can reach.
- **In-module tests** - `test "name" { assert(...) }` run by `frost --test`.

## Documentation

- [docs/authoring.md](docs/authoring.md): the dense practical guide to writing
  correct Frost fast, and the best single document to start from
- [docs/tour.md](docs/tour.md): a hands-on tour of the language by example
- [docs/coming-from-rust.md](docs/coming-from-rust.md): a detailed guide for
  Rust programmers, mapping every reflex to its Frost equivalent
- [docs/syntax-design.md](docs/syntax-design.md): why the syntax is shaped the
  way it is, difference by difference against Rust
- [docs/philosophy.md](docs/philosophy.md): design philosophy, goals and
  non-goals, and why Frost is data-oriented rather than object-oriented
- [docs/memory-safety.md](docs/memory-safety.md): how Frost guarantees memory
  safety without a garbage collector or lifetime annotations
- [docs/c-compatibility.md](docs/c-compatibility.md): calling C and the C backend
- [docs/architecture.md](docs/architecture.md): the compiler pipeline and what
  the native backend supports today
- [docs/native-pools.md](docs/native-pools.md): a spike writing a generational
  pool in pure Frost, and the plan to move the memory model out of the C runtime
- [docs/allocators.md](docs/allocators.md): the layered allocator strategy (arena,
  pool, and the one platform call), and where it sits on the roadmap
- [docs/spec.md](docs/spec.md): the language reference and grammar
- [docs/roadmap.md](docs/roadmap.md): what is left, in the order to do it
- [docs/separate-compilation.md](docs/separate-compilation.md): the module
  boundary, one object per module, and what `--incremental` rebuilds
- [docs/callbacks.md](docs/callbacks.md): callbacks with a typed context, the
  design and the record of building it

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

## Building and running

Requires a Rust toolchain and a C compiler (gcc or clang) for linking.

```bash
cargo build --release

# Compile the file, link it, and run it natively
frost program.frost

# Compile to an object file, or link to a named executable
frost --native -o program.o program.frost
frost --link -o program program.frost

# Emit portable C instead of using Cranelift
frost --emit-c -o program.c program.frost

# Rebuild only the modules an edit reaches, keeping objects under .frost-build
frost --link --incremental -o program program.frost

# Link with no C standard library (a static-arena program needs no runtime)
frost --link --freestanding -o program program.frost

# Interpret the typed IR directly (the reference oracle)
frost --run-ir program.frost

# Run the file's `test` blocks
frost --test program.frost
```

## Examples

Runnable programs live in `examples/native/`. Start with `game_world.frost` (an
entity system) and `pool_linked_list.frost` (handles as links).

## Project structure

```
frost/
├── src/
│   ├── lib.rs          # library exports
│   ├── lexer.rs        # tokenizer
│   ├── parser.rs       # AST parser
│   ├── imports.rs      # import resolution and module privacy
│   ├── interface.rs    # what a caller needs to compile against a module
│   ├── build_cache.rs  # what --incremental remembers between builds
│   ├── regions.rs      # arena regions and frame escapes
│   ├── param_modes.rs  # parameter modes to references and call borrows
│   ├── callbacks.rs    # callback registrations and their declaration rules
│   ├── c_abi.rs        # how C returns a struct, per target
│   ├── ownership.rs    # move, borrow, and linearity checking
│   ├── ir.rs           # typed IR definitions
│   ├── ir_build.rs     # AST to typed IR lowering
│   ├── ir_typecheck.rs # type checking on the IR
│   ├── ir_codegen.rs   # IR to Cranelift native backend
│   ├── ir_c.rs         # IR to portable C backend
│   ├── ir_interp.rs    # direct IR interpreter (differential oracle)
│   ├── types.rs        # type definitions
│   └── bin/frost.rs    # the compiler CLI
├── runtime/            # small C runtime (pools, bounds check, assert)
├── bootstrap/          # a Frost-subset compiler written in Frost
├── examples/native/    # example programs
└── docs/               # documentation
```

## Tests

```bash
cargo test
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
