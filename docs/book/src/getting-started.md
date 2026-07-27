# Getting started

```bash
cargo build --release          # build the compiler
just install                   # put it on PATH as `frost`
```

Requires a Rust toolchain and a C compiler (gcc or clang) for linking.

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

An import is looked for beside the importing file, then on `-L` and
`FROST_PATH`, then in the project's `frost.json`, then in the bundled `std/`.
See [finding a module](impl/modules.md).

`just install-self` puts the self-hosted compiler on PATH as `frostc`. The two
names are deliberate: this project compares the two compilers constantly, and
one name would mean the second install hides the first with no way to tell which
one ran.

## Editor support

```bash
just install-editor    # link the VS Code extension, then reload the window
```

A `.frost` file gets syntax highlighting, snippets for the declaration forms,
and validation for `frost.json`. A fenced block tagged `frost` in a markdown
file is highlighted the same way. `Ctrl+Shift+B` runs every compiler check over
the open file and puts what it finds in the Problems panel, on the line that
caused it.

The extension is in `.vscode/frost` rather than on the marketplace, so it is
linked rather than installed. `just editor-dir` prints where this VS Code reads
its extensions, which is not `~/.vscode/extensions` when VS Code is a portable
install. The grammar's keywords and builtins are held to the compiler's own
lists by a test, so the two cannot drift apart.

## Tests

```bash
cargo test              # everything, including both self-hosting fixpoints
just test-interfaces    # the whole suite again, built from module interfaces
just bench-scaling      # how the pipeline scales
just bench-incremental  # what --incremental saves
```

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
└── docs/book/    # this book
```
