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

`just install-self` puts the self-hosted compiler on PATH as `frostc`. Two
names keep both compilers on PATH at once, so you can run a program through
each and compare what you get.

## Editor support

```bash
just install-editor    # link the VS Code extension, then reload the window
```

A `.frost` file gets syntax highlighting, snippets for the declaration forms,
and validation for `frost.json`. A fenced block tagged `frost` in a markdown
file is highlighted the same way.

Everything else comes from the compiler. The extension starts one `frostc lsp`
and asks it: the reports as you type and the fixes they carry, go to
definition, declaration, type definition and implementation, hover, references,
the outline, workspace search, rename, completion and what a call takes,
folding, selection ranges, document links, colouring by what a name is, inlay
hints, code lenses, the call and type hierarchies, and layout of a file, of a
range, or of a line as you close it. The passes that build a program are what
answer, so a report the editor underlines is a report the build refuses on.

For a `.frost` file the extension turns on four settings a stock window leaves
off: linked editing, layout as you type, semantic highlighting, and inlay
hints. Say otherwise in your own settings to turn one back off.

`Ctrl+Shift+B` runs every compiler check over the open file and puts what it
finds in the Problems panel, on the line that caused it.

The extension lives in `.vscode/frost` instead of the marketplace, and
`just install-editor` links it into place. `just editor-dir` prints the
directory this VS Code reads extensions from, which for a portable install is
somewhere other than `~/.vscode/extensions`. A test holds the grammar's
keywords and builtins to the compiler's own lists.

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
├── runtime/      # the support runtime: aborts and assertions in Frost, IO in C
├── lib/          # the engine, in Frost: platform, renderer, engine
├── tools/        # the wgpu binding generator
├── examples/     # runnable programs
├── bench/        # the benchmark generator
├── tests/        # the harness
├── .vscode/      # editor settings, and the VS Code grammar for .frost
└── docs/book/    # this book
```
