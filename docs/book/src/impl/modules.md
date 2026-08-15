# Finding a module

`import "x.frost"` names a file. The first place looked is beside the importing
file, which covers a program's own files, and four more places follow, so
a library can be named as "the slab, wherever it is installed" instead of as
`../../std/slab.frost`.

An import also says what the importing file may name: its own declarations plus
the exported names of the modules it imports directly. Importing is not
transitive, so a file that uses a name has the module exporting it in its own
import list.

`import "list.frost" (insert as list_insert)` reads one of those names under
another, which is the answer when two modules you cannot edit export the same
name. The rename belongs to the file that wrote it. See section 5.5 of
[declarations.md](../reference/declarations.md).

## The order

1. Beside the importing file. Always tried, and always tried first, because a
   file's neighbours are the most specific thing it could mean.
2. `-L DIR` on the command line, repeatable, in the order given.
3. `FROST_PATH`, split the way the platform splits a path list.
4. The project manifest, the nearest `frost.json` at or above the file named on
   the command line, which may declare `paths`.
5. The standard library.

Command line beats environment beats project file, which is the order of how
deliberately each one was said. The standard library is last, so a project can
shadow it by putting a file of the same name somewhere earlier.

## The manifest

`frost.json`, optional, in the entry file's directory or any directory above it:

```json
{ "name": "demo", "paths": ["lib", "vendor/things"] }
```

Every field is optional. `paths` are relative to the manifest. Three more say
something about the project rather than about where a file is found: `layers`
lists its directories lowest first, and a file may import from its own layer or
one declared before it and from no later one; `prefixes` maps a directory to the
prefix its exported names share, which `frost lint` holds them to; `generated`
names the files a program of the project writes.

It answers one question about an import, where a library lives. It carries no
versions and fetches no dependencies from anywhere, since compiling a program
takes neither and each is a decision that would be hard to take back. The format
is JSON, the same serde and JSON that interfaces and build records already use.

## A file this project writes

```json
{ "generated": [
    { "output": "lib/renderer/wgpu.frost",
      "from":   "tools/wgpu_bindgen.frost",
      "inputs": ["lib/renderer/wgpu/webgpu.json"] }] }
```

`frost generate` compiles each `from` with the compiler that was asked, runs it
with the output path first and the inputs after, and puts a `.frost` output
through the formatter. `frost generate --check` writes somewhere else and
compares the bytes, so a checkout can be held to generated files that are not
stale. Every path is relative to the manifest, the way a search directory is.

The generator is an ordinary Frost program taking a file to write and files to
read, so it compiles, runs and reads on its own without knowing a manifest
exists. What it writes stays a file in the tree: a reader opens it, a diff shows
what a schema change did to it, and the compiler that reads it afterward is the
ordinary one.

That is the whole of build-time generation here, and the shape is the argument.
A metaprogram that injected declarations into a compilation would produce the
same bindings invisibly, and every check in this compiler defaults toward
refusing what it cannot see. Generating source on disk instead keeps the
compile-time layer a function from shapes to values and refusals, and it is what
made a gap in `emit_handles` visible as a diff rather than as a resize that
failed on someone's machine.

## The standard library

`std/` in this repository. It is found, in order, at `FROST_STD` if that is set,
then `std` beside the compiler, the layout an install has, then two directories
up from the compiler, the layout `cargo build` produces. The binary lands in
`target/debug` and the library is at the repository root.

So `import "option.frost"` works from anywhere with nothing declared.

The standard library is written in Frost. `std/option.frost` is a generic enum,
built out of the language's own generic enums. `std/slab.frost` is the
generational pool, whose storage and free list are ordinary struct fields.
`std/strings.frost` is walks over `str`, whose only primitives are `str_len` and
`s[i]`. `std/io.frost` wraps the runtime's emit helpers so a first program can
skip declaring `printf` itself.

A module's `test` blocks are its own. They are compiled when that file is the
one named to `--test`, and dropped when it is imported, so a library with tests
keeps the test harness out of every program that uses it and `--test` on a
program runs the program's own tests. `frost --test std/` runs the standard
library's own.

## A file spliced in while compiling

`include_str("path")` puts a file's bytes into the program as a string literal.

```frost,sketch
SHADER :: include_str("shape.wgsl")
```

The path is relative to the file the call is written in, and it takes one string
literal, so what gets read is settled while the program is being compiled.

The splice happens between the lexer and the parser: four tokens become one
string literal, and everything downstream of the lexer sees an ordinary one. The
type checker sees a `str`, the region walk sees a literal, and both backends
emit it the way they emit any other. That is the whole implementation, and it is
why an included file may hold any bytes at all: they arrive as text.

Carriage returns are dropped as the bytes come in, so a file checked out with
Windows line endings and one checked out with Unix line endings splice the same
program. A shader whose text differed by line ending would otherwise hash
differently on two machines and force a rebuild on each.

Which is the other half. An included file is part of the module's hash, so
`--incremental` rebuilds a module when a file it reads changes, the same way it
rebuilds when the module's own source changes. The include paths are read off
the tokens rather than off the parsed tree, because whether a cached module is
stale is answered before it is parsed.

`lib/renderer/lit.frost` uses it for the shader it hands the GPU, which is what
keeps the shader a `.wgsl` file an editor highlights rather than a string
constant in the middle of Frost.

## Identity

A module's identity is the path relative to the root it was found under, with
that root's label in front. Private symbol names are mangled from it and the
build cache is keyed on it, so an identity that varied by install path would
make a cached object unportable and leave two machines disagreeing about a
symbol.

- a file in the project is `lib/slab.frost`, the path a reader would write.
- a standard library module is `std/option.frost` wherever the standard library
  is installed.
- a file from `-L`, `FROST_PATH` or the manifest is named relative to that
  directory.

That keeps `--incremental` and separate compilation working across machines, and
it is why a search root carries a label alongside its path. See
[separate-compilation.md](separate-compilation.md).
