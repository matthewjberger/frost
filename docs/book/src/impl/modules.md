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
[spec.md](../reference/conformance.md).

## The order

1. Beside the importing file. Always tried, and always tried first, because a
   file's neighbours are the most specific thing it could mean.
2. `-L DIR` on the command line, repeatable, in the order given.
3. `FROST_PATH`, split the way the platform splits a path list.
4. The project manifest, `frost.json` beside the file named on the command
   line, which may declare `paths`.
5. The standard library.

Command line beats environment beats project file, which is the order of how
deliberately each one was said. The standard library is last, so a project can
shadow it by putting a file of the same name somewhere earlier.

## The manifest

`frost.json`, optional, beside the entry file:

```json
{ "name": "demo", "paths": ["lib", "vendor/things"] }
```

Both fields are optional. `paths` are relative to the manifest.

It answers one question, where a library lives. It carries no versions and
fetches no dependencies from anywhere, since compiling a program takes neither
and each is a decision that would be hard to take back. The format is JSON, the
same serde and JSON that interfaces and build records already use.

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
