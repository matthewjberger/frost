# Finding a module

`import "x.frost"` used to mean one thing: a file beside the one that wrote the
import. That is right for a program's own files and useless for a library,
because there is no way to say "the slab, wherever it is installed" without
writing `../../std/slab.frost` and hoping.

There are now four more places to look, and one rule about identity that makes
them safe.

## The order

1. **Beside the importing file.** Always tried first, and always tried, because
   a file's neighbours are the most specific thing it could mean and because
   every program written before this relied on it.
2. **`-L DIR` on the command line**, repeatable, in the order given.
3. **`FROST_PATH`**, split the way the platform splits a path list.
4. **The project manifest**, `frost.json` beside the file named on the command
   line, which may declare `paths`.
5. **The standard library.**

Command line beats environment beats project file, which is the order of how
deliberately each one was said. The standard library is last so a project can
always shadow it by putting a file of the same name somewhere earlier.

## The manifest

`frost.json`, optional, beside the entry file:

```json
{ "name": "demo", "paths": ["lib", "vendor/things"] }
```

Both fields are optional. `paths` are relative to the manifest.

It answers exactly one question, where a library lives, and deliberately not the
others a manifest could grow into. There are no versions and no dependencies
fetched from anywhere, because none of that is needed to compile a program and
each of it is a decision that would be hard to take back. JSON rather than a
format of its own, because interfaces and build records are already serde and
JSON and a second format is a second thing to learn.

## The standard library

`std/` in this repository. It is found, in order, at `FROST_STD` if that is set,
then `std` beside the compiler, which is what an installed layout looks like,
then two directories up from the compiler, which is what `cargo build` produces:
the binary lands in `target/debug` and the library is at the repository root.

So `import "maybe.frost"` works from anywhere with nothing declared.

What is in it is Frost, not compiler magic. `std/maybe.frost` is a generic enum,
and it exists because the language grew generic enums rather than because the
compiler has an optional type. `std/slab.frost` is the generational pool, whose
storage and free list are ordinary struct fields.

## Identity, which is the part that has to be right

A module's identity is not where the machine keeps it. Private symbol names are
mangled from it and the build cache is keyed on it, so if identity varied by
install path then a cached object would not be portable and two machines would
disagree about a symbol.

So identity is **the path relative to the root the module was found under, with
that root's label in front**:

- a file in the project stays `lib/slab.frost`, exactly as before;
- a standard library module is `std/maybe.frost` wherever the standard library
  is installed;
- a file from `-L`, `FROST_PATH` or the manifest is named relative to that
  directory.

That is what keeps `--incremental` and separate compilation working across
machines, and it is why the search roots carry a label rather than only a path.
See [separate-compilation.md](separate-compilation.md).
