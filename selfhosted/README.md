# The Frost compiler, written in Frost

A compiler for Frost, written in Frost, across thirteen modules. It lexes,
parses, type-checks, and emits either a C translation unit or x86-64 assembly.
`frost.frost` is the driver; the rest is named below.

**This is the compiler Frost is for.** The Rust compiler in `src/` is the
bootstrap: it compiles stage 0, and it is the differential oracle the tests
compare against. Both are scaffolding. What a person writing Frost is meant to
run is this one, which is why it is held to the full language and to the same
speed promise rather than to a lower bar. Where it is behind, that is a port
waiting on [docs/roadmap.md](../docs/roadmap.md) rather than a divergence.

**It self-hosts, twice over.** It compiles its own source; a compiler built from
that output compiles the same source again; the two outputs are byte-identical.
That three-stage fixpoint holds through both of its backends, and both are
checked on every build by `self_hosting_is_a_fixpoint` and
`native_self_hosting_is_a_fixpoint` in `tests/native.rs`. Through the assembly
backend there is no C compiler anywhere in the loop. The fixpoint is how the
compiler is checked, not what it is for.

It is written the way the language wants a compiler written: Frost-native arenas
for the tokens, the AST and the symbol tables, integer indices instead of heap
pointers between nodes, and free functions over that data. No closures, no
dynamic collections, every reference second-class, and its memory comes from the
language's own allocator rather than a runtime pool.

## What it implements

Not a toy subset. It checks its own programs rather than deferring to whatever
compiles its output, which it has to, because through the assembly backend
nothing downstream would catch a mistake.

- **Types and layout.** `i64`, `i8`, `u8`, `bool`, pointers `^T`, structs
  passed and returned by value, and `e^`, `e[i]`, `ptr_to`, `ptr_cast`,
  `sizeof`.
- **Ownership and linearity.** Use after move, and `linear` values that must be
  consumed exactly once, with a read parameter borrowing and `move` consuming.
- **Generics by monomorphization** over `$T`, on structs and functions,
  instantiated once per concrete type argument.
- **Enums with payloads** and `match` over them.
- **Allocation sources.** `uses A` on a function, `with a { }` around a call.
- **Regions.** A `with` block is a region and an arena pointer may not outlive
  it.
- **Failure sets.** `-> T ! E` and `?`.
- **Imports.** `import "path"` joins another file's declarations to this one.
- **`extern fn`** for C linkage, which is how it does its own IO.

Both backends emit from the same checked program: C through
`frost_emit_*` helpers, or x86-64 assembly directly.

## What is not here yet

All of it is a work list, in order, as items 6 through 18 of
[docs/roadmap.md](../docs/roadmap.md). Derived by reading `src/types.rs` against
this file's type codes and the bootstrap's flags against `main`, rather than
from memory:

- **A command line.** This is the largest one and the nearest to a user. It
  reads `FROST_INPUT`, writes to standard output, and links nothing. No file
  argument, no `-o`, no `--native`, no `--link`.
- **Most of the scalar types.** `i16`, `i32`, `isize`, `u16`, `u32`, `u64`,
  `usize`, `f32`, `f64`, `str`. The floats need SSE in the assembly backend,
  which has none.
- **Arrays and slices**, so a fixed buffer is a `malloc` and a pointer.
- **Value generics** (`$N: usize`) and compile-time function arguments (`$f`),
  since a template here carries one type parameter rather than a list.
- **Generic enums**, **handles and pools**, **distinct types**.
- **Callbacks with a typed context.**
- **C functions returning a struct by value**, which needs the per-target ABI
  classification in `src/c_abi.rs`.
- **`test` blocks**, so a file carrying one does not compile.
- **Diagnostics with a file, line and column.** Errors here have none.
- **Module search paths**, so the standard library cannot be imported by name.
- **Speed parity**: parallel code generation, separate compilation,
  `--incremental`.

## How it works

1. **The lexer** turns the source into a flat `Arena<Token>` in one pass. It
   drops whitespace and `//` comments, reads integer and string literals,
   classifies keyword spellings to their own token kinds, and lexes operators
   greedily, so `::`, `:=`, `->`, `<=` and `&&` are single tokens. Each token
   keeps the byte range it came from, which is how identifiers stay interned by
   range.
2. **The AST is an arena of `Node` records.** `Arena<$T>` is a bump allocator
   over one `malloc`, written in the language itself. `arena_push` appends and
   returns an index; `arena_at` turns an index back into a `^Node`. A node names
   its children by index, which is the data-oriented replacement for pointers
   between heap nodes, and sibling statements, parameters and arguments thread
   through a `next` field.
3. **Names are byte ranges.** An identifier occurrence is an offset and a length
   into the source. Locals re-emit those bytes, so a local becomes a named C
   variable and no local symbol table is needed. Function names intern through
   one global table to a small integer.
4. **The parser cursor** lives in a `Parser` struct threaded by `mut` through
   the recursive-descent functions, which is how you carry mutable state where
   references are second-class.
5. **Code generation** walks the arena dispatching on `node.kind`. The C backend
   emits text through runtime helpers; the assembly backend emits x86-64
   directly, with its own register allocation, stack frames and calling
   convention.

## Building and running

```
just selfhost-build                       # build it with the bootstrap compiler
just selfhost-run  examples/selfhosted/hello.frost   # compile a file, via its C backend
just selfhost-native examples/selfhosted/hello.frost # via its assembly backend
just selfhost-check                       # the three-stage fixpoint
just selfhost-test                        # every self-hosting check
```

By hand, the compiler reads `FROST_INPUT` and writes to standard output:

```
frost --link -o selfhosted/frost.exe selfhosted/frost.frost
FROST_INPUT=program.frost ./selfhosted/frost.exe > out.c
cc out.c -o out && ./out

FROST_BACKEND=asm FROST_INPUT=program.frost ./selfhosted/frost.exe > out.s
cc out.s -o out && ./out
```

## The modules

In dependency order, which is also a topological order of what calls what. Each
file states what it is about at the top and lists what it offers on one `export`
line.

| module | lines | what it is |
| --- | --- | --- |
| `core.frost` | 444 | externs, the constant tables, the records, the arena, `Parser` |
| `lexer.frost` | 181 | source bytes to a flat arena of tokens, in one pass |
| `cursor.frost` | 46 | the token cursor everything reading tokens goes through |
| `imports.frost` | 203 | every file's text into one buffer, dependencies first |
| `names.frost` | 428 | interning, visibility, synthesized names, type codes, AST constructors |
| `types.frost` | 358 | typing, and the checks that ride on it: moves, linearity, field and call types |
| `parser.frost` | 1,373 | recursive descent, and the re-parse that instantiates a generic |
| `emit.frost` | 69 | what both backends emit through |
| `layout.frost` | 70 | sizes, alignments, field offsets |
| `emit_c.frost` | 794 | the C backend |
| `emit_asm.frost` | 864 | the x86-64 backend |
| `regions.frost` | 251 | the region check |
| `frost.frost` | 206 | the driver |

The order is acyclic: no module names anything from a module below it. The
assembly backend does not depend on the C one, which is what `emit.frost` is
for. How the boundaries were drawn is item 6 of
[docs/roadmap.md](../docs/roadmap.md).

## Where the measurements live

[docs/self-hosting.md](../docs/self-hosting.md) has the checklist in dependency
order, what each item cost, the compile-speed numbers, and the bugs that only
the assembly backend could have found. That last part is the useful bit: C needs
neither struct offsets nor a scope discipline of its own, so emitting assembly
surfaced three real bugs the C backend had been papering over.
