# The Frost compiler, written in Frost

A compiler for Frost, written in Frost. It lexes, parses, type-checks, and emits
either a C translation unit or x86-64 assembly. `frost.frost` is the driver.

**This is the compiler Frost is for.** The Rust compiler in `src/` is the
bootstrap: it compiles stage 0, and it is the differential oracle the tests
compare against. Both are scaffolding. What a person writing Frost is meant to
run is this one, which is why it is held to the full language and to the same
speed promise rather than to a lower bar. Where it is behind, that is a port
waiting rather than a divergence.

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

- **Types and layout.** `i64`, `i8`, `u8`, `bool`, `str`, pointers `^T`, fixed
  arrays `[N]T`, slices `[]T`, `Handle<T>`, structs passed and returned by
  value, and `e^`, `e[i]`, `ptr_to`, `ptr_cast`, `sizeof`.
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
- **`extern fn`** for C linkage, which is how it does its own IO, and
  `safe extern fn` for one audited not to need an `unsafe` block at its calls.
- **`test` blocks**, run by `--test`, which reports each test and summarises.
- **Diagnostics carrying a file, line and column**, since a compiler that
  refuses a program owes you the position.

Both backends emit from the same checked program: C through
`frost_emit_*` helpers, or x86-64 assembly directly.

## What is not here yet

The target is the full language, so everything below is work waiting rather than
a decision. Each line was checked by compiling a program that uses the feature,
rather than from memory, because this list had drifted from the compiler once
already.

- **A function may not be named after a keyword.** `print` is a statement
  keyword here, so `std/io.frost`, which exports a function called `print`,
  cannot be compiled by this compiler at all. Retiring the `print` statement in
  favour of the library function is what unblocks the standard library, and is
  the single change that buys the most.
- **Value generics** (`$N: usize`) and **more than one generic parameter**,
  since a template here carries one type parameter rather than a list. This is
  what `std/slab.frost` needs.
- **Generic enums.**
- **`distinct` types.**
- **Compile-time function arguments** (`$f`), and calling through a function
  **pointer held in a parameter**: `f(x)` reads as a call to a function named
  `f` rather than through the value.
- **Callbacks with a typed context.**
- **C functions returning a struct by value**, which needs the per-target ABI
  classification in `src/c_abi.rs`. The declaration parses; the ABI does not.
- **Module search paths**, so the standard library cannot be imported by name.
- **Speed parity**: parallel code generation, separate compilation,
  `--incremental`. The front end is already fast; these are what the bootstrap
  has and this does not.

Each unsupported form above is refused with a position rather than misparsed:
they used to run into the function parser and die inside the arena with an
out-of-range index, far from the source that caused it.

What the earlier version of this list got wrong, since the same drift is easy to
repeat: `str`, arrays, slices, `Handle<T>`, `test` blocks, and diagnostics
carrying a file, line and column are all present and were listed as missing.

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

Each file states what it is about at the top and lists what it offers on one
`export` line, so the shape of the compiler is readable from the imports.

`core` holds the externs, the constant tables, the records and the arena.
`lexer` and `cursor` turn source into tokens and read them. `imports` lays every
file's text into one buffer. `names` interns and resolves. `types` does the
typing and the checks that ride on it. `parser` is recursive descent. `layout`
works out sizes and offsets, `emit` is what both backends write through, and
`emit_c` and `emit_asm` are the backends. `regions` is the region check, and
`frost` is the driver.

The import order is acyclic, and the assembly backend does not depend on the C
one, which is what `emit` is for.

## Where the measurements live

[docs/self-hosting.md](../docs/self-hosting.md) has the checklist in dependency
order, what each item cost, the compile-speed numbers, and the bugs that only
the assembly backend could have found. That last part is the useful bit: C needs
neither struct offsets nor a scope discipline of its own, so emitting assembly
surfaced three real bugs the C backend had been papering over.
