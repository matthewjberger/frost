# Mini-Frost: a compiler written in Frost

`minifrost.frost` is a small compiler, written in Frost, for a Frost-like
subset language. It reads a Mini-Frost program, builds an AST, and emits a C
translation unit; compiling and running that C reproduces the program's output.
So it is a real code generator written in Frost that emits native code, the same
way Frost's own primary backend emits C and links it.

It is deliberately scoped to what the native language expresses today, and it is
written the way the language wants you to write a compiler: a Frost-native arena
for the AST, integer indices instead of heap pointers between nodes, and free
functions over that data. There are no closures and no dynamic collections, and
every reference is second-class. Its memory comes from the language's own
allocator, not a runtime pool.

## The language it compiles

A subset of Frost's own syntax.

- Definitions `name :: fn(p: i64, q: i64) -> i64 { ... }`, called as `name(args)`
  inside any expression, including recursively. Parameter and return types are
  parsed (everything is `i64` today) and every function lowers to an `int64_t` C
  function.
- The function named `main` is the entry point and becomes C `main`.
- Locals `x := expr` and `mut x : i64 = expr`, reassignment `x = expr`. Each
  local becomes a named C variable of its own.
- `return expr`.
- Arithmetic (`+ - * / %`) with precedence, and parentheses.
- Comparisons (`< > <= >= == !=`) and boolean `&&` and `||`.
- `print expr` (a temporary convenience, to be replaced by `extern` calls once
  those are supported).
- `if (expr) { ... } else { ... }` and `while (expr) { ... }`.

Identifiers are letter runs that may include underscores, uppercase letters, and
digits after the first character (`sum`, `count`, `fib`, `is_main`, `Node`).

## How it works

1. **A lexer runs first,** turning the source into a flat `Arena<Token>` in one
   pass through `frost_byte_at`. It drops whitespace and `//` comments, reads
   integer and string literals, classifies identifier runs that spell a keyword
   to their own token kind, and lexes every operator and punctuation mark
   greedily (so `::`, `:=`, `->`, `<=`, `&&` are single tokens). Each token keeps
   the byte range it came from, which is how identifiers stay interned by range.
   The parser then reads token kinds rather than raw bytes.
2. **The AST lives in a Frost-native arena of `Node` records.** The arena is a
   generic `Arena<$T>`, a bump allocator over one `malloc`, written in the
   language itself (`docs/allocators.md`): `arena_push` appends a node and
   returns its index, `arena_at` turns an index back into a `^Node`. A node names
   its children by that index, the data-oriented replacement for pointers between
   heap nodes, and sibling statements, parameters, and call arguments are threaded
   through a `next` field into singly linked lists. No runtime pool is involved.
3. **Locals are named; function names are interned.** An identifier occurrence
   is a byte range into the source (offset plus length). A local reference stores
   that range on its AST node, and code generation re-emits the bytes, so a local
   becomes a named C variable and no local symbol table is needed. Function names
   go through one global table that interns each range to a small integer, and a
   function is emitted as `mf_<index>`, which gives a stable name and keeps user
   names from colliding with the C runtime.
4. **The mutable parser cursor** lives in a `Parser` struct threaded by `&mut`
   through the recursive-descent functions, which is the second-class-reference
   way to carry mutable state without storing a borrow anywhere.
5. **Code generation** walks the arena, dispatching on `node.kind`, and emits C
   text through the runtime helpers (`frost_emit_str`, `frost_emit_int`, and
   `frost_emit_char` for identifier bytes). Each function becomes a C function
   taking named `int64_t` parameters, and its locals are declared where they are
   bound, so recursion just works with no explicit frame. Prototypes are emitted
   ahead of the definitions, so functions may call each other in any order. The
   function `main` is emitted as C `main`.

## Building and running

```
frost --link -o minifrost bootstrap/minifrost.frost
./minifrost > out.c          # the Frost-written compiler emits C
cc out.c -o out && ./out     # compile the emitted C to native and run it
```

The embedded sample program defines a recursive `fib` and a recursive `fact`,
then a `main` that prints `fib(10)` (55), `fact(5)` (120), and the sum `1..10`
(55) accumulated in variables named `sum` and `index`, so the compiled output
prints:

```
55
120
55
```

The compiler itself produces identical C through both of Frost's backends
(`--link` and `--emit-c`), which the differential test checks. The
`bootstrap_minifrost_emits_working_c` test in `tests/native.rs` runs the whole
pipeline on every build: it compiles the Frost-written compiler, runs it to emit
C, compiles that C, runs the result, and checks the output.

## Scope and what is next

This is a working code generator written in Frost: source to AST to emitted C to
a native executable. It is not a full self-hosting compiler, because it accepts a
small subset rather than all of Frost. What it establishes is that the
data-oriented native language is expressive enough to write real compiler-shaped
programs end to end, including functions, recursion, multi-character names
resolved through symbol tables, and code emission. Widening the accepted language
further (more types, structs, expressions as statements) toward the full surface
is the path from here to compiling Frost itself.
