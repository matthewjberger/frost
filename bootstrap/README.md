# Mini-Frost: a compiler written in Frost

`minifrost.frost` is a small compiler, written in Frost, for a Frost-like
subset language. It reads a Mini-Frost program, builds an AST, and emits a C
translation unit; compiling and running that C reproduces the program's output.
So it is a real code generator written in Frost that emits native code, the same
way Frost's own primary backend emits C and links it.

It is deliberately scoped to what the native language expresses today, and it is
written the way the language wants you to write a compiler: a pool-backed arena
for the AST, integer indices instead of heap pointers between nodes, and free
functions over that data. There are no closures and no dynamic collections, and
every reference is second-class.

## The language it compiles

- Functions `fn name(p, q) { ... }`, with multi-character names and parameters,
  called as `name(args)` inside any expression, including recursively.
- `return expr`.
- The function named `main` is the entry point and becomes C `main`.
- Multi-character integer variables, local to each function.
- Arithmetic (`+ - * / %`) with precedence, and parentheses.
- Comparisons (`< > <= >= == !=`).
- `let name = expr` and `name = expr`.
- `print expr`.
- `if (expr) { ... } else { ... }` and `while (expr) { ... }`.
- Statements separated by whitespace or `;`.

Identifiers are lowercase letter runs (`sum`, `count`, `fib`), no longer limited
to single letters.

## How it works

1. **Lexing and parsing are fused** into a recursive-descent parser that reads
   source bytes directly through two tiny runtime helpers, `frost_byte_at` and
   `frost_str_len`. Source is held as a `^i8` pointer, which is a copy type, so
   it threads through the parser freely.
2. **The AST is a pool of `Node` records.** Each node names its children by
   their arena index, the data-oriented replacement for pointers between heap
   nodes. Sibling statements, function parameters, and call arguments are each
   linked through a `next` field into their own singly linked lists. The pool
   gives a growable, zero-initialized arena with no manual allocation.
3. **Names are interned through two symbol tables.** An identifier occurrence is
   a byte range into the source (offset plus length). Because the language has no
   string type, the tables intern those ranges to small integer indices by
   comparing bytes directly, the data-oriented stand-in for a hash map keyed by
   strings. Each function has its own local table, so its variables become
   `v[0]`, `v[1]`, ... in that function's frame, and a single global table maps
   function names to the `mf_<index>` of their emitted C functions. The tables
   live in their own pools, so no fixed-size array has to be sized in advance.
4. **The mutable parser cursor** lives in a `Parser` struct threaded by `&mut`
   through the recursive-descent functions, which is the second-class-reference
   way to carry mutable state without storing a borrow anywhere.
5. **Code generation** walks the arena, dispatching on `node.kind`, and emits C
   text through two runtime helpers (`frost_emit_str`, `frost_emit_int`). Each
   Mini-Frost function becomes a C function that takes `int64_t` parameters and
   holds its variables in its own fixed `int64_t v[64]`, indexed by the local
   symbol table, so calls get a fresh frame and recursion works. Prototypes are
   emitted ahead of the definitions, so functions may call each other in any
   order. The function `main` is emitted as C `main`.

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
