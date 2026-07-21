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

- Integer variables `a` through `z`.
- Arithmetic (`+ - * / %`) with precedence, and parentheses.
- Comparisons (`< > <= >= == !=`).
- `let v = expr` and `v = expr`.
- `print expr`.
- `if (expr) { ... } else { ... }` and `while (expr) { ... }`.
- Statements separated by whitespace or `;`.

## How it works

1. **Lexing and parsing are fused** into a recursive-descent parser that reads
   source bytes directly through two tiny runtime helpers, `frost_byte_at` and
   `frost_str_len`. Source is held as a `^i8` pointer, which is a copy type, so
   it threads through the parser freely.
2. **The AST is a pool of `Node` records.** Each node names its children by
   their arena index, the data-oriented replacement for pointers between heap
   nodes. Sibling statements are linked through a `next` field. The pool gives a
   growable, zero-initialized arena with no manual allocation.
3. **The mutable parser cursor** lives in a `Parser` struct threaded by `&mut`
   through the recursive-descent functions, which is the second-class-reference
   way to carry mutable state without storing a borrow anywhere.
4. **Code generation** walks the arena, dispatching on `node.kind`, and emits C
   text through two runtime helpers (`frost_emit_str`, `frost_emit_int`).
   Variables become slots in a fixed C array, and Mini-Frost statements map
   directly to C statements.

## Building and running

```
frost --link -o minifrost bootstrap/minifrost.frost
./minifrost > out.c          # the Frost-written compiler emits C
cc out.c -o out && ./out     # compile the emitted C to native and run it
```

The embedded sample program computes the sum `1..10` (55), `5!` (120),
`fib(10)` by iteration (55), and takes an `if` branch (111), so the compiled
output prints:

```
55
120
55
111
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
programs end to end, including code emission. Widening the accepted language
(functions, more types, structs) toward the full surface is the path from here
to compiling Frost itself.
