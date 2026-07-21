# Mini-Frost: a compiler written in Frost

`minifrost.frost` is a small compiler, written in Frost, for a Frost-like
subset language. It is the self-hosting milestone: a non-trivial, compiler-
shaped program written entirely in the data-oriented native surface, compiled
to native code by the Frost compiler, and exercised by the test suite.

It is deliberately scoped to what the native language expresses today, and it is
written the way the language wants you to write a compiler: a pool-backed arena
for the AST, integer indices instead of heap pointers between nodes, fixed
arrays for the variable environment, and free functions over that data. There
are no closures and no dynamic collections, and every reference is second-class.

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
4. **Evaluation** walks the arena, dispatching on `node.kind` with `match`, and
   keeps variables in a fixed `[26]i64` environment.

## Building and running

```
frost --link -o minifrost bootstrap/minifrost.frost
./minifrost
```

The embedded sample program computes the sum `1..10` (55), `5!` (120),
`fib(10)` by iteration (55), and takes an `if` branch (111), so a correct run
prints:

```
55
120
55
111
```

The same program produces identical output through the portable C backend
(`--emit-c`). The IR interpreter declines it by design, because it uses pools
and pointers, which the interpreter does not model. The
`bootstrap_minifrost_compiler_runs` test in `tests/native.rs` compiles and runs
it on every build.

## Scope and what is next

This is the front-to-evaluation pipeline of a compiler (source to AST to result)
written in Frost. It is not yet a Frost compiler emitting native code from
Frost; that would additionally require a code generator written in Frost, which
is a larger undertaking. What it does establish is that the data-oriented native
language is expressive enough to write real compiler-shaped programs: arena
allocation, tree building and walking, and recursive descent all work in the
intended style. Extending the accepted language (functions, more types) and,
eventually, adding a backend are the natural next steps.
