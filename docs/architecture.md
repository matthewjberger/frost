# Frost Architecture

This document describes how the Frost compiler is structured today and the
direction it is moving. It is kept honest: it states what works, what is
partial, and what is not built yet.

## Pipeline

```
Source (.frost)
      |
      v
   Lexer            src/lexer.rs      -> tokens
      |
      v
   Parser           src/parser.rs     -> AST
      |
      +-----------------------------+
      |                             |
      v                             v
  Bytecode compiler            Typed IR
  src/compiler.rs              src/ir.rs, src/ir_build.rs
      |                             |
      v                             v
  Bytecode VM                  Native codegen
  src/typed_vm.rs              src/ir_codegen.rs (Cranelift)
      |                             |
      v                             v
  Interpreted run              Object file -> linked executable
```

The **bytecode VM** is the mature, default path used by the REPL and by
`frost file.frost`. It is broad and covered by a large unit-test suite.

The **typed IR** is the new spine for native compilation and, over time, the
single place where type checking and ownership/linearity checking will be
discharged. `--native` / `--link` lower the AST to the IR and emit machine
code from it via Cranelift; `--emit-c` lowers the same IR to portable C.
Because both native backends emit from one IR, a differential test compiles
each program through both and asserts their output matches.

## Typed IR

The IR (`src/ir.rs`) is a typed, CFG-based intermediate representation in the
spirit of a compiler "middle end" (MIR):

- A module is a set of functions and extern declarations.
- Each function has typed locals, a list of basic blocks, and an entry block.
- Each block is a sequence of statements ending in a terminator
  (`return`, `jump`, conditional `branch`, or `unreachable`).
- Values are explicit operands (a constant or a local). Every operand has a
  concrete type, so lowering never has to guess widths or signedness.
- Short-circuit `&&` / `||` and `if`/`else` expressions are lowered to
  explicit control flow, not special-cased in the backend.
- Address-taken locals are marked `in_memory`; the backend gives them stack
  slots. `&`, `&mut`, and `^` (dereference) lower to address-of, load, and
  store.

Lowering (`src/ir_build.rs`) folds light bidirectional type inference into the
translation so each value carries a real type. Anything outside the supported
subset fails loudly with a `native backend: ...` error rather than emitting
incorrect code.

## Native backends

`src/ir_codegen.rs` emits a relocatable object from the IR via Cranelift and
links it with the system C toolchain. `src/ir_c.rs` emits portable C from the
same IR (`--emit-c`), which the system C compiler builds. Both use the
correct type and operation for each value because the IR is fully typed, and
`tests/native.rs` checks that the two backends agree on every program.

**Working today**, verified by running native binaries (`tests/native.rs`):

- Integer arithmetic at every width with correct signedness, float
  arithmetic, bitwise and shift operators.
- Comparisons (signed / unsigned / float) and boolean logic with
  short-circuit evaluation.
- `if` / `else` expressions, `while`, `for`-over-range, `break`, `continue`.
- Functions, recursion, and direct calls.
- Sign / zero extension and truncation casts between integer widths, and
  integer/float conversions.
- `extern fn` C interop, including string-literal arguments with escape
  sequences (e.g. `puts`, `printf`).
- References and pointers: `&`, `&mut`, `^` dereference read/write, and
  pointer/reference parameters (e.g. `swap(a: ^i64, b: ^i64)`,
  `increment(x: &mut i64)`).
- Structs: layout with correct field alignment, construction, field read
  and write, references to structs and to fields, mutation through
  `&mut Struct`, whole-struct copy (`copy := p`), passing aggregates by
  value (copied at the call boundary so a callee's mutations do not affect
  the caller), and returning aggregates by value (via a hidden out-pointer),
  including passing an aggregate-returning call directly as an argument.
- Fixed-size arrays: array literals, indexed read and write with static or
  runtime indices, and references to arrays (e.g. `sum(a: &[5]i64)`). Every
  index is bounds-checked against the statically-known length; an out-of-range
  index aborts (see [memory-safety.md](memory-safety.md)).
- Enums and tagged unions: construction, and `match` over a value or a
  reference with enum-variant patterns (binding payload fields), integer
  literal patterns, identifier binding, and wildcard.
- Tuple patterns in `match` (e.g. `match (i % 3, i % 5) { case (0, 0): ... }`),
  with literal, wildcard, and identifier-binding sub-patterns.
- Function pointers: a function used as a value becomes its address, a
  `fn(...) -> T` parameter or local holds one, and calling through it is an
  indirect call. This is the design's "function pointers, not closures"
  higher-order story (`apply(f: fn(i64) -> i64, x: i64)`).
- `defer`: function-scoped, run in LIFO order at each return and at the
  trailing expression. A `return` nested inside a branch alongside `defer`
  is rejected (it would need runtime tracking), so defers always run.

### Generational handles and pools

A small C runtime (`runtime/frost_runtime.c`) provides generational pools
that both backends link automatically: `pool_new(capacity, elem_size)`,
`pool_alloc`, `pool_get`, `pool_free`, `pool_contains`, `handle_index`, and
`handle_generation`. This is the design's handle/pool proposition working
natively without a garbage collector.

The interface is deliberately scalar-only: a pool is an opaque pointer and a
handle is a packed `i64` (`generation << 32 | index`). Nothing is passed or
returned by aggregate value, so the runtime's natural C ABI matches Frost's
internal aggregate-return convention (a hidden out-pointer) with no ABI
negotiation — the identical compiled runtime links into both the Cranelift
and C backends, which is why they agree bit for bit.

The generational guarantee is what the differential test pins down: freeing a
slot bumps its generation, so a later allocation reuses the slot at a higher
generation and the original handle reports `pool_contains == 0`. A stale
handle can never silently read a live value. Storing the *handle* is fine (it
is plain copyable data, not a reference); the borrow you get by dereferencing
through the pool is what stays second-class.

Today the pool's memory management lives in the raw runtime, but the typed
surface over it is now an ordinary Frost library (see below), not a set of
privileged builtins.

### Generic functions, specialization, and sizeof

The native path monomorphizes generic functions. A function is generic when a
parameter is typed `$T`; it is kept out of normal lowering and specialized on
demand. At each call site the concrete substitution is inferred from the
argument types, a specialized name is mangled (`identity__i64`), and a
worklist drives specialization to fixpoint — so transitive generics and
multiple instantiations of the same function are all emitted once. Substitution
rewrites both `TypeParam(T)` and the bare `Struct("T")` the parser produces for
later uses of a type parameter. This works for generics over scalars, structs
by value, and references.

`sizeof(T)` lowers to a compile-time integer from the IR's layout, and because
substitution runs first, `sizeof(T)` inside a generic function becomes
`sizeof(Concrete)` and then a constant. When a type parameter can't be inferred
from a value argument, it is declared `$T: Type` and passed explicitly at the
call as `$Concrete` (type parameters are then erased from the specialized ABI).
Together these turn the pool into a Frost library:
`make_pool($T: Type, cap) -> ^u8` sizes itself with `sizeof(T)` and is called
`make_pool($Entity, 16)`, and `insert(pool, value: $T) -> Handle<T>` copies the
inferred element type in — no manual element size, no privileged builtin
(`examples/native/generic_pool_library.frost`).

Generic structs monomorphize the same way. `Foo<Args>` in type position is
encoded as a struct name that carries its arguments (`Pair<i64>`); because a
struct name is only a layout-registry key and aggregates are byte buffers, the
name never has to be a valid identifier. A pre-pass discovers every instance
used across signatures, fields, and bodies, substitutes the generic struct's
fields, and registers a concrete layout to fixpoint (so nested instances
resolve). Construction uses the annotated instance type. This works over
scalars and structs, with multiple type parameters, array fields of the
parameter, by-reference passing, and nesting inside other structs.

**Not yet in the native backend** (these fail loudly, they are not
silently miscompiled): slices, capturing closures (the design deliberately
uses function pointers instead), hashmaps, `comptime` blocks/loops, and
explicit type arguments (a type parameter is inferred from a value or borrow,
not passed as `f<T>(...)`). These run on the bytecode VM.

The emitted C is an internal detail, not an interface for external C callers,
so Frost function names are prefixed (`frost_`) to avoid C keyword clashes;
`extern` names and `main` are left untouched so FFI and the entry point link.
Frost-to-C interop (`extern fn`) works on both the Cranelift and C paths.

This replaces the previous AST-walking `codegen.rs`, which treated most
values as `i64`, hardcoded `if`-expression result types, resolved struct
field offsets by first-name match, and emitted `iconst 0` for anything it did
not handle.

## Direction

See [philosophy.md](philosophy.md) for the design philosophy, goals and
non-goals, and why Frost is data-oriented rather than object-oriented.

Frost is being reshaped toward a data-oriented language with:

- Plain data (copy/move), **linear resources** that must be consumed exactly
  once, and generational **handles** into explicit pools.
- Second-class references: borrows exist only as parameter modes and
  dereference-scoped temporaries; they cannot be stored or returned.
- Free functions only, with signatures that declare their effects.
- The typed IR as the single point where ownership, borrow, and linearity
  checking are discharged, with the bytecode VM serving as the differential
  oracle for the native backends.

## Ownership checking

`src/ownership.rs` runs after parsing and enforces two rules:

- **Second-class references.** A reference (`&T` / `&mut T`) cannot be stored
  in a struct or enum field, and cannot be returned from a function or
  extern. Reference *parameters* are allowed, and `Handle<T>` (a generational
  index, not a reference) can be stored and returned freely. Because
  references cannot escape, borrow analysis stays scope-local.
- **Borrow exclusivity.** A `&mut` borrow is exclusive: a variable cannot be
  borrowed as mutable more than once, or as both shared and mutable, within a
  single call. Multiple shared `&` borrows are fine. Because references are
  second-class, this per-call check is enough to keep mutable aliasing out.
- **Move checking.** Per function body, a value of a move type (a struct,
  enum, string, or slice — anything not `Copy`) is consumed when it is passed
  by value, assigned, or returned; using it again is a use-after-move error.
  Borrowing (`&x`), field access (`x.f`), and dereference do not consume, and
  copy types (integers, floats, bools, pointers, references, handles) are
  never moved.
- **Linear resources.** A struct or enum declared `linear`
  (`File :: linear struct { ... }`) is a resource that must be consumed
  exactly once: the move checker's use-after-move rule gives "at most once",
  and a linear value that is still live at the end of the function that owns
  it is a "never consumed" error. Consuming means moving it onward — returning
  it, passing it by value to another function (the terminal consumer is
  typically an `extern`, which takes ownership across the FFI boundary), or
  `match`ing it (a `match` on a linear value destructures and consumes it).
  This is how the design replaces `Drop`: cleanup is an obligation the type
  system tracks rather than an implicit call. It also makes a linear error
  enum non-ignorable — a `linear enum` returned from a fallible function must
  be matched (or otherwise consumed), so a failure cannot be silently dropped.

### Roadmap

1. Extend ownership to move tracking and borrow exclusivity on the IR
   (second-class references are already enforced).
2. A real type-checking pass on the IR, replacing the currently bypassed
   `typechecker.rs`.
3. Linear resources with path-sensitive consumption, and error enums that
   linearity makes non-ignorable.
4. Handle-dereference-as-borrow. *(Done: `Handle<T>` is a first-class native
   type (a packed i64), and `pool[handle]` is a place — read/write fields
   through it, copy the element out, or take `&`/`&mut` of it. The borrow is
   second-class for free: storing it in a field or returning it is already
   rejected, so a handle-deref borrow cannot escape.)*
5. Struct/array/enum by-value passing and tuple patterns in the native
   backend. *(Done: all three, plus nested aggregates and arrays of structs.)*
6. Generics and specialization-only comptime (monomorphization). *(Done:
   generic functions, generic structs (incl. nested `Pair<Pair<i64>>`, factory
   functions returning instances, construction inference, and generic-over-
   instance), `sizeof`, and explicit type arguments (`fn($T: Type, ...)` called
   `f($Concrete, ...)`, with type parameters erased from the specialized ABI);
   the pool typed surface is now a Frost library.)*
7. Bounds-checked array indexing. *(Done: every fixed-size array index is
   checked against the statically-known length and aborts on out-of-range.)*
8. Source locations in errors. *(Done for the lexer and parser: errors carry
   `line`/`column`. Remaining: spans on AST nodes so ownership and IR-lowering
   errors are located too — a large mechanical change of modest value now that
   syntax errors are located.)*
9. Eventual self-hosting of the compiler in Frost.
