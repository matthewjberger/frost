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
  and write, references to structs and to fields, and mutation through
  `&mut Struct` (e.g. `scale(p: &mut Point, factor: i64)`).
- Fixed-size arrays: array literals, indexed read and write with static or
  runtime indices, and references to arrays (e.g. `sum(a: &[5]i64)`).
- Enums and tagged unions: construction, and `match` over a value or a
  reference with enum-variant patterns (binding payload fields), integer
  literal patterns, identifier binding, and wildcard.

**Not yet in the native backend** (these fail loudly, they are not
silently miscompiled): passing or returning structs, arrays, and enums by
value; tuple patterns in `match`; slices, closures, hashmaps, `defer`,
`comptime`, and generics. These run on the bytecode VM.

This replaces the previous AST-walking `codegen.rs`, which treated most
values as `i64`, hardcoded `if`-expression result types, resolved struct
field offsets by first-name match, and emitted `iconst 0` for anything it did
not handle.

## Direction

Frost is being reshaped toward a data-oriented language with:

- Plain data (copy/move), **linear resources** that must be consumed exactly
  once, and generational **handles** into explicit pools.
- Second-class references: borrows exist only as parameter modes and
  dereference-scoped temporaries; they cannot be stored or returned.
- Free functions only, with signatures that declare their effects.
- The typed IR as the single point where ownership, borrow, and linearity
  checking are discharged, with the bytecode VM serving as the differential
  oracle for the native backends.

### Roadmap

1. Structs, enums / tagged unions and `match`, and arrays in the IR and
   native backend.
2. A real type-checking pass on the IR, replacing the currently bypassed
   `typechecker.rs`.
3. Ownership: move tracking, borrow exclusivity, and second-class reference
   enforcement on the IR.
4. Linear resources with path-sensitive consumption, and error enums that
   linearity makes non-ignorable.
5. Handle-dereference-as-borrow, unifying pool handles with the region
   checker.
6. A C backend emitting from the same IR, plus differential testing across
   the VM, Cranelift, and C.
7. Eventual self-hosting of the compiler in Frost.
