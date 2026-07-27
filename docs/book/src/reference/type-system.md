# 4. The type system

Frost is statically typed with light local inference. Every binding, parameter,
and expression has a compile-time type.

- `:=` infers a local's type from its initializer. `:` gives it explicitly.
- Function parameter and return types are always explicit.
- A binary operation requires compatible operand types. Integer widths widen to
  a common width, and a comparison yields `bool`.
- A call requires the argument count and types to match the signature. Passing
  an aggregate where a reference is expected, or the reverse, is an error.

Type checking runs on the typed intermediate representation after lowering
(`src/ir_typecheck.rs`). It validates operand types, call arity, and that a
non-`void` function returns a value on every path.
