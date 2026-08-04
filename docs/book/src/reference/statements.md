# 7. Statements

A block `{ Stmt* }` is a sequence of statements and is itself an expression whose
value is its trailing expression (or `void`).

- Expression statement, an expression evaluated for effect.
- Binding, the forms in 5.1.
- Assignment, `Place = Expr`, where `Place` is a `var` local, a field, an
  index, or a dereference.
- `return`, `return`, `return Expr`, or `return Expr ( "," Expr )+` in a
  function whose signature is a return type list (5.2a).
- `while`, `while ( Cond ) Block`.
- `for`, `for name in Expr Block` walks `name` over the value of `Expr`, which
  is a range, a slice `[]T`, a fixed array `[N]T`, or a `str` (yielding its
  bytes). `for index, name in Expr Block` names the position as well.

  Over a sequence this is the index-and-bound loop written out, not an iterator
  protocol: nothing is called per element, there is no trait to implement, and
  `break` and `continue` mean what they do in any other loop. The element binds
  the way a parameter of its type would, so an aggregate is borrowed and a
  scalar is copied. The sequence is evaluated once and its length read once
  before the first step, so a call in that position happens once and a body that
  appends to the same container does not walk what it just added.

  A name followed by `{` is a struct literal everywhere else, so the literal is
  not available in the `Expr` of a `for`, whose brace opens the body.
- `break` and `continue` are loop control.
- `with`, `with name Block` names the allocation source every `uses` call inside
  the block draws from, and makes the block that source's region (8a).
- `defer`, `defer Stmt` runs `Stmt` where the function leaves, last deferred
  first. Only at the top level of a function body, and not run by `break` or
  `continue` (chapter 9.3).

There is no print statement. Writing output is `import "io.frost"` and a call,
one writer per type ([text-and-io.md](../std/text-and-io.md)).
