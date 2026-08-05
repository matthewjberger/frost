# 7. Statements

A block `{ Stmt* }` is a sequence of statements and is itself an expression whose
value is its trailing expression, when it has one.

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

  `for name in live_slots(c) Block` walks the slots of a generational
  container that hold an element, in slot order, and
  `for rank, name in live_slots(c) Block` counts them as it goes.
  `live_slots(c)` is the subject of a `for` and may be written nowhere else.
  See 10.1b.
- `break` and `continue` are loop control.
- `with`, `with name Block` names the allocation source every `uses` call inside
  the block draws from, and makes the block that source's region (8a).
- `defer`, `defer Stmt` runs `Stmt` where the function leaves, last deferred
  first. Only at the top level of a function body, and not run by `break` or
  `continue` (chapter 9.3).
- `errdefer`, `errdefer Stmt` runs `Stmt` where the function leaves through its
  failure set, and nowhere else. Same rule about the top level of a body, same
  exits, and one list with `defer`, so the two run in the order they were
  written, last first, whichever kind each is. A function with no failure set
  has no exit for one to name, so an `errdefer` in one is refused.

  What it is for is the resource a `?` steps over. `f := open()?` followed by
  `errdefer close(f)` says the failure path closes `f`; the straight-line path
  still owes a consumption, so the body's own `close(f)` is the first one rather
  than a second. An `errdefer` on its own does not answer for a resource, and a
  body that leaves with an answer without consuming it is the ordinary leak
  (chapter 9).

There is no print statement. Writing output is `import "io.frost"` and a call,
one writer per type ([text-and-io.md](../std/text-and-io.md)).
