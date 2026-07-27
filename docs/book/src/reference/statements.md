# 7. Statements

A block `{ Stmt* }` is a sequence of statements and is itself an expression whose
value is its trailing expression (or `void`).

- Expression statement, an expression evaluated for effect.
- Binding, the forms in 5.1.
- Assignment, `Place = Expr`, where `Place` is a `mut` local, a field, an
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
- `defer`, `defer Stmt` runs `Stmt` at scope exit, LIFO (chapter 9.3).
- `print Expr` writes one value and a newline to standard output: an integer as
  `%lld`, a float as `%g`, and a `str` or `^i8` as its bytes.
- `print STRING ( "," Expr )*` writes a line built from a format literal. Each
  `{}` in the literal is a hole filled by the value at the same position, and
  `{{` and `}}` are one brace each. The count of holes and the count of values
  have to agree.

  ```frost
  print "hp {} of {}", entity.hp, entity.max
  ```

  The literal is read by the compiler, which splits it into the pieces to write
  where the statement is written. No format exists at run time, nothing parses
  one, and the values are written by their types, so the printable set is closed
  and lives in the compiler: the integer widths, the floats, `bool`, `Handle`,
  `str` and `^i8`. Anything else is an error naming the type. This is the same
  arrangement as `print` being a statement keyword rather than a library
  function.
