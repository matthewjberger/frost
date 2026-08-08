# 6. Expressions

## 6.1 Primary expressions

The primary expressions are integer, float, string, and boolean literals,
identifiers, parenthesized expressions `( Expr )`, and array literals, either the
listed form `[ e, ... ]` or the repeat form `[ e ; N ]` for `N` copies of `e`
(the way a large or zeroed backing buffer is written, e.g. `[0; 256]`). The
count is an integer, a constant, or a value parameter of the generic the literal
is written in (11.1a).

A call may go through a value rather than a name. A parameter, a binding, or a
struct field of function-pointer type is called by writing the call on it:

```frost,sketch
System :: struct { run: fn(mut World), stage: i64 }

systems[index].run(world)
held := systems[index].run
held(world)
```

The parameters of an indirect call travel exactly as the signature writes them:
`fn(mut World)` borrows, so the world is not consumed by being handed to one.

## 6.2 Operators

Prefix `-` (negate) and `!` (logical not). Binary operators, grouped by the
precedence in 14.1, are `||`, `&&`, `==` `!=`, `<` `<=` `>` `>=`, `|`, `&`, `<<`
`>>`, `+` `-`, `*` `/` `%`, and the range operators `..` and `..=`. All binary
operators are left-associative.

`!` applies to `bool` and answers the opposite of it. Given anything else it is
refused, naming the type it was given. `x == false` says the same as `!x`, and
both are available.

## 6.3 References and dereference

- `ptr_to(place)` the address of a place. There is no borrow operator: a
  borrow is what a parameter mode means, inserted at the call.
- `expr^` dereferences a raw pointer to its pointee value and is assignable
  (`p^ = v`). Member access through a raw pointer is written `p^.field`.
- A borrowed parameter needs no dereference at all, whatever its type. `p.field`
  reads and writes a field of a borrowed aggregate, and `p = q` on a `mut`
  parameter assigns the whole value through the borrow rather than rebinding
  anything local. Naming a `mut` parameter always means the caller's value.
- Binding a parameter to a name (`x := p`) binds a *copy* of what it holds, so
  writing through `x` does not reach back to the caller. A second name for the
  same place is `ref x := p`, and a call answering with a `ref T` (3.3) hands
  one out. A parameter of a `linear` type cannot be bound this way at all, since
  a copy would be a second owner of something consumed exactly once.

## 6.4 Calls, indexing, and field access

- `f(a, b, ...)` calls a function or function pointer.
- `a[i]` indexes an array, slice, or pool (for a pool, `i` is a `Handle`).
- `e.field` accesses a struct field or, on an enum place, a variant field.

## 6.5 Construction

```
Point { x = 1, y = 2 }                // struct literal (fields use =)
Shape::Circle { radius = 5 }          // enum variant with payload
Shape::Player                         // unit variant
.Circle { radius = 5 }                // the enum comes from the context
.Player                               // the same, with no payload
{ x = 1, y = 2 }                      // the struct comes from the context
```

Struct and enum-variant construction are recognized only when the operand to the
left of `{` or `::` is a bare identifier.

`.Variant` names a variant without naming its enum, and takes the enum from the
type the surrounding code expects. It is the construction counterpart of the
`case .Variant` a pattern writes (6.7).

The contexts that supply a type are the ones that state it:

| Context | What supplies the enum |
| --- | --- |
| `c : Color = .Red` | the annotation |
| `paint(.Red)` | the parameter's declared type |
| `Theme { primary = .Red }` | the field's declared type |
| `return .Circle { radius = r }` | the function's declared return type |
| `c = .Blue` | the type of the place assigned to |
| `wheel : [3]Color = [.Red, .Green, .Blue]` | the array's element type |

In a function with a failure set the return is two types, so `.Denied` names a
variant of the failure set when it has one and a variant of the value type
otherwise. That is how `return .Denied` fails and `return .Some { value = 3 }`
succeeds in the same function.

A dot with nothing to take its enum from is an error naming the variant.
`c := .Red` has no annotation and no context, so it is rejected and the fix is
`c : Color = .Red` or `c := Color::Red`.

`{ x = 1, y = 2 }` is a struct literal that leaves out a type name the context
already carries. It reads from the same contexts the leading dot does, and the
two nest: a literal's field supplies the type of a literal written inside it, so
`{ from = { x = 0, y = 0 }, to = { x = 3, y = 4 } }` and
`{ at = { x = 7, y = 0 }, colour = .Green }` both resolve all the way down.

Every field is still named. There is no positional literal, here or anywhere
else in the language. A field's name is what says where the value lands, and
the layout that shows the reader is goal 1 of
[philosophy.md](../design/philosophy.md).

A literal with nothing to take its type from is an error, the same as a bare
dot. `p := { x = 1, y = 2 }` is rejected and the fix is `p : Point = { .. }` or
`p := Point { .. }`.

Either form written as an argument to a function declared later in the program
resolves once every signature is parsed, fields and all.

A literal must write every field. There is no partial construction, no
`..rest`, and no implicit zero. A missing field is an error that names it.

## 6.6 `if` expression

```
if ( Cond ) Block
if ( Cond ) Block else Block
```

The condition is parenthesized. `if` is an expression. Both arms are blocks and
their trailing expressions are the value.

An `if` answers with a value when both of its arms do. An arm whose block ends
in a statement answers with nothing, and then the whole `if` answers with
nothing however the other arm ended, so

```frost,sketch
if (queued) { spawn(world) } else { report(world) }
```

is an ordinary statement. An `if` with no `else` never answers with a value,
since the path where the condition was false has none to give.

## 6.7 `match` expression

```
match Scrutinee {
    case Pattern : Expr
    case Pattern : Block
    ...
}
```

An arm is `case`, a pattern, `:`, then an expression or block. A `{` after the
colon opens a block, so an arm that answers with an unnamed struct literal names
its type. There is no separator between arms. An arm ends where the next `case`
or the closing `}` begins. Patterns:

- Variant, shorthand, `.Variant` or `.Variant { field, field }`, binding each
  named field to a same-named local.
- Variant, qualified, `Enum::Variant` with the same optional field list.
- Value, a whole number or a boolean (`case 90:`, `case true:`), or a name a
  `::` declaration settled on a whole number (`case CH_0:`).
- Range, `a..b` half-open or `a..=b` inclusive, over whole numbers.
- Tuple, `( P, P, ... )`.
- Wildcard, `_`.

A name in a pattern is the value it stands for, and a name that stands for no
constant is refused. `_` is the arm that covers the rest, and the only one.

What a `case` covers is a set a reader can count, so a decimal and a piece of
text are both refused. A decimal covers one of the reals, and text is compared
rather than counted. `if (x == 1.5)` and `if (x == "hi")` are the spellings.

An arm may name several patterns separated by `|`, and its body runs for any of
them:

```frost
Step :: enum { Left, Right, Up, Down }

sideways :: fn(k: Step) -> i64 {
    match k {
        case .Left | .Right: 1
        case .Up: 2
        case _: 0
    }
}
```

What such an arm covers is the union of its alternatives, so the rule that
every variant is covered goes on counting. Three shapes may not be an
alternative. A variant pattern binding payload fields may not, because two
variants hold two shapes and a name reading a field out of them would mean two
things. Such a pattern takes an arm of its own. `_` and a bare identifier may
not, because each already covers everything.

A range arm covers a span of whole numbers, with the two spellings meaning what
they do after `in` (6.9). Both ends are whole numbers, written out or named by
a `::` declaration, which is the one position where a name in a pattern stands
for a value rather than binding what was matched:

```frost
CH_0 :: 48
CH_9 :: 57
CH_UPPER_A :: 65
CH_UPPER_Z :: 90
CH_LOWER_A :: 97
CH_LOWER_Z :: 122

kind_of :: fn(c: i64) -> i64 {
    match c {
        case CH_LOWER_A..=CH_LOWER_Z | CH_UPPER_A..=CH_UPPER_Z: 1
        case CH_0..=CH_9: 2
        case 0 | 5..10: 3
        case _: 0
    }
}
```

A range never removes the need for a `case _`. Proving that a run of spans
leaves no whole number out is analysis this language does not carry, so the arm
naming the rest is what says the match is finished.

An arm every value of which the arms above it already take is refused where it
is written. What an arm covers is the union of what its alternatives name, read
against the union of every arm above it: `case 1..5:`, `case 5..10:`,
`case 3..7:` refuses the third, because between them the first two take every
value it has. Since `_` covers everything, an arm below one is refused by that
same rule.

An alternative and a range are both refused inside a tuple pattern, which
compares one value per part.

`match` works over a value or a reference. Matching a value of a `linear` type
consumes it (chapter 9). An arm consumes it once however many patterns the arm
names, since the alternatives are one arm.

## 6.8 `sizeof`, `cast`, `typename`, and `unsafe`

`sizeof`, `typename` and `type_id` are builtin names, recognized where one is
called with a type argument the same way `ptr_to` and `cast` are recognized at
a call, and each stays usable as an ordinary identifier elsewhere.

- `sizeof(T)` is a compile-time constant.
- `cast($T, value)` converts a scalar to `T` where the conversion loses
  something (3.1a). It is safe and needs no block.
- `type_id(T)` is a number standing for the type, and `typename(T)` is its name
  as the source spells it, as a `str`. Both are compile-time constants, fixed
  where the type is known, so a generic asked inside its own body answers with
  what it was instantiated with rather than with the name of its parameter. A
  distinct type answers with its own name, not its representation's.
- `unsafe { ... }` is a block, and it is the only place four operations may be
  written: reading or writing through a raw pointer, `ptr_cast`, `slice_from`,
  and calling an `extern fn` that is not marked `safe`. Outside one each is a
  compile error. Chapter 6a is the rule, the list, and `--audit-unsafe`.

The block's value is its trailing expression, the same as any other block, so a
gated operation's result leaves one as `p := unsafe { ptr_cast($T, slot) }`.

## 6.9 Ranges

`a..b` is half-open, `a..=b` inclusive. Ranges appear in `for` and are the
lowest-binding binary form.
