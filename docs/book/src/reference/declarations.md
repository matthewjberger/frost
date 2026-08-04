# 5. Declarations and bindings

A program is a sequence of statements. The top-level meaningful statements are
declarations. These are constants (including functions, structs, and enums),
externs, and imports.

## 5.1 Binding forms

| Form | Meaning |
| --- | --- |
| `name := expr` | bind a local, type inferred |
| `name : Type = expr` | bind a local, type explicit |
| `mut name := expr` / `mut name : Type = expr` | bind a mutable local |
| `a, b := call()` | bind the several values of one call (5.2a) |
| `ref name := place` | bind a borrow of a place, not a copy of it |
| `NAME :: expr` | declare a constant, evaluated once |

Bindings are immutable unless `mut`. A `mut` local is reassigned with `=`.

`ref name := place` binds a borrow rather than a copy, so writing through the
name writes the place: `ref entry := table[index]` then `entry.count = 0` is a
write to the element. The right-hand side has to be a place (a name, a field, an
index, a dereference, or a call answering with a `ref T`), since there is nothing
to borrow otherwise. A `ref` binding is a borrow like any other: it may not be
stored, and it may not outlive what it names (chapter 8).

The parser distinguishes `name : Type = ...` (a typed binding) from
`name :: ...` (a constant) by one token of lookahead after the first `:`. A
second `:` means a constant.

## 5.2 Constants and items

`::` declares a constant. Functions, structs, enums, and type aliases are all
constants:

```
NAME  :: <expr>                       // value constant
f     :: fn(params) -> R { body }     // function
Point :: struct { x: i64, y: i64 }    // struct
Shape :: enum { A, B { n: i64 } }     // enum
Meters :: distinct i64                // distinct type
```

A value constant's `<expr>` is an integer constant expression: an integer
literal, an earlier constant by name, or those combined with the integer
operators `+ - * / % << >> & |` and parentheses, folded to a value at compile
time. So `STRIDE :: POSITION + NORMAL + UV` and `MASK :: 1 << FLAG_BIT` name a
computed value, and the result may be used where a compile-time integer is
required, such as an array length `[STRIDE]f32`. A constant whose value is a
single other name (`X :: Y`) is not written, since `Enum::Variant` shares that
shape. A constant that refers to one name combines it with an operator.

A `linear` qualifier may precede `struct` or `enum` (chapter 9).

An `inline` qualifier may precede `fn` (`f :: inline fn(...) -> R { ... }`). It
asks the C backend to force the function inline (`static inline
__attribute__((always_inline))`). The assembly backend, which does not inline,
ignores it. It changes no semantics, only whether the C compiler is obliged to
fold the call rather than merely permitted to.

## 5.2a Multiple return values

A function returns several values by declaring a return type list, and the
caller binds them by name:

```
divide :: fn(a: i64, b: i64) -> (quotient: i64, remainder: i64) {
    return a / b, a % b
}

quotient, remainder := divide(17, 5)      // 3 and 2
```

The list holds two or more values. `-> T` is how one value is returned and
`-> (T)` is an error that says so.

**Naming the values.** A value may be written `name: Type`, and a list names all
of its values or none of them:

```
divide :: fn(a: i64, b: i64) -> (quotient: i64, remainder: i64)   // named
divide :: fn(a: i64, b: i64) -> (i64, i64)                        // not
```

A name says which value is which at the definition, which is the whole reason to
have one, and it is also the field name a `return` can write:

```
split :: fn(value: i64) -> (high: i64, low: i64) {
    return { high = value / 256, low = value % 256 }
}
```

That is the inferred literal of 6.5 over the struct the list becomes, so the
`return` reads the way the signature does and cannot silently swap two values of
the same type. A `return` by name in a function whose list names nothing is an
error, since there would be no field names to write. Two names the same in one
list is also an error.

A name is not a variable. It is not in scope in the body, there is nothing to
assign to it, and there is no bare `return` that hands back whatever the names
hold. That is Go's naked return and Odin's implicit result, and it is the hidden
control flow [philosophy.md](../design/philosophy.md) rules out: what a function answers
with is written at the `return` that answers.

`return` lists the values in order, or names them, and it is required either
way: a trailing expression is one value, so a function with a return type list
ends every path with a `return`. A `return` that lists a different number of
values than the list has is a compile error, as is one that lists several values
in a function that returns one.

`mut` goes in front of any name the body goes on to write:

```
magnitude, mut negative := classify(value)
negative = false
```

**There is no tuple type.** The list is not a value, cannot be named, stored in
a field, passed as an argument, or returned from anything but the function that
declares it, and `(A, B)` is not a type anywhere else in the grammar. A call
that returns several values is bound by a list of names and used nowhere else.
Binding it to a single name is a compile error that says so. That restriction is
what keeps the layout of every value in a program something the reader named
(goal 1 of [philosophy.md](../design/philosophy.md)): a program that wants to pass a pair
around declares a struct and gets a name for it.

What the compiler does with the list is give it one struct, whose fields are the
names the signature gave, or `value0`, `value1` and so on in order when it gave
none. The signature becomes a plain return of
that struct, the `return` becomes a literal of it, and the binding becomes the
call bound to a temporary and one field read per name. Nothing after the front
end sees a return type list, which is why every backend and the C ABI handle one
with no code of their own. In the bootstrap compiler two functions returning
the same list under the same names share the struct, since its name is derived
from both. The self-hosted compiler makes one per function. Neither is
observable.

A return type list does not combine with a failure set: `-> (A, B) ! E` is
rejected, because a fallible function answers with one value or one error. A
function that wants both returns a struct it names. A return type list on a
function with a compile-time parameter is rejected for the same reason it is one
struct rather than one per specialization.

## 5.2b Failure sets

A function that can fail says so in its signature, with the type it fails with:

```frost
Parse :: struct { at: i64, code: i64 }

digit :: fn(text: str, index: i64) -> i64 ! Parse {
    byte := text[index]
    if (byte < 48 || byte > 57) {
        return { at = index, code = byte }
    }
    byte - 48
}
```

`-> T ! E` reads "answers with a T, or fails with an E". `E` is a type the
program declares, a struct or an enum, and nothing about it is built in: there
is no error interface to implement, no backtrace, no allocation, and no boxing.
A failure is a value of a type you wrote, so what a failure carries is what you
put in it.

**What the compiler makes of it.** The signature becomes one enum with two
variants:

```frost
Result :: enum { Ok { value: T }, Err { error: E } }
```

That enum is what the function returns, and it is where the names at a `match`
come from: `value` is the field the `Ok` variant carries and `error` is the
field `Err` carries. A field like `error.at` is a field of `Parse`, the type
this program declared. Nothing downstream knows failure sets exist, which is
why every backend and the C ABI handle a fallible function with no code of
their own.

**Returning one or the other.** A `return` whose expression builds the failure
type is the failure, and anything else is the value:

- `return Parse { at = 3, code = 0 }` names the failure type, and is the
  failure.
- `return .Denied` names no type, and is the failure when the failure type is
  an enum with that variant.
- `return { at = 3, code = 0 }` names no type either, and is the failure when
  the value type is not itself a struct or an enum, since then only the failure
  can be written that way. When both are aggregates, name the one you mean.
- The body's trailing expression is the value, the same as in any other
  function.

**`?` hands a failure up.** A call followed by `?` is the value it answered
with, or an immediate return of its failure from the function the `?` is
written in:

```frost
number :: fn(text: str) -> i64 ! Parse {
    mut total : i64 = 0
    mut index : i64 = 0
    while (index < str_len(text)) {
        d := digit(text, index)?
        total = total * 10 + d
        index = index + 1
    }
    total
}
```

`?` is only allowed in a function that declares a failure set, since it has
nowhere to hand a failure to otherwise, and the two failure types have to be
the same one. There is no conversion, and no `From` to write: a function that
calls something failing differently `match`es it and returns the failure it
declares.

**Reading the answer.** The caller matches, and a match on an enum covers every
variant (6.7):

```frost
match number(text) {
    case .Ok { value }: { print_int_line(value) }
    case .Err { error }: { print_int_line(error.at) }
}
```

**A resource survives a failure.** A result carrying a `linear` value is itself
linear, so it must be consumed, and matching it is what consumes it (chapter 9).
Ignoring a call that answers with one is refused, since the resource would be
dropped where nothing named it.

```frost
open :: fn(n: i64) -> File ! Denied { ... }

use_it :: fn(n: i64) -> i64 {
    match open(n) {
        case .Ok { value }: close(value)     // consumes the File
        case .Err { error }: error.code
    }
}
```

**There is no other error channel.** No exceptions, no panics to catch, no
error return codes to check by convention, and no ignoring a failure by
accident: what a function can fail with is in its signature, and the caller
either matches it or hands it up with `?`. A failure that is not one, a bug in
the program rather than a condition in the world, is an assertion, and an
assertion aborts.

`-> (A, B) ! E` is rejected: a fallible function answers with one value or one
failure. A function that wants both returns a struct it names.

## 5.3 Externs and imports

```
name :: extern fn(params) -> R        // foreign function (chapter 12)
name :: extern fn(params)             // foreign function returning void
import "path"                         // bring another source file into scope
```

## 5.4 Tests

A `test` block declares a named unit test.

```
test "name" { Stmt* }
```

`test` is a contextual keyword, recognized only when followed by a string
literal and a block, so it remains usable as an ordinary identifier elsewhere.
Inside a test, `assert(cond)` aborts the test when `cond` is false. `frost --test
file.frost` compiles the file, runs each test in declaration order, and exits
non-zero if any assertion fails.

## 5.5 Modules and exports

A source file is a module. `import "path"` brings another file in, resolved
relative to the importing file, and each file is pulled in once even through a
diamond of imports.

Top-level items are private to their file by default. A file lists what it offers
with an `export` line at the top.

```frost
export area, Shape

Shape :: enum { Circle { r: i64 }, Rect { w: i64, h: i64 } }
area :: fn(s: Shape) -> i64 { ... }
scale :: fn(x: i64) -> i64 { ... }   // private, not exported
```

Only the names on an `export` line are visible to importers. A file with no
`export` line exports nothing. A private item is fully usable inside its own
file, so an exported function may call a private helper, but an importer cannot
name that helper, and two files may share a private name without colliding.
There is no `pub` and no per-item visibility marker. The `export` line is the
only control, and struct fields are always public (3.2).

**An import says what a file may name.** A file sees the names it declares and
the exported names of the modules it imports *directly*, and nothing else.
Importing is not transitive: if `a.frost` imports `b.frost` and `b.frost`
imports `c.frost`, then `a.frost` cannot name what `c.frost` exports until it
imports `c.frost` itself.

That makes the list at the top of a file the list of what it depends on, which
is the only reason to have one. Without it a file could call a function from a
module it never named, and an import line could be deleted with the build still
passing.

The two compilers reach it differently. The bootstrap splices every module into
one program, so it compares what each file used against what that file imported.
The self-hosted compiler resolves a name by scanning declarations with a
visibility rule, so the import becomes an edge that rule has to cross, and an
unimported name is never found.

The exported namespace is flat, and a name carries its own prefix by convention
(`vec3_add`, not a qualified `math.add`), which keeps it a single token to search
for.

Two modules exporting the same name is not itself a problem, since a file that
imports one of them sees one name. It is a problem when one file imports both
*and writes the name*, which is a compile error naming both modules rather than
a silent choice between them. The fix is usually to prefix one of them, since a
name carrying its own prefix is what the flat namespace runs on.

**Reading a name under another.** When neither module is yours to edit, an
import may say what to call what it brings in:

```
import "list.frost" (insert as list_insert)
import "tree.frost" (insert as tree_insert)
```

Everything else still arrives under its own name. Only the names listed are
renamed. A rename belongs to the file that wrote it, so another importer of
`list.frost` still says `insert`. Renaming a name the module does not export is
an error saying so.

This is the last resort, not a style. A renamed call no longer greps back to its
definition, which is the one thing the flat namespace is for, so it is worth it
only when the alternative is not being able to use two libraries at once.
