# 5. Declarations and bindings

A program is a sequence of statements. The top-level meaningful statements are
declarations. These are constants (including functions, structs, and enums),
externs, and imports.

## 5.1 Binding forms

| Form | Meaning |
| --- | --- |
| `name := expr` | bind a local, type inferred |
| `name : Type = expr` | bind a local, type explicit |
| `var name := expr` / `var name : Type = expr` | bind an assignable local |
| `a, b := call()` | bind the several values of one call (5.2a) |
| `ref name := place` | bind a borrow of a place |
| `NAME :: expr` | declare a constant, evaluated once |

Bindings are immutable unless `var`. A `var` local is reassigned with `=`.
`mut` is not a binding form: it is the parameter mode of 3.3, and a
`mut` written at statement position is refused with the words that say so.

`ref name := place` binds a borrow rather than a copy, so writing through the
name writes the place: `ref entry := table[index]` then `entry.count = 0` is a
write to the element. The right-hand side has to be a place (a name, a field, an
index, a dereference, or a call answering with a `ref T`). A `ref` binding is a
borrow like any other: it may not be stored, and it may not outlive what it
names (chapter 8).

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

`main` is the entry, and it has one shape:

```frost
main :: fn() -> i64 {
    0
}
```

Its caller is the C runtime. It calls `main` with the argument count and the
argument vector, which a Frost `main` declares neither of, so a `main` that
declared a parameter would be handed whatever the platform left in that
register:

```frost,refused
main :: fn(count: i64) -> i64 {
    count
}
```

> 'main' takes no parameters, and this one takes 1; what a call to it would
> supply is whatever the platform left in a register

A `uses` clause is counted with the written parameters, since a capability
becomes one (8a.2), and `main` drawing one has nowhere to draw from.

`main` answers the process exit code, so it answers `i64`. A `main` that can
fail, or that answers an aggregate, or that answers nothing, is refused:

```frost,refused
Pair :: struct { a: i64, b: i64 }

main :: fn() -> Pair {
    Pair { a = 1, b = 2 }
}
```

> 'main' is called by the C runtime and its answer is the process exit code, so
> it answers i64

A value constant's `<expr>` is worked out at compile time. An integer one is an
integer literal, an earlier constant by name, a call the compiler works out
(5.2c), or those combined with the integer operators `+ - * / % << >> & |` and
parentheses, folded to a value. A constant may also hold a run of values, a set
of named ones, or a run of bytes, which 5.2c writes out. So
`STRIDE :: POSITION + NORMAL + UV` and `MASK :: 1 << FLAG_BIT` name a
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

Every value is written `name: Type`, and a list that leaves one out is an error:

```
divide :: fn(a: i64, b: i64) -> (quotient: i64, remainder: i64)   // the form
divide :: fn(a: i64, b: i64) -> (i64, i64)                        // an error
```

A name says which value is which at the definition, and it is the field name a
`return` can write:

```
split :: fn(value: i64) -> (high: i64, low: i64) {
    return { high = value / 256, low = value % 256 }
}
```

That is the inferred literal of 6.5 over the struct the list becomes. Two names
the same in one list is an error.

Both forms of `return` are legal. A function whose values share a type returns
by name, and a function that is a table of answers returns by order:
`mnemonic_of` in `selfhosted/assemble.frost` is forty-five lines of
`return M_ADDQ, 0`.

A name in the list is a label for one of the values. It is out of scope in the
body, there is nothing to assign to it, and there is no bare `return` that hands
back whatever the names hold. Each `return` writes the values it answers with.

`return` lists the values in order, or names them, and it is required either
way: a trailing expression is one value, so a function with a return type list
ends every path with a `return`. A `return` that lists a different number of
values than the list has is a compile error, as is one that lists several values
in a function that returns one.

`var` goes in front of any name the body goes on to write:

```
magnitude, var negative := classify(value)
negative = false
```

A `_` takes a value the caller has no use for. Any number of them may sit in one
list, in any position:

```
high, _ := split(4096)
_, low := split(770)
```

A `_` may not take a resource. A `linear` value is consumed exactly once
(chapter 9), so it has to land on a name and be consumed there, and dropping one
is refused where the `_` is written:

```
this `_` drops a 'File', which is consumed exactly once; bind it to a name and
consume it
```

`_` is the wildcard token of 2.3 and never a binding name, and `_` is not an
expression. A list of one is a list, so a single value is discarded the same
way, with `_ := call()` (chapter 7).

The struct a list becomes holds a resource when one of its values does, and it
is the one aggregate that carries no obligation of its own: the lowering builds
it at the `return`, takes it apart at the binding, and reads every field exactly
once, so it owes what its fields owe and each of those lands on a name the
binding introduced.

There is no tuple type. The list is not a value, cannot be named, stored in a
field, passed as an argument, or returned from anything but the function that
declares it, and `(A, B)` is not a type anywhere else in the grammar. A call
that returns several values is bound by a list of names and used nowhere else.
Binding it to a single name is a compile error that says so. A program that
wants to pass a pair around declares a struct and gets a name for it.

The compiler gives the list one struct, whose fields are the names the signature
gave. The signature becomes a plain return of that struct, the `return` becomes
a literal of it, and the binding becomes the call bound to a temporary and one
field read per name. Nothing after the front end sees a return type list.

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
A failure is a value of a type you wrote, and it carries what you put in it.

The signature becomes one enum with two variants:

```frost,sketch
Result :: enum { Ok { value: T }, Err { error: E } }
```

The function returns that enum, and the names at a `match` come from it:
`value` is the field the `Ok` variant carries and `error` is the field `Err`
carries. A field like `error.at` is a field of `Parse`, the type this program
declared. Nothing downstream knows failure sets exist.

A `return` whose expression builds the failure type is the failure, and anything
else is the value:

- `return Parse { at = 3, code = 0 }` names the failure type, and is the
  failure.
- `return .Denied` names no type, and is the failure when the failure type is
  an enum with that variant.
- `return { at = 3, code = 0 }` names no type either, and is the failure when
  the value type is not itself a struct or an enum, since then only the failure
  can be written that way. When both are aggregates, name the one you mean.
- The body's trailing expression is the value, the same as in any other
  function.

A call followed by `?` is the value it answered with, or an immediate return of
its failure from the function the `?` is written in:

```frost,sketch
number :: fn(text: str) -> i64 ! Parse {
    var total : i64 = 0
    var index : i64 = 0
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

The caller matches, and a match on an enum covers every variant (6.7):

```frost,sketch
match number(text) {
    case .Ok { value }: { print("{}\n", value) }
    case .Err { error }: { print("{}\n", error.at) }
}
```

A result carrying a `linear` value is itself linear, so it must be consumed, and
matching it consumes it (chapter 9). Ignoring a call that answers with one is
refused, since the resource would be dropped where nothing named it.

```frost,sketch
open :: fn(n: i64) -> File ! Denied { ... }

use_it :: fn(n: i64) -> i64 {
    match open(n) {
        case .Ok { value }: close(value)     // consumes the File
        case .Err { error }: error.code
    }
}
```

There is no other error channel. No exceptions, no panics to catch, no error
return codes to check by convention, and no ignoring a failure by accident: a
function's signature says what it can fail with, and the caller either matches
it or hands it up with `?`. A failure that is not one, a bug in the program
rather than a condition in the world, is an assertion, and an assertion aborts.

`-> (A, B) ! E` is rejected: a fallible function answers with one value or one
failure. A function that wants both returns a struct it names.

## 5.2c Calls the compiler works out

A call written where a compile-time value is read is run before the program is.
The two places one is read are a constant's value and an array's length.

```frost
round_up :: fn(value: i64, to: i64) -> i64 { (value + to - 1) / to * to }

next_power_of_two :: fn(n: i64) -> i64 {
    var held: i64 = 1
    while (held < n) { held = held * 2 }
    held
}

LANES :: round_up(300, 64)                       // 320
Table :: struct { slots: [next_power_of_two(300)]i64 }
```

The function is an ordinary one. Nothing marks it, and the same function is
called while the program runs wherever a program calls it. Where the call is
written decides that it is worked out early.

Such a call may do the whole-number half of the language: parameters and
locals, integer arithmetic and comparison, `if`, `while`, `break`, `continue`,
`return`, a trailing expression, and calls to other functions written the same
way. `&&` and `||` answer without asking the right side when the left one
settles it, as they do where the program runs.

A value may also be a run of values, a set of named ones, or a run of bytes, so
a lookup table can be settled before the program runs:

```frost
Point :: struct { x: i64, y: i64 }

TABLE  :: [1, 2, 4, 8]
ORIGIN :: Point { x = 3, y = 4 }
NAME   :: "hello"

SLOTS  :: TABLE[2]        // 4
DOWN   :: ORIGIN.y        // 4
LETTER :: NAME[1]         // 101, the byte
WIDTH  :: str_len(NAME)   // 5

Sized :: struct { bytes: [TABLE[3]]u8 }
```

An array literal, a struct literal, a string literal, `[value; n]`, an index, a
field, and `str_len` and `slice_len` over one are all worked out. An index is
checked where it is written, so reading past the end is a compile error naming
the index and the length.

A run is held by the compiler once it is worked out. An element that is itself a
call runs once, however many times the run is named, and the held value outlives
the names that built it.

Everything else is refused, naming what stopped it:

- A function that reaches itself, directly or through others.
- A call into the world. A function this program does not declare has no body to
  read, so an `extern`, and anything that reaches one, stops the call. Reading a
  file, printing, and allocating are all out.
- A pointer, an `unsafe` block, a `match`, a `for`, a `defer`, a `?`. Each is
  named where it is written.
- A number with a fraction. A compile-time value is a whole number or a yes or
  no.
- More than a million steps, or calls nested deeper than thirty-two.

A call names a function the file can name: what it declares, and what the files
it imports export.

`wrap_add`, `wrap_sub` and `wrap_mul` are worked out here too. They are the way
a value leaves its range on purpose, and a hash folded before the program runs
comes out the same as one computed while it does.

A compile-time number is read in three places and a call may stand in all of
them: a constant's value, an array's length, and the value argument a generic
takes, whether written in a type (`Slab<Entity, next_power_of_two(300)>`) or at
a call (`slab_insert($Entity, $next_power_of_two(300), ...)`).

Every argument has to be known where the call is written. So a call over a
generic's own size parameter, `Grid :: struct($N: usize) { cells: [pow2(N)]i64 }`,
is refused and the parameter is named: `N` is bound at the instantiation, which
is later than the declaration the call is written in. The caller-side form hands
the rounded number in: `Grid<pow2(300)>`. Arithmetic over a size parameter keeps
working, and `[(N + 63) / 64]i64` is how a length over one is written (3.2).

## 5.3 Externs and imports

```
name :: extern fn(params) -> R        // foreign function (chapter 12)
name :: extern fn(params)             // foreign function returning nothing
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

```frost,sketch
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

An import says what a file may name. A file sees the names it declares and the
exported names of the modules it imports *directly*, and nothing else.
Importing is not transitive: if `a.frost` imports `b.frost` and `b.frost`
imports `c.frost`, then `a.frost` cannot name what `c.frost` exports until it
imports `c.frost` itself.

The exported namespace is flat, and a name carries its own prefix by convention
(`vec3_add`, not a qualified `math.add`).

Two modules may export the same name, and a file that imports one of them sees
one name. A file that imports both *and writes the name* is a compile error
naming both modules. The fix is usually to prefix one of them.

When neither module is yours to edit, an import may say what to call what it
brings in:

```
import "list.frost" (insert as list_insert)
import "tree.frost" (insert as tree_insert)
```

Everything else still arrives under its own name. Only the names listed are
renamed. A rename belongs to the file that wrote it, so another importer of
`list.frost` still says `insert`. Renaming a name the module does not export is
an error saying so.

A renamed call no longer greps back to its definition.
