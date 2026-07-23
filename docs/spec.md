# The Frost Language Specification

This is the reference specification for the Frost language as implemented by the
native, data-oriented compiler (the `--native`, `--link`, `--emit-c`, and
`--run-ir` paths). It is normative for that language. There is one language and
one parser. The older bytecode VM and every syntactic form that existed only to
serve it have been removed, so the parser accepts exactly what this document
describes.

The specification has two halves. The prose chapters (1 through 12) describe the
lexical structure, types, declarations, expressions, statements, and static
semantics. The grammar chapter (13) gives the syntax in one place, in a
disciplined EBNF that the hand-written recursive-descent parser is held against.

## Contents

1. Notation and conformance
2. Lexical structure
3. Types
4. The type system
5. Declarations and bindings
6. Expressions
7. Statements
8. Ownership and borrowing
9. Linear resources
10. Handles, pools, and the memory model
11. Generics and compile-time specialization
12. The foreign function interface
13. Grammar
14. Appendix: precedence, keywords, escapes

---

## 1. Notation and conformance

### 1.1 Grammar notation

Grammar rules use EBNF:

- `x y` is `x` followed by `y`. `x | y` is `x` or `y`.
- `x?` optional, `x*` zero or more, `x+` one or more, `( ... )` groups.
- Terminals are literal spellings in `code font` (`::`, `fn`, `->`) or token
  classes in UPPERCASE (`IDENT`, `INTEGER`, `STRING`).
- Nonterminals are `PascalCase`.

### 1.2 Parsing discipline

The language is parsed by recursive descent with a Pratt (precedence-climbing)
expression parser and **bounded lookahead**. Statement and type selection is
decided by the first one to three tokens. The specific lookahead each decision
uses is stated in the grammar. Expression parsing is driven by the operator
precedence table in 14.1. The parser does not backtrack past a committed
production, with one bounded exception. It may scan ahead a fixed number of
tokens to decide whether a parenthesized group is a function parameter list
(13.6).

This discipline is the contract. The reference parser (`src/parser.rs`), the
self-hosted parser (`bootstrap/`), and this grammar are three views of one
language. A disagreement between them is a bug in whichever diverges from the
intent expressed here. A conformance test (`tests/grammar.rs`) feeds a corpus of
accepted and rejected forms through the reference parser on every build, so the
grammar and the parser cannot drift apart unnoticed.

### 1.3 Conformance

A conforming program is one this grammar accepts and whose static semantics
(chapters 4, 8, 9, 10, 11) hold. A conforming implementation rejects
non-conforming programs with a diagnostic and compiles conforming programs to
code with the behavior described here. Constructs marked *unchecked* (raw
pointer operations, `extern` calls) place their correctness obligation on the
programmer.

---

## 2. Lexical structure

### 2.1 Source

Source text is UTF-8. Tokens are formed by maximal munch. At each point the
lexer takes the longest token that matches. Identifiers are ASCII,
`[A-Za-z_][A-Za-z0-9_]*`. There are no Unicode identifiers.

### 2.2 Whitespace and comments

Whitespace (space, tab, carriage return, newline) separates tokens and is
otherwise insignificant. Frost is not whitespace-sensitive and has no automatic
semicolon insertion. Statement terminators (`;`) are always optional. There are
two comment forms:

- Line comment, `//` to end of line.
- Block comment, `/* ... */`. Block comments do not nest, and an unterminated
  block comment is an error.

### 2.3 Identifiers and the wildcard

```
IDENT = ( LETTER | "_" ) ( LETTER | DIGIT | "_" )*
```

The single underscore `_` is a distinct token, the wildcard, and is not a
binding name.

### 2.4 Keywords

Reserved words of the specified language:

```
fn struct enum match case if else while for in mut return break continue
defer extern import linear distinct type unsafe sizeof
```

Reserved primitive type names, each its own token:

```
i8 i16 i32 i64 isize   u8 u16 u32 u64 usize   f32 f64   bool str void
```

`test` and `export` are not reserved. They are recognized contextually, `test`
only at the start of a top-level test declaration and `export` only on a
top-level export line, so both remain usable as ordinary identifiers elsewhere.
`Type` (capitalized), used in `$T: Type` (chapter 11), is likewise an ordinary
identifier recognized in that position, not a keyword.

### 2.5 Literals

**Integer**. `INTEGER = DIGIT+`. Decimal only. No digit separators, no
hexadecimal, octal, or binary prefixes. Integer literals are non-negative. A
negative value is the prefix `-` applied to one. An integer literal takes its
type from context, defaulting to `i64`.

**Float**. `FLOAT = DIGIT+ "." DIGIT+`, with an optional `f` or `f32` suffix that
makes it an `f32`, otherwise it is `f64`. A `.` is only taken as a decimal point
when the following character is not another `.`, so `0..10` lexes as a range.
There is no exponent notation and no leading-dot form.

**String**. Delimited by `"`, with escapes `\n`, `\t`, `\r`, `\0`, `\\`, `\"`,
`\'`. Any other escape is an error. There are no numeric or Unicode escapes. A
string literal has type `str` (3.7) and denotes a view of its bytes. Where `^i8`
is expected it instead denotes a pointer to the same bytes with a trailing NUL,
which is how string literals interoperate with C.

**Boolean**. `true`, `false`, of type `bool`.

### 2.6 Operators and punctuation

```
::  :=  :   =   ->  ..  ..=  .   ^   $   ?   #
+   -   *   /   %   &   |   &&  ||  <<  >>
==  !=  <   <=  >   >=  !
(   )   {   }   [   ]   ,   ;
```

`>>` is a single shift token that the
parser splits when it closes nested generic arguments (11.4).

---

## 3. Types

### 3.1 Scalar types

| Type | Meaning | Size (bytes) |
| --- | --- | --- |
| `i8` `i16` `i32` `i64` `isize` | signed integers | 1, 2, 4, 8, 8 |
| `u8` `u16` `u32` `u64` `usize` | unsigned integers | 1, 2, 4, 8, 8 |
| `f32` `f64` | IEEE floats | 4, 8 |
| `bool` | boolean | 1 |
| `void` | the unit/empty type | 0 |

All scalar types are **copy** types (chapter 8). Integer arithmetic wraps at the
type width with two's-complement semantics and is never checked for overflow.
Mixed-width integer arithmetic is permitted. The narrower operand widens to the
wider type.

### 3.2 Aggregate types

- **Structs** `Name`, declared `Name :: struct { field: T, ... }`, are exactly
  their fields in declaration order, with natural alignment.
- **Enums** `Name`, declared `Name :: enum { Variant, Variant { f: T }, ... }`,
  are a discriminant plus the active variant's payload. Variants may be unit or
  carry named fields, and one enum may mix both. An enum takes type parameters
  exactly as a struct does, `Maybe :: enum($T: Type) { Nothing, Just { value: T } }`,
  and instantiates the same way (chapter 11).
- **Fixed arrays** `[N]T` are `N` contiguous `T`. The length is part of the type
  and every index is bounds-checked (10.4).
- **Slices** `[]T` are a pointer/length view of a run of `T`, sixteen bytes and a
  copy value, the same fat-pointer shape as `str` (which is `[]u8`). An array
  coerces to a slice of the whole array, `s[i]` is bounds-checked against the
  runtime length (10.4), and `slice_len(s)` reads the length in constant time.

Aggregates are **move** types (chapter 8), copied by value at call and return
boundaries unless passed by borrow, with no `Copy` derive.

Frost has no visibility modifiers. There is no `pub` and no private. Every struct
field is public and reachable, and there is nothing to specify.

### 3.3 Borrows and pointer types

A borrow is not a type a program writes. It is what a parameter mode means:
an unmarked parameter of a non-copy type is read-borrowed, `mut` is
write-borrowed, and `move` takes the value (chapter 8). There is no `&` or `&mut`
in the surface, so a borrow has nowhere to be written down and nowhere to be
stored, which is what makes it second-class by construction rather than by rule.

- `^T` raw pointer, unchecked, for FFI and low-level libraries.

`ptr_to(place)` yields a `^T` to a place. `ptr_cast($T, p)` reinterprets a
pointer as `^T` at no runtime cost. These are the low-level tools an allocator
uses to hand back typed memory from a byte buffer; ordinary code does not need
them, and a pointer carries no safety guarantee once it is formed.

A pointer or a slice that names storage in the current frame may not be
returned, and neither may one into an arena outlive its region (chapter 8).

### 3.4 Handle and pool types

`Handle<T>` names an element of a pool of `T` (chapter 10), a small copy value
(index plus generation), not a pointer, that unlike a reference may be stored in
fields and returned.

A pool is not a built-in type. It is an ordinary struct a program writes for
itself, an array of storage indexed by `Handle<T>` (chapter 10.1). The compiler
provides the pieces to build one, not the pool itself.

### 3.5 Function types

`fn(T1, ...) -> R` is a function pointer. There are no closure types. A
function-typed value is always a plain pointer to a function.

A parameter of a function type may be written `mut T`, which means the same
reference the `mut` mode means on a declared parameter (chapter 8). It has to be
sayable here because the surface has no reference type to write instead. An
unmarked parameter is the type as written, and `move T` is the type as written
too, allowed so a function type can be read beside the declaration it describes.

### 3.6 Other type forms

- `distinct T` is a nominal type with `T`'s representation, not interchangeable
  with `T`.
- `?T` is an optional `T`.
- `$T` is a type parameter (chapter 11).
- `Name<T, ...>` is a generic instantiation (chapter 11).

### 3.7 Strings

`str` is an immutable, non-owning view of a run of bytes. It is a pointer and a
length, sixteen bytes, laid out as the byte pointer at offset 0 and the length
(a `usize`) at offset 8. It owns nothing, so it is a **copy** type (chapter 8),
freely duplicated with no move and nothing to release. In this it is the byte
form of a slice (`[]u8`).

- A string literal is a `str` pointing into read-only data, with the length
  fixed at compile time.
- `str_len(s)` returns the byte length in constant time, reading the length
  field rather than scanning.
- `s[i]` reads the byte at index `i` as a `u8` and is bounds-checked against the
  length (10.4), the same rule as array indexing.
- Passing a `str` to a function copies the pointer and length by value.

`str` carries no NUL terminator and may contain a NUL byte. Crossing to C is
therefore an explicit conversion, not an automatic one. The single affordance is
the string literal, which the compiler also emits NUL-terminated so that a
literal used where `^i8` is expected passes as a C string at no cost (2.5).

Owned or growable text is not a language type. It is an ordinary struct over an
array or a pool that a program borrows as a `str`.

---

## 4. The type system

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

---

## 5. Declarations and bindings

A program is a sequence of statements. The top-level meaningful statements are
declarations. These are constants (including functions, structs, and enums),
externs, and imports.

### 5.1 Binding forms

| Form | Meaning |
| --- | --- |
| `name := expr` | bind a local, type inferred |
| `name : Type = expr` | bind a local, type explicit |
| `mut name := expr` / `mut name : Type = expr` | bind a mutable local |
| `NAME :: expr` | declare a constant, evaluated once |

Bindings are immutable unless `mut`. A `mut` local is reassigned with `=`.

The parser distinguishes `name : Type = ...` (a typed binding) from
`name :: ...` (a constant) by one token of lookahead after the first `:`. A
second `:` means a constant.

### 5.2 Constants and items

`::` declares a constant. Functions, structs, enums, and type aliases are all
constants:

```
NAME  :: <expr>                       // value constant
f     :: fn(params) -> R { body }     // function
Point :: struct { x: i64, y: i64 }    // struct
Shape :: enum { A, B { n: i64 } }     // enum
Meters :: distinct i64                // distinct type
```

A `linear` qualifier may precede `struct` or `enum` (chapter 9).

### 5.3 Externs and imports

```
name :: extern fn(params) -> R        // foreign function (chapter 12)
name :: extern fn(params)             // foreign function returning void
import "path"                         // bring another source file into scope
```

### 5.4 Tests

A `test` block declares a named unit test.

```
test "name" { Stmt* }
```

`test` is a contextual keyword, recognized only when followed by a string
literal and a block, so it remains usable as an ordinary identifier elsewhere.
Inside a test, `assert(cond)` aborts the test when `cond` is false. `frost --test
file.frost` compiles the file, runs each test in declaration order, and exits
non-zero if any assertion fails.

### 5.5 Modules and exports

A source file is a module. `import "path"` brings another file in, resolved
relative to the importing file, and each file is pulled in once even through a
diamond of imports.

Top-level items are private to their file by default. A file lists what it offers
with an `export` line at the top.

```
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

---

## 6. Expressions

### 6.1 Primary expressions

The primary expressions are integer, float, string, and boolean literals,
identifiers, parenthesized expressions `( Expr )`, and array literals, either the
listed form `[ e, ... ]` or the repeat form `[ e ; N ]` for `N` copies of `e`
(the way a large or zeroed backing buffer is written, e.g. `[0; 256]`).

### 6.2 Operators

Prefix `-` (negate) and `!` (logical not). Binary operators, grouped by the
precedence in 14.1, are `||`, `&&`, `==` `!=`, `<` `<=` `>` `>=`, `|`, `&`, `<<`
`>>`, `+` `-`, `*` `/` `%`, and the range operators `..` and `..=`. All binary
operators are left-associative.

### 6.3 References and dereference

- `ptr_to(place)` the address of a place. There is no borrow operator: a
  borrow is what a parameter mode means, inserted at the call.
- `expr^` dereferences a raw pointer to its pointee value and is assignable
  (`p^ = v`). Member access through a raw pointer is written `p^.field`.
- A borrowed parameter needs no dereference at all, whatever its type. `p.field`
  reads and writes a field of a borrowed aggregate, and `p = q` on a `mut`
  parameter assigns the whole value through the borrow rather than rebinding
  anything local. Naming a `mut` parameter always means the caller's value.

### 6.4 Calls, indexing, and field access

- `f(a, b, ...)` calls a function or function pointer.
- `a[i]` indexes an array, slice, or pool (for a pool, `i` is a `Handle`).
- `e.field` accesses a struct field or, on an enum place, a variant field.

### 6.5 Construction

```
Point { x = 1, y = 2 }                // struct literal (fields use =)
Shape::Circle { radius = 5 }          // enum variant with payload
Shape::Player                         // unit variant
```

Struct and enum-variant construction are recognized only when the operand to the
left of `{` or `::` is a bare identifier.

**A literal must write every field.** There is no partial construction, no
`..rest`, and no implicit zero. A field left out would name storage nothing
wrote, and reading it afterwards would read whatever was there, which is exactly
the shape chapter 8 exists to make unrepresentable. A missing field is an error
that names it.

### 6.6 `if` expression

```
if ( Cond ) Block
if ( Cond ) Block else Block
```

The condition is parenthesized. `if` is an expression. Both arms are blocks and
their trailing expressions are the value.

### 6.7 `match` expression

```
match Scrutinee {
    case Pattern : Expr
    case Pattern : Block
    ...
}
```

An arm is `case`, a pattern, `:`, then an expression or block. There is no
separator between arms. An arm ends where the next `case` or the closing `}`
begins. Patterns:

- Variant, shorthand, `.Variant` or `.Variant { field, field }`, binding each
  named field to a same-named local.
- Variant, qualified, `Enum::Variant` with the same optional field list.
- Value, an integer, float, string, or boolean literal (`case 90:`).
- Tuple, `( P, P, ... )`.
- Binding, a bare identifier.
- Wildcard, `_`.

`match` works over a value or a reference. Matching a value of a `linear` type
consumes it (chapter 9).

### 6.8 `sizeof` and `unsafe`

- `sizeof(T)` is a compile-time constant.
- `unsafe { ... }` is a block whose body may use unchecked operations.

### 6.9 Ranges

`a..b` is half-open, `a..=b` inclusive. Ranges appear in `for` and are the
lowest-binding binary form.

---

## 7. Statements

A block `{ Stmt* }` is a sequence of statements and is itself an expression whose
value is its trailing expression (or `void`).

- **Expression statement**, an expression evaluated for effect.
- **Binding**, the forms in 5.1.
- **Assignment**, `Place = Expr`, where `Place` is a `mut` local, a field, an
  index, or a dereference.
- **`return`**, `return` or `return Expr`.
- **`while`**, `while ( Cond ) Block`.
- **`for`**, `for name in Expr Block` iterates `name` over the value of `Expr`,
  normally a range.
- **`break`** and **`continue`** are loop control.
- **`defer`**, `defer Stmt` runs `Stmt` at scope exit, LIFO (chapter 9.3).

---

## 8. Ownership and borrowing

Frost is memory-safe without a garbage collector and without lifetimes. The
borrow rules run after parsing (`src/ownership.rs`).

### 8.1 Copy and move

Each type is **copy** or **move**. Scalars, pointers, function pointers, handles,
strings, and slices are copy: a slice and a `str` are a pointer and a length, and
copying one copies that pair rather than what it names. Structs and enums are
move. A
move value is consumed when passed by value, assigned, or returned. Using it
after is a use-after-move error. There is no `Copy`/`Clone` derive and no
implicit deep copy. A second copy of an aggregate is constructed explicitly.

### 8.2 Second-class borrows

A borrow exists only as a parameter mode, so it cannot be stored in a struct or
enum field, placed in an array, or returned: there is no syntax that would name
one. Because a borrow cannot escape its call, borrow analysis is scope-local, and
Frost has no lifetime annotations, lifetime variables, or borrow regions.

A raw pointer can escape, which is what it is for, so the two ways one could
outlive its storage are checked rather than forbidden. A function may not answer
with a pointer or a slice that names its own frame, and an arena pointer may not
outlive the `with` block that owns the arena. A `uses` function may hand one back
to its caller, whose region checks it, but may not store one into a parameter.

### 8.3 Borrow exclusivity

Within one call, a value may be read-borrowed any number of times or write-
borrowed exactly once, never both at once. Passing the same variable to two `mut`
parameters of one call is rejected. This per-call check suffices to prevent
mutable aliasing precisely because borrows cannot escape.

### 8.4 Reference escape through returns

A function or `extern` whose return type contains a reference is rejected.
`Handle<T>` is not a reference and may be returned and stored freely, the
intended replacement for an escaping borrow.

---

## 9. Linear resources

### 9.1 The linear rule

A struct or enum declared `linear` must be consumed **exactly once**. The move
rule gives "at most once". Linearity adds "at least once". A linear value still
live at the end of its owning scope is a compile error.

### 9.2 Consuming

A linear value is consumed by moving it onward, whether by returning it, passing
it by value (often to an `extern` that takes ownership across the FFI boundary),
or `match`ing it. This replaces destructors. Cleanup is a tracked obligation,
never an implicit call, and a `linear enum` returned from a fallible function
cannot be silently dropped, so errors are non-ignorable by construction.

### 9.3 `defer`

`defer` runs a statement at scope exit in LIFO order, for local best-effort
cleanup. Owned resources should be `linear` types, which the compiler checks,
rather than relying on `defer`.

---

## 10. Handles, pools, and the memory model

### 10.1 Pools

A pool is a contiguous, fixed-capacity arena of same-typed elements addressed by
`Handle<T>` rather than pointer. A pool is not a language type: it is a struct a
program writes for itself, holding the storage and a free list, with the
generational `(generation << 32) | index` handle and the stale-handle check as
ordinary code. `examples/native/generic_slab.frost` is a pool generic over both
element type and capacity, built on value generics (`[N]T` storage, 11.1a) and
slices (`slice_len` to recover the capacity). The compiler provides the pieces,
arrays, handles, value generics, `ptr_to`/`ptr_cast`, and the byte buffer, not
the pool itself. This is the data-oriented memory model expressed in the
language, and the direction is written up in `docs/native-pools.md` and
`docs/allocators.md`.

The runtime in `runtime/frost_runtime.c` offers a ready-made generational pool
(`pool_new`, `pool_alloc`, `pool_get`, `pool_free`, `pool_contains`,
`pool_destroy`), reachable as an opt-in library by declaring the functions with
`extern fn`, the way `malloc` is. Nothing about it is compiler-special. When a
pool from it is indexed by a `Handle<T>`, `pool[handle]` lowers to its `pool_get`
(10.2).

### 10.2 `pool[handle]` is a place

`pool[handle]` is a place. Read a field, write a field, copy the element out, or
pass it to a parameter, which borrows it. The borrow obtained is second-class and
cannot escape the call.

### 10.3 Generational safety

A handle carries the generation of the slot it was minted for. Freeing a slot
increments its generation. A lookup whose handle generation does not match the
slot's current generation fails rather than returning the slot's new occupant, so
a stale handle can never read or write freed-and-reused data.

### 10.4 Bounds checking

Every fixed-array index is checked against the statically known length. An
out-of-range index aborts (`frost_bounds_check`). There is no unchecked-index
form.

### 10.5 The six guarantees

| Guarantee | Mechanism |
| --- | --- |
| No dangling references | references are second-class (8.2) |
| No use-after-move | move checking (8.1) |
| No mutable aliasing | per-call borrow exclusivity (8.3) |
| No leaked resources | linear consume-exactly-once (9.1) |
| No use-after-free of pooled data | generational handles (10.3) |
| No out-of-bounds array access | bounds checking (10.4) |

Raw pointers (`^T`) are outside these guarantees by design.

---

## 11. Generics and compile-time specialization

### 11.1 Type parameters

A type parameter is written `$T`. It may appear on a function's parameters and on
a struct or enum declaration:

```
Pair  :: struct($T: Type) { first: T, second: T }
Maybe :: enum($T: Type) { Nothing, Just { value: T } }
make_pair :: fn(a: $T, b: $T) -> Pair<T> { Pair { first = a, second = b } }
```

A generic literal carries no arguments of its own, so which instance it is comes
from the context: an annotation, or the type of the parameter it is passed to.

```
m : Maybe<i64> = Maybe::Just { value = 42 }     // the annotation names it
unwrap_or($i64, Maybe::Nothing, 7)              // the parameter names it
```

In a parameter or struct type-parameter position, `$` IDENT `:` is followed by
the contextual word `Type` (or the keyword `type`). In a function's parameter
list it may instead be followed by a function signature, which declares a
compile-time function parameter (11.1b).

### 11.1a Value parameters

A parameter written `$N: usize` is a value parameter rather than a type
parameter. It is a compile-time integer, and its main use is sizing a fixed
array:

```
Slab :: struct($T: Type, $N: usize) { storage: [N]T, used: i64 }
world : Slab<Entity, 4> = ...
```

An instantiation supplies an integer where a value parameter stands
(`Slab<Entity, 4>`), and monomorphization resolves `[N]T` to the concrete
`[4]Entity` for that instance. Value parameters are erased from the specialized
type the same way type parameters are.

**A function takes them too**, which is what lets an operation over a sized
aggregate be written once rather than once per size:

```
slab_reset :: fn($T: Type, $N: usize, mut s: Slab<T, N>) {
    mut i : i64 = 0
    while (i < N) { s.generations[i] = 0  i = i + 1 }
}

slab_reset($Entity, $4, world)
```

Inside the body the name stands for the integer wherever it appears, in a type
(`[N]T`) and in an expression (`i < N`) alike. `examples/native/lib/slab.frost`
is a generational pool written this way, generic over both element type and
capacity.

### 11.1b Compile-time function parameters

A parameter written `$f: Type` whose argument names a declared function is a
compile-time function parameter. The specialization calls it directly, with no
function pointer and no indirect call:

```
ascending :: fn(a: i64, b: i64) -> bool { a < b }

best :: fn($T: Type, $before: Type, move x: $T, move y: $T) -> $T {
    mut result := x
    if (before(y, result)) { result = y }
    result
}

smallest := best($i64, $ascending, 7, 3)
```

Written `$f: Type` the parameter accepts a function of any signature, and a
mismatch surfaces inside the specialized body. Writing the signature instead
states what the argument has to be:

```
best :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T, move y: $T) -> $T
```

The bound is checked at the call, with that call's type arguments substituted
into it, so `T` in the bound means what it means at the call. A function whose
signature differs, or a type where a function is required, is an error reported
against the parameter list.

This is the only form of bound in the language, and it bounds one parameter kind
against one signature. It is not a trait, has no coherence or orphan rules, and
involves no solving.

### 11.2 Monomorphization

Generics specialize at compile time. Each concrete instantiation compiles to its
own code, with no runtime dispatch and no boxing. Type parameters are erased from
the specialized ABI once monomorphization chooses concrete types.

### 11.3 Explicit type arguments

There is no turbofish. A type is passed as an ordinary argument by writing `$`
before it, which forms a type value:

```
make_pool :: fn($T: Type, capacity: i64) -> ^u8 { pool_new(capacity, sizeof(T)) }
world := make_pool($Entity, 16)
```

### 11.4 Nested generic arguments

Generic arguments are delimited by `<` and `>`. Because `>>` lexes as one shift
token, the parser splits it when it closes two nested argument lists, so
`Pair<Pair<i64>>` parses correctly. This splitting is wired into the `Handle<T>`
and `Name<...>` type forms.

### 11.5 No traits

There are no traits, and therefore no `where` clauses, associated types, trait
objects, or dynamic dispatch. A generic body type-checks once specialized.

To abstract over an operation, pass it as a compile-time function parameter
(11.1b), which keeps the call direct. A type parameter written `$T: Type` carries
no bound of its own: what a body requires of `T` is whatever its code does with
it, and that is checked when the specialization is compiled.

---

## 12. The foreign function interface

An `extern fn` declares a function with C linkage:

```
printf :: extern fn(fmt: ^i8, value: i64) -> i32
malloc :: extern fn(size: i64) -> ^u8
```

Frost scalar types map to the natural C types and `^T` is a C pointer. String
literals denote NUL-terminated bytes for `^i8` parameters. An `extern` takes
parameter modes like any other function.

**Aggregate parameters and aggregate returns are not symmetric**, and the
asymmetry is deliberate.

- An aggregate **parameter** is passed as a pointer to the value, so
  `close :: extern fn(f: File)` links against a C `void close(File*)`. This is a
  convention rather than the C ABI, chosen because most C APIs take structs by
  pointer, and it is what lets a `linear` resource have a terminal consumer
  across the boundary. Passing a struct to C by value has no spelling.
- An aggregate **return** is by value, following the target's real C ABI: in
  registers where that target's rule says so, and through a hidden pointer where
  it does not. A return could not have been a convention, because `-> Ctx` has
  to mean what C means by it and `-> ^Ctx` is how a returned pointer is written.

### 12.1 Callbacks

An `extern` whose parameter list has a `$` parameter bound to a function
signature is a **callback registration**:

```
Ctx :: struct { hits: i64 }

on_event         :: fn(mut ctx: Ctx, code: i64) { ctx.hits = ctx.hits + code }
register_handler :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64
```

The handler's **first parameter is the context** and must be written `mut`.
Whichever parameter of the extern has that type is the one the context is taken
from, found by type rather than by position because C libraries put the userdata
on either side of the function pointer, and it must be taken by `move`. The call
passes the handler's address and the context's address; there is no generated
trampoline, because a `mut` parameter is already a pointer in the signature and
Frost and C share a calling convention.

The context moving in is what makes this safe rather than merely typed: the
caller cannot touch the context while the callback can fire. A registration is
normally a `linear` value, so it must be consumed, and the region check refuses
to let it leave the frame that holds its context.

The FFI is otherwise asymmetric. **Frost calls C, but C does not call Frost**,
except through a registered callback, which is the one place a C library holds a
Frost function pointer. There is no stable exported ABI and no attribute to
expose a Frost function to a C caller. The emitted C is an internal lowering,
not an interface.

---

## 13. Grammar

This chapter is the complete syntax of the specified language. Legacy forms the
current parser also accepts are listed in 13.9 and are not part of the language.

### 13.1 Program and statements

```
Program   = Statement*

Statement =
      "return" Expr? ";"?
    | "defer" Statement
    | "for" IDENT "in" Expr Block
    | "while" "(" Expr ")" Block
    | "break" ";"?
    | "continue" ";"?
    | "import" STRING ";"?
    | "mut" IDENT ( ":=" Expr | ":" Type "=" Expr ) ";"?
    | IDENT ":=" Expr ";"?
    | IDENT ":" Type "=" Expr ";"?           // lookahead: ":" not followed by ":"
    | IDENT "::" ConstBody ";"?
    | Expr ( "=" Expr )? ";"?                 // expression statement or assignment
```

The `mut` / `:=` / `: =` / `::` forms are selected by the token after the
identifier. These are `:=` (inferred binding), `:` then a non-`:` (typed
binding), and `::` (constant). The last alternative covers expression statements
and assignments to a place.

### 13.2 Constants and items

```
ConstBody =
      "linear"? "struct" GenericParams? "{" StructFields? "}"
    | "linear"? "enum" GenericParams? "{" EnumVariants? "}"
    | "distinct" Type
    | "extern" "fn" "(" Params? ")" ( "->" Type )?
    | Expr                                    // function literal, or a value

GenericParams = "(" TypeParam ( "," TypeParam )* ")"
TypeParam     = "$" IDENT ":" ( "Type" | "type" )

StructFields  = StructField ( "," StructField )* ","?
StructField   = IDENT ":" Type

EnumVariants  = EnumVariant ( "," EnumVariant )* ","?
EnumVariant   = IDENT ( "{" ( IDENT ":" Type ( "," IDENT ":" Type )* )? "}" )?

Params        = Param ( "," Param )*
Param         = ParamMode? "$" IDENT ":" ( "Type" | "type" | "usize" | ProcType )
              | ParamMode? IDENT ( ":" Type )?
ParamMode     = "mut" | "move"
ProcType      = "fn" "(" ( ProcParam ( "," ProcParam )* )? ")" ( "->" Type )?
ProcParam     = ParamMode? Type
```

A `Name :: fn(...) { ... }` item is the `Expr` alternative of `ConstBody`, whose
expression is a function literal (13.6).

### 13.3 Blocks

```
Block = "{" Statement* "}"
```

The trailing expression of a block, if any, is its value.

### 13.4 Expressions

Expressions are parsed by precedence climbing. `Expr` denotes an expression at
the lowest precedence. The operator table in 14.1 governs grouping.

```
Expr    = Prefix ( InfixOp Expr )*           // resolved by precedence (14.1)

Prefix =
      Primary
    | "-" Expr
    | "!" Expr
    | "&" "mut"? Expr                         // borrow / mutable borrow
    | "$" Type                                // type value (11.3)
    | "sizeof" "(" Type ")"

Primary =
      INTEGER | FLOAT | STRING | "true" | "false"
    | IDENT
    | "(" Grouped                             // group, tuple, or function literal
    | "[" ( Expr ( "," Expr )* )? "]"         // array literal
    | "[" Expr ";" INTEGER "]"                // repeat array literal
    | IfExpr
    | MatchExpr
    | "fn" "(" Params? ")" ( "->" Type )? Block
    | "unsafe" Block
```

Postfix and infix forms, applied by the precedence loop:

```
InfixOp = "||" | "&&" | "==" | "!=" | "<" | "<=" | ">" | ">="
        | "|"  | "&"  | "<<" | ">>" | "+" | "-" | "*" | "/" | "%"

Postfix =
      Expr ".." Expr                          // range (half-open)
    | Expr "..=" Expr                         // range (inclusive)
    | Expr "[" Expr "]"                       // index
    | Expr "(" ( Expr ( "," Expr )* )? ")"    // call
    | Expr "." IDENT                          // field access
    | Expr "^"                                // dereference (assignable place)
    | IDENT "{" StructInit? "}"              // struct literal (bare identifier)
    | IDENT "::" IDENT ( "{" StructInit? "}" )?   // enum variant (bare identifier)

StructInit = IDENT "=" Expr ( "," IDENT "=" Expr )* ","?
```

`Expr "^"` and the struct/enum-init forms are only entered when the left operand
is the appropriate shape (a place for `^`, a bare identifier for `{`/`::`). The
struct-literal `{` is disambiguated from a `match` body by checking that the
token after `{` is not `case`.

### 13.5 `if` and `match`

```
IfExpr    = "if" "(" Expr ")" Block ( "else" Block )?

MatchExpr = "match" Expr "{" MatchArm* "}"
MatchArm  = "case" Pattern ":" ( Block | Expr )

Pattern =
      "_"
    | INTEGER | FLOAT | STRING | "true" | "false"
    | "." IDENT ( "{" IDENT ( "," IDENT )* "}" )?
    | IDENT "::" IDENT ( "{" IDENT ( "," IDENT )* "}" )?
    | "(" Pattern ( "," Pattern )* ")"
    | IDENT
```

### 13.6 Parenthesized groups and function literals

A `(` begins one of three things, chosen by a bounded look-ahead scan
(`looks_like_function_params`, at most a fixed number of tokens, depth-tracked):

```
Grouped =
      ")" ( "->" Type )? Block               // zero-parameter function literal
    | ")"                                     // empty tuple  ()
    | Params ")" ( ( "->" Type )? Block )?    // function literal (if a body follows)
    | Expr ( "," Expr )* ")"                  // tuple, or a parenthesized expression
```

A `:` at group depth zero marks a parameter list. When a parameter-shaped group
is not followed by a body, its contents are reinterpreted as expressions (a
single expression, or a tuple).

### 13.7 Types

```
Type =
      "i8" | "i16" | "i32" | "i64" | "isize"
    | "u8" | "u16" | "u32" | "u64" | "usize"
    | "f32" | "f64" | "bool" | "str" | "void"
    | "^" Type                               // raw pointer
    | "&" "mut"? Type                        // reference
    | "[" "]" Type                           // slice
    | "[" INTEGER "]" Type                   // array (size first)
    | "[" Type ";" INTEGER "]"               // array (element first)
    | "fn" "(" ( ProcParam ( "," ProcParam )* )? ")" ( "->" Type )?
    | "distinct" Type
    | "?" Type
    | "Handle" "<" Type ">"
    | IDENT "<" Type ( "," Type )* ">"       // generic instantiation
    | IDENT                                  // named type
    | "$" IDENT                              // type parameter
```

A type is a single prefix-constructed form. Nesting comes from the recursive
constructors (`^`, `[]`, `?`, `distinct`, `fn`), not a postfix loop. Closing
`>` in the generic forms accepts a split `>>` (11.4).

### 13.8 Comparison and equality precedence

The precedence ladder (14.1) places comparison tighter than equality and the
bitwise operators tighter than comparison, which differs from C. Write explicit
parentheses in mixed expressions. A conformance-minded style parenthesizes any
combination of `==`/`!=`, the comparisons, and the bitwise operators.

---

## 14. Appendix

### 14.1 Operator precedence

Lowest to highest binding. All binary operators are left-associative. This table
is normative and matches the reference parser's precedence mapping.

| Level | Operators | Notes |
| --- | --- | --- |
| Range | `..` `..=` | range construction |
| LogicalOr | `\|\|` | |
| LogicalAnd | `&&` | tighter than `\|\|` |
| Equals | `==` `!=` | |
| Comparison | `<` `<=` `>` `>=` | tighter than equality |
| BitwiseOr | `\|` | |
| BitwiseAnd | `&` | |
| Shift | `<<` `>>` | |
| Sum | `+` `-` | |
| Product | `*` `/` `%` | |
| Prefix | `-` `!` | unary |
| Call / Index / Access | `f(...)` `a[i]` `.` `^` `::` | tightest |

### 14.2 Keywords

Specified language:

```
fn struct enum match case if else while for in mut return break continue
defer extern import linear distinct type unsafe sizeof
```

Primitive type names are `i8 i16 i32 i64 isize u8 u16 u32 u64 usize f32 f64 bool
str void`. The wildcard is `_`. `test` and `export` are contextual, not reserved.

### 14.3 String escapes

`\n` `\t` `\r` `\0` `\\` `\"` `\'`. Any other escape is an error.

### 14.4 Related documents

- [tour.md](tour.md), the language by example.
- [coming-from-rust.md](coming-from-rust.md), a guide for Rust programmers.
- [memory-safety.md](memory-safety.md), the safety guarantees in depth.
- [philosophy.md](philosophy.md), the design rationale.
- [architecture.md](architecture.md), the compiler pipeline.
