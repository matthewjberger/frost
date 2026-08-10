# Why the syntax reads this way

Frost's syntax is chosen so that code is cheap to parse, cheap to grep, and hard
to write almost-right. Every rule below follows from three decisions: fewer
symbols that mean more than one thing, fewer special-case grammar rules, and
nothing invisible. Rust is the comparison throughout, because it is where most
readers are coming from and because it often chose the other way.

A few points below are properties of the grammar that the implementation has not
fully caught up to. Those are marked inline, and everything else describes the
language as it compiles today. For the normative rules see
[the reference](../reference/conformance.md), and for the form-by-form Rust
mapping, including the table of everything below in one place, see
[coming-from-rust.md](../coming-from-rust.md).

## 1. Uniform declaration syntax

`MAX :: 10`, `add :: fn(...)`, `Point :: struct {...}`, `Shape :: enum {...}`,
and `name :: extern fn(...)` are all the same grammar production,
`identifier :: value`. Rust has five different keyword-first forms (`const`,
`fn`, `struct`, `enum`, and `extern` blocks), each with its own parse rules.

The name always comes first and left-aligned. Answering "where is Point
defined" is searching for `Point ::`, no matter what kind of thing Point is. In
Rust you need to know what kind of item you are looking for before you can grep
for it.

One grammar rule instead of five: a smaller parser, a smaller spec, and one
pattern to write.

## 2. Functions as values, anonymous functions for free

In Rust a named function and an anonymous function are two different grammatical
things:

```rust
fn add(a: i64, b: i64) -> i64 { a + b }    // item syntax, keyword-first
let f = |a: i64, b: i64| -> i64 { a + b }; // closure syntax, pipes
```

The `fn` form is not an expression. You cannot lift it out of item position and
drop it into an expression to get a value, so Rust needed a second, separate
syntax (the pipe form) for a function as a value, with its own type story
(closures are anonymous, unnameable types, distinct from `fn` pointers) and
coercion rules between the two.

In Frost, the named function is already a binding of a value:

```frost
add :: fn(a: i64, b: i64) -> i64 { a + b }
```

This is the same shape as `MAX :: 10`. The right-hand side,
`fn(a: i64, b: i64) -> i64 { a + b }`, is a complete function-valued expression.
The binding names it. So the anonymous form falls out of the grammar. Delete
the name and what remains is already a legal expression.

```frost,sketch
apply(fn(a: i64, b: i64) -> i64 { a + b }, 3, 4)

callbacks := [
    fn(x: i64) -> i64 { x + 1 },
    fn(x: i64) -> i64 { x * 2 },
]
```

There is one function-literal syntax, and a named function is that literal given
a name.

*Implementation status.* The forms above compile and run on both native
backends. An anonymous function literal is lifted to a synthetic top-level
function and referenced by its address, so passing one inline or binding it to a
name both work. There is no capture, which makes this lambda lifting.

The caveat is capture. Anonymous functions come free, and closures in the
capturing sense do not. Whether `fn(x) { x + y }` may capture `y` from the
enclosing scope, and how (by value, by borrow, with what lifetime), is a
semantic decision the uniform syntax does not answer. Jai and Odin, which use
this syntax, mostly punt on capture. Nested function literals cannot close over
locals, precisely because capture drags in the ownership questions that forced
Rust's closure machinery (`Fn`/`FnMut`/`FnOnce`, `move`) into existence. A
borrow-checked language has to pick, either no capture (functions are plain
pointers) or explicit capture lists, which add syntax but keep everything
written down. Frost takes the first path.

## 3. `:=` for declaration, `=` for assignment

In Rust `let x = 5` and `x = 5` do different things but look almost
identical, differing only by a keyword. Writing `let` twice is legal and
silently creates a second variable that shadows the first.

Frost spells introduction (`:=`) and mutation (`=`) with different operators, so
the two intents are distinct in the grammar instead of distinguished by a
keyword. Redeclaring a variable, or assigning to one that was never introduced,
can then be a diagnostic instead of a silent success.

*Implementation status.* Assigning with `=` to a name that was never introduced
is already a located compile error. Redeclaring a name with `:=` still shadows,
as it does in Rust. Whether to forbid that shadowing is an open design choice the
distinct operators make easy to enforce either way.

## 4. `=` for struct fields frees `:` for types

`Point { x = 1, y = 2 }` uses `=` for field initialization, so `:` only ever
means type ascription. Rust overloads `:` for both type annotation and struct
field init, which is part of why Rust never shipped general type ascription. The
grammar collides. One symbol, one meaning helps a parser and a reader alike.

## 5. Mandatory parentheses on conditions

`if (x > 5)` rather than `if x > 5`. It sounds like a downgrade, but Rust's
paren-free `if` created a real ambiguity. In `if x == Foo { }`, is `Foo {` the
start of a struct literal or the start of the if-body? Rust resolves it with a
special rule banning struct literals in condition position. `if (cond) { }` is
context-free with no such carve-out. Slightly more typing, one fewer special
rule. (Frost still uses one small local look-ahead elsewhere, to tell a struct
literal from a `match` body.)

## 6. Postfix deref `p^`

Dereference chains read left to right in evaluation order with no wrapping
parentheses:

```
p^.field
```

Rust's prefix `*` produces `(*p).field` in raw form, which is why Rust added
invisible auto-deref through `.` to make it livable. Postfix deref stays
explicit and stays readable, so the language needs no auto-deref machinery, and
a reader (or a generator) never has to reason about where the compiler inserted
a deref.

## 7. One pointer type `^T`

Rust has `*const T` and `*mut T`. Frost has one raw pointer type, `^T`, and moves
mutability to bindings and borrows where it already lives. Less redundant state
to keep in sync, and less to write.

## 8. `foo($u32)` instead of turbofish

The turbofish `foo::<T>()` exists because `foo<T>()` collides with `a < b > (c)`.
With angle brackets, Rust cannot tell `<` for generics apart from `<` for
less-than in expression position. Frost passes a type as an ordinary argument
marked with `$` (`foo($u32)`), so the `<` disambiguation never arises.

## 9. A multi-return names its values

A function that answers with several values declares a return type list, and
every value in the list has a name:

```frost,sketch
split :: fn(value: i64) -> (high: i64, low: i64) {   // the type list
    return value / 256, value % 256                  // by order
    return { high = value / 256, low = value % 256 } // by name
}
high, low := split(4096)                             // taken apart
```

Return by order where the function is a table of answers: `mnemonic_of` in
`selfhosted/assemble.frost` is forty-five consecutive lines of
`return M_ADDQ, 0`, and naming the fields on each would triple its width.
Return by name where two values share a type and could be swapped without
anyone noticing, as in `mat4_inverse`, which answers `(inverse: Mat4, ok: bool)`.

The parentheses on the type list are load-bearing. Without them, a signature
carrying `uses Arena<256>` puts two comma-separated lists back to back, and a
return type that is itself `fn(i64) -> i64` makes a comma in return position
ambiguous.

An unnamed list, `-> (i64, i64)`, is refused:

```
a return type list names every value; write `-> (name: T, name: T)`
```

Go, Odin, Jai, C# and Swift all make the names optional. Rust and Zig have no
multi-return at all: you return a struct, and its fields are named. Where the
question has been revisited the direction is toward names, and C# 7 added named
tuple elements because `Item1` and `Item2` failed readers. Frost has no
positional struct literal and no positional variant payload, so an unnamed
return list would be the one aggregate whose fields nobody names.

Use `_` for a value you have no use for:

```frost,sketch
high, _ := split(4096)
_, low := split(770)
```

The value is still read, into storage the compiler names, so a linear value
taken by a `_` still owes a consumer.

The struct the list becomes has no name a program can write, and there is no
tuple type. Exposing that struct would make `(A, B)` a type, and a pair could
then travel through a program with nobody naming the aggregate.

## 10. A stated layout is a word

`packed struct` and `field: T align(16)` say what the memory looks like. Three
spellings are possible: an attribute, a sigil, or a word.

An attribute, `#[repr(packed)]`, needs a second grammar with its own bracket, and
that grammar then invites everything else that could be attached to a
declaration. Frost has no attributes, and adding one for a single feature opens
a whole namespace.

A sigil is shorter. `~struct` is a symbol a reader has to look up, and it
collides with the rule that a symbol in Frost is an operator.

A word needs no new grammar. `packed` sits where `linear` sits, which is the
position every marker on a type declaration already takes, so a reader who knows
one knows the other. Neither word is reserved: `packed` marks the declaration
only where `struct` follows it, and `align` only where `(` follows it. Reserving
them would cost every program that has a local called `packed`, and
`std/slab.frost` is one. The shape after a word decides what the word means, the
same way `flags`, `value`, `test` and `export` read.

There is one form for alignment, on a field, and none for the declaration. A
struct's alignment is the widest its fields ask for, so a second form saying the
same thing about the whole type would let a program write two answers to one
question.

## 11. A compile-time call is marked by where it is written

Zig writes `comptime`, C++ writes `constexpr`, Jai writes `#run`. Frost writes
nothing: `LANES :: round_up(300, 64)` is a constant, a constant is worked out
before the program runs, and that is the whole rule.

The alternative is a marker on the function, and a marker on a function is a
second kind of function. `constexpr` says the body is *allowed* to run early,
which means every library author has to decide, for every function, whether to
promise it. A promise that can be made can be broken, so it becomes an ABI
question, and the answer has been rewritten in three C++ standards.

The position already carries the information. A constant's value and an array's
length are the two places a compile-time value is read, and both are worked out
before the program runs. The vocabulary there is a call as well as arithmetic.
The same function is called normally wherever a program calls it normally.

The cost is that a call the compiler cannot work out is refused, with no
fallback to running it later. Falling back would make `LANES` a number in one
place and a call in another, which is two meanings for one declaration.

## 12. A vector is an array

Zig writes `@Vector(4, f32)`, Odin writes `#simd[4]f32`, Rust writes
`Simd<f32, 4>`. Frost writes `[4]f32`, which is the array it already had.

A separate vector type would need its own layout rule, its own ABI answer, its
own coercion to and from the array it is shaped like, and a spelling. Every one
of those is a place for the two to disagree. The array already answers all four,
and a fixed array of numbers is exactly what a vector register holds, so the
type carries no information the array does not.

Elsewhere the separate type buys a place to hang the operators without giving
arrays elementwise arithmetic. Frost gives arrays the arithmetic instead.
`+` over two arrays is otherwise an error, so nothing else changes meaning.

Two rules keep it from becoming a hidden loop. The length is a power of two and
the whole vector is at most sixty-four bytes, which is a register's worth. Past
that an operator would be a loop the reader never wrote, and hidden control flow
is ruled out everywhere else here.

A program wanting elementwise arithmetic over a thousand numbers writes the
loop. That is the same cost the length rule imposes, and the loop was going to
happen either way.

## 13. A `case` says what it covers, and nothing else

An arm may name several patterns joined by `|`, and an arm over whole numbers
may name a span. Both follow one rule: a match's coverage is readable at its
arms.

Without a span, the way to write `case 1..10:` is a wildcard arm holding an
`if`, which is an arm that misstates its own coverage: the wildcard claims
everything else while the body splits it invisibly. `case 1..10:` puts the span
where coverage lives, so a grep over `case` shows the whole shape of the match.
Both stay decidable: the covered set of an
alternative list is the union of its parts, and a span covers what it names, so
the rule about covering every variant goes on counting.

The same rule decides three more questions. A name in a pattern is the value it
stands for, never a binding: `case CH_0:` next to `case CH_0..=CH_9:` cannot
mean compare in one and bind in the other, and `_` is the arm that covers the
rest. A decimal and a piece of text are refused, because covering one of the
reals is a claim nobody can act on and text is compared rather than counted. And
an arm is read against the union of every arm above it, as a reader does looking
down the arms, so an arm nothing reaches is named where it is written whether
one earlier arm took its values or three did between them.

There are no guards. `case n if n > 5:` puts an expression in pattern position,
and a guarded arm covers nothing the compiler can count, so exhaustiveness would
need a second rule for arms it has to ignore. Write the `if` inside the arm.

Patterns do not nest either. `case Shape::Line { start: Shape::Point { x } }` binds through
two levels. A second `match`, or a `.` on the bound field, says the same thing
and leaves every field access where a grep for the field name finds it.

## What the syntax costs

Bounds come from a closed list. `$T` can carry a `where` clause after the
signature, drawn from `is_numeric`, `is_integer`, `is_float`, `is_struct`,
`is_array`, `is_slice`, `is_pointer` and `is_linear`, combined with `&&`, `||`
and `!`. Each one is a question the compiler already answers for itself, to pick
an integer or a floating point instruction, to decide how wide a value is and
whether it travels by address, and to know what has to be consumed. A program
cannot extend the list, so a requirement outside it has nowhere to be written
down and shows up at instantiation instead, pointing into the generic's body
instead of at the caller's line. Opening the vocabulary would mean bound
solving, coherence, and the front-end cost of both.

A compile-time function parameter narrows that: `$before: fn(T, T) -> bool`
declares the signature it needs, and a mismatch is reported against the
parameter list. A capability bundle covers what can be *done* with a type. The
full rules are in [generics.md](../reference/generics.md).

Mandatory parentheses and explicit consumers cost typing, and buy a grammar with
no carve-outs in it.

Capture is unanswered. The uniform function syntax gives anonymous functions and
leaves the closure-capture question open, and answering it later means either no
capture or capture lists.

## Rust and Frost side by side

| Property | Rust | Frost |
|---|---|---|
| Declaration forms | 5 keyword-first grammars | 1 (`name :: value`) |
| Grep-ability of definitions | Must know item kind | `Name ::` finds anything |
| Anonymous functions | Separate closure syntax | Declaration form minus the name |
| Decl vs assignment | Keyword (`let`), silent shadowing | `:=` vs `=`, distinct in the grammar |
| Meaning of `:` | Types and field init | Types only |
| `if` grammar | Special struct-literal ban | Parenthesized condition |
| Deref chains | Auto-deref magic | Explicit postfix `^` |
| Raw pointer types | Two | One |
| Generic call syntax | Turbofish workaround | `$` sigil, no ambiguity |
