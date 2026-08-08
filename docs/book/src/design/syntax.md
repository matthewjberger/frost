# Why the syntax reads this way

Frost's syntax is chosen so that code is cheap to parse, cheap to grep, and hard
to write almost-right. Every rule below is the payoff of the same three
decisions: fewer symbols that mean more than one thing, fewer special-case
grammar rules, and nothing invisible. Rust is the comparison throughout, because
it is where most readers are coming from and because it made the opposite choice
often enough to be instructive.

A few points below are properties of the grammar that the implementation has not
fully caught up to. Those are marked inline; everything else describes the
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

One grammar rule instead of five. A smaller parser, a smaller spec, and one
pattern to write rather than five keyword orderings to keep straight.

Functions become ordinary values by construction. See the next section.

## 2. Functions as values, anonymous functions for free

In Rust a named function and an anonymous function are two different grammatical
things:

```rust
fn add(a: i64, b: i64) -> i64 { a + b }    // item syntax, keyword-first
let f = |a: i64, b: i64| -> i64 { a + b }; // closure syntax, pipes
```

The `fn` form is not an expression. You cannot lift it out of item position and
drop it into an expression to get a value, so Rust needed a second, separate
syntax (the pipe form) for "function as a value," with its own type story
(closures are anonymous, unnameable types, distinct from `fn` pointers) and
coercion rules between the two.

In Frost, the named function is already a binding of a value:

```frost
add :: fn(a: i64, b: i64) -> i64 { a + b }
```

This is the same shape as `MAX :: 10`. The right-hand side,
`fn(a: i64, b: i64) -> i64 { a + b }`, is a complete function-valued expression.
The binding just names it. So the anonymous form falls out of the grammar. Delete
the name and what remains is already a legal expression.

```frost,sketch
apply(fn(a: i64, b: i64) -> i64 { a + b }, 3, 4)

callbacks := [
    fn(x: i64) -> i64 { x + 1 },
    fn(x: i64) -> i64 { x * 2 },
]
```

There is one function-literal syntax, and "named function" is that literal given
a name. Nobody had to design anonymous functions as a feature. They are what the
grammar produces when you omit the name.

*Implementation status.* The forms above compile and run on both native
backends. An anonymous function literal is lifted to a synthetic top-level
function and referenced by its address, so passing one inline or binding it to a
name both work. There is no capture, which is what keeps this pure lambda
lifting rather than a closure.

Caveat. Functions, not closures. This gives anonymous functions for free,
but not closures in the capturing sense. Whether `fn(x) { x + y }` may capture
`y` from the enclosing scope, and how (by value, by borrow, with what lifetime),
is a semantic decision the uniform syntax does not answer. Jai and Odin, which
use this syntax, mostly punt on capture. Nested function literals cannot close
over locals, precisely because capture drags in the ownership questions that
forced Rust's closure machinery (`Fn`/`FnMut`/`FnOnce`, `move`) into existence. A
borrow-checked language has to pick, either no capture (functions are plain
pointers, maximally simple), or explicit capture lists, which reintroduce some
syntax but keep the "everything is written down" property. Frost currently takes
the first path.

## 3. `:=` for declaration, `=` for assignment

In Rust `let x = 5` and `x = 5` do different things but look almost
identical, differing only by a keyword. Writing `let` twice is legal and
silently creates a second variable that shadows the first.

Frost spells introduction (`:=`) and mutation (`=`) with different operators, so
the two intents are distinct in the grammar rather than distinguished by a
keyword. This is the mechanism by which redeclaring a variable, or assigning to
one that was never introduced, can be turned into a diagnostic instead of a
silent success.

*Implementation status.* Assigning with `=` to a name that was never introduced
is already a located compile error. Redeclaring a name with `:=` still shadows,
as it does in Rust. Whether to forbid that shadowing is an open design choice the
distinct operators make easy to enforce either way.

## 4. `=` for struct fields frees `:` for types

`Point { x = 1, y = 2 }` uses `=` for field initialization, so `:` only ever
means type ascription. Rust overloads `:` for both type annotation and struct
field init, which is part of why Rust never shipped general type ascription. The
grammar collides. One symbol, one meaning is the kind of local unambiguity that
helps a parser and a reader alike. When you see `:` in Frost it always means the
same thing.

## 5. Mandatory parentheses on conditions

`if (x > 5)` rather than `if x > 5`. It sounds like a downgrade, but Rust's
paren-free `if` created a real ambiguity. In `if x == Foo { }`, is `Foo {` the
start of a struct literal or the start of the if-body? Rust resolves it with a
special rule banning struct literals in condition position. `if (cond) { }` is
context-free with no such carve-out. Slightly more typing, a meaningfully simpler
rule. (Frost still uses one small local look-ahead elsewhere, to tell a struct
literal from a `match` body, so this trade buys simplicity, not its total
absence.)

## 6. Postfix deref `p^`

Dereference chains read left to right in evaluation order with no wrapping
parentheses:

```
p^.field
```

Rust's prefix `*` produces `(*p).field` in raw form, which is why Rust added
invisible auto-deref through `.` to make it livable. Postfix deref is explicit
and ergonomic at once, so the language needs no auto-deref machinery, and a
reader (or a generator) never has to reason about where the compiler inserted a
deref.

## 7. One pointer type `^T`

Rust has `*const T` and `*mut T`. Frost has one raw pointer type, `^T`, and moves
mutability to bindings and borrows where it already lives. Less redundant state
to keep in sync, and less to write.

## 8. `foo($u32)` instead of turbofish

The turbofish `foo::<T>()` exists because `foo<T>()` collides with `a < b > (c)`.
With angle brackets, Rust cannot tell `<` for generics apart from `<` for
less-than in expression position. Frost passes a type as an ordinary argument
marked with `$` (`foo($u32)`), sidestepping the `<` disambiguation entirely, one
of the ugliest corners of Rust's grammar.

## 9. How many spellings a multi-return has

A function that answers with several values is written four ways over its
lifetime:

```frost,sketch
split :: fn(value: i64) -> (high: i64, low: i64) {   // the type list
    return value / 256, value % 256                  // by order
    return { high = value / 256, low = value % 256 } // by name
}
high, low := split(4096)                             // taken apart
```

Four spellings of one idea is more ceremony than any keyword in the language
carries, so each was asked what it pays for. What the parsers say first:

- The braced `return` is the inferred struct literal of 6.5, over the struct
  the list becomes. The bootstrap rewrites an unnamed `StructInit` to name that
  struct; the self-hosted parser sets the expected type to it and reads an
  ordinary literal. So this spelling costs no grammar of its own.
- The destructure is a statement form until lowering and nothing afterwards.
  The bootstrap carries a `LetMultiple` node that `lower_multiple_returns`
  expands into a binding of a temporary plus one field read per name; the
  self-hosted parser emits those statements as it reads the names, so it has no
  node at all.
- The struct has no name a program can write. The bootstrap derives one from
  the rendered types, so two functions with the same list share it; the
  self-hosted compiler makes one per function. Both refuse any attempt to write
  the name down.

The type list stays: without the parentheses a signature carrying
`uses Arena<256>` after it puts two comma-separated lists back to back, and a
return type that is itself `fn(i64) -> i64` makes a comma in return position
ambiguous. The destructure is the payoff and has no cheaper form.

Both `return` forms stay, and the corpus is what settled it. `mat4_inverse` in
`std/math.frost` answers `(inverse: Mat4, ok: bool)` and `tally` in
`examples/tour.frost` answers `(total: i64, strongest: i64)`, where two `i64`
values could silently swap and the names are the guard. `mnemonic_of` in
`selfhosted/assemble.frost` is forty-five consecutive lines of
`return M_ADDQ, 0`, and writing those as `return { op = M_ADDQ, cc = 0 }` makes
a lookup table three times as wide for nothing. Forcing either form out makes
real code in the tree worse, so both earn their place.

What does not earn its place is the unnamed list, `-> (i64, i64)`. Its only
effect would be to call the fields `value0` and `value1`, names the compiler
picked and no program is allowed to write. **A return type list names every
value**, which deletes a rule rather than adding one: there is no `valueN`
synthesis in either parser, and no refusal guarding it.

```
a return type list names every value; write `-> (name: T, name: T)`
```

Frost is alone in that. Go, Odin, Jai, C# and Swift all make the names optional,
and Rust and Zig sidestep the question by having no multi-return at all: you
return a struct, and its fields are named. The direction of travel where it has
been revisited is toward names, though. C# 7 added named tuple elements because
`Item1` and `Item2` failed readers, which is the same failure as `value0`, and
Go's optionality is tangled with naked return, a job Frost's names do not have.
The precedent that decides it is internal: Frost has no positional struct
literal and no positional variant payload, so the return type list was the one
aggregate whose fields a person had not named.

At the other end, `_` takes a value the caller has no use for:

```frost,sketch
high, _ := split(4096)
_, low := split(770)
```

The list binds one name per value, so without this a caller wanting the first
has to invent a name for the rest, and the corpus did exactly that three times
under the name `unused`, which is a live binding somebody can read by mistake.
The value is still read into storage the compiler names, so a linear one taken
by a `_` is still owed a consumer.

The alternative worth naming and refusing is exposing the synthesized struct as
a nameable type, which would make the multi-return an ordinary struct return
with sugar at the call site. It removes a concept rather than renaming one, and
it contradicts goal 1: `(A, B)` would become a type, and a program could pass a
pair around without anyone naming the aggregate. The two compilers also derive
different names and share the struct on different terms, so exposing it means
picking one and rewriting the other.

## 10. A stated layout is a word, not a sigil or an attribute

`packed struct` and `field: T align(16)` say what the memory looks like. Three
spellings were available and the surface already answers which one to take.

An attribute, `#[repr(packed)]`, needs a second grammar with its own bracket, and
that grammar then invites everything else that could be attached to a
declaration. Frost has no attributes and adding one for a single feature buys a
whole namespace nobody asked for.

A sigil is shorter and unreadable. `~struct` is a symbol a reader has to look up,
and it collides with the rule that a symbol in Frost is an operator.

A word costs nothing. `packed` sits where `linear` sits, which is the position
every marker on a type declaration already takes, so a reader who knows one
knows the other. Neither word is reserved: `packed` marks the declaration only
where `struct` follows it, and `align` only where `(` follows it. Reserving them
would cost every program that has a local called `packed`, and `std/slab.frost`
is one. The shape after a word is what says what it means, which is how `flags`,
`value`, `test` and `export` already read.

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
length are the two places a compile-time value is read, and both were already
worked out before the program ran; what changed is that the vocabulary there is
now a call rather than only arithmetic. The same function is called normally
wherever a program calls it normally.

What that costs is that a call which cannot be worked out is a refusal rather
than a fallback to running it later. That is the trade taken on purpose: falling
back would mean `LANES` was a number in one place and a call in another, which
is two meanings for one declaration.

## 12. A vector is an array, not a type of its own

Zig writes `@Vector(4, f32)`, Odin writes `#simd[4]f32`, Rust writes
`Simd<f32, 4>`. Frost writes `[4]f32`, which is the array it already had.

A separate vector type would need its own layout rule, its own ABI answer, its
own coercion to and from the array it is shaped like, and a spelling. Every one
of those is a place for the two to disagree. The array already answers all four,
and a fixed array of numbers is exactly what a vector register holds, so the
type carries no information the array did not.

What the separate type buys elsewhere is a place to hang the operators without
giving arrays elementwise arithmetic. Frost gives arrays the arithmetic instead,
which costs nothing: `+` over two arrays was an error, so no program changes
meaning.

Two rules keep it from becoming a hidden loop. The length is a power of two and
the whole vector is at most sixty-four bytes, which is a register's worth. Past
that an operator would be a loop the reader never wrote, and hidden control flow
is what this document rules out everywhere else.

The trade is that a program wanting elementwise arithmetic over a thousand
numbers writes the loop. That is the same trade the length rule makes, and the
loop is the thing that was going to happen either way.

## 13. A `case` says what it covers, and nothing else

An arm may name several patterns joined by `|`, and an arm over whole numbers
may name a span. Both were added on one axis: a match's coverage should be
readable at its arms.

The span is where that shows. Without it the way to write `case 1..10:` is a
wildcard arm holding an `if`, which is an arm that misstates its own coverage:
the wildcard claims everything else while the body splits it invisibly. `case
1..10:` puts the span where coverage lives, so a grep over `case` shows the
whole shape of the match. Both stay decidable: the covered set of an
alternative list is the union of its parts, and a span covers what it names, so
the rule about covering every variant goes on counting.

The same axis settled three older questions the same way. A name in a pattern
is the value it stands for, never a binding: `case CH_0:` next to
`case CH_0..=CH_9:` cannot mean compare in one and bind in the other, and `_`
is the arm that covers the rest. A decimal and a piece of text are refused,
because covering one of the reals is a claim nobody can act on and text is
compared rather than counted. And an arm is read against the union of every arm
above it, which is what a reader does looking down the arms, so an arm nothing
reaches is named where it is written whether one earlier arm took its values or
three did between them.

There are no guards. `case n if n > 5:` puts an expression in pattern position,
and a guarded arm covers nothing the compiler can count, so exhaustiveness would
need a second rule for arms it has to ignore. Write the `if` inside the arm.

Patterns do not nest either. `case .Line { start: .Point { x } }` binds through
two levels; a second `match`, or a `.` on the bound field, says the same thing
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

## Summary

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

Almost every difference reduces context-sensitivity, overloaded symbols, or
invisible compiler behavior. That is the design thesis: code that is cheap to
parse, cheap to grep, and hard to write almost-right.
