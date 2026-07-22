# Frost vs Rust: Syntax Design Advantages

This is an analysis of the syntax differences between Rust and Frost, read
through one lens. Frost is a minimal, borrow-checked, procedurally oriented
language designed so that code is cheap to parse, cheap to grep, and hard to
generate subtly wrong. That last property matters when a large language model is
a primary author, but none of it is model-specific. It is the ordinary payoff of
fewer symbols that mean multiple things, fewer special-case grammar rules, and
nothing invisible.

A note on honesty. A few points below are properties of the grammar and the
design that the implementation has not fully caught up to yet. Those are marked
inline. Everything else describes the language as it compiles today. For the
normative rules see [spec.md](spec.md). For a broader Rust-to-Frost guide see
[coming-from-rust.md](coming-from-rust.md).

## The Rosetta table

| Rust | Frost |
|---|---|
| `let x = 5;` | `x := 5` |
| `let mut x = 5;` | `mut x := 5` |
| `let x: i64 = 5;` | `x : i64 = 5` |
| `const MAX: i64 = 10;` | `MAX :: 10` |
| `fn add(a: i64, b: i64) -> i64 { a + b }` | `add :: fn(a: i64, b: i64) -> i64 { a + b }` |
| `struct Point { x: i64, y: i64 }` | `Point :: struct { x: i64, y: i64 }` |
| `enum Shape { Circle { r: i64 }, .. }` | `Shape :: enum { Circle { r: i64 }, .. }` |
| `Point { x: 1, y: 2 }` | `Point { x = 1, y = 2 }` |
| `match s { Shape::Circle { r } => .. }` | `match s { case .Circle { r }: .. }` |
| `if x > 5 { a } else { b }` | `if (x > 5) { a } else { b }` |
| `for i in 0..n { }` | `for i in 0..n { }` |
| `while cond { }` | `while (cond) { }` |
| `&x`, `&mut x` | nothing: a borrow is what a parameter mode means |
| `&T`, `&mut T` (parameter) | `x: T` (read), `mut x: T` (write) |
| `*p` (deref) | `p^` |
| `*const T`, `*mut T` | `^T` |
| `fn(i64) -> i64` (fn pointer) | `fn(i64) -> i64` |
| `Box<T>` / `Rc<T>` / arena index | `Handle<T>` into a pool |
| `impl Drop for T` | `T :: linear struct { .. }` plus a consumer |
| generics with `<T: Trait>` | `$T` type parameters, no bounds |
| `foo::<u32>()` (turbofish) | `foo($u32, ..)` |
| `extern "C" { .. }` | `name :: extern fn(..) -> ..` |

## 1. Uniform declaration syntax

`MAX :: 10`, `add :: fn(...)`, `Point :: struct {...}`, `Shape :: enum {...}`,
and `name :: extern fn(...)` are all the same grammar production,
`identifier :: value`. Rust has five different keyword-first forms (`const`,
`fn`, `struct`, `enum`, and `extern` blocks), each with its own parse rules.

The uniform form means the following.

**The name always comes first and left-aligned.** Answering "where is Point
defined" is searching for `Point ::`, no matter what kind of thing Point is. In
Rust you need to know what kind of item you are looking for before you can grep
for it.

**One grammar rule instead of five.** A smaller parser, a smaller spec, and for
a model author, one pattern to emit rather than five keyword orderings to keep
straight.

**Functions become ordinary values by construction.** See the next section.

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

```
add :: fn(a: i64, b: i64) -> i64 { a + b }
```

This is the same shape as `MAX :: 10`. The right-hand side,
`fn(a: i64, b: i64) -> i64 { a + b }`, is a complete function-valued expression.
The binding just names it. So the anonymous form falls out of the grammar. Delete
the name and what remains is already a legal expression.

```
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

**Caveat. Functions, not closures.** This gives anonymous functions for free,
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

In Rust `let x = 5` and `x = 5` do very different things but look almost
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
grammar collides. One symbol, one meaning is exactly the kind of local
unambiguity that helps both parsers and model authors. When you see `:` in Frost
it always means the same thing.

## 5. Mandatory parentheses on conditions

`if (x > 5)` rather than `if x > 5`. It sounds like a downgrade, but Rust's
paren-free `if` created a real ambiguity. In `if x == Foo { }`, is `Foo {` the
start of a struct literal or the start of the if-body? Rust resolves it with a
special rule banning struct literals in condition position. `if (cond) { }` is
context-free with no such carve-out. Slightly more typing, a meaningfully simpler
rule. (Frost still uses one small local look-ahead elsewhere, to tell a struct
literal from a `match` body, so this trade buys simplicity, not its total
absence.)

## 6. Postfix deref: `p^`

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

## 7. One pointer type: `^T`

Rust has `*const T` and `*mut T`. Frost has one raw pointer type, `^T`, and moves
mutability to bindings and borrows where it already lives. Less redundant state
to keep consistent, less to write, less to keep in sync.

## 8. Inferred variant shorthand: `.Circle`

When the compiler already knows the scrutinee is a `Shape`, you do not repeat
`Shape` in the pattern:

```
match s {
    case .Circle { radius }: ...
    case .Rect { width, height }: ...
}
```

This Swift-style inferred enum scoping is fewer tokens, and pattern code does not
break when the enum is renamed. Every arm keeps working.

## 9. `foo($u32)` instead of turbofish

The turbofish `foo::<T>()` exists because `foo<T>()` collides with `a < b > (c)`.
With angle brackets, Rust cannot tell `<` for generics apart from `<` for
less-than in expression position. Frost passes a type as an ordinary argument
marked with `$` (`foo($u32)`), sidestepping the `<` disambiguation entirely, one
of the ugliest corners of Rust's grammar.

## 10. Linear structs instead of `Drop`

This one is semantic more than syntactic, but the surface effect matters. Nothing
invisible runs at scope exit. Rust's `Drop` fires automatically and silently when
a value leaves scope. A Frost `linear` type makes you call the cleanup yourself,
and forgetting to consume a linear value is a type error at the point of the leak
rather than an implicit destructor quietly firing. For generated code,
"everything that happens is written down" is an auditability win. Nothing happens
that is not literally in the source.

## 11. No visibility modifiers

Frost has no `pub`, no private, and no visibility keywords at all. Every struct
field is public, and there is nothing to specify. Rust threads `pub` through
fields, functions, modules, and re-exports, with `pub(crate)` and `pub(super)`
refinements on top. Frost drops the entire axis. A struct is its fields and they
are all reachable, so there is one fewer decision per field and one fewer piece
of grammar. This also means the module system needs no visibility rules. Bringing
a file in with `import` makes its names available, and that is the whole story.

## Honest tradeoffs

**Unbounded generics.** `$T` with no bounds means generic errors surface at
instantiation, not declaration. This is the C++ template experience that Rust's
trait bounds were designed to fix, where error messages point into the generic's
body instead of at the call site's contract. If the corpus author is a model
iterating against compiler output, that may be acceptable, but it is a real cost.

**Ergonomics traded for predictability.** Mandatory parentheses and explicit
consumers cost some human ergonomics in exchange for machine predictability. That
is a coherent trade given the audience, but it is a trade, not a free win.

**Capture semantics still owed.** The uniform function syntax defers rather than
solves the closure-capture question. A borrow-checked language eventually has to
answer it, and the answer will reintroduce either a restriction (no capture) or
some syntax (capture lists).

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
| Cleanup | Invisible `Drop` at scope exit | Explicit consumer, enforced by linearity |
| Visibility | `pub`, `pub(crate)`, private by default | None, all fields public |

Almost every difference reduces context-sensitivity, overloaded symbols, or
invisible compiler behavior. That is the design thesis. Code that is cheap to
parse, cheap to grep, and hard to generate almost-right-but-subtly-wrong.
