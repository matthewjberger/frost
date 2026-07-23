# Frost for Rust Programmers

This guide explains Frost to someone who already thinks in Rust. It assumes you
are comfortable with ownership, borrows, lifetimes, traits, `Drop`, and
monomorphization, and it spends its time on where Frost agrees with Rust, where
it deliberately diverges, and how to translate the Rust idioms you reach for by
reflex.

Read [philosophy.md](philosophy.md) for the reasoning behind the design and
[memory-safety.md](memory-safety.md) for the safety argument in full. This
document is the practical bridge.

## The one idea to internalize first

Rust makes references safe with a borrow checker built on **lifetimes**. A
reference carries a region `'a`, and the compiler proves no reference outlives
its referent. That machinery is the price of letting references be first-class
values you can store in structs, return from functions, and thread through data
structures.

Frost makes a different trade. Borrows are **second-class**, and they are not a
type at all: there is no `&` in the language. How a parameter is passed is
written on the parameter (`p: T` reads, `mut p: T` mutates, `move p: T` takes),
and the call site writes nothing. Because a borrow can only ever be a parameter,
it cannot be stored in a field, put in an array, or returned, and the shapes that
would let it escape are not expressible rather than merely rejected. So there is
nothing to annotate and nothing to infer. Frost has **no lifetimes, no `'a`, no
borrow regions, and no lifetime elision** because it does not need them.

Everything else about the borrow system follows from that single decision.
Where Rust reaches for a reference that must live somewhere (a graph node, a
back-pointer, a cache), Frost reaches for a **generational handle** into a pool
instead. If you keep that substitution in mind, most of the surprises below stop
being surprises.

## The 60-second Rosetta table

| Rust | Frost |
| --- | --- |
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
| `&x`, `&mut x` (at a call) | nothing, the callee's mode decides |
| `fn f(x: &T)`, `fn f(x: &mut T)` | `f :: fn(x: T)`, `f :: fn(mut x: T)` |
| `*p` (deref) | `p^` |
| `*const T`, `*mut T` | `^T`, and `ptr_to(x)` takes one |
| `fn(i64) -> i64` (fn pointer) | `fn(i64) -> i64` |
| `Box<T>` / `Rc<T>` / arena index | `Handle<T>` into a pool |
| `impl Drop for T` | `T :: linear struct { .. }` plus a consumer |
| generics with `<T: Trait>` | `$T` type parameters, unbounded |
| a trait method a generic calls | `$f: fn(..) -> ..` compile-time parameter |
| `foo::<u32>()` (turbofish) | `foo($u32, ..)` |
| `extern "C" { .. }` | `name :: extern fn(..) -> ..` |

## Declarations and bindings

There is no `let`. A name is introduced with one of three operators:

- `x := expr` binds a local with an inferred type. This is your everyday `let`.
- `x : Type = expr` binds a local with an explicit type.
- `NAME :: expr` declares a **constant**, evaluated once. Functions, structs,
  enums, and top-level items are all constants, which is why every function is
  written `name :: fn(..)`.

Bindings are immutable by default, exactly as in Rust. `mut` makes a local
assignable:

```
mut total : i64 = 0
total = total + 1
```

A block is an expression whose value is its trailing expression, the same rule
as Rust. `return` exists for early exit. Statements are separated by newlines.
Semicolons are not required. Line comments start with `//`, as in Rust.

Program entry is `main :: fn() -> i64`, and its return value is the process exit
code, which is why the examples end in a bare `0`.

There are no attributes, no `#[derive(..)]`, and no macros. What you write is
what runs.

## Functions, and the absence of methods

Frost is data-oriented, not object-oriented. There are **no methods, no `self`,
no `impl` blocks, and no traits.** Behavior lives in free functions that take
their data as parameters:

```
Vec3 :: struct { x: i64, y: i64, z: i64 }

dot :: fn(a: Vec3, b: Vec3) -> i64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}
```

Both parameters are borrowed to read, which is what an unmarked parameter means,
so nothing is copied and nothing is consumed.

Where Rust would write `a.dot(&b)`, Frost writes `dot(a, b)`. This is not a
missing feature. It is the design. Separating data from the code that walks it
is what keeps the memory layout visible and the control flow explicit.

Higher-order code uses function pointers, covered below. There are no closures.

## Types and arithmetic

The scalar types are what you expect. They are `i8`, `i16`, `i32`, `i64`,
`isize`, their unsigned `u*` counterparts, `f32`, `f64`, and `bool`. These are
all copy types.

The difference that will bite first is that **arithmetic wraps at the type width
with two's-complement semantics, and is never checked for overflow.** Rust panics on
overflow in debug and wraps in release. Frost always wraps, like Rust's
`wrapping_add` family. A `u8` holding `200` plus `100` is `44`, and an `i32` at
`2000000000` doubled is `-294967296`. Do not rely on overflow being caught.
(There are no `_` digit separators, either.)

Mixed-width integer arithmetic is permitted, with the narrower operand widening
to the wider one, so `an_i32 + an_i64` is an `i64`. This is looser than Rust,
which would reject the mismatch and make you write an `as` cast.

`str` is a byte-slice view, a pointer and a length, and it is the analogue of
Rust's `&str` rather than `String`. It owns nothing, so it is a copy type you can
duplicate freely. A string literal is a `str` into read-only data. `str_len(s)`
is the constant-time length and `s[i]` is a bounds-checked `u8`, the same
indexing rule as arrays. Unlike Rust there is no owned `String` in the language
and no UTF-8 method library, `str` is just bytes. An owned or growable buffer is
something you build as a struct over an array or pool and borrow back as a `str`.

Crossing to C is explicit, since a `str` carries no NUL terminator. The one
shortcut is the string literal, which the compiler also lays down NUL-terminated,
so a literal passed where `^i8` is expected reaches C as a plain pointer at no
cost. That is why the FFI examples below pass `"..."` straight to `printf`.

Aggregates (`struct`, `enum`, fixed arrays) pass and return **by value**,
copied at the call boundary, unless you pass a borrow. There is no implicit
boxing and no hidden heap allocation anywhere.

Fixed-size arrays are written `[N]T` and **every index is bounds-checked**. An
out-of-range access aborts at runtime rather than reading past the end. This is
always on, with no `get`/`get_unchecked` split.

## Structs, enums, and pattern matching

Structs and enums are plain data. Construction uses `=` for fields, not `:`:

```
Point :: struct { x: i64, y: i64 }
p := Point { x = 3, y = 4 }

Shape :: enum {
    Circle { radius: i64 },
    Rect { width: i64, height: i64 },
}
c := Shape::Circle { radius = 5 }
```

Enum variants may be unit variants or carry named fields, and a single enum can
mix both:

```
Kind :: enum { Player, Enemy { damage: i64 }, Pickup { amount: i64 } }
```

`match` is the workhorse, and its arm syntax is `case <pattern>: <expr>`. A
variant pattern leads with a dot and binds fields by name:

```
delta :: fn(k: Kind) -> i64 {
    match k {
        case .Player: 0
        case .Enemy { damage }: 0 - damage
        case .Pickup { amount }: amount
    }
}
```

`match` also works over scalar values and over tuples, with `_` as the
wildcard, which covers the common Rust idioms:

```
label :: fn(score: i64) -> i64 {
    match score {
        case 90: 4
        case 80: 3
        case _: 0
    }
}

fizz :: fn(i: i64) -> i64 {
    match (i % 3, i % 5) {
        case (0, 0): 15
        case (0, _): 3
        case (_, 0): 5
        case (_, _): i
    }
}
```

`match` can be taken over a value or over a reference. Matching a value of a
`linear` type consumes it (see below). There is no `#[derive(Debug)]`,
`PartialEq`, or the rest. Equality and printing are not free, and printing is
usually done by handing values to C's `printf`.

## The borrow system without lifetimes

This is the section a Rust programmer should read twice.

Shared and exclusive borrows mean what they mean in Rust, and the exclusivity
rule is familiar: within a single call you may borrow a variable to read many
times or to mutate exactly once, never both. What is different is where you write
it. There is no `&`. The mode is a property of the parameter:

| Rust | Frost | means |
| --- | --- | --- |
| `fn f(x: &T)` | `f :: fn(x: T)` | borrowed to read |
| `fn f(x: &mut T)` | `f :: fn(mut x: T)` | borrowed to mutate in place |
| `fn f(x: T)` | `f :: fn(move x: T)` | ownership transferred |

The call is `f(x)` in all three cases. Which one it is comes from the signature
you can go read, not from a sigil at the call, and the exclusivity check reads
that signature too.

Borrows are also **second-class**, which here means something stronger than
"rejected": the shapes are not expressible. There is no reference type to write
in a struct field or a return position, so:

- A borrow cannot be stored in a struct or enum field.
- A borrow cannot be returned from a function.
- A borrow cannot be put in an array or otherwise made to outlive the call.

You pass data in by borrow, operate on it, and the borrow dies at the end of the
call. Because it can never escape, the analysis is entirely scope-local. There is
nothing like Rust's `fn longest<'a>(x: &'a str, y: &'a str) -> &'a str` because
you cannot return a borrow at all.

```
scale :: fn(mut p: Point, k: i64) {
    p.x = p.x * k          // field access on a borrowed struct is direct
    p.y = p.y * k
}

main :: fn() -> i64 {
    mut p := Point { x = 3, y = 4 }
    scale(p, 2)            // no '&mut' here
    p.x
}
```

Deref rules to keep straight, since they differ from Rust's `*`:

- On a borrowed aggregate, member access is direct, as in `p.x` where the
  parameter is `mut p: Point`. There is no `(*p).x`.
- On a raw pointer, the postfix `^` operator reads or writes the pointee. Given
  `a: ^i64`, `a^` is the value and `a^ = 7` writes it.

The Rust move you cannot make is "return a borrow into my own data" or "stash a
borrow for later." When you feel that reflex, that is the signal to switch from
borrows to handles.

### Raw pointers are the escape hatch

`^T` is a raw pointer, the analogue of `*const T` / `*mut T`. It is unchecked
and exists for FFI and for building low-level libraries (the pool runtime is
one). You dereference it with `^`, and field access through it is written
explicitly:

```
pe : ^Entity = pool_get(world, handle)
pe^.hp = pe^.hp - 25
```

Raw pointers are where you step outside the safety guarantees, exactly as
`unsafe` raw pointers are in Rust. The difference is that in Frost the common
case (a pool of long-lived objects) is served by safe handles, so you reach for
raw pointers far less often than you might expect.

## Moves, copies, and linear resources

Move semantics match your Rust intuition. A non-copy value (a struct, enum, or
other aggregate) is **moved** when passed by value, assigned, or returned, and
using it afterward is a compile error:

```
buf := make_buffer()
consume(buf)
// consume(buf)   // error: use of moved value 'buf'
```

Copy-ness is decided by the **type category**, not by a `Copy` derive. Scalars,
pointers, references, and handles are copy. Aggregates are move. There is no
`#[derive(Clone)]` and no `.clone()`. If you want a second copy of an aggregate,
you construct one.

The larger divergence is how cleanup works. Frost has **no `Drop`**. In its
place is the `linear` qualifier, which changes the affine rule (use *at most*
once) into a linear rule (use *exactly* once):

```
File :: linear struct { fd: i64 }
open  :: fn(n: i64) -> File { File { fd = n } }
close :: extern fn(f: File)          // terminal consumer, takes ownership

run :: fn() {
    f := open(3)
    close(f)          // consumes f exactly once
    // close(f)        // error: use of moved value
}                      // if f were still live here: error, linear value never consumed
```

A `linear` value that reaches the end of its scope without being consumed is a
compile error, the mirror image of a leaked `Drop`. Consuming means moving it
onward. That means returning it, passing it by value (often to an `extern` that
takes ownership across the FFI boundary), or `match`ing it.

Two consequences a Rust programmer will appreciate:

- Cleanup is a checked obligation you can see in the code, not an implicit call
  that runs at a brace you have to imagine. There is no drop order to reason
  about and no `mem::forget` footgun. Forgetting is simply a compile error.
- A `linear enum` returned from a fallible function **cannot be ignored**. Where
  Rust leans on `#[must_use]` as a lint, Frost makes must-use a type rule.
  The result has to be consumed, so a failure cannot be silently dropped.

### `defer` for scope-exit actions

Because there is no `Drop`, the RAII-guard pattern is replaced by `defer`,
which runs a statement when the scope exits, in last-in-first-out order:

```
work :: fn() {
    defer printf("cleanup\n", 0)   // runs on the way out
    // ... body ...
}
```

Think of it as Go's `defer` rather than a Rust guard object. For resources with
real ownership, prefer a `linear` type. Use `defer` for local, best-effort
scope-exit actions.

## Handles and pools: the replacement for `Rc`, `Arc`, and back-references

This is where you put everything that Rust would model with `Rc<RefCell<T>>`,
`Arc`, a `Vec<T>` plus indices, or a graph of references. Long-lived, shared,
or interlinked data lives in a **pool** and is named by a **`Handle<T>`**, a
small copyable value that is an index plus a generation, not a pointer.

```
pool_new   :: extern fn(capacity: i64, elem_size: i64) -> ^u8
pool_alloc :: extern fn(pool: ^u8, value: ^u8) -> i64
pool_get   :: extern fn(pool: ^u8, handle: i64) -> ^u8

Entity :: struct { hp: i64, mana: i64 }

world := pool_new(16, 16)
mut hero := Entity { hp = 100, mana = 30 }
h : Handle<Entity> = pool_alloc(world, &hero)

printf("%lld\n", world[h].hp)       // 100
world[h].hp = world[h].hp - 25      // pool[handle] is a place you can write
```

`pool[handle]` is a **place**. You can read a field, write a field, copy the
element out, or take a `&`/`&mut` of it. The borrow you get is second-class like
any other, so it cannot escape the pool operation. The subscript lowers to the
pool runtime, so the `pool_*` functions it uses (`pool_get` here) must be
declared as `extern fn`, as shown. The generic pool wrappers in
`examples/native/` package this so you can write `make_pool($Entity, 16)`
instead of wiring the runtime by hand.

The generation is what makes this safe without a borrow checker. Freeing a slot
bumps its generation counter. A handle carries the generation it was minted
with, and a lookup with a stale generation fails instead of reading whoever
reused the slot. This is the same idea as the `slotmap` and `generational-arena`
crates in the Rust ecosystem, promoted to a language primitive. It gives you the
"weak reference that safely goes dangling" behavior without any reference
counting or runtime borrow tracking.

The mental substitution is direct. A `Handle<T>` is what you store in fields and
return from functions, precisely the things a `&T` may not do. A linked list, a
scene graph, or an entity system is a pool of nodes linked by handles.

### The six memory-safety guarantees

Frost claims memory safety without a garbage collector and without lifetimes.
The guarantees, and how each is achieved, are:

| Guarantee | Mechanism |
| --- | --- |
| No dangling references | References are second-class and cannot escape their scope |
| No use-after-move | Move checking on every non-copy value |
| No mutable aliasing | Per-call borrow exclusivity (enough because borrows cannot escape) |
| No leaks of resources | `linear` values must be consumed exactly once |
| No use-after-free of pooled data | Generational handles reject stale lookups |
| No out-of-bounds array access | Every fixed-array index is bounds-checked |

Raw pointers (`^T`) sit outside this set by design. They are the explicit,
opt-in escape hatch, the way `unsafe` raw pointers are in Rust.

## Generics without traits

Generics monomorphize, exactly as in Rust. Each instantiation is compiled to
specialized code with no runtime dispatch. The differences are in how you spell
them and, more importantly, in what you cannot say.

A type parameter is written `$T`. It can appear on functions and on structs:

```
Pair :: struct($T: Type) { first: T, second: T }

make_pair :: fn(a: $T, b: $T) -> Pair<T> { Pair { first = a, second = b } }

swap :: fn(mut a: $T, mut b: $T) {
    t := a
    a = b
    b = t
}
```

The type parameter is usually inferred from a value argument, the way Rust
infers `T` from a call. When it cannot be (for example a function that only uses
`sizeof(T)` and never takes a `T` by value), declare it as `$T: Type` and pass
the type explicitly at the call site with a leading `$`, which is Frost's
equivalent of the turbofish:

```
make_pool :: fn($T: Type, capacity: i64) -> ^u8 {
    pool_new(capacity, sizeof(T))
}

world := make_pool($Entity, 16)     // like make_pool::<Entity>(16)
```

`sizeof(T)` is a compile-time constant, so a generic function can size its own
type parameter. Type parameters are erased after monomorphization and carry no
runtime cost. They drive the specialization and then vanish from the ABI.

There are **no traits, so no `where` clauses, no associated types, and no `dyn
Trait`.** A generic function is generic over any type its body actually
type-checks against once specialized.

Where Rust would write `T: Ord` and call `a.cmp(&b)`, Frost takes the operation
as a compile-time function parameter, and that parameter can declare the
signature it needs:

```
ascending :: fn(a: i64, b: i64) -> bool { a < b }

best :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T, move y: $T) -> $T {
    mut result := x
    if (before(y, result)) { result = y }
    result
}

smallest := best($i64, $ascending, 7, 3)
```

The bound is checked at the call with that call's type arguments substituted in,
so a wrong signature is an error against the parameter list rather than a
message pointing inside the specialized body. The call to `before` is direct,
not through a pointer, because the specialization knows which function it is.
This is the one form of bound in the language, and it is a comparison of one
signature against another, not a solver.

What you still cannot state is a requirement on `T` itself, such as "numeric".
That surfaces when the specialization is compiled. The upside of the whole
arrangement is that there is no trait-resolution machinery, no coherence rules,
and no orphan problem.

## Function pointers, not closures

Functions are values. A parameter of type `fn(..) -> T` holds one, and you call
it directly:

```
apply :: fn(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }
double :: fn(x: i64) -> i64 { x * 2 }

apply(double, 21)    // 42
```

There are **no capturing closures**, and therefore no `Fn` / `FnMut` / `FnOnce`
distinction and no closure environment. Where a Rust closure would capture
state, you pass that state explicitly as another argument. This keeps every
indirect call a plain function pointer with no hidden allocation and no captured
lifetimes to reason about. In practice, callback-style code threads a context
value alongside the function pointer, the same pattern C uses.

Registering a callback **with a C library** is the one case where that pattern
gets language support, because it is the case where the context outlives the
call. It is written as a `$` function parameter on an `extern` plus a context
taken by `move`, and it is closer to Rust's `Box::into_raw` plus a
`extern "C" fn` shim than to a closure: the context is handed over, the caller
cannot touch it while the callback can fire, and getting it back is what
unregistration is for. Unlike the Rust version there is no `unsafe` and no raw
pointer in what you write. See chapter 12.1 of [spec.md](spec.md).

## Compile-time evaluation

Frost has no general compile-time interpreter and no macros. The compile-time
machinery is `sizeof(T)` as a constant and monomorphization, which is driven by
three kinds of `$` parameter: a type (`$T: Type`), an integer (`$N: usize`, which
is Rust's const generics as values and is what sizes a `[N]T` field), and a
function (`$f: fn(..) -> ..`). All three work on functions as well as structs, so
an operation over a sized aggregate is written once rather than once per size,
and unlike Rust's const generics the integer is usable as a plain value in the
body rather than only in a type. A generic function or struct is stamped out once
per distinct set of those arguments at the call site (chapter 11 of the spec),
the way Rust monomorphizes generics. That is the whole of it. There is nothing
here that corresponds to `const fn` or macro expansion.

## Calling C

FFI is a first-class, zero-glue path. Declaring an external function is a
constant whose value is an `extern fn`:

```
printf :: extern fn(fmt: ^i8, value: i64) -> i32
malloc :: extern fn(size: i64) -> ^u8
```

Frost scalar types map to the natural C types, aggregates pass by the platform
ABI, and a `^T` is a C pointer. This is how the examples reach `printf`,
`malloc`, and the pool runtime.

One asymmetry to note, coming from Rust's `extern "C"` and `#[no_mangle]`, is
that **Frost calls C, but C does not call Frost.** There is no stable exported ABI
and no attribute to expose a Frost function to a C caller. The C that the
compiler emits internally is a lowering detail, not an interface. If you need a
library other languages link against, that is out of scope. The asymmetry is
what keeps the backend simple. See [c-compatibility.md](c-compatibility.md) for
the full type mapping.

## Modules

Source is brought together with `import`. There is no `mod`, no `pub`/`use`
visibility system, and no crate graph in the Rust sense. This is the least
developed part of the surface today. For now, think of a program as a set of
files pulled in by `import`, not as a module tree with visibility rules.

## What a Rust program leans on that Frost omits, and what to use instead

| Rust feature | Frost approach |
| --- | --- |
| Lifetimes (`'a`) | Not needed; references are second-class and cannot escape |
| Traits, `impl Trait`, `dyn Trait` | None; pass function pointers, or write concrete code |
| Trait bounds, `where`, associated types | None; generics are unbounded and check per instantiation |
| Methods, `self`, `impl` blocks | Free functions that take the data as a parameter |
| `Drop`, RAII guards | `linear` types (consume exactly once), plus `defer` |
| Closures, `Fn`/`FnMut`/`FnOnce` | Function pointers plus an explicit context argument |
| `Box`, `Rc`, `Arc`, `RefCell` | Pools and `Handle<T>` (generational indices) |
| `Vec`, `HashMap`, `String` | Fixed arrays and pools; no general collections or string library |
| `#[derive(..)]`, macros, attributes | None; write what you need explicitly |
| `?`, `Result`, `#[must_use]` | `linear enum` returns that must be consumed |
| Overflow checks in debug | None; arithmetic always wraps at width |
| `unsafe` blocks and raw pointers | `^T` raw pointers as the explicit escape hatch |
| `pub`, `pub(crate)`, field privacy | None; every struct field is public |
| Async, generics over const, GATs | Out of scope |

## Gotchas checklist for the first hour

- `if` and `while` conditions need parentheses, as in `if (x > 5) { .. }`.
- Struct fields are set with `=`, not `:`, as in `Point { x = 1, y = 2 }`.
- Match arms are `case <pattern>: <expr>`, and variant patterns lead with a dot,
  as in `case .Circle { radius }:`.
- There is no `let`. Use `:=`, `:`, or `::`.
- Every function, type, and constant is declared with `::`.
- To deref a raw pointer, use postfix `^`, as in `a^`, `p^.field`. A borrowed
  parameter needs no sigil, so field access on one is direct, as in `p.field`,
  and assigning to the whole of a `mut` parameter is just `p = q`.
- You cannot return or store a borrow. Use a `Handle<T>` for anything that must
  live beyond the call.
- A `linear` value must be consumed on every path, or it is a compile error.
- Integer arithmetic wraps. Do not rely on overflow being caught.

## A worked example: a tiny entity system

This is the idiom you will use constantly, the Frost answer to a `Vec` of
objects with cross-references. Entities live in a pool, are named by handles,
and are mutated in place through the pool.

```
printf     :: extern fn(fmt: ^i8, value: i64) -> i32
pool_new   :: extern fn(capacity: i64, elem_size: i64) -> ^u8
pool_alloc :: extern fn(pool: ^u8, value: ^u8) -> i64
pool_get   :: extern fn(pool: ^u8, handle: i64) -> ^u8
pool_free  :: extern fn(pool: ^u8, handle: i64) -> i64

Kind :: enum { Player, Enemy { damage: i64 }, Pickup { amount: i64 } }
Entity :: struct { hp: i64, kind: Kind }

delta :: fn(k: Kind) -> i64 {
    match k {
        case .Player: 0
        case .Enemy { damage }: 0 - damage
        case .Pickup { amount }: amount
    }
}

main :: fn() -> i64 {
    world := pool_new(16, 24)

    mut player := Entity { hp = 100, kind = Kind::Player }
    ph := pool_alloc(world, ptr_to(player))
    mut goblin := Entity { hp = 30, kind = Kind::Enemy { damage = 15 } }
    gh := pool_alloc(world, ptr_to(goblin))

    pe : ^Entity = pool_get(world, ph)
    ge : ^Entity = pool_get(world, gh)

    pe^.hp = pe^.hp + delta(ge^.kind)     // player takes the goblin's damage
    printf("%lld\n", pe^.hp)              // 85

    pool_free(world, gh)                  // gh is now stale; its generation bumped
    0
}
```

Notice what is doing the work. Entities are stored by value in the pool,
handles are the things that get passed around and stored, borrows (the argument
to `delta`) last only for the duration of a call, and freeing a slot invalidates old
handles by generation rather than by any lifetime the compiler had to track.
That is the whole model. Once it clicks, the absence of lifetimes stops feeling
like something missing and starts feeling like something removed.

## Where to go next

- [philosophy.md](philosophy.md), why the language is shaped this way.
- [memory-safety.md](memory-safety.md), the safety guarantees in depth.
- [c-compatibility.md](c-compatibility.md), the C type mapping and FFI details.
- [architecture.md](architecture.md), the compiler pipeline, the typed IR, and
  the three backends that must agree.
- `examples/native/`, runnable programs, starting with `game_world.frost` and
  `pool_linked_list.frost`.
```
