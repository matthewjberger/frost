# Frost for Rust programmers

This guide explains Frost to someone who already thinks in Rust. It assumes you
are comfortable with ownership, borrows, lifetimes, traits, `Drop`, and
monomorphization, and it translates the Rust idioms you reach for by reflex.

Read [philosophy.md](design/philosophy.md) for the reasoning behind the design and
[memory-safety.md](design/memory-safety.md) for the safety argument in full.

## Second-class borrows instead of lifetimes

Rust makes references safe with a borrow checker built on lifetimes. A
reference carries a region `'a`, and the compiler proves no reference outlives
its referent. That machinery is the price of letting references be first-class
values you can store in structs, return from functions, and thread through data
structures.

Frost's everyday borrow is second-class. There is no `&` in the language. How a
parameter is passed is written on the parameter (`p: T` reads, `mut p: T`
mutates, `move p: T` takes), and the call site writes nothing. A borrow obtained
that way cannot be stored in a field, put in an array, or returned, and the
shapes that would let it escape have no spelling. So there is nothing to
annotate and nothing to infer. Frost has no lifetimes, no `'a`, no borrow
regions, and no lifetime elision.

The one borrow a program writes down is `ref T`, covered below. It is checked:
it may be returned, and it may not be stored.

Everything else about the borrow system follows from that decision. Where Rust
reaches for a reference that must live somewhere (a graph node, a back-pointer,
a cache), Frost reaches for a generational handle into a slab.

## The Rosetta table

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
| `A \| B => ..` and `1..=9 => ..` | `case .A \| .B:` and `case 1..=9:` |
| `n if n > 5 => ..` | `case _: if (x > 5) { .. }` |
| `n => ..` (bind the value) | `case _:` and read the matched name |
| `if x > 5 { a } else { b }` | `if (x > 5) { a } else { b }` |
| `for i in 0..n { }` | `for i in 0..n { }` |
| `for x in &xs { }` | `for x in xs { }` |
| `for (i, x) in xs.iter().enumerate()` | `for i, x in xs { }` |
| `println!("{}", x)` | `import "io.frost"` and `print("{}\n", x)` |
| `fn f() -> (i64, i64)` (a tuple) | `f :: fn() -> (q: i64, r: i64)`, and no tuple type |
| `let (q, r) = divide(a, b);` | `q, r := divide(a, b)` |
| `let (q, _) = divide(a, b);` | `q, _ := divide(a, b)` |
| named results optional (Go, Odin) | `-> (quotient: i64, remainder: i64)`, every value named |
| `Shape::Circle { r: 5 }` | `Shape::Circle { radius = 5 }`, or `.Circle { radius = 5 }` |
| `Point { x: 1, y: 2 }` | `Point { x = 1, y = 2 }`, or `{ x = 1, y = 2 }` |
| `Point(1, 2)` (tuple struct) | nothing; every field is named |
| `struct Meters(i64);` (newtype) | `Meters :: distinct i64`, no field to unwrap |
| `while cond { }` | `while (cond) { }` |
| `&x`, `&mut x` (at a call) | nothing, the callee's mode decides |
| `fn f(x: &T)`, `fn f(x: &mut T)` | `f :: fn(x: T)`, `f :: fn(mut x: T)` |
| `fn f(..) -> &T` | `f :: fn(..) -> ref T`, and `ref x := place` binds one |
| `*p` (deref) | `p^` |
| `*const T`, `*mut T` | `^T`, and `ptr_to(x)` takes one |
| `fn(i64) -> i64` (fn pointer) | `fn(i64) -> i64` |
| `Box<T>` / `Rc<T>` / arena index | `Handle<T>` into a slab |
| `impl Drop for T` | `T :: linear struct { .. }` plus a consumer |
| `<T: Trait>` | `$T` plus a `where` bound, or a capability bundle |
| a trait method a generic calls | `$f: fn(..) -> ..` compile-time parameter |
| `foo::<u32>()` (turbofish) | `foo($u32, ..)` |
| `extern "C" { .. }` | `name :: extern fn(..) -> ..` |
| `#[repr(packed)]` | `Name :: packed struct { .. }`, and `field: T align(16)` |
| `const fn f(..)` at a `const` | any `fn`, worked out where a constant reads it |
| `Simd<f32, 4>` / `@Vector(4, f32)` | `[4]f32`, and `a + b` is once per lane |
| `a.wrapping_add(b)` | `wrap_add(a, b)` |

## Declarations and bindings

There is no `let`. A name is introduced with one of three operators:

- `x := expr` binds a local with an inferred type. This is your everyday `let`.
- `x : Type = expr` binds a local with an explicit type.
- `NAME :: expr` declares a constant, evaluated once. Functions, structs,
  enums, and top-level items are all constants, which is why every function is
  written `name :: fn(..)`.

Bindings are immutable by default, exactly as in Rust, and `mut` declares one
the body may assign again. It is the same word a parameter mode carries, and
which of the two it means comes from where it is written: in a parameter list
`mut p` is the caller's value, and at a statement `mut x := 3` is a slot this
frame owns.

Rust's difficulty with `mut` is that a pattern `mut x` and a reference type
`&mut T` are two ideas under one word. Frost has no `&` or `&mut` in the
surface, so `mut` appears only as a mode on a name being introduced, and the
syntax around the name says whose storage it is:

```frost,inside
mut total : i64 = 0
total = total + 1
```

A block is an expression whose value is its trailing expression, the same rule
as Rust. `return` exists for early exit. Statements are separated by newlines.
Semicolons are not required. Line comments start with `//`, as in Rust.

Program entry is `main :: fn() -> i64`, and its return value is the process exit
code, which is why the examples end in a bare `0`.

There are no attributes, no `#[derive(..)]`, and no macros.

Printing is a library call, so `println!` has no equivalent.
`import "io.frost"` brings in `print`, which takes a format string and one value
for each `{}` in it. The compiler checks the count where the call is written and
picks the writer for each value while it compiles the call, so the program does
one direct write per value ([text-and-io.md](std/text-and-io.md)).

## Functions, and the absence of methods

Frost is data-oriented. There are no methods, no `self`, no `impl` blocks, and
no traits. Behavior lives in free functions that take their data as parameters:

```frost
Vec3 :: struct { x: i64, y: i64, z: i64 }

dot :: fn(a: Vec3, b: Vec3) -> i64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}
```

An unmarked parameter is borrowed to read, so nothing here is copied and nothing
is consumed.

Where Rust would write `a.dot(&b)`, Frost writes `dot(a, b)`. Higher-order code
uses function pointers, covered below. There are no closures.

## Returning several values

Rust returns a tuple. Frost has no tuple type, so a function that answers with
more than one value declares a return type list and the caller binds the values
by name:

```frost,sketch
divide :: fn(a: i64, b: i64) -> (quotient: i64, remainder: i64) {
    return a / b, a % b
}

quotient, remainder := divide(17, 5)
```

Every value in the list is named. Go, Odin and Jai make those names optional;
Frost requires them. The names say which value is which at the declaration,
`frost api` shows them, and a `return` by name writes them as fields:

```frost
split :: fn(value: i64) -> (high: i64, low: i64) {
    return { high = value / 256, low = value % 256 }
}
```

A name in the list is a label for one of the values. Go's named results also let
a naked `return` hand back whatever those names hold, and Frost has no such
form. The `return` is required either way: a trailing expression is one value.
`mut` goes in front of any name the body writes afterwards, as in
`magnitude, mut negative := classify(value)`, and `_` takes a value the caller
has no use for, as in `quotient, _ := divide(17, 5)`.

The list itself is never a value. `(i64, i64)` is not a type, so it cannot be
stored in a field, passed as an argument, or bound to one name. A program that
wants to pass a pair around declares a struct, so every aggregate in a Frost
program has a name its author chose. A fallible function still
answers with one value, so `-> (A, B) ! E` is rejected and a function that wants
both returns a struct it names.

## Distinct types instead of the newtype

Rust's newtype is a tuple struct you wrap and unwrap: `Meters(3)` going in and
`m.0` coming out. Frost's is a type declaration:

```frost
Meters :: distinct i64
Feet   :: distinct i64
```

The representation is the inner type, so arithmetic, layout and the C ABI are
`i64`'s and there is nothing to unwrap. The name buys the same thing the newtype
buys: a `Meters` cannot be built from a bare number or from a `Feet`.

The check runs in one direction, where the newtype's runs in both. Going out is
free, so `print("{}\n", m)` and `n : i64 = m` both work, because a
`Meters` is an `i64` in memory. Going in is checked, so a value that means
something else cannot become a `Meters` by accident. There is no cast in either
direction.

## Types and arithmetic

The scalar types are `i8`, `i16`, `i32`, `i64`, `isize`, their unsigned `u*`
counterparts, `f32`, `f64`, and `bool`. These are all copy types.

Integer arithmetic that leaves the range of the type it is computed at aborts
and says where, on every backend. Rust panics in debug and wraps in release.
Frost aborts in both, so a build that ran is a build whose arithmetic held.
`wrap_add`, `wrap_sub` and `wrap_mul` keep the low bits, for a hash or a counter
that is meant to leave the range, and they are the analogue of Rust's
`wrapping_*` family. An `_` may sit between digits, as in Rust, so `1_000_000`
reads in groups.

A fixed array of numbers takes the arithmetic operators, once per lane:
`a + b` over two `[4]f32` is four adds, and `a * 2.0` is that number in every
lane. Where Rust has `Simd<f32, 4>` and Zig has `@Vector`, Frost has the fixed
array it already had and no SIMD type. The length is a power of two and
the whole thing is at most a register's sixty-four bytes, so an operator stays a
handful of instructions.

Mixed-width integer arithmetic is permitted, with the narrower operand widening
to the wider one, so `an_i32 + an_i64` is an `i64`. This is looser than Rust,
which would reject the mismatch and make you write an `as` cast.

`str` is a byte-slice view, a pointer and a length, and it is the analogue of
Rust's `&str`. It owns nothing, so it is a copy type you can duplicate freely. A
string literal is a `str` into read-only data. `str_len(s)` is the
constant-time length and `s[i]` is a bounds-checked `u8`, the same indexing rule
as arrays. Frost has no owned `String` and no UTF-8 method library. A `str` is
bytes. An owned or growable buffer is something you build as a struct over an
array or slab and borrow back as a `str`.

Crossing to C is explicit, since a `str` carries no NUL terminator. The one
shortcut is the string literal, which the compiler also lays down NUL-terminated,
so a literal passed where `^i8` is expected reaches C as a plain pointer at no
cost. That is why the FFI examples below pass `"..."` straight to `printf`.

Structs and enums pass and return by value, copied at the call boundary, unless
you pass a borrow. Slices and fixed arrays are copy: a slice is a pointer and a
length, and copying one duplicates that pair and leaves the elements where they
are. There is no implicit boxing and no hidden heap allocation anywhere.

Fixed-size arrays are written `[N]T` and every index is bounds-checked. An
out-of-range access aborts at runtime. This is always on, with no
`get`/`get_unchecked` split.

## Structs, enums, and pattern matching

Structs and enums are plain data. Construction uses `=` for fields, where Rust
uses `:`:

```frost,sketch
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

```frost,sketch
Kind :: enum { Player, Enemy { damage: i64 }, Pickup { amount: i64 } }
```

A `match` arm is written `case <pattern>: <expr>`. A variant pattern leads with
a dot and binds fields by name:

```frost,sketch
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

```frost
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

Or-patterns and range patterns carry over as they read in Rust: `case .Left |
.Right:` runs one body for either variant, and `case 1..10:` or `case 'a'..='z'`
written out as numbers covers a span. There are no guards. An `if` inside the
arm is the spelling, and 13 of [syntax.md](design/syntax.md) covers the
reasoning. An alternative may not bind payload fields, and a range never removes
the need for a `case _`.

One thing that does not carry over: a bare name is not a binding. `case n:` in
Rust catches everything and names it. In Frost a name in a pattern is the value
it stands for, so `case CH_0:` compares against that constant, and a name
standing for no constant is refused. `_` is the arm that covers the rest, and
the matched value already has a name to read. Decimals and strings are refused
in a pattern too.

Matching a value of a `linear` type consumes it (see below). There is no
`#[derive(Debug)]`, `PartialEq`, or the rest. Equality is written out, and
printing is a call to a writer named for the type.

## The borrow system without lifetimes

Shared and exclusive borrows mean what they mean in Rust, and the exclusivity
rule is familiar. Within a single call you may borrow a variable to read many
times or to mutate exactly once, never both. The place you write it differs.
There is no `&`. The mode is a property of the parameter:

| Rust | Frost | means |
| --- | --- | --- |
| `fn f(x: &T)` | `f :: fn(x: T)` | borrowed to read |
| `fn f(x: &mut T)` | `f :: fn(mut x: T)` | borrowed to mutate in place |
| `fn f(x: T)` | `f :: fn(move x: T)` | ownership transferred |

The call is `f(x)` in all three cases. Which one it is comes from the signature
you can go read, and the exclusivity check reads that signature too.

```frost,sketch
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

A borrow obtained from a parameter mode is implicit, and an implicit borrow
cannot escape. There is no reference type to write down, so such a borrow cannot
be stored in a struct or enum field, put in an array, or returned. You pass data
in by borrow, operate on it, and the borrow dies at the end of the call.

### `ref T`, the borrow you write down

The exception is explicit and checked. `ref name := place` binds a borrow of a
place, and a function may declare `-> ref T`:

```frost,sketch
at :: fn(points: []Point, index: i64) -> ref Point {
    ref result := points[index]
    result
}

held := at(storage, 1)
held.x = 9                 // writes the element, not a copy of it
```

That is Rust's `fn get(&mut self, i: usize) -> &mut T` without the lifetime, and
it lets a container hand back an element instead of a read-and-write pair. It
may not be stored: no struct field, no array element, no container. So
`fn longest<'a>(x: &'a str, y: &'a str) -> &'a str` translates directly, and a
cache of borrows still does not. That is where you switch to handles. See 3.3 of
[types.md](reference/types.md) and chapter 8 of
[ownership.md](reference/ownership.md).

Deref rules to keep straight, since they differ from Rust's `*`:

- On a borrowed aggregate, member access is direct, as in `p.x` where the
  parameter is `mut p: Point`. There is no `(*p).x`.
- On a raw pointer, the postfix `^` operator reads or writes the pointee. Given
  `a: ^i64`, `a^` is the value and `a^ = 7` writes it.

### Raw pointers are the escape hatch

`^T` is a raw pointer, the analogue of `*const T` / `*mut T`. It is unchecked
and exists for FFI and for building low-level libraries. `ptr_to(place)` takes
one, and taking an address is safe. Reading through one with the postfix `^`
is what belongs in an `unsafe` block:

```frost,sketch
mut hero := Entity { hp = 100, mana = 30 }
pe : ^Entity = ptr_to(hero)
unsafe { pe^.hp = pe^.hp - 25 }
```

Raw pointers are where you step outside the safety guarantees, exactly as
`unsafe` raw pointers are in Rust. In Frost the common case, a slab of
long-lived objects, is served by safe handles, so raw pointers stay in FFI and
low-level library code.

## Moves, copies, and linear resources

Move semantics match your Rust intuition. A non-copy value (a struct or an
enum) is moved when passed by value, assigned, or returned, and using it
afterward is a compile error:

```frost,sketch
buf := make_buffer()
consume(buf)
// consume(buf)   // error: use of moved value 'buf'
```

Copy-ness is decided by the type category, where Rust uses a `Copy` derive.
Scalars, pointers, handles, `str`, slices, and fixed arrays are copy. Structs
and enums are move. There is no `#[derive(Clone)]` and no `.clone()`. If you
want a second copy of a struct, you construct one.

The larger divergence is how cleanup works. Frost has no `Drop`. In its
place is the `linear` qualifier, which changes the affine rule (use *at most*
once) into a linear rule (use *exactly* once):

```frost
import "io.frost"
File :: linear struct { fd: i64 }
open  :: fn(n: i64) -> File { File { fd = n } }
close :: fn(move f: File) -> i64 { f.fd }   // terminal consumer

run :: fn() {
    f := open(3)
    print("{}\n", close(f))   // consumes f exactly once
    // close(f)         // error: use of moved value
}                       // f still live here: linear value never consumed
```

A `linear` value that reaches the end of its scope without being consumed is a
compile error, the mirror image of a leaked `Drop`. Consuming means moving it
onward: returning it, passing it by value (often to an `extern` that takes
ownership across the FFI boundary), or `match`ing it. There is no drop order to
reason about and no `mem::forget` footgun.

A fallible function that answers with a `linear` value cannot be ignored either.
Where Rust leans on `#[must_use]` as a lint, Frost makes must-use a type rule:
the result carrying a resource is itself linear, so dropping the call drops the
resource, and that is a compile error.

### `defer` for actions on the way out

Because there is no `Drop`, the RAII-guard pattern is replaced by `defer`,
which runs a statement where the function leaves, last written first. It goes
at the top level of a body, and `break` and `continue` do not run one:

```frost
import "io.frost"
work :: fn() {
    defer print("cleanup\n")   // runs on the way out
    // ... body ...
}
```

It is Go's `defer`. For resources with real ownership, prefer a `linear` type.
Use `defer` for local, best-effort actions on the way out.

## Errors, without `Result<T, E>` being a library type

Rust's `Result` is a type in the library and everything around it is machinery
you compose: `?` plus `From` for conversion, `Box<dyn Error>` to erase the type,
`#[must_use]` to nag about ignoring one. Frost puts the same idea in the
signature and leaves the machinery out.

```frost,sketch
Parse :: struct { at: i64, code: i64 }

digit :: fn(text: str, index: i64) -> i64 ! Parse { .. }

d := digit(text, index)?

match number(text) {
    case .Ok { value }: { print("{}\n", value) }
    case .Err { error }: { print("{}\n", error.at) }
}
```

`-> i64 ! Parse` is "answers with an i64, or fails with a Parse", the compiler
makes that signature into one enum with `Ok { value }` and `Err { error }`
variants, and `?` hands a failure up. 5.2b of
[declarations.md](reference/declarations.md) is the mechanism in full. Four
things differ from Rust.

`Parse` is a plain struct the program declared. There is no `Error` trait to
implement, no `source()`, no backtrace, no allocation, and no `Box<dyn Error>`
to erase anything into.

`?` does not convert. The failure type of the call and the failure type of the
function it is written in have to be the same one, where Rust would insert a
`From` impl. A function that calls something failing differently `match`es it
and returns the failure it declares, so a function fails with what its signature
says.

`#[must_use]` has no equivalent. A result carrying a `linear` value is itself
linear, so ignoring the call is a compile error. A result carrying an ordinary
value may be ignored, exactly as in Rust with the lint off.

There is no `panic!` to catch, no `unwind`, and no `catch_unwind`. A failure
that is a bug, as opposed to a condition in the world, is an assertion, and an
assertion aborts. There is also no `unwrap`, because the match on the enum
covers every variant the way any other match does.

## Handles and slabs, the replacement for `Rc`, `Arc`, and back-references

This is where you put everything that Rust would model with `Rc<RefCell<T>>`,
`Arc`, a `Vec<T>` plus indices, or a graph of references. Long-lived, shared,
or interlinked data lives in a slab and is named by a `Handle<T>`, a small
copyable value holding an index and a generation.

The slab lives in the library, with no runtime behind it. `std/slab.frost` is
ordinary Frost: the storage, the free list, the generation
counters, and the packing of an index and a generation into one handle are all
written out, the way `slotmap` and `generational-arena` are written out in Rust.
The compiler adds the subscript, since handing back a validated reference into
storage is the one operation a language with second-class borrows cannot write
for itself. It offers that subscript for any struct that is slab-shaped: a
`storage` array beside a parallel `generations` array.

```frost
import "io.frost"
import "slab.frost"

Entity :: struct { hp: i64, mana: i64 }

main :: fn() -> i64 {
    mut world : Slab<Entity, 16> = slab_new()
    slab_reset(world)

    hero := slab_insert(world, Entity { hp = 100, mana = 30 })

    print("{}\n", world[hero].hp)        // 100
    world[hero].hp = world[hero].hp - 25  // the subscript is a place to write
    print("{}\n", world[hero].hp)        // 75
    0
}
```

`world[handle]` is a place. You can read a field, write a field, copy the
element out, or pass it to a function, which borrows it under the same parameter
modes as anything else, so that borrow cannot escape the call either.

The generation makes this safe without a borrow checker. Releasing a slot bumps
its generation counter. A handle carries the generation it was minted with, and
a lookup with a stale generation aborts instead of reading whoever reused the
slot. You get the behavior of a weak reference that goes dangling safely, with
no reference counting and no runtime borrow tracking.

The mental substitution is direct. You store a `Handle<T>` in a field and
return one from a function, precisely the things a `&T` may not do. A linked
list, a scene graph, or an entity system is a slab of nodes linked by handles.
`std/columns.frost` is the same handle scheme over a structure-of-arrays layout,
and chapter 10 of [handles-and-pools.md](reference/handles-and-pools.md) covers
both.

Frost's memory safety, with no garbage collector and no lifetimes, rests on six
guarantees, each with a mechanism behind it and raw pointers (`^T`) outside
the set. [memory-safety.md](design/memory-safety.md) states them and argues
each one.

## Generics without traits

Generics monomorphize, exactly as in Rust. Each instantiation is compiled to
specialized code with no runtime dispatch. The differences are in the spelling
and in what you cannot say.

A type parameter is written `$T`. It can appear on functions and on structs:

```frost
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

```frost,sketch
bytes_for :: fn($T: Type, count: i64) -> i64 { count * sizeof(T) }

n := bytes_for($Entity, 16)     // like bytes_for::<Entity>(16)
```

`sizeof(T)` is a compile-time constant, so a generic function can size its own
type parameter. Type parameters are erased after monomorphization: they drive
the specialization and then vanish from the ABI.

There are no traits, no associated types, and no `dyn Trait`. A generic
function is generic over any type its body type-checks against once
specialized.

There is a `where` clause. It asks what the compiler already knows about a type,
over a fixed vocabulary: `is_numeric`, `is_integer`, `is_float`, `is_struct`,
`is_array`, `is_slice`, `is_pointer`, `is_linear`, combined with `&&`, `||` and
`!`.

```frost
twice :: fn($T: Type, v: $T) -> T where is_numeric(T) { v + v }
```

Nothing registers into that vocabulary and nothing implements it, so there is
no coherence rule, no orphan rule, and no solver. The bound is a precondition
checked at the call, so the error lands on the caller that chose the type.

Where Rust would write `T: Ord` and call `a.cmp(&b)`, Frost passes the
operation. For a single operation that is a compile-time function parameter,
which can declare the signature it needs:

```frost,sketch
ascending :: fn(a: i64, b: i64) -> bool { a < b }

best :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T, move y: $T) -> $T {
    mut result := x
    if (before(y, result)) { result = y }
    result
}

smallest := best($i64, $ascending, 7, 3)
```

The signature is checked at the call with that call's type arguments substituted
in, so a wrong one is an error against the parameter list, and the message
points there instead of inside the specialized body. The call to `before` is
direct, with no pointer, because the specialization knows which function it is.

### A trait, and what replaces it

In Rust the operations travel together, attached to the type:

```rust
trait Ordering {
    fn less(a: &Self, b: &Self) -> bool;
    fn equal(a: &Self, b: &Self) -> bool;
}

impl Ordering for i64 {
    fn less(a: &i64, b: &i64) -> bool { a < b }
    fn equal(a: &i64, b: &i64) -> bool { a == b }
}

fn sort<T: Ordering>(items: &mut [T]) { /* calls T::less(a, b) */ }

sort(&mut numbers);
```

The implementation is attached to the type, found by a lookup, and unnamed at
the call. Frost's replacement is a capability bundle: the same operations in a
generic struct whose fields are functions, an implementation that is a plain
constant of it, and a call that names which constant it means.

```frost,sketch
Ordering :: struct($T: Type) { less: fn(T, T) -> bool, equal: fn(T, T) -> bool }

i64_ascending :: Ordering<i64> { less = i64_less, equal = i64_equal }

sort :: fn($T: Type, $ops: Ordering<T>, mut items: []T) {
    ...
    if (ops.less(items[j], items[j - 1])) { ... }
}

sort($i64_ascending, numbers)
```

11.4b of [generics.md](reference/generics.md) has the mechanism, including how
`ops.less(a, b)` folds to a direct call to `i64_less` so the specialization
holds no function pointer.

Two orderings for one type. In Rust this is the newtype dance or a wrapper. In
Frost it is a second constant, `i64_descending`, and the two never collide,
because neither was ever implicit.

Runtime dispatch comes from the same declaration. Drop the `$` and `ops` is an
ordinary value that can be chosen while the program runs, stored in an array, or
swapped. Rust needs `dyn Trait` and a second signature for that.

Composing bounds is a struct with struct fields, where Rust writes `T: A + B`,
and the body reads `ops.ordering.less(a, b)`.

Every call carries the bundle: `sort($i64_ascending, numbers)` where Rust
writes `sort(&mut numbers)`. In exchange the call site says which comparison it
used, and `i64_ascending` greps to exactly one definition.

A bundle is never found for you. There is no lookup, so there is nothing to be
coherent about, no orphan rule, and no solver. You also cannot state a
requirement on `T` itself beyond the `where` vocabulary above. Anything narrower
surfaces when the specialization is compiled.

## Function pointers and an explicit context

Functions are values. A parameter of type `fn(..) -> T` holds one, and you call
it directly:

```frost,inside
apply :: fn(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }
double :: fn(x: i64) -> i64 { x * 2 }

apply(double, 21)    // 42
```

There are no capturing closures, and therefore no `Fn` / `FnMut` / `FnOnce`
distinction and no closure environment. Where a Rust closure would capture
state, you pass that state explicitly as another argument. This keeps every
indirect call a plain function pointer with no hidden allocation and no captured
lifetimes to reason about. In practice, callback-style code threads a context
value alongside the function pointer, the same pattern C uses.

Registering a callback with a C library is the one case where that pattern
gets language support, because it is the case where the context outlives the
call. It is written as a `$` function parameter on an `extern` plus a context
taken by `move`, and it sits closer to Rust's `Box::into_raw` plus an
`extern "C" fn` shim than to a closure. The context is handed over, the caller
cannot touch it while the callback can fire, and unregistering gives it back.
The Frost version has no `unsafe` and no raw pointer in what you write. See
[callbacks.md](design/callbacks.md).

## Compile-time evaluation

Frost has no general compile-time interpreter and no macros. The compile-time
machinery is `sizeof(T)` as a constant and monomorphization, driven by five
kinds of `$` parameter: a type (`$T: Type`), an integer (`$N: usize`, which is
Rust's const generics as values and sizes a `[N]T` field), a function
(`$f: fn(..) -> ..`), a capability bundle (`$ops: Ordering<T>`), and a list of
arguments (`args: $...`). They work on functions as well as structs, so an
operation over a sized aggregate is written once and covers every size. Where
Rust's const generics keep the integer in the type, Frost lets the body use it
as a plain value. A generic is stamped out once per distinct set of those
arguments at the call site (chapter 11 of the spec), the way Rust monomorphizes
generics.

A list covers what a variadic macro covers in Rust, without being a macro. A
`for` over one unrolls, `list[K]` names an element, and an `if` over a type
predicate keeps the branch that survives for that element:

```frost,sketch
printall :: fn(args: $...) {
    for value in args {
        if (is_float(value)) {
            print("{}\n", value)
        } else if (is_slice(value)) {
            print("{}\n", value)
        } else {
            print("{}\n", value)
        }
    }
}

printall(1, 2.5, "three")
```

A call written where a compile-time value is read is worked out before the
program runs, which is where `const fn` lands. Nothing marks the function:
`LANES :: round_up(300, 64)` and `[next_power_of_two(300)]u8` call ordinary
functions, and the position runs them early. Rust asks the author to promise
`const` on the definition. Frost reads the position, so no function has two
kinds.

Such a call may use the whole-number half of the language, plus the three things
built out of it: a run of values, a set of named ones, and a run of bytes. So a
lookup table is decided before the program runs, and `TABLE[2]` reads out of it
with the index checked where it is written. `wrap_add`, `wrap_sub` and
`wrap_mul` fold too, which is Rust's `wrapping_*` family in a `const fn`.

Everything else is refused where it is written: a function that reaches itself,
a call into the world, a pointer, a number with a fraction, and a bound of a
million steps. A refused call stays refused, so one declaration is never a
number in one place and a call in another.

The compile-time layer stops well short of a language of its own. Every
construct walks a list whose length the call fixed, or a body whose steps are
bounded, so expansion costs what the program's text costs. There is nothing that
corresponds to a procedural macro.

## Calling C

FFI needs no glue. Declaring an external function is a
constant whose value is an `extern fn`:

```frost
printf :: extern fn(fmt: ^i8, value: i64) -> i32
malloc :: extern fn(size: i64) -> ^u8

main :: fn() -> i64 {
    unsafe { printf("%lld\n", 42) }
    0
}
```

Frost scalar types map to the natural C types, aggregates pass by the platform
ABI, and a `^T` is a C pointer. Calling C is unchecked, so the call goes in an
`unsafe` block, which makes `unsafe` the complete list of places to look when
memory has been corrupted. A foreign function that takes and returns numbers
with no pointer anywhere is declared `safe extern fn` and needs no block, which
is how `std/math.frost` reaches `sqrtf`.

Coming from Rust's `extern "C"` and `#[no_mangle]`, the counterpart is the same
`extern fn` declaration written with a body. An ordinary Frost function is
emitted under a name the compiler chose, so C cannot call it; one written this
way keeps the name it was written under and C links against it. The names
beginning `frost_rt_` and `frost_u_` are the runtime's and the compiler's, and
a definition taking either is refused. See
[c-compatibility.md](impl/c-compatibility.md) for the full type mapping.

## Modules

A module is a file, brought in with `import "x.frost"`. There is no `mod`, no
crate graph, and no visibility modifier: a file's `export` line is the complete
set of names another file may use from it, and everything else is private and
mangled so it cannot collide. An import stops at the file that writes it, so
each file names what it uses. Two imported modules exporting the same name is an
error at the use, and `(old as new)` renames one of them for the importing file
only. See [modules.md](impl/modules.md).

## What a Rust program leans on that Frost omits, and what to use instead

| Rust feature | Frost approach |
| --- | --- |
| Lifetimes (`'a`) | Not needed; an implicit borrow cannot escape and `ref T` is checked without one |
| Traits, `impl Trait`, `dyn Trait` | None; capability bundles, function pointers, or concrete code |
| Trait bounds, associated types | `where` bounds over a closed vocabulary of what a type *is* (11.4a); no associated types |
| Methods, `self`, `impl` blocks | Free functions that take the data as a parameter |
| `Drop`, RAII guards | `linear` types (consume exactly once), plus `defer` |
| Closures, `Fn`/`FnMut`/`FnOnce` | Function pointers plus an explicit context argument |
| `Box`, `Rc`, `Arc`, `RefCell` | Slabs and `Handle<T>` (generational indices) |
| `Vec`, `HashMap`, `String` | `std/vec.frost`, `std/map.frost`, `str` and `std/strings.frost` |
| `#[derive(..)]`, macros, attributes | None; write what you need explicitly |
| `println!` | `import "io.frost"`, one writer per type |
| `?`, `Result`, `#[must_use]` | `-> T ! E` failure sets and `?`; a result carrying a `linear` value must be consumed |
| Overflow checks in debug | Overflow aborts in every build; `wrap_add` and friends wrap on purpose |
| `unsafe` blocks and raw pointers | `unsafe` blocks around `^T` and `extern` calls |
| `pub`, `pub(crate)`, field privacy | An `export` line per file; every struct field is public |
| Async, generics over const, GATs | Out of scope |

## Gotchas checklist for the first hour

- `if` and `while` conditions need parentheses, as in `if (x > 5) { .. }`.
- There is no `let`. Use `:=`, `:`, or `::`, and every function, type, and
  constant is declared with `::`.
- Struct fields are set with `=`, as in `Point { x = 1, y = 2 }`, where Rust
  uses `:`.
- Match arms are `case <pattern>: <expr>`, and variant patterns lead with a dot,
  as in `case .Circle { radius }:`.
- A variant can leave its enum out where the type is already stated, as in
  `paint(.Red)` or `c : Color = .Red`. Rust has this only in a `use`. Here it
  reads from the context the way the `case .Red` of a match does.
- A struct literal can leave its name out the same way, as in
  `p : Point = { x = 1, y = 2 }`. The field names stay: there is no tuple struct
  and no positional literal, so a value never lands in a field by counting.
- `for x in xs` walks a slice, an array or a `str` with no iterator and no
  `.iter()`, and `for index, x in xs` names the position too. It is the
  index-and-bound loop written out, so `break` and `continue` mean what they
  always do and nothing is called per element.
- There is no `&`. Borrowing is written on the parameter, not at the call.
- To deref a raw pointer, use postfix `^`, as in `a^`, `p^.field`. A borrowed
  parameter needs no sigil, so field access on one is direct, as in `p.field`,
  and assigning to the whole of a `mut` parameter is written `p = q`.
- You cannot store a borrow, and only a `ref T` may be returned. Use a
  `Handle<T>` for anything that must live beyond the call and be kept.
- A `linear` value must be consumed on every path, or it is a compile error.
- Integer overflow aborts in every build. Reach for `wrap_add` and friends where
  the arithmetic is meant to wrap.
- There is no `pub`. Visibility is the `export` line at the top of a file, and
  struct fields are always public.
- Calling an `extern fn` needs an `unsafe` block unless it is a `safe extern`.

## A worked example, a tiny entity system

This is the idiom you will use constantly, the Frost answer to a `Vec` of
objects with cross-references. Entities live in a slab, are named by handles,
and are mutated in place through the slab.

```frost
import "io.frost"
import "slab.frost"

Kind   :: enum { Player, Enemy { damage: i64 }, Pickup { amount: i64 } }
Entity :: struct { hp: i64, kind: Kind }

delta :: fn(e: Entity) -> i64 {
    match e.kind {
        case .Player: 0
        case .Enemy { damage }: 0 - damage
        case .Pickup { amount }: amount
    }
}

main :: fn() -> i64 {
    mut world : Slab<Entity, 16> = slab_new()
    slab_reset(world)

    player := slab_insert(world,
        Entity { hp = 100, kind = .Player })
    goblin := slab_insert(world,
        Entity { hp = 30, kind = .Enemy { damage = 15 } })

    // The player takes the goblin's damage, written straight into the slot.
    world[player].hp = world[player].hp + delta(world[goblin])
    print("hp {}\n", world[player].hp)                 // 85

    slab_release(world, goblin)
    print("{}\n", slab_alive(world, goblin))   // 0, the handle is stale
    0
}
```

Entities are stored by value in the slab, handles are the things that get passed
around and stored, the borrow of `world[goblin]` lasts only for the call to
`delta`, and releasing a slot invalidates old handles by generation.

## Where to go next

- [tour.md](tour.md), the same language walked through end to end.
- [philosophy.md](design/philosophy.md), why the language is shaped this way.
- [memory-safety.md](design/memory-safety.md), the safety guarantees in depth.
- [patterns.md](patterns.md), what the language rewards and what it merely
  permits.
- [c-compatibility.md](impl/c-compatibility.md), the C type mapping and FFI details.
- [architecture.md](impl/architecture.md), the compiler pipeline, the typed IR, and
  the three backends that must agree.
- `examples/native/`, runnable programs, starting with `game_world.frost` (the
  entity-component system) and `pool_linked_list.frost`.
