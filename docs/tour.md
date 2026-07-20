# A Tour of Frost

This is a hands-on walk through the language by example. Every snippet below
compiles and runs on both native backends. For the *why* behind these choices,
see [philosophy.md](philosophy.md); for the safety rules, see
[memory-safety.md](memory-safety.md).

Run a program with:

```
frost program.frost --link -o program && ./program
```

## Declarations, values, and functions

`::` declares a constant (including a function). `:=` binds an inferred local,
`:` gives an explicit type, and `mut` makes a binding assignable. There are no
methods — behavior lives in free functions.

```
printf :: extern fn(fmt: ^i8, value: i64) -> i32

square :: fn(x: i64) -> i64 { x * x }

main :: fn() -> i64 {
    n := 6                 // inferred i64
    mut total : i64 = 0    // explicit, mutable
    for i in 0..n {
        total = total + square(i)
    }
    printf("%lld\n", total)   // 55
    0
}
```

Integer widths (`i8`..`i64`, `u8`..`u64`), floats (`f32`, `f64`), and `bool`
are all value (copy) types. Control flow is `if`/`else` (an expression),
`while`, `for … in a..b`, `break`, `continue`, and `match`.

## Structs and enums: plain data

A `struct` is just its fields; an `enum` is a tagged union with payloads.
Neither carries methods.

```
Point :: struct { x: i64, y: i64 }

Shape :: enum {
    Circle { radius: i64 },
    Rect { width: i64, height: i64 },
}

area :: fn(s: &Shape) -> i64 {
    match s {
        case .Circle { radius }: 3 * radius * radius
        case .Rect { width, height }: width * height
    }
}

main :: fn() -> i64 {
    p := Point { x = 3, y = 4 }
    printf("%lld\n", p.x + p.y)              // 7
    printf("%lld\n", area(&Shape::Rect { width = 4, height = 5 }))  // 20
    0
}
```

Structs pass and return by value (copied at the boundary), and `match` works
over a value or a reference, binding payload fields.

## References are second-class

`&T` and `&mut T` are borrows. They exist only as parameters and short-lived
temporaries — you **cannot** store one in a field or return one. That single
rule is why Frost needs no lifetime annotations.

```
scale :: fn(p: &mut Point, k: i64) {   // borrow to mutate in place
    p.x = p.x * k
    p.y = p.y * k
}

// Bad :: struct { r: &Point }         // rejected: reference stored in a field
// bad :: fn(p: &Point) -> &Point { p } // rejected: reference returned
```

Raw pointers `^T` exist as an explicit, unchecked escape hatch for FFI.

## Move checking and linear resources

Non-`Copy` values (structs, enums) *move* when passed by value; using one again
is a compile error. A `linear` type must be consumed **exactly once**, which is
how Frost replaces destructors:

```
File :: linear struct { fd: i64 }
open  :: fn(n: i64) -> File { File { fd = n } }
close :: extern fn(f: File)     // terminal consumer, takes ownership

run :: fn() {
    f := open(3)
    close(f)      // consumes f exactly once
    // close(f)   // error: use of moved value 'f'
}                 // dropping f without consuming would also be an error
```

A `linear enum` returned from a fallible function therefore cannot be ignored —
errors are non-ignorable by construction.

## Generational handles and pools

Long-lived data lives in a pool and is referenced by a `Handle<T>` — a small
copyable value, not a pointer. `pool[handle]` is a *place*: read and write
through it, or borrow it. A freed-and-reused slot bumps its generation, so an
old handle can never read the new occupant.

```
pool_new    :: extern fn(capacity: i64, elem_size: i64) -> ^u8
pool_alloc  :: extern fn(pool: ^u8, value: ^u8) -> i64
pool_get    :: extern fn(pool: ^u8, handle: i64) -> ^u8

Entity :: struct { hp: i64, mana: i64 }

main :: fn() -> i64 {
    world := pool_new(16, 16)
    mut hero := Entity { hp = 100, mana = 30 }
    h : Handle<Entity> = pool_alloc(world, &hero)

    printf("%lld\n", world[h].hp)   // 100
    world[h].hp = world[h].hp - 25
    printf("%lld\n", world[h].hp)   // 75
    0
}
```

The borrow you get from `&world[h]` is second-class, so it cannot escape the
scope where the pool operation is valid.

## Generics: specialize at compile time

Generic functions and structs monomorphize — there is no runtime dispatch. A
type parameter is written `$T`:

```
Pair :: struct($T: Type) { first: T, second: T }

make_pair :: fn(a: $T, b: $T) -> Pair<T> { Pair { first = a, second = b } }
swap      :: fn(a: &mut $T, b: &mut $T) { t := a^  a^ = b^  b^ = t }

main :: fn() -> i64 {
    p := make_pair(3, 4)               // Pair<i64> inferred
    printf("%lld\n", p.first + p.second)   // 7

    mut x : i64 = 1
    mut y : i64 = 2
    swap(&mut x, &mut y)
    printf("%lld\n", x)                // 2
    0
}
```

`sizeof(T)` is a compile-time constant, so a generic function can size its own
type parameter. When a type parameter can't be inferred from a value argument —
for example a function that only uses `sizeof(T)` — declare it as `$T: Type` and
pass the type explicitly with a leading `$`:

```
make_pool :: fn($T: Type, capacity: i64) -> ^u8 {
    pool_new(capacity, sizeof(T))     // T is a compile-time type
}

main :: fn() -> i64 {
    world := make_pool($Entity, 16)   // pass the type with $
    ...
}
```

Type parameters are erased — they drive monomorphization (`sizeof`, the return
type, annotations in the body) and carry no runtime cost. This is how the typed
pool wrappers work as an ordinary Frost library, with no dummy value needed.

## Higher-order code: function pointers, not closures

Functions are values; a `fn(...) -> T` parameter holds one. There are no
capturing closures.

```
apply :: fn(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }
double :: fn(x: i64) -> i64 { x * 2 }

main :: fn() -> i64 {
    printf("%lld\n", apply(double, 21))   // 42
    0
}
```

## Arrays are bounds-checked

A fixed-size array `[N]T` knows its length, and every index is checked:

```
main :: fn() -> i64 {
    arr := [10, 20, 30]
    printf("%lld\n", arr[2])   // 30
    // arr[5]                   // aborts: index 5 out of bounds for length 3
    0
}
```

## Calling C

`extern fn` links against any C library with the natural ABI — this is how the
examples reach `printf`, `malloc`, and the pool runtime. See
[c-compatibility.md](c-compatibility.md).

## Where to next

- Runnable programs live in `examples/native/` — start with `game_world.frost`
  (an entity system) and `pool_linked_list.frost` (handles as links).
- [architecture.md](architecture.md) explains the compiler pipeline and exactly
  what the native path supports today.
