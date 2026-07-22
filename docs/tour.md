# A Tour of Frost

This is a hands-on walk through the language by example. Every snippet below
compiles and runs on both native backends. For the *why* behind these choices,
see [philosophy.md](philosophy.md). For the safety rules, see
[memory-safety.md](memory-safety.md).

Run a program with:

```
frost program.frost --link -o program && ./program
```

## Declarations, values, and functions

`::` declares a constant (including a function). `:=` binds an inferred local,
`:` gives an explicit type, and `mut` makes a binding assignable. There are no
methods. Behavior lives in free functions.

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
`while`, `for i in a..b`, `break`, `continue`, and `match`.

## Structs and enums: plain data

A `struct` is just its fields. An `enum` is a tagged union with payloads.
Neither carries methods.

```
Point :: struct { x: i64, y: i64 }

Shape :: enum {
    Circle { radius: i64 },
    Rect { width: i64, height: i64 },
}

area :: fn(s: Shape) -> i64 {
    match s {
        case .Circle { radius }: 3 * radius * radius
        case .Rect { width, height }: width * height
    }
}

main :: fn() -> i64 {
    p := Point { x = 3, y = 4 }
    printf("%lld\n", p.x + p.y)              // 7
    printf("%lld\n", area(Shape::Rect { width = 4, height = 5 }))  // 20
    0
}
```

Structs pass and return by value, and `match` binds payload fields.

## Borrowing is a parameter mode

There is no `&` in the language. How a parameter is passed is a property of the
parameter, written on its declaration, and the call site says nothing:

| mode | written | means |
| --- | --- | --- |
| read | `p: Point` | borrowed to read, the default |
| write | `mut p: Point` | borrowed to mutate in place |
| move | `move p: Point` | ownership transferred |

```
scale :: fn(mut p: Point, k: i64) {   // borrowed to mutate in place
    p.x = p.x * k
    p.y = p.y * k
}

main :: fn() -> i64 {
    mut p := Point { x = 3, y = 4 }
    scale(p, 2)                       // no sigil at the call
    printf("%lld\n", p.x)             // 6
    0
}
```

Because a borrow is only ever a parameter, it cannot be stored in a field or
returned. There is no reference type to write in either position. That single
rule is why Frost needs no lifetime annotations.

Raw pointers `^T` exist as an explicit, unchecked escape hatch for FFI, and
`ptr_to(x)` is how you take one.

## Move checking and linear resources

Non-`Copy` values (structs, enums) *move* when passed by value. Using one again
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

A `linear enum` returned from a fallible function therefore cannot be ignored,
errors are non-ignorable by construction.

## Generational handles and pools

Long-lived data lives in a pool and is referenced by a `Handle<T>`, a small
copyable value, not a pointer. `pool[handle]` is a *place*. Read and write
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
    h : Handle<Entity> = pool_alloc(world, ptr_to(hero))

    printf("%lld\n", world[h].hp)   // 100
    world[h].hp = world[h].hp - 25
    printf("%lld\n", world[h].hp)   // 75
    0
}
```

Passing `world[h]` to a function borrows it, and that borrow is a parameter mode
like any other, so it cannot escape the scope where the pool operation is valid.

## Generics: specialize at compile time

Generic functions and structs monomorphize, so there is no runtime dispatch. A
type parameter is written `$T`:

```
Pair :: struct($T: Type) { first: T, second: T }

make_pair :: fn(a: $T, b: $T) -> Pair<T> { Pair { first = a, second = b } }
swap      :: fn(mut a: $T, mut b: $T) { t := a  a = b  b = t }

main :: fn() -> i64 {
    p := make_pair(3, 4)               // Pair<i64> inferred
    printf("%lld\n", p.first + p.second)   // 7

    mut x : i64 = 1
    mut y : i64 = 2
    swap(x, y)
    printf("%lld\n", x)                // 2
    0
}
```

`sizeof(T)` is a compile-time constant, so a generic function can size its own
type parameter. When a type parameter can't be inferred from a value argument
(for example a function that only uses `sizeof(T)`), declare it as `$T: Type`
and pass the type explicitly with a leading `$`:

```
make_pool :: fn($T: Type, capacity: i64) -> ^u8 {
    pool_new(capacity, sizeof(T))     // T is a compile-time type
}

main :: fn() -> i64 {
    world := make_pool($Entity, 16)   // pass the type with $
    ...
}
```

Type parameters are erased, and they drive monomorphization (`sizeof`, the return
type, annotations in the body) and carry no runtime cost. This is how the typed
pool wrappers work as an ordinary Frost library, with no dummy value needed.

## Higher-order code: no traits, no closures

A generic algorithm takes the operation it needs as a compile-time function
parameter, which is Frost's answer to what a trait bound expresses. The
parameter can state the signature it requires, and the call inside the
specialization is direct rather than through a pointer:

```
ascending :: fn(a: i64, b: i64) -> bool { a < b }

best :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T, move y: $T) -> $T {
    mut result := x
    if (before(y, result)) { result = y }
    result
}

main :: fn() -> i64 {
    printf("%lld\n", best($i64, $ascending, 7, 3))   // 3
    0
}
```

When the function genuinely varies at runtime, it is an ordinary value: a
`fn(...) -> T` parameter holds a pointer. There are no capturing closures.

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

## Tests

A `test` block is a named unit test, and `assert` fails it when the condition is
false. Run every test in a file with `frost --test file.frost`.

```
add :: fn(a: i64, b: i64) -> i64 { a + b }

test "addition" {
    assert(add(2, 3) == 5)
    assert(add(0, 0) == 0)
}
```

The runner compiles the file, runs each test, and prints one line per test. A
failing assertion aborts that test and the run exits non-zero, so `frost --test`
works as a build gate.

## Calling C

`extern fn` links against any C library with the natural ABI. This is how the
examples reach `printf`, `malloc`, and the pool runtime. See
[c-compatibility.md](c-compatibility.md).

## Where to next

- Runnable programs live in `examples/native/`. Start with `game_world.frost`
  (an entity system) and `pool_linked_list.frost` (handles as links).
- [architecture.md](architecture.md) explains the compiler pipeline and exactly
  what the native path supports today.
