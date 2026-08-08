# Learn Frost in Y minutes

Frost is a systems language for data-oriented programs. It is compiled ahead of
time, has no garbage collector, no runtime, no exceptions, and no hidden
allocation. It is memory-safe without lifetimes: a borrow is a parameter mode
and cannot escape the call it was made for.

This page covers the whole language in order, then the standard library, as
code. The test suite compiles every block below, so the compiler accepts
everything written here. Blocks marked `sketch` show a shape and skip
compilation.

Run a file with `frost program.frost`. Keep the executable with
`frost program.frost --link -o program`.

## Comments, declarations, and bindings

```frost
// A line comment. There is no block comment.

// `::` declares a constant, and a function is one.
MAX :: 100
GREETING :: "hello"
PI :: 3.14159

double :: fn(x: i64) -> i64 { x * 2 }

main :: fn() -> i64 {
    // `:=` binds a name to a value and infers the type.
    count := 7

    // `:` states the type instead of inferring it.
    total : i64 = 0

    // `var` makes a binding assignable. Without it a binding is fixed.
    var running : i64 = 0
    running = running + count

    // A block is an expression whose value is its trailing expression, so a
    // function body ends with the value it answers with. No `return` needed.
    0
}
```

There are no methods and no `self`. Write behaviour as free functions over
plain structs.

## Primitive types

```frost
main :: fn() -> i64 {
    // Signed and unsigned integers, by width.
    a : i8 = -1
    b : i16 = -2
    c : i32 = -3
    d : i64 = -4
    e : u8 = 1
    f : u16 = 2
    g : u32 = 3
    h : u64 = 4

    // Pointer-width integers.
    i : isize = -5
    j : usize = 5

    // Floats and truth values.
    k : f32 = 1.5
    l : f64 = 2.5
    m : bool = true
    n : bool = false

    // Text is a byte slice: an address and a length.
    o : str = "text"

    // A fixed array, and a slice of one.
    p : [3]i64 = [1, 2, 3]
    q : []i64 = p

    // The repeat form, for a large or zeroed buffer.
    r : [8]i64 = [0; 8]
    0
}
```

Arithmetic traps on overflow on every backend. Use `wrap_add`, `wrap_sub` and
`wrap_mul` where the arithmetic is meant to wrap, as in a hash function.

```frost
import "io.frost"

main :: fn() -> i64 {
    var hash : u64 = 1469598103934665603
    hash = wrap_mul(hash, 1099511628211)
    print("{}\n", hash)
    0
}
```

A `distinct` type has the representation of the type it is written over and an
identity of its own, so a length and a count are two types.

```frost
import "io.frost"

Meters :: distinct i64
Feet :: distinct i64

main :: fn() -> i64 {
    d : Meters = 10
    print("{}\n", d + d)   // still Meters
    0
}
```

## Operators

Prefix `-` negates a number and `!` negates a truth value. There is no
truthiness. `!` takes a `bool`, and `count == 0` is how you ask whether a number
is zero.

```frost
import "io.frost"

ready :: fn(n: i64) -> bool { n > 0 }

main :: fn() -> i64 {
    a := 7
    b := 2

    print("{}\n", a + b)
    print("{}\n", a - b)
    print("{}\n", a * b)
    print("{}\n", a / b)      // 3, integer division
    print("{}\n", a % b)      // 1

    print("{}\n", a < b)
    print("{}\n", a == b)
    print("{}\n", a != b)

    print("{}\n", a & b)      // bitwise
    print("{}\n", a | b)
    print("{}\n", a << 2)
    print("{}\n", a >> 1)

    print("{}\n", ready(a) && ready(b))
    print("{}\n", ready(a) || ready(b))
    print("{}\n", !ready(a))
    print("{}\n", ready(a) == false)   // the same thing, said the other way
    0
}
```

Spacing tells the two uses of `!` apart. `!ready` is a negation, and
`-> T ! E` marks a failure set.

## Control flow

`if` is an expression, and its condition is parenthesized.

```frost
import "io.frost"

classify :: fn(n: i64) -> i64 {
    // Both arms answer with a value, so the `if` does.
    if (n < 0) { -1 } else { 1 }
}

main :: fn() -> i64 {
    print("{}\n", classify(-5))

    // As a statement, an arm may end in a statement and answer with nothing.
    var total : i64 = 0
    if (total == 0) {
        total = 10
    } else {
        total = 20
    }

    // `while`.
    var index : i64 = 0
    while (index < 3) {
        index = index + 1
    }

    // `for` over a range. `..` is half-open, `..=` includes the end.
    for step in 0..3 {
        total = total + step
    }
    for step in 0..=3 {
        total = total + step
    }

    // `for` over a sequence. This is the index-and-bound loop written out:
    // no iterator, nothing to implement, nothing called per element.
    numbers : [4]i64 = [10, 20, 30, 40]
    for value in numbers {
        total = total + value
    }
    // The position as well.
    for at, value in numbers {
        total = total + at * value
    }

    // `break` and `continue` mean what they always mean.
    for value in numbers {
        if (value == 20) { continue }
        if (value == 40) { break }
        total = total + value
    }
    print("{}\n", total)
    0
}
```

## `match`

An arm is `case`, a pattern, `:`, then an expression or a block. There is no
separator between arms, and a match over an enum has to cover every variant.

```frost
import "io.frost"

Shape :: enum {
    Circle { radius: i64 },
    Rect { width: i64, height: i64 },
    Point,
}

area :: fn(s: Shape) -> i64 {
    match s {
        // The payload's fields bind to names of their own.
        case .Circle { radius }: 3 * radius * radius
        case .Rect { width, height }: width * height
        case .Point: 0
    }
}

kind_of :: fn(byte: i64) -> i64 {
    match byte {
        // A range arm, and several patterns for one arm.
        case 97..=122 | 65..=90: 1
        case 48..=57: 2
        // A match over whole numbers is not countable, so it needs `_`.
        case _: 0
    }
}

main :: fn() -> i64 {
    print("{}\n", area(Shape::Rect { width = 4, height = 5 }))
    print("{}\n", kind_of(100))
    0
}
```

A name in a pattern is the value it stands for, so a constant matches the value
it was declared with:

```frost
import "io.frost"

CH_0 :: 48
CH_9 :: 57

digit :: fn(byte: i64) -> bool {
    match byte {
        case CH_0..=CH_9: true
        case _: false
    }
}

main :: fn() -> i64 {
    print("{}\n", digit(50))
    0
}
```

## Structs and enums

```frost
import "io.frost"

Point :: struct { x: i64, y: i64 }

Colour :: enum { Red, Green, Blue }

main :: fn() -> i64 {
    // Every field is named. There is no positional literal, anywhere.
    p := Point { x = 3, y = 4 }
    print("{}\n", p.x + p.y)

    // A variant may leave its enum out where the type is already known, and a
    // struct literal may leave its name out the same way.
    c : Colour = .Green
    q : Point = { x = 1, y = 2 }
    print("{}\n", q.x)
    0
}
```

The contexts that supply the type are the ones that state it: an annotation, a
parameter's declared type, a field's declared type, a declared return type, the
place being assigned to, and an array's element type. Where none of those is
present, the compiler reports an error naming the type it could not resolve.

Layout can be stated where it has to match something outside the program.

```frost
import "io.frost"

// `packed` lays fields end to end, with no padding between them.
Header :: packed struct { tag: u8, length: u32 }

// `align(N)` after a field's type is what that field starts at a multiple of.
// A struct's own alignment is the widest any field asks for.
Uniform :: struct { time: f32, offset: i64 align(16) }

main :: fn() -> i64 {
    print("{}\n", sizeof(Header))    // 5: one byte and four, end to end
    print("{}\n", sizeof(Uniform))   // 32: the align(16) gives the struct one
    0
}
```

## Functions

How a parameter travels is written on the parameter, and the call site writes
the argument alone. There is no `&` in the language.

| mode | written | means |
| --- | --- | --- |
| read | `p: Point` | borrowed to read, the default |
| write | `mut p: Point` | borrowed to write, in place |
| move | `move p: Point` | ownership transferred |

```frost
import "io.frost"

Point :: struct { x: i64, y: i64 }

length_squared :: fn(p: Point) -> i64 { p.x * p.x + p.y * p.y }

scale :: fn(mut p: Point, by: i64) {
    p.x = p.x * by
    p.y = p.y * by
}

main :: fn() -> i64 {
    var p := Point { x = 3, y = 4 }
    print("{}\n", length_squared(p))
    scale(p, 2)                        // no sigil at the call
    print("{}\n", p.x)
    0
}
```

A function may answer with more than one value, by order or by name.

```frost
import "io.frost"

divide :: fn(a: i64, b: i64) -> (quotient: i64, remainder: i64) {
    return a / b, a % b
}

split :: fn(value: i64) -> (high: i64, low: i64) {
    return { high = value / 256, low = value % 256 }
}

main :: fn() -> i64 {
    quotient, remainder := divide(17, 5)
    print("{}\n{}\n", quotient, remainder)

    high, low := split(600)
    print("{}\n{}\n", high, low)
    0
}
```

There is no tuple type behind that. A return type list cannot be stored in a
field, passed as an argument, or bound to one name. To pass a pair around,
declare a struct.

A function is a value, and a parameter may hold one. There are no capturing
closures.

```frost
import "io.frost"

apply :: fn(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }
double :: fn(x: i64) -> i64 { x * 2 }

main :: fn() -> i64 {
    print("{}\n", apply(double, 21))
    0
}
```

## Ownership, moves, and linear types

A struct or enum passed by value *moves*. Using it again is refused where it is
written. A `linear` type must be consumed exactly once, and takes the place of a
destructor.

```frost
import "io.frost"

File :: linear struct { handle: i64 }

open :: fn(handle: i64) -> File { File { handle = handle } }
close :: fn(move f: File) -> i64 { f.handle }

main :: fn() -> i64 {
    f := open(3)
    print("{}\n", close(f))   // consumed exactly once
    // close(f)               // refused: 'f' was already consumed
    0
}                             // leaving without consuming it is refused too
```

`defer` runs a statement where the function leaves, last written first.

```frost
import "io.frost"

main :: fn() -> i64 {
    defer print("second\n")
    defer print("first\n")
    print("body\n")
    0
}
```

## Errors

`-> T ! E` reads "answers with a `T`, or fails with an `E`". `E` is an ordinary
type the program declared. There is no error interface, no backtrace, no
allocation, and no boxing.

```frost
import "io.frost"

Fault :: enum { Negative, TooLarge }

checked :: fn(n: i64) -> i64 ! Fault {
    if (n < 0) { return .Negative }
    if (n > 100) { return .TooLarge }
    n * 2
}

// `?` hands a failure up. The two failure types have to be the same one:
// there is no conversion to write and no `From` to implement.
twice :: fn(n: i64) -> i64 ! Fault {
    first := checked(n)?
    checked(first)?
}

main :: fn() -> i64 {
    // The signature becomes one enum with `Ok { value }` and `Err { error }`,
    // so reading the answer means saying what happens when there is not one.
    match twice(5) {
        case .Ok { value }: print("{}\n", value)
        case .Err { error }: print("failed\n")
    }
    0
}
```

A call that can fail answers with the two-variant enum, and an expression
statement reads neither arm, so writing one for effect alone is refused.
`_ := call()` says the answer was meant to go unread.

`errdefer` runs only where the function leaves through its failure set, which
releases a resource that a `?` would otherwise step over.

## Returning a borrow with `ref T`

The borrow a parameter mode gives is implicit, and an implicit borrow cannot
escape: it may not be stored in a field, put in an array, or returned. That rule
takes the place of lifetimes.

`ref T` is the explicit, checked exception. It binds a second name for a place,
and a function may answer with one, so a container can hand back the element
itself.

```frost
import "io.frost"

at :: fn(values: []i64, index: i64) -> ref i64 {
    ref found := values[index]
    found
}

main :: fn() -> i64 {
    var storage : [3]i64 = [1, 2, 3]
    ref second := storage[1]
    second = 20                     // written through the borrow
    print("{}\n", storage[1])

    held := at(storage, 2)
    print("{}\n", held)             // reading a borrow reads what it borrows
    0
}
```

A `ref T` is a checked address. Reading and writing through it needs no
`unsafe`, and it may still not be stored in a field or an array. A slice `[]T`
*is* storable, and the frame and region checks keep a stored one from outliving
what it views.

## Generics specialize at compile time

A type parameter is written `$T`. Everything monomorphizes, so there is no
runtime dispatch and no boxing.

```frost
import "io.frost"

Pair :: struct($T: Type) { first: T, second: T }

make_pair :: fn(a: $T, b: $T) -> Pair<T> { Pair { first = a, second = b } }
swap :: fn(mut a: $T, mut b: $T) { held := a  a = b  b = held }

// A value parameter is a compile-time integer, which is what sizes an array.
Buffer :: struct($T: Type, $N: usize) { data: [N]T, used: i64 }

main :: fn() -> i64 {
    p := make_pair(3, 4)
    print("{}\n", p.first + p.second)

    var x : i64 = 1
    var y : i64 = 2
    swap(x, y)
    print("{}\n", x)
    0
}
```

Where a type parameter cannot be read off a value argument, pass it with a
leading `$`.

```frost
import "io.frost"

bytes_for :: fn($T: Type, count: i64) -> i64 { count * sizeof(T) }

main :: fn() -> i64 {
    print("{}\n", bytes_for($i64, 16))
    0
}
```

A generic may state what it needs of a type, drawing on a fixed vocabulary of
questions the compiler already answers. There is nothing to register and nothing
to implement.

```frost
import "io.frost"

twice :: fn($T: Type, v: $T) -> T where is_numeric(T) { v + v }

main :: fn() -> i64 {
    print("{}\n", twice($i64, 21))
    0
}
```

## Passing an operation as an argument

An algorithm takes the operation it needs as a compile-time function parameter.
The call inside the specialization goes straight to the function the caller
named, with no pointer to go through.

```frost
import "io.frost"

ascending :: fn(a: i64, b: i64) -> bool { a < b }

smaller :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T, move y: $T) -> $T {
    var held := x
    if (before(y, held)) { held = y }
    held
}

main :: fn() -> i64 {
    print("{}\n", smaller($i64, $ascending, 7, 3))
    0
}
```

When several operations travel together they go in a struct whose fields are
functions. That is a capability bundle, and it takes the place of a trait. A
bundle is a type, an implementation is a constant of that type, and the call
names which one it means. Nothing is registered, nothing is searched for, and
there is no coherence rule.

```frost,sketch
Ordering :: struct($T: Type) {
    less: fn(T, T) -> bool,
    equal: fn(T, T) -> bool,
}

i64_ascending :: Ordering<i64> { less = i64_less, equal = i64_equal }

sort :: fn($T: Type, $ops: Ordering<T>, mut items: []T) { ... }

sort($i64, $i64_ascending, view)
```

Dropping the `$` gives the runtime version of the same declaration: a
`fn(...) -> T` parameter holds a pointer, and a bundle written that way is a
struct holding several.

## A compile-time list of arguments

A parameter written `args: $...` takes as many arguments as the call gives it,
of whatever types they are. The `for` over it is unrolled where it is written,
and the body is compiled once per element.

```frost
import "io.frost"

show_all :: fn(args: $...) {
    for value in args {
        print("{}\n", value)
    }
}

main :: fn() -> i64 {
    show_all(1, 2.5, "three")
    0
}
```

There is no compile-time string parsing, no recursion, and no unbounded loop.
Every compile-time `for` walks a list whose length the call fixed.

## Compile-time questions about a type

`sizeof`, `typename` and `type_id` are compile-time constants, and `fields(T)`
walks a struct's layout. Use it to derive a vertex format or a descriptor table
from the struct that the data already lives in.

```frost
import "io.frost"

Vertex :: struct { x: f32, y: f32, id: i64 }

main :: fn() -> i64 {
    print("{}\n", sizeof(Vertex))
    print("{}\n", typename(Vertex))
    print("{}\n", field_count(Vertex))
    for field in fields(Vertex) {
        print("{}\n{}\n", offset_of(field), sizeof(field))
    }
    0
}
```

## Handles, slabs, and columns

Long-lived data lives in a container and is named by a `Handle`, a small
copyable value. A freed slot bumps its generation, so a stale handle fails the
check instead of reading the new occupant.

```frost
import "slab.frost"
import "io.frost"

Entity :: struct { hp: i64, mana: i64 }

main :: fn() -> i64 {
    var world : Slab<Entity, 16> = slab_new()
    slab_reset($Entity, $16, world)

    hero := slab_insert($Entity, $16, world, Entity { hp = 100, mana = 30 })

    // The subscript is a checked place: it reads and it is written to.
    print("{}\n", world[hero].hp)
    world[hero].hp = world[hero].hp - 25
    print("{}\n", world[hero].hp)

    slab_release($Entity, $16, world, hero)
    print("{}\n", slab_alive($Entity, $16, world, hero))   // 0
    0
}
```

`columns<T, N>` is the same container with each field in its own array, so a
pass reading one field across many elements walks a tight column. To move a
system to structure-of-arrays, change the type and the call prefix.

```frost
import "columns.frost"
import "io.frost"

Particle :: struct { x: i64, y: i64 }

main :: fn() -> i64 {
    var world : columns<Particle, 8> = columns_new()
    columns_reset($Particle, $8, world)
    h := columns_insert($Particle, $8, world, Particle { x = 10, y = 1 })

    print("{}\n", world[h].x)
    world[h].x = 100
    print("{}\n", world[h].x)
    0
}
```

## Where memory comes from

An allocator is an ordinary struct a program declares, and an allocation is an
ordinary call. `uses A` on a function says it draws a capability of type `A`,
and `with a { ... }` says which one for every call inside the block. The block
is also the region: a pointer into `a` may not leave it.

```frost
import "io.frost"

Arena :: struct($N: usize) { data: [N]u8, offset: i64 }

alloc_int :: fn($N: usize, mut a: Arena<N>) -> ^i64 {
    slot := ptr_to(a.data[a.offset])
    a.offset = a.offset + 8
    unsafe { ptr_cast($i64, slot) }
}

make_two :: fn() -> i64 uses Arena<256> {
    p := alloc_int($256, arena)
    unsafe { p^ = 10 }
    q := alloc_int($256, arena)
    unsafe { q^ = 32 }
    unsafe { p^ + q^ }
}

main :: fn() -> i64 {
    var scratch : Arena<256> = Arena { data = [0; 256], offset = 0 }
    with scratch {
        print("{}\n", make_two())
    }
    0
}
```

The body reaches the capability by the type's own name with the first letter
lowercased. That is why `make_two` writes `arena` while its signature declares
no parameter by that name.

## Unchecked operations

Four operations may be written only inside an `unsafe` block: reading or writing
through a raw pointer, `ptr_cast`, `slice_from`, and calling an `extern fn` that
is not marked `safe`. Everything else in the language is checked.

```frost
import "io.frost"

main :: fn() -> i64 {
    var value : i64 = 41

    // Taking an address is safe. Reading through one is not.
    p := ptr_to(value)
    unsafe { p^ = p^ + 1 }
    print("{}\n", value)

    // A block's value is its trailing expression, so a gated result leaves one.
    held := unsafe { p^ }
    print("{}\n", held)
    0
}
```

`cast($T, value)` converts between scalars where the conversion loses something.
It needs no block, since the result is defined for every input.

```frost
import "io.frost"

main :: fn() -> i64 {
    wide : i64 = 300
    narrow := cast($u8, wide)
    print("{}\n", narrow)
    0
}
```

## Talking to C

```frost,sketch
// A declaration, and the name is the C symbol.
printf :: extern fn(fmt: ^i8, value: i64) -> i32

// `safe` says a call needs no `unsafe` block.
clock :: safe extern fn() -> i64

// `value` hands a struct to C by value rather than by address.
draw :: extern fn(value rect: Rect)
```

## Modules

A file is a module. `import` names another file whose declarations join this
one's, and `export` says which of this file's names leave it. The namespace is
flat, so a name is the same name everywhere and one grep finds every use of it.
Two modules exporting one name is a refusal.

```frost,sketch
import "io.frost"
import "../lib/engine/world.frost"

export spawn, despawn
```

## Tests

```frost
import "io.frost"

add :: fn(a: i64, b: i64) -> i64 { a + b }

test "adding two numbers" {
    assert(add(2, 2) == 4)
}

main :: fn() -> i64 { 0 }
```

## Lane-wise arithmetic on a small array

A small array of numbers takes the arithmetic operators lane by lane. Frost
defines the operators over `[4]f32` itself, so there is no SIMD type to reach
for. (The growable `Vec<T>` in the standard library is a heap container, and
unrelated to this.)

```frost
import "io.frost"

main :: fn() -> i64 {
    a : [4]f32 = [1.0, 2.0, 3.0, 4.0]
    b : [4]f32 = [10.0, 20.0, 30.0, 40.0]
    c := a + b
    print("{}\n", c[0])
    0
}
```

## What Frost leaves out, and what to write instead

No garbage collector, no runtime, no exceptions, no panics to catch, no
destructors, no traits, no interfaces, no inheritance, no methods, no operator
overloading, no capturing closures, no iterators, no macros, no lifetimes, no
tuples, no null, no implicit conversion, no truthiness, no positional struct
literals, and no hidden allocation.

Each has a replacement above: a `linear` type instead of a destructor, a
capability bundle instead of a trait, a compile-time function parameter instead
of a closure, an index loop instead of an iterator, a generational handle
instead of a pointer into a container, and a failure set in the signature
instead of an exception.

## The standard library

`std/` is twenty files of ordinary Frost, compiled the way your own modules are.
Nothing is imported for you: a program that prints says `import "io.frost"`.
Everything below is covered in full under
[the standard library](std/index.md).

### Text

`str` is a byte slice, and `strings.frost` asks questions of one.

```frost
import "io.frost"
import "strings.frost"

main :: fn() -> i64 {
    name := "frost.frost"
    print("{}\n", str_len(name))                  // 11
    print("{}\n", str_ends_with(name, ".frost"))  // 1
    print("{}\n", str_index_of(name, "."))        // 5
    print("{}\n", str_slice(name, 0, 5))          // frost
    print("{}\n", str_to_i64("42"))               // 42
    0
}
```

### A growable array

`Vec<T>` is `linear`, so a vector nothing frees is a compile error.
`vec_slice` hands out the live elements as a bounds-checked slice.

```frost
import "io.frost"
import "vec.frost"

main :: fn() -> i64 {
    var scores := vec_new($i64, 4)
    vec_push($i64, scores, 10)
    vec_push($i64, scores, 30)
    vec_push($i64, scores, 20)

    var total : i64 = 0
    for value in vec_slice($i64, scores) {
        total = total + value
    }
    print("{} in {}\n", total, vec_len($i64, scores))   // 60 in 3

    vec_free($i64, scores)
    0
}
```

`fixed.frost` is the same array over storage you hand it, for a container that
lives in an arena.

### A hash map

Keys of any type. `$text_keys` is the `Hashing<K>` bundle for text, and hashing
and comparison fold to direct calls at each site.

```frost
import "io.frost"
import "map.frost"

main :: fn() -> i64 {
    var ages := map_new($Text, $i64, 8)
    map_put($Text, $i64, $text_keys, ages, text("ada"), 36)
    map_put($Text, $i64, $text_keys, ages, text("alan"), 41)

    print("{}\n", map_get($Text, $i64, $text_keys, ages, text("ada"), 0))
    print("{}\n", map_has($Text, $i64, $text_keys, ages, text("grace")))
    print("{}\n", map_len($Text, $i64, ages))
    map_free($Text, $i64, ages)
    0
}
```

`map_get` takes the value to answer with when the key is absent. Ask `map_has`
where absence and a zero differ.

### A value that may be absent

`Option<T>` is a generic enum, with no compiler support behind it.

```frost
import "io.frost"
import "option.frost"

lookup :: fn(key: i64) -> Option<i64> {
    if (key == 1) { return option_some($i64, 100) }
    option_none($i64)
}

main :: fn() -> i64 {
    hit := lookup(1)
    miss := lookup(2)
    print("{}\n", option_unwrap_or($i64, hit, 0))     // 100
    print("{}\n", option_unwrap_or($i64, miss, 0))    // 0
    0
}
```

### Sorting

`sort` orders a slice in place and takes an `Ordering<T>`, a struct of two
functions, as a compile-time argument.

```frost
import "io.frost"
import "ordering.frost"
import "sort.frost"

main :: fn() -> i64 {
    var numbers : [5]i64 = [4, 1, 5, 3, 2]
    view : []i64 = numbers
    sort($i64, $i64_ascending, view)
    for value in view {
        print("{} ", value)      // 1 2 3 4 5
    }
    print("\n")
    0
}
```

`i64_descending`, `f64_ascending` and `f64_descending` come with it, and
`sort_vec` does the same to a `Vec<T>`.

### Assembling text

`print` writes a line. `Builder` collects bytes so a program can build a string
and keep it.

```frost
import "io.frost"
import "format.frost"

main :: fn() -> i64 {
    var line := builder_new(64)
    builder_str_value(line, "hp ")
    builder_int(line, 75)
    builder_str_value(line, "/100")
    print("{}\n", builder_str(line))   // hp 75/100
    builder_free(line)
    0
}
```

### Files and JSON

`fs_read` answers with a `ReadResult` holding the whole file. `json_parse`
reads text into a flat array of nodes reached by index.

```frost
import "io.frost"
import "fs.frost"
import "json.frost"

main :: fn() -> i64 {
    fs_write("hero.json", "{\"hp\": 75, \"name\": \"ada\"}")

    result := fs_read("hero.json")
    var document := json_parse(result.text)
    root := json_root(document)
    hp := json_member(document, root, "hp")
    print("{}\n", json_number(document, hp))   // 75

    json_free(document)
    fs_free(result)
    fs_remove("hero.json")
    0
}
```

### Math

`math.frost` is vectors, matrices and quaternions at `f32`, and
`math64.frost` is the same library at `f64` with a `d` on every name.

```frost
import "io.frost"
import "math.frost"

main :: fn() -> i64 {
    up := vec3(0.0, 1.0, 0.0)
    over := vec3(1.0, 0.0, 0.0)
    turn := quat_from_axis_angle(up, radians(90.0))
    spun := quat_rotate_vec3(turn, over)
    print("{}\n", spun.z * 1000.0)   // -1000, +X turned onto -Z
    0
}
```

### An entity-component system

`ecs.frost` stores entities by archetype, so a system reading one component
walks a packed column.

```frost
import "io.frost"
import "ecs.frost"

Health :: struct { points: i64 }

main :: fn() -> i64 {
    var world := ecs_new()
    health := ecs_register($Health, world)

    hero := ecs_spawn(world)
    ecs_add($Health, world, hero, health, Health { points = 100 })
    ecs_set($Health, world, hero, health, Health { points = 75 })

    held := ecs_get($Health, world, hero, health)
    print("{}\n", held.points)   // 75

    ecs_free(world)
    0
}
```

`ecs_slice` hands a whole column to a system, and `snapshot.frost` writes a
world to bytes and reads it back.

### Threads

`spawn` takes a function and a context pointer, `join` waits, and `atomic_add`
is the one shared-memory operation.

```frost
import "io.frost"
import "thread.frost"

Counter :: struct { total: i64 }

bump :: fn(context: ^u8) {
    counter := unsafe { ptr_cast($Counter, context) }
    var index : i64 = 0
    while (index < 1000) {
        unsafe { atomic_add(ptr_to(counter^.total), 1) }
        index = index + 1
    }
}

main :: fn() -> i64 {
    var counter := Counter { total = 0 }
    handle := unsafe { spawn(bump, ptr_cast($u8, ptr_to(counter))) }
    bump(unsafe { ptr_cast($u8, ptr_to(counter)) })
    join(handle)
    print("{}\n", counter.total)   // 2000
    0
}
```

This is the reasonable-C floor. Nothing here is checked for data races.

## Where to go next

- [A tour of Frost](tour.md), the same ground at a slower pace.
- [Coming from Rust](coming-from-rust.md), if Rust is where you are arriving
  from.
- [The language reference](reference/conformance.md), which states every rule in
  full.
- [The standard library](std/index.md), for the twenty modules in full.
- [Design philosophy](design/philosophy.md), for the reasoning behind the
  absences above.
