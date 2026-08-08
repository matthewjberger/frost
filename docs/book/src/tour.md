# A tour of Frost

This chapter walks through the language by example. The test suite compiles
every block below, so the compiler accepts what is written here. Blocks marked
`sketch` show a shape and skip compilation. `examples/tour.frost` is a runnable
program over most of the same ground. For the reasoning behind these choices,
see [philosophy.md](design/philosophy.md). For the safety rules, see
[memory-safety.md](design/memory-safety.md). Each section links the reference
chapter that states its rules in full.

`frost program.frost` compiles and runs a program in one step. To keep the
executable:

```
frost program.frost --link -o program && ./program
```

## Declarations, values, and functions

`::` declares a constant (including a function). `:=` binds an inferred local,
`:` gives an explicit type, and `var` makes a binding assignable. There are no
methods. Behavior lives in free functions.

```frost
import "io.frost"

square :: fn(x: i64) -> i64 { x * x }

main :: fn() -> i64 {
    n := 6                 // inferred i64
    var total : i64 = 0    // explicit, mutable
    for i in 0..n {
        total = total + square(i)
    }
    print("{}\n", total)  // 55
    0
}
```

Output goes through the library. `import "io.frost"` brings in `print`, which
takes a format string and as many values as the line names. Each `{}` is where
the next value goes, and the value's type decides how it is written:

```frost,sketch
print("hp {} of {}\n", entity.hp, entity.max)
```

An integer, a float, a `bool` and a `str` all go in a hole. Nothing is appended,
so a line ends with the `\n` written into the literal. The compiler checks the
count where the call is written and picks the writer for each value while it
compiles the call, so the program does one direct write per value. Printing
lives entirely in the library ([text-and-io.md](std/text-and-io.md)).

Integer widths (`i8`..`i64`, `u8`..`u64`), floats (`f32`, `f64`), and `bool`
are all value (copy) types. Control flow is `if`/`else` (an expression),
`while`, `for`, `break`, `continue`, and `match`.

A `for` walks a range, and it walks a sequence the same way:

```frost,sketch
var total : i64 = 0
for value in numbers {          // a slice, an array, or a `str`
    total = total + value
}
for index, value in numbers {   // the position as well
    total = total + index * value
}
```

Frost has no iterator protocol. `for value in numbers` is the index-and-bound
loop written out, so the backend sees what you would have written by hand. The
sequence is evaluated once, so a call in that position happens once however many
elements it answers with.

A function that answers with more than one value declares a return type list,
and the caller binds the values by name:

```frost,sketch
divide :: fn(a: i64, b: i64) -> (quotient: i64, remainder: i64) {
    return a / b, a % b
}

quotient, remainder := divide(17, 5)      // 3 and 2
```

Every value in the list carries a name, which says which is which and gives
`return` by name a field to write:
`return { high = value / 256, low = value % 256 }`. Reach for a named return
when two values share a type and could be swapped. Return by order where the
function is a short table of answers.

Frost has no tuple type. `(i64, i64)` is a return type list, and it appears in
a signature and nowhere else: you cannot store it in a field, pass it as an
argument, or bind it to one name. To pass a pair around, declare a struct. A
name in the list labels one of the values. There is nothing to assign to it,
and no bare `return` that hands back whatever it holds, so a function's answer
is always written out at the `return`. The full rules are 5.2a of
[declarations.md](reference/declarations.md).

## Structs and enums, plain data

A `struct` is just its fields. An `enum` is a tagged union with payloads.
Neither carries methods.

```frost
import "io.frost"

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
    print("{}\n", p.x + p.y)                                     // 7
    print("{}\n", area(Shape::Rect { width = 4, height = 5 }))   // 20
    0
}
```

Structs pass and return by value, and `match` binds payload fields.

Wherever the type is already stated, a variant can leave its enum name out, the
same shorthand a `case .Circle` arm uses:

```frost,sketch
s : Shape = .Circle { radius = 4 }                    // the annotation says which
print("{}\n", area(.Rect { width = 4, height = 5 })) // the parameter does
round :: fn(r: i64) -> Shape { return .Circle { radius = r } }   // the return
```

A struct literal leaves its name out the same way, and the two nest, each inner
literal taking its type from the field it fills:

```frost,sketch
p : Point = { x = 3, y = 4 }
```

Where there is nothing to read the type from, as in a bare `c := .Red` or
`p := { x = 1, y = 2 }`, the compiler reports what it could not resolve. The
field names always stay. Frost has no positional literal, so the name says
where a value lands.

## Borrowing is a parameter mode

There is no `&` in the language. How a parameter is passed is a property of the
parameter, written on its declaration, and the call site says nothing:

| mode | written | means |
| --- | --- | --- |
| read | `p: Point` | borrowed to read, the default |
| write | `mut p: Point` | borrowed to mutate in place |
| move | `move p: Point` | ownership transferred |

```frost,sketch
import "io.frost"

scale :: fn(mut p: Point, k: i64) {   // borrowed to mutate in place
    p.x = p.x * k
    p.y = p.y * k
}

main :: fn() -> i64 {
    var p := Point { x = 3, y = 4 }
    scale(p, 2)                       // no sigil at the call
    print("{}\n", p.x)               // 6
    0
}
```

A parameter mode gives you an implicit borrow, and an implicit borrow cannot
escape: it may not be stored in a field, put in an array, or returned. A borrow
that cannot leave the call needs nothing said about how long it lives, which is
why Frost has no lifetimes.

`ref T` is the explicit, checked exception. `ref name := place` binds a borrow
of a place instead of copying it, and a function may answer with one, so a
container can hand back an element:

```frost,sketch
import "io.frost"

at :: fn(points: []Point, index: i64) -> ref Point {
    ref result := points[index]
    result
}

main :: fn() -> i64 {
    var storage : [3]Point = [Point { x = 0, y = 0 }; 3]
    held := at(storage, 1)
    held.x = 9
    print("{}\n", storage[1].x)  // 9, written through the borrow
    0
}
```

A `ref T` is a checked address: reading and writing through it needs no
`unsafe`, and it still may not be stored in a struct field, an array, or any
other container. Chapter 3.3 of [types.md](reference/types.md) has the rule and
chapter 8 of [ownership.md](reference/ownership.md) has the checks.

A slice is storable. `[]T` is an address with a length beside it and an ordinary
type, so a struct field may hold one, and a parser holds views into a buffer it
does not own. The frame and region checks refuse a function that answers with a
view whose storage they cannot trace. See
[memory-safety.md](design/memory-safety.md).

Raw pointers `^T` exist as an explicit, unchecked escape hatch for FFI, and
`ptr_to(x)` is how you take one.

## Move checking and linear resources

Non-`Copy` values (structs, enums) *move* when passed by value. Using one again
is a compile error. A `linear` type must be consumed exactly once, which is
how Frost replaces destructors:

```frost
import "io.frost"
File :: linear struct { fd: i64 }
open  :: fn(n: i64) -> File { File { fd = n } }
close :: fn(move f: File) -> i64 { f.fd }   // terminal consumer

run :: fn() {
    f := open(3)
    print("{}\n", close(f))   // consumes f exactly once
    // close(f)                // error: use of moved value 'f'
}                              // dropping f without consuming would also be an error
```

Chapter 9 of [linear.md](reference/linear.md) has the consume rules, including
what happens on a branch that consumes on one path and not the other.

## Errors are values, in the signature

A function that can fail says what it fails with, and `?` hands a failure up:

```frost
Parse :: struct { at: i64, code: i64 }

digit :: fn(text: str, index: i64) -> i64 ! Parse {
    byte := text[index]
    if (byte < 48 || byte > 57) {
        return { at = index, code = byte }
    }
    byte - 48
}

number :: fn(text: str) -> i64 ! Parse {
    var total : i64 = 0
    var index : i64 = 0
    while (index < str_len(text)) {
        total = total * 10 + digit(text, index)?
        index = index + 1
    }
    total
}
```

`-> i64 ! Parse` reads "answers with an i64, or fails with a Parse". `Parse` is
an ordinary struct this program declared. There is no error interface to
implement, no backtrace, no allocation, and no boxing. The compiler turns the
signature into one enum with two variants, `Ok { value }` and `Err { error }`,
which is where the names come from at the match:

```frost,sketch
match number(text) {
    case .Ok { value }: { print("{}\n", value) }
    case .Err { error }: { print("{}\n", error.at) }
}
```

A match on an enum covers every variant, so reading the value means saying what
happens when there is not one. The two failure types at a `?` have to be the
same type, because there is no conversion and no `From` to write. A result
carrying a `linear` value is itself linear, so a fallible function that answers
with a resource cannot be called and ignored.

Frost has no exceptions, no panics to catch, and no error codes checked by
convention. A failure that is a bug rather than a condition in the world is an
assertion, and an assertion aborts. Chapter 5.2b of
[declarations.md](reference/declarations.md) is the full account.

## Generational handles and slabs

Long-lived data lives in a slab, and you reach it through a `Handle<T>`, a
small copyable value. A freed-and-reused slot bumps its generation, so an old
handle can never read the new occupant.

The slab lives in the library. `std/slab.frost` is ordinary Frost: storage, a
free list, generation counters, and the packing of a slot index and a generation
into one handle, all written out. The compiler supplies one piece, the place
`world[handle]`, which it offers for a slab-shaped struct (a `storage` array
beside a parallel `generations` array). Borrows are second-class, so the
library cannot write that piece for itself.

```frost
import "slab.frost"
import "io.frost"

Entity :: struct { hp: i64, mana: i64 }

main :: fn() -> i64 {
    var world : Slab<Entity, 16> = slab_new()
    slab_reset(world)

    hero := slab_insert(world, Entity { hp = 100, mana = 30 })

    print("{}\n", world[hero].hp)        // 100
    world[hero].hp = world[hero].hp - 25  // the subscript is a place to write
    print("{}\n", world[hero].hp)        // 75

    slab_release(world, hero)
    print("{}\n", slab_alive(world, hero))   // 0, the generation moved on
    0
}
```

The arrays are written out at construction because a struct literal cannot run
code, and `slab_reset` fills in the free list. Passing `world[hero]` to a
function borrows the element, and that borrow is a parameter mode like any
other. `examples/native/entity_system.frost` is this over an entity enum, and
chapter 10 of [handles-and-pools.md](reference/handles-and-pools.md) states the
generational rule.

## Structure-of-arrays with columns

`columns<T, N>` stores each field of `T` in its own array rather than storing
whole elements back to back, so a pass reading one field across many elements
walks a tight column. The slab's generational handle carries over unchanged:
`c[h]` is still a checked place, and `c.field` is the whole column, a slice for
a hot loop:

```frost
import "columns.frost"
import "io.frost"

Particle :: struct { x: i64, y: i64 }

main :: fn() -> i64 {
    var world : columns<Particle, 8> = columns_new()
    columns_reset(world)
    h := columns_insert(world, Particle { x = 10, y = 1 })

    print("{}\n", world[h].x)     // 10, checked at the handle's slot
    world[h].x = 100               // scatter a field back to the slot
    // world.x is the whole [8]i64 column, and coerces to a []i64 slice
    0
}
```

Moving a system from a slab to structure-of-arrays is changing `Slab<T, N>` to
`columns<T, N>` and the `slab_` prefix to `columns_`. See
[pools-and-columns.md](design/pools-and-columns.md).

## Generics specialize at compile time

Generic functions and structs monomorphize, so there is no runtime dispatch. A
type parameter is written `$T`:

```frost
import "io.frost"

Pair :: struct($T: Type) { first: T, second: T }

make_pair :: fn(a: $T, b: $T) -> Pair<T> { Pair { first = a, second = b } }
swap      :: fn(mut a: $T, mut b: $T) { t := a  a = b  b = t }

main :: fn() -> i64 {
    p := make_pair(3, 4)               // Pair<i64> inferred
    print("{}\n", p.first + p.second) // 7

    var x : i64 = 1
    var y : i64 = 2
    swap(x, y)
    print("{}\n", x)                  // 2
    0
}
```

`sizeof(T)` is a compile-time constant, so a generic function can size its own
type parameter. When a type parameter can't be inferred from a value argument
(for example a function that only uses `sizeof(T)`), declare it as `$T: Type`
and pass the type explicitly with a leading `$`:

```frost,sketch
bytes_for :: fn($T: Type, count: i64) -> i64 { count * sizeof(T) }

main :: fn() -> i64 {
    print("{}\n", bytes_for($Entity, 16))   // pass the type with $
    0
}
```

A parameter may also be an integer, written `$N: usize`, which sizes the `[N]T`
field of a `Slab<T, N>`. Type parameters are erased: they drive
monomorphization and carry no runtime cost.

A generic can also state what it needs of a type, drawn from a fixed vocabulary
of questions the compiler already answers about a type:

```frost
twice :: fn($T: Type, v: $T) -> T where is_numeric(T) { v + v }
```

Nothing registers into that vocabulary and nothing implements it. The bound is
a precondition, checked against the line the caller wrote. The vocabulary is
11.4a of [generics.md](reference/generics.md).

## Passing an operation as an argument

A generic algorithm takes the operation it needs as a compile-time function
parameter, which covers what you would write a trait bound for elsewhere. The
parameter can state the signature it requires, and the call inside the
specialization is direct, with no pointer to go through:

```frost
import "io.frost"
ascending :: fn(a: i64, b: i64) -> bool { a < b }

best :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T, move y: $T) -> $T {
    var result := x
    if (before(y, result)) { result = y }
    result
}

main :: fn() -> i64 {
    print("{}\n", best($ascending, 7, 3))   // 3
    0
}
```

When several operations travel together, they go in a struct whose fields are
functions. That struct is a capability bundle, and it stands where a trait
would:

```frost,sketch
Ordering :: struct($T: Type) {
    less: fn(T, T) -> bool,
    equal: fn(T, T) -> bool,
}

i64_ascending :: Ordering<i64> { less = i64_less, equal = i64_equal }

sort :: fn($T: Type, $ops: Ordering<T>, mut items: []T) {
    ...
    if (ops.less(items[j], items[j - 1])) { ... }
}

sort($i64_ascending, view)
```

The bundle is a type, an implementation is a constant of it, and the call names
which one it means. Nothing registers, nothing is searched for, and there is no
coherence rule to learn. Since `$ops` is a compile-time argument,
`ops.less(a, b)` folds to a direct call to `i64_less`, and the specialization
holds no function pointer at all. `std/ordering.frost` and `std/sort.frost` are
this written out, and 11.4b of [generics.md](reference/generics.md) has the
rest.

When the operation varies at runtime, drop the `$` and the same declaration
gives an ordinary value. A `fn(...) -> T` parameter holds a pointer, and a
bundle without the `$` is a struct holding several. There are no capturing
closures.

```frost
import "io.frost"

apply :: fn(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }
double :: fn(x: i64) -> i64 { x * 2 }

main :: fn() -> i64 {
    print("{}\n", apply(double, 21))   // 42
    0
}
```

## A compile-time list of arguments

A parameter written `args: $...` takes as many arguments as the call gives it,
of whatever types they are:

```frost
import "io.frost"

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

main :: fn() -> i64 {
    printall(1, 2.5, "three")
    0
}
```

The `for` is unrolled where it is written. The body is compiled once per element
with `value` standing for that element, so the program does three writes of
three different types. The `if` over what a type is gets its answer at expansion
time, and the branch that cannot run is dropped before anything checks it, so
one body serves elements of different types.

There is no compile-time string parsing, no recursion and no unbounded loop.
Everything here walks a list whose length the call fixed, so expansion costs
what the program's own text costs and no more. 11.1c of
[generics.md](reference/generics.md) has `args[0]`, handing a list on, and
`g(T) for T in list` in an argument position.

## A table over a type's fields

A vertex format, a uniform layout and a descriptor table are all the same
thing: offsets and sizes over a struct you already declared. The compiler
worked those numbers out to lay the struct out, so write the table over them:

```frost
import "math.frost"
import "io.frost"

Vertex :: struct { position: Vec3, normal: Vec3, uv: Vec2, id: i64 }

main :: fn() -> i64 {
    print("{}\n", field_count(Vertex))
    for field in fields(Vertex) {
        print("{}\n{}\n", offset_of(field), sizeof(field))
        if (is_float(field)) { print("{}\n", 1) } else { print("{}\n", 0) }
    }
    0
}
```

This `for` is unrolled too: the body is compiled once per field. `T` may be a
type parameter, so one description serves every type a call names, and the
table cannot drift from the struct because it is written once.

A field is a name to ask questions of. `offset_of`, `sizeof` and the type
predicates are the questions available, and naming a field anywhere else is an
error. A field's name cannot be read, which keeps this to layout (11.1d of
[generics.md](reference/generics.md)).

## Arrays are bounds-checked

A fixed-size array `[N]T` knows its length, and every index is checked:

```frost
import "io.frost"

main :: fn() -> i64 {
    arr := [10, 20, 30]
    print("{}\n", arr[2])   // 30
    // arr[5]                // aborts: index 5 out of bounds for length 3
    0
}
```

## Tests

A `test` block is a named unit test, and `assert` fails it when the condition is
false. Run every test in a file with `frost --test file.frost`.

```frost
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

`extern fn` links against any C library with the natural ABI, with no glue. A
string literal is laid down NUL-terminated as well, so it reaches a `^i8`
parameter as a plain C string with no conversion.

```frost
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    unsafe { printf("%lld\n", 42) }
    0
}
```

Calling C is unchecked, so the call sits in an `unsafe` block. The blocks in a
program are the complete list of places to look when memory has been corrupted.
A foreign function that takes and answers with numbers, with no pointer
anywhere, is declared `safe extern fn` and needs no block. That is how
`std/math.frost` reaches `sqrtf`. See
[c-compatibility.md](impl/c-compatibility.md).

## The standard library

The library under `std/` is ordinary Frost, imported by name (`import
"math.frost"`). It carries `str` helpers, a growable `Vec<T>` and a hash map,
file and formatted IO, an `Ordering<T>` bundle with a sort that takes one, the
`slab` and `columns` containers, an archetype ECS, and a graphics-math library
of vectors, matrices, and quaternions. The math library ships at both
precisions, `std/math.frost` in `f32` and `std/math64.frost` in `f64` with a `d`
on every name. It is described in [math.md](std/math.md), and
`examples/native/math_transform.frost` puts it through a model-view-projection.

## Navigating a Frost codebase

Every top-level thing is declared with the same shape, `name :: value`, so the
name is always first and left-aligned. Reading an unfamiliar program comes down
to four habits:

- Find any definition, of any kind, by grepping for the name followed by `::`.
  `rg "Entity ::"` finds the struct, function, enum, or constant named `Entity`,
  and you do not need to know which kind it is first.
- A file's public surface is its `export` line at the top. Read that to see what
  a module offers without scanning every declaration.
- A file's dependencies are its `import` lines at the top, and an import is not
  transitive, so that list is complete.
- `::` declares a constant or item, `:=` introduces a local, `:` annotates a
  type, and `=` assigns. Each symbol means one thing, so nothing has to be read
  in context to know what it did.

The exported namespace is flat and a name carries its own prefix by convention
(`vec3_add` in place of a qualified `math.add`), so every name is a single token
to search for. [modules.md](impl/modules.md) covers imports, exports, and the
rename escape hatch for a collision.

## A complete program

Entities in a slab, behavior as a free function over an enum, and identity as a
handle that stays checkable after the slot it names is reused.

```frost
import "slab.frost"
import "io.frost"

Kind   :: enum { Player, Enemy { damage: i64 } }
Entity :: struct { hp: i64, kind: Kind }

delta :: fn(e: Entity) -> i64 {
    match e.kind {
        case .Player: 0
        case .Enemy { damage }: 0 - damage
    }
}

main :: fn() -> i64 {
    var world : Slab<Entity, 16> = slab_new()
    slab_reset(world)

    player := slab_insert(world,
        Entity { hp = 100, kind = .Player })
    goblin := slab_insert(world,
        Entity { hp = 30, kind = .Enemy { damage = 15 } })

    world[player].hp = world[player].hp + delta(world[goblin])
    print("hp {}\n", world[player].hp)                 // 85

    slab_release(world, goblin)
    print("{}\n", slab_alive(world, goblin))   // 0
    0
}
```

You pass and store handles. The borrow of `world[goblin]` lasts only for the
call to `delta`, and releasing a slot invalidates the handles to it by
generation.

## Where to next

- Runnable programs live in `examples/native/`. Start with `game_world.frost`
  (the entity-component system) and `generic_slab.frost` (the slab written out
  in full).
- [coming-from-rust.md](coming-from-rust.md) if you already think in Rust.
- [patterns.md](patterns.md) for what the language rewards and what it merely
  permits.
- [architecture.md](impl/architecture.md) explains the compiler pipeline and
  what the native path supports.
