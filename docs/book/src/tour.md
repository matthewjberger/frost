# A tour of Frost

This is a walk through the language by example. Every snippet below compiles and
runs on both native backends, and `examples/tour.frost` is a runnable program
over most of the same ground. For the *why* behind these choices, see
[philosophy.md](design/philosophy.md). For the safety rules, see
[memory-safety.md](design/memory-safety.md). Each section links the reference
chapter that states its rules in full.

`frost program.frost` compiles and runs a program in one step. To keep the
executable:

```
frost program.frost --link -o program && ./program
```

## Declarations, values, and functions

`::` declares a constant (including a function). `:=` binds an inferred local,
`:` gives an explicit type, and `mut` makes a binding assignable. There are no
methods. Behavior lives in free functions.

```frost
square :: fn(x: i64) -> i64 { x * x }

main :: fn() -> i64 {
    n := 6                 // inferred i64
    mut total : i64 = 0    // explicit, mutable
    for i in 0..n {
        total = total + square(i)
    }
    print total            // 55
    0
}
```

`print` is a statement, not a library function, so showing a number needs
nothing declared and nothing imported. `print expr` writes one value and a
newline. The format form fills each `{}` from the values after the literal:

```frost
print "hp {} of {}", entity.hp, entity.max
```

The compiler reads that literal and splits it into the pieces to write where the
statement is written, so no format exists at run time and nothing parses one.
The printable set is closed and lives in the compiler: the integer widths, the
floats, `bool`, `Handle`, `str` and `^i8`. Anything else is an error naming the
type ([statements.md](reference/statements.md)).

Integer widths (`i8`..`i64`, `u8`..`u64`), floats (`f32`, `f64`), and `bool`
are all value (copy) types. Control flow is `if`/`else` (an expression),
`while`, `for`, `break`, `continue`, and `match`.

A `for` walks a range, and it walks a sequence the same way:

```frost
mut total : i64 = 0
for value in numbers {          // a slice, an array, or a `str`
    total = total + value
}
for index, value in numbers {   // the position as well
    total = total + index * value
}
```

There is no iterator and nothing to implement. `for value in numbers` is the
index-and-bound loop written out, so what the backend sees is what you would
have written by hand. The sequence is evaluated once, so a call in that position
happens once however many elements it answers with.

A function that answers with more than one value declares a return type list,
and the caller binds the values by name:

```frost
divide :: fn(a: i64, b: i64) -> (quotient: i64, remainder: i64) {
    return a / b, a % b
}

quotient, remainder := divide(17, 5)      // 3 and 2
```

Naming the values is optional and worth it whenever two of them share a type. A
named list can also be returned by name rather than by order, as in
`return { high = value / 256, low = value % 256 }`.

There is no tuple type behind any of that. `(i64, i64)` is a return type list
and nothing else: it cannot be stored in a field, passed as an argument, or
bound to one name. A program that wants to pass a pair around declares a struct,
so every aggregate has a name its author chose. A returned name is not a
variable either: there is nothing to assign to it and no bare `return` that
hands back whatever it holds, so what a function answers with is always written
at the `return`. The full rules are 5.2a of
[declarations.md](reference/declarations.md).

## Structs and enums, plain data

A `struct` is just its fields. An `enum` is a tagged union with payloads.
Neither carries methods.

```frost
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
    print p.x + p.y                                     // 7
    print area(Shape::Rect { width = 4, height = 5 })   // 20
    0
}
```

Structs pass and return by value, and `match` binds payload fields.

A variant can leave its enum out wherever the type is already stated, which is
the construction counterpart of the `case .Circle` an arm writes:

```frost
s : Shape = .Circle { radius = 4 }                 // the annotation says which
print area(.Rect { width = 4, height = 5 })        // the parameter does
round :: fn(r: i64) -> Shape { return .Circle { radius = r } }   // the return
```

A struct literal leaves its name out the same way, and the two nest, each inner
literal taking its type from the field it fills:

```frost
p : Point = { x = 3, y = 4 }
```

Where there is nothing to read the type from, as in a bare `c := .Red` or
`p := { x = 1, y = 2 }`, it is an error naming what could not be resolved rather
than a guess. What never goes away is the field names: there is no positional
literal in Frost, because the name is what says where a value lands.

## Borrowing is a parameter mode

There is no `&` in the language. How a parameter is passed is a property of the
parameter, written on its declaration, and the call site says nothing:

| mode | written | means |
| --- | --- | --- |
| read | `p: Point` | borrowed to read, the default |
| write | `mut p: Point` | borrowed to mutate in place |
| move | `move p: Point` | ownership transferred |

```frost
scale :: fn(mut p: Point, k: i64) {   // borrowed to mutate in place
    p.x = p.x * k
    p.y = p.y * k
}

main :: fn() -> i64 {
    mut p := Point { x = 3, y = 4 }
    scale(p, 2)                       // no sigil at the call
    print p.x                         // 6
    0
}
```

The borrow a parameter mode gives you is implicit, and an implicit borrow cannot
escape: it may not be stored in a field, put in an array, or returned. That is
the rule that removes lifetimes, because a borrow that cannot leave the call
needs nothing said about how long it lives.

The explicit, checked exception is `ref T`. `ref name := place` binds a borrow
of a place rather than a copy of it, and a function may answer with one, which
is what lets a container hand back an element instead of a copy of it:

```frost
at :: fn(points: []Point, index: i64) -> ref Point {
    ref result := points[index]
    result
}

main :: fn() -> i64 {
    mut storage : [3]Point = [Point { x = 0, y = 0 }; 3]
    held := at(storage, 1)
    held.x = 9
    print storage[1].x       // 9, written through the borrow
    0
}
```

A `ref T` is a checked address: reading and writing through it needs no
`unsafe`, and it still may not be stored in a struct field, an array, or any
other container. Chapter 3.3 of [types.md](reference/types.md) has the rule and
chapter 8 of [ownership.md](reference/ownership.md) has the checks.

A slice is not held that way. `[]T` is an address with a length beside it and an
ordinary storable type, so a struct field may hold one, which is what a parser
holding views into a buffer it does not own is made of. What keeps a stored
slice honest is not the type but the frame and region checks, which refuse a
function that answers with a view whose storage they cannot trace. See
[memory-safety.md](design/memory-safety.md).

Raw pointers `^T` exist as an explicit, unchecked escape hatch for FFI, and
`ptr_to(x)` is how you take one.

## Move checking and linear resources

Non-`Copy` values (structs, enums) *move* when passed by value. Using one again
is a compile error. A `linear` type must be consumed exactly once, which is
how Frost replaces destructors:

```frost
File :: linear struct { fd: i64 }
open  :: fn(n: i64) -> File { File { fd = n } }
close :: fn(move f: File) -> i64 { f.fd }   // terminal consumer

run :: fn() {
    f := open(3)
    print close(f)   // consumes f exactly once
    // close(f)      // error: use of moved value 'f'
}                    // dropping f without consuming would also be an error
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
    mut total : i64 = 0
    mut index : i64 = 0
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

```frost
match number(text) {
    case .Ok { value }: { print value }
    case .Err { error }: { print error.at }
}
```

A match on an enum covers every variant, so there is no way to read the value
without saying what happens when there is not one. The two failure types at a
`?` have to be the same one, because there is no conversion and no `From` to
write. And a result carrying a `linear` value is itself linear, so a fallible
function that answers with a resource cannot be called and ignored.

That is the whole error story. No exceptions, no panics to catch, no error codes
checked by convention. A failure that is a bug rather than a condition in the
world is an assertion, and an assertion aborts. Chapter 5.2b of
[declarations.md](reference/declarations.md) is the full account.

## Generational handles and slabs

Long-lived data lives in a slab and is referenced by a `Handle<T>`, a small
copyable value, not a pointer. A freed-and-reused slot bumps its generation, so
an old handle can never read the new occupant.

A slab is not a language type. `std/slab.frost` is an ordinary Frost library:
storage, a free list, generation counters, and the packing of a slot index and a
generation into one handle, all written out. What the compiler supplies is the
one piece that cannot be written where borrows are second-class, the place
`world[handle]`, which it offers because the struct is slab-shaped (a `storage`
array beside a parallel `generations` array).

```frost
import "slab.frost"

Entity :: struct { hp: i64, mana: i64 }

main :: fn() -> i64 {
    mut world : Slab<Entity, 16> = Slab {
        storage = [Entity { hp = 0, mana = 0 }; 16],
        generations = [0; 16],
        free_list = [0; 16],
        free_count = 0,
    }
    slab_reset($Entity, $16, world)

    hero := slab_insert($Entity, $16, world, Entity { hp = 100, mana = 30 })

    print world[hero].hp                  // 100
    world[hero].hp = world[hero].hp - 25  // the subscript is a place to write
    print world[hero].hp                  // 75

    slab_release($Entity, $16, world, hero)
    print slab_alive($Entity, $16, world, hero)   // 0, the generation moved on
    0
}
```

The arrays are written out at construction because a struct literal cannot run
code, and `slab_reset` is what fills in the free list. Passing `world[hero]` to
a function borrows the element, and that borrow is a parameter mode like any
other. `examples/native/entity_system.frost` is this over an entity enum, and
chapter 10 of [handles-and-pools.md](reference/handles-and-pools.md) states the
generational rule.

## Structure-of-arrays with columns

`columns<T, N>` stores each field of `T` in its own array rather than storing
whole elements back to back, so a pass reading one field across many elements
walks a tight column. It keeps the slab's generational handle unchanged, so
`c[h]` is still a checked place, and `c.field` is the whole column, a slice for
a hot loop:

```frost
import "columns.frost"

Particle :: struct { x: i64, y: i64 }

main :: fn() -> i64 {
    mut world : columns<Particle, 8> = columns_new()
    columns_reset($Particle, $8, world)
    h := columns_insert($Particle, $8, world, Particle { x = 10, y = 1 })

    print world[h].x               // 10, checked at the handle's slot
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
Pair :: struct($T: Type) { first: T, second: T }

make_pair :: fn(a: $T, b: $T) -> Pair<T> { Pair { first = a, second = b } }
swap      :: fn(mut a: $T, mut b: $T) { t := a  a = b  b = t }

main :: fn() -> i64 {
    p := make_pair(3, 4)               // Pair<i64> inferred
    print p.first + p.second           // 7

    mut x : i64 = 1
    mut y : i64 = 2
    swap(x, y)
    print x                            // 2
    0
}
```

`sizeof(T)` is a compile-time constant, so a generic function can size its own
type parameter. When a type parameter can't be inferred from a value argument
(for example a function that only uses `sizeof(T)`), declare it as `$T: Type`
and pass the type explicitly with a leading `$`:

```frost
bytes_for :: fn($T: Type, count: i64) -> i64 { count * sizeof(T) }

main :: fn() -> i64 {
    print bytes_for($Entity, 16)       // pass the type with $
    0
}
```

A parameter may also be an integer, written `$N: usize`, which is what sizes the
`[N]T` field of a `Slab<T, N>`. Type parameters are erased, drive
monomorphization, and carry no runtime cost.

A generic may also say what it needs of a type, over a fixed vocabulary of
questions the compiler already answers about one:

```frost
twice :: fn($T: Type, v: $T) -> T where is_numeric(T) { v + v }
```

Nothing registers into that vocabulary and nothing implements it, so the bound
is a precondition checked against the line the caller wrote. The vocabulary is
11.4a of [generics.md](reference/generics.md).

## Higher-order code without traits or closures

A generic algorithm takes the operation it needs as a compile-time function
parameter, which is Frost's answer to what a trait bound expresses. The
parameter can state the signature it requires, and the call inside the
specialization is direct rather than through a pointer:

```frost
ascending :: fn(a: i64, b: i64) -> bool { a < b }

best :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T, move y: $T) -> $T {
    mut result := x
    if (before(y, result)) { result = y }
    result
}

main :: fn() -> i64 {
    print best($i64, $ascending, 7, 3)   // 3
    0
}
```

When several operations travel together, they go in a struct whose fields are
functions. That is a capability bundle, and it is what stands in for a trait:

```frost
Ordering :: struct($T: Type) {
    less: fn(T, T) -> bool,
    equal: fn(T, T) -> bool,
}

i64_ascending :: Ordering<i64> { less = i64_less, equal = i64_equal }

sort :: fn($T: Type, $ops: Ordering<T>, mut items: []T) {
    ...
    if (ops.less(items[j], items[j - 1])) { ... }
}

sort($i64, $i64_ascending, view)
```

The bundle is a type, an implementation is a constant of it, and the call names
which one it means. Nothing registers, nothing is searched for, and there is no
coherence rule to learn, because the answer is written at the call. Since `$ops`
is a compile-time argument, `ops.less(a, b)` folds to a direct call to
`i64_less`, and the specialization holds no function pointer at all.
`std/ordering.frost` and `std/sort.frost` are this written out, and 11.4b of
[generics.md](reference/generics.md) has the rest, including what dropping the
`$` gives you.

When the operation genuinely varies at runtime, drop the `$` and the same
declaration gives an ordinary value. A `fn(...) -> T` parameter holds a pointer,
and a bundle without the `$` is a struct holding several. There are no capturing
closures.

```frost
apply :: fn(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }
double :: fn(x: i64) -> i64 { x * 2 }

main :: fn() -> i64 {
    print apply(double, 21)   // 42
    0
}
```

## A compile-time list of arguments

A parameter written `args: $...` takes as many arguments as the call gives it,
of whatever types they are:

```frost
printall :: fn(args: $...) {
    for value in args {
        print value
    }
}

main :: fn() -> i64 {
    printall(1, 2.5, 9)
    0
}
```

The `for` is not a loop. The body is compiled once per element with `value`
standing for that element, so what runs is three `print`s of three different
types. An `if` over what a type is gets its answer at expansion time too, and
the branch that cannot run is dropped before anything checks it, which is what
lets one body serve elements of different types.

There is no compile-time string parsing, no recursion and no unbounded loop:
everything here walks a list whose length the call fixed, so expansion costs
what the program's own text costs and no more. 11.1c of
[generics.md](reference/generics.md) has `args[0]`, handing a list on, and
`g(T) for T in list` in an argument position.

## A table over a type's fields

A vertex format, a uniform layout and a descriptor table are all the same
thing: offsets and sizes over a struct you already declared. The compiler worked
those numbers out to lay the struct out, so write the table over them rather
than beside them:

```frost
Vertex :: struct { position: Vec3, normal: Vec3, uv: Vec2, id: i64 }

main :: fn() -> i64 {
    print field_count(Vertex)
    for field in fields(Vertex) {
        print offset_of(field)
        print sizeof(field)
        if (is_float(field)) { print 1 } else { print 0 }
    }
    0
}
```

The `for` is not a loop here either: the body is compiled once per field. `T`
may be a type parameter, so one description serves every type a call names, and
the table cannot drift from the struct because it is not written twice.

A field is not a value. `offset_of`, `sizeof` and the type predicates are what
may be asked of one, and naming it anywhere else is an error. There is no
reading a field's name, which is what keeps this a layout question rather than a
second language for asking about types (11.1d of
[generics.md](reference/generics.md)).

## Arrays are bounds-checked

A fixed-size array `[N]T` knows its length, and every index is checked:

```frost
main :: fn() -> i64 {
    arr := [10, 20, 30]
    print arr[2]     // 30
    // arr[5]         // aborts: index 5 out of bounds for length 3
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
parameter as a plain C string at no cost.

```frost
printf :: extern fn(fmt: ^i8, value: i64) -> i32

main :: fn() -> i64 {
    unsafe { printf("%lld\n", 42) }
    0
}
```

Calling C is unchecked, so the call sits in an `unsafe` block. That is the point
of the block: it is the complete list of places to look when memory has been
corrupted. A foreign function that cannot go wrong, one taking and returning
numbers with no pointer anywhere, is declared `safe extern fn` and needs no
block, which is how `std/math.frost` reaches `sqrtf`. See
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
(`vec3_add` rather than a qualified `math.add`), which is what keeps every name
a single token to search for. [modules.md](impl/modules.md) covers imports,
exports, and the rename escape hatch for a collision.

## A complete program

Entities in a slab, behavior as a free function over an enum, and identity as a
handle that stays checkable after the slot it names is reused.

```frost
import "slab.frost"

Kind   :: enum { Player, Enemy { damage: i64 } }
Entity :: struct { hp: i64, kind: Kind }

delta :: fn(e: Entity) -> i64 {
    match e.kind {
        case .Player: 0
        case .Enemy { damage }: 0 - damage
    }
}

main :: fn() -> i64 {
    mut world : Slab<Entity, 16> = Slab {
        storage = [Entity { hp = 0, kind = Kind::Player }; 16],
        generations = [0; 16],
        free_list = [0; 16],
        free_count = 0,
    }
    slab_reset($Entity, $16, world)

    player := slab_insert($Entity, $16, world,
        Entity { hp = 100, kind = .Player })
    goblin := slab_insert($Entity, $16, world,
        Entity { hp = 30, kind = .Enemy { damage = 15 } })

    world[player].hp = world[player].hp + delta(world[goblin])
    print "hp {}", world[player].hp                  // 85

    slab_release($Entity, $16, world, goblin)
    print slab_alive($Entity, $16, world, goblin)    // 0
    0
}
```

Handles are what get passed around and stored, the borrow of `world[goblin]`
lasts only for the call to `delta`, and releasing a slot invalidates the handles
to it by generation rather than by a lifetime the compiler had to track.

## Where to next

- Runnable programs live in `examples/native/`. Start with `game_world.frost`
  (the entity-component system) and `generic_slab.frost` (the slab written out
  in full).
- [coming-from-rust.md](coming-from-rust.md) if you already think in Rust.
- [patterns.md](patterns.md) for what the language rewards and what it merely
  permits.
- [architecture.md](impl/architecture.md) explains the compiler pipeline and
  exactly what the native path supports today.
