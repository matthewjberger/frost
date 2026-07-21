# Writing Frost

This is the practical guide to writing correct Frost quickly. It is dense on
purpose, so it is a good single document to hand a new author, human or model,
before they write their first program. For the reasoning behind the choices see
[syntax-design.md](syntax-design.md), for the full rules see [spec.md](spec.md),
and for a Rust-to-Frost mapping see [coming-from-rust.md](coming-from-rust.md).

## What Frost is

Frost is a small, statically typed, data-oriented systems language. A program is
plain data (structs and enums) plus free functions that transform it. There are
no methods, no inheritance, no classes, and no `self`. It is memory-safe without
a garbage collector and without lifetime annotations. It compiles to native code
through Cranelift or through portable C.

## Navigating a Frost codebase

Every top-level thing is declared with the same shape, `name :: value`, so the
name is always first and left-aligned. This makes the codebase trivially
greppable.

- Find any definition, of any kind, by grepping for the name followed by `::`.
  `rg "Entity ::"` finds the struct, function, enum, or constant named `Entity`,
  and you do not need to know which kind it is first.
- A file's public surface is its `export` line at the top. Read that to see
  what a module offers without scanning every declaration.
- A file's dependencies are its `import` lines at the top.
- `::` means "declare a constant or item", `:=` means "introduce a local", `:`
  means "type annotation", and `=` means "assign". Each symbol means exactly one
  thing, so reading is unambiguous.

## Declarations and bindings

```
MAX   :: 100                          // a constant
add   :: fn(a: i64, b: i64) -> i64 { a + b }   // a function
Point :: struct { x: i64, y: i64 }    // a struct
Color :: enum { Red, Green, Blue }    // an enum

x := 5                                // local, type inferred
y : i64 = 5                           // local, type explicit
mut total : i64 = 0                   // mutable local
total = total + 1                     // assignment
```

There is no `let`. Bindings are immutable unless marked `mut`. A block is an
expression whose value is its trailing expression, and `return` exists for early
exit. Line comments start with `//`. Program entry is `main :: fn() -> i64`, and
its return value is the process exit code.

## Types

Scalars are `i8 i16 i32 i64 isize`, `u8 u16 u32 u64 usize`, `f32 f64`, `bool`,
and `void`. Integer arithmetic wraps at the type width and is never overflow
checked. Aggregates are structs, enums, fixed arrays `[N]T`, and slices `[]T`.
References are `&T` and `&mut T`, a raw pointer is `^T`, and `Handle<T>` names an
element of a pool.

## Structs, enums, and match

Fields are set with `=`, not `:`. Struct fields are always public, there is no
visibility on fields.

```
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
    c := Shape::Circle { radius = 5 }
    area(&c)
}
```

Match arms are `case pattern: body`. Variant patterns lead with a dot and bind
fields by name. Value patterns (`case 90:`), tuple patterns (`case (0, _):`),
and the wildcard `_` all work.

## Ownership in one page

- **References are second-class.** A `&T` or `&mut T` may be a parameter or a
  short-lived local, but it cannot be stored in a field, put in an array, or
  returned. This is why Frost needs no lifetimes. When you want something that
  outlives a call, use a `Handle<T>`, not a reference.
- **Move checking.** Structs, enums, and slices move when passed by value,
  assigned, or returned. Using one after it moves is an error. Scalars,
  pointers, handles, and `str` are copy.
- **`str` is a byte view.** A string literal is a `str`, a pointer plus length
  that owns nothing. `str_len(s)` is its length and `s[i]` is a bounds-checked
  `u8`. Passing `"..."` where `^i8` is expected hands C a NUL-terminated pointer,
  which is how the literal reaches `printf` and friends.
- **Linear resources replace `Drop`.** A `linear struct` or `linear enum` must be
  consumed exactly once. Consume it by returning it, passing it by value, or
  matching it. Forgetting is a compile error.
- **Handles and pools** are how you model long-lived, shared, or linked data. A
  `Pool<T>` is a contiguous arena, made with `pool_new($T, capacity)`, and a
  `Handle<T>` (an index plus a generation) is a copy value you store and pass.
  `pool_alloc(pool, value)` returns a handle, `pool[handle]` is a place you read,
  write, or borrow, `pool_contains(pool, handle)` tests liveness, and
  `pool_free(pool, handle)` releases a slot. No `extern` declarations are needed.
  A freed slot bumps its generation, so a stale handle can never read the new
  occupant. A `Pool<T>` is a linear resource, so you must `pool_destroy(pool)` it
  exactly once; forgetting is a compile error.

## Generics

A type parameter is written `$T`, with no bounds and no traits. It usually
infers from a value argument. When it cannot, declare it `$T: Type` and pass the
type at the call site with a leading `$`.

```
Pair :: struct($T: Type) { first: T, second: T }
make_pair :: fn(a: $T, b: $T) -> Pair<T> { Pair { first = a, second = b } }

world : Pool<Entity> = pool_new($Entity, 16)
```

Generics monomorphize, so there is no runtime dispatch.

## Functions as values

Higher-order code uses function pointers, not closures. A `fn(...) -> T`
parameter holds one. A function literal is an expression, so an anonymous
function is just the declaration form without the name.

```
apply :: fn(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }

apply(fn(a: i64) -> i64 { a + 1 }, 41)          // inline anonymous function
ops := [fn(a: i64) -> i64 { a + 1 }, fn(a: i64) -> i64 { a * 2 }]
```

There is no capture. Pass any state a function needs as another argument.

## Modules

A file is a module. `import "path"` brings another file in. Top-level items are
private by default, and a file lists what it offers with an `export` line at the
top. There is no `pub` anywhere.

```
// geometry.frost
export area, Shape
Shape :: enum { Circle { r: i64 }, Rect { w: i64, h: i64 } }
area :: fn(s: &Shape) -> i64 { ... }
scale :: fn(x: i64) -> i64 { ... }   // private, not exported

// main.frost
import "geometry.frost"
main :: fn() -> i64 { area(&Shape::Circle { r = 5 }) }
```

A private item is usable inside its own file, so `area` may call `scale`, but an
importer cannot name `scale`.

## Tests

A `test` block is a named unit test. `assert(cond)` fails it when `cond` is
false. Run every test in a file with `frost --test file.frost`.

```
add :: fn(a: i64, b: i64) -> i64 { a + b }
test "addition" { assert(add(2, 3) == 5) }
```

## Calling C

`extern fn` links against any C library with the natural ABI, with no glue.
String literals are C-compatible for `^i8` parameters. Frost calls C, but C does
not call Frost.

```
printf :: extern fn(fmt: ^i8, value: i64) -> i32
```

## First-hour gotchas

- `if` and `while` conditions need parentheses. `if (x > 5) { }`.
- Struct fields are set with `=`, not `:`. `Point { x = 1, y = 2 }`.
- Match arms are `case pattern: body`, and variant patterns lead with a dot.
- There is no `let`. Use `:=`, `:`, or `::`.
- Deref a reference-to-scalar or a raw pointer with postfix `^`. `p^`, `p^.field`.
  Field access through a reference to a struct is direct, `r.field`.
- You cannot return or store a `&T`. Use a `Handle<T>`.
- A `linear` value must be consumed on every path.
- Integer arithmetic wraps. Do not rely on overflow being caught.
- There is no `pub`. Control visibility with the `export` line, not per item.

## Why this is easy to write correctly

Every property above serves one goal, which is code that is cheap to parse,
cheap to grep, and hard to get subtly wrong.

- One declaration form means one pattern to emit and one way to search.
- One meaning per symbol (`::`, `:=`, `:`, `=`, `^`) removes the guesswork about
  what a punctuation mark does in context.
- Nothing runs invisibly. There is no `Drop` at scope exit, no auto-deref, no
  hidden allocation, and no implicit copy of a large value. What happens is
  written down.
- Second-class references remove lifetimes, so there is no annotation to get
  wrong. When a reference will not do, the answer is always a handle.
- Private-by-default with an explicit export line means a module's surface is one
  line you can read, and nothing leaks by accident.

## A complete program

```
printf :: extern fn(fmt: ^i8, value: i64) -> i32

Kind   :: enum { Player, Enemy { damage: i64 } }
Entity :: struct { hp: i64, kind: Kind }

delta :: fn(k: &Kind) -> i64 {
    match k {
        case .Player: 0
        case .Enemy { damage }: 0 - damage
    }
}

main :: fn() -> i64 {
    world : Pool<Entity> = pool_new($Entity, 16)
    player := pool_alloc(world, Entity { hp = 100, kind = Kind::Player })
    goblin := pool_alloc(
        world,
        Entity { hp = 30, kind = Kind::Enemy { damage = 15 } },
    )

    world[player].hp = world[player].hp + delta(&world[goblin].kind)
    printf("%lld\n", world[player].hp)          // 85
    pool_destroy(world)
    0
}
```
