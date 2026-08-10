# 3. Types

## 3.1 Scalar types

| Type | Meaning | Size (bytes) |
| --- | --- | --- |
| `i8` `i16` `i32` `i64` `isize` | signed integers | 1, 2, 4, 8, 8 |
| `u8` `u16` `u32` `u64` `usize` | unsigned integers | 1, 2, 4, 8, 8 |
| `f32` `f64` | IEEE floats | 4, 8 |
| `bool` | boolean | 1 |

A function that declares no return has the empty type, which the compilers call
`void`. It has no surface spelling: leaving the `->` off is how the empty type
is written.

All scalar types are copy types (chapter 8). Integer arithmetic whose result
does not fit the type it is computed at aborts rather than wrapping: add,
subtract, multiply, divide or take the remainder by zero, divide the lowest
signed value by minus one, negate that value, or shift by more than the width.
`wrap_add(a, b)`, `wrap_sub(a, b)` and `wrap_mul(a, b)` keep the low bits and
drop the rest. Mixed-width integer arithmetic is permitted. The narrower operand
widens to the wider type.

## 3.1a Conversions

A value converts to another scalar type on its own only where the destination
is at least as wide. `i32` to `i64`, `u16` to `u32`, `f32` to `f64`, any
integer to a float, and a change of sign at one width all happen silently at an
assignment, an argument, a return, or a field.

A conversion that narrows is refused, and the diagnostic names both types:

```
count : i32 = total    // total is an i64
                ^ this is a i64 and a i32 is wanted, which cannot hold all of
                  one; write cast($i32, ...) to say the loss is meant
```

`cast($T, value)` is the conversion written out loud. It converts between any
two scalars: integer to narrower integer (truncating the high bits), signed to
unsigned and back (reinterpreting the bit pattern), float to integer
(truncating toward zero), integer to float, and `f64` to `f32`. The result is
`T`. It never checks and never traps.

A literal takes its type from the context. `-1` written where an `i32` is wanted
is an `i32`, and `1.5` where an `f32` is wanted is an `f32`, so neither is a
narrowing and neither needs a cast. A literal too large for the type it is
written at is a compile error.

`cast($T, value)` also names a distinct type for a value that already has its
representation, which is the one conversion it makes that is not between
numbers: `cast($Key, sdl_scancode())` where `Key` is `distinct i64`, and
`cast($Adapter, p)` where `Adapter` is `distinct ^u8` and `p` is one. No bits
move; the value takes the name its declaration gives that representation. This
is the boundary a foreign number or handle crosses at, and it is written once
where the value arrives (3.6a).

`cast` is otherwise for scalars. Reinterpreting a pointer as a pointer to
something else is `ptr_cast`, which lives inside `unsafe` (6.8).

A `^T` and an integer convert to each other in either direction, since a pointer
is an address and an address is a whole number. A call into C hands the address
over, and address arithmetic reads it back. A pointer reaches no other scalar:
`f : f64 = p` and `b : bool = p` are refused and the diagnostic names the
pointer.

## 3.2 Aggregate types

- Structs `Name`, declared `Name :: struct { field: T, ... }`, are exactly
  their fields in declaration order, with natural alignment. A declaration may
  state the layout instead of taking it (3.2a).
- Enums `Name`, declared `Name :: enum { Variant, Variant { f: T }, ... }`,
  are a discriminant plus the active variant's payload. Variants may be unit or
  carry named fields, and one enum may mix both. An enum takes type parameters
  exactly as a struct does, `Option :: enum($T: Type) { None, Some { value: T } }`,
  and instantiates the same way (chapter 11).
- Fixed arrays `[N]T` are `N` contiguous `T`. The length is part of the type
  and every index is bounds-checked (10.4). It is a number, a name standing for
  one (a module constant, or a size parameter a generic binds), a call the
  compiler works out (5.2c), or `+ - * / %` and brackets over those.
  `[(N + 63) / 64]i64` is one word of bits per sixty-four slots, the form
  `Slab<T, N>` carries its liveness in (10.1b), and
  `[next_power_of_two(300)]u8` is a length a function decided. A call is run
  where it is written, so every argument has to be known there: a size parameter
  a generic has not bound yet is refused and named.
- Slices `[]T` are a pointer/length view of a run of `T`, sixteen bytes and a
  copy value, the same fat-pointer shape as `str` (which is `[]u8`). An array
  coerces to a slice of the whole array, `s[i]` is bounds-checked against the
  runtime length (10.4), and `slice_len(s)` reads the length in constant time.
  `slice_len` of a fixed array is its length too, which its type already says.
  A parameter of array type is a borrow of the caller's array (chapter 8), so
  the slice made from one views the caller's storage: a write through it lands
  in the argument, and handing the slice back out of the call is a view of
  storage that outlives it. The coercion holds in every position that takes a
  slice: a binding with an annotation, an assignment, a field of a literal, an
  argument, and a `return`. The array's type carries the length, and the borrow
  still names that type.

Aggregates are move types (chapter 8), copied by value at call and return
boundaries unless passed by borrow, with no `Copy` derive.

Frost has no visibility modifiers. There is no `pub` and no private. Every
struct field is reachable.

## 3.2a Stated layout

A struct takes its layout from its fields. Two forms let a declaration state a
layout instead.

```frost,sketch
Header :: packed struct { magic: u32, kind: u8, length: u32 }
Uniform :: struct { time: f32, view: Matrix4 align(16) }
```

`packed struct` puts every field at the next byte and gives the type an
alignment of one. `Header` above is nine bytes, at offsets 0, 4 and 5.

`align(N)` after a field's type is the multiple that field starts at, in place
of what its type would ask for. A struct's own alignment is the widest any of
its fields asks for, so `align(16)` on a field also gives the struct that
alignment, and there is no form that states an alignment for a whole
declaration.

`align(N)` takes a power of two, since an address is a multiple of one or of
nothing. `align` inside a `packed struct` is refused: packed pads no field and
`align` asks for this one to be padded.

`packed` and `align` carry meaning without being reserved (2.4). A local, a
field and a parameter may still be called either. The `struct` after `packed`
marks the declaration, and the `(` after `align` marks the field form.

A stated layout reaches every backend, so `sizeof`, `offset_of` and a read
through a field answer the same on all of them.

## 3.2b Vectors

The arithmetic operators are defined over a fixed array of numbers, once per
lane:

```frost,inside
a : [4]f32 = [1.0, 2.0, 3.0, 4.0]
b : [4]f32 = [5.0, 6.0, 7.0, 8.0]
sum := a + b            // [6.0, 8.0, 10.0, 12.0]
scaled := a * 2.0       // [2.0, 4.0, 6.0, 8.0]
flipped := -a           // [-1.0, -2.0, -3.0, -4.0]
```

There is no separate SIMD type and no marker. `[4]f32` is the fixed array of
3.2, and `a + b` is elementwise because both sides are arrays of numbers. The
growable `Vec<T>` of the standard library is a different thing entirely, a heap
container.

`+`, `-`, `*` and `/` are defined for any numeric element. `%`, `&`, `|`, `<<`
and `>>` are defined where the element is a whole number. Unary `-` negates
every lane. A number written on either side is that number in every lane. Two
vectors are compared nowhere: a comparison answers one yes or no, and a vector
of them is a mask, which is a type this language does not have.

A lane does what the number would do. A float lane rounds as one `f32` does, and
a whole-number lane aborts on overflow and says where (10.4).

The shape is held to a register's worth. The length is a power of two, and the
whole vector is at most 64 bytes. Both are refused where they do not hold,
naming the length or the width.

The backend decides what a lane becomes. The self-hosted compiler's assembly
backend emits `addps`, `subps`, `mulps` and `divps` for a vector of `f32`, and
the `pd` forms for one of `f64`, sixteen bytes at a time. Both C backends write
the lanes out and the C compiler folds them back: `a * b + a` over `[4]f32`
comes out of `gcc -O2` as `mulps` and `addps`.

## 3.3 Borrows and pointer types

A parameter mode means a borrow: an unmarked parameter of a non-copy type is
read-borrowed, `mut` is write-borrowed, and `move` takes the value
(chapter 8). There is no `&` or `&mut` in the surface, so a borrow has nowhere
to be stored in a struct and nowhere to be written in a field.

- `ref T` a borrow of a place, the one borrow a program writes down. It appears
  as a return type, and `ref name := place` binds one (5.1). It is a checked
  address: reading and writing through it needs no `unsafe`, and the region and
  frame checks of chapter 8 govern where it may go. It may not be stored: no
  struct field, no array element, no container.
- `^T` raw pointer, unchecked, for FFI and low-level libraries.

`ref T` lets an accessor hand back a place rather than a copy:

```frost,sketch
arena_at :: fn($T: Type, a: Arena<T>, index: i64) -> ref T { ... }

ref entry := arena_at(p.tokens, index)
entry.kind = TokenKind::Ident
```

The borrow it answers with is the caller's to read and write and nobody's to
keep.

`ptr_to(place)` yields a `^T` to a place. `ptr_cast($T, p)` reinterprets a
pointer as `^T` at no runtime cost. An allocator uses both to hand back typed
memory from a byte buffer. A pointer carries no safety guarantee once it is
formed.

A pointer or a slice that names storage in the current frame may not be
returned, and neither may one into an arena outlive its region (chapter 8).

## 3.4 Handle and pool types

`Handle<T>` names an element of a pool of `T` (chapter 10). It is a small copy
value, an index plus a generation, and it may be stored in fields and returned.

A pool is not a built-in type. It is an ordinary struct a program writes for
itself, an array of storage indexed by `Handle<T>` (chapter 10.1). The compiler
provides the pieces to build one, and `std/slab.frost` is one written out in the
language.

`columns<T, N>` is a compiler-synthesized structure-of-arrays container for `N`
elements of struct `T`, one array per field of `T` plus a generational free list
and a record of which slots hold an element, addressed by `Handle<T>` the same
way a pool is (chapter 10.1a) and walked by `for slot in live_slots(c)` (10.1b).
It cannot be written as a library, since "one array per field of `T`" is not
expressible over an arbitrary `T`.

## 3.5 Function types

`fn(T1, ...) -> R` is a function pointer. There are no closure types. A
function-typed value is always a plain pointer to a function.

A parameter of a function type may be written `mut T`, which means the same
reference the `mut` mode means on a declared parameter (chapter 8). An unmarked
parameter is the type as written, and `move T` is the type as written too, so a
function type can be read beside the declaration it describes.

## 3.6 Other type forms

There is no built-in optional. A generic enum (3.2) writes one in the language:
`Option :: enum($T: Type) { None, Some { value: T } }`.

- `distinct T` is a nominal type with `T`'s representation, built only from
  itself (3.6a).
- `flags T { ... }` is a nominal set of named bits over an integer (3.6b).
- `$T` is a type parameter (chapter 11).
- `Name<T, ...>` is a generic instantiation (chapter 11).

## 3.6a Distinct types

`Meters :: distinct i64` declares a type with `i64`'s representation and a name
of its own. Size, alignment, arithmetic and the C ABI all follow the inner type.
Identity does not: a `Meters` is not an `i64`, and it is not a `Feet` declared
the same way.

The rule is one-directional. A distinct type is built only from itself, so
neither the representation nor another distinct type over it will do:

```
Meters :: distinct i64
Feet   :: distinct i64

m : Meters = 3            // a literal takes the type the context wants
n : i64    = 3
m = n                     // error: a distinct type is not its representation
add_meters(m, f)          // error, where f is a Feet
```

Reading one as its representation is allowed, and a call into C reads it that
way: a `Meters` is an `i64` in memory. So `printf("%lld\n", m)` works, and
`n : i64 = m` works. The other direction is checked, wherever it is written: a
binding, a return, an argument, an assignment and a comparison all ask it, and a
bare number or a value that means something else cannot become a `Meters`. A
body's last expression is its answer and is asked the same question, so leaving
the word `return` off does not skip it.

`cast($Meters, n)` is how a number is meant to become one, and it is the only
way (3.1).

A literal is exempt: it has no type of its own until the context gives it one,
so `m : Meters = 3` holds.

Arithmetic answers with the distinct type, so adding two lengths gives a length
and the sum goes back where either operand came from. Everything else about the
value follows the representation: its size, its alignment, which register it
travels in, and how it crosses to C.

## 3.6b Flags

`InitFlags :: flags u32 { ... }` declares a named set of bits, each written
`Name :: number` on a line of its own.

```
InitFlags :: flags u32 {
    Audio :: 16
    Video :: 32
    Events :: 16384
    Gamepad :: 8192
}

WindowFlags :: flags u64 {
    Resizable :: 32
    HighPixelDensity :: 8192
}

chosen := InitFlags::Video | InitFlags::Events
sdl_init(chosen)
window_create("Frost", 960, 540,
    WindowFlags::Resizable | WindowFlags::HighPixelDensity)

if (flags_has(chosen, InitFlags::Video)) { ... }
```

The representation is written, and is an integer type. Each bit's number is
written out as well, so a set states the numbers a C header fixed.

A bit is declared with `::`, the way every value named under a type is (5.2d),
and one bit goes on a line. What a bit may hold is where the two blocks differ:
a number a C header fixed, where a value named under a type is an expression of
that type.

Each bit is named under the type: `InitFlags::Video`. There is no prefix
convention and no loose constants beside the declaration.

A flags type is nominal exactly as a `distinct` type is. An `InitFlags` where a
`WindowFlags` is wanted is a compile error, and so is a bare `u32`. A number
written where a flags value belongs is refused, where a distinct type would take
the type from the context.

```
f : InitFlags = 48        // error, the names are what one is built from
sdl_init(48)              // error, the same
```

Two values of the type combine with `|` and narrow with `&`, and both answer
with the type. They compare with `==` and `!=`. Everything else is refused:
adding two sets, ordering them, or shifting one along is a question about the
number underneath. Both operands have to be the same set, so
`InitFlags::Video | WindowFlags::Resizable` is an error.

`flags_has(set, wanted)` answers whether every bit of `wanted` is on in `set`.
Both are values of the same flags type.

Reading one as its representation is allowed, and a call into C reads it that
way, the same as for a distinct type.

Printing a flags value writes the number, so `print("{}\n", set)` shows the bits
as one integer. The names are not available at run time, and a program that
wants to show which bits are set writes that loop itself. `flags_has` makes each
test one call.

`flags` is not a keyword. It is recognized in a declaration by the shape that
follows it (a scalar type and then a brace), so a parameter, a local or a field
may still be called `flags`.

## 3.7 Strings

`str` is an immutable, non-owning view of a run of bytes. It is a pointer and a
length, sixteen bytes, laid out as the byte pointer at offset 0 and the length
(a `usize`) at offset 8. It owns nothing, so it is a copy type (chapter 8),
freely duplicated with no move and nothing to release. It is the byte form of a
slice (`[]u8`).

- A string literal is a `str` pointing into read-only data, with the length
  fixed at compile time.
- `str_len(s)` returns the byte length in constant time, reading the length
  field. Since `str` *is* `[]u8`, `str_len` reads either spelling, and
  `slice_len` reads a `str` the same way it reads any other run. Both refuse
  what carries no length: a struct, a scalar, or a raw pointer, which is a run's
  address with nothing beside it saying how long the run is. `str_len` of a run
  whose elements are not bytes is refused as well, and `slice_len` is the word
  for that length.
- `s[i]` reads the byte at index `i` as a `u8` and is bounds-checked against the
  length (10.4), the same rule as array indexing.
- Passing a `str` to a function copies the pointer and length by value.

`str` carries no NUL terminator and may contain a NUL byte. Crossing to C is
therefore an explicit conversion. The single affordance is the string literal,
which the compiler also emits NUL-terminated so that a literal used where `^i8`
is expected passes as a C string at no cost (2.5).

Owned or growable text is not a language type. It is an ordinary struct over an
array or a pool that a program borrows as a `str`.
