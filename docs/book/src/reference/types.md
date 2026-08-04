# 3. Types

## 3.1 Scalar types

| Type | Meaning | Size (bytes) |
| --- | --- | --- |
| `i8` `i16` `i32` `i64` `isize` | signed integers | 1, 2, 4, 8, 8 |
| `u8` `u16` `u32` `u64` `usize` | unsigned integers | 1, 2, 4, 8, 8 |
| `f32` `f64` | IEEE floats | 4, 8 |
| `bool` | boolean | 1 |

A function that declares no return has the empty type inside the compilers,
where it goes by `void`. It has no surface spelling: leaving the `->` off is
how the empty type is written.

All scalar types are copy types (chapter 8). Integer arithmetic whose result
does not fit the type it is computed at aborts rather than wrapping: add,
subtract, multiply, divide or take the remainder by zero, divide the lowest
signed value by minus one, negate that value, or shift by more than the width.
`wrap_add(a, b)`, `wrap_sub(a, b)` and `wrap_mul(a, b)` keep the low bits and
drop the rest, for the cases where leaving the range is what was wanted.
Mixed-width integer arithmetic is permitted. The narrower operand widens to the
wider type.

## 3.1a Conversions

A value converts to another scalar type on its own only when the destination
can hold every value the source can. `i32` to `i64`, `u16` to `u32`, `f32` to
`f64`, and any integer to a float wide enough for it all happen silently at an
assignment, an argument, a return, or a field. Nothing is lost, so nothing has
to be said.

Every other conversion is refused, and the diagnostic names both types:

```
count : i32 = total    // total is an i64
                ^ this is a i64 and a i32 is wanted, which cannot hold all of
                  one; write cast($i32, ...) to say the loss is meant
```

`cast($T, value)` is the conversion written out loud. It converts between any
two scalars: integer to narrower integer (truncating the high bits), signed to
unsigned and back (reinterpreting the bit pattern), float to integer
(truncating toward zero), integer to float, and `f64` to `f32`. The result is
`T`. It never checks and never traps, which is why it has to be written: the
loss is the caller's to intend.

A literal is not converted, it is typed. `-1` written where an `i32` is wanted
is an `i32`, and `1.5` where an `f32` is wanted is an `f32`, so neither is a
narrowing and neither needs a cast. A literal too large for the type it is
written at is a compile error rather than a silent wrap.

`cast` is for scalars. Reinterpreting a pointer is `ptr_cast` and lives inside
`unsafe` (6.8); the two are not spellings of each other.

## 3.2 Aggregate types

- Structs `Name`, declared `Name :: struct { field: T, ... }`, are exactly
  their fields in declaration order, with natural alignment.
- Enums `Name`, declared `Name :: enum { Variant, Variant { f: T }, ... }`,
  are a discriminant plus the active variant's payload. Variants may be unit or
  carry named fields, and one enum may mix both. An enum takes type parameters
  exactly as a struct does, `Option :: enum($T: Type) { None, Some { value: T } }`,
  and instantiates the same way (chapter 11).
- Fixed arrays `[N]T` are `N` contiguous `T`. The length is part of the type
  and every index is bounds-checked (10.4).
- Slices `[]T` are a pointer/length view of a run of `T`, sixteen bytes and a
  copy value, the same fat-pointer shape as `str` (which is `[]u8`). An array
  coerces to a slice of the whole array, `s[i]` is bounds-checked against the
  runtime length (10.4), and `slice_len(s)` reads the length in constant time.

Aggregates are move types (chapter 8), copied by value at call and return
boundaries unless passed by borrow, with no `Copy` derive.

Frost has no visibility modifiers. There is no `pub` and no private. Every struct
field is public and reachable, and there is nothing to specify.

## 3.3 Borrows and pointer types

A borrow is mostly not a type a program writes. It is what a parameter mode
means: an unmarked parameter of a non-copy type is read-borrowed, `mut` is
write-borrowed, and `move` takes the value (chapter 8). There is no `&` or `&mut`
in the surface, so a borrow has nowhere to be stored in a struct and nowhere to
be written in a field, which is what makes it second-class by construction
rather than by rule.

- `ref T` a borrow of a place, the one borrow a program writes down. It appears
  as a return type, and `ref name := place` binds one (5.1). It is a checked
  address: reading and writing through it needs no `unsafe`, unlike `^T`, and
  the region and frame checks of chapter 8 govern where it may go. What it may
  not do is be stored: no struct field, no array element, no container.
- `^T` raw pointer, unchecked, for FFI and low-level libraries.

`ref T` is what lets an accessor hand back a place rather than a copy:

```frost
arena_at :: fn($T: Type, a: Arena<T>, index: i64) -> ref T { ... }

ref entry := arena_at(p.tokens, index)
entry.kind = TokenKind::Ident
```

Without it a container's element could only be read out and written back, and
every accessor over one would be a pair of functions or an `unsafe` block over a
raw pointer. The borrow it answers with is the caller's to read and write and
nobody's to keep.

`ptr_to(place)` yields a `^T` to a place. `ptr_cast($T, p)` reinterprets a
pointer as `^T` at no runtime cost. These are the low-level tools an allocator
uses to hand back typed memory from a byte buffer. Ordinary code does not need
them, and a pointer carries no safety guarantee once it is formed.

A pointer or a slice that names storage in the current frame may not be
returned, and neither may one into an arena outlive its region (chapter 8).

## 3.4 Handle and pool types

`Handle<T>` names an element of a pool of `T` (chapter 10), a small copy value
(index plus generation), not a pointer, that unlike a reference may be stored in
fields and returned.

A pool is not a built-in type. It is an ordinary struct a program writes for
itself, an array of storage indexed by `Handle<T>` (chapter 10.1). The compiler
provides the pieces to build one, not the pool itself, and `std/slab.frost` is
one written out in the language.

`columns<T, N>` is a compiler-synthesized structure-of-arrays container for `N`
elements of struct `T`, one array per field of `T` plus a generational free list,
addressed by `Handle<T>` the same way a pool is (chapter 10.1a). Unlike a pool it
cannot be written as a library, since "one array per field of `T`" is not
expressible over an arbitrary `T`.

## 3.5 Function types

`fn(T1, ...) -> R` is a function pointer. There are no closure types. A
function-typed value is always a plain pointer to a function.

A parameter of a function type may be written `mut T`, which means the same
reference the `mut` mode means on a declared parameter (chapter 8). It has to be
sayable here because the surface has no reference type to write instead. An
unmarked parameter is the type as written, and `move T` is the type as written
too, allowed so a function type can be read beside the declaration it describes.

## 3.6 Other type forms

There is no built-in optional, and none is needed: a generic enum (3.2) writes
one in the language. `Option :: enum($T: Type) { None, Some { value: T } }` is
that type, with nothing special about it.

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

Reading one as its representation is allowed, and is what a call into C is: a
`Meters` is an `i64` in memory and nothing is at stake going that way. So
`printf("%lld\n", m)` works, and `n : i64 = m` works. What the name protects is
what goes *in*: a bare number, or a value that means something else, cannot
become a `Meters` by accident. There is no cast, because there is nothing a cast
would be needed for in the direction that is checked.

A literal is exempt because it has no type of its own until the context gives it
one, which is what makes `m : Meters = 3` read the way it should.

Arithmetic answers with the distinct type, so adding two lengths gives a length
and the sum goes back where either operand came from. Everything else about the
value follows the representation: its size, its alignment, which register it
travels in, and how it crosses to C.

## 3.6b Flags

`InitFlags :: flags u32 { Video = 32, Audio = 16 }` declares a named set of
bits. It sits between an enum, which is a closed set of alternatives with
exactly one of them held, and a `distinct` integer, which combines with `|` but
has nothing tying loose constants to the type.

```
InitFlags :: flags u32 {
    Audio   = 16,
    Video   = 32,
    Events  = 16384,
    Gamepad = 8192,
}

WindowFlags :: flags u64 {
    Resizable        = 32,
    HighPixelDensity = 8192,
}

chosen := InitFlags::Video | InitFlags::Events
sdl_init(chosen)
window_create("Frost", 960, 540,
    WindowFlags::Resizable | WindowFlags::HighPixelDensity)

if (flags_has(chosen, InitFlags::Video)) { ... }
```

The representation is written, and is an integer type. These numbers mirror a C
header's, so a compiler that chose them would be choosing different ones.

Each bit is named under the type: `InitFlags::Video`. There is no prefix
convention and no loose constants beside the declaration.

A flags type is nominal exactly as a `distinct` type is. An `InitFlags` where a
`WindowFlags` is wanted is a compile error, and so is a bare `u32`. It is
*stricter* than a distinct type in one way: a number written where a flags value
belongs is refused rather than taking the type from the context, because the
names are the whole content of the declaration.

```
f : InitFlags = 48        // error, the names are what one is built from
sdl_init(48)              // error, the same
```

Two values of the type combine with `|` and narrow with `&`, and both answer
with the type, which is what lets a combination be passed with nothing written
down to say what it is. They compare with `==` and `!=`. Everything else is
refused: adding two sets, ordering them, or shifting one along is a question
about the number underneath, and the declaration exists to say that the number
is not what this is. Both operands have to be the same set, so
`InitFlags::Video | WindowFlags::Resizable` is an error rather than a number
wearing one of the two names.

`flags_has(set, wanted)` answers whether every bit of `wanted` is on in `set`.
Both are values of the same flags type.

Reading one as its representation is allowed, and is what a call into C is, the
same way it is for a distinct type.

Printing a flags value writes the number: going out to the representation is
free, so `print_int_line(set)` shows the bits as one integer. The names are not
available at run time, and a program that wants to show which bits are set
writes that loop itself. `flags_has` makes each test one call.

`flags` is not a keyword. It is recognized in a declaration by the shape that
follows it (a scalar type and then a brace), so a parameter, a local or a field
may still be called `flags`.

## 3.7 Strings

`str` is an immutable, non-owning view of a run of bytes. It is a pointer and a
length, sixteen bytes, laid out as the byte pointer at offset 0 and the length
(a `usize`) at offset 8. It owns nothing, so it is a copy type (chapter 8),
freely duplicated with no move and nothing to release. In this it is the byte
form of a slice (`[]u8`).

- A string literal is a `str` pointing into read-only data, with the length
  fixed at compile time.
- `str_len(s)` returns the byte length in constant time, reading the length
  field rather than scanning.
- `s[i]` reads the byte at index `i` as a `u8` and is bounds-checked against the
  length (10.4), the same rule as array indexing.
- Passing a `str` to a function copies the pointer and length by value.

`str` carries no NUL terminator and may contain a NUL byte. Crossing to C is
therefore an explicit conversion, not an automatic one. The single affordance is
the string literal, which the compiler also emits NUL-terminated so that a
literal used where `^i8` is expected passes as a C string at no cost (2.5).

Owned or growable text is not a language type. It is an ordinary struct over an
array or a pool that a program borrows as a `str`.
