# Patterns, and what to write instead

What the language rewards and what it merely permits. Everything here is legal
Frost, and the antipatterns compile. What makes them antipatterns is that the
compiler stops helping you the moment you write them, and the help is the whole
reason the rules are there.

Each one is written from something that happened in this repository, and the
"instead" is what the code says now.

## A set of alternatives is an enum, not a run of constants

**Antipattern.**

```frost
PLAIN    :: 0
COMMENT  :: 1
KEYWORD  :: 2
TYPE     :: 3
FUNCTION :: 4

colour_of :: fn(class: i64) -> str { ... }
```

This is legal and it is what a C header looks like, which is why it keeps
appearing. What it costs is three things.

The parameter says nothing. `class: i64` accepts a row count, a file descriptor,
or `-1`. Nothing is checked, because there is nothing to check against.

It cannot be matched. `match` over an enum is exhaustive, so adding a variant
stops every match that does not handle it from compiling. A run of constants has
no such set, so the `if` chain that reads them silently keeps its old answer for
a value it has never seen.

The prefix is doing the type's job. `PLAIN` and `COMMENT` belong together
because they start with nothing in common at all. What holds them together is a
comment above them and everyone remembering.

**Instead.**

```frost
Class :: enum { Plain, Comment, Keyword, Type, Function, Number, String, Punct }

colour_of :: fn(class: Class) -> str {
    if (class == Class::Comment) { return "#6a9955" }
    ...
}
```

Now the parameter takes one of eight things, a caller cannot pass a row count,
and the set has a name.

**Where an ordering matters, match rather than fall through.**

```frost
// Antipattern: a stage the chain does not name silently sorts as the last one.
stage_order :: fn(s: Stage) -> i64 {
    if (s == Stage::First) { return 0 }
    if (s == Stage::Update) { return 1 }
    2
}

// Instead: a match covers every variant or it does not compile, so adding a
// stage is a compile error here rather than a system that runs in the wrong one.
stage_order :: fn(s: Stage) -> i64 {
    match s {
        case .First: 0
        case .Update: 1
        case .Last: 2
    }
}
```

And do not keep a `STAGE_COUNT :: 3` beside the enum. It is a third thing to
remember, and the loop that used it can read the bound off the systems it has.

## A set of bits is a `flags` declaration, not a run of constants either

**Antipattern.** The same run of constants, with `|` between them:

```frost
INIT_VIDEO   :: 32
INIT_AUDIO   :: 16
WINDOW_RESIZABLE :: 32

sdl_init(INIT_VIDEO | INIT_AUDIO)
```

`INIT_VIDEO` and `WINDOW_RESIZABLE` are both 32, so handing one where the other
belongs is a program that compiles and does the wrong thing. An enum is no help:
an enum holds exactly one alternative, and this holds several.

**Instead.**

```frost
InitFlags :: flags u32 {
    Audio  = 16,
    Video  = 32,
    Events = 16384,
}

sdl_init(InitFlags::Video | InitFlags::Audio)
if (flags_has(chosen, InitFlags::Video)) { ... }
```

The numbers are still C's, written down, because they have to be. What changed
is that the bits are named under a type. Two flags types are not
interchangeable, a bare number is refused, and `|` over two of one still answers
with one, so a combination goes straight into a call with no annotation.

The operators a set does not answer are refused: `+`, `<`, `<<` on a bit set are
questions about the number underneath, and the declaration exists to say that
the number is not what this is.

This replaced two `distinct` types and nine loose constants in
`examples/graphics/sdl.frost`, and five families of them in the generated wgpu
binding.

## When constants are right

Two cases, both in this repository:

- **Numbers a foreign header owns that are not a set.** The generated wgpu
  binding has several hundred enum discriminants, because they must be exactly
  what C says they are and Frost's enum picks its own.
- **Arithmetic.** The compiler's `STRUCT_BASE`, `POINTER_BASE` and the rest are
  added to indices to make type codes. They are numbers being used as numbers.

## Three forms, one sentence each

- `enum` is a closed set of alternatives, exactly one of them held. Matching on
  one is exhaustive.
- `flags` is a named set of bits over an integer, several of them held at once,
  combined with `|` and asked with `flags_has`.
- `distinct` is one integer with a meaning: a `Meters`, an `EntityId`. It
  answers to arithmetic, which the other two do not.

## A handle is a type, not a pointer

**Antipattern.**

```frost
wgpuDeviceCreateBuffer :: extern fn(handle: ^u8, descriptor: ^u8) -> ^u8
```

Every handle in the API is the same type, so a `Buffer` goes where a `Device`
belongs and the first anyone hears of it is a crash inside a driver.

**Instead.**

```frost
Device :: distinct ^u8
Buffer :: distinct ^u8

device_create_buffer :: fn(handle: Device, descriptor: ^BufferDescriptor) -> Buffer
```

A distinct type is not its representation, so passing a `Buffer` where a
`Device` belongs is a compile error, and so is passing a raw pointer to either.

**What this does not give you.** A distinct type can still be built from its
representation in a function's trailing expression, so a one-line function can
mint one from any pointer. Treat the guarantee as "these do not get mixed up"
rather than as "these cannot be forged".

## A wrapper that changes nothing vouches for nothing

**Antipattern.**

```frost
device_create_buffer :: fn(handle: ^u8, descriptor: ^u8) -> ^u8 {
    unsafe { wgpuDeviceCreateBuffer(handle, descriptor) }
}
```

The signature is the extern's signature. The call is as dangerous as it was. All
that moved is the word `unsafe`, out of the caller's sight, and the gate exists
so that word is *the complete list of places to look when something has
corrupted memory*. A wrapper like this shortens the list without shortening
the danger.

**Instead**, a wrapper earns the name by narrowing the signature:

```frost
// The count decides the size, and what comes back carries its length.
heap_slice :: fn($T: Type, count: i64) -> []T

// The path is text; the NUL is added here rather than promised by every caller.
fs_read :: fn(path: str) -> ReadResult

// C answers with nothing when it fails, and a handle of nothing is not one.
device_get_queue :: fn(handle: Device) -> Queue
```

Each of these takes something a caller cannot get wrong and hands back something
the language can check. If you cannot narrow the signature, leave the `unsafe`
at the call site where a reader will find it.

`--audit-unsafe` reports the two ways a block can vouch for nothing: holding no
unchecked operation, and sitting inside another that already covers it. It is
off by default.

## Own a resource with `linear`, not with a comment

**Antipattern.**

```frost
// The caller frees this with vec_free.
Vec :: struct($T: Type) { storage: []T, len: i64, cap: i64 }
```

The comment is the only thing holding the obligation. Forget the call and the
program leaks, silently, until it runs long enough to matter.

**Instead.**

```frost
Vec :: linear struct($T: Type) { storage: []T, len: i64, cap: i64 }
```

A linear value must be consumed exactly once, and `vec_free` taking `move v` is
what consumes it. Forgetting is a compile error. A struct holding a linear value
is linear too, so a `World` holding `Vec`s inherits the obligation without being
told.

This caught two real bugs the day it started working: two constructors that
built a container only to steal its fields and abandon the shell, and both tools
leaking a file buffer on every early-return path.

## Count the blocks in the test

Linearity catches a value nobody consumed. It does not catch a free that gives
back less than it took, because that is one function's arithmetic rather than a
program's shape:

```frost
// Antipattern: freeing a column allocated three replacement blocks so that a
// freed column could be reused. Every caller threw the column away.
column_free :: fn(mut c: Column) {
    heap_release($u8, c.data)
    c.data = heap_bytes(1)
    ...
}
```

Nothing failed. The tests passed. It leaked three blocks per column.

**Instead**, make the count observable and assert on it:

```frost
test "a world gives back every block it took" {
    before := heap_live()
    mut world := ecs_new()
    ecs_free(world)
    assert(heap_live() == before)
}
```

`heap_live` is the number of blocks the runtime has out. Every container in
`std/` has a test of this shape, and a leak is now a failing test rather than
something a profiler finds later.

Take the baseline after the first round of whatever you are measuring, so
what the test measures is the loop rather than the one-time setup.

## Reading a container's element by value makes a second owner

**Antipattern.**

```frost
table := vec_slice($Table, world.tables)[slot.table]
```

If `Table` owns anything, this is a copy of the owner: two values now believe
they hold the same storage.

**Instead.**

```frost
ref table := vec_slice($Table, world.tables)[slot.table]
```

`ref` binds a borrow of the place rather than a copy of the value. Once the
containers were linear the compiler started refusing the first form, which is
how the sixteen of them in the ECS were found.

Related: binding a *local* struct to a second name moves it.

```frost
root := ecs_spawn(world)
mut parent := root          // root is gone from here on
mut parent := Entity { id = root.id, generation = root.generation }   // instead
```

Binding a *parameter* to a name is the other way round: it copies, so writing
through the binding does not reach the caller. That is what makes a function
like `mask_with` read the way it should.

```frost
mask_with :: fn(m: Mask, index: i64) -> Mask {
    mut out := m                  // a copy of the caller's mask
    out.words[index / 64] = ...
    out
}
```

## A signature that takes `^i8` is asking for a promise

**Antipattern.**

```frost
print_cstr :: fn(text: ^i8)
fs_read :: fn(path: ^i8) -> ReadResult
```

Both are promising the bytes end in a NUL. Nothing checks it, and the call site
needs no `unsafe` to make the promise, so the danger is invisible at both ends.

**Instead**, take a `str`, which carries its length:

```frost
print_str :: fn(text: str)
fs_read :: fn(path: str) -> ReadResult
```

Where C needs a terminator, add it once inside the wrapper rather than asking
every caller for it. The remaining raw pointers in `std/` are in `mem.frost`,
where a raw pointer is the point, and in `thread.frost`, where the OS needs an
untyped context.

## An arity in a name is a missing language feature

**Antipattern.**

```frost
for_each1 :: fn($A: Type, $body: fn(mut []A, i64), ...)
for_each2 :: fn($A: Type, $B: Type, $body: fn(mut []A, mut []B, i64), ...)
for_each3 :: fn($A: Type, $B: Type, $C: Type, ...)
```

The number in the name is there because each arity needed its own declaration,
so a system reading four components had no call to make. Three functions, one
idea, and a ceiling nobody chose.

**Instead**, a compile-time list decides the arity:

```frost
for_each :: fn($body: Type, mut world: World, f: Filters, types: $...) {
    for T in types {
        query_with(q, component_of($T, world))
    }
    while (query_next(world, q)) {
        body(query_column($T, world, q, component_of($T, world))
            for T in types, q.count)
    }
}
```

`for_each($integrate, world, no_filters(), $Position, $Velocity)` emits exactly
what `for_each2` emitted, and a fourth component is a fourth element and a
fourth parameter.

What made this possible was three things the language did not have: a list may
hold types, a list may be handed on by naming it, and `g(T) for T in list` in an
argument list is one argument per element. The first version of this pattern was
written as "the arity limit is fine, three is enough". It was not enough, and
the fix was in the compiler rather than in the library.

## Generic code: what the compilers disagree about

These are shapes where the two compilers have differed, so prefer the form that
both accept.

- **Write a match as `match value { case .Variant: ... }`.** Parentheses around
  the scrutinee are not the syntax, and the error you get from the wrong form
  names a later line.
- **Do not write through a read parameter.** `into[at] = x` where `into: str` is
  a read borrow lowered differently by each backend. Bind `mut destination :=
  into` first, which is what you meant anyway.
- **A `(` or `[` that opens a line starts a statement.** `(table.mask & mask)`
  on its own line is not a call of the line above it.
- **An imported name inside an array literal** was not rewritten by the import
  pass until recently. A call in an array literal that reports an unknown
  variable is the shape to suspect.

## Where these came from

Every antipattern above was written by someone in this repository, mostly by
the same person, in the same week, while adding the feature the pattern belongs
to. They are not hypothetical, and none of them was caught by review. What
caught them was making the compiler able to say so: linearity, the unsafe audit,
the exhaustive match, and a heap counter a test can read.

The lesson worth keeping is the last one. When a class of mistake is invisible,
the fix is not to be more careful. It is to make it visible.
