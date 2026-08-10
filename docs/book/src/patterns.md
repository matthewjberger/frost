# Patterns, and what to write instead

Everything in this chapter is legal Frost, and the antipatterns compile. They
are antipatterns because the compiler stops helping you the moment you write
one.

## Use an enum for a set of alternatives

**Antipattern.**

```frost,sketch
PLAIN    :: 0
COMMENT  :: 1
KEYWORD  :: 2
TYPE     :: 3
FUNCTION :: 4

colour_of :: fn(class: i64) -> str { ... }
```

This is legal, and a C header looks exactly like it, which is why it keeps
appearing. It costs you three things.

The parameter says nothing. `class: i64` accepts a row count, a file descriptor,
or `-1`. Nothing is checked, because there is nothing to check against.

It cannot be matched. `match` over an enum is exhaustive, so adding a variant
stops every match that does not handle it from compiling. A run of constants has
no such set, so the `if` chain that reads them silently keeps its old answer for
a value it has never seen.

The prefix is doing the type's job. Only a comment above them and everyone's
memory holds `PLAIN` and `COMMENT` together.

**Instead.**

```frost,sketch
Class :: enum { Plain, Comment, Keyword, Type, Function, Number, String, Punct }

colour_of :: fn(class: Class) -> str {
    if (class == Class::Comment) { return "#6a9955" }
    ...
}
```

Now the parameter takes one of eight things, a caller cannot pass a row count,
and the set has a name.

**Where an ordering matters, match on the enum.**

```frost,sketch
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
        case Stage::First: 0
        case Stage::Update: 1
        case Stage::Last: 2
    }
}
```

And do not keep a `STAGE_COUNT :: 3` beside the enum. It is a third thing to
remember, and the loop that used it can read the bound off the systems it has.

## Use `flags` for a set of bits

**Antipattern.** The same run of constants, with `|` between them:

```frost,sketch
INIT_VIDEO   :: 32
INIT_AUDIO   :: 16
WINDOW_RESIZABLE :: 32

sdl_init(INIT_VIDEO | INIT_AUDIO)
```

`INIT_VIDEO` and `WINDOW_RESIZABLE` are both 32, so handing one where the other
belongs is a program that compiles and does the wrong thing. An enum is no help
here. An enum holds exactly one alternative, and this holds several.

**Instead.**

```frost,sketch
InitFlags :: flags u32 {
    Audio :: 16
    Video :: 32
    Events :: 16384
}

sdl_init(InitFlags::Video | InitFlags::Audio)
if (flags_has(chosen, InitFlags::Video)) { ... }
```

The numbers are still C's, written down, because they have to be. The bits are
named under a type. Two flags types are not interchangeable, a bare number is
refused, and `|` over two of one still answers with one, so a combination goes
straight into a call with no annotation.

`+`, `<` and `<<` on a bit set are refused. They are questions about the number
underneath, and a `flags` declaration says the value is a set of named bits.

`lib/platform/sdl.frost` declares its initialisation and window bits this way,
and the generated wgpu binding declares five families of them.

## When constants are right

Two cases, both in this repository.

A foreign header owns numbers that do not form a set. The generated wgpu binding
has several hundred enum discriminants, because they must be exactly what C says
they are and Frost's enum picks its own.

The numbers are arithmetic. The compiler's `STRUCT_BASE`, `POINTER_BASE` and the
rest are added to indices to make type codes. They are numbers being used as
numbers.

## Three forms, one sentence each

- `enum` is a closed set of alternatives, exactly one of them held. Matching on
  one is exhaustive.
- `flags` is a named set of bits over an integer, several of them held at once,
  combined with `|` and asked with `flags_has`.
- `distinct` is one integer with a meaning: a `Meters`, an `EntityId`. It
  answers to arithmetic, and `enum` and `flags` do not.

## Give each handle a distinct type

**Antipattern.**

```frost
wgpuDeviceCreateBuffer :: extern fn(handle: ^u8, descriptor: ^u8) -> ^u8
```

Every handle in the API is the same type, so a `Buffer` goes where a `Device`
belongs and the first anyone hears of it is a crash inside a driver.

**Instead.**

```frost,sketch
Device :: distinct ^u8
Buffer :: distinct ^u8

device_create_buffer :: fn(handle: Device, descriptor: ^BufferDescriptor) -> Buffer
```

A distinct type is a type of its own, so passing a `Buffer` where a `Device`
belongs is a compile error, and so is passing a raw pointer to either.

A distinct type can still be built from its representation in a function's
trailing expression, so a one-line function can mint one from any pointer. The
guarantee covers accidental mixing. Forging one takes a line written on purpose.

## A wrapper that changes nothing vouches for nothing

**Antipattern.**

```frost,sketch
device_create_buffer :: fn(handle: ^u8, descriptor: ^u8) -> ^u8 {
    unsafe { wgpuDeviceCreateBuffer(handle, descriptor) }
}
```

The signature is the extern's signature, and the call is as dangerous as it was.
All that moved is the word `unsafe`, out of the caller's sight. That word is the
complete list of places to look when something has corrupted memory, and a
wrapper like this shortens the list without shortening the danger.

**Instead**, narrow the signature:

```frost,sketch
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

Every build reports the two ways a block can vouch for nothing: holding no
unchecked operation, and sitting inside another that already covers it.
`--audit-unsafe` turns that report into a failure, which is what holds a tree
to zero of them.

## Own a resource with `linear`

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

A linear value must be consumed exactly once, and `vec_free` consumes one by
taking `move v`. Forgetting is a compile error. A struct holding a linear value
is linear too, so a `World` holding `Vec`s inherits the obligation without being
told.

## Count the blocks in the test

Linearity catches a value nobody consumed. A free that gives back less than it
took is one function's arithmetic, and linearity does not see it:

```frost,sketch
// Antipattern: freeing a column allocated three replacement blocks so that a
// freed column could be reused. Every caller threw the column away.
column_free :: fn(mut c: Column) {
    heap_release(c.data)
    c.data = heap_bytes(1)
    ...
}
```

Nothing fails, the tests pass, and the program leaks three blocks per column.

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
`std/` that takes from the heap has a test of this shape, so a leak shows up as
a failing test.

Take the baseline after the first round of whatever you are measuring, which
leaves the one-time setup outside the count.

## Reading a container's element by value makes a second owner

**Antipattern.**

```frost,sketch
table := vec_slice(world.tables)[slot.table]
```

If `Table` owns anything, this is a copy of the owner, and two values now
believe they hold the same storage.

**Instead.**

```frost,sketch
ref table := vec_slice(world.tables)[slot.table]
```

`ref` binds a borrow of the place, so there is still one owner. With linear
containers the compiler refuses the first form outright.

Binding a *local* struct to a second name moves it.

```frost,sketch
root := ecs_spawn(world)
mut parent := root          // root is gone from here on
mut parent := Entity { id = root.id, generation = root.generation }   // instead
```

Binding a *parameter* to a name is the other way round. It copies, so writing
through the binding does not reach the caller, which is how a function like
`mask_with` gets a mask of its own to edit.

```frost,sketch
mask_with :: fn(m: Mask, index: i64) -> Mask {
    mut out := m                  // a copy of the caller's mask
    out.words[index / 64] = ...
    out
}
```

## A signature that takes `^i8` is asking for a promise

**Antipattern.**

```frost,sketch
log_line :: fn(text: ^i8)
fs_read :: fn(path: ^i8) -> ReadResult
```

Both are promising the bytes end in a NUL. Nothing checks it, and the call site
needs no `unsafe` to make the promise, so the danger is invisible at both ends.

**Instead**, take a `str`, which carries its length:

```frost,sketch
log_line :: fn(text: str)
fs_read :: fn(path: str) -> ReadResult
```

Where C needs a terminator, add it once inside the wrapper. The raw pointers
left in `std/` are where a run of bytes is the subject: `mem.frost`, the byte
stores in `ecs.frost`, `fs.frost` and `snapshot.frost`, and `thread.frost`,
where the OS needs an untyped context.

## An arity in a name is a missing language feature

**Antipattern.**

```frost,sketch
for_each1 :: fn($A: Type, $body: fn(mut []A, i64), ...)
for_each2 :: fn($A: Type, $B: Type, $body: fn(mut []A, mut []B, i64), ...)
for_each3 :: fn($A: Type, $B: Type, $C: Type, ...)
```

The number in the name is there because each arity needs its own declaration, so
a system reading four components has no call to make. Three functions carry one
idea, and the ceiling sits wherever the last declaration stopped.

**Instead**, a compile-time list decides the arity:

```frost,sketch
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
what `for_each2` emits, and a fourth component is a fourth element and a fourth
parameter.

Three language rules carry this: a compile-time list may hold types, a list may
be handed on by naming it, and `g(T) for T in list` in an argument list expands
to one argument per element.

## Shapes that misread, and the form to write

Four shapes read differently than they look, and the report you get points
somewhere else. Write the plain form.

- Write a match as `match value { case Enum::Variant: ... }`. The scrutinee takes no
  parentheses, and the error you get from the other form names a later line.
- Do not write through a read parameter. `into[at] = x` where `into: str` is a
  read borrow lowered differently by each backend. Bind `mut destination :=
  into` first.
- A `(` or `[` that opens a line starts a statement. `(table.mask & mask)` on
  its own line is a statement of its own, not a call of the line above it.
- Suspect an array literal when an imported name reads as undeclared. A call
  inside one that reports an unknown variable is that shape.

