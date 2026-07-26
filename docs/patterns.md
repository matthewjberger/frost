# Patterns, and what to write instead

What the language rewards and what it merely permits. Everything here is legal
Frost; the antipatterns compile. What makes them antipatterns is that the
compiler stops helping you the moment you write them, and the help is the whole
reason the rules are there.

Each one is written from something that happened in this repository, and the
"instead" is what the code says now.

---

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
appearing. What it costs:

- **The parameter says nothing.** `class: i64` accepts a row count, a file
  descriptor, or `-1`. Nothing is checked, because there is nothing to check
  against.
- **It cannot be matched.** `match` over an enum is exhaustive: add a variant
  and every match that does not handle it stops compiling. A run of constants
  has no such set, so the `if` chain that reads them silently keeps its old
  answer for a value it has never seen.
- **The prefix is doing the type's job.** `PLAIN` and `COMMENT` belong together
  because they start with nothing in common at all; they are held together by a
  comment above them and by everyone remembering.

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

## When constants are right

Three cases, all in this repository:

- **Numbers a foreign header owns.** `examples/graphics/wgpu.frost` has several
  hundred, because they must be exactly what C says they are. An enum's
  discriminants are the compiler's to choose, so an enum here would be a
  different number wearing the right name.
- **Bit masks.** SDL's `INIT_VIDEO` and `WINDOW_RESIZABLE` are combined with
  `|`, and an enum is a choice of one alternative rather than a set of several.
  A `distinct` type is what gives these the safety an enum would have:

  ```frost
  InitFlags :: distinct u32
  WindowFlags :: distinct u64

  mut flags : WindowFlags = WINDOW_RESIZABLE | WINDOW_HIGH_PIXEL_DENSITY
  ```

  Neither can be handed where the other belongs, and `|` over one still answers
  with one.
- **Arithmetic.** The compiler's `STRUCT_BASE`, `POINTER_BASE` and the rest are
  added to indices to make type codes. They are numbers being used as numbers.

---

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
mint one from any pointer. Treat the guarantee as "these do not get mixed up",
not as "these cannot be forged".

---

## A wrapper that changes nothing vouches for nothing

**Antipattern.**

```frost
device_create_buffer :: fn(handle: ^u8, descriptor: ^u8) -> ^u8 {
    unsafe { wgpuDeviceCreateBuffer(handle, descriptor) }
}
```

The signature is the extern's signature. The call is as dangerous as it was. All
that moved is the word `unsafe`, out of the caller's sight — and the gate exists
precisely so that word is *the complete list of places to look when something
has corrupted memory*. A wrapper like this shortens the list without shortening
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

---

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

Take the baseline **after** the first round of whatever you are measuring, so
what the test measures is the loop rather than the one-time setup.

---

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

Related: binding a struct to a second name moves it.

```frost
root := ecs_spawn(world)
mut parent := root          // root is gone from here on
mut parent := Entity { id = root.id, generation = root.generation }   // instead
```

---

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

---

## Generic code: what the compilers disagree about

These are not style. They are shapes where the two compilers have differed, so
prefer the form that both accept.

- **Write a match as `match value { case .Variant: ... }`.** Parentheses around
  the scrutinee are not the syntax, and the error you get from the wrong form
  names a later line.
- **Do not write through a read parameter.** `into[at] = x` where `into: str` is
  a read borrow lowered differently by each backend; bind `mut destination :=
  into` first, which is what you meant anyway.
- **A `(` or `[` that opens a line starts a statement.** `(table.mask & mask)`
  on its own line is not a call of the line above it.
- **An imported name inside an array literal** was not rewritten by the import
  pass until recently; if a call in an array literal reports an unknown
  variable, that is the shape to suspect.

---

## Where these came from

Every antipattern above was written by someone in this repository — mostly by
the same person, in the same week, while adding the feature the pattern belongs
to. They are not hypothetical, and none of them was caught by review. What
caught them was making the compiler able to say so: linearity, the unsafe audit,
the exhaustive match, and a heap counter a test can read.

The lesson worth keeping is the last one. When a class of mistake is invisible,
the fix is not to be more careful. It is to make it visible.
