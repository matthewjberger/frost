# How Frost guarantees memory safety

Frost is memory-safe without a garbage collector and without lifetime
annotations. Safety is enforced entirely at compile time by a pass that runs
after parsing and before any code is generated (`src/ownership.rs`), backed by a
type system that makes the dangerous shapes *unrepresentable* rather than merely
*checked*.

This document explains each guarantee, why it holds, and where it is enforced.
Frost removes the need for a borrow checker's hardest machinery (lifetime
inference, region variables) by making references second-class. Moves,
exclusivity, resource cleanup, and dangling-pointer freedom all follow from that
one decision plus a small number of local rules.

## The six guarantees

1. Nothing that borrows storage outlives it. A borrow taken by a parameter
   mode is implicit, lasts exactly one call, and has nowhere to be written down,
   so it cannot escape at all. `ref T` is the explicit exception, the one borrow
   a program writes, and it may be returned, so it is held instead to storage
   that already outlives the call. A function answering with a borrowed view
   (`ref T`, a raw pointer, or a slice) may not hand back one formed from its
   own frame, whether by returning it, by storing it where the call cannot see,
   or by ending with it (`src/regions.rs`, the frame check). A view it was
   handed names storage the caller owns and passes back out freely. What no
   borrow of either kind may do is be stored: not in a struct field, not in an
   array element, not in a container.

   The region check asks the same question about an arena. A pointer into one
   may not outlive the `with` block that owns it, and a `uses` function may hand
   one back to its caller, where that caller's own region check catches it, but
   may not store one into a parameter.
2. No use-after-move. A non-`Copy` value is consumed when moved. Using it
   again is a compile error.
3. No mutable aliasing. Within a call, a value cannot be passed to two
   `mut` parameters at once, nor to a `mut` and a read parameter at the same
   time.
4. No leaked resources. A `linear` value must be consumed exactly once. A
   live-but-unconsumed linear value at end of scope is a compile error.
5. No use-after-free through a stale handle. A generational handle whose slot
   has been freed and reused reports "not contained". It can never silently read
   a live value.
6. No out-of-bounds array access. Every array index is bounds-checked against
   the array's statically-known length. An out-of-range index aborts with a
   diagnostic rather than reading or writing past the array.

One case is not covered. A raw pointer is unchecked once it is out of the frame and
region checks, which is what `^T` is for. It carries no guarantee, and a program
that casts one with `ptr_cast` and reads through it is on its own.

The first four hold statically. The fifth uses a runtime generation check that
stays cheap (one integer compare) because the static rules keep handles honest.
A handle is plain copyable data, not a reference, so the compiler never has to
track its lifetime. The sixth is a single compile-time-known length compare on
each array access.

## 1. Second-class borrows, so no dangling pointers

A borrow is second-class: it may be passed and it may be returned, and it may
never be stored. Mostly it is not even a type a program writes. It is what a
*parameter mode* means. `x: T` borrows to read, `mut x: T` borrows to mutate,
`move x: T` takes ownership, and the call site writes no sigil at all.

```frost
read  :: fn(x: i64) -> i64 { x }        // borrowed to read
bump  :: fn(mut x: i64) { x = x + 1 }   // borrowed to mutate in place
eat   :: fn(move p: Point) -> i64 { p.x }

bump(n)                                  // no '&mut' at the call
```

A parameter-mode borrow cannot escape because there is nowhere to put it. A
reference-typed struct field is rejected, an enum variant's field the same, and
so is an `extern` that returns one, all by `check_ownership` reading
`Type::contains_reference()` over the declared types. No container can hold one
either, since a container's element type is a field type.

The exception a program can write is `ref T`, a returnable borrow of a place.
An accessor over a container needs it. `arena_at` hands back the element rather
than a copy of it, so a caller writes `entry.kind = ...` through the borrow, and
without it every such accessor would be a read-and-write-back pair or an
`unsafe` block over a raw pointer. So a return position is deliberately allowed,
and what keeps it sound is the frame check rather than the type: a function that
answers with a borrowed view may not answer with one built from its own frame.
`ref T` is still storable nowhere.

That is why there are no lifetimes to infer and no lifetime annotations. The
question a lifetime variable answers, how long the storage behind this borrow
lives, is replaced by a provenance question with two answers, this frame or the
caller's, which a single pass over the function reads off the shape of the code.
The borrow analysis stays scope-local.

The same rule is what makes `pool[handle]` sound (see section 5). Passing
`pool[handle]` to a function borrows it under that function's parameter mode,
and that borrow is second-class like any other. You cannot stash it in a struct
or return it, so it cannot dangle past the pool operation.

Enforced in `check_ownership` via `Type::contains_reference()` on declared
struct and enum field types and on an `extern`'s return type, and in
`check_frame_escapes` for what a Frost function answers with.

## 2. Move checking, so no use-after-move

Every type is either copy or move, and the line is drawn by
`Type::is_copy()`. Copy is the scalars (integers, floats, bools), raw pointers,
borrows, function pointers, handles, `str`, a slice, and a fixed array whatever
its element type. Move is structs and enums, and a distinct type is whichever
the type it is represented by is. A move-typed value is *consumed* when it is:

- passed by value to a function,
- assigned to another binding, or
- returned.

Using it afterward is a use-after-move error:

```frost
p := Point { x = 1, y = 2 }
a := take(p)      // p moved into take
b := take(p)      // error: use of moved value 'p'
```

Passing to a read or `mut` parameter, field read (`x.f`), and dereference (`p^`)
do not consume, so the common read patterns are unaffected. Only a `move`
parameter takes the value. Copy types are never moved, so
`add(x, x)` with integer `x` is fine.

Enforced per function body by `MoveChecker` in `src/ownership.rs`, which tracks a
set of moved bindings and their types.

## 3. Borrow exclusivity, so no mutable aliasing

A `mut` borrow is exclusive. Within a single call the checker rejects:

- passing the same variable to two `mut` parameters, and
- passing it to a `mut` parameter and a read parameter at once.

```
// add :: fn(mut a: i64, mut b: i64)   mix :: fn(a: i64, mut b: i64)
add(x, x)   // rejected: aliased mutable borrows
mix(x, x)   // rejected: shared and mutable borrow of the same value
add(x, y)   // fine: distinct variables
sum(x, x)   // fine: multiple shared borrows
```

Which argument is a borrow and which kind it is comes from the callee's
parameter modes, so the check reads the signature rather than the call's syntax.

This per-call check is sufficient, not merely necessary, because borrows are
second-class. A borrow cannot be saved to be aliased later, so there is no
cross-call aliasing to reason about. The question "who else holds a reference to
this?" collapses to "what does this one call borrow?".

Enforced by `check_borrow_exclusivity` per call-site argument list.

## 4. Linear resources, so no leaks, no double-free, and non-ignorable errors

A struct or enum declared `linear` is a resource that must be consumed
exactly once:

```frost
File :: linear struct { handle: i64 }
open  :: fn() -> File { File { handle = 1 } }
close :: extern fn(f: File)              // terminal consumer, across the FFI boundary
```

- At most once comes from the move checker (section 2). Consuming a linear
  value moves it, so a second use is a use-after-move error, and there is no
  double-free.
- At least once is the new rule. A linear value still live at the end of the
  function that owns it is a "never consumed" error, and there is no leak.

Consuming means moving the value onward, returning it, passing it by value to
another function (typically an `extern` that takes ownership across the FFI
boundary), or `match`ing it (a `match` on a linear value destructures and
consumes it). This is how Frost replaces `Drop`. Cleanup is an obligation the
type system tracks, not an implicit call inserted behind your back.

A `linear enum` returned from a fallible function is a non-ignorable error. You
cannot drop it on the floor, so a failure must be matched (or otherwise
consumed), and silently swallowing an error becomes a compile error.

Enforced by the same `MoveChecker`, tracking which linear bindings remain live at
scope exit.

## 5. Generational handles, so no use-after-free through the heap

Long-lived data lives in a pool and is referred to by a generational
handle (`Handle<T>`), not a raw pointer. A handle is a packed `(index,
generation)` pair, which is plain copyable data you *can* freely store and return
(unlike a reference).

- `slab_insert` puts a value in a free slot and returns a handle carrying that
  slot's current generation.
- `slab_release` bumps the slot's generation and returns it to the free list.
- Any later access checks the handle's generation against the slot's current
  generation. If they differ, the handle is stale. `slab_alive` answers
  false, and reading `world[h]` aborts rather than returning the new occupant.

```frost
h := slab_insert($Entity, $8, world, entity)    // slot 0, generation 0
slab_release($Entity, $8, world, h)             // slot 0 now generation 1
slab_insert($Entity, $8, world, other)          // reuses slot 0 at generation 1
slab_alive($Entity, $8, world, h)               // false, the old handle can never
                                                // read the new occupant
```

Those operations are ordinary Frost, not compiler builtins or a runtime.
`std/slab.frost` is the whole implementation, generic over element type and
capacity, and `examples/native/generic_slab.frost` is the same thing written out
as a worked example. The only part the compiler supplies is the validated
place-deref `world[h]`, which is inline index and generation arithmetic against
the struct's own fields rather than a call.

This is the memory-safety property a raw pointer cannot give you. After a free
and reuse, the *bit pattern* of the old handle no longer matches, so it cannot be
used to read or corrupt whatever now occupies the slot. That is safe
use-after-free detection without a GC and without reference counting.

### Handle-dereference-as-borrow

`pool[handle]` is a place. You can read and write fields through it
(`world[h].hp = 60`), copy the element out (`e := world[h]`), or pass it to a
function, which borrows it under that function's parameter mode. The element
type is recovered from the handle's `Handle<T>`, so the pool itself stays a raw
pointer.

The borrow you get is second-class (section 1), so there is nowhere to put it
that would let it escape the region where the pool operation is valid. Handles
unify with the borrow discipline. The *handle* is data you keep. The *borrow*
through it is a scoped thing the language gives you no way to save.

## 6. Bounds-checked indexing, so no out-of-bounds access

A fixed-size array `[N]T` carries its length `N` in its type, so every index
expression `a[i]` is compiled with a check against that known length before the
address is computed:

```frost
arr := [10, 20, 30]
arr[5]   // aborts: "frost: index 5 out of bounds for length 3"
```

The check is a single call to a small runtime routine
(`frost_rt_bounds_check(index, length)`) that aborts if the index is out of range.
The comparison is unsigned, so a negative index (which would wrap to a huge
unsigned value) is caught too. Valid accesses are unaffected. A silent
out-of-bounds read or write, the classic C memory-safety hole, becomes a loud,
deterministic abort.

Pool access does not need this check. `pool[handle]` is guarded by the
generational check instead (section 5).

## Why this is enough, and why it is small

Traditional borrow checking spends most of its complexity on lifetimes,
inferring how long each reference is valid, relating those regions to each other,
and threading them through generics. Frost pays a different price up front, that
a borrow cannot be stored and a returned one is judged by where its storage came
from, and in exchange deletes that entire machinery.

| Hazard                       | How Frost removes it                                   |
| ---------------------------- | ------------------------------------------------------ |
| Dangling reference           | A borrow is unstorable, a returned one is not this frame's |
| Use-after-move               | Move checking on non-`Copy` values                     |
| Mutable aliasing             | Per-call borrow exclusivity (sufficient, not just necessary) |
| Leak / double-free / drop    | Linear resources: consume exactly once                 |
| Use-after-free via heap      | Generational handles: stale handle detected at access  |
| Out-of-bounds access         | Array length is known; every index is bounds-checked   |
| Ignored error                | Linear error enums are non-ignorable                   |

None of these requires lifetime variables, region inference, or a runtime GC.
The analysis is a single AST pass. The only per-value runtime cost is one integer
compare per handle access and one per array index.

## Where the unchecked work lives

A container has to allocate, and allocating needs two things the compiler cannot
check: a call into C for the bytes, and a reinterpretation of those bytes as a
typed pointer. Written at each site that is three `unsafe` blocks per container
and the count-times-size arithmetic repeated wherever it is easy to get wrong.

`std/mem.frost` is that floor, and it is the only file in the standard library's
containers that contains an `unsafe` block. It hands back a slice rather than a
pointer:

```frost
keys := heap_slice($i64, capacity)     // []i64, not ^i64
```

A slice carries its length, so every later access through it is bounds-checked,
which is what leaves `std/vec.frost` and `std/map.frost` with no `unsafe` of
their own: the whole body of a hash map is ordinary safe code, and the only
unchecked operations are the allocation and the release. For a container whose
element width is decided while the program runs, `heap_bytes`, `bytes_at` and
`bytes_as` are the same move: the byte arithmetic is written once, and what
comes back is a bounds-checked `[]T`. `std/ecs.frost` is written entirely on
those, so it too has no `unsafe` block.

This is the same shape `arena_at` has: push the unchecked operation down into
one audited function and hand back something the language can check.

## What is not yet guarded

A few honest gaps in the current implementation:

- Raw pointers (`^T`) are an explicit escape hatch, used for FFI and the pool
  runtime's internals. They are `Copy` and unchecked, exactly like C pointers,
  and code that uses them takes on the corresponding responsibility. The safe
  surface of borrows, handles, and linear resources is what the guarantees above
  cover. `check_frame_escapes` narrows the hatch. A raw pointer formed from this
  frame's own storage, including one taken with `ptr_to` or a slice over a local
  array, cannot be returned.
- A callback's guarantee stops at the C boundary. The Frost side is checked.
  The context moves in and comes back out, the registration is `linear` so
  forgetting to unregister is a compile error, and the region check holds the
  registration to the frame that holds its context, so no Frost code can read
  the context while the callback might fire. None of that says anything about
  the library's own threading, and a library that keeps the pointer after
  unregistration is outside what the compiler can see. See
  [callbacks.md](callbacks.md).
- The static checks run on the AST, so integer overflow follows the backend's
  C semantics (wrapping for unsigned, two's-complement for signed) rather than
  trapping.

These are implementation gaps, not holes in the design. The design's job is to
make the safe constructs (borrows, handles, linear resources) impossible to
misuse.
