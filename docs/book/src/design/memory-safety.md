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
   (`ref T`, a raw pointer, or a slice) hands one back only where the check can
   trace its storage to a parameter or an allocation capability. Storage it
   cannot trace is refused, whether the view leaves by being returned, by being
   stored where the call cannot see, or by being what the block ends with
   (`src/regions.rs`, and `check_frame_escapes` in `selfhosted/regions.frost`;
   both compilers refuse the same programs). A view the function was handed names
   storage the caller owns and passes back out freely. What no borrow of either
   kind may do is be stored: not in a struct field, not in an array element, not
   in a container.

   Refusing what it cannot trace is the point rather than an implementation
   detail. The check answered "this does not name my frame" for every expression
   form nobody had taught it, which made each road a view could travel a hole
   until someone wrote it down: an ordinary call, a call through a function
   pointer, an assignment into a local, a `return` inside a match arm, the
   address of a `move` parameter. Every one of those compiled and handed back a
   view of a dead frame. A check whose soundness rests on having enumerated
   every shape is a list, not a proof.

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
lives, is replaced by a provenance question with three answers, which a single
pass over the function reads off the shape of the code: storage this call did
not create, storage this frame owns, and neither shown. A view leaves the call
only on the first. The third is what makes the pass a proof rather than a list,
since a value built out of parts is worth its shortest-lived part and a shape
the walk cannot follow has no shortest-lived part to name. The borrow analysis
stays scope-local.

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
| Dangling reference           | A borrow is unstorable, a returned one is traced to storage that outlives the call |
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

The guarantees above are what the checks prove. This is what they do not, stated
so nobody has to find out by reading the passes.

- Raw pointers (`^T`) are an explicit escape hatch, used for FFI and the pool
  runtime's internals. They are `Copy` and unchecked, exactly like C pointers,
  and code that uses them takes on the corresponding responsibility. The safe
  surface of borrows, handles, and linear resources is what the guarantees above
  cover. `check_frame_escapes` narrows the hatch: a raw pointer whose storage
  cannot be traced past the call cannot leave it.
- `slice_from($T, p, n)` is a trusted primitive. It asserts that `n` elements of
  `T` live at `p`, and nothing checks that assertion. Every bounds-check
  guarantee downstream of a slice is conditional on it, which is why the call is
  gated on an `unsafe` block and why `std/mem.frost` is the one place in the
  containers that writes one.

  What is checked is the part of that claim which can be: `n` is refused where
  it is negative. The bounds check compares unsigned, so one comparison answers
  for a negative index as well as for one past the end, and the same cast read a
  negative length as enormous and let every index through. A length is settled
  once where the slice is built while an access happens in a loop, so it is
  answered for there. Two allocation shapes are checked for the same reason:
  `count * sizeof(T)` is refused where the product would wrap, since a wrapped
  size asks for fewer bytes than the caller believes and the slice over that
  block then reads past its end with every access reporting as bounds-checked,
  and an allocation that fails aborts rather than answering with a null a caller
  would wrap in a slice. A negative count and a wrapped size are not
  unverifiable claims but meaningless ones, which is why they are refused rather
  than trusted.
- The roughly 150 hand-written `unsafe` blocks in the standard library, the
  compiler and the examples are audited rather than proven. `Vec`, `Map` and the
  ECS are ordinary safe code resting on `std/mem.frost` being right.
- Two places reached through different raw pointers are read as apart, so
  `f(p^, q^)` with `p` and `q` holding one address passes the exclusivity check.
  Both dereferences are gated on an `unsafe` block, which is the reason this is
  left where it is.
- Integer overflow wraps rather than trapping. An index computed with arithmetic
  that overflows wraps to some other number, and the bounds check then runs on
  that number: the read stays inside the array and lands on the wrong element. A
  wrong answer, not a wrong address.
- There is no stack-depth guard. Unbounded recursion runs the stack out and the
  process dies however the host decides.
- A callback's guarantee stops at the C boundary. The Frost side is checked.
  The context moves in and comes back out, the registration is `linear` so
  forgetting to unregister is a compile error, and the region check holds the
  registration to the frame that holds its context, so no Frost code can read
  the context while the callback might fire. None of that says anything about
  the library's own threading, and a library that keeps the pointer after
  unregistration is outside what the compiler can see. See
  [callbacks.md](callbacks.md).
- There is no data-race story. `std/thread.frost` is the reasonable-C floor:
  the spawner owns the context and must keep it alive until the join, and shared
  state goes through `atomic_add` or the program races.

Two things the language rules out by construction rather than by checking, which
is why they are not on the list above. A binding cannot be declared without a
value, so there is no uninitialized read to catch. And an implicit borrow has no
type to write down, so no expression stores one.
