# How Frost Guarantees Memory Safety

Frost is memory-safe **without a garbage collector and without lifetime
annotations**. Safety is enforced entirely at compile time by a pass that runs
after parsing and before any code is generated (`src/ownership.rs`), backed by a
type system that makes the dangerous shapes *unrepresentable* rather than merely
*checked*.

This document explains each guarantee, why it holds, and where it is enforced.
Frost removes the need for a borrow checker's hardest machinery (lifetime
inference, region variables) by making references **second-class**. Moves,
exclusivity, resource cleanup, and dangling-pointer freedom all follow from that
one decision plus a small number of local rules.

## The six guarantees

1. **Nothing that borrows storage outlives it.** A borrow can never outlive the
   value it names. Borrows exist only as parameter modes and so last exactly one
   call, and the two things that could carry storage out of a frame are checked
   instead of forbidden: a raw pointer formed from a local, and a slice over one,
   may not be returned (`src/regions.rs`, the frame check). A pointer a function
   was handed is not its frame's and passes back out freely.

   The same check covers arenas. A raw pointer into an arena may not outlive the
   `with` block that owns it, and a `uses` function may hand one back to its
   caller, whose region checks it, but may not store one into a parameter.
2. **No use-after-move.** A non-`Copy` value is consumed when moved. Using it
   again is a compile error.
3. **No mutable aliasing.** Within a call, a value cannot be passed to two
   `mut` parameters at once, nor to a `mut` and a read parameter at the same
   time.
4. **No leaked resources.** A `linear` value must be consumed exactly once. A
   live-but-unconsumed linear value at end of scope is a compile error.
5. **No use-after-free through a stale handle.** A generational handle whose slot
   has been freed and reused reports "not contained". It can never silently read
   a live value.
6. **No out-of-bounds array access.** Every array index is bounds-checked against
   the array's statically-known length. An out-of-range index aborts with a
   diagnostic rather than reading or writing past the array.

What is not covered: a raw pointer is unchecked once it is out of the frame and
region checks, which is what `^T` is for. It carries no guarantee, and a program
that casts one with `ptr_cast` and reads through it is on its own.

The first four hold statically. The fifth uses a runtime generation check that
stays cheap (one integer compare) because the static rules keep handles honest.
A handle is plain copyable data, not a reference, so the compiler never has to
track its lifetime. The sixth is a single compile-time-known length compare on
each array access.

---

## 1. Second-class borrows, so no dangling pointers

A borrow is **second-class**: it exists only as a *parameter mode*, and there is
no reference type in the surface language to write anywhere else. `x: T` borrows
to read, `mut x: T` borrows to mutate, `move x: T` takes ownership, and the call
site writes no sigil at all.

```
read  :: fn(x: i64) -> i64 { x }        // borrowed to read
bump  :: fn(mut x: i64) { x = x + 1 }   // borrowed to mutate in place
eat   :: fn(move p: Point) -> i64 { p.x }

bump(n)                                  // no '&mut' at the call
```

Since the only place a borrow can appear is a parameter, the shapes that would
let one escape are not expressible. There is no way to write a reference-typed
struct field, and no way to write a reference return type, so a borrow can never
outlive the call it was created for. There is nothing to outlive, so there are
**no lifetimes to infer and no lifetime annotations**. That is what lets the
borrow analysis stay entirely scope-local.

The lowering still forms reference types internally, and `check_ownership` still
rejects them in fields and return positions, which is what keeps a synthesized
reference from escaping either.

The same rule is what makes `pool[handle]` sound (see section 5). Passing
`pool[handle]` to a function borrows it under that function's parameter mode,
and that borrow is second-class like any other. You cannot stash it in a struct
or return it, so it cannot dangle past the pool operation.

Enforced in `check_ownership` via `Type::contains_reference()` on declared
struct/enum field types and function return signatures.

## 2. Move checking, so no use-after-move

Every type is either **Copy** (integers, floats, bools, raw pointers,
references, function pointers, handles) or a **move** type (structs, enums,
strings, arrays of move types). A move-typed value is *consumed* when it is:

- passed by value to a function,
- assigned to another binding, or
- returned.

Using it afterward is a use-after-move error:

```
p := Point { x = 1, y = 2 }
a := take(p)      // p moved into take
b := take(p)      // error: use of moved value 'p'
```

Passing to a read or `mut` parameter, field read (`x.f`), and dereference (`p^`)
do **not** consume, so the common read patterns are unaffected. Only a `move`
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

A struct or enum declared `linear` is a **resource** that must be consumed
**exactly once**:

```
File :: linear struct { handle: i64 }
open  :: fn() -> File { File { handle = 1 } }
close :: extern fn(f: File)              // terminal consumer, across the FFI boundary
```

- **At most once** comes from the move checker (section 2). Consuming a linear
  value moves it, so a second use is a use-after-move error, and there is no
  double-free.
- **At least once** is the new rule. A linear value still live at the end of the
  function that owns it is a "never consumed" error, and there is no leak.

Consuming means moving the value onward, returning it, passing it by value to
another function (typically an `extern` that takes ownership across the FFI
boundary), or `match`ing it (a `match` on a linear value destructures and
consumes it). This is how Frost **replaces `Drop`**. Cleanup is an obligation the
type system tracks, not an implicit call inserted behind your back.

There is a useful consequence. A `linear enum` returned from a fallible function
is a **non-ignorable error**. You cannot drop it on the floor, so a failure must
be matched (or otherwise consumed), and silently swallowing an error becomes a
compile error.

Enforced by the same `MoveChecker`, tracking which linear bindings remain live at
scope exit.

## 5. Generational handles, so no use-after-free through the heap

Long-lived data lives in a **pool** and is referred to by a **generational
handle** (`Handle<T>`), not a raw pointer. A handle is a packed `(index,
generation)` pair, which is plain copyable data you *can* freely store and return
(unlike a reference).

- `pool_alloc` puts a value in a free slot and returns a handle carrying that
  slot's current generation.
- `pool_free` bumps the slot's generation and returns it to the free list.
- Any later access checks the handle's generation against the slot's current
  generation. If they differ, the handle is **stale** and the access fails
  (`pool_contains` returns 0, a checked get returns nothing).

```
h := pool_alloc(world, ptr_to(entity))   // slot 0, generation 0
pool_free(world, h)                      // slot 0 now generation 1
pool_alloc(world, ptr_to(other))         // reuses slot 0 at generation 1
pool_contains(world, h)           // 0, the old handle can never read the new occupant
```

This is the memory-safety property a raw pointer cannot give you. After a free
and reuse, the *bit pattern* of the old handle no longer matches, so it cannot be
used to read or corrupt whatever now occupies the slot. That is safe
use-after-free detection without a GC and without reference counting.

### Handle-dereference-as-borrow

`pool[handle]` is a **place**. You can read and write fields through it
(`world[h].hp = 60`), copy the element out (`e := world[h]`), or pass it to a
function, which borrows it under that function's parameter mode. The element
type is recovered from the handle's `Handle<T>`, so the pool itself stays a raw
pointer.

The borrow you get is **second-class** (section 1), so there is nowhere to put it
that would let it escape the region where the pool operation is valid. Handles
unify with the borrow discipline. The *handle* is data you keep. The *borrow*
through it is a scoped thing the language gives you no way to save.

---

## 6. Bounds-checked indexing, so no out-of-bounds access

A fixed-size array `[N]T` carries its length `N` in its type, so every index
expression `a[i]` is compiled with a check against that known length before the
address is computed:

```
arr := [10, 20, 30]
arr[5]   // aborts: "frost: index 5 out of bounds for length 3"
```

The check is a single call to a small runtime routine
(`frost_bounds_check(index, length)`) that aborts if the index is out of range.
The comparison is unsigned, so a negative index (which would wrap to a huge
unsigned value) is caught too. Valid accesses are unaffected. A silent
out-of-bounds read or write, the classic C memory-safety hole, becomes a loud,
deterministic abort.

Pool access does not need this check. `pool[handle]` is guarded by the
generational check instead (section 5).

## Why this is enough, and why it is small

Traditional borrow checking spends most of its complexity on **lifetimes**,
inferring how long each reference is valid, relating those regions to each other,
and threading them through generics. Frost pays a different price up front, that
references cannot escape, and in exchange deletes that entire machinery.

| Hazard                       | How Frost removes it                                   |
| ---------------------------- | ------------------------------------------------------ |
| Dangling reference           | References are second-class and cannot outlive the call |
| Use-after-move               | Move checking on non-`Copy` values                     |
| Mutable aliasing             | Per-call borrow exclusivity (sufficient, not just necessary) |
| Leak / double-free / drop    | Linear resources: consume exactly once                 |
| Use-after-free via heap      | Generational handles: stale handle detected at access  |
| Out-of-bounds access         | Array length is known; every index is bounds-checked   |
| Ignored error                | Linear error enums are non-ignorable                   |

None of these requires lifetime variables, region inference, or a runtime GC.
The analysis is a single AST pass. The only per-value runtime cost is one integer
compare per handle access and one per array index.

## What is not yet guarded

A few honest gaps in the current implementation:

- **Raw pointers** (`^T`) are an explicit escape hatch, used for FFI and the pool
  runtime's internals. They are `Copy` and unchecked, exactly like C pointers,
  and code that uses them takes on the corresponding responsibility. The safe
  surface of borrows, handles, and linear resources is what the guarantees above
  cover. `check_frame_escapes` narrows the hatch: a raw pointer formed from this
  frame's own storage, including one taken with `ptr_to` or a slice over a local
  array, cannot be returned.
- **Callbacks used to be an unsafe API** and are no longer, which is worth
  recording here because it was the one place the implementation contradicted
  the goal above. A callback is now a compile-time function argument plus a
  typed context the caller moves in and gets back on unregistration, with the
  registration `linear` so forgetting to unregister is a compile error, and the
  region check holding the registration to the frame that holds its context.
  Nothing in a program that uses one names a `^u8`. See
  [callbacks.md](callbacks.md).

  What that guarantee is *not*: it says no Frost code holds the context while
  the callback can fire, and it says the registration cannot outlive the
  storage. It says nothing about the C library's own threading, and a library
  that keeps the pointer after unregistration is outside anything the compiler
  can see.
- **Uninitialized reads are gone from aggregate construction**, which is worth
  noting because they were not. A struct or enum-variant literal that left a
  field out used to compile, and the storage that field named was never written,
  so reading it read whatever was on the stack. A literal now has to write every
  field, named in the error when it does not.
- The static checks run on the AST, so **integer overflow** follows the backend's
  C semantics (wrapping for unsigned, two's-complement for signed) rather than
  trapping.

These are implementation gaps, not holes in the design. The design's job is to
make the safe constructs (borrows, handles, linear resources) impossible to
misuse, and that is what the six guarantees above deliver.
