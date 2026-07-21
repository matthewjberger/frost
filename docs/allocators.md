# Allocation strategy

The pool is one allocator, not the whole story. A data-oriented language needs a
small, layered set of allocators, and the pool is the fixed-size-recycling member
of that set. This note sets the allocator direction so the pool work
(`docs/native-pools.md`) sits on a real foundation rather than a single `malloc`
at the edge.

The design follows Ginger Bill's *Memory Allocation Strategies* (the Odin
allocator model): allocators are explicit, composable, and operate over plain
bytes, and freeing in bulk beats freeing per object. Frost already carries the
seeds of this. `Arena` is a parseable but inert type, `Context` is a dead type,
and the removed `push_context` and `push_allocator` forms were an ambient
allocator that was tried and dropped. That history points the way.

## The layers

Every allocator operates over a `[]u8` of backing memory. The layers differ only
in how they hand that memory out and take it back.

1. **Backing memory** is a `[]u8`. Its root is either a static or stack array
   (no operating system involved) or a single acquisition from the OS. Nothing
   below this layer allocates.

2. **The OS layer** is one platform primitive, `os_alloc(size) -> []u8` and
   `os_free(buffer)`, implemented per platform (`mmap`/`munmap`,
   `VirtualAlloc`/`VirtualFree`). This is the single irreducible platform extern.
   A freestanding program skips it entirely and roots every allocator in a static
   buffer.

3. **The arena (linear or bump) allocator is the workhorse.** An `Arena` holds a
   backing `[]u8` and a bump offset. Allocation is an aligned bump of the offset;
   there is no per-object free. You free everything at once by resetting the
   offset, or you save a marker and roll back to it (a stack discipline). This is
   O(1), fragmentation-free, and it eliminates a whole class of leak and
   use-after-free bugs because lifetime is batched. An arena is a linear resource
   in Frost's sense: created once, used freely, destroyed exactly once, which the
   move checker now enforces for `Pool<T>` and would enforce here identically.

4. **The pool allocator is the fixed-size specialization.** Same-sized blocks, a
   free list, and individual free and reuse. This is Frost's `Pool<T>`, with one
   addition over the textbook pool: a per-slot generation counter, so a stale
   handle to a reused slot is caught rather than silently reading the new
   occupant. A pool is backed by an arena or a static buffer, not by its own OS
   call.

5. **A general allocator** (arbitrary sizes, individual free, a free list or
   buddy scheme) exists only for the rare case that genuinely needs it. It is the
   last resort, not the default, because it reintroduces fragmentation and the
   per-object-free bugs the arena avoids.

The important inversion from a malloc-centric world: the arena is the primary
allocator and bulk free is the primary lifetime tool. The pool and the general
allocator are specializations you reach for by need, and both are backed by an
arena or a static buffer rather than by the OS directly.

## The allocator interface

To make backing memory swappable, an allocator is a value, not a hardcoded call.
Frost has function pointers and no closures, so the Odin shape fits without a
vtable:

```
Allocator :: struct {
    allocate: fn(^u8, usize, usize) -> ^u8,   // data, size, align
    release:  fn(^u8, ^u8),                    // data, pointer
    data:     ^u8,                             // the allocator's own state
}
```

A pool or a growable buffer takes an `Allocator` and never names `malloc`. Swap
the allocator and the same data structure runs against an arena, a static buffer,
or the OS. Compile-time selection (a data structure generic over its allocator
type) is the zero-cost alternative when no runtime swap is needed; the runtime
interface above is what enables swapping a temporary arena in for a scope, which
is the pattern that makes the model worth having.

## Explicit, not ambient

Odin threads a `context.allocator` implicitly so callees allocate without being
handed an allocator. Frost should not. The language's stance is that nothing runs
invisibly, and an ambient allocator is exactly the hidden, thread-local state that
stance rules out. Frost already removed `push_context` and `push_allocator`, and
it already threads mutable working state explicitly by second-class `&mut`
reference (the bootstrap compiler passes its `Parser` this way). So an arena is
passed as a `&mut Arena` parameter, the same as any other working set. That is
in character, and it costs no more than the threading the language already does.

The one convenience worth keeping from the context idea is the **temporary
arena**: a scratch arena that a caller resets at a known boundary (per frame, per
request), so transient allocations cost nothing to free. That is a use of arena
reset, not a reason for ambient state.

## Prerequisites

Two language features gate the allocator stack, and both are already on the pool
roadmap:

- **Slices** (`[]T`, and specifically `[]u8`). An allocator operates over a byte
  slice, a pointer plus a length. This is **done**: `[]T` is a fat-pointer view
  in the native backend, an array coerces to a slice, indexing is bounds-checked,
  and `slice_len` reads the length (`docs/architecture.md`,
  `examples/native/slices.frost`). It subsumes the raw `^u8` the current pool
  runtime passes around.
- **Value generics** (`$N` as a value). A static arena is `Arena<N>` and a static
  pool is `Pool<T, N>`, both sized at compile time with no allocation at all.
  This is the same feature the pool roadmap needs for `[N]T` storage.

## Disposition of the vestigial types

- **`Arena`** becomes the real bump allocator described here (layer 3), rather
  than the inert placeholder it is today. It keeps its name.
- **`Context`** is removed. It is dead (not even parseable), and the ambient
  context it stood for is explicitly rejected above in favor of passing
  allocators by reference.

## Where this sits in the roadmap

The allocator stack is the foundation under the pool work. Merged with the pool
roadmap, the order is:

1. **Slices** in the native backend (`[]T`, `[]u8`), the byte view every
   allocator needs. *(Done.)*
2. **Value generics** (`$N`), so arenas and pools are sized without allocation.
3. **Arena allocator** (revive `Arena`): a linear bump allocator over a `[]u8`,
   with reset and marker rollback.
4. **The allocator interface** and the pool rebuilt over an arena or static
   buffer in Frost, with `pool[handle]` staying the one compiler-supported place
   deref.
5. **The OS extern** (`os_alloc`/`os_free`) and a freestanding demo that roots
   everything in a static arena with no libc.
6. **Retire the C pool runtime** and remove `Context`, freeing the reserved names
   and leaving C only for genuine FFI.

The payoff is the same as for the pool, one layer down: the memory model is Frost
code the language can inspect and own, fixed and static configurations reach
freestanding targets, and the only thing left in C is the single platform call
that asks the operating system for a block of bytes.
