# Allocation strategy

A data-oriented language needs a small, layered set of allocators, and the pool
is the fixed-size-recycling member of that set. This chapter is the layer model
and the reasoning behind it. How to write an
allocation source, what `uses` and `with` mean, and what the region check
refuses are in the reference chapter on
[allocation and regions](../reference/allocation-and-regions.md).

The design follows Ginger Bill's *Memory Allocation Strategies*, the Odin
allocator model, on two of its three points. Allocators are explicit and they
operate over plain bytes, and freeing in bulk beats freeing per object. The
third point, an allocator threaded through an ambient context, is the one Frost
rejects, for the reason in the last section.

## The layers

Every allocator operates over a `[]u8` of backing memory. The layers differ only
in how they hand that memory out and take it back.

1. Backing memory is a `[]u8`. Its root is either a static or stack array (no
   operating system involved) or a single acquisition from the OS. Nothing below
   this layer allocates.

2. The OS layer is one call, and a program that has one declares it as an
   ordinary `extern`. `examples/native/dynamic_arena.frost` roots its backing in
   `malloc` and frees it in a consumer the compiler makes it call. That is the
   whole platform dependency, and a freestanding program (`frost --link
   --freestanding`) has none of it. It roots every allocator in a static buffer
   and imports only the platform's exit function.

3. The arena, a bump allocator, is the default. It holds a backing `[]u8` and an
   offset. Allocation is an aligned bump of the offset. There is no per-object
   free. You free everything at once by resetting the offset, or you save a
   marker and roll back to it, a stack discipline. That is O(1),
   fragmentation-free, and it removes a class of leak and use-after-free bug by
   batching lifetime rather than tracking it per object. `std/arena.frost` is
   one over a fixed `[N]u8` inside the struct, handing out `[]T` runs so that
   what is built on it is bounds-checked, with `ptr_to`, `ptr_cast` and
   `slice_from` as the only primitives the compiler supplies.
   `examples/native/arena.frost` is the same thing written out by hand, which is
   what the layer looks like with nothing under it.

   `std/fixed.frost` is the container over a carved run: `Vec<T>` with the
   allocator taken out, so a growable array can live in a scratch region and the
   region check refuses it outliving one. A `Vec` keeps the heap, and neither
   type carries a field naming which allocator it came from.

   An arena whose backing came from the OS is declared `linear`, so the compiler
   requires it be destroyed exactly once and forgetting is a compile error. That
   is what keeps a dynamic allocation from leaking without a collector. An arena
   over a static or stack buffer has nothing to destroy and is a plain struct.

4. The pool is the fixed-size specialization. Same-sized slots, a free list, and
   individual free and reuse, with one addition over the textbook pool: a
   per-slot generation counter, so a stale handle to a reused slot is caught
   rather than silently reading the new occupant. `std/slab.frost` is that, and
   it is Frost rather than a compiler type. A pool is backed by an arena or a
   static buffer, never by its own OS call, which is what lets a fixed-capacity
   one live entirely inside a struct. See
   [pools-and-columns.md](pools-and-columns.md).

5. A general allocator, arbitrary sizes with individual free over a free list or
   a buddy scheme, exists for the rare case that genuinely needs it. It is the
   last resort rather than the default, because it reintroduces fragmentation
   and the per-object-free bugs the arena avoids.

The inversion from a malloc-centric world is that the arena is the primary
allocator and bulk free is the primary lifetime tool. The pool and the general
allocator are specializations reached for by need, and both are backed by an
arena or a static buffer rather than by the OS directly.

## Explicit, not ambient

Odin threads a `context.allocator` implicitly, so a callee allocates without
being handed an allocator. Frost does not. Nothing in the language runs
invisibly, and an ambient allocator is exactly the hidden, thread-local state
that rules out. An earlier `push_context` and `push_allocator` pair was that
idea tried and removed.

What replaced it is a declaration rather than a convention. A function says
`uses Arena<256>` in its signature, which gives its body one implicit write
parameter reached by the source type's own name lowercased, and a `with arena
{ ... }` block around a call supplies it. A call that draws a source neither its
caller holds nor a surrounding block provides is refused at the call. The
allocator is still threaded explicitly in the sense that matters. It is a real
parameter, inserted at a place the compiler can name, and a program cannot
allocate from a source it never asked for.

Two things come from making it a declaration instead of a plain parameter. A
function that only passes an arena through to something deeper stops having to
name it in its own signature, which is what made the ambient version attractive
in the first place. And the `with` block is a region with a scope the compiler
knows, which is what the escape check reasons about. A pointer into that arena
may not outlive the block, and that check would have nothing to say about a
thread-local whose boundaries the language cannot see.

The convenience worth keeping from the context idea is the scratch arena, one a
caller resets at a known boundary, per frame or per request, so transient
allocations cost nothing to free. That is a use of arena reset, not a reason for
ambient state, and a function may draw several sources at once for it.

## An allocator as a value

Where the backing has to be chosen while the program runs, an allocator is
written as a value rather than a call. Frost has function pointers and no
closures, so the Odin shape fits with no vtable and no compiler support at all,
because it is an ordinary struct:

```frost
Allocator :: struct {
    take:  fn(^u8, i64) -> ^u8,   // state, size
    state: ^u8,                   // the allocator's own state
}
```

A container takes one of those and never names `malloc`, so the same code runs
against an arena, a static buffer, or the OS.
`examples/native/allocator.frost` is a bump allocator behind that interface.

This is a library pattern and not the language's mechanism, and the distinction
is worth keeping straight. `uses` and `with` decide *which* source a call draws
from, at compile time, and pay nothing. The struct above decides it at run time
and costs an indirect call, so it earns its place only when the answer is not
known until then. A data structure generic over its allocator type is the middle
that costs nothing, and it covers most of what a swap is wanted for.

## What is left in C

One call that asks the operating system for a block of bytes, in the programs
that ask at all. Everything above it, the bump logic, the free list, the
generation counters, and the reset discipline, is Frost the language can inspect
and check. Fixed and static configurations reach freestanding targets with
nothing underneath them.
