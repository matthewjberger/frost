# Allocation strategy

Frost's allocators come in a small stack of layers, and the pool is the
fixed-size-recycling member of that stack. How to write an allocation source,
what `uses` and `with` mean, and what the region check refuses are in the
reference chapter on
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
   batching lifetime instead of tracking it per object. `std/arena.frost` is
   one over a `[]u8` it was handed, handing out `[]T` runs so that what is built
   on it is bounds-checked, with `ptr_to`, `ptr_cast` and `slice_from` as the
   only primitives the compiler supplies. A run starts on `alignof(T)`, so a
   type asking for more than a word gets it.
   `examples/native/arena.frost` is the same thing written out by hand, so it
   shows the layer with nothing under it.

   `std/fixed.frost` is the container over a carved run: `Vec<T>` with the
   allocator taken out, so a growable array can live in a scratch region and the
   region check refuses it outliving one. A `Vec` keeps the heap, and neither
   type carries a field naming which allocator it came from.

   An arena whose backing came from the OS is declared `linear`, so the compiler
   requires it be destroyed exactly once and forgetting is a compile error. That
   keeps a dynamic allocation from leaking without a collector. An arena over a
   static or stack buffer has nothing to destroy and is a plain struct.

4. The pool is the fixed-size specialization. Same-sized slots, a free list, and
   individual free and reuse, with one addition over the textbook pool: a
   per-slot generation counter, so a stale handle to a reused slot is caught
   instead of silently reading the new occupant. `std/slab.frost` is that,
   written in Frost. A pool is backed by an arena or a static buffer, never by
   its own OS call, so a fixed-capacity one can live entirely inside a struct.
   See [pools-and-columns.md](pools-and-columns.md).

5. A general allocator, arbitrary sizes with individual free over a free list or
   a buddy scheme, exists for the rare case that genuinely needs it. Reach for
   it last, because it reintroduces fragmentation and the per-object-free bugs
   the arena avoids.

The arrangement inverts a malloc-centric one. The arena is the primary
allocator and bulk free is the primary lifetime tool.

## The allocator a function draws from is declared

Odin threads a `context.allocator` implicitly, so a callee allocates without
being handed an allocator. Frost does not. Nothing in the language runs
invisibly, and an ambient allocator is exactly the hidden, thread-local state
that rules out.

In its place Frost has a declaration. A function says `uses Arena<256>` in its
signature, which gives its body one implicit write parameter reached by the
source type's own name with its first letter lowercased, and a
`with arena { ... }` block around a call supplies it. A call that draws a source
neither its caller holds nor a surrounding block provides is refused at the
call. The allocator is still
threaded explicitly in the sense that matters. It is a real parameter, inserted
at a place the compiler can name, and a program cannot allocate from a source it
never asked for.

Two things come from making it a declaration instead of a plain parameter. A
function that only passes an arena through to something deeper stops having to
name it in its own signature, which is the convenience the ambient version
offers. And the `with` block is a region with a scope the compiler knows, which
gives the escape check something to reason about. A pointer into that arena may
not outlive the block, and that check would have nothing to say about a
thread-local whose boundaries the language cannot see.

The convenience worth keeping from the context idea is the scratch arena, one a
caller resets at a known boundary, per frame or per request, so transient
allocations cost nothing to free. That is a use of arena reset, and a function
may draw several sources at once for it.

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

This is a library pattern, separate from the language's mechanism. `uses` and
`with` decide *which* source a call draws from, at compile time, and pay
nothing. The struct above decides it at run time and costs an indirect call, so
use it when the answer is not known until then.

The middle of those two is a data structure generic over its allocator type,
which costs nothing and covers most of what a swap is wanted for. That is a
capability bundle: `std/allocation.frost` declares `Allocation<A>` holding
`take`, `resize` and `give` over `[]u8`, and `carve`, `carve_grow` and
`carve_give` are the typed face over them. The bundle arrives as a compile-time
argument, so a call through one of its fields is a direct call to the function
that field names, and the same body draws from the heap or from an arena:

```frost
import "allocation.frost"
import "arena.frost"

main :: fn() -> i64 {
    var h := heap_state()
    var run := carve($i64, $heap_source, h, 4)
    carve_give($heap_source, h, run)

    var backing: [64]u8 = [0; 64]
    var a := arena_over(backing)
    var scratch := carve($i64, $arena_source, a, 4)
    scratch[0]
}
```

What the checks say about such a call is what they say about any call whose body
they cannot see: the answer is worth the shortest-lived argument that could have
reached it. Carving from the caller's arena hands back the caller's storage;
carving from one built in this frame is refused, in the same words either
compiler gives for a bare pointer out of a frame.

### Which container draws from which source

`Vec<T>` draws from the heap and `Fixed<T>` draws from a run somebody else
carved. That split is the answer rather than a stop on the way to one container
parameterized by its source, and it follows from what a bump allocator is.

A container that grows asks its source for a bigger run holding what the old one
held. An arena answers by carving a second run and copying, because it has
nowhere to grow into, and the old run stays out until the whole arena resets. So
a container that grows out of an arena spends the arena at the rate it doubles,
and the run it abandons is live storage nothing will hand back before the reset.
A container in a scratch region is sized where it is carved instead, which is
what `Fixed<T>` is: capacity is the run's length, a push past the end aborts at
the index that reached past it, and there is no allocator under it to ask for
more. Both programs in the tree that fill a container out of an arena carve a run
whose size is known before the loop.

That leaves the heap as the only source a growing container draws from, and the
heap keeps no state, so `Vec<T>` names no allocator and carries no field saying
where its block came from. A capability a value has to be used with the same way
for its whole life belongs in that value's type, which is what `Map<K, V, ops>`
does with its hashing; the allocation source of a growing container is not one of
those, because there is only ever one of it.

`arena_source` fills the third field of `Allocation<A>` with a resize, so
`carve_grow($arena_source, a, run, n)` is written down and abandons the old run.
Whether `Arena` stops being an `Allocation<A>` or the bundle splits into a
resizing and a non-resizing one is open.

## What is left in C

One call that asks the operating system for a block of bytes, in the programs
that ask at all. Everything above it, the bump logic, the free list, the
generation counters, and the reset discipline, is Frost the language can inspect
and check. Fixed and static configurations reach freestanding targets with
nothing underneath them.
