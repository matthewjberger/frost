# 10. Handles, pools, and the memory model

## 10.1 Pools

A pool is a contiguous, fixed-capacity arena of same-typed elements addressed by
`Handle<T>` rather than pointer. A pool is not a language type: it is a struct a
program writes for itself, holding the storage and a free list, with the
generational `(generation << 32) | index` handle and the stale-handle check as
ordinary code. `examples/native/generic_slab.frost` is a pool generic over both
element type and capacity, built on value generics (`[N]T` storage, 11.1a) and
slices (`slice_len` to recover the capacity). The compiler provides the pieces,
arrays, handles, value generics, `ptr_to`/`ptr_cast`, and the byte buffer, not
the pool itself. This is the data-oriented memory model expressed in the
language, and the direction is written up in
[pools-and-columns.md](../design/pools-and-columns.md) and
[allocators.md](../design/allocators.md).

The runtime (`runtime/frost_runtime.c`) has no pool in it. It holds aborts,
assertions, IO, and the checks a handle deref calls (`frost_rt_bounds_check`,
`frost_rt_generation_check`, and `frost_rt_slot`, which runs both and answers
with the validated index). Nothing in it allocates or hands out a slot. The pool
the standard library offers is
`std/slab.frost`, ordinary Frost: a `Slab<T, N>` carrying `storage`,
`generations`, `free_list` and `free_count`, with `slab_reset`, `slab_full`,
`slab_insert`, `slab_alive` and `slab_release` written out as ordinary
functions over them. Reaching one through a handle is the one part the compiler
supplies (10.2).

## 10.1a Structure-of-arrays columns

`columns<T, N>` is the structure-of-arrays transpose of a pool. Where a pool (or
slab) stores whole elements contiguously, `columns<T, N>` stores each field of
`T` in its own `[N]` array, so a pass reading one field across many elements
walks a tight column rather than striding over the fields it does not want. It is
a compiler-synthesized type, not a library struct: for a `T` with fields
`f1..fn`, the compiler reflects over them and lays out one `[N]` array per field
named after that field, plus the same `generations` / `free_list` / `free_count`
bookkeeping a pool carries.

Naming each column after its field gives two accesses from existing machinery.
`c.field` is ordinary field access yielding the whole `[N]t` array, which coerces
to a `[]t` slice for a hot loop at no cost. `c[handle].field` selects the column
and then indexes it at the handle's slot, the mirror of a pool (which indexes
storage and then selects the field), under the same bounds-and-generation check
(10.3). Construction (`columns_new()`), the deref `c[handle].field`, and the
element scatter `c[handle] = value` are compiler-supplied for the reason
`pool[handle]` is (10.2): they select a column before indexing, which a
second-class borrow cannot express. Everything else is a library,
`std/columns.frost`, mirroring `std/slab.frost`. The reserved field names are
`storage`, `generations`, `free_list`, `free_count`, `live_words`, and
`live_count`. See [pools-and-columns.md](../design/pools-and-columns.md).

`slab_new()` is a zeroed `Slab<T, N>` of the type the context wants, the twin of
`columns_new()`. A slab's arrays have lengths worked out from `N`, so writing
them out at every construction was the worst part of using one, and the
`live_words` array's length is `(N + 63) / 64`, which is a number a reader
should not have to work out. Construct with `slab_new()` and then `slab_reset`,
which is the contract a columns container already had.

## 10.1b `for slot in live(c)`

`c.field` is every slot, released ones included, and nothing about it says which
of them hold an element. `c[handle].field` is one slot and is checked. So the
shortest loop over a column is the one that reads storage nobody put anything in,
which is wasted work for an integration step and a wrong answer for a sum.

`live(c)` is what that loop should have said:

```frost
for slot in live(c) {
    c.velocity[slot] = c.velocity[slot] + c.accel[slot] * dt
}
```

Two characters longer than `for slot in 0..N`, and it does not need `N` in
scope. The body is unchanged: `slot` is a number, columns are indexed with it,
and no generation is read, because the walk answered that question by finding
the slot.

`for rank, slot in live(c)` counts the elements as it goes, in the same order
`for index, name in` reads in, which is what compacting into a packed buffer
wants:

```frost
for rank, slot in live(c) {
    upload[rank] = c.position[slot]
}
```

`live(c)` is the subject of a `for` and may be written nowhere else. There is no
sequence value, nothing to bind, and nothing to hand on. Its subject is a name or
a field of one, since the container is read where it stands rather than bound.
A `Slab<T, N>` carries the same record and is walked the same way, with
`s.storage[slot]` where a columns container has `c.field[slot]`.

The container records which slots hold an element as one bit each in
`live_words`, set by `columns_insert` and cleared by `columns_release`. The walk
reads those words: one that is zero passes over sixty-four slots on a single
test, and one with bits set gives up its lowest, clears it, and goes round. No
slot is asked whether it holds an element and no empty slot is reached. `break`
and `continue` mean what they mean in any other loop.

What the form cannot say, which is why the raw column walk stays:

- A column as a contiguous slice. `c.position` is a `[]Vec3` over all `N` slots,
  which is what a bulk copy, a GPU upload or a C call takes. A live walk hands
  out one slot at a time and can never produce a run.
- Any order but ascending, once. Reverse, stride, two-pointer, binary search over
  a sorted column.
- A neighbour. `c.x[slot - 1]` is writable, but that slot may hold nothing and
  the walk does not say. A prefix sum or a finite difference belongs to a
  container the program knows is packed.
- Two containers in lockstep. There is no zip, and two walks have unrelated
  bits. A parallel array outside the container indexed by the same `slot` is
  fine, which is most of what zip is wanted for.
- Inserting or releasing into the container being walked. That edits the words
  the walk is reading.
- A container that is never fragmented, where every word is full and the
  per-word test buys nothing against `for slot in 0..N`.

## 10.2 `pool[handle]` is a place

Which struct is a pool is decided by shape, not by a declaration. A struct is
slab-shaped when it declares a `storage` array and a parallel `generations`
array (`slab_shaped_base`, `src/ir_build.rs`), and indexing one by a `Handle<T>`
is generated inline rather than lowered to a call:

- the handle is read back as the `i64` it is at the ABI, its low 32 bits the
  slot index and its high 32 bits the generation;
- the index goes through `frost_rt_bounds_check` against the length the
  `storage` array was declared with;
- the generation goes through `frost_rt_generation_check` against
  `generations[index]`;
- what is left is the address of `storage[index]`.

`c[handle].field` on a `columns<T, N>` (10.1a) runs the same two checks and
scales the checked index into the named column instead.

`pool[handle]` is a place. Read a field, write a field, copy the element out, or
pass it to a parameter, which borrows it. The borrow obtained is second-class and
cannot escape the call.

## 10.3 Generational safety

A handle carries the generation of the slot it was minted for. Freeing a slot
increments its generation. A lookup whose handle generation does not match the
slot's current generation aborts rather than returning the slot's new occupant,
so a stale handle can never read or write freed-and-reused data.

The top 32 bits of a handle hold the container's number in the upper seven and
the slot's generation in the lower twenty-four, and the two are compared as one
word. So a slot's count may reach 2^24 - 2 and no further: one more would carry
into the container's number and the slot would answer for a container it is not
in. A slot that reaches the bound is retired rather than returned to the free
list, so the container loses one place and stays correct. Reaching it takes
sixteen million releases of a single slot.

### A handle names its container

A generation says a slot has been reused. It says nothing about which container
the slot is in, and two containers of the same element type and capacity are the
ordinary shape: `active` and `pending`, `current` and `next`. A handle from one
used against the other has an index in range on both, and its generation matches
whenever the two slots have been released the same number of times. Right after
both are reset that is every slot, because every generation is zero. The state
with no protection at all is the one a program starts in.

`slab_reset` and `columns_reset` take a number for the container from
`frost_rt_container_id` and stamp it into every generation, so a handle carries
which container minted it. A deref against another container aborts, saying so.

It costs nothing where a handle is read. The number sits in the same word as the
generation, the deref already compares that word, and a handle is still an `i64`
that converts freely. A container stores no handles, only `generations`, so no
layout changes: `columns<T, N>` is what it was.

Two containers share a number once in a hundred and twenty-seven, and a program
that resets a container in a loop comes back round to the same one after that
many. Where it is wrong it is wrong the way it was before, and everywhere else a
handle that was silently read is an abort that names the reason.

The direction of that failure is the point. The check compares the slot's
ever-increasing count against a sign-extended 32-bit value, so a count past the
bound can never equal an older handle's generation: what a spent slot produces
is a handle nothing accepts, never a stale handle something does. `std/slab.frost`
and `std/columns.frost` retire the slot; the pool examples under `examples/` are
written for the shape and say so.

### A pool holds data, not resources

An element type may not be `linear`, and may not hold anything that is. A slot is
emptied by bumping a generation and filled again by an insert that overwrites
what was there, so nothing consumes the element that leaves: the pool carries one
obligation and its slots carry none. Nor can a consumer be written to make up the
difference, since releasing each element means consuming `s.storage[i]` around a
loop, and a move inside a loop is refused because nothing says the indexes
differ.

What is asked about is the slot table, which is the elements a handle addresses.
For a slab that is `storage`, and another run the struct happens to carry is not
one: a field holding resources makes the struct a resource by the ordinary linear
rule, and that is where it is answered for. A `columns` container has no
`storage`, since one array per field of the element is what it is, so every column
is asked and `generations`, `free_list` and `free_count` are not.

It is refused where the container is declared, in both compilers, for a `Slab`, a
`columns`, and a container of that shape written out by hand alike.
What to write instead is either a handle in the slot and the resource outside the
pool, or the elements beside the pool: one array of offsets giving each element
its range into a single run that owns the whole of it, which is one allocation for
the lot and a linear scan to walk.

### What the check costs, and how to pay it once

`pool[handle]` reads the slot's generation and compares it on every access. For a
handful of lookups that is nothing. For a pass that walks the same elements over
and over it is paid once a hop for an answer that cannot change while the pass
runs, and it is measurable: over tens of millions of hops summing one field, the
handle form costs about twice what the same loop over a plain array does. That is
roughly a nanosecond a hop, which is most of a loop that does nothing else and a
rounding error in one that does anything.

`slab_slot` is how that is paid once. It checks the generation, answers with the
slot the handle names, or `-1` where the handle is stale. Indexing `storage` with
that number is an ordinary array access: bounds-checked like any other, compared
against no generation.

```frost
slot := slab_slot($Unit, $1024, pool, handle)
if (slot >= 0) {
    var round : i64 = 0
    while (round < many) {
        total = total + pool.storage[slot].hp
        round = round + 1
    }
}
```

The same walk through a slot gives back most of the check's cost. What is left
over the plain array is the second index rather than the check: a slot table is
one more load than walking storage directly.

What it costs in the other direction is the guarantee. A slot is a number, and
nothing says it still names the element it named. Release that slot while a loop
holds its number and the loop reads whatever moved in, which is the reading a
handle exists to refuse. So this belongs where a walk owns its data for the
length of the walk, and the handle belongs everywhere else. It is a word at the
site for that reason, the same way leaving the range is `wrap_add` rather than a
compiler flag.

## 10.4 Bounds checking

Every fixed-array index is checked against the statically known length. An
out-of-range index aborts (`frost_rt_bounds_check`). There is no unchecked-index
form.

## 10.5 The six guarantees

| Guarantee | Mechanism |
| --- | --- |
| No dangling references | an implicit borrow cannot escape (8.2); a `ref` is frame- and region-checked (8.4) |
| No use-after-move | move checking (8.1) |
| No mutable aliasing | per-call borrow exclusivity (8.3) |
| No leaked resources | linear consume-exactly-once (9.1) |
| No use-after-free of pooled data | generational handles (10.3) |
| No out-of-bounds array access | bounds checking (10.4) |

Raw pointers (`^T`) are outside these guarantees by design.
