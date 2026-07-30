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
`storage`, `generations`, `free_list`, and `free_count`. See
[pools-and-columns.md](../design/pools-and-columns.md).

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

The generation occupies the top 32 bits of the handle and is read back
sign-extended, so a slot's count may reach 2^31 - 1 and no further: past that
the generation packed into a handle no longer equals the slot's own count, and
the slot would hand out handles that were stale the moment they were made. A
slot that reaches the bound is retired rather than returned to the free list, so
the container loses one place and stays correct. Reaching it takes two billion
releases of a single slot.

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

So it is refused where the container is declared, in both compilers, for a
`Slab`, a `columns`, and a container of that shape written out by hand alike.
What to write instead is either a handle in the slot and the resource outside the
pool, or the elements beside the pool: one array of offsets giving each element
its range into a single run that owns the whole of it, which is one allocation for
the lot and a linear scan to walk.

### What the check costs, and how to pay it once

`pool[handle]` reads the slot's generation and compares it on every access. For a
handful of lookups that is nothing. For a pass that walks the same elements over
and over it is paid once a hop for an answer that cannot change while the pass
runs, and it is measurable: over twenty million hops, summing one field, the
handle form takes 53 ms against 26 ms for the same loop over a plain array. About
a nanosecond a hop, and most of a loop that does nothing else.

`slab_slot` is how that is paid once. It checks the generation, answers with the
slot the handle names, or `-1` where the handle is stale. Indexing `storage` with
that number is an ordinary array access: bounds-checked like any other, compared
against no generation.

```frost
slot := slab_slot($Unit, $1024, pool, handle)
if (slot >= 0) {
    mut round : i64 = 0
    while (round < many) {
        total = total + pool.storage[slot].hp
        round = round + 1
    }
}
```

The same twenty million hops through a slot take 33 ms, so about three quarters
of the check's cost goes. What is left over the plain array is the second index
rather than the check: a slot table is one more load than walking storage
directly.

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
