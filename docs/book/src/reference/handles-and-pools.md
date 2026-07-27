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
language, and the direction is written up in `docs/native-pools.md` and
`docs/allocators.md`.

The runtime in `runtime/frost_runtime.c` offers a ready-made generational pool
(`pool_new`, `pool_alloc`, `pool_get`, `pool_free`, `pool_contains`,
`pool_destroy`), reachable as an opt-in library by declaring the functions with
`extern fn`, the way `malloc` is. Nothing about it is compiler-special. When a
pool from it is indexed by a `Handle<T>`, `pool[handle]` lowers to its `pool_get`
(10.2).

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
`docs/native-pools.md`.

## 10.2 `pool[handle]` is a place

`pool[handle]` is a place. Read a field, write a field, copy the element out, or
pass it to a parameter, which borrows it. The borrow obtained is second-class and
cannot escape the call.

## 10.3 Generational safety

A handle carries the generation of the slot it was minted for. Freeing a slot
increments its generation. A lookup whose handle generation does not match the
slot's current generation fails rather than returning the slot's new occupant, so
a stale handle can never read or write freed-and-reused data.

## 10.4 Bounds checking

Every fixed-array index is checked against the statically known length. An
out-of-range index aborts (`frost_rt_bounds_check`). There is no unchecked-index
form.

## 10.5 The six guarantees

| Guarantee | Mechanism |
| --- | --- |
| No dangling references | references are second-class (8.2) |
| No use-after-move | move checking (8.1) |
| No mutable aliasing | per-call borrow exclusivity (8.3) |
| No leaked resources | linear consume-exactly-once (9.1) |
| No use-after-free of pooled data | generational handles (10.3) |
| No out-of-bounds array access | bounds checking (10.4) |

Raw pointers (`^T`) are outside these guarantees by design.
