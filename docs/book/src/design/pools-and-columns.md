# Pools, slabs, and columns

Pools and generational handles are Frost's memory model, and the pool itself is
written in Frost.

## The shape

A slab is a Frost struct with Frost operations over it, generic over element
type and capacity: `std/slab.frost`, with the same thing written out longhand in
`examples/native/generic_slab.frost`. Every example and every test uses it.
The runtime defines no `pool_new`, `pool_alloc`, `pool_get`, `pool_contains`,
`pool_free`, `pool_destroy`, `handle_index` or `handle_generation`, and the
compiler emits no implicit `pool_get` when a `Handle` indexes something that is
not slab-shaped. The runtime holds the bounds and generation aborts, the
assertions and the IO helpers, and the aborts are Frost too
(`runtime/runtime.frost`).

A struct is recognized as slab-shaped by having a `storage` array beside a
parallel `generations` array, and `p[handle]` against one compiles to inline
index and generation arithmetic over those two fields. No call, and the same
class of code generation arrays already get.

## The alternative

The other design is a `Pool<T>` built into the compiler, lowering to an opaque
`^u8` and calling a C runtime. That runtime is tiny and portable, and libc is
the most portable ABI there is, so at run time it costs nothing. It costs
elsewhere. The pool becomes C a Frost program cannot read or write, the language
cannot reach freestanding targets without libc, and the pool's behavior is
defined outside the language's own safety story.

The dependency to shed is the allocator. A pool runtime in C needs `malloc`,
`free`, `memcpy`, and pointer arithmetic. The generational free list is integer
and pointer manipulation the language already does. Only the allocation is
irreducible, and pools are fixed-capacity by design, so that their element
pointers stay stable, which means a pool needs no heap allocation at all. It can
live in a caller-provided buffer, or inside the struct.

The layer under this one, the arena as the primary allocator with the pool as
its fixed-size specialization, is in [allocators.md](allocators.md).

## What the compiler still supplies

The place-deref is compiler-supported. The columns container below requires
that, since selecting a column before indexing it is something a function
returning a borrow cannot say. So a pool is Frost storage, Frost logic, and a
thin compiler-generated accessor. Raw writes into the backing array are still
unsafe, and they sit in Frost the type checker reads rather than in an opaque C
file.

## Columns, the slab transposed

The slab stores whole elements back to back, an array of structs. A system that
reads one field across many elements, a physics step touching every position, a
render pass reading every transform, strides over the other fields it does not
want and pays a cache miss for each. The data-oriented answer is to store each
field in its own contiguous array, a structure of arrays, so that pass walks one
tight column. `columns<T, N>` is that transpose of the slab, and it keeps the
slab's generational safety unchanged.

`columns<T, N>` is declared in `std/columns.frost`, out of a `for` over
`fields(T)` in its own body (11.1d.1). For a struct `T` with fields `f1..fn`, it
lays out one `[N]` array per field, each named after the field, plus the same
`generations` / `free_list` / `free_count` bookkeeping the slab carries, and a
record of which slots hold an element:

```frost
import "math.frost"
Particle :: struct { position: Vec3, velocity: Vec3, mass: f32 }

// columns<Particle, 1024> is, in effect:
//   { position:   [1024]Vec3,
//     velocity:   [1024]Vec3,
//     mass:       [1024]f32,
//     generations:[1024]i64, free_list: [1024]i64, free_count: i64,
//     live_words:  [16]i64,   live_count: i64 }
```

"One array per field of `T`" is what a walk in a struct's body says, so the
layout is a library declaration and the compiler holds no shape of its own for
it. What the compiler still supplies is the two access forms, since each selects
a column before indexing and that is not writable where a struct is one value.
Each column is named after its field, so both use machinery that already
exists:

- `c.field` is the whole column. It is ordinary field access that yields the
  `[N]t` array, and an array coerces to a slice for free, so `c.position` passed
  to a `[]Vec3` parameter is the hot-loop view with no copy and no special form.
- `c[handle].field` is the checked single element. It selects the column
  first and then indexes it at the handle's slot, the mirror image of the slab
  (which indexes storage first and then selects the field). The slot comes from
  the same `frost_rt_slot` bounds-and-generation check the slab uses, so a stale
  handle aborts here exactly as it does there.

Construction (`columns_zeroed()`, a zeroed container), the deref `c[handle].field`
for read and write, and the element scatter `c[handle] = value` (a per-field
store at the validated slot) are the compiler-supplied pieces, for the same
reason `pool[handle]` is. They select a column before indexing, which a function
returning a second-class borrow cannot express. Everything else is an ordinary
Frost library. `std/columns.frost` provides `columns_reset`, `columns_full`,
`columns_len`, `columns_insert`, `columns_alive`, and `columns_release`, field
for field the
same as `std/slab.frost`, so moving a system from arrays-of-structs to
structure-of-arrays is changing `Slab<T, N>` to `columns<T, N>` and the `slab_`
prefix to `columns_`, and nothing else in the calling code.

### Walking the ones that hold something

`c.field` is every slot and says nothing about which of them are filled, and
`c[handle].field` is one slot and is checked. A loop from `0` to `N` therefore
reads storage nobody put anything in, which wastes work for an integration step
and answers wrongly for a sum. `generations` cannot tell you which slots are
filled either: an insert does not touch it, so a slot at generation zero may be
live or may never have held anything, and the free list carries liveness in
release order.

`live_words` is that knowledge in the order a column is stored in, one bit per
slot, set by `columns_insert` and cleared by `columns_release`. `for slot in
live_slots(c)` walks it: a word of zeroes passes over sixty-four slots on one
test, a word with bits set gives up its lowest, clears it, and goes round. No
empty slot is reached.

```frost,sketch
for slot in live_slots(c) {
    c.velocity[slot] = c.velocity[slot] + c.accel[slot] * dt
}
```

Two characters longer than `for slot in 0..N`, needing no `N` in scope, and the
body reads the same either way. The raw walk stays for the container a program
knows is packed, for a column handed to C or a GPU as one contiguous run, and
for any order but ascending. 10.1b lists what the walk cannot say.

The reserved element-field names are `storage`, `generations`, `free_list`,
`free_count`, `live_words`, and `live_count`, which would confuse the structural
recognizer that tells a columns container from a slab. The one operation not
provided is the whole-element gather `value := c[handle]`, because recovering a
`T` from the separate columns needs the element type the layout does not store.
Read the fields you want through `c[handle].field` instead.

It is built in both compilers (the Rust bootstrap in `src/ir/build.rs`, the
self-hosted compiler in `selfhosted/`), on both backends each, to a
byte-identical self-hosting fixpoint, because the language is defined by the two
compilers agreeing. `examples/selfhosted/soa_particles.frost` is a worked
example.
