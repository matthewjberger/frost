# Pools, slabs, and columns

Frost's thesis is that pools and generational handles *are* the memory model, so
where the pool itself lives is a question about whether the language can carry
its own thesis. It lives in Frost.

## The shape

A slab is a Frost struct with Frost operations over it, generic over element
type and capacity: `std/slab.frost`, with the same thing written out longhand in
`examples/native/generic_slab.frost`. Every example and every test uses it.
`runtime/frost_runtime.c` defines no `pool_new`, `pool_alloc`, `pool_get`,
`pool_contains`, `pool_free`, `pool_destroy`, `handle_index` or
`handle_generation`, and the compiler emits no implicit `pool_get` when a
`Handle` indexes something that is not slab-shaped. What is left of the runtime
is about a hundred lines: bounds and generation aborts, assertions, and IO
helpers.

A struct is recognized as slab-shaped by having a `storage` array beside a
parallel `generations` array, and `p[handle]` against one compiles to inline
index and generation arithmetic over those two fields. No call, and the same
class of code generation arrays already get.

The argument below is what decided that shape, and the distinction underlying
it is the part that applies again elsewhere. How the spike that proved it out
went, and the five steps that took the pool out of C, are in
[history.md](../appendix/history.md).

## The question it answers

The alternative was a `Pool<T>` built into the compiler, lowering to an opaque
`^u8` and calling a C runtime. That runtime is tiny and portable, and libc is the
most portable ABI there is, so functionally it costs nothing. The concern is
strategic. It makes the flagship data-oriented primitive uninspectable,
unwritable C, which means the language cannot demonstrate its own core idea in
itself, cannot reach freestanding targets without libc, and defines the pool's
behavior outside its
own safety story.

The distinction is that the dependency worth shedding was the *allocator*,
not the *C*. Look at what the pool runtime used from C: `malloc`,
`free`, `memcpy`, and pointer arithmetic. The generational free list is pure
integer and pointer manipulation the language can already do.
Only the allocation was irreducible, and pools are fixed-capacity by design (so
their element pointers stay stable), which means a pool does not even need a heap
allocation. It can live in a caller-provided buffer, or, as it turned out,
inside the struct.

The layer under this one, the arena as the primary allocator with the pool as
its fixed-size specialization, is in [allocators.md](allocators.md).

## The honest caveat

The pool is never a pure library. The place-deref stays compiler-supported,
which for the columns container below it has to be, since selecting a column
before indexing it is not something a function returning a borrow can say. So
the pool is Frost storage plus Frost
logic plus a thin compiler-generated accessor. But that accessor is inline
address math, not a runtime, and the unsafe floor (raw writes into the backing
array) moves from an opaque C file into auditable, type-integrated language code.
That relocation, not elimination, of unsafety is the point.

## The columns container, the same model transposed

The slab stores whole elements back to back, an array of structs. A system that
reads one field across many elements, a physics step touching every position, a
render pass reading every transform, strides over the other fields it does not
want and pays a cache miss for each. The data-oriented answer is to store each
field in its own contiguous array, a structure of arrays, so that pass walks one
tight column. `columns<T, N>` is that transpose of the slab, and it keeps the
slab's generational safety unchanged.

`columns<T, N>` is a compiler-synthesized type. For a struct `T` with fields
`f1..fn`, it lays out one `[N]` array per field, each named after the field, plus
the same `generations` / `free_list` / `free_count` bookkeeping the slab carries:

```frost
Particle :: struct { position: Vec3, velocity: Vec3, mass: f32 }

// columns<Particle, 1024> is, in effect:
//   { position:   [1024]Vec3,
//     velocity:   [1024]Vec3,
//     mass:       [1024]f32,
//     generations:[1024]i64, free_list: [1024]i64, free_count: i64 }
```

The layout cannot be written in library Frost, because "one array per field of
`T`" is not a thing the type system can say about an arbitrary `T`. So the
compiler reflects over `T`'s fields and synthesizes it, the way it synthesizes a
generic struct instance. Naming each column after its field is what makes both
access patterns fall out of machinery that already exists:

- `c.field` is the whole column. It is ordinary field access that yields the
  `[N]t` array, and an array coerces to a slice for free, so `c.position` passed
  to a `[]Vec3` parameter is the hot-loop view with no copy and no special form.
- `c[handle].field` is the checked single element. It selects the column
  first and then indexes it at the handle's slot, the mirror image of the slab
  (which indexes storage first and then selects the field). The slot comes from
  the same `frost_rt_slot` bounds-and-generation check the slab uses, so a stale
  handle aborts here exactly as it does there.

Construction (`columns_new()`, a zeroed container), the deref `c[handle].field`
for read and write, and the element scatter `c[handle] = value` (a per-field
store at the validated slot) are the compiler-supplied pieces, for the same
reason `pool[handle]` is. They select a column before indexing, which a function
returning a second-class borrow cannot express. Everything else is an ordinary
Frost library. `std/columns.frost` provides `columns_reset`, `columns_full`,
`columns_insert`, `columns_alive`, and `columns_release`, field for field the
same as `std/slab.frost`, so moving a system from arrays-of-structs to
structure-of-arrays is changing `Slab<T, N>` to `columns<T, N>` and the `slab_`
prefix to `columns_`, and nothing else in the calling code.

The reserved element-field names are `storage`, `generations`, `free_list`, and
`free_count`, which would confuse the structural recognizer that tells a columns
container from a slab. The one operation not provided is the whole-element gather
`value := c[handle]`, because recovering a `T` from the separate columns needs
the element type the layout does not store. Read the fields you want through
`c[handle].field` instead.

It is built in both compilers (the Rust bootstrap in `src/ir_build.rs`, the
self-hosted compiler in `selfhosted/`), on both backends each, to a
byte-identical self-hosting fixpoint, because the language is defined by the two
compilers agreeing. `examples/selfhosted/soa_particles.frost` is a worked
example.
