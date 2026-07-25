# Pools, the C runtime, and the road to a native memory model

Frost's thesis is that pools and generational handles *are* the memory model. So
a fair question is whether the pool primitive living in the C runtime
(`runtime/frost_runtime.c`, reached through the compiler's `Pool<T>` type) holds
the language back. This note records what a spike found and what the engine
should do long-term.

## Done, and what it took

The pool has left C. `runtime/frost_runtime.c` no longer defines
`pool_new`, `pool_alloc`, `pool_get`, `pool_contains`, `pool_free`,
`pool_destroy`, `handle_index` or `handle_generation`, and the compiler no
longer emits an implicit call to `pool_get` when a `Handle` indexes something
that is not slab-shaped. A slab is a Frost struct with Frost operations over it,
generic over element type and capacity, `examples/native/lib/slab.frost`. Every
example and every test uses that, and the runtime shrank by about a hundred
lines to bounds and generation aborts, assertions, and IO helpers.

The reasoning below is kept because it is the argument that decided the shape,
and because the distinction underlying it is the reusable part.

## The question it answered

`Pool<T>` used to be a compiler type that lowered to an opaque `^u8` and called
a C runtime. The runtime was tiny and portable, and libc is the most portable
ABI there is, so functionally it cost nothing. The concern was strategic. The
flagship data-oriented primitive was uninspectable, unwritable C, which meant the
language could not demonstrate its own core idea in itself, could not reach
freestanding targets without libc, and defined the pool's behavior outside its
own safety story.

The distinction is that the dependency worth shedding was the *allocator*,
not the *C*. Look at what the pool runtime actually used from C: `malloc`,
`free`, `memcpy`, and pointer arithmetic. The generational free list is pure
integer and pointer manipulation the language can already do.
Only the allocation was irreducible, and pools are fixed-capacity by design (so
their element pointers stay stable), which means a pool does not even need a heap
allocation. It can live in a caller-provided buffer, or, as it turned out,
inside the struct.

## The spike, a generational pool in pure Frost

`examples/native/native_pool.frost` is a full generational pool written entirely
in Frost, with no runtime support for the pool itself. The storage, the free
list, the generation counters, the packed `(generation << 32) | index` handles,
and the stale-handle check are all ordinary Frost code over a fixed-size array
inside a struct:

```
Slab :: struct {
    storage: [4]Entity,
    generations: [4]i64,
    free_list: [4]i64,
    free_count: i64,
}
```

It compiles and runs identically through both backends, and it reproduces
generational safety end to end. After a slot is released and reused, the old
handle reads as dead. The whole data-oriented model is expressible in the
language today with zero runtime.

## What the spike proved

The DOD memory model is native-expressible. Everything the pool logic needs,
arrays inside structs, bounds-checked element read and write, aggregate element
assignment, and integer packing, already works. This is strong evidence that
Frost can own its memory model rather than borrow it from the C runtime.

## What the spike surfaced

Writing the pool in Frost ran straight into the ways the C-backed `Pool<T>`
currently pollutes the language:

1. `Pool` is a reserved type name. Naming a struct `Pool` fails with "Expected
   '<' after 'Pool'", because the compiler reserved `Pool` for `Pool<T>`. The
   spike had to call its struct `Slab`.
2. `pool_*` and `handle_*` are reserved function names. Naming a function
   `pool_alloc`, `pool_get`, `pool_free`, or `handle_index` collides at link time
   with the always-linked C runtime ("multiple definition of `pool_alloc`"). The
   spike had to prefix its helpers.

Both are symptoms of the same thing. The compiler-plus-C-runtime pool claims
global names that user code cannot use. A Frost-native pool has neither problem.

Three ergonomic gaps also showed up:

3. No place-deref as a library. `pool[handle].field = x`, an in-place,
   generation-checked place, cannot be a library function, because a function
   cannot return a reference into the storage (references are second-class). The
   spike mutates through read-modify-write instead. This is the one slice that
   genuinely needs compiler help.
4. No value generics. The slab is hard-coded to capacity `4`. A general
   `Slab<T, N>` needs `$N` as a *value* type parameter, which Frost does not have
   yet (it has `$T` for types only).
5. Verbose construction. Building the struct means writing out the full
   `storage`, `generations`, and `free_list` array literals. A zeroed or default
   aggregate construction would remove the boilerplate.

## What the engine should do long-term

The pool should be a Frost-native aggregate with a thin compiler-supported
accessor, not a C runtime. Concretely:

- Storage is a Frost aggregate. `Pool<T, N>` becomes a generic struct
  (`storage: [N]T`, `generations: [N]u32`, `free_list: [N]u32`, `free_count`),
  pure Frost, fixed capacity, no allocation.
- Bookkeeping is Frost code. `pool_new`, `pool_alloc`, `pool_free`, and
  `pool_contains` become ordinary free functions over that struct. The spike
  already proved these work.
- `pool[handle]` stays compiler-supported, but calls nothing. Instead of the
  C `pool_get`, the compiler generates inline address arithmetic into the
  struct's `storage` field plus a generation check against `generations[index]`,
  aborting on mismatch the same way bounds checking does. This is the same class
  of codegen the compiler already does for arrays, with no runtime call.
- Dynamic pools are an opt-in at the edge. A heap-backed pool is a pool over
  a caller-provided buffer or one `extern` allocation. Fixed-capacity pools, the
  common case, need nothing external. The allocator layer under this, an arena as
  the primary allocator with the pool as its fixed-size specialization, is in
  `docs/allocators.md`.
- The C runtime shrinks. What stays in C is `printf`-style FFI, the string
  and emit helpers the self-hosted compiler uses, `frost_assert`, and
  `frost_bounds_check`. The pool functions and the generational logic leave C,
  and `Pool` / `pool_*` stop being reserved names.

This is the right boundary because it dogfoods the model (the pool is Frost code),
keeps the ergonomic `pool[handle].field = x` (the one thing that needs the
compiler), removes the libc floor for the memory model so fixed pools reach
freestanding targets, and folds the generational check into the compiler's own
abort discipline rather than a foreign definition.

### Roadmap

1. Value generics (`$N` as a value parameter). *(Done.)* A struct takes
   `$N: usize`, sizes `[N]T` with it, and instantiates concretely
   (`Slab<Entity, 4>`). See `examples/native/generic_slab.frost`, a generational
   pool generic over both element type and capacity.
2. Compiler place-deref over a Frost aggregate. *(Done.)* A struct with a
   `storage` array and a parallel `generations` array is recognized as
   slab-shaped, and `p[handle]` compiles to inline index and generation
   arithmetic against those fields rather than a call.
3. Operations generic over the capacity. *(Done.)* A function takes `$N:
   usize` the way a struct already could, so `slab_reset`, `slab_insert` and the
   rest are written once rather than once per size, and the name stands for the
   integer in the body as well as in a type. That is what turned the example
   into a library: `examples/native/lib/slab.frost`.

   The item as originally written asked for zeroed or default construction to
   remove the array-literal boilerplate. The repeat literal already does that
   (`storage = [Node { value = 0, next = 0 }; 16]`), and an implicit zero would
   contradict the rule that a literal writes every field.
4. Remove the compiler-special pool surface, freeing the `Pool` type name and
   the `pool_*` function names for user code. *(Done.)*
5. Take the pool out of the C runtime. *(Done.)* Every example and test uses
   the Frost slab, the runtime no longer defines any `pool_*` function, and
   indexing by a handle requires a slab-shaped struct rather than falling back
   to an implicit `pool_get`. The runtime that is always linked is now bounds
   and generation aborts, assertions, and the IO helpers the self-hosted compiler
   uses, with one `malloc` left in `frost_read_file`.

This roadmap is a slice of the larger allocator plan in `docs/allocators.md`,
which puts the pool on top of an arena and pushes the one remaining OS call to
the edge. The merged ordering there starts with slices and value generics, the
same two features this roadmap needs.

### The honest caveat

The pool is never a pure library. The place-deref must stay compiler-supported,
because "return a validated reference into storage" cannot be expressed when
references are second-class. So the long-term pool is Frost storage plus Frost
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

```
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
  the same `frost_slot` bounds-and-generation check the slab uses, so a stale
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
