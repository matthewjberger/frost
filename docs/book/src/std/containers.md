# Containers

Five modules: a growable vector, a hash map, a generational slab, the
structure-of-arrays version of that slab, and an optional. They share one shape.
Storage is either a heap block from [mem.frost](mem.md) or a fixed array in the
struct itself, elements are reached through a bounds-checked slice or a
generation-checked handle, and what a caller does with an element is written
where the caller is rather than handed to the container as a closure.

## `std/vec.frost`, a growable array

```frost
Vec :: linear struct($T: Type) {
    storage: []T,
    len: i64,
    cap: i64,
}
```

`Vec<T>` is `linear`, so the compiler refuses a vector that nothing frees. The
storage is a slice of the whole block rather than a raw pointer, which is why
`vec.frost` contains no `unsafe` of its own: every read and write in the file
goes through the bounds-checked path, and the allocation it stands on lives in
`mem.frost`.

| Call | What it does |
| --- | --- |
| `vec_new($T, capacity) -> Vec<T>` | An empty vector with room reserved |
| `vec_free($T, move v)` | Releases the block. Consumes the vector |
| `vec_len($T, v) -> i64` | How many elements are live |
| `vec_slice($T, v) -> []T` | The live elements, bounds-checked |
| `vec_push($T, mut v, move value)` | Appends, doubling the storage when full |
| `vec_get($T, v, index) -> T` | The element, by copy |
| `vec_set($T, mut v, index, move value)` | Overwrites the element |
| `vec_clear($T, mut v)` | Forgets the elements, keeps the storage |

`vec_slice` is the face everything above the container uses. It hands out the
live prefix rather than the block, so a vector with room for sixty-four holding
two elements gives a slice of length two. Writing through it writes into the
vector:

```frost,sketch
var v := vec_new($VecPoint, 2)
vec_push($VecPoint, v, VecPoint { x = 1, y = 2 })
vec_push($VecPoint, v, VecPoint { x = 3, y = 4 })
var held := vec_slice($VecPoint, v)
held[1].x = 30
assert(vec_get($VecPoint, v, 1).x == 30)
vec_free($VecPoint, v)
```

The slice views the storage rather than copying it, which is also why a slice
taken before a push is not valid after one: a push that fills the capacity
reallocates.

`vec_clear` sets the length to zero and leaves the block alone, so a per-frame
buffer is reused rather than reallocated. The test asserts `heap_live()` is
unchanged across the clear and the next push, which is the part that would
otherwise regress silently.

## `std/map.frost`, a hash map

Open addressing with linear probing, from keys of any type to values of any
type. Three parallel heap slices: the keys, the values, and a state byte per
slot. `Map<K, V>` is `linear`.

```frost
Slot :: enum { Empty, Used, Gone }
```

`Gone` is a tombstone, left where a key was removed. A probe run passes through
one rather than stopping at it, so keys placed after a removed key are still
found. The table doubles when it is half full, which keeps probe runs short, and
the capacity is a power of two so the starting slot is a mask rather than a
division.

| Call | What it does |
| --- | --- |
| `map_new($K, $V, capacity) -> Map<K, V>` | An empty table, at least eight slots, doubled up to the capacity asked for |
| `map_free($K, $V, move m)` | Releases all three runs. Consumes the map |
| `map_len($K, $V, m) -> i64` | How many keys are in it |
| `map_put($K, $V, $ops, mut m, key, move value)` | Inserts or overwrites, growing if half full |
| `map_get($K, $V, $ops, m, key, move fallback) -> V` | The value, or the fallback when the key is absent |
| `map_has($K, $V, $ops, m, key) -> bool` | Whether the key is there |
| `map_remove($K, $V, $ops, mut m, key) -> bool` | Removes it, leaving a tombstone. Answers whether it was there |
| `map_clear($K, $V, mut m)` | Empties every slot, keeps the storage |

`map_get` takes a fallback rather than answering an `Option`, so the common read
is one call with no match around it. `map_has` is there for the case where
absence and a zero value are different answers.

### `Hashing<K>`

What it means to hash a key and to compare two of them is a value:

```frost
Hashing :: struct($K: Type) {
    hash: fn(K) -> i64,
    equal: fn(K, K) -> bool,
}
```

`$ops` is a compile-time argument, so the hash and the comparison fold to direct
calls and the map holds no function pointer. This is the shape
[sort.md](sort.md) describes for `Ordering<T>`, with two different fields.

The library ships `i64_keys` and `text_keys`, built from `i64_hash`/`i64_same`
and `text_hash`/`text_same`. A key of any other type is a constant written where
it is needed, and the module's own tests do exactly that for a two-field struct:

```frost,sketch
Cell :: struct { x: i64, y: i64 }

cell_hash :: fn(c: Cell) -> i64 {
    (i64_hash(c.x) + (i64_hash(c.y) * 31)) & 4611686018427387903
}
cell_same :: fn(a: Cell, b: Cell) -> bool { a.x == b.x && a.y == b.y }
cell_keys :: Hashing<Cell> { hash = cell_hash, equal = cell_same }

map_put($Cell, $i64, $cell_keys, grid, Cell { x = 1, y = 2 }, 12)
```

`i64_hash` scrambles the number so keys differing only in their low bits do not
land in one run of slots. `str_hash` is a multiply-and-add over the bytes,
masked to stay positive because the slot is a remainder of it. Both are
exported, along with `str_same`, for building a `Hashing` over a type that
contains one.

### `Text`, and why it exists

A string key goes through a one-field wrapper:

```frost,sketch
Text :: struct { bytes: str }

var ages := map_new($Text, $i64, 8)
map_put($Text, $i64, $text_keys, ages, text("ada"), 36)
held := map_get($Text, $i64, $text_keys, ages, text("ada"), 0)
```

The wrapper is not about semantics. A `str` would work as a key on its own, and
`str_hash` and `str_same` are written over `str` directly. The wrapper is there
because the key run is a `[]K`, and a run of slices is a shape the self-hosted
compiler does not yet hold. Wrapping the slice in a struct puts a single
concrete layout in the array. When that gap closes,
[roadmap.md](../roadmap.md) is where it is tracked.

The hash is over the contents, so two strings that read the same are one key
however each was built, and a string key survives the table growing. Both are
tested.

## `std/slab.frost`, the generational slab

Fixed-capacity storage with handles that go stale when the slot they name is
reused. No allocation, no runtime call, generic over element type and capacity.

```frost
Slab :: struct($T: Type, $N: usize) {
    storage: [N]T,
    generations: [N]i64,
    free_list: [N]i64,
    free_count: i64,
}
```

| Call | What it does |
| --- | --- |
| `slab_reset($T, $N, mut s)` | Every slot free, every generation back to zero |
| `slab_full($T, $N, s) -> bool` | Whether there is any room left |
| `slab_insert($T, $N, mut s, move value) -> Handle<T>` | Takes a free slot. The caller checks `slab_full` first |
| `slab_alive($T, $N, s, handle) -> bool` | Whether the handle names a live slot |
| `slab_release($T, $N, mut s, handle) -> bool` | Frees the slot, bumping its generation |

A handle is `Handle<T>` to callers and an `i64` inside, with the slot index in
the low thirty-two bits and the generation above them. The two convert freely,
because a handle is an `i64` at the ABI. `slab_release` bumps the generation,
which is what makes every outstanding handle to that slot stale: a later read
through one aborts rather than seeing the next tenant.

There is no in-band "no handle" value. A handle that means nothing is exactly
what generations exist to stop being readable, so the caller asks `slab_full`
before inserting instead.

### Why `world[handle].hp = 75` works

The compiler recognizes a struct as slab-shaped when it has a `storage` array
and a parallel `generations` array, and it supplies the indexing:
`world[handle]` checks the generation, aborts on a mismatch, and otherwise names
the place in `storage`. That place-deref is the only part of this file the
compiler provides. The storage, the free list and all five operations above are
ordinary Frost you can read, copy, or replace.

It has to be compiler-supplied because "return a validated reference into
storage" cannot be written in a language where references are second-class. The
rule is section 10.2 of
[handles-and-pools.md](../reference/handles-and-pools.md).

```frost,sketch
var world : Slab<Entity, 8> = slab_new()
slab_reset($Entity, $8, world)

h := slab_insert($Entity, $8, world, Entity { hp = 100, mana = 30 })
print_int_line(world[h].hp)     // 100
world[h].hp = 75
print_int_line(world[h].hp)     // 75
slab_release($Entity, $8, world, h)
print_int_line(world[h].hp)     // aborts: the handle is stale
```

A slab is written out with its arrays and then reset, because construction
cannot run code. `examples/native/generic_pool_library.frost` uses the same
source at `Slab<Entity, 8>` and `Slab<Tile, 4>` in one program, which are two
different types with two different layouts and no shared type-erased storage.

## `std/columns.frost`, the same thing transposed

`columns<T, N>` stores each field of `T` in its own contiguous array, so a
system reading one field across many elements strides one column rather than
gathering whole structs. The handle scheme is the slab's, unchanged.

| Call | What it does |
| --- | --- |
| `columns_reset($T, $N, mut c)` | Every slot free, every generation back to zero |
| `columns_full($T, $N, c) -> bool` | Whether there is any room left |
| `columns_insert($T, $N, mut c, move value) -> Handle<T>` | Scatters the element into the columns |
| `columns_alive($T, $N, c, handle) -> bool` | Whether the handle names a live slot |
| `columns_release($T, $N, mut c, handle) -> bool` | Frees the slot, bumping its generation |

The type itself is synthesized: `columns<T, N>` is one `[N]field` array per
field of `T`, plus the same `generations`, `free_list` and `free_count`
bookkeeping a slab has. `columns_new()` builds a zeroed one. The deref
`c[handle].field` and the element scatter `c[handle] = value` are
compiler-supplied, because both select a column before indexing and that is not
writable where a struct is one value. Everything in the file is ordinary Frost
over those, mirroring
`std/slab.frost` line for line, so switching a system from `Slab<T, N>` to
`columns<T, N>` changes the container token and the `slab_` prefix and nothing
else.

```frost,sketch
Particle :: struct { x: i64, y: i64 }

var world : columns<Particle, 8> = columns_new()
columns_reset($Particle, $8, world)

a := columns_insert($Particle, $8, world, Particle { x = 10, y = 1 })
columns_insert($Particle, $8, world, Particle { x = 20, y = 2 })

print_int_line(world[a].x + world[a].y)   // one element, generation-checked
print_int_line(sum_x(world.x))            // the whole x column, as a []i64
```

`world.x` is the column, handed out as a slice, which is what a data-oriented
inner loop wants. `examples/selfhosted/soa_particles.frost` is that program in
full, and it compiles under the self-hosted compiler on both backends.

[pools-and-columns.md](../design/pools-and-columns.md) is why the container is
shaped this way.

## `std/option.frost`, a value that may be absent

```frost
Option :: enum($T: Type) { None, Some { value: T } }
```

An ordinary generic enum, written in Frost like the rest of the library.
There is no null and no built-in optional type. The language grew generic enums
so that this could be written in Frost rather than built into the compiler, and
the whole module is the declaration above plus four functions over it.

| Call | What it does |
| --- | --- |
| `option_some($T, move value) -> Option<T>` | The present case |
| `option_none($T) -> Option<T>` | The absent case |
| `option_is_some($T, o) -> bool` | Whether there is a value |
| `option_unwrap_or($T, o, move fallback) -> T` | The value, or the fallback |

A `match` has to cover every variant, so forgetting to handle the absent case is
a compile error rather than a crash:

```frost,sketch
match found {
    case .None: 0
    case .Some { value }: value * 2
}
```

`option_unwrap_or` takes its fallback by value rather than lazily. There are no
closures to defer it with, and taking a function instead would hide the cost of
computing a fallback that is usually not used.

Nothing else in `std/` returns an `Option`. `map_get` takes a fallback,
`str_index_of` answers -1, and `json_member` answers -1, each because the caller
almost always tests the result immediately and a match at every call site would
buy nothing.
