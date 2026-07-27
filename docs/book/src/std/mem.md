# Typed allocation

`std/mem.frost` is the floor every other container stands on. It gets a block of
bytes from the C allocator and hands it back as a typed pointer or, better, as a
slice. Nothing else in `std/` calls the allocator directly.

Two operations have to happen together to allocate anything: calling C for the
bytes, and reinterpreting those bytes as a typed pointer. Written at each site
that is three `unsafe` blocks per container plus the count-times-size arithmetic
repeated wherever it is easiest to get wrong. Written once here it is one block
per function, and a caller reads `keys := heap_array($i64, cap)` with no
`unsafe` of its own, because an unsafe block is a perimeter: calling a function
that contains one does not require one.

`heap_slice` is the stronger of the two forms and the one to reach for. A slice
carries its length, so every later access through it is bounds-checked and the
container above it is ordinary safe code. `std/vec.frost` and `std/map.frost`
both hold slices rather than raw pointers, and neither file contains an `unsafe`
block.

## What it vouches for

Each function here is a claim the module is making on the caller's behalf, and
the claims are small enough to check by reading the file.

`heap_array` and `heap_slice` promise that the block is at least `count *
sizeof(T)` bytes and correctly aligned for `T`, which follows from `malloc`.
A count of zero still allocates room for one element, so a container never holds
a pointer to nothing and a zero-capacity vector still has somewhere to put its
first push.

`heap_grow` and `heap_grow_slice` take the old block by `move`, because
`realloc` may return a different address and the old pointer is not valid
afterwards. The move rule turns "do not use the old pointer" from a comment into
a compile error.

`bytes_as` is the end of an erasure. A run of bytes does not know how many
elements it holds, so the caller says, and what comes back is a bounds-checked
`[]T` from there on. This is how the ECS's columns, whose element width is
decided while the program runs, get back into the typed world.

`slice_prefix` is what makes a container's live length visible: the storage
slice is as long as the capacity, and the prefix is as long as the count, so
`vec_slice` hands out two elements out of sixty-four rather than sixty-four.

## The calls

| Call | What it does |
| --- | --- |
| `heap_array($T, count) -> ^T` | Room for `count` elements of `T` |
| `heap_slice($T, count) -> []T` | The same block as a bounds-checked `[]T` |
| `heap_grow($T, move block, count) -> ^T` | Resize, answering where it now is |
| `heap_grow_slice($T, move held, count) -> []T` | The same for a slice |
| `heap_release($T, move block)` | Give the block back |
| `heap_release_slice($T, move held)` | Give a slice's block back |
| `slice_prefix($T, held, count) -> []T` | The first `count` elements |
| `heap_bytes(size) -> ^u8` | A run of bytes with a width decided at runtime |
| `heap_grow_bytes(move block, size) -> ^u8` | Resize such a run |
| `bytes_at(block, offset) -> ^u8` | One position in a run of bytes |
| `bytes_as($T, block, count) -> []T` | A run of bytes read as `count` of `T` |
| `heap_zero(destination, size)` | Fill with zero |
| `heap_copy(destination, source, size)` | Copy `size` bytes |
| `heap_live() -> i64` | How many blocks are currently out |

## Counting blocks

`heap_live` answers how many blocks the runtime has handed out and not taken
back. A container that frees what it took leaves the count where it found it, so
a test can say a leak happened at the moment it happens rather than waiting for
a long-running program to notice.

Every allocating module in `std/` has a test of this shape, and it is the
cheapest test in the library to write:

```frost
test "a vector gives back every block it took, however far it grew" {
    before := heap_live()
    mut v := vec_new($i64, 1)
    mut i : i64 = 0
    while (i < 500) {
        vec_push($i64, v, i)
        i = i + 1
    }
    vec_free($i64, v)
    assert(heap_live() == before)
}
```

The five hundred pushes take the vector through nine reallocations, and the
assertion holds only if each `heap_grow_slice` released what it replaced. The
same test over `std/map.frost` covers the three parallel runs a map grows, where
freeing two of the three would pass every functional test in the file.

A grow does not change the count, which is what the `mem.frost` tests check
directly:

```frost
test "what a block took is what releasing it gives back" {
    before := heap_live()
    mut held := heap_slice($i64, 16)
    assert(heap_live() == before + 1)
    held[0] = 1
    mut bigger := heap_grow_slice($i64, held, 64)
    assert(heap_live() == before + 1)
    heap_release_slice($i64, bigger)
    assert(heap_live() == before)
}
```

`before` is read rather than assumed to be zero, so the test says what it means
whatever else the program has already allocated.

## Tests

```bash
frost --test std/mem.frost
```

Nine blocks, covering the typed path, the byte-width path, the zero count, a
struct element (where an element index and a byte offset are different numbers),
and the two block-counting tests above.
