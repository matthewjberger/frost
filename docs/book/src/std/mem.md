# Typed allocation

`std/mem.frost` is the floor every other container stands on. It gets a block of
bytes from the C allocator and hands it back as a typed pointer or as a slice.
Nothing else in `std/` calls the allocator directly.

Allocating takes two operations together: calling C for the bytes, and
reinterpreting those bytes as a typed pointer. Each function here does both
inside one `unsafe` block and does the count-times-size arithmetic once, so a
caller writes `keys := heap_array($i64, cap)` with no `unsafe` of its own. An
unsafe block is a perimeter: calling a function that contains one does not
require one.

`heap_slice` is the form to reach for. A slice carries its length, so every
later access through it is bounds-checked and the container above it is ordinary
safe code. `std/vec.frost` and `std/map.frost` both hold slices, and neither
file contains an `unsafe` block.

## What each call gives you

`heap_array` and `heap_slice` give a block of at least `count * sizeof(T)`
bytes, aligned for `T`, which follows from `malloc`. A count of zero allocates
room for one element, so a container always holds a pointer to storage and a
zero-capacity vector has somewhere to put its first push.

`heap_grow` and `heap_grow_slice` take the old block by `move`, because
`realloc` may return a different address and the old pointer is dead afterwards.
Reading the old block after the call is a compile error.

`bytes_as` takes a run of bytes and the element count the caller knows, and
answers a bounds-checked `[]T`. The ECS's columns, whose element width is
decided while the program runs, come back into the typed world through it.

`slice_prefix` cuts a container's live length out of its storage. The storage
slice is as long as the capacity and the prefix is as long as the count, so
`vec_slice` over a vector holding two elements in a block of sixty-four hands
out two.

## What the module refuses

A count is the caller's word for how many elements are there. Three parts of
that word are refused outright.

A negative length. Every access through a slice is bounds-checked, and the check
compares unsigned so one comparison answers for a negative index as well as for
one past the end. That same cast reads a negative length as enormous, which
would leave a slice built with one unchecked at every access, so the length is
answered for at the one place a slice is built.

A view longer than the run it came from. `slice_prefix` and `slice_range` cut
from a slice, so the length of the run is known and both ends are checked
against it. `slice_span` and `slice_chunk` clamp before they cut.

A size that wraps. `count * sizeof(T)` at a large count wraps to a small number,
so the allocator would hand back a small block while the slice over it carried
the count that was asked for, and every read past the block's real end would be
checked against the wrong number and pass.

An allocation that fails aborts. These calls have no way to say they ran out of
memory, and every caller wraps what comes back in a slice without looking.

One part stays the caller's word: that `count` elements really do live at the
pointer.

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
a test sees a leak at the moment it happens.

Every allocating module in `std/` has a test of this shape:

```frost
test "a vector gives back every block it took, however far it grew" {
    before := heap_live()
    var v := vec_new($i64, 1)
    var i : i64 = 0
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

A grow leaves the count where it was. The `mem.frost` tests check that directly:

```frost
test "what a block took is what releasing it gives back" {
    before := heap_live()
    var held := heap_slice($i64, 16)
    assert(heap_live() == before + 1)
    held[0] = 1
    var bigger := heap_grow_slice($i64, held, 64)
    assert(heap_live() == before + 1)
    heap_release_slice($i64, bigger)
    assert(heap_live() == before)
}
```

`before` is read at the start, so the test says what it means whatever else the
program has already allocated.

## `std/arena.frost`, the other allocator

An arena is the second way a program gets storage, and the one a scratch region
uses. It holds its own bytes and an offset:

```frost
Arena :: struct($N: usize) {
    data: [N]u8,
    offset: i64,
}
```

`Arena<4096>` is 4096 bytes and an offset, and nothing under it allocates. A
program builds one where it wants the storage to live and hands it to the calls
that draw from it.

| Call | What it does |
| --- | --- |
| `arena_carve($T, $N, mut a, count) -> []T` | A run of `count` elements, taken from the front of what is left |
| `arena_mark($N, a) -> i64` | Where the arena is now, to roll back to |
| `arena_reset($N, mut a, mark)` | Everything carved since the mark, reclaimed |
| `arena_used($N, a) -> i64` | How many bytes are out |

`arena_carve` hands back a `[]T`, so everything built on it is bounds-checked,
and the one `unsafe` block in the file is the reinterpret from bytes to `T`. A
run starts at the next multiple of 8, the alignment of every type laid out
without `align(N)` written on it. There is no `alignof` to ask, so a type
wanting more than that is a gap this does not fill.

Freeing is by the block: `arena_reset` puts the offset back and the next carve
takes the same bytes. That is the whole lifetime story. The container over a
carved run ([fixed.frost](containers.md)) owns nothing and frees nothing. What
stops a run outliving the arena is the region check, in
[allocation-and-regions.md](../reference/allocation-and-regions.md).

## Tests

```bash
frost --test std/mem.frost
```

Thirteen blocks, covering the slice views, the typed path, the byte-width path,
the zero count, a struct element (where an element index and a byte offset are
different numbers), and the two block-counting tests.
