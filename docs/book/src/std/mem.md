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
| `as_bytes($T, held) -> []u8` | A run of `T` read as the bytes it occupies |
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
uses. It is a view of bytes somebody else owns, and an offset into them:

```frost
Arena :: struct {
    data: []u8,
    offset: i64,
}
```

A program builds the backing where it wants the storage to live and hands the
view over, so the arena itself allocates nothing:

```frost,sketch
var backing: [4096]u8 = [0; 4096]
var scratch := arena_over(backing)
```

| Call | What it does |
| --- | --- |
| `arena_over(backing) -> Arena` | An arena over a run of bytes |
| `arena_carve($T, mut a, count) -> []T` | A run of `count` elements, taken from the front of what is left |
| `arena_mark(a) -> i64` | Where the arena is now, to roll back to |
| `arena_reset(mut a, mark)` | Everything carved since the mark, reclaimed |
| `arena_used(a) -> i64` | How many bytes are out |
| `arena_left(a) -> i64` | How many are still there |
| `arena_take(mut a, size, align) -> []u8` | The byte-level carve the typed one is built on |
| `arena_resize(mut a, block, size, align) -> []u8` | A bigger run holding what the old one held |
| `arena_give(mut a, block)` | Nothing; an arena reclaims by reset |

`arena_carve` hands back a `[]T`, so everything built on it is bounds-checked,
and there is no `unsafe` block in the file at all: `arena_take` answers for the
run against the backing and `bytes_as` holds the one reinterpret. A run starts
on `alignof(T)`, so a type wanting more than a word gets it.

Freeing is by the block: `arena_reset` puts the offset back and the next carve
takes the same bytes. That is the whole lifetime story. The container over a
carved run ([fixed.frost](containers.md)) owns nothing and frees nothing. What
stops a run outliving the arena is the region check, in
[allocation-and-regions.md](../reference/allocation-and-regions.md).

## `std/allocation.frost`, either one

A function that should work against whichever source it is given takes the
source as a compile-time argument. `Allocation<A>` is the capability bundle for
that: `take`, `resize` and `give`, all over `[]u8`.

| Call | What it does |
| --- | --- |
| `carve($T, $A, $source, mut a, count) -> []T` | Room for `count` elements of `T` |
| `carve_grow($T, $A, $source, mut a, held, count) -> []T` | A bigger run holding what the old one held |
| `carve_give($T, $A, $source, mut a, held)` | The run handed back |

Two sources ship with it. `heap_source` over `Heap`, whose state counts the
blocks that are out, and `arena_source` over `Arena`, which is the three arena
calls above under the bundle's names.

```frost
import "allocation.frost"

main :: fn() -> i64 {
    var h := heap_state()
    var run := carve($i64, $Heap, $heap_source, h, 4)
    run[0] = 11
    carve_give($i64, $Heap, $heap_source, h, run)
    h.taken
}
```

The bundle is a compile-time argument, so a call through one of its fields is a
direct call to the function that field names: nothing is loaded and nothing is
dispatched. What a run carved this way may outlive is the same question the
region and frame checks ask of a call to `arena_carve` directly, and they answer
it the same way, because a call whose body they cannot see is worth the
shortest-lived argument that could have reached its answer.

## Tests

```bash
frost --test std/mem.frost
```

Thirteen blocks, covering the slice views, the typed path, the byte-width path,
the zero count, a struct element (where an element index and a byte offset are
different numbers), and the two block-counting tests.
