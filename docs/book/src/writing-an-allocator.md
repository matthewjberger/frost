# Writing an allocator

An allocator in Frost is a struct you declare and a call you write. The compiler
supplies three primitives (`ptr_to`, `ptr_cast`, `slice_from`) and two forms
that say where a program's storage comes from and how long what it hands out may
live. Those two are `uses`, which lets a function declare that it allocates
without taking an allocator as a parameter, and `with`, which says which
allocator a run of calls draws from and bounds the life of everything they
carve.

`examples/native/custom_allocator.frost` is the whole of what follows as one
program that builds and runs. `std/arena.frost` is the same allocator written
once for everyone, and `std/fixed.frost` is the container over what it hands
out.

## The allocator

Backing bytes and how far into them the next value goes:

```frost
Bump :: struct($N: usize) {
    data: [N]u8,
    offset: i64,
}
```

`Bump<1024>` is 1024 bytes plus a number, all inside the struct, so a program
that builds one on the stack allocates nothing at run time. Where the backing
has to come from the operating system instead, the struct holds a `[]u8` it was
handed and one `extern` call gets it. That is the only difference, and
`examples/native/dynamic_arena.frost` is that version.

Two operations take storage back. A mark is where the allocator is now, and a
reset winds it back to one:

```frost
Bump :: struct($N: usize) { data: [N]u8, offset: i64 }

bump_mark :: fn($N: usize, b: Bump<N>) -> i64 {
    b.offset
}

bump_reset :: fn($N: usize, mut b: Bump<N>, mark: i64) {
    assert(mark >= 0 && mark <= b.offset)
    b.offset = mark
}
```

There is no per-value free. One reset gives back everything carved since the
mark, in O(1).

## The carve

The one operation that turns bytes into values. It hands back a `[]T`, so
everything built on top of it is bounds-checked and the allocator is the only
place that has to be read carefully:

```frost
import "mem.frost"

Bump :: struct($N: usize) { data: [N]u8, offset: i64 }

bump_carve :: fn($T: Type, $N: usize, mut b: Bump<N>, count: i64) -> []T {
    align := alignof(T)
    start := (b.offset + align - 1) / align * align
    width := count * sizeof(T)
    bytes := slice_range(b.data, start, width)
    b.offset = start + width
    unsafe { slice_from($T, ptr_cast($T, ptr_to(bytes[0])), count) }
}
```

`slice_range` answers for the run against the buffer, so asking for more than is
left aborts here instead of handing back a view that reaches past the end.
Arithmetic traps on overflow, so a `count * sizeof(T)` that would wrap aborts
too.

The `unsafe` block is the reinterpret from bytes to `T`, and it is the only one
in the file. A caller of `bump_carve` writes none of its own, because the
unchecked step is inside.

The run starts on `alignof(T)`, which is what the compiler laid `T` out to, so
a type carrying `align(16)` is carved onto sixteen without the allocator being
told about it.

`mut b` moves the offset in the caller's allocator, so the bump is still there
after the call returns.

## Declaring that a function allocates

A function that carves can take the allocator as an ordinary parameter, and for
one call that is the whole story. `uses` lets a function that only passes an
allocator down leave it out of the parameter list:

```frost,sketch
gather :: fn(world: []i64, over: i64) -> []i64 uses Bump<1024> {
    mut kept := fixed_over(bump_carve($i64, bump, slice_len(world)))
    for value in world {
        if (value > over) { fixed_push(kept, value) }
    }
    fixed_slice(kept)
}
```

`uses Bump<1024>` in the signature says the function allocates from one. The
body reaches the allocator as `bump`: the type's own name with the first letter
lowercased. No parameter is written for it, and no argument is passed at the
call.

The `[]i64` this answers with views the allocator, and handing one back out of a
`uses` function is allowed. The allocator belongs to whoever supplied it, so the
`with` block at the caller holds that view to the allocator's life.

One limitation to know before reaching for `uses` in a library. The type after
it is concrete. `uses Bump<1024>` names one allocator type at one size, and
there is no way to write "whatever allocator my caller has". So
`std/arena.frost` takes `mut a: Arena<N>` as a parameter and leaves `uses` to
the program, which knows which allocator it built.

## Supplying it

```frost,sketch
main :: fn() -> i64 {
    mut scratch: Bump<1024> = Bump { data = [0; 1024], offset = 0 }
    mut round: i64 = 0
    while (round < 3) {
        mark := bump_mark(scratch)
        with scratch {
            kept := gather(world, round * 3)
            print("{}\n", total_of(kept))
        }
        bump_reset(scratch, mark)
        round = round + 1
    }
    0
}
```

`with scratch { ... }` says which allocator the calls inside draw from. The
block allocates nothing and constructs nothing. `scratch` is built the way any
other value is, and the block answers the question a `uses` call asks.

A call that draws an allocator with neither a `with` block around it nor a
`uses` on its caller is refused where it is written, naming both ways to fix it.

## The block is also the region

The `with` block bounds how long anything carved inside it may live. A view of
the allocator may not be stored in a binding declared outside the block,
returned, or handed back as the block's value:

```frost,refused
Bump :: struct($N: usize) { data: [N]u8, offset: i64 }

carve_one :: fn($N: usize, mut b: Bump<N>) -> ^i64 {
    slot := ptr_to(b.data[b.offset])
    b.offset = b.offset + 8
    unsafe { ptr_cast($i64, slot) }
}

main :: fn() -> i64 {
    mut scratch: Bump<1024> = Bump { data = [0; 1024], offset = 0 }
    mut sink: i64 = 0
    mut escaped: ^i64 = ptr_to(sink)
    with scratch {
        escaped = carve_one($1024, scratch)
    }
    unsafe { escaped^ }
}
```

```
region: a pointer into arena 'scratch' escapes its region by being stored
outside it; it may not outlive the arena
```

The same is true one field down. A container built inside the block holds a view
of the allocator, so the container may not leave either, and reading its storage
back out one field at a time is refused for the same reason. Anything that
carries no storage leaves freely: a count, a total, a copy of an element.

## The container over it

A growable array over a carved run is `Fixed<T>` in
[std/fixed.frost](std/containers.md). It never allocates, so it fits an
allocator with no realloc. It is handed a run and fills it, a push past the end
aborts at the index that reaches past it, and it owns nothing, so there is
nothing to free. `Vec<T>` keeps the heap, and neither type carries a field
saying which allocator it came from.

```frost,sketch
with scratch {
    mut visible := fixed_over(arena_carve($Sprite, scratch, 64))
    fixed_push(visible, sprite)
}
```

A function that should work against whichever source it is handed takes the
source as a compile-time argument instead of naming one. `Allocation<A>` in
[std/allocation.frost](std/mem.md) is the bundle for that, and `arena_source`
puts an arena behind it, so the same body draws from the arena here and from the
heap elsewhere at no cost either way.

```frost,sketch
with scratch {
    mut visible := carve($Sprite, $arena_source, scratch, 64)
}
```

## Choosing per call site instead

Everything above binds the allocator to a `uses` declaration. Where a caller has
to vary it, the function that takes memory is named at the call and the state
travels beside it. There is no vtable and no compiler support behind it: a call
names the function it goes to, so the name is a compile-time argument.

```frost,sketch
alloc :: fn($S: Type, $take: fn(mut S, i64) -> ^u8, mut state: S, size: i64)
    -> ^u8 {
    take(state, size)
}
```

`examples/native/allocator.frost` is a bump allocator behind that interface. The
function is named at the call, so each allocation is a direct call and the state
is what varies.

## Where to go next

[Allocation sources and regions](reference/allocation-and-regions.md) is the
reference for `uses`, `with`, and exactly what the region check refuses.
[Allocation strategy](design/allocators.md) is the layer model: why the arena is
the default, where a pool fits, and why nothing is threaded through an ambient
context.
