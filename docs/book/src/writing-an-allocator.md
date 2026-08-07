# Writing an allocator

An allocator in Frost is a struct you declare and a call you write. Nothing
about it is a compiler feature: the compiler supplies three primitives
(`ptr_to`, `ptr_cast`, `slice_from`) and two forms that say where a program's
storage comes from and how long what it hands out may live. Those two are
`uses`, which lets a function declare that it allocates without taking an
allocator as a parameter, and `with`, which says which allocator a run of calls
draws from and bounds the life of everything they carve.

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
handed and one `extern` call gets it; that is the only difference, and
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

There is no per-value free, and that is the point: everything carved since a
mark goes back in one step, which is O(1) and cannot leak half of what it took.

## The carve

The one operation that turns bytes into values. It hands back a `[]T` rather
than a `^T`, so everything built on top of it is bounds-checked and the
allocator is the only place that has to be read carefully:

```frost
import "mem.frost"

Bump :: struct($N: usize) { data: [N]u8, offset: i64 }

bump_carve :: fn($T: Type, $N: usize, mut b: Bump<N>, count: i64) -> []T {
    start := (b.offset + 7) / 8 * 8
    width := count * sizeof(T)
    bytes := slice_range($u8, b.data, start, width)
    b.offset = start + width
    unsafe { slice_from($T, ptr_cast($T, ptr_to(bytes[0])), count) }
}
```

Four things are worth reading closely.

`slice_range` answers for the run against the buffer, so asking for more than is
left aborts here rather than handing back a view that reaches past the end.
Arithmetic traps on overflow, so a `count * sizeof(T)` that would wrap aborts
too.

The `unsafe` block is the reinterpret from bytes to `T`, and it is the only one
in the file. An `unsafe` block is a perimeter rather than a mode: a caller of
`bump_carve` writes none of its own, because the unchecked step is inside.

The run starts at the next multiple of 8. There is no `alignof` to ask, so an
allocator handing out types that want more alignment than that needs the number
written into it.

And `mut b` is what makes the bump visible: the offset moves in the caller's
allocator rather than in a copy.

## Declaring that a function allocates

A function that carves can take the allocator as an ordinary parameter, and for
one call that is the whole story. What `uses` adds is that a function which only
passes an allocator down no longer has to name it:

```frost,sketch
gather :: fn(world: []i64, over: i64) -> []i64 uses Bump<1024> {
    var kept := fixed_over($i64, bump_carve($i64, $1024, bump, slice_len(world)))
    for value in world {
        if (value > over) { fixed_push($i64, kept, value) }
    }
    fixed_slice($i64, kept)
}
```

`uses Bump<1024>` says so in the signature. The body reaches the allocator as
`bump`: the type's own name with the first letter lowercased. No parameter is
written for it, and no argument is passed at the call.

The `[]i64` this answers with views the allocator, and handing one back out of a
`uses` function is allowed. The allocator belongs to whoever supplied it, so the
`with` block at the caller is where that view is held to the allocator's life.

One limitation to know before reaching for `uses` in a library. The type after
it is concrete: `uses Bump<1024>` names one allocator type at one size, and
there is no way to write "whatever allocator my caller has". That is why
`std/arena.frost` takes `mut a: Arena<N>` as a parameter and leaves `uses` to
the program, which knows which allocator it built.

## Supplying it

```frost,sketch
main :: fn() -> i64 {
    var scratch: Bump<1024> = Bump { data = [0; 1024], offset = 0 }
    var round: i64 = 0
    while (round < 3) {
        mark := bump_mark($1024, scratch)
        with scratch {
            kept := gather(world, round * 3)
            print("{}\n", total_of(kept))
        }
        bump_reset($1024, scratch, mark)
        round = round + 1
    }
    0
}
```

`with scratch { ... }` says which allocator the calls inside draw from. It
allocates nothing and constructs nothing: `scratch` is built the way any other
value is, and the block only answers the question a `uses` call asks.

A call that draws an allocator with neither a `with` block around it nor a
`uses` on its caller is refused where it is written, naming both ways to fix it.

## The block is also the region

This is the part that makes the whole thing checkable rather than a convention.
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
    var scratch: Bump<1024> = Bump { data = [0; 1024], offset = 0 }
    var sink: i64 = 0
    var escaped: ^i64 = ptr_to(sink)
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
back out one field at a time is refused for the same reason. What does leave
freely is anything that carries no storage: a count, a total, a copy of an
element.

## The container over it

A growable array over a carved run is `Fixed<T>` in
[std/fixed.frost](std/containers.md). It never allocates, so it fits an
allocator with no realloc: it is handed a run and fills it, a push past the end
aborts at the index that reaches past it, and it owns nothing so there is
nothing to free. `Vec<T>` keeps the heap, and neither type carries a field
saying which allocator it came from.

```frost,sketch
with scratch {
    var visible := fixed_over($Sprite, arena_carve($Sprite, $4096, scratch, 64))
    fixed_push($Sprite, visible, sprite)
}
```

## Choosing at run time instead

Everything above decides the allocator at compile time and pays nothing for it.
Where the answer is not known until the program runs, an allocator is a value: a
function pointer and the state it works on, with no vtable and no compiler
support, because it is an ordinary struct.

```frost
Allocator :: struct {
    take: fn(^u8, i64) -> ^u8,
    state: ^u8,
}
```

`examples/native/allocator.frost` is a bump allocator behind that interface. It
costs an indirect call per allocation, so it earns its place when the backing
really is chosen at run time and not before.

## Where to go next

[Allocation sources and regions](reference/allocation-and-regions.md) is the
reference for `uses`, `with`, and exactly what the region check refuses.
[Allocation strategy](design/allocators.md) is the layer model: why the arena is
the default, where a pool fits, and why nothing is threaded through an ambient
context.
