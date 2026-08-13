# 8a. Allocation sources and regions

An allocator in Frost is an ordinary struct a program declares, and an
allocation is an ordinary call (see
[allocators.md](../design/allocators.md)). Two forms carry an allocator through
a program without a parameter on every signature, and a check keeps what it
hands out inside the allocator's lifetime.

`uses A` on a function says the function draws an allocation capability of type
`A`. `with a { ... }` says which one, for every call inside the block. The
block is also the region: a raw pointer into `a` may not leave it.

## 8a.1 Declaring a capability

`uses` follows the return type in a signature (13.6), and takes one or more
types:

```frost
Arena :: struct($N: usize) { data: [N]u8, offset: i64 }

alloc_int :: fn($N: usize, mut a: Arena<N>) -> ^i64 {
    slot := ptr_to(a.data[a.offset])
    a.offset = a.offset + 8
    unsafe { ptr_cast($i64, slot) }
}

make_two :: fn() -> i64 uses Arena<256> {
    p := alloc_int(arena)
    unsafe { p^ = 10 }
    q := alloc_int(arena)
    unsafe { q^ = 32 }
    unsafe { p^ + q^ }
}
```

`Arena` is the struct the two lines above declare, and `uses` takes a `Pool`, a
`Scratch` or a `Bump` the same way.

The return type is optional. `fn() uses Arena<256> { ... }` is a signature, and
so is `fn() -> i64 uses Arena<256>, Scratch<64>`.

## 8a.2 The name the body reaches it by

Each capability is an implicit parameter, appended to the function's own
parameters in the order the `uses` list wrote them, taken by write borrow so a
callee that bumps an offset is bumping the caller's arena.

The body reaches it by the capability type's own name with the first letter
lowercased, ignoring any generic arguments. `Arena<256>` binds `arena`,
`Scratch<64>` binds `scratch`, `Bump` binds `bump`. `make_two` above writes
`alloc_int(arena)` with no parameter called `arena` anywhere in its signature.

A function drawing two sources tells them apart by those names:

```frost,sketch
Scratch :: struct($N: usize) { data: [N]u8, offset: i64 }

take_arena :: fn(mut a: Arena<256>) -> i64 {
    a.offset = a.offset + 8
    a.offset
}

take_scratch :: fn(mut s: Scratch<64>) -> i64 {
    s.offset = s.offset + 1
    s.offset
}

both :: fn() -> i64 uses Arena<256>, Scratch<64> {
    take_arena(arena) + take_scratch(scratch)
}
```

## 8a.3 What a call supplies

A call to a `uses` function takes one extra argument per capability the callee
draws, and the compiler supplies each. It may draw from:

- the capability parameters the enclosing function itself holds, if it is a
  `uses` function;
- the arenas named by the `with` blocks around the call, the innermost winning
  where two carry the same name.

Each capability is matched by the name of 8a.2. A callee drawing
`Arena<256>, Scratch<64>` from inside a function that holds both finds `arena`
and `scratch` by name and forwards them, whichever order the caller declared
them in.

A callee drawing exactly one takes the innermost source, whatever it is named,
so a `with scratch` block supplies a `uses Arena<256>` call. A callee drawing
several is matched by name alone.

A call with nothing to draw from is refused, naming the two ways to fix it:

```
calling 'make' needs an allocation capability; declare `uses Arena<256>` on
the caller or wrap the call in a `with` block
```

A call with sources in scope but not the one it wanted is refused with the
wanted name, its type, and the list of what is in scope there instead.

A function literal written inside another function's body draws nothing. The
enclosing function's capabilities lie outside it.

## 8a.4 `with` provides one, and is a region

```
"with" IDENT Block
```

The identifier names a variable the enclosing scope already owns and can write.
The arena is built the way any other value is, and the block says which calls
inside it are supplied from it.

```frost,sketch
main :: fn() -> i64 {
    mut arena : Arena<256> = Arena { data = [0; 256], offset = 0 }
    mut result : i64 = 0
    with arena {
        result = forward()
    }
    print("{}\n", result)
    0
}
```

Blocks nest, and a call inside both is supplied by the inner one:

```frost,sketch
with arena {
    with scratch {
        result = forwards()        // draws both, by name
        result = result + only_one()   // draws one, takes scratch
    }
}
```

The block bounds how long anything carved out of the arena may live. That bound
is the region.

## 8a.5 What the region check refuses

A view carved out of an arena is region-bound, a `[]T` and a `str` as much as a
`^T`. The check (`src/check/regions.rs`) is a flow walk over the body. Frost
has no global arenas and no closures, so a `^T` points only into an arena the
function was handed directly, and a walk can follow that provenance.

Two regions exist:

- the body of a `with a { ... }` block, whose arena is `a`;
- the body of a `uses A` function, whose arena is the implicit capability.

A value is a pointer into the region when it is `ptr_to(...)`, `slice_from(...)`
or `ptr_cast(...)` over the arena, or over something already bound to it; a
binding already holding one; a call to a function that either draws this arena
or is passed it and answers with something holding a view, whether that is a
bare `^T`, a `[]T`, a `str`, or a struct with one of those inside; a struct, an
enum variant or an array literal any of whose values is one; whichever of those
an `if`, a `match` or an `unsafe` block ends with; or a read back through a
pointer to one.

The last two cover a container built in an arena. A `Vec`-shaped value built in
the region carries the arena's storage under a field name, so the literal that
builds it is a region pointer and so is the whole binding it lands in. Reading
a field back out is one too, where what is read can carry storage: `held.view`
off a bound binding is the arena's run, and `held.count` beside it is a number
and leaves freely. A read whose type the walk cannot name is refused.

Inside a `with` block, such a value escapes three ways, and each is an error:
being returned; being assigned to a place rooted outside the region; and being
the block's trailing expression, which flows to the enclosing scope.

```frost,refused
Arena :: struct($N: usize) { data: [N]u8, offset: i64 }

alloc_int :: fn($N: usize, mut a: Arena<N>) -> ^i64 {
    slot := ptr_to(a.data[a.offset])
    a.offset = a.offset + 8
    unsafe { ptr_cast($i64, slot) }
}

main :: fn() -> i64 {
    mut arena: Arena<256> = Arena { data = [0; 256], offset = 0 }
    mut escaped: ^i64 = ptr_to(arena.offset)
    with arena {
        escaped = alloc_int(arena) // error: escapes its region
    }
    unsafe { escaped^ }
}
```

Storing one in a binding declared inside the region is fine. That binding dies
with the region, and the check follows it: anything later read out of it is a
region pointer too.

Inside a `uses` function the rule differs in one place. Returning an arena
pointer is allowed: that pointer's region is the caller's `with` block, and the
check runs there. Writing one into a parameter is refused, since the parameter
belongs to a frame that outlives this one:

```frost,refused
Arena :: struct($N: usize) { data: [N]u8, offset: i64 }
Reg :: struct { ptr: ^i64 }

alloc_int :: fn($N: usize, mut a: Arena<N>) -> ^i64 {
    slot := ptr_to(a.data[a.offset])
    a.offset = a.offset + 8
    unsafe { ptr_cast($i64, slot) }
}

stash :: fn(mut r: Reg) -> i64 uses Arena<256> {
    r.ptr = alloc_int(arena) // error: escapes its region
    0
}
```

Every refusal names the arena and the way the pointer got out, in the same
words and at the same place from both compilers:

```
region: a pointer into arena 'arena' escapes its region by being returned;
it may not outlive the arena
```

The four ways it ends are `being returned`, `being stored outside it`, `being
stored into a parameter` (the `uses` case above), and `being the block's value`.

## 8a.5a A container in an arena

`std/arena.frost` is the arena, and `std/fixed.frost` is the container over what
it carves. `Fixed<T>` holds a `[]T` somebody else owns and a count of how much
of it is live. It is a type of its own, beside `Vec<T>`, which calls the heap
in three places a bump allocator has no answer for.

`Fixed<T>` owns nothing, so it is not `linear`: it has no block to give back,
and the arena reclaims what it handed out when the block ends. Its capacity is
the run's length, so a push past the end is an index past the end of a slice
and aborts where it happens. It may not outlive its storage, which the region
check enforces:

```frost,sketch
draw_frame :: fn(mut scratch: Arena, world: []Sprite) -> i64 {
    mut total : i64 = 0
    with scratch {
        run := arena_carve($Sprite, scratch, 16)
        mut visible := fixed_over(run)
        for sprite in world { fixed_push(visible, sprite) }
        total = tally(fixed_slice(visible))
    }
    total
}
```

`visible` holds a view of the arena, so assigning it to a binding outside the
block, reading its storage out one field at a time, or ending the block with it
are all refused. `total` is a number and leaves. `examples/scratch_frame.frost`
is the whole program.

This is the arena half of 8.2. The frame half is checked beside it, in
`src/check/regions.rs` and in `check_frame_escapes` in
`selfhosted/regions.frost`, and refused on the same grounds, with one
difference in where the burden sits. The arena check asks whether a pointer is
known to come from the region and refuses it when it is. The frame check asks
whether a view is known to come from somewhere that outlives the call and
refuses it when it is not. A view whose storage the walk cannot trace is
refused. Both compilers refuse the same programs.

## 8a.5b The type after `uses`

The type after `uses` is concrete. `uses Arena` names one allocator type, and a
signature has no way to stand for whatever allocator the caller has, so a
library function leaves `uses` to the program that built the allocator.
`std/arena.frost` takes `mut a: Arena` as an ordinary parameter. Inside a `with`
block the region check keys off the block, so a call there is checked either
way.

What stands in for it is a capability bundle taken as an ordinary compile-time
argument. `std/allocation.frost` declares `Allocation<A>` over `take`, `resize`
and `give`, and a function generic over its source writes them out:

```frost,sketch
carve :: fn(
    $T: Type,
    $A: Type,
    $source: Allocation<A>,
    mut a: A,
    count: i64
) -> []T
```

A call to a function supplied at compile time, whether a bundle's field or a
`$f` argument, is one whose body neither check can see. What it answers with is
worth the shortest-lived argument that could have reached it, which is the rule
already used for a named function the walk cannot follow: a callee can only
build a view out of what it was handed or out of storage that outlives the call,
and a callee handing back a view of its own frame is caught where that callee is
itself checked. So `carve` over the caller's arena is accepted, and `carve` over
an allocator built in this frame is refused with the sentence a bare pointer out
of a frame gets:

```
region: a pointer into the frame of 'leak' is the call's answer; the storage it
names dies when the call returns
```

What the concrete `uses` type still costs is notation. A container generic over
its source takes `$A`, `$source` and `mut a` as parameters rather than drawing
them, so `Vec<T>` is the heap container and `Fixed<T>` beside it is the one over
storage somebody else owns.

[Writing an allocator](../writing-an-allocator.md) is the worked version of this
chapter: one program that declares an allocator, carves from it, draws it
through `uses`, supplies it with `with`, and shows the refusal.

## 8a.6 The extent of the notation

The `uses` list is the whole of what a signature says about allocation. A
pointer type carries no region, and a lifetime is never written. Supply is
lexical: a call draws from the capabilities the enclosing function holds and
from the `with` blocks around it, and a call with neither in reach is an error.
