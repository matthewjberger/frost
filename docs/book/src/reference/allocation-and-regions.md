# 8a. Allocation sources and regions

An allocator in Frost is an ordinary struct a program declares, and an
allocation is an ordinary call (see
[allocators.md](../design/allocators.md)). What this chapter adds is a way to
stop threading one through every signature by hand, and a check that stops what
it hands out from outliving it.

Two forms do it. `uses A` on a function says the function draws an allocation
capability of type `A`. `with a { ... }` says which one, for every call inside
the block. The block is also the region: a raw pointer into `a` may not leave
it.

## 8a.1 Declaring a capability

`uses` follows the return type in a signature (13.6), and takes one or more
types:

```frost
Arena :: struct($N: usize) { data: [N]u8, offset: i64 }

make_two :: fn() -> i64 uses Arena<256> {
    p := alloc_int(arena)
    p^ = 10
    q := alloc_int(arena)
    q^ = 32
    p^ + q^
}
```

Nothing about `Arena` is built in. It is the struct the two lines above declare,
and `uses` would take a `Pool`, a `Scratch` or a `Bump` the same way.

A return type is not required. `fn() uses Arena<256> { ... }` is a signature,
and so is `fn() -> i64 uses Arena<256>, Scratch<64>`.

## 8a.2 The name the body reaches it by

Each capability is an implicit parameter, appended to the function's own
parameters in the order the `uses` list wrote them, taken by write borrow so a
callee that bumps an offset is bumping the caller's arena.

The body reaches it by the capability type's own name with the first letter
lowercased, ignoring any generic arguments. `Arena<256>` binds `arena`,
`Scratch<64>` binds `scratch`, `Bump` binds `bump`. That is why `make_two` above
writes `alloc_int(arena)` without a parameter called `arena` anywhere in its
signature.

Deriving the name from the type is what lets a function drawing two sources tell
them apart:

```frost
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
draws, and the compiler supplies each. What it may draw from at the call is:

- the capability parameters the enclosing function itself holds, if it is a
  `uses` function;
- the arenas named by the `with` blocks around the call, the innermost winning
  where two carry the same name.

Each capability is matched by the name of 8a.2. A callee drawing
`Arena<256>, Scratch<64>` from inside a function that holds both finds `arena`
and `scratch` by name and forwards them, whichever order the caller declared
them in.

**A callee drawing exactly one takes the innermost source, whatever it is
named.** This is the rule that makes the feature usable: a `with scratch` block
supplies a `uses Arena<256>` call, because there is only one thing the call
could have meant. A callee drawing several has no such fallback, since the names
are the only thing telling them apart.

A call with nothing to draw from is refused, naming the two ways to fix it:

```
calling 'make' needs an allocation capability; declare `uses Arena<256>` on
the caller or wrap the call in a `with` block
```

A call with sources in scope but not the one it wanted is refused with the
wanted name, its type, and the list of what is in scope there instead.

A function literal written inside another function's body draws nothing. It
cannot see the enclosing capability, because it is not the enclosing function.

## 8a.4 `with` provides one, and is a region

```
"with" IDENT Block
```

The identifier names a variable the enclosing scope already owns and can write.
`with` allocates nothing and constructs nothing: the arena is built the way any
other value is, and the block only says which calls inside it are supplied from
it.

```frost
main :: fn() -> i64 {
    mut arena : Arena<256> = Arena { data = [0; 256], offset = 0 }
    mut result : i64 = 0
    with arena {
        result = forward()
    }
    print result
    0
}
```

Blocks nest, and a call inside both is supplied by the inner one:

```frost
with arena {
    with scratch {
        result = forwards()        // draws both, by name
        result = result + only_one()   // draws one, takes scratch
    }
}
```

The block bounds how long the compiler lets anything carved out of the arena
live, and that is what makes it a region.

## 8a.5 What the region check refuses

A raw pointer carved out of an arena is region-bound, and the check
(`src/regions.rs`) is a flow question rather than a lifetime system. Frost has
no global arenas and no closures, so a `^T` can only point into an arena the
function was handed directly, which makes provenance something a walk over the
body can follow.

Two regions exist:

- the body of a `with a { ... }` block, whose arena is `a`;
- the body of a `uses A` function, whose arena is the implicit capability.

A value is a pointer into the region when it is `ptr_to(...)` or
`slice_from(...)` over the arena, or over something already bound to it; a
binding already holding one; a call to a pointer-returning function that either
draws this arena or is passed it; whichever of those an `if`, a `match` or an
`unsafe` block ends with; or a read back through a pointer to one.

Inside a `with` block, such a value escapes three ways, and each is an error:
being returned; being assigned to a place rooted outside the region; and being
the block's trailing expression, since that value flows to the enclosing
scope.

```frost
main :: fn() -> i64 {
    mut arena : Arena<256> = Arena { data = [0; 256], offset = 0 }
    mut escaped : ^i64 = ptr_to(arena.offset)
    with arena {
        escaped = alloc_int(arena)     // error: escapes its region
    }
    escaped^
}
```

Storing one in a binding *declared inside* the region is fine. That binding dies
with the region, and the check follows it: anything later read out of it is a
region pointer too.

Inside a `uses` function the rule differs in one place. Returning an arena
pointer is allowed, because the caller's `with` block is where that pointer's
region actually is and the check runs there. Writing one into a parameter is
not, since the parameter belongs to a frame that outlives this one:

```frost
Reg :: struct { ptr: ^i64 }

stash :: fn(mut r: Reg) -> i64 uses Arena<256> {
    r.ptr = alloc_int(arena)      // error: escapes its region
    0
}
```

Every refusal reads the same way, naming the arena and how the pointer got out:

```
region: a pointer into arena 'arena' escapes its region by being returned;
it may not outlive the arena
```

This is the arena half of 8.2. The frame half is checked beside it in
`src/regions.rs` and refused on the same grounds, with one difference in where
the burden sits. The arena check asks whether a pointer is known to come from the
region and refuses it when it is; the frame check asks whether a view is known to
come from somewhere that outlives the call and refuses it when it is not. A view
whose storage the walk cannot trace is refused, so a road nobody wrote down is a
refusal rather than a leak. What that costs is measured rather than assumed:
across the standard library, the self-hosted compiler and the examples it refuses
nothing.

## 8a.6 What this is not

There are no lifetime annotations, no region variables on pointer types, and
nothing to write in a signature beyond the `uses` list. There is also no
inference across a program: a call is supplied from what is lexically around it,
and a call with nothing around it is an error rather than a search. What a
function allocates from is either on its own signature or in the block the
caller wrote.
