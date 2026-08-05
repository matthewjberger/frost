# What the probes found

Three questions asked of the compiler rather than of the design notes. Every
claim here is a program that was run; the programs are in the harness where they
are worth keeping.

## The region checker refuses one thing, and it is the right thing

The reported problem was that the checker rejects valid code when it cannot
trace provenance through deeply nested types. It does not. Six shapes, all
accepted:

| shape | written | answer |
| --- | --- | --- |
| one field | `b.room` | accepted |
| two fields | `o.held.room` | accepted |
| three fields | `c.b.a.room` | accepted |
| through an element | `b.slots[at].room` | accepted |
| from a branch | `if (pick) { return b.one }` then `b.two` | accepted |
| through a call | `inner(b)` where `inner` answers `b.room` | accepted |
| an array field | `d.a` where `a: [4]i64` | accepted |
| from the heap | `heap_slice($i64, 4)` | accepted |

Depth of nesting is not what the checker traces on. What it traces is where the
storage came from, and it has two answers it is happy with: a parameter, reached
through any number of fields, elements and calls, or an allocation.

The refusal class is one shape:

> a view whose storage is the calling function's own frame, on any path.

```frost
view :: fn() -> []i64 {
    var local : [4]i64 = [0; 4]
    local
}
```

> region: a pointer into the frame of 'view' is the call's answer; the storage
> it names dies when the call returns

That refusal is correct, and it holds for a function that answers with a frame
view on one path and a parameter view on another. Refusing the mixed case is
conservative, and it is the conservative direction that is sound: the answer is
a view of the frame on that path whatever the other path does.

No provenance annotation is designed, because no refusal was found that needed
one. Designing surface for a problem that does not reproduce would be surface
with no caller.

### Blast radius

The strongest form of the property — adding a field to struct `S` cannot change
the accept or reject status of any function that names neither `S` nor a type
containing `S` — holds, and is what the checker's shape predicts: provenance is
traced through the places a function names, and a struct it does not name is not
one of them.

```frost
Unrelated :: struct { x: i64, y: [8]f32 }   // y added
Box :: struct { room: []i64 }
Slot :: struct { held: Box }
view :: fn(s: Slot) -> []i64 { s.held.room }
```

Adding `y` changes nothing about `view`, and nothing about a function that
answers with a frame view. What a field addition does change is layout, which is
global and loud by design, and the type errors that follow name the field.

## Callee determinism holds for a name, not for a value

The thesis is that the text of a call site fully determines what it calls. Asked
of the compiler, it is true of a call to a name and false of a call to a value.
All four of these compile and run:

| shape | written | answers |
| --- | --- | --- |
| parameter | `run :: fn(f: fn(i64, i64) -> i64) -> i64 { f(2, 3) }` | 5 |
| local | `f := add` then `f(2, 3)` | 5 |
| reassigned local | `var f := add`, `f = mul`, `f(2, 3)` | 6 |
| struct field | `t.op(2, 3)` | 5 |

So the receiver-inference problem is not removed, it is relocated to dataflow: a
reader of `f(2, 3)` has to find what `f` was bound to, and with `var` that is a
question about every assignment that reaches the call.

The honest statement of the thesis is therefore scoped:

> A call whose callee is written as a name is fully determined by the text of
> the call site: there is one namespace, no overloading, no methods and no
> traits, so the name is the callee. A call whose callee is an expression names
> a value, and what that value is, is a dataflow question like any other.

The scope is worth what it costs. Forbidding function values entirely would make
the unscoped claim true and would take the callback tables, the render graph's
pass functions and the ECS's system lists with it. Restricting them to a binding
that is never reassigned would keep those and cost the `var f` case, which the
corpus does not use. That is the cheapest strengthening available and it is a
language change, so it is written down here rather than made.

Inside a generic body the same rule holds with one addition: a call to a `$f`
compile-time parameter is resolved where the generic is expanded, so it is
determined by the instantiation's text rather than the template's. `where`
bounds do not dispatch — they are preconditions the compiler already answers, so
nothing about them makes a call resolve differently per type.

## Allocation reaches the process through one door

The invariant behind "nothing allocates behind your back", stated precisely:

> Every allocation is a call to `frost_rt_heap_alloc`, `frost_rt_heap_realloc`
> or the arena, all of which are `extern fn` rather than `safe extern fn`, so
> reaching them takes an `unsafe` block. The only safe way through is a named
> function in `std/mem.frost`, and the only safe way to an arena is a `uses`
> capability written into the signature.

Verified by inspection: `std/mem.frost` is the one file in the tree that
declares those externs, and nothing else in the corpus names them. A function
that allocates therefore either calls something named in `std/mem.frost`, or
declares `uses Arena`, or has an `unsafe` block in it. All three are visible in
the text.
