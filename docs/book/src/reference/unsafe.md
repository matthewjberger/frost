# 6a. Unchecked operations and `unsafe`

## 6a.1 What the gate is

Every other check in the compiler proves something about a program it can see
all of: a value is not used after it moves (8.1), a `linear` resource is
consumed exactly once (9.1), an arena pointer does not outlive its region (8a),
an index is inside the array (10.4). A few operations reach memory that none of
that covers, and those are the ones an `unsafe` block is for.

The block does not enable anything a program could not otherwise write. It is
the other way around: the operations below are *refused* outside one, which is
what makes the set of `unsafe` blocks in a program the complete list of places
to look when memory has been corrupted. Without the refusal the block would be a
comment.

## 6a.2 The operations that need one

```frost
unsafe { Stmt* }
```

Four things belong inside one, and each is a compile error outside one
(`src/unsafety.rs`, and `check_unsafety` in `selfhosted/regions.frost`).

- Reaching through a raw pointer: `p^`, `p^.field`, and `p[i]` where `p` is a
  `^T`. A `^T` carries no length and no proof that what it names is live.
- `ptr_cast($T, p)`, which asserts that the bytes at an address are a `T`.
- `slice_from($T, p, n)`, which asserts that `n` elements of `T` live at `p`.
  Reads through the slice it yields are checked, so the assertion is the whole
  of what is unchecked.
- Calling an `extern fn` that is not marked `safe`. The body is C, outside every
  check the compiler makes.

The error names the operation and the line: "reading through a raw pointer is
unchecked, so it belongs in an `unsafe` block".

## 6a.3 What does not need one

Forming an address is not gated. Using it is. `ptr_to(place)` names a place the
compiler can already see, and the frame and region checks of 8.2 hold what it
yields to storage that outlives it, so `ptr_to` is written in ordinary code and
the block goes around the read, the write, or the cast:

```frost
alloc_int :: fn(mut a: Arena<256>) -> ^i64 {
    slot := ptr_to(a.data[a.offset])
    a.offset = a.offset + sizeof(i64)
    unsafe { ptr_cast($i64, slot) }
}
```

Indexing an array or a slice does not need one either, since both know their
length and both are checked (10.4). Nor does reaching a field through a borrow,
which is why a `mut world: World` parameter writes `world.tick = 0` with no
block around it, and nor does `pool[handle]`, which is checked twice (10.2). A
`ref T` (3.3) is read and written with no block, and that is the difference
between it and a `^T`.

Nesting is allowed and means nothing extra. An `unsafe` block inside another is
already covered by the outer one.

## 6a.4 `safe extern fn`

A C function is unchecked by default, so every call to one is gated. Some are
not worth gating. `frost_rt_emit_int` writes a number to standard output;
`sqrtf` takes a float and answers with a float. A call that cannot corrupt
memory is not a place to look for corruption, and listing it among the ones that
can makes the list worth less.

`safe extern fn` is where that judgement is written down, once, at the
declaration:

```frost
sqrtf :: safe extern fn(x: f32) -> f32
sinf  :: safe extern fn(x: f32) -> f32

frost_rt_emit_int :: safe extern fn(value: i64)
```

Calls to one need no block. The word is the author's assertion that this
function was read and cannot corrupt memory, and the reason it goes on the
declaration rather than at each call is that the assertion is about the
function, not about the call. `std/math.frost`, `std/io.frost` and
`selfhosted/core.frost` are where the standard library and the compiler use it;
`malloc` in `selfhosted/core.frost` is one, because handing back memory corrupts
none.

`safe` is a keyword (2.4) and means nothing anywhere else. An `extern fn`
without it is gated, and there is no per-call opt-out.

## 6a.5 What the check refuses when it cannot tell

Three of the four operations are recognized by their shape and need no type. The
index rule is the one that has to know whether the base is a raw pointer, since
an array, a slice and a `str` each carry a length and are checked while a `^T` is
not.

A base whose type the pass cannot name is refused rather than allowed. A gate
that lets the unknown through reports what it happened to recognize, and the list
of `unsafe` blocks is then worth nothing. What keeps the rule from refusing
ordinary code is that the pass names a base in the shapes programs write: a
call's return type off the declaration, an element's off its array, slice, `str`
or pointer, a pointee's off the pointer, a field's off the struct, a block's off
its last statement, and a literal's off itself. Across `selfhosted/frost.frost`
and `std/ecs.frost` that leaves no unnameable base, so the strict rule costs
nothing.

The walk lists every statement and expression form rather than ending in a
wildcard, so a form nobody handled is a compile error in the compiler instead of
a hole in the gate. `print` was that hole: every gated operation written under
one compiled, and a program holding no `unsafe` block at all read far out of
bounds through a raw pointer and died on the access.

The gate runs on every build. `FROST_CHECK_UNSAFE=0` turns it off, which exists
for compiling older sources that have not marked their unchecked operations yet
and is not part of the language.

## 6a.6 `--audit-unsafe`

The gate says a block is missing. The audit says a block is idle. Two shapes
vouch for nothing, and `--audit-unsafe` reports both:

- a block holding no gated operation at all;
- a block written inside another, which already covers what is in it.

It is off by default, and passing the flag makes each report a build error. A
build pays for the checks that keep a program correct. This one keeps it tidy,
and tidiness is not worth a pass over the source on every compile.

## 6a.7 Where the guarantee stops

`^T` is outside the six guarantees of 10.5 by design, and so is what a C
function does with what it is handed. What the language promises is narrower and
checkable: nothing else in it can touch memory it has not been shown to own, and
everything that can is inside an `unsafe` block or behind a `safe extern`
declaration. Both are greppable.
