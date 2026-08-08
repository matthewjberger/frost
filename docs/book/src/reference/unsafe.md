# 6a. Unchecked operations and `unsafe`

## 6a.1 What the gate is

Every other check in the compiler proves something about a program it can see
all of: a value is not used after it moves (8.1), a `linear` resource is
consumed exactly once (9.1), an arena pointer does not outlive its region (8a),
an index is inside the array (10.4). A few operations reach memory that none of
those checks covers, and an `unsafe` block is where they are written.

The operations below are *refused* outside a block. The set of `unsafe` blocks
in a program is the complete list of places to look when memory has been
corrupted.

## 6a.2 The operations that need one

```frost,sketch
unsafe { Stmt* }
```

Four things belong inside one, and each is a compile error outside one
(`src/check/unsafety.rs`, and `check_unsafety` in `selfhosted/regions.frost`).

- Reaching through a raw pointer: `p^`, `p^.field`, and `p[i]` where `p` is a
  `^T`. A `^T` is an address alone, and the length and the liveness of what it
  names are unknown to the compiler.
- `ptr_cast($T, p)`, which asserts that the bytes at an address are a `T`.
- `slice_from($T, p, n)`, which asserts that `n` elements of `T` live at `p`.
  Reads through the slice it yields are checked, so the assertion is the whole
  of what is unchecked.
- Calling an `extern fn` that is not marked `safe`. The body is C, outside every
  check the compiler makes.

The error names the operation and the line: "reading through a raw pointer is
unchecked, so it belongs in an `unsafe` block".

## 6a.3 Operations written in ordinary code

Forming an address is ordinary code, and reaching through it is gated.
`ptr_to(place)` names a place the compiler can already see, and the frame and
region checks of 8.2 hold what it yields to storage that outlives it, so
`ptr_to` is written in ordinary code and the block goes around the read, the
write, or the cast:

```frost,sketch
alloc_int :: fn(mut a: Arena<256>) -> ^i64 {
    slot := ptr_to(a.data[a.offset])
    a.offset = a.offset + sizeof(i64)
    unsafe { ptr_cast($i64, slot) }
}
```

Indexing an array or a slice is ordinary code, since both know their length and
both are checked (10.4). So is reaching a field through a borrow, so a
`mut world: World` parameter writes `world.tick = 0` on its own, and so is
`pool[handle]`, which is checked twice (10.2). A `ref T` (3.3) is read and
written in ordinary code.

Nesting is allowed. An `unsafe` block inside another is already covered by the
outer one.

## 6a.4 `safe extern fn`

A C function is unchecked by default, so every call to one is gated. `safe`
marks a function that cannot corrupt memory. `frost_rt_emit_int` writes a number
to standard output. `sqrtf` takes a float and answers with a float.

`safe extern fn` records that judgement once, at the declaration:

```frost
sqrtf :: safe extern fn(x: f32) -> f32
sinf  :: safe extern fn(x: f32) -> f32

frost_rt_emit_int :: safe extern fn(value: i64)
```

Calls to one are ordinary code. The word is the author's assertion that this
function was read and cannot corrupt memory, and it goes on the declaration
because the assertion is about the function. `std/math.frost`, `std/io.frost`
and `selfhosted/core.frost` are where the standard library and the compiler use
it. `malloc` in `selfhosted/core.frost` is one, because handing back memory
corrupts none.

`safe` is a keyword (2.4) and carries this meaning only here. An `extern fn`
without it is gated, and the gate has no per-call opt-out.

## 6a.5 What the check refuses when it cannot tell

Three of the four operations are recognized by their shape and need no type. The
index rule is the one that has to know whether the base is a raw pointer, since
an array, a slice and a `str` each carry a length and are checked while a `^T` is
not.

A base whose type the pass cannot name is refused. The pass names a base in the
shapes programs write: a call's return type off the declaration, an element's
off its array, slice, `str` or pointer, a pointee's off the pointer, a field's
off the struct, a block's off its last statement, and a literal's off itself.

The walk lists every statement and expression form, so a form nobody handled is
a compile error in the compiler.

The gate runs on every build, and no flag turns it off.

## 6a.6 `--audit-unsafe`

The gate reports a missing block. `--audit-unsafe` reports a block that gates
nothing, in two shapes:

- a block holding no gated operation at all;
- a block written inside another, which already covers what is in it.

It is off by default, and passing the flag makes each report a build error.

## 6a.7 Where the guarantee stops

`^T` is outside the six guarantees of 10.5, and so is a C function's use of what
it is handed. Everything else in the language touches only memory it
has been shown to own, and every operation that reaches further is inside an
`unsafe` block or behind a `safe extern` declaration.
