# 8. Ownership and borrowing

Frost is memory-safe without a garbage collector and without lifetimes. The
borrow rules run after parsing (`src/check/ownership.rs`).

## 8.1 Copy and move

Each type is copy or move. Scalars, pointers, function pointers, handles,
strings, slices, and fixed arrays are copy. A slice and a `str` are a pointer
and a length, and copying one copies that pair rather than what it names. A
fixed array `[N]T` is copy as well, so passing one does not consume it. Structs
and enums are move, and `is_copy` in `src/types.rs` is the whole list.

A move value is consumed when passed by value, assigned, or returned. Using it
after is a use-after-move error. There is no `Copy`/`Clone` derive and no
implicit deep copy. A second copy of an aggregate is constructed explicitly.

## 8.2 Second-class borrows

An implicit borrow is a parameter mode and nothing else, so it cannot be stored
in a struct or enum field, placed in an array, or returned: there is no syntax
that would name one. The one borrow a program writes down is `ref T` (3.3),
which may be returned and may not be stored, and which the checks below hold to
what outlives the call. Neither kind carries a lifetime annotation, a lifetime
variable, or a borrow region, and there is nothing to write in a signature about
how long one lives.

A raw pointer can escape, which is what it is for, so the two ways one could
outlive its storage are checked rather than forbidden. A function may not hand
out a pointer or a slice that names its own frame, by answering with it or by
writing it anywhere the call cannot see, and an arena view may not outlive the
`with` block that owns the arena. What decides whether a value is a view of the
storage beside it is the type the position expects, since an array becoming a
`[]T` is a view being formed and only the other side says so. A `uses` function may hand one back
to its caller, whose region checks it, but may not store one into a parameter.
Chapter 8a is that check in full, along with the `uses` and `with` forms it runs
over.

A third way is checked separately, because no scope expresses it: the storage a
view names can move while everything around it stays alive. A container that
fills asks the allocator for a wider block and gives the old one back, so a view
taken before that names storage the allocator has taken.

```frost,sketch
view := vec_slice($i64, v)
vec_push($i64, v, 1)         // fills, so the block is replaced
print_int_line(view[0])      // refused
```

What makes it visible from the call is two summaries, worked out for every
function in the program: which run under a parameter its answer views, and which
run under a parameter it replaces. Both are runs of field names, since a
container with more than one run grows one while a caller holds a view of
another, and that is ordinary. Taking the view again after the growth is what
clears it.

## 8.3 Borrow exclusivity

Within one call, a value may be read-borrowed any number of times or write-
borrowed exactly once, never both at once. Passing the same variable to two `mut`
parameters of one call is rejected. This per-call check suffices to prevent
mutable aliasing precisely because an implicit borrow cannot escape the call it
was made for.

Two places overlap unless something says they are apart. Fields are apart when
the names differ. Two indexes are apart only when both are numbers and the
numbers differ, so `xs[0]` and `xs[1]` are two elements while `xs[i]` and `xs[j]`
are one until the program says otherwise: they name the same slot whenever `i`
and `j` evaluate the same, and nothing in a compiler can rule that out.

The one pair this does not separate is two places reached through different raw
pointers. `p^` and `q^` are one place whenever `p` and `q` hold one address.
Reaching through a `^T` is gated on an `unsafe` block (6a), which is where that
sits.

## 8.4 Reference escape through returns

An implicit borrow cannot be returned, because there is nothing to write. It is
what a parameter mode means (3.3) and has no type of its own, so a signature has
no way to say it hands one back.

`ref T` is the explicit exception, and it is checked rather than refused. A
function may answer with one: `arena_at(...) -> ref T` is the reason it exists,
and reaching into `std/slab.frost`'s storage is what it is for. What holds it to
something that outlives the call is the pair of checks in 8.2, applied to a
`ref` the same way they are applied to a `^T` or a slice. A `ref` naming storage
in the returning function's own frame is refused, and one into an arena may not
outlive the `with` block that owns the arena (8a). Storing one is refused
separately, whatever its provenance: no struct field, no array element, no
container.

An `extern` is the one place the old blanket rule survives. Its return type may
not contain a reference at all, since no check on this side governs what a C
function's answer names.

`Handle<T>` is not a reference and may be returned and stored freely, which is
what a program reaches for when what it wants to keep has to outlive the call
that produced it.
