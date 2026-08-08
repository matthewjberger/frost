# 8. Ownership and borrowing

Frost is memory-safe, and the ownership and borrow rules below give the
guarantee. They run after parsing (`src/check/ownership.rs`).

## 8.1 Copy and move

Each type is copy or move. Scalars, pointers, function pointers, handles,
strings, slices, and fixed arrays are copy. A slice and a `str` are a pointer
and a length, and copying one copies that pair. A fixed array `[N]T` is copy as
well, so passing one leaves it live. Structs and enums are move, and `is_copy`
in `src/types.rs` is the whole list.

A move value is consumed when passed by value, assigned, or returned. Using it
after is a use-after-move error. A second copy of an aggregate is constructed
explicitly, and the language has no `Copy`/`Clone` derive and no implicit deep
copy.

## 8.2 Second-class borrows

An implicit borrow is a parameter mode and nothing else, so it cannot be stored
in a struct or enum field, placed in an array, or returned: there is no syntax
that would name one. The one borrow a program writes down is `ref T` (3.3),
which may be returned and may not be stored, and which the checks below hold to
what outlives the call. Neither kind carries a lifetime annotation, a lifetime
variable, or a borrow region, and there is nothing to write in a signature about
how long one lives.

A raw pointer can escape, so the two ways one could outlive its storage are
checked. A function may not hand out a pointer or a slice that names its own
frame, by answering with it or by writing it anywhere the call cannot see, and
an arena view may not outlive the `with` block that owns the arena. The type the
position expects decides whether a value is a view of the storage beside it,
since an array becoming a `[]T` is a view being formed and only the other side
says so. A `uses` function may hand one back to its caller, whose region checks
it, and may not store one into a parameter. Chapter 8a is that check in full,
along with the `uses` and `with` forms it runs over.

A third way is checked separately, because no scope expresses it: the storage a
view names can move while everything around it stays alive. A container that
fills asks the allocator for a wider block and gives the old one back, so a view
taken before that names storage the allocator has taken.

```frost,sketch
view := vec_slice($i64, v)
vec_push($i64, v, 1)         // fills, so the block is replaced
print("{}\n", view[0])      // refused
```

Two summaries make it visible from the call, worked out for every function in
the program: which run under a parameter its answer views, and which run under a
parameter it replaces. Both are runs of field names, since a
container with more than one run may grow one while a caller holds a view of
another. Taking the view again after the growth clears it.

## 8.3 Borrow exclusivity

Within one call, a value may be read-borrowed any number of times or write-
borrowed exactly once, never both at once. Passing the same variable to two `mut`
parameters of one call is rejected. The check is per call, and an implicit
borrow cannot escape the call it was made for, so mutable aliasing cannot arise.

Two places overlap unless something says they are apart. Fields are apart when
the names differ. Two indexes are apart only when both are numbers and the
numbers differ, so `xs[0]` and `xs[1]` are two elements while `xs[i]` and `xs[j]`
are one place: they name the same slot whenever `i` and `j` evaluate the same.

Places reached through different raw pointers stay one place. `p^` and `q^` are
one place whenever `p` and `q` hold one address. Reaching through a `^T` is
gated on an `unsafe` block (6a).

## 8.4 Reference escape through returns

An implicit borrow cannot be returned, because there is nothing to write. It is
a parameter mode (3.3) with no type of its own, so a signature has no way to say
it hands one back.

`ref T` is the explicit exception, and it is checked. A function may answer with
one: `arena_at(...) -> ref T` reaches into an arena, and `std/slab.frost`
reaches into its own storage the same way. The pair of checks in 8.2 holds it to
something that outlives the call, applied to a `ref` the same way it is applied
to a `^T` or a slice. A `ref` naming storage in the returning
function's own frame is refused, and one into an arena may not outlive the
`with` block that owns the arena (8a). Storing one is refused
separately, whatever its provenance: no struct field, no array element, no
container.

An `extern` return type may not contain a reference at all, since no check on
this side governs what a C function's answer names.

`Handle<T>` may be returned and stored freely, so a program keeps one when the
value has to outlive the call that produced it.
