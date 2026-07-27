# 8. Ownership and borrowing

Frost is memory-safe without a garbage collector and without lifetimes. The
borrow rules run after parsing (`src/ownership.rs`).

## 8.1 Copy and move

Each type is copy or move. Scalars, pointers, function pointers, handles,
strings, and slices are copy: a slice and a `str` are a pointer and a length, and
copying one copies that pair rather than what it names. Structs and enums are
move. A
move value is consumed when passed by value, assigned, or returned. Using it
after is a use-after-move error. There is no `Copy`/`Clone` derive and no
implicit deep copy. A second copy of an aggregate is constructed explicitly.

## 8.2 Second-class borrows

A borrow exists only as a parameter mode, so it cannot be stored in a struct or
enum field, placed in an array, or returned: there is no syntax that would name
one. Because a borrow cannot escape its call, borrow analysis is scope-local, and
Frost has no lifetime annotations, lifetime variables, or borrow regions.

A raw pointer can escape, which is what it is for, so the two ways one could
outlive its storage are checked rather than forbidden. A function may not answer
with a pointer or a slice that names its own frame, and an arena pointer may not
outlive the `with` block that owns the arena. A `uses` function may hand one back
to its caller, whose region checks it, but may not store one into a parameter.

## 8.3 Borrow exclusivity

Within one call, a value may be read-borrowed any number of times or write-
borrowed exactly once, never both at once. Passing the same variable to two `mut`
parameters of one call is rejected. This per-call check suffices to prevent
mutable aliasing precisely because borrows cannot escape.

## 8.4 Reference escape through returns

A function or `extern` whose return type contains a reference is rejected.
`Handle<T>` is not a reference and may be returned and stored freely, the
intended replacement for an escaping borrow.
