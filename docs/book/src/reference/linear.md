# 9. Linear resources

## 9.1 The linear rule

A struct or enum declared `linear` must be consumed exactly once. The move
rule gives "at most once". Linearity adds "at least once". A linear value still
live at the end of its owning scope is a compile error.

A resource held inside something else is one too, so a struct with a linear
field, a fixed array of resources, and a generic instantiated with one all carry
the obligation. An instantiation is asked about what it was bound to rather than
what its template declares, since a template's field names a parameter bound to
nothing. A slice is not one of these: it looks at storage it does not own.

The count is kept per *place* rather than per name, which is what makes it hold
for a resource inside something else. It crosses a call as well: a function
carries what it consumes through the parameters it only borrows, and a call site
reads that against the places the caller wrote, so `once(h)` consuming its
parameter's `.file` gives up `h.file` and a second `once(h)` is refused. Fields
carry across, elements do not; "What is not yet guarded" in
[memory-safety.md](../design/memory-safety.md) says why. `close(h.file)` consumes part of `h`, so a
second `close(h.file)` is a second consumption and so is consuming `h` after it.
Two separate fields, and two elements whose indexes are known apart, are
different storage and may each be consumed once. Assigning a place gives it back,
along with everything it definitely covers; writing a part of something that may
already have been given away is refused, since that storage belongs to whoever it
went to. The two directions take opposite sides of what the checks cannot tell:
an element at an index nobody knows is not revived, and is not written into.

A pool may not hold resources. A slot is emptied by bumping a generation and
filled again by an insert that overwrites it, so nothing consumes the element
that leaves and no consumer can be written that would. Chapter 10.3 has the shape
and what to write instead.

## 9.2 Consuming

A linear value is consumed by moving it onward, whether by returning it, passing
it by value (often to an `extern` that takes ownership across the FFI boundary),
or `match`ing it. This replaces destructors. Cleanup is a tracked obligation,
never an implicit call, and a `linear enum` returned from a fallible function
cannot be silently dropped, so errors are non-ignorable by construction.

## 9.3 `defer`

`defer Stmt` runs `Stmt` where the function leaves, last deferred first.

Function exit rather than scope exit, and the difference is worth writing down.
A `defer` belongs at the top level of a function body. It runs before every
`return`, wherever that `return` is written, where the body falls off the end,
and where a `?` hands a failure on, since that is the function leaving too. It
does not run on `break` or `continue`, which leave a loop rather than the
function.

Written inside any block, whether that is an `if`, a loop body, an `unsafe`
block or a bare one, it is refused by both compilers with the same message. Down
there it would mean something other than it reads: it would run past the end of
the block it was written in, and one written in a loop would run once
afterwards, with whatever the loop left behind, rather than once a turn.
Refusing is the answer because neither of those is what anyone writing it meant.

What the function answers with is worked out before the deferred statements run
and is held in a name of its own, so a deferred statement cannot change what the
caller receives. Both compilers do this and both are held to it by a test.

A deferred statement is written out again at each exit, so the names it mentions
are read there. A name it mentions that is bound again below the `defer` is
therefore refused: the copy would read that later binding rather than the one in
scope where the `defer` was written, and on a path that never reached the later
binding it would read something nothing had written.

Its arguments are read where it runs, not where it was written, so a variable
reassigned after the `defer` changes what the deferred statement is given:

```frost,sketch
var f : i64 = 1
defer close(f)
f = 2               // closes 2
```

This is where Frost and Go differ. Go evaluates a deferred call's arguments at
the `defer` and would close 1.

Nothing here is checked. There is no unwinding in Frost, so `defer` is not
about a path an abort takes; what it does not do is notice that you never wrote
one. Owned resources should be `linear` types, which the compiler does check.
`defer` is for the bookkeeping that has no owner to be consumed: restoring a
saved index, resetting an arena to a mark, unwinding a flag around a call.

## 9.3a `errdefer`

`errdefer Stmt` runs `Stmt` where the function leaves through its failure set,
and nowhere else. Everything 9.3 says about where one may be written, which
exits it answers to, and when its arguments are read holds for it unchanged. A
function with no failure set has no exit for one to name, and an `errdefer` in
one is refused by both compilers with the same sentence.

`defer` and `errdefer` share one list, so they run in the order they were
written, last first, whichever kind each is. Two lists would have made the order
between a `defer` and an `errdefer` depend on which list was drained first,
which is a rule nobody would remember and nothing would remind them of.

What it is for is the resource a `?` steps over:

```frost,sketch
work :: fn() -> i64 ! FileError {
    f := open()?
    errdefer close(f)
    value := read(f)?      // leaves here on a failure, and `f` is closed
    close(f)               // leaves here with an answer, and `f` is closed
    value
}
```

Without the `errdefer`, the second `?` hands a failure on with `f` still open,
and there is nowhere to write the close: after the `?` is too late and before it
is too early. A `defer close(f)` would close it on both paths and then the
straight-line `close(f)` is a second consumption, which linearity refuses.

That is what the checker is taught. An `errdefer` answers for the failure path
alone: the value stays the straight-line path's to consume, so the `close(f)`
below is the first consumption rather than the second, and a resource live at a
`?` is no longer a leak. What it deliberately does not do is answer for the
whole obligation. An `errdefer close(f)` with no `close(f)` below it leaves with
an answer and the resource still open, and that is the ordinary refusal:

```
h := opened(1)
^ linear value 'h' is never consumed
```
