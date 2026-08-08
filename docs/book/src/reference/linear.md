# 9. Linear resources

## 9.1 The linear rule

A struct or enum declared `linear` must be consumed exactly once. The move
rule gives "at most once". Linearity adds "at least once". A linear value still
live at the end of its owning scope is a compile error.

A resource held inside something else is one too, so a struct with a linear
field, a fixed array of resources, and a generic instantiated with one all carry
the obligation. An instantiation is asked about the types it was bound to, since
a template's field names a parameter bound to nothing. A slice views storage
owned elsewhere, and the obligation stays with the owner.

The count is kept per *place*, so it holds for a resource inside something else.
It crosses a call as well: a function carries what it consumes through the
parameters it only borrows, and a call site reads that against the places the
caller wrote, so `once(h)` consuming its parameter's `.file` gives up `h.file`
and a second `once(h)` is refused. Fields carry across, elements do not.
"Where the checks stop" in [memory-safety.md](../design/memory-safety.md) says
why.
`close(h.file)` consumes part of `h`, so a second `close(h.file)` is a second
consumption and so is consuming `h` after it. Two separate fields, and two
elements whose indexes are known apart, are different storage and may each be
consumed once. Assigning a place gives it back, along with everything it
definitely covers. Writing a part of something that may already have been given
away is refused, since that storage belongs to whoever it went to. An element at
an index nobody knows is neither revived nor written into.

A pool may not hold resources. A slot is emptied by bumping a generation and
filled again by an insert that overwrites it, so nothing consumes the element
that leaves and no consumer can be written that would. Chapter 10.3 has the shape
and what to write instead.

## 9.2 Consuming

A linear value is consumed by moving it onward, whether by returning it, passing
it by value (often to an `extern` that takes ownership across the FFI boundary),
or `match`ing it. This replaces destructors. Cleanup is a tracked obligation,
and the compiler makes no call of its own. A `linear enum` returned from a
fallible function is consumed like any other resource, so an error is read by
construction.

## 9.3 `defer`

`defer Stmt` runs `Stmt` where the function leaves, last deferred first.

The exit it answers to is the function's. A `defer` belongs at the top level of
a function body. It runs before every `return`, wherever that `return` is
written, where the body falls off the end, and where a `?` hands a failure on,
since that is the function leaving too. `break` and `continue` leave a loop, and
neither runs a deferred statement.

Written inside any block, whether that is an `if`, a loop body, an `unsafe`
block or a bare one, it is refused.

The function's answer is worked out before the deferred statements run and held
in a name of its own, so a deferred statement cannot change what the caller
receives.

A deferred statement is written out again at each exit, so the names it mentions
are read there. A name it mentions that is bound again below the `defer` is
therefore refused: the copy would read that later binding, and on a path that
never reached the later binding it would read something nothing had written.

Its arguments are read where it runs, so a variable reassigned after the
`defer` changes what the deferred statement is given:

```frost,sketch
var f : i64 = 1
defer close(f)
f = 2               // closes 2
```

Go evaluates a deferred call's arguments at the `defer` and would close 1.

Nothing here is checked. Frost has no unwinding, so a `defer` answers to the
exits above and to nothing else, and a missing `defer` compiles. An owned
resource is a `linear` type, which the compiler does check. `defer` is for the
bookkeeping that has no owner to be consumed: restoring a saved index, resetting
an arena to a mark, unwinding a flag around a call.

## 9.3a `errdefer`

`errdefer Stmt` runs `Stmt` where the function leaves through its failure set,
and nowhere else. Everything 9.3 says about where one may be written, which
exits it answers to, and when its arguments are read holds for it unchanged. A
function with no failure set has no exit for one to name, and an `errdefer` in
one is refused.

`defer` and `errdefer` share one list, so they run in the order they were
written, last first, whichever kind each is.

It is for the resource a `?` steps over:

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

An `errdefer` answers for the failure path alone. The value stays the
straight-line path's to consume, so the `close(f)` below is the first
consumption, and a resource live at a `?` is consumed on the failure path. The
whole obligation is still the body's. An `errdefer close(f)` with no `close(f)`
below it leaves with an answer and the resource still open, which is the
ordinary refusal:

```
h := opened(1)
^ linear value 'h' is not consumed on every path before return
```
