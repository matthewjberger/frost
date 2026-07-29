# 9. Linear resources

## 9.1 The linear rule

A struct or enum declared `linear` must be consumed exactly once. The move
rule gives "at most once". Linearity adds "at least once". A linear value still
live at the end of its owning scope is a compile error.

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

Written inside an `if` or a loop body it is refused, by both compilers. Down
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

```frost
mut f : i64 = 1
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
