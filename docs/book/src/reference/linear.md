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

`defer` runs a statement at scope exit in LIFO order, for local best-effort
cleanup. Owned resources should be `linear` types, which the compiler checks,
rather than relying on `defer`.
