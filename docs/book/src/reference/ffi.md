# 12. The foreign function interface

An `extern fn` declares a function with C linkage:

```frost
printf :: extern fn(fmt: ^i8, value: i64) -> i32
malloc :: extern fn(size: i64) -> ^u8
```

Frost scalar types map to the natural C types and `^T` is a C pointer. String
literals denote NUL-terminated bytes for `^i8` parameters. An `extern` takes
parameter modes like any other function.

Aggregate parameters and aggregate returns are not symmetric, and the
asymmetry is deliberate.

- An aggregate parameter is passed as a pointer to the value, so
  `close :: extern fn(f: File)` links against a C `void close(File*)`. This is a
  convention rather than the C ABI, chosen because most C APIs take structs by
  pointer, and it is what lets a `linear` resource have a terminal consumer
  across the boundary.
- A parameter written `value` is passed as C passes a struct instead, following
  the target's real ABI, so `label :: extern fn(value v: View)` links against a
  C `void label(View)`. `value` is a word rather than a keyword and is only a
  mode where a mode can appear, so a parameter may still be named `value`. It
  says how the bytes cross rather than what the caller gives up: C receives a
  copy and the argument is borrowed, exactly as an unmarked one is.
- An aggregate return is by value, following the target's real C ABI: in
  registers where that target's rule says so, and through a hidden pointer where
  it does not. A return could not have been a convention, because `-> Ctx` has
  to mean what C means by it and `-> ^Ctx` is how a returned pointer is written.

## 12.1 Callbacks

An `extern` whose parameter list has a `$` parameter bound to a function
signature is a callback registration:

```frost
Ctx :: struct { hits: i64 }

on_event         :: fn(mut ctx: Ctx, code: i64) { ctx.hits = ctx.hits + code }
register_handler :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64
```

The handler's context is its one `mut` parameter, wherever in the signature it
is written, so a library that passes the userdata last is declared as
`fn(i32, i64, mut Ctx)`. A handler with no `mut` parameter is not a callback,
and one with more than one does not say which parameter is the context.
Whichever parameter of the extern has that type is the one the context is taken
from, found by type rather than by position because C libraries put the userdata
on either side of the function pointer, and it must be taken by `move`. The call
passes the handler's address and the context's address. There is no generated
trampoline, because a `mut` parameter is already a pointer in the signature and
Frost and C share a calling convention.

The context moving in is what makes this safe rather than merely typed: the
caller cannot touch the context while the callback can fire. A registration is
normally a `linear` value, so it must be consumed, and the region check refuses
to let it leave the frame that holds its context.

The FFI is otherwise asymmetric. Frost calls C, but C does not call Frost,
except through a registered callback, which is the one place a C library holds a
Frost function pointer. There is no stable exported ABI and no attribute to
expose a Frost function to a C caller. The emitted C is an internal lowering,
not an interface.
