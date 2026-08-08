# Callbacks with a typed context

How a Frost function and a Frost context cross into a C library that takes a
callback, and why the crossing needs no generated code. It runs against a real C
callback API on both backends, in the test
`a_callback_registered_with_a_c_library_runs`, which links a small C library
that stores a `(callback, userdata)` pair and calls it back later.

## What the C idiom costs

Goal 2 in [philosophy.md](philosophy.md) says safety comes from making dangerous
shapes unrepresentable. Without a callback form of its own, the only way to write
one is the C idiom: a function pointer beside an untyped `^u8` the callee casts
back. Every piece of that idiom is already in the language:

- `fn(T1, ...) -> R` is a function pointer type
  ([types.md](../reference/types.md) 3.5), and a named function used as a value
  lowers to `IrRvalue::FunctionAddress` in `src/ir.rs`.
- `ptr_cast($T, p)` reinterprets a pointer at no runtime cost (types.md 3.3).
- `^T` carries no guarantee once formed, which the reference says outright.

So a callback is writable and it is entirely outside every check the language
has. `src/check/regions.rs` reasons about arena pointers by provenance, and its
argument is stated in its own header comment: "Frost has no global arenas and no
closures, so a `^T` can only point into an arena a function was handed directly."
A `^u8` handed to a C library and called back through later is precisely the case
that argument does not cover. `src/check/ownership.rs` cannot see through it either.

Under that idiom every callback-shaped API is an unsafe API, because the only
expression of a callback is a raw escape hatch. The surface `&` removal pushed
that shape out of the language, and the C boundary is where it comes back.

## The shape

Closures stay a non-goal. A callback is a compile-time function argument plus a
typed context the caller owns:

```frost
Ctx :: struct { hits: i64 }

on_event :: fn(mut ctx: Ctx, code: i64) { ctx.hits = ctx.hits + code }

register :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64
```

Nothing is generated for the crossing. A `mut` parameter is already a pointer in
the signature, and Frost and C share a calling convention, so `on_event`
compiled for Frost *is* the `void (*)(void*, int64_t)` the library wants. What
the compiler does at a registration is pass the handler's address and the
context's address, and there is no cast anywhere in the program. There is no
trampoline because there is no cast for one to hold.

## What the declaration says

### The handler is a bound compile-time parameter

A `$handler` parameter carrying a function bound on an `extern fn` is the
complete statement of "this extern takes a callback". Nothing is written beside
it, and in particular no capability: `uses Arena` means a real implicit
parameter is supplied at the call, which `src/lower/allocation_sources.rs` inserts,
and a callback needs no such parameter. A `uses CallbackAbi` would be a keyword
pretending to be a capability and a second thing to keep in step with the first.

The form is the bound one, `$handler: fn(mut Ctx, i64)`
([generics.md](../reference/generics.md) 11.1b), so the handler's signature is
checked against what the library expects at the call, by the code in
`src/ir/build.rs` that already checks compile-time signatures. Plain `$handler:
Type` would say only that the argument is a type.

### The extern's C signature

A `$handler` parameter contributes exactly one C argument, the callback pointer,
in the position it is written. The context contributes the `void*`, in
the position it is written. So the declaration is written in the order C wants
and the mapping needs no further rule:

```frost,sketch
register :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64
```

becomes `int64_t register(void (*)(void*, int64_t), void*)`.

The context is the parameter whose type is the type of the handler's context.
Position does not identify it, and must not, because libraries put the userdata
on either side of the function pointer.

The handler's context is its one `mut` parameter, wherever it is written, and
every other parameter is a callback argument that C passes through. What
identifies it is being the one parameter the handler can write. So both of these
are registrations, and the second is the order wgpu-native and most modern C
APIs take:

```frost,sketch
register :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64
request  :: extern fn($handler: fn(i32, i64, mut Ctx), move ctx: Ctx) -> i64
```

A declaration where no extern parameter has the context's type is an error at
the declaration, not at the call. So is a handler with no `mut` parameter: a
callback that cannot write its context is a callback that cannot do anything,
and reading one is the case a plain function pointer with no context already
covers. So is a handler with more than one, since then nothing says which of
them the library is being asked to keep.

### Ownership of the context

Registration moves the context in, unregistration moves it back out, and the
registration is a `linear` value.

```frost,sketch
Ctx          :: struct { hits: i64 }
Registration :: linear struct { token: i64 }

on_event   :: fn(mut ctx: Ctx, code: i64) { ctx.hits = ctx.hits + code }
register_handler   :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64
unregister_handler :: extern fn(token: i64) -> Ctx
unregister         :: fn(move r: Registration) -> Ctx { unregister_handler(r.token) }
```

Three things follow from moving rather than borrowing.

- No new machinery. A borrow held by C after the call returns is nothing the
  language has. `ref T` is returnable, but it is handed to a caller whose frame
  already outlives it, while a registered context is held by code the compiler
  cannot see, so making that safe means inventing the region annotation the
  whole design is built on not having. Moving needs nothing new.
  `check_ownership` already stops the caller touching a moved value, and
  `check_linearity` already forces a `linear` value to be consumed exactly once.
- Aliasing. While registered, the callback may fire at any moment, so the caller
  must not be reading or writing the context. Having moved it in, the caller
  cannot.
- Forgetting to unregister becomes a compile error, which is a real bug class
  in every C callback API, and a dangling callback into a freed context is the
  exact failure this is meant to prevent.

The fire-and-forget case, where the library never hands the callback back, does
not get an exception. The registration is still linear, and a program that means
to abandon it says so with a terminal consumer that takes it and returns nothing.

## Where the context lives

One question decides whether the crossing is safe: where the context lives while
the callback can fire.

`move ctx: Ctx` hands the value to the extern, and the extern keeps a pointer to
it. So the storage the pointer names has to outlive the call, and a moved
argument is a value in the caller's frame. `src/check/regions.rs` and
`check_frame_escapes` between them already reject a pointer into the current
frame being returned, stored into a parameter, or carried out inside a struct.
Handing one to an extern that keeps it is the case a callback adds, and it is
the one those checks were not written for.

So a registration carries one obligation, and it is the whole safety argument:

> The context argument of a callback registration must name storage that outlives
> the registration.

One way to meet it is to require the context to live in an arena or a pool and
reject a place in the current frame. That rule does not survive contact with the
language it is a rule for. A context is a value of a struct type, and a value
lives where it is bound. Putting one in an arena means holding a `^Ctx`, and
then the registration's context parameter is a pointer instead of a moved value
and the ownership argument above evaporates. The rule would reject every program
anyone could write.

The obligation is satisfied from the other end. A `Registration` is `linear`,
so `check_linearity` already forces it to be consumed exactly once in the
function that made it. A context in that same frame therefore outlives the
registration by construction, and the frame is exactly the right place for it.
What is left to stop is the registration *leaving* that function by some other
road, which is the same shape `src/check/regions.rs` already enforces for pointers:
returned, stored where the call cannot see, or handed back as the call's answer.

So the rule needs no new kind of check. A registration whose context is rooted
in this frame counts as a value that points into this frame, and the three roads
out are closed by the code that was already closing them. Linearity closes the
fourth, which is not consuming it at all.

Without that rule the crossing is type-safe and the program still has a dangling
pointer.

## The limits

A registration lives in the frame that holds its context, so a callback whose
context has to outlive that frame has no spelling here.

Reentrancy is the caller's problem. Nothing stops a callback from calling back
into code that reaches the same context. Moving the context into the
registration means no Frost code holds it, which says nothing about the C
library's own threading.
