# Callbacks with a typed context

How a Frost function and a Frost context cross into a C library that takes a
callback, and why the crossing needs no generated code. It runs against a real C
callback API on both backends, in the test
`a_callback_registered_with_a_c_library_runs`, which links a small C library
that stores a `(callback, userdata)` pair and calls it back later.

## The contradiction it exists to remove

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

The result was that every callback-shaped API in Frost was an unsafe API, not
because callbacks are unsafe but because the only expression of one was a raw
escape hatch. That is the inversion the surface `&` removal was meant to prevent,
reappearing at the C boundary.

## The shape

Closures stay a non-goal. Capture is not the answer. A context written down is.
A callback is a compile-time function argument plus a typed context the caller
owns:

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

Which form of compile-time parameter it is follows from that. Not `$handler:
Type`, which says only "some type", but the bound form `$handler: fn(mut Ctx,
i64)` ([generics.md](../reference/generics.md) 11.1b), so the handler's
signature is checked against what the library expects at the call, by the code
in `src/ir/build.rs` that already checks compile-time signatures.

### The extern's C signature

A `$handler` parameter contributes exactly one C argument, the callback pointer,
in the position it is written. The context contributes the `void*`, in
the position it is written. So the declaration is written in the order C wants
and the mapping needs no further rule:

```frost,sketch
register :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64
```

becomes `int64_t register(void (*)(void*, int64_t), void*)`.

Which parameter is the context is not positional and must not be, because
libraries put the userdata on either side of the function pointer. It is the
parameter whose type is the type of the handler's context.

The handler's context is its one `mut` parameter, wherever it is written, and
every other parameter is a callback argument that C passes through. Position is
not what identifies it. Being the one parameter the handler can write is. So
both of these are registrations, and the second is the order wgpu-native and
most modern C APIs take:

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

Registration moves it in, unregistration moves it back out, and the registration
is a `linear` value. Not a borrow.

```frost,sketch
Ctx          :: struct { hits: i64 }
Registration :: linear struct { token: i64 }

on_event   :: fn(mut ctx: Ctx, code: i64) { ctx.hits = ctx.hits + code }
register_handler   :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64
unregister_handler :: extern fn(token: i64) -> Ctx
unregister         :: fn(move r: Registration) -> Ctx { unregister_handler(r.token) }
```

Three things fall out, and each is a reason to prefer moving over borrowing.

- No new machinery. A borrow held by C after the call returns is nothing the
  language has. `ref T` is returnable, but it is handed to a caller whose frame
  already outlives it, while a registered context is held by code the compiler
  cannot see, so making that safe means inventing the region annotation the
  whole design is built on not having. Moving needs nothing new.
  `check_ownership` already stops the caller touching a moved value, and
  `check_linearity` already forces a `linear` value to be consumed exactly once.
- The aliasing guarantee is the one you want. While registered, the callback
  may fire at any moment, so the caller must not be reading or writing the
  context. Having moved it in, the caller cannot.
- Forgetting to unregister becomes a compile error, which is a real bug class
  in every C callback API, and a dangling callback into a freed context is the
  exact failure this is meant to prevent.

The fire-and-forget case, where the library never hands the callback back, does
not get an exception. The registration is still linear, and a program that means
to abandon it says so with a terminal consumer that takes it and returns nothing.
"I am deliberately leaking this" is worth having to write.

## Where the context lives

The rules above are the easy part. One question decides whether the feature is
safe or merely tidier, and it is where the context lives while the callback can
fire.

`move ctx: Ctx` hands the value to the extern, and the extern keeps a pointer to
it. So the storage the pointer names has to outlive the call, and a moved
argument is a value in the caller's frame. `src/check/regions.rs` and
`check_frame_escapes` between them already reject a pointer into the current
frame being returned, stored into a parameter, or carried out inside a struct.
What they did not reject was one being *handed to an extern that keeps it*,
because until callbacks nothing in the language could keep one.

So the feature adds exactly one obligation, and it is the whole safety argument:

> The context argument of a callback registration must name storage that outlives
> the registration.

The obvious answer, that the context therefore has to live in an arena or a
pool and a place in the current frame is rejected, is wrong, and it is wrong in
a way worth understanding, because it does not survive contact with the language
it is a rule for. A context is a value of a struct type, and a value lives where
it is bound. Putting one in an arena means holding a `^Ctx`, and then the
registration's context parameter is a pointer rather than a moved value and the
ownership argument above evaporates. That rule would reject every program anyone
could write.

The obligation is satisfied from the other end. A `Registration` is `linear`,
so `check_linearity` already forces it to be consumed exactly once in the
function that made it. A context in that same frame therefore outlives the
registration by construction, and the frame is exactly the right place for it.
What is left to stop is the registration *leaving* that function by some other
road, which is the same shape `src/check/regions.rs` already enforces for pointers:
returned, stored where the call cannot see, or handed back as the call's answer.

So the rule is not a new kind of check. A registration whose context is rooted in
this frame counts as a value that points into this frame, and the three roads out
are closed by the code that was already closing them. Linearity closes the
fourth, which is not consuming it at all.

That is what makes this different from the C idiom rather than a prettier
spelling of it. Without it the crossing is type-safe and the program still has a
dangling pointer.

## What is not settled

Two questions are open.

- What `token` holds for a library whose unregister takes something other
  than an integer. `Registration` is an ordinary `linear struct` a binding
  author writes, and the compiler knows nothing about it. That works for a
  library that hands back an integer. A library that hands back a pointer or a
  struct has not been tried.
- Whether a context that outlives its frame is worth supporting. Everything
  here confines the registration to the frame that holds the context, which
  covers a registration whose life is a scope and not one whose life is not. A
  pool's `Handle<T>` is the obvious way to lift that, and it is not obvious yet
  that any binding wants it.
- Reentrancy. Nothing here stops a callback from calling back into code that
  reaches the same context. Moving the context in means no *Frost* code holds it,
  which is the guarantee being claimed, and it is worth being precise that it is
  not a guarantee about the C library's own threading.
