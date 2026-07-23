# Callbacks with a typed context

This is the design and the record of building it. It is item 3 in
[roadmap.md](roadmap.md), and the same method that produced
[separate-compilation.md](separate-compilation.md) is used here: work the design
against the code that exists until it either survives or does not, before
writing any of it. Step 1 of the five at the bottom is built; the rest is
design.

## The contradiction it exists to remove

Goal 2 in [philosophy.md](philosophy.md) says safety comes from making dangerous
shapes unrepresentable. The only way to write a callback today is the C idiom: a
function pointer plus an untyped `^u8` the callee casts back. Every piece of that
already exists in the language, verified against the source rather than
remembered:

- `fn(T1, ...) -> R` is a function pointer type (spec 3.5), and a named function
  used as a value lowers to `IrRvalue::FunctionAddress` in `src/ir.rs`.
- `ptr_cast($T, p)` reinterprets a pointer at no runtime cost (spec 3.3).
- `^T` carries no guarantee once formed, which the spec says outright.

So a callback is writable and it is entirely outside every check the language
has. `src/regions.rs` reasons about arena pointers by provenance, and its
argument is stated in its own header comment: "Frost has no global arenas and no
closures, so a `^T` can only point into an arena a function was handed directly."
A `^u8` handed to a C library and called back through later is precisely the case
that argument does not cover. `src/ownership.rs` cannot see through it either.

The result is that every callback-shaped API in Frost is an unsafe API, not
because callbacks are unsafe but because the only expression of one is a raw
escape hatch. That is the inversion the surface `&` removal was meant to prevent,
reappearing at the C boundary.

## The shape

Closures stay a non-goal. Capture is not the answer; a context written down is.
A callback is a compile-time function argument plus a typed context the caller
owns:

```
Ctx :: struct { hits: i64 }

on_event :: fn(mut ctx: Ctx, code: i64) { ctx.hits = ctx.hits + code }

register :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64
```

The compiler emits a trampoline per `(handler, context type)` pair with the C ABI
the library expects, and that trampoline is the only code that casts the `^u8`
back to `^Ctx`. The cast still happens, because C requires it, but it is
generated once per instantiation inside code nobody writes, rather than appearing
at every call site in code everybody reads.

## Three questions the roadmap left, and the answers

### Is the spelling `uses CallbackAbi`, or is it inferred?

**Inferred, and `uses CallbackAbi` is dropped.** A `$handler` parameter with a
function bound on an `extern fn` is already the complete statement of "this
extern wants a trampoline". A `uses` clause beside it would be a second thing
that has to be kept in step with the first, and it would be a capability that
grants nothing: `uses Arena` means a real implicit parameter is supplied at the
call, which `src/allocation_sources.rs` inserts. A capability that supplies
nothing is a keyword pretending to be a capability.

This also settles which form of compile-time parameter it is. Not `$handler:
Type`, which says only "some type", but the bound form from item 2 of the
roadmap, `$handler: fn(mut Ctx, i64)`, so the handler's signature is checked
against what the library expects at the call, by the code already in
`src/ir_build.rs` that checks compile-time signatures.

### What does the extern's C signature become?

A `$handler` parameter contributes exactly one C argument, the trampoline
pointer, in the position it is written. The context contributes the `void*`, in
the position it is written. So the declaration is written in the order C wants
and the mapping needs no further rule:

```
register :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> i64
```

becomes `int64_t register(void (*)(void*, int64_t), void*)`.

**Which parameter is the context** is not positional and must not be, because
libraries put the userdata on either side of the function pointer. It is the
parameter whose type is the type of the handler's **first** parameter. That is
also the definition that makes the trampoline derivable: the handler's first
parameter is the context, and every parameter after it is a callback argument
that C passes through.

A declaration where no parameter has that type is an error at the declaration,
not at the call. So is a handler whose first parameter is not `mut`: a callback
that cannot write its context is a callback that cannot do anything, and reading
one is the case a plain function pointer with no context already covers.

### Who owns the context?

Registration moves it in, unregistration moves it back out, and the registration
is a `linear` value. Not a borrow.

```
Ctx          :: struct { hits: i64 }
Registration :: linear struct { token: i64 }

on_event   :: fn(mut ctx: Ctx, code: i64) { ctx.hits = ctx.hits + code }
register   :: extern fn($handler: fn(mut Ctx, i64), move ctx: Ctx) -> Registration
unregister :: fn(move r: Registration) -> Ctx
```

Three things fall out, and each is a reason to prefer moving over borrowing.

- **No new machinery.** A borrow that outlives its call would be the first thing
  in the language that does, and inventing it means inventing the region
  annotation the whole design is built on not having. Moving needs nothing new:
  `check_ownership` already stops the caller touching a moved value, and
  `check_linearity` already forces a `linear` value to be consumed exactly once.
- **The aliasing guarantee is the one you want.** While registered, the callback
  may fire at any moment, so the caller must not be reading or writing the
  context. Having moved it in, the caller cannot.
- **Forgetting to unregister becomes a compile error**, which is a real bug class
  in every C callback API, and a dangling callback into a freed context is the
  exact failure this is meant to prevent.

The fire-and-forget case, where the library never hands the callback back, does
not get an exception. The registration is still linear, and a program that means
to abandon it says so with a terminal consumer that takes it and returns nothing.
"I am deliberately leaking this" is worth having to write.

## What the design as written down does not survive

The three answers above are the easy part. Working them against the code turns up
one thing that the roadmap's version of this design does not handle, and it is
the thing that decides whether the feature is safe or merely tidier.

**Where does the context live while the callback can fire?**

`move ctx: Ctx` hands the value to the extern, and the extern keeps a pointer to
it. So the storage the pointer names has to outlive the call, and a moved
argument today is a value in the caller's frame. `src/regions.rs` and
`check_frame_escapes` between them already reject a pointer into the current
frame being returned, stored into a parameter, or carried out inside a struct.
They do not reject one being *handed to an extern that keeps it*, because nothing
in the language could keep it until now.

So the feature adds exactly one obligation, and it is the whole safety argument:

> The context argument of a callback registration must name storage that outlives
> the registration. A place in the current frame does not.

Which means the context has to come from somewhere with a longer life, and Frost
already has the two somewheres: an arena entered with `with`, whose region rule
is enforced in `src/regions.rs`, or a pool, whose `Handle<T>` is a copy value
that may be stored and returned by design (spec 3.4). Both are checkable with the
machinery that exists. A frame temporary is not, and is rejected.

That obligation is what makes this different from the C idiom rather than a
prettier spelling of it. Without it the trampoline is type-safe and the program
still has a dangling pointer.

**A second, smaller consequence.** The registration is linear and the context is
inside the region, so `unregister` returning the context by value has to be a
move out of the region and into the caller's, which is the ordinary case the
region check already covers. But a `Registration` that outlives its `with` block
would strand a pointer into a dead arena, so `Registration` must be region-bound
in the same way a `^T` into that arena is. That is not a new rule either: it is
`src/regions.rs` treating a linear registration as arena-derived, which it can
do because provenance is a flow question and the registration is produced from
the context.

## The order to build it in

Each step is meant to be landable on its own, against the differential oracle and
both self-hosting fixpoints, in the way the separate-compilation steps were.

1. **Parse the declaration.** *Done.* `$handler: fn(...)` and parameter modes
   are accepted on an `extern fn`, and `src/callbacks.rs` checks at the
   declaration that the handler's first parameter is the context and written
   `mut`, that some parameter of the extern has that type, and that it is taken
   by `move`. This comes first, and not the safety check, because nothing can
   tell a callback registration from any other extern call until the declaration
   says so, which is the ordering the first draft of this list got backwards.

   Two things it turned up.

   **A function type could not say a mode.** `fn(T1, ...) -> R` parsed types and
   nothing else, so the bound `fn(mut Ctx, i64)` could not be written at all: a
   `mut` parameter is a reference in the signature and the surface deliberately
   has no way to write a reference type. `mut` is now a marker inside a function
   type and means the reference the mode means. Unmarked stays the type as
   written, which is what the spec says a function type is and what every
   existing bound already means; changing that would have broken
   `$before: fn(T, T) -> bool` in the same edit.

   **The ownership guarantee cost nothing**, which is the claim above being
   true rather than merely argued. `src/param_modes.rs` and `src/ownership.rs`
   already read an extern's parameter list the same way they read a function's,
   so `move ctx: Ctx` on an extern makes the argument a move with no further
   work, and a program that registers a context and then reads it is rejected
   by the pass that was already there.
2. **Reject the unsafe case.** Extend `check_frame_escapes` so that a place in
   the current frame passed as the context of a registration is an error. This
   is the whole safety argument and it lands before anything is emitted, so
   there is a window where the feature can only say no, which is the right way
   round. Breakable on purpose to confirm it fails.
3. **Emit the trampoline.** One function per `(handler, context type)` pair, with
   `IrFunction::local` set, since it is called only from the object that
   registered it, exactly as a specialization is. Its body is one `ptr_cast` and
   one call. The naming and the dedup are the same worklist that specializations
   already use in `src/ir_build.rs`.
4. **Lower the call.** The `$handler` argument becomes
   `IrRvalue::FunctionAddress` of the trampoline; the context argument becomes a
   pointer to its storage. Everything else about the extern call is unchanged.
5. **Make the registration region-bound.** Teach `src/regions.rs` that a value
   produced from an arena-derived context is itself arena-derived, so a
   registration cannot outlive the region its context lives in.

Steps 1 and 2 are parsing and a check and land before anything is emitted.
Step 5 is the one that is easy to declare done without being done, so it wants
the same treatment the interface closure check got: break it deliberately and
confirm it fails before trusting it.

## What is still open

- **What `token` holds** for a library whose unregister takes something other
  than an integer. The `Registration` above is an ordinary `linear struct` a
  binding author writes, so it can hold whatever the library hands back, and the
  only real question is whether the compiler needs to know anything about it at
  all. On the design above it does not, which is the answer this document
  prefers, but no binding has been written against it yet.
- **Whether a pool context is worth supporting in the first version.** An arena
  context is the smaller case and covers a registration whose life is a scope. A
  `Handle<T>` context covers one whose life is not, and it is not obvious yet
  that any binding wants it.
- **Reentrancy.** Nothing here stops a callback from calling back into code that
  reaches the same context. Moving the context in means no *Frost* code holds it,
  which is the guarantee being claimed, and it is worth being precise that it is
  not a guarantee about the C library's own threading.
