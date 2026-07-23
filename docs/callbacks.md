# Callbacks with a typed context

This is the design and the record of building it. It is item 3 in
[roadmap.md](roadmap.md), and the same method that produced
[separate-compilation.md](separate-compilation.md) is used here: work the design
against the code that exists until it either survives or does not, before
writing any of it. All five steps at the bottom are built, and a Frost handler
with a Frost context now runs through a real C callback API.

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

The design assumed the compiler would emit a trampoline per
`(handler, context type)` pair, holding the one cast from the untyped userdata
back to `^Ctx` so that nobody has to write it. **Building it showed the cast does
not have to happen at all.** A `mut` parameter is already a pointer in the
signature, and Frost and C share a calling convention, so `on_event` compiled for
Frost *is* the `void (*)(void*, int64_t)` the library wants. What the compiler
does at a registration is pass the handler's address and the context's address.
There is no generated code and no cast anywhere in the program.

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
> the registration.

The first answer written here was that the context therefore has to live in an
arena or a pool, and a place in the current frame is rejected. **That answer is
wrong, and it is wrong in a way worth recording**, because it does not survive
contact with the language it is a rule for. A context is a value of a struct
type, and a value lives where it is bound; putting one in an arena means holding
a `^Ctx`, and then the registration's context parameter is a pointer rather than
a moved value and the ownership argument above evaporates. The rule would have
rejected every program anyone could write.

**The obligation is satisfied from the other end.** A `Registration` is `linear`,
so `check_linearity` already forces it to be consumed exactly once in the
function that made it. A context in that same frame therefore outlives the
registration by construction, and the frame is exactly the right place for it.
What is left to stop is the registration *leaving* that function by some other
road, which is the same shape `src/regions.rs` already enforces for pointers:
returned, stored where the call cannot see, or handed back as the call's answer.

So the rule is not a new kind of check. A registration whose context is rooted in
this frame counts as a value that points into this frame, and the three roads out
are closed by the code that was already closing them. Linearity closes the
fourth, which is not consuming it at all.

That is what makes this different from the C idiom rather than a prettier
spelling of it. Without it the trampoline is type-safe and the program still has
a dangling pointer.

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
2. **Close the roads out.** *Done.* `check_frame_escapes` now treats a
   registration whose context is rooted in this frame as a value that points
   into this frame, so it cannot be returned, stored where the call cannot see,
   or be the call's answer. `callback_registrations` in `src/callbacks.rs` is
   what tells it which calls those are and which argument is the context.

   This lands before anything is emitted, so there is a window where the feature
   can only say no, which is the right way round. It was checked in both
   directions: a registration returned out of the frame holding its context is
   rejected, and the shape the design is for, registering and unregistering in
   one frame, gets past every check the language has and stops only at lowering,
   which is steps 3 and 4. Emptying the registration table makes the first of
   those stop failing, which is the confirmation that the check is what catches
   it rather than something downstream.
3. **Emit the trampoline.** *There is no trampoline, and finding that out is the
   whole of this step.* The plan was one function per `(handler, context type)`
   pair holding the one `ptr_cast` nobody should have to write. But the
   handler's context parameter is `mut`, so it is already a pointer in the
   signature, and a Frost function and a C function use the same calling
   convention. `on_event` compiled for Frost is bit for bit the
   `void (*)(void*, int64_t)` the library wants. The cast the design set out to
   hide inside generated code does not exist, so there is nothing to generate
   and nothing to dedup.
4. **Lower the call.** *Done.* The `$handler` argument becomes
   `IrRvalue::FunctionAddress` of the handler itself, and the context argument
   becomes its address. An extern's C-facing parameter types are computed in
   `extern_parameter_types` in `src/ir_build.rs`, since a registration's C
   signature is not what its Frost declaration says literally. Everything else
   about the extern call is unchanged, on both backends.
5. **Run one.** *Done.* `a_callback_registered_with_a_c_library_runs` compiles a
   small C library that stores a `(callback, userdata)` pair and calls it back
   later, links it, and registers a Frost handler with a Frost context against
   it. The handler runs, writes the Frost struct through the pointer the library
   kept, and the library reads the updated value back out. That is the claim in
   step 3 being true rather than argued: if the ABI did not line up, this is
   where it would crash.

Steps 1 and 2 are parsing and a check and landed before anything was emitted,
which is the order that let the safety rule be found to be wrong while it was
still cheap to change.

## What is still open

- **How the caller gets its context back, and it is the big one.** The context
  goes in by `move`, so the name it was bound to is dead to `check_ownership`
  for the rest of the function, and the callback's updates are written into that
  exact storage. So the caller cannot read what its own callback did. The C
  library in the end-to-end test reads the value back out, which proves the ABI
  but is not a program anyone wants to write.

  The roadmap's sketch had `unregister :: fn(move r: Registration) -> Ctx`, and
  that does not work as written: a `Registration` holding a `Ctx` field holds a
  copy, and the copy is not the storage the library wrote through. The answer is
  probably that unregistration is a recognized form the way registration is, and
  gives the context back by name rather than by value. Nothing here is blocked
  on it: the callback runs and the safety rules hold. But the feature is not
  finished until a program can read its own context.
- **What `token` holds** for a library whose unregister takes something other
  than an integer. The `Registration` in the end-to-end test is an ordinary
  `linear struct` a binding author writes and the compiler knows nothing about
  it, which is the answer this document prefers and which held up for a library
  returning an integer. A library that hands back a pointer or a struct has not
  been tried.
- **Whether a context that outlives its frame is worth supporting.** Everything
  here confines the registration to the frame that holds the context, which
  covers a registration whose life is a scope and not one whose life is not. A
  pool's `Handle<T>` is the obvious way to lift that, and it is not obvious yet
  that any binding wants it.
- **Reentrancy.** Nothing here stops a callback from calling back into code that
  reaches the same context. Moving the context in means no *Frost* code holds it,
  which is the guarantee being claimed, and it is worth being precise that it is
  not a guarantee about the C library's own threading.
