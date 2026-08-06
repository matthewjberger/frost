# 12. The foreign function interface

An `extern fn` declares a function with C linkage:

```frost
printf :: extern fn(fmt: ^i8, value: i64) -> i32
malloc :: extern fn(size: i64) -> ^u8
```

Frost scalar types map to the natural C types and `^T` is a C pointer. String
literals denote NUL-terminated bytes for `^i8` parameters. An `extern` takes
parameter modes like any other function.

## 12.1 Nothing is watching the signature

The compiler never sees the C header. It takes the declaration as given, and a
declaration that does not match the C one produces a program that compiles,
links, runs, and is wrong. There is no diagnostic for this and there cannot be
one. If you have read the rest of this book and formed the habit of expecting
the compiler to catch you, drop it here: **the boundary is the one place in
Frost where being wrong is silent.**

The mistake that costs the most is the one in 12.3: omitting `value` where the
C function takes a struct by value. The callee then reads a pointer as though it
were the struct.

What does help:

- Generate the binding from the header rather than writing it out. Then the
  question moves to keeping the generator in step, which is a thing you can
  test.
- Run every declaration at least once. A binding that compiles proves nothing.
  Four ABI bugs in this compiler's own history were invisible until something
  executed the call.

## 12.2 Calling one needs `unsafe`

An `extern` is arbitrary C, so a call to one is refused outside an `unsafe`
block, alongside the other two operations the language cannot check:

```frost,sketch
count :: fn(text: ^i8) -> i64 {
    unsafe { strlen(text) }
}
```

A declaration whose signature cannot corrupt memory whatever it is handed is
written `safe extern fn`, and calls to it need no block:

```frost
sqrtf :: safe extern fn(x: f32) -> f32
```

The judgement is made once, at the declaration, and it is about the signature
rather than the function. `sqrtf` takes a float and answers a float, so no
argument it can be given reaches memory. `malloc :: safe extern fn(size: i64)
-> ^u8` is also sound, because it is reading through the pointer it answers with
that is unchecked, and that read is gated where it happens. Anything taking a
`^T` it will read through, or a length beside a pointer, is not `safe`. See
[6a.4](unsafe.md#6a4-safe-extern-fn).

## 12.3 Aggregate parameters, and the mistake to avoid

Aggregate parameters and aggregate returns are not symmetric.

**A parameter written `value` is passed as C passes a struct**, following the
target's real ABI, so `label :: extern fn(value v: View)` links against a C
`void label(View)`. `value` is a word rather than a keyword and is only a mode
where a mode can appear, so a parameter may still be named `value`. It says how
the bytes cross rather than what the caller gives up: C receives a copy and the
argument is borrowed, exactly as an unmarked one is.

**An unmarked aggregate parameter is passed as a pointer to the value**, so
`close :: extern fn(f: File)` links against a C `void close(File*)`. This is a
convention rather than the C ABI. It is what lets a `linear` resource have a
terminal consumer across the boundary, and it suits the older style of C API
that takes a context struct by address.

That default is not a claim about C APIs in general. In the largest binding
measured against it, 356 declarations covering the whole surface of a game
engine, 206 take at least one aggregate parameter and **none of them wanted the
default**: handles, vectors, tagged unions and wire structs of 8 to 40 bytes,
which is exactly the range a modern C ABI passes in registers. Treat `value` as
the common case and the default as the one for `linear` resources and for APIs
written in the older style.

So, plainly:

> If the C declaration takes a struct by value, the Frost declaration needs
> `value`. Getting this wrong is not caught by anything: it compiles, it links,
> it runs, and the callee reads a pointer as though it were the struct.

**An aggregate return is by value**, following the target's real C ABI: in
registers where that target's rule says so, and through a hidden pointer where
it does not. A return could not have been a convention, because `-> Ctx` has to
mean what C means by it and `-> ^Ctx` is how a returned pointer is written.

## 12.4 Which integer for a length

`size_t`, `uintptr_t` and `ptrdiff_t` are all one machine word on every target
Frost supports, and `i64`, `isize`, `u64` and `usize` are all eight bytes passed
identically. So the choice cannot break the call, and it is not an ABI question.

**Use `i64`.** Everything in Frost that is a length already is one: `slice_len`
answers `i64`, `sizeof` answers `i64`, and an arena's count is `i64`. Declaring
a length as `usize` does not cost you a cast, since Frost's scalar types convert
freely, but it does mean a value that flows between the two starts changing
which arithmetic it gets.

That is the real difference, and it is a semantic one rather than a layout one.
Comparison, division, remainder and right shift follow the type: `u64` and
`usize` get the unsigned form and everything else gets the signed one, matching
what C does with the same types. That only diverges past 2^63, which no real
length reaches, so for lengths the two are interchangeable and `i64` is the one
that matches the rest of the language.

Reserve `u64` and `usize` for a value that is genuinely an unsigned number and
can pass 2^63: a bit pattern, a hash, a raw address you are doing arithmetic on.
There, the type is not decoration, it is what makes `>` and `/` mean the right
thing.

## 12.5 Linking the library

A declaration says what to call. `--libs` says what to link it against, and is
repeatable:

```
frost --link --libs "C:/SDL3/SDL3.dll" -o window.exe window.frost
frost --link --libs=-lSDL3 -o window window.frost
```

What `--libs` names is passed to the linker in the order written, after the
program's own objects and ahead of the platform's libraries, which is the order
a linker resolves in: a library that itself needs libm has to be seen before
libm is.

## 12.5a Going the other way: a Frost function C calls by name

The same declaration with a body is a definition rather than a declaration, and
what it adds is the name. A Frost function is emitted under a name the compiler
chose, so C cannot call it; one written this way is emitted under the name it
was written under:

```frost
frost_demo_double :: extern fn(value: i64) -> i64 {
    value * 2
}
```

That is `int64_t frost_demo_double(int64_t)` in the emitted C and
`.globl frost_demo_double` in the emitted assembly, from both compilers and all
four backends. An ordinary function beside it is emitted as `frost_u_double` or
`mf_7`, which is what keeps two Frost functions of the same name in different
modules apart.

The body is ordinary Frost, and calling one from Frost needs no `unsafe`: the
function is written here, so there is nothing unaudited about the call. What is
outside the language is only who else may name it.

This is what lets Frost supply a symbol something else already calls: a
callback a C library takes by name rather than by pointer, an entry point a
platform expects, and the compiler's own runtime.

Two name spaces are not yours to define into. A name beginning with `frost_rt_`
is the runtime's and one beginning with `frost_u_` is the compiler's, and a
definition keeping either would replace something every program calls, since the
runtime is linked into every program:

```frost,refused
frost_rt_check_index :: extern fn(index: i64, length: i64) -> i64 {
    index
}
```

> 'frost_rt_check_index' keeps the name it is written under, and 'frost_rt_' and
> 'frost_u_' are the runtime's and the compiler's own, so a definition here would
> replace what every program calls

The rule is about definitions, not declarations: `frost_rt_die :: safe extern
fn()` says the runtime has one and calls it, which is what a Frost program that
wants to end the way a failed check ends does.

## 12.6 Callbacks

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
