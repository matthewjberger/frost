# C compatibility

Frost has two relationships with C:

1. Frost calls C (`extern fn`), a supported feature. This is
   how Frost reaches `printf`, `malloc`, the support runtime, and any C
   library.
2. Frost lowers *through* C (`--emit-c`), an internal implementation detail
   of one backend. The emitted C is a compilation target the system C
   compiler reads.

Frost calls C, and C calling Frost is a non-goal, which leaves the emitted C a
simple lowering (char buffers, mangled names) under no obligation to a stable
ABI.

## 1. Frost calls C with `extern fn`

An `extern fn` declares a function implemented outside Frost, linked at build
time. It is available on both native backends (Cranelift and C).

```frost,sketch
printf :: extern fn(fmt: ^i8, value: i64) -> i32
malloc :: extern fn(size: i64) -> ^u8
free   :: extern fn(ptr: ^u8)

main :: fn() -> i64 {
    printf("%lld\n", 42)
    0
}
```

- Names are preserved. An `extern` symbol keeps its exact name (`printf`
  stays `printf`) so it links against the real C library. Only *non-extern*
  Frost functions are mangled (see below).
- Types map to the natural C ABI. Scalars map to their `<stdint.h>`
  equivalents (`i32` to `int32_t`, `u8` to `uint8_t`, `f64` to `double`), and
  a raw pointer (`^T`) and a borrowed parameter both map to pointers. An `extern`
  signature is therefore a direct description of the C function's ABI.
- Aggregate parameters are passed by pointer. A `struct`/`enum`/array
  parameter to an `extern fn` is passed as a pointer to the value. So
  `close :: extern fn(f: File)` links against a C `void close(File* f)`. This is
  a convention of Frost's, and it is how a `linear` resource's terminal consumer
  works natively: the `extern` takes ownership across the boundary, receiving a
  pointer to the moved-in aggregate. It suits the older style of C API that
  takes a context struct by address. A modern one takes a struct by value, which
  is the note under `value` below.
- Aggregate returns from an `extern` follow the real C ABI. An
  `extern fn(...) -> Ctx` returns whatever the target's C compiler returns:
  in registers when the rule says so and through a hidden pointer when it does
  not. Frost's own uniform out-pointer convention stops at the boundary.
  `src/c_abi.rs` has the three rules and what each was checked against.

  The two lines above answer differently. A struct parameter to an extern
  is a pointer by convention, a struct return is by value with the real ABI.
  A return could not have been a convention, because `-> Ctx` has to mean what C
  means by it and `-> ^Ctx` is how a returned pointer is written. A parameter had
  a choice, and a `linear` resource needs the pointer form.
- A parameter written `value` is passed to C the way C passes a struct.
  `set_label :: extern fn(handle: ^u8, value label: View)` links against
  `void set_label(void*, View)`, with the bytes split across registers or pushed
  on the stack by the same target rule the return uses. `src/c_abi.rs` has both
  classifications side by side.

  A modern binding wants `value` on nearly every aggregate. In the largest one
  measured, 356 declarations covering the whole surface of a game engine, 206
  take at least one aggregate parameter and every one of them wants `value`:
  handles, vectors, tagged unions and wire structs of 8 to 40 bytes, which is
  the range a modern C ABI passes in registers. Omitting `value` where
  C takes a struct by value is silent wrong code, and nothing in the program
  knows the C signature, so no diagnostic is possible.

  `value` is a contextual word, so a parameter may still be called `value`. A
  mode is followed by the name and a name is followed by its type, which tells
  the two apart. The mode says how the bytes cross. C receives a copy, so the
  caller still holds its own value and the argument is borrowed exactly as an
  unmarked one is.

  All four paths emit it: both of the bootstrap's backends and both of the
  self-hosted compiler's. The two C backends hand the C compiler a real struct
  type and let it apply the rule. The two that generate code have to know the
  rule, so `src/c_abi.rs` classifies for Cranelift and the same classification
  is written out again in `selfhosted/emit_asm.frost`, where one argument
  becomes the one or several slots the target wants.

  The three shapes an argument takes, and the whole rule:

  | | Windows | System V |
  | --- | --- | --- |
  | 1, 2, 4 or 8 bytes | one integer register, whatever it holds | by eightbyte |
  | up to 16 bytes | address of a copy the caller makes | one or two registers, each integer or SSE by what reaches it |
  | over 16 bytes | address of a copy the caller makes | pushed onto the stack |

  C gives the callee its own parameter, so a callee that writes to it must not
  write through to the caller's value, and the test that says so passes a struct
  to a function that assigns to its parameter and then reads the caller's copy
  back.
- A function type says the same thing about its own parameters, so
  `fn(i32, value View, i64)` is a callback C hands a struct to, and a Frost
  function written `handler :: fn(status: i32, value message: View, tail: i64)`
  is compiled to receive one. That is the other direction: the caller takes the
  struct apart, the callee puts it back together, and neither needs a
  trampoline, which is the same claim [callbacks.md](../design/callbacks.md) makes about
  the simpler shapes.

  wgpu's callbacks need this. Declared as one pointer instead, a sixteen-byte
  struct matches what Windows hands a callee and reads every argument after it
  out of the wrong register on System V.
- Freestanding is a separate axis. Everything on this page is about calling
  C and about the C backend. Whether the *executable* needs libc once it is
  running is a different question, answered by `--freestanding`. See
  [build-modes.md](build-modes.md).
- The linker gets a real C compiler. Both backends finish by invoking
  `cc`/`gcc`/`clang` (or `cl` on MSVC), so C symbols resolve normally and you can
  pass extra libraries with `--libs`.

`extern fn` reaches libc, the OS syscalls and any third-party C library, with no
FFI glue code.

### An `extern` call is gated, and `safe extern fn` is how a binding lifts it

Calling an `extern fn` is refused outside an `unsafe` block. The compiler has
checked the Frost side of the call and nothing whatever about the other side, so
what the declaration says the C function does is a claim, and the block is where
someone writes that claim down. [unsafe.md](../reference/unsafe.md) has the rule
and the rest of what the gate covers.

`safe extern fn` marks one declaration audited, and calls to it need no block.
The audit is about the signature. `sqrtf :: safe extern fn(x: f32) -> f32` in
`std/math.frost` takes and returns a number and touches no memory of the
caller's, so there is nothing a call site could get wrong that the type checker
has not already caught. `malloc :: safe extern fn(size: i64) -> ^u8` in
`selfhosted/core.frost` is safe for a different reason: it hands memory back and
reads none of the caller's, so it cannot corrupt what the caller holds. It still
returns a raw pointer, and reading through one is gated on its own.

A declaration taking a pointer usually cannot be marked safe, because the
callee's read is bounded by something the signature does not say.
`frost_rt_emit_bytes :: extern fn(data: ^u8, length: i64)` stays gated for that
reason, and its one caller hands it a `str` whose length it already knows, so
one `unsafe` at that call covers every emit.

When every declaration in a binding file is either `safe` or reached through a
wrapper that establishes what the C side needs, that file is a perimeter: a
program using the binding writes no `unsafe` of its own, and the file is the
complete list of places to look when memory is corrupted. The generated wgpu
binding is written that way, with a safe wrapper per call, so a program that
draws a triangle writes none for the graphics API.

### The support runtime is two files, and most of it is Frost

Every program links a runtime.

`runtime/runtime.frost` is Frost. It holds the checks a program compiles to: the
bounds check an index becomes, the length and span checks a slice becomes, the
size check an allocation becomes, the generation check a handle becomes, the trap
arithmetic that cannot answer ends on, and the byte and memory helpers the
standard library reaches for. Each function there is written `extern fn` with a
body, so it is emitted under the name it was written under and emitted code
calls it by that name (12.5a). Both compilers compile it with their own front
end into a cached object and link it beside the C one.

Two rules hold that file. Nothing in it may need what it provides: a bounds
check that indexes something calls itself, so nothing there indexes an array and
a number is written a digit at a time, with no buffer to index. And it bottoms
out in the C file beside it, reached through a handful of `extern` declarations
that are the whole of its contact with C.

`runtime/frost_runtime.c` holds the rest, and this is the size it stays.

Almost all of it is built around a variable that lives for the whole program and
that every call sees the same copy of: the emit buffer `-o` writes through, the
counters `--test` sums, the recovery stack a parse escapes to, the block count a
leak check reads, the arguments a constructor captures before `main`. Frost has
constants and locals and nothing at module scope that a function writes to,
because a value's lifetime is a place in the program. The region check and the
move checker are built on that rule: a `^T` points into an arena in
scope, and a place belongs to a frame. Adding a variable that outlives every
frame would give both of them a second case, paid for by every program, to serve
these few functions. So they stay here.

Beside them sit the `setjmp`/`longjmp` escapes, whose `setjmp` has to own its own
call frame; the stack-guard handlers, which are platform APIs; and the three
functions that are an `#if` on the target. Each says why it is there.

Where a runtime function's state has an explicit form, that is the direction to
take it, because it removes an implicit global from *programs*. Output is a
stream something names. A block count belongs to the allocation source that
handed the block out. The arguments are an input to the program. Each of those
is a language improvement first and a smaller runtime second.

The pool is in neither: it is written in Frost as an ordinary library, so nothing
in the runtime allocates or owns one. See
[pools-and-columns.md](../design/pools-and-columns.md). Programs reach the
runtime through the same `extern fn` mechanism:

```frost
frost_rt_bounds_check :: extern fn(index: i64, length: i64)
frost_rt_assert       :: extern fn(cond: bool)
frost_rt_read_file    :: extern fn(path: ^i8) -> ^i8
```

Its interface is scalar-only. Nothing is passed or returned by aggregate value,
so the runtime's *natural* C ABI matches Frost's internal aggregate convention
with no negotiation, and the same compiled runtime links into both backends,
which agree bit for bit.

The memory model is a library. A slab is a Frost struct with Frost operations
over it (`std/slab.frost` in the standard library, and
`examples/native/generic_slab.frost` for one written out in full), which is why
fixed-capacity storage works under `--freestanding` where there is no libc at
all. C holds the aborts, the assertions and the IO.

## 2. Frost lowers through C with `--emit-c`

`--emit-c` selects the portable-C backend instead of Cranelift. It emits a
single `.c` file and compiles it with the system C compiler. This exists for
portability (anywhere with a C compiler) and as the second half of the
differential oracle. Every test program is compiled through *both* Cranelift
and C, run, and the outputs are asserted equal.

The emitted C is an internal lowering, and it looks like one:

- Aggregates are byte buffers. A struct/enum/array local is emitted as
  `_Alignas(16) unsigned char _7[N];` and accessed through pointer casts, with
  no named C `struct` for it. This is why a Frost struct type's *name* is only
  ever a layout-registry key inside the compiler. It never has to be a valid C
  identifier, so monomorphized names like `Pair<i64>` work with no extra
  escaping.
- Aggregate returns use a hidden out-pointer. A Frost function returning a
  struct compiles to `void f(..., char* __ret)` and `memcpy`s the result into
  `__ret`. An `extern` follows the real C ABI instead, as section 1 sets out.
- Non-extern names are mangled. Every Frost function that isn't `extern` and
  isn't `main` is prefixed (`frost_`) so it can never collide with a C keyword or
  library symbol. `extern` names and `main` are left untouched so FFI and the
  entry point link.
- Function prototypes are emitted up front, so forward references and mutual
  recursion compile regardless of definition order.

With C calling Frost a non-goal, the backend picks whatever lowering is simplest
and fastest to emit. Stable C-callable exports would be a separate, opt-in
surface, leaving the internal lowering free to change.

## What "C compatible" means here

| Direction        | Supported? | Mechanism                                            |
| ---------------- | ---------- | ---------------------------------------------------- |
| Frost calls C    | Yes        | `extern fn`, natural C ABI, real linker              |
| Frost links C    | Yes        | support runtime + `--libs`, compiled and linked by `cc` |
| Frost emits C    | Yes        | `--emit-c`, an internal lowering / differential oracle |
| C calls Frost    | No (non-goal) | emitted C is mangled internal detail, not an API   |

Frost calls C going out and uses C as a portable assembler going down. It
promises C nothing coming in.

## Building

```
frost program.frost --link -o program            # Cranelift backend, links an executable
frost program.frost --emit-c --link -o program   # C backend, same result via emitted C
frost program.frost --emit-c -o program.c         # just emit the C, don't link
frost program.frost --link -o program --libs -lm  # link extra libraries
frost program.frost --link --incremental -o program  # rebuild only what changed
```

Both `--link` paths compile and link both halves of the runtime on their own, so
the bounds and generation checks, the assertions and the IO helpers are there
with no extra flags. `FROST_RUNTIME` and `FROST_RUNTIME_FROST` say where they
are for a checkout you are not standing in.
