# C compatibility

Frost has two distinct relationships with C, and it is important to keep them
separate:

1. Frost calls C (`extern fn`), a supported feature. This is
   how Frost reaches `printf`, `malloc`, the pool runtime, and any C library.
2. Frost lowers *through* C (`--emit-c`), an internal implementation detail
   of one backend. The emitted C is a compilation target, not an interface
   for external C code to call into.

The design is deliberately asymmetric. Frost calling C matters, and C calling
Frost does not. Keeping that asymmetry is what lets the emitted C stay a simple
lowering (char buffers, mangled names) without owing anyone a stable ABI.

## 1. Frost calls C with `extern fn`

An `extern fn` declares a function implemented outside Frost, linked at build
time. It is available on both native backends (Cranelift and C).

```frost
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
  parameter to an `extern fn` is passed as a pointer to the value, not
  by-value-in-registers. So `close :: extern fn(f: File)` links against a C
  `void close(File* f)`. This matches the common C convention (most APIs take
  structs by pointer) and is how a `linear` resource's terminal consumer works
  natively. The `extern` takes ownership across the boundary, receiving a
  pointer to the moved-in aggregate.
- Aggregate returns from an `extern` follow the real C ABI. An
  `extern fn(...) -> Ctx` returns whatever the target's C compiler returns:
  in registers when the rule says so and through a hidden pointer when it does
  not. Frost's own uniform out-pointer convention is not imposed on C, because C
  does not use it. `src/c_abi.rs` has the three rules and where they were checked
  against, and item 4 of [roadmap.md](roadmap.md) has the table.

  Note the asymmetry with the line above it. A struct parameter to an extern
  is a pointer by convention, a struct return is by value with the real ABI.
  A return could not have been a convention, because `-> Ctx` has to mean what C
  means by it and `-> ^Ctx` is how a returned pointer is written. A parameter had
  a choice, and passing by pointer is what most C APIs want.
- A parameter written `value` is passed to C the way C passes a struct.
  `set_label :: extern fn(handle: ^u8, value label: View)` links against
  `void set_label(void*, View)`, with the bytes split across registers or pushed
  on the stack by the same target rule the return uses. `src/c_abi.rs` has both
  classifications side by side.

  `value` is a word rather than a keyword, so a parameter may still be called
  `value`; what tells them apart is that a mode is followed by the name and a
  name is followed by its type. It says how the bytes cross, not what the caller
  gives up: C receives a copy, so the caller still holds its own value and the
  argument is borrowed exactly as an unmarked one is. That copy is real, and a
  callee that writes to its parameter is writing to its own.

  All four paths emit it: both of the bootstrap's backends and both of the
  self-hosted compiler's. The two C backends hand the C compiler a real struct
  type and let it apply the rule. The two that generate code have to know the
  rule, so `src/c_abi.rs` classifies for Cranelift and the same classification
  is written out again in `selfhosted/emit_asm.frost`, where one argument
  becomes the one or several slots the target wants.

  The three shapes an argument takes, which is the whole of the rule:

  | | Windows | System V |
  | --- | --- | --- |
  | 1, 2, 4 or 8 bytes | one integer register, whatever it holds | by eightbyte |
  | up to 16 bytes | address of a copy the caller makes | one or two registers, each integer or SSE by what reaches it |
  | over 16 bytes | address of a copy the caller makes | pushed onto the stack |

  The copy matters. C gives the callee its own parameter, so a callee that
  writes to it must not write through to the caller's value, and the test that
  says so passes a struct to a function that assigns to its parameter and then
  reads the caller's copy back.
- A function type says the same thing about its own parameters, so
  `fn(i32, value View, i64)` is a callback C hands a struct to, and a Frost
  function written `handler :: fn(status: i32, value message: View, tail: i64)`
  is compiled to receive one. That is the other direction: the caller takes the
  struct apart, the callee puts it back together, and neither needs a
  trampoline, which is the same claim [callbacks.md](callbacks.md) makes about
  the simpler shapes.

  This is what wgpu's callbacks need. Without it the struct had to be declared
  as one pointer, which is what Windows hands a sixteen-byte struct to a callee
  as, so it worked there and read every argument after it out of the wrong
  register on System V.
- Freestanding is a separate axis. Everything on this page is about calling
  C and about the C backend. Whether the *executable* needs libc once it is
  running is a different question, answered by `--freestanding`. See
  [build-modes.md](build-modes.md).
- The linker gets a real C compiler. Both backends finish by invoking
  `cc`/`gcc`/`clang` (or `cl` on MSVC), so C symbols resolve normally and you can
  pass extra libraries with `--libs`.

This is the interop that carries real weight. Frost programs get the entire C
ecosystem (libc, OS syscalls, third-party libraries) through `extern fn`, with
no FFI glue code.

### The pool runtime is itself just linked C

The generational pool runtime (`runtime/frost_runtime.c`) is an ordinary C file
that both backends link automatically. Programs reach it through the same
`extern fn` mechanism:

```frost
frost_rt_bounds_check :: extern fn(index: i64, length: i64)
frost_rt_assert       :: extern fn(cond: bool)
frost_rt_read_file    :: extern fn(path: ^i8) -> ^i8
```

Its interface is intentionally scalar-only. Nothing is passed or returned by
aggregate value, so the runtime's *natural* C ABI matches Frost's internal
aggregate convention with zero negotiation. That is also why the identical
compiled runtime links into both backends and they agree bit for bit.

The memory model is not in here. A slab is a Frost struct with Frost
operations over it (`examples/native/lib/slab.frost`), which is why
fixed-capacity storage works under `--freestanding` where there is no libc at
all. What C holds is bounds and generation aborts, assertions, and the IO
helpers.

## 2. Frost lowers through C with `--emit-c`

`--emit-c` selects the portable-C backend instead of Cranelift. It emits a
single `.c` file and compiles it with the system C compiler. This exists for
portability (anywhere with a C compiler) and as the second half of the
differential oracle. Every test program is compiled through *both* Cranelift
and C, run, and the outputs are asserted equal. Two independent backends that
must agree catch miscompilations that a single backend would hide.

The emitted C is an internal lowering, and it looks like one:

- Aggregates are byte buffers. A struct/enum/array local is emitted as
  `_Alignas(16) unsigned char _7[N];` and accessed through pointer casts, not as
  a named C `struct`. This is why a Frost struct type's *name* is only ever a
  layout-registry key inside the compiler. It never has to be a valid C
  identifier, which is what lets monomorphized names like `Pair<i64>` work with
  no extra escaping.
- Aggregate returns use a hidden out-pointer. A Frost function returning a
  struct compiles to `void f(..., char* __ret)` and `memcpy`s the result into
  `__ret`. An `extern` is different, and deliberately so. See below.
- Non-extern names are mangled. Every Frost function that isn't `extern` and
  isn't `main` is prefixed (`frost_`) so it can never collide with a C keyword or
  library symbol. `extern` names and `main` are left untouched so FFI and the
  entry point link.
- Function prototypes are emitted up front, so forward references and mutual
  recursion compile regardless of definition order.

Because of the mangling, the byte-buffer aggregates, and the out-pointer return
convention, the emitted C is not a clean header you would hand to a C
programmer. That is intentional. Since C calling Frost is a non-goal, the backend
is free to pick whatever lowering is simplest and fastest to emit. If stable
C-callable exports ever become a goal, they would be a separate, opt-in surface
rather than a property the internal lowering has to preserve.

## What "C compatible" means here

| Direction        | Supported? | Mechanism                                            |
| ---------------- | ---------- | ---------------------------------------------------- |
| Frost calls C    | Yes        | `extern fn`, natural C ABI, real linker              |
| Frost links C    | Yes        | pool runtime + `--libs`, compiled and linked by `cc` |
| Frost emits C    | Yes        | `--emit-c`, an internal lowering / differential oracle |
| C calls Frost    | No (non-goal) | emitted C is mangled internal detail, not an API   |

Frost speaks C fluently going out and uses C as a portable assembler
going down, but it does not promise C anything coming in.

## Building

```
frost program.frost --link -o program            # Cranelift backend, links an executable
frost program.frost --emit-c --link -o program   # C backend, same result via emitted C
frost program.frost --emit-c -o program.c         # just emit the C, don't link
frost program.frost --link -o program --libs -lm  # link extra libraries
frost program.frost --link --incremental -o program  # rebuild only what changed
```

Both `--link` paths automatically compile and link `runtime/frost_runtime.c`, so
pool programs work without any extra flags.
