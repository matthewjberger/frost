# Frost architecture

This document describes how the bootstrap compiler in `src/` is structured and
the direction it is moving. The compiler written in Frost is
[a document of its own](self-hosted.md).

## Pipeline

```
Source (.frost)
      |
      v
   Lexer            src/lexer.rs        -> tokens
      |
      v
   Parser           src/parser.rs       -> AST
      |
      v
   Import resolver  src/imports.rs      -> one flat, module-scoped AST
      |                                    src/interface.rs derives and checks
      |                                    each module's interface alongside
      |                                    src/source_map.rs records which file
      |                                    each position came from
      v
   Ownership check  src/ownership.rs
      |
      v
   Typed IR         src/ir.rs, src/ir_build.rs, src/ir_typecheck.rs
      |
      v
   Linearity check  src/ir_ownership.rs   (dataflow over the IR CFG)
      |
      +--------------------+--------------------+
      v                    v                    v
  Cranelift            Portable C           IR interpreter
  src/ir_codegen.rs    src/ir_c.rs          src/ir_interp.rs
      |                    |                    |
      v                    v                    v
  object -> exe        C -> exe             direct run
```

The typed IR is the single intermediate representation. Reference and move
checking are discharged before it on the AST, type checking and the linear
consume discipline are discharged on the IR itself, and every backend emits
from it.
`--native` / `--link` lower to the IR and emit machine code via Cranelift.
`--emit-c` lowers the same IR to portable C. `--run-ir` interprets the IR
directly. With no flag, `frost file.frost` compiles, links, and runs the program
natively. Because all three backends emit from one IR, a differential test runs
each program through them and asserts their output matches.

There is one execution surface. An earlier bytecode VM was retired once the
native path covered the language, so the data-oriented native language is the
only language.

Which backend runs, what the executable depends on at run time, and what the
compiler itself is written in are three separate questions that sound alike.
[build-modes.md](build-modes.md) separates them.

## Modules

`src/imports.rs` reads each imported file once, renames the top-level names the
file does not `export`, and splices the result into one statement list. A
module's private names are tagged `__m<tag>_<name>`, where the tag is an FNV-1a
hash of the module's path relative to the project root, so a module's symbols
are a property of the module rather than of the order it was reached in. The tag
is undone in diagnostics by `demangle_private_names`, which lives next to the
code that applies it.

Positions carry a file id into `src/source_map.rs`, stamped during import
resolution and, for the entry file, by the driver. Without it a diagnostic from
an imported module would name a line number in a flattened program that matches
no file the reader has open.

A specialization additionally carries the call that asked for it and the name
the reader wrote, so an error inside a stamped-out body leads with the line they
wrote rather than a line in a template they may not own, and never shows a
mangled symbol.

`src/interface.rs` derives what a caller would need to compile against a module
without seeing the rest of it, and checks it. The checks run under
`FROST_CHECK_INTERFACES`, which the test suite sets on every compilation, and
the whole suite runs a second time under `FROST_BUILD_FROM_INTERFACES`, which
reduces every imported module to its interface.

`src/build_cache.rs` is what makes that pay. Under `--incremental` it keeps a
record and an object per module, and a module whose own source and whose
imported interfaces are unchanged is neither parsed nor code generated: it
contributes what the record already holds and its object is linked.

What it contributes is signatures rather than bodies. `Statement::Declared` is a
Frost function's signature with no body, which is all a caller needs for a
function it is not going to emit, and it is not an `extern` because an extern
means C linkage and a C ABI. Generic bodies still come, because the caller is
what stamps out the template. See
[separate-compilation.md](separate-compilation.md) for the fingerprint rule.

## Code generation is parallel

`src/ir_codegen.rs` builds and compiles each function on its own thread, then
defines them into the object serially, since a module is one mutable thing.
Functions are handed out from a shared atomic cursor rather than split into
equal chunks, because cost per function varies by more than an order of
magnitude and the expensive ones sit next to each other. Results are sorted back
into module order so a build's output does not depend on how threads
interleaved. `FROST_THREADS` caps the pool and `FROST_TIMINGS` reports the
split between declaring, generating, defining and emitting.

## Typed IR

The IR (`src/ir.rs`) is a typed, CFG-based intermediate representation in the
spirit of a compiler "middle end" (MIR):

- A module is a set of functions and extern declarations.
- Each function has typed locals, a list of basic blocks, and an entry block.
- Each block is a sequence of statements ending in a terminator
  (`return`, `jump`, conditional `branch`, or `unreachable`).
- Values are explicit operands (a constant or a local). Every operand has a
  concrete type, so lowering never has to guess widths or signedness.
- Short-circuit `&&` / `||` and `if`/`else` expressions are lowered to
  explicit control flow, not special-cased in the backend.
- Address-taken locals are marked `in_memory`. The backend gives them stack
  slots. `&`, `&mut`, and `^` (dereference) lower to address-of, load, and
  store.

Lowering (`src/ir_build.rs`) folds light bidirectional type inference into the
translation so each value carries a real type. Anything outside the supported
subset fails loudly with a `native backend: ...` error rather than emitting
incorrect code.

## The C ABI is classified, not assumed

`src/c_abi.rs` decides how C returns a struct, which is not how Frost returns
one. Frost returns every aggregate through a hidden out-pointer, uniformly. C
returns a small one in registers and a large one through a pointer, and where
the line falls depends on the target and, on Windows, not at all on the field
types even though it does everywhere else. So an `extern fn` returning a struct
is classified per target rather than pushed through Frost's own convention.

The Cranelift backend builds the signature from that classification and writes
the returned registers into the caller's storage. The C backend does not
reimplement any of it: it declares a real struct type, field for field with
explicit padding, and lets the C compiler classify it. An aggregate *parameter*
to an extern stays a pointer by convention, which is a different kind of answer
and is why the two are described separately in
[c-compatibility.md](c-compatibility.md).

## Native backends

`src/ir_codegen.rs` emits a relocatable object from the IR via Cranelift and
links it with the system C toolchain. `src/ir_c.rs` emits portable C from the
same IR (`--emit-c`), which the system C compiler builds. Both use the
correct type and operation for each value because the IR is fully typed, and
`tests/native.rs` checks that the two backends agree on every program.

Working today, verified by running native binaries (`tests/native.rs`):

- Integer arithmetic at every width with correct signedness, float
  arithmetic, bitwise and shift operators.
- Comparisons (signed / unsigned / float) and boolean logic with
  short-circuit evaluation.
- `if` / `else` expressions, `while`, `for`-over-range, `break`, `continue`.
- Functions, recursion, and direct calls.
- Sign / zero extension and truncation casts between integer widths, and
  integer/float conversions.
- `extern fn` C interop, including string-literal arguments with escape
  sequences (e.g. `puts`, `printf`), C functions that return a struct by value
  (classified per target by `src/c_abi.rs`), and callback registration, where a
  `$` function parameter plus a context taken by `move` hands a C library a
  Frost function pointer and a typed context.
- `str`, a byte-slice view (pointer plus length): string-literal values,
  `str_len` in constant time, bounds-checked byte indexing `s[i]`, and passing
  and returning `str` by value.
- `[]T` slices, the same fat-pointer view generalized to any element: an array
  coerces to a slice of the whole array (`view : []i64 = arr`, or an array
  passed to a `[]T` parameter), `s[i]` is bounds-checked against the runtime
  length, `slice_len(s)` reads the length, and slices pass and return by value.
- Borrows and pointers: parameter modes (`increment(mut x: i64)`), `^`
  dereference read/write, and raw pointer parameters (e.g.
  `swap(a: ^i64, b: ^i64)`). The surface has no `&`. `lower_param_modes`
  synthesizes the reference types the rest of the pipeline handles.
- Structs: layout with correct field alignment, construction, field read
  and write, borrowed struct and field parameters, mutation through a
  `mut` parameter, whole-struct copy (`copy := p`), passing aggregates by
  value (copied at the call boundary so a callee's mutations do not affect
  the caller), and returning aggregates by value (via a hidden out-pointer),
  including passing an aggregate-returning call directly as an argument.
- Fixed-size arrays: array literals, indexed read and write with static or
  runtime indices, and borrowed array parameters (e.g. `sum(a: [5]i64)`). Every
  index is bounds-checked against the statically-known length. An out-of-range
  index aborts (see [memory-safety.md](../design/memory-safety.md)).
- Enums and tagged unions: construction, and `match` over a value or a
  reference with enum-variant patterns (binding payload fields), integer
  literal patterns, identifier binding, and wildcard.
- Tuple patterns in `match` (e.g. `match (i % 3, i % 5) { case (0, 0): ... }`),
  with literal, wildcard, and identifier-binding sub-patterns.
- Function pointers: a function used as a value becomes its address, a
  `fn(...) -> T` parameter or local holds one, and calling through it is an
  indirect call. This is the design's "function pointers, not closures"
  higher-order story (`apply(f: fn(i64) -> i64, x: i64)`).
- `defer`: function-scoped, run in LIFO order at each return and at the
  trailing expression. A `return` nested inside a branch alongside `defer`
  is rejected (it would need runtime tracking), so defers always run.

### Generic functions, specialization, and sizeof

The native path monomorphizes generic functions. A function is generic when a
parameter is typed `$T`. It is kept out of normal lowering and specialized on
demand. At each call site the concrete substitution is inferred from the
argument types, a specialized name is mangled (`identity__i64`), and a
worklist drives specialization to fixpoint, so transitive generics and
multiple instantiations of the same function are all emitted once. Substitution
rewrites both `TypeParam(T)` and the bare `Struct("T")` the parser produces for
later uses of a type parameter. This works for generics over scalars, structs
by value, and references.

`sizeof(T)` lowers to a compile-time integer from the IR's layout, and because
substitution runs first, `sizeof(T)` inside a generic function becomes
`sizeof(Concrete)` and then a constant. When a type parameter can't be inferred
from a value argument, it is declared `$T: Type` and passed explicitly at the
call as `$Concrete` (type parameters are then erased from the specialized ABI).
Together these turn the pool into a Frost library.
`make_pool($T: Type, cap) -> ^u8` sizes itself with `sizeof(T)` and is called
`make_pool($Entity, 16)`, and `insert(pool, value: $T) -> Handle<T>` copies the
inferred element type in, with no manual element size and no privileged builtin
(`examples/native/generic_pool_library.frost`).

Generic structs monomorphize the same way. `Foo<Args>` in type position is
encoded as a struct name that carries its arguments (`Pair<i64>`). Because a
struct name is only a layout-registry key and aggregates are byte buffers, the
name never has to be a valid identifier. A pre-pass discovers every instance
used across signatures, fields, and bodies, substitutes the generic struct's
fields, and registers a concrete layout to fixpoint (so nested instances
resolve). Construction uses the annotated instance type. This works over
scalars and structs, with multiple type parameters, array fields of the
parameter, by-reference passing, and nesting inside other structs.

`columns<T, N>` is synthesized by the same pre-pass by reflecting over `T`'s
fields rather than substituting a template: for each field it registers one
`[N]field` array named after the field, plus the `generations` / `free_list` /
`free_count` free list a slab carries. The deref `c[handle].field` and the
scatter `c[handle] = value` lower to the slab's bounds-and-generation check
(`frost_rt_slot`) reused verbatim, selecting the column before indexing it, and
`columns_new()` zero-initializes. It is the structure-of-arrays sibling of the
slab. See [pools-and-columns.md](../design/pools-and-columns.md).

Growable storage is a library rather than a backend feature. `std/vec.frost` is
one heap block that doubles when it fills, presented as a `[]T` so every access
goes through the bounds-checked slice path, and `std/map.frost`,
`std/json.frost` and `std/format.frost` are written on top of it. Capturing
closures are absent by design, since the language uses function pointers and
non-capturing function literals, both of which the native backend supports.
There is no other backend to fall back to, so a construct a backend cannot lower
is a compile error rather than silently miscompiled code.

The emitted C is an internal detail, not an interface for external C callers,
so Frost function names are prefixed (`frost_`) to avoid C keyword clashes.
`extern` names and `main` are left untouched so FFI and the entry point link.
Frost-to-C interop (`extern fn`) works on both the Cranelift and C paths.

This replaces the previous AST-walking `codegen.rs`, which treated most
values as `i64`, hardcoded `if`-expression result types, resolved struct
field offsets by first-name match, and emitted `iconst 0` for anything it did
not handle.

## Direction

See [philosophy.md](../design/philosophy.md) for the design philosophy, goals and
non-goals, and why Frost is data-oriented rather than object-oriented.

Frost is being reshaped toward a data-oriented language with:

- Plain data (copy/move), linear resources that must be consumed exactly
  once, and generational handles into explicit pools.
- Parameter modes rather than reference syntax: unmarked reads, `mut`
  writes, `move` takes ownership, and the compiler inserts the borrow at the
  call. `&`/`&mut` are not surface syntax, so a borrow has nowhere to be stored
  and is second-class by construction (`src/param_modes.rs`).
- Regions without lifetimes: a `with arena { }` block owns an arena, and a
  raw pointer into it may not outlive the block. A function's frame is checked
  the same way, so a pointer or slice naming a local cannot be returned
  (`src/regions.rs`).
- Allocation sources: `uses A, B` draws allocation capabilities, threaded as
  implicit parameters and supplied by the `with` blocks that provide them,
  matched by the name each is reached by (`src/allocation_sources.rs`).
- Failure sets: `-> T ! E` says how a function fails and `?` hands a failure
  on, desugared to an ordinary enum and match (`src/failure_sets.rs`).
- Compile-time arguments: `$T` for types, `$N` for values, and `$f` for a
  function, so a generic algorithm calls its comparator directly rather than
  through a pointer. A function argument may declare the signature it needs
  (`$before: fn(T, T) -> bool`), checked at the call with that call's type
  arguments substituted in. That is the only bound in the language and it is not
  a trait system.
- Free functions only, with signatures that declare their effects.
- The typed IR as the single point where ownership, borrow, and linearity
  checking are discharged, cross-checked by three independent execution paths:
  the Cranelift native backend, the portable C backend, and a direct IR
  interpreter that all must agree.

## Ownership checking

`src/ownership.rs` runs after parsing, over one top-level item at a time. The
rules it enforces are the language's rather than the pass's:
[ownership.md](../reference/ownership.md) has second-class borrows, mutable
exclusivity and the move rule, and [linear.md](../reference/linear.md) has
consume-exactly-once. What this pass does with them:

- It refuses a reference in a struct field or an enum variant's field, and a
  reference returned from an `extern`. A reference returned from a Frost
  function is allowed, because the frame-escape check in `src/regions.rs` holds
  it to storage that outlives the call. The reference types the pass sees at all
  are synthesized by `lower_param_modes`, since the surface has none.
- It checks the arguments of each call against each other, reporting a variable
  passed to two `mut` parameters, or to a `mut` and a read parameter, in one
  call.
- It walks each function body tracking which names have been moved out of,
  reporting a use of one afterwards. A name is a move name when its type is not
  `Copy`, which `Type::is_copy` in `src/types.rs` answers: structs and enums
  move, and integers, floats, bools, pointers, references, handles, arrays,
  `str` and slices copy, so a slice passed by value leaves the caller's own
  slice usable.
- It reports a move inside a loop body, which would be a use after move on the
  next iteration, and a linear value consumed inside one, which would be a
  second consumption.

Two things about how it reports. It checks every top-level item rather than
stopping at the first failure, so a program with a move error in three functions
names all three. And it remembers what it has already said, because past a move
the state stays moved and every later mention of that name would otherwise fail
the same way.

That gives "at most once" for a `linear` resource. The other half, the leak
check that makes it exactly once, is discharged on the IR.

## Linearity checking on the IR

`src/ir_ownership.rs` discharges the "consumed exactly once" discipline as a
dataflow pass over each function's control-flow graph, which is where the design
always intended ownership to be checked. Lowering marks a local as linear when
its type is a `linear` struct or enum, emits an `own` marker where such a value
is constructed, and emits a `consume` marker where it is moved (an identifier
read, or an aggregate passed by value, which lowers to an address and would
otherwise be invisible). Both markers are metadata that every backend skips.

The pass runs a forward dataflow to a fixpoint over an unowned / owned / consumed
lattice, joining at merge points, so it handles `if`, `match`, and loop back
edges directly rather than by structured approximation. It reports a value
consumed more than once, consumed before it holds a resource, or a linear local
still owned on a path to a return (a leak), each located at the source line the
value was created on. A leak is caught here. A use-after-move is caught on the
AST. Both point at a line.
