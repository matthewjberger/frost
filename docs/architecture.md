# Frost Architecture

This document describes how the Frost compiler is structured today and the
direction it is moving. It is kept honest. It states what works, what is
partial, and what is not built yet.

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

The **typed IR** is the single spine. Reference and move checking are discharged
before it on the AST, type checking and the linear consume discipline are
discharged on the IR itself, and every backend emits from it.
`--native` / `--link` lower to the IR and emit machine code via Cranelift.
`--emit-c` lowers the same IR to portable C. `--run-ir` interprets the IR
directly. With no flag, `frost file.frost` compiles, links, and runs the program
natively. Because all three backends emit from one IR, a differential test runs
each program through them and asserts their output matches.

There is one execution surface. An earlier bytecode VM was retired once the
native path covered the language, so the data-oriented native language is the
only language.

## Modules

`src/imports.rs` reads each imported file once, renames the top-level names the
file does not `export`, and splices the result into one statement list. A
module's private names are tagged `__m<tag>_<name>`, where the tag is an FNV-1a
hash of the module's path relative to the project root, so a module's symbols
are a property of the module rather than of the order it was reached in. The tag
is undone in diagnostics by `demangle_private_names`, which lives next to the
code that applies it.

Positions carry a file id into `src/source_map.rs`, stamped during import
resolution, which is the only place that knows which file a position belongs to.
Without it a diagnostic from an imported module would name a line number in a
flattened program that matches no file the reader has open.

`src/interface.rs` derives what a caller would need to compile against a module
without seeing the rest of it, and checks it. The checks run under
`FROST_CHECK_INTERFACES`, which the test suite sets on every compilation, and
the whole suite runs a second time under `FROST_BUILD_FROM_INTERFACES`, which
reduces every imported module to its interface.

`src/build_cache.rs` is what makes that pay. Under `--incremental` it keeps a
record and an object per module, and a module whose own source and whose
imported interfaces are unchanged is neither parsed nor code generated: it
contributes the interface the record already holds and its object is linked. See
[separate-compilation.md](separate-compilation.md) for the fingerprint rule and
what is still whole-program.

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

**Working today**, verified by running native binaries (`tests/native.rs`):

- Integer arithmetic at every width with correct signedness, float
  arithmetic, bitwise and shift operators.
- Comparisons (signed / unsigned / float) and boolean logic with
  short-circuit evaluation.
- `if` / `else` expressions, `while`, `for`-over-range, `break`, `continue`.
- Functions, recursion, and direct calls.
- Sign / zero extension and truncation casts between integer widths, and
  integer/float conversions.
- `extern fn` C interop, including string-literal arguments with escape
  sequences (e.g. `puts`, `printf`).
- `str`, a byte-slice view (pointer plus length): string-literal values,
  `str_len` in constant time, bounds-checked byte indexing `s[i]`, and passing
  and returning `str` by value.
- `[]T` slices, the same fat-pointer view generalized to any element: an array
  coerces to a slice of the whole array (`view : []i64 = arr`, or an array
  passed to a `[]T` parameter), `s[i]` is bounds-checked against the runtime
  length, `slice_len(s)` reads the length, and slices pass and return by value.
- Borrows and pointers: parameter modes (`increment(mut x: i64)`), `^`
  dereference read/write, and raw pointer parameters (e.g.
  `swap(a: ^i64, b: ^i64)`). The surface has no `&`; `lower_param_modes`
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
  index aborts (see [memory-safety.md](memory-safety.md)).
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

### Generational handles and pools

A small C runtime (`runtime/frost_runtime.c`) provides generational pools
that both backends link automatically, `pool_new(capacity, elem_size)`,
`pool_alloc`, `pool_get`, `pool_free`, `pool_contains`, `handle_index`, and
`handle_generation`. This is the design's handle/pool proposition working
natively without a garbage collector.

The interface is deliberately scalar-only. A pool is an opaque pointer and a
handle is a packed `i64` (`generation << 32 | index`). Nothing is passed or
returned by aggregate value, so the runtime's natural C ABI matches Frost's
internal aggregate-return convention (a hidden out-pointer) with no ABI
negotiation, and the identical compiled runtime links into both the Cranelift
and C backends, which is why they agree bit for bit.

The generational guarantee is what the differential test pins down. Freeing a
slot bumps its generation, so a later allocation reuses the slot at a higher
generation and the original handle reports `pool_contains == 0`. A stale
handle can never silently read a live value. Storing the *handle* is fine (it
is plain copyable data, not a reference). The borrow you get by dereferencing
through the pool is what stays second-class.

Today the pool's memory management lives in the raw runtime, but the typed
surface over it is now an ordinary Frost library (see below), not a set of
privileged builtins.

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

**Not yet in the native backend** (these fail loudly, they are not silently
miscompiled): growable or heap-backed collections. Capturing closures are absent by design,
since the language uses function pointers and non-capturing function literals,
both of which the native backend supports. There is no other backend to fall
back to, so an unsupported construct is a compile error.

The emitted C is an internal detail, not an interface for external C callers,
so Frost function names are prefixed (`frost_`) to avoid C keyword clashes.
`extern` names and `main` are left untouched so FFI and the entry point link.
Frost-to-C interop (`extern fn`) works on both the Cranelift and C paths.

This replaces the previous AST-walking `codegen.rs`, which treated most
values as `i64`, hardcoded `if`-expression result types, resolved struct
field offsets by first-name match, and emitted `iconst 0` for anything it did
not handle.

## Direction

See [philosophy.md](philosophy.md) for the design philosophy, goals and
non-goals, and why Frost is data-oriented rather than object-oriented.

Frost is being reshaped toward a data-oriented language with:

- Plain data (copy/move), **linear resources** that must be consumed exactly
  once, and generational **handles** into explicit pools.
- **Parameter modes** rather than reference syntax: unmarked reads, `mut`
  writes, `move` takes ownership, and the compiler inserts the borrow at the
  call. `&`/`&mut` are not surface syntax, so a borrow has nowhere to be stored
  and is second-class by construction (`src/param_modes.rs`).
- **Regions** without lifetimes: a `with arena { }` block owns an arena, and a
  raw pointer into it may not outlive the block. A function's frame is checked
  the same way, so a pointer or slice naming a local cannot be returned
  (`src/regions.rs`).
- **Allocation sources**: `uses A` draws an allocation capability, threaded as
  an implicit parameter and supplied by the `with` block that provides it
  (`src/allocation_sources.rs`).
- **Failure sets**: `-> T ! E` says how a function fails and `?` hands a failure
  on, desugared to an ordinary enum and match (`src/failure_sets.rs`).
- **Compile-time arguments**: `$T` for types, `$N` for values, and `$f` for a
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

`src/ownership.rs` runs after parsing and enforces two rules:

- **Second-class borrows.** The reference types this pass sees are synthesized
  by `lower_param_modes`, since the surface has none. One cannot be stored in a
  struct or enum field, and cannot be returned from a function or extern.
  Reference *parameters* are the point, and `Handle<T>` (a generational index,
  not a reference) can be stored and returned freely. Because a borrow cannot
  escape, borrow analysis stays scope-local.
- **Borrow exclusivity.** A `mut` borrow is exclusive. A variable cannot be
  passed to more than one `mut` parameter, or to both a `mut` and a read
  parameter, within a single call. Multiple read borrows are fine. Because
  borrows are second-class, this per-call check is enough to keep mutable
  aliasing out.
- **Move checking.** Per function body, a value of a move type (a struct,
  enum, or slice, anything not `Copy`) is consumed when it is passed to a
  `move` parameter, assigned, or returned. Using it again is a use-after-move
  error. A read or `mut` parameter, field access (`x.f`), and dereference do
  not consume, and copy types (integers, floats, bools, pointers, references,
  handles, and `str`) are never
  moved.
- **Linear resources.** A struct or enum declared `linear`
  (`File :: linear struct { ... }`) is a resource that must be consumed
  exactly once. The move checker's use-after-move rule gives "at most once".
  The "exactly once" half, the leak check, is discharged separately on the IR
  (see below). Consuming means moving it onward, returning it, passing it by
  value to another function (the terminal consumer is typically an `extern`,
  which takes ownership across the FFI boundary), or `match`ing it (a `match`
  on a linear value destructures and consumes it). This is how the design
  replaces `Drop`. Cleanup is an obligation the type system tracks rather than
  an implicit call. It also makes a linear error enum non-ignorable, since a
  `linear enum` returned from a fallible function must be matched (or otherwise
  consumed), so a failure cannot be silently dropped.

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
value was created on. A leak is caught here; a use-after-move is caught on the
AST; both point at a line.

### Roadmap

1. Discharge ownership on the IR. *(Partly done: the linear consume discipline
   now runs as a CFG dataflow pass in `src/ir_ownership.rs`. Move tracking and
   borrow exclusivity stay on the AST, where the move-versus-borrow distinction
   the IR erases is still visible; second-class borrows keep that analysis
   scope-local.)*
2. A real type-checking pass on the IR. *(Done: `src/ir_typecheck.rs` runs on
   the typed IR after lowering and before either backend. It validates local
   and block id ranges, direct and indirect call arity against the gathered
   signatures, numeric operands for arithmetic and indexing, and that non-void
   functions return a value. It also enforces the IR's pointer discipline: loads,
   stores, field access, and element access all go through a pointer-typed
   operand; casts stay between numeric types; and an indirect call targets a
   function-pointer value. These checks hold across the whole native corpus, so a
   lowering bug that produced a load from a non-pointer or a cast to a struct
   would be caught before codegen rather than miscompiled.)*
3. Linear resources with path-sensitive consumption, and error enums that
   linearity makes non-ignorable.
4. Handle-dereference-as-borrow, and a first-class pool type. *(Done: `Handle<T>`
   is a first-class native type (a packed i64), and `pool[handle]` is a place.
   Read/write fields through it, copy the element out, or pass it to a function,
   which borrows it under that function's parameter mode. The borrow is
   second-class like any other, so a handle-deref borrow has nowhere to escape
   to.
   A pool is not a compiler type. A program writes its own: a value-generic
   struct of `[N]T` storage plus a generational free list, all Frost code
   (`examples/native/generic_slab.frost`), on top of slices, value generics, and
   `ptr_to`/`ptr_cast`. The runtime pool functions are an opt-in `extern`
   library, like `malloc`, and `pool[handle]` lowers to `pool_get` when a pool is
   indexed by a handle. See `docs/native-pools.md` and `docs/allocators.md`.)*
5. Struct/array/enum by-value passing and tuple patterns in the native
   backend. *(Done: all three, plus nested aggregates and arrays of structs.)*
6. Generics by monomorphization. *(Done:
   generic functions, generic structs (incl. nested `Pair<Pair<i64>>`, factory
   functions returning instances, construction inference, and generic-over-
   instance), `sizeof`, and explicit type arguments (`fn($T: Type, ...)` called
   `f($Concrete, ...)`, with type parameters erased from the specialized ABI).
   Value parameters too (`struct($T: Type, $N: usize)` sizing a `[N]T` field),
   resolved to a concrete array size at instantiation, so a slab can be generic
   over both element type and capacity.)*
7. Bounds-checked array indexing. *(Done: every fixed-size array index is
   checked against the statically-known length and aborts on out-of-range.)*
8. Source locations in errors. *(Done: the lexer and parser carry
   `line`/`column`, and a `Spanned<T>` wrapper attaches a source position to
   every statement, so ownership and IR-lowering errors report the exact source
   line and column, not just the enclosing function. A position also carries the
   file it came from, so an error inside an imported module names that module
   rather than a line number in the flattened program.)*
9. A third differential oracle. *(Done: `src/ir_interp.rs` interprets the typed
   IR directly, exposed through `--run-ir`. It validates scalar arithmetic,
   control flow, recursion, and function pointers against the Cranelift and C
   backends, and declines cleanly on memory and pool operations rather than
   guessing.)*
10. Self-hosting the compiler in Frost. *(Done: `bootstrap/frost.frost` compiles
    itself to a byte-identical fixpoint through both its C backend and its own
    native x86-64 backend, so there is a path with no C compiler in the loop. It
    is written in the data-oriented native surface, a pool-backed AST arena with
    integer node indices instead of pointers, and carries imports and modules,
    failure sets, enums with payloads, and generics. See
    [self-hosting.md](self-hosting.md).)*
11. Parser error recovery. *(Done: the parser recovers at statement boundaries
    instead of stopping at the first error, at the top level and inside function
    bodies alike, so one malformed statement no longer discards the rest of the
    file or the rest of the enclosing block. `parse_recovering` returns the
    statements that parsed plus a `Diagnostic` per error, and the plain `parse`
    entry point reports them all at once. Synchronization skips to the next
    statement start (a declaration, an assignment, or a leading keyword) and
    always makes progress, so recovery cannot loop. This is the foundation an
    editor integration would build on, though the language server itself is not
    yet in scope.)*
12. Parallel code generation. *(Done: `src/ir_codegen.rs` builds and compiles
    functions across every core from a shared work queue. 385 ms to 55 ms on
    sixteen threads at 10,401 functions, which took a full native build of 58k
    lines from 1.11 s to 353 ms. The two false starts are recorded in
    [roadmap.md](roadmap.md) because both came from trusting a summary statistic
    over a per-worker measurement.)*
13. Callbacks with a typed context. *(Done. An `extern fn` with a `$handler`
    parameter bound to a function signature is a callback registration:
    `src/callbacks.rs` checks the declaration, `src/regions.rs` holds the
    registration to the frame that holds its context, and `src/ir_build.rs`
    passes the handler's address and the context's address. There is no
    trampoline and no cast, because a `mut` parameter is already a pointer and
    Frost and C share a calling convention. [callbacks.md](callbacks.md) has the
    design and the one thing it does not yet do.)*
14. Separate compilation. *(Done. A module is a file, its interface is its
    `export` line, and a specialization is emitted in the module that
    instantiates it. On `--link` each module is its own object, and
    `--incremental` rebuilds a module only when its own source or an imported
    interface changes, which is a hash over the interface with generic bodies
    kept and ordinary ones blanked.
    [separate-compilation.md](separate-compilation.md) tracks it step by step,
    including what was found to be wrong about the original design.)*
