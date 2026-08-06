# How it got here

The pool, the callback form, the compiler and its separate compilation were each
built in stages, and the stages turned up things their plans did not have. The
current designs are in [pools-and-columns.md](../design/pools-and-columns.md),
[callbacks.md](../design/callbacks.md),
[architecture.md](../impl/architecture.md) and
[separate-compilation.md](../impl/separate-compilation.md). This is what happened
on the way to them, kept out of those documents so each one states the language
and the compiler as they are.

## The pool spike

Before the pool was Frost it was a compiler type, `Pool<T>`, lowering to an
opaque `^u8` and calling a C runtime. `examples/native/native_pool.frost` was
the spike that decided whether the language could own it instead: a full
generational pool, with the storage, the free list, the generation counters, the
packed `(generation << 32) | index` handles and the stale-handle check all
ordinary Frost over a fixed-size array inside a struct.

```frost
import "ecs.frost"
Slab :: struct {
    storage: [4]Entity,
    generations: [4]i64,
    free_list: [4]i64,
    free_count: i64,
}
```

It compiled and ran identically through both backends and reproduced
generational safety end to end. After a slot was released and reused, the old
handle read as dead. Everything the logic needs, arrays inside structs,
bounds-checked element read and write, aggregate element assignment, and integer
packing, already worked, which is what made the whole memory model expressible
in the language with no runtime.

Writing it also ran straight into the ways the C-backed pool sat in the
language's way. `Pool` was a reserved type name, so the spike had to call its
struct `Slab`, and `pool_*` and `handle_*` were reserved function names that
collided at link time with the always-linked runtime, so its helpers needed a
prefix. Both were one thing: a compiler-plus-runtime pool claiming global names
that user code could not have.

Three gaps showed up beside them. A place-deref could not be written as a
library function, because a function cannot return a reference into storage, so
the spike mutated by read-modify-write instead. The slab was hard-coded to
capacity 4, since `$N` as a value parameter did not exist yet. And construction
meant writing out each backing array as a literal.

The plan that came out of it ran in five steps, all since done. Value generics,
so `$N` sizes a field and the struct instantiates concretely. Compiler
place-deref over a Frost aggregate, recognized by a `storage` array beside a
parallel `generations` array. Operations generic over the capacity, which is
what turned the example into the library `std/slab.frost` rather than one slab
per size. Removing the compiler-special pool surface, which freed the `Pool` and
`pool_*` names. And taking the pool out of the C runtime, after which the
always-linked runtime is bounds and generation aborts, assertions, and the IO
helpers the self-hosted compiler uses.

One step changed as it landed. It had asked for zeroed or default construction
to remove the array-literal boilerplate. The repeat literal already covers that
(`storage = [Node { value = 0, next = 0 }; 16]`), and an implicit zero would
contradict the rule that a literal writes every field, so nothing was added.

## Building callbacks

The callback design arrived with three questions open, and answering them
changed two of the answers the roadmap had assumed.

`uses CallbackAbi` was dropped rather than decided. A `$handler` parameter with
a function bound on an `extern fn` is already the whole statement that the
extern takes a callback, so a `uses` clause beside it would be a second thing to
keep in step with the first, granting nothing. `uses Arena` supplies a real
implicit parameter at the call. A capability that supplies nothing is a keyword
pretending to be one.

Which parameter carries the context was originally "the first one". That made
the wgpu-native shape, where the userdata follows the callback arguments,
undeclarable, and the way around it was to put the same function pointer in a
struct field where none of the checks apply, so the rule cost safety rather than
buying it. It is now the parameter whose type is the type of the handler's
context, wherever it is written.

The trampoline does not exist. The plan was one generated function per
`(handler, context type)` pair, holding the single `ptr_cast` from untyped
userdata back to `^Ctx` so that nobody would have to write it. But a `mut`
parameter is already a pointer in the signature, and a Frost function and a C
function use the same calling convention, so `on_event` compiled for Frost is
bit for bit the `void (*)(void*, int64_t)` the library wants. The cast the
design set out to hide inside generated code was never there, so there was
nothing to generate and nothing to deduplicate.

Getting the context back needed something unrelated to callbacks. The roadmap's
sketch had the `Registration` hold a `Ctx` field, and that does not work: the
field is a copy, not the storage the library wrote through. What works is
unregistration as an ordinary extern handing the context back by value,
`unregister_handler :: extern fn(token: i64) -> Ctx`, wrapped in the Frost
function that consumes the linear registration. That needed an `extern fn`
returning a struct by value, which is `src/c_abi.rs` classifying return types
the way the target's C compiler does.

The build order was parse, check, then emit, and that order is the point. Step 1
accepted the declaration and checked its shape, because nothing can tell a
registration from any other extern call until the declaration says so. It turned
up that a function type could not say a mode at all: `fn(T1, ...) -> R` parsed
types and nothing else, so the bound `fn(mut Ctx, i64)` could not be written.
`mut` became a marker inside a function type, with unmarked left meaning the
type as written so that `$before: fn(T, T) -> bool` kept working. Step 2 closed
the roads out of the frame, landing before anything was emitted, so there was a
window in which the feature could only say no. Steps 3 and 4 were the trampoline
that was not there and the lowering of the call. Step 5 ran one: a small C
library that stores a `(callback, userdata)` pair and calls it back later,
linked and driven from Frost in `a_callback_registered_with_a_c_library_runs`.
If the ABI did not line up, that is where it would crash.

## Building the compiler

It sat under "Linearity checking on the IR" and every item was marked done.

1. Discharge ownership on the IR. *(Partly done: the linear consume discipline
   now runs as a CFG dataflow pass in `src/ir_ownership.rs`. Move tracking and
   borrow exclusivity stay on the AST, where the move-versus-borrow distinction
   the IR erases is still visible. Second-class borrows keep that analysis
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
   (`examples/native/generic_slab.frost`, and `std/slab.frost` in the standard
   library), on top of slices, value generics, and `ptr_to`/`ptr_cast`.
   CORRECTED WHILE MOVING: the original text said the runtime pool functions are
   an opt-in `extern` library and that `pool[handle]` lowers to `pool_get`.
   Neither is true. `slab_shaped_base` in `src/ir_build.rs` recognizes a struct
   holding a `storage` array beside a parallel `generations` array, and the
   index lowers inline to the bounds-and-generation check `frost_rt_slot`. The
   original also cited `docs/native-pools.md` and `docs/allocators.md`, which are
   now `../design/pools-and-columns.md` and `../design/allocators.md`.)*
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
10. Self-hosting the compiler in Frost. *(Done as a fixpoint, in progress as the
    product: `selfhosted/frost.frost` compiles itself to a byte-identical
    fixpoint through both its C backend and its own native x86-64 backend, so
    there is a path with no C compiler in the loop. It is written in the
    data-oriented native surface, a pool-backed AST arena with integer node
    indices instead of pointers, and carries imports and modules, failure sets,
    enums with payloads, and generics. It is the compiler Frost is for, and
    `architecture.md` describes the bootstrap that builds its stage 0. See
    [the self-hosted compiler](../impl/self-hosted.md).)*
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
    sixteen threads at 10,401 functions, and a full native build of 58k lines
    in 353 ms, measured when it landed. [roadmap.md](../roadmap.md) says how to
    sweep it again, and why a shared cursor beats splitting the function list
    into equal chunks.)*
13. Callbacks with a typed context. *(Done. An `extern fn` with a `$handler`
    parameter bound to a function signature is a callback registration:
    `src/callbacks.rs` checks the declaration, `src/regions.rs` holds the
    registration to the frame that holds its context, and `src/ir_build.rs`
    passes the handler's address and the context's address. There is no
    trampoline and no cast, because a `mut` parameter is already a pointer and
    Frost and C share a calling convention.
    [callbacks.md](../design/callbacks.md) has the design.)*
14. The C ABI for struct returns. *(Done. `src/c_abi.rs` classifies an
    `extern fn`'s aggregate return the way the target's C compiler does, per
    target, because Frost's uniform hidden out-pointer is Frost's own
    convention and C does not share it. The C backend defers to the C compiler
    instead by declaring a real struct type.
    [c-compatibility.md](../impl/c-compatibility.md) has the rules.)*
15. Separate compilation. *(Done. A module is a file, its interface is its
    `export` line, and a specialization is emitted in the module that
    instantiates it. On `--link` each module is its own object, and
    `--incremental` rebuilds a module only when its own source or an imported
    interface changes, which is a hash over the interface with generic bodies
    kept and ordinary ones blanked.
    [separate-compilation.md](../impl/separate-compilation.md) tracks it step by
    step, including what was found to be wrong about the original design.)*

---

## Building separate compilation

1. Make symbol names a property of the module. *Done.* The tag is an FNV-1a
   hash of the module's path relative to the project root, computed in
   `module_tag` in `src/imports.rs`. FNV is written out rather than taken from
   the standard library because the hash has to mean the same thing in every
   build of the compiler, and `DefaultHasher` promises only consistency within
   one version. The test compiles the same module reached first in one program
   and second in another and compares the tags, and it was checked against both
   failure modes: a traversal-order counter fails it, and so does a constant.
2. Write the interface out and read it back, while still compiling the whole
   program. *Done.* `src/interface.rs` derives a `ModuleInterface` at the one
   place a module is parsed, which is what stops it drifting from the source it
   describes. Three checks run under `FROST_CHECK_INTERFACES`, which the test
   suite sets on every compilation: it survives a JSON round trip, it declares
   everything it exports, and it is closed, meaning every name a carried
   declaration reaches and this module declares is carried too. Serialization is
   serde and JSON, marked replaceable.
3. Make monomorphization per-module. *Prerequisite done, the rest not
   started.* `expand_generic_structs` and the specialization loop in
   `src/ir_build.rs` walk every statement in the flattened program, and the
   blocker was that flattening threw away which module a statement came from.
   That is fixed: a `Position` now carries a file id into `src/source_map.rs`,
   stamped during import resolution, so every statement knows its module. It
   earns its place immediately rather than sitting as scaffolding, because it is
   also what lets a diagnostic name the file it came from, which after
   flattening it previously could not.

   *Attribution done.* Every specialization now records the module that asked
   for it, inherited by anything it goes on to instantiate. Emitted code is
   unchanged, verified by hashing the C and the object before and after.
   `FROST_MODULE_REPORT=1` reports how many specializations a single object
   emits, how many separate objects would emit, and how many are instantiated by
   more than one module, which is the measurement that decides whether the
   private-copy choice ever needs revisiting.

   What is left is to seed the worklist per module rather than from the whole
   flattened program, which only becomes observable at step 4.

   One thing to get right, because it is easy to state the step wrongly. While
   the compiler still emits one object, per-module copies cannot actually be
   emitted: two definitions of `Stack<i64>` in one object file is a duplicate
   symbol, not a fold. So `emitted: HashSet<String>` in `src/ir_build.rs` stays
   global for as long as the output is one object, and what step 3 changes is
   only *how the worklist is seeded*. The copies become real at step 4, when
   each module emits its own object, and that is also when their linkage becomes
   module-local. Step 3 is therefore a refactor whose output is byte-identical,
   which is exactly the kind the fixpoint tests are there to hold.
4. Compile a module from interfaces alone. *Available as an oracle.*
   `FROST_BUILD_FROM_INTERFACES=1` makes an imported module contribute what its
   interface says and nothing else, so a program that still behaves identically
   is evidence that the interface is sufficient. A module's own `import` lines
   are kept, since an interface carries declarations and not dependencies.

   The first thing this found was a live bug that had nothing to do with
   interfaces. The renamer walked a function's parameters and body but skipped
   its return signature, so a module exporting a function that returned an
   unexported type produced a name the importer could not resolve, and such a
   program did not compile.

   The whole test suite runs a second time under it, in CI and as
   `just test-interfaces`, so the sufficiency claim is checked on every commit
   rather than the day the compiler starts relying on it. That gate was itself
   checked by breaking the interface closure and confirming it fails.

   *Per-module objects done.* On the link path each module is now its own
   compilation unit: `IrModule::split_by_module` produces one part per module,
   each becomes its own object, and the linker resolves the calls between them.
   `--native -o x.o` still writes the single object its `-o` names, since that
   output is one file by contract.

   Three things this forced, each of which is the design becoming real:
   - Specializations are module-local. Two modules that instantiate the same
     generic each emit their own private copy, so exporting them would be a
     duplicate symbol. `IrFunction::local` says so and the backend declares it
     `Linkage::Local`.
   - The dedup had to become per-module. With one global dedup the first
     module to ask for `wrap<i64>` got it and the second module's object
     referenced a symbol that was not there. `build_module_per_module` keys the
     dedup by module and name.
   - Cross-module calls are declared, not externed. A part carries the other
     parts' functions in `imported` and declares them with the same signature
     builder that built the definitions. Describing them as externs would lose
     the hidden out-pointer an aggregate return uses, and the two objects would
     silently disagree about the ABI.
5. Cache and skip. *Done.* `--incremental` keeps a record and an object per
   module under `--build-dir`, and a module whose own source and whose imported
   interfaces are all unchanged is never parsed and never code generated: it
   contributes the interface the record already holds, and its object is linked
   rather than built.

   Three things it forced:
   - The import graph has to be walked before anything is spliced, bottom
     up, because whether a module can be skipped depends on the interfaces below
     it. The walk parses only the modules it cannot answer for, and hands those
     parses to the splice rather than repeating them.
   - The record carries the import list, even though an interface carries
     declarations and not dependencies. Deciding whether to skip a module means
     knowing what it imports before it has been read.
   - A file id could not go into a record. It is handed out in registration
     order, so an interface written down with one in it means something else in
     the process that reads it back, and module attribution is exactly what
     reads it. Interfaces are written with it zeroed and restamped on load.

   The first version of this step spliced the interface as it stood, which
   carries the bodies of exported functions whether or not they are generic, so
   the front end walked every body in the program even though it emitted none of
   them. `Statement::Declared` fixed that, and the declaration form was most of
   the win: it took the skipped path from 309 ms of compiler work to 110 ms.

---

## Porting the compiler to Frost

The self-hosted compiler checks its own programs now rather than deferring to
whatever compiles its output. In dependency order:

1. Self type-checking. Required before (3), because once the self-hosted
   compiler stops emitting C there is no C compiler behind it to catch anything.

   Done, and free. Every check was measured by running the build before and after
   it back to back: the self-compile time is identical either way. The absolute
   number moves with machine load (23 ms on a quiet machine, about 69 ms on a
   busy one), so compare builds against each other rather than against a number
   written down here.

   - Undefined calls and argument count. A name becomes defined when its `fn` or
     `extern fn` is parsed, so a name only ever called does not exist. Every node
     lives in one arena, so a linear scan reaches every call.
   - Undefined variables. `lookup_local` answers with a type and defaults an
     unknown name to `TY_I64`, which cannot tell "an i64" from "not declared",
     so `local_exists` checks the scope instead.
   - Unknown struct fields, naming the struct and the field.
   - Return types, against the function's declared return type.
   - Argument types, against the parameter list.
   - Assignment types, against the place.

   Compatibility is deliberately lenient where the emitted C is lenient. Scalars
   and pointers convert freely and a type parameter matches anything, because it
   is a placeholder. A struct only matches the same struct, which is where real
   mistakes show up. Two false positives had to be designed around, since
   frost.frost must keep passing its own checks: a generic template's
   parameter types are placeholders bound per instantiation and carry nothing to
   check against, and an auto-borrowed argument is a value whose address is taken
   at the call, so it answers to the pointee rather than the pointer.

   Left: the scope tracking is emit-time rather than a separate semantic pass,
   which is enough for these checks but would need reworking for flow-sensitive
   ones.

2. Ownership and linearity. Done, and also free.

   - Use after move. A struct handed to a parameter that does not borrow is
     moved out of the caller, so reading it afterwards reads a value that was
     given away.
   - Linear types. `linear struct` marks a resource. `is_linear` is recorded on
     the definition, and at the end of each body every linear value must have
     been handed on, by being returned or passed to a parameter that takes
     ownership. Together with the move check this is linearity proper, consumed
     exactly once: never consumed is a leak, consumed twice is a use after move.

   Note the shape of a real consumer. A read parameter of struct type borrows,
   so it does not consume. Consuming takes `move`, as in
   `close :: extern fn(move f: File)`. A function that takes a linear value by
   `move` and only reads a field out of it is correctly rejected, because the
   resource dies there.

3. Native backend. Done, and self-hosting with no C compiler in the loop.

   The compiler emits assembly for its own source, that assembly assembles into
   a compiler, and that compiler emits the same assembly byte for byte.
   `native_self_hosting_is_a_fixpoint` checks it.

   Two bugs had to be found to get there, both invisible to the C backend
   because C needs neither struct offsets nor a scope discipline of its own:

   - `type_size` counted a struct's fields instead of measuring them, so every
     type holding a struct field was sized wrong. `Parser` carries ten arenas
     and an arena is three words, not one.
   - `lookup_local` and `local_slot` answered with the first binding of a name
     rather than the most recent. `emit_program` binds `node` to an i64 in one
     loop and to a `^Node` in the next, so `node^.next` typed as a scalar and
     read offset zero, and the walk over the top-level list never reached its
     end.
   - The prologue read the hidden struct-result pointer out of `%rcx`, which is
     where Windows puts the first argument but not System V. That self-hosted on
     Windows and segfaulted on Linux, and only in the second stage, since the
     first-stage compiler is built by the reference compiler and never runs this
     assembly. When a stage fails, check which one before assuming the emitter
     crashed.
4. Allocation sources. Done. `uses A` on a function and `with a { }` around
   a call, mirroring `src/allocation_sources.rs`.

   The capability is an implicit trailing parameter that borrows its source, so
   after parsing the function is ordinary and only the call sites know to supply
   one. A `uses` function forwards what it was handed, a `with` block supplies
   the arena it names by address, and a call with neither in reach is rejected.

   This needed a name the compiler makes up rather than reads, since the binding
   is the capability type's name lowercased and no source spells it. A name here
   is an offset and a length into the source everywhere, so rather than add a
   second kind of name, the source is copied into a buffer with room past its
   terminator and synthesized names are written there. The lexer stops at the
   terminator, so nothing in that room is ever read as code.

   Fixing this also removed a limitation worth naming: a plain function used to
   lower to `int64_t` whatever it returned, so it could not return a pointer.
   It now emits its declared return type.

5. Regions. Done, mirroring `src/regions.rs`. A `with` block is a region and
   a raw pointer derived from its arena may not be stored outside the block or
   returned. A binding declared inside may hold one, since it dies with the
   block, and reading through it is the point.

   No lifetimes and no region types on pointers, just a walk over the block
   tracking which names hold arena pointers. A dereference deliberately does not
   propagate: it reads the value there, and reading out of the region is what
   the region is for.

   It is interprocedural: the body of every `uses` function is a region too,
   whose arena belongs to whoever supplied it. Such a function may hand a
   pointer back to its caller, where the caller's region checks it, but may not
   store one into a parameter, which outlives the call.

6. Failure sets. Done. `-> T ! E` says a function answers with a T or fails
   with an E, and `e?` hands the failure on.

   Both lower to what the compiler already had. A failure set is a struct
   carrying which of the two it holds beside both payloads, rather than the
   reference compiler's Result enum, so no enum-with-payload machinery was
   needed: the two are never both live, but reading either one is then a plain
   field of a plain struct, which every backend already does. A `return` that
   builds the error type is the failure side and anything else is the value
   side, matching how the reference reads it.

   `e?` becomes a binding, a test and a return, queued for the block being
   parsed and emitted ahead of the statement the `?` was written in. That is
   what lets a `?` sit anywhere an expression can rather than only where a
   statement can.

   This needed names the compiler makes up, which the room past the source
   already provides: `__Result<n>` per failure set, `__try<n>` per `?`, and the
   three field names.

7. Imports. Done. `import "path"` names a file whose declarations join this
   one's.

   Every file's text reaches one buffer, so a name stays what it is everywhere
   else in this compiler, an offset and a length into a single source, and
   nothing downstream learns that more than one file was involved. A file lands
   there once however many times it is named, which is what separates this from
   a textual include and what makes a diamond one copy and a cycle terminate.
   Imports are found by token rather than by scanning text, so the word
   `import` inside a string or a comment is not one, and the placement is
   post-order: a file follows everything it depends on, because a struct's name
   is resolved where it is written rather than looked up later.

   Fixing this turned up a real gap: a local bound by `:=` to a call of a
   generic function took the template's return type, which still mentions the
   type parameter, so two arenas over different elements came out as the same
   type. The concrete return type per instantiation was already computed for the
   native backend. It now runs before either backend, so a call answers with it
   everywhere.

   `export a, b` lists what a file offers. A top-level name not listed is the
   file's own, so two modules may each keep one of the same name and neither
   sees the other's.

   Visibility is settled where a name is interned rather than by renaming what
   a module keeps private. A declaration's own offset says which file wrote it,
   so nothing has to carry an owner, and a lookup matches a name only when the
   two are in the same file or the declaring file exported it. A program built
   from one file has one module, so the first comparison answers yes and this
   costs nothing.

8. Enums with payloads. Done. `Kind :: enum { Player, Enemy { damage: i64 } }`,
   `Kind::Enemy { damage = 15 }`, and `case .Enemy { damage }:` in a match.

   An enum is a struct carrying a tag beside every variant's fields, each field
   named for the variant it came from, so two variants may each carry a
   `damage` and they stay apart. A variant is then a tag value and a set of
   field names rather than a type of its own, which is why construction, field
   access and matching all reduce to what the compiler already did with structs
   and integers. The tag says which variant is live, so the variants share their
   offset and an enum is its tag plus its widest variant. The C backend writes
   that as an anonymous union of anonymous structs, one per variant, which leaves
   every field reachable by the flat name it was declared under.

   A variant pattern also made `match` an expression rather than a statement. It
   binds a name, each arm assigns to it, and the match is queued for the block
   the same way a `?` is. An arm ending in something with no value is left alone,
   since it was not producing one.

   Making `match` an expression turned up a bug in the road both it and `?`
   take. A statement queued while a condition was being parsed was placed
   before the loop, so a `?` in a `while` condition was asked once and answered
   the same for ever. The loop now carries what its condition needs and runs it
   every time round, with the test moved inside.

   Two more bugs came out of running everything through both backends and
   comparing, which is now a test of its own:

   - A generic function used with a type that no generic struct was written
     with was called and never emitted, so the program failed to link.
     Instantiation was driven by the struct instances, and a generic function
     had to ride on one. It is driven by the distinct types the generics are
     used with now, gathered from the instances and from the calls. That also
     closes a second hole: two generic structs written with the same type would
     have had every generic function emitted twice.
   - The native backend treated every scalar as a word, so `^i8` indexing strode
     eight bytes at a time and `sizeof(i8)` answered 8. A byte-wide type is one
     byte now, struct fields sit on their own alignment, and a byte is loaded
     and stored with byte instructions. The C backend was always right here,
     which is how this survived so long.

Param modes were already done before the list started. The self-hosted compiler
lowers `mut`/`move`/read to pointers and inserts the borrow at call sites.

---

## How each port was done

The pattern that worked for param modes:

1. Add the capability to the self-hosted compiler while leaving frost.frost
   unchanged, so the fixpoint stays byte-identical and the change is inert.
2. Migrate frost.frost to use it.
3. Verify against the bootstrap compiler with the differential tests, and keep
   `self_hosting_is_a_fixpoint` green.
4. Commit each stage separately.

The bootstrap stays the oracle throughout. Note what that oracle does and does
not check. It says the two compilers agree, not that either is right. The
mixed-width arithmetic bug survived it, because every backend agreed on the
wrong answer. A port needs a program with expected output too.
