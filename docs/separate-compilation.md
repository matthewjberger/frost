# Separate compilation

This started as the design and is now also the record of building it. All five
steps at the bottom are done. What a module's compiled artifact contains, and
what has to be in it for a caller to compile against it without seeing the body,
is the question the whole document is answering.

The short version of the result: `frost --link --incremental` gives each module
its own object and rebuilds a module only when its own source or an imported
interface changes.

## Why the shape has to change

Goal 8 in [philosophy.md](philosophy.md) makes compilation speed a promise. The
constant factor is already handled: a full native build runs at about 166,000
lines per second, which clears the bar. This is the other half, and it is not a
constant factor.

`src/imports.rs` flattens every import into one AST. `resolve_imports` reads each
imported file, parses it, renames its private names, and splices the statements
into a single `Vec<Spanned<Statement>>`. Every pass after that runs over the
whole program. So a program's cost is whole-program by construction, and the two
things that grow worst are the two that most want bounding:

- **Monomorphization runs to fixpoint over everything.** A specialization is
  emitted once per program. Change one line in `main` and every specialization in
  every dependency is computed again.
- **Nothing can be compiled independently**, so nothing can be cached and
  incremental rebuild does not exist. Touching a leaf costs the same as a clean
  build.

Invisible at five thousand lines. The whole story at five hundred thousand.

## What a module is

**A module is a file.** That is already true of `import`, already true of the
`export` line, and inventing a second notion of module would mean two visibility
rules to explain instead of one.

**A module's interface is its `export` line.** Also already true. `private_renames`
in `src/imports.rs` mangles every top-level name a file does not export, which
means the export list is already the complete set of names another file can
depend on. Separate compilation does not need a new declaration form; it needs
the existing one to be written down in a file rather than reconstructed by
re-reading the source.

**A specialization is emitted in the module that instantiates it.** Not in the
module that declares the generic, which cannot know what its callers will ask
for, and not once per program, which is the thing being fixed. Two modules that
both instantiate `Stack<i64>` each emit their own copy, with module-local
linkage, and the duplicate code is the price of a module's work depending only
on the module.

An earlier version of this said the linker folds those copies, which is what
C++ and Rust do with COMDAT and weak symbols. **That is not available here.**
`cranelift_module::Linkage` has exactly `Import`, `Local`, `Preemptible`,
`Hidden` and `Export`, with no weak or COMDAT variant, so there is nothing to
ask the object writer for. Two options follow and the first is chosen:

- **Emit a private copy per module** (`Linkage::Local`), once each module emits
  its own object. Needs no backend work at all, and duplicate specializations
  cost code size rather than correctness. Note this only becomes possible with
  separate objects: two copies in one object file is a duplicate symbol, so
  while the compiler emits a single object it must keep deduplicating exactly as
  it does today.
- Teach `cranelift-object` to emit COMDAT sections and add a `Linkage` variant
  for it. Better output, upstream work, and not on the critical path. Worth
  revisiting only if duplicated specializations measurably matter.

Finding this out is why the step order below is worth trusting: it was a design
assumption that survived being written down and did not survive being checked
against the API.

## What the artifact contains

Compiling a module produces two things: an object file, and an **interface** the
compiler reads instead of the source when something imports it.

The interface has to carry everything a caller's passes consult about an imported
name today. Working through the pipeline, that is:

| what | why it must be in the interface | consulted by |
| --- | --- | --- |
| exported function signatures | call arity and types, and the return type | `ir_typecheck`, `ir_build` |
| parameter **modes** | the borrow at a call site is inserted from the callee's mode, not from the call | `param_modes`, `ownership` |
| exported struct layouts | field offsets, sizes, alignment | `ir_build`, both backends |
| exported enum layouts | tag values, payload offsets | same |
| which types are `linear` | consume-exactly-once is checked at the caller | `check_linearity` |
| failure sets (`uses`) | `try` lowering and the result type | `lower_failure_sets` |
| allocation capabilities (`uses Arena`) | the implicit parameter a caller has to supply | `lower_allocation_sources` |
| **generic bodies** | a specialization is stamped out at the caller, so the caller needs the AST | `ir_build` |
| compile-time parameter signatures | the bound checked at the call | `ir_build` |

The last one on the list is the one that decides the flavour of the whole design.
**A generic's body is part of its interface.** There is no way around it while
monomorphization is the only implementation of generics: the caller is what
chooses the type arguments, so the caller is what has to instantiate the template.
This is the same bargain C++ headers and Rust `#[inline]`/generic MIR make, and it
means an interface is not merely a list of signatures.

That has one consequence worth stating plainly rather than discovering later:
**changing a generic's body is a change to the module's interface**, and every
module that instantiates it has to be rebuilt. Changing a non-generic body is
not. So the interface should be hashed with the bodies of exported generics
included and the bodies of everything else excluded, and that hash is what
downstream rebuilds key on.

## What has to change first, and it is not the file format

Two things in the current pipeline are wrong for this in ways that a serialization
format cannot paper over.

**Private name mangling depends on traversal order.** `resolve_into` hands out
`module_tag` by the order files are visited, so a private `helper` becomes
`__m3_helper` in one program and `__m7_helper` in another. A module's symbol names
therefore are not a property of the module, which is exactly what separate
compilation requires. The tag has to be derived from the module's own identity, a
hash of its canonical path relative to the project root, so that a module compiled
alone and a module compiled as part of a program produce the same symbols.

**Monomorphization is a whole-program fixpoint.** `expand_generic_structs` and the
specialization loop in `src/ir_build.rs` walk every statement in the flattened
program. Both need to become per-module, seeded by what that module's own code
instantiates, with the templates read from interfaces rather than found inline.
The dedup set that is currently `emitted: HashSet<String>` becomes per-module, and
duplicate specializations across modules are resolved by the linker rather than by
the compiler.

Neither is a large change. Both are invasive in the sense that they touch the
assumption every later pass rests on, which is why they come before anything else
rather than after.

## The order to build it in

1. **Make symbol names a property of the module.** *Done.* The tag is an FNV-1a
   hash of the module's path relative to the project root, computed in
   `module_tag` in `src/imports.rs`. FNV is written out rather than taken from
   the standard library because the hash has to mean the same thing in every
   build of the compiler, and `DefaultHasher` promises only consistency within
   one version. The test compiles the same module reached first in one program
   and second in another and compares the tags, and it was checked against both
   failure modes: a traversal-order counter fails it, and so does a constant.
2. **Write the interface out and read it back**, while still compiling the whole
   program. *Done.* `src/interface.rs` derives a `ModuleInterface` at the one
   place a module is parsed, which is what stops it drifting from the source it
   describes. Three checks run under `FROST_CHECK_INTERFACES`, which the test
   suite sets on every compilation: it survives a JSON round trip, it declares
   everything it exports, and it is closed, meaning every name a carried
   declaration reaches and this module declares is carried too. Serialization is
   serde and JSON, marked replaceable.
3. **Make monomorphization per-module.** *Prerequisite done, the rest not
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
   private-copy choice above ever needs revisiting.

   What is left is to seed the worklist per module rather than from the whole
   flattened program, which only becomes observable at step 4.

   One thing to get right, because it is easy to state the step wrongly. While
   the compiler still emits **one** object, per-module copies cannot actually be
   emitted: two definitions of `Stack<i64>` in one object file is a duplicate
   symbol, not a fold. So `emitted: HashSet<String>` in `src/ir_build.rs` stays
   global for as long as the output is one object, and what step 3 changes is
   only *how the worklist is seeded*. The copies become real at step 4, when
   each module emits its own object, and that is also when their linkage becomes
   module-local. Step 3 is therefore a refactor whose output is byte-identical,
   which is exactly the kind that should be landed against the fixpoint tests.
4. **Compile a module from interfaces alone.** *Available as an oracle.*
   `FROST_BUILD_FROM_INTERFACES=1` makes an imported module contribute what its
   interface says and nothing else, so a program that still behaves identically
   is evidence that the interface is sufficient. A module's own `import` lines
   are kept, since an interface carries declarations and not dependencies.

   The first thing this found was a live bug that had nothing to do with
   interfaces: the renamer walked a function's parameters and body but skipped
   its return signature, so a module exporting a function that returned an
   unexported type produced a name the importer could not resolve, and such a
   program simply did not compile.

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
   - **Specializations are module-local.** Two modules that instantiate the same
     generic each emit their own private copy, so exporting them would be a
     duplicate symbol. `IrFunction::local` says so and the backend declares it
     `Linkage::Local`.
   - **The dedup had to become per-module.** With one global dedup the first
     module to ask for `wrap<i64>` got it and the second module's object
     referenced a symbol that was not there. `build_module_per_module` keys the
     dedup by module and name.
   - **Cross-module calls are declared, not externed.** A part carries the other
     parts' functions in `imported` and declares them with the same signature
     builder that built the definitions. Describing them as externs would lose
     the hidden out-pointer an aggregate return uses, and the two objects would
     silently disagree about the ABI.
5. **Cache and skip.** *Done.* `--incremental` keeps a record and an object per
   module under `--build-dir`, and a module whose own source and whose imported
   interfaces are all unchanged is never parsed and never code generated: it
   contributes the interface the record already holds, and its object is linked
   rather than built.

   The decision is a fingerprint, in `src/build_cache.rs`. A module's is a hash
   of its own source together with the interface hash of every module reachable
   through its imports, transitively, since a generic this module instantiates
   can instantiate one from further down. A module's *interface* hash is taken
   over the interface with the bodies of ordinary functions blanked and the
   bodies of generics kept, which is the distinction this document has been
   claiming since the top and is the only thing that makes the cache worth
   having: an ordinary body is what a module can change without rebuilding
   anything else, and it is most of what anyone edits.

   Three things it forced:
   - **The import graph has to be walked before anything is spliced**, bottom
     up, because whether a module can be skipped depends on the interfaces below
     it. The walk parses only the modules it cannot answer for, and hands those
     parses to the splice rather than repeating them.
   - **The record carries the import list**, even though an interface carries
     declarations and not dependencies. Deciding whether to skip a module means
     knowing what it imports before it has been read.
   - **A file id could not go into a record.** It is handed out in registration
     order, so an interface written down with one in it means something else in
     the process that reads it back, and module attribution is exactly what
     reads it. Interfaces are written with it zeroed and restamped on load.

   **What it is worth**, from `just bench-incremental` on 9,484 lines across 65
   files, one of which changed:

   | | full | incremental |
   | --- | --- | --- |
   | build | 668 ms | 399 ms |

   About 90 ms of each is process start and the linker, which no amount of
   skipping removes, so the compiler's own work goes from 578 ms to 309 ms.

   **Where the rest of it still goes, and it is worth naming precisely.** A
   skipped module still contributes its interface to the flattened program, and
   an interface carries the bodies of exported functions, generic or not. So the
   front end still walks every exported body of every module in the program even
   when it emits none of them. Removing that means a declaration form that
   carries a full Frost signature and no body: not `extern`, whose ABI loses the
   hidden out-pointer an aggregate return uses and which has nowhere to put
   parameter modes, `uses` sets or linearity. That is the next lever and it is a
   change to every pass that matches on a function declaration, which is why it
   is named here rather than done quietly as part of this step.

## Open questions

- **Does the self-hosted compiler grow this too?** `bootstrap/frost.frost` has no
  incremental or separate compilation either. The reference compiler is the one
  under a speed promise, and the self-hosted one is under a self-hosting promise,
  so the answer is probably no. Worth deciding rather than drifting into.
- **What is a project root?** *Settled, smallest answer.* The directory of the
  file named on the command line. A manifest would make it robust and nothing
  needs one yet. Note what this costs: the same library imported from two
  different roots has two identities and two caches.
- **Interfaces in what format?** *Settled, replaceably.* serde and JSON, for
  both interfaces and build records. The one requirement the design imposes is
  that it can hold a generic's AST, which rules out anything shaped like a C
  header and points at serializing the existing types, which is what this does.
- **What invalidates a record beyond content?** Nothing yet. A record does not
  name the compiler that wrote it, so changing the compiler and keeping a build
  directory reuses objects the new compiler would not have emitted. Deleting the
  build directory is the current answer, and a compiler version in the record is
  the obvious one.
- **`--test` does not use the cache.** A module answered for from a record is
  never read far enough to know it has `test` blocks, so `--incremental` refuses
  to combine with `--test` rather than silently running fewer tests.
