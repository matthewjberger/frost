# Separate compilation

What a module's compiled artifact contains, and what has to be in it for a
caller to compile against it without seeing the body, is the question this
document answers.

`frost --link --incremental` gives each module its own object and rebuilds a
module only when its own source or an imported interface changes.

## Why the shape has to change

Goal 9 in [philosophy.md](../design/philosophy.md) makes compilation speed a
promise. The constant factor is already handled: the bootstrap's full native
build clears the bar, which `just bench-scaling` is what says. This is the other
half, and it is not a constant factor. [roadmap.md](../roadmap.md) lists the
recipes.

`src/imports.rs` flattens every import into one AST. `resolve_imports` reads each
imported file, parses it, renames its private names, and splices the statements
into a single `Vec<Spanned<Statement>>`. Every pass after that runs over the
whole program. So a program's cost is whole-program by construction, and the two
things that grow worst are the two that most want bounding:

- Monomorphization runs to fixpoint over everything. A specialization is
  emitted once per program. Change one line in `main` and every specialization in
  every dependency is computed again.
- Nothing can be compiled independently, so nothing can be cached and
  incremental rebuild does not exist. Touching a leaf costs the same as a clean
  build.

Invisible at five thousand lines. The dominant cost at five hundred thousand.

## What a module is

A module is a file. That is already true of `import`, already true of the
`export` line, and inventing a second notion of module would mean two visibility
rules to explain instead of one.

A module's interface is its `export` line. Also already true. `private_renames`
in `src/imports.rs` mangles every top-level name a file does not export, which
means the export list is already the complete set of names another file can
depend on. Separate compilation does not need a new declaration form. It needs
the existing one to be written down in a file rather than reconstructed by
re-reading the source.

A specialization is emitted in the module that instantiates it. Not in the
module that declares the generic, which cannot know what its callers will ask
for, and not once per program, which is the thing being fixed. Two modules that
both instantiate `Stack<i64>` each emit their own copy, with module-local
linkage, and the duplicate code is the price of a module's work depending only
on the module.

An earlier version of this said the linker folds those copies, which is what
C++ and Rust do with COMDAT and weak symbols. That is not available here.
`cranelift_module::Linkage` has exactly `Import`, `Local`, `Preemptible`,
`Hidden` and `Export`, with no weak or COMDAT variant, so there is nothing to
ask the object writer for. Two options follow and the first is chosen:

- Emit a private copy per module (`Linkage::Local`). Needs no backend work at
  all, and duplicate specializations cost code size rather than correctness.
  This only works with separate objects: two copies in one object file is a
  duplicate symbol, not a fold, so the single-object path (`--native -o x.o`)
  still deduplicates across the whole program.
- Teach `cranelift-object` to emit COMDAT sections and add a `Linkage` variant
  for it. Better output, upstream work, and not on the critical path. Worth
  revisiting only if duplicated specializations measurably matter, which
  `FROST_MODULE_REPORT=1` is what would say: it counts how many specializations
  one object emits, how many separate objects emit, and how many more than one
  module asks for.

The fold was a design assumption that survived being written down and did not
survive being checked against the API.

## What the artifact contains

Compiling a module produces two things: an object file, and an interface the
compiler reads instead of the source when something imports it.

The interface has to carry everything a caller's passes consult about an imported
name today. Working through the pipeline, that is:

| what | why it must be in the interface | consulted by |
| --- | --- | --- |
| exported function signatures | call arity and types, and the return type | `ir::typecheck`, `ir::build` |
| parameter **modes** | the borrow at a call site is inserted from the callee's mode, not from the call | `param_modes`, `ownership` |
| exported struct layouts | field offsets, sizes, alignment | `ir::build`, both backends |
| exported enum layouts | tag values, payload offsets | same |
| which types are `linear` | consume-exactly-once is checked at the caller | `check_linearity` |
| failure sets (`-> T ! E`) | `?` lowering and the result type | `lower_failure_sets` |
| allocation capabilities (`uses Arena`) | the implicit parameter a caller has to supply | `lower_allocation_sources` |
| **generic bodies** | a specialization is stamped out at the caller, so the caller needs the AST | `ir::build` |
| compile-time parameter signatures | the bound checked at the call | `ir::build` |

The last one on the list is the one that decides the flavour of the whole design.
A generic's body is part of its interface. There is no way around it while
monomorphization is the only implementation of generics. The caller is what
chooses the type arguments, so the caller is what has to instantiate the template.
This is the same bargain C++ headers and Rust `#[inline]`/generic MIR make, and
it puts the bodies of exported generics in the interface alongside the
signatures.

That has one consequence worth stating plainly rather than discovering later.
Changing a generic's body is a change to the module's interface, and every
module that instantiates it has to be rebuilt. Changing a non-generic body is
not. So the interface should be hashed with the bodies of exported generics
included and the bodies of everything else excluded, and that hash is what
downstream rebuilds key on.

## What had to change first, and it was not the file format

Two things in the pipeline were wrong for this in ways a serialization format
could not paper over, and both touched the assumption every later pass rests on,
which is why they came before anything else.

Private name mangling depended on traversal order. `resolve_into` handed out
`module_tag` by the order files were visited, so a private `helper` became
`__m3_helper` in one program and `__m7_helper` in another. A module's symbol
names were therefore not a property of the module, which is exactly what
separate compilation requires. The tag is now an FNV-1a hash of the module's
path relative to the project root, so a module compiled alone and a module
compiled as part of a program produce the same symbols. FNV is written out
rather than taken from the standard library because the hash has to mean the
same thing in every build of the compiler, and `DefaultHasher` promises only
consistency within one version.

Monomorphization was a whole-program fixpoint. `expand_generic_structs` and the
specialization loop in `src/ir/build.rs` walked every statement in the flattened
program, and flattening had thrown away which module a statement came from. A
`Position` now carries a file id into `src/source_map.rs`, stamped during import
resolution, so every statement knows its module and every specialization records
the module that asked for it. The dedup that was one `emitted: HashSet<String>`
is keyed by module and name, and duplicate specializations across modules are
private copies rather than a fold.

## What it does now

Compiling a module writes an interface and an object, and `--incremental` skips
the modules an edit cannot reach.

`src/interface.rs` derives a `ModuleInterface` at the one place a module is
parsed, which is what stops it drifting from the source it describes. Three
checks run under `FROST_CHECK_INTERFACES`, which the test suite sets on every
compilation: it survives a JSON round trip, it declares everything it exports,
and it is closed, meaning every name a carried declaration reaches and this
module declares is carried too. `FROST_BUILD_FROM_INTERFACES=1` goes further and
reduces every imported module to what its interface says, and the whole test
suite runs a second time under it as `just test-interfaces`, so the sufficiency
claim is checked on every commit rather than the day the compiler starts relying
on it. That gate was itself checked by breaking the interface closure and
confirming it fails.

On the link path each module is its own compilation unit.
`IrModule::split_by_module` produces one part per module, each becomes its own
object, and the linker resolves the calls between them. `--native -o x.o` still
writes the single object its `-o` names, since that output is one file by
contract. Cross-module calls are declared rather than externed: a part carries
the other parts' functions in `imported` and declares them with the same
signature builder that built the definitions, because describing them as externs
would lose the hidden out-pointer an aggregate return uses and the two objects
would silently disagree about the ABI.

`--incremental` keeps a record and an object per module under `--build-dir`. The
decision is a fingerprint, in `src/build_cache.rs`: a hash of a module's own
source together with the interface hash of every module reachable through its
imports, transitively, since a generic this module instantiates can instantiate
one from further down. A module's interface hash is taken over the interface
with the bodies of ordinary functions blanked and the bodies of generics kept,
which is the distinction this document has been claiming since the top and is
the only thing that makes the cache worth having: an ordinary body is what a
module can change without rebuilding anything else, and it is most of what
anyone edits.

A skipped module contributes signatures, not bodies. `Statement::Declared` is a
Frost function's signature with no body, produced by `as_declaration` for the
functions whose bodies a caller does not need. A generic keeps its body,
because the caller stamps out the template. A type keeps its fields, because the
caller lays out its own frame with them. It is not an `extern`, which would have
been the tempting reuse: an extern means C linkage and a C ABI, which loses the
hidden out-pointer and has nowhere to put parameter modes, `uses` sets or
linearity.

What it is worth to the bootstrap is what `just bench-incremental` answers: it
builds a program across many files, changes one, and times a full build against
an incremental one. Read the two against each other rather than either alone,
since process start and the linker sit inside both and no amount of skipping
removes them. Against the compiler's own work the incremental build is a small
fraction of the full one, and the declaration form is most of that difference:
splicing bodies instead leaves the front end walking code it will never emit,
which the recipe's third column shows costs most of the win.

## Open questions

- Does the self-hosted compiler grow this too? *Done, and by a smaller route.*
  `selfhosted/frost.frost` is the compiler people will use, so the
  edit-compile loop goal 9 promises about is the one it runs. It emits one unit
  per module and assembles each to its own object, and it needs none of the
  machinery above to decide what is stale: the cache key is the emitted unit
  itself, since the compiler has just written what a module compiles to. See
  [the self-hosted compiler](self-hosted.md).
- What is a project root? *Settled, smallest answer.* The directory of the
  file named on the command line, or of the `frost.json` beside it. A module
  found through `-L`, `FROST_PATH` or the manifest is named relative to the root
  it was found under, and that label is what keeps its identity the same on
  another machine. See [modules.md](modules.md).
- Interfaces in what format? *Settled, replaceably.* serde and JSON, for
  both interfaces and build records. The one requirement the design imposes is
  that it can hold a generic's AST, which rules out anything shaped like a C
  header and points at serializing the existing types, which is what this does.
- What invalidates a record beyond content? Nothing yet. A record does not
  name the compiler that wrote it, so changing the compiler and keeping a build
  directory reuses objects the new compiler would not have emitted. Deleting the
  build directory is the current answer, and a compiler version in the record is
  the obvious one.
- `--test` does not use the cache. A module answered for from a record is
  never read far enough to know it has `test` blocks, so `--incremental` refuses
  to combine with `--test` rather than silently running fewer tests.
