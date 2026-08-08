# Separate compilation

A module's compiled artifact is an object file and an interface. The interface
carries everything a caller's passes consult about an imported name, so a
caller compiles against it while the body stays unread.

`frost --link --incremental` gives each module its own object and rebuilds a
module only when its own source or an imported interface changes.

## The whole-program pipeline

Goal 9 in [philosophy.md](../design/philosophy.md) makes compilation speed a
promise. The bootstrap's full native build covers the constant factor, which
`just bench-scaling` measures. Separate compilation covers the half that grows
with the program, and `just bench-incremental` measures that.

`src/modules/imports.rs` flattens every import into one AST. `resolve_imports`
reads each imported file, parses it, renames its private names, and splices the
statements into a single `Vec<Spanned<Statement>>`. Every pass after that runs
over the whole program, so a program's cost is whole-program by construction,
and two costs grow with it:

- Monomorphization runs to fixpoint over everything. A specialization is
  emitted once per program. Change one line in `main` and every specialization
  in every dependency is computed again.
- A flattened program leaves nothing to cache, so touching a leaf costs a clean
  build.

At five thousand lines both are invisible. At five hundred thousand they
dominate.

## What a module is

A module is a file, which is what `import` and the `export` line already name.

A module's interface is its `export` line. `private_renames` in
`src/modules/imports.rs` mangles every top-level name a file leaves off that
line, so the export list is the complete set of names another file can depend
on. Separate compilation stores that list in a file, and the compiler reads it
in place of the source.

A specialization is emitted in the module that instantiates it. The module
declaring the generic has no way to know what its callers will ask for, and the
caller is what chooses the type arguments. Two modules that both instantiate
`Stack<i64>` each emit their own copy, with module-local linkage, and the
duplicate code is the price of a module's work depending only on the module.

Folding those copies at link time, the way C++ and Rust do with COMDAT and weak
symbols, takes a linkage the object writer can be asked for.
`cranelift_module::Linkage` has exactly `Import`, `Local`, `Preemptible`,
`Hidden` and `Export`, with no weak or COMDAT variant. Two options follow, and
the compiler takes the first:

- Emit a private copy per module (`Linkage::Local`). This needs no backend work
  at all, and duplicate specializations cost code size. It works with separate
  objects only: two copies in one object file are a duplicate symbol, so the
  single-object path (`--native -o x.o`) still deduplicates across the whole
  program.
- Teach `cranelift-object` to emit COMDAT sections and add a `Linkage` variant
  for it, which is upstream work. `FROST_MODULE_REPORT=1` measures what it
  would buy: it counts how many specializations one object emits, how many
  separate objects emit, and how many more than one module asks for.

## What the artifact contains

Compiling a module produces two things: an object file, and an interface the
compiler reads instead of the source when something imports it.

The interface carries everything a caller's passes consult about an imported
name. Working through the pipeline, that is:

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

The last row shapes the whole design. A generic's body is part of its
interface. Monomorphization is the only implementation of generics, and the
caller is what chooses the type arguments, so the caller is what instantiates
the template and needs the AST to do it. This is the same bargain C++ headers
and Rust `#[inline]`/generic MIR make, and it puts the bodies of exported
generics in the interface alongside the signatures.

Changing a generic's body changes the module's interface, and every module that
instantiates it is rebuilt. Changing an ordinary body leaves the interface
alone. So the interface hash covers the bodies of exported generics and skips
every other body, and that hash is what downstream rebuilds key on.

## Symbols and specializations belong to the module

A module's private symbol names are a property of the module. `resolve_into`
tags each module with an FNV-1a hash of its path relative to the project root,
so a private `helper` gets the same mangled name whether the module is compiled
alone or as part of a program. The compiler carries its own FNV because the
hash has to mean the same thing in every build of the compiler, and
`DefaultHasher` promises consistency within one version.

Monomorphization records the module that asked for each specialization. A
`Position` carries a file id into `src/source_map.rs`, stamped during import
resolution, so every statement knows its module. `expand_generic_structs` and
the specialization loop in `src/ir/build.rs` walk the flattened program with
that id in hand, and the dedup is keyed by module and name. Duplicate
specializations across modules are private copies.

## Interfaces, objects and the cache

Compiling a module writes an interface and an object, and `--incremental` skips
the modules an edit cannot reach.

`src/modules/interface.rs` derives a `ModuleInterface` at the one place a module
is parsed, from the same AST the rest of the pipeline runs on. Three checks run
under `FROST_CHECK_INTERFACES`, which the test suite sets on every compilation:
it survives a JSON round trip, it declares everything it exports, and it is
closed, meaning every name a carried declaration reaches and this module
declares is carried too. `FROST_BUILD_FROM_INTERFACES=1` goes further and
reduces every imported module to what its interface says. The whole test suite
runs a second time under it as `just test-interfaces`, so the sufficiency claim
is checked on every commit. A broken closure fails the gate.

On the link path each module is its own compilation unit.
`IrModule::split_by_module` produces one part per module, each becomes its own
object, and the linker resolves the calls between them. `--native -o x.o` still
writes the single object its `-o` names, since that output is one file by
contract. A part carries the other parts' functions in `imported` and declares
them with the same signature builder that built the definitions, so the two
objects agree on the ABI. Describing them as externs would lose the hidden
out-pointer an aggregate return uses.

`--incremental` keeps a record and an object per module under `--build-dir`. The
decision is a fingerprint, in `src/modules/build_cache.rs`: a hash of a module's
own source together with the interface hash of every module reachable through
its imports, transitively, since a generic this module instantiates can
instantiate one from further down. A module's interface hash is taken over the
interface with the bodies of ordinary functions blanked and the bodies of
generics kept. An ordinary body is what a module can change while every other
module stays cached, and it is most of what anyone edits.

A skipped module contributes signatures. `Statement::Declared` is a Frost
function's signature with no body, produced by `as_declaration` for the
functions whose bodies a caller does not need. A generic keeps its body,
because the caller stamps out the template. A type keeps its fields, because the
caller lays out its own frame with them. `extern` means C linkage and a C ABI,
which loses the hidden out-pointer and has nowhere to put parameter modes,
`uses` sets or linearity, so a declaration is its own statement form.

`just bench-incremental` measures what the cache is worth to the bootstrap: it
builds a program across many files, changes one, and times a full build against
an incremental one. Read the two against each other, since process start and the
linker sit inside both and no amount of skipping removes them. Against the
compiler's own work the incremental build is a small fraction of the full one,
and the declaration form is most of that difference. The recipe's third column
splices bodies in place of declarations, which leaves the front end walking code
it will never emit and costs most of the win.

## Edges of the design

- The self-hosted compiler gets there by a smaller route.
  `selfhosted/frost.frost` is the compiler people will use, so the
  edit-compile loop goal 9 promises about is the one it runs. It emits one unit
  per module and assembles each to its own object, and it decides what is stale
  from the emitted unit itself, since the compiler has just written what a
  module compiles to. See [the self-hosted compiler](self-hosted.md).
- A project root is the directory of the file named on the command line, or of
  the `frost.json` beside it. A module found through `-L`, `FROST_PATH` or the
  manifest is named relative to the root it was found under, and that label
  keeps its identity the same on another machine. See
  [modules.md](modules.md).
- Interfaces and build records are serde and JSON. The format has to hold a
  generic's AST, which rules out anything shaped like a C header and points at
  serializing the existing types.
- Content is the only thing that invalidates a record. A record does not name
  the compiler that wrote it, so changing the compiler and keeping a build
  directory reuses objects the new compiler would not have emitted. Deleting the
  build directory is the current answer, and a compiler version in the record is
  the obvious one.
- `--test` does not use the cache. A module answered for from a record is
  never read far enough to know it has `test` blocks, so `--incremental` refuses
  to combine with `--test`, which would otherwise run fewer tests than the files
  hold.
