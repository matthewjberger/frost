# Separate compilation

This is the design, not the implementation. Item 3 in [roadmap.md](roadmap.md)
says not to start it by writing code, and this document is what it asks for
instead: what a module's compiled artifact contains, and what has to be in it for
a caller to compile against it without seeing the body.

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
both instantiate `Stack<i64>` each emit it, and the linker folds them. That is
the standard arrangement and it is what makes a module's work depend only on the
module.

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

1. **Make symbol names a property of the module.** Content-derived module tags,
   no format work, no new files. Verifiable on its own: the same module imported
   from two different programs produces byte-identical symbols.
2. **Write the interface out and read it back**, while still compiling the whole
   program. The compiler produces an interface per module and then ignores it,
   except for a check that reading it back gives the same answers as reading the
   source. This is the differential oracle for this feature, and it is what keeps
   the class of bug that "passes the test suite and corrupts output" out.
3. **Make monomorphization per-module.** Still one build, still one object, but
   the specialization sets are computed per module and the linker folds the
   duplicates. Correctness is carried by the existing tests and the fixpoint.
4. **Compile a module from interfaces alone.** Only now does the compiler stop
   reading imported source. Everything it needs is already written down and
   already verified to agree with the source.
5. **Cache and skip.** Rebuild a module only when its own source or an imported
   interface hash changes. This is the step that pays, and it pays only because
   the four before it made it a scheduling question rather than a correctness one.

## Open questions

- **Does the self-hosted compiler grow this too?** `bootstrap/frost.frost` has no
  incremental or separate compilation either. The reference compiler is the one
  under a speed promise, and the self-hosted one is under a self-hosting promise,
  so the answer is probably no. Worth deciding rather than drifting into.
- **What is a project root?** Module identity is a path relative to something, and
  right now there is nothing. The smallest answer that works is the directory of
  the file named on the command line, and the smallest thing that would make it
  robust is a manifest. Neither is urgent, but the first one has to be picked
  before step 1.
- **Interfaces in what format?** Deliberately last, because it is the least
  interesting decision here and the easiest to change. The one requirement the
  design imposes is that it can hold a generic's AST, which rules out anything
  shaped like a C header and points at serializing the existing types.
