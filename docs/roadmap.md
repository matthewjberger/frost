# What is left, and the order to do it in

Goal 9 in [philosophy.md](philosophy.md) makes compilation speed a promise
rather than a happy accident, and that promise has a bill. This is the bill,
sequenced so that nothing here gets built twice.

## The target

Competitive with Jai and Odin, which in practice means a full build in the
100,000 lines per second range rather than merely "fast for a compiler". That is
a number to measure against, not a feeling.

Where the bootstrap stands, from `just bench-scaling` on 58,107 lines:

| stage | rate |
| --- | --- |
| front end (`--emit-c`, 318 ms) | ~183,000 lines/sec |
| full build (`--native`, 349 ms) | ~166,000 lines/sec |

Both clear the bar. Code generation is 64 ms of that 349 ms build, with the
front end holding the rest, which is what says where the next hour of work
belongs. Cranelift is not the problem.

A benchmark is easy to get wrong in ways that read as a compiler result.
Generated programs that name a function `f32` time a parse error. Programs whose
`main` holds thousands of call sites make parallel code generation look like it
does nothing, because one function is one thread however many cores there are.
Re-run the benchmark before trusting any of it, and read what it generates.

## What is left

One thing, in the compiler people will run.

**Speed parity for the self-hosted compiler.** The numbers above are the
bootstrap's. The Frost compiler compiles a program in one pass on one thread,
whole-program: no parallel code generation, no per-module objects, no build
cache, no `--incremental`. Its front end is already fast, so this is backend and
build work rather than language work. See
[separate-compilation.md](separate-compilation.md) for what each piece means and
[self-hosting.md](self-hosting.md) for where its time goes.

The pieces, in the order they unlock each other:

1. **Per-module objects.** A module is already a compilation unit in the
   bootstrap: its interface is its `export` line, and a specialization is
   emitted in the module that instantiates it. The self-hosted compiler splices
   every module into one AST and emits one translation unit.
2. **A build cache keyed by module identity.** `module_tag_of` hashes what a
   module is rather than where it sat in a traversal, which is what makes a
   cache key possible at all.
3. **`--incremental`.** Rebuild a module when its own source or an imported
   interface changes, and not otherwise.
4. **Parallel code generation.** The type system is local and
   signature-based, so once signatures are collected, functions are independent.
   That is a large part of why the language is shaped the way it is.

Everything else the two compilers are held to is done: they accept the same
language, and what says so is a suite of programs run through both rather than a
claim. See [../selfhosted/README.md](../selfhosted/README.md).

## What the speed work already taught

The findings that cost the most to learn, and that apply again to whoever builds
the same thing in the Frost compiler.

**A generic's body is part of its interface.** The caller chooses the type
arguments, so the caller instantiates the template, so changing a generic's body
is an interface change and rebuilds every module that instantiates it. Changing
an ordinary body is not. That distinction is what a module fingerprint has to
encode, and it is the whole reason the cache pays.

**A skipped module contributes signatures, not bodies.** A module that
contributes its interface as it stands makes the front end walk bodies it will
never emit. `Statement::Declared` is a function's signature with no body, which
is what a module offers for every function whose body a caller does not need. A
generic still offers its body, because the caller is what stamps out the
template. In the bootstrap that took the skipped path from 309 ms of compiler
work to about 110 ms.

**A declared signature is not an `extern`.** An extern means C linkage and a C
ABI, and loses the hidden out-pointer an aggregate return uses along with
parameter modes, `uses` sets and linearity. It rides in `IrModule::imported`,
which already means "declared here, defined in another object".

**Private symbol names have to be a property of the module.** When they depend
on import traversal order, a private `helper` is `__m3_helper` in one program
and `__m7_helper` in another, and nothing downstream can be cached by module.

**Cranelift has no weak or COMDAT linkage**, so duplicate specializations across
modules are not folded and each module emits its own private copy.
`FROST_MODULE_REPORT=1` measures what that costs, and that measurement is the
only thing that would justify revisiting it.

**A shared cursor beats a static split.** Cost per function varies by more than
an order of magnitude and the expensive ones sit next to each other in a module,
so splitting the function list into equal chunks leaves one thread holding all
of them while the rest finish early. Handing out one function at a time from an
atomic cursor does not. On 10,401 functions, with `FROST_THREADS=n`:

| threads | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| code generation | 385 ms | 218 ms | 111 ms | 65 ms | 55 ms |

That is 7.0x on a machine with eight physical cores, which is about as close to
linear as this gets. Two things are needed to reach it: one ISA per thread
rather than one shared, and per-thread contexts, since the object is the only
serial part and only the defining step touches it.

## What is deliberately not on this list

**A trait system.** Two things that a trait bound expresses are in the language
already, and neither is a solver. What a type *is* is a `where` clause over a
closed vocabulary of questions the compiler answers for itself, checked at the
call (spec 11.4a). What can be *done* with a type is a capability bundle, a
struct whose fields are functions, passed as a compile-time argument and folded
to direct calls (spec 11.4b). Both are named at the call site. Nothing registers
into either, so there is nothing to be coherent about, no orphan rule, and no
method lookup. What is deliberately missing is inference: a bundle is never
found for you.

**Folding duplicate specializations across objects.** See the Cranelift note
above. It is measurable, and it is not measured to matter.

**Anything that would make expansion a language of its own.** A compile-time
list (spec 11.1c) unrolls a `for`, names an element, and answers an `if` over a
type predicate, and that is the whole of it. No compile-time string parsing, no
recursion, no unbounded loop, and nothing that reads the world: every construct
walks a list whose length the call fixed, so what expansion costs is bounded by
the program's own text.
