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

**Speed parity for the self-hosted compiler.** The Frost compiler compiles a
program in one pass on one thread, whole-program: no parallel code generation,
no per-module objects, no build cache, no `--incremental`. This is backend and
build work rather than language work. See
[separate-compilation.md](separate-compilation.md) for what each piece means and
[self-hosting.md](self-hosting.md) for where its time goes.

Where it stands on its own source, 14,273 lines, from `just bench-selfhost`:

| compiler and backend | full build |
| --- | --- |
| bootstrap front end (`--emit-c`) | ~72,000 lines/sec |
| self-hosted, C backend | ~145,000 lines/sec |
| self-hosted, assembly backend | ~130,000 lines/sec |

Both self-hosted backends clear the 100,000 lines per second bar on a full
build, so what is left of this item is about not rebuilding what did not change,
and about using more than one core, rather than about raw throughput. Warm runs
only: the first run after a build measures the file cache. Measure
before optimizing anything here: the assembly backend read as eight times slower
than that until the measurement stopped going through a pipe, and the one real
find under it was a slot lookup that added up every local's size at each mention
of a name, quadratic in a function's locals. Recording the slot with the local
took the assembly backend from 612 ms to 122 ms on this source.

The pieces, in the order they unlock each other:

1. **Per-module objects.** Done. `--incremental` emits one assembly unit per
   module and assembles each to its own object. A function goes to the module
   that declared it, a specialization to the module that declared its template,
   and a string or a float to the module whose source wrote it, all decided by
   comparing an offset against the module's range in the one source buffer.
2. **A build cache.** Done, and smaller than it was going to be. The cache key
   is the emitted assembly itself: the compiler has just written what a module
   compiles to, so whether that module's object is stale is a comparison of
   those bytes against last build's. No source hash, no interface fingerprint,
   no dependency graph, because the answer is already in hand. The C runtime is
   cached the same way, since it is a compilation unit like any other.
3. **`--incremental`.** Done, `just bench-selfhost-incremental`:

   | build | |
   | --- | --- |
   | whole program | ~1,500 ms |
   | incremental, first build | ~1,060 ms |
   | incremental, nothing changed | ~330 ms |

   Even a first build is faster than the whole-program one, because the
   assembler runs go out at once, and every build after it costs a fifth. The
   compiler it produces is byte for byte the one the whole-program build
   produces, which a test checks rather than a claim.
4. **Parallel work.** Partly done, and the part that pays. The assembler runs
   go out together, one OS thread each, which halves a first build: the
   machine-code step is where a build's time is, not the compiler. The type
   system is local and signature-based, so emitting could be parallel too, but
   the compiler writes straight to one file at a time and emitting is 150 ms of
   a 1,060 ms build, so buffering a unit in memory to parallelize it would be
   work spent where the time is not.

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
