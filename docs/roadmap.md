# What is left, and the order to do it in

Goal 9 in [philosophy.md](philosophy.md) makes compilation speed a promise
rather than a happy accident, and that promise has a bill. This is the bill,
sequenced so that nothing here gets built twice.

## The target

Competitive with Jai and Odin, which in practice means a full build in the
100,000 lines per second range rather than merely "fast for a compiler". That is
a number to measure against, not a feeling. Both compilers clear it; the
measurements are below.

## What is left

Two things, both contained, neither on the critical path of anything else.

**Extended const-eval, for layout tables.** A vertex format, a shader's uniform
layout and a descriptor table are all the same shape: a table of offsets and
sizes derived from a struct the program already declares. Today they are written
out by hand beside the struct and drift from it. The compiler knows every one of
those numbers, because it laid the struct out.

The pieces this needs are mostly built. `sizeof(T)` is a compile-time constant,
a compile-time list expands (11.1c) with a `for` that unrolls and an `if` over a
type predicate that prunes, and `columns<T, N>` already reflects over a struct's
fields to synthesize one array per field. What is missing is reaching a field's
*offset* and *type* from an expansion, so a table can be written once over
whatever fields the struct has.

The line to hold is the one 11.1c holds: no compile-time string parsing, no
recursion, no unbounded loop. A layout table is a walk over a field list whose
length the struct fixes, which is the same bounded shape, so this stays inside
the rule rather than reopening it. Field reflection by name is the thing to
refuse: `has_field(T, "position")` is the string-keyed predicate 11.4a already
ruled out, and a table built by walking every field needs no name to be written
as a literal anywhere.

**`--incremental` for the self-hosted C backend.** The assembly backend splits a
program into one unit per module and assembles only what changed. The C backend
still emits one translation unit, so a build through it pays for the whole
program every time, and that is the slower of the two paths to begin with: a C
compile of the emitted 13,000 lines is about 1,200 ms against about 750 ms to
assemble.

The design is already proven, and the split is the same one: a function goes to
the module that declared it, a specialization to the module that declared its
template. Two things differ from the assembly case and are the whole of the
work. A C unit needs the type definitions and the prototypes of everything it
calls, where an assembly unit needs neither, so each unit carries the shared
declarations and its own bodies. And a string literal is emitted inside the
function that holds it rather than in a data section, which removes the question
of where the data goes.

Everything the two compilers are held to is done: they accept the same language,
and what says so is a suite of programs run through both rather than a claim.
See [../selfhosted/README.md](../selfhosted/README.md).

## What is done, and what it cost

The speed bill this file was opened for is paid. Both compilers clear the target
on a full build, and the self-hosted one rebuilds only what changed.

**Where the bootstrap stands**, from `just bench-scaling` on 58,107 lines:

| stage | rate |
| --- | --- |
| front end (`--emit-c`, 318 ms) | ~183,000 lines/sec |
| full build (`--native`, 349 ms) | ~166,000 lines/sec |

Code generation is 64 ms of that 349 ms build, with the front end holding the
rest. Cranelift is not the problem.

**Where the self-hosted compiler stands** on its own source, 14,273 lines, from
`just bench-selfhost`, warm runs only since the first run after a build measures
the file cache:

| compiler and backend | full build |
| --- | --- |
| bootstrap front end (`--emit-c`) | ~72,000 lines/sec |
| self-hosted, C backend | ~145,000 lines/sec |
| self-hosted, assembly backend | ~130,000 lines/sec |

**And what a rebuild costs**, from `just bench-selfhost-incremental`:

| build | |
| --- | --- |
| whole program | ~1,500 ms |
| incremental, first build | ~1,060 ms |
| incremental, nothing changed | ~330 ms |

The four pieces that took it there:

1. **Per-module objects.** `--incremental` emits one assembly unit per module
   and assembles each to its own object. A function goes to the module that
   declared it, a specialization to the module that declared its template, and a
   string or a float to the module whose source wrote it, all decided by
   comparing an offset against the module's range in the one source buffer.
2. **A build cache**, smaller than it was going to be. The cache key is the
   emitted assembly itself: the compiler has just written what a module compiles
   to, so whether that module's object is stale is a comparison of those bytes
   against last build's. No source hash, no interface fingerprint, no dependency
   graph, because the answer is already in hand. The C runtime is cached the
   same way, being a compilation unit like any other.
3. **`--incremental`**, which produces byte for byte the compiler the
   whole-program build produces. A test checks that rather than a claim.
4. **Parallel work**, the part of it that pays. The assembler runs go out
   together, one OS thread each, which is why even a first build beats the
   whole-program one: the machine-code step is where a build's time is, not the
   compiler. Emitting could be parallel too, since the type system is local and
   signature-based, but the compiler writes straight to one file at a time and
   emitting is 150 ms of a 1,060 ms build, so buffering a unit in memory to
   parallelize it would be work spent where the time is not.

A benchmark is easy to get wrong in ways that read as a compiler result.
Generated programs that name a function `f32` time a parse error. Programs whose
`main` holds thousands of call sites make parallel code generation look like it
does nothing, because one function is one thread however many cores there are.
Measuring through a pipe made both self-hosted backends read as ten times slower
and hid the difference between them entirely. Re-run the benchmark before
trusting any of it, and read what it generates.

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

**SIMD types and intrinsics.** The layout work that makes vectorization
possible is already done and is the half that matters: `columns<T, N>` gives a
C compiler separate homogeneous arrays with nothing aliasing between them, and
the small math functions are marked `inline` so it sees through them. The C
backend is the performance path by the same decision that left the assembly
backend without an inliner, and clang at `-O2` vectorizes what that layout
allows. An `f32x4` in the language would put the work where it pays least: the
assembly backend would need vector registers, a vector ABI and per-instruction
emission, and it is the portable-and-fast path rather than the peak one. The
vocabulary is also the wrong shape for this language, since shuffles, blends and
masks are open-ended and target-specific where every other vocabulary here is
closed. A program that truly needs intrinsics writes the kernel in C and calls
it, which the FFI already carries. Revisit when a measured Frost program says
vector width is its limit, and check first whether the C compiler vectorized it
and why not.

**Anything that would make expansion a language of its own.** A compile-time
list (spec 11.1c) unrolls a `for`, names an element, and answers an `if` over a
type predicate, and that is the whole of it. No compile-time string parsing, no
recursion, no unbounded loop, and nothing that reads the world: every construct
walks a list whose length the call fixed, so what expansion costs is bounded by
the program's own text.
