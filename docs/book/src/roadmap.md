# What is left

## What is left

**A container of resources is checked at both ends now, except where it is
freed.** Handing one out is refused: a function may answer with a resource out
of a borrowed parameter only where the place can be named by a run of fields,
which is what a caller can be told about, so `vec_get<File>` no longer compiles.
Dropping one is refused by a bound: `is_linear` joined the `where` vocabulary,
so a function that must not take a resource can say so, and the library
functions that would overwrite or forget an element hold themselves to elements
that are not resources. `vec_get`, `vec_set`, `vec_clear`, `map_put`, `map_get`,
`map_clear`, `option_is_some` and `option_unwrap_or` all carry the bound.
`map_get` stays a defaulting read for plain values rather than growing a second
fallible-accessor convention beside the sentinels the containers already use:
presence is `map_has`.

What is left is `vec_free` and `map_free`. They release the storage and nothing
consumes the elements, and they cannot carry the bound: `Vec<T>` and
`Map<K, V>` are themselves resources, so the free is the only thing that
consumes one, and a bound there would leave a container of resources with no way
to be discharged at all. A slot is reached by a number worked out while the
program runs, so there is no place a check could name to say the elements were
consumed. Doing it before the block goes is the caller's obligation, and
`std/ecs.frost` is what that looks like: a `Vec<Table>` where `Table` is a
resource, released one element at a time through a `ref` borrow before the
storage goes. The shape is writable, it is written, and it is not checked.

The pool rule refuses `Slab<File>` outright for the same shape, which it can
because a pool is recognized by its declaration rather than by a call.

What a type holds is read from the types a program *forms* rather than the ones
it spells out. A call that answers with an instantiation makes one without
anyone writing its name, so `held := option_some($File, ...)` used to leave
`Option<File>` ordinary data and the obligation on the resource inside it went
in and did not come out. Three things had to be true together: the ownership
pass types a call as the callee's return type with that call's own type
arguments put in, a variant's payload is held by its enum the way a field is
held by its struct, and the walk runs over the instantiations specialization
forms rather than over the source alone.

The wgpu binding is generated with a safe wrapper per call now, so a program
that draws a triangle writes no `unsafe` of its own for the graphics API: the
perimeter is one generated file. What is left there is the same shape as the
compiler's byte access, a perimeter rather than a proof, since a descriptor is
a pointer C reads without checking.

Everything else this document was opened for is built. The two compilers accept
the same language, and what says so is a suite of programs run through both
rather than a claim. See [the self-hosted compiler](impl/self-hosted.md).

## The speed promise, and how to check it

Goal 9 in [philosophy.md](design/philosophy.md) makes compilation speed a
promise rather than a happy accident: competitive with Jai and Odin, which in
practice means a full build in the 100,000 lines per second range rather than
merely "fast for a compiler".

That is a thing to measure, not to record. A number written down here is a
number about one machine on one day, and it goes stale without anyone noticing
it has. So run it:

| what | recipe |
| --- | --- |
| the bootstrap, front end and full build, over a range of program sizes | `just bench-scaling` |
| the self-hosted compiler on its own source, both backends | `just bench-selfhost` |
| what a rebuild costs against a whole-program build | `just bench-selfhost-incremental` |

Take warm runs only: the first run after a build measures the file cache.

Both compilers clear the target, and the assembly backend is the faster of the
self-hosted pair by a wide margin, since it writes machine code rather than
handing text to a C compiler. An incremental rebuild that changed nothing costs
a fraction of a whole-program build; the fraction is what the recipe answers.

The six pieces that took it there:

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
4. **Both backends split.** The C backend emits one unit per module too. It
   differs from the assembly case in exactly two ways, and they were the whole
   of the work: a C unit cannot call what it has not seen, so every unit carries
   the runtime prototypes, the type definitions and a prototype for every
   function and specialization in the program, with only the bodies differing;
   and the mark that says a type is already written has to be cleared per unit,
   since a per-module build writes the definitions several times.
5. **A table over a type's fields** (spec 11.1d), which is what the layout
   tables a renderer writes by hand are. `for field in fields(T)` expands once
   per field, and `offset_of`, `sizeof`, the type predicates and `field_count`
   are what may be asked of one. Every one of those is a number the compiler
   worked out to lay the type out, so a table written this way cannot drift from
   the struct. Reflection by name stays refused: a field's name is not readable,
   which keeps this a layout question rather than a second language for asking
   about types.
6. **Parallel work**, the part of it that pays. The assembler runs go out
   together, one OS thread each, which is why even a first build beats the
   whole-program one: the machine-code step is where a build's time is, not the
   compiler. Emitting could be parallel too, since the type system is local and
   signature-based, but the compiler writes straight to one file at a time and
   emitting is a small share of a build, so buffering a unit in memory to
   parallelize it would be work spent where the time is not. Check that share
   again before believing it, since it is what the decision rests on.

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
template. In the bootstrap that cut the work a skipped module costs to roughly a
third.

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
atomic cursor does not. `FROST_THREADS=n` sweeps it on a program with enough
functions to matter, and code generation scales close to linearly in physical
cores, with hyperthreads adding little. Two things are needed to reach that:
one ISA per thread rather than one shared, and per-thread contexts, since the
object is the only serial part and only the defining step touches it.

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
