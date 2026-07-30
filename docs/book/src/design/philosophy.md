# Design philosophy, goals, and non-goals

Frost is a data-oriented systems language. It is built on the view that a
program is a description of *how data is laid out and transformed*, not a society
of objects exchanging messages. Every major decision in the language follows from
taking that seriously.

## Data-oriented design, not object-oriented programming

Object-oriented programming organizes code around objects, bundles of data
and the methods that act on them, related by inheritance, reached through
references, and freed by destructors or a garbage collector. It optimizes for a
particular human intuition ("a `Player` *is a* `Character` and *has an*
`Inventory`") and pays for it in indirection, virtual dispatch, pointer chasing,
scattered allocations, hidden lifetimes, and cache-hostile memory layouts.

Data-oriented design starts from the opposite question. What does the data
look like, and how does it flow? The layout of memory is the primary design
artifact, and the code is written to transform that layout efficiently. The
practical consequences run through all of Frost.

| Concern            | Object-oriented default            | Frost (data-oriented)                                  |
| ------------------ | ---------------------------------- | ------------------------------------------------------ |
| Grouping           | Objects bundle data + behavior     | **Plain structs** hold data; **free functions** transform it |
| Reuse / hierarchy  | Inheritance, virtual methods       | **Composition** and **`match` on enums**; no inheritance |
| Polymorphism       | Dynamic dispatch (vtables)         | **Monomorphized generics** + **function pointers** (static) |
| Identity / linking | References everywhere              | **Generational handles** into pools; references are second-class |
| Lifetime           | GC or destructors                  | **Linear resources** consumed exactly once; no hidden `Drop` |
| Memory             | Per-object heap allocation         | **Pools / explicit allocation**; contiguous, predictable layout |

The common thread is that behavior is not attached to data. A struct is just
its fields. To do something with it you call a free function and pass it in.
There are no methods, no `self`, no inheritance, no vtables. That is the design,
not a limitation it works around. Separating data from the code that walks it is
what makes the layout visible, the control flow explicit, and the machine's
actual work predictable.

### Which OOP features are deliberately absent

- No classes, methods, or `self`. Structs are data. Functions are functions.
- No inheritance or interfaces. Reuse comes from composition and generics.
- No virtual dispatch. Higher-order code uses function pointers (chosen
  explicitly), and polymorphism is resolved at compile time by monomorphization.
- No garbage collector and no destructors. Cleanup is a *linear* obligation
  the type system tracks, and long-lived data is addressed by *handles* rather
  than by references the runtime must keep alive.
- No implicit anything. No hidden allocations, no hidden copies of large
  values beyond an explicit move, no hidden control flow behind an operator.

## Goals

1. Make the data layout the design. The programmer should always be able to
   see how a value is represented and where it lives. Aggregates have a defined,
   inspectable layout, pools are contiguous, and nothing is boxed implicitly.
2. Memory safety without a garbage collector and without lifetime
   annotations. Safety comes from making the dangerous shapes unrepresentable
   (borrowing is a parameter mode, and the one borrow a program writes down,
   `ref T`, has nowhere to be stored) and from a few local rules (moves,
   exclusivity, linearity, generational handles, and the frame and region
   checks that hold a returned borrow to storage outliving the call). See
   [memory-safety.md](memory-safety.md).
3. Zero-cost, static polymorphism. Generics monomorphize, and a function
   that varies is a compile-time argument (`$f`) rather than a pointer, so the
   call in the inner loop is direct. How *many* arguments a call makes can vary
   the same way: a compile-time list decides an arity, so a query over any
   number of components is one function rather than one per count, and it emits
   what a hand-written one does. You pay for abstraction at compile time,
   not at run time. Function pointers remain for the cases that are genuinely
   dynamic, and they are honest about costing an indirect call. Neither backend
   devirtualizes one, by design, because goal 7 says the lowering is what you
   read rather than what an optimizer left behind.
4. Cleanup as a tracked obligation, not an implicit call. `linear` resources
   must be consumed exactly once. This replaces `Drop` and finalizers with
   something the compiler checks and the reader can see, and it makes error
   values non-ignorable.
5. Speak C fluently going out. `extern fn` reaches the entire C ecosystem
   with no glue, including the two shapes that are easy to leave out and then
   be stuck without: a function that returns a struct by value, which follows
   the target's real C ABI rather than Frost's own convention, and a callback,
   which is a `$` function parameter plus a context taken by `move` rather than
   a raw `^u8`. See [c-compatibility.md](../impl/c-compatibility.md) and
   [callbacks.md](callbacks.md).
6. One typed IR, three execution paths, kept honest. The AST lowers to a
   single typed IR from which a Cranelift backend and a portable C backend emit
   and an IR interpreter runs directly, and a differential test puts every
   program through them and asserts they agree. Independent paths that must
   match catch miscompilations a single backend would hide.
7. Predictability over cleverness. The generated code should be something you
   can reason about. Simple, explicit lowerings are preferred to clever ones.
8. The two compilers accept the same language. The Rust bootstrap and the
   self-hosted compiler are held to the same surface, and a divergence is a bug
   in whichever one is behind rather than a division of labour. This is what
   makes the language buildable from a Rust toolchain and nothing else: the
   bootstrap compiles the Frost compiler, the Frost compiler compiles itself,
   and no seed binary has to exist or be trusted. Every feature lands in both,
   and what still differs between them is listed in
   [the self-hosted compiler](../impl/self-hosted.md).
9. Compilation stays fast as programs grow. Not fast at the sizes tested so
   far, which any compiler manages, but fast on a curve that does not turn over.
   This is a promise rather than a happy accident, and it has a bill attached.
   Whole-program monomorphization with imports flattened into one AST is the
   shape that contradicts it, so separate compilation is an obligation this goal
   takes on. The bar is the one Jai and Odin set, a full build in the 100,000
   lines per second range, and a full native build clears it. That is the
   constant factor, and it is done. So is the shape. Each module is its own
   object and `--incremental` rebuilds only the modules an edit can reach. Both
   are measured by commands rather than remembered, `just bench-scaling` and
   `just bench-incremental`, which is deliberate: a rate written into a document
   is a rate about one machine on one day, and it goes stale without anyone
   noticing it has. See [self-hosting.md](../impl/self-hosted.md) and
   [separate-compilation.md](../impl/separate-compilation.md).
   [roadmap.md](../roadmap.md) is the surrounding order of work.

## Non-goals

- Not object-oriented. No inheritance, methods, or dynamic dispatch will be
  added to make Frost feel like an OO language. If a problem wants an object
  graph reached by references, that is a sign to reshape it around data and
  handles.
- Not a garbage-collected language. Automatic, tracing reclamation is out of
  scope. Ownership is explicit through moves, linear resources, and pools.
- Not lifetime-annotated. Frost will not grow lifetime variables or region
  syntax. The second-class-reference rule is the deliberate trade that removes
  the need for them.
- Not a stable C-callable ABI. Frost calls C, and there is no attribute that
  exposes a Frost function to a C caller. The emitted C is an internal lowering,
  not an interface anyone should link against, and that asymmetry is what keeps
  the backend simple. The one place C does call Frost is a registered callback,
  which works because the compiler picked the function and its signature rather
  than a user promising one, so it is a hole in the direction of travel and not
  in the ABI promise.
- Not maximally general. Frost intentionally omits capturing closures
  (function pointers and compile-time function arguments instead), exceptions
  (errors are failure sets and linear enum returns), and implicit conversions
  that hide cost.
- Not trait-based. There are no traits, no coherence rules, and no bound
  solving. A `where` clause is a precondition over a fixed vocabulary of
  questions the compiler already answers about a type, not a set of operations
  a type registers into, so nothing implements it and there is nothing to
  resolve. A generic algorithm takes the functions it needs as compile-time
  arguments, which is what makes its calls direct, and what other languages
  call an interface is a struct whose fields are function pointers. Do not reach for a generic
  sort, hash or equality that works over everything. Write the one you need over
  the layout you have, and pass `$compare` or `$hash` when it varies. The
  self-hosted compiler is 5,000 lines and wanted a generic function three times,
  which is the shape of the language working rather than a gap in it.

  This is also what keeps the front end near-linear. Coherence checking, bound
  solving and method resolution are among the passes that dominate other
  compilers' front ends, and their absence is measured in
  [self-hosting.md](../impl/self-hosted.md), not assumed.
- Not access-controlled per declaration. There is no `pub`, no private
  marker on any item, and no `pub(crate)` refinement of one. Every struct field
  is public, so encapsulation by field privacy is out of scope. A module does
  have a surface, and it is one line: an `export` at the top of a file lists
  what the file offers, everything else in it is private, and an import is not
  transitive. One line to read rather than a marker per declaration, and the
  reasoning is in
  [declarations.md](../reference/declarations.md).
- Not a research vehicle for a novel type theory. The type system is a means
  to make data-oriented code safe and fast, not an end in itself.

## Why this project exists

Frost is the language. It is not a prototype for a later one, not a spike, and
not an exercise whose lessons get carried somewhere else. The goal is a real,
finished implementation of this design: ownership without lifetimes, linear
resources replacing `Drop`, generational handles unified with a borrow
discipline, a typed IR feeding multiple backends, and specialization-only
generics. The bar for done is that those hold up against real code and a real
code generator, and then that people can write real programs with them.

That is why the differential oracle matters. The point is that its independent
backends *agree*, which is strong evidence that the semantics are pinned down
rather than accidental. Every proposition in the design has been driven to
running, differential-tested code.
