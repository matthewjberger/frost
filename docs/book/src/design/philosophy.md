# Design philosophy, goals, and non-goals

Frost is a data-oriented systems language. A program is a description of *how
data is laid out and transformed*, and every major decision in the language
follows from taking that seriously.

## Data-oriented design

Object-oriented programming organizes code around objects, bundles of data
and the methods that act on them, related by inheritance, reached through
references, and freed by destructors or a garbage collector. It optimizes for a
particular human intuition ("a `Player` *is a* `Character` and *has an*
`Inventory`") and pays for it in indirection, virtual dispatch, pointer chasing,
scattered allocations, hidden lifetimes, and cache-hostile memory layouts.

Data-oriented design starts from the shape of the data and the way it flows. The
layout of memory is the primary design artifact, and the code is written to
transform that layout efficiently.

| Concern            | Object-oriented default            | Frost (data-oriented)                                  |
| ------------------ | ---------------------------------- | ------------------------------------------------------ |
| Grouping           | Objects bundle data + behavior     | **Plain structs** hold data; **free functions** transform it |
| Reuse / hierarchy  | Inheritance, virtual methods       | **Composition** and **`match` on enums**; no inheritance |
| Polymorphism       | Dynamic dispatch (vtables)         | **Monomorphized generics**; the call names the function it goes to |
| Identity / linking | References everywhere              | **Generational handles** into pools; references are second-class |
| Lifetime           | GC or destructors                  | **Linear resources** consumed exactly once; no hidden `Drop` |
| Memory             | Per-object heap allocation         | **Pools / explicit allocation**; contiguous, predictable layout |

Behavior lives apart from data. A struct is its fields. To do something with it
you call a free function and pass it in. Separating data from the code that
walks it keeps the layout visible and the machine's actual work predictable.

### Object-oriented features Frost omits

- No classes, methods, or `self`. Structs are data. Functions are functions.
- No inheritance or interfaces. Reuse comes from composition and generics.
- No virtual dispatch, and nothing to devirtualize. A call names the function
  it goes to, so higher-order code names it at the call as a compile-time
  argument and polymorphism is resolved by monomorphization. A `fn(..)` value is
  still built and handed to C, which is where a callback is registered.
- No garbage collector and no destructors. Cleanup is a *linear* obligation
  the type system tracks, and long-lived data is addressed by *handles* rather
  than by references the runtime must keep alive.
- No implicit anything. No hidden allocations, no hidden copies of large
  values beyond an explicit move, no hidden control flow behind an operator.

## Goals

1. Make the data layout the design. You can always see how a value is
   represented and where it lives. Aggregates have a defined, inspectable
   layout, pools are contiguous, and nothing is boxed implicitly.
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
   what a hand-written one does. You pay for abstraction while you compile.
   A call names the function it goes to, so there is no dispatch to devirtualize
   and nothing about what runs is decided by what a value held. Function
   pointers remain as values: they are built, stored and handed to C, which is
   where a callback is registered.
4. Cleanup as a tracked obligation. `linear` resources must be consumed
   exactly once. This replaces `Drop` and finalizers with something the
   compiler checks and the reader can see, and it makes error values
   non-ignorable.
5. Speak C fluently going out. `extern fn` reaches any C library with no glue,
   including the two shapes that are easy to leave out and then be stuck
   without: a function that returns a struct by value, which follows the
   target's real C ABI rather than Frost's own convention, and a callback,
   which is a `$` function parameter plus a context taken by `move`. See
   [c-compatibility.md](../impl/c-compatibility.md) and
   [callbacks.md](callbacks.md).
6. One typed IR, three execution paths, kept honest. The AST lowers to a
   single typed IR from which a Cranelift backend and a portable C backend emit
   and an IR interpreter runs directly. A differential test puts every program
   through all three and asserts they agree. Independent paths that must match
   catch miscompilations a single backend would hide.
7. Predictability over cleverness. You should be able to reason about the
   generated code. Simple, explicit lowerings are preferred to clever ones.
8. The two compilers accept the same language. The Rust bootstrap and the
   self-hosted compiler are held to the same surface, and a divergence is a bug
   in whichever one is behind. The language therefore builds from a Rust
   toolchain and nothing else: the bootstrap compiles the Frost compiler, the
   Frost compiler compiles itself, and no seed binary has to exist or be
   trusted. Every feature lands in both, and what still differs between them is
   listed in [the self-hosted compiler](../impl/self-hosted.md).
9. Compilation stays fast as programs grow. The rate has to hold on a curve
   that does not turn over, at the sizes a real program reaches. The goal has a
   bill attached. Whole-program monomorphization with imports flattened into one
   AST is the shape that contradicts it, so separate compilation is an
   obligation this goal takes on. The bar is the one Jai and Odin set, a full
   build in the 100,000 lines per second range, and a full native build clears
   it. Each module is its own object and `--incremental` rebuilds only the
   modules an edit can reach. `just bench-scaling` and `just bench-incremental`
   measure both, because a rate written into a document is a rate about one
   machine on one day, and it goes stale without anyone noticing it has. See
   [self-hosting.md](../impl/self-hosted.md) and
   [separate-compilation.md](../impl/separate-compilation.md).

## Non-goals

- Not object-oriented. No inheritance, methods, or dynamic dispatch will be
  added to make Frost feel like an OO language. A problem that wants an object
  graph reached by references is a problem to reshape around data and handles.
- Not a garbage-collected language. Automatic, tracing reclamation is out of
  scope. Ownership is explicit through moves, linear resources, and pools.
- Not lifetime-annotated. Frost will not grow lifetime variables or region
  syntax. The second-class-reference rule is the trade that removes the need for
  them.
- Not a stable C-callable ABI. Frost calls C, and there is no attribute that
  exposes a Frost function to a C caller. The emitted C is an internal lowering
  that nothing should link against, and that asymmetry keeps the backend simple.
  The one place C does call Frost is a registered callback, which works because
  the compiler picked the function and its signature rather than a user
  promising one, so the hole is in the direction of travel and the ABI promise
  stands.
- Not maximally general. Frost omits capturing closures (function pointers and
  compile-time function arguments instead), exceptions (errors are failure sets
  and linear enum returns), and implicit conversions that hide cost.
- Not trait-based. There are no traits, no coherence rules, and no bound
  solving. A `where` clause is a precondition over a fixed vocabulary of
  questions the compiler already answers about a type, so nothing implements it
  and there is nothing to resolve. A generic algorithm takes the functions it
  needs as compile-time arguments, so its calls are direct, and what other
  languages call an interface is a struct whose fields are function pointers.
  Do not reach for a generic sort, hash or equality that works over everything.
  Write the one you need over the layout you have, and pass
  `$compare` or `$hash` when it varies. The self-hosted compiler is forty
  thousand lines and wanted a generic function three times.

  Coherence checking, bound solving and method resolution dominate other
  compilers' front ends. Leaving them out keeps this front end near-linear.
  The measurement is in [self-hosting.md](../impl/self-hosted.md).
- Not access-controlled per declaration. There is no `pub`, no private
  marker on any item, and no `pub(crate)` refinement of one. Every struct field
  is public, so encapsulation by field privacy is out of scope. A module does
  have a surface, and it is one line: an `export` at the top of a file lists
  what the file offers, everything else in it is private, and an import is not
  transitive. The reasoning is in
  [declarations.md](../reference/declarations.md).
- Not a research vehicle for a novel type theory. The type system is there to
  make data-oriented code safe and fast.
