# Design Philosophy, Goals, and Non-Goals

Frost is a **data-oriented** systems language. It is built on the belief that a
program is, first and last, a description of *how data is laid out and
transformed* — not a society of objects exchanging messages. Every major
decision in the language follows from taking that seriously.

## Data-oriented design, not object-oriented programming

Object-oriented programming organizes code around **objects**: bundles of data
and the methods that act on them, related by inheritance, reached through
references, and freed by destructors or a garbage collector. It optimizes for a
particular kind of human intuition — "a `Player` *is a* `Character` and *has an*
`Inventory`" — and pays for it in indirection: virtual dispatch, pointer chasing,
scattered allocations, hidden lifetimes, and cache-hostile memory layouts.

Data-oriented design starts from the opposite question: **what does the data
look like, and how does it flow?** The layout of memory is the primary design
artifact; the code is written to transform that layout efficiently. The
practical consequences run through all of Frost:

| Concern            | Object-oriented default            | Frost (data-oriented)                                  |
| ------------------ | ---------------------------------- | ------------------------------------------------------ |
| Grouping           | Objects bundle data + behavior     | **Plain structs** hold data; **free functions** transform it |
| Reuse / hierarchy  | Inheritance, virtual methods       | **Composition** and **`match` on enums**; no inheritance |
| Polymorphism       | Dynamic dispatch (vtables)         | **Monomorphized generics** + **function pointers** (static) |
| Identity / linking | References everywhere              | **Generational handles** into pools; references are second-class |
| Lifetime           | GC or destructors                  | **Linear resources** consumed exactly once; no hidden `Drop` |
| Memory             | Per-object heap allocation         | **Pools / explicit allocation**; contiguous, predictable layout |

The through-line: **behavior is not attached to data.** A struct is just its
fields. To do something with it you call a free function and pass it in. There
are no methods, no `self`, no inheritance, no vtables. This is not a limitation
Frost tolerates — it is the point. Separating data from the code that walks it is
what makes the layout visible, the control flow explicit, and the machine's
actual work predictable.

### Concretely, what OOP features are *deliberately absent*

- **No classes, methods, or `self`.** Structs are data; functions are functions.
- **No inheritance or interfaces.** Reuse comes from composition and generics.
- **No virtual dispatch.** Higher-order code uses function pointers (chosen
  explicitly), and polymorphism is resolved at compile time by monomorphization.
- **No garbage collector and no destructors.** Cleanup is a *linear* obligation
  the type system tracks; long-lived data is addressed by *handles*, not by
  references the runtime must keep alive.
- **No implicit anything.** No hidden allocations, no hidden copies of large
  values beyond an explicit move, no hidden control flow behind an operator.

## Goals

1. **Make the data layout the design.** The programmer should always be able to
   see how a value is represented and where it lives. Aggregates have a defined,
   inspectable layout; pools are contiguous; nothing is boxed behind your back.
2. **Memory safety without a garbage collector and without lifetime
   annotations.** Safety comes from making the dangerous shapes unrepresentable
   (second-class references) and from a few local rules (moves, exclusivity,
   linearity, generational handles). See [memory-safety.md](memory-safety.md).
3. **Zero-cost, static polymorphism.** Generics monomorphize; higher-order code
   is function pointers. You pay for abstraction at compile time, not at run
   time.
4. **Cleanup as a tracked obligation, not magic.** `linear` resources must be
   consumed exactly once. This replaces `Drop`/finalizers with something the
   compiler checks and the reader can see, and it makes error values
   non-ignorable.
5. **Speak C fluently going out.** `extern fn` reaches the entire C ecosystem
   with no glue. See [c-compatibility.md](c-compatibility.md).
6. **One typed IR, two backends, kept honest.** The AST lowers to a single typed
   IR from which both a Cranelift backend and a portable C backend emit; a
   differential test compiles every program through both and asserts they agree.
   Two independent backends that must match catch miscompilations a single
   backend would hide.
7. **Predictability over cleverness.** The generated code should be something you
   can reason about. Simple, explicit lowerings are preferred to clever ones.

## Non-goals

- **Not object-oriented.** No inheritance, methods, or dynamic dispatch will be
  added to make Frost feel like an OO language. If a problem wants an object
  graph reached by references, that is a sign to reshape it around data and
  handles.
- **Not a garbage-collected language.** Automatic, tracing reclamation is out of
  scope. Ownership is explicit (moves, linear resources, pools).
- **Not lifetime-annotated.** Frost will not grow lifetime variables or region
  syntax. The second-class-reference rule is the deliberate trade that removes
  the need for them.
- **Not a stable C-callable ABI (C → Frost).** Frost calls C; C does not call
  Frost. The emitted C is an internal lowering, free to be ugly, and is not an
  interface anyone should link against. (This asymmetry is what keeps the
  backend simple.)
- **Not maximally general.** Frost intentionally omits capturing closures
  (function pointers instead), exceptions (errors are linear enum returns), and
  implicit conversions that hide cost.
- **Not a research vehicle for a novel type theory.** The type system is a means
  to make data-oriented code safe and fast, not an end in itself.

## Why this project exists: de-risking a from-scratch language

Frost is being completed **in place** as a de-risking vehicle. The goal is to
prove out the *hard* parts of this design on a real, running implementation —
ownership without lifetimes, linear resources replacing `Drop`, generational
handles unified with a borrow discipline, a typed IR feeding multiple backends,
specialization-only generics — before building a new language "in a similar
way." The bar for "done" is not features for their own sake; it is **confidence
that the bones of the design hold up** when they meet real code and a real code
generator.

That is why the differential oracle matters so much here: the point is not just
that Frost runs, but that its two independent backends *agree*, which is strong
evidence that the semantics are pinned down rather than accidental. Every
proposition in the design has been driven to running, differential-tested code.
The rest is engineering.

## The one-sentence version

**Frost organizes programs around data and the functions that transform it —
plain structs, free functions, enums with `match`, monomorphized generics,
generational handles, and linearly-tracked resources — and is memory-safe
without a garbage collector or lifetime annotations because references are
second-class and therefore cannot escape.**
