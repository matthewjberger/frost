# Summary

[The Frost programming language](index.md)

---

# Learning Frost

- [Getting started](getting-started.md)
- [A tour of Frost](tour.md)
- [Coming from Rust](coming-from-rust.md)
- [Patterns, and what to write instead](patterns.md)
- [Writing an allocator](writing-an-allocator.md)

---

# The language reference

- [Notation and conformance](reference/conformance.md)
- [Lexical structure](reference/lexical.md)
- [Types](reference/types.md)
- [The type system](reference/type-system.md)
- [Declarations and bindings](reference/declarations.md)
- [Expressions](reference/expressions.md)
- [Unchecked operations and `unsafe`](reference/unsafe.md)
- [Statements](reference/statements.md)
- [Ownership and borrowing](reference/ownership.md)
- [Allocation sources and regions](reference/allocation-and-regions.md)
- [Linear resources](reference/linear.md)
- [Handles, pools, and the memory model](reference/handles-and-pools.md)
- [Generics and compile-time specialization](reference/generics.md)
- [The foreign function interface](reference/ffi.md)
- [Grammar](reference/grammar.md)
- [Precedence, keywords, and escapes](reference/precedence.md)

---

# Why it is the way it is

- [Design philosophy, goals, and non-goals](design/philosophy.md)
- [Why the syntax reads this way](design/syntax.md)
- [How Frost guarantees memory safety](design/memory-safety.md)
- [Pools, slabs, and columns](design/pools-and-columns.md)
- [Allocation strategy](design/allocators.md)
- [Callbacks with a typed context](design/callbacks.md)

- [Writing Frost with a model](design/cost-of-error.md)
---

# The standard library

- [What is in std/](std/index.md)
- [Typed allocation](std/mem.md)
- [Containers](std/containers.md)
- [Text, files, and JSON](std/text-and-io.md)
- [Sorting, and orderings as values](std/sort.md)
- [Math](std/math.md)
- [The entity-component system](std/ecs.md)
- [Threads](std/thread.md)
- [Graphics bindings](std/graphics.md)

---

# The implementation

- [Native, freestanding, self-hosted](impl/build-modes.md)
- [Compiler architecture](impl/architecture.md)
- [Finding a module](impl/modules.md)
- [Separate compilation](impl/separate-compilation.md)
- [The self-hosted compiler](impl/self-hosted.md)
- [C compatibility](impl/c-compatibility.md)

- [What the probes found](impl/findings.md)
- [Where a statement ends](impl/line-boundaries.md)
---

# What is left

- [Roadmap](roadmap.md)

---

# Appendices

- [The command line and the environment](appendix/cli.md)
- [How it got here](appendix/history.md)
