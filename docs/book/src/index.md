# The Frost programming language

A data-oriented systems language that is memory-safe with no garbage collector
and no lifetimes, and compiles itself.

A Frost program is plain data and free functions that transform it. There are no
classes and nothing is allocated behind your back. It compiles to native code
through Cranelift or to portable C, and the compiler is written in Frost.

## Borrows are parameter modes

A borrow is what a parameter mode means, so there is no reference type to write
and no lifetime to describe.

```frost,sketch
wound :: fn(mut e: Entity, amount: i64) { e.hp = e.hp - amount }
```

`mut` borrows for the call and mutates in place. An unmarked parameter borrows
to read. `move` takes ownership. There is no `&` anywhere in the language, so a
borrow has nothing to be stored in and nowhere to escape to. That one rule is
why Frost needs no lifetime annotations, and the rest of the design grows from
it.

| in place of | Frost has |
| --- | --- |
| lifetimes on references | borrows that are parameter modes and cannot escape |
| a garbage collector | arenas and pools you can see |
| a long-lived pointer into a collection | a generational handle, a copy value that goes stale rather than dangling |
| destructors | linear resources, consumed exactly once, checked at compile time |
| exceptions and `Result` plumbing | failure sets, `-> T ! E` and `?` |
| dynamic dispatch | monomorphized generics, so the inner-loop call is direct |
| classes and methods | plain structs and free functions |

## A short tour

What Frost has instead of the usual machinery: borrows that are parameter modes,
a resource the compiler counts, a failure that travels in the signature, a `for`
with no iterator to implement, a function that answers with two values and no
tuple type behind them, and literals that leave out a type the context already
carries while every field keeps its name. [A tour of Frost](tour.md) walks all
of it, and the program behind it runs:

```bash
frost examples/tour.frost          # compile, link, and run
```

That is `examples/tour.frost`. A test compiles it and checks what it prints.

## What it does

The language has structs and tagged enums, `match` with payload and tuple
patterns that has to cover every variant or say what the rest do, and generics
that monomorphize over types, values, and functions, so a call in an inner loop
stays direct rather than going through a pointer. A `for` walks a range or a
sequence as the index-and-bound loop it stands for, a function answers with
several values through a return type list rather than a tuple type, and a
literal leaves out a type the context already carries, while every field keeps
its name because the name is what says where the value lands. A resource that
must be released is marked `linear` and the compiler counts it, consumed once on
every path out or the program does not build. Long-lived data lives in pools
addressed by generational handles, a region check keeps a pointer from
outliving the block or the stack frame it points into, and a value constant can
be a folded integer expression. There are no visibility modifiers and no
methods.

It calls C without a binding layer. An `extern fn` links against a C library
with the natural ABI, including one that returns a struct by value and one that
takes a Frost function as a callback with a typed context.

One typed intermediate representation feeds three backends, a Cranelift native
path, a portable C path, and a small interpreter. A differential test runs every
program through all three and checks the answers match, so a lowering bug shows
up as a disagreement rather than as a wrong binary.

The compiler is written in Frost. `selfhosted/frost.frost` reproduces itself
byte for byte through its own C backend and its own x86-64 assembly backend, so
a build can go from source to a running compiler with no C compiler in the loop.
A full native build clears the 100,000 lines per second the speed promise asks
for, with code generation spread across cores, and `--incremental` rebuilds only
the modules an edit can reach. `just bench-scaling` is what says so on the
machine in front of you.

The standard library is ordinary Frost. It has length-carrying strings, a
growable `Vec` and a hash map, file and formatted output, a sort, the slab and
structure-of-arrays `columns` containers, an archetype entity-component system,
and vector, matrix, and quaternion math at both single and double precision.

## Status

Everything above works today and is checked by the test suite on every commit,
including both self-hosting fixpoints and the three-backend differential run.
The compiler has compiled itself.

The two compilers accept the same language. They are held to it by running the
same programs through both and comparing what each accepts, and every form the
language has is a test that both of them compile.

## How to read this book

[A tour of Frost](tour.md) is the language by example, one feature at a time,
and is the place to start. [Coming from Rust](coming-from-rust.md) is the same
ground for someone who already thinks in ownership and borrows.
[Patterns](patterns.md) is what to write instead, once the syntax is familiar.

The [language reference](reference/conformance.md) is normative. Everything
under [why it is the way it is](design/philosophy.md) is the reasoning behind
it, and everything under [the implementation](impl/build-modes.md) is how the
compiler does it.
