# The Frost programming language

Frost is a systems language for programs that care where their data lives. It is
memory-safe without a garbage collector and without lifetime annotations, it
compiles to native code, and its compiler is written in Frost.

A program is structs and free functions that transform them. Every allocation is
one you asked for. This book teaches the language, then documents it: start with
[the tour](tour.md), and use [the reference](reference/conformance.md) when you
need the exact rule.

## Borrows are parameter modes

A parameter mode is the whole of what a borrow is.

```frost,sketch
wound :: fn(mut e: Entity, amount: i64) { e.hp = e.hp - amount }
```

`mut` borrows the caller's value for the call and mutates it in place. An
unmarked parameter borrows it to read. `move` takes ownership. A borrow lasts
for the call, and you cannot store one, so there is nothing to annotate.

Anything that has to outlive a call needs another way to be named, and that is
what pools and generational handles are for. One borrow does have a spelling:
`ref T` can be returned, and the compiler checks it against the frame and the
region it came from, so an accessor can hand back a place instead of a copy.

Here is what Frost uses in place of the machinery you may be expecting:

| in place of | Frost has |
| --- | --- |
| lifetimes on references | borrows that are parameter modes and cannot escape |
| a garbage collector | arenas and pools you can see |
| a long-lived pointer into a collection | a generational handle, a copy value that goes stale instead of dangling |
| destructors | linear resources, consumed exactly once, checked at compile time |
| exceptions and `Result` plumbing | failure sets, `-> T ! E` and `?` |
| dynamic dispatch | monomorphized generics, so the inner-loop call is direct |
| classes and methods | plain structs and free functions |

## The language in one paragraph

Frost has structs and tagged enums, `match` with payload and tuple patterns that
must cover every variant or say what the rest do, and generics that monomorphize
over types, values, and functions. A `for` walks a range or a sequence, and it
compiles to the index-and-bound loop it stands for. A function can answer with
several values through a return type list, and every value in that list has a
name. A literal can leave out a type the context already carries, while every
field keeps its name, because the name says where the value lands. Mark a
resource `linear` and the compiler counts it: consumed once on every path out,
or the program does not build. Long-lived data lives in pools addressed by
generational handles. A region check keeps a pointer from outliving the block or
the stack frame it points into. There are no visibility modifiers and no
methods.

Frost calls C directly. An `extern fn` links against a C library with the
natural ABI, including functions that return a struct by value and functions
that take a Frost function as a callback with a typed context.

## What is in the box

One typed intermediate representation feeds three backends: a Cranelift native
path, a portable C path, and a small interpreter. A differential test runs every
program through all three and checks that the answers match.

The compiler is written in Frost. `selfhosted/frost.frost` reproduces itself
byte for byte through its own C backend and its own x86-64 assembly backend, so
a build can go from source to a running compiler with no C compiler in the loop.
A full native build clears 100,000 lines per second, code generation is spread
across cores, and `--incremental` rebuilds only the modules an edit can reach.
Run `just bench-scaling` to see the numbers on your own machine.

The standard library is ordinary Frost: length-carrying strings, a growable
`Vec` and a hash map, file and formatted output, a sort, the slab and
structure-of-arrays `columns` containers, an archetype entity-component system,
and vector, matrix, and quaternion math at single and double precision.

## Trying it

```bash
frost examples/tour.frost          # compile, link, and run
```

`examples/tour.frost` is the program behind [the tour](tour.md). A test compiles
it and checks what it prints.

## How to read this book

[A tour of Frost](tour.md) is the language by example, one feature at a time,
and is the place to start. [Coming from Rust](coming-from-rust.md) covers the
same ground for someone who already thinks in ownership and borrows.
[Patterns](patterns.md) shows what to write once the syntax is familiar.

The [language reference](reference/conformance.md) is normative. The chapters
under [why it is the way it is](design/philosophy.md) give the reasoning behind
the rules, and the chapters under [the implementation](impl/build-modes.md)
describe how the compiler works.
