# Writing Frost with a model

Frost is built to be worked on by a language model as well as by a person. This
page says what that claim rests on and what it does not. Every figure is from
`just bench-loop`; every property named is a test in the suite or a program you
can run.

**One scoping sentence first.** There is almost no Frost in the world, so a model
writing it is working from a specification held in context rather than from a
corpus it has read. The argument here is therefore about error rate per token of
context spent, not about familiarity. A port of a large program changes that:
grep for a prefix returns working call sites, and the story moves from
specification to retrieval.

## The larger good is the cost of a wrong guess, not the reading

Naming properties reduce misreading. They are worth having and Frost has them.
But the failure that costs an agent a session is not misreading an existing
line, it is writing a new one confidently and wrongly. What decides the cost of
that is how fast the guess is falsified, and by what.

Three shapes of confidently wrong code used to compile and produce a working
binary that did the wrong thing. Each is now a compile error in both compilers,
pinned by a test:

| written | what it did | now |
| --- | --- | --- |
| `add(1 2)` — comma dropped | compiled, answered the same as `add(1, 2)` | refused |
| a continuation line whose leading `+` was dropped | compiled, answered 10 where 30 was meant | refused |
| `fn(v: Absent)` — a type name nothing declares | compiled | refused |

None of those was found by reading the compiler. Each was found by writing the
smallest program that would expose it and running it, which is the loop this
page is about.

### The loop, measured

On an AMD Ryzen 7 7800X3D, median of five runs:

| | median |
| --- | --- |
| one file, checked to an object | 12 ms |
| one file, compiled and its tests run | 9 ms |
| the largest module, checked | 487 ms |
| `frost fmt --check` over the whole corpus | 121 ms |
| `frost lint` over the standard library | 110 ms |
| the self-hosting fixpoint, both compilers | 12.5 s |

A question about one file is answered in about a hundredth of a second. That is
the number the claim "cheaper to probe than to reason" rests on, and
`bench/loop.ps1` recomputes it.

Behind those numbers: two compilers that must agree, three backends that must
agree, and a self-hosting fixpoint that must reproduce byte for byte. A wrong
hypothesis has four independent chances to be caught.

## What the diagnostics do

The comparison worth making is against a modern compiler's full rendered output,
not against an error code. rustc's spans, notes and helps are excellent and
Frost does not beat them at explaining. What Frost has that is worth naming is
the channel:

- One invocation reports every independent fault its checks found, not the first
  pass's worth, and reports each fault once.
- `--diagnostics=json` writes one object per report: file, line and column, the
  same place as a byte offset, severity, message, other places, and a structured
  edit where one exists. Both compilers write the same records.
- `frost fix` applies the edits that can be applied unread, read back out of
  that channel.
- A name nothing declares carries the nearest name that does, when one name is
  nearer than every other.

## The name model

There is one namespace, no overloading, no methods, no traits and no
inheritance. A call written as a name is fully determined by the text of the
call site: the name is the callee.

That claim is scoped, and the scope is measured rather than asserted. A call
whose callee is an *expression* — a parameter, a local, a struct field — names a
value, and what that value is, is a dataflow question like any other. All four
forms compile and run today. See [What the probes found](../impl/findings.md).

`frost api <prefix>` gives back the narrowing a `.` provides in a method
language: it prints the exported surface a prefix reaches, with signatures, in
text or JSON.

## Where this sits

The combination is C's name model — one flat namespace, a call site that names
its callee — with C's failure severity removed and C's diagnostics replaced.
Nothing else occupies that spot. It is not a claim to be the most greppable
language; greppability is a property of the name model, and C already had it.
What C did not have is that a wrong guess fails to compile instead of corrupting
memory, and that the compiler hands back an edit.

## Field addition, once

Changing data layout is global and loud on purpose. Adding a field to a struct
is a change every function that names that struct sees, and the errors that
follow name the field.

The blast radius is bounded, and the bound is the strong form: adding a field to
struct `S` does not change the accept or reject status of any function that names
neither `S` nor a type containing `S`. That is tested rather than hoped for.

## What is not claimed

- Not that the diagnostics explain better than rustc's. They are in the house
  voice and they carry machine-applicable edits; that is the whole of it.
- Not that every call is determined by its own text. A call through a value is
  not, and the language keeps function values because the callback tables, the
  render graph and the ECS need them.
- Not that the line-boundary rule is safe by construction. A leading operator
  still continues the line above; the hazard that made it dangerous is refused
  by a check standing next to the rule rather than removed from the grammar.
  See [Where a statement ends](../impl/line-boundaries.md).
- Not that any of this is validated at scale. The corpus is the compiler, the
  standard library and the examples. A large port is what would test it.
