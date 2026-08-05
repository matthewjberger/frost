# Where a statement ends

A line break ends a statement. An expression that runs past one continues when
the next line opens with an operator, and a leading `-` opens a new statement
rather than continuing with a subtraction. This is what the rule costs, what was
measured, and what was done about it.

## The hazard, measured

A single token at a line boundary decided the answer, silently:

```frost
x := 10
    + 20        // x is 30
```

```frost
x := 10
    20          // x is 10, and nothing was said
```

Both compilers accepted the second and answered 10. Three neighbouring shapes
were probed at the same time:

| edit | before | now |
| --- | --- | --- |
| drop the leading `+` | compiled, answered 10 | refused |
| add a leading `+` in front of a statement | parse error | parse error |
| change the `+` to a `-` | refused, with the rule named | unchanged |
| drop a `+` inside brackets | parse error | unchanged |

Only the first was open, and only outside brackets: inside any bracket a line
break is insignificant already, and the same drop leaves two expressions side by
side, which does not parse.

## Four designs, and the one taken

**(a) Continuation by syntactic incompleteness.** A line that cannot be a
complete statement continues. Trailing-operator style. Rejected: it reproduces
the hazard with the polarity flipped. Dropping a trailing `+` silently
terminates the statement, which is the same failure read in a mirror, and the
corpus would have to be rewritten to trailing operators to get there.

**(b) Free wrapping inside brackets, incompleteness outside.** The bracket half
is already true. The outside half is (a) and carries (a)'s hazard.

**(c) Explicit statement terminators.** A semicolon after every statement. Kills
the hazard outright: a dropped operator leaves a statement that does not
terminate. Costs a terminator on every line of a 60,000-line corpus, and the
language deliberately does not have one.

**(d) Explicit continuation marker.** A trailing `\` or similar. Unambiguous in
both directions, and it is a token whose only job is to be dropped.

**What was taken: keep the rule, and make the hazard fail to parse.** Every
statement of a block begins at the same column. A statement indented past its
neighbours reads as a continuation of the line above, so a continuation that
parses as a statement of its own is refused:

> this line is indented past the statement above it, so it reads as continuing
> that line, and it begins a statement of its own. An expression broken over
> lines carries the operator that joins them onto the second, or is written
> inside brackets

The whole corpus passes unchanged, in both compilers, because the formatter
already writes every statement of a block at one column.

## What this is and is not

It closes the measured hazard: the dropped-operator edit no longer changes the
statement count silently, it fails to parse. It does not establish the grammar
property the strongest version of this asks for — *the parse of line N never
depends on the leading token of line N+1* — because a leading `+` still
continues the line above. The check is a guard standing next to the rule, not a
replacement for it.

Reaching the grammar property means design (c) or (d) and a corpus migration.
The measured cost of not having it is now one refused program rather than one
wrong answer, which is why the guard was worth landing on its own.

A statement that shares the line its block opens on sets no column, since it is
indented relative to nothing:

```frost
main :: fn() -> i64 { print_int_line(1)
    0 }
```
