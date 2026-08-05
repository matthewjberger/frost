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

**What was taken: a line break ends a statement, and brackets are where an
expression runs on.** Outside brackets a line cannot open with an operator that
joins it to the line above; inside brackets a line break says nothing, which was
already true. The 129 places in the tree that used the old form were wrapped in
the brackets their expressions already needed, mechanically, by the migration
kept in `tests/grammar.rs`.

## The grammar property

> The parse of line N never depends on the token that opens line N+1.

By construction rather than by a check standing beside a rule. Outside brackets
a line is a statement and the expression on it ends at the break, so nothing
after it can change what it means. Inside brackets a line break carries no
meaning at all, so there is nothing for a token to change.

## No mirror hazard

Every single-token edit at a boundary, for every operator that could carry an
expression across one, held by
`no_operator_carries_an_expression_across_a_line_outside_brackets` and its
neighbours:

| edit | outside brackets | inside brackets |
| --- | --- | --- |
| add a leading operator | refused, naming the operator | a parse error |
| drop a leading operator | the shape does not exist | a parse error |
| drop a trailing operator | the shape does not exist | a parse error |
| alter `+` to `-` | a statement whose value nothing reads, refused | a subtraction, which is an ordinary edit |

Nothing in that table silently changes how many statements the surrounding text
parses as. The trailing-operator hazard that ruled out design (a) does not arise
because there are no trailing-operator continuations to drop.

The same wrapped expression is run through the Cranelift backend, the C backend
and the IR interpreter, and all three answer the same.

## Wrapping

Long expressions wrap freely inside any bracket, at any width:

```frost
kept :: fn(a: i64, b: i64, c: i64) -> bool {
    (a > 0
        && b > 0
        && c > 0)
}
```

The brackets are the cost. They are one character at each end of an expression
that was already going to be parenthesised by anyone reading it twice.

## A statement that shares its block's opening line

Sets no column, since it is indented relative to nothing:

```frost
main :: fn() -> i64 { print_int_line(1)
    0 }
```
