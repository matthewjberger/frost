# Where a statement ends

A line break ends a statement. An expression runs past one inside brackets, and
where the line above was left unfinished by an operator written at its end.

## The hazard it removes

The alternative every brace-free language reaches for is continuation by leading
operator, and under it a single token at a line boundary decides the answer
silently:

```frost,sketch
x := 10
    + 20        // x is 30
```

```frost,sketch
x := 10
    20          // x is 10, and nothing was said
```

Under a leading-operator rule the second of those compiles and answers 10.
Nothing in the program says the `+` went missing, and the four single-token
edits around it behave differently from each other:

| edit | under leading-operator continuation | in Frost |
| --- | --- | --- |
| drop the leading `+` | compiles, answers 10 | refused |
| add a leading `+` in front of a statement | parse error | parse error |
| change the `+` to a `-` | a subtraction, silently | refused, with the rule named |
| drop a `+` inside brackets | parse error | parse error |

Only the first row is a silent change of meaning, and only outside brackets:
inside any bracket a line break is insignificant, so the same drop leaves two
expressions side by side, which does not parse.

## Four rules, and the one taken

(a) Continuation by syntactic incompleteness. A line that cannot be a
complete statement continues. Trailing-operator style. Rejected: it reproduces
the hazard with the polarity flipped. Dropping a trailing `+` silently
terminates the statement, and the corpus would have to be rewritten to trailing
operators to get there.

(b) Free wrapping inside brackets, incompleteness outside. The bracket half
is already true. The outside half is (a) and carries (a)'s hazard.

(c) Explicit statement terminators. A semicolon after every statement. Kills
the hazard outright: a dropped operator leaves a statement that does not
terminate. Costs a terminator on every line of a 60,000-line corpus, and the
language deliberately does not have one.

(d) Explicit continuation marker. A trailing `\` or similar. Unambiguous in
both directions, and it is a token whose only job is to be dropped.

The rule Frost takes: a line break ends a statement, and brackets are where an
expression runs on. Outside brackets a line cannot open with an operator that
joins it to the line above, and a line indented past the statement above it is
refused for reading as a continuation while parsing as a statement. Inside
brackets a line break says nothing. An expression too long for a line is wrapped
in the brackets a reader would have put round it anyway, or carries the operator
at the end of the line it continues from, which is where `count -` says a
subtraction is meant.

## The grammar property

> The parse of line N never depends on the token that opens line N+1.

That holds by construction, with no check standing beside a rule. Outside
brackets a line is a statement and the expression on it ends at the break, so
nothing after it can change what it means. Inside brackets a line break carries
no meaning at all, so there is nothing for a token to change.

## Single-token edits at a boundary

The test `no_operator_carries_an_expression_across_a_line_outside_brackets` and
its neighbours run every single-token edit at a boundary, for every operator
that could carry an expression across one:

| edit | outside brackets | inside brackets |
| --- | --- | --- |
| add a leading operator | refused, naming the operator | a parse error |
| drop a leading operator | the shape does not exist | a parse error |
| drop a trailing operator | refused, naming the indentation | a parse error |
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

The brackets are the cost: one character at each end of an expression that was
already going to be parenthesised by anyone reading it twice.

## A statement that shares its block's opening line

Sets no column, since it is indented relative to nothing:

```frost
import "io.frost"
main :: fn() -> i64 { print("{}\n", 1)
    0 }
```
