# 1. Notation and conformance

## 1.1 Grammar notation

Grammar rules use EBNF:

- `x y` is `x` followed by `y`. `x | y` is `x` or `y`.
- `x?` optional, `x*` zero or more, `x+` one or more, `( ... )` groups.
- Terminals are literal spellings in `code font` (`::`, `fn`, `->`) or token
  classes in UPPERCASE (`IDENT`, `INTEGER`, `STRING`).
- Nonterminals are `PascalCase`.

## 1.2 Parsing discipline

The language is parsed by recursive descent with a Pratt (precedence-climbing)
expression parser and bounded lookahead. Statement and type selection is
decided by the first one to three tokens. The specific lookahead each decision
uses is stated in the grammar. Expression parsing is driven by the operator
precedence table in 14.1. The parser does not backtrack past a committed
production, with one bounded exception. It may scan ahead a fixed number of
tokens to decide whether a parenthesized group is a function parameter list
(13.6).

This discipline is the contract. The reference parser (`src/parser.rs`), the
self-hosted parser (`selfhosted/`), and this grammar are three views of one
language. A disagreement between them is a bug in whichever diverges from the
intent expressed here.

`tests/grammar.rs` is a corpus test, and it is worth knowing what it does and
does not catch. It runs a fixed list of sources through the reference parser and
asserts that each one parses, and a shorter list that each one does not. It
never reads chapter 13. A form the grammar describes and the parser rejects is
caught only if someone put that form in the corpus, and a form the parser
accepts and the grammar never mentions is not caught at all. The corpus stops a
parser change from silently breaking what it covers. Keeping this chapter honest
is a reading job.

## 1.3 Conformance

A conforming program is one this grammar accepts and whose static semantics
(chapters 4, 8, 9, 10, 11) hold. A conforming implementation rejects
non-conforming programs with a diagnostic and compiles conforming programs to
code with the behavior described here. Constructs marked *unchecked* (raw
pointer operations, `extern` calls) place their correctness obligation on the
programmer.
