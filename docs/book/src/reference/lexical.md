# 2. Lexical structure

## 2.1 Source

Source text is UTF-8. Tokens are formed by maximal munch. At each point the
lexer takes the longest token that matches. Identifiers are ASCII,
`[A-Za-z_][A-Za-z0-9_]*`. There are no Unicode identifiers.

## 2.2 Whitespace and comments

Whitespace (space, tab, carriage return, newline) separates tokens and is
otherwise insignificant. There is no automatic semicolon insertion. Statement
terminators (`;`) are always optional.

A line break decides one thing. A token that opens a line and could begin a
statement of its own begins one, and the line above ends at the break. Three
tokens are in that position, `(`, `[`, and `-`, so

```frost,sketch
table := tables[slot.table]
(table.mask & mask) != 0

count = 4
-total
```

is four statements: a binding and a comparison, then an assignment and a
negation. Every other operator has no prefix form, so a line opening with one
continues the line above. To spell a subtraction across a line break, put the
`-` at the end of the first line:

```frost,sketch
held := count -
    total
```

This is the only rule whitespace has.

There are two comment forms:

- Line comment, `//` to end of line.
- Block comment, `/* ... */`. Block comments do not nest, and an unterminated
  block comment is an error.

## 2.3 Identifiers and the wildcard

```
IDENT = ( LETTER | "_" ) ( LETTER | DIGIT | "_" )*
```

The single underscore `_` is a distinct token, the wildcard, and never names a
binding. It stands in a `match` arm that covers the rest (6.7), in a binding
list for a value the caller has no use for (5.2a), and on its own before a `:=`
for an answer meant to go unread (chapter 7). Everywhere else it has nowhere to
parse: `_` is refused where an expression is expected. It is also refused as
one alternative of a `case` naming several (6.7).

`..`, `..=` and `|` carry their usual meanings inside a `case` pattern: the
first two open a span of whole numbers and the third joins alternatives (6.7).
The lexer reads them there as it reads them anywhere.

## 2.4 Keywords

Reserved words of the specified language, the keyword table of `src/lexer.rs`
in full:

```
break case continue defer distinct else enum errdefer extern fn for if
import in inline linear match move mut ref return safe struct
type unsafe uses var where while with
```

The primitive type names are predeclared identifiers:

```
i8 i16 i32 i64 isize   u8 u16 u32 u64 usize   f32 f64   bool str
```

Each means its type wherever a type is read, ahead of any declaration going by
the same name, so the meaning cannot be redeclared, and each stays usable as
an ordinary binding or field name. `void` is a type only inside the compilers;
no surface program writes it.

`true` and `false`, the boolean literals of 2.5, are predeclared the same way:
identifiers to the lexer, always the booleans in expression position. A
binding, parameter or constant declared by either name is refused where the
declaration is read.

Several more words carry meaning without being reserved, so each stays usable
as an ordinary identifier. `test` is read only at the start of a top-level test
declaration (5.4) and `export` only on a top-level export line (5.5). `flags`
is read as a declaration only when a scalar type and a brace follow it (3.6b),
and `value` only as a parameter mode, where a name follows it (chapter 12).
`packed` marks a declaration only where `struct` follows it, and `align` marks
a field's alignment only where `(` follows it (3.2a). `format` marks a
parameter that takes a string literal, and only where a name follows it
(11.1c.0).
`Type` (capitalized), used in `$T: Type` (chapter 11), is likewise an ordinary
identifier recognized in that position. The lowercase `type` is a keyword.
The type builtins `sizeof`, `typename` and `type_id` are ordinary names read
as the builtin only where one is called with a type argument (6.8), the way
`ptr_to` and the other builtin functions are read at a call.

## 2.5 Literals

Integer. `INTEGER = DECIMAL | HEX | BINARY`, where
`DECIMAL = DIGIT (DIGIT | "_")*`, `HEX = "0" ("x" | "X") (HEXDIGIT | "_")+` and
`BINARY = "0" ("b" | "B") (BINDIGIT | "_")+`. There is no octal prefix. An
underscore may sit between digits and is dropped before the number is read, so
`0xFF_FF` and `1_000_000` are `65535` and `1000000`. Integer literals are
non-negative; a negative value is the prefix `-` applied to one. An integer
literal takes its type from context, defaulting to `i64`.

A hex or binary literal is read as unsigned and reinterpreted, so a full
sixty-four bit mask can be written. `0xFFFFFFFFFFFFFFFF` is the all-ones value,
past what an `i64` holds as a positive number. A literal that does not fit the
type it is written at is refused (3.2).

Float. `FLOAT = DIGIT (DIGIT | "_")* ("." DIGIT (DIGIT | "_")*)? EXPONENT?`,
where `EXPONENT = ("e" | "E") ("+" | "-")? DIGIT+`, with an optional `f` or
`f32` suffix that makes it an `f32`, otherwise it is `f64`. Either a fraction
or an exponent makes the number a float, so `1e3` is a float and `1` is an
integer. A `.` is only taken as a decimal point when the next character is not
`.`, so `0..10` lexes as a range. An `e` is only taken as an exponent when
digits, or a sign and then digits, follow it. There is no leading-dot form.

String. Delimited by `"`, with escapes `\n`, `\t`, `\r`, `\0`, `\\`, `\"`,
`\'`. Any other escape is an error. There are no numeric or Unicode escapes. A
string literal has type `str` (3.7) and denotes a view of its bytes. Where `^i8`
is expected it denotes a pointer to the same bytes with a trailing NUL, the form
an `extern` function reads.

Boolean. `true`, `false`, of type `bool`.

## 2.6 Operators and punctuation

```
::  :=  :   =   ->  ..  ..=  ...  .   ^   $   ?
+   -   *   /   %   &   |   &&  ||  <<  >>
==  !=  <   <=  >   >=  !
(   )   {   }   [   ]   ,   ;
```

`>>` is a single shift token that the parser splits when it closes nested
generic arguments (11.4).
