# 2. Lexical structure

## 2.1 Source

Source text is UTF-8. Tokens are formed by maximal munch. At each point the
lexer takes the longest token that matches. Identifiers are ASCII,
`[A-Za-z_][A-Za-z0-9_]*`. There are no Unicode identifiers.

## 2.2 Whitespace and comments

Whitespace (space, tab, carriage return, newline) separates tokens and is
otherwise insignificant. Frost is not whitespace-sensitive and has no automatic
semicolon insertion. Statement terminators (`;`) are always optional.

A line break does decide one thing. A token that opens a line and could begin a
statement of its own does begin one rather than continuing the line above. Three
tokens are in that position, `(`, `[`, and `-`, so

```frost
table := tables[slot.table]
(table.mask & mask) != 0

count = 4
-total
```

is four statements: no call of the first line's value, and no subtraction from
`4`. Every other operator has no prefix form, so a line opening with one can
only be a continuation and is read as one. To spell a subtraction across a line
break, put the `-` at the end of the first line:

```frost
held := count -
    total
```

This is the only rule whitespace has. It exists because a statement beginning
with a parenthesis or a negation is otherwise indistinguishable from a call or a
subtraction written across two lines, and the reading that agrees with the
indentation is the one a reader means.

There are two comment forms:

- Line comment, `//` to end of line.
- Block comment, `/* ... */`. Block comments do not nest, and an unterminated
  block comment is an error.

## 2.3 Identifiers and the wildcard

```
IDENT = ( LETTER | "_" ) ( LETTER | DIGIT | "_" )*
```

The single underscore `_` is a distinct token, the wildcard, and is not a
binding name.

## 2.4 Keywords

Reserved words of the specified language, the keyword table of `src/lexer.rs`
in full:

```
break case continue defer distinct else enum extern false fn for if import
in inline linear match move mut ref return safe struct true
type unsafe uses where while with
```

`true` and `false` are reserved here and are the boolean literals of 2.5.

Reserved primitive type names, each its own token:

```
i8 i16 i32 i64 isize   u8 u16 u32 u64 usize   f32 f64   bool str void
```

Four words carry meaning without being reserved, so each stays usable as an
ordinary identifier. `test` is read only at the start of a top-level test
declaration (5.4) and `export` only on a top-level export line (5.5). `flags` is
read as a declaration only when a scalar type and a brace follow it (3.6b), and
`value` only as a parameter mode, where a name follows it (chapter 12). `Type`
(capitalized), used in `$T: Type` (chapter 11), is likewise an ordinary
identifier recognized in that position, unlike the lowercase keyword `type`.

The type builtins `sizeof`, `typename` and `type_id` are not reserved either:
each is an ordinary name read as the builtin only where it is called with a
type argument (6.8), the way `ptr_to` and the other builtin functions are read
at a call.

## 2.5 Literals

Integer. `INTEGER = DECIMAL | HEX | BINARY`, where
`DECIMAL = DIGIT (DIGIT | "_")*`, `HEX = "0" ("x" | "X") (HEXDIGIT | "_")+` and
`BINARY = "0" ("b" | "B") (BINDIGIT | "_")+`. There is no octal prefix. An
underscore may sit between digits and is dropped before the number is read, so
`0xFF_FF` and `1_000_000` are `65535` and `1000000`. Integer literals are
non-negative; a negative value is the prefix `-` applied to one. An integer
literal takes its type from context, defaulting to `i64`.

A hex or binary literal is read as unsigned and reinterpreted, so the whole of
a sixty-four bit mask can be written: `0xFFFFFFFFFFFFFFFF` is the all-ones
sentinel a C header spells that way, and it is past what an `i64` holds as a
positive number. A literal that does not fit the type it is written at is
refused rather than truncated (3.2).

Float. `FLOAT = DIGIT (DIGIT | "_")* ("." DIGIT (DIGIT | "_")*)? EXPONENT?`,
where `EXPONENT = ("e" | "E") ("+" | "-")? DIGIT+`, with an optional `f` or
`f32` suffix that makes it an `f32`, otherwise it is `f64`. Either a fraction
or an exponent makes the number a float, so `1e3` is one, and `1` is not. A `.`
is only taken as a decimal point when the following character is not another
`.`, so `0..10` lexes as a range. An `e` is only taken as an exponent when
digits, or a sign and then digits, follow it. There is no leading-dot form.

String. Delimited by `"`, with escapes `\n`, `\t`, `\r`, `\0`, `\\`, `\"`,
`\'`. Any other escape is an error. There are no numeric or Unicode escapes. A
string literal has type `str` (3.7) and denotes a view of its bytes. Where `^i8`
is expected it instead denotes a pointer to the same bytes with a trailing NUL,
which is how string literals interoperate with C.

Boolean. `true`, `false`, of type `bool`.

## 2.6 Operators and punctuation

```
::  :=  :   =   ->  ..  ..=  .   ^   $   ?   #
+   -   *   /   %   &   |   &&  ||  <<  >>
==  !=  <   <=  >   >=  !
(   )   {   }   [   ]   ,   ;
```

`>>` is a single shift token that the
parser splits when it closes nested generic arguments (11.4).
