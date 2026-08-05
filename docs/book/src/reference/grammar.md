# 13. Grammar

This chapter is the syntax of the specified language. Where a production here
and `src/parser.rs` disagree, the parser is the one that runs, and either the
production or the parser is wrong.

`IDENT`, `INTEGER`, `FLOAT` and `STRING` are terminals and are spelled out in
2.5. `INTEGER` covers decimal, `0x` hex and `0b` binary, each of which may
carry `_` between digits; `FLOAT` covers a fraction, an `e` exponent, or both.

## 13.1 Program and statements

The top level holds declarations. A binding, a loop, and an expression
statement live inside blocks, and either compiler refuses one written at file
scope by naming what may stand there.

```
Program   = TopLevel*

TopLevel =
      "import" STRING ImportRenames? ";"?
    | ExportLine
    | TestBlock
    | IDENT "::" ConstBody ";"?

Statement =
      TopLevel
    | "return" ( Expr ( "," Expr )* )? ";"?
    | "defer" Statement
    | "for" IDENT ( "," IDENT )? "in" ( Expr | LiveWalk ) Block
    | "while" "(" Expr ")" Block
    | "with" IDENT Block                      // a region, 8a
    | "break" ";"?
    | "continue" ";"?
    | "var" IDENT ( ":=" Expr | ":" Type "=" Expr ) ";"?
    | "ref" IDENT ":=" Expr ";"?             // bind a borrow of a place (5.1)
    | MultiNames ":=" Expr ";"?              // several values from one call
    | IDENT ":=" Expr ";"?
    | IDENT ":" Type "=" Expr ";"?           // lookahead: ":" not followed by ":"
    | Expr ( "=" Expr )? ";"?                 // expression statement or assignment
```

```
MultiNames = MultiName ( "," MultiName )+
MultiName  = "var"? IDENT | "_"

ImportRenames = "(" IDENT "as" IDENT ( "," IDENT "as" IDENT )* ","? ")"
ExportLine    = "export" IDENT ( "," IDENT )*
TestBlock     = "test" STRING Block
```

`export`, `test` and the `as` of a rename are ordinary identifiers the parser
reads by what follows them (2.4), so each stays usable as a name elsewhere.
`export` is read only when an identifier follows it, `test` only when a string
literal and a `{` follow it, and a rename list only when its `(` opens on the
same line as the import path.

A name followed by a comma at statement position is a list binding and nothing
else, which is what tells the two `:=` forms apart.

The `var` / `:=` / `: =` / `::` forms are selected by the token after the
identifier. These are `:=` (inferred binding), `:` then a non-`:` (typed
binding), and `::` (constant). The last alternative covers expression statements
and assignments to a place. `mut` never opens a statement: it is the parameter
mode, and a local that is reassigned is declared with `var`.

## 13.2 Constants and items

```
ConstBody =
      "linear"? "struct" GenericParams? "{" StructFields? "}"
    | "linear"? "enum" GenericParams? "{" EnumVariants? "}"
    | "distinct" Type
    | "flags" IntegerType "{" FlagBits? "}"
                                              // IntegerType is one of the
                                              // i8..usize names of 13.7
    | "extern" "fn" "(" Params? ")" ( "->" Type )?
    | Expr                                    // function literal, or a value

GenericParams = "(" TypeParam ( "," TypeParam )* ")"
TypeParam     = "$" IDENT ":" ( "Type" | "type" )

StructFields  = StructField ( "," StructField )* ","?
StructField   = IDENT ":" Type

EnumVariants  = EnumVariant ( "," EnumVariant )* ","?
EnumVariant   = IDENT ( "{" ( IDENT ":" Type ( "," IDENT ":" Type )* )? "}" )?

FlagBits      = FlagBit ( "," FlagBit )* ","?
FlagBit       = IDENT "=" INTEGER

Params        = Param ( "," Param )*
Param         = ParamMode? "$" IDENT ":" ( "Type" | "type" | "usize" | ProcType )
              | IDENT ":" "$" "..."          // compile-time list, last (11.1c)
              | ParamMode? IDENT ( ":" Type )?
ParamMode     = "mut" | "move" | "value"
ProcType      = "fn" "(" ( ProcParam ( "," ProcParam )* )? ")" ( "->" Type )?
ProcParam     = ParamMode? Type
```

A `Name :: fn(...) { ... }` item is the `Expr` alternative of `ConstBody`, whose
expression is a function literal (13.6).

`value` (chapter 12) is a word rather than a keyword, so a parameter may still
be named `value`. What tells the two apart is that a mode is followed by the
parameter's name and a name is followed by its type.

`args: $...` takes no type, since its length and its element types arrive with
each call, and it is last, because anything after it would have nothing to say
which side of the list it belonged to.

## 13.3 Blocks

```
Block = "{" Statement* "}"
```

The trailing expression of a block, if any, is its value.

## 13.4 Expressions

Expressions are parsed by precedence climbing. `Expr` denotes an expression at
the lowest precedence. The operator table in 14.1 governs grouping.

```
Expr    = Prefix ( InfixOp Expr )*           // resolved by precedence (14.1)

Prefix =
      Primary
    | "-" Expr
    | "!" Expr
    | "$" Type                                // type value (11.3)
    | TypeBuiltin "(" "$"? Type ")"           // sizeof / typename / type_id (6.8)

TypeBuiltin = "sizeof" | "typename" | "type_id"

Primary =
      INTEGER | FLOAT | STRING | "true" | "false"
    | IDENT
    | "(" Grouped                             // group, tuple, or function literal
    | "[" ( Expr ( "," Expr )* )? "]"         // array literal
    | "[" Expr ";" ( INTEGER | IDENT ) "]"    // repeat array literal
    | IDENT "<" TypeArgs ">" "{" FieldInits? "}"  // generic literal (11.1)
    | IfExpr
    | MatchExpr
    | "." IDENT ( "{" FieldInits? "}" )?      // inferred variant (6.5)
    | "{" FieldInits? "}"                     // inferred struct literal (6.5)
    | "fn" "(" Params? ")" ReturnSig? Block
    | "unsafe" Block

FieldInits = IDENT "=" Expr ( "," IDENT "=" Expr )* ","?
```

Postfix and infix forms, applied by the precedence loop:

```
InfixOp = "||" | "&&" | "==" | "!=" | "<" | "<=" | ">" | ">="
        | "|"  | "&"  | "<<" | ">>" | "+" | "-" | "*" | "/" | "%"

Postfix =
      Expr ".." Expr                          // range (half-open)
    | Expr "..=" Expr                         // range (inclusive)
    | Expr "[" Expr "]"                       // index
    | Expr "(" Arguments? ")"                 // call
    | Expr "." IDENT                          // field access
    | Expr "^"                                // dereference (assignable place)
    | Expr "?"                                // hand a failure up (5.2b)
    | IDENT "{" StructInit? "}"              // struct literal (bare identifier)
    | IDENT "::" IDENT ( "{" StructInit? "}" )?   // enum variant (bare identifier)

Arguments  = Argument ( "," Argument )* ","?
Argument   = Expr ( "for" IDENT "in" IDENT )?  // one argument per list element
StructInit = IDENT "=" Expr ( "," IDENT "=" Expr )* ","?
```

`Expr "^"` and the struct/enum-init forms are only entered when the left operand
is the appropriate shape (a place for `^`, a bare identifier for `{`/`::`). The
struct-literal `{` is disambiguated from a `match` body by checking that the
token after `{` is not `case`.

`SizeExpr = SizeTerm ( ("+" | "-") SizeTerm )*`,
`SizeTerm = SizeAtom ( ("*" | "/" | "%") SizeAtom )*`,
`SizeAtom = INTEGER | IDENT | "(" SizeExpr ")"`. A length is arithmetic and
nothing else (3.4): the `[Type ";" INTEGER]` form is entered instead when the
token after the `[` is followed by a `;`.

`LiveWalk = "live" "(" Place ")"`, where `Place` is a name or a field of one
(10.1b). It is written only after the `in` of a `for`: it is the subject of a
walk, so there is no value for it to be anywhere else.

The `for` form of `Argument` is `g(T) for T in list` (11.1c): the expression is
written once and the call takes one argument per element of the compile-time
list, with the named variable standing for that element. An argument list is the
only place it may be written, because what it produces is an argument count.

## 13.5 `if` and `match`

```
IfExpr    = "if" "(" Expr ")" Block ( "else" Block )?

MatchExpr = "match" Expr "{" MatchArm* "}"
MatchArm  = "case" Pattern ":" ( Block | Expr )

Pattern =
      "_"
    | INTEGER | FLOAT | STRING | "true" | "false"
    | "." IDENT ( "{" IDENT ( "," IDENT )* "}" )?
    | IDENT "::" IDENT ( "{" IDENT ( "," IDENT )* "}" )?
    | "(" Pattern ( "," Pattern )* ")"
    | IDENT
```

## 13.6 Parenthesized groups and function literals

A `(` begins one of three things, chosen by a bounded look-ahead scan
(`looks_like_function_params`, at most a fixed number of tokens, depth-tracked):

```
Grouped =
      ")" ReturnSig? Block                    // zero-parameter function literal
    | ")"                                     // empty tuple  ()
    | Params ")" ( ReturnSig? Block )?        // function literal (if a body follows)
    | Expr ( "," Expr )* ")"                  // tuple, or a parenthesized expression

ReturnSig   = ReturnType? UsesClause* WhereClause?
ReturnType  = "->" ( Type ( "!" Type )? | ReturnList )
UsesClause  = "uses" Type ( "," Type )*
WhereClause = "where" Expr
ReturnList  = "(" ReturnValue "," ReturnValue ( "," ReturnValue )* ")"
ReturnValue = IDENT ":" Type
```

Every part of a `ReturnSig` is optional and they are read in that order, so
`fn() uses Arena<256> { }` and `fn($T: Type, v: $T) -> T where is_numeric(T)`
are both signatures.

A `ReturnList` is the return type list of 5.2a. It holds two or more values,
names every one of them, and does not combine with the `!` of a failure set.

A `UsesClause` draws one allocation capability per type (8a). Each is an
implicit parameter the body reaches by the type's own name with the first letter
lowercased, and a call supplies one argument per capability, found by that name
among what the caller holds and the `with` blocks around the call. A callee
drawing exactly one takes the innermost source whatever it is named.

A `WhereClause` is the bound of 11.4a. Its expression is read with struct
literals switched off, so the `{` that follows opens the body rather than a
literal.

A `:` at group depth zero marks a parameter list. When a parameter-shaped group
is not followed by a body, its contents are reinterpreted as expressions (a
single expression, or a tuple).

## 13.7 Types

```
Type =
      "i8" | "i16" | "i32" | "i64" | "isize"
    | "u8" | "u16" | "u32" | "u64" | "usize"
    | "f32" | "f64" | "bool" | "str"
    | "^" Type                               // raw pointer
    | "ref" Type                             // returnable borrow (3.3)
    | "[" "]" Type                           // slice
    | "[" SizeExpr "]" Type                  // array (length first)
    | "[" Type ";" INTEGER "]"               // array (element first)
    | "fn" "(" ( ProcParam ( "," ProcParam )* )? ")" ( "->" Type )?
    | "distinct" Type
    | "Handle" "<" Type ">"
    | IDENT "<" Type ( "," Type )* ">"       // generic instantiation
    | IDENT                                  // named type
    | "$" IDENT                              // type parameter
```

A type is a single prefix-constructed form. Nesting comes from the recursive
constructors (`^`, `ref`, `[]`, `distinct`, `fn`), not a postfix loop. Closing
`>` in the generic forms accepts a split `>>` (11.4).

There is no `&` in either position, which is why neither this production nor
`Prefix` in 13.4 has one. `&x` and `&mut x` are refused where an expression is
expected, with an error pointing at the parameter mode or at `ptr_to`, and `&T`
and `&mut T` are refused where a type is expected (3.3). The one place the
parser reads them is an internal re-parse: `type_from_string` sets a flag that
accepts them, so the compiler can read back the reference types its own
parameter-mode lowering wrote out. No program text reaches that path.

## 13.8 Comparison and equality precedence

The precedence ladder (14.1) places comparison tighter than equality and the
bitwise operators tighter than comparison, which differs from C. Write explicit
parentheses in mixed expressions. A conformance-minded style parenthesizes any
combination of `==`/`!=`, the comparisons, and the bitwise operators.
