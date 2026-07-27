# 13. Grammar

This chapter is the complete syntax of the specified language. Legacy forms the
current parser also accepts are listed in 13.9 and are not part of the language.

## 13.1 Program and statements

```
Program   = Statement*

Statement =
      "return" ( Expr ( "," Expr )* )? ";"?
    | "defer" Statement
    | "for" IDENT ( "," IDENT )? "in" Expr Block
    | "while" "(" Expr ")" Block
    | "break" ";"?
    | "continue" ";"?
    | "import" STRING ";"?
    | "mut" IDENT ( ":=" Expr | ":" Type "=" Expr ) ";"?
    | MultiNames ":=" Expr ";"?              // several values from one call
    | IDENT ":=" Expr ";"?
    | IDENT ":" Type "=" Expr ";"?           // lookahead: ":" not followed by ":"
    | IDENT "::" ConstBody ";"?
    | "print" Expr ( "," Expr )* ";"?         // a value, or a format and its values
    | Expr ( "=" Expr )? ";"?                 // expression statement or assignment
```

```
MultiNames = MultiName ( "," MultiName )+
MultiName  = "mut"? IDENT
```

A name followed by a comma at statement position is a list binding and nothing
else, which is what tells the two `:=` forms apart.

The `mut` / `:=` / `: =` / `::` forms are selected by the token after the
identifier. These are `:=` (inferred binding), `:` then a non-`:` (typed
binding), and `::` (constant). The last alternative covers expression statements
and assignments to a place.

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
              | ParamMode? IDENT ( ":" Type )?
ParamMode     = "mut" | "move"
ProcType      = "fn" "(" ( ProcParam ( "," ProcParam )* )? ")" ( "->" Type )?
ProcParam     = ParamMode? Type
```

A `Name :: fn(...) { ... }` item is the `Expr` alternative of `ConstBody`, whose
expression is a function literal (13.6).

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
    | "&" "mut"? Expr                         // borrow / mutable borrow
    | "$" Type                                // type value (11.3)
    | "sizeof" "(" Type ")"

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
    | Expr "(" ( Expr ( "," Expr )* )? ")"    // call
    | Expr "." IDENT                          // field access
    | Expr "^"                                // dereference (assignable place)
    | IDENT "{" StructInit? "}"              // struct literal (bare identifier)
    | IDENT "::" IDENT ( "{" StructInit? "}" )?   // enum variant (bare identifier)

StructInit = IDENT "=" Expr ( "," IDENT "=" Expr )* ","?
```

`Expr "^"` and the struct/enum-init forms are only entered when the left operand
is the appropriate shape (a place for `^`, a bare identifier for `{`/`::`). The
struct-literal `{` is disambiguated from a `match` body by checking that the
token after `{` is not `case`.

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

ReturnSig = "->" ( Type ( "!" Type )? | ReturnList ) ( "uses" Type ( "," Type )* )?
ReturnList = "(" ReturnValue "," ReturnValue ( "," ReturnValue )* ")"
ReturnValue = ( IDENT ":" )? Type
```

A `ReturnList` is the return type list of 5.2a. It holds two or more values,
names all of them or none, and does not combine with the `!` of a failure set.

A `uses` list draws one allocation capability per type. Each is an implicit
parameter the body reaches by the type's own name with the first letter
lowercased, and a call supplies one argument per capability, found by that name
among what the caller holds and the `with` blocks around the call. A callee
drawing exactly one takes the innermost source whatever it is named.

A `:` at group depth zero marks a parameter list. When a parameter-shaped group
is not followed by a body, its contents are reinterpreted as expressions (a
single expression, or a tuple).

## 13.7 Types

```
Type =
      "i8" | "i16" | "i32" | "i64" | "isize"
    | "u8" | "u16" | "u32" | "u64" | "usize"
    | "f32" | "f64" | "bool" | "str" | "void"
    | "^" Type                               // raw pointer
    | "&" "mut"? Type                        // reference
    | "[" "]" Type                           // slice
    | "[" INTEGER "]" Type                   // array (size first)
    | "[" Type ";" INTEGER "]"               // array (element first)
    | "fn" "(" ( ProcParam ( "," ProcParam )* )? ")" ( "->" Type )?
    | "distinct" Type
    | "Handle" "<" Type ">"
    | IDENT "<" Type ( "," Type )* ">"       // generic instantiation
    | IDENT                                  // named type
    | "$" IDENT                              // type parameter
```

A type is a single prefix-constructed form. Nesting comes from the recursive
constructors (`^`, `[]`, `?`, `distinct`, `fn`), not a postfix loop. Closing
`>` in the generic forms accepts a split `>>` (11.4).

## 13.8 Comparison and equality precedence

The precedence ladder (14.1) places comparison tighter than equality and the
bitwise operators tighter than comparison, which differs from C. Write explicit
parentheses in mixed expressions. A conformance-minded style parenthesizes any
combination of `==`/`!=`, the comparisons, and the bitwise operators.
