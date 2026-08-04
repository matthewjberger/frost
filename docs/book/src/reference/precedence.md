# 14. Appendix

## 14.1 Operator precedence

Lowest to highest binding. All binary operators are left-associative. This table
is normative and matches the reference parser's precedence mapping.

| Level | Operators | Notes |
| --- | --- | --- |
| Range | `..` `..=` | range construction |
| LogicalOr | `\|\|` | |
| LogicalAnd | `&&` | tighter than `\|\|` |
| Equals | `==` `!=` | |
| Comparison | `<` `<=` `>` `>=` | tighter than equality |
| BitwiseOr | `\|` | |
| BitwiseAnd | `&` | |
| Shift | `<<` `>>` | |
| Sum | `+` `-` | |
| Product | `*` `/` `%` | |
| Prefix | `-` `!` | unary |
| Call / Index / Access | `f(...)` `a[i]` `.` `^` `::` | tightest |

## 14.2 Keywords

Specified language, matching the keyword table of `src/lexer.rs` (2.4):

```
break case continue defer distinct else enum extern false fn for if import
in inline linear match move mut ref return safe struct true
type unsafe uses var where while with
```

Primitive type names are `i8 i16 i32 i64 isize u8 u16 u32 u64 usize f32 f64 bool
str void`. The wildcard is `_`. `test`, `export`, `flags` and `value` are
contextual, not reserved, and so is the capitalized `Type` of `$T: Type`.

## 14.3 String escapes

`\n` `\t` `\r` `\0` `\\` `\"` `\'`. Any other escape is an error.

## 14.4 Related documents

- [tour.md](../tour.md), the language by example.
- [coming-from-rust.md](../coming-from-rust.md), a guide for Rust programmers.
- [memory-safety.md](../design/memory-safety.md), the safety guarantees in depth.
- [philosophy.md](../design/philosophy.md), the design rationale.
- [architecture.md](../impl/architecture.md), the compiler pipeline.
