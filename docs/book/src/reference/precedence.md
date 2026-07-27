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

Specified language:

```
fn struct enum match case if else while for in mut return break continue
defer extern import linear distinct type unsafe sizeof
```

Primitive type names are `i8 i16 i32 i64 isize u8 u16 u32 u64 usize f32 f64 bool
str void`. The wildcard is `_`. `test`, `export` and `flags` are contextual, not
reserved.

## 14.3 String escapes

`\n` `\t` `\r` `\0` `\\` `\"` `\'`. Any other escape is an error.

## 14.4 Related documents

- [tour.md](../tour.md), the language by example.
- [coming-from-rust.md](../coming-from-rust.md), a guide for Rust programmers.
- [memory-safety.md](../design/memory-safety.md), the safety guarantees in depth.
- [philosophy.md](../design/philosophy.md), the design rationale.
- [architecture.md](../impl/architecture.md), the compiler pipeline.
