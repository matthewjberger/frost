# Frost Language Specification

**Version 0.1.0**

Frost is a statically-oriented programming language combining Odin/Jai-style syntax with Rust-inspired immutability defaults. It compiles to bytecode executed by a stack-based virtual machine.

---

## Table of Contents

1. [Lexical Structure](#1-lexical-structure)
2. [Types](#2-types)
3. [Declarations](#3-declarations)
4. [Expressions](#4-expressions)
5. [Statements](#5-statements)
6. [Operators](#6-operators)
7. [Control Flow](#7-control-flow)
8. [Functions](#8-functions)
9. [Structs](#9-structs)
10. [Enums](#10-enums)
11. [Pointers and References](#11-pointers-and-references)
12. [Ownership and Borrowing](#12-ownership-and-borrowing)
13. [Arrays and Collections](#13-arrays-and-collections)
14. [Built-in Functions](#14-built-in-functions)
15. [Foreign Function Interface](#15-foreign-function-interface)
16. [Runtime Semantics](#16-runtime-semantics)
17. [Grammar Summary](#17-grammar-summary)

---

## 1. Lexical Structure

### 1.1 Keywords

```
break    case        comptime    continue    defer    distinct    else    enum
extern   false       fn          for         if       in          mut     match
return   sizeof      struct      true        unsafe   using       while
```

### 1.2 Type Keywords

```
i8    i16    i32    i64
u8    u16    u32    u64
f32   f64
bool  str    void
Arena    Pool    Handle
```

### 1.3 Operators and Punctuation

| Symbol | Name |
|--------|------|
| `+` | Addition |
| `-` | Subtraction / Negation |
| `*` | Multiplication |
| `/` | Division |
| `%` | Modulo |
| `==` | Equal |
| `!=` | Not Equal |
| `<` | Less Than |
| `>` | Greater Than |
| `<=` | Less Than or Equal |
| `>=` | Greater Than or Equal |
| `!` | Logical NOT |
| `&&` | Logical AND (short-circuiting) |
| `||` | Logical OR (short-circuiting) |
| `&` | Bitwise AND / Address-of |
| `|` | Bitwise OR |
| `<<` | Shift Left |
| `>>` | Shift Right |
| `^` | Dereference (postfix) |
| `:=` | Declaration with inference |
| `=` | Assignment |
| `::` | Constant declaration |
| `:` | Type annotation |
| `->` | Return type annotation |
| `.` | Field access |
| `..` | Range |
| `;` | Statement terminator |
| `,` | Separator |
| `(` `)` | Grouping / Function call |
| `{` `}` | Block / Struct / HashMap |
| `[` `]` | Array / Index |
| `_` | Wildcard pattern |

### 1.4 Identifiers

Identifiers begin with a letter or underscore, followed by letters, digits, or underscores.

```
identifier = (letter | "_") (letter | digit | "_")*
letter     = "a".."z" | "A".."Z"
digit      = "0".."9"
```

Examples: `x`, `count`, `my_variable`, `Vec3`, `_private`, `x1`

### 1.5 Literals

**Integer Literals**
```frost
42
0
-17
```

**Float Literals**
```frost
3.14
0.5
-2.718
```

**String Literals**
```frost
"hello"
"hello world"
"with \"escapes\""
```

**Boolean Literals**
```frost
true
false
```

### 1.6 Comments

**Single-line comments:**
```frost
// This is a single-line comment
x := 5  // Inline comment
```

**Block comments:**
```frost
/* This is a block comment */

/*
   Multi-line
   block comment
*/
```

---

## 2. Types

### 2.1 Primitive Types

| Type | Size | Description |
|------|------|-------------|
| `i8` | 1 byte | Signed 8-bit integer |
| `i16` | 2 bytes | Signed 16-bit integer |
| `i32` | 4 bytes | Signed 32-bit integer |
| `i64` | 8 bytes | Signed 64-bit integer |
| `u8` | 1 byte | Unsigned 8-bit integer |
| `u16` | 2 bytes | Unsigned 16-bit integer |
| `u32` | 4 bytes | Unsigned 32-bit integer |
| `u64` | 8 bytes | Unsigned 64-bit integer |
| `f32` | 4 bytes | 32-bit floating point |
| `f64` | 8 bytes | 64-bit floating point |
| `bool` | 1 byte | Boolean (true/false) |
| `str` | 16 bytes | String (pointer + length) |
| `void` | 0 bytes | No value |

### 2.2 Pointer Types

Pointer types are written with the caret prefix:

```frost
^i64        // Pointer to i64
^str        // Pointer to string
^^i32       // Pointer to pointer to i32
```

### 2.3 Reference Types

Reference types represent borrowed values:

```frost
&i64        // Immutable reference to i64
&mut i64    // Mutable reference to i64
&Point      // Immutable reference to Point struct
&mut str    // Mutable reference to string
```

References are created with the borrow operators (`&` and `&mut`) and dereferenced with the postfix `^` operator.

### 2.4 Array Types

Fixed-size arrays:
```frost
[10]i64     // Array of 10 i64 values
[5][5]f32   // 2D array (5x5 matrix of f32)
```

Slice types (dynamic size):
```frost
[]i64       // Slice of i64
[]str       // Slice of strings
```

### 2.5 Function Types

```frost
fn(i64, i64) -> i64    // Function taking two i64, returning i64
fn(f32) -> f32         // Function taking f32, returning f32
fn()                   // Function with no params, no return
```

### 2.6 Struct Types

Named struct types are declared with the `struct` keyword:

```frost
Point :: struct {
    x: i64,
    y: i64,
}
```

### 2.7 Enum Types

Named enum types are declared with the `enum` keyword:

```frost
Color :: enum {
    Red,
    Green,
    Blue,
}
```

### 2.8 Distinct Types

Distinct types create type-safe wrappers:

```frost
UserId :: distinct i64
Temperature :: distinct f32
```

### 2.9 Arena Type

Arenas provide region-based memory allocation:

```frost
frame : Arena = Arena::new(megabytes(4));
ptr := frame.alloc(Position { x = 0.0, y = 0.0 });
frame.reset();  // Free all allocations at once
```

### 2.10 Pool and Handle Types

Pools provide object storage with generational handles:

```frost
entities : Pool<Entity> = Pool::new(1024);
handle := entities.alloc(Entity { health = 100 });

// Safe access - returns null if handle is stale
if let Some(e) = entities.get(handle) {
    print(e.health);
}

entities.free(handle);
entities.get(handle);  // Returns null (generation mismatch)
```

Handle properties:
- `Handle<T>` is Copy (just two u32s: index + generation)
- Can be stored in structs and returned from functions
- Access is O(1) with generation check
- Dangling handles return null, never crash

### 2.11 Optional Type

Optional types represent values that may or may not exist:

```frost
?i64       // Optional i64
?Point     // Optional Point
```

### 2.12 Type Inference

When no explicit type is provided, types are inferred from the initializer:

| Literal | Inferred Type |
|---------|---------------|
| `42` | `i64` |
| `3.14` | `f64` |
| `"hello"` | `str` |
| `true` / `false` | `bool` |
| `[1, 2, 3]` | Array |

---

## 3. Declarations

### 3.1 Variable Declarations

**Immutable with type inference:**
```frost
x := 5
name := "Alice"
```

**Immutable with explicit type:**
```frost
x : i64 = 42
ratio : f64 = 0.5
```

**Mutable variables:**
```frost
mut counter := 0
mut value : i64 = 100
```

Variables are immutable by default. Use `mut` to allow reassignment.

### 3.2 Constant Declarations

Constants use the `::` operator and cannot be reassigned:

```frost
PI :: 3.14159
MAX_SIZE :: 1024
GREETING :: "Hello"
```

### 3.3 Function Declarations

Functions are first-class values assigned to variables or constants:

```frost
add := fn(a, b) { a + b }

multiply :: fn(x: i64, y: i64) -> i64 {
    x * y
}
```

### 3.4 Struct Declarations

```frost
Vec3 :: struct {
    x: f32,
    y: f32,
    z: f32,
}

Person :: struct {
    name: str,
    age: i64,
}
```

### 3.5 Enum Declarations

```frost
Direction :: enum {
    North,
    South,
    East,
    West,
}

Status :: enum {
    Ok,
    Error,
    Pending,
}
```

---

## 4. Expressions

### 4.1 Primary Expressions

**Identifiers:**
```frost
x
myVariable
```

**Literals:**
```frost
42
3.14
"string"
true
```

**Grouped expressions:**
```frost
(x + y) * z
```

### 4.2 Function Calls

```frost
add(5, 3)
print("Hello")
len(array)
```

### 4.3 Field Access

```frost
point.x
person.name
nested.inner.value
```

### 4.4 Index Expressions

```frost
array[0]
matrix[i][j]
map["key"]
```

### 4.5 Struct Initialization

```frost
Point { x = 10, y = 20 }
Vec3 { x = 1.0, y = 2.0, z = 3.0 }
```

### 4.6 Array Literals

```frost
[1, 2, 3]
[1 + 2, 3 * 4, 5]
[]
```

### 4.7 HashMap Literals

```frost
{ "name": "Alice", "age": 30 }
{ 1: "one", 2: "two" }
{}
```

### 4.8 Function Literals

```frost
fn(x, y) { x + y }
fn(n: i64) -> i64 { n * 2 }
fn(x: f64) -> f64 { x * x }
```

### 4.9 If Expressions

If expressions return values:

```frost
result := if (x > 0) { "positive" } else { "non-positive" }

max := if (a > b) { a } else { b }
```

### 4.10 Address-of Expression

```frost
&x
&array[0]
```

### 4.11 Dereference Expression

```frost
ptr^
pointer^
```

### 4.12 Sizeof Expression

```frost
sizeof(i64)      // 8
sizeof(^i32)     // 8 (pointer size)
sizeof(Vec3)     // Size of struct
```

### 4.13 Range Expressions

Range expressions are used in for loops:

```frost
0..10      // 0 to 9 (exclusive end)
start..end
```

### 4.14 Scoped Identifiers

For enum variants:

```frost
Color::Red
Direction::North
```

---

## 5. Statements

### 5.1 Expression Statements

Any expression followed by a semicolon:

```frost
print("hello");
x + y;
```

### 5.2 Declaration Statements

```frost
x := 5;
mut counter := 0;
PI :: 3.14;
```

### 5.3 Assignment Statements

```frost
x = 10;
point.x = 5;
ptr^ = 42;
```

Assignment requires the target to be mutable or a pointer dereference.

### 5.4 Return Statements

```frost
return 42
return value
return
```

### 5.5 Defer Statements

Defer executes a statement when the enclosing scope exits (LIFO order):

```frost
defer print("cleanup")
defer close(file)
```

Multiple defers execute in reverse order:

```frost
defer print("first")   // Executes third
defer print("second")  // Executes second
defer print("third")   // Executes first
```

### 5.6 For Statements

```frost
for i in 0..10 {
    print(i)
}

for x in start..end {
    // body
}
```

### 5.7 Break Statements

Exit the innermost loop immediately:

```frost
for i in 0..100 {
    if (i == 10) {
        break
    }
    print(i)
}
// Prints 0-9, then exits
```

### 5.8 Continue Statements

Skip to the next iteration of the innermost loop:

```frost
for i in 0..10 {
    if (i % 2 == 0) {
        continue
    }
    print(i)
}
// Prints only odd numbers: 1, 3, 5, 7, 9
```

---

## 6. Operators

### 6.1 Arithmetic Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition | `5 + 3` |
| `-` | Subtraction | `5 - 3` |
| `*` | Multiplication | `5 * 3` |
| `/` | Division | `10 / 2` |
| `%` | Modulo | `10 % 3` |

### 6.2 Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equal | `x == y` |
| `!=` | Not equal | `x != y` |
| `<` | Less than | `x < y` |
| `>` | Greater than | `x > y` |
| `<=` | Less than or equal | `x <= y` |
| `>=` | Greater than or equal | `x >= y` |

### 6.3 Logical Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `&&` | Logical AND (short-circuiting) | `a && b` |
| `||` | Logical OR (short-circuiting) | `a || b` |
| `!` | Logical NOT | `!flag` |

Short-circuit evaluation:
- `&&` evaluates the right operand only if the left is truthy
- `||` evaluates the right operand only if the left is falsy

```frost
x > 0 && x < 10       // true if x is in range (1, 9)
name == "" || len(name) > 100  // true if invalid
```

### 6.4 Unary Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `-` | Negation | `-x` |
| `!` | Logical NOT | `!flag` |
| `&` | Address-of | `&x` |

### 6.5 Postfix Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `^` | Dereference | `ptr^` |
| `()` | Function call | `f(x)` |
| `[]` | Index | `arr[i]` |
| `.` | Field access | `obj.field` |

### 6.6 Operator Precedence

From lowest to highest:

1. Range (`..`)
2. Logical OR (`||`)
3. Logical AND (`&&`)
4. Equality (`==`, `!=`)
5. Comparison (`<`, `>`, `<=`, `>=`)
6. Additive (`+`, `-`)
7. Multiplicative (`*`, `/`, `%`)
8. Unary prefix (`!`, `-`, `&`, `sizeof`)
9. Postfix (`()`, `[]`, `.`, `^`, `::`)

---

## 7. Control Flow

### 7.1 If Expressions

Basic if:
```frost
if (condition) {
    // then branch
}
```

If-else:
```frost
if (condition) {
    // then branch
} else {
    // else branch
}
```

If as expression:
```frost
result := if (x > 0) { x } else { -x }
```

Else if chains:
```frost
if (x > 0) {
    "positive"
} else if (x < 0) {
    "negative"
} else {
    "zero"
}
```

Multiple conditions:
```frost
if (score >= 90) {
    "A"
} else if (score >= 80) {
    "B"
} else if (score >= 70) {
    "C"
} else if (score >= 60) {
    "D"
} else {
    "F"
}
```

### 7.2 For Loops

Range-based for loop:
```frost
for i in 0..10 {
    print(i)
}
```

The range `start..end` iterates from `start` to `end - 1` (exclusive end).

Nested loops:
```frost
for i in 0..10 {
    for j in 0..10 {
        // body
    }
}
```

### 7.3 While Loops

Condition-based loops:
```frost
mut i := 0;
while (i < 10) {
    print(i);
    i = i + 1;
}
```

### 7.4 Break and Continue

**Break** exits the innermost loop:
```frost
for i in 0..100 {
    if (i == 10) {
        break  // Exit loop when i reaches 10
    }
}
```

**Continue** skips to the next iteration:
```frost
for i in 0..10 {
    if (i % 2 == 0) {
        continue  // Skip even numbers
    }
    print(i)  // Only prints odd numbers
}
```

### 7.5 Match Expressions

Match expressions provide pattern matching:

```frost
result := match x {
    case 1: "one"
    case 2: "two"
    case _: "other"
}
```

Match on enum variants:

```frost
Result :: enum {
    Ok { value: i64 },
    Err { code: i64 },
}

msg := match result {
    case .Ok { value }: value
    case .Err { code }: 0 - code
}
```

### 7.6 Pattern Matching

Patterns can be:

**Literals:**
```frost
match x {
    case 0: "zero"
    case 1: "one"
    case _: "other"
}
```

**Wildcards:**
```frost
match x {
    case _: "anything"
}
```

**Identifiers (binding):**
```frost
match x {
    case n: n * 2  // Binds x to n
}
```

**Enum variants:**
```frost
match result {
    case .Ok { value }: value
    case .Err { code }: code
}
```

**Tuples:**
```frost
match (a, b) {
    case (0, 0): "both zero"
    case (0, _): "a is zero"
    case (_, 0): "b is zero"
    case (_, _): "neither zero"
}
```

### 7.7 Tuple Expressions

Tuples group multiple values:

```frost
pair := (1, 2)
triple := ("hello", 42, true)
```

Tuples are commonly used with match for multi-value pattern matching:

```frost
for i in 1..101 {
    result := match (i % 3, i % 5) {
        case (0, 0): "FizzBuzz"
        case (0, _): "Fizz"
        case (_, 0): "Buzz"
        case (_, _): i
    };
    print(result);
}
```

---

## 8. Functions

### 8.1 Function Syntax

Anonymous function:
```frost
fn(parameters) { body }
fn(parameters) -> return_type { body }
```

Named function (via binding):
```frost
add := fn(a, b) { a + b }

multiply :: fn(x: i64, y: i64) -> i64 {
    x * y
}
```

### 8.2 Typed Functions

Functions with type annotations:

```frost
fn(x: i64, y: i64) -> i64 { x + y }

operation :: fn(a: i64, b: i64) -> i64 {
    return a * b
}
```

### 8.3 Parameters

Untyped parameters:
```frost
fn(a, b) { a + b }
```

Typed parameters:
```frost
fn(a: i64, b: i64) { a + b }
```

### 8.4 Return Types

Implicit return (last expression):
```frost
fn(x) { x * 2 }    // Returns x * 2
```

Explicit return type:
```frost
fn(x: i64) -> i64 { x * 2 }
```

Explicit return statement:
```frost
fn(x) {
    if (x < 0) { return 0 }
    x * 2
}
```

### 8.5 Closures

Functions capture variables from enclosing scopes:

```frost
make_adder := fn(n) {
    fn(x) { x + n }    // Captures n
}

add5 := make_adder(5)
print(add5(3))         // 8
```

### 8.6 Recursion

Functions can call themselves:

```frost
factorial := fn(n) {
    if (n <= 1) { 1 } else { n * factorial(n - 1) }
}

fib := fn(n) {
    if (n < 2) { n } else { fib(n - 1) + fib(n - 2) }
}
```

---

## 9. Structs

### 9.1 Declaration

```frost
Point :: struct {
    x: i64,
    y: i64,
}

Vec3 :: struct {
    x: f32,
    y: f32,
    z: f32,
}
```

### 9.2 Initialization

```frost
p := Point { x = 10, y = 20 }
v := Vec3 { x = 1.0, y = 2.0, z = 3.0 }
```

### 9.3 Field Access

```frost
x_value := p.x
y_value := p.y
```

### 9.4 Field Assignment

```frost
p.x = 100
p.y = 200
```

Note: Struct field assignment modifies the variable holding the struct.

---

## 10. Enums

### 10.1 Simple Enums

Simple enums have variants without data:

```frost
Color :: enum {
    Red,
    Green,
    Blue,
}

Status :: enum {
    Ok,
    Error,
    Pending,
}
```

### 10.2 Tagged Unions (Enums with Data)

Enum variants can carry named fields:

```frost
Result :: enum {
    Ok { value: i64 },
    Err { code: i64, message: str },
    None,
}

Shape :: enum {
    Circle { radius: f64 },
    Rectangle { width: f64, height: f64 },
    Point,
}
```

### 10.3 Variant Access

```frost
color := Color::Red
status := Status::Ok
```

### 10.4 Tagged Union Construction

```frost
r := Result::Ok { value = 42 }
err := Result::Err { code = 404, message = "Not found" }
circle := Shape::Circle { radius = 5.0 }
```

### 10.5 Representation

Enum variants are represented as consecutive integers starting from 0:
- `Color::Red` = 0
- `Color::Green` = 1
- `Color::Blue` = 2

Tagged unions are represented as a tag (u32) plus field values.

---

## 11. Pointers and References

### 11.1 Pointer Types

```frost
p : ^i64       // Pointer to i64
pp : ^^i64     // Pointer to pointer to i64
```

### 11.2 Reference Types

References are safe borrows of values:

```frost
r : &i64       // Immutable reference to i64
mr : &mut i64  // Mutable reference to i64
```

### 11.3 Borrow Operators

**Immutable borrow:**
```frost
x := 42
r := &x        // r is an immutable reference to x
print(r^)      // Read through reference
```

**Mutable borrow:**
```frost
mut y := 100
mr := &mut y   // mr is a mutable reference to y
mr^ = 200      // Write through mutable reference
```

### 11.4 Dereference Operator

The dereference operator is postfix `^`:

```frost
value := r^    // Read value at reference/pointer
mr^ = 100      // Write value through mutable reference/pointer
```

### 11.5 Pointer Arithmetic

Pointer arithmetic is not directly supported. Use array indexing instead.

---

## 12. Ownership and Borrowing

Frost implements an ownership system inspired by Rust, providing memory safety without garbage collection.

### 12.1 Ownership Rules

1. Each value has a single owner (the variable that holds it)
2. When the owner goes out of scope, the value is dropped
3. Ownership can be transferred (moved) to another variable

### 12.2 Copy vs Move Types

**Copy types** are duplicated when assigned or passed to functions:
- All primitive types: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `f32`, `f64`, `bool`
- References: `&T`, `&mut T`
- Pointers: `^T`
- Function types: `fn(...) -> T`

```frost
x := 42
y := x     // x is copied, both x and y are valid
print(x)   // OK
print(y)   // OK
```

**Move types** transfer ownership when assigned or passed:
- Strings: `str`
- Structs
- Enums
- Arrays and slices

```frost
Point :: struct { x: i64, y: i64 }

p := Point { x = 10, y = 20 }
q := p     // p is moved to q
// print(p.x)  // ERROR: use of moved value 'p'
print(q.x)    // OK
```

### 12.3 Borrowing Rules

Borrowing allows temporary access to a value without taking ownership:

1. **Multiple immutable borrows are allowed:**
```frost
x := 42
r1 := &x
r2 := &x   // OK: multiple immutable borrows
print(r1^ + r2^)
```

2. **Mutable borrows are exclusive:**
```frost
mut x := 42
r := &mut x
// s := &mut x  // ERROR: cannot borrow as mutable more than once
// t := &x      // ERROR: cannot borrow as immutable while mutably borrowed
```

3. **Cannot borrow moved values:**
```frost
Point :: struct { x: i64, y: i64 }
take :: fn(p: Point) { print(p.x) }

p := Point { x = 1, y = 2 }
take(p)     // p is moved
// r := &p  // ERROR: cannot borrow moved value
```

4. **Mutable borrows require mutable variables:**
```frost
x := 42       // immutable
// r := &mut x  // ERROR: cannot mutably borrow immutable variable

mut y := 42   // mutable
r := &mut y   // OK
```

5. **References are second-class citizens:**
```frost
// ERROR: functions cannot return references
bad :: fn() -> &i64 {
    x := 42
    return &x  // would be dangling
}

// ERROR: cannot store references in structs
BadStruct :: struct {
    ref: &i64,  // ERROR
}

// ERROR: cannot store references in arrays
refs := [&a, &b, &c];  // ERROR

// OK: return owned values instead
good :: fn() -> i64 {
    x := 42
    return x
}

// OK: functions can accept references as parameters
read :: fn(r: &i64) -> i64 {
    r^
}

// OK: use Handle<T> for persistent references
entities : Pool<Entity> = Pool::new(1024);
handle := entities.alloc(entity);  // Can be stored, returned, etc.
```

This restriction ensures memory safety without requiring lifetime annotations.

### 12.4 Scope-based Lifetimes

Borrows are valid until the end of their scope:

```frost
mut x := 10
{
    r := &x      // borrow starts
    print(r^)
}                // borrow ends here
x = 20           // OK: no active borrows
```

### 12.5 Field Access and Ownership

Accessing fields of a struct does not move the struct:

```frost
Point :: struct { x: i64, y: i64 }

p := Point { x = 10, y = 20 }
sum := p.x + p.y   // OK: field access doesn't move
print(p.x)         // OK: p is still valid
```

### 12.6 Automatic Drop

Values that implement drop are automatically cleaned up when they go out of scope:

```frost
{
    s := some_struct();   // s is created
    // use s...
}                         // s is automatically dropped here
```

In loops, values defined within the loop body are dropped at the end of each iteration:

```frost
for i in 0..10 {
    temp := create_value()   // created
    use(temp)
}                            // temp dropped each iteration
```

### 12.7 Unsafe Blocks

Unsafe blocks disable certain safety checks:

```frost
result := unsafe {
    // Inside unsafe:
    // - Can return references from functions
    // - Other reference restrictions relaxed
    dangerous_operation()
}
```

Unsafe blocks should be used sparingly and only when necessary for FFI or low-level operations.

---

## 13. Arrays and Collections

### 13.1 Array Literals

```frost
numbers := [1, 2, 3, 4, 5]
empty := []
mixed := [1 + 2, 3 * 4]
```

### 13.2 Array Indexing

```frost
first := numbers[0]
last := numbers[len(numbers) - 1]
```

### 13.3 HashMap Literals

```frost
map := { "name": "Alice", "city": "Boston" }
scores := { "alice": 100, "bob": 85 }
empty_map := {}
```

### 13.4 HashMap Indexing

```frost
name := map["name"]
score := scores["alice"]
```

---

## 14. Built-in Functions

### 14.1 Array Functions

| Function | Description | Example |
|----------|-------------|---------|
| `len(array)` | Returns length | `len([1,2,3])` = 3 |
| `first(array)` | Returns first element | `first([1,2,3])` = 1 |
| `last(array)` | Returns last element | `last([1,2,3])` = 3 |
| `rest(array)` | Returns all but first | `rest([1,2,3])` = [2,3] |
| `push(array, value)` | Appends value | `push([1,2], 3)` = [1,2,3] |

### 14.2 I/O Functions

| Function | Description | Example |
|----------|-------------|---------|
| `print(value)` | Prints to stdout | `print("hello")` |

### 14.3 Memory Management

Memory is managed automatically through ownership and the Drop system (see Section 12). There are no manual allocation or deallocation functions - values are allocated when created and freed automatically when they go out of scope.

---

## 15. Foreign Function Interface

### 15.1 Extern Function Declarations

External C functions can be declared using `extern fn`:

```frost
puts :: extern fn(s: ^i8) -> i32
printf :: extern fn(fmt: ^i8) -> i32
malloc :: extern fn(size: u64) -> ^u8
free :: extern fn(ptr: ^u8)
```

When compiled to native code, these link against the C library:

```frost
puts :: extern fn(s: ^i8) -> i32

main :: fn() -> i64 {
    puts("Hello from Frost!");
    0
}
```

Compile and link:
```bash
frost --link -o hello hello.frost
./hello
```

### 15.2 Native Function Registration (Bytecode VM)

Native functions are registered from the host language (Rust):

```rust
let mut registry = NativeRegistry::new();

registry.register("sin", 1, |args, _handles| {
    let value = args[0].as_f64();
    Ok(Value64::Float(value.sin()))
});

registry.register("custom_add", 2, |args, _handles| {
    let a = args[0].as_i64();
    let b = args[1].as_i64();
    Ok(Value64::Integer(a + b))
});
```

### 15.2 Calling Native Functions

From Frost code, native functions are called like regular functions:

```frost
result := sin(3.14159)
sum := custom_add(10, 20)
```

### 15.3 Handle Registry

For opaque native objects:

```rust
// Store a native object
let handle = handles.store(MyNativeType { ... });

// Retrieve later
if let Some(obj) = handles.get::<MyNativeType>(handle) {
    // Use obj
}

// Free when done
handles.free(handle);
```

### 15.4 Value Marshaling

All values are passed as `Value64`:

| Frost Type | Value64 Variant |
|------------|-----------------|
| Integer | `Value64::Integer(i64)` |
| Float | `Value64::Float(f64)` |
| Boolean | `Value64::Bool(bool)` |
| Null | `Value64::Null` |
| Heap objects | `Value64::HeapRef(u32)` |

---

## 16. Runtime Semantics

### 16.1 Execution Model

Frost supports two execution modes:

**Bytecode VM** (default):
- Interpreted execution via stack-based virtual machine
- Full feature support including arenas, pools, handles
- Good for development and REPL

**Native Compilation** (via Cranelift):
- Compiles to native machine code
- Produces real executables
- Links against libc for I/O

```bash
# Bytecode VM (default)
frost program.frost

# Native compilation
frost --native -o program.o program.frost    # Object file
frost --link -o program program.frost        # Executable
```

### 16.2 Bytecode VM Details

The bytecode VM provides:
- **Stack**: 2,048 value slots
- **Globals**: 65,536 variable slots
- **Call frames**: 1,024 maximum depth

### 16.3 Value Representation

All values are 64 bits:

```
Value64:
  - Integer(i64)      64-bit signed integer
  - Float(f64)        64-bit floating point
  - Bool(bool)        Boolean
  - Null              Null value
  - HeapRef(u32)      Reference to heap object
```

### 16.4 Heap Objects

Complex values are heap-allocated:

- `String` - String data
- `Array` - Dynamic array of values
- `HashMap` - Key-value map
- `Closure` - Function with captured variables
- `Struct` - Struct instance with fields
- `NativeFunction` - FFI function reference
- `NativeHandle` - Opaque native object

### 16.5 Truthiness

Values evaluate to boolean in conditions:

| Value | Truthy |
|-------|--------|
| `true` | Yes |
| `false` | No |
| `null` | No |
| Integer 0 | No |
| Other integers | Yes |
| All other values | Yes |

### 16.6 Mutability

Variables are immutable by default:

```frost
x := 5
x = 10      // ERROR: cannot assign to immutable variable

mut y := 5
y = 10      // OK
```

### 16.7 Scope

Variables are lexically scoped:

```frost
x := 1
{
    x := 2    // Shadows outer x
    print(x)  // 2
}
print(x)      // 1
```

### 16.8 Defer Execution

Defer statements execute in LIFO order when scope exits:

```frost
{
    defer print("A")
    defer print("B")
    defer print("C")
    // Prints: C, B, A when block exits
}
```

---

## 17. Grammar Summary

```ebnf
program        = statement* ;

statement      = let_statement
               | const_statement
               | struct_decl
               | enum_decl
               | return_statement
               | defer_statement
               | for_statement
               | break_statement
               | continue_statement
               | assignment
               | expression_stmt ;

break_statement = "break" ";" ;
continue_statement = "continue" ";" ;

let_statement  = ["mut"] IDENT (":=" expression | ":" type "=" expression) ";" ;
const_statement = IDENT "::" expression ";" ;
struct_decl    = IDENT "::" "struct" "{" struct_fields "}" ;
enum_decl      = IDENT "::" "enum" "{" enum_variants "}" ;
enum_variant   = IDENT ["{" struct_fields "}"] ;
return_statement = "return" [expression] ;
defer_statement = "defer" statement ;
for_statement  = "for" IDENT "in" range "{" statement* "}" ;
assignment     = lvalue "=" expression ";" ;
expression_stmt = expression ";" ;

expression     = if_expr
               | fn_expr
               | match_expr
               | infix_expr ;

if_expr        = "if" "(" expression ")" block ["else" (if_expr | block)] ;
fn_expr        = "fn" "(" [params] ")" ["->" type] block ;
match_expr     = "match" expression "{" match_case+ "}" ;
match_case     = "case" pattern ":" (block | expression) ;

pattern        = "_"
               | literal
               | IDENT
               | "." IDENT ["{" bindings "}"]
               | IDENT "::" IDENT ["{" bindings "}"]
               | "(" pattern ("," pattern)* ")" ;

bindings       = IDENT ("," IDENT)* ;

infix_expr     = prefix_expr (binary_op prefix_expr)* ;
prefix_expr    = unary_op* postfix_expr ;
postfix_expr   = primary (call_args | index | "." IDENT | "^")* ;

primary        = IDENT
               | INTEGER
               | FLOAT
               | STRING
               | "true" | "false"
               | "(" expression ")"
               | "(" expression "," expression+ ")"
               | array_literal
               | hashmap_literal
               | struct_init
               | scoped_ident
               | enum_variant_init
               | "sizeof" "(" type ")"
               | "&" expression
               | "&" "mut" expression ;

array_literal  = "[" [expression ("," expression)*] "]" ;
hashmap_literal = "{" [pair ("," pair)*] "}" ;
struct_init    = IDENT "{" [field_init ("," field_init)*] "}" ;
enum_variant_init = IDENT "::" IDENT "{" [field_init ("," field_init)*] "}" ;
scoped_ident   = IDENT "::" IDENT ;

range          = expression ".." expression ;
call_args      = "(" [expression ("," expression)*] ")" ;
index          = "[" expression "]" ;

type           = primitive_type
               | "^" type
               | "&" type
               | "&" "mut" type
               | "[" [INTEGER] "]" type
               | "fn" "(" [type_list] ")" ["->" type]
               | IDENT ;

primitive_type = "i8" | "i16" | "i32" | "i64"
               | "u8" | "u16" | "u32" | "u64"
               | "f32" | "f64"
               | "bool" | "str" | "void" ;

binary_op      = "+" | "-" | "*" | "/" | "%"
               | "==" | "!=" | "<" | ">" | "<=" | ">="
               | "&&" | "||"
               | ".." ;

unary_op       = "-" | "!" | "&" ;

block          = "{" statement* [expression] "}" ;
```

---

## Appendix A: Example Programs

### A.1 Hello World

```frost
print("Hello, World!")
```

### A.2 Factorial

```frost
factorial := fn(n) {
    if (n <= 1) { 1 } else { n * factorial(n - 1) }
}

print(factorial(5))  // 120
```

### A.3 FizzBuzz

Using match with tuple patterns:

```frost
for i in 1..101 {
    result := match (i % 3, i % 5) {
        case (0, 0): "FizzBuzz"
        case (0, _): "Fizz"
        case (_, 0): "Buzz"
        case (_, _): i
    };
    print(result);
}
```

Using if-else (alternative):

```frost
for i in 1..101 {
    if (i % 15 == 0) {
        print("FizzBuzz")
    } else if (i % 3 == 0) {
        print("Fizz")
    } else if (i % 5 == 0) {
        print("Buzz")
    } else {
        print(i)
    }
}
```

### A.4 Higher-Order Functions

```frost
make_adder := fn(n) {
    fn(x) { x + n }
}

add5 := make_adder(5)
add10 := make_adder(10)

print(add5(3))   // 8
print(add10(3))  // 13
```

### A.5 Structs

```frost
Point :: struct {
    x: i64,
    y: i64,
}

distance_squared := fn(p: Point) -> i64 {
    p.x * p.x + p.y * p.y
}

origin := Point { x = 0, y = 0 }
p := Point { x = 3, y = 4 }

print(distance_squared(p))  // 25
```

### A.6 Working with Arrays

```frost
numbers := [1, 2, 3, 4, 5]

print(len(numbers))    // 5
print(first(numbers))  // 1
print(last(numbers))   // 5

more := push(numbers, 6)
print(len(more))       // 6
```

### A.7 Defer for Cleanup

```frost
process := fn() {
    print("Starting")
    defer print("Cleanup 1")
    defer print("Cleanup 2")
    print("Working")
    42
}

result := process()
// Output:
// Starting
// Working
// Cleanup 2
// Cleanup 1
```

### A.8 Pointers

```frost
swap := fn(a: ^i64, b: ^i64) {
    temp := a^
    a^ = b^
    b^ = temp
}

mut x := 10
mut y := 20
swap(&x, &y)
print(x)  // 20
print(y)  // 10
```

### A.9 Tagged Unions with Pattern Matching

```frost
Result :: enum {
    Ok { value: i64 },
    Err { code: i64 },
}

unwrap := fn(r: Result) -> i64 {
    match r {
        case .Ok { value }: value
        case .Err { code }: {
            print("Error occurred");
            0 - code
        }
    }
}

success := Result::Ok { value = 42 }
failure := Result::Err { code = 404 }

print(unwrap(success))  // 42
print(unwrap(failure))  // Error occurred, -404
```

---

## Appendix B: Opcode Reference

| Opcode | Operands | Description |
|--------|----------|-------------|
| `Constant` | index | Push constant |
| `Pop` | - | Discard top |
| `Add`, `Sub`, `Mul`, `Div`, `Mod` | - | Generic arithmetic |
| `AddI64`, `SubI64`, etc. | - | Typed integer arithmetic |
| `AddF64`, `SubF64`, etc. | - | Typed float arithmetic |
| `True`, `False`, `Null` | - | Push literals |
| `Equal`, `NotEqual`, `GreaterThan` | - | Comparison |
| `EqualI64`, etc. | - | Typed comparison |
| `Minus`, `Bang` | - | Unary operators |
| `NegateI64`, `NegateF64` | - | Typed negation |
| `Jump` | addr | Unconditional jump |
| `JumpNotTruthy` | addr | Conditional jump |
| `GetGlobal`, `SetGlobal` | index | Global variable access |
| `GetLocal`, `SetLocal` | index | Local variable access |
| `GetBuiltin` | index | Built-in function |
| `GetNative` | index | Native function |
| `GetFree` | index | Closure free variable |
| `Array` | count | Create array |
| `Hash` | count | Create hashmap |
| `Index` | - | Array/map access |
| `Call` | argc | Function call |
| `Closure` | fn, free | Create closure |
| `ReturnValue`, `Return` | - | Return from function |
| `CurrentClosure` | - | Self-reference |
| `LoadPtr`, `StorePtr` | - | Pointer operations |
| `AddressOfLocal`, `AddressOfGlobal` | index | Take address |
| `Alloc`, `Free` | - | Memory management |
| `StructAlloc` | size | Allocate struct |
| `StructGet`, `StructSet` | offset | Struct field access |
| `TaggedUnionAlloc` | num_fields | Allocate tagged union |
| `TaggedUnionSetTag` | tag | Set tag on tagged union |
| `TaggedUnionGetTag` | - | Get tag from tagged union |
| `TaggedUnionGetField` | offset | Get field from tagged union |
| `TaggedUnionSetField` | offset | Set field on tagged union |
| `TupleAlloc` | size | Allocate tuple (pops N values) |
| `TupleGet` | index | Get element from tuple |
| `Dup` | - | Duplicate top of stack |
| `Drop` | - | Drop a heap-allocated value (ownership cleanup) |

---

*End of Frost Language Specification*
