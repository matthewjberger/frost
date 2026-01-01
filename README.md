# Frost

A statically-oriented programming language combining the simplicity of Odin and Jai with Rust-inspired immutability. Frost compiles to bytecode and runs on a fast virtual machine.

## Features

- **Immutable by default** - Variables are immutable unless declared with `mut`
- **Odin-style syntax** - `:=` for inference, `::` for constants, `proc` for typed functions
- **First-class functions** - Closures, higher-order functions, recursion
- **Structs** - User-defined data types
- **Defer** - Cleanup statements that run at scope exit
- **Bytecode VM** - Fast execution with a stack-based virtual machine

## Quick Start

### Build

```bash
cargo build --release
```

### REPL

```bash
cargo run --release -p repl
```

```
> x := 5
> y := x * 2
> y
10
> factorial := fn(n) { if (n <= 1) { 1 } else { n * factorial(n - 1) } }
> factorial(5)
120
```

### Run Examples

```bash
cargo run --release -p frost --bin frost -- examples/hello_world.frost
cargo run --release -p frost --bin frost -- examples/fizzbuzz.frost
cargo run --release -p frost --bin frost -- examples/recursion.frost
```

### Run Benchmarks

```bash
cargo run --release -p frost --bin benchmark
```

## Language Guide

### Variables

```rust
// Immutable (default)
x := 5
name := "Frost"

// Mutable
mut counter := 0
counter = counter + 1

// With type annotation
age : i64 = 25
mut score : i64 = 0
```

### Constants

```rust
// Compile-time constants (Odin-style ::)
PI :: 3.14159
MAX_SIZE :: 1024
GREETING :: "Hello!"
```

### Functions

```rust
// Basic function
add := fn(a, b) { a + b }

// With explicit return
multiply := fn(a, b) {
    return a * b
}

// Recursive
factorial := fn(n) {
    if (n <= 1) { 1 } else { n * factorial(n - 1) }
}

// Higher-order
apply_twice := fn(f, x) { f(f(x)) }
```

### Procedures (Typed Functions)

```rust
// Odin-style procedures with type annotations
add :: proc(a: i64, b: i64) -> i64 {
    a + b
}

is_even :: proc(n: i64) -> bool {
    n % 2 == 0
}
```

### Structs

```rust
// Define a struct
Point :: struct {
    x: i64,
    y: i64,
}

// Create instances
origin := Point { x = 0, y = 0 }
p := Point { x = 10, y = 20 }

// Access fields
print(p.x)
print(p.y)
```

### Control Flow

```rust
// If expressions (they return values!)
result := if (x > 5) { "big" } else { "small" }

// For loops with ranges
mut sum := 0
for i in 0..10 {
    sum = sum + i
}
```

### Arrays

```rust
numbers := [1, 2, 3, 4, 5]

// Access
first_num := numbers[0]

// Built-in functions
print(len(numbers))    // 5
print(first(numbers))  // 1
print(last(numbers))   // 5
print(rest(numbers))   // [2, 3, 4, 5]

extended := push(numbers, 6)  // [1, 2, 3, 4, 5, 6]
```

### HashMaps

```rust
ages := {
    "Alice": 30,
    "Bob": 25,
}

print(ages["Alice"])  // 30
```

### Closures

```rust
make_adder := fn(n) {
    fn(x) { x + n }
}

add5 := make_adder(5)
print(add5(10))  // 15
```

### Defer

```rust
// Deferred statements run at scope exit (LIFO order)
test := fn() {
    mut x := 0
    defer x = x + 1
    defer x = x * 10
    x = 5
    x
}
// Returns 5, but x becomes 51 after defers run
```

### Strings

```rust
greeting := "Hello"
name := "World"
message := greeting + ", " + name + "!"
print(message)
print(len(message))
```

## Types

| Type | Description |
|------|-------------|
| `i64` | 64-bit signed integer |
| `f64` | 64-bit floating point |
| `bool` | Boolean (`true`/`false`) |
| `str` | String |
| `^T` | Pointer to T |
| `[N]T` | Fixed-size array |
| `[]T` | Slice |
| `proc(...) -> T` | Procedure type |

## Built-in Functions

| Function | Description |
|----------|-------------|
| `len(x)` | Length of array or string |
| `first(arr)` | First element of array |
| `last(arr)` | Last element of array |
| `rest(arr)` | All but first element |
| `push(arr, x)` | New array with x appended |
| `print(x)` | Print value to stdout |

## Examples

The `examples/` directory contains many sample programs:

| Example | Description |
|---------|-------------|
| `hello_world.frost` | Basic hello world |
| `variables.frost` | Variables and mutability |
| `constants.frost` | Compile-time constants |
| `functions.frost` | Functions and recursion |
| `procedures.frost` | Typed procedures |
| `structs.frost` | Struct definitions |
| `arrays.frost` | Array operations |
| `control_flow.frost` | If/else expressions |
| `loops.frost` | For loops with ranges |
| `closures.frost` | Closures and captures |
| `defer.frost` | Defer statements |
| `hashmaps.frost` | HashMap usage |
| `strings.frost` | String operations |
| `recursion.frost` | Recursive algorithms |
| `higher_order.frost` | Map, filter, reduce |
| `math.frost` | Math utilities |
| `algorithms.frost` | Common algorithms |
| `fizzbuzz.frost` | FizzBuzz |
| `primes.frost` | Prime numbers |
| `statistics.frost` | Basic statistics |

## Project Structure

```
frost/
├── src/
│   ├── lib.rs          # Library exports
│   ├── lexer.rs        # Tokenizer
│   ├── parser.rs       # AST parser
│   ├── compiler.rs     # Bytecode compiler
│   ├── typed_vm.rs     # Virtual machine
│   ├── typechecker.rs  # Type checking
│   ├── types.rs        # Type definitions
│   └── value.rs        # Runtime values
├── repl/
│   └── src/main.rs     # Interactive REPL
├── examples/           # Example programs
└── Cargo.toml
```

## Running Tests

```bash
cargo test
```

## Benchmarks

```bash
cargo run --release -p frost --bin benchmark
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
