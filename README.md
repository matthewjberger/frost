# Frost

A statically-typed programming language with Rust-inspired ownership and immutability. Frost compiles to bytecode for rapid development or native code via Cranelift for production.

## Features

- **Immutable by default** - Variables are immutable unless declared with `mut`
- **Clean syntax** - `:=` for inference, `::` for constants, `fn` for functions
- **Ownership and borrowing** - Memory safety without garbage collection, enforced by a pass that runs before compilation
- **Second-class references** - References can't be stored in structs/enums or returned, eliminating lifetime annotations
- **Move checking** - Use-after-move of a non-`Copy` value is a compile error
- **Linear resources** - A `linear` struct or enum must be consumed exactly once, replacing `Drop` with a tracked obligation
- **Generational handles and pools** - Long-lived data lives in a pool and is referenced by a `Handle<T>`; `pool[handle]` is a place whose borrow is second-class, and a stale handle can never read a reused slot
- **Generics via monomorphization** - Generic functions *and* generic structs specialize at compile time, plus a native `sizeof`; the pool typed surface is an ordinary Frost library
- **Enums with data** - Tagged unions with pattern matching via `match`, including tuple patterns
- **Typed IR with two native backends** - The AST lowers to a typed IR that both a Cranelift backend and a portable C backend emit from; a differential test checks the two agree
- **Structs** - User-defined data types with field access, by-value passing and returning, and references
- **Calls C directly** - `extern fn` links against libc and any C library on both backends

## Documentation

- [docs/philosophy.md](docs/philosophy.md) — design philosophy, goals and
  non-goals, and why Frost is data-oriented rather than object-oriented
- [docs/memory-safety.md](docs/memory-safety.md) — how Frost guarantees memory
  safety without a garbage collector or lifetime annotations
- [docs/c-compatibility.md](docs/c-compatibility.md) — calling C and the C backend
- [docs/architecture.md](docs/architecture.md) — the compiler pipeline and what
  the native backend supports today versus the bytecode VM
- [docs/memory_model.md](docs/memory_model.md) — copy / move / linear type categories
- [docs/spec.md](docs/spec.md) — the detailed language reference

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

### Run Examples (Bytecode VM)

```bash
cargo run --release -p frost --bin frost -- examples/hello_world.frost
cargo run --release -p frost --bin frost -- examples/fizzbuzz.frost
cargo run --release -p frost --bin frost -- examples/recursion.frost
```

### Native Compilation (Cranelift)

```bash
# Compile to object file
cargo run --release -p frost --bin frost -- --native -o program.o program.frost

# Compile and link to executable
cargo run --release -p frost --bin frost -- --link -o program program.frost

# Run the native executable
./program
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
// Compile-time constants (::)
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

### Typed Functions

```rust
// Functions with type annotations
add :: fn(a: i64, b: i64) -> i64 {
    a + b
}

is_even :: fn(n: i64) -> bool {
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

### Enums

```rust
// Simple enum
Color :: enum {
    Red,
    Green,
    Blue,
}

color := Color::Red

// Tagged union (enum with data)
Result :: enum {
    Ok { value: i64 },
    Err { code: i64, message: str },
}

success := Result::Ok { value = 42 }
failure := Result::Err { code = 404, message = "Not found" }
```

### Match (Pattern Matching)

```rust
// Match on enum variants
msg := match result {
    case .Ok { value }: value
    case .Err { code }: 0 - code
}

// Match on values
grade := match score {
    case 90: "A"
    case 80: "B"
    case _: "C"
}

// Tuple patterns
result := match (x % 3, x % 5) {
    case (0, 0): "FizzBuzz"
    case (0, _): "Fizz"
    case (_, 0): "Buzz"
    case (_, _): x
}
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

### References

```rust
// Immutable reference
read_point :: fn(p: &Point) -> i64 {
    p.x + p.y
}

// Mutable reference
increment :: fn(x: &mut i64) {
    x^ = x^ + 1
}

mut value := 10
increment(&mut value)
print(value)  // 11
```

References are second-class citizens: they cannot be stored in structs, returned from functions, or stored in arrays. This eliminates the need for lifetime annotations.

### Linear Resources

A struct or enum declared `linear` is a resource that must be consumed exactly
once. It is a compile error to let a linear value go out of scope without
consuming it, and (as with any move type) to use it after it has been moved.

```rust
File :: linear struct { handle: i64 }

open :: fn(path: str) -> File { ... }

// The terminal consumer takes ownership across the FFI boundary.
close :: extern fn(f: File)

use_file :: fn() {
    f := open("data.txt")
    // ... use f ...
    close(f)              // consumes f; forgetting this line is an error
    // close(f)           // ERROR: use of moved value 'f'
}
```

Consuming a linear value means moving it onward: returning it, or passing it by
value to another function. This replaces `Drop` with an obligation the compiler
tracks, so cleanup can never be silently forgotten or run twice.

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
| `i8`, `i16`, `i32`, `i64` | Signed integers |
| `u8`, `u16`, `u32`, `u64` | Unsigned integers |
| `f32`, `f64` | Floating point |
| `bool` | Boolean (`true`/`false`) |
| `str` | String |
| `^T` | Pointer to T |
| `&T` | Immutable reference |
| `&mut T` | Mutable reference |
| `[N]T` | Fixed-size array |
| `[]T` | Slice |
| `fn(...) -> T` | Function type |

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
| `procedures.frost` | Typed functions |
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
│   ├── typechecker.rs  # Type checking (being reworked onto the IR)
│   ├── ir.rs           # Typed IR definitions
│   ├── ir_build.rs     # AST -> typed IR lowering
│   ├── ir_codegen.rs   # Typed IR -> Cranelift native backend
│   ├── types.rs        # Type definitions
│   └── value.rs        # Runtime values
├── repl/
│   └── src/main.rs     # Interactive REPL
├── bootstrap/          # Self-hosted compiler (Frost in Frost)
├── examples/           # Example programs
├── docs/               # Documentation
└── Cargo.toml
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the authoritative,
up-to-date description of the compiler, including the typed IR and the exact
subset the native backend supports today.

Frost has a **dual-backend architecture** designed for different use cases:

```
Source Code (.frost)
       │
       ▼
┌─────────────┐
│   Lexer     │  lexer.rs
└─────────────┘
       │ tokens
       ▼
┌─────────────┐
│   Parser    │  parser.rs
└─────────────┘
       │ AST
       ▼
┌─────────────┐
│ Typechecker │  typechecker.rs
└─────────────┘
       │ AST
       ▼
   ┌───┴───┐
   │       │
   ▼       ▼
┌──────┐ ┌──────────┐
│  VM  │ │ Cranelift│
│Backend │ │ Backend  │
└──────┘ └──────────┘
   │           │
   ▼           ▼
Bytecode    Native
Execution   Binary
```

### Two Backends

1. **Bytecode VM** (`compiler.rs` → `typed_vm.rs`)
   - Compiles Frost source to custom bytecode
   - Executed by an interpreter written in Rust
   - Used for: REPL, rapid iteration, debugging

2. **Native Codegen** (`ir_build.rs` + `ir_codegen.rs` via Cranelift)
   - Lowers the AST to a typed IR, then emits machine code with Cranelift
   - Produces `.o` object files or linked executables
   - Supports a growing subset (scalars, control flow, functions, casts,
     `extern` C interop, references and pointers); anything outside that
     subset fails loudly rather than miscompiling. See
     [docs/architecture.md](docs/architecture.md).

### Why Both?

| Aspect | Bytecode VM | Native |
|--------|-------------|--------|
| Compile time | ~instant | slower |
| Runtime speed | interpreted | fast |
| Debugging | easier | harder |
| Hot reload | possible | requires recompile |
| Distribution | needs runtime | standalone binary |

This is similar to how Java has interpreter + JIT, or how Lua has bytecode VM + LuaJIT.

For game development, you'd use the VM during development for fast iteration, then compile native for shipping.

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
