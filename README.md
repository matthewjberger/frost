# Frost

Frost is a small programming language, made after working through ["Writing an Interpreter in Go"](https://interpreterbook.com/).

## REPL

```bash
cargo run --release
```

## Example
```bash
let add = fn(a, b, c, d) { return a + b + c + d };
frost ❄️> add(1, 2, 3, 4);
10
frost ❄️> let addThree = fn(x) { return x + 3 };
frost ❄️> addThree(3);
6
frost ❄️> let max = fn(x, y) { if (x > y) { x } else { y } };
frost ❄️> max(5, 10)
10
frost ❄️> let factorial = fn(n) { if (n == 0) { 1 } else { n * factorial(n - 1) } };
frost ❄️> factorial(5)
120
```

## License
[MIT](https://choosealicense.com/licenses/mit/)