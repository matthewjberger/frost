# Frost ❄️

Frost is a small programming language, made after working through ["Writing an Interpreter in Go"](https://interpreterbook.com/).

## REPL

```bash
cargo run --release
```

## Example
```bash
frost ❄️> let x = 5;
frost ❄️> let b = x + 5 * 3;
frost ❄️> b
20
frost ❄️> let add = fn(a, b, c, d) { return a + b + c + d };
frost ❄️> add(1, 2, 3, 4);
10
frost ❄️> let factorial = fn(n) { if (n == 0) { 1 } else { n * factorial(n - 1) } };
frost ❄️> factorial(5)
120
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
