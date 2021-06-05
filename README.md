# Frost ❄️

Frost is a small programming language, made after working through ["Writing an Interpreter in Go"](https://interpreterbook.com/).

## REPL

```bash
cargo run --release
```

## Example

```bash
> let x = 5;
> let b = x + 5 * 3;
> b
20
> let add = fn(a, b, c, d) { return a + b + c + d };
> add(1, 2, 3, 4);
10
> let factorial = fn(n) { if (n == 0) { 1 } else { n * factorial(n - 1) } };
> factorial(5)
120
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
