# Generates the programs behind the scaling table in docs/self-hosting.md.
# Two curves: straight-line code, which measures the pipeline against lines,
# and generic instantiation, which measures it against specializations.
import os
import sys


def straight_line(count):
    out = ["printf :: extern fn(fmt: ^i8, value: i64) -> i32", ""]
    for index in range(count):
        out += [
            f"S{index} :: struct {{ a: i64, b: i64 }}",
            f"work{index} :: fn(x: i64, y: i64) -> i64 {{",
            "    mut acc : i64 = x",
            "    mut j : i64 = 0",
            "    while (j < 4) { acc = acc + y * j  j = j + 1 }",
            f"    s := S{index} {{ a = acc, b = y }}",
            "    s.a + s.b",
            "}",
            "",
        ]
    out += ["main :: fn() -> i64 {", "    mut total : i64 = 0"]
    out += [f"    total = total + work{index}({index}, 2)" for index in range(count)]
    out += ['    printf("%lld\n", total)', "    0", "}"]
    return "\n".join(out) + "\n"


def specializations(types, generics):
    out = ["printf :: extern fn(fmt: ^i8, value: i64) -> i32", ""]
    out += [f"T{t} :: struct {{ v: i64 }}" for t in range(types)]
    out += [f"g{g} :: fn(move x: $T) -> i64 {{ 1 }}" for g in range(generics)]
    out += ["main :: fn() -> i64 {", "    mut total : i64 = 0"]
    for t in range(types):
        for g in range(generics):
            out.append(f"    total = total + g{g}(T{t} {{ v = {t} }})")
    out += ['    printf("%lld\n", total)', "    0", "}"]
    return "\n".join(out) + "\n"


directory = sys.argv[1] if len(sys.argv) > 1 else "bench/generated"
os.makedirs(directory, exist_ok=True)
for count in (100, 400, 1600, 6400):
    path = os.path.join(directory, f"lines_{count}.frost")
    with open(path, "w") as handle:
        handle.write(straight_line(count))
    print(path)
for types, generics in ((40, 16), (80, 32), (160, 64)):
    path = os.path.join(directory, f"mono_{types * generics}.frost")
    with open(path, "w") as handle:
        handle.write(specializations(types, generics))
    print(path)
