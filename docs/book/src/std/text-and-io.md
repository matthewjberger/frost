# Text, files, and JSON

Five modules that between them are enough to read a file, pick it apart, build
an answer and write it back out. `tools/wgpu_bindgen.frost` uses four of the
five and nothing else, which is the shortest demonstration that they add up.

They all agree on one thing: a `str` is a pointer and a length, never a
NUL-terminated buffer. A writer taking a `^i8` promises the bytes end in a NUL,
and that is a promise the caller makes silently and the compiler cannot check. A
`str` carries its length, so there is nothing to promise. Where C insists on a
terminator, the copy is made inside the module, once, rather than by every
caller.

## `std/io.frost`, writing to standard output

```frost
export print_int, print_char, print_str, print_bool, print_f64,
    print_int_line, print_str_line, print_bool_line, print_f64_line
```

| Call | What it writes |
| --- | --- |
| `print_int(value)` | A signed integer in base ten |
| `print_char(byte)` | One byte |
| `print_str(text)` | A `str`, in a single call |
| `print_f64(value)` | A float, the way C writes `%g` |
| `print_bool(value)` | `1` or `0`, the number a mask or a flag would print |
| `print_int_line(value)` | An integer and a newline |
| `print_str_line(text)` | A `str` and a newline |
| `print_f64_line(value)` | A float and a newline |
| `print_bool_line(value)` | `1` or `0` and a newline |

This module is the whole of how a program writes output: there is no print
statement in the language. Without it a program declares the C function it
wants itself, which is why so many older examples open with
`printf :: extern fn(fmt: ^i8, value: i64) -> i32`. That is honest about the
FFI, but it is a strange first line for a first program, and it drags in
`printf`'s format string, which is one more thing to get wrong when what was
wanted was to print a number.

These go through the runtime's write helpers, which are pinned to standard
output rather than following the compiler's emit target, so a program that
redirects emitted text still prints where a reader looks. There is no format
string and nothing is variadic. Three of the four externs behind them are
`safe extern`, because each takes a number and there is nothing a caller can
hand one that misbehaves. The byte writer takes a pointer and a length, so its
one call sits in an `unsafe` block inside `print_str`, where the length comes
from the same `str` as the pointer and the two cannot disagree.

```frost
import "io.frost"

main :: fn() -> i64 {
    print_str_line("hello")
    print_int_line(42)
    0
}
```

A line built from several values is several calls, with the newline written by
the one `_line` call at the end:

```frost
print_str("hp ")
print_int(entity.hp)
print_str(" of ")
print_int_line(entity.max)
```

## `std/strings.frost`, questions about text

Walks over `str`. `str_len` and `s[i]` are the only primitives, and everything
here is ordinary Frost over them, so every one of these is a bounds-checked walk
with no allocation anywhere.

| Call | What it answers |
| --- | --- |
| `str_is_empty(s) -> bool` | Whether the length is zero |
| `str_eq(a, b) -> bool` | Whether the two hold the same bytes |
| `str_starts_with(s, prefix) -> bool` | Whether `s` begins with it |
| `str_ends_with(s, suffix) -> bool` | Whether `s` ends with it |
| `str_index_of(s, needle) -> i64` | The first occurrence, or -1 |
| `str_contains(s, needle) -> bool` | Whether it occurs at all |
| `str_count(s, byte) -> i64` | How many times that byte appears |
| `str_to_i64(s) -> i64` | A decimal integer, with an optional leading `-` |
| `str_to_f64(s) -> f64` | A decimal number, fraction and `e` exponent included |
| `str_byte_is_digit(byte) -> bool` | Whether the byte is `0` through `9` |
| `str_byte_is_space(byte) -> bool` | Space, tab, newline or carriage return |

Nothing here returns a new string. Building one needs storage, and this module
does not get to decide where a caller's storage comes from. What it offers is
answers about strings, and building goes through a `Builder` or an arena the
caller owns.

`str_index_of` answers an out-of-band -1 rather than an optional, because a
caller almost always compares it to -1 immediately, and an enum here would cost
a match at every call for nothing.

`str_to_i64` answers 0 for anything it cannot read, which is what C's `atoi`
does. It does not distinguish `"0"` from "not a number". A caller that needs the
difference checks the bytes first, which is what `str_byte_is_digit` is for.
`str_to_f64` answers 0.0 on the same terms.

`str_to_f64` reads the digits into one integer and turns where the point sat
into a power of ten, so a number costs one scaling rather than a rounding per
digit. Where the integer is under 2^53 and the power is one of the twenty-three
a double holds exactly, both sides are exact and the answer is the nearest
double to what was written, which is what lets its tests compare with `==`
rather than a tolerance. Past that the scaling goes in steps and the last place
can differ.

## `std/format.frost`, the builder

A `Builder` is a `Vec<u8>` under a name that says what it is for. Everything
here appends to one, so a caller assembles a line from parts and prints or
writes it once rather than one `print_int` at a time.

```frost
Builder :: struct { bytes: Vec<u8> }
```

| Call | What it does |
| --- | --- |
| `builder_new(capacity) -> Builder` | An empty buffer with room reserved |
| `builder_free(move b)` | Releases the storage |
| `builder_len(b) -> i64` | How many bytes have been written |
| `builder_str(b) -> str` | The bytes so far, as a `str` |
| `builder_byte(mut b, byte)` | Appends one byte |
| `builder_str_value(mut b, text)` | Appends a `str`, byte for byte |
| `builder_int(mut b, value)` | Appends a signed integer in base ten |
| `builder_uint(mut b, value)` | Appends a non-negative integer's digits |
| `builder_clear(mut b)` | Forgets the bytes, keeps the storage |

```frost
mut out := builder_new(256)
builder_str_value(out, "frames: ")
builder_int(out, count)
builder_byte(out, 10)
print_str(builder_str(out))
builder_free(out)
```

`builder_str` views the builder's storage rather than copying it, so it is valid
only until the next append, which is the usual rule for a view into a growable
thing.

`builder_int` produces its digits least significant first into a twenty-byte
stack array and then appends them in order, which is how it avoids needing to
know the length before it starts. Twenty is enough for any `i64`, and the sign
is written before the digits by `builder_int` rather than by `builder_uint`.

`builder_clear` keeps the block, which is what makes a builder worth hoisting
out of a loop: one allocation and a clear per iteration rather than an
allocation per iteration.

## `std/fs.frost`, whole files

```frost
ReadResult :: linear struct {
    text: str,
    buffer: ^u8,
    ok: bool,
}
```

| Call | What it does |
| --- | --- |
| `fs_read(path) -> ReadResult` | Reads the whole file into a fresh heap block |
| `fs_free(move result)` | Releases that block. Consumes the result |
| `fs_write(path, text) -> bool` | Writes the whole file. Answers whether it worked |
| `fs_exists(path) -> bool` | Whether the path is there |

A read owns the buffer its text borrows, so `ReadResult` is `linear` and the
compiler refuses a read whose buffer nothing frees. `ok` is false when the file
could not be opened, in which case `text` is empty. The `text` is a view into
`buffer`, so nothing may use it after `fs_free`.

```frost
read := fs_read("webgpu.json")
if (read.ok == false) {
    fs_free(read)
    return 1
}
... read.text ...
fs_free(read)
```

A path is a `str` like any other text. C reads one until a NUL and a `str` has
none, so `fs.frost` makes the terminated copy itself, in one private function
all four calls go through, rather than taking a `^i8` and trusting every caller.

## `std/json.frost`, a JSON reader

The document is parsed into one flat array of nodes and every reference is an
index into it. A node never owns another, so nothing here is recursive except
the walk, and the whole document is freed by freeing one vector.

```frost
JsonKind :: enum { Null, True, False, Number, String, Array, Object }
```

| Call | What it does |
| --- | --- |
| `json_parse(source) -> Json` | Parses a whole document |
| `json_free(move document)` | Releases the node vector |
| `json_root(document) -> i64` | The root node, or -1 if the parse failed |
| `json_kind(mut document, node) -> JsonKind` | What the node is |
| `json_number(mut document, node) -> i64` | A number node's integer part |
| `json_real(mut document, node) -> f64` | The whole of a number node |
| `json_is_null(mut document, node) -> bool` | Whether it is the null node |
| `json_text_eq(mut document, node, text) -> bool` | Whether a string node holds exactly this text |
| `json_text_off(mut document, node) -> i64` | Where a string node's bytes start in the source |
| `json_text_len(mut document, node) -> i64` | How many bytes they are |
| `json_member(mut document, object, name) -> i64` | An object member's value, or -1 |
| `json_at(mut document, container, index) -> i64` | The nth element of an array or object |
| `json_count(mut document, container) -> i64` | How many children it has |

`JsonNode` is exported too, so a caller can hold one, but the accessors above
are how a reader normally works. Each takes an index and answers a safe value
for a negative one, so a walk through a document that turns out not to have the
member it wanted returns -1 all the way up rather than aborting partway.

Strings are not copied. A string node records where it sits in the source, so
`json_text_eq` compares against bytes already in memory and the parse allocates
nothing per string. That is also why the accessors take offsets and lengths
rather than handing out a `str`: the node names a span of a buffer the caller
still owns.

A number is read twice over. `json_number` answers with the integer part,
which is what a count or a size is, and `json_real` answers with the whole of
it, which is what a measurement is. A number node records where its digits sit
the same way a string node does, so the second of those is `str_to_f64` over
bytes already in memory and the parse still copies nothing per number.

Children are chained with a `first` and a `next` index the way a linked list is,
so an array of a thousand elements costs a thousand nodes and no reallocation of
the parent. `json_at` walks that chain, so reading an array element by element
with `json_at` is quadratic. Reading it by following `next` is not, and that is
what the node fields are exported for.

## Where they add up

`tools/wgpu_bindgen.frost` reads `lib/renderer/wgpu/webgpu.json` with
`fs_read`, walks it with `json_member` and `json_at`, classifies bytes with
`str_byte_is_digit` and friends, assembles the whole generated module in two
`Builder`s, and writes it with `fs_write`. That is a code generator in eleven
hundred lines, importing four of these five modules and nothing else. See
[graphics.md](graphics.md).

## Tests

`std/strings.frost` has six test blocks:

```bash
frost --test std/strings.frost
```

`io.frost`, `format.frost`, `fs.frost` and `json.frost` have none of their own.
They are covered where they are used: the bindgen exercises the last three end
to end on a 119 KB document every time `just bindgen` runs, and a difference in
any of them shows up as a generated file that does not compile.
