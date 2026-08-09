# Text, files, and JSON

Five modules that between them are enough to read a file, pick it apart, build
an answer and write it back out. `tools/wgpu_bindgen.frost` uses all five and
nothing else.

They all agree on one thing: a `str` is a pointer and a length. A writer taking
a `^i8` relies on the bytes ending in a NUL, a promise the caller makes silently
and the compiler cannot check. A `str` carries its length, so there is nothing
to promise. Where C insists on a terminator, the module makes the terminated
copy inside itself, once, on behalf of every caller.

## `std/io.frost`, writing to standard output

```frost
export to_stdout, print, write, LINE
```

Frost has no print statement, so this module is how a program writes output. The
other route is to declare the C function you want yourself, which is why so many
examples open with `printf :: extern fn(fmt: ^i8, value: i64) -> i32`. That line
is honest about the FFI, and it drags in `printf`'s format string, which is one
more thing to get wrong when what was wanted was to print a number.

```frost
import "io.frost"

main :: fn() -> i64 {
    print("hello\n")
    print("hp {} of {}\n", 12, 20)
    0
}
```

`print` takes a format string and as many values as the line names. It writes
the literal out, with each `{}` replaced by the next value:

| Written | Means |
| --- | --- |
| `{}` | The next value |
| `{{` | One `{` |
| `}}` | One `}` |
| `}` on its own | One `}` |

A value may be a signed or unsigned integer of any width, a float of either
width, a `bool`, or a `str`. An integer is written in base ten, a float the way
C writes `%g`, and a `bool` as `1` or `0`, the number a mask or a flag would
print. Nothing is appended. A line ends with the `\n` you write into the
literal.

There are no specifiers. `{}` says where a value goes, and the value's type says
how it is written.

### What the compiler checks, and what runs

`format` on the parameter settles the count where the call is written:

```frost,sketch
print :: fn(format fmt: str, args: $...)
```

A `format` parameter must be given a string literal, and the holes that literal
opens are counted against the compile-time list that follows it. A call that
names more holes than it gives values, gives more values than it names holes,
opens a hole it never closes, or hands over a value no writer takes, is refused
at the line the call is written on:

```
this format string opens 2 hole(s) and the call gives 1 value(s)
a '{' in a format string opens a hole or stands for one brace, so write '{}' or '{{'
a format string is written as a literal, since how many values follow it is settled where the call is written
a format string writes a number, a yes or no, or a str, and this is a Point
```

`args: $...` is an ordinary [compile-time list](../reference/generics.md), so
the body walks it with `for` and asks each value's type what it is. Those
questions are answered while the body is expanded and the arms that lose are
deleted, so a call compiles to one direct write per value with the choice
already made. Nothing is dispatched at run time and nothing is boxed.

At run time the literal's own bytes are still walked, looking for the next hole.

### One line, one write

The line is composed before any of it leaves. `print("a {} b {}\n", x, y)` is
one write, not five: the digits, the float and the runs of literal text all land
in a buffer on the stack, and that buffer goes out once.

Nothing is held back after the call returns. There is no buffer between calls,
so `print` never needs a `flush`, a program that stops partway keeps everything
it had already printed, and `print` always arrives in order with the compiler's
own emitted text. To batch output, name a destination that batches it.

A line longer than the buffer leaves in pieces of that size. Nothing is lost and
nothing allocates.

### Where the bytes go

A destination is a function taking bytes.

```frost,sketch
to_file :: fn(text: str) { ... }
write(to_file, "frame {} took {}ms\n", index, elapsed)
```

`print` is `write` to `to_stdout`. A destination is called once per formatted
line, so it always sees a whole line and what it does with one is its own
business.

`to_stdout` goes through the runtime's byte writer, which always writes to
standard output whatever the compiler's emit target is, so a program that
redirects emitted text still prints where a reader looks. That writer takes a
pointer and a length, so its one call sits in an `unsafe` block, where the
length comes from the same `str` as the pointer and the two cannot disagree.

Integers are written by `std/io.frost` itself. A float is the one thing it asks
the C library for, through a runtime helper that spells `%g` into a buffer, so
the line is still finished before anything leaves.

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
| `str_slice(s, from, count) -> str` | A view of `count` bytes from `from` |
| `str_span(s, from, count) -> str` | The same, stopping at the end |

Every call here answers a question about text you already have. Building a new
string needs storage, and this module leaves the choice of storage to the
caller, so building goes through a `Builder` or an arena the caller owns.

`str_index_of` answers an out-of-band -1, so a caller compares the result to -1
where it stands. An enum here would cost a match at every call.

`str_to_i64` answers 0 for anything it cannot read, the way C's `atoi` does. It
reads `"0"` and "not a number" the same way. A caller that needs the difference
checks the bytes first with `str_byte_is_digit`. `str_to_f64` answers 0.0 on the
same terms.

`str_to_f64` reads the digits into one integer and turns where the point sat
into a power of ten, so a number costs one scaling and not a rounding per digit.
Where the integer is under 2^53 and the power is one of the twenty-three a
double holds exactly, both sides are exact and the answer is the nearest double
to what was written, so its tests compare with `==` and need no tolerance. Past
that the scaling goes in steps and the last place can differ.

## `std/format.frost`, the builder

A `Builder` is a `Vec<u8>` under a name that says what it is for. Everything
here appends to one, so a caller assembles a line from its parts and prints or
writes it once.

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

```frost,sketch
mut out := builder_new(256)
builder_str_value(out, "frames: ")
builder_int(out, count)
builder_byte(out, 10)
print("{}", builder_str(out))
builder_free(out)
```

`builder_str` views the builder's storage, so it is valid until the next append,
which is the usual rule for a view into a growable thing.

`builder_int` produces its digits least significant first into a twenty-byte
stack array and then appends them in order, so it never needs the length before
it starts. Twenty is enough for any `i64`. `builder_int` writes the sign before
the digits, and `builder_uint` writes digits alone.

`builder_clear` keeps the block, so a builder hoisted out of a loop costs one
allocation and a clear per iteration.

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
| `fs_remove(path) -> bool` | Deletes it. Answers whether it worked |

A read owns the buffer its text borrows, so `ReadResult` is `linear` and the
compiler refuses a read whose buffer nothing frees. `ok` is false when the file
could not be opened, in which case `text` is empty. The `text` is a view into
`buffer`, so nothing may use it after `fs_free`.

```frost,sketch
read := fs_read("webgpu.json")
if (read.ok == false) {
    fs_free(read)
    return 1
}
... read.text ...
fs_free(read)
```

A path is a `str` like any other text. C reads a path until a NUL, so
`fs.frost` makes the terminated copy itself, in one private function all four
calls go through.

## `std/json.frost`, a JSON reader

The document is parsed into one flat array of nodes and every reference is an
index into it. A node never owns another, so the walk is the only recursive part
and the whole document is freed by freeing one vector.

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

`JsonNode` is exported too, so a caller can hold one, though a reader normally
works through the accessors above. Each takes an index and answers a safe value
for a negative one, so a walk through a document missing the member it wanted
answers -1 all the way up and finishes.

A string node records where it sits in the source, so `json_text_eq` compares
against bytes already in memory and the parse allocates nothing per string. The
accessors take offsets and lengths for the same reason: the node names a span of
a buffer the caller still owns.

A number is read twice over. `json_number` answers with the integer part, for a
count or a size, and `json_real` answers with the whole of it, for a
measurement. A number node records where its digits sit the same way a string
node does, so the second of those is `str_to_f64` over bytes already in memory
and the parse still copies nothing per number.

Children are chained with a `first` and a `next` index the way a linked list is,
so an array of a thousand elements costs a thousand nodes and no reallocation of
the parent. `json_at` walks that chain, so reading an array element by element
with `json_at` is quadratic. Following `next` yourself reads it in linear time,
and the node fields are exported for that walk.

## The five together

`tools/wgpu_bindgen.frost` reads `lib/renderer/wgpu/webgpu.json` with
`fs_read`, walks it with `json_member` and `json_at`, classifies bytes with
`str_byte_is_digit` and friends, assembles the whole generated module in three
`Builder`s, and writes it with `fs_write`. That is a code generator in fourteen
hundred lines, importing these five modules and nothing else. See
[graphics.md](graphics.md).

## Tests

`std/strings.frost` has twelve test blocks and `std/fs.frost` two:

```bash
frost --test std/strings.frost
```

`io.frost`, `format.frost` and `json.frost` are covered where they are used. The
bindgen exercises those three and `fs.frost` end to end on a 182 KB document
every time `just bindgen` runs, and a difference in any of them shows up as a
generated file that does not compile.
