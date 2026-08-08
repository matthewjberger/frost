# 11. Generics and compile-time specialization

## 11.1 Type parameters

A type parameter is written `$T`. It may appear on a function's parameters and on
a struct or enum declaration:

```frost
Pair  :: struct($T: Type) { first: T, second: T }
Option :: enum($T: Type) { None, Some { value: T } }
make_pair :: fn(a: $T, b: $T) -> Pair<T> { Pair { first = a, second = b } }
```

A parameter may name the type parameter inside a generic type's argument list,
where it binds to whatever that instance was made with:

```frost,sketch
first :: fn(a: Pair<$T>) -> T { a.first }
```

A caller that is itself generic hands its own argument straight on this way, so
`fn(a: Pair<$T>) -> i64 { width(first(a)) }` compiles `width` for the element
rather than for the pair.

A generic literal usually carries no arguments of its own, and which instance it
is comes from the context: an annotation, or the type of the parameter it is
passed to.

```
m : Option<i64> = Option::Some { value = 42 }     // the annotation names it
unwrap_or($i64, Option::None, 7)              // the parameter names it
```

Where there is no context to read it from, the literal says which instance it
is:

```
p := Pair<i64, bool> { first = 7, second = true }
```

Both forms name every field. There is no positional struct literal, generic or
otherwise.

In a parameter or struct type-parameter position, `$` IDENT `:` is followed by
the contextual word `Type` (or the keyword `type`). In a function's parameter
list it may instead be followed by a function signature, which declares a
compile-time function parameter (11.1b).

## 11.1a Value parameters

A parameter written `$N: usize` is a value parameter. It is a compile-time
integer, and its main use is sizing a fixed array:

```frost,sketch
Slab :: struct($T: Type, $N: usize) { storage: [N]T, used: i64 }
world : Slab<Entity, 4> = ...
```

An instantiation supplies an integer where a value parameter stands
(`Slab<Entity, 4>`), and monomorphization resolves `[N]T` to the concrete
`[4]Entity` for that instance. Value parameters are erased from the specialized
type the same way type parameters are.

A repeat literal takes one as its count, which is how a generic's backing array
is filled without naming a size:

```frost
filled :: fn($T: Type, $N: usize, value: $T) -> Buffer<T, N> {
    Buffer { items = [value; N], count = N }
}
```

A function takes them too, so an operation over a sized aggregate is written
once and serves every size:

```frost,sketch
slab_reset :: fn($T: Type, $N: usize, mut s: Slab<T, N>) {
    var i : i64 = 0
    while (i < N) { s.generations[i] = 0  i = i + 1 }
}

slab_reset($Entity, $4, world)
```

Inside the body the name stands for the integer wherever it appears, in a type
(`[N]T`) and in an expression (`i < N`) alike. `std/slab.frost` is a
generational pool written this way, generic over both element type and capacity,
and `examples/native/generic_slab.frost` is the same shape as a single file.

## 11.1b Compile-time function parameters

A parameter written `$f: Type` whose argument names a declared function is a
compile-time function parameter. The specialization calls it directly, with no
function pointer and no indirect call:

```frost,sketch
ascending :: fn(a: i64, b: i64) -> bool { a < b }

best :: fn($T: Type, $before: Type, move x: $T, move y: $T) -> $T {
    var result := x
    if (before(y, result)) { result = y }
    result
}

smallest := best($i64, $ascending, 7, 3)
```

Written `$f: Type` the parameter accepts a function of any signature, and a
mismatch surfaces inside the specialized body. Writing the signature instead
states what the argument has to be:

```frost,sketch
best :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T, move y: $T) -> $T
```

The bound is checked at the call, with that call's type arguments substituted
into it, so `T` in the bound means what it means at the call. A function whose
signature differs, or a type where a function is required, is an error reported
against the parameter list.

This is the only form of bound in the language, and it bounds one parameter kind
against one signature. There is no coherence rule, no orphan rule, and no
solving.

## 11.1c Compile-time argument lists

A parameter written `args: $...` takes every argument past the parameters
written before it. Each call has its own count and its own types, and the
specialization takes one ordinary parameter per element.

```frost,sketch
widths :: fn(args: $...) {
    for value in args {
        if (is_slice(value)) {
            print("{} bytes\n", slice_len(value))
        } else if (is_float(value)) {
            print("{} rounds to {}\n", value, cast($i64, value))
        } else {
            print("{}\n", value)
        }
    }
}

widths(1, 2.5, "three")
```

The list is always last.

A `for` over the list unrolls. The body is written once and compiled once per
element, with the loop's name standing for that element. There is no loop at run
time and no index. Every other `for` is the ordinary loop of chapter 7.

`list[K]` names the Kth element. The index has to be a literal, since which
element it is has to be known while the body is being expanded. An index past
what the call gave is an error against the call.

An `if` over a type predicate is decided at expansion time. Inside a
specialization, a condition from the 11.4a vocabulary asked of a parameter is
answered while the body is expanded, and the branch that cannot run is dropped
before anything checks it:

```frost
import "io.frost"

show :: fn(args: $...) {
    for value in args {
        if (is_slice(value)) {
            print("{} bytes\n", slice_len(value))
        } else {
            print("{}\n", value)
        }
    }
}
```

One body serves elements of different types.

Each element is evaluated once, however many times the unrolled body names it,
because the specialization takes it as an ordinary parameter and the call passes
it once.

An element may be a type. `f($Position, $Velocity)` gives the list two types. A
type element takes no parameter and is evaluated nowhere. It leaves behind a
name the body writes where a type belongs, so a `for` over the list may write
`sizeof(T)`, `[]T` and `T` as a generic argument. A list may hold both kinds.

Naming the list in an argument list hands over its elements. One generic passes
its list on to another this way:

```frost
passed_on :: fn(values: $...) -> i64 {
    total(values)
}
```

`g(T) for T in list` in an argument list is one argument per element, with the
element's name standing for it. The call's arity is the list's length, and an
argument list is the only place a list may be written this way:

```frost,sketch
for_each :: fn($body: Type, mut world: World, f: Filters, types: $...) {
    ...
    body(query_column($T, world, q, component_of($T, world))
        for T in types, q.count)
}
```

## 11.1c.0 `format`, a literal counted against the list

A parameter written `format name: str` must be given a string literal, and the
holes that literal opens are counted against the compile-time list declared
after it. `std/io.frost` is the one declaration that uses it:

```frost,sketch
print :: fn(format fmt: str, args: $...)
```

A `{}` in the literal opens a hole, `{{` and `}}` stand for one brace each, and
a `{` that does neither is a fault. Four things are refused where the call is
written:

| Written | Refused because |
| --- | --- |
| `print("{} of {}\n", 1)` | Two holes, one value |
| `print("{}\n", 1, 2)` | One hole, two values |
| `print("{ x }\n", 1)` | A `{` that opens neither a hole nor a brace |
| `print(chosen, 1)` | The count comes from the literal, so it has to be one |

The function that has the parameter decides what a hole accepts: `print` takes
a number, a `bool` or a `str`, and refuses anything else with the type it was
given.

The word is contextual, the same way `value` and `mut` are. A parameter named
`format` still parses as a parameter named `format`, since the word only takes
effect when a name follows it.

## 11.1c.1 `type_id`

`type_id(T)` is a number the build gives that type: the same wherever the type
is written and different for every other type. It has no meaning outside the
build, and nothing is promised about which number a type gets.

It keys a table by type in a program whose contents are decided while it runs.
`std/ecs.frost` registers a component under a type and is given an index in
return, and `type_id` lets a query later name the component by writing the type.

Expansion has no recursion, no unbounded loop, and nothing that reads the world.
Every construct here iterates a list whose length is known once the generic is
instantiated, so the program's own text bounds the cost of expansion.

A literal is read where a `format` parameter takes one, and a constant or a
length may be a call the build runs early. Both are bounded: the reader counts
the holes in one literal, and the evaluator runs a fixed number of steps to a
depth it will not exceed. Neither hands the program a string it computed.

## 11.1d Walking a type's fields

A `for` over `fields(T)` is decided at expansion time, the same as a `for` over
a compile-time list. The body is written once and compiled once per field of
`T`, with the loop's name standing for that field:

```frost,sketch
Vertex :: struct { position: Vec3, normal: Vec3, uv: Vec2, id: i64 }

describe :: fn($T: Type, mut out: []Attribute) -> i64 {
    var index : i64 = 0
    for field in fields(T) {
        out[index] = Attribute {
            offset = offset_of(field),
            size = sizeof(field),
            floating = is_float(field),
        }
        index = index + 1
    }
    index
}
```

A field is not a value. It is asked about, and this is the whole of what may be
asked:

| | |
| --- | --- |
| `offset_of(field)` | where it sits in the type that declares it |
| `sizeof(field)` | how wide what it holds is |
| the 11.4a predicates | what kind of type it holds |
| `field_count(T)` | how many fields a type has, which sizes a table |

Every one of those is a number the compiler worked out to lay the type out.
Naming a field anywhere else is an error.

There is no reflection by name. `has_field(T, "position")` is the string-keyed
predicate 11.4a rules out, and a field's name is not readable at all.

The bound is the same one 11.1c holds: the list a `for` walks is the struct's
own field list, so its length is fixed by a declaration. No recursion, no
unbounded loop, and nothing that reads the world.

## 11.2 Monomorphization

Generics specialize at compile time. Each concrete instantiation compiles to its
own code, with no runtime dispatch and no boxing. Type parameters are erased
from the specialized ABI once monomorphization chooses concrete types.

## 11.3 Explicit type arguments

There is no turbofish. A type is passed as an ordinary argument by writing `$`
before it, which forms a type value:

```frost,sketch
stride :: fn($T: Type, count: i64) -> i64 { count * sizeof(T) }
bytes := stride($Entity, 16)
```

## 11.4 Nested generic arguments

Generic arguments are delimited by `<` and `>`. Because `>>` lexes as one shift
token, the parser splits it when it closes two nested argument lists, so
`Pair<Pair<i64>>` parses correctly. This splitting is wired into the `Handle<T>`
and `Name<...>` type forms.

## 11.4a Bounds

A generic may say what it needs of its compile-time parameters, with a `where`
clause after the signature:

```frost
twice :: fn($T: Type, v: $T) -> T where is_numeric(T) { v + v }
first :: fn($T: Type, xs: []T) -> T where is_numeric(T) && !is_pointer(T) {
    xs[0]
}
```

The bound is read at each call, with that call's arguments in hand, so a type
that cannot work is refused against the line the caller wrote.

The vocabulary is fixed and closed:

| bound | holds for |
| --- | --- |
| `is_numeric(T)` | an integer or a float |
| `is_integer(T)` | an integer of any width, signed or unsigned |
| `is_float(T)` | `f32` or `f64` |
| `is_struct(T)` | a struct or an enum |
| `is_array(T)` | a fixed array `[N]T` |
| `is_slice(T)` | a slice `[]T`, which includes `str` |
| `is_pointer(T)` | a raw pointer or a borrow |
| `is_linear(T)` | a resource: a type that must be consumed exactly once |

`vec_set` writes into a slot while whatever was there goes unconsumed, so it is
declared `where !is_linear(T)`, and a `Vec<File>` is refused at the call the
reader wrote. A resource element is reached through `vec_slice`, where
`ref e := vec_slice(v)[i]` stays a borrow.

Terms combine with `&&`, `||` and `!`. A distinct type answers as what it is
represented by.

There is no bound keyed by a name, such as asking whether a type has a field
called `position`.

A bound answers what a type is. Capability bundles (11.4b) say what can be done
with it.

## 11.4b Capability bundles

A capability bundle is a generic struct whose fields are functions. It says what
can be done with a type:

```frost
Ordering :: struct($T: Type) {
    less: fn(T, T) -> bool,
    equal: fn(T, T) -> bool,
}
```

An implementation is a constant of it, and a constant is its value wherever it
is named:

```frost
i64_less  :: fn(a: i64, b: i64) -> bool { a < b }
i64_equal :: fn(a: i64, b: i64) -> bool { a == b }

i64_ascending :: Ordering<i64> { less = i64_less, equal = i64_equal }
```

A generic that needs the operations takes the bundle as a compile-time
argument, and the call names which one it means:

```frost,sketch
sort :: fn($T: Type, $ops: Ordering<T>, mut items: []T) {
    ...
    if (ops.less(items[j], items[j - 1])) { ... }
}

sort($i64, $i64_ascending, view)
```

Because `$ops` is a compile-time argument, `ops.less(a, b)` folds to a direct
call to `i64_less`. The specialization holds no function pointer, loads nothing,
and dispatches on nothing.

Dropping the `$` gives the runtime form from the same declaration:

```frost,sketch
sort_at_runtime :: fn(ops: Ordering<i64>, mut items: []i64) { ... }
```

In that form `ops` is an ordinary value: it can be chosen while the program
runs, stored in an array, or swapped, and the calls go through the pointers it
holds. There is no separate feature and no second spelling of the bundle type.

Two orderings over one type are two constants:

```frost
i64_descending :: Ordering<i64> { less = i64_greater, equal = i64_equal }
```

Composition is a struct with struct fields, and the body reads
`ops.ordering.less(a, b)`.

`std/ordering.frost` and `std/sort.frost` are this written out.

The declared type is checked at the call: an argument that is a constant of
another type, or a name that is not a constant at all, is refused against the
line the caller wrote.

## 11.5 No traits

A bound is not a trait. Nothing registers into it, nothing implements it, there
is no set to name, and there is nothing to resolve, so there is no coherence
rule, no orphan rule, and no method lookup. There are no associated types, no
trait objects, and no dynamic dispatch. A generic body type-checks once
specialized.

A capability bundle (11.4b) stands in its place. A bundle's implementation is a
constant, named at the call: `i64_ascending` greps to one definition, and a
program that wants a second ordering writes a second constant.

For a single operation there is no need for a bundle at all. A compile-time
function parameter (11.1b) passes one function and keeps the call direct.

## 11.6 Modules and imports

A module is a file. `import "x.frost"` splices that file's declarations into the
program, and a file's `export` line is the complete set of names another file
can use from it. Everything else is private and mangled so it cannot collide.

An import is looked for beside the importing file first, then in directories
given with `-L`, then in `FROST_PATH`, then in those a `frost.json` beside the
entry file declares, then in the standard library. A module's identity is its
path relative to whichever of those it was found under, and private symbol names
and the build cache are keyed on it. See
[modules.md](../impl/modules.md).
