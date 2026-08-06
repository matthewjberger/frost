# The standard library

`std/` is twenty files of Frost, compiled the way your own modules are
compiled. Nothing in it is a compiler intrinsic a program does not also get:
`Vec` and `Map` are structs over a heap slice, `Option` is a generic enum, and
`Slab` and `columns` use the handle indexing of section 10.2 of
[handles-and-pools.md](../reference/handles-and-pools.md), which any struct of
the same shape gets on the same terms.

Where the library reaches outside the language it does it the way a program
would, with an `extern` declaration. `mem.frost` declares the C allocator,
`io.frost` the runtime's write helpers, `fs.frost` its file calls,
`thread.frost` its threading ones, and `math.frost` and `math64.frost` the C
transcendentals. `ecs.frost` and `slab.frost` add one runtime call each, for
reporting and for the identifier a container stamps into its handles. Those
eight files hold every `extern` in the library. The other twelve declare none,
which is why they are ordinary safe code.

Nothing is imported implicitly. A program that wants to print says
`import "io.frost"`, and the file is found on the standard library search path
with nothing declared and no manifest. See
[modules.md](../impl/modules.md) for the five places an import looks and why the
standard library is the last of them.

| Module | What it is | Page |
| --- | --- | --- |
| `mem.frost` | Typed heap allocation, and the block counter the leak tests use | [mem.md](mem.md) |
| `arena.frost` | A bump allocator over a fixed buffer, carving `[]T` runs | [mem.md](mem.md) |
| `vec.frost` | A growable array, `linear`, over one heap block | [containers.md](containers.md) |
| `fixed.frost` | The same array over storage it does not own, an arena run among them | [containers.md](containers.md) |
| `map.frost` | A hash map by open addressing, with `Hashing<K>` | [containers.md](containers.md) |
| `slab.frost` | The generational slab, and `Handle<T>` | [containers.md](containers.md) |
| `columns.frost` | The same slab stored structure-of-arrays | [containers.md](containers.md) |
| `option.frost` | `Option<T>`, an ordinary generic enum | [containers.md](containers.md) |
| `ordering.frost` | `Ordering<T>`, and four orderings over numbers | [sort.md](sort.md) |
| `sort.frost` | Sorting a slice or a vector in place | [sort.md](sort.md) |
| `strings.frost` | Questions about `str`: equality, search, counting, parsing | [text-and-io.md](text-and-io.md) |
| `format.frost` | `Builder`, a byte buffer text is assembled into | [text-and-io.md](text-and-io.md) |
| `io.frost` | Writing to standard output without declaring `printf` | [text-and-io.md](text-and-io.md) |
| `fs.frost` | Reading and writing whole files | [text-and-io.md](text-and-io.md) |
| `json.frost` | A JSON reader over a flat node array | [text-and-io.md](text-and-io.md) |
| `thread.frost` | Spawn, join, and an atomic add | [thread.md](thread.md) |
| `math.frost` | Vectors, matrices and quaternions at `f32` | [math.md](math.md) |
| `math64.frost` | The same library at `f64` | [math.md](math.md) |
| `snapshot.frost` | A world written to bytes and read back, refusing a mismatched registry | [ecs.md](ecs.md) |
| `ecs.frost` | An archetype entity-component system | [ecs.md](ecs.md) |

[graphics.md](graphics.md) covers `lib/` and `examples/graphics/`, which are not part of
`std/`. It is SDL3 and WebGPU bound to Frost, and it is the worked example of
what a binding to a real C library looks like.

## What it stands on

Almost everything allocating in the library goes through `mem.frost`, which is
where the `unsafe` is concentrated. `arena.frost` is the one other allocator,
and its single `unsafe` block is the reinterpret from bytes to `T`.
`vec.frost`, `fixed.frost` and `map.frost` hold slices rather than raw pointers
and contain no `unsafe` block of their own.
`slab.frost` and `columns.frost` allocate nothing at all. `io.frost`,
`fs.frost` and `thread.frost` each wrap a handful of `extern` declarations so a
program that prints, reads a file or starts a thread names none of them.

Two patterns recur and are worth learning once. A capability is a struct whose
fields are functions, passed as a compile-time argument and folded at the call:
`Ordering<T>` for the sort and `Hashing<K>` for the map. A container hands out
its elements as a bounds-checked slice rather than taking a callback:
`vec_slice`, `ecs_slice`, `world.x` on a `columns`. Both come out of the same
decision, which [philosophy.md](../design/philosophy.md) argues for: no traits
and no capturing closures, so the body goes where it is written and the dispatch
folds away.

## Running the tests

A module's `test` blocks are its own. They are compiled when that file is the
one named to `--test`, and dropped when it is imported, so a library with tests
does not drag the harness into every program that uses it.

```bash
frost --test std/
```

runs every `.frost` file under `std/`, each in its own process, and prints how
many files failed. A single module is the same command with the file named:

```bash
frost --test std/map.frost
```

Eight of the seventeen carry tests: `ecs.frost` (72 blocks), `math.frost` and
`math64.frost` (20 each), `map.frost` (12), `mem.frost` (9), `strings.frost`
(6), `vec.frost` (5) and `sort.frost` (3). The rest are covered where they are
used, which each page says.
