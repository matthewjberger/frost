# Sorting, and orderings as values

`std/sort.frost` orders a slice in place. `std/ordering.frost` says what "in
order" means. They are two files because the caller passes in what counts as
ordered.

## An ordering is a struct

```frost
Ordering :: struct($T: Type) {
    less: fn(T, T) -> bool,
    equal: fn(T, T) -> bool,
}
```

An implementation is an ordinary constant of that struct:

```frost
Ordering :: struct($T: Type) {
    less: fn(T, T) -> bool,
    equal: fn(T, T) -> bool,
}

i64_less :: fn(a: i64, b: i64) -> bool { a < b }
i64_greater :: fn(a: i64, b: i64) -> bool { a > b }
i64_equal :: fn(a: i64, b: i64) -> bool { a == b }

i64_ascending :: Ordering<i64> { less = i64_less, equal = i64_equal }
i64_descending :: Ordering<i64> { less = i64_greater, equal = i64_equal }
```

There is no registry, no lookup and no coherence rule. Two orderings over the
same type are two constants, and they do not conflict.

`std/ordering.frost` ships four of them:

| Constant | What it orders by |
| --- | --- |
| `i64_ascending` | `i64`, smallest first |
| `i64_descending` | `i64`, largest first |
| `f64_ascending` | `f64`, smallest first |
| `f64_descending` | `f64`, largest first |

and the six comparison functions they are built from: `i64_less`, `i64_greater`,
`i64_equal`, `f64_less`, `f64_greater`, `f64_equal`. A key of any other type is
a fifth constant, written where it is needed:

```frost,sketch
by_hp :: fn(a: Unit, b: Unit) -> bool { a.hp < b.hp }
same_hp :: fn(a: Unit, b: Unit) -> bool { a.hp == b.hp }
weakest_first :: Ordering<Unit> { less = by_hp, equal = same_hp }
```

## The sort takes it at compile time

```frost,sketch
sort :: fn($T: Type, $ops: Ordering<T>, mut items: []T)
sort_vec :: fn($T: Type, $ops: Ordering<T>, mut v: Vec<T>)
```

`$ops` is a compile-time argument, so `ops.less(a, b)` in the inner loop folds
to a direct call to whichever function that constant's field names. The
comparison ends up inside the loop, and the sorted slice holds no function
pointer.

```frost,sketch
mut items := [5, 2, 9, 1, 7]
sort($i64_ascending, items)     // 1 2 5 7 9
sort($i64_descending, items)    // 9 7 5 2 1
```

`sort_vec` is the same walk over a vector's live elements, which it reaches
through `vec_slice`, so a vector with room for sixty-four and three elements in
it sorts three.

```frost,sketch
mut v := vec_new($i64, 4)
vec_push(v, 3)
vec_push(v, 1)
vec_push(v, 2)
sort_vec($i64_ascending, v)
assert(vec_get(v, 0) == 1)
vec_free(v)
```

## The algorithm

Insertion sort while `high - low < 12`, quicksort above that. Quicksort is not
stable, and it takes the middle element as its pivot, which keeps already-sorted
and reverse-sorted input off the quadratic path a first-element pivot falls
into. Each of the two recursive calls is guarded on the partition having shrunk
the range it was given.

Only `sort` and `sort_vec` are exported. `insertion`, `quicksort` and `swap` are
private to the file.

## The same shape elsewhere

`Ordering<T>` is the smallest capability bundle in the library.
`Hashing<K>` in [containers.md](containers.md) is the same struct over a hash
and an equality, passed the same way and folded the same way. Higher-order code
in Frost takes this shape throughout: a struct of functions, handed over as a
compile-time argument and folded at the call.

Section 11.4b of [generics.md](../reference/generics.md) is the language rule,
and [philosophy.md](../design/philosophy.md) says why there are no traits.

## Tests

```bash
frost --test std/sort.frost
```

Three blocks: a short run through the insertion path in both directions, a
fifteen-element run that reaches quicksort, and a vector sorted over its live
prefix. `std/ordering.frost` has no tests of its own, since its contents are
four struct constants and six one-line comparisons that the sort's tests
exercise.
