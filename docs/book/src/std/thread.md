# Threads

`std/thread.frost` is three functions over the C threading calls: start a
thread, wait for it, and add to a shared word without a lock.

```frost
export spawn, join, atomic_add
```

| Call | What it does |
| --- | --- |
| `spawn(body: fn(^u8), context: ^u8) -> i64` | Runs `body(context)` on a new thread, answering a handle |
| `join(handle: i64)` | Waits for that thread to finish |
| `atomic_add(cell: ^i64, amount: i64) -> i64` | Adds to a shared word, answering its value before the add |

Each of the three wraps one `extern`, so the spawn, the join and the atomic add
need no `unsafe` block at their call sites. That is the whole module. What still
needs one is the context: `ptr_cast` and a dereference are gated wherever they
are written, and passing a typed struct through a `^u8` needs both.

## What crosses

A thread body is `fn(^u8)`. The context is a raw pointer because a thread
crosses into C, where the body is a `void (*)(void*)` and the context is its
`void*`. The body casts it back to the type it knows:

```frost
import "io.frost"
import "thread.frost"

Work :: struct { start: i64, count: i64, total: ^i64 }

worker :: fn(raw: ^u8) {
    w := unsafe { ptr_cast($Work, raw) }
    start := unsafe { w^.start }
    count := unsafe { w^.count }
    cell := unsafe { w^.total }
    mut i : i64 = 0
    while (i < count) {
        atomic_add(cell, start + i)
        i = i + 1
    }
}

main :: fn() -> i64 {
    mut total : i64 = 0
    mut w1 := Work { start = 0, count = 500, total = ptr_to(total) }
    mut w2 := Work { start = 500, count = 500, total = ptr_to(total) }
    t1 := spawn(worker, unsafe { ptr_cast($u8, ptr_to(w1)) })
    t2 := spawn(worker, unsafe { ptr_cast($u8, ptr_to(w2)) })
    join(t1)
    join(t2)
    print_int_line(total)     // 499500
    0
}
```

The two work descriptions live in `main`'s frame, and `main` does not read
`total` until after both joins. That is the arrangement the module requires, and
it is the caller's to get right.

## What is guaranteed, and what is not

This is the reasonable-C floor, not a checked concurrency model. Frost's
memory-safety guarantees are about a single thread's accesses. Nothing in the
type system tracks which thread owns a value, whether a `mut` borrow is shared
across threads, or whether two threads write the same field.

Three obligations sit with the caller, and the compiler will not check any of
them:

The context must outlive the thread. `spawn` takes an address and returns
immediately, so a context in a frame that returns before the join is a pointer
to a dead frame. Keeping it in the spawner's own frame and joining before that
frame returns is what the example above does.

Anything two threads touch at once goes through `atomic_add`, or the program
races. Reading a plain `i64` that another thread is writing is undefined, the
same as it is in C.

There is no specified memory model beyond what `atomic_add` itself provides. The
library does not define an ordering between a write on one thread and a read on
another, and it offers no fences, no mutexes and no condition variables. What it
offers is: a spawn starts a thread, a join waits for it to finish, and an atomic
add on a single word is not torn by a concurrent one.

This program is the one `self_hosted_threads_share_a_counter` in
`tests/native.rs` writes out, compiles through the C backend and runs, checking
the total is 499500. An exact total is what says the atomic held and the join
waited: a race would produce a smaller number, and a missed join would produce a
smaller one too.

## What is not here

No thread pool, no channels, no locks, no thread-local storage, and no way for a
thread body to return a value. A body communicates by writing into memory the
spawner owns, and the spawner reads it after the join.
