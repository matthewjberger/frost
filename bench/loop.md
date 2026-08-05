# What the loop costs

Produced by `just bench-loop`, which is `bench/loop.ps1`. Median of five runs
after one thrown away. Re-run it rather than trusting the table: these are one
machine's readings on one day.

Hardware: AMD Ryzen 7 7800X3D, 64 GB. Commit 96b6763.

| measurement | median ms |
| --- | --- |
| one file, checked to an object | 12 |
| one file, compiled and its tests run | 9 |
| largest module, checked to an object | 487 |
| `frost fmt --check` over the corpus | 121 |
| `frost lint` over the standard library | 110 |
| `frost api` over the standard library | 10 |
| self-hosting fixpoint, both generations | 12,503 |

The figure the thesis rests on is the first two: a question about one file is
answered in about a hundredth of a second, so asking the compiler is cheaper
than reasoning about the answer by a wide margin. Nothing here is over the two
second mark that was set as worth flagging, so no profile is attached.

The fixpoint is the outlier and it is the only measurement that is not
interactive: it compiles the whole compiler twice.
