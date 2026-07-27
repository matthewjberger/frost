# The Frost compiler, written in Frost

A compiler for Frost, written in Frost. It lexes, parses, type-checks, and emits
either a C translation unit or x86-64 assembly. `frost.frost` is the driver.

What it implements, how it works, where the two compilers differ, and how to
build and run it are in the book:
[docs/book/src/impl/self-hosted.md](../docs/book/src/impl/self-hosted.md).

```bash
just selfhost-build     # build it with the bootstrap
just install-self       # put it on PATH as `frostc`
just selfhost-test      # every self-hosting check, including both fixpoints
```
