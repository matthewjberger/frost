# The command line and the environment

Two programs answer to a command line here. `frost` is the bootstrap compiler,
written in Rust, and `frostc` is the self-hosted one, written in Frost. They
accept the same language and overlapping but different flags, so every entry
below says which one reads it. The names come from `just install` and
`just install-self`, which put the two on PATH under separate names so that
installing one leaves the other reachable.

## The bootstrap compiler

`frost [options] <file.frost>`. The file is the only positional argument and it
is required. With `--test` it may be a directory.

| Flag | What it does |
| --- | --- |
| `-n`, `--native` | Lower the IR through Cranelift and write a relocatable object |
| `-l`, `--link` | Link an executable. Implies `--native` unless `--emit-c` is also given |
| `-o`, `--output <PATH>` | Where the result goes, used verbatim |
| `--libs <LIB>` | An object file or library to pass to the linker after the program's own. Repeatable |
| `--emit-c` | Emit a C translation unit from the same IR instead of using Cranelift |
| `--run-ir` | Interpret the typed IR and print what the program printed |
| `--test` | Build the file's `test` blocks into an executable and run it |
| `--freestanding` | Link with no libc: a minimal runtime and a custom entry point |
| `--incremental` | Reuse a module's cached object unless its source or an imported interface changed |
| `--audit-unsafe` | Fail the build when an `unsafe` block vouches for nothing. Every build already warns about one |
| `--build-dir <DIR>` | Where `--incremental` keeps interfaces and objects. Defaults to `.frost-build` |
| `-L`, `--lib-path <DIR>` | A directory to find imports under, after the importing file's own. Repeatable |
| `-h`, `--help` | The flag list |

There is no `--version`.

The modes are tried in one order and the first that matches wins: `--test`,
then `--run-ir`, then `--emit-c`, then `--native` or `--link`, then the bare
run. So `--test` still reads `--emit-c` and runs its bodies through whichever
backend that names, and `--run-ir` ignores `-o` and exits 3 when the interpreter
declines the program.
[Native, freestanding, self-hosted](../impl/build-modes.md) says which backend
means what.

`frost program.frost` with no flags at all compiles through Cranelift into a
temporary object, links a temporary executable, runs it, deletes both, and
exits with what the program returned.

`--incremental` is rejected without `--link`, since a module is a compilation
unit only when the objects are linked, and rejected alongside `--test`,
`--emit-c` or `--run-ir`. See
[separate compilation](../impl/separate-compilation.md) for what it compares and
what it stores.

`--freestanding` reaches only the native link. The `--emit-c --link` path, the
`--test` link and the bare run all link the ordinary runtime, and the flag is
refused outright when the toolchain found is MSVC `cl`.

`--libs` applies to every path that links, including `--test` and the bare run,
which is how an example links SDL3.

### What the output is called

Given `-o`, the path is used exactly as written, on every platform. No
extension is appended and none is replaced. A Unix `--link -o frost.exe` build
produces an ELF binary named `frost.exe`, which `just install-self` relies on.

Without `-o`, and only then, the compiler derives a name from the source file's
stem:

| Mode | Derived name |
| --- | --- |
| `--link` | `stem.exe` on Windows, `stem` elsewhere |
| `--native` alone | `stem.o` |
| `--emit-c` alone | `stem.c` |

A `--link` build writes one object per module as `<output>.<index>.o` and
removes them after the link, so those intermediates are named for the
executable and two builds of one program to two outputs leave each other's
alone.

### Tests

`frost --test file.frost` compiles the file plus a generated `main` that runs
each `test` block through the runtime's test runner, links it, runs it, and
exits 1 if any test failed. A file with no `test` blocks says so and exits 0.

`--test` also takes a directory, and that is how the standard library's tests
are run:

```bash
frost --test std/
```

Every `.frost` file under the directory, recursively and in sorted order, is
compiled and run as its own process, and the run ends with a count of how many
files failed. Each file gets its own process, so a test that crashes outright
takes down only its own file, and two files are two programs that may define
the same names. Only `-L` and `--libs` are carried through to the per-file runs.

## The self-hosted compiler

`frostc` reads a smaller and slightly different set. Its own `--help` prints
the same list. This is that list with what each flag implies.

| Flag | What it does |
| --- | --- |
| `-o <PATH>` | Where the result goes. Without it the emitted text goes to standard output |
| `-L <DIR>` | A directory to find imports under. Repeatable |
| `--emit-c` | Emit a C translation unit. This is the default for a build that does not link |
| `--emit-asm` | Emit x86-64 assembly |
| `-n`, `--native` | Encode that assembly into an object file |
| `-l`, `--link` | Assemble and link an executable. With no backend flag this picks the assembly one |
| `--libs <ITEM>` | An object file or library to pass to the linker. Repeatable |
| `--incremental` | One object per module, assembling only what changed. Implies `--link` |
| `--build-dir <DIR>` | Where those objects are kept. Defaults to `.frost-build` |
| `--assemble` | Read an assembly file and write the object it stands for, compiling nothing |
| `--test` | Build and run the file's `test` blocks. Implies `--link` |
| `--audit-unsafe` | Fail the build when an `unsafe` block vouches for nothing. Every build already warns about one |
| `-h`, `--help` | The flag list |

The differences from the bootstrap: emitting on its own goes through C, output
goes to standard output when nothing asked for a file, a `--link` build with no
`-o` writes `a.exe` on Windows and `a.out` elsewhere, `--test` takes a file and
never a directory, and the `--freestanding` and `--run-ir` flags have no path
here, which the compiler says in those words rather than calling them unknown.
An argument starting with `-` that none of the above claims ends the run with an
error.

`--audit-unsafe` exists on both compilers and means the same thing on each: a
build fails if any `unsafe` block covers no operation that needed one. The
report is not what the flag turns on. Every build names such a block already,
and the flag is what turns naming one into a refusal.

## Environment variables

### Read by both compilers

| Variable | Effect |
| --- | --- |
| `FROST_PATH` | Import search directories. The bootstrap splits it the way the platform splits a path list and places it after `-L` and before `frost.json`; the self-hosted compiler takes it as one directory, ahead of `std` and `.` |
| `FROST_STD` | Where the bundled standard library is. Wins over a `std` beside the compiler and over the one at the repository root |
| `FROST_RUNTIME_FROST` | Where the runtime's Frost half is. Both compilers otherwise look for it beside themselves and then up the directories a checkout puts it under. The runtime is two files: `runtime.frost` holds the checks an index and a slice compile to, and `frost_runtime.c` holds what cannot be written in Frost |

The rest of the search order is [finding a module](../impl/modules.md).

### Set by both compilers

| Variable | Effect |
| --- | --- |
| `FROST_COMPILER` | The resolved path of the compiler that ran a program under `frost run`, put in that program's environment. A build program written in Frost drives the compiler that started it rather than whichever one is on PATH, which is what lets one checkout hold two compilers and have each build with itself |

### Read by the bootstrap compiler only

| Variable | Effect |
| --- | --- |
| `FROST_THREADS` | Cap on code-generation threads. Defaults to the machine's parallelism; a value that is not a positive number is ignored |
| `FROST_TIMINGS` | Anything but `0`: report the split between generating code for each function and writing the object, on stderr |
| `FROST_MODULE_REPORT` | Anything but `0`: report how many duplicated specializations separate compilation costs, on stderr |
| `FROST_CHECK_INTERFACES` | Anything but `0`: write each module's interface out, read it back, and check it says what the source says, while still compiling from source |
| `FROST_BUILD_FROM_INTERFACES` | Anything but `0`: an imported module contributes what its interface says it contributes and nothing else |

The last two are the oracles behind `--incremental`, described in
[separate compilation](../impl/separate-compilation.md). Both are off in an
ordinary build.

### Read by the self-hosted compiler only

| Variable | Effect |
| --- | --- |
| `FROST_INPUT` | The file to compile, when the command line named none |
| `FROST_BACKEND` | `asm` picks the assembly emitter and `c` the C one, when no backend flag was given |
| `FROST_RUNTIME` | Where the runtime's C stub is. Looked for beside the compiler and then up the directories a checkout puts it under. Set it to compile from a checkout you are standing outside of. The bootstrap needs no such variable, since it carries that file inside itself |
| `FROST_ABI` | `sysv` or `win` overrides the host's calling convention, so either target's output can be read from either host |
| `FROST_OBJECT` | `elf` or `coff` overrides the host's object format, the way `FROST_ABI` overrides the convention |
| `FROST_QUERY` | Answer an editor's question about the checked program instead of building it: `symbols`, `definition NAME`, `fields NAME`, or `local FN NAME`. Answers go to stderr, one line each, and nothing is emitted |
| `CC` | The C compiler the emitted C and the link go to. Defaults to `gcc` on Windows and `cc` elsewhere |

The fixpoint checks and the `just selfhost-*` recipes drive the compiler with
`FROST_INPUT` and `FROST_BACKEND`, so each one answers for an argument the
command line left out and a flag on the command line wins.

### Read by the tests and the justfile

| Variable | Effect |
| --- | --- |
| `FROST_REQUIRE_LINKER` | `tests/native.rs`: set, a missing C toolchain fails the run instead of quietly skipping every test that links |
| `FROST_BENCH` | `just bench-selfhost`: which file to measure. Defaults to `selfhosted/frost.frost` |
| `FROST_BIN` | Where `just install` and `just install-self` put the binaries. Defaults to `~/.cargo/bin` |
| `SDL3_DIR` | `just app window` and `just app triangle` on Windows: the directory holding `SDL3.dll` |

## The just recipes

The repository is built, tested and measured with the justfile. The recipes
worth reaching for:

| Recipe | What it does |
| --- | --- |
| `just build` | Builds the project in release mode |
| `just install` | Builds the bootstrap compiler and puts it on PATH as `frost` |
| `just install-self` | Builds the self-hosted compiler and puts it on PATH as `frostc` |
| `just run FILE` | Compiles and runs a frost file |
| `just generate` | Writes every file `frost.json` says a program of this project writes |
| `just generate-check` | Says whether each of those is what its generator would write, and exits nonzero when one is not |
| `just compile FILE` | Compiles a frost file to a native executable |
| `just compile-c FILE` | Compiles a frost file through the C backend instead of the native one |
| `just check-file FILE` | Checks a frost file without producing an executable, for the editor |
| `just test` | Runs all tests |
| `just test-interfaces` | Runs all tests with every imported module reduced to what its interface says it offers |
| `just lint` | Runs the linter and displays warnings |
| `just format` | Formats the code |
| `just examples` | Lists the example programs |
| `just examples-run` | Builds and runs every example, checking they all still work |
| `just install-editor` | Links the VS Code syntax highlighting for `.frost` files |
| `just selfhost-build` | Builds the self-hosted compiler |
| `just selfhost-run FILE` | Compiles a frost file with the self-hosted compiler, via its C backend |
| `just selfhost-native FILE` | Compiles a file with the self-hosted native backend, then assembles and runs it |
| `just selfhost-check` | Checks the self-hosted compiler reproduces itself exactly |
| `just selfhost-native-check` | Checks the compiler built from its own assembly reproduces that assembly exactly |
| `just selfhost-test` | Runs every self-hosting check: fixpoint, emitted C, native backend, own errors |
| `just bench FILE` | Reports how long a build takes, compiler work against linking |
| `just bench-scaling` | Measures how the pipeline scales, in lines and in specializations |
| `just bench-incremental` | Measures what `--incremental` saves on a program spread over modules |
| `just bench-selfhost` | Measures the self-hosted compiler against the bootstrap on one source |
| `just app window`, `just app triangle` | Opens an SDL3 window, and draws a wgpu triangle in one |

`just` on its own lists every recipe, including the version and release ones
left out here.

## Subcommands

Read from the first argument, before any flag. `frost <file.frost>` and every
flag above keep their meaning alongside them.

### `frost run <file> [args...]`

Compiles the file, runs what it built, and exits on what the program returned.
Everything written after the file belongs to the program, so a program's own
`--check` is answered by the program rather than by the compiler.

The executable is temporary and is taken away again, the way `--test` builds
one. Both compilers accept this and both set `FROST_COMPILER` for the program
they run.

This is what a build program written in Frost is started with, and it is how
`frost generate` runs the generators a project declares.

### `frost fmt <paths...>`

Writes the one rendering of every file named. A directory is every `.frost` file
under it, and `-` is standard input, whose rendering goes to standard output.
`--check` writes nothing, names the files that are not already formatted, and
exits nonzero.

The rendering settles the space inside a line, the indentation in front of it,
how many blank lines sit between two of them, the brace that opens a block, and
the newline a file ends with. It keeps every token on the line it was written
on, because which line a token is on is meaning here. See
[Where a statement ends](../design/line-boundaries.md).

Both compilers write the same bytes, which a test over the whole corpus holds.

### `frost lint <paths...>`

Reports what is worth a look and refuses nothing. `frost lint` exits nonzero
when it finds any, so a project can hold a tree to none of them.

- an `unsafe` block that holds nothing unchecked
- a function nothing reaches
- an exported name outside the prefix its directory declares, where `frost.json`
  declares one under `prefixes`

### `frost fix <file>`

Applies every edit the reports carry that can be applied unread. The edits are
read back out of `--diagnostics=json`, so anything it applies is something a
reader could have applied by hand from the same output.

### `frost api <prefix> [paths...]`

Prints the exported names beginning with a prefix, each with its signature as it
was written. `--json` writes one object per name. With no paths it walks the
directory it is run in.

A flat namespace has no `.` to narrow a guess with, and a family is named by its
prefix here, so this asks for that narrowing directly.

### `frost generate [--check]`

Writes every file the project's `frost.json` says a program of its own writes,
in the order declared. `--check` writes each one somewhere else instead and says
whether it matches what is on disk, exiting nonzero when one does not, which is
what holds a checkout to generated files that are not stale.

A step always writes, and staleness is decided from content rather than from
timestamps, because a checkout stamps every file with the time it was made.

The generator is compiled and run by the compiler that was asked, so a checkout
holding two compilers regenerates with whichever one it was given. It is handed
the output path first and the declared inputs after. A `.frost` output then goes
through `frost fmt`, so what a check compares is what a build would leave on
disk. Both compilers accept this and write the same files.

See [Finding a module](../impl/modules.md) for how a generator is declared.

## Diagnostics as JSON

`--diagnostics=json` writes one report per line as an object: the file, the line
and column, the same place as a byte offset, the severity, the message, the
other places the report is about, and the edit that answers it where there is
one. Both compilers write the same records.

A report about a name nothing declares carries the nearest declared name, when
one name is nearer than every other.
