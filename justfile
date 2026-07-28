set windows-shell := ["powershell.exe"]
export RUST_BACKTRACE := "1"

# Where `install` and `install-self` put the two compilers. `~/.cargo/bin` is
# already on PATH for anyone who can build this repo, so the default needs no
# setup. FROST_BIN overrides it. Windows sets USERPROFILE and not always HOME.
homedir := env_var_or_default("HOME", env_var_or_default("USERPROFILE", "."))
bindir := env_var_or_default("FROST_BIN", join(homedir, ".cargo", "bin"))

# Displays the list of available commands
@just:
    just --list

# Builds the project in release mode
build:
    cargo build -r

# Builds the bootstrap compiler, with the standard library beside it (Windows)
[windows]
install: build
    New-Item -ItemType Directory -Force "{{bindir}}" | Out-Null
    Copy-Item -Force target/release/frost.exe (Join-Path "{{bindir}}" frost.exe)
    New-Item -ItemType Directory -Force (Join-Path "{{bindir}}" std) | Out-Null
    Copy-Item -Force std/*.frost (Join-Path "{{bindir}}" std)
    New-Item -ItemType Directory -Force (Join-Path "{{bindir}}" runtime) | Out-Null
    Copy-Item -Force runtime/frost_runtime.c (Join-Path "{{bindir}}" runtime)
    Write-Host "frost -> {{bindir}}"

# Builds the bootstrap compiler, with the standard library beside it (Unix)
[unix]
install: build
    mkdir -p "{{bindir}}/std" "{{bindir}}/runtime"
    cp target/release/frost "{{bindir}}/frost"
    cp std/*.frost "{{bindir}}/std/"
    cp runtime/frost_runtime.c "{{bindir}}/runtime/"
    echo "frost -> {{bindir}}"

# Builds the self-hosted compiler, with the standard library beside it (Windows)
[windows]
install-self: selfhost-build
    New-Item -ItemType Directory -Force "{{bindir}}" | Out-Null
    Copy-Item -Force selfhosted/frost.exe (Join-Path "{{bindir}}" frostc.exe)
    New-Item -ItemType Directory -Force (Join-Path "{{bindir}}" std) | Out-Null
    Copy-Item -Force std/*.frost (Join-Path "{{bindir}}" std)
    New-Item -ItemType Directory -Force (Join-Path "{{bindir}}" runtime) | Out-Null
    Copy-Item -Force runtime/frost_runtime.c (Join-Path "{{bindir}}" runtime)
    Write-Host "frostc -> {{bindir}}"

# Builds the self-hosted compiler and puts it on PATH as `frostc` (Unix)
#
# The source really is `frost.exe` here. `-o` names the output verbatim on every
# platform, and `selfhost-build` asks for that name, so a Unix build produces an
# ELF binary called `frost.exe`.
[unix]
install-self: selfhost-build
    mkdir -p "{{bindir}}/std" "{{bindir}}/runtime"
    cp selfhosted/frost.exe "{{bindir}}/frostc"
    cp std/*.frost "{{bindir}}/std/"
    cp runtime/frost_runtime.c "{{bindir}}/runtime/"
    echo "frostc -> {{bindir}}"

# Removes both compilers from PATH (Windows)
[windows]
uninstall:
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path "{{bindir}}" frost.exe), (Join-Path "{{bindir}}" frostc.exe)
    Write-Host "removed frost and frostc from {{bindir}}"

# Removes both compilers from PATH (Unix)
[unix]
uninstall:
    rm -f "{{bindir}}/frost" "{{bindir}}/frostc"
    echo "removed frost and frostc from {{bindir}}"

# Runs cargo check and format check
check:
    cargo check --all --tests
    cargo fmt --all -- --check

# Generates and opens documentation
docs:
    cargo doc --open -p frost

# Fixes linting issues automatically
fix:
    cargo clippy --all --tests --fix

# Formats the code using cargo fmt
format:
    cargo fmt --all

# Install development tools
install-tools:
    cargo install cargo-license
    cargo install cargo-deny
    cargo install cargo-machete
    cargo install git-cliff

# Install git hooks (Windows)
[windows]
install-hooks:
    Copy-Item -Path hooks/pre-commit -Destination .git/hooks/pre-commit -Force

# Install git hooks (Unix)
[unix]
install-hooks:
    cp hooks/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

# Uninstall git hooks (Windows)
[windows]
uninstall-hooks:
    Remove-Item -Path .git/hooks/pre-commit -Force -ErrorAction SilentlyContinue

# Uninstall git hooks (Unix)
[unix]
uninstall-hooks:
    rm -f .git/hooks/pre-commit

# Checks a frost file without producing an executable, for the editor (Windows)
[windows]
check-file file:
    @New-Item -ItemType Directory -Force .frost-build | Out-Null; $env:RUST_BACKTRACE = "0"; cargo run -r -q -p frost --bin frost -- --native -o .frost-build/check.o {{file}}

# Checks a frost file without producing an executable, for the editor (Unix)
[unix]
check-file file:
    @mkdir -p .frost-build && RUST_BACKTRACE=0 cargo run -r -q -p frost --bin frost -- --native -o .frost-build/check.o {{file}}

# Prints where this VS Code keeps its extensions, portable installs included (Windows)
[windows]
editor-dir:
    @$cli = Get-Command code -ErrorAction SilentlyContinue; if (-not $cli) { throw "code is not on PATH" }; $root = Split-Path (Split-Path $cli.Source -Parent) -Parent; if (Test-Path "$root\data\extensions") { "$root\data\extensions" } else { "$env:USERPROFILE\.vscode\extensions" }

# Prints where this VS Code keeps its extensions, portable installs included (Unix)
[unix]
editor-dir:
    #!/usr/bin/env bash
    set -euo pipefail
    CLI=$(command -v code) || { echo "code is not on PATH" >&2; exit 1; }
    ROOT=$(dirname "$(dirname "$(readlink -f "$CLI")")")
    if [ -d "$ROOT/data/extensions" ]; then echo "$ROOT/data/extensions"; else echo "$HOME/.vscode/extensions"; fi

# Install the VS Code syntax highlighting for .frost files (Windows)
[windows]
install-editor:
    $dir = just editor-dir; $target = Join-Path $dir "frost"; New-Item -ItemType Directory -Force $dir | Out-Null; Remove-Item -Recurse -Force $target -ErrorAction Ignore; Copy-Item -Recurse -Force .vscode\frost $target; Write-Host "Installed to $target. Reload the VS Code window to pick it up." -ForegroundColor Green

# Install the VS Code syntax highlighting for .frost files (Unix)
[unix]
install-editor:
    #!/usr/bin/env bash
    set -euo pipefail
    DIR=$(just editor-dir)
    mkdir -p "$DIR"
    rm -rf "$DIR/frost"
    ln -s "$PWD/.vscode/frost" "$DIR/frost"
    echo "Linked $DIR/frost. Reload the VS Code window to pick it up."

# Remove the VS Code syntax highlighting for .frost files (Windows)
[windows]
uninstall-editor:
    $dir = just editor-dir; Remove-Item -Recurse -Force (Join-Path $dir "frost") -ErrorAction Ignore

# Remove the VS Code syntax highlighting for .frost files (Unix)
[unix]
uninstall-editor:
    rm -rf "$(just editor-dir)/frost"

# Runs linter and displays warnings
lint:
    cargo clippy --all --tests -- -D warnings

# Compiles and runs a frost file
run file:
    cargo run -r -q -p frost --bin frost -- --link -o {{file}}.exe {{file}}
    ./{{file}}.exe

# Compiles a frost file to a native executable
compile file:
    cargo run -r -q -p frost --bin frost -- --link -o {{file}}.exe {{file}}

# Compiles a frost file through the C backend instead of the native one
compile-c file:
    cargo run -r -q -p frost --bin frost -- --emit-c --link -o {{file}}.exe {{file}}

# Lists the example programs
[unix]
examples:
    @echo "full frost (just run):"; ls examples/native/*.frost | sed 's|.*/||; s|\.frost$||' | sed 's/^/  /'
    @echo "self-hosted subset (just selfhost-native):"; ls examples/selfhosted/*.frost | sed 's|.*/||; s|\.frost$||' | sed 's/^/  /'

# Lists the example programs
[windows]
examples:
    @Write-Host "full frost (just run):"; Get-ChildItem examples/native/*.frost | ForEach-Object { "  " + $_.BaseName }
    @Write-Host "self-hosted subset (just selfhost-native):"; Get-ChildItem examples/selfhosted/*.frost | ForEach-Object { "  " + $_.BaseName }

# Builds and runs every example, checking they all still work
[unix]
examples-run:
    #!/usr/bin/env bash
    set -euo pipefail
    for f in examples/native/*.frost; do
        echo "== $f"
        cargo run -r -q -p frost --bin frost -- --link -o "$f.exe" "$f"
        "./$f.exe"
        rm -f "$f.exe"
    done

# Builds and runs every example, checking they all still work
[windows]
examples-run:
    Get-ChildItem examples/native/*.frost | ForEach-Object { Write-Host "== $_"; cargo run -r -q -p frost --bin frost -- --link -o "$_.exe" "$_"; & "$($_.FullName).exe"; Remove-Item "$($_.FullName).exe" -Force }

# Opens a window with SDL3. SDL3_DIR overrides where SDL3.dll is found (Windows)
[windows]
window:
    $dir = if ($env:SDL3_DIR) { $env:SDL3_DIR } else { "examples/graphics" }; if (-not (Test-Path "$dir/SDL3.dll")) { throw "no SDL3.dll in $dir. Set SDL3_DIR to a directory containing it." }; cargo run -r -q -p frost --bin frost -- --link --libs "$dir/SDL3.dll" -o examples/graphics/window.exe examples/graphics/window.frost; if ($dir -ne "examples/graphics") { Copy-Item "$dir/SDL3.dll" examples/graphics -Force }; & ./examples/graphics/window.exe

# Opens a window with SDL3, from the system package libsdl3-dev / sdl3 (Unix)
[unix]
window:
    cargo run -r -q -p frost --bin frost -- --link --libs=-lSDL3 -o examples/graphics/window examples/graphics/window.frost
    ./examples/graphics/window

# Fetches the libraries the graphics examples link against, and the schema the
# wgpu binding is generated from. Everything it writes is gitignored, so a fresh
# checkout runs this once and `just triangle` works.
#
# The three versions are pinned rather than tracking whatever is newest, and the
# last two are not free to move apart. A wgpu release ships the schema it was
# built from as `webgpu.yml`, the binding generator reads json, and the json
# lives in the header repository. So the commit below is the one that release
# vendored, read from its own submodule rather than chosen:
#
#     gh api repos/gfx-rs/wgpu-native/contents/ffi/webgpu-headers?ref=<tag>
#
# Taking whatever the header repository has newest instead gives a binding a
# release older than it does not answer to. That is not a build failure at the
# boundary, it is a field named something else three hundred lines into a
# program that was compiling yesterday.
sdl_version := "3.4.12"
wgpu_version := "v29.0.1.1"
webgpu_headers_rev := "673658bc2bd70ec39fc55ebe6bb0173cf6d0a603"

[windows]
deps:
    #!powershell.exe -NoProfile
    $ErrorActionPreference = "Stop"
    $ProgressPreference = "SilentlyContinue"
    $work = Join-Path $env:TEMP "frost-deps"
    New-Item -ItemType Directory -Force $work | Out-Null
    New-Item -ItemType Directory -Force "examples/graphics/wgpu" | Out-Null
    Write-Host "SDL {{sdl_version}}"
    $sdl = "https://github.com/libsdl-org/SDL/releases/download/release-{{sdl_version}}/SDL3-devel-{{sdl_version}}-mingw.zip"
    Invoke-WebRequest -Uri $sdl -OutFile "$work/sdl.zip" -UseBasicParsing
    Remove-Item "$work/sdl" -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive "$work/sdl.zip" "$work/sdl" -Force
    $root = "$work/sdl/SDL3-{{sdl_version}}/x86_64-w64-mingw32"
    Copy-Item "$root/bin/SDL3.dll" examples/graphics -Force
    Copy-Item "$root/lib/libSDL3.dll.a" examples/graphics/SDL3.lib -Force
    Remove-Item "examples/graphics/SDL3" -Recurse -Force -ErrorAction SilentlyContinue
    Copy-Item "$root/include/SDL3" examples/graphics/SDL3 -Recurse -Force
    Write-Host "wgpu-native {{wgpu_version}}"
    $wgpu = "https://github.com/gfx-rs/wgpu-native/releases/download/{{wgpu_version}}/wgpu-windows-x86_64-gnu-release.zip"
    Invoke-WebRequest -Uri $wgpu -OutFile "$work/wgpu.zip" -UseBasicParsing
    Remove-Item "$work/wgpu" -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive "$work/wgpu.zip" "$work/wgpu" -Force
    Copy-Item "$work/wgpu/lib/wgpu_native.dll" examples/graphics/wgpu -Force
    Copy-Item "$work/wgpu/lib/libwgpu_native.dll.a" examples/graphics/wgpu -Force
    Copy-Item "$work/wgpu/include" examples/graphics/wgpu -Recurse -Force
    Write-Host "webgpu.json at {{webgpu_headers_rev}}"
    $json = "https://raw.githubusercontent.com/webgpu-native/webgpu-headers/{{webgpu_headers_rev}}/webgpu.json"
    Invoke-WebRequest -Uri $json -OutFile "examples/graphics/wgpu/webgpu.json" -UseBasicParsing
    Write-Host "ready. run: just bindgen; just triangle"

[unix]
deps:
    #!/usr/bin/env sh
    set -e
    work="${TMPDIR:-/tmp}/frost-deps"
    mkdir -p "$work" examples/graphics/wgpu
    case "$(uname -s)-$(uname -m)" in
      Linux-x86_64)  target=wgpu-linux-x86_64-release.zip ;;
      Linux-aarch64) target=wgpu-linux-aarch64-release.zip ;;
      Darwin-x86_64) target=wgpu-macos-x86_64-release.zip ;;
      Darwin-arm64)  target=wgpu-macos-aarch64-release.zip ;;
      *) echo "no wgpu-native build published for $(uname -s)-$(uname -m)" >&2; exit 1 ;;
    esac
    echo "wgpu-native {{wgpu_version}} ($target)"
    curl -fsSL -o "$work/wgpu.zip" \
      "https://github.com/gfx-rs/wgpu-native/releases/download/{{wgpu_version}}/$target"
    rm -rf "$work/wgpu" && mkdir -p "$work/wgpu"
    unzip -q "$work/wgpu.zip" -d "$work/wgpu"
    cp "$work"/wgpu/lib/libwgpu_native.* examples/graphics/wgpu/
    cp -r "$work/wgpu/include" examples/graphics/wgpu/
    echo "SDL3 headers are the system package's here."
    echo "webgpu.json at {{webgpu_headers_rev}}"
    curl -fsSL -o examples/graphics/wgpu/webgpu.json \
      "https://raw.githubusercontent.com/webgpu-native/webgpu-headers/{{webgpu_headers_rev}}/webgpu.json"
    echo "SDL3 comes from the system here: install libsdl3-dev or sdl3."
    echo "ready. run: just bindgen; just triangle"

# A spinning triangle through the platform and the renderer (Windows)
[windows]
scene:
    $sdl = if ($env:SDL3_DIR) { $env:SDL3_DIR } else { "examples/graphics" }; cargo run -r -q -p frost --bin frost -- --link --libs "$sdl/SDL3.dll" --libs "examples/graphics/wgpu/wgpu_native.dll" -o examples/graphics/scene.exe examples/graphics/scene.frost; Copy-Item examples/graphics/wgpu/wgpu_native.dll examples/graphics -Force; & ./examples/graphics/scene.exe

# A spinning triangle through the platform and the renderer (Unix)
[unix]
scene:
    cargo run -r -q -p frost --bin frost -- --link --libs=-lSDL3 --libs=-lwgpu_native -o examples/graphics/scene examples/graphics/scene.frost
    ./examples/graphics/scene

# A field of lit primitives, each turning about its own axis (Windows)
[windows]
spinning:
    $sdl = if ($env:SDL3_DIR) { $env:SDL3_DIR } else { "examples/graphics" }; cargo run -r -q -p frost --bin frost -- --link --libs "$sdl/SDL3.dll" --libs "examples/graphics/wgpu/wgpu_native.dll" -o examples/graphics/spinning.exe examples/graphics/spinning.frost; Copy-Item examples/graphics/wgpu/wgpu_native.dll examples/graphics -Force; & ./examples/graphics/spinning.exe

# A field of lit primitives, each turning about its own axis (Unix)
[unix]
spinning:
    cargo run -r -q -p frost --bin frost -- --link --libs=-lSDL3 --libs=-lwgpu_native -o examples/graphics/spinning examples/graphics/spinning.frost
    ./examples/graphics/spinning

# Opens a window and reports what the platform layer saw (Windows)
[windows]
input:
    $dir = if ($env:SDL3_DIR) { $env:SDL3_DIR } else { "examples/graphics" }; if (-not (Test-Path "$dir/SDL3.dll")) { throw "no SDL3.dll in $dir. Run `just deps`, or set SDL3_DIR." }; cargo run -r -q -p frost --bin frost -- --link --libs "$dir/SDL3.dll" -o examples/graphics/input.exe examples/graphics/input.frost; if ($dir -ne "examples/graphics") { Copy-Item "$dir/SDL3.dll" examples/graphics -Force }; & ./examples/graphics/input.exe

# Opens a window and reports what the platform layer saw (Unix)
[unix]
input:
    cargo run -r -q -p frost --bin frost -- --link --libs=-lSDL3 -o examples/graphics/input examples/graphics/input.frost
    ./examples/graphics/input

# Regenerates the wgpu bindings from webgpu.json
bindgen:
    cargo run -r -q -p frost --bin frost -- --link -o tools/wgpu_bindgen.exe tools/wgpu_bindgen.frost
    ./tools/wgpu_bindgen.exe

# Redraws docs/book/src/tour.svg, the picture of examples/tour.frost the README shows
tour-image:
    cargo run -r -q -p frost --bin frost -- --link -o tools/highlight.exe tools/highlight.frost
    ./tools/highlight.exe

# Draws a triangle with wgpu in an SDL3 window (Windows)
[windows]
triangle:
    $sdl = if ($env:SDL3_DIR) { $env:SDL3_DIR } else { "examples/graphics" }; cargo run -r -q -p frost --bin frost -- --link --libs "$sdl/SDL3.dll" --libs "examples/graphics/wgpu/wgpu_native.dll" -o examples/graphics/triangle.exe examples/graphics/triangle.frost; Copy-Item examples/graphics/wgpu/wgpu_native.dll examples/graphics -Force; & ./examples/graphics/triangle.exe

# Draws a triangle with wgpu in an SDL3 window, needs SDL3 and wgpu-native (Unix)
[unix]
triangle:
    cargo run -r -q -p frost --bin frost -- --link --libs=-lSDL3 --libs=-lwgpu_native -o examples/graphics/triangle examples/graphics/triangle.frost
    ./examples/graphics/triangle

# Builds the self-hosted compiler (frost written in frost)
#
# Through the C backend rather than the native one. That takes about six seconds
# against six hundred milliseconds, and it is deliberate: the compiler it
# produces is two and a half times faster on everything it goes on to do,
# because a C compiler inlines and allocates registers where Cranelift does not.
# The two routes are held to emitting the same bytes by
# both_routes_build_the_same_compiler, which is the only thing in the suite that
# builds a compiler the way this one ships.
selfhost-build:
    cargo run -r -q -p frost --bin frost -- --link --emit-c -o selfhosted/frost.exe selfhosted/frost.frost

# Compiles a frost file with the self-hosted compiler, via its C backend (Unix)
[unix]
selfhost-run file: selfhost-build
    FROST_INPUT={{file}} ./selfhosted/frost.exe

# Compiles a frost file with the self-hosted compiler, via its C backend (Windows)
[windows]
selfhost-run file: selfhost-build
    $env:FROST_INPUT = "{{file}}"; ./selfhosted/frost.exe

# Compiles a frost file with the self-hosted native backend, then assembles and runs it (Unix)
[unix]
selfhost-native file: selfhost-build
    #!/usr/bin/env bash
    set -euo pipefail
    FROST_BACKEND=asm FROST_INPUT={{file}} ./selfhosted/frost.exe > {{file}}.s
    cc {{file}}.s runtime/frost_runtime.c -o {{file}}.exe
    ./{{file}}.exe

# Compiles a frost file with the self-hosted native backend, then assembles and runs it (Windows)
[windows]
selfhost-native file: selfhost-build
    $env:FROST_BACKEND = "asm"; $env:FROST_INPUT = "{{file}}"; $asm = & ./selfhosted/frost.exe; $env:FROST_BACKEND = $null; [System.IO.File]::WriteAllLines((Resolve-Path .).Path + "/{{file}}.s", $asm); gcc "{{file}}.s" runtime/frost_runtime.c -o "{{file}}.exe"; & "./{{file}}.exe"

# Runs every self-hosted example through the native backend (Windows)
[windows]
selfhost-examples: selfhost-build
    Get-ChildItem examples/selfhosted/*.frost | ForEach-Object { Write-Host "== $($_.Name)"; $env:FROST_BACKEND = "asm"; $env:FROST_INPUT = $_.FullName; $asm = & ./selfhosted/frost.exe; if ($LASTEXITCODE -ne 0) { throw "the self-hosted compiler failed on $($_.Name)" }; $env:FROST_BACKEND = $null; [System.IO.File]::WriteAllLines($_.FullName + ".s", $asm); gcc ($_.FullName + ".s") runtime/frost_runtime.c -o ($_.FullName + ".exe"); & ($_.FullName + ".exe"); Remove-Item ($_.FullName + ".s"), ($_.FullName + ".exe") -Force }

# Runs every self-hosted example through the native backend (Unix)
[unix]
selfhost-examples: selfhost-build
    #!/usr/bin/env bash
    set -euo pipefail
    for f in examples/selfhosted/*.frost; do
        echo "== $f"
        FROST_BACKEND=asm FROST_INPUT="$f" ./selfhosted/frost.exe > "$f.s"
        cc "$f.s" runtime/frost_runtime.c -o "$f.exe"
        "./$f.exe"
        rm -f "$f.s" "$f.exe"
    done

# Checks the self-hosted compiler reproduces itself exactly (three-stage fixpoint)
selfhost-check:
    cargo test -r -p frost --test native self_hosting_is_a_fixpoint -- --nocapture

# Checks the compiler built from its own assembly reproduces that assembly exactly
selfhost-native-check:
    cargo test -r -p frost --test native native_self_hosting_is_a_fixpoint -- --nocapture

# Runs every self-hosting check: fixpoint, emitted C, native backend, own errors
selfhost-test:
    cargo test -r -p frost --test native self_host -- --nocapture
    cargo test -r -p frost --test native self_hosted -- --nocapture

# Reports how long a build takes, compiler work versus linking (Unix)
[unix]
bench file:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build -r -q -p frost --bin frost
    echo "compile only:"; time ./target/release/frost --native -o /tmp/bench.o {{file}}
    echo "with link:";    time ./target/release/frost --link -o /tmp/bench.exe {{file}}

# Reports how long a build takes, compiler work versus linking (Windows)
[windows]
bench file:
    cargo build -r -q -p frost --bin frost
    Write-Host "compile only:"; Measure-Command { ./target/release/frost.exe --native -o "$env:TEMP/bench.o" {{file}} } | Select-Object -ExpandProperty TotalMilliseconds
    Write-Host "with link:"; Measure-Command { ./target/release/frost.exe --link -o "$env:TEMP/bench.exe" {{file}} } | Select-Object -ExpandProperty TotalMilliseconds

# Measures how the pipeline scales, in lines and in specializations (Unix)
[unix]
bench-scaling:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build -r -q -p frost --bin frost
    python3 bench/generate.py bench/generated > /dev/null
    for f in bench/generated/*.frost; do
        printf "%-34s %6s lines  " "$(basename "$f")" "$(wc -l < "$f")"
        printf "front end "; /usr/bin/time -f "%e s" ./target/release/frost --emit-c -o /tmp/bench.c "$f" 2>&1 >/dev/null | tail -1
        printf "%-34s %6s        native    " "" ""
        /usr/bin/time -f "%e s" ./target/release/frost --native -o /tmp/bench.o "$f" 2>&1 >/dev/null | tail -1
    done

# Measures how the pipeline scales, in lines and in specializations (Windows)
[windows]
bench-scaling:
    cargo build -r -q -p frost --bin frost
    python bench/generate.py bench/generated | Out-Null
    Get-ChildItem bench/generated/*.frost | ForEach-Object { $lines = (Get-Content $_ | Measure-Object -Line).Lines; $front = (Measure-Command { ./target/release/frost.exe --emit-c -o "$env:TEMP/bench.c" $_.FullName }).TotalMilliseconds; $native = (Measure-Command { ./target/release/frost.exe --native -o "$env:TEMP/bench.o" $_.FullName }).TotalMilliseconds; "{0,-24} {1,7} lines  front end {2,7:N0} ms  native {3,7:N0} ms" -f $_.BaseName, $lines, $front, $native }

# Measures what --incremental saves on a program spread over modules (Unix)
[unix]
bench-incremental:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build -r -q -p frost --bin frost
    python3 bench/generate.py bench/generated > /dev/null
    rm -rf /tmp/frost-bench-build
    echo "full:"; time ./target/release/frost --link -o /tmp/bench.exe bench/generated/modules.frost
    ./target/release/frost --link --incremental --build-dir /tmp/frost-bench-build -o /tmp/bench.exe bench/generated/modules.frost > /dev/null
    echo "incremental:"; time ./target/release/frost --link --incremental --build-dir /tmp/frost-bench-build -o /tmp/bench.exe bench/generated/modules.frost

# Measures what --incremental saves on a program spread over modules (Windows)
[windows]
bench-incremental:
    cargo build -r -q -p frost --bin frost
    python bench/generate.py bench/generated | Out-Null
    Remove-Item -Recurse -Force "$env:TEMP/frost-bench-build" -ErrorAction Ignore
    $full = (Measure-Command { ./target/release/frost.exe --link -o "$env:TEMP/bench.exe" bench/generated/modules.frost }).TotalMilliseconds; ./target/release/frost.exe --link --incremental --build-dir "$env:TEMP/frost-bench-build" -o "$env:TEMP/bench.exe" bench/generated/modules.frost | Out-Null; $again = (Measure-Command { ./target/release/frost.exe --link --incremental --build-dir "$env:TEMP/frost-bench-build" -o "$env:TEMP/bench.exe" bench/generated/modules.frost }).TotalMilliseconds; "{0,-14} {1,7:N0} ms" -f "full", $full; "{0,-14} {1,7:N0} ms" -f "incremental", $again

# Measures the self-hosted compiler against the bootstrap on one source, so
# "speed parity" is a number rather than a feeling (Unix)
[unix]
bench-selfhost: selfhost-build
    #!/usr/bin/env bash
    set -euo pipefail
    cargo build -r -q -p frost --bin frost
    file="${FROST_BENCH:-selfhosted/frost.frost}"
    lines=$(cat $(dirname "$file")/*.frost | wc -l)
    echo "$file, $lines lines"
    printf "  bootstrap   front end  "; /usr/bin/time -f "%e s" ./target/release/frost --emit-c -o /tmp/bench.c "$file" 2>&1 >/dev/null | tail -1
    printf "  self-hosted C          "; /usr/bin/time -f "%e s" env FROST_INPUT="$file" ./selfhosted/frost.exe -o /tmp/bench.c 2>&1 >/dev/null | tail -1
    printf "  self-hosted assembly   "; /usr/bin/time -f "%e s" env FROST_INPUT="$file" FROST_BACKEND=asm ./selfhosted/frost.exe -o /tmp/bench.s 2>&1 >/dev/null | tail -1

# Measures the self-hosted compiler against the bootstrap on one source (Windows)
[windows]
bench-selfhost: selfhost-build
    cargo build -r -q -p frost --bin frost
    $file = if ($env:FROST_BENCH) { $env:FROST_BENCH } else { "selfhosted/frost.frost" }; $dir = Split-Path -Parent $file; if (-not $dir) { $dir = "." }; $lines = (Get-Content (Join-Path $dir "*.frost") | Measure-Object -Line).Lines; "{0}, {1} lines" -f $file, $lines; $boot = (Measure-Command { ./target/release/frost.exe --emit-c -o "$env:TEMP/bench.c" $file }).TotalMilliseconds; $env:FROST_INPUT = $file; $env:FROST_BACKEND = $null; $shc = (Measure-Command { ./selfhosted/frost.exe -o "$env:TEMP/bench.c" }).TotalMilliseconds; $env:FROST_BACKEND = "asm"; $sha = (Measure-Command { ./selfhosted/frost.exe -o "$env:TEMP/bench.s" }).TotalMilliseconds; $env:FROST_BACKEND = $null; $env:FROST_INPUT = $null; "  {0,-22} {1,7:N0} ms  {2,8:N0} lines/sec" -f "bootstrap front end", $boot, ($lines / $boot * 1000); "  {0,-22} {1,7:N0} ms  {2,8:N0} lines/sec" -f "self-hosted C", $shc, ($lines / $shc * 1000); "  {0,-22} {1,7:N0} ms  {2,8:N0} lines/sec" -f "self-hosted assembly", $sha, ($lines / $sha * 1000)

# Measures what the self-hosted compiler's --incremental saves (Windows)
[windows]
bench-selfhost-incremental: selfhost-build
    Remove-Item -Recurse -Force "$env:TEMP/frost-sh-build" -ErrorAction Ignore; $whole = (Measure-Command { ./selfhosted/frost.exe --link -o "$env:TEMP/whole.exe" selfhosted/frost.frost }).TotalMilliseconds; $first = (Measure-Command { ./selfhosted/frost.exe --incremental --build-dir "$env:TEMP/frost-sh-build" -o "$env:TEMP/inc.exe" selfhosted/frost.frost }).TotalMilliseconds; $again = (Measure-Command { ./selfhosted/frost.exe --incremental --build-dir "$env:TEMP/frost-sh-build" -o "$env:TEMP/inc.exe" selfhosted/frost.frost }).TotalMilliseconds; Remove-Item -Recurse -Force "$env:TEMP/frost-sh-build-c" -ErrorAction Ignore; $cwhole = (Measure-Command { ./selfhosted/frost.exe --emit-c --link -o "$env:TEMP/cwhole.exe" selfhosted/frost.frost }).TotalMilliseconds; $cfirst = (Measure-Command { ./selfhosted/frost.exe --emit-c --incremental --build-dir "$env:TEMP/frost-sh-build-c" -o "$env:TEMP/cinc.exe" selfhosted/frost.frost }).TotalMilliseconds; $cagain = (Measure-Command { ./selfhosted/frost.exe --emit-c --incremental --build-dir "$env:TEMP/frost-sh-build-c" -o "$env:TEMP/cinc.exe" selfhosted/frost.frost }).TotalMilliseconds; "{0,-34} {1,7:N0} ms" -f "assembly, whole program", $whole; "{0,-34} {1,7:N0} ms" -f "assembly, incremental first", $first; "{0,-34} {1,7:N0} ms" -f "assembly, incremental unchanged", $again; "{0,-34} {1,7:N0} ms" -f "C, whole program", $cwhole; "{0,-34} {1,7:N0} ms" -f "C, incremental first", $cfirst; "{0,-34} {1,7:N0} ms" -f "C, incremental unchanged", $cagain

# Measures what the self-hosted compiler's --incremental saves (Unix)
[unix]
bench-selfhost-incremental: selfhost-build
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf /tmp/frost-sh-build
    rm -rf /tmp/frost-sh-build-c
    echo "assembly, whole program:"; time ./selfhosted/frost.exe --link -o /tmp/whole selfhosted/frost.frost
    echo "assembly, incremental first:"; time ./selfhosted/frost.exe --incremental --build-dir /tmp/frost-sh-build -o /tmp/inc selfhosted/frost.frost
    echo "assembly, incremental unchanged:"; time ./selfhosted/frost.exe --incremental --build-dir /tmp/frost-sh-build -o /tmp/inc selfhosted/frost.frost
    echo "C, whole program:"; time ./selfhosted/frost.exe --emit-c --link -o /tmp/cwhole selfhosted/frost.frost
    echo "C, incremental first:"; time ./selfhosted/frost.exe --emit-c --incremental --build-dir /tmp/frost-sh-build-c -o /tmp/cinc selfhosted/frost.frost
    echo "C, incremental unchanged:"; time ./selfhosted/frost.exe --emit-c --incremental --build-dir /tmp/frost-sh-build-c -o /tmp/cinc selfhosted/frost.frost

# Runs all tests
test:
    cargo test -p frost -- --nocapture

# Runs all tests with every imported module reduced to what its interface says
# it offers, which is what --incremental relies on (Unix)
[unix]
test-interfaces:
    FROST_BUILD_FROM_INTERFACES=1 cargo test -p frost

# Runs all tests with every imported module reduced to what its interface says
# it offers, which is what --incremental relies on (Windows)
[windows]
test-interfaces:
    $env:FROST_BUILD_FROM_INTERFACES = "1"; cargo test -p frost; $env:FROST_BUILD_FROM_INTERFACES = $null

# Checks for unused dependencies
udeps:
    cargo machete

# Prints a table of all dependencies and their licenses
licenses:
    cargo license

# Checks for problematic licenses in dependencies
licenses-check:
    cargo deny check licenses

# Displays version information for Rust tools
@versions:
    rustc --version
    cargo fmt -- --version
    cargo clippy -- --version

# Watches for changes and runs tests
watch:
    cargo watch -x 'test -p frost'

# Generates changelog using git-cliff
changelog:
    git cliff -o CHANGELOG.md

# Shows the last tagged commit
show-tag:
    git describe --tags --abbrev=0

# Shows the current version from Cargo.toml (Windows)
[windows]
show-version:
    "v" + (Select-String -Path 'Cargo.toml' -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches.Groups[1].Value

# Shows the current version from Cargo.toml (Unix)
[unix]
show-version:
    @echo "v$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')"

# Deletes a git tag locally and remotely
strip-tag tag:
    git tag -d {{tag}}
    git push origin :refs/tags/{{tag}}

# Pushes a version tag and commits (Windows)
[windows]
push-version:
    $version = (Select-String -Path 'Cargo.toml' -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches.Groups[1].Value; git push origin "v$version"; git push

# Pushes a version tag and commits (Unix)
[unix]
push-version:
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
    git push origin "v$VERSION"
    git push

# Creates a GitHub release for the current version (Windows)
[windows]
publish-release:
    $version = (Select-String -Path 'Cargo.toml' -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches.Groups[1].Value; gh release create "v$version" --title "frost-v$version" --notes-file CHANGELOG.md

# Creates a GitHub release for the current version (Unix)
[unix]
publish-release:
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
    gh release create "v$VERSION" --title "frost-v$VERSION" --notes-file CHANGELOG.md

# Shows the GitHub release for the current version (Windows)
[windows]
show-release:
    $version = (Select-String -Path 'Cargo.toml' -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches.Groups[1].Value; gh release view "v$version"

# Shows the GitHub release for the current version (Unix)
[unix]
show-release:
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
    gh release view "v$VERSION"

# Deletes a GitHub release (by tag, e.g. v0.1.11) (Windows)
[windows]
strip-release tag:
    gh release delete {{tag}} --yes
    Write-Host ""
    Write-Host "To delete the git tag as well, run:" -ForegroundColor Green
    Write-Host "  just strip-tag {{tag}}" -ForegroundColor Green

# Deletes a GitHub release (by tag, e.g. v0.1.11) (Unix)
[unix]
strip-release tag:
    gh release delete {{tag}} --yes
    @echo ""
    @echo "To delete the git tag as well, run:"
    @echo "  just strip-tag {{tag}}"

# Bumps the minor version, updates changelog, and creates a git tag (Windows)
[windows]
bump-minor-version:
    $currentVersion = (Select-String -Path 'Cargo.toml' -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches.Groups[1].Value; $parts = $currentVersion.Split('.'); $newMinor = [int]$parts[1] + 1; $newVersion = "$($parts[0]).$newMinor.0"; Write-Host "Bumping version from $currentVersion to $newVersion"; (Get-Content 'Cargo.toml') -replace "^version = `"$currentVersion`"", "version = `"$newVersion`"" | Set-Content 'Cargo.toml'; git add Cargo.toml; git commit -m "chore: bump version to v$newVersion"; git cliff --tag "v$newVersion" -o CHANGELOG.md; git add CHANGELOG.md; git commit -m "chore: update changelog for v$newVersion"; git tag "v$newVersion"; Write-Host ""; Write-Host "Version bumped and tagged! To push, run:" -ForegroundColor Green; Write-Host "  just push-version" -ForegroundColor Green

# Bumps the minor version, updates changelog, and creates a git tag (Unix)
[unix]
bump-minor-version:
    #!/usr/bin/env bash
    set -euo pipefail
    CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
    IFS='.' read -ra PARTS <<< "$CURRENT_VERSION"
    NEW_MINOR=$((PARTS[1] + 1))
    NEW_VERSION="${PARTS[0]}.$NEW_MINOR.0"
    echo "Bumping version from $CURRENT_VERSION to $NEW_VERSION"
    sed -i "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" Cargo.toml
    git add Cargo.toml
    git commit -m "chore: bump version to v$NEW_VERSION"
    git cliff --tag "v$NEW_VERSION" -o CHANGELOG.md
    git add CHANGELOG.md
    git commit -m "chore: update changelog for v$NEW_VERSION"
    git tag "v$NEW_VERSION"
    echo ""
    echo "Version bumped and tagged! To push, run:"
    echo "  just push-version"

# Bumps the major version, updates changelog, and creates a git tag (Windows)
[windows]
bump-major-version:
    $currentVersion = (Select-String -Path 'Cargo.toml' -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches.Groups[1].Value; $parts = $currentVersion.Split('.'); $newMajor = [int]$parts[0] + 1; $newVersion = "$newMajor.0.0"; Write-Host "Bumping version from $currentVersion to $newVersion"; (Get-Content 'Cargo.toml') -replace "^version = `"$currentVersion`"", "version = `"$newVersion`"" | Set-Content 'Cargo.toml'; git add Cargo.toml; git commit -m "chore: bump version to v$newVersion"; git cliff --tag "v$newVersion" -o CHANGELOG.md; git add CHANGELOG.md; git commit -m "chore: update changelog for v$newVersion"; git tag "v$newVersion"; Write-Host ""; Write-Host "Version bumped and tagged! To push, run:" -ForegroundColor Green; Write-Host "  just push-version" -ForegroundColor Green

# Bumps the major version, updates changelog, and creates a git tag (Unix)
[unix]
bump-major-version:
    #!/usr/bin/env bash
    set -euo pipefail
    CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
    IFS='.' read -ra PARTS <<< "$CURRENT_VERSION"
    NEW_MAJOR=$((PARTS[0] + 1))
    NEW_VERSION="$NEW_MAJOR.0.0"
    echo "Bumping version from $CURRENT_VERSION to $NEW_VERSION"
    sed -i "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" Cargo.toml
    git add Cargo.toml
    git commit -m "chore: bump version to v$NEW_VERSION"
    git cliff --tag "v$NEW_VERSION" -o CHANGELOG.md
    git add CHANGELOG.md
    git commit -m "chore: update changelog for v$NEW_VERSION"
    git tag "v$NEW_VERSION"
    echo ""
    echo "Version bumped and tagged! To push, run:"
    echo "  just push-version"

# Bumps the patch version, updates changelog, and creates a git tag (Windows)
[windows]
bump-patch-version:
    $currentVersion = (Select-String -Path 'Cargo.toml' -Pattern '^version = "(.+)"' | Select-Object -First 1).Matches.Groups[1].Value; $parts = $currentVersion.Split('.'); $newPatch = [int]$parts[2] + 1; $newVersion = "$($parts[0]).$($parts[1]).$newPatch"; Write-Host "Bumping version from $currentVersion to $newVersion"; (Get-Content 'Cargo.toml') -replace "^version = `"$currentVersion`"", "version = `"$newVersion`"" | Set-Content 'Cargo.toml'; git add Cargo.toml; git commit -m "chore: bump version to v$newVersion"; git cliff --tag "v$newVersion" -o CHANGELOG.md; git add CHANGELOG.md; git commit -m "chore: update changelog for v$newVersion"; git tag "v$newVersion"; Write-Host ""; Write-Host "Version bumped and tagged! To push, run:" -ForegroundColor Green; Write-Host "  just push-version" -ForegroundColor Green

# Bumps the patch version, updates changelog, and creates a git tag (Unix)
[unix]
bump-patch-version:
    #!/usr/bin/env bash
    set -euo pipefail
    CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
    IFS='.' read -ra PARTS <<< "$CURRENT_VERSION"
    NEW_PATCH=$((PARTS[2] + 1))
    NEW_VERSION="${PARTS[0]}.${PARTS[1]}.$NEW_PATCH"
    echo "Bumping version from $CURRENT_VERSION to $NEW_VERSION"
    sed -i "s/^version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" Cargo.toml
    git add Cargo.toml
    git commit -m "chore: bump version to v$NEW_VERSION"
    git cliff --tag "v$NEW_VERSION" -o CHANGELOG.md
    git add CHANGELOG.md
    git commit -m "chore: update changelog for v$NEW_VERSION"
    git tag "v$NEW_VERSION"
    echo ""
    echo "Version bumped and tagged! To push, run:"
    echo "  just push-version"
