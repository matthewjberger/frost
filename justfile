set windows-shell := ["powershell.exe"]
export RUST_BACKTRACE := "1"

# Displays the list of available commands
@just:
    just --list

# Builds the project in release mode
build:
    cargo build -r

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

# Builds the self-hosted compiler (frost written in frost)
selfhost-build:
    cargo run -r -q -p frost --bin frost -- --link -o bootstrap/frost.exe bootstrap/frost.frost

# Compiles a frost file with the self-hosted compiler, via its C backend (Unix)
[unix]
selfhost-run file: selfhost-build
    FROST_INPUT={{file}} ./bootstrap/frost.exe

# Compiles a frost file with the self-hosted compiler, via its C backend (Windows)
[windows]
selfhost-run file: selfhost-build
    $env:FROST_INPUT = "{{file}}"; ./bootstrap/frost.exe

# Compiles a frost file with the self-hosted native backend, then assembles and runs it (Unix)
[unix]
selfhost-native file: selfhost-build
    #!/usr/bin/env bash
    set -euo pipefail
    FROST_BACKEND=asm FROST_INPUT={{file}} ./bootstrap/frost.exe > {{file}}.s
    cc {{file}}.s -o {{file}}.exe
    ./{{file}}.exe

# Compiles a frost file with the self-hosted native backend, then assembles and runs it (Windows)
[windows]
selfhost-native file: selfhost-build
    $env:FROST_BACKEND = "asm"; $env:FROST_INPUT = "{{file}}"; $asm = & ./bootstrap/frost.exe; $env:FROST_BACKEND = $null; [System.IO.File]::WriteAllLines((Resolve-Path .).Path + "/{{file}}.s", $asm); gcc "{{file}}.s" -o "{{file}}.exe"; & "./{{file}}.exe"

# Runs every self-hosted example through the native backend (Windows)
[windows]
selfhost-examples: selfhost-build
    Get-ChildItem examples/selfhosted/*.frost | ForEach-Object { Write-Host "== $($_.Name)"; $env:FROST_BACKEND = "asm"; $env:FROST_INPUT = $_.FullName; $asm = & ./bootstrap/frost.exe; $env:FROST_BACKEND = $null; [System.IO.File]::WriteAllLines($_.FullName + ".s", $asm); gcc ($_.FullName + ".s") -o ($_.FullName + ".exe"); & ($_.FullName + ".exe"); Remove-Item ($_.FullName + ".s"), ($_.FullName + ".exe") -Force }

# Runs every self-hosted example through the native backend (Unix)
[unix]
selfhost-examples: selfhost-build
    #!/usr/bin/env bash
    set -euo pipefail
    for f in examples/selfhosted/*.frost; do
        echo "== $f"
        FROST_BACKEND=asm FROST_INPUT="$f" ./bootstrap/frost.exe > "$f.s"
        cc "$f.s" -o "$f.exe"
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

# Runs all tests
test:
    cargo test -p frost -- --nocapture

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
