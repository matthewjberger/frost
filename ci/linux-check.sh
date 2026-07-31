#!/bin/sh
# The gates the Linux runner runs, in the order it runs them. Meant for a
# container: `just ci-linux` is what starts one. See `.github/workflows/rust.yml`
# for the job this mirrors.
set -e

apt-get update -qq
# gcc and binutils are what a build and the assembler oracle need. clang is what
# the ELF oracle needs, and without it that test skips itself, which is the
# difference between checking the object this compiler writes and not.
apt-get install -y -qq gcc binutils clang
rustup component add clippy >/dev/null 2>&1 || true

# The runner leaves the target directory inside the checkout, which is how
# `bundled_std` finds the standard library: two directories up from the compiler
# binary. This build puts the target directory elsewhere so the host's own is
# left alone, so the library has to be named instead. Without this a handful of
# tests fail for a reason belonging to this script rather than to the code.
export FROST_STD=/w/std
# A toolchain that went missing would otherwise hide behind tests that skip
# themselves, and the run would be green for the wrong reason.
export FROST_REQUIRE_LINKER=1

echo "=== build"
cargo build --verbose >/dev/null

echo "=== clippy"
cargo clippy --all-targets -- -D warnings

echo "=== tests"
cargo test

echo "=== tests, imports built from what their interfaces say"
FROST_BUILD_FROM_INTERFACES=1 cargo test
