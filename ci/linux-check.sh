#!/bin/sh
# The gates the Linux runner runs, in the order it runs them. Meant for a
# container: `just ci-linux` is what starts one. See `.github/workflows/rust.yml`
# for the job this mirrors.
set -e

apt-get update -qq
# gcc and binutils are what a build and the assembler oracle need. clang is what
# the ELF oracle needs, and without it that test skips itself, which is the
# difference between checking the object this compiler writes and not. nodejs
# runs the two drivers that ask the editor support its questions.
apt-get install -y -qq gcc binutils clang nodejs
rustup component add clippy >/dev/null 2>&1 || true

# The runner leaves the target directory inside the checkout, which is how
# `bundled_std` finds the standard library: two directories up from the compiler
# binary. This build puts the target directory elsewhere so the host's own is
# left alone, so the library has to be named instead. Without this a handful of
# tests fail for a reason belonging to this script rather than to the code.
export FROST_STD=/w/std
# The Frost half of the runtime is found the same way and moves for the same
# reason. It is also the one file allowed to define names in the runtime's own
# name space, and that permission is given to whichever file this compiler
# resolved as the runtime, so a compiler that cannot find it refuses the runtime
# for holding the names it exists to hold.
export FROST_RUNTIME_FROST=/w/runtime/runtime.frost
# A toolchain that went missing would otherwise hide behind tests that skip
# themselves, and the run would be green for the wrong reason.
export FROST_REQUIRE_LINKER=1
# And so would a missing JavaScript runtime, which is what the editor test
# needs to reach either half of the editor support.
export FROST_REQUIRE_NODE=1

echo "=== build"
cargo build --verbose >/dev/null

echo "=== clippy"
cargo clippy --all-targets -- -D warnings

echo "=== tests"
cargo test

echo "=== tests, imports built from what their interfaces say"
FROST_BUILD_FROM_INTERFACES=1 cargo test
