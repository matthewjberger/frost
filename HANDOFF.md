# Handoff

## Where this is

Item 1 of the LLM-workability brief, "batch diagnostics and a machine-readable
fix channel". Its three stated done conditions are met, the whole suite is green
and both fixpoints hold, so the item is finished except for the one part named
below.

Landed, in four commits on `main`:

- `59a93f1` one run names every fault the checks find, and names each once;
  `--diagnostics=json`
- `7b30ae9` `frost fix` applies the edits the reports carry (also carries the
  CI fix: three `installed_layout` call sites ran a just-copied compiler
  without `run_when_no_longer_busy`, which is the ETXTBSY that failed
  `both_compilers_call_a_c_function_answering_with_a_struct` upstream)
- `25d5192` the self-hosted checks keep walking, and a call is reported where
  it is
- `ba56769` a literal of a name nothing declares is refused, not a crash

## What is left of item 1

`--diagnostics=json` exists in the bootstrap only. The self-hosted compiler
still writes caret reports and nothing else.

Why it was left: the self-hosted compiler composes a report piece by piece
straight to stderr (`report_where` writes the header, the source line and the
caret, then the caller writes the message in pieces, then
`frost_rt_recover_note`, `frost_rt_recover_escape` or `frost_rt_die` ends the
line). A JSON record needs the pieces held until the record is closed, which
means a buffer in the runtime rather than an edit at each of the hundred-odd
report sites.

The shape to build, which needs no report site to change:

1. `runtime/frost_runtime.c`: a JSON mode flag, set once at startup. While it
   is on and a record is open, `frost_rt_error_bytes` and `frost_rt_error_int`
   append to a buffer with JSON escaping instead of writing to stderr.
2. `frost_rt_json_place(path, line, column, offset)` opens a record. Called
   from `report_where` in `selfhosted/imports.frost`, which already computes
   all four; in JSON mode it calls this and returns rather than writing the
   caret block.
3. `frost_rt_recover_note`, `frost_rt_recover_escape` and `frost_rt_die` close
   the open record: one object per line, the field names the bootstrap writes
   (`Report` in `src/diagnostic.rs`).
4. The `fix` field, declared at the report site rather than derived from the
   message text, so the rule lives beside the message: a `report_fix(offset,
   length, replacement)` after the two `mut`-on-a-local reports in
   `selfhosted/parser.frost` covers what the bootstrap offers today.
5. Pin it beside `the_json_reports_round_trip_through_frost_fix` in
   `tests/native.rs`: both compilers write the same records for one program.

Then: `cargo test -r` green, and the fixpoint checked by
`cargo test -r --test native -- self_hosting_is_a_fixpoint
native_self_hosting_is_a_fixpoint`.

## Findings for the item-8 rewrite, verified by probe and not fixed

- **An undeclared type name in a signature is accepted by both compilers.**
  `selfhosted/names.frost`'s `type_code_for` answers `TY_I64` for a name nothing
  declares; the bootstrap's `parse_type` answers `Type::Struct(name)`. So
  `takes :: fn(v: Absent) -> i64 { 0 }` compiles under both, and a hallucinated
  type name in a parameter is silent unless something later happens to ask.
  This is the largest hole found against the thesis and it is a language change:
  it would refuse programs that compile today. The literal case
  (`Absent { a = 2 }`) is closed; the signature case is not.
- **The frame-escape report diverges in wording.** Bootstrap: `region: a
  pointer into the frame of 'escapes' is the call's answer; the storage it names
  dies when the call returns`. Self-hosted: `a pointer into this frame is the
  block's value; the storage it names dies when the call returns`. Both refuse
  the program, so `REFUSED_BY_BOTH` passes on its substring. Pre-existing.
- **A call to an undefined name is worded differently.** Bootstrap: `unknown
  variable 'nosuch'` (it resolves the callee as a name). Self-hosted: `call to
  undefined function 'nosuch'`, which is the better of the two. Both now point
  at the same column.
- **Neither compiler checks a bool against an integer parameter.**
  `takes :: fn(v: i64) -> i64 { v }` called as `takes(true)` compiles under
  both, and so does `1 + true`. Consistent across the two, so not a divergence,
  but it is a hole in the type check.
- **Fault order differs between the compilers** and is not worth forcing: which
  fault comes first follows which walk found it, and the bootstrap checks
  ownership before lowering while the self-hosted compiler counts arguments
  first. `both_compilers_name_the_same_faults_whatever_found_them` compares the
  set, which is the part that matters.
