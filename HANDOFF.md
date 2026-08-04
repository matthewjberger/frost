# Handoff

## Where this is

Item 1 of the LLM-workability brief, "batch diagnostics and a machine-readable
fix channel". Its three stated done conditions are met and the whole suite is
green, so the item is finished except for one part named below.

Landed, in three commits:

- `59a93f1` one run names every fault the checks find, and names each once;
  `--diagnostics=json`
- `7b30ae9` `frost fix` applies the edits the reports carry
- `25d5192` the self-hosted checks keep walking, and a call is reported where
  it is

## What is left of item 1

`--diagnostics=json` exists in the bootstrap only. The self-hosted compiler
still writes caret reports and nothing else.

Why it was left: the self-hosted compiler composes a report piece by piece
straight to stderr (`report_where` writes the header, the source line and the
caret, then the caller writes the message in pieces, then
`frost_rt_recover_note`/`_escape`/`frost_rt_die` ends the line). A JSON record
needs the pieces held until the record is closed, which means a buffer in the
runtime rather than a change at each of the hundred-odd report sites.

The shape to build, which needs no report site to change:

1. `runtime/frost_runtime.c`: a JSON mode flag, set once at startup. While it
   is on and a record is open, `frost_rt_error_bytes` and `frost_rt_error_int`
   append to a buffer with JSON escaping instead of writing to stderr.
2. `frost_rt_json_place(path, line, column, offset)`: opens a record. Called
   from `report_where` in `selfhosted/imports.frost`, which already computes
   all four; in JSON mode it calls this and returns instead of writing the
   caret block.
3. `frost_rt_recover_note`, `frost_rt_recover_escape` and `frost_rt_die` close
   the open record: one object per line, the same field names the bootstrap
   writes (`src/diagnostic.rs`, `Report`).
4. The `fix` field: declared at the report site rather than derived from the
   message text, so the rule lives with the message. A `report_fix(offset,
   length, replacement)` after the two `mut`-on-a-local reports in
   `selfhosted/parser.frost` (lines near 2884 and 2923) covers what the
   bootstrap offers today.
5. Pin it: a test beside `the_json_reports_round_trip_through_frost_fix` in
   `tests/native.rs` asserting both compilers write the same records for one
   program.

The fixpoint must be green afterwards: `cargo test -r --test native -- 
self_hosting_is_a_fixpoint native_self_hosting_is_a_fixpoint`.

## Findings to carry into the report or a later item

Verified by probe, not fixed here:

- An undeclared type name is accepted by both compilers. The self-hosted
  `type_code_for` (`selfhosted/names.frost`) answers `TY_I64` for a name
  nothing declares; the bootstrap's `parse_type` answers `Type::Struct(name)`.
  So `takes :: fn(v: Absent) -> i64 { 0 }` compiles under both, and a
  hallucinated type name in a parameter is silent. This is the largest hole
  found this session against the thesis, and it is a language change: it would
  refuse programs that compile today.
- The frame-escape report diverges in wording between the compilers. Bootstrap:
  `region: a pointer into the frame of 'escapes' is the call's answer; ...`.
  Self-hosted: `a pointer into this frame is the block's value; ...`. Both
  refuse the program, so `REFUSED_BY_BOTH` passes on a substring; the words
  differ. Pre-existing.
- A struct literal naming an undeclared type crashed the self-hosted compiler
  with `an arena was indexed out of range: -16` (`struct_index_of` of type 0,
  with `STRUCT_BASE` at 16). Not reproduced since the call-position change;
  re-probe with `main :: fn() -> i64 { held := Absent { a = 2 }\n 0 }` before
  calling it gone.
