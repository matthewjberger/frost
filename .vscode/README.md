# Editor support

`.vscode/frost` is a VS Code extension for the Frost language. It contributes:

- **A TextMate grammar** for `.frost`, plus the comment, bracket and indentation
  rules. It covers comments, strings and their escapes, numbers, every keyword
  and primitive type, `$T` type parameters, `name :: fn` and `name :: struct`
  declaration heads, `Type::Variant` and `.Variant` patterns, the compiler
  builtins, calls, and field access.
- **An injection grammar** so a fenced block tagged `frost` in a markdown file
  is highlighted with the same rules.
- **Snippets** for the declaration forms, `match`, the loops, `with`, `unsafe`,
  `defer` and `test`.
- **A schema** for `frost.json`, so the manifest completes and validates.
- **Formatting**, which runs `frost fmt -` over the buffer, so Format Document
  and format-on-save write what a build's `frost fmt --check` accepts. The
  editor and the build run the same code rather than keeping two of it.
  `frost.compilerPath` names the compiler; a bare name is looked up on PATH.
- **A problem matcher**, `$frost`, which turns a located compiler diagnostic
  into an entry in the Problems panel.

`.vscode/tasks.json` uses that matcher. `Ctrl+Shift+B` runs every compiler check
over the open file, and errors land on the line that caused them.

## Installing

VS Code will not load an extension out of a workspace folder, so it has to be
linked or copied into the extensions directory once:

```
just install-editor
```

Then reload the window (`Ctrl+Shift+P`, "Developer: Reload Window"). If `Frost`
still is not in the language list, quit VS Code and reopen it, since the
extension scan is cached, and a change to `package.json` needs the manifest read
again rather than only the window redrawn. To remove it, `just uninstall-editor`.

The extensions directory is not always `~/.vscode/extensions`. A portable
install (which is what scoop and the zip download give you) keeps its extensions
next to the executable in `data/extensions` instead, and copying to the wrong
one silently does nothing. `just editor-dir` prints the one this `code` on
`PATH` actually reads, which is what the install recipe uses.

The copy on Windows is a snapshot, so rerun `just install-editor` after changing
anything here. The symlink on Linux and macOS picks up changes on the next
reload.

## Working on the grammar

`Ctrl+Shift+P`, "Developer: Inspect Editor Tokens and Scopes" shows the scope
stack under the cursor, which is how to check a rule fires where it should. The
rules live in `.vscode/frost/syntaxes/frost.tmLanguage.json` and are tried in
the order the top-level `patterns` array lists them, leftmost match winning.

The keyword and builtin lists are not maintained by hand. `tests/editor_grammar.rs`
reads this grammar and compares it against `KEYWORD_NAMES` and
`BUILTIN_FUNCTIONS` in the compiler, failing with the words that drifted in
either direction. Adding a keyword to the lexer and not to the grammar is a
test failure, which is how `assert` and `str_len` were found missing.

Two rules are conventions rather than grammar, because the language cannot
distinguish them and the corpus can. A capitalized name reads as a type, and
`::` written tight (`Maybe::Just`) reads as enum construction where `::` written
spaced reads as a declaration head. Both hold across every Frost file in the
tree. A formatter that normalized the spacing around `::` would break the second
one everywhere at once.
