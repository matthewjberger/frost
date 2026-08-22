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
- **Everything the compiler knows**, from one `frostc lsp` started once and
  asked over its own streams. The passes that build a program are what answer,
  so a report the editor underlines is a report the build refuses on, and a
  definition is the row in the symbol table rather than a line a pattern
  matched. What it serves:

  - Reports in the Problems panel, published as you type. A burst of keystrokes
    is one check: what is already on the way in says which change is the last
    one. A file an import reports about gets its own entry.
  - Quick fixes, from the edit a report carries. The compiler works out the span
    to replace and what goes there, and says whether it applies the edit unread;
    one it is sure of is the preferred action. `Frost: Apply every fix the
    reports offer` applies all of those at once.
  - Go to definition, hover with the declaration head and the comment above it,
    find references, and the highlight under every other place a name is
    written.
  - The outline, and workspace search over every open file.
  - Rename, which changes every place a name is written: the namespace is flat,
    so a name means one thing across the whole program.
  - Completion over the names the workspace declares, each carrying the head its
    author wrote.
  - Folding ranges, and formatting, which is the same code `frostc fmt --check`
    runs in a build.

- **A problem matcher**, `$frost`, which turns a located compiler report into an
  entry in the Problems panel.

The compiler it talks to is the **self-hosted** one. The bootstrap is a complete
Frost compiler and its one job is to build that one; a tool is written in Frost
and lives there alone. `just install-self` puts it on PATH as `frostc`, which is
what `frost.compilerPath` names by default. A bare name is looked up on PATH.

`Frost: Restart the language server` starts a fresh one, which is what to reach
for after rebuilding the compiler.

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
