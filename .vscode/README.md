# Editor support

`.vscode/frost` is a VS Code extension for the Frost language: a TextMate
grammar for `.frost` files plus the comment, bracket and indentation rules. It
covers comments, strings and their escapes, numbers, every keyword and primitive
type, `$T` type parameters, `name :: fn` and `name :: struct` declaration heads,
`.Variant` patterns, the compiler builtins (`ptr_to`, `ptr_cast`, `slice_from`,
`slice_len`), calls, and field access.

VS Code will not load an extension out of a workspace folder, so it has to be
linked or copied into the extensions directory once:

```
just install-editor
```

Then reload the window (`Ctrl+Shift+P`, "Developer: Reload Window"); if `Frost`
still is not in the language list, quit VS Code and reopen it, since the
extension scan is cached. To remove it, `just uninstall-editor`.

The extensions directory is not always `~/.vscode/extensions`. A portable
install (which is what scoop and the zip download give you) keeps its extensions
next to the executable in `data/extensions` instead, and copying to the wrong
one silently does nothing. `just editor-dir` prints the one this `code` on
`PATH` actually reads, which is what the install recipe uses.

The copy on Windows is a snapshot, so rerun `just install-editor` after changing
the grammar. The symlink on Linux and macOS picks up changes on the next reload.

## Working on the grammar

`Ctrl+Shift+P`, "Developer: Inspect Editor Tokens and Scopes" shows the scope
stack under the cursor, which is how to check a rule fires where it should. The
rules live in `.vscode/frost/syntaxes/frost.tmLanguage.json` and are tried in
the order the top-level `patterns` array lists them, leftmost match winning.
