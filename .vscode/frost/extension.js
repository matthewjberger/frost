// Navigation over the declaration syntax alone: every declaration head opens
// its line and the namespace is flat, so a workspace search for
// `^\s*name\s*::` is definition lookup. No language server, no dependencies;
// the extension stays a plain file copy under `just install-editor`.
//
// What it cannot answer from the text it asks the compiler, and the compiler it
// asks is the self-hosted one. The bootstrap's job is to build that compiler and
// to be held to the same language; a tool written in Frost lives in the
// self-hosted compiler alone and has no twin there.
const vscode = require("vscode");

const FUNCTION_HEAD =
  /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*::\s*(?:(?:safe|extern|inline|unsafe)\s+)*fn\b/;
const TYPE_HEAD =
  /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*::\s*(?:(?:linear|distinct)\s+)*(struct|enum|flags|type|distinct)\b/;
const TEST_HEAD = /^\s*test\s+"([^"]*)"/;
// A constant head is spaced where `Type::Variant` is written tight, the same
// rule the grammar uses to tell the two apart.
const CONSTANT_HEAD = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s+::/;

const TYPE_KINDS = {
  struct: vscode.SymbolKind.Struct,
  enum: vscode.SymbolKind.Enum,
  flags: vscode.SymbolKind.Enum,
  type: vscode.SymbolKind.Interface,
  distinct: vscode.SymbolKind.Class,
};

const EXCLUDED_DIRECTORIES = [".frost-build", "target", ".git"];

// How the compiler ends a line of JSON, either way it was written.
const LINE_BREAKS = /\r?\n/;

const files = new Map();
let ready = Promise.resolve();

function isExcluded(uri) {
  const segments = uri.fsPath.split(/[\\/]/);
  return EXCLUDED_DIRECTORIES.some((directory) => segments.includes(directory));
}

function classifyHead(line, lineIndex) {
  let match = line.match(FUNCTION_HEAD);
  if (match) {
    return headOf(match[1], vscode.SymbolKind.Function, line, lineIndex);
  }
  match = line.match(TYPE_HEAD);
  if (match) {
    return headOf(match[1], TYPE_KINDS[match[2]], line, lineIndex);
  }
  match = line.match(TEST_HEAD);
  if (match) {
    const head = headOf("test", vscode.SymbolKind.Event, line, lineIndex);
    head.name = `test "${match[1]}"`;
    return head;
  }
  match = line.match(CONSTANT_HEAD);
  if (match) {
    return headOf(match[1], vscode.SymbolKind.Constant, line, lineIndex);
  }
  return null;
}

function headOf(name, kind, line, lineIndex) {
  const startCharacter = line.indexOf(name);
  return {
    name,
    kind,
    line: lineIndex,
    startCharacter,
    endCharacter: startCharacter + name.length,
    endLine: lineIndex,
  };
}

// One pass over a line: how its braces move the depth, and which of its columns
// are code rather than the inside of a string or a comment. Both answers come
// from the same walk, so a reader of one cannot disagree with a reader of the
// other about where a string ends.
function readLine(line, state) {
  let delta = 0;
  const code = [];
  let open = state.inString || state.inBlockComment ? -1 : 0;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (state.inBlockComment) {
      if (character === "*" && line[index + 1] === "/") {
        state.inBlockComment = false;
        index += 1;
        open = index + 1;
      }
      continue;
    }
    if (state.inString) {
      if (character === "\\") {
        index += 1;
      } else if (character === '"') {
        state.inString = false;
        open = index + 1;
      }
      continue;
    }
    if (character === "/" && line[index + 1] === "/") {
      if (open >= 0) {
        code.push([open, index]);
        open = -1;
      }
      return { delta, code };
    }
    if (character === "/" && line[index + 1] === "*") {
      state.inBlockComment = true;
      if (open >= 0) {
        code.push([open, index]);
        open = -1;
      }
      index += 1;
      continue;
    }
    if (character === '"') {
      state.inString = true;
      if (open >= 0) {
        code.push([open, index]);
        open = -1;
      }
    } else if (character === "{") {
      delta += 1;
    } else if (character === "}") {
      delta -= 1;
    }
  }
  if (open >= 0) {
    code.push([open, line.length]);
  }
  return { delta, code };
}

function braceDelta(line, state) {
  return readLine(line, state).delta;
}

// Where a name is written as a name. A file's lines are walked in order, since
// a string or a block comment opened on one line runs into the next.
function codeRunsOf(lines) {
  const state = { inString: false, inBlockComment: false };
  return lines.map((line) => readLine(line, state).code);
}

function insideCode(runs, column, length) {
  return runs.some(([start, end]) => column >= start && column + length <= end);
}

// One pass over the file, tracking strings, comments and brace depth, so a
// head is only read at the top level and a body's extent is known for the
// outline. A head that never opens a brace ends on its own line.
function scanDeclarations(lines) {
  const declarations = [];
  const state = { inString: false, inBlockComment: false };
  let depth = 0;
  let current = null;
  let started = false;
  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex];
    if (depth === 0 && !state.inString && !state.inBlockComment) {
      const head = classifyHead(line, lineIndex);
      if (head) {
        if (current) {
          current.endLine = started ? lineIndex - 1 : current.line;
          declarations.push(current);
        }
        current = head;
        started = false;
      }
    }
    depth += braceDelta(line, state);
    if (depth < 0) {
      depth = 0;
    }
    if (current) {
      if (depth > 0) {
        started = true;
      } else if (started) {
        current.endLine = lineIndex;
        declarations.push(current);
        current = null;
      }
    }
  }
  if (current) {
    current.endLine = started ? lines.length - 1 : current.line;
    declarations.push(current);
  }
  return declarations;
}

function setFile(uri, text) {
  const lines = text.split(/\r?\n/);
  files.set(uri.toString(), {
    uri,
    lines,
    declarations: scanDeclarations(lines),
  });
}

async function indexFile(uri) {
  if (isExcluded(uri)) {
    return;
  }
  let bytes;
  try {
    bytes = await vscode.workspace.fs.readFile(uri);
  } catch {
    files.delete(uri.toString());
    return;
  }
  setFile(uri, new TextDecoder("utf-8").decode(bytes));
}

function indexDocument(document) {
  if (document.uri.scheme !== "file" || isExcluded(document.uri)) {
    return;
  }
  setFile(document.uri, document.getText());
}

async function indexWorkspace() {
  const uris = await vscode.workspace.findFiles(
    "**/*.frost",
    "{**/.frost-build/**,**/target/**,**/.git/**}"
  );
  await Promise.all(uris.map(indexFile));
  for (const document of vscode.workspace.textDocuments) {
    if (document.languageId === "frost") {
      indexDocument(document);
    }
  }
}

// A field or member access cannot resolve without types, so a word directly
// after a single dot is skipped rather than guessed at. Two dots are a range,
// whose right operand is an ordinary name.
function wordAt(document, position) {
  const range = document.getWordRangeAtPosition(position);
  if (!range) {
    return null;
  }
  let word = document.getText(range);
  if (word.startsWith("$")) {
    word = word.slice(1);
  }
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(word)) {
    return null;
  }
  const start = range.start;
  if (start.character > 0) {
    const before = document.getText(
      new vscode.Range(
        start.line,
        Math.max(0, start.character - 2),
        start.line,
        start.character
      )
    );
    if (before.endsWith(".") && !before.endsWith("..")) {
      return null;
    }
  }
  return word;
}

function lookup(name) {
  const results = [];
  for (const entry of files.values()) {
    for (const declaration of entry.declarations) {
      if (declaration.name === name) {
        results.push({ entry, declaration });
      }
    }
  }
  return results;
}

function locationOf(entry, declaration) {
  return new vscode.Location(
    entry.uri,
    new vscode.Range(
      declaration.line,
      declaration.startCharacter,
      declaration.line,
      declaration.endCharacter
    )
  );
}

function signatureOf(entry, declaration) {
  const lines = entry.lines;
  let text = lines[declaration.line];
  if (declaration.kind === vscode.SymbolKind.Function) {
    let depth = parenthesisDelta(text);
    let lineIndex = declaration.line;
    while (
      depth > 0 &&
      lineIndex + 1 < lines.length &&
      lineIndex - declaration.line < 12
    ) {
      lineIndex += 1;
      text += "\n" + lines[lineIndex];
      depth += parenthesisDelta(lines[lineIndex]);
    }
  }
  return text.replace(/\s*\{\s*$/, "");
}

function parenthesisDelta(line) {
  let delta = 0;
  for (const character of line) {
    if (character === "(") {
      delta += 1;
    } else if (character === ")") {
      delta -= 1;
    }
  }
  return delta;
}

function commentAbove(lines, headLine) {
  const collected = [];
  for (let lineIndex = headLine - 1; lineIndex >= 0; lineIndex -= 1) {
    const match = lines[lineIndex].match(/^\s*\/\/ ?(.*)$/);
    if (!match) {
      break;
    }
    collected.unshift(match[1]);
  }
  return collected.join("\n").trim();
}

function matchesQuery(needle, haystack) {
  let position = 0;
  for (const character of needle) {
    position = haystack.indexOf(character, position);
    if (position < 0) {
      return false;
    }
    position += 1;
  }
  return true;
}

const documentSymbolProvider = {
  provideDocumentSymbols(document) {
    const lines = document.getText().split(/\r?\n/);
    return scanDeclarations(lines).map((declaration) => {
      const endLine = Math.min(declaration.endLine, lines.length - 1);
      const symbol = new vscode.DocumentSymbol(
        declaration.name,
        "",
        declaration.kind,
        new vscode.Range(
          declaration.line,
          0,
          endLine,
          lines[endLine].length
        ),
        new vscode.Range(
          declaration.line,
          declaration.startCharacter,
          declaration.line,
          declaration.endCharacter
        )
      );
      return symbol;
    });
  },
};

const definitionProvider = {
  async provideDefinition(document, position) {
    await ready;
    const word = wordAt(document, position);
    if (!word) {
      return undefined;
    }
    return lookup(word).map(({ entry, declaration }) =>
      locationOf(entry, declaration)
    );
  },
};

const workspaceSymbolProvider = {
  async provideWorkspaceSymbols(query) {
    await ready;
    const needle = query.toLowerCase();
    const symbols = [];
    for (const entry of files.values()) {
      const container = vscode.workspace.asRelativePath(entry.uri);
      for (const declaration of entry.declarations) {
        if (!matchesQuery(needle, declaration.name.toLowerCase())) {
          continue;
        }
        symbols.push(
          new vscode.SymbolInformation(
            declaration.name,
            declaration.kind,
            container,
            locationOf(entry, declaration)
          )
        );
      }
    }
    return symbols;
  },
};

const referenceProvider = {
  async provideReferences(document, position, context) {
    await ready;
    const range = document.getWordRangeAtPosition(position);
    if (!range) {
      return [];
    }
    let word = document.getText(range);
    if (word.startsWith("$")) {
      word = word.slice(1);
    }
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(word)) {
      return [];
    }
    const pattern = new RegExp("\\b" + word + "\\b", "g");
    const locations = [];
    for (const entry of files.values()) {
      // A name written in a comment or inside a string is prose, not a use of
      // the declaration. Counted, `// vec_push is what to call` sent a reader
      // to a line that names nothing.
      const runs = codeRunsOf(entry.lines);
      for (let lineIndex = 0; lineIndex < entry.lines.length; lineIndex += 1) {
        const line = entry.lines[lineIndex];
        pattern.lastIndex = 0;
        let match;
        while ((match = pattern.exec(line)) !== null) {
          if (!insideCode(runs[lineIndex], match.index, word.length)) {
            continue;
          }
          if (
            !context.includeDeclaration &&
            entry.declarations.some(
              (declaration) =>
                declaration.line === lineIndex &&
                declaration.startCharacter === match.index &&
                declaration.name === word
            )
          ) {
            continue;
          }
          locations.push(
            new vscode.Location(
              entry.uri,
              new vscode.Range(
                lineIndex,
                match.index,
                lineIndex,
                match.index + word.length
              )
            )
          );
        }
      }
    }
    return locations;
  },
};

const hoverProvider = {
  async provideHover(document, position) {
    await ready;
    const word = wordAt(document, position);
    if (!word) {
      return undefined;
    }
    const hits = lookup(word);
    if (hits.length === 0) {
      return undefined;
    }
    const local = hits.find(
      ({ entry }) => entry.uri.toString() === document.uri.toString()
    );
    const { entry, declaration } = local || hits[0];
    const markdown = new vscode.MarkdownString();
    markdown.appendCodeblock(signatureOf(entry, declaration), "frost");
    const comment = commentAbove(entry.lines, declaration.line);
    if (comment) {
      markdown.appendMarkdown(comment);
    }
    return new vscode.Hover(markdown);
  },
};

// Formatting is `frost fmt -`: the compiler reads the buffer on standard input
// and writes its one rendering to standard output, so the editor and the build
// agree by running the same code rather than by keeping two of it. The whole
// document is replaced.
//
// A compiler that cannot be run leaves the buffer alone and says so. Said
// rather than swallowed: returning nothing is what an already-formatted buffer
// returns too, so a `frost` that is not on PATH looked exactly like a file with
// nothing to change, and the setting to point at one is named in the message.
const documentFormattingProvider = {
  provideDocumentFormattingEdits(document) {
    const compiler = vscode.workspace
      .getConfiguration("frost")
      .get("compilerPath", "frost");
    let written;
    try {
      written = require("child_process").execFileSync(compiler, ["fmt", "-"], {
        input: document.getText(),
        encoding: "utf8",
        maxBuffer: 64 * 1024 * 1024,
      });
    } catch (error) {
      const reason =
        error && error.code === "ENOENT"
          ? `'${compiler}' was not found; run 'just install-self' or set frost.compilerPath to the self-hosted compiler`
          : (error && error.stderr) || (error && error.message) || "it failed";
      vscode.window.showErrorMessage(`frost fmt did not run: ${reason}`);
      return undefined;
    }
    if (written === document.getText()) {
      return undefined;
    }
    const whole = new vscode.Range(
      document.positionAt(0),
      document.positionAt(document.getText().length)
    );
    return [vscode.TextEdit.replace(whole, written)];
  },
};

// What the compiler says about a file, published where the editor shows
// problems. Two passes, because they answer different questions: a check reports
// what a build refuses on and what it warns about, and `lint` reports the
// findings a build says nothing about at all.
const findings = vscode.languages.createDiagnosticCollection("frost");

// The edits the reports carried, by file, for the code actions to offer. Held
// beside the diagnostics rather than on them: what the editor hands back to a
// code action is its own copy of a diagnostic, so a property put on one here
// does not survive the trip.
const offered = new Map();

// The compiler this extension runs. The self-hosted one, which is where the
// tools live: the bootstrap's job is to build it, and a tool written in Frost
// has no twin there. `just install-self` puts it on PATH under this name.
function compilerPath() {
  return vscode.workspace
    .getConfiguration("frost")
    .get("compilerPath", "frostc");
}

// The compiler's output, whichever stream it wrote on and whether or not it
// ended well. A refused build is a nonzero exit and its reports are what to
// read, so the failure carries the answer.
//
// Run without waiting on it. A check and a lint are a compiler apiece, and run
// where the editor waits they are two builds between one keystroke and the
// next.
function runCompiler(arguments_, options) {
  const held = {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    ...(options || {}),
  };
  return new Promise((answer) => {
    require("child_process").execFile(
      compilerPath(),
      arguments_,
      held,
      (error, out, err) => {
        answer({
          out: typeof out === "string" ? out : "",
          err: typeof err === "string" ? err : "",
          failed: Boolean(error),
        });
      }
    );
  });
}

function reportsIn(text) {
  const held = [];
  for (const line of (text || "").split(LINE_BREAKS)) {
    if (!line.trim().startsWith("{")) {
      continue;
    }
    try {
      held.push(JSON.parse(line));
    } catch (error) {
      continue;
    }
  }
  return held;
}

// A span in a report counts bytes and a position counts UTF-16 units, and the
// two agree only while every byte is one. A file holding a character above
// ASCII is read by its line and column instead, which is exact either way.
function bytesAreUnits(text) {
  return !/[^\x00-\x7F]/.test(text);
}

// What to underline. A report's own span is a point at the place it names, so
// the span of the edit it offers is preferred where there is one, and the word
// under the place where there is neither. A zero-width range draws nothing.
function rangeOf(document, report, ascii) {
  const at = new vscode.Position(
    Math.max(0, (report.line || 1) - 1),
    Math.max(0, (report.column || 1) - 1)
  );
  const spans = [report.fix && report.fix.span, report.span];
  if (ascii) {
    for (const span of spans) {
      if (Array.isArray(span) && span[1] > span[0]) {
        return new vscode.Range(
          document.positionAt(span[0]),
          document.positionAt(span[1])
        );
      }
    }
  }
  return document.getWordRangeAtPosition(at) || new vscode.Range(at, at);
}

async function publishFindings(document) {
  if (document.languageId !== "frost" || document.uri.scheme !== "file") {
    return;
  }
  const path = document.uri.fsPath;
  // The check writes its reports on the error stream and `lint` writes its own
  // on the output stream. Neither writes a file: a build that is refused stops
  // before it emits, and one that is not was asked for no output.
  // Run where the project is. The last two roots a compiler searches are named
  // relative to the directory it was started in, so a project that keeps its own
  // libraries beside its manifest is only reachable from there.
  const folder = vscode.workspace.getWorkspaceFolder(document.uri);
  const at = folder ? { cwd: folder.uri.fsPath } : {};
  const [check, lint] = await Promise.all([
    runCompiler([path, "--diagnostics=json"], at),
    runCompiler(["lint", "--diagnostics=json", path], at),
  ]);
  const checked = check.err;
  const linted = lint.out;
  const ascii = bytesAreUnits(document.getText());
  const found = [];
  const fixes = [];
  const seen = new Set();
  for (const report of reportsIn(checked).concat(reportsIn(linted))) {
    // The two passes overlap on the warnings, which both of them find.
    const key = `${report.line}:${report.column}:${report.message}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    const range = rangeOf(document, report, ascii);
    const severity =
      report.severity === "error"
        ? vscode.DiagnosticSeverity.Error
        : vscode.DiagnosticSeverity.Warning;
    const held = new vscode.Diagnostic(
      range,
      report.message || "",
      severity
    );
    held.source = "frost";
    // Another place the same report is about, as a link the reader can follow.
    if (Array.isArray(report.related) && report.related.length > 0) {
      held.relatedInformation = report.related
        .filter((place) => place && typeof place.line === "number")
        .map((place) => {
          const where = new vscode.Position(
            Math.max(0, place.line - 1),
            Math.max(0, (place.column || 1) - 1)
          );
          return new vscode.DiagnosticRelatedInformation(
            new vscode.Location(
              place.file ? vscode.Uri.file(place.file) : document.uri,
              new vscode.Range(where, where)
            ),
            place.message || ""
          );
        });
    }
    found.push(held);
    if (report.fix && ascii && Array.isArray(report.fix.span)) {
      fixes.push({
        message: held.message,
        range,
        edit: new vscode.Range(
          document.positionAt(report.fix.span[0]),
          document.positionAt(report.fix.span[1])
        ),
        replacement: report.fix.replacement || "",
        certain: report.fix.certain === true,
      });
    }
  }
  findings.set(document.uri, found);
  offered.set(document.uri.toString(), fixes);
}

// The edit a report carried, offered where the report is. The compiler worked
// it out; without this it was written into the JSON and read by nothing.
const codeActionProvider = {
  provideCodeActions(document, range, context) {
    const held = offered.get(document.uri.toString()) || [];
    const actions = [];
    for (const fix of held) {
      if (!fix.range.intersection(range)) {
        continue;
      }
      const shown = fix.replacement
        ? `Replace with '${fix.replacement}'`
        : "Remove this";
      const action = new vscode.CodeAction(
        shown,
        vscode.CodeActionKind.QuickFix
      );
      action.edit = new vscode.WorkspaceEdit();
      action.edit.replace(document.uri, fix.edit, fix.replacement);
      // A fix the compiler applies unread is the one to reach for, and one it
      // offers is a guess at what was meant.
      action.isPreferred = fix.certain;
      action.diagnostics = context.diagnostics.filter(
        (diagnostic) => diagnostic.message === fix.message
      );
      actions.push(action);
    }
    return actions;
  },
};

// The exported surface under a prefix, from `frost api`, which reads the
// program rather than the text. Asked once for the first two characters and
// narrowed here, since the answer for `ve` holds every answer for `vec_`.
const surface = new Map();

// A function's head without the body, which an `inline fn` carries on the same
// line. The brace that opens a body is the first one written outside every
// bracket, so a `where` clause naming a call keeps its own and a body holding
// one does not put the cut inside itself.
//
// Only a function. A struct's braces hold its fields, which is what a reader
// asking about a struct wants to read.
function withoutBody(text) {
  if (!/::\s*(?:[a-z]+\s+)*fn\b/.test(text)) {
    return text;
  }
  let depth = 0;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === "(" || character === "[") {
      depth += 1;
    } else if (character === ")" || character === "]") {
      depth -= 1;
    } else if (character === "{" && depth === 0) {
      return text.slice(0, index).trimEnd();
    }
  }
  return text;
}

async function surfaceUnder(prefix, directory) {
  const key = `${directory} ${prefix}`;
  const cached = surface.get(key);
  if (cached && Date.now() - cached.at < 5000) {
    return cached.items;
  }
  const written = (await runCompiler(["api", prefix], { cwd: directory })).out;
  const lines = written.split(LINE_BREAKS);
  const items = [];
  for (let index = 0; index < lines.length; index += 1) {
    if (!/^(.+):(\d+)$/.test(lines[index])) {
      continue;
    }
    // A head carries over the lines its parameters run onto, and the blank line
    // after it is what ends it.
    const parts = [];
    let at = index + 1;
    while (at < lines.length && lines[at].trim() !== "") {
      parts.push(lines[at].trim());
      at += 1;
    }
    const signature = withoutBody(parts.join(" "));
    const named = signature.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*::/);
    if (!named) {
      continue;
    }
    items.push({ name: named[1], signature, where: lines[index] });
  }
  surface.set(key, { at: Date.now(), items });
  return items;
}

const completionItemProvider = {
  async provideCompletionItems(document, position) {
    const range = document.getWordRangeAtPosition(position);
    const written = range
      ? document.getText(new vscode.Range(range.start, position))
      : "";
    if (written.length < 2) {
      return undefined;
    }
    const folder = vscode.workspace.getWorkspaceFolder(document.uri);
    const directory = folder
      ? folder.uri.fsPath
      : require("path").dirname(document.uri.fsPath);
    const under = await surfaceUnder(written.slice(0, 2), directory);
    return under
      .filter((held) => held.name.startsWith(written))
      .map((held) => {
        const item = new vscode.CompletionItem(
          held.name,
          held.signature.includes(":: fn")
            ? vscode.CompletionItemKind.Function
            : vscode.CompletionItemKind.Variable
        );
        item.detail = held.signature;
        item.documentation = new vscode.MarkdownString(held.where);
        return item;
      });
  },
};

// Every edit the compiler applies unread, applied at once. The reports are the
// ones already published, so this offers exactly what the lightbulbs offer and
// asks the compiler nothing new.
//
// Highest offset first, so applying one leaves the offsets of the ones not yet
// applied standing, and two edits over the same bytes are one edit twice.
async function applyEveryFix() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.languageId !== "frost") {
    return;
  }
  const held = (offered.get(editor.document.uri.toString()) || []).filter(
    (fix) => fix.certain
  );
  if (held.length === 0) {
    vscode.window.showInformationMessage("frost: nothing to apply here");
    return;
  }
  held.sort((one, other) => other.edit.start.compareTo(one.edit.start));
  const edit = new vscode.WorkspaceEdit();
  let last = null;
  let written = 0;
  for (const fix of held) {
    if (last && fix.edit.end.compareTo(last) > 0) {
      continue;
    }
    edit.replace(editor.document.uri, fix.edit, fix.replacement);
    last = fix.edit.start;
    written += 1;
  }
  await vscode.workspace.applyEdit(edit);
  await publishFindings(editor.document);
  vscode.window.showInformationMessage(
    `frost: applied ${written} fix(es)`
  );
}

function activate(context) {
  ready = indexWorkspace().catch(() => undefined);
  context.subscriptions.push(findings);

  const selector = { language: "frost" };
  const timers = new Map();

  context.subscriptions.push(
    vscode.languages.registerDocumentFormattingEditProvider(
      selector,
      documentFormattingProvider
    ),
    vscode.languages.registerDocumentSymbolProvider(
      selector,
      documentSymbolProvider
    ),
    vscode.languages.registerDefinitionProvider(selector, definitionProvider),
    vscode.languages.registerWorkspaceSymbolProvider(workspaceSymbolProvider),
    vscode.languages.registerReferenceProvider(selector, referenceProvider),
    vscode.languages.registerHoverProvider(selector, hoverProvider),
    vscode.languages.registerCodeActionsProvider(selector, codeActionProvider, {
      providedCodeActionKinds: [vscode.CodeActionKind.QuickFix],
    }),
    vscode.languages.registerCompletionItemProvider(
      selector,
      completionItemProvider
    ),
    vscode.commands.registerCommand("frost.applyEveryFix", applyEveryFix),
    vscode.workspace.onDidChangeTextDocument((event) => {
      if (event.document.languageId !== "frost") {
        return;
      }
      const key = event.document.uri.toString();
      clearTimeout(timers.get(key));
      timers.set(
        key,
        setTimeout(() => {
          timers.delete(key);
          indexDocument(event.document);
        }, 250)
      );
    })
  );

  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument(publishFindings),
    vscode.workspace.onDidOpenTextDocument(publishFindings)
  );
  for (const open of vscode.workspace.textDocuments) {
    publishFindings(open);
  }

  const watcher = vscode.workspace.createFileSystemWatcher("**/*.frost");
  watcher.onDidCreate(indexFile, null, context.subscriptions);
  watcher.onDidChange(indexFile, null, context.subscriptions);
  watcher.onDidDelete(
    (uri) => {
      files.delete(uri.toString());
      offered.delete(uri.toString());
      findings.delete(uri);
    },
    null,
    context.subscriptions
  );
  context.subscriptions.push(watcher);
}

module.exports = { activate };
