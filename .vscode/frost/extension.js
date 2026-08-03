// Navigation over the declaration syntax alone: every declaration head opens
// its line and the namespace is flat, so a workspace search for
// `^\s*name\s*::` is definition lookup. No language server, no dependencies;
// the extension stays a plain file copy under `just install-editor`.
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

// One pass over the file, tracking strings, comments and brace depth, so a
// head is only read at the top level and a body's extent is known for the
// outline. A head that never opens a brace ends on its own line.
function braceDelta(line, state) {
  let delta = 0;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (state.inBlockComment) {
      if (character === "*" && line[index + 1] === "/") {
        state.inBlockComment = false;
        index += 1;
      }
      continue;
    }
    if (state.inString) {
      if (character === "\\") {
        index += 1;
      } else if (character === '"') {
        state.inString = false;
      }
      continue;
    }
    if (character === "/" && line[index + 1] === "/") {
      break;
    }
    if (character === "/" && line[index + 1] === "*") {
      state.inBlockComment = true;
      index += 1;
      continue;
    }
    if (character === '"') {
      state.inString = true;
    } else if (character === "{") {
      delta += 1;
    } else if (character === "}") {
      delta -= 1;
    }
  }
  return delta;
}

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
      for (let lineIndex = 0; lineIndex < entry.lines.length; lineIndex += 1) {
        const line = entry.lines[lineIndex];
        pattern.lastIndex = 0;
        let match;
        while ((match = pattern.exec(line)) !== null) {
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

function activate(context) {
  ready = indexWorkspace().catch(() => undefined);

  const selector = { language: "frost" };
  const timers = new Map();

  context.subscriptions.push(
    vscode.languages.registerDocumentSymbolProvider(
      selector,
      documentSymbolProvider
    ),
    vscode.languages.registerDefinitionProvider(selector, definitionProvider),
    vscode.languages.registerWorkspaceSymbolProvider(workspaceSymbolProvider),
    vscode.languages.registerReferenceProvider(selector, referenceProvider),
    vscode.languages.registerHoverProvider(selector, hoverProvider),
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

  const watcher = vscode.workspace.createFileSystemWatcher("**/*.frost");
  watcher.onDidCreate(indexFile, null, context.subscriptions);
  watcher.onDidChange(indexFile, null, context.subscriptions);
  watcher.onDidDelete(
    (uri) => files.delete(uri.toString()),
    null,
    context.subscriptions
  );
  context.subscriptions.push(watcher);
}

module.exports = { activate };
