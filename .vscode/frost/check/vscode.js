// Enough of the VS Code API for the extension to activate and answer, so the
// half of the editor support that runs inside the editor can be driven without
// one.
//
// What this catches is what a protocol driver cannot: a name that does not
// exist, an API called wrongly at activation, a reply built with the wrong
// constructor. None of that shows until the extension host loads the file, and
// then it shows as nothing working at all.

class Position {
  constructor(line, character) {
    this.line = line;
    this.character = character;
  }
  compareTo(other) {
    if (this.line !== other.line) {
      return this.line - other.line;
    }
    return this.character - other.character;
  }
}

class Range {
  constructor(start, end) {
    this.start = start;
    this.end = end;
  }
  intersection(other) {
    const after = this.start.compareTo(other.end) > 0;
    const before = this.end.compareTo(other.start) < 0;
    return after || before ? undefined : this;
  }
}

const PLAIN =
  "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~/";

class Uri {
  constructor(text, fsPath) {
    this.text = text;
    this.scheme = text.split(":")[0];
    this.fsPath = fsPath;
  }
  static parse(text) {
    let path = decodeURIComponent(text.replace(/^file:\/\//, ""));
    if (/^\/[A-Za-z]:/.test(path)) {
      path = path.slice(1);
    }
    return new Uri(text, path);
  }
  static file(path) {
    const held = path.replace(/\\/g, "/");
    let text = "file://";
    if (!held.startsWith("/")) {
      text += "/";
    }
    for (const byte of Buffer.from(held, "utf8")) {
      const one = String.fromCharCode(byte);
      text += PLAIN.includes(one)
        ? one
        : "%" + byte.toString(16).toUpperCase().padStart(2, "0");
    }
    return new Uri(text, path);
  }
  toString() {
    return this.text;
  }
}

class Diagnostic {
  constructor(range, message, severity) {
    Object.assign(this, { range, message, severity });
  }
}

class Location {
  constructor(uri, range) {
    this.uri = uri;
    this.range = range;
  }
}

class Hover {
  constructor(contents) {
    this.contents = contents;
  }
}

class MarkdownString {
  constructor(value) {
    this.value = value;
  }
}

class DocumentHighlight {
  constructor(range) {
    this.range = range;
  }
}

class DocumentSymbol {
  constructor(name, detail, kind, range, selectionRange) {
    Object.assign(this, { name, detail, kind, range, selectionRange });
  }
}

class SymbolInformation {
  constructor(name, kind, container, location) {
    Object.assign(this, { name, kind, container, location });
  }
}

class TextEdit {
  static replace(range, newText) {
    return { range, newText };
  }
}

class FoldingRange {
  constructor(start, end) {
    this.start = start;
    this.end = end;
  }
}

class CodeAction {
  constructor(title, kind) {
    this.title = title;
    this.kind = kind;
  }
}

class WorkspaceEdit {
  constructor() {
    this.edits = [];
  }
  replace(uri, range, text) {
    this.edits.push({ uri, range, text });
  }
}

class SignatureHelp {
  constructor() {
    this.signatures = [];
  }
}

class SignatureInformation {
  constructor(label) {
    this.label = label;
    this.parameters = [];
  }
}

class ParameterInformation {
  constructor(label) {
    this.label = label;
  }
}

class SemanticTokensLegend {
  constructor(tokenTypes, tokenModifiers) {
    this.tokenTypes = tokenTypes;
    this.tokenModifiers = tokenModifiers;
  }
}

class SemanticTokens {
  constructor(data) {
    this.data = data;
  }
}

class SelectionRange {
  constructor(range, parent) {
    this.range = range;
    this.parent = parent;
  }
}

class DocumentLink {
  constructor(range, target) {
    this.range = range;
    this.target = target;
  }
}

class LinkedEditingRanges {
  constructor(ranges) {
    this.ranges = ranges;
  }
}

class CompletionItem {
  constructor(label, kind) {
    this.label = label;
    this.kind = kind;
  }
}

const registered = {};
const diagnostics = new Map();
const shown = [];
const workspaceFolders = [];
const documents = [];
const watchers = [];

function keep(name) {
  return (...held) => {
    registered[name] = registered[name] || [];
    registered[name].push(held);
    return { dispose() {} };
  };
}

function ignore() {
  return { dispose() {} };
}

module.exports = {
  Position,
  Range,
  Uri,
  Diagnostic,
  DiagnosticSeverity: { Error: 0, Warning: 1 },
  Location,
  Hover,
  MarkdownString,
  DocumentHighlight,
  DocumentSymbol,
  SymbolInformation,
  TextEdit,
  FoldingRange,
  CodeAction,
  CodeActionKind: { QuickFix: "quickfix" },
  WorkspaceEdit,
  SignatureHelp,
  SignatureInformation,
  ParameterInformation,
  CompletionItem,
  SelectionRange,
  SemanticTokensLegend,
  SemanticTokens,
  DocumentLink,
  LinkedEditingRanges,
  languages: {
    createDiagnosticCollection() {
      return {
        set(uri, held) {
          diagnostics.set(uri.toString(), held);
        },
        delete(uri) {
          diagnostics.delete(uri.toString());
        },
        clear() {
          diagnostics.clear();
        },
        dispose() {},
      };
    },
    registerDefinitionProvider: keep("definition"),
    registerDeclarationProvider: keep("declaration"),
    registerTypeDefinitionProvider: keep("typeDefinition"),
    registerImplementationProvider: keep("implementation"),
    registerSelectionRangeProvider: keep("selectionRange"),
    registerDocumentLinkProvider: keep("documentLink"),
    registerLinkedEditingRangeProvider: keep("linkedEditing"),
    registerHoverProvider: keep("hover"),
    registerReferenceProvider: keep("references"),
    registerDocumentHighlightProvider: keep("highlight"),
    registerDocumentSymbolProvider: keep("documentSymbol"),
    registerWorkspaceSymbolProvider: keep("workspaceSymbol"),
    registerDocumentFormattingEditProvider: keep("formatting"),
    registerDocumentRangeFormattingEditProvider: keep("rangeFormatting"),
    registerOnTypeFormattingEditProvider: keep("onTypeFormatting"),
    registerFoldingRangeProvider: keep("folding"),
    registerDocumentSemanticTokensProvider: keep("semanticTokens"),
    registerRenameProvider: keep("rename"),
    registerCompletionItemProvider: keep("completion"),
    registerSignatureHelpProvider: keep("signatureHelp"),
    registerCodeActionsProvider: keep("codeAction"),
  },
  commands: { registerCommand: keep("command") },
  workspace: {
    workspaceFolders,
    textDocuments: documents,
    getConfiguration() {
      return {
        get: (_name, fallback) => process.env.FROST_COMPILER || fallback,
      };
    },
    getWorkspaceFolder() {
      return workspaceFolders[0];
    },
    onDidOpenTextDocument: ignore,
    onDidChangeTextDocument: ignore,
    onDidSaveTextDocument: ignore,
    onDidCloseTextDocument: ignore,
    onDidChangeConfiguration: ignore,
    createFileSystemWatcher(pattern) {
      const held = { pattern, created: [], changed: [], deleted: [] };
      watchers.push(held);
      return {
        onDidCreate: (one) => held.created.push(one),
        onDidChange: (one) => held.changed.push(one),
        onDidDelete: (one) => held.deleted.push(one),
        dispose() {},
      };
    },
    applyEdit(edit) {
      shown.push(["edit", edit.edits.length]);
      return Promise.resolve(true);
    },
  },
  window: {
    activeTextEditor: undefined,
    showErrorMessage(text) {
      shown.push(["error", text]);
    },
    showInformationMessage(text) {
      shown.push(["info", text]);
    },
  },
  held: { registered, diagnostics, shown, workspaceFolders, documents, watchers },
};
