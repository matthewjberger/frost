// The editor's half of the Frost language server.
//
// Everything an answer needs is what the compiler already built, so the
// compiler answers: `frostc lsp` is one process, started once, asked over its
// own streams. What is here is the wiring between that conversation and what
// VS Code hands a provider.
//
// Written against the protocol directly rather than through a client library,
// so the extension stays one file with nothing to install beside it and `just
// install-editor` stays a copy. The protocol is a header carrying a byte count,
// then that many bytes of JSON-RPC.
//
// The compiler it talks to is the self-hosted one. The bootstrap's job is to
// build that compiler and to be held to the same language; a tool written in
// Frost lives in the self-hosted compiler alone and has no twin there.

const vscode = require("vscode");
const { spawn } = require("child_process");

const findings = vscode.languages.createDiagnosticCollection("frost");

// Where the self-hosted compiler is. It carries the server, the formatter, the
// check and the lint; a bare name is looked up on PATH.
function compilerPath() {
  const held = vscode.workspace
    .getConfiguration("frost")
    .get("compilerPath", "frostc");
  return held && held.trim() ? held.trim() : "frostc";
}

// The conversation with one server process.
//
// A request carries a number that its answer carries back, so the two are
// matched by that number rather than by order: the server answers a question
// about one file while a change to another is still on its way in.
class Server {
  constructor() {
    this.child = null;
    this.next = 1;
    this.waiting = new Map();
    this.held = Buffer.alloc(0);
    this.open = new Set();
  }

  start() {
    if (this.child) {
      return true;
    }
    const folder = vscode.workspace.workspaceFolders
      ? vscode.workspace.workspaceFolders[0].uri.fsPath
      : undefined;
    let child;
    try {
      child = spawn(compilerPath(), ["lsp"], {
        cwd: folder,
        stdio: ["pipe", "pipe", "pipe"],
      });
    } catch (error) {
      this.report(error);
      return false;
    }
    this.child = child;
    // Whatever half a message the one before it left behind is not the start of
    // this one's first.
    this.held = Buffer.alloc(0);
    child.on("error", (error) => this.report(error));
    // Each handler answers for the process it was given, and says nothing once
    // a later one has taken its place. A restart starts the next server while
    // the one before it is still ending, and its exit arrives after: without
    // this, it forgets the buffers the new server was just told about, and the
    // reader is left with a server that answers nothing.
    child.stdout.on("data", (piece) => {
      if (this.child !== child) {
        return;
      }
      this.read(piece);
    });
    child.on("exit", () => {
      if (this.child !== child) {
        return;
      }
      this.child = null;
      for (const settle of this.waiting.values()) {
        settle(null);
      }
      this.waiting.clear();
      this.open.clear();
    });
    this.send({
      jsonrpc: "2.0",
      id: this.next++,
      method: "initialize",
      params: {
        processId: process.pid,
        rootUri: folder ? vscode.Uri.file(folder).toString() : null,
        capabilities: {},
      },
    });
    this.send({ jsonrpc: "2.0", method: "initialized", params: {} });
    return true;
  }

  report(error) {
    vscode.window.showErrorMessage(
      `frost: cannot start '${compilerPath()} lsp' (${error.message}). ` +
        "Run `just install-self`, or set frost.compilerPath."
    );
  }

  stop() {
    if (!this.child) {
      return;
    }
    this.send({ jsonrpc: "2.0", method: "exit", params: {} });
    this.child.stdin.end();
    this.child = null;
    this.open.clear();
  }

  send(message) {
    if (!this.child) {
      return;
    }
    const body = Buffer.from(JSON.stringify(message), "utf8");
    this.child.stdin.write(`Content-Length: ${body.length}\r\n\r\n`);
    this.child.stdin.write(body);
  }

  // Whole messages out of the stream, which arrives in whatever pieces the pipe
  // hands over: a header may come without its body, and two answers may come at
  // once.
  read(piece) {
    this.held = Buffer.concat([this.held, piece]);
    for (;;) {
      const head = this.held.indexOf("\r\n\r\n");
      if (head < 0) {
        return;
      }
      const header = this.held.slice(0, head).toString("ascii");
      const named = /content-length:\s*(\d+)/i.exec(header);
      if (!named) {
        this.held = this.held.slice(head + 4);
        continue;
      }
      const length = Number(named[1]);
      if (this.held.length < head + 4 + length) {
        return;
      }
      const body = this.held.slice(head + 4, head + 4 + length);
      this.held = this.held.slice(head + 4 + length);
      try {
        this.take(JSON.parse(body.toString("utf8")));
      } catch (error) {
        continue;
      }
    }
  }

  take(message) {
    if (message.id !== undefined && this.waiting.has(message.id)) {
      const settle = this.waiting.get(message.id);
      this.waiting.delete(message.id);
      settle(message.result === undefined ? null : message.result);
      return;
    }
    if (message.method === "textDocument/publishDiagnostics") {
      publish(message.params);
    }
  }

  notify(method, params) {
    if (!this.start()) {
      return;
    }
    this.send({ jsonrpc: "2.0", method, params });
  }

  request(method, params) {
    if (!this.start()) {
      return Promise.resolve(null);
    }
    const id = this.next++;
    return new Promise((settle) => {
      this.waiting.set(id, settle);
      this.send({ jsonrpc: "2.0", id, method, params });
      // A server that never answers must not leave a provider waiting: VS Code
      // shows a spinner until one of the two happens.
      setTimeout(() => {
        if (this.waiting.has(id)) {
          this.waiting.delete(id);
          settle(null);
        }
      }, 20000);
    });
  }

  // The server answers about the buffer on screen, so it is told what that is
  // before anything is asked about it.
  opened(document) {
    const uri = document.uri.toString();
    if (this.open.has(uri)) {
      return;
    }
    this.open.add(uri);
    this.notify("textDocument/didOpen", {
      textDocument: {
        uri,
        languageId: "frost",
        version: document.version,
        text: document.getText(),
      },
    });
  }

  changed(document) {
    if (!this.open.has(document.uri.toString())) {
      this.opened(document);
      return;
    }
    this.notify("textDocument/didChange", {
      textDocument: {
        uri: document.uri.toString(),
        version: document.version,
      },
      contentChanges: [{ text: document.getText() }],
    });
  }

  closed(document) {
    const uri = document.uri.toString();
    if (!this.open.has(uri)) {
      return;
    }
    this.open.delete(uri);
    this.notify("textDocument/didClose", { textDocument: { uri } });
  }

  saved(document) {
    this.notify("textDocument/didSave", {
      textDocument: { uri: document.uri.toString() },
    });
  }
}

const server = new Server();

function watched(document) {
  return document.languageId === "frost" && document.uri.scheme === "file";
}

function named(document) {
  return { textDocument: { uri: document.uri.toString() } };
}

function at(position) {
  return { line: position.line, character: position.character };
}

function positionOf(held) {
  return new vscode.Position(held.line, held.character);
}

function rangeOf(held) {
  return new vscode.Range(positionOf(held.start), positionOf(held.end));
}

function locationOf(held) {
  return new vscode.Location(vscode.Uri.parse(held.uri), rangeOf(held.range));
}

// A workspace edit out of the shape the protocol writes one in.
function editOf(held) {
  const edit = new vscode.WorkspaceEdit();
  if (!held || !held.changes) {
    return edit;
  }
  for (const [uri, edits] of Object.entries(held.changes)) {
    for (const one of edits) {
      edit.replace(vscode.Uri.parse(uri), rangeOf(one.range), one.newText);
    }
  }
  return edit;
}

// Whether an edit lands where a report is, which is what pairs the two so a
// lightbulb shows under the squiggle it answers.
function touches(range, held) {
  if (!held || !held.changes) {
    return false;
  }
  for (const edits of Object.values(held.changes)) {
    for (const one of edits) {
      if (range.intersection(rangeOf(one.range))) {
        return true;
      }
    }
  }
  return false;
}

// The reports the server publishes. It sends one message per file it has
// something to say about, and an empty one for a file it no longer has.
function publish(params) {
  const uri = vscode.Uri.parse(params.uri);
  findings.set(
    uri,
    (params.diagnostics || []).map((report) => {
      const held = new vscode.Diagnostic(
        rangeOf(report.range),
        report.message || "",
        report.severity === 2
          ? vscode.DiagnosticSeverity.Warning
          : vscode.DiagnosticSeverity.Error
      );
      held.source = report.source || "frost";
      return held;
    })
  );
}

const definitionProvider = {
  async provideDefinition(document, position) {
    const held = await server.request("textDocument/definition", {
      ...named(document),
      position: at(position),
    });
    return held ? locationOf(held) : undefined;
  },
};

const declarationProvider = {
  async provideDeclaration(document, position) {
    const held = await server.request("textDocument/declaration", {
      ...named(document),
      position: at(position),
    });
    return held ? locationOf(held) : undefined;
  },
};

const typeDefinitionProvider = {
  async provideTypeDefinition(document, position) {
    const held = await server.request("textDocument/typeDefinition", {
      ...named(document),
      position: at(position),
    });
    return held ? locationOf(held) : undefined;
  },
};

const implementationProvider = {
  async provideImplementation(document, position) {
    const held = await server.request("textDocument/implementation", {
      ...named(document),
      position: at(position),
    });
    return Array.isArray(held) ? held.map(locationOf) : [];
  },
};

// The chain the server writes, turned inside out: VS Code holds the parent as
// an object of its own, built from the outside in.
function widening(held) {
  if (!held) {
    return undefined;
  }
  const steps = [];
  for (let walk = held; walk; walk = walk.parent) {
    steps.push(rangeOf(walk.range));
  }
  let built;
  for (let index = steps.length - 1; index >= 0; index--) {
    built = new vscode.SelectionRange(steps[index], built);
  }
  return built;
}

const selectionRangeProvider = {
  async provideSelectionRanges(document, positions) {
    const held = await server.request("textDocument/selectionRange", {
      ...named(document),
      positions: positions.map(at),
    });
    return Array.isArray(held) ? held.map(widening).filter(Boolean) : [];
  },
};

const documentLinkProvider = {
  async provideDocumentLinks(document) {
    const held = await server.request("textDocument/documentLink", named(document));
    if (!Array.isArray(held)) {
      return [];
    }
    return held.map(
      (one) =>
        new vscode.DocumentLink(rangeOf(one.range), vscode.Uri.parse(one.target))
    );
  },
};

const linkedEditingProvider = {
  async provideLinkedEditingRanges(document, position) {
    const held = await server.request("textDocument/linkedEditingRange", {
      ...named(document),
      position: at(position),
    });
    if (!held || !Array.isArray(held.ranges) || held.ranges.length === 0) {
      return undefined;
    }
    return new vscode.LinkedEditingRanges(held.ranges.map(rangeOf));
  },
};

const hoverProvider = {
  async provideHover(document, position) {
    const held = await server.request("textDocument/hover", {
      ...named(document),
      position: at(position),
    });
    if (!held || !held.contents) {
      return undefined;
    }
    return new vscode.Hover(
      new vscode.MarkdownString(held.contents.value || "")
    );
  },
};

const referenceProvider = {
  async provideReferences(document, position) {
    const held = await server.request("textDocument/references", {
      ...named(document),
      position: at(position),
      context: { includeDeclaration: true },
    });
    return (held || []).map(locationOf);
  },
};

const documentHighlightProvider = {
  async provideDocumentHighlights(document, position) {
    const held = await server.request("textDocument/documentHighlight", {
      ...named(document),
      position: at(position),
    });
    return (held || []).map(
      (one) => new vscode.DocumentHighlight(rangeOf(one.range))
    );
  },
};

// The protocol numbers its kinds from one and VS Code numbers the same list
// from zero, so every kind that crosses loses one on the way in.
const documentSymbolProvider = {
  async provideDocumentSymbols(document) {
    const held = await server.request(
      "textDocument/documentSymbol",
      named(document)
    );
    return (held || []).map(
      (one) =>
        new vscode.DocumentSymbol(
          one.name,
          "",
          one.kind - 1,
          rangeOf(one.range),
          rangeOf(one.selectionRange)
        )
    );
  },
};

const workspaceSymbolProvider = {
  async provideWorkspaceSymbols(query) {
    const held = await server.request("workspace/symbol", { query });
    return (held || []).map(
      (one) =>
        new vscode.SymbolInformation(
          one.name,
          one.kind - 1,
          "",
          locationOf(one.location)
        )
    );
  },
};

const documentFormattingProvider = {
  async provideDocumentFormattingEdits(document) {
    const held = await server.request("textDocument/formatting", {
      ...named(document),
      options: { tabSize: 4, insertSpaces: true },
    });
    return (held || []).map((one) =>
      vscode.TextEdit.replace(rangeOf(one.range), one.newText)
    );
  },
};

const foldingRangeProvider = {
  async provideFoldingRanges(document) {
    const held = await server.request(
      "textDocument/foldingRange",
      named(document)
    );
    return (held || []).map(
      (one) => new vscode.FoldingRange(one.startLine, one.endLine)
    );
  },
};

const renameProvider = {
  async prepareRename(document, position) {
    const held = await server.request("textDocument/prepareRename", {
      ...named(document),
      position: at(position),
    });
    if (!held) {
      throw new Error("frost: there is no name here to rename");
    }
    return rangeOf(held);
  },
  async provideRenameEdits(document, position, newName) {
    const held = await server.request("textDocument/rename", {
      ...named(document),
      position: at(position),
      newName,
    });
    return held && held.changes ? editOf(held) : undefined;
  },
};

const completionItemProvider = {
  async provideCompletionItems(document, position) {
    const held = await server.request("textDocument/completion", {
      ...named(document),
      position: at(position),
    });
    if (!held) {
      return undefined;
    }
    return (held.items || []).map((one) => {
      const item = new vscode.CompletionItem(one.label, one.kind - 1);
      if (one.detail) {
        item.detail = one.detail;
      }
      return item;
    });
  },
};

// The edits the reports carry, offered where the reports are. The compiler
// worked each of them out; without this they were written and read by nothing.
// The head of the call being written, and which parameter the caret is in. The
// server reads it back from the caret rather than out of a parse, since a call
// with its arguments half written is one the parser refuses and that is exactly
// when a reader wants to be told what goes there.
const signatureHelpProvider = {
  async provideSignatureHelp(document, position) {
    const held = await server.request("textDocument/signatureHelp", {
      ...named(document),
      position: at(position),
    });
    if (!held || !held.signatures || held.signatures.length === 0) {
      return undefined;
    }
    const answer = new vscode.SignatureHelp();
    answer.signatures = held.signatures.map((one) => {
      const signature = new vscode.SignatureInformation(one.label);
      signature.parameters = (one.parameters || []).map(
        (each) => new vscode.ParameterInformation(each.label)
      );
      return signature;
    });
    answer.activeSignature = held.activeSignature || 0;
    answer.activeParameter = held.activeParameter || 0;
    return answer;
  },
};

const codeActionProvider = {
  async provideCodeActions(document, range, context) {
    const held = await server.request("textDocument/codeAction", {
      ...named(document),
      range: { start: at(range.start), end: at(range.end) },
      context: { diagnostics: [] },
    });
    return (held || []).map((one) => {
      const action = new vscode.CodeAction(
        one.title,
        vscode.CodeActionKind.QuickFix
      );
      // A fix the compiler applies unread is the one to reach for, and one it
      // offers is a guess at what was meant.
      action.isPreferred = one.isPreferred === true;
      action.edit = editOf(one.edit);
      action.diagnostics = context.diagnostics.filter((diagnostic) =>
        touches(diagnostic.range, one.edit)
      );
      return action;
    });
  },
};

// Every edit the compiler applies unread, applied at once.
//
// Highest offset first, so applying one leaves the offsets of the ones not yet
// applied standing, and two edits over the same bytes are one edit twice.
async function applyEveryFix() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || !watched(editor.document)) {
    return;
  }
  const document = editor.document;
  const last = document.lineAt(document.lineCount - 1).range.end;
  const held = await server.request("textDocument/codeAction", {
    ...named(document),
    range: { start: { line: 0, character: 0 }, end: at(last) },
    context: { diagnostics: [] },
  });
  const edits = [];
  for (const one of (held || []).filter((each) => each.isPreferred === true)) {
    for (const [uri, changes] of Object.entries(
      (one.edit && one.edit.changes) || {}
    )) {
      for (const each of changes) {
        edits.push({ uri, range: rangeOf(each.range), text: each.newText });
      }
    }
  }
  if (edits.length === 0) {
    vscode.window.showInformationMessage("frost: nothing to apply here");
    return;
  }
  edits.sort((one, other) => other.range.start.compareTo(one.range.start));
  const edit = new vscode.WorkspaceEdit();
  let above = null;
  let written = 0;
  for (const one of edits) {
    if (above && one.range.end.compareTo(above) > 0) {
      continue;
    }
    edit.replace(vscode.Uri.parse(one.uri), one.range, one.text);
    above = one.range.start;
    written += 1;
  }
  await vscode.workspace.applyEdit(edit);
  vscode.window.showInformationMessage(`frost: applied ${written} fix(es)`);
}

// A fresh server, told about every buffer that is open. What it was told about
// the old one goes with it.
function restart() {
  server.stop();
  findings.clear();
  for (const open of vscode.workspace.textDocuments) {
    if (watched(open)) {
      server.opened(open);
    }
  }
}

function activate(context) {
  const selector = { language: "frost", scheme: "file" };
  context.subscriptions.push(findings);

  context.subscriptions.push(
    vscode.languages.registerDefinitionProvider(selector, definitionProvider),
    vscode.languages.registerDeclarationProvider(selector, declarationProvider),
    vscode.languages.registerTypeDefinitionProvider(
      selector,
      typeDefinitionProvider
    ),
    vscode.languages.registerImplementationProvider(
      selector,
      implementationProvider
    ),
    vscode.languages.registerSelectionRangeProvider(
      selector,
      selectionRangeProvider
    ),
    vscode.languages.registerDocumentLinkProvider(
      selector,
      documentLinkProvider
    ),
    vscode.languages.registerLinkedEditingRangeProvider(
      selector,
      linkedEditingProvider
    ),
    vscode.languages.registerHoverProvider(selector, hoverProvider),
    vscode.languages.registerReferenceProvider(selector, referenceProvider),
    vscode.languages.registerDocumentHighlightProvider(
      selector,
      documentHighlightProvider
    ),
    vscode.languages.registerDocumentSymbolProvider(
      selector,
      documentSymbolProvider
    ),
    vscode.languages.registerWorkspaceSymbolProvider(workspaceSymbolProvider),
    vscode.languages.registerDocumentFormattingEditProvider(
      selector,
      documentFormattingProvider
    ),
    vscode.languages.registerFoldingRangeProvider(
      selector,
      foldingRangeProvider
    ),
    vscode.languages.registerRenameProvider(selector, renameProvider),
    vscode.languages.registerCompletionItemProvider(
      selector,
      completionItemProvider
    ),
    vscode.languages.registerSignatureHelpProvider(
      selector,
      signatureHelpProvider,
      "(",
      ","
    ),
    vscode.languages.registerCodeActionsProvider(selector, codeActionProvider, {
      providedCodeActionKinds: [vscode.CodeActionKind.QuickFix],
    }),
    vscode.commands.registerCommand("frost.applyEveryFix", applyEveryFix),
    vscode.commands.registerCommand("frost.restartServer", restart)
  );

  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument((document) => {
      if (watched(document)) {
        server.opened(document);
      }
    }),
    vscode.workspace.onDidChangeTextDocument((event) => {
      if (watched(event.document)) {
        server.changed(event.document);
      }
    }),
    vscode.workspace.onDidSaveTextDocument((document) => {
      if (watched(document)) {
        server.saved(document);
      }
    }),
    vscode.workspace.onDidCloseTextDocument((document) => {
      if (watched(document)) {
        server.closed(document);
        findings.delete(document.uri);
      }
    }),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("frost.compilerPath")) {
        restart();
      }
    })
  );

  for (const open of vscode.workspace.textDocuments) {
    if (watched(open)) {
      server.opened(open);
    }
  }
}

function deactivate() {
  server.stop();
}

module.exports = { activate, deactivate };
