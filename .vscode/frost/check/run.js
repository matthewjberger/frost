// Loads the extension against a stub VS Code, activates it, and asks every
// provider it registers, over a workspace it writes itself.
//
// The server has its own gate: a driver that speaks the protocol on a pipe.
// This is the other half. An extension whose `activate` throws registers
// nothing and the editor says nothing about it, so the failure a reader sees is
// that the language has no support at all.
//
//     node .vscode/frost/check/run.js [compiler]
//
// The compiler defaults to `frostc` on PATH, which is what `just install-self`
// puts there and what the extension asks for.

const fs = require("fs");
const os = require("os");
const path = require("path");
const Module = require("module");

const here = __dirname;
const root = path.resolve(here, "..", "..", "..");

if (process.argv[2]) {
  process.env.FROST_COMPILER = process.argv[2];
}

// Every server the extension starts, kept so the gate can weigh whether it
// ends. A server that outlives the editor holds the compiler binary open, and
// the next build cannot write over it.
const child_process = require("child_process");
const started = [];
const spawned = Object.assign({}, child_process, {
  spawn(...held) {
    const child = child_process.spawn(...held);
    started.push(child);
    return child;
  },
});

// `require("vscode")` inside the extension resolves to the stub beside this,
// and `require("child_process")` to the wrapper above it.
const load = Module._load;
Module._load = function (request, parent, isMain) {
  if (request === "vscode") {
    return require(path.join(here, "vscode.js"));
  }
  if (request === "child_process") {
    return spawned;
  }
  return load(request, parent, isMain);
};

const vscode = require(path.join(here, "vscode.js"));
const extension = require(path.join(root, ".vscode", "frost", "extension.js"));
const held = vscode.held;

// A workspace of its own, so what an answer names is what this wrote and
// nothing a previous run left behind.
const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "frost-editor-"));
const sample = path.join(workspace, "sample.frost");
const TEXT = [
  "// What a greeting costs.",
  "//",
  "// Counted rather than measured.",
  "greeting_cost :: fn(n: i64) -> i64 {",
  "    n * 2",
  "}",
  "",
  "Point :: struct { x: i64, y: i64 }",
  "",
  "main :: fn() -> i64 {",
  "    var total: i64 = greeting_cost(2)",
  "    total",
  "}",
  "",
].join("\n");
fs.writeFileSync(sample, TEXT, "utf8");

const uri = vscode.Uri.file(sample);
const lines = TEXT.split("\n");
// The buffer, as the editor holds one: what it says now, which a change
// rewrites.
let held_text = TEXT;
const document = {
  languageId: "frost",
  uri,
  version: 1,
  lineCount: lines.length,
  getText: () => held_text,
  lineAt: (index) => ({
    range: { end: new vscode.Position(index, lines[index].length) },
  }),
};

held.workspaceFolders.push({ uri: vscode.Uri.file(workspace) });
held.documents.push(document);

function provider(name) {
  const entry = held.registered[name];
  if (!entry) {
    throw new Error(`nothing registered a ${name} provider`);
  }
  // A workspace-symbol provider is registered on its own; every other one
  // carries a selector in front of it.
  return entry[0].length === 1 ? entry[0][0] : entry[0][1];
}

function at(needle, offset = 0) {
  const index = TEXT.indexOf(needle) + offset;
  const line = TEXT.slice(0, index).split("\n").length - 1;
  return new vscode.Position(
    line,
    index - (TEXT.lastIndexOf("\n", index - 1) + 1)
  );
}

// A path out of a URI is written with the separator the URI uses, and the one
// this wrote is written with the platform's.
function same_file(one, other) {
  return (
    one.replace(/\\/g, "/").toLowerCase() ===
    other.replace(/\\/g, "/").toLowerCase()
  );
}

const wrong = [];

function want(label, held, ok, shown) {
  const passed = ok(held);
  console.log(`${passed ? "ok  " : "BAD "} ${label.padEnd(16)} ${shown(held)}`);
  if (!passed) {
    wrong.push(label);
  }
}

const some = (held) => Array.isArray(held) && held.length > 0;
const count = (held) => (Array.isArray(held) ? String(held.length) : "-");

async function main() {
  extension.activate({ subscriptions: [] });
  const names = Object.keys(held.registered).sort();
  want(
    "registered",
    names,
    (found) => found.length >= 26,
    (found) => found.join(" ")
  );

  // `activate` tells the server about the buffer; the first answer follows.
  await new Promise((settle) => setTimeout(settle, 5000));

  want(
    "diagnostics",
    held.diagnostics.get(uri.toString()) || [],
    (found) => found.some((one) => one.message.includes("`mut`")),
    (found) => found.map((one) => one.message.slice(0, 40)).join(" | ")
  );

  const caret = at("greeting_cost(2)", 2);
  want(
    "definition",
    await provider("definition").provideDefinition(document, caret),
    (found) =>
      found && same_file(found.uri.fsPath, sample) && found.range.start.line === 3,
    (found) => (found ? `${found.uri.fsPath}:${found.range.start.line}` : "none")
  );
  want(
    "declaration",
    await provider("declaration").provideDeclaration(document, caret),
    (found) => found && found.range.start.line === 3,
    (found) => (found ? "line " + found.range.start.line : "none")
  );
  want(
    "type definition",
    await provider("typeDefinition").provideTypeDefinition(
      document,
      at("var total", 5)
    ),
    // Nothing: the local is an `i64`, and a builtin has no declaration to be
    // taken to. A provider that invents one fails here.
    (found) => found === undefined,
    (found) => (found ? "line " + found.range.start.line : "nothing")
  );
  want(
    "implementation",
    await provider("implementation").provideImplementation(document, caret),
    (found) => Array.isArray(found) && found.length === 1,
    (found) => (Array.isArray(found) ? String(found.length) : "none")
  );
  want(
    "selection range",
    await provider("selectionRange").provideSelectionRanges(document, [caret]),
    (found) =>
      Array.isArray(found) && found.length === 1 && found[0].parent !== undefined,
    (found) => {
      let walk = found && found[0];
      let steps = 0;
      while (walk) {
        steps = steps + 1;
        walk = walk.parent;
      }
      return steps + " steps";
    }
  );
  want(
    "document link",
    await provider("documentLink").provideDocumentLinks(document),
    // None: this file imports nothing.
    (found) => Array.isArray(found) && found.length === 0,
    (found) => (Array.isArray(found) ? String(found.length) : "none")
  );
  want(
    "linked editing",
    await provider("linkedEditing").provideLinkedEditingRanges(document, caret),
    (found) => found && found.ranges.length === 2,
    (found) => (found ? found.ranges.length + " ranges" : "none")
  );
  want(
    "hover",
    await provider("hover").provideHover(document, caret),
    (found) =>
      found && found.contents.value.includes("greeting_cost :: fn(n: i64)"),
    (found) => (found ? JSON.stringify(found.contents.value.slice(0, 44)) : "none")
  );
  want(
    "references",
    await provider("references").provideReferences(document, caret),
    (found) => found.length === 2,
    count
  );
  want(
    "highlight",
    await provider("highlight").provideDocumentHighlights(document, caret),
    (found) => found.length === 2,
    count
  );
  want(
    "documentSymbol",
    await provider("documentSymbol").provideDocumentSymbols(document),
    (found) => found.length === 3,
    (found) => found.map((one) => `${one.name}:${one.kind}`).join(" ")
  );
  want(
    "workspaceSymbol",
    await provider("workspaceSymbol").provideWorkspaceSymbols("point"),
    (found) => found.length === 1 && found[0].name === "Point",
    (found) => found.map((one) => one.name).join(" ")
  );
  want(
    "formatting",
    await provider("formatting").provideDocumentFormattingEdits(document),
    some,
    count
  );
  const rung = await provider("callHierarchy").prepareCallHierarchy(
    document,
    caret
  );
  want(
    "prepare call hierarchy",
    rung,
    (found) => Array.isArray(found) && found[0].name === "greeting_cost",
    (found) => (Array.isArray(found) && found[0] ? found[0].name : "none")
  );
  want(
    "incoming calls",
    await provider("callHierarchy").provideCallHierarchyIncomingCalls(rung[0]),
    (found) => Array.isArray(found) && found.length === 1,
    (found) =>
      Array.isArray(found) ? found.map((one) => one.from.name).join(" ") : "none"
  );
  const typed = await provider("typeHierarchy").prepareTypeHierarchy(
    document,
    at("Point :: struct", 2)
  );
  want(
    "prepare type hierarchy",
    typed,
    (found) => Array.isArray(found) && found[0].name === "Point",
    (found) => (Array.isArray(found) && found[0] ? found[0].name : "none")
  );
  want(
    "supertypes",
    await provider("typeHierarchy").provideTypeHierarchySupertypes(typed[0]),
    // None: nothing in this file holds a field of a Point.
    (found) => Array.isArray(found) && found.length === 0,
    count
  );
  want(
    "inlay hints",
    await provider("inlayHint").provideInlayHints(
      document,
      new vscode.Range(at("greeting_cost :: fn"), at("total", 5))
    ),
    (found) => Array.isArray(found) && found.length === 1,
    (found) =>
      Array.isArray(found) ? found.map((one) => one.label).join(" ") : "none"
  );
  const lenses = await provider("codeLens").provideCodeLenses(document);
  want(
    "code lenses",
    lenses,
    (found) => Array.isArray(found) && found.length === 3,
    count
  );
  want(
    "code lens resolve",
    await provider("codeLens").resolveCodeLens(lenses[0]),
    (found) => found && found.command && found.command.title.includes("use"),
    (found) => (found && found.command ? found.command.title : "none")
  );
  want(
    "semantic legend",
    held.registered.semanticTokens[0][2],
    (found) =>
      found.tokenTypes.join(" ") ===
        "keyword function type parameter variable property string number" &&
      found.tokenModifiers.join(" ") === "declaration",
    (found) => found.tokenTypes.length + " types"
  );
  want(
    "semantic tokens",
    await provider("semanticTokens").provideDocumentSemanticTokens(document),
    (found) => found && found.data.length % 5 === 0 && found.data.length > 20,
    (found) => (found ? found.data.length / 5 + " tokens" : "none")
  );
  want(
    "folding",
    await provider("folding").provideFoldingRanges(document),
    some,
    count
  );
  want(
    "range formatting",
    await provider("rangeFormatting").provideDocumentRangeFormattingEdits(
      document,
      new vscode.Range(at("greeting_cost :: fn"), at("    n * 2", 9))
    ),
    (found) => Array.isArray(found) && found.length === 1,
    count
  );
  want(
    "on type formatting",
    await provider("onTypeFormatting").provideOnTypeFormattingEdits(
      document,
      at("}", 1),
      "}"
    ),
    // None: the brace is already on the column its opener is on.
    (found) => Array.isArray(found) && found.length === 0,
    count
  );
  want(
    "watcher",
    held.watchers,
    (found) =>
      found.length === 1 &&
      found[0].created.length === 1 &&
      found[0].changed.length === 1 &&
      found[0].deleted.length === 1,
    (found) => (found.length ? found[0].pattern : "none")
  );
  want(
    "prepareRename",
    await provider("rename").prepareRename(document, caret),
    (found) => found && found.start.line === 10,
    (found) => (found ? JSON.stringify(found) : "none")
  );
  want(
    "rename",
    await provider("rename").provideRenameEdits(document, caret, "cost_of"),
    (found) => found && found.edits.length === 2,
    (found) => (found ? String(found.edits.length) : "none")
  );
  const offered = await provider("completion").provideCompletionItems(
    document,
    at("greeting_cost(2)", 6)
  );
  want(
    "completion resolve",
    await provider("completion").resolveCompletionItem(
      offered.find((one) => one.label === "greeting_cost")
    ),
    (found) => found && found.documentation && found.documentation.value,
    (found) =>
      found && found.documentation
        ? JSON.stringify(found.documentation.value.slice(0, 26))
        : "none"
  );
  want(
    "completion",
    await provider("completion").provideCompletionItems(
      document,
      at("greeting_cost(2)", 6)
    ),
    (found) => found.some((one) => one.label === "greeting_cost"),
    (found) => found.map((one) => `${one.label}:${one.kind}`).join(" ")
  );
  want(
    "signatureHelp",
    await provider("signatureHelp").provideSignatureHelp(
      document,
      at("greeting_cost(2)", 14)
    ),
    (found) =>
      found &&
      found.signatures.length === 1 &&
      found.signatures[0].parameters.length === 1,
    (found) => (found ? found.signatures[0].label : "none")
  );
  want(
    "codeAction",
    await provider("codeAction").provideCodeActions(
      document,
      new vscode.Range(at("var total"), at("var total", 3)),
      { diagnostics: held.diagnostics.get(uri.toString()) || [] }
    ),
    (found) => found.length === 1 && found[0].edit.edits.length === 1,
    (found) => found.map((one) => one.title).join(" | ")
  );

  vscode.window.activeTextEditor = { document };
  const command = held.registered.command.find(
    (one) => one[0] === "frost.applyEveryFix"
  )[1];
  await command();
  want(
    "applyEveryFix",
    held.shown,
    (found) => found.some((one) => one[0] === "edit" && one[1] === 1),
    (found) => JSON.stringify(found)
  );

  // A server that stopped, on purpose or otherwise, is started again by the
  // next thing asked of it, and what it was told about the open buffers goes
  // to the new one. Without that a reader whose server died sees the language
  // lose its support with nothing said about it.
  const restart = held.registered.command.find(
    (one) => one[0] === "frost.restartServer"
  )[1];
  await restart();
  await new Promise((settle) => setTimeout(settle, 4000));
  want(
    "restarted",
    await provider("hover").provideHover(document, caret),
    (found) =>
      found && found.contents.value.includes("greeting_cost :: fn(n: i64)"),
    (found) => (found ? "answering" : "silent")
  );
  want(
    "restarted diagnostics",
    held.diagnostics.get(uri.toString()) || [],
    (found) => found.some((one) => one.message.includes("`mut`")),
    (found) => found.length + " diagnostics"
  );

  // One keystroke, carried as the range it changed rather than as the file.
  // The word the report names is the word it replaces, so what the editor is
  // told goes away.
  held_text = TEXT.replace("    var total", "    mut total");
  document.version = 2;
  for (const handler of held.listeners.change || []) {
    handler({
      document,
      contentChanges: [
        {
          range: new vscode.Range(at("    var", 4), at("    var", 7)),
          rangeLength: 3,
          text: "mut",
        },
      ],
    });
  }
  await new Promise((settle) => setTimeout(settle, 5000));
  want(
    "one change of a line",
    held.diagnostics.get(uri.toString()) || [],
    (found) => found.length === 0,
    (found) => found.length + " diagnostics"
  );

  extension.deactivate();
  // What deactivate leaves behind. The workspace cannot be removed while a
  // process holds it as its own directory, and a server still running holds
  // the compiler binary too, so every one it started must have ended.
  const ended = await Promise.all(
    started.map(
      (child) =>
        new Promise((settle) => {
          if (child.exitCode !== null || child.signalCode !== null) {
            settle(true);
            return;
          }
          const waited = setTimeout(() => settle(false), 4000);
          child.on("close", () => {
            clearTimeout(waited);
            settle(true);
          });
        })
    )
  );
  want(
    "servers end",
    ended,
    (held) => held.length > 0 && held.every((one) => one),
    (held) => `${held.filter((one) => one).length} of ${held.length} ended`
  );
  try {
    fs.rmSync(workspace, { recursive: true, force: true });
  } catch (problem) {
    console.log(`(left ${workspace} behind)`);
  }
  if (wrong.length > 0) {
    console.error(`\n${wrong.length} wrong: ${wrong.join(", ")}`);
    return 1;
  }
  console.log("\nthe editor half answers");
  return 0;
}

main().then(
  (code) => process.exit(code),
  (problem) => {
    console.error("FAILED:", problem && problem.stack);
    process.exit(1);
  }
);
