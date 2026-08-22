// Speaks the protocol to `frostc lsp` on a pipe and weighs every answer.
//
// The extension has its own gate beside this one. This is the half below it:
// the server itself, driven with no editor and no extension in the way, so a
// wrong answer is localized to the compiler rather than to the glue.
//
// Spoken one message at a time, waiting for what each one is due, the way an
// editor speaks it. A check a change asks for runs once the pipe is quiet, so
// a driver that writes the whole conversation at once is answered as one
// burst and never sees a second check at all.
//
//     node .vscode/frost/check/server.js [compiler]
//
// The compiler defaults to `frostc` on PATH.

const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawn } = require("child_process");

const COMPILER = process.argv[2] || process.env.FROST_COMPILER || "frostc";
const CRLF = "\r\n";
const PATIENCE = 60000;

const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "frost-server-"));
const sample = path.join(workspace, "sample.frost");

const CLEAN = [
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
  "    p := Point { x = 1, y = 2 }",
  "    greeting_cost(p.x) + greeting_cost(p.y)",
  "}",
  "",
].join("\n");

// The same file with a closing brace pushed off the column it belongs on.
const BADLY = CLEAN.replace("    n * 2\n}", "    n * 2\n  }");

// One fault, and one the compiler carries a fix for.
const FAULTY = CLEAN.replace("    p := Point", "    var p := Point");

fs.writeFileSync(sample, CLEAN, "utf8");

// A line whose characters are wider than a byte. The compiler counts a column
// in bytes and the editor counts one in UTF-16 units, so a name after this
// string sits at two different numbers and only one of them is the answer.
const wide = path.join(workspace, "wide.frost");
const WIDE = [
  "wide :: fn(text: str, n: i64) -> i64 { n }",
  "",
  "/*",
  "wide_dropped :: fn() -> i64 { 2 }",
  "A brace that opens nothing: {",
  "*/",
  "",
  "main :: fn() -> i64 {",
  '    wide("héllo 🌍", 1) + wide("x", 2)',
  "}",
  "",
].join("\n");
const WIDE_FAULT = WIDE.replace('wide("x", 2)', "nosuch(2)");
fs.writeFileSync(wide, WIDE, "utf8");

// A pair: what one file says wrong is reported under that file while the
// other is the one being read.
const shared = path.join(workspace, "shared.frost");
const reader = path.join(workspace, "reader.frost");
fs.writeFileSync(
  shared,
  [
    "tally :: fn(n: i64) -> i64 {",
    "    var total: i64 = n",
    "    total",
    "}",
    "",
    '/* A brace, a quote and the name, all of them text: { " tally */',
    "",
    "Weight :: struct { grams: i64 }",
    "",
    "Crate :: struct { load: Weight, count: i64 }",
    "",
    "heavier :: fn(w: Weight) -> i64 { w.grams + 1 }",
    "",
  ].join("\n"),
  "utf8"
);
// The name is written in a comment and inside a string here as well, and a
// rename that reaches either of them writes over text that is not the name.
const READER = [
  'import "shared.frost"',
  "",
  "// tally is what this calls.",
  "start :: fn() -> i64 {",
  '    label := "tally"',
  "    held := tally(",
  "        /* ) */",
  "        2)",
  "    w := Weight { grams = 3 }",
  "    tally(1) + held + heavier(w)",
  "}",
  "",
].join("\n");
fs.writeFileSync(reader, READER, "utf8");

const PLAIN =
  "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~/";

function uri_of(where) {
  const held = where.split(path.sep).join("/");
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
  return text;
}

const uri = uri_of(sample);
const empty_uri = uri_of(path.join(workspace, "empty.frost"));
const unwritten_uri = uri_of(path.join(workspace, "never-written.frost"));
const unopened_uri = uri_of(path.join(workspace, "never-opened.frost"));
const wide_uri = uri_of(wide);
const shared_uri = uri_of(shared);
const reader_uri = uri_of(reader);

function at(text, needle, offset = 0) {
  const index = text.indexOf(needle) + offset;
  const upto = text.slice(0, index);
  const line = upto.split("\n").length - 1;
  const start = upto.lastIndexOf("\n") + 1;
  return { line, character: index - start };
}

const caret = at(CLEAN, "    greeting_cost(p.x)", 6);
const declared = at(CLEAN, "greeting_cost :: fn", 2);

function framed(message) {
  const body = Buffer.from(JSON.stringify(message), "utf8");
  return Buffer.concat([
    Buffer.from("Content-Length: " + body.length + CRLF + CRLF, "ascii"),
    body,
  ]);
}

// The conversation, held open. Each message goes out on its own, and what it
// is due comes back before the next one goes.
class Talk {
  constructor(child) {
    this.child = child;
    this.buffered = Buffer.alloc(0);
    this.replies = new Map();
    this.published = [];
    this.taken = 0;
    this.complained = [];
    this.woken = [];
    this.broken = undefined;
    child.stdout.on("data", (piece) => this.arrived(piece));
    child.stderr.on("data", (piece) => this.complained.push(piece));
    // A compiler that is not there is a failure with a name, rather than an
    // error thrown out of a handler that nothing is waiting on.
    child.on("error", (problem) => {
      this.broken = problem;
      for (const wake of this.woken.slice()) {
        wake();
      }
    });
  }

  arrived(piece) {
    this.buffered = Buffer.concat([this.buffered, piece]);
    for (;;) {
      const head = this.buffered.indexOf(CRLF + CRLF, 0, "ascii");
      if (head < 0) {
        break;
      }
      const header = this.buffered.slice(0, head).toString("ascii");
      const said = /content-length:[ ]*([0-9]+)/i.exec(header);
      if (!said) {
        break;
      }
      const length = Number(said[1]);
      const start = head + 4;
      if (this.buffered.length < start + length) {
        break;
      }
      const body = this.buffered.slice(start, start + length).toString("utf8");
      this.buffered = this.buffered.slice(start + length);
      const message = JSON.parse(body);
      if (message.id !== undefined && message.method === undefined) {
        this.replies.set(message.id, message);
      } else if (message.method === "textDocument/publishDiagnostics") {
        this.published.push(message.params);
      }
      for (const wake of this.woken.slice()) {
        wake();
      }
    }
  }

  // Settles once `ready` answers true, or gives up after PATIENCE.
  until(ready, what) {
    return new Promise((settle, fail) => {
      const wake = () => {
        if (this.broken) {
          clearTimeout(patience);
          this.woken = this.woken.filter((one) => one !== wake);
          fail(new Error("could not run `" + COMPILER + " lsp`: " + this.broken.message));
          return;
        }
        const found = ready();
        if (found !== undefined && found !== false) {
          clearTimeout(patience);
          this.woken = this.woken.filter((one) => one !== wake);
          settle(found);
        }
      };
      const patience = setTimeout(() => {
        this.woken = this.woken.filter((one) => one !== wake);
        fail(new Error("waited " + PATIENCE + "ms for " + what));
      }, PATIENCE);
      this.woken.push(wake);
      wake();
    });
  }

  say(message) {
    this.child.stdin.write(framed(message));
  }

  ask(id, method, params) {
    this.say({ jsonrpc: "2.0", id, method, params });
    return this.until(() => this.replies.get(id), "a reply to " + method).then(
      (held) => held.result
    );
  }

  // The reply itself, error and all, for a request the server may refuse.
  attempt(id, method, params) {
    this.say({ jsonrpc: "2.0", id, method, params });
    return this.until(() => this.replies.get(id), "a reply to " + method);
  }

  tell(method, params) {
    this.say({ jsonrpc: "2.0", method, params });
  }

  // A notification that asks for a check, and the notification the check
  // answers with.
  told(method, params, about) {
    const already = this.taken;
    this.say({ jsonrpc: "2.0", method, params });
    return this.until(() => {
      for (let index = already; index < this.published.length; index++) {
        if (this.published[index].uri === about) {
          this.taken = index + 1;
          return this.published[index];
        }
      }
      return false;
    }, "diagnostics for " + method);
  }
}

const wrong = [];

function want(label, held, ok, shown) {
  let passed = false;
  let seen = "";
  try {
    passed = ok(held);
    seen = shown(held);
  } catch (problem) {
    seen = "threw: " + problem.message;
  }
  console.log((passed ? "ok  " : "BAD ") + " " + label.padEnd(18) + " " + seen);
  if (!passed) {
    wrong.push(label);
  }
}

const brief = (held) => JSON.stringify(held).slice(0, 96);
const many = (held) => (Array.isArray(held) ? String(held.length) : "none");
const doc = (params) => Object.assign({ textDocument: { uri } }, params);

async function main() {
  const child = spawn(COMPILER, ["lsp"], { cwd: workspace });
  const talk = new Talk(child);
  const ended = new Promise((settle) => child.on("close", settle));

  want(
    "initialize",
    await talk.ask(1, "initialize", {
      rootUri: uri_of(workspace),
      capabilities: {},
    }),
    (held) =>
      held &&
      held.capabilities &&
      held.capabilities.hoverProvider &&
      held.capabilities.definitionProvider &&
      held.capabilities.renameProvider &&
      held.capabilities.textDocumentSync.change === 2 &&
      held.capabilities.diagnosticProvider.workspaceDiagnostics === true &&
      held.capabilities.semanticTokensProvider.legend.tokenTypes.join(" ") ===
        "keyword function type parameter variable property string number" &&
      held.capabilities.semanticTokensProvider.legend.tokenModifiers.join(
        " "
      ) === "declaration",
    (held) =>
      held ? Object.keys(held.capabilities).length + " capabilities" : "none"
  );
  talk.tell("initialized", {});

  want(
    "opens quiet",
    await talk.told(
      "textDocument/didOpen",
      { textDocument: { uri, languageId: "frost", version: 1, text: CLEAN } },
      uri
    ),
    (held) => held.diagnostics.length === 0,
    (held) => held.diagnostics.length + " diagnostics"
  );

  want(
    "definition",
    await talk.ask(2, "textDocument/definition", doc({ position: caret })),
    (held) => held && held.uri === uri && held.range.start.line === 3,
    (held) => (held ? "line " + held.range.start.line : "none")
  );
  want(
    "hover",
    await talk.ask(3, "textDocument/hover", doc({ position: caret })),
    (held) =>
      held && held.contents.value.includes("greeting_cost :: fn(n: i64) -> i64"),
    (held) => (held ? JSON.stringify(held.contents.value.slice(0, 46)) : "none")
  );
  want(
    "references",
    await talk.ask(
      4,
      "textDocument/references",
      doc({ position: declared, context: { includeDeclaration: true } })
    ),
    (held) => Array.isArray(held) && held.length === 3,
    many
  );
  want(
    "highlight",
    await talk.ask(
      5,
      "textDocument/documentHighlight",
      doc({ position: declared })
    ),
    (held) => Array.isArray(held) && held.length === 3,
    many
  );
  want(
    "documentSymbol",
    await talk.ask(6, "textDocument/documentSymbol", doc({})),
    (held) => Array.isArray(held) && held.length === 3,
    (held) =>
      Array.isArray(held) ? held.map((one) => one.name).join(" ") : "none"
  );
  want(
    "workspaceSymbol",
    await talk.ask(7, "workspace/symbol", { query: "point" }),
    (held) => Array.isArray(held) && held.some((one) => one.name === "Point"),
    (held) =>
      Array.isArray(held) ? held.map((one) => one.name).join(" ") : "none"
  );
  want(
    "formatting",
    await talk.ask(8, "textDocument/formatting", doc({ options: {} })),
    (held) => Array.isArray(held),
    many
  );
  want(
    "folding",
    await talk.ask(9, "textDocument/foldingRange", doc({})),
    (held) => Array.isArray(held) && held.length >= 2,
    many
  );
  want(
    "prepareRename",
    await talk.ask(10, "textDocument/prepareRename", doc({ position: declared })),
    (held) => held && held.start.line === 3,
    brief
  );
  want(
    "rename",
    await talk.ask(
      11,
      "textDocument/rename",
      doc({ position: declared, newName: "cost_of" })
    ),
    (held) =>
      held &&
      held.changes &&
      held.changes[uri] &&
      held.changes[uri].length === 3,
    (held) =>
      held && held.changes && held.changes[uri]
        ? held.changes[uri].length + " edits"
        : "none"
  );
  const items = (held) => (Array.isArray(held) ? held : (held || {}).items);
  want(
    "completion",
    await talk.ask(
      12,
      "textDocument/completion",
      doc({ position: at(CLEAN, "    greeting_cost(p.x)", 9) })
    ),
    (held) => {
      const found = items(held);
      return (
        Array.isArray(found) && found.some((one) => one.label === "greeting_cost")
      );
    },
    (held) => many(items(held))
  );
  want(
    "signatureHelp",
    await talk.ask(
      13,
      "textDocument/signatureHelp",
      doc({ position: at(CLEAN, "    greeting_cost(p.x)", 18) })
    ),
    (held) =>
      held &&
      Array.isArray(held.signatures) &&
      held.signatures.length === 1 &&
      held.signatures[0].label.includes("greeting_cost"),
    (held) =>
      held && held.signatures && held.signatures[0]
        ? held.signatures[0].label
        : "none"
  );

  want(
    "declaration",
    await talk.ask(34, "textDocument/declaration", doc({ position: caret })),
    (held) => held && held.uri === uri && held.range.start.line === 3,
    (held) => (held ? "line " + held.range.start.line : "none")
  );
  want(
    "type definition",
    await talk.ask(35, "textDocument/typeDefinition", {
      textDocument: { uri },
      position: at(CLEAN, "    p := Point", 4),
    }),
    (held) => held && held.uri === uri && held.range.start.line === 7,
    (held) => (held ? "line " + held.range.start.line : "none")
  );
  want(
    "selection range",
    await talk.ask(37, "textDocument/selectionRange", {
      textDocument: { uri },
      positions: [caret],
    }),
    (held) =>
      Array.isArray(held) &&
      held.length === 1 &&
      held[0].range.start.line === 11 &&
      held[0].parent &&
      held[0].parent.parent &&
      held[0].parent.parent.parent === undefined,
    (held) => {
      let walk = held && held[0];
      const steps = [];
      while (walk) {
        steps.push(walk.range.end.line - walk.range.start.line);
        walk = walk.parent;
      }
      return steps.join(" then ");
    }
  );
  want(
    "pulled diagnostics",
    await talk.ask(58, "textDocument/diagnostic", doc({})),
    (held) =>
      held && held.kind === "full" && Array.isArray(held.items) &&
      held.items.length === 0,
    (held) => (held ? held.kind + ", " + held.items.length + " items" : "none")
  );
  want(
    "pulled workspace diagnostics",
    await talk.ask(59, "workspace/diagnostic", { previousResultIds: [] }),
    (held) =>
      held &&
      Array.isArray(held.items) &&
      held.items.some((one) => one.uri === uri),
    (held) =>
      held && held.items ? held.items.length + " files" : "none"
  );
  // One keystroke, carrying only what it changed.
  want(
    "one change of a line",
    await talk.told(
      "textDocument/didChange",
      {
        textDocument: { uri, version: 12 },
        contentChanges: [
          {
            range: {
              start: at(CLEAN, "    n * 2", 8),
              end: at(CLEAN, "    n * 2", 9),
            },
            text: "nosuch",
          },
        ],
      },
      uri
    ),
    (held) =>
      held.diagnostics.length === 1 &&
      held.diagnostics[0].message.includes("nosuch"),
    (held) =>
      held.diagnostics.length > 0
        ? JSON.stringify(held.diagnostics[0].message.slice(0, 34))
        : "none"
  );
  want(
    "the change is taken back",
    await talk.told(
      "textDocument/didChange",
      {
        textDocument: { uri, version: 13 },
        contentChanges: [
          {
            range: {
              start: at(CLEAN, "    n * 2", 8),
              end: { line: 4, character: 14 },
            },
            text: "2",
          },
        ],
      },
      uri
    ),
    (held) => held.diagnostics.length === 0,
    (held) => held.diagnostics.length + " diagnostics"
  );
  want(
    "inlay hints",
    await talk.ask(46, "textDocument/inlayHint", {
      textDocument: { uri },
      range: {
        start: { line: 0, character: 0 },
        end: { line: 12, character: 0 },
      },
    }),
    (held) =>
      Array.isArray(held) &&
      held.length === 2 &&
      held[0].label === "n:" &&
      held[0].position.line === 11,
    (held) =>
      Array.isArray(held)
        ? held.map((one) => one.label + "@" + one.position.character).join(" ")
        : "none"
  );
  const lenses = await talk.ask(47, "textDocument/codeLens", doc({}));
  want(
    "code lenses",
    lenses,
    (held) =>
      Array.isArray(held) &&
      held.length === 3 &&
      held[0].data.name === "greeting_cost",
    (held) =>
      Array.isArray(held) ? held.map((one) => one.data.name).join(" ") : "none"
  );
  want(
    "code lens resolve",
    await talk.ask(48, "codeLens/resolve", lenses[0]),
    (held) => held && held.command && held.command.title === "2 uses",
    (held) => (held && held.command ? held.command.title : "none")
  );
  want(
    "semantic tokens",
    await talk.ask(45, "textDocument/semanticTokens/full", doc({})),
    (held) => {
      if (!held || !Array.isArray(held.data) || held.data.length % 5 !== 0) {
        return false;
      }
      // The declaration of greeting_cost: line 3, column 0, 13 wide, a
      // function, and declared rather than used.
      return (
        held.data[0] === 3 &&
        held.data[1] === 0 &&
        held.data[2] === 13 &&
        held.data[3] === 1 &&
        held.data[4] === 1
      );
    },
    (held) =>
      held && Array.isArray(held.data)
        ? held.data.length / 5 + " tokens, first " + held.data.slice(0, 5)
        : "none"
  );

  // Laying out part of a file, and a line as it is typed.
  await talk.told(
    "textDocument/didChange",
    {
      textDocument: { uri, version: 10 },
      contentChanges: [{ text: BADLY }],
    },
    uri
  );
  want(
    "range formatting",
    await talk.ask(41, "textDocument/rangeFormatting", {
      textDocument: { uri },
      range: {
        start: { line: 3, character: 0 },
        end: { line: 5, character: 3 },
      },
      options: { tabSize: 4, insertSpaces: true },
    }),
    (held) =>
      Array.isArray(held) &&
      held.length === 1 &&
      held[0].range.start.line === 3 &&
      held[0].newText.includes("\n}") &&
      held[0].newText.includes("  }") === false,
    (held) =>
      Array.isArray(held) && held[0]
        ? JSON.stringify(held[0].newText.slice(-6))
        : "none"
  );
  want(
    "on type formatting",
    await talk.ask(42, "textDocument/onTypeFormatting", {
      textDocument: { uri },
      position: at(BADLY, "  }", 3),
      ch: "}",
      options: { tabSize: 4, insertSpaces: true },
    }),
    (held) =>
      Array.isArray(held) && held.length === 1 && held[0].newText === "",
    (held) =>
      Array.isArray(held) && held[0]
        ? JSON.stringify(held[0].newText)
        : JSON.stringify(held)
  );
  await talk.told(
    "textDocument/didChange",
    {
      textDocument: { uri, version: 11 },
      contentChanges: [{ text: CLEAN }],
    },
    uri
  );
  want(
    "completion resolve",
    await talk.ask(43, "completionItem/resolve", {
      label: "greeting_cost",
      kind: 3,
    }),
    (held) =>
      held &&
      held.label === "greeting_cost" &&
      held.detail.includes("greeting_cost :: fn") &&
      held.documentation.value.includes("What a greeting costs"),
    (held) =>
      held && held.documentation
        ? JSON.stringify(held.documentation.value.slice(0, 30))
        : "none"
  );
  talk.say({ jsonrpc: "2.0", method: "$/cancelRequest", params: { id: 44 } });
  want(
    "cancelled request",
    await talk.attempt(44, "textDocument/hover", doc({ position: caret })),
    (held) => held.error !== undefined && held.error.code === -32800,
    (held) => (held.error ? "code " + held.error.code : "answered")
  );
  want(
    "watched files",
    await talk.told(
      "workspace/didChangeWatchedFiles",
      { changes: [{ uri: shared_uri, type: 2 }] },
      uri
    ),
    (held) => Array.isArray(held.diagnostics),
    (held) => held.diagnostics.length + " diagnostics"
  );

  // A fault arrives.
  want(
    "diagnostics",
    await talk.told(
      "textDocument/didChange",
      {
        textDocument: { uri, version: 2 },
        contentChanges: [{ text: FAULTY }],
      },
      uri
    ),
    (held) =>
      held.diagnostics.length >= 1 &&
      held.diagnostics.some((said) => said.message.includes("mut")),
    (held) =>
      held.diagnostics.length > 0
        ? JSON.stringify(held.diagnostics[0].message.slice(0, 42))
        : "none"
  );
  want(
    "codeAction",
    await talk.ask(
      14,
      "textDocument/codeAction",
      doc({
        range: {
          start: at(FAULTY, "    var p := Point", 4),
          end: at(FAULTY, "    var p := Point", 7),
        },
        context: { diagnostics: [] },
      })
    ),
    (held) => Array.isArray(held) && held.length >= 1 && held[0].edit,
    (held) =>
      Array.isArray(held) ? held.map((one) => one.title).join(" | ") : "none"
  );

  // The fault goes away, and so does what was said about it.
  want(
    "cleared",
    await talk.told(
      "textDocument/didChange",
      {
        textDocument: { uri, version: 3 },
        contentChanges: [{ text: CLEAN }],
      },
      uri
    ),
    (held) => held.diagnostics.length === 0,
    (held) => held.diagnostics.length + " diagnostics"
  );

  // Nothing below is well formed. The server answers what it can and stays up.
  const rough = [
    ["unknown method", 15, "textDocument/nosuchmethod", doc({})],
    [
      "past the end",
      16,
      "textDocument/hover",
      doc({ position: { line: 9999, character: 0 } }),
    ],
    [
      "unopened file",
      17,
      "textDocument/hover",
      { textDocument: { uri: unwritten_uri }, position: { line: 0, character: 0 } },
    ],
    [
      "bad uri",
      18,
      "textDocument/hover",
      { textDocument: { uri: "not-a-uri" }, position: { line: 0, character: 0 } },
    ],
    ["no position", 19, "textDocument/definition", { textDocument: { uri } }],
    ["no document", 20, "textDocument/documentSymbol", {}],
    [
      "negative position",
      21,
      "textDocument/hover",
      doc({ position: { line: -1, character: -1 } }),
    ],
  ];
  for (const [label, id, method, params] of rough) {
    want(
      "survives " + label,
      await talk.attempt(id, method, params),
      (held) => held !== undefined && held.jsonrpc === "2.0",
      (held) =>
        held.error ? "error " + held.error.code : brief(held.result)
    );
  }

  talk.tell("textDocument/didChange", {
    textDocument: { uri: unopened_uri, version: 1 },
    contentChanges: [{ text: "" }],
  });
  want(
    "survives empty file",
    await talk.told(
      "textDocument/didOpen",
      {
        textDocument: {
          uri: empty_uri,
          languageId: "frost",
          version: 1,
          text: "",
        },
      },
      empty_uri
    ),
    (held) => Array.isArray(held.diagnostics),
    (held) => held.diagnostics.length + " diagnostics"
  );
  talk.tell("$/cancelRequest", { id: 99 });
  talk.tell("workspace/didChangeConfiguration", { settings: {} });

  want(
    "still answering",
    await talk.ask(22, "textDocument/hover", doc({ position: caret })),
    (held) => held && held.contents.value.includes("greeting_cost"),
    (held) => (held ? "yes" : "no")
  );

  // Half-typed. A reader in the middle of writing a function still wants the
  // outline and the names, so a file that does not parse has to be answered
  // rather than go blank.
  want(
    "half-typed reported",
    await talk.told(
      "textDocument/didChange",
      {
        textDocument: { uri, version: 5 },
        contentChanges: [{ text: CLEAN + "\nlater :: fn(n: i64" }],
      },
      uri
    ),
    (held) => held.diagnostics.length >= 1,
    (held) => held.diagnostics.length + " diagnostics"
  );
  want(
    "half-typed still named",
    await talk.ask(27, "textDocument/documentSymbol", doc({})),
    (held) => Array.isArray(held) && held.length >= 3,
    (held) =>
      Array.isArray(held) ? held.map((one) => one.name).join(" ") : "none"
  );

  // Closed while something is still underlined. What was said about a file no
  // open document reports about goes with the document that said it.
  want(
    "underlined again",
    await talk.told(
      "textDocument/didChange",
      {
        textDocument: { uri, version: 4 },
        contentChanges: [{ text: FAULTY }],
      },
      uri
    ),
    (held) => held.diagnostics.length >= 1,
    (held) => held.diagnostics.length + " diagnostics"
  );
  want(
    "closing takes it back",
    await talk.told("textDocument/didClose", doc({}), uri),
    (held) => held.diagnostics.length === 0,
    (held) => held.diagnostics.length + " diagnostics"
  );
  // A report and a caret on a line of wide characters.
  want(
    "wide line reported",
    await talk.told(
      "textDocument/didOpen",
      {
        textDocument: {
          uri: wide_uri,
          languageId: "frost",
          version: 1,
          text: WIDE_FAULT,
        },
      },
      wide_uri
    ),
    (held) =>
      held.diagnostics.length >= 1 &&
      held.diagnostics[0].range.start.character ===
        at(WIDE_FAULT, "nosuch(2)").character,
    (held) =>
      held.diagnostics.length > 0
        ? "character " +
          held.diagnostics[0].range.start.character +
          ", wanted " +
          at(WIDE_FAULT, "nosuch(2)").character
        : "none"
  );
  await talk.told(
    "textDocument/didChange",
    {
      textDocument: { uri: wide_uri, version: 2 },
      contentChanges: [{ text: WIDE }],
    },
    wide_uri
  );
  want(
    "names past a comment",
    await talk.ask(31, "textDocument/documentSymbol", {
      textDocument: { uri: wide_uri },
    }),
    (held) =>
      Array.isArray(held) &&
      held.length === 2 &&
      held.every((one) => one.name !== "wide_dropped"),
    (held) =>
      Array.isArray(held) ? held.map((one) => one.name).join(" ") : "none"
  );
  want(
    "finds nothing in a comment",
    await talk.ask(32, "workspace/symbol", { query: "wide_dropped" }),
    (held) => Array.isArray(held) && held.length === 0,
    many
  );
  want(
    "offers nothing from a comment",
    await talk.ask(33, "textDocument/completion", {
      textDocument: { uri: wide_uri },
      position: at(WIDE, 'wide("x"', 4),
    }),
    (held) => {
      const found = Array.isArray(held) ? held : (held || {}).items;
      return (
        Array.isArray(found) &&
        found.every((one) => one.label !== "wide_dropped")
      );
    },
    (held) => {
      const found = Array.isArray(held) ? held : (held || {}).items;
      return Array.isArray(found)
        ? found.map((one) => one.label).join(" ")
        : "none";
    }
  );
  want(
    "folds past a comment",
    await talk.ask(29, "textDocument/foldingRange", {
      textDocument: { uri: wide_uri },
    }),
    (held) => Array.isArray(held) && held.length === 1,
    many
  );
  want(
    "wide line caret",
    await talk.ask(26, "textDocument/definition", {
      textDocument: { uri: wide_uri },
      position: at(WIDE, 'wide("x"', 1),
    }),
    (held) => held && held.uri === wide_uri && held.range.start.line === 0,
    (held) => (held ? "line " + held.range.start.line : "none")
  );

  // What one file says wrong, reported while another is the one being read.
  want(
    "reports the import",
    await talk.told(
      "textDocument/didOpen",
      {
        textDocument: {
          uri: reader_uri,
          languageId: "frost",
          version: 1,
          text: READER,
        },
      },
      shared_uri
    ),
    (held) => held.diagnostics.length >= 1,
    (held) => held.diagnostics.length + " diagnostics on shared.frost"
  );
  want(
    "definition across files",
    await talk.ask(
      24,
      "textDocument/definition",
      {
        textDocument: { uri: reader_uri },
        position: at(READER, "    tally(1)", 5),
      }
    ),
    (held) => held && held.uri === shared_uri && held.range.start.line === 0,
    (held) => (held ? held.uri.split("/").pop() + ":" + held.range.start.line : "none")
  );
  want(
    "rename across files",
    await talk.ask(25, "textDocument/rename", {
      textDocument: { uri: reader_uri },
      position: at(READER, "    tally(1)", 5),
      newName: "counted",
    }),
    (held) =>
      held &&
      held.changes &&
      held.changes[shared_uri] &&
      held.changes[shared_uri].length === 1 &&
      held.changes[reader_uri] &&
      held.changes[reader_uri].length === 2,
    (held) =>
      held && held.changes
        ? Object.keys(held.changes)
            .map((one) => one.split("/").pop() + ":" + held.changes[one].length)
            .join(" ")
        : "none"
  );
  want(
    "signature past a comment",
    await talk.ask(30, "textDocument/signatureHelp", {
      textDocument: { uri: reader_uri },
      position: at(READER, "        2)", 8),
    }),
    (held) =>
      held &&
      Array.isArray(held.signatures) &&
      held.signatures.length === 1 &&
      held.signatures[0].label.includes("tally"),
    (held) =>
      held && held.signatures && held.signatures[0]
        ? held.signatures[0].label
        : "none"
  );
  want(
    "implementation",
    await talk.ask(36, "textDocument/implementation", {
      textDocument: { uri: reader_uri },
      position: at(READER, "    w := Weight", 9),
    }),
    (held) =>
      Array.isArray(held) &&
      held.length === 1 &&
      held[0].uri === shared_uri,
    many
  );
  const hierarchy = await talk.ask(
    49,
    "textDocument/prepareCallHierarchy",
    {
      textDocument: { uri: reader_uri },
      position: at(READER, "    tally(1)", 5),
    }
  );
  want(
    "prepare call hierarchy",
    hierarchy,
    (held) =>
      Array.isArray(held) &&
      held.length === 1 &&
      held[0].name === "tally" &&
      held[0].uri === shared_uri,
    (held) => (Array.isArray(held) && held[0] ? held[0].name : "none")
  );
  want(
    "incoming calls",
    await talk.ask(50, "callHierarchy/incomingCalls", { item: hierarchy[0] }),
    (held) =>
      Array.isArray(held) &&
      held.length === 1 &&
      held[0].from.name === "start" &&
      held[0].fromRanges.length === 2,
    (held) =>
      Array.isArray(held)
        ? held
            .map((one) => one.from.name + ":" + one.fromRanges.length)
            .join(" ")
        : "none"
  );
  const caller = await talk.ask(51, "textDocument/prepareCallHierarchy", {
    textDocument: { uri: reader_uri },
    position: at(READER, "start :: fn", 2),
  });
  want(
    "outgoing calls",
    await talk.ask(52, "callHierarchy/outgoingCalls", { item: caller[0] }),
    (held) =>
      Array.isArray(held) &&
      held.length === 2 &&
      held.some((one) => one.to.name === "tally") &&
      held.some((one) => one.to.name === "heavier"),
    (held) =>
      Array.isArray(held) ? held.map((one) => one.to.name).join(" ") : "none"
  );
  const held_type = await talk.ask(53, "textDocument/prepareTypeHierarchy", {
    textDocument: { uri: reader_uri },
    position: at(READER, "    w := Weight", 9),
  });
  want(
    "prepare type hierarchy",
    held_type,
    (held) =>
      Array.isArray(held) && held.length === 1 && held[0].name === "Weight",
    (held) => (Array.isArray(held) && held[0] ? held[0].name : "none")
  );
  want(
    "subtypes",
    await talk.ask(54, "typeHierarchy/subtypes", { item: held_type[0] }),
    (held) => Array.isArray(held) && held.length === 0,
    many
  );
  want(
    "supertypes",
    await talk.ask(55, "typeHierarchy/supertypes", { item: held_type[0] }),
    (held) =>
      Array.isArray(held) && held.length === 1 && held[0].name === "Crate",
    (held) =>
      Array.isArray(held) ? held.map((one) => one.name).join(" ") : "none"
  );
  const crate = await talk.ask(56, "textDocument/prepareTypeHierarchy", {
    textDocument: { uri: reader_uri },
    position: at(READER, "    w := Weight", 9),
  });
  want(
    "subtypes of a holder",
    await talk.ask(57, "typeHierarchy/subtypes", {
      item: Object.assign({}, crate[0], { name: "Crate" }),
    }),
    (held) =>
      Array.isArray(held) && held.length === 1 && held[0].name === "Weight",
    (held) =>
      Array.isArray(held) ? held.map((one) => one.name).join(" ") : "none"
  );
  want(
    "implementation of a function",
    await talk.ask(40, "textDocument/implementation", {
      textDocument: { uri: reader_uri },
      position: at(READER, "    tally(1)", 5),
    }),
    (held) =>
      Array.isArray(held) &&
      held.length === 1 &&
      held[0].uri === shared_uri &&
      held[0].range.start.line === 0,
    many
  );
  want(
    "document link",
    await talk.ask(38, "textDocument/documentLink", {
      textDocument: { uri: reader_uri },
    }),
    (held) =>
      Array.isArray(held) &&
      held.length === 1 &&
      held[0].target === shared_uri &&
      held[0].range.start.line === 0,
    (held) =>
      Array.isArray(held) && held[0] ? held[0].target.split("/").pop() : "none"
  );
  want(
    "linked editing",
    await talk.ask(39, "textDocument/linkedEditingRange", {
      textDocument: { uri: reader_uri },
      position: at(READER, "    tally(1)", 5),
    }),
    (held) => held && Array.isArray(held.ranges) && held.ranges.length === 2,
    (held) => (held && held.ranges ? held.ranges.length + " ranges" : "none")
  );
  want(
    "references across files",
    await talk.ask(28, "textDocument/references", {
      textDocument: { uri: reader_uri },
      position: at(READER, "    tally(1)", 5),
      context: { includeDeclaration: true },
    }),
    (held) =>
      Array.isArray(held) &&
      held.length === 3 &&
      held.some((one) => one.uri === shared_uri) &&
      held.some((one) => one.uri === reader_uri),
    many
  );
  want(
    "closing takes back the import",
    await talk.told(
      "textDocument/didClose",
      { textDocument: { uri: reader_uri } },
      shared_uri
    ),
    (held) => held.diagnostics.length === 0,
    (held) => held.diagnostics.length + " diagnostics on shared.frost"
  );

  want(
    "shutdown",
    await talk.attempt(23, "shutdown", {}),
    (held) => held !== undefined,
    brief
  );
  talk.tell("exit", {});
  child.stdin.end();

  const code = await ended;
  const stderr = Buffer.concat(talk.complained).toString("utf8");
  want(
    "exit",
    code,
    (held) => held === 0,
    (held) => "code " + held
  );
  want(
    "stderr",
    stderr,
    (held) => held.trim() === "",
    (held) => (held.trim() ? JSON.stringify(held.slice(0, 90)) : "silent")
  );

  try {
    fs.rmSync(workspace, { recursive: true, force: true });
  } catch (problem) {
    console.log("(left " + workspace + " behind)");
  }

  if (wrong.length > 0) {
    console.error("\n" + wrong.length + " wrong: " + wrong.join(", "));
    return 1;
  }
  console.log("\nthe server answers");
  return 0;
}

main().then(
  (code) => process.exit(code),
  (problem) => {
    console.error("FAILED:", problem && problem.stack);
    process.exit(1);
  }
);
