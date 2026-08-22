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

// One fault, and one the compiler carries a fix for.
const FAULTY = CLEAN.replace("    p := Point", "    var p := Point");

fs.writeFileSync(sample, CLEAN, "utf8");

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
  ].join("\n"),
  "utf8"
);
const READER = [
  'import "shared.frost"',
  "",
  "start :: fn() -> i64 {",
  "    tally(1)",
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
      held.capabilities.textDocumentSync !== undefined,
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
      held.changes[shared_uri].length >= 1 &&
      held.changes[reader_uri] &&
      held.changes[reader_uri].length >= 1,
    (held) =>
      held && held.changes
        ? Object.keys(held.changes)
            .map((one) => one.split("/").pop() + ":" + held.changes[one].length)
            .join(" ")
        : "none"
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
