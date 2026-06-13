// Functional smoke test for the pi host-mode JS extensions.
//
// Drives benchmaker/swebench/pi_ext/{remote_exec_all,register_provider}.js the
// way pi would: a mock `pi` captures registered tools/providers, and a mock
// localhost bridge actually executes the shell commands the tools emit (in a
// temp dir, exactly as _ExecBridge anchors at cwd). This verifies the real
// read/write/edit shell wiring, not just that the files parse.
//
// Usage: node pi_ext_smoke.mjs <pi_ext_dir>
// Exits 0 on success; prints "FAIL: ..." and exits 1 otherwise.

import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { mkdtempSync, readFileSync, writeFileSync, rmSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const EXT_DIR = process.argv[2];
if (!EXT_DIR) {
  console.error("FAIL: missing <pi_ext_dir> argument");
  process.exit(1);
}

const failures = [];
function check(cond, msg) {
  if (!cond) failures.push(msg);
}
function eq(actual, expected, msg) {
  if (actual !== expected) failures.push(`${msg}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

// --- load an ESM extension whose on-disk name ends in .js ------------------ //
const workdir = mkdtempSync(join(tmpdir(), "pi-ext-smoke-"));
async function loadExt(name) {
  const src = readFileSync(join(EXT_DIR, name), "utf-8");
  const mjs = join(workdir, name.replace(/\.js$/, ".mjs"));
  writeFileSync(mjs, src);
  return (await import(pathToFileURL(mjs))).default;
}

// --- mock pi --------------------------------------------------------------- //
function makePi() {
  const tools = new Map();
  const providers = [];
  return {
    registerTool(def) { tools.set(def.name, def); },
    registerProvider(name, config) { providers.push({ name, config }); },
    on() {},
    tools,
    providers,
  };
}

// --- mock bridge: run the emitted shell command in `sandbox`, like _ExecBridge ---
const sandbox = mkdtempSync(join(tmpdir(), "pi-ext-sandbox-"));
const commands = [];
const server = createServer((req, res) => {
  let body = "";
  req.on("data", (c) => (body += c));
  req.on("end", () => {
    const { command } = JSON.parse(body);
    commands.push(command);
    const full = `cd '${sandbox}' && ${command}`;
    let out = { return_code: 0, stdout: "", stderr: "" };
    try {
      out.stdout = execFileSync("sh", ["-c", full], { encoding: "utf-8" });
    } catch (e) {
      out.return_code = typeof e.status === "number" ? e.status : 1;
      out.stdout = e.stdout ? e.stdout.toString() : "";
      out.stderr = e.stderr ? e.stderr.toString() : String(e);
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(out));
  });
});

function assertCanonical(result, label) {
  check(result && Array.isArray(result.content), `${label}: result.content must be an array`);
  check(result && result.content[0] && result.content[0].type === "text",
    `${label}: result.content[0] must be a text block`);
  check(result && typeof result.details === "object",
    `${label}: result.details must be present`);
}
const textOf = (r) => (r && r.content && r.content[0] ? r.content[0].text : undefined);

async function main() {
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const { port } = server.address();
  process.env.PI_EXEC_BRIDGE = `http://127.0.0.1:${port}`;
  process.env.PI_EXEC_CWD = sandbox;

  // ---- remote_exec_all.js: bash + read + write + edit, all routed --------- //
  const allExt = await loadExt("remote_exec_all.js");
  const pi = makePi();
  allExt(pi);
  for (const t of ["bash", "read", "write", "edit"]) {
    check(pi.tools.has(t), `remote_exec_all must register tool "${t}"`);
  }

  const bash = pi.tools.get("bash");
  const read = pi.tools.get("read");
  const write = pi.tools.get("write");
  const edit = pi.tools.get("edit");

  // write -> read round-trip
  let r = await write.execute("1", { path: "a.txt", content: "hello\nworld\n" }, undefined, undefined, {});
  assertCanonical(r, "write");
  eq(r.details.exitCode, 0, "write exitCode");
  eq(readFileSync(join(sandbox, "a.txt"), "utf-8"), "hello\nworld\n", "write landed in sandbox file");

  r = await read.execute("2", { path: "a.txt" }, undefined, undefined, {});
  assertCanonical(r, "read");
  eq(textOf(r), "hello\nworld\n", "read returns file content verbatim");

  // write nested dirs are created
  r = await write.execute("3", { path: "sub/dir/b.txt", content: "x" }, undefined, undefined, {});
  eq(r.details.exitCode, 0, "write nested exitCode");
  eq(readFileSync(join(sandbox, "sub/dir/b.txt"), "utf-8"), "x", "write creates parent dirs");

  // read offset/limit (1-indexed lines)
  await write.execute("4", { path: "lines.txt", content: "l1\nl2\nl3\nl4\nl5\n" }, undefined, undefined, {});
  r = await read.execute("5", { path: "lines.txt", offset: 2, limit: 2 }, undefined, undefined, {});
  eq(textOf(r), "l2\nl3\n", "read offset+limit slices lines");
  r = await read.execute("6", { path: "lines.txt", offset: 4 }, undefined, undefined, {});
  eq(textOf(r), "l4\nl5\n", "read offset-only tails from line");

  // edit: exact unique replacement
  r = await edit.execute("7", { path: "a.txt", edits: [{ oldText: "world", newText: "earth" }] }, undefined, undefined, {});
  assertCanonical(r, "edit");
  eq(r.details.exitCode, 0, "edit exitCode ok");
  eq(readFileSync(join(sandbox, "a.txt"), "utf-8"), "hello\nearth\n", "edit applied in sandbox file");

  // edit: multiple edits in one call
  await write.execute("8", { path: "multi.txt", content: "AAA BBB CCC" }, undefined, undefined, {});
  r = await edit.execute("9", { path: "multi.txt",
    edits: [{ oldText: "AAA", newText: "111" }, { oldText: "CCC", newText: "333" }] }, undefined, undefined, {});
  eq(r.details.exitCode, 0, "multi-edit exitCode ok");
  eq(readFileSync(join(sandbox, "multi.txt"), "utf-8"), "111 BBB 333", "multi-edit applied");

  // edit: non-unique oldText -> error, file untouched
  await write.execute("10", { path: "dup.txt", content: "zz zz" }, undefined, undefined, {});
  r = await edit.execute("11", { path: "dup.txt", edits: [{ oldText: "zz", newText: "q" }] }, undefined, undefined, {});
  eq(r.details.exitCode, 1, "edit non-unique -> exitCode 1");
  check(/not unique/i.test(textOf(r)), "edit non-unique message");
  eq(readFileSync(join(sandbox, "dup.txt"), "utf-8"), "zz zz", "edit non-unique leaves file untouched");

  // edit: oldText not found -> error
  r = await edit.execute("12", { path: "a.txt", edits: [{ oldText: "nope", newText: "x" }] }, undefined, undefined, {});
  eq(r.details.exitCode, 1, "edit not-found -> exitCode 1");

  // edit: edits[] arriving as a JSON string is tolerated
  await write.execute("13", { path: "js.txt", content: "foo" }, undefined, undefined, {});
  r = await edit.execute("14", { path: "js.txt", edits: JSON.stringify([{ oldText: "foo", newText: "bar" }]) }, undefined, undefined, {});
  eq(r.details.exitCode, 0, "edit tolerates stringified edits[]");
  eq(readFileSync(join(sandbox, "js.txt"), "utf-8"), "bar", "edit stringified-edits applied");

  // bash: returncode-prefixed combined output
  r = await bash.execute("15", { command: "echo hi" }, undefined, undefined, {});
  assertCanonical(r, "bash");
  check(/^returncode: 0\nhi/.test(textOf(r)), "bash returncode prefix + stdout");

  // ---- register_provider.js: env -> pi.registerProvider ------------------- //
  process.env.PI_BENCH_PROVIDER = "bench";
  process.env.PI_BENCH_BASE_URL = "https://api.example/v1";
  process.env.PI_BENCH_MODEL = "zai-org/GLM";
  process.env.PI_BENCH_CONTEXT_WINDOW = "200000";
  process.env.PI_BENCH_MAX_TOKENS = "4096";
  const regExt = await loadExt("register_provider.js");
  const pi2 = makePi();
  regExt(pi2);
  eq(pi2.providers.length, 1, "register_provider registers exactly one provider");
  const reg = pi2.providers[0];
  eq(reg.name, "bench", "provider name");
  eq(reg.config.api, "openai-completions", "provider api");
  eq(reg.config.baseUrl, "https://api.example/v1", "provider baseUrl");
  eq(reg.config.apiKey, "$OPENAI_API_KEY", "provider apiKey stays a $-ref");
  const m = reg.config.models[0];
  eq(m.id, "zai-org/GLM", "model id");
  eq(m.contextWindow, 200000, "model contextWindow from env");
  eq(m.maxTokens, 4096, "model maxTokens from env");
  // dynamic registerProvider needs the full model config (no defaults filled in)
  for (const f of ["name", "reasoning", "input", "cost"]) {
    check(m[f] !== undefined, `model config must spell out "${f}"`);
  }

  // register_provider is a no-op without the env (e.g. older pi / unset)
  delete process.env.PI_BENCH_PROVIDER;
  const pi3 = makePi();
  regExt(pi3);
  eq(pi3.providers.length, 0, "register_provider no-ops without PI_BENCH_PROVIDER");
}

main()
  .catch((e) => failures.push(`threw: ${e && e.stack ? e.stack : e}`))
  .finally(() => {
    server.close();
    try { rmSync(workdir, { recursive: true, force: true }); } catch {}
    try { rmSync(sandbox, { recursive: true, force: true }); } catch {}
    if (failures.length) {
      for (const f of failures) console.error(`FAIL: ${f}`);
      process.exit(1);
    }
    console.log("OK: pi_ext smoke passed");
    process.exit(0);
  });
