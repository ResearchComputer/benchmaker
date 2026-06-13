// Route ALL of pi's core tools — bash, read, write, edit — into the task
// environment (/testbed) via the localhost bridge, so a host-loop pi run has the
// same four tools a sandbox-loop pi run gets (its builtins). This is the opt-in
// tool-parity variant of remote_exec.js, which overrides only `bash`.
//
// Every file action is expressed as a shell command POSTed to $PI_EXEC_BRIDGE
// (the same bridge `bash` uses), so edits land in the environment, never on the
// host filesystem. read = cat/sed; write = base64-decode; edit = read-modify-
// write of exact-text replacements (the file is read from and written back to
// the environment; only the in-memory string patch happens host-side).
//
// The host agent activates this by setting settings.json `tools` to
// ["bash","read","write","edit"] AND loading this file instead of remote_exec.js
// (loading both would double-register `bash`). No-op without a bridge.
export default function (pi) {
  const bridgeUrl = process.env.PI_EXEC_BRIDGE;
  if (!bridgeUrl) {
    return; // no bridge configured — leave the built-in tools in place
  }

  async function remoteExec(command, timeoutSec) {
    const resp = await fetch(`${bridgeUrl}/exec`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ command, timeout: timeoutSec || 600 }),
    });
    if (!resp.ok) {
      return { return_code: -1, stdout: "", stderr: `bridge HTTP ${resp.status}` };
    }
    return await resp.json();
  }

  // POSIX single-quote a value for safe interpolation into a shell command.
  const sq = (s) => `'${String(s).replace(/'/g, `'\\''`)}'`;
  const rcOf = (res) => (typeof res.return_code === "number" ? res.return_code : 0);
  const textResult = (text, exitCode) => ({
    content: [{ type: "text", text }],
    details: { exitCode },
  });
  const joinStreams = (res) =>
    (res.stdout || "") +
    (res.stderr ? (res.stdout ? "\n" : "") + res.stderr : "");

  // --- bash: identical contract to remote_exec.js -------------------------- //
  pi.registerTool({
    name: "bash",
    label: "Bash (remote)",
    description:
      "Execute a shell command in the task environment (/testbed). Combined " +
      "stdout+stderr is returned. The working directory does not persist " +
      "between calls; filesystem edits do.",
    parameters: {
      type: "object",
      properties: {
        command: { type: "string", description: "The shell command to run." },
        timeout: { type: "number", description: "Timeout in seconds." },
      },
      required: ["command"],
    },
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const res = await remoteExec(params.command, params.timeout);
      const rc = rcOf(res);
      return textResult(`returncode: ${rc}\n${joinStreams(res)}`, rc);
    },
  });

  // --- read: cat (optionally line-sliced) in the environment --------------- //
  pi.registerTool({
    name: "read",
    label: "Read (remote)",
    description:
      "Read the contents of a file in the task environment (/testbed). Use " +
      "offset/limit (1-indexed lines) for large files.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Path to the file to read." },
        offset: { type: "number", description: "1-indexed line to start from." },
        limit: { type: "number", description: "Maximum number of lines to read." },
      },
      required: ["path"],
    },
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const { path, offset, limit } = params;
      let cmd;
      const start = offset && offset > 0 ? Math.floor(offset) : 1;
      if (limit && limit > 0) {
        const end = start + Math.floor(limit) - 1;
        cmd = `sed -n ${sq(`${start},${end}p`)} ${sq(path)}`;
      } else if (start > 1) {
        cmd = `tail -n +${start} ${sq(path)}`;
      } else {
        cmd = `cat ${sq(path)}`;
      }
      const res = await remoteExec(cmd);
      const rc = rcOf(res);
      if (rc !== 0) {
        return textResult(
          `read: cannot read ${path}\n${joinStreams(res) || `exit ${rc}`}`,
          rc,
        );
      }
      // Return the file contents verbatim (no returncode prefix), matching the
      // built-in read tool the sandbox-loop run uses.
      return textResult(res.stdout || "", 0);
    },
  });

  // --- write: base64-safe whole-file write into the environment ------------ //
  pi.registerTool({
    name: "write",
    label: "Write (remote)",
    description:
      "Write content to a file in the task environment (/testbed), creating " +
      "parent directories. Creates the file or overwrites it entirely.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Path to the file to write." },
        content: { type: "string", description: "Full content to write." },
      },
      required: ["path", "content"],
    },
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const { path } = params;
      const b64 = Buffer.from(String(params.content ?? ""), "utf-8").toString("base64");
      const cmd =
        `mkdir -p "$(dirname ${sq(path)})" && ` +
        `printf %s ${sq(b64)} | base64 -d > ${sq(path)}`;
      const res = await remoteExec(cmd);
      const rc = rcOf(res);
      if (rc !== 0) {
        return textResult(
          `write: failed for ${path}\n${joinStreams(res) || `exit ${rc}`}`,
          rc,
        );
      }
      return textResult(`Wrote ${path}`, 0);
    },
  });

  // --- edit: exact-text replacement (read -> patch in memory -> write back) - //
  function normalizeEdits(params) {
    let edits = params && params.edits;
    // Some models send edits[] as a JSON-encoded string; others send a single
    // top-level {oldText,newText}. Accept both (mirrors pi's prepareArguments).
    if (typeof edits === "string") {
      try {
        const parsed = JSON.parse(edits);
        if (Array.isArray(parsed)) edits = parsed;
      } catch (_) {}
    }
    if (!Array.isArray(edits)) {
      if (params && typeof params.oldText === "string" && typeof params.newText === "string") {
        return [{ oldText: params.oldText, newText: params.newText }];
      }
      return [];
    }
    return edits;
  }

  pi.registerTool({
    name: "edit",
    label: "Edit (remote)",
    description:
      "Edit a single file in the task environment (/testbed) using exact text " +
      "replacement. Each edits[].oldText must match a unique region of the file.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Path to the file to edit." },
        edits: {
          type: "array",
          description: "One or more exact-text replacements.",
          items: {
            type: "object",
            properties: {
              oldText: { type: "string", description: "Exact, unique text to replace." },
              newText: { type: "string", description: "Replacement text." },
            },
            required: ["oldText", "newText"],
          },
        },
      },
      required: ["path", "edits"],
    },
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const path = params.path;
      const edits = normalizeEdits(params);
      if (edits.length === 0) {
        return textResult("edit: no edits provided", 1);
      }
      const readRes = await remoteExec(`cat ${sq(path)}`);
      const readRc = rcOf(readRes);
      if (readRc !== 0) {
        return textResult(
          `edit: cannot read ${path}\n${joinStreams(readRes) || `exit ${readRc}`}`,
          readRc || 1,
        );
      }
      let content = readRes.stdout || "";
      for (let i = 0; i < edits.length; i++) {
        const { oldText, newText } = edits[i];
        if (typeof oldText !== "string" || typeof newText !== "string") {
          return textResult(`edit ${i}: oldText and newText must be strings`, 1);
        }
        const first = content.indexOf(oldText);
        if (first === -1) {
          return textResult(`edit ${i}: oldText not found in ${path}`, 1);
        }
        if (content.indexOf(oldText, first + oldText.length) !== -1) {
          return textResult(
            `edit ${i}: oldText is not unique in ${path}; add surrounding context`,
            1,
          );
        }
        content = content.slice(0, first) + newText + content.slice(first + oldText.length);
      }
      const b64 = Buffer.from(content, "utf-8").toString("base64");
      const writeRes = await remoteExec(`printf %s ${sq(b64)} | base64 -d > ${sq(path)}`);
      const writeRc = rcOf(writeRes);
      if (writeRc !== 0) {
        return textResult(
          `edit: write-back failed for ${path}\n${joinStreams(writeRes) || `exit ${writeRc}`}`,
          writeRc || 1,
        );
      }
      return textResult(`Applied ${edits.length} edit(s) to ${path}`, 0);
    },
  });
}
