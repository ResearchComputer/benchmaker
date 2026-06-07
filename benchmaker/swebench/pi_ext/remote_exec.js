export default function (pi) {
  const bridgeUrl = process.env.PI_EXEC_BRIDGE;
  if (!bridgeUrl) {
    // No bridge configured — leave the built-in bash tool in place.
    return;
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
      const out =
        (res.stdout || "") +
        (res.stderr ? (res.stdout ? "\n" : "") + res.stderr : "");
      const rc = typeof res.return_code === "number" ? res.return_code : 0;
      return {
        output: `returncode: ${rc}\n${out}`,
        exitCode: rc,
      };
    },
  });
}
