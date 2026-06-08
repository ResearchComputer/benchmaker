// Cap pi's agentic turns.
//
// pi has no native turn/step limit — its agent loop runs until the model stops
// calling tools. This extension reads PI_MAX_TURNS from the environment and,
// once that many turns have completed, aborts the agent loop so pi stops. The
// in-tree edits already on disk remain for harbor's verifier to grade.
//
// It is a no-op when PI_MAX_TURNS is unset / <= 0, and degrades gracefully (no
// cap) if the running pi build does not emit `turn_end` or does not expose
// `ctx.abort`/`ctx.shutdown` — so loading it is always safe. Counting the
// extension's own `turn_end` events (rather than trusting an event's turnIndex)
// keeps it correct regardless of 0- vs 1-based indexing.
//
// Loaded by benchmaker.swebench.pi_agent: PiContainerAgent passes it via
// `--extension`; PiHostAgent stages it into the agent extensions dir.
export default function (pi) {
  const raw = process.env.PI_MAX_TURNS;
  const maxTurns = raw ? parseInt(raw, 10) : 0;
  if (!Number.isFinite(maxTurns) || maxTurns <= 0) {
    return; // no cap configured
  }
  if (!pi || typeof pi.on !== "function") {
    return; // unexpected API shape; do nothing rather than crash pi
  }

  let completed = 0;
  let aborted = false;

  const stop = (ctx) => {
    if (aborted) return;
    aborted = true;
    try {
      process.stderr.write(
        `[max_turns] reached cap of ${maxTurns} turns; stopping agent\n`,
      );
    } catch (_) {}
    try {
      if (ctx && typeof ctx.abort === "function") {
        ctx.abort();
      } else if (ctx && typeof ctx.shutdown === "function") {
        ctx.shutdown();
      }
    } catch (_) {}
  };

  pi.on("turn_end", (_event, ctx) => {
    completed += 1;
    if (completed >= maxTurns) {
      stop(ctx);
    }
  });
}
