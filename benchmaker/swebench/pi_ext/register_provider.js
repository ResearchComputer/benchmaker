// Register the benchmaker OpenAI-compatible provider straight from the
// environment, instead of relying on a staged models.json.
//
// Why: pi resolves its config dir (and therefore models.json) via Node's
// os.homedir(), while the Python side stages that file by expanding $HOME with
// bash. When the two disagree — common when pi runs on the host with a custom
// HOME — pi never sees the custom provider and dies with `Unknown provider
// "bench"`. Reading the endpoint/model out of env and calling pi.registerProvider
// here removes that filesystem dependency entirely.
//
// pi.registerProvider during initial extension load is queued and applied once
// the runner binds its context, so `--provider <name> --model <id>` resolves.
// The apiKey is kept as a `$OPENAI_API_KEY` reference; pi expands `$VAR` config
// values from the environment at request time (resolve-config-value.ts), so the
// secret never has to be written anywhere.
//
// No-op (leaving any staged models.json to do the job) when the running pi build
// predates pi.registerProvider, or when PI_BENCH_* is not set. Staged by
// benchmaker.swebench.pi_agent.PiHostAgent for `--agent pi-host`.
export default function (pi) {
  if (!pi || typeof pi.registerProvider !== "function") {
    return; // older pi build without dynamic providers — rely on models.json
  }

  const provider = process.env.PI_BENCH_PROVIDER;
  const baseUrl = process.env.PI_BENCH_BASE_URL;
  const model = process.env.PI_BENCH_MODEL;
  if (!provider || !baseUrl || !model) {
    return; // nothing to register
  }

  const toInt = (raw, fallback) => {
    const n = raw ? parseInt(raw, 10) : NaN;
    return Number.isFinite(n) && n > 0 ? n : fallback;
  };
  const contextWindow = toInt(process.env.PI_BENCH_CONTEXT_WINDOW, 128000);
  const maxTokens = toInt(process.env.PI_BENCH_MAX_TOKENS, 8192);
  // A `$VAR` reference by default, so the key stays out of process args/files.
  const apiKey = process.env.PI_BENCH_API_KEY_REF || "$OPENAI_API_KEY";

  try {
    // The dynamic registerProvider path does not fill in model defaults the way
    // models.json parsing does, so spell out the full model config.
    pi.registerProvider(provider, {
      baseUrl,
      api: "openai-completions",
      apiKey,
      models: [
        {
          id: model,
          name: model,
          reasoning: false,
          input: ["text"],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow,
          maxTokens,
        },
      ],
    });
  } catch (err) {
    try {
      process.stderr.write(`[register_provider] failed: ${err}\n`);
    } catch (_) {}
  }
}
