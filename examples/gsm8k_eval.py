"""End-to-end: stream GSM8K from HuggingFace, hit an OpenAI-compatible
endpoint, grade each answer with a regex scorer that extracts the final number.

Requires:  pip install -e .[hf]

Usage — explicit endpoint:
    python -m examples.gsm8k_eval \\
        --url http://localhost:8000/v1/chat/completions \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --max-items 200 --rps 4 --duration 5m

Usage — endpoint from .env (OPENAI_API_BASE_URL / OPENAI_COMPATIBLE_MODEL /
OPENAI_API_KEY), one-liner:
    python -m examples.gsm8k_eval --max-items 200 --rps 4 --duration 5m

The dataset's `answer` field looks like:

    Reasoning step 1...
    Reasoning step 2...
    #### 42

`HFDatasetWorkload(preset="gsm8k")` transforms that to just `"42"` via
`reference_transform="gsm8k_answer"`. The regex scorer below pulls the last
integer out of the model's reply and compares it to the reference.
"""

import argparse
import asyncio
import re
import sys

from benchmaker import (
    BenchConfig,
    BenchRunner,
    EvalWorkloadType,
    HFDatasetWorkload,
    OpenAIChatWorkloadType,
    Scorer,
    correctness_hook,
    parse_rate_spec,
)
from benchmaker.core.load import parse_duration


def gsm8k_numeric_scorer() -> Scorer:
    """Pull the last integer out of the model output and compare to reference."""
    last_int = re.compile(r"-?\d+(?:\.\d+)?")

    def _score(reference, prediction):
        matches = last_int.findall((prediction or "").replace(",", ""))
        if not matches:
            return {"correct": 0.0, "answered": 0.0}
        # GSM8K answers are integers; reference comes pre-stripped from the preset.
        try:
            pred_value = float(matches[-1])
            ref_value = float(str(reference).replace(",", "").strip())
        except ValueError:
            return {"correct": 0.0, "answered": 1.0}
        return {
            "correct": 1.0 if pred_value == ref_value else 0.0,
            "answered": 1.0,
        }

    return _score


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None,
                    help="defaults to $OPENAI_API_BASE_URL + /chat/completions")
    ap.add_argument("--model", default=None,
                    help="defaults to $OPENAI_COMPATIBLE_MODEL / $OPENAI_MODEL")
    ap.add_argument("--api-key", default=None,
                    help="defaults to $OPENAI_API_KEY")
    ap.add_argument("--dotenv", default=".env",
                    help="path to .env (default: .env in cwd; '' to disable)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--rps", default="4")
    ap.add_argument("--duration", default="10m")
    ap.add_argument("--bundle", default=None)
    args = ap.parse_args()

    workload = HFDatasetWorkload(
        preset="gsm8k",
        split=args.split,
        max_items=args.max_items,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    base = OpenAIChatWorkloadType.from_env(
        url=args.url, model=args.model, api_key=args.api_key,
        dotenv_path=(args.dotenv or None),
        max_tokens=args.max_tokens, temperature=0.0,
    )
    wt = EvalWorkloadType(base)
    hook = correctness_hook(gsm8k_numeric_scorer())

    load = parse_rate_spec(args.rps, duration_s=parse_duration(args.duration))
    runner = BenchRunner(BenchConfig(
        workload_type=wt, workload=workload, load=load,
        post_hooks=[hook], timeout_s=600.0,
    ))
    await runner.run()
    runner.metrics.render(sys.stdout)
    if args.bundle:
        path = runner.write_bundle(args.bundle)
        print(f"\nwrote bundle: {path}")


if __name__ == "__main__":
    asyncio.run(main())
