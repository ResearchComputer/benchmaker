"""HuggingFace `datasets` workload.

Yields per-request dicts shaped for downstream LLM workload-types — usually
`{"prompt": ..., "reference": ..., <extra_fields>: ...}`. Pair with
`OpenAIChatWorkloadType` (which promotes `prompt` → `messages`) and
`EvalWorkloadType` + `correctness_hook` (which reads `reference`).

The `datasets` library is an optional dependency. Install with:

    pip install -e .[hf]
"""

from __future__ import annotations

import random
import re
from typing import Any, Callable, Iterable, Mapping, Optional, Union

from benchmaker.workloads.datasets import Workload


# --------------------------------------------------------------------------- #
# Reference transforms (postprocessing on the raw reference field)
# --------------------------------------------------------------------------- #


def _gsm8k_answer(text: str) -> str:
    """GSM8K answers end in `#### <number>`; return that final number."""
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return m.group(0) if m else text.strip()


def _strip(text: str) -> str:
    return text.strip()


def _first_line(text: str) -> str:
    return text.split("\n", 1)[0].strip()


def _lower_strip(text: str) -> str:
    return text.strip().lower()


_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "identity":    lambda s: s,
    "strip":       _strip,
    "lower_strip": _lower_strip,
    "first_line":  _first_line,
    "gsm8k":       _gsm8k_answer,
    "gsm8k_answer": _gsm8k_answer,
}


def get_transform(name: str) -> Callable[[str], str]:
    try:
        return _TRANSFORMS[name]
    except KeyError:
        raise ValueError(
            f"unknown reference_transform {name!r}; known: {sorted(_TRANSFORMS)}"
        )


# --------------------------------------------------------------------------- #
# Dataset presets (path + subset + field names)
# --------------------------------------------------------------------------- #


PresetSpec = dict[str, Any]

_PRESETS: dict[str, PresetSpec] = {
    "gsm8k": {
        "path": "gsm8k",
        "name": "main",
        "split": "test",
        "prompt_field": "question",
        "reference_field": "answer",
        "reference_transform": "gsm8k_answer",
    },
    "mmlu": {
        # MMLU has many configs; default to "all". Override with `name=...`.
        "path": "cais/mmlu",
        "name": "all",
        "split": "test",
        # MMLU rows: question, choices (list[str]), answer (int idx). We build
        # a multiple-choice prompt via a template and ship the letter answer.
        "prompt_template": (
            "{question}\n"
            "A. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}\n"
            "Answer with a single letter (A, B, C, or D)."
        ),
        "prompt_template_fields": {
            "question":  "question",
            "choice_a":  ("choices", 0),
            "choice_b":  ("choices", 1),
            "choice_c":  ("choices", 2),
            "choice_d":  ("choices", 3),
        },
        "reference_field": "answer",
        "reference_transform": "mmlu_letter",
    },
    "humaneval": {
        "path": "openai_humaneval",
        "split": "test",
        "prompt_field": "prompt",
        "reference_field": "canonical_solution",
        "extra_fields": ("task_id", "test", "entry_point"),
    },
}

# Late-bound transforms that need argument bindings.
_TRANSFORMS["mmlu_letter"] = lambda v: ("ABCD"[int(v)] if isinstance(v, int)
                                        or (isinstance(v, str) and v.isdigit())
                                        else str(v).strip().upper()[:1])


def list_presets() -> list[str]:
    return sorted(_PRESETS)


def get_preset(name: str) -> PresetSpec:
    try:
        return dict(_PRESETS[name])
    except KeyError:
        raise ValueError(
            f"unknown HF preset {name!r}; known: {list_presets()}"
        )


# --------------------------------------------------------------------------- #
# The workload
# --------------------------------------------------------------------------- #


class HFDatasetWorkload(Workload):
    """Load a HuggingFace dataset and yield `{prompt, reference, ...}` items.

    Args:
        path: HuggingFace dataset id (`"gsm8k"`, `"cais/mmlu"`, ...) — required
            unless `preset` is given.
        name: dataset config / subset (`"main"` for gsm8k, etc.).
        split: split name (default `"test"`).
        preset: one of `list_presets()`. Fills in `path/name/split/field`
            defaults; explicit kwargs override.

        prompt_field: source row field whose value becomes `"prompt"`.
            Mutually exclusive with `prompt_template`.
        prompt_template: format string referencing fields from
            `prompt_template_fields`. Use this when the prompt is composed of
            several columns (MMLU, BBH, etc.).
        prompt_template_fields: maps `{template_var: row_field}` (or
            `(row_field, index)` to index into a list field). Required when
            `prompt_template` is set.

        reference_field: source row field whose value becomes `"reference"`.
            Omit to skip producing a reference (e.g. for non-eval generation).
        reference_transform: callable `(value) -> str` or a registered
            transform name (`"gsm8k_answer"`, `"strip"`, ...).

        extra_fields: row fields to carry through unchanged on each item
            (handy for `EvalWorkloadType(extra_meta_keys=...)`).

        max_items: stop after this many items.
        shuffle: permute once at construction (non-streaming only).
        seed: RNG seed for shuffle.
        streaming: use `datasets`'s streaming mode (no shuffle, no len()).
        loop: cycle the dataset when exhausted (non-streaming only).
        cache_dir / token / trust_remote_code: forwarded to `load_dataset`.
    """

    name: str = "hf"

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        name: Optional[str] = None,
        split: str = "test",
        preset: Optional[str] = None,

        prompt_field: Optional[str] = "prompt",
        prompt_template: Optional[str] = None,
        prompt_template_fields: Optional[Mapping[str, Any]] = None,

        reference_field: Optional[str] = "reference",
        reference_transform: Optional[Union[str, Callable[[Any], str]]] = None,

        extra_fields: Iterable[str] = (),

        max_items: Optional[int] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        streaming: bool = False,
        loop: bool = False,

        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        trust_remote_code: bool = False,

        workload_name: Optional[str] = None,
    ):
        spec: PresetSpec = {}
        if preset:
            spec = get_preset(preset)

        path = path or spec.get("path")
        if not path:
            raise ValueError("HFDatasetWorkload requires `path` (or a `preset`)")

        name = name if name is not None else spec.get("name")
        # `split` is keyword-only with a default ("test"). Only fall back to
        # the preset's value if the caller didn't explicitly override.
        if split == "test" and "split" in spec:
            split = spec["split"]

        if prompt_template is None and spec.get("prompt_template"):
            prompt_template = spec["prompt_template"]
            prompt_template_fields = (prompt_template_fields
                                      or spec.get("prompt_template_fields"))
            prompt_field = None
        elif prompt_field == "prompt" and "prompt_field" in spec:
            prompt_field = spec["prompt_field"]

        if reference_field == "reference" and "reference_field" in spec:
            reference_field = spec["reference_field"]

        if reference_transform is None and "reference_transform" in spec:
            reference_transform = spec["reference_transform"]

        if isinstance(reference_transform, str):
            reference_transform = get_transform(reference_transform)

        if prompt_template and not prompt_template_fields:
            raise ValueError(
                "prompt_template requires prompt_template_fields to map "
                "template vars onto row fields"
            )
        if prompt_template and prompt_field:
            # Caller supplied both; template wins (prompt_field auto-cleared
            # above when preset injected the template).
            prompt_field = None
        if not prompt_template and not prompt_field:
            raise ValueError("Either prompt_field or prompt_template must be set")

        extra_fields = tuple(extra_fields) or tuple(spec.get("extra_fields") or ())

        self._path = path
        self._config_name = name
        self._split = split
        self._prompt_field = prompt_field
        self._prompt_template = prompt_template
        self._prompt_template_fields = dict(prompt_template_fields or {})
        self._reference_field = reference_field
        self._reference_transform = reference_transform
        self._extra_fields = tuple(extra_fields)
        self._max = max_items
        self._loop = loop
        self._streaming = streaming
        self._cache_dir = cache_dir
        self._token = token
        self._trust_remote_code = trust_remote_code

        if workload_name:
            self.name = workload_name
        else:
            tag = path
            if name:
                tag = f"{tag}:{name}"
            self.name = f"hf:{tag}/{split}"

        self._iter: Any = None
        self._n = 0
        self._shuffle = shuffle
        self._seed = seed

    # ---- HF wiring ---- #

    def _load(self) -> Any:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "HFDatasetWorkload requires the `datasets` package. "
                "Install with `pip install datasets` or `pip install -e .[hf]`."
            ) from e

        kwargs: dict[str, Any] = {"split": self._split}
        if self._config_name is not None:
            kwargs["name"] = self._config_name
        if self._cache_dir is not None:
            kwargs["cache_dir"] = self._cache_dir
        if self._token is not None:
            kwargs["token"] = self._token
        if self._trust_remote_code:
            kwargs["trust_remote_code"] = True
        if self._streaming:
            kwargs["streaming"] = True
        return load_dataset(self._path, **kwargs)

    def _ensure_iter(self) -> None:
        if self._iter is not None:
            return
        ds = self._load()
        if self._streaming:
            self._iter = iter(ds)
            self._source = None
        else:
            indices = list(range(len(ds)))
            if self._shuffle:
                random.Random(self._seed).shuffle(indices)
            self._source = ds
            self._indices = indices
            self._cursor = 0
            self._iter = self._yield_indexed()

    def _yield_indexed(self):
        while True:
            if self._cursor >= len(self._indices):
                if not self._loop:
                    return
                self._cursor = 0
            idx = self._indices[self._cursor]
            self._cursor += 1
            yield self._source[idx]

    # ---- item shaping ---- #

    def _shape(self, row: Mapping[str, Any]) -> dict[str, Any]:
        item: dict[str, Any] = {}
        if self._prompt_template:
            ctx: dict[str, Any] = {}
            for var, src in self._prompt_template_fields.items():
                ctx[var] = _lookup(row, src)
            item["prompt"] = self._prompt_template.format(**ctx)
        elif self._prompt_field:
            item["prompt"] = row[self._prompt_field]

        if self._reference_field and self._reference_field in row:
            value = row[self._reference_field]
            if self._reference_transform is not None:
                value = self._reference_transform(value)
            item["reference"] = value

        for f in self._extra_fields:
            if f in row:
                item[f] = row[f]
        return item

    # ---- Workload protocol ---- #

    async def next_item(self) -> Any:
        if self._max is not None and self._n >= self._max:
            raise StopAsyncIteration
        self._ensure_iter()
        try:
            row = next(self._iter)
        except StopIteration:
            raise StopAsyncIteration
        self._n += 1
        return self._shape(row)

    async def aclose(self) -> None:
        self._iter = None
        return None


def _lookup(row: Mapping[str, Any], src: Any) -> Any:
    """Resolve a source spec against a row.

    `src` is either a field name (`"question"`) or a tuple
    `(field_name, index, ...)` for indexing into list/dict fields.
    """
    if isinstance(src, str):
        return row[src]
    if isinstance(src, tuple) and src:
        value = row[src[0]]
        for key in src[1:]:
            value = value[key]
        return value
    raise TypeError(f"bad template field spec {src!r}")
