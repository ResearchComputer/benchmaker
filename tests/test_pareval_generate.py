from benchmaker.pareval.generate import extract_code


def test_extract_fenced_with_lang():
    reply = "Sure:\n```cpp\nint f() { return 1; }\n```\nDone."
    assert extract_code(reply) == "int f() { return 1; }"

def test_extract_fenced_no_lang():
    reply = "```\nint f() {}\n```"
    assert extract_code(reply) == "int f() {}"

def test_extract_first_of_multiple_blocks():
    reply = "```cpp\nA\n```\nthen\n```cpp\nB\n```"
    assert extract_code(reply) == "A"

def test_extract_unfenced_returns_stripped():
    reply = "   int f() {}\n"
    assert extract_code(reply) == "int f() {}"


from benchmaker.pareval.generate import (
    assemble_generated_code, patch_no_inline, split_preamble,
)

PREAMBLE = "#include <omp.h>\nstruct Point { double x, y; };\n"
SIG = "double closestPair(std::vector<Point> const& points) {"
STUB = PREAMBLE + SIG

def test_split_preamble_keeps_context_above_signature():
    pre, sig = split_preamble(STUB)
    assert sig == SIG
    assert "struct Point" in pre and "#include <omp.h>" in pre

def test_assemble_retains_preamble_and_dedupes_signature():
    completion = SIG + "\n    return 0.0;\n}"
    src = assemble_generated_code(STUB, completion)
    assert "struct Point" in src
    assert src.count("closestPair(std::vector<Point> const& points)") == 1
    assert "return 0.0;" in src and "NO_INLINE" in src

def test_assemble_body_only_reply():
    body = "    return 0.0;\n}"
    src = assemble_generated_code(STUB, body)
    assert "struct Point" in src
    assert src.count("closestPair(") == 1
    assert "double NO_INLINE closestPair(" in src
    assert "return 0.0;" in src

def test_assemble_strips_preamble_the_model_re_emitted():
    completion = "#include <omp.h>\nstruct Point { double x, y; };\n" + SIG + "\n  return 1;\n}"
    src = assemble_generated_code(STUB, completion)
    assert src.count("struct Point") == 1

def test_patch_no_inline_inserts_after_return_type():
    assert patch_no_inline("double closestPair(std::vector<Point> const& p) {") == \
        "double NO_INLINE closestPair(std::vector<Point> const& p) {"


from benchmaker.pareval.dataset import ParEvalPrompt
from benchmaker.pareval.generate import generate_one

PROMPT = ParEvalPrompt(name="x", problem_type="t", language="cpp",
                       parallelism_model="omp",
                       prompt="int sum(std::vector<int> const& v) {")

async def test_generate_one_extracts_and_assembles():
    async def fake_send(messages):
        return ("```cpp\nint sum(std::vector<int> const& v) { return 0; }\n```", None)
    rec = await generate_one(fake_send, PROMPT, sample_idx=2)
    assert rec["name"] == "x" and rec["sample_idx"] == 2
    assert rec["error"] is None
    assert "NO_INLINE" in rec["generated_code"]
    assert rec["generated_code"].count("sum(") == 1

async def test_generate_one_records_error_on_send_failure():
    async def boom(messages):
        raise RuntimeError("model down")
    rec = await generate_one(boom, PROMPT, sample_idx=0)
    assert rec["error"] and "model down" in rec["error"]
    assert rec.get("generated_code") in (None, "")
