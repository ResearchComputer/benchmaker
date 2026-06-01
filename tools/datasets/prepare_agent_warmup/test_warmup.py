"""Unit tests for the warmup-dataset normalizers + SWE-bench pure helpers.

Run from this directory:  pytest test_warmup.py -q
These cover the format logic only; the live SWE-bench rollout/verify path needs
flash-sandbox + the swebench package and is not exercised here.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P
import swebench_runner as S


# --------------------------- protocol: reasoning --------------------------- #

def test_split_reasoning():
    r, c = P.split_reasoning("<think>my plan</think>\nThe answer is 42.")
    assert r == "my plan"
    assert c == "The answer is 42."
    r2, c2 = P.split_reasoning("no think here")
    assert r2 is None and c2 == "no think here"


# --------------------------- normalize_oai_messages ------------------------ #

def test_normalize_claude_style():
    row = {"category": "coding", "model": "claude-opus-4-6", "messages": [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "is 5,12,13 a right triangle?"},
        {"role": "assistant", "content": "<think>5^2+12^2=13^2</think>Yes."},
    ]}
    rec = P.normalize_oai_messages(row, source="claude", id_prefix="claude:code",
                                   row_index=3, meta_keys=("category", "model"))
    assert P.validate(rec) is None
    assert rec.id == "claude:code:row3"
    a = rec.messages[-1]
    assert a["role"] == "assistant" and a["content"] == "Yes."
    assert a["reasoning"] == "5^2+12^2=13^2"
    assert rec.meta["category"] == "coding" and rec.meta["model"] == "claude-opus-4-6"
    assert rec.verified is False


# --------------------------- normalize_hermes ------------------------------ #

def test_normalize_hermes_tool_pairing():
    row = {
        "id": "abc",
        "tools": json.dumps([{"name": "write_file",
                              "description": "w", "parameters": {"type": "object"}}]),
        "conversations": [
            {"from": "system", "value": "tools: <tools>...</tools>"},
            {"from": "human", "value": "make a file"},
            {"from": "gpt", "value": "<think>plan</think>Sure.\n"
                                     "<tool_call>\n{\"name\": \"write_file\", "
                                     "\"arguments\": {\"path\": \"a.py\"}}\n</tool_call>"},
            {"from": "tool", "value": "<tool_response>\n{\"tool_call_id\": "
                                      "\"srv-1\", \"name\": \"write_file\", "
                                      "\"content\": {\"bytes\": 10}}\n</tool_response>"},
            {"from": "gpt", "value": "Done."},
        ],
    }
    rec = P.normalize_hermes(row, source="hermes", id_prefix="hermes:glm", row_index=0)
    assert P.validate(rec) is None
    assert rec.tools and rec.tools[0]["type"] == "function"
    assert rec.tools[0]["function"]["name"] == "write_file"
    asst = rec.messages[2]
    assert asst["reasoning"] == "plan" and asst["content"] == "Sure."
    call = asst["tool_calls"][0]
    assert call["function"]["name"] == "write_file"
    # tool result must reference the assistant call's synthesized id (positional)
    tool = rec.messages[3]
    assert tool["role"] == "tool" and tool["tool_call_id"] == call["id"]


# --------------------------- normalize_pi_traces --------------------------- #

def test_normalize_pi_traces():
    traces = [
        {"type": "session", "id": "sess-1"},
        {"type": "message", "message": {"role": "user",
            "content": [{"type": "text", "text": "fix the bug"}]}},
        {"type": "message", "message": {"role": "assistant", "model": "qwen",
            "content": [
                {"type": "thinking", "thinking": "let me look"},
                {"type": "text", "text": "I'll list files."},
                {"type": "toolCall", "id": "tc-1", "name": "bash",
                 "arguments": {"command": "ls", "path": "", "timeout": None}},
            ]}},
        {"type": "message", "message": {"role": "toolResult", "toolCallId": "tc-1",
            "toolName": "bash", "content": [{"type": "text", "text": "a.py"}]}},
        {"type": "message", "message": {"role": "assistant",
            "content": [{"type": "text", "text": "done"}]}},
    ]
    rec = P.normalize_pi_traces({"traces": traces, "session_id": "sess-1"},
                                source="pi", id_prefix="pi", row_index=0)
    assert P.validate(rec) is None
    assert rec.id == "pi:sess-1"
    assert rec.meta.get("model") == "qwen"
    asst = rec.messages[1]
    assert asst["reasoning"] == "let me look"
    tc = asst["tool_calls"][0]
    assert tc["function"]["name"] == "bash"
    # null/empty args were dropped; command preserved
    assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}
    assert rec.messages[2]["role"] == "tool"
    assert rec.messages[2]["tool_call_id"] == "tc-1"


def test_validate_catches_bad_records():
    bad = P.WarmupRecord(id="", source="x", messages=[{"role": "user", "content": "hi"}])
    assert P.validate(bad) is not None
    dup = P.WarmupRecord(id="z", source="x", messages=[
        {"role": "user", "content": "hi"},
        P.assistant_msg(None, tool_calls=[P.make_tool_call("c", "bash", {})]),
        P.assistant_msg(None, tool_calls=[P.make_tool_call("c", "bash", {})]),
    ])
    assert "duplicate" in (P.validate(dup) or "")


# --------------------------- swebench pure helpers ------------------------- #

def test_parse_tool_call_and_record():
    tc = {"id": "x1", "type": "function",
          "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}
    name, args = S._parse_tool_call(tc)
    assert name == "bash" and args == {"command": "ls"}

    msg = {"role": "assistant", "content": "looking",
           "reasoning_content": "hmm", "tool_calls": [tc]}
    rec_msg = S._to_record_assistant(msg)
    assert rec_msg["reasoning"] == "hmm"
    assert rec_msg["tool_calls"][0]["function"]["name"] == "bash"
    api_msg = S._strip_for_api(msg)
    assert "reasoning_content" not in api_msg and api_msg["tool_calls"]


def test_format_and_truncate():
    out = S._format_exec({"exit_code": 0, "stdout": "hello", "stderr": ""})
    assert "exit_code=0" in out and "hello" in out
    long = S._truncate("x" * 100, 20)
    assert "truncated" in long and len(long) < 100


def test_build_initial_messages():
    msgs = S.build_initial_messages({"repo": "a/b", "problem_statement": "boom",
                                     "hints_text": "look here"})
    assert msgs[0]["role"] == "system"
    assert "boom" in msgs[1]["content"] and "look here" in msgs[1]["content"]
