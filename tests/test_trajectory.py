# tests/test_trajectory.py
"""Unit tests for benchmaker.swebench.trajectory (pure converter + key derivation)."""
from __future__ import annotations

import json

from benchmaker.swebench import trajectory as T


def test_recorded_turn_roundtrip():
    turn = T.RecordedTurn(
        index=0, content="hi", reasoning="because",
        tool_calls=[{"id": "call_1", "name": "bash", "arguments": {"command": "ls"}}],
        finish_reason="tool_calls",
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    )
    rebuilt = T.RecordedTurn.from_dict(json.loads(json.dumps(turn.to_dict())))
    assert rebuilt == turn


def test_trajectory_roundtrip():
    traj = T.Trajectory(
        key="django__django-1", instance_id="django__django-1", model="m",
        trial="django__django-1__abc",
        turns=[T.RecordedTurn(0, "x", None, [], "stop", {})],
    )
    rebuilt = T.Trajectory.from_dict(json.loads(json.dumps(traj.to_dict())))
    assert rebuilt == traj
    assert traj.to_dict()["n_turns"] == 1


def test_task_key_parses_instance_id_from_prompt():
    messages = [
        {"role": "system", "content": "you are pi"},
        {"role": "user", "content": "Fix the bug.\n\n# Task: astropy__astropy-12907\nRepository: astropy"},
    ]
    assert T.task_key_from_messages(messages) == "astropy__astropy-12907"


def test_task_key_handles_list_content_and_hash_fallback():
    messages = [{"role": "user", "content": [{"type": "text", "text": "no task marker here"}]}]
    key = T.task_key_from_messages(messages)
    assert key.startswith("sha1:") and len(key) > 10


def _pi_log(*events) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"


def test_parse_pi_conversation_extracts_turns_and_key():
    log = _pi_log(
        {"type": "session", "id": "s1"},
        {"type": "message_start", "message": {"role": "user", "content": [
            {"type": "text", "text": "Fix it.\n# Task: django__django-11095\nRepository: django"}]}},
        "{ this is a corrupt line",  # must be skipped, not raise
        {"type": "turn_end", "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "let me look"},
                {"type": "text", "text": "investigating"},
                {"type": "toolCall", "id": "call_a", "name": "bash",
                 "arguments": {"command": "ls"}},
            ],
            "stopReason": "toolUse", "model": "zai-org/GLM-4.7-Flash",
            "usage": {"input": 1513, "output": 124, "cacheRead": 0, "cacheWrite": 5,
                      "totalTokens": 1637, "cost": {"total": 0.0}}}},
        {"type": "turn_end", "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "stopReason": "endTurn", "model": "zai-org/GLM-4.7-Flash",
            "usage": {"input": 2000, "output": 10, "totalTokens": 2010}}},
    )
    traj = T.parse_pi_conversation(log)
    assert traj.key == "django__django-11095"
    assert traj.instance_id == "django__django-11095"
    assert traj.model == "zai-org/GLM-4.7-Flash"
    assert len(traj.turns) == 2

    t0 = traj.turns[0]
    assert t0.index == 0
    assert t0.content == "investigating"
    assert t0.reasoning == "let me look"
    assert t0.finish_reason == "tool_calls"
    assert t0.tool_calls == [{"id": "call_a", "name": "bash", "arguments": {"command": "ls"}}]
    assert t0.usage == {"prompt_tokens": 1513, "completion_tokens": 124,
                        "total_tokens": 1637, "cache_read": 0, "cache_write": 5, "cost": 0.0}

    t1 = traj.turns[1]
    assert t1.finish_reason == "stop" and t1.content == "done"


def _pi_session(*events) -> str:
    return "\n".join(e if isinstance(e, str) else json.dumps(e) for e in events) + "\n"


def test_parse_pi_session_extracts_turns_key_and_tool_status():
    # pi's on-disk session log: one {"type":"message"} envelope per line, with
    # role in {user, assistant, toolResult}. This is what survives when a run is
    # killed at the wall-clock cap before pi-host.log is flushed.
    sess = _pi_session(
        {"type": "session", "version": 3, "id": "s1"},
        {"type": "model_change", "provider": "bench", "modelId": "m"},
        {"type": "message", "message": {"role": "user", "content": [
            {"type": "text", "text": "Fix it.\n# Task: django__django-11095\nRepository: django"}]}},
        "{ corrupt line",  # must be skipped, not raise
        {"type": "message", "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "let me look"},
                {"type": "text", "text": "investigating"},
                {"type": "toolCall", "id": "call_a", "name": "bash",
                 "arguments": {"command": "ls"}}],
            "stopReason": "toolUse", "model": "zai-org/GLM-4.7-Flash",
            "usage": {"input": 1513, "output": 124, "cacheRead": 0, "cacheWrite": 5,
                      "totalTokens": 1637, "cost": {"total": 0.0}}}},
        {"type": "message", "message": {
            "role": "toolResult", "toolCallId": "call_a", "toolName": "bash",
            "content": [{"type": "text", "text": "returncode: 0\nok"}], "isError": False}},
        {"type": "message", "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "stopReason": "endTurn", "model": "zai-org/GLM-4.7-Flash",
            "usage": {"input": 2000, "output": 10, "totalTokens": 2010}}},
    )
    traj = T.parse_pi_session(sess)
    assert traj.key == "django__django-11095"
    assert traj.instance_id == "django__django-11095"
    assert traj.model == "zai-org/GLM-4.7-Flash"
    assert len(traj.turns) == 2

    t0 = traj.turns[0]
    assert t0.index == 0
    assert t0.content == "investigating"
    assert t0.reasoning == "let me look"
    assert t0.finish_reason == "tool_calls"
    assert t0.tool_calls == [{"id": "call_a", "name": "bash", "arguments": {"command": "ls"}}]
    assert t0.usage == {"prompt_tokens": 1513, "completion_tokens": 124,
                        "total_tokens": 1637, "cache_read": 0, "cache_write": 5, "cost": 0.0}
    # tool result status comes from the toolResult message's content
    assert t0.tool_results == [{"name": "bash", "status": 0}]

    assert traj.turns[1].finish_reason == "stop" and traj.turns[1].content == "done"


def test_parse_pi_session_empty_is_safe():
    traj = T.parse_pi_session("")
    assert traj.turns == [] and traj.model == ""


def test_drop_nonfinal_actionless_turns_elides_and_reindexes():
    turns = [
        T.RecordedTurn(0, "", None, [{"id": "c", "name": "bash", "arguments": {}}],
                       "tool_calls", {}),
        T.RecordedTurn(1, "", None, [], "stop", {}),    # action-less mid turn (truncation/empty)
        T.RecordedTurn(2, "thinking only", None, [], "stop", {}),  # also action-less, dropped
        T.RecordedTurn(3, "", None, [{"id": "d", "name": "write", "arguments": {}}],
                       "tool_calls", {}),
        T.RecordedTurn(4, "done", None, [], "stop", {}),  # genuine final turn — always kept
    ]
    kept = T.drop_nonfinal_actionless_turns(turns)
    assert [t.tool_calls and t.tool_calls[0]["name"] or t.finish_reason for t in kept] == \
        ["bash", "write", "stop"]
    assert [t.index for t in kept] == [0, 1, 2]   # reindexed contiguously


def test_drop_keeps_final_turn_even_if_actionless():
    turns = [T.RecordedTurn(0, "done", None, [], "stop", {})]
    assert T.drop_nonfinal_actionless_turns(turns) == turns


def test_parse_pi_conversation_drops_midconversation_actionless_turn():
    # A truncated/empty assistant turn (no tool call) mid-conversation halts pi on
    # replay; the converter must elide it so the agent reaches the real fix.
    log = _pi_log(
        {"type": "message_start", "message": {"role": "user",
         "content": "Fix.\n# Task: x__x-1\n"}},
        {"type": "turn_end", "message": {"role": "assistant",       # truncated reasoning, no tool
         "content": [{"type": "thinking", "thinking": "long cut-off reasoning"}],
         "stopReason": "maxTokens", "model": "m",
         "usage": {"input": 1, "output": 8192, "totalTokens": 8193}}},
        {"type": "turn_end", "message": {"role": "assistant",
         "content": [{"type": "toolCall", "id": "c1", "name": "write",
                      "arguments": {"path": "f.py", "content": "fix"}}],
         "stopReason": "toolUse", "model": "m", "usage": {"input": 2, "output": 3, "totalTokens": 5}}},
        {"type": "turn_end", "message": {"role": "assistant",
         "content": [{"type": "text", "text": "done"}], "stopReason": "endTurn",
         "model": "m", "usage": {"input": 4, "output": 1, "totalTokens": 5}}},
    )
    traj = T.parse_pi_conversation(log)
    assert len(traj.turns) == 2                       # the action-less turn is gone
    assert traj.turns[0].tool_calls[0]["name"] == "write"
    assert traj.turns[1].finish_reason == "stop"


def test_load_store_drops_actionless_turns_in_existing_jsonl(tmp_path):
    # Pre-fix jsonl on disk still carries a mid-conversation action-less turn;
    # load_store must elide it (the original job dir is usually gone, so
    # re-conversion is not an option).
    out = tmp_path / "t.jsonl"
    traj = T.Trajectory(key="k", instance_id="k", model="m", turns=[
        T.RecordedTurn(0, "", None, [{"id": "c", "name": "bash", "arguments": {}}], "tool_calls", {}),
        T.RecordedTurn(1, "", None, [], "stop", {}),   # action-less mid turn
        T.RecordedTurn(2, "done", None, [], "stop", {}),
    ])
    out.write_text(json.dumps(traj.to_dict()) + "\n")
    loaded = T.load_store(out)["k"]
    assert len(loaded.turns) == 2
    assert loaded.turns[0].tool_calls[0]["name"] == "bash"


def test_parse_pi_conversation_empty_log_is_safe():
    traj = T.parse_pi_conversation("")
    assert traj.turns == [] and traj.key == _key_helper("")


def test_parse_pi_conversation_skips_malformed_turn_end():
    log = _pi_log(
        {"type": "message_start", "message": {"role": "user", "content": "# Task: x__x-1\n"}},
        {"type": "turn_end"},                       # no message key
        {"type": "turn_end", "message": None},      # null message
        {"type": "turn_end", "message": {"role": "assistant",
            "content": [{"type": "text", "text": "ok"}], "stopReason": "endTurn",
            "model": "m", "usage": {"input": 1, "output": 1, "totalTokens": 2}}},
    )
    traj = T.parse_pi_conversation(log)
    assert len(traj.turns) == 1 and traj.turns[0].content == "ok"


def _key_helper(text: str) -> str:
    return T._key_from_text(text)


def _write_job(tmp_path, trial, log_text):
    d = tmp_path / trial / "agent"
    d.mkdir(parents=True)
    (d / "pi-container.log").write_text(log_text)


def test_convert_job_and_load_store(tmp_path):
    log_a = _pi_log(
        {"type": "message_start", "message": {"role": "user",
         "content": "x\n# Task: aa__aa-1\n"}},
        {"type": "turn_end", "message": {"role": "assistant",
         "content": [{"type": "text", "text": "a0"}], "stopReason": "endTurn",
         "model": "m", "usage": {"input": 1, "output": 1, "totalTokens": 2}}},
    )
    log_b = _pi_log(
        {"type": "message_start", "message": {"role": "user",
         "content": "y\n# Task: bb__bb-2\n"}},
        {"type": "turn_end", "message": {"role": "assistant",
         "content": [{"type": "toolCall", "id": "c1", "name": "bash",
                      "arguments": {"command": "ls"}}], "stopReason": "toolUse",
         "model": "m", "usage": {"input": 3, "output": 4, "totalTokens": 7}}},
    )
    _write_job(tmp_path, "aa__aa-1__x", log_a)
    _write_job(tmp_path, "bb__bb-2__y", log_b)

    out = tmp_path / "replay-trajectories.jsonl"
    n = T.convert_job(tmp_path, out)
    assert n == 2
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2

    store = T.load_store(out)
    assert set(store) == {"aa__aa-1", "bb__bb-2"}
    assert store["aa__aa-1"].trial == "aa__aa-1__x"
    assert store["bb__bb-2"].turns[0].tool_calls[0]["name"] == "bash"


def test_load_store_last_wins_on_duplicate_key(tmp_path):
    out = tmp_path / "t.jsonl"
    a = T.Trajectory(key="k", instance_id="k", model="m",
                     turns=[T.RecordedTurn(0, "first", None, [], "stop", {})])
    b = T.Trajectory(key="k", instance_id="k", model="m",
                     turns=[T.RecordedTurn(0, "second", None, [], "stop", {})])
    out.write_text(json.dumps(a.to_dict()) + "\n" + json.dumps(b.to_dict()) + "\n")
    store = T.load_store(out)
    assert store["k"].turns[0].content == "second"


def test_convert_job_empty_dir_writes_no_file(tmp_path):
    out = tmp_path / "out.jsonl"
    n = T.convert_job(tmp_path, out)
    assert n == 0
    assert not out.exists()


def test_task_key_placeholder_id_falls_back_to_hash():
    # Real recorded runs emit "# Task: ?"; distinct prompts must NOT collide.
    a = [{"role": "user", "content": "Fix A.\n# Task: ?\nRepository: x"}]
    b = [{"role": "user", "content": "Fix B.\n# Task: ?\nRepository: y"}]
    ka, kb = T.task_key_from_messages(a), T.task_key_from_messages(b)
    assert ka.startswith("sha1:") and kb.startswith("sha1:")
    assert ka != kb


def test_convert_job_placeholder_ids_dont_collide(tmp_path):
    log_a = _pi_log(
        {"type": "message_start", "message": {"role": "user", "content": "Fix A.\n# Task: ?\n"}},
        {"type": "turn_end", "message": {"role": "assistant",
         "content": [{"type": "text", "text": "a"}], "stopReason": "endTurn",
         "model": "m", "usage": {"input": 1, "output": 1, "totalTokens": 2}}},
    )
    log_b = _pi_log(
        {"type": "message_start", "message": {"role": "user", "content": "Fix B.\n# Task: ?\n"}},
        {"type": "turn_end", "message": {"role": "assistant",
         "content": [{"type": "text", "text": "b"}], "stopReason": "endTurn",
         "model": "m", "usage": {"input": 1, "output": 1, "totalTokens": 2}}},
    )
    _write_job(tmp_path, "astropy__astropy-1__aaa", log_a)
    _write_job(tmp_path, "django__django-2__bbb", log_b)
    out = tmp_path / "t.jsonl"
    assert T.convert_job(tmp_path, out) == 2
    store = T.load_store(out)
    assert len(store) == 2  # distinct prompts -> distinct keys, no collision on "?"
    iids = sorted(t.instance_id for t in store.values())
    assert iids == ["astropy__astropy-1", "django__django-2"]


def test_sha1_key_agrees_between_list_and_string_content():
    # The recorded log stores the prompt as a list-of-blocks; pi's HTTP request
    # sends it as a plain string. Both must hash to the SAME replay key, or every
    # placeholder-id task becomes a permanent replay miss.
    text = "Fix A.\n# Task: ?\nRepository: x\n## Problem statement\nsome bug\n"
    list_msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    str_msgs = [{"role": "user", "content": text}]
    assert T.task_key_from_messages(list_msgs) == T.task_key_from_messages(str_msgs)


def test_convert_job_handles_pi_host_logs(tmp_path):
    log_text = _pi_log(
        {"type": "message_start", "message": {"role": "user", "content": "Fix.\n# Task: x__x-9\n"}},
        {"type": "turn_end", "message": {"role": "assistant",
         "content": [{"type": "text", "text": "ok"}], "stopReason": "endTurn",
         "model": "m", "usage": {"input": 1, "output": 1, "totalTokens": 2}}},
    )
    d = tmp_path / "x__x-9__zzz" / "agent"
    d.mkdir(parents=True)
    (d / "pi-host.log").write_text(log_text)
    out = tmp_path / "t.jsonl"
    assert T.convert_job(tmp_path, out) == 1
    assert set(T.load_store(out)) == {"x__x-9"}


from benchmaker.swebench.trajectory import derive_tool_status, RecordedTurn, parse_pi_conversation


def test_derive_tool_status_bash_returncode_prefix():
    assert derive_tool_status("bash", "returncode: 0\nhi\n") == 0
    assert derive_tool_status("bash", "returncode: 1\nTraceback...\n") == 1
    assert derive_tool_status("bash", "returncode: -1\n") == -1
    assert derive_tool_status("bash", "returncode: 0\nCommand exited with code 5") == 0


def test_derive_tool_status_bash_container_trailer_fallback():
    assert derive_tool_status("bash", "boom\nCommand exited with code 2") == 2


def test_derive_tool_status_bash_malformed_is_none():
    assert derive_tool_status("bash", "no status line here") is None


def test_derive_tool_status_file_tools():
    assert derive_tool_status("read", "file contents\n") == 0
    assert derive_tool_status("read", "read: cannot read /x\nexit 1") == 1
    assert derive_tool_status("write", "Wrote /x") == 0
    assert derive_tool_status("write", "write: failed for /x\nexit 1") == 1
    assert derive_tool_status("edit", "Applied 2 edit(s) to /x") == 0
    assert derive_tool_status("edit", "edit 0: oldText not found in /x") == 1
    assert derive_tool_status("edit", "edit: no edits provided") == 1
    assert derive_tool_status("edit", "edit: cannot read /x") == 1


def test_derive_tool_status_unknown_tool_is_none():
    assert derive_tool_status("mystery", "whatever") is None


def test_recorded_turn_tool_results_roundtrip():
    t = RecordedTurn(0, "c", None, [{"id": "x", "name": "bash", "arguments": {}}],
                     "tool_calls", {}, tool_results=[{"name": "bash", "status": 0}])
    assert t.to_dict()["tool_results"] == [{"name": "bash", "status": 0}]
    back = RecordedTurn.from_dict(t.to_dict())
    assert back.tool_results == [{"name": "bash", "status": 0}]


def test_recorded_turn_tool_results_defaults_empty_for_old_dict():
    back = RecordedTurn.from_dict({"index": 0, "content": "c", "tool_calls": []})
    assert back.tool_results == []
    assert RecordedTurn(0, "c", None, [], "stop", {}).tool_results == []


def test_parse_captures_tool_results_aligned_by_id():
    log = "\n".join([
        json.dumps({"type": "message_start", "message": {"role": "user",
               "content": [{"type": "text", "text": "# Task: aa__aa-1\n"}]}}),
        json.dumps({"type": "turn_end", "message": {"role": "assistant", "model": "m",
               "content": [{"type": "toolCall", "id": "c0", "name": "bash",
                            "arguments": {"command": "pytest"}}]}}),
        json.dumps({"type": "tool_execution_end", "toolCallId": "c0", "toolName": "bash",
               "isError": False, "result": {"content": [{"type": "text",
                            "text": "returncode: 1\nFAILED\n"}]}}),
        json.dumps({"type": "turn_end", "message": {"role": "assistant",
               "content": [{"type": "text", "text": "done"}]}}),
    ])
    traj = parse_pi_conversation(log)
    assert traj.turns[0].tool_results == [{"name": "bash", "status": 1}]
    assert traj.turns[-1].tool_results == []
