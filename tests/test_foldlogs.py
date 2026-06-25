import json
import os
import tempfile
from benchmaker.swebench.foldlogs import fold_lines, fold_file, REDUNDANT

def _line(obj):
    return (json.dumps(obj) + "\n").encode()

def _log():
    # message_start, two message_update deltas, a message_end, a byte-identical
    # turn_end, and a tool_execution_end.
    msg = {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
    return [
        _line({"type": "message_start"}),
        _line({"type": "message_update", "message": {"role": "assistant", "content": []}}),
        _line({"type": "message_update", "message": msg}),
        _line({"type": "message_end", "message": msg}),
        _line({"type": "turn_end", "message": msg}),
        _line({"type": "tool_execution_end", "toolCallId": "t1"}),
    ]

def test_default_drop_removes_all_redundant():
    kept, missing = fold_lines(_log())
    assert missing == set()
    types = [json.loads(b)["type"] for b in kept]
    # message_update, turn_end (byte-identical to message_end), agent_end dropped
    assert types == ["message_start", "message_end", "tool_execution_end"]

def test_drop_set_can_retain_turn_end():
    kept, missing = fold_lines(_log(), drop={"message_update", "agent_end"})
    assert missing == set()
    types = [json.loads(b)["type"] for b in kept]
    assert "turn_end" in types
    assert "message_update" not in types
    assert types == ["message_start", "message_end", "turn_end", "tool_execution_end"]

def test_retained_turn_end_with_unique_content_is_not_missing():
    """When turn_end is excluded from `drop` (retained) and carries content not
    present in any message_end, folding must NOT report it as missing — the line
    is kept, so its content survives. Guards the keep-loop fast-path inventory
    symmetry (regression: false SKIPPED_UNSAFE)."""
    end_msg = {"role": "assistant", "content": [{"type": "text", "text": "ended"}]}
    uniq_msg = {"role": "assistant", "content": [{"type": "text", "text": "UNIQUE turn content"}]}
    log = [
        _line({"type": "message_start"}),
        _line({"type": "message_update", "message": end_msg}),
        _line({"type": "message_end", "message": end_msg}),
        _line({"type": "turn_end", "message": uniq_msg}),
        _line({"type": "tool_execution_end", "toolCallId": "t1"}),
    ]
    kept, missing = fold_lines(log, drop={"message_update", "agent_end"})
    assert missing == set()
    types = [json.loads(b)["type"] for b in kept]
    assert "turn_end" in types
    # the retained turn_end carries the unique content
    turn_msgs = [json.loads(b)["message"] for b in kept if json.loads(b)["type"] == "turn_end"]
    assert uniq_msg in turn_msgs

def test_fold_lines_reports_missing_for_hardkilled_stream():
    """A stream where the final streamed partial carries unique content but the
    closing message_end differs from it: dropping message_update loses that unique
    partial, so `missing` must be non-empty (unsafe).

    A stream that closes with a message_end is treated as "ended", so its last
    message_update is NOT kept (keep_mu_idx only retains finals of streams with no
    message_end). If that partial's content differs from the message_end's, it is
    lost -> the fold is unsafe.
    """
    diff_end = {"role": "assistant", "content": [{"type": "text", "text": "the ended message"}]}
    uniq_partial = {"role": "assistant", "content": [{"type": "text", "text": "STREAMED unique mid"}]}
    log = [
        _line({"type": "message_start"}),
        _line({"type": "message_update", "message": uniq_partial}),
        _line({"type": "message_end", "message": diff_end}),
    ]
    kept, missing = fold_lines(log)
    assert missing  # uniq_partial content is lost when message_update dropped

def test_fold_file_skipped_unsafe_leaves_file_unchanged():
    """fold_file returns SKIPPED_UNSAFE and does not rewrite when folding would
    lose unique content."""
    diff_end = {"role": "assistant", "content": [{"type": "text", "text": "the ended message"}]}
    uniq_partial = {"role": "assistant", "content": [{"type": "text", "text": "STREAMED unique mid"}]}
    log = [
        _line({"type": "message_start"}),
        _line({"type": "message_update", "message": uniq_partial}),
        _line({"type": "message_end", "message": diff_end}),
    ]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as fh:
        for line in log:
            fh.write(line)
        path = fh.name
    try:
        before = open(path, "rb").read()
        status, orig_bytes, new_bytes, n = fold_file(path)
        assert status == "SKIPPED_UNSAFE"
        assert new_bytes == orig_bytes  # reported unchanged
        assert n >= 1
        assert open(path, "rb").read() == before  # file untouched
    finally:
        if os.path.exists(path):
            os.unlink(path)

def test_fold_file_returns_folded_and_shrinks():
    """fold_file writes a smaller file and returns FOLDED status."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as fh:
        for line in _log():
            fh.write(line)
        path = fh.name
    try:
        orig_size = os.path.getsize(path)
        status, orig_bytes, new_bytes, n = fold_file(path)
        assert status == "FOLDED"
        assert new_bytes < orig_bytes
        assert orig_bytes == orig_size
        assert n == 0
        # file on disk is now smaller
        assert os.path.getsize(path) == new_bytes
        # folded file only has the kept types
        with open(path, "rb") as fh:
            types = [json.loads(b)["type"] for b in fh.readlines()]
        assert types == ["message_start", "message_end", "tool_execution_end"]
    finally:
        if os.path.exists(path):
            os.unlink(path)
