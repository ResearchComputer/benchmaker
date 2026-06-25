import json, os

def _w(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(obj if isinstance(obj, str) else json.dumps(obj))

def _jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def _result(task="django__django-11999", trial="django__django-11999__K75PXvM",
            reward=1.0, mode="host", exit_status="ok"):
    return {
        "task_name": task, "trial_name": trial, "source": "swebench-verified",
        "config": {"task": {"path": f"datasets/swebench-verified/{task}"},
                   "agent": {"model_name": "zai-org/GLM-4.7-Flash-rocm",
                             "kwargs": {"exec_timeout_s": 20.0}},
                   "timeout_multiplier": 4.0,
                   "environment": {"kwargs": {"backend_type": "docker"}}},
        "agent_info": {"model_info": {"name": "GLM-4.7-Flash-rocm"}},
        "agent_result": {"metadata": {"mode": mode, "exit_status": exit_status,
                                      "exec_count": 81}},
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": None,
        "environment_setup": {"started_at": "2026-06-19T18:01:29.0Z",
                              "finished_at": "2026-06-19T18:01:29.5Z"},
        "agent_setup": {"started_at": "2026-06-19T18:01:29.5Z",
                        "finished_at": "2026-06-19T18:01:29.7Z"},
        "agent_execution": {"started_at": "2026-06-19T18:01:29.7Z",
                            "finished_at": "2026-06-19T18:01:55Z"},
        "verifier": {"started_at": "2026-06-19T18:01:55Z",
                     "finished_at": "2026-06-19T18:02:23Z"},
        "started_at": "2026-06-19T18:01:29.0Z",
        "finished_at": "2026-06-19T18:02:24.0Z",
    }

def _report(task="django__django-11999", resolved=True):
    return {task: {"resolved": resolved,
                   "tests_status": {"FAIL_TO_PASS": {"success": ["t_a"], "failure": []},
                                    "PASS_TO_PASS": {"success": ["t_b"], "failure": []}}}}

def make_pi_host_trial(root, task="django__django-11999",
                       trial="django__django-11999__K75PXvM"):
    td = os.path.join(root, trial)
    _w(os.path.join(td, "result.json"), _result(task, trial, mode="host"))
    _w(os.path.join(td, "config.json"), _result(task, trial)["config"])
    _w(os.path.join(td, "verifier", "reward.txt"), "1\n")
    _w(os.path.join(td, "verifier", "report.json"), _report(task))
    sess = os.path.join(td, "agent", "pi-home", ".pi", "agent", "sessions",
                        "--enc--", "2026-06-19T18-01-30_uuid.jsonl")
    _jsonl(sess, [
        {"type": "session", "id": "uuid", "version": 3},
        {"type": "message", "message": {"role": "user", "timestamp": 1781782380000,
                                         "content": [{"type": "text", "text": "fix"}]}},
        {"type": "message", "message": {"role": "assistant", "timestamp": 1781782381000,
                                         "content": [{"type": "toolCall", "name": "bash"}]}},
        {"type": "message", "message": {"role": "toolResult", "timestamp": 1781782382500,
                                        "content": [{"type": "text", "text": "ok"}]}},
    ])
    _jsonl(os.path.join(td, "agent", "timeline-spans.jsonl"),
           [{"name": "sandbox_exec", "duration_s": 1.5, "rc": 0}])
    # A real pi-host.log (one tool-call turn, matching the session jsonl above).
    # clean prefers the session jsonl and discards this log; the legacy
    # trajectory collector parses it directly.
    a = {"role": "assistant", "timestamp": 1781782381000,
         "content": [{"type": "toolCall", "name": "bash"}]}
    _jsonl(os.path.join(td, "agent", "pi-host.log"), [
        {"type": "turn_end", "message": a},
    ])
    return td

def make_pi_container_trial(root, task="astropy__astropy-7606",
                            trial="astropy__astropy-7606__B8sCm3m"):
    td = os.path.join(root, trial)
    _w(os.path.join(td, "result.json"), _result(task, trial, mode="container"))
    _w(os.path.join(td, "config.json"), _result(task, trial, mode="container")["config"])
    _w(os.path.join(td, "verifier", "reward.txt"), "1\n")
    _w(os.path.join(td, "verifier", "report.json"), _report(task))
    a = {"role": "assistant", "timestamp": 1781782381000,
         "content": [{"type": "toolCall", "name": "bash"}]}
    tr = {"role": "toolResult", "timestamp": 1781782382500,
          "content": [{"type": "text", "text": "ok"}]}
    # streaming deltas (the bulk that folds away); intermediate partials never
    # become the stream's final partial, so they don't enter the safety inventory.
    deltas = [{"type": "message_update",
               "message": {"role": "assistant",
                           "content": [{"type": "text", "text": "x" * 80}]}}
              for _ in range(50)]
    _jsonl(os.path.join(td, "agent", "pi-container.log"), [
        {"type": "message_start"},
        *deltas,
        {"type": "message_update", "message": a},   # final partial == message_end content
        {"type": "message_end", "message": a},
        {"type": "turn_end", "message": a},
        {"type": "tool_execution_end", "toolCallId": "t1", "toolName": "bash",
         "result": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "message_end", "message": tr},
        {"type": "turn_end", "message": tr},
    ])
    return td

def make_pi_container_killed_trial(root, task="astropy__astropy-7606",
                                   trial="astropy__astropy-7606__KiLL3d0"):
    """A pi-container.log whose stream is hard-killed: the final streaming
    partial (message_update) carries UNIQUE content and never closes with a
    matching message_end. Folding away message_update would lose that content,
    so the safety fallback must keep the raw log verbatim."""
    td = os.path.join(root, trial)
    _w(os.path.join(td, "result.json"), _result(task, trial, mode="container"))
    _w(os.path.join(td, "verifier", "report.json"), _report(task))
    partial = {"role": "assistant", "timestamp": 1781782381000,
               "content": [{"type": "text", "text": "partial only, never closed"}]}
    _jsonl(os.path.join(td, "agent", "pi-container.log"), [
        {"type": "message_start"},
        {"type": "message_update", "message": partial},   # killed before message_end
    ])
    return td

def make_pi_host_log_only_trial(root, task="matplotlib__matplotlib-25122",
                                trial="matplotlib__matplotlib-25122__QrBK5jY"):
    td = os.path.join(root, trial)
    _w(os.path.join(td, "result.json"), _result(task, trial, mode="host"))
    _w(os.path.join(td, "verifier", "report.json"), _report(task))
    _w(os.path.join(td, "verifier", "reward.txt"), "1\n")
    a = {"role": "assistant", "timestamp": 1781782381000,
         "content": [{"type": "toolCall", "name": "bash"}]}
    tr = {"role": "toolResult", "timestamp": 1781782382500,
          "content": [{"type": "text", "text": "ok"}]}
    _jsonl(os.path.join(td, "agent", "pi-host.log"), [
        {"type": "message_start"},
        {"type": "message_update", "message": a},
        {"type": "message_end", "message": a},
        {"type": "turn_end", "message": a},
        {"type": "tool_execution_end", "toolCallId": "t1", "toolName": "bash",
         "result": {"content": [{"type": "text", "text": "ok"}]}},
        {"type": "message_end", "message": tr},
        {"type": "turn_end", "message": tr},
    ])
    return td

def make_failure_trial(root, task="django__django-16100",
                       trial="django__django-16100__PpdSN7s"):
    td = os.path.join(root, trial)
    r = _result(task, trial, reward=None, exit_status="error")
    r["exception_info"] = {"exception_type": "RuntimeError"}
    r["verifier_result"] = {"rewards": {}}
    _w(os.path.join(td, "result.json"), r)
    _w(os.path.join(td, "exception.txt"), "Traceback ... RuntimeError\n")
    os.makedirs(os.path.join(td, "agent"), exist_ok=True)  # empty agent dir
    return td
