"""``sandbox`` recipe — benchmark a Flash Sandbox endpoint (exec/create/lifecycle)."""

from __future__ import annotations

import json
from typing import Any, Optional

import click

from benchmaker.recipes import register
from benchmaker.recipes._cli_shared import parse_headers
from benchmaker.recipes.base import BuildResult, Recipe, SharedOpts
from benchmaker.workloads.datasets import StaticWorkload
from benchmaker.workloads.sandbox import PREFIX_CLUSTER, SandboxWorkloadType


class SandboxRecipe(Recipe):
    name = "sandbox"
    help = ("Benchmark a Flash Sandbox endpoint. operation=exec reuses one "
            "sandbox per run; create makes a fresh pod per request; lifecycle "
            "times the full create -> exec -> delete sequence per request; "
            "file does PUT/GET /files write/read-back verification.")

    def options(self) -> list:
        return [
            click.option("--base-url", "base_url", required=True,
                         help="Flash Sandbox base URL (e.g. http://localhost:8080)."),
            click.option("--operation", type=click.Choice(["exec", "create", "lifecycle", "file"]),
                         default="exec", show_default=True),
            click.option("--command", "-c", "command", multiple=True,
                         help="Command to exec (repeatable -> one item each). "
                              "A bare string runs via 'sh -c'. exec/lifecycle only."),
            click.option("--image", default=None,
                         help="Sandbox image for the create spec (e.g. alpine:3.20)."),
            click.option("--spec-json", "spec_json", default=None,
                         help="JSON object merged into the sandbox create spec "
                              "(overrides --image)."),
            click.option("--endpoint-prefix", "endpoint_prefix", default=PREFIX_CLUSTER,
                         show_default=True,
                         help="API prefix ('/sandboxes' cluster, '/native/sandboxes' node)."),
            click.option("--ttl-seconds", "ttl_seconds", type=int, default=None,
                         help="Server-side reap TTL for created sandboxes."),
            click.option("--persistent/--no-persistent", default=False,
                         help="Use /pshell so cd/export persist across exec calls."),
            click.option("--sandbox-id", "sandbox_id", default=None,
                         help="Target an existing sandbox instead of creating one "
                              "(exec only)."),
            click.option("--header", "-H", "header", multiple=True,
                         help="Extra header 'Name: value'. Repeatable."),
        ]

    def build(self, shared: SharedOpts, *, base_url, operation, command, image,
              spec_json, endpoint_prefix, ttl_seconds, persistent, sandbox_id,
              header) -> BuildResult:
        spec: dict[str, Any] = {}
        if image:
            spec["image"] = image
        if spec_json:
            extra = json.loads(spec_json)
            if not isinstance(extra, dict):
                raise click.BadParameter("--spec-json must be a JSON object")
            spec.update(extra)

        headers = parse_headers(header)
        wt_kwargs: dict[str, Any] = dict(
            base_url=base_url,
            operation=operation,
            spec=(spec or None),
            endpoint_prefix=endpoint_prefix,
            persistent=persistent,
            timeout_s=shared.timeout_s,
            ttl_seconds=ttl_seconds,
            headers=(headers or None),
        )
        if operation == "exec" and sandbox_id:
            wt_kwargs["sandbox_id"] = sandbox_id

        # In exec mode with no explicit command, fall back to a default so a
        # None workload item is runnable.
        default_command: Optional[Any] = None
        if operation in ("exec", "lifecycle") and not command:
            default_command = "true"
            wt_kwargs["default_command"] = default_command

        wt = SandboxWorkloadType(**wt_kwargs)

        if command:
            items = list(command)
            workload = StaticWorkload(items=items)
            workload_spec: Any = {"type": "static", "items": items}
        else:
            workload = StaticWorkload()
            workload_spec = None

        source_config: dict = {
            "workload_type": {
                "type": "sandbox",
                "base_url": base_url,
                "operation": operation,
                "spec": spec or None,
                "endpoint_prefix": endpoint_prefix,
                "persistent": persistent,
                "ttl_seconds": ttl_seconds,
                "timeout_s": shared.timeout_s,
            },
        }
        if sandbox_id:
            source_config["workload_type"]["sandbox_id"] = sandbox_id
        if default_command is not None:
            source_config["workload_type"]["default_command"] = default_command
        if workload_spec is not None:
            source_config["workload"] = workload_spec

        return BuildResult(
            workload_type=wt,
            workload=workload,
            source_config=source_config,
        )


register(SandboxRecipe())
