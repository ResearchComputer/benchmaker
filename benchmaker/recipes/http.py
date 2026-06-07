"""``http`` recipe — one-off benchmark of a single HTTP endpoint."""

from __future__ import annotations

import json
from typing import Any

import click

from benchmaker.recipes import register
from benchmaker.recipes._cli_shared import parse_headers
from benchmaker.recipes.base import BuildResult, Recipe, SharedOpts
from benchmaker.workloads.datasets import StaticWorkload
from benchmaker.workloads.http import HttpWorkloadType


class HttpRecipe(Recipe):
    name = "http"
    help = "One-off benchmark of a single HTTP endpoint (no config file)."

    def options(self) -> list:
        return [
            click.option("--url", required=True, help="Target URL."),
            click.option("--method", default="GET", help="HTTP method."),
            click.option("--header", "-H", "header", multiple=True,
                         help="Header 'Name: value'. Repeatable."),
            click.option("--json-body", "json_body", default=None,
                         help="JSON body string (sent as one static item)."),
            click.option("--data", default=None,
                         help="Raw body string (sent as one static item)."),
        ]

    def build(self, shared: SharedOpts, *, url: str, method: str,
              header: tuple[str, ...], json_body: str | None,
              data: str | None) -> BuildResult:
        headers = parse_headers(header)
        wt = HttpWorkloadType(
            url=url,
            method=method,
            headers=headers,
            timeout_s=shared.timeout_s,
        )

        workload = StaticWorkload()
        workload_spec: Any = None
        if json_body is not None:
            items = [json.loads(json_body)]
            workload = StaticWorkload(items=items)
            workload_spec = {"type": "static", "items": items}
        elif data is not None:
            workload = StaticWorkload(items=[data.encode("utf-8")])
            workload_spec = {"type": "static", "items": [data]}

        source_config: dict = {
            "workload_type": {
                "type": "http",
                "url": url,
                "method": method,
                "headers": headers,
                "timeout_s": shared.timeout_s,
            },
        }
        if workload_spec is not None:
            source_config["workload"] = workload_spec

        return BuildResult(
            workload_type=wt,
            workload=workload,
            source_config=source_config,
        )


register(HttpRecipe())
