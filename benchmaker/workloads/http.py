"""Generic HTTP workload-type.

Interprets workload items flexibly:

  * `None`            → fire the base request unchanged.
  * `bytes`           → use as raw body.
  * `str`             → use as raw body (utf-8 encoded).
  * `dict`            → use as JSON body, UNLESS keys overlap with Request fields
                        (`body`/`json`/`params`/`headers`/`url`/`method`/`meta`),
                        in which case those fields are applied directly. This
                        lets a dataset fully customize the request when needed.
"""

from __future__ import annotations

from typing import Any, Optional

from benchmaker.types import Request
from benchmaker.workloads.base import WorkloadType


_REQUEST_KEYS = {"body", "json", "params", "headers", "url", "method", "meta", "timeout_s"}


class HttpWorkloadType(WorkloadType):
    name = "http"

    def __init__(
        self,
        url: str = "",
        method: str = "GET",
        headers: Optional[dict[str, str]] = None,
        params: Optional[dict[str, Any]] = None,
        timeout_s: Optional[float] = None,
        name: str = "http",
    ):
        self.name = name
        self._url = url
        self._method = method
        self._headers = headers or {}
        self._params = params or {}
        self._timeout_s = timeout_s

    async def make_request(self, item: Any) -> Request:
        req = Request(
            method=self._method,
            url=self._url,
            headers=dict(self._headers),
            params=dict(self._params),
            timeout_s=self._timeout_s,
        )
        if item is None:
            return req
        if isinstance(item, (bytes, bytearray)):
            req.body = bytes(item)
            return req
        if isinstance(item, str):
            req.body = item.encode("utf-8")
            return req
        if isinstance(item, dict):
            if _REQUEST_KEYS & item.keys():
                # Treat as a Request override.
                for k, v in item.items():
                    if k == "headers":
                        req.headers.update(v)
                    elif k == "params":
                        req.params.update(v)
                    elif k == "meta":
                        req.meta.update(v)
                    elif k in _REQUEST_KEYS:
                        setattr(req, k, v)
                    # ignore unknown keys silently
                return req
            # Otherwise treat the whole dict as the JSON body.
            req.json = item
            return req
        raise TypeError(f"HttpWorkloadType cannot interpret item of type {type(item).__name__}")
