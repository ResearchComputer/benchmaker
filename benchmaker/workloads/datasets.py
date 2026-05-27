"""Workloads = datasets / input sources.

A `Workload` is the source of per-request data. It yields opaque items that a
`WorkloadType` knows how to interpret. Built-ins cover the common shapes;
custom workloads are a one-method subclass.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Optional, Union


class Workload(ABC):
    """A source of per-request items (a dataset, replay log, prompt list, …)."""

    name: str = "workload"

    @abstractmethod
    async def next_item(self) -> Any:
        """Return the next item. Raise `StopAsyncIteration` to halt the bench."""

    async def aclose(self) -> None:
        return None


class StaticWorkload(Workload):
    """Cycle through a fixed list of items forever (or `max_items` times).

    Use this for trivial benchmarks ("send the same JSON repeatedly") or when
    your dataset fits in memory.
    """

    def __init__(self, items: Iterable[Any] = (None,), name: str = "static",
                 shuffle: bool = False, seed: Optional[int] = None,
                 max_items: Optional[int] = None):
        self.name = name
        items = list(items)
        if not items:
            items = [None]
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(items)
        self._items = items
        self._cycle = itertools.cycle(items)
        self._max = max_items
        self._n = 0

    async def next_item(self) -> Any:
        if self._max is not None and self._n >= self._max:
            raise StopAsyncIteration
        self._n += 1
        return next(self._cycle)


class JsonlWorkload(Workload):
    """Stream items from a JSONL file.

    Args:
        path: path to .jsonl file.
        field: if set, yield only this top-level field of each line (e.g. "prompt").
        loop: if True (default), restart the file when exhausted.
        max_items: cap on total items yielded across loops.
    """

    def __init__(self, path: str, field: Optional[str] = None, loop: bool = True,
                 max_items: Optional[int] = None, name: Optional[str] = None):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.name = name or f"jsonl:{os.path.basename(path)}"
        self._path = path
        self._field = field
        self._loop = loop
        self._max = max_items
        self._n = 0
        self._fh = open(path, "r")
        self._lock = asyncio.Lock()

    async def next_item(self) -> Any:
        if self._max is not None and self._n >= self._max:
            raise StopAsyncIteration
        async with self._lock:
            line = self._fh.readline()
            if not line:
                if not self._loop:
                    raise StopAsyncIteration
                self._fh.seek(0)
                line = self._fh.readline()
                if not line:
                    raise StopAsyncIteration  # empty file
            obj = json.loads(line)
            if self._field is not None:
                obj = obj[self._field]
            self._n += 1
            return obj

    async def aclose(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class CallableWorkload(Workload):
    """Wrap a callable (sync or async) that returns the next item on each call.

    The callable can be a closure that walks a generator, queries an external
    service, etc. Raise `StopAsyncIteration` from the callable to halt.
    """

    def __init__(self, fn: Callable[[], Union[Any, Awaitable[Any]]],
                 name: str = "callable"):
        self.name = name
        self._fn = fn

    async def next_item(self) -> Any:
        result = self._fn()
        if hasattr(result, "__await__"):
            return await result  # type: ignore[return-value]
        return result


class IterableWorkload(Workload):
    """Wrap any (async) iterable as a workload. Halts when iterable is exhausted."""

    def __init__(self, iterable: Union[Iterable[Any], AsyncIterator[Any]],
                 name: str = "iterable", loop: bool = False):
        self.name = name
        self._loop = loop
        if hasattr(iterable, "__aiter__"):
            self._aiter = iterable.__aiter__()  # type: ignore[union-attr]
            self._items_cache: Optional[list] = None
            self._is_async = True
        else:
            self._items_cache = list(iterable)  # type: ignore[arg-type]
            self._iter = iter(self._items_cache)
            self._is_async = False

    async def next_item(self) -> Any:
        if self._is_async:
            try:
                return await self._aiter.__anext__()
            except StopAsyncIteration:
                raise
        try:
            return next(self._iter)
        except StopIteration:
            if self._loop and self._items_cache:
                self._iter = iter(self._items_cache)
                return next(self._iter)
            raise StopAsyncIteration
