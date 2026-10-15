# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Voice-clone reference encoding in the API server, shared by TTS models.

A voice-cloning request needs artifacts computed from its reference clip, such
as codec codes and a speaker embedding. Computed inside the engine, they stall
the AR step of every running request; computed per request, a clip that many
requests share is encoded again each time. ``ReferenceEncodingService`` runs a
model-provided batch encoder in the API server before the request is submitted:

* content-addressed: callers key entries by a digest of the decoded clip
  (``content_key``), so one clip sent through different locators shares one entry;
* cached: least recently used entries beyond a byte budget are evicted;
* single-flight: concurrent requests for one uncached clip await one encode;
* micro-batched: cold clips arriving within a short window share one encoder
  call, run in a worker thread off the event loop.

The model's encoder owns the computation, including what it batches, so an
artifact equals what that encoder returns for the clip on its own.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Iterator
from typing import Any, Generic, TypeVar

import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_BATCH_WINDOW_MS = 10.0
DEFAULT_MAX_BATCH = 32
DEFAULT_CACHE_BYTES = 256 * 1024 * 1024


class MicroBatcher(Generic[T, R]):
    """Coalesce items submitted within a window into batched calls of a sync function.

    ``encode_batch_fn(items)`` runs in a worker thread and returns one result
    (or exception) per item, in order: a list releases the whole batch at once,
    a generator releases each item as it is yielded.
    """

    def __init__(
        self,
        encode_batch_fn: Callable[[list[T]], list[R | BaseException] | Iterator[R | BaseException]],
        *,
        window_ms: float = DEFAULT_BATCH_WINDOW_MS,
        max_batch: int = DEFAULT_MAX_BATCH,
        max_inflight: int = 1,
    ) -> None:
        self._encode_batch_fn = encode_batch_fn
        self._window_s = max(0.0, float(window_ms) / 1000.0)
        self._max_batch = max(1, int(max_batch))
        # Batches encoded at once. Items that arrive while every slot is busy
        # are queued and form the next batch.
        self._max_inflight = max(1, int(max_inflight))
        self._queue: asyncio.Queue | None = None
        self._slots: asyncio.Semaphore | None = None
        self._inflight: set[asyncio.Task] = set()
        self._drainer: asyncio.Task | None = None

    def _ensure_started(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._slots is None:
            self._slots = asyncio.Semaphore(self._max_inflight)
        if self._drainer is None or self._drainer.done():
            self._drainer = asyncio.create_task(self._drain_loop())

    async def submit(self, item: T) -> R:
        self._ensure_started()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((item, fut))  # type: ignore[union-attr]
        return await fut

    async def _drain_loop(self) -> None:
        assert self._queue is not None and self._slots is not None
        while True:
            await self._slots.acquire()
            try:
                jobs = await self._coalesce(await self._queue.get())
            except BaseException:
                self._slots.release()
                raise
            task = asyncio.create_task(self._run_and_release(jobs))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)
            # Do not retain the batch here while waiting for new work.
            del jobs

    async def _run_and_release(self, jobs: list[tuple[T, asyncio.Future]]) -> None:
        try:
            await self._run_batch(jobs)
        finally:
            del jobs
            assert self._slots is not None
            self._slots.release()

    async def _coalesce(self, first: tuple[T, asyncio.Future]) -> list[tuple[T, asyncio.Future]]:
        """Group ``first`` with jobs arriving within the batch window."""
        assert self._queue is not None
        jobs = [first]
        if self._window_s > 0:
            deadline = asyncio.get_running_loop().time() + self._window_s
            while len(jobs) < self._max_batch:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    jobs.append(await asyncio.wait_for(self._queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
        else:
            # window=0: coalesce only what is already queued, never wait.
            while len(jobs) < self._max_batch:
                try:
                    jobs.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
        return jobs

    async def _run_batch(self, jobs: list[tuple[T, asyncio.Future]]) -> None:
        items = [item for item, _ in jobs]
        futs = [fut for _, fut in jobs]
        loop = asyncio.get_running_loop()

        def settle(fut: asyncio.Future, res: Any) -> None:
            if fut.done():
                return
            if isinstance(res, BaseException):
                fut.set_exception(res)
            else:
                fut.set_result(res)

        def run() -> list | None:
            results = self._encode_batch_fn(items)
            if isinstance(results, list):
                return results
            # A generator hands each item over as soon as it is ready instead
            # of holding the whole batch back for its slowest item.
            for fut, res in zip(futs, results):
                loop.call_soon_threadsafe(settle, fut, res)
            return None

        try:
            results = await asyncio.to_thread(run)
        except Exception as exc:  # noqa: BLE001 - every waiter still pending gets the failure
            results = [exc] * len(futs)
        if results is None:
            # Let the handovers queued by the worker run first.
            await asyncio.sleep(0)
            results = [RuntimeError("reference encoder returned no result for this item")] * len(futs)
        for fut, res in zip(futs, results):
            settle(fut, res)

    async def aclose(self) -> None:
        if self._drainer is not None:
            self._drainer.cancel()
            try:
                await self._drainer
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._drainer = None
        for task in list(self._inflight):
            task.cancel()


def content_key(namespace: str, waveform: np.ndarray, sample_rate: int, *extra: Hashable) -> str:
    """Digest of a decoded clip, its sample rate and the encoder's ``namespace``."""
    digest = hashlib.sha256()
    digest.update(namespace.encode())
    digest.update(str(int(sample_rate)).encode())
    for item in extra:
        digest.update(repr(item).encode())
    wav = np.ascontiguousarray(waveform, dtype=np.float32)
    digest.update(str(wav.shape).encode())
    digest.update(wav.tobytes())
    return digest.hexdigest()


class ReferenceEncodingService(Generic[T, R]):
    """Content-addressed, byte-bounded, single-flight, micro-batched reference artifacts."""

    def __init__(
        self,
        encode_batch: Callable[[list[T]], list[R | BaseException] | Iterator[R | BaseException]],
        *,
        nbytes: Callable[[R], int],
        max_bytes: int = DEFAULT_CACHE_BYTES,
        window_ms: float = DEFAULT_BATCH_WINDOW_MS,
        max_batch: int = DEFAULT_MAX_BATCH,
        max_inflight: int = 1,
        name: str = "reference",
    ) -> None:
        self._batcher: MicroBatcher[T, R] = MicroBatcher(
            encode_batch, window_ms=window_ms, max_batch=max_batch, max_inflight=max_inflight
        )
        self._nbytes = nbytes
        self._max_bytes = max(0, int(max_bytes))
        self._cache: OrderedDict[str, tuple[R, int]] = OrderedDict()
        self._cached_bytes = 0
        self._inflight: dict[str, asyncio.Task] = {}
        self._name = name
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def cached_bytes(self) -> int:
        return self._cached_bytes

    def lookup(self, key: str) -> R | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        self._cache.move_to_end(key)
        return entry[0]

    def _store(self, key: str, artifact: R) -> None:
        size = int(self._nbytes(artifact))
        if size > self._max_bytes:
            return
        old = self._cache.pop(key, None)
        if old is not None:
            self._cached_bytes -= old[1]
        self._cache[key] = (artifact, size)
        self._cached_bytes += size
        while self._cached_bytes > self._max_bytes:
            _, (_, evicted) = self._cache.popitem(last=False)
            self._cached_bytes -= evicted

    async def get(self, key: str, item: T) -> R:
        """The artifact for ``item`` keyed ``key``: cached, joined in flight, or encoded."""
        cached = self.lookup(key)
        if cached is not None:
            self.hits += 1
            return cached
        task = self._inflight.get(key)
        if task is None:
            self.misses += 1
            task = asyncio.ensure_future(self._encode(key, item))
            self._inflight[key] = task
            task.add_done_callback(self._retire(key))
        # A cancelled caller must not cancel the encode other callers share.
        return await asyncio.shield(task)

    def _retire(self, key: str) -> Callable[[asyncio.Task], None]:
        def retire(_task: asyncio.Task) -> None:
            self._inflight.pop(key, None)

        return retire

    async def _encode(self, key: str, item: T) -> R:
        artifact = await self._batcher.submit(item)
        self._store(key, artifact)
        return artifact

    def forget(self, keys: Iterable[str]) -> None:
        for key in keys:
            old = self._cache.pop(key, None)
            if old is not None:
                self._cached_bytes -= old[1]

    async def aclose(self) -> None:
        await self._batcher.aclose()

    def stats(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "entries": len(self._cache),
            "bytes": self._cached_bytes,
            "max_bytes": self._max_bytes,
            "hits": self.hits,
            "misses": self.misses,
        }
