# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference-audio code encoding for the MOSS-TTS family.

The serving layer owns one :class:`MossTTSReferenceEncoder` per
``(processor, variant, n_vq, sr_target)`` tuple. Each instance layers three
small collaborators:

* :class:`_PooledReferenceEncoder` — a small pool of daemon workers that run
  the codec encoder. Concurrent cold encodes run in parallel (the MOSS audio
  tokenizer sits on CPU and releases the GIL during its forward), which is
  what the ``asyncio.to_thread`` path did before this module existed. When a
  wait window is configured the workers additionally coalesce queued jobs into
  a single batched forward, isolating per-item failures with a per-item retry.
* :class:`_CachedReferenceEncoder` — a content-addressed CPU LRU that stores
  compact ``int32`` code tensors and does single-flight de-duplication so
  concurrent misses for the same audio share one real codec encode.
* :class:`MossTTSReferenceEncoder` — the async facade used by the OpenAI
  speech serving layer.

Caching is keyed off the *locator* (the ``ref_audio`` request string) via
:func:`vllm_omni.utils.reference_cache_key.locator_content_key`, not the
decoded waveform. The key is derived without decoding the audio, so a cache
hit returns immediately without resolving, resampling, or re-encoding the
reference — the expensive resolve happens only on a miss, inside the
single-flight leader.

On worker count: the audio tokenizer runs on CPU (deliberately kept off the
GPU to spare memory next to the ~8B talker + codec) and a single codec forward
already fans out across every core via torch's global intra-op thread pool. So
running several encodes at once does not add real parallelism — it just makes
them contend for the same cores and memory bandwidth, which measured *slower*
than encoding serially on a many-core host. The default is therefore a single
worker; ``num_workers`` (env ``MOSS_REF_AUDIO_ENCODE_WORKERS``) can be raised
for hosts where each encode under-uses the CPU, or for a GPU codec paired with
a non-zero wait window to coalesce forwards. The robust win here is the cache,
not concurrency: identical/repeated speakers skip resolve + encode entirely.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.utils.reference_cache_key import locator_content_key

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level defaults (kept small; runtime tunables live on the classes).
# ---------------------------------------------------------------------------

_DEFAULT_MAX_BATCH_SIZE = 8
# CPU codec: no wait window by default so requests are not delayed waiting for
# a batch to fill. Set a non-zero window (and it will coalesce) only when the
# codec runs somewhere batching actually pays off, e.g. a GPU.
_DEFAULT_MAX_BATCH_WAIT_MS = 0
_DEFAULT_CACHE_MAX_ITEMS = 8192
_DEFAULT_CACHE_MAX_BYTES = 64 * 1024 * 1024

_CACHE_ENV_VAR = "MOSS_REF_AUDIO_CACHE"
_CACHE_DISABLED_TOKENS = frozenset({"0", "false", "no", "off", ""})
_ENCODE_WORKERS_ENV_VAR = "MOSS_REF_AUDIO_ENCODE_WORKERS"

# Default parallel codec encodes. A single codec forward already saturates the
# CPU via torch's shared intra-op pool, so extra workers tend to contend rather
# than speed things up; benchmarking a many-core host showed 8 workers running
# *slower* than 1 under high concurrency. Default to serial and let operators
# opt into parallelism (``MOSS_REF_AUDIO_ENCODE_WORKERS``) where it pays off
# (CPU-underutilizing encodes, or a GPU codec with a wait window).
_DEFAULT_ENCODE_WORKERS = 1

_SHUTDOWN_SENTINEL: object = object()


def _cache_enabled_from_env() -> bool:
    """Interpret ``MOSS_REF_AUDIO_CACHE`` as an ops kill switch (default on)."""
    value = os.environ.get(_CACHE_ENV_VAR)
    if value is None:
        return True
    return value.strip().lower() not in _CACHE_DISABLED_TOKENS


def _encode_workers_from_env() -> int | None:
    """Read ``MOSS_REF_AUDIO_ENCODE_WORKERS`` as an ops override (>= 1)."""
    value = os.environ.get(_ENCODE_WORKERS_ENV_VAR)
    if value is None:
        return None
    try:
        workers = int(value.strip())
    except ValueError:
        logger.warning(
            "Ignoring non-integer %s=%r", _ENCODE_WORKERS_ENV_VAR, value
        )
        return None
    if workers < 1:
        logger.warning(
            "Ignoring %s=%r (must be >= 1)", _ENCODE_WORKERS_ENV_VAR, value
        )
        return None
    return workers


# ---------------------------------------------------------------------------
# Waveform preparation and hashing helpers.
# ---------------------------------------------------------------------------

def _as_waveform_tensor(wav_list: Any) -> torch.Tensor:
    """Return a CPU float32 waveform with shape ``[channels, samples]``."""
    wav = torch.as_tensor(wav_list, dtype=torch.float32, device="cpu")
    if wav.ndim == 0:
        wav = wav.reshape(1, 1)
    elif wav.ndim == 1:
        wav = wav.unsqueeze(0)
    elif wav.ndim == 2:
        # Some resolvers hand back ``[samples, channels]`` for stereo refs;
        # re-orient when the "long" axis is samples.
        if wav.shape[0] > wav.shape[1] and wav.shape[1] <= 8:
            wav = wav.transpose(0, 1)
    else:
        raise ValueError(
            f"reference audio must be 1D or 2D, got shape {tuple(wav.shape)}"
        )
    return wav.contiguous()


def _namespace_key(
    content_key: str,
    *,
    variant: str,
    n_vq: int,
    sr_target: int,
) -> str:
    """Namespace a content key so different codec configs never collide."""
    return f"moss_tts:{variant}:nq{int(n_vq)}:sr{int(sr_target)}:{content_key}"


def _prepare_waveform(
    wav: torch.Tensor,
    sample_rate: int,
    sr_target: int,
) -> torch.Tensor:
    """Return a CPU float32 waveform resampled to ``sr_target`` if needed."""
    wav = wav.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if int(sample_rate) != int(sr_target):
        # Local import: torchaudio only pulled in when a resample is needed.
        import torchaudio

        wav = torchaudio.functional.resample(
            wav,
            orig_freq=int(sample_rate),
            new_freq=int(sr_target),
        )
    return wav.contiguous()


def _stored_codes(codes: torch.Tensor) -> torch.Tensor:
    """Compact CPU representation used inside the LRU (int32)."""
    return codes.detach().to(device="cpu", dtype=torch.int32).contiguous()


def _return_codes(codes: torch.Tensor) -> torch.Tensor:
    """Detach + clone as CPU long so callers can freely mutate the result."""
    return codes.detach().to(device="cpu", dtype=torch.long).contiguous().clone()


def _set_fresh_exception(
    future: concurrent.futures.Future,
    message: str,
    cause: BaseException | None = None,
) -> None:
    """Assign a *new* exception per waiter so tracebacks never race.

    A shared exception instance would let concurrent ``future.result()`` calls
    mutate the same traceback object during re-raise.
    """
    if future.done():
        return
    if cause is None:
        future.set_exception(RuntimeError(message))
    else:
        future.set_exception(RuntimeError(f"{message}: {cause}"))


# ---------------------------------------------------------------------------
# Pooled reference encoder (worker pool + queue).
# ---------------------------------------------------------------------------

@dataclass
class _EncodeJob:
    wav: torch.Tensor
    sample_rate: int
    n_vq: int
    future: concurrent.futures.Future


class _WorkerShutdown(Exception):
    """Internal signal raised inside ``_drain_batch`` on shutdown."""


class _PooledReferenceEncoder:
    """Run codec encodes on a small pool of daemon workers.

    Callers submit one waveform at a time via :meth:`submit`; ``num_workers``
    daemon threads drain the shared queue and call
    ``processor.encode_audios_from_wav`` on the jobs they pick up. With the
    default zero wait window each worker encodes a single job, so up to
    ``num_workers`` encodes run in parallel. With a non-zero window a worker
    coalesces jobs that share ``(sample_rate, n_vq)`` into one batched forward;
    a failing batch falls back to per-item encodes so one bad waveform only
    fails its own future.

    De-duplication of concurrent same-content requests is *not* handled here
    -- that responsibility lives on :class:`_CachedReferenceEncoder`, which
    keeps this class focused on parallelism, batching and error isolation.
    """

    #: Reference audio longer than this is rejected before it reaches the
    #: worker; matches the Higgs cap and bounds batch-padding memory.
    MAX_REFERENCE_SECONDS = 100.0

    #: A single encode runs in well under a second; a result this late means
    #: the worker died or wedged, so fail the request instead of hanging the
    #: request slot forever.
    ENCODE_TIMEOUT_S = 120.0

    def __init__(
        self,
        processor: Any,
        *,
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
        max_batch_wait_ms: int = _DEFAULT_MAX_BATCH_WAIT_MS,
        num_workers: int | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")

        self._processor = processor
        self._max_batch_size = int(max_batch_size)
        self._max_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        # Default to a single worker (see _DEFAULT_ENCODE_WORKERS): on the CPU
        # codec extra workers contend rather than parallelize, and a single
        # worker is also what lets a non-zero wait window coalesce jobs into one
        # batched forward for a GPU codec.
        if num_workers is None:
            num_workers = _DEFAULT_ENCODE_WORKERS
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        self._num_workers = int(num_workers)
        self._queue: queue.Queue[_EncodeJob | object] = queue.Queue()
        self._closed = threading.Event()
        self._workers = [
            threading.Thread(
                target=self._worker_loop,
                name=f"moss-tts-ref-encode-{index}",
                daemon=True,
            )
            for index in range(self._num_workers)
        ]
        for worker in self._workers:
            worker.start()

    # -- Public API ---------------------------------------------------------

    @classmethod
    def check_reference_duration(
        cls, wav: torch.Tensor, sample_rate: int
    ) -> None:
        duration = int(wav.shape[-1]) / max(int(sample_rate), 1)
        if duration > cls.MAX_REFERENCE_SECONDS:
            raise ValueError(
                f"reference audio is {duration:.1f}s long, limit is "
                f"{cls.MAX_REFERENCE_SECONDS:.0f}s"
            )

    def submit(
        self,
        wav: torch.Tensor,
        *,
        sample_rate: int,
        n_vq: int,
    ) -> concurrent.futures.Future:
        """Enqueue an encode job; returns a future resolving to a code tensor."""
        future: concurrent.futures.Future = concurrent.futures.Future()
        if self._closed.is_set():
            _set_fresh_exception(future, "MOSS-TTS reference encoder is closed")
            return future
        self._queue.put(
            _EncodeJob(
                wav=wav,
                sample_rate=int(sample_rate),
                n_vq=int(n_vq),
                future=future,
            )
        )
        return future

    async def encode(
        self,
        wav: torch.Tensor,
        *,
        sample_rate: int,
        n_vq: int,
    ) -> torch.Tensor:
        """Async wrapper around :meth:`submit` with a hard timeout guard."""
        future = self.submit(wav, sample_rate=sample_rate, n_vq=n_vq)
        # ``asyncio.shield`` protects the future from cancellation of the
        # awaiting coroutine: the codec forward has already been queued and
        # cancelling it doesn't help other requests waiting behind it.
        return await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(future)),
            timeout=self.ENCODE_TIMEOUT_S,
        )

    def close(self, *, join_timeout_s: float = 5.0) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        # One sentinel per worker so each drains and exits.
        for _ in self._workers:
            self._queue.put(_SHUTDOWN_SENTINEL)
        deadline = time.monotonic() + float(join_timeout_s)
        for worker in self._workers:
            if worker.is_alive():
                worker.join(timeout=max(deadline - time.monotonic(), 0.0))
        # Fail anything that raced past the sentinels; harmless if empty.
        self._fail_queued_jobs("MOSS-TTS reference encoder is closed")

    # -- Worker plumbing ----------------------------------------------------

    def _fail_queued_jobs(self, message: str) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, _EncodeJob):
                _set_fresh_exception(item.future, message)

    def _worker_loop(self) -> None:
        while True:
            try:
                batch = self._drain_batch()
            except _WorkerShutdown:
                return

            try:
                self._encode_batch(batch)
            except BaseException as exc:
                # Should never happen — ``_encode_batch`` catches internally
                # and falls back to per-item encodes — but keep a hard net so
                # a worker can never die and orphan every waiting future.
                logger.exception(
                    "MOSS-TTS reference-encode worker failed a batch"
                )
                for job in batch:
                    _set_fresh_exception(
                        job.future,
                        "reference encode worker failed",
                        exc,
                    )

    def _drain_batch(self) -> list[_EncodeJob]:
        first = self._queue.get()
        if first is _SHUTDOWN_SENTINEL:
            raise _WorkerShutdown

        batch: list[_EncodeJob] = [first]  # type: ignore[list-item]
        deadline = time.monotonic() + self._max_wait_s
        while len(batch) < self._max_batch_size:
            try:
                if self._max_wait_s > 0:
                    remaining_s = deadline - time.monotonic()
                    if remaining_s <= 0:
                        break
                    item = self._queue.get(timeout=remaining_s)
                else:
                    item = self._queue.get_nowait()
            except queue.Empty:
                break

            if item is _SHUTDOWN_SENTINEL:
                # Put the sentinel back so a sibling worker (or this worker's
                # next loop) sees it and shuts down too.
                self._queue.put(_SHUTDOWN_SENTINEL)
                break
            batch.append(item)  # type: ignore[arg-type]
        return batch

    def _encode_batch(self, batch: list[_EncodeJob]) -> None:
        # Group by ``(sample_rate, n_vq)``: the underlying processor cannot
        # mix reference audio at different rates or codec depths in one call.
        groups: dict[tuple[int, int], list[_EncodeJob]] = {}
        for job in batch:
            groups.setdefault((job.sample_rate, job.n_vq), []).append(job)
        for (sample_rate, n_vq), jobs in groups.items():
            self._encode_group(jobs, sample_rate=sample_rate, n_vq=n_vq)

    def _encode_group(
        self,
        jobs: list[_EncodeJob],
        *,
        sample_rate: int,
        n_vq: int,
    ) -> None:
        try:
            with torch.no_grad():
                codes_list = self._processor.encode_audios_from_wav(
                    [job.wav for job in jobs],
                    sampling_rate=sample_rate,
                    n_vq=n_vq,
                )
        except BaseException:
            # A single bad ref shouldn't take down the whole batch — retry
            # each item on its own so failures stay isolated. (Only reachable
            # when a wait window coalesced more than one job.)
            logger.exception(
                "MOSS-TTS batched reference encode failed "
                "(batch=%d, sr=%d, n_vq=%d); retrying per item",
                len(jobs),
                sample_rate,
                n_vq,
            )
            self._encode_jobs_isolated(
                jobs, sample_rate=sample_rate, n_vq=n_vq
            )
            return

        if len(codes_list) != len(jobs):
            message = (
                f"reference encode returned {len(codes_list)} results "
                f"for a batch of {len(jobs)}"
            )
            for job in jobs:
                _set_fresh_exception(job.future, message)
            return

        for job, codes in zip(jobs, codes_list):
            if codes is None:
                _set_fresh_exception(
                    job.future, "reference encode produced no codes"
                )
                continue
            job.future.set_result(codes)

    def _encode_jobs_isolated(
        self,
        jobs: list[_EncodeJob],
        *,
        sample_rate: int,
        n_vq: int,
    ) -> None:
        for job in jobs:
            try:
                with torch.no_grad():
                    codes_list = self._processor.encode_audios_from_wav(
                        [job.wav],
                        sampling_rate=sample_rate,
                        n_vq=n_vq,
                    )
            except BaseException as exc:
                _set_fresh_exception(job.future, "reference encode failed", exc)
                continue

            if not codes_list or codes_list[0] is None:
                _set_fresh_exception(
                    job.future, "reference encode produced no codes"
                )
                continue
            job.future.set_result(codes_list[0])


# ---------------------------------------------------------------------------
# LRU + single-flight cache in front of the pooled encoder.
# ---------------------------------------------------------------------------

class _CachedReferenceEncoder:
    """CPU-int32 LRU cache and single-flight in front of a pooled encoder.

    Every path (miss, hit, follower) returns an *independent* CPU long tensor
    so downstream code can freely mutate it without corrupting a shared cache
    entry. Codes are stored as ``int32`` (lossless for typical codebook
    values in ``[0, 1023]``) to keep the byte budget small.

    The cache is fed by an async ``encode_fn`` callback, not a resolved
    waveform: the leader runs ``encode_fn`` (resolve + prepare + codec) exactly
    once per key, so a cache hit or a merged follower never resolves the
    reference at all.
    """

    #: Cadence of the periodic stats log; class attr so it is easy to tune.
    LOG_INTERVAL_S = 60.0

    #: Followers on a merged encode wait a little longer than the leader's
    #: hard timeout so a slow-but-not-hung leader still delivers a result.
    _FOLLOWER_TIMEOUT_S = _PooledReferenceEncoder.ENCODE_TIMEOUT_S + 10.0

    def __init__(
        self,
        *,
        max_items: int = _DEFAULT_CACHE_MAX_ITEMS,
        max_bytes: int = _DEFAULT_CACHE_MAX_BYTES,
    ) -> None:
        # Fail fast on non-positive capacities: a zero/negative bound would
        # make the eviction loop churn or the LRU never store anything and
        # thereby silently disable caching.
        if max_items < 1:
            raise ValueError(f"max_items must be >= 1, got {max_items}")
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")

        self._max_items = int(max_items)
        self._max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cache_bytes = 0
        self._inflight: dict[str, concurrent.futures.Future] = {}
        self._hits = 0
        self._misses = 0
        self._merged = 0
        self._last_log_time = 0.0
        self._leader_tasks: set[asyncio.Task[None]] = set()

    async def encode(
        self,
        cache_key: str,
        encode_fn: Callable[[], Awaitable[torch.Tensor]],
        *,
        desc: str,
    ) -> torch.Tensor:
        """Return codes for ``cache_key``, running ``encode_fn`` at most once.

        ``encode_fn`` is an async callable that produces the raw code tensor
        (typically: resolve the locator, prepare the waveform, run the codec).
        It runs only on a genuine miss, inside the single-flight leader.
        """
        cached: torch.Tensor | None = None
        follower_future: concurrent.futures.Future | None = None
        leader_future: concurrent.futures.Future | None = None

        with self._lock:
            entry = self._cache.get(cache_key)
            if entry is not None:
                self._cache.move_to_end(cache_key)
                self._hits += 1
                cached = entry
            elif cache_key in self._inflight:
                self._merged += 1
                follower_future = self._inflight[cache_key]
            else:
                self._misses += 1
                leader_future = concurrent.futures.Future()
                self._inflight[cache_key] = leader_future

        if cached is not None:
            self._maybe_log_stats()
            return _return_codes(cached)

        if follower_future is not None:
            return await self._await_stored_codes(
                follower_future, desc=desc, wrap_failures=True
            )

        # Leader path. The encode runs in an independent task so that
        # cancelling *this* caller does not cancel the shared encode — a
        # follower that arrives before it finishes still merges onto it.
        assert leader_future is not None
        self._track_leader_task(
            asyncio.create_task(
                self._run_leader_encode(
                    cache_key=cache_key,
                    encode_fn=encode_fn,
                    leader_future=leader_future,
                )
            )
        )
        return await self._await_stored_codes(
            leader_future, desc=desc, wrap_failures=False
        )

    def _track_leader_task(self, task: asyncio.Task[None]) -> None:
        self._leader_tasks.add(task)
        task.add_done_callback(self._leader_tasks.discard)

    async def _run_leader_encode(
        self,
        *,
        cache_key: str,
        encode_fn: Callable[[], Awaitable[torch.Tensor]],
        leader_future: concurrent.futures.Future,
    ) -> None:
        try:
            result = await encode_fn()
            stored = _stored_codes(result)
        except asyncio.CancelledError as exc:
            with self._lock:
                self._inflight.pop(cache_key, None)
            if not leader_future.done():
                leader_future.set_exception(exc)
            raise
        except BaseException as exc:
            with self._lock:
                self._inflight.pop(cache_key, None)
            if not leader_future.done():
                leader_future.set_exception(exc)
            return

        with self._lock:
            self._put_locked(cache_key, stored)
            self._inflight.pop(cache_key, None)
        if not leader_future.done():
            leader_future.set_result(stored)
        self._maybe_log_stats()

    async def _await_stored_codes(
        self,
        future: concurrent.futures.Future,
        *,
        desc: str,
        wrap_failures: bool,
    ) -> torch.Tensor:
        try:
            stored = await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(future)),
                timeout=self._FOLLOWER_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            raise
        except Exception as cause:
            if wrap_failures:
                # Fresh exception per follower: sharing one exception object
                # lets concurrent re-raises corrupt the traceback (same
                # lesson as ``_set_fresh_exception``).
                raise RuntimeError(
                    f"reference encode failed for {desc}: {cause}"
                ) from cause
            raise
        return _return_codes(stored)

    def _put_locked(self, key: str, tensor: torch.Tensor) -> None:
        size = int(tensor.numel() * tensor.element_size())
        if size > self._max_bytes:
            # Refuse to admit a single entry larger than the whole budget:
            # otherwise the eviction loop below would immediately drop it
            # and every future access would miss again anyway.
            return
        old = self._cache.pop(key, None)
        if old is not None:
            self._cache_bytes -= int(old.numel() * old.element_size())
        self._cache[key] = tensor
        self._cache.move_to_end(key)
        self._cache_bytes += size
        while self._cache and (
            len(self._cache) > self._max_items
            or self._cache_bytes > self._max_bytes
        ):
            _, evicted = self._cache.popitem(last=False)
            self._cache_bytes -= int(evicted.numel() * evicted.element_size())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "merged": self._merged,
                "entries": len(self._cache),
                "bytes": self._cache_bytes,
                "inflight": len(self._inflight),
            }

    def _maybe_log_stats(self) -> None:
        # Rate-limited stats log; double-checked under the lock so parallel
        # requests don't spam the logger.
        now = time.monotonic()
        with self._lock:
            if now - self._last_log_time < self.LOG_INTERVAL_S:
                return
            self._last_log_time = now
            snapshot = (
                self._hits,
                self._misses,
                self._merged,
                len(self._cache),
                self._cache_bytes,
                len(self._inflight),
            )
        logger.info(
            "MOSS-TTS ref cache: hits=%d misses=%d merged=%d "
            "entries=%d bytes=%d inflight=%d",
            *snapshot,
        )


# ---------------------------------------------------------------------------
# Public async facade used by the OpenAI speech serving layer.
# ---------------------------------------------------------------------------

class MossTTSReferenceEncoder:
    """Async facade tying together preprocessing, pooling and caching.

    One instance is held per ``(processor, variant, n_vq, sr_target)`` tuple
    by the serving layer so the daemon worker pool and content-addressed LRU
    are shared across every request that reuses that MOSS-TTS variant.
    """

    def __init__(
        self,
        processor: Any,
        *,
        variant: str,
        n_vq: int,
        sr_target: int,
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
        max_batch_wait_ms: int = _DEFAULT_MAX_BATCH_WAIT_MS,
        num_workers: int | None = None,
        cache_max_items: int = _DEFAULT_CACHE_MAX_ITEMS,
        cache_max_bytes: int = _DEFAULT_CACHE_MAX_BYTES,
        enable_cache: bool | None = None,
    ) -> None:
        self._variant = str(variant)
        self._n_vq = int(n_vq)
        self._sr_target = int(sr_target)
        self._closed_stats: dict[str, Any] | None = None
        self._pooled = _PooledReferenceEncoder(
            processor,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            num_workers=num_workers,
        )

        cache_on = (
            _cache_enabled_from_env()
            if enable_cache is None
            else bool(enable_cache)
        )
        if cache_on:
            self._cached: _CachedReferenceEncoder | None = (
                _CachedReferenceEncoder(
                    max_items=cache_max_items,
                    max_bytes=cache_max_bytes,
                )
            )
        else:
            logger.info(
                "MOSS-TTS reference-audio cache disabled via %s; "
                "pooled encoding only",
                _CACHE_ENV_VAR,
            )
            self._cached = None

    async def encode_reference_codes(
        self,
        ref_str: str,
        *,
        resolve_ref_audio: Callable[[str], Awaitable[tuple[list, int]]],
    ) -> torch.Tensor:
        """Resolve, encode, and cache a reference audio into MOSS codes.

        The cache key is derived from the ``ref_str`` locator up front
        (:func:`locator_content_key`), so a cache hit returns without ever
        calling ``resolve_ref_audio``. On a miss the single-flight leader
        resolves, resamples and runs the codec exactly once.

        The returned tensor is a fresh CPU ``torch.long`` clone regardless of
        cache temperature, so the caller may mutate it freely.
        """
        cache_key = _namespace_key(
            locator_content_key(ref_str),
            variant=self._variant,
            n_vq=self._n_vq,
            sr_target=self._sr_target,
        )
        desc = repr(str(ref_str)[:64])

        async def _encode_fn() -> torch.Tensor:
            wav_list, sample_rate = await resolve_ref_audio(ref_str)
            wav = _as_waveform_tensor(wav_list)
            # Reject over-long refs before the codec (and, on the cache path,
            # before anything is stored) — a 100+ s ref must never be admitted.
            _PooledReferenceEncoder.check_reference_duration(
                wav, int(sample_rate)
            )
            prepared = _prepare_waveform(wav, int(sample_rate), self._sr_target)
            return await self._pooled.encode(
                prepared, sample_rate=self._sr_target, n_vq=self._n_vq
            )

        if self._cached is not None:
            return await self._cached.encode(cache_key, _encode_fn, desc=desc)

        # Cache-off path: no LRU, no dedup — concurrent same-content requests
        # each run through the codec (matches sglang's behaviour when
        # ``MOSS_REF_AUDIO_CACHE`` is disabled).
        return _return_codes(await _encode_fn())

    def close(self, *, join_timeout_s: float = 5.0) -> None:
        if self._closed_stats is None:
            self._closed_stats = self.stats()
        self._pooled.close(join_timeout_s=join_timeout_s)
        self._cached = None

    def stats(self) -> dict[str, Any]:
        if self._cached is None:
            if self._closed_stats is not None:
                return dict(self._closed_stats)
            return {
                "hits": 0,
                "misses": 0,
                "merged": 0,
                "entries": 0,
                "bytes": 0,
                "inflight": 0,
                "cache_enabled": False,
            }
        return {**self._cached.stats(), "cache_enabled": True}


# ---------------------------------------------------------------------------
# Factory (used by the serving layer's per-variant getter).
# ---------------------------------------------------------------------------

def create_reference_encoder(
    processor: Any,
    *,
    variant: str,
    n_vq: int,
    sr_target: int,
    max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
    max_batch_wait_ms: int = _DEFAULT_MAX_BATCH_WAIT_MS,
    num_workers: int | None = None,
    cache_max_items: int = _DEFAULT_CACHE_MAX_ITEMS,
    cache_max_bytes: int = _DEFAULT_CACHE_MAX_BYTES,
    enable_cache: bool | None = None,
) -> MossTTSReferenceEncoder:
    if num_workers is None:
        num_workers = _encode_workers_from_env()
    return MossTTSReferenceEncoder(
        processor,
        variant=variant,
        n_vq=n_vq,
        sr_target=sr_target,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        num_workers=num_workers,
        cache_max_items=cache_max_items,
        cache_max_bytes=cache_max_bytes,
        enable_cache=enable_cache,
    )
