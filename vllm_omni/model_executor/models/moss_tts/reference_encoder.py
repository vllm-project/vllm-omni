# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Reference-audio encoding + speaker cache for the MOSS-TTS-family talker.

This lives in the model package (not the shared serving layer) so all
MOSS-specific reference handling stays with the model — mirroring how Fish
Speech (``dac_encoder.encode_reference_audio_codes``), CosyVoice3, and
Qwen3-TTS keep their reference/speaker extraction next to the model rather than
in ``serving_speech.py``. The serving layer constructs one
:class:`MossReferenceEncoder` per server (lazily, alongside the upstream MOSS
processor) and calls :meth:`MossReferenceEncoder.encode` with its generic
helpers (the audio resolver, the artifact-key lookup, and the process-wide
speaker cache).

On top of that cache the encoder adds content-addressed keys (the same clip
arriving via different locators shares one entry), single-flight (concurrent
requests for one uncached clip join a single encode), and micro-batched
encoding (cold encodes arriving close together share one processor forward).

Kept import-light (``asyncio`` / ``hashlib`` / ``numpy`` / ``torch`` plus the logger)
so importing it from the API-server process does not pull the talker/codec.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from vllm.logger import init_logger

from vllm_omni.model_executor.models.moss_tts.reference_encoder_graphs import (
    FINE_BATCH_SIZES,
    FINE_BUCKET_SECONDS,
    MossReferenceEncoderGraphs,
    install_windowed_attention,
)
from vllm_omni.model_executor.models.moss_tts.shared_reference_encoder import (
    SharedReferenceEncoderClient,
    SharedReferenceEncoderHost,
    SharedReferenceEncoderStartupError,
    elect_host,
    socket_path_fits,
)

logger = init_logger(__name__)

# Coalescing defaults. Kept as module constants (not env/CLI) to match the
# ``_REF_AUDIO_RESOLVE_CACHE_MAX_*`` convention in serving_speech.py.
_REF_ENCODE_BATCH_WINDOW_MS = 10.0
_REF_ENCODE_MAX_BATCH = 32
# Experiment switches for cold-reference latency (see the A/B notes):
# batch window in ms; batches in flight per API process with a shared encoder;
# "high" puts reference encodes on the highest-priority CUDA stream.
_REF_ENCODE_WINDOW_ENV = "VLLM_OMNI_MOSS_REF_BATCH_WINDOW_MS"
_REF_ENCODE_INFLIGHT_ENV = "VLLM_OMNI_MOSS_REF_INFLIGHT"
_REF_ENCODE_PRIORITY_ENV = "VLLM_OMNI_MOSS_REF_STREAM_PRIORITY"
# "1" resamples and loudness-normalizes clips on the encoder's GPU instead of
# the API process's CPU (not bit-identical to the CPU path).
_REF_ENCODE_GPU_PREP_ENV = "VLLM_OMNI_MOSS_REF_GPU_PREP"

_INT32_MAX = 2**31
# Inline (data: URI) references already resolved and encoded, keyed by the
# serving resolve key. Codes are small; the URI itself is not retained.
_INLINE_INDEX_MAX_ENTRIES = 4096
# The shared code directory is bounded like the in-process index: any API
# process that stores a new entry periodically drops the least recently used
# files and stale temporary files. Every use of an entry, in memory or from
# the file, refreshes the file's mtime at most once per interval.
_SHARED_CODES_MAX_ENTRIES = _INLINE_INDEX_MAX_ENTRIES
_SHARED_CODES_PRUNE_EVERY = 64
_SHARED_CODES_STALE_TMP_S = 60.0
_SHARED_CODES_TOUCH_S = 1.0
# Directory that API processes of one server share, so an inline reference
# encoded by any frontend is reused by all. Multi-API servers use a directory
# scoped to the server by default; this overrides it, and an empty value
# turns sharing off.
_SHARED_CODES_DIR_ENV = "VLLM_OMNI_MOSS_REF_CODES_SHARED_DIR"
# "0" keeps one reference encoder per API process even with a shared directory.
_SHARED_ENCODER_ENV = "VLLM_OMNI_MOSS_REF_SHARED_ENCODER"
_shared_role: str | None = None
# Concurrent encode workers in the shared host, each with its own CUDA graphs.
_SHARED_WORKERS_ENV = "VLLM_OMNI_MOSS_REF_ENCODER_WORKERS"
_SHARED_WORKER_BATCH_SECONDS = 96.0


def shared_codes_dir(server_dir: str | None) -> str | None:
    """Reference-code directory shared by this server's API processes, if any.

    ``server_dir`` is the multi-API server's scratch directory (None with one
    API process).
    """
    explicit = os.environ.get(_SHARED_CODES_DIR_ENV)
    if explicit is not None:
        return explicit or None
    return os.path.join(server_dir, "moss-ref-codes") if server_dir else None


def shared_encoder_enabled(shared_dir: str | None) -> bool:
    """Whether API processes sharing ``shared_dir`` also share one encoder."""
    if not shared_dir or os.environ.get(_SHARED_ENCODER_ENV, "1") == "0":
        return False
    return socket_path_fits(shared_dir)


def shared_encoder_role(shared_dir: str | None) -> str:
    """``"host"``, ``"client"`` or ``""`` (no shared encoder) for this process.

    Decided once, before the processor places its tokenizer on the GPU, so
    client processes can leave it on the CPU.
    """
    global _shared_role
    if _shared_role is None:
        _shared_role = ""
        if shared_dir and shared_encoder_enabled(shared_dir):
            os.makedirs(shared_dir, exist_ok=True)
            _shared_role = "host" if elect_host(shared_dir) else "client"
        elif shared_dir and os.environ.get(_SHARED_ENCODER_ENV, "1") != "0":
            logger.warning(
                "MOSS shared reference encoder disabled: %s is too long a path for its Unix socket; "
                "each API process encodes its own references",
                shared_dir,
            )
    return _shared_role


# Opt in to reference graphs; existing deployments keep the eager encoder.
_REF_ENCODE_GRAPHS_ENV = "VLLM_OMNI_MOSS_REF_GRAPHS"


def reference_graphs_enabled() -> bool:
    return os.environ.get(_REF_ENCODE_GRAPHS_ENV, "0") == "1"


def reference_stream_priority() -> int:
    """CUDA stream priority for reference encodes (lower is more urgent)."""
    if os.environ.get(_REF_ENCODE_PRIORITY_ENV, "") == "high" and torch.cuda.is_available():
        return torch.cuda.Stream.priority_range()[1]
    return 0


# "sdpa" keeps the tokenizer's dense masked SDPA attention. By default the
# reference encoder attends through a local-window flash kernel, the backend
# the tokenizer config selects (flash_attention_2) when flash-attn is present.
_REF_ENCODE_ATTN_ENV = "VLLM_OMNI_MOSS_REF_ATTN"
# "0" captures the encoder's eager kernels. By default the captured encoder is
# compiled first, which fuses its many small elementwise kernels.
_REF_ENCODE_COMPILE_ENV = "VLLM_OMNI_MOSS_REF_COMPILE"

_Waveform = NDArray[np.float32] | list[float]


@lru_cache(maxsize=16)
def _reference_resampler(sr: int, sr_target: int) -> torch.nn.Module:
    import torchaudio

    # Match functional.resample's float32 kernel construction, rather than
    # Resample's default float64 construction followed by a float32 cast.
    return torchaudio.transforms.Resample(sr, sr_target, dtype=torch.float32)


def _prep_wav_sync(waveform: _Waveform, sr: int, sr_target: int) -> torch.Tensor:
    """Tensor-ise + resample one clip to ``sr_target`` (the blocking prep)."""
    # Keep the tensor independent from the server-side waveform/cache buffer.
    wav_np = np.array(waveform, dtype=np.float32, order="C", copy=True)
    wav = torch.from_numpy(wav_np)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != sr_target:
        wav = _reference_resampler(sr, sr_target)(wav)
    return wav


def _inline_reference_key(ref_str: str) -> str | None:
    """SHA-1 of a ``data:`` URI, matching the serving resolve key for it."""
    if not isinstance(ref_str, str) or ref_str[:5].lower() != "data:":
        return None
    return hashlib.sha1(ref_str.encode("utf-8")).hexdigest()


def _to_compact_codes(codes: torch.Tensor) -> torch.Tensor:
    """Downcast RVQ codes to int32 for compact caching."""
    codes = codes.detach().cpu().contiguous()
    if codes.numel() == 0:
        return codes.to(torch.int32)
    hi = int(codes.max().item())
    lo = int(codes.min().item())
    if -_INT32_MAX <= lo and hi < _INT32_MAX:
        return codes.to(torch.int32)
    logger.warning("MOSS ref codes out of int32 range (min=%d max=%d); caching as int64", lo, hi)
    return codes.to(torch.int64)


def _clone_out(codes: torch.Tensor) -> torch.Tensor:
    """Return an independent int64 copy for the caller."""
    return codes.to(torch.int64, copy=True)


def _registered_voice(voice_name: str | None, voice_created_at: int) -> tuple[str | None, int]:
    """Return ``(name, created_at)`` for a registered uploaded voice, else ``(None, 0)``.

    The OpenAI speech API requires a ``voice`` field, and callers often send
    placeholders such as "default" for ref-audio voice cloning. Only registered
    uploaded voices have a positive created_at timestamp; other names must not
    key the cache because the timbre comes from ref_audio.

    The name is lowercased here so every derived key agrees on one spelling:
    if the cache key kept the caller's casing while the flight key normalized
    it, two concurrent callers differing only by case would share a flight but
    populate different cache slots.
    """
    name = voice_name.strip().lower() if isinstance(voice_name, str) else ""
    created_at = int(voice_created_at)
    if name and created_at > 0:
        return name, created_at
    return None, 0


class _RefEncodeBatcher:
    """Coalesce cold reference encodes into batched processor forwards."""

    def __init__(
        self,
        encode_batch_fn: Callable[[list[tuple[_Waveform, int]]], list],
        *,
        window_ms: float,
        max_batch: int,
        max_inflight: int = 1,
    ):
        # encode_batch_fn: sync, takes [(waveform, sr), ...], returns a list of
        # (codes_tensor | Exception) aligned to the input order.
        self._encode_batch_fn = encode_batch_fn
        self._window_s = max(0.0, float(window_ms) / 1000.0)
        self._max_batch = max(1, int(max_batch))
        # Batches encoded at once. Clips that arrive while every slot is busy
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

    async def submit(self, waveform: _Waveform, sr: int) -> torch.Tensor:
        self._ensure_started()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((waveform, sr, fut))  # type: ignore[union-attr]
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

    async def _run_and_release(self, jobs: list[tuple[_Waveform, int, asyncio.Future]]) -> None:
        try:
            await self._run_batch(jobs)
        finally:
            del jobs
            assert self._slots is not None
            self._slots.release()

    async def _coalesce(self, first: tuple) -> list[tuple]:
        """Group ``first`` with jobs arriving within the batch window."""
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

    async def _run_batch(self, jobs: list[tuple[_Waveform, int, asyncio.Future]]) -> None:
        payload = [(waveform, sr) for waveform, sr, _ in jobs]
        futs = [fut for _, _, fut in jobs]
        try:
            results = await asyncio.to_thread(self._encode_batch_fn, payload)
        except Exception as exc:  # noqa: BLE001 — propagate the batch failure to every waiter
            for fut in futs:
                if not fut.done():
                    fut.set_exception(exc)
            return
        for fut, res in zip(futs, results):
            if fut.done():
                continue
            if isinstance(res, BaseException):
                fut.set_exception(res)
            else:
                fut.set_result(res)

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


class MossReferenceEncoder:
    """Content-addressed, single-flight, micro-batched reference-audio encoder."""

    def __init__(
        self,
        processor: Any,
        *,
        variant: str,
        n_vq: int,
        sr_target: int,
        speaker_cache: Any,
        window_ms: float | None = None,
        max_batch: int = _REF_ENCODE_MAX_BATCH,
        server_dir: str | None = None,
    ):
        self._processor = processor
        self._n_vq = int(n_vq)
        self._sr_target = int(sr_target)
        self._speaker_cache = speaker_cache
        # ``created_at`` and the audio-content name vary per request; the
        # model_type namespaces the whole family so a moss_tts server never
        # collides with another model's speaker-cache entries.
        self._model_type = f"moss_tts_{variant}_nq{int(n_vq)}"
        self._inflight: dict[str | tuple[str, str], asyncio.Task] = {}
        self._inline_index: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._shared_codes_dir = shared_codes_dir(server_dir)
        self._shared_codes_stores = 0
        # Last mtime refresh of each indexed shared entry (monotonic seconds).
        self._shared_codes_touched: dict[str, float] = {}
        if self._shared_codes_dir:
            os.makedirs(self._shared_codes_dir, exist_ok=True)
        # Realtime encodes with its own codec and never shares the encoder.
        self._shares_encoder = type(self)._encode_prepared is MossReferenceEncoder._encode_prepared and (
            shared_encoder_enabled(self._shared_codes_dir)
        )
        # This process's own encodes replay its graphs, which hold one batch
        # at a time; only a shared encoder's workers run batches concurrently.
        self._local_lock = threading.Lock()
        self._shared_host: SharedReferenceEncoderHost | None = None
        self._shared_client: SharedReferenceEncoderClient | None = None
        self._graphs: MossReferenceEncoderGraphs | None = None
        self._attention_installed = False
        self._graphs_enabled = reference_graphs_enabled()
        if window_ms is None:
            window_ms = float(os.environ.get(_REF_ENCODE_WINDOW_ENV, _REF_ENCODE_BATCH_WINDOW_MS))
        # A shared encoder serves several batches at once (one per host
        # worker); a process's own graphs replay one batch at a time.
        inflight = 1
        if self._shares_encoder:
            inflight = max(1, int(os.environ.get(_REF_ENCODE_INFLIGHT_ENV, "1")))
        self._local_stream: torch.cuda.Stream | None = None
        self._gpu_prep = os.environ.get(_REF_ENCODE_GPU_PREP_ENV, "0") == "1"
        self._gpu_resamplers: dict[torch.device, torch.nn.Module] = {}
        self._batcher = _RefEncodeBatcher(
            self._encode_batch_sync, window_ms=window_ms, max_batch=max_batch, max_inflight=inflight
        )

    def _make_cache_key(self, name: str, created_at: int) -> tuple:
        return self._speaker_cache.make_cache_key(name, model_type=self._model_type, created_at=int(created_at))

    async def encode(
        self,
        ref_str: str,
        *,
        resolve_ref_audio: Callable[[str], Awaitable[tuple[_Waveform, int, str]]],
        get_artifact_key: Callable[[str], str | None],
        voice_name: str | None = None,
        voice_created_at: int = 0,
    ) -> tuple[torch.Tensor, str | None]:
        """Encode one reference clip into MOSS RVQ codes, reusing the cache.

        ``resolve_ref_audio`` maps ``ref_str`` to ``(waveform, sr, cache_key)``
        where *cache_key* is the content-aware resolve key (it folds mtime/size
        for local files). ``get_artifact_key`` maps that resolve key to the
        waveform-content artifact key, or ``None`` when unknown.

        Returns ``(codes, resolve_cache_key)``; the key is ``None`` when the
        clip was served from the named-voice cache without resolving (the
        caller salts those requests with ``voice_created_at`` instead).
        """
        voice_name, created_at = _registered_voice(voice_name, voice_created_at)
        inline_key: str | None = None

        if voice_name:
            # A named voice has a stable key that does not depend on the
            # resolved audio, so the cache can be checked before resolving.
            flight_key = f"voice:{voice_name}:{created_at}"
            cached = self._speaker_cache.get(self._make_cache_key(voice_name, created_at))
            if cached is not None:
                return _clone_out(cached["codes"]), None
        else:
            inline_key = _inline_reference_key(ref_str)
            if inline_key is not None:
                codes = self._inline_index.get(inline_key)
                if codes is None:
                    codes = self._load_shared_codes(inline_key)
                if codes is not None:
                    # A data: URI is immutable content, and its serving
                    # resolve key is this same digest: skip the resolve hop.
                    self._inline_index.move_to_end(inline_key)
                    self._touch_shared_codes(inline_key)
                    return _clone_out(codes), inline_key
            # Anonymous refs have no pre-resolve hot path: the content key must
            # come from the resolve itself (mtime/size from a single stat) so an
            # on-disk edit invalidates the cached codes; the flight body
            # re-checks the speaker cache right after resolving, which is cheap
            # when the resolve cache is warm. The flight key is the request-side
            # reference (not the content hash), so concurrent requests for the
            # same ref_str also share the resolve/download, not just the encode.
            # Key by the locator itself: equality is exact, and hashing a
            # large inline data URI once per request object is much cheaper
            # than a separate SHA-1 of its UTF-8 copy on the event loop.
            flight_key = ("ref", ref_str)

        codes, resolve_key = await self._single_flight(
            flight_key,
            lambda: self._resolve_and_encode(ref_str, resolve_ref_audio, get_artifact_key, voice_name, created_at),
        )
        if not voice_name and inline_key is not None and resolve_key == inline_key:
            # Only index when the resolver's key is exactly this digest.
            self._remember_inline(inline_key, codes)
            self._store_shared_codes(inline_key, codes)
        return _clone_out(codes), resolve_key

    def _remember_inline(self, key: str, codes: torch.Tensor) -> None:
        self._inline_index[key] = codes
        self._inline_index.move_to_end(key)
        while len(self._inline_index) > _INLINE_INDEX_MAX_ENTRIES:
            evicted, _ = self._inline_index.popitem(last=False)
            self._shared_codes_touched.pop(evicted, None)

    def _load_shared_codes(self, key: str) -> torch.Tensor | None:
        if not self._shared_codes_dir:
            return None
        try:
            array = np.load(os.path.join(self._shared_codes_dir, key + ".npy"), allow_pickle=False)
        except (OSError, ValueError):
            return None
        if array.ndim != 2 or array.shape[1] != self._n_vq or array.dtype not in (np.int32, np.int64):
            return None
        codes = torch.from_numpy(array)
        self._remember_inline(key, codes)
        return codes

    def _touch_shared_codes(self, key: str) -> None:
        """Refresh a used shared entry's mtime so pruning keeps it, at most once per interval."""
        if not self._shared_codes_dir:
            return
        now = time.monotonic()
        if now - self._shared_codes_touched.get(key, -_SHARED_CODES_TOUCH_S) < _SHARED_CODES_TOUCH_S:
            return
        self._shared_codes_touched[key] = now
        with contextlib.suppress(OSError):
            os.utime(os.path.join(self._shared_codes_dir, key + ".npy"))

    def _store_shared_codes(self, key: str, codes: torch.Tensor) -> None:
        if not self._shared_codes_dir:
            return
        path = os.path.join(self._shared_codes_dir, key + ".npy")
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "wb") as handle:
                np.save(handle, codes.detach().cpu().numpy(), allow_pickle=False)
            os.replace(tmp, path)
        except OSError:
            logger.warning("MOSS ref codes: could not write shared entry %s", path, exc_info=True)
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            return
        if self._shared_codes_stores % _SHARED_CODES_PRUNE_EVERY == 0:
            self._prune_shared_codes()
        self._shared_codes_stores += 1

    def _prune_shared_codes(self) -> None:
        """Keep the most recently used shared entries, across all API processes."""
        assert self._shared_codes_dir is not None
        entries: list[tuple[float, str]] = []
        now = time.time()
        try:
            with os.scandir(self._shared_codes_dir) as scan:
                for entry in scan:
                    try:
                        mtime = entry.stat().st_mtime
                    except OSError:
                        continue  # removed by another process
                    if entry.name.endswith(".npy"):
                        entries.append((mtime, entry.path))
                    elif entry.name.endswith(".tmp") and now - mtime > _SHARED_CODES_STALE_TMP_S:
                        with contextlib.suppress(OSError):
                            os.unlink(entry.path)
        except OSError:
            return
        excess = len(entries) - _SHARED_CODES_MAX_ENTRIES
        if excess <= 0:
            return
        # Prune a little below the bound so the scan runs rarely.
        entries.sort()
        for _, stale in entries[: excess + _SHARED_CODES_PRUNE_EVERY]:
            with contextlib.suppress(OSError):
                os.unlink(stale)

    async def _single_flight(
        self,
        flight_key: str | tuple[str, str],
        start_flight: Callable[[], Awaitable[tuple[torch.Tensor, str]]],
    ) -> tuple[torch.Tensor, str]:
        """Join the in-flight encode for ``flight_key``, starting one if absent.

        The shared task is awaited through ``shield`` so a caller cancelling
        its own request does not cancel the flight (asyncio would otherwise
        propagate the cancel into the awaited task and take down every other
        waiter with it).
        """
        task = self._inflight.get(flight_key)
        if task is not None:
            return await asyncio.shield(task)

        task = asyncio.create_task(start_flight())
        self._inflight[flight_key] = task

        # Retire the slot when the flight *completes*, not when the creating
        # caller returns: if the creator is cancelled the shielded task keeps
        # running, and popping the slot early would let the next arrival start
        # a duplicate resolve/encode instead of joining this one. Identity
        # guard: only drop the slot if it still holds *our* task (a later
        # request may have replaced it after ours completed).
        def _retire(t: asyncio.Task, key: str | tuple[str, str] = flight_key) -> None:
            if self._inflight.get(key) is t:
                self._inflight.pop(key, None)

        task.add_done_callback(_retire)
        return await asyncio.shield(task)

    async def _resolve_and_encode(
        self,
        ref_str: str,
        resolve_ref_audio: Callable[[str], Awaitable[tuple[_Waveform, int, str]]],
        get_artifact_key: Callable[[str], str | None],
        voice_name: str | None,
        created_at: int,
    ) -> tuple[torch.Tensor, str]:
        """Flight body: resolve → re-check cache by content hash → batch-encode."""
        waveform, sr, resolve_key = await resolve_ref_audio(ref_str)

        # The content hash is available now that the clip is resolved; this also
        # catches the case where another flight populated the cache in between.
        if voice_name:
            key = self._make_cache_key(voice_name, created_at)
        else:
            artifact_key = get_artifact_key(resolve_key)
            key_name = ("ref:" + artifact_key) if artifact_key else ("ref:" + resolve_key)
            key = self._make_cache_key(key_name, 0)

        cached = self._speaker_cache.get(key)
        if cached is not None:
            return cached["codes"], resolve_key

        codes = await self._batcher.submit(waveform, sr)
        compact = _to_compact_codes(codes)
        self._speaker_cache.put(key, {"codes": compact})
        logger.debug(
            "MOSS ref encode STORE key=%s shape=%s dtype=%s",
            key,
            tuple(compact.shape),
            compact.dtype,
        )
        return compact, resolve_key

    def _encode_batch_sync(self, payload: list[tuple[_Waveform, int]]) -> list:
        """Worker-thread body: prep each clip, then one batched forward."""
        n = len(payload)
        results: list = [None] * n
        prepared: list[torch.Tensor] = []
        prepared_idx: list[int] = []
        for i, (waveform, sr) in enumerate(payload):
            try:
                prepared.append(self._encoder_input(_prep_wav_sync(waveform, sr, self._sr_target)))
                prepared_idx.append(i)
            except Exception as exc:  # noqa: BLE001 — isolate this clip's failure
                results[i] = exc
        if not prepared:
            return results

        for orig_i, codes in zip(prepared_idx, self._encode_shared_or_local(prepared)):
            results[orig_i] = codes
        return results

    def prepare(self) -> None:
        """Capture the CUDA graphs (or start the shared encoder) before the first request."""
        role = shared_encoder_role(self._shared_codes_dir) if self._shares_encoder else ""
        if role == "host":
            self._host().wait_until_ready()
        elif role == "client":
            if self._shared_client is None:
                self._shared_client = SharedReferenceEncoderClient(self._shared_codes_dir)
            self._shared_client.wait_until_ready()
        elif role == "":
            with self._local_lock:
                self._reference_graphs()

    def _host(self) -> SharedReferenceEncoderHost:
        if self._shared_host is None:
            # Graph capture runs in the background; clips queue until it ends.
            self._shared_host = SharedReferenceEncoderHost(
                self._shared_codes_dir,
                self._host_workers(),
                stream_priority=reference_stream_priority(),
                batch_window_ms=float(os.environ.get("VLLM_OMNI_MOSS_REF_HOST_WINDOW_MS", "0")),
            )
        return self._shared_host

    def _encode_shared_or_local(self, prepared: list[torch.Tensor]) -> list:
        role = shared_encoder_role(self._shared_codes_dir) if self._shares_encoder else ""
        if role == "host":
            return self._host().encode(prepared)
        place_tokenizer = False
        if role == "client":
            if self._shared_client is None:
                self._shared_client = SharedReferenceEncoderClient(self._shared_codes_dir)
            try:
                return self._shared_client.encode(prepared)
            except SharedReferenceEncoderStartupError:
                raise
            except Exception:  # noqa: BLE001 — the host is unreachable: encode here
                logger.warning("MOSS shared reference encoder unavailable; encoding locally", exc_info=True)
                place_tokenizer = True
        with self._local_lock:
            # Moving the tokenizer rewrites its parameters one by one: with
            # several batches in flight, another one must not encode until the
            # whole module is on the device (its stream also follows the device).
            if place_tokenizer:
                self._place_tokenizer_for_local_encode()
            with self._local_stream_context():
                return self._encode_local(prepared)

    def _local_stream_context(self):
        tokenizer = getattr(self._processor, "audio_tokenizer", None)
        params = next(tokenizer.parameters(), None) if isinstance(tokenizer, torch.nn.Module) else None
        if params is None or params.device.type != "cuda":
            return contextlib.nullcontext()
        if self._local_stream is None:
            self._local_stream = torch.cuda.Stream(device=params.device, priority=reference_stream_priority())
        return torch.cuda.stream(self._local_stream)

    def _host_workers(self) -> list:
        """Encode workers of the shared host, each with its own graphs and stream."""
        count = max(1, int(os.environ.get(_SHARED_WORKERS_ENV, "4")))
        if not self._graph_capable():
            return [(self._encode_local, None)] * count
        workers = []
        for _ in range(count):
            graphs = MossReferenceEncoderGraphs(
                self._processor.audio_tokenizer,
                n_vq=self._n_vq,
                batch_sizes=FINE_BATCH_SIZES,
                bucket_seconds=FINE_BUCKET_SECONDS,
                max_batch_seconds=_SHARED_WORKER_BATCH_SECONDS,
                compile_core=self._compile_graphs(),
                batched_transfer=os.environ.get("VLLM_OMNI_MOSS_REF_BATCHED_TRANSFER", "0") == "1",
                singleton_bucket_seconds=self._singleton_bucket_seconds(),
            )
            workers.append(((lambda wavs, g=graphs: self._encode_local(wavs, g)), graphs.capture))
        return workers

    def _encode_local(self, prepared: list[torch.Tensor], graphs: MossReferenceEncoderGraphs | None = None) -> list:
        """One batched forward, retried per clip so one bad clip fails alone."""
        try:
            return list(self._encode_prepared(prepared, graphs))
        except Exception:  # noqa: BLE001 — batch forward failed; retry item-by-item
            logger.warning("MOSS ref batch encode (n=%d) failed; falling back to per-item", len(prepared))
        results: list = []
        for wav in prepared:
            try:
                results.append(self._encode_prepared([wav], graphs)[0])
            except Exception as exc:  # noqa: BLE001 — isolate this clip's failure
                results.append(exc)
        return results

    def _place_tokenizer_for_local_encode(self) -> None:
        """A client process keeps the tokenizer off the GPU until it must encode."""
        tokenizer = getattr(self._processor, "audio_tokenizer", None)
        device = getattr(self._processor, "_vllm_omni_ref_encoder_device", None)
        if tokenizer is not None and device is not None and next(tokenizer.parameters()).device != device:
            self._processor.audio_tokenizer = tokenizer.to(device)

    def _encode_prepared(
        self, prepared: list[torch.Tensor], graphs: MossReferenceEncoderGraphs | None = None
    ) -> list[torch.Tensor]:
        # ``prepared`` clips come from _encoder_input, at ``_clip_rate()``.
        graphs = graphs or self._reference_graphs()
        if graphs is None:
            with torch.no_grad():
                return self._processor.encode_audios_from_wav(
                    prepared, sampling_rate=self._clip_rate(), n_vq=self._n_vq
                )
        codes = graphs.encode([self._finish_prepare(wav) for wav in prepared])
        missing = [i for i, c in enumerate(codes) if c is None]
        if missing:
            with torch.no_grad():
                eager = self._processor.encode_audios_from_wav(
                    [prepared[i] for i in missing], sampling_rate=self._clip_rate(), n_vq=self._n_vq
                )
            for i, c in zip(missing, eager):
                codes[i] = c
        return codes

    def _graph_capable(self) -> bool:
        """True when the tokenizer can run graphs; installs windowed attention once."""
        if not self._graphs_enabled:
            return False
        tokenizer = getattr(self._processor, "audio_tokenizer", None)
        device = next(tokenizer.parameters()).device if tokenizer is not None else None
        if device is None or device.type != "cuda" or not hasattr(tokenizer, "_codec_inference_autocast"):
            self._graphs_enabled = False
            return False
        if not self._attention_installed and os.environ.get(_REF_ENCODE_ATTN_ENV, "flash") == "flash":
            logger.info(
                "MOSS reference encoder: %d attention layers use local-window flash attention",
                install_windowed_attention(
                    tokenizer,
                    fa_version=int(os.environ.get("VLLM_OMNI_MOSS_REF_FA_VERSION", "2")),
                    cache_rope=os.environ.get("VLLM_OMNI_MOSS_REF_CACHE_ROPE", "0") == "1",
                    skip_padded_query_mask=os.environ.get("VLLM_OMNI_MOSS_REF_SKIP_PADDED_QUERY_MASK", "0") == "1",
                ),
            )
        self._attention_installed = True
        return True

    @staticmethod
    def _compile_graphs() -> bool:
        return os.environ.get(_REF_ENCODE_COMPILE_ENV, "1") != "0"

    @staticmethod
    def _singleton_bucket_seconds() -> tuple[float, ...] | None:
        step_ms = int(os.environ.get("VLLM_OMNI_MOSS_REF_SINGLETON_BUCKET_MS", "0"))
        if step_ms == 0:
            return None
        if not 80 <= step_ms <= 1000:
            raise ValueError("MOSS reference singleton bucket step must be 80..1000 ms, or 0 to disable")
        return tuple(ms / 1000 for ms in range(3000, 10001, step_ms))

    def _reference_graphs(self) -> MossReferenceEncoderGraphs | None:
        """Capture on first use, on the encode worker thread that replays them."""
        if self._graphs is None:
            if not self._graph_capable():
                return None
            graphs = MossReferenceEncoderGraphs(
                self._processor.audio_tokenizer,
                n_vq=self._n_vq,
                compile_core=self._compile_graphs(),
                batched_transfer=os.environ.get("VLLM_OMNI_MOSS_REF_BATCHED_TRANSFER", "0") == "1",
                singleton_bucket_seconds=self._singleton_bucket_seconds(),
            )
            graphs.capture()
            self._graphs = graphs
        return self._graphs

    def _tokenizer_rate(self) -> int:
        return int(getattr(getattr(self._processor, "model_config", None), "sampling_rate", self._sr_target))

    def _encoder_input(self, wav: torch.Tensor) -> torch.Tensor:
        """A prepped clip at the tokenizer rate, one or two channels.

        This is the costly half of the processor's per-clip preparation; it runs
        in the process that received the request, even with a shared encoder.
        """
        import torchaudio

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 2:
            wav = wav[:2]
        if self._clip_rate() != self._sr_target:
            # Channels resample independently, so a mono clip is resampled
            # once before it is duplicated; the result is the same.
            wav = torchaudio.functional.resample(wav, self._sr_target, self._tokenizer_rate())
        return wav

    def _clip_rate(self) -> int:
        """Sample rate of the clips _encoder_input hands to the encoder."""
        return self._sr_target if self._gpu_prep else self._tokenizer_rate()

    def _finish_prepare(self, wav: torch.Tensor) -> torch.Tensor:
        """The rest of the processor's preparation: two channels, normalized loudness."""
        if self._gpu_prep:
            device = next(self._processor.audio_tokenizer.parameters()).device
            wav = wav.to(device)
            if self._clip_rate() != self._tokenizer_rate():
                wav = self._gpu_resampler(device)(wav)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        return self._processor.loudness_normalize(wav)

    def _gpu_resampler(self, device: torch.device) -> torch.nn.Module:
        resampler = self._gpu_resamplers.get(device)
        if resampler is None:
            import torchaudio

            # The same windowed-sinc filter as torchaudio.functional.resample's defaults.
            resampler = torchaudio.transforms.Resample(self._sr_target, self._tokenizer_rate()).to(device)
            self._gpu_resamplers[device] = resampler
        return resampler

    def _processor_prepare(self, wav: torch.Tensor) -> torch.Tensor:
        """Per-clip preparation of the processor's ``encode_audios_from_wav``."""
        return self._finish_prepare(self._encoder_input(wav))

    async def aclose(self) -> None:
        """Release the batcher drainer + any in-flight encodes (tests/shutdown)."""
        for task in list(self._inflight.values()):
            task.cancel()
        self._inflight.clear()
        await self._batcher.aclose()


class MossRealtimeReferenceEncoder(MossReferenceEncoder):
    """Use the realtime codec with the shared reference cache and batcher."""

    def _encoder_input(self, wav: torch.Tensor) -> torch.Tensor:
        return wav  # the realtime codec takes the clip at the working rate

    def _encode_prepared(self, prepared: list[torch.Tensor], graphs: Any = None) -> list[torch.Tensor]:
        with torch.no_grad():
            encoded = self._processor.batch_encode([wav.squeeze(0) for wav in prepared], num_quantizers=self._n_vq)
        return [
            encoded.audio_codes[:, i, : int(length.item())].transpose(0, 1).contiguous()
            for i, length in enumerate(encoded.audio_codes_lengths)
        ]


def build_reference_encoder(
    processor: Any,
    *,
    variant: str,
    speaker_cache: Any,
    server_dir: str | None = None,
) -> MossReferenceEncoder:
    """Build the per-server encoder for a MOSS-TTS ``variant``.

    Derives the encode geometry (``n_vq`` and the working sample rate) from the
    upstream processor's ``model_config``; Realtime takes its codec directly
    and uses 16 codebooks at 24 kHz. Variant knowledge stays in the model package.
    """
    if variant == "realtime":
        return MossRealtimeReferenceEncoder(
            processor,
            variant=variant,
            n_vq=16,
            sr_target=24000,
            speaker_cache=speaker_cache,
            server_dir=server_dir,
        )
    n_vq = int(getattr(processor.model_config, "n_vq", 32))
    # Local-v1.5 encodes reference audio at a fixed 24 kHz working rate
    # regardless of its 48 kHz stereo *output* codec -- mirrors the offline
    # example's hardcoded encode_audios_from_wav(sampling_rate=24000) for this
    # variant; proc.model_config.sampling_rate there is the output rate
    # (48000), the wrong value to resample the reference into.
    sr_target = 24000 if variant == "local" else int(getattr(processor.model_config, "sampling_rate", 24000))
    return MossReferenceEncoder(
        processor,
        variant=variant,
        n_vq=n_vq,
        sr_target=sr_target,
        speaker_cache=speaker_cache,
        server_dir=server_dir,
    )


async def encode_request_references(
    encoder: MossReferenceEncoder,
    ref_audio: str,
    ref_audio_2: str | None = None,
    *,
    resolve_ref_audio: Callable[[str], Awaitable[tuple[_Waveform, int, str]]],
    get_artifact_key: Callable[[str], str | None],
    voice_name: str | None = None,
    voice_created_at: int = 0,
) -> tuple[list[torch.Tensor], dict[int, str]]:
    """Encode a request's reference clip(s) into MOSS RVQ code tensors.

    ``ref_audio_2`` is the TTSD second speaker; pass ``None`` for the
    single-speaker variants. The named-voice cache key is
    ``(voice_name, created_at)`` and ignores the clip content, so only slot 0
    (the reference that actually belongs to the uploaded voice) may use it —
    speaker 2 is a different clip and stays content-addressed, or it would
    silently reuse speaker 1's codes.

    Returns ``(codes_per_speaker, resolve_keys)`` where ``resolve_keys`` maps
    the reference slot (0 = ``ref_audio``, 1 = ``ref_audio_2``) to its
    content-aware resolve key, for salting the KV prefix cache. A dict rather
    than an append list because the two-speaker encodes run concurrently, so
    completion order is not slot order.
    """
    resolve_keys: dict[int, str] = {}

    async def encode_one(ref_str: str, *, named_voice: bool, slot: int) -> torch.Tensor:
        codes, resolve_key = await encoder.encode(
            ref_str,
            resolve_ref_audio=resolve_ref_audio,
            get_artifact_key=get_artifact_key,
            voice_name=voice_name if named_voice else None,
            voice_created_at=voice_created_at if named_voice else 0,
        )
        if resolve_key is not None:
            resolve_keys[slot] = resolve_key
        return codes

    if ref_audio_2:
        # Encode both speakers concurrently so they land in the same batch
        # window / share single-flight instead of serializing.
        refs = list(
            await asyncio.gather(
                encode_one(ref_audio, named_voice=True, slot=0),
                encode_one(ref_audio_2, named_voice=False, slot=1),
            )
        )
    else:
        refs = [await encode_one(ref_audio, named_voice=True, slot=0)]
    return refs, resolve_keys
