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

Kept import-light (only ``asyncio`` / ``hashlib`` / ``torch`` plus the logger)
so importing it from the API-server process does not pull the talker/codec.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Coalescing defaults. Kept as module constants (not env/CLI) to match the
# ``_REF_AUDIO_RESOLVE_CACHE_MAX_*`` convention in serving_speech.py.
_REF_ENCODE_BATCH_WINDOW_MS = 10.0
_REF_ENCODE_MAX_BATCH = 8

_INT32_MAX = 2**31


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def _prep_wav_sync(wav_list: list, sr: int, sr_target: int) -> torch.Tensor:
    """Tensor-ise + resample one clip to ``sr_target`` (the blocking prep)."""
    wav = torch.tensor(wav_list, dtype=torch.float32)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != sr_target:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, sr_target)
    return wav


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
        encode_batch_fn: Callable[[list[tuple[list, int]]], list],
        *,
        window_ms: float,
        max_batch: int,
    ):
        # encode_batch_fn: sync, takes [(wav_list, sr), ...], returns a list of
        # (codes_tensor | Exception) aligned to the input order.
        self._encode_batch_fn = encode_batch_fn
        self._window_s = max(0.0, float(window_ms) / 1000.0)
        self._max_batch = max(1, int(max_batch))
        self._queue: asyncio.Queue | None = None
        self._drainer: asyncio.Task | None = None

    def _ensure_started(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._drainer is None or self._drainer.done():
            self._drainer = asyncio.create_task(self._drain_loop())

    async def submit(self, wav_list: list, sr: int) -> torch.Tensor:
        self._ensure_started()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((wav_list, sr, fut))  # type: ignore[union-attr]
        return await fut

    async def _drain_loop(self) -> None:
        assert self._queue is not None
        while True:
            first = await self._queue.get()
            jobs = await self._coalesce(first)
            await self._run_batch(jobs)

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

    async def _run_batch(self, jobs: list[tuple[list, int, asyncio.Future]]) -> None:
        payload = [(wav_list, sr) for wav_list, sr, _ in jobs]
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
        window_ms: float = _REF_ENCODE_BATCH_WINDOW_MS,
        max_batch: int = _REF_ENCODE_MAX_BATCH,
    ):
        self._processor = processor
        self._n_vq = int(n_vq)
        self._sr_target = int(sr_target)
        self._speaker_cache = speaker_cache
        # ``created_at`` and the audio-content name vary per request; the
        # model_type namespaces the whole family so a moss_tts server never
        # collides with another model's speaker-cache entries.
        self._model_type = f"moss_tts_{variant}_nq{int(n_vq)}"
        self._inflight: dict[str, asyncio.Task] = {}
        self._batcher = _RefEncodeBatcher(self._encode_batch_sync, window_ms=window_ms, max_batch=max_batch)

    def _make_cache_key(self, name: str, created_at: int) -> tuple:
        return self._speaker_cache.make_cache_key(name, model_type=self._model_type, created_at=int(created_at))

    async def encode(
        self,
        ref_str: str,
        *,
        resolve_ref_audio: Callable[[str], Awaitable[tuple[list, int, str]]],
        get_artifact_key: Callable[[str], str | None],
        voice_name: str | None = None,
        voice_created_at: int = 0,
    ) -> tuple[torch.Tensor, str | None]:
        """Encode one reference clip into MOSS RVQ codes, reusing the cache.

        ``resolve_ref_audio`` maps ``ref_str`` to ``(wav_list, sr, cache_key)``
        where *cache_key* is the content-aware resolve key (it folds mtime/size
        for local files). ``get_artifact_key`` maps that resolve key to the
        waveform-content artifact key, or ``None`` when unknown.

        Returns ``(codes, resolve_cache_key)``; the key is ``None`` when the
        clip was served from the named-voice cache without resolving (the
        caller salts those requests with ``voice_created_at`` instead).
        """
        voice_name, created_at = _registered_voice(voice_name, voice_created_at)

        if voice_name:
            # A named voice has a stable key that does not depend on the
            # resolved audio, so the cache can be checked before resolving.
            flight_key = f"voice:{voice_name}:{created_at}"
            cached = self._speaker_cache.get(self._make_cache_key(voice_name, created_at))
            if cached is not None:
                return _clone_out(cached["codes"]), None
        else:
            # Anonymous refs have no pre-resolve hot path: the content key must
            # come from the resolve itself (mtime/size from a single stat) so an
            # on-disk edit invalidates the cached codes; the flight body
            # re-checks the speaker cache right after resolving, which is cheap
            # when the resolve cache is warm. The flight key is the request-side
            # reference (not the content hash), so concurrent requests for the
            # same ref_str also share the resolve/download, not just the encode.
            flight_key = "ref:" + _sha1(ref_str)

        codes, resolve_key = await self._single_flight(
            flight_key,
            lambda: self._resolve_and_encode(ref_str, resolve_ref_audio, get_artifact_key, voice_name, created_at),
        )
        return _clone_out(codes), resolve_key

    async def _single_flight(
        self,
        flight_key: str,
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
        def _retire(t: asyncio.Task, key: str = flight_key) -> None:
            if self._inflight.get(key) is t:
                self._inflight.pop(key, None)

        task.add_done_callback(_retire)
        return await asyncio.shield(task)

    async def _resolve_and_encode(
        self,
        ref_str: str,
        resolve_ref_audio: Callable[[str], Awaitable[tuple[list, int, str]]],
        get_artifact_key: Callable[[str], str | None],
        voice_name: str | None,
        created_at: int,
    ) -> tuple[torch.Tensor, str]:
        """Flight body: resolve → re-check cache by content hash → batch-encode."""
        wav_list, sr, resolve_key = await resolve_ref_audio(ref_str)

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

        codes = await self._batcher.submit(wav_list, sr)
        compact = _to_compact_codes(codes)
        self._speaker_cache.put(key, {"codes": compact})
        logger.debug(
            "MOSS ref encode STORE key=%s shape=%s dtype=%s",
            key,
            tuple(compact.shape),
            compact.dtype,
        )
        return compact, resolve_key

    def _encode_batch_sync(self, payload: list[tuple[list, int]]) -> list:
        """Worker-thread body: prep each clip, then one batched forward."""
        n = len(payload)
        results: list = [None] * n
        prepared: list[torch.Tensor] = []
        prepared_idx: list[int] = []
        for i, (wav_list, sr) in enumerate(payload):
            try:
                prepared.append(_prep_wav_sync(wav_list, sr, self._sr_target))
                prepared_idx.append(i)
            except Exception as exc:  # noqa: BLE001 — isolate this clip's failure
                results[i] = exc
        if not prepared:
            return results

        try:
            codes_list = self._encode_prepared(prepared)
            for local_i, orig_i in enumerate(prepared_idx):
                results[orig_i] = codes_list[local_i]
        except Exception:  # noqa: BLE001 — batch forward failed; retry item-by-item
            logger.warning("MOSS ref batch encode (n=%d) failed; falling back to per-item", len(prepared))
            for local_i, orig_i in enumerate(prepared_idx):
                try:
                    results[orig_i] = self._encode_prepared([prepared[local_i]])[0]
                except Exception as exc:  # noqa: BLE001 — isolate this clip's failure
                    results[orig_i] = exc
        return results

    def _encode_prepared(self, prepared: list[torch.Tensor]) -> list[torch.Tensor]:
        with torch.no_grad():
            return self._processor.encode_audios_from_wav(prepared, sampling_rate=self._sr_target, n_vq=self._n_vq)

    async def aclose(self) -> None:
        """Release the batcher drainer + any in-flight encodes (tests/shutdown)."""
        for task in list(self._inflight.values()):
            task.cancel()
        self._inflight.clear()
        await self._batcher.aclose()


def build_reference_encoder(
    processor: Any,
    *,
    variant: str,
    speaker_cache: Any,
) -> MossReferenceEncoder:
    """Build the per-server encoder for a MOSS-TTS ``variant``.

    Derives the encode geometry (``n_vq`` and the working sample rate) from the
    upstream processor's ``model_config`` so the variant knowledge stays in the
    model package rather than in ``serving_speech.py``.
    """
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
    )


async def encode_request_references(
    encoder: MossReferenceEncoder,
    ref_audio: str,
    ref_audio_2: str | None = None,
    *,
    resolve_ref_audio: Callable[[str], Awaitable[tuple[list, int, str]]],
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
