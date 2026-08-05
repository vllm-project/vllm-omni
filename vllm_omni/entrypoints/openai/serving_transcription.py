# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``/v1/audio/transcriptions`` with word-level alignment.

Upstream vLLM already defines the whole OpenAI verbose surface --
``TranscriptionResponseVerbose.words``, ``TranscriptionWord`` and the
``timestamp_granularities[]`` field -- but nothing ever populates ``words``.
This subclass fills that gap with the shared Qwen3 forced aligner that #4034
landed for streaming TTS, pointed at the *input* audio and the ASR hypothesis
instead of generated audio and its source text.

Alignment runs only when the caller asks for ``word`` granularity, so the
default transcription path is unchanged and pays nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import multiprocessing
import threading
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranscriptionWord,
)
from vllm.entrypoints.speech_to_text.transcription.serving import OpenAIServingTranscription
from vllm.logger import init_logger

from vllm_omni.model_executor.stage_input_processors.qwen3_asr_align import (
    ALIGNER_MODEL,
    ALIGNER_STAGE_NAME,
    attach_aligner_audio,
)
from vllm_omni.utils.forced_aligner import ForcedAlignerConfig, ForcedAlignerLoadError, _decode_timestamps
from vllm_omni.utils.forced_aligner import align as forced_align

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastapi import Request
    from vllm.entrypoints.openai.protocol import ErrorResponse

    TranscriptionResult = (
        TranscriptionResponse | TranscriptionResponseVerbose | AsyncGenerator[str, None] | ErrorResponse
    )

logger = init_logger(__name__)

#: The Qwen3 forced aligner consumes 16 kHz mono audio.
ALIGNER_SAMPLE_RATE = 16000

#: How often to log decode-reuse hit/miss stats.
_REUSE_STATS_EVERY = 50

#: Per-request collector for the aligner stage's pooling results. Set by the
#: request coroutine, appended to by the output filter running in the same
#: task, so no request-id bookkeeping is needed to pair them.
_pooling_sink: ContextVar[list[Any] | None] = ContextVar("_omni_aligner_pooling_sink", default=None)

#: Recycle a decode worker after this many clips, to bound any per-decode leak
#: in the audio stack. High enough that respawn cost is noise.
_DECODE_WORKER_MAX_TASKS = 512


def _float32_to_int16_pcm(pcm: np.ndarray) -> bytes:
    return (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _to_int16_pcm(audio_data: bytes, target_sr: int = ALIGNER_SAMPLE_RATE) -> bytes:
    """Decode an uploaded audio file to signed-int16 LE mono PCM at ``target_sr``.

    ``align()`` takes raw PCM bytes, but the endpoint receives whatever
    container the client uploaded (mp3, wav, flac, ...).
    """
    import soundfile

    with io.BytesIO(audio_data) as buf:
        pcm, sr = soundfile.read(buf, dtype="float32", always_2d=False)

    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)

    if sr != target_sr:
        # Linear resample. The aligner quantises to a coarse timestamp grid
        # (tens of ms), so interpolation error here is well below its
        # resolution and a higher-quality resampler would not change results.
        new_len = int(round(len(pcm) * (target_sr / float(sr))))
        pcm = np.interp(
            np.linspace(0, len(pcm), new_len, endpoint=False),
            np.arange(len(pcm)),
            pcm,
        ).astype(np.float32)

    return _float32_to_int16_pcm(pcm)


def _decode_audio_worker(audio_data: bytes, sample_rate: int, max_duration_s: float | None):
    """Decode + resample in a worker *process*, dodging the GIL.

    Module-level so it is picklable for a ProcessPoolExecutor. Returns the
    waveform; chunking stays in the parent because it is cheap and its policy
    lives on the serving object.
    """
    from vllm.multimodal.media.audio import load_audio

    with io.BytesIO(audio_data) as buf:
        return load_audio(buf, sr=sample_rate, max_duration_s=max_duration_s)


def _audio_key(audio_data: bytes) -> bytes:
    """Content key linking a decode to the request that asked for it.

    ~30 us on a 500 KB upload, against the ~75 ms decode it lets us skip.
    """
    return hashlib.blake2b(audio_data, digest_size=16).digest()


def _duration_seconds(audio_data: bytes) -> float:
    """Wall duration of an uploaded audio file, for the verbose envelope."""
    import soundfile

    with io.BytesIO(audio_data) as buf:
        info = soundfile.info(buf)
    return float(info.duration)


class _AlignerOutputFilter:
    """Keeps the aligner stage's pooling result out of the transcript stream.

    The ASR stage and the aligner stage are both declared final outputs, so the
    engine emits two results per request. Upstream's transcription loop reads
    ``outputs[0].text`` off whatever arrives and has no notion of a second,
    differently-typed terminal output -- there is no mechanism yet for a stage
    whose result goes to the client instead of downstream (RFC #4468). Splitting
    the stream here keeps that concern in the entrypoint: pooling results are
    captured for the alignment decode, everything else passes through untouched.

    Delegates everything else to the real client, so the generative paths that
    share this engine are unaffected.
    """

    def __init__(self, inner: Any, sink: Any) -> None:
        self._inner = inner
        self._sink = sink

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        stream = self._inner.generate(*args, **kwargs)

        async def _filtered() -> Any:
            async for out in stream:
                outputs = getattr(out, "outputs", None)
                first = outputs[0] if outputs else None
                # A pooling result carries data, not text. Duck-typing rather
                # than importing PoolingOutput keeps this tolerant of the
                # several shapes the omni layer can hand back.
                if first is not None and not hasattr(first, "text") and hasattr(first, "data"):
                    # Whole output, not just the pooling payload: the decode
                    # needs prompt_token_ids to locate the timestamp markers.
                    self._sink(out)
                    continue
                yield out

        return _filtered()


class OmniServingTranscription(OpenAIServingTranscription):
    """Upstream transcription serving plus optional forced alignment."""

    #: Cap on in-flight decoded-audio handoffs, which bounds host memory: each
    #: entry is one decoded waveform (~1.9 MB for a 30s 16 kHz clip), so 1024
    #: is ~2 GB worst case.
    #:
    #: This has to exceed peak request concurrency, not just cover leaks. Every
    #: in-flight request holds an entry between its decode and its alignment,
    #: so a cap below concurrency evicts live handoffs and silently sends those
    #: requests back to decoding the upload a second time -- measured as a 6%
    #: miss rate at concurrency 384 when this was 256.
    _MAX_PENDING_DECODES = 1024

    def __init__(
        self,
        *args: Any,
        forced_aligner_config: ForcedAlignerConfig | None = None,
        audio_decode_procs: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.forced_aligner_config = forced_aligner_config
        self._decoded: OrderedDict[bytes, list[tuple[np.ndarray, float]]] = OrderedDict()
        self._pending = 0
        self._decoded_lock = threading.Lock()
        self._reuse_hits = 0
        self._reuse_misses = 0
        self._aligner_stage_cache: bool | None = None
        self._pooling_results: OrderedDict[str, Any] = OrderedDict()
        self._aligned_count = 0
        self._decode_constants: tuple[int, int, float] | tuple[()] | None = None
        if self._has_aligner_stage:
            self.engine_client = _AlignerOutputFilter(self.engine_client, self._capture_pooling_output)

        # Audio decode is GIL-bound: ~40 decodes/s for a 30s 48 kHz stereo mp3
        # no matter how many *threads* upstream's preprocess pool gets (it
        # plateaus past ~4), which lands below what the GPU can consume.
        # Processes escape the GIL; the decoded array comes back over IPC
        # (~0.4 ms for 30s @ 16 kHz float32), cheap next to the ~75 ms saved.
        # Measured 19.9 -> 52.6 req/s at concurrency 384 with this on.
        #
        # Opt-in: 0 keeps upstream's thread pool. Under --omni this is the only
        # lever available, since vllm-omni rejects --api-server-count.
        self._decode_pool: ProcessPoolExecutor | None = None
        if audio_decode_procs > 0:
            # forkserver, not the default fork: this process has CUDA
            # initialized (the in-process aligner lives here), and forking a
            # CUDA-initialized process leaves the child with an unusable
            # context. The workers touch no CUDA today, so fork happens to
            # survive, but it is not a property worth depending on.
            #
            # max_tasks_per_child bounds any per-decode leak in the audio
            # stack; the pool is built here rather than on first use so a
            # misconfiguration fails at startup.
            self._decode_pool = ProcessPoolExecutor(
                max_workers=audio_decode_procs,
                mp_context=multiprocessing.get_context("forkserver"),
                max_tasks_per_child=_DECODE_WORKER_MAX_TASKS,
            )
            self._decode_and_chunk_speech_async = self._decode_and_chunk_speech_in_proc
            logger.info("Audio decode: %d worker processes (forkserver)", audio_decode_procs)

    async def _decode_and_chunk_speech_in_proc(self, audio_data: bytes) -> tuple[list[np.ndarray], float]:
        """Process-pool replacement for upstream's thread-pool decode.

        Mirrors ``SpeechToTextBaseServing._decode_and_chunk_speech``: only the
        decode moves to a subprocess; chunking is numpy-cheap and its policy
        lives on ``self.asr_config``.
        """
        from vllm.multimodal.audio import get_audio_duration, split_audio

        loop = asyncio.get_running_loop()
        try:
            y, sr = await loop.run_in_executor(
                self._decode_pool,
                _decode_audio_worker,
                audio_data,
                self.asr_config.sample_rate,
                self.max_audio_decode_duration_s,
            )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Invalid or unsupported audio file.") from exc

        duration = get_audio_duration(y=y, sr=sr)
        do_split = self.asr_config.allow_audio_chunking and (
            self.asr_config.max_audio_clip_s is not None and duration > self.asr_config.max_audio_clip_s
        )
        if not do_split:
            chunks = [y]
        else:
            chunks = split_audio(
                audio_data=y,
                sample_rate=int(sr),
                max_clip_duration_s=self.asr_config.max_audio_clip_s,
                overlap_duration_s=self.asr_config.overlap_chunk_second,
                min_energy_window_size=self.asr_config.min_energy_split_window_size,
            )

        self._stash_decoded(audio_data, chunks, duration)
        return chunks, duration

    def shutdown(self) -> None:
        if self._decode_pool is not None:
            self._decode_pool.shutdown(wait=False)
        super().shutdown()

    def _decode_and_chunk_speech(self, audio_data: bytes) -> tuple[list[np.ndarray], float]:
        """Stash the decoded waveform so alignment does not decode it again.

        Decoding a 30s 48 kHz stereo mp3 costs ~75 ms and caps out around 40
        decodes/s across the whole process (GIL-bound past ~4 threads), so
        paying it twice per request roughly halves the ceiling of the
        alignment path. Upstream already resamples to the model sample rate,
        which is exactly what the aligner wants.

        Runs on upstream's preprocess executor, not the event loop.
        """
        chunks, duration = super()._decode_and_chunk_speech(audio_data)

        # Only the unchunked case is reusable: chunks overlap by design, so
        # rejoining them would not reconstruct the original waveform.
        self._stash_decoded(audio_data, chunks, duration)
        return chunks, duration

    def _stash_decoded(self, audio_data: bytes, chunks: list[np.ndarray], duration: float) -> None:
        """Hand the decoded waveform to this request's alignment pass.

        A *list* per key, not a single slot: the key is a content hash, so
        concurrent requests carrying identical audio (retries, replayed
        fixtures, the same file submitted twice) collide. With one slot each
        decode clobbers the last and only one consumer can pop it -- measured
        65% miss under load, i.e. two thirds of requests decoding twice.
        Queueing one entry per decode makes hits track decodes exactly.
        """
        if not self._aligner_enabled or len(chunks) != 1:
            return
        with self._decoded_lock:
            self._decoded.setdefault(_audio_key(audio_data), []).append((chunks[0], duration))
            self._pending += 1
            # Bound only against leakage (a request erroring between decode and
            # pickup); drop oldest first.
            while self._pending > self._MAX_PENDING_DECODES:
                oldest_key = next(iter(self._decoded))
                self._decoded[oldest_key].pop(0)
                self._pending -= 1
                if not self._decoded[oldest_key]:
                    del self._decoded[oldest_key]

    def _take_decoded(self, audio_data: bytes) -> tuple[np.ndarray, float] | None:
        with self._decoded_lock:
            key = _audio_key(audio_data)
            queued = self._decoded.get(key)
            hit = None
            if queued:
                hit = queued.pop(0)
                self._pending -= 1
                if not queued:
                    del self._decoded[key]
            self._reuse_hits += hit is not None
            self._reuse_misses += hit is None
            total = self._reuse_hits + self._reuse_misses
            if total % _REUSE_STATS_EVERY == 0:
                # A miss means that request decodes the upload a second time.
                # Should be ~0%; anything else means the handoff is leaking and
                # the alignment path is paying full decode cost.
                logger.info(
                    "Alignment decode reuse: %d hits, %d misses (%.1f%% miss), pending=%d",
                    self._reuse_hits,
                    self._reuse_misses,
                    100.0 * self._reuse_misses / total,
                    self._pending,
                )
        return hit

    @property
    def _has_aligner_stage(self) -> bool:
        """Whether the running pipeline includes a forced-aligner stage.

        Read off the engine's own stage metadata rather than a flag, so the
        topology stays the single source of truth: deploying ``qwen3_asr``
        instead of ``qwen3_asr_align`` turns this off with no other change.
        """
        if self._aligner_stage_cache is None:
            # engine_client is the AsyncOmni facade; the stage metadata lives on
            # the AsyncOmniEngine it wraps.
            meta: Any = None
            for holder in (self.engine_client, getattr(self.engine_client, "engine", None)):
                meta = getattr(holder, "stage_metadata", None)
                if meta:
                    break
            self._aligner_stage_cache = any(getattr(m, "model_stage", None) == ALIGNER_STAGE_NAME for m in meta or [])
            logger.info("Forced-aligner stage detected: %s", self._aligner_stage_cache)
        return self._aligner_stage_cache

    def _capture_pooling_output(self, output: Any) -> None:
        """Route the aligner stage's result to the request that is waiting for it.

        A context-local sink rather than a request-id lookup: the engine's id
        for the aligner stage's result is not the one this method could match
        against, and the sink is set by the same task that will consume it, so
        the handoff cannot pair the wrong request with the wrong logits.
        """
        with self._decoded_lock:
            self._aligned_count += 1
            if self._aligned_count % _REUSE_STATS_EVERY == 0:
                logger.info("Aligner stage: %d requests aligned", self._aligned_count)
        sink = _pooling_sink.get()
        if sink is not None:
            sink.append(output)

    @property
    def _aligner_enabled(self) -> bool:
        """Whether anything downstream still needs the decoded waveform."""
        return self.forced_aligner_config is not None or self._has_aligner_stage

    def _peek_decoded(self, audio_data: bytes) -> tuple[np.ndarray, float] | None:
        """Read this request's decoded waveform without consuming the handoff.

        The stage path reads the waveform while building the prompt but the
        entry is still reclaimed later by ``_take_decoded``, so the accounting
        stays identical to the sidecar's. Peeking the head entry is safe even
        when concurrent requests collide on the key: the key is a content hash,
        so every queued entry under it holds the same audio.
        """
        with self._decoded_lock:
            queued = self._decoded.get(_audio_key(audio_data))
            return queued[0] if queued else None

    async def _preprocess_speech_to_text(
        self,
        request: TranscriptionRequest,
        audio_data: bytes,
        request_id: str,
    ) -> tuple[list[Any], float]:
        """Carry the decoded waveform to the aligner stage on the prompt itself.

        By the time a prompt reaches a downstream stage the audio has been
        turned into processed features (``mm_kwargs``); the raw waveform the
        aligner needs is gone, and the framework's default stage input
        processor looks for a ``multi_modal_data`` key that the transcription
        entrypoint never puts there. Attaching the waveform that stage 0's
        decode already produced keeps the pipeline to one decode per request
        without a second lookup path.
        """
        engine_inputs, duration = await super()._preprocess_speech_to_text(request, audio_data, request_id)
        if self._has_aligner_stage:
            peeked = self._peek_decoded(audio_data)
            if peeked is not None:
                for engine_input in engine_inputs:
                    attach_aligner_audio(engine_input, peeked[0], self.asr_config.sample_rate)
            else:
                # Chunked audio has no single reusable waveform; the stage
                # degrades to a transcript without words rather than failing.
                logger.debug("No reusable decode for %s; aligner stage will skip", request_id)
        return engine_inputs, duration

    def _aligner_decode_constants(self) -> tuple[int, int, float] | None:
        """``(classify_num, timestamp_token_id, timestamp_segment_time_ms)``.

        Read from the aligner checkpoint's config rather than hard-coded: the
        marker grid is a property of the weights, and a mismatch would silently
        shift every timestamp rather than fail.
        """
        if self._decode_constants is None:
            try:
                from transformers import AutoConfig

                cfg = AutoConfig.from_pretrained(ALIGNER_MODEL, trust_remote_code=True)
                thinker = getattr(cfg, "thinker_config", None)
                self._decode_constants = (
                    int(thinker.classify_num),
                    int(cfg.timestamp_token_id),
                    float(cfg.timestamp_segment_time),
                )
            except Exception:  # noqa: BLE001
                logger.exception("Could not read aligner decode constants; word timestamps disabled")
                self._decode_constants = ()
        return self._decode_constants or None

    def _words_from_stage(
        self,
        result: TranscriptionResponseVerbose,
        request: TranscriptionRequest,
        text: str,
        duration_ms: float,
    ) -> TranscriptionResult:
        """Decode the aligner stage's logits into OpenAI ``words``.

        Degrades to a transcript without words rather than failing a request
        whose text is already correct; a missing alignment is worth strictly
        less than the transcript it would otherwise take down.
        """
        from vllm_omni.utils.qwen3_force_align_processor import segment_words

        sink = _pooling_sink.get() or []
        if not sink:
            logger.warning("Aligner stage produced no result for this request; returning transcript without words")
            return result

        constants = self._aligner_decode_constants()
        if constants is None:
            return result
        classify_num, timestamp_token_id, segment_time_ms = constants

        output = sink[0]
        prompt_token_ids = getattr(output, "prompt_token_ids", None) or []
        positions = [i for i, tid in enumerate(prompt_token_ids) if tid == timestamp_token_id]
        if not positions:
            logger.warning("No timestamp markers in the aligner prompt; returning transcript without words")
            return result

        # Re-segment the same post-processed transcript the stage segmented, so
        # the word list and the marker pairs line up. Both sides call this one
        # function on this one string, which is what keeps them in step.
        words = segment_words(text, getattr(request, "language", None))
        try:
            timestamps = _decode_timestamps(
                logits=output.outputs[0].data,
                words=words,
                timestamp_positions=positions,
                classify_num=classify_num,
                audio_duration_ms=duration_ms,
                timestamp_segment_time_ms=segment_time_ms,
            )
        except (ValueError, IndexError, RuntimeError):
            logger.exception("word alignment decode failed; returning transcript without words")
            return result

        result.words = [
            TranscriptionWord(word=t.word, start=t.start_ms / 1000.0, end=t.end_ms / 1000.0) for t in timestamps
        ]
        return result

    @staticmethod
    def _wants_word_timestamps(request: TranscriptionRequest) -> bool:
        return "word" in (request.timestamp_granularities or []) and request.response_format == "verbose_json"

    async def create_transcription(
        self,
        audio_data: bytes,
        request: TranscriptionRequest,
        raw_request: Request | None = None,
    ) -> TranscriptionResult:
        wants_words = self._wants_word_timestamps(request)

        # Upstream rejects verbose_json outright unless the model implements
        # segment timestamps. Word alignment does not need that: the aligner
        # derives timings from the audio, not from the model's timestamp
        # tokens. So for a word-only request against such a model, run the
        # inner call as plain json and build the verbose envelope here.
        synthesize_verbose = wants_words and not self.model_cls.supports_segment_timestamp
        inner_request = request.model_copy(update={"response_format": "json"}) if synthesize_verbose else request

        # finally, not a trailing statement: the handoff must be reclaimed even
        # when the inner call raises, or a failing request parks its waveform
        # until the size cap evicts it.
        decoded = None
        # Fresh per-request collector for the aligner stage's output; the
        # filter appends into whichever sink is current when it runs. No reset:
        # each request handler runs in its own task context, so the binding dies
        # with the request rather than leaking into the next one.
        if self._has_aligner_stage:
            _pooling_sink.set([])
        try:
            result = await super().create_transcription(
                audio_data=audio_data,
                request=inner_request,
                raw_request=raw_request,
            )
        finally:
            if self._aligner_enabled:
                decoded = self._take_decoded(audio_data)

        if not wants_words:
            return result
        # An error response from the inner call passes straight through.
        if synthesize_verbose and isinstance(result, TranscriptionResponse):
            duration = decoded[1] if decoded is not None else await asyncio.to_thread(_duration_seconds, audio_data)
            result = TranscriptionResponseVerbose(
                # Qwen3-ASR auto-detects and does not report the language back,
                # so echo the request hint and fall back to "auto".
                language=getattr(request, "language", None) or "auto",
                text=result.text,
                duration=str(duration),
                segments=None,
            )
        # Streaming responses are async generators; aligning them needs a
        # per-sentence hook rather than a whole-response pass. Left for the
        # streaming surface.
        if not isinstance(result, TranscriptionResponseVerbose):
            return result
        text = (result.text or "").strip()
        if not text:
            result.words = []
            return result

        if self._has_aligner_stage:
            # The aligner already ran inside the pipeline; its logits are
            # waiting, so this is a decode rather than a second inference pass.
            duration_ms = (decoded[1] if decoded is not None else float(result.duration or 0.0)) * 1000.0
            return self._words_from_stage(result, request, text, duration_ms)

        if self.forced_aligner_config is None:
            logger.warning(
                "word timestamps requested but no forced aligner configured; pass --forced-aligner to enable them."
            )
            return result

        try:
            if decoded is not None:
                # Fast path: reuse the waveform the ASR pass already decoded.
                pcm = await asyncio.to_thread(_float32_to_int16_pcm, decoded[0])
                sample_rate = self.asr_config.sample_rate
            else:
                # Chunked audio, or the decode handoff missed. Decode our own
                # copy, off-thread so it cannot block the loop driving every
                # other in-flight request.
                pcm = await asyncio.to_thread(_to_int16_pcm, audio_data)
                sample_rate = ALIGNER_SAMPLE_RATE
        except (OSError, ValueError, RuntimeError):
            # Bad/unsupported container, or a resample that could not be
            # completed. Degrade to a transcript without words rather than
            # failing a request whose text is already correct. Deliberately
            # narrow: anything else here is a bug and should surface.
            logger.exception("word alignment: could not decode audio")
            return result

        try:
            timestamps = await forced_align(
                audio=pcm,
                text=text,
                sample_rate=sample_rate,
                config=self.forced_aligner_config,
                language=getattr(request, "language", None),
            )
        except ForcedAlignerLoadError:
            # Permanent until restart; surface rather than silently degrading
            # every subsequent request to a response with no words.
            raise

        if timestamps is None:
            logger.warning("word alignment failed for this request; returning transcript without words")
            return result

        result.words = [
            TranscriptionWord(word=t.word, start=t.start_ms / 1000.0, end=t.end_ms / 1000.0) for t in timestamps
        ]
        return result
