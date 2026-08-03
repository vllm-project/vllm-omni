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
import os
import threading
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
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

from vllm_omni.utils.forced_aligner import ForcedAlignerConfig, ForcedAlignerLoadError
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


class OmniServingTranscription(OpenAIServingTranscription):
    """Upstream transcription serving plus optional forced alignment."""

    #: Cap on in-flight decoded-audio handoffs. Entries are popped by the
    #: request that produced them; this only bounds leakage if one errors out
    #: between decode and pickup.
    _MAX_PENDING_DECODES = 256

    def __init__(self, *args: Any, forced_aligner_config: ForcedAlignerConfig | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.forced_aligner_config = forced_aligner_config
        self._decoded: OrderedDict[bytes, list[tuple[np.ndarray, float]]] = OrderedDict()
        self._pending = 0
        self._decoded_lock = threading.Lock()
        self._reuse_hits = 0
        self._reuse_misses = 0

        # Audio decode is GIL-bound: measured ~40 decodes/s for a 30s 48 kHz
        # stereo mp3 no matter how many *threads* upstream's preprocess pool
        # gets (it plateaus past ~4). That ceiling sits below what the GPU can
        # consume, so the server ends up decode-bound rather than GPU-bound.
        # Processes escape the GIL; the decoded array comes back over IPC
        # (~1.9 MB for 30s @ 16 kHz float32), which is cheap next to the ~75 ms
        # of CPU it saves.
        #
        # Opt-in: 0 keeps upstream's thread pool. Under --omni this is the only
        # lever available, since vllm-omni rejects --api-server-count.
        n_procs = int(os.getenv("VLLM_OMNI_AUDIO_DECODE_PROCS", "0"))
        self._decode_pool: ProcessPoolExecutor | None = None
        if n_procs > 0:
            self._decode_pool = ProcessPoolExecutor(max_workers=n_procs)
            self._decode_and_chunk_speech_async = self._decode_and_chunk_speech_in_proc
            logger.info("Audio decode: using %d worker processes (GIL-free)", n_procs)

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
        if self.forced_aligner_config is None or len(chunks) != 1:
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

        result = await super().create_transcription(
            audio_data=audio_data,
            request=inner_request,
            raw_request=raw_request,
        )

        # Always reclaim the handoff, even on paths that will not align, so a
        # rejected or errored request cannot leave its waveform parked.
        decoded = self._take_decoded(audio_data) if self.forced_aligner_config is not None else None

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
        if self.forced_aligner_config is None:
            logger.warning(
                "word timestamps requested but no forced aligner configured; pass --forced-aligner to enable them."
            )
            return result
        text = (result.text or "").strip()
        if not text:
            result.words = []
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
