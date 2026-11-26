# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional server-configured ASR executor for Qwen heard-text calibration."""

from __future__ import annotations

import asyncio
import io
import time
import wave

import httpx
import numpy as np
from vllm.logger import init_logger

from vllm_omni.engine.duplex.session.history_calibration import HeardTextSnapshot
from vllm_omni.model_executor.models.qwen3_omni.duplex.history import asr_prefix

logger = init_logger(__name__)


class QwenAsrCalibration:
    """Share an inference concurrency limit across sessions; never load a model here.

    The deployment operator owns the transcription URL/model. It implements
    OpenAI's audio/transcriptions request shape and returns {"text": ...}.
    The framework deadline includes queueing, HTTP and prefix matching.
    """

    def __init__(self, url: str, model: str, timeout_s: float, *, max_concurrency: int = 4) -> None:
        if max_concurrency <= 0:
            raise ValueError("ASR concurrency must be positive")
        self.url = url
        self.model = model
        self.timeout_s = timeout_s
        self.slots = asyncio.Semaphore(max_concurrency)

    async def __call__(self, snapshot: HeardTextSnapshot) -> int | None:
        started = phase_started = time.monotonic()
        phase = "preprocess"
        timings = dict.fromkeys(("preprocess", "queue", "http", "match"), 0.0)
        status = "refused"
        try:
            # The temporal margin also guards ASR completion of partial words.
            samples = max(0, (snapshot.played_ms - 160) * snapshot.sample_rate_hz // 1000)
            audio = np.frombuffer(snapshot.pcm_f32le, dtype="<f4")[:samples]
            if audio.size < snapshot.sample_rate_hz // 2:
                status = "short_audio"
                return None
            if not np.isfinite(audio).all():
                status = "invalid_audio"
                return None
            pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
            wav = io.BytesIO()
            with wave.open(wav, "wb") as stream:
                stream.setparams((1, 2, snapshot.sample_rate_hz, 0, "NONE", "not compressed"))
                stream.writeframes(pcm)
            timings[phase] = time.monotonic() - phase_started
            phase, phase_started = "queue", time.monotonic()
            async with self.slots:
                timings[phase] = time.monotonic() - phase_started
                phase, phase_started = "http", time.monotonic()
                async with httpx.AsyncClient(timeout=self.timeout_s, trust_env=False) as client:
                    response = await client.post(
                        self.url,
                        data={"model": self.model, "response_format": "json"},
                        files={"file": ("heard.wav", wav.getvalue(), "audio/wav")},
                    )
                    response.raise_for_status()
                    transcript = response.json().get("text")
            if not isinstance(transcript, str) or len(transcript) > 8192:
                status = "invalid_transcript"
                return None
            timings[phase] = time.monotonic() - phase_started
            phase, phase_started = "match", time.monotonic()
            result = await asyncio.to_thread(asr_prefix, snapshot.text, transcript)
            status = result.reason or "accepted"
            return result.char_end if result.accepted else None
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            status = type(exc).__name__
            raise
        finally:
            timings[phase] = time.monotonic() - phase_started
            # HTTP includes server queueing and transport, not just inference.
            # On cancellation, phase identifies where the task was waiting.
            logger.info(
                "Qwen history ASR response=%s played_ms=%d status=%s phase=%s "
                "preprocess_ms=%.1f queue_ms=%.1f http_ms=%.1f match_ms=%.1f total_ms=%.1f",
                snapshot.response_id,
                snapshot.played_ms,
                status,
                phase,
                *(timings[name] * 1000 for name in ("preprocess", "queue", "http", "match")),
                (time.monotonic() - started) * 1000,
            )
