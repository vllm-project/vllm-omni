from __future__ import annotations

import binascii
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pybase64 as base64

SILERO_VAD_MIN_THRESHOLD = 0.15


class ServerVADUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SileroVADConfig:
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    min_speech_duration_ms: int = 96


@dataclass(frozen=True, slots=True)
class StreamingVADResult:
    is_speech: bool
    speech_active: bool
    speech_started: bool = False
    speech_stopped: bool = False
    speech_probability: float = 0.0
    speech_start_ms: int | None = None
    speech_end_ms: int | None = None


class SileroStreamingVAD:
    _SAMPLE_RATE_HZ = 16_000
    _WINDOW_SAMPLES = 512

    def __init__(
        self,
        config: SileroVADConfig,
        *,
        frame_scorer: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        self.config = config
        self._frame_scorer = frame_scorer
        self._model: object | None = None
        self._clear_stream_state()

    def _clear_stream_state(self) -> None:
        self._pending = np.empty(0, dtype=np.float32)
        self._speech_active = False
        self._candidate_samples = 0
        self._candidate_start_sample: int | None = None
        self._silence_samples = 0
        self._processed_samples = 0
        self._resample_rate_hz: int | None = None
        self._resample_remainder = 0

    def reset(self) -> None:
        self._clear_stream_state()
        reset_states = getattr(self._model, "reset_states", None)
        if callable(reset_states):
            reset_states()

    def process_base64(
        self,
        audio: object,
        *,
        fmt: object,
        sample_rate_hz: object,
    ) -> StreamingVADResult:
        if fmt != "pcm_f32le" or not isinstance(audio, str):
            raise ValueError("Silero server VAD requires decoded pcm_f32le audio")
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Silero server VAD received invalid base64 audio") from exc
        if len(raw) % 4:
            raise ValueError("Silero server VAD received an incomplete pcm_f32le frame")
        samples = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
        rate = int(sample_rate_hz) if isinstance(sample_rate_hz, int | float) else self._SAMPLE_RATE_HZ
        if rate <= 0:
            raise ValueError("Silero server VAD requires a positive sample rate")
        if rate != self._SAMPLE_RATE_HZ:
            if self._resample_rate_hz != rate:
                self._resample_rate_hz = rate
                self._resample_remainder = 0
            numerator = samples.size * self._SAMPLE_RATE_HZ + self._resample_remainder
            target_size, self._resample_remainder = divmod(numerator, rate)
            if target_size == 0:
                samples = np.empty(0, dtype=np.float32)
            elif samples.size == 1:
                samples = np.full(target_size, samples[0], dtype=np.float32)
            else:
                source_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=True)
                target_x = np.linspace(0.0, 1.0, num=target_size, endpoint=True)
                samples = np.interp(target_x, source_x, samples).astype(np.float32)
        else:
            self._resample_rate_hz = None
            self._resample_remainder = 0
        return self.process(samples)

    def process(self, samples: np.ndarray) -> StreamingVADResult:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size:
            self._pending = np.concatenate((self._pending, samples))

        started = False
        stopped = False
        start_ms: int | None = None
        end_ms: int | None = None
        max_probability = 0.0
        contained_speech = self._speech_active
        min_speech_samples = self.config.min_speech_duration_ms * 16
        min_silence_samples = self.config.silence_duration_ms * 16
        negative_threshold = max(0.0, self.config.threshold - SILERO_VAD_MIN_THRESHOLD)

        while self._pending.size >= self._WINDOW_SAMPLES:
            frame = self._pending[: self._WINDOW_SAMPLES]
            self._pending = self._pending[self._WINDOW_SAMPLES :]
            frame_start = self._processed_samples
            self._processed_samples += self._WINDOW_SAMPLES
            probability = min(1.0, max(0.0, float(self._score_frame(frame))))
            max_probability = max(max_probability, probability)

            if self._speech_active:
                contained_speech = True
                if probability < negative_threshold:
                    self._silence_samples += self._WINDOW_SAMPLES
                    if self._silence_samples >= min_silence_samples:
                        speech_end_sample = self._processed_samples - self._silence_samples
                        self._speech_active = False
                        self._silence_samples = 0
                        stopped = True
                        end_ms = max(0, int(round(speech_end_sample / 16)))
                else:
                    self._silence_samples = 0
                continue

            if probability >= self.config.threshold:
                if self._candidate_samples == 0:
                    self._candidate_start_sample = frame_start
                self._candidate_samples += self._WINDOW_SAMPLES
                if self._candidate_samples >= min_speech_samples:
                    candidate_start = self._candidate_start_sample or 0
                    prefix_samples = self.config.prefix_padding_ms * 16
                    self._speech_active = True
                    self._candidate_samples = self._silence_samples = 0
                    self._candidate_start_sample = None
                    contained_speech = True
                    started = True
                    start_ms = max(0, int(round((candidate_start - prefix_samples) / 16)))
            else:
                self._candidate_samples = 0
                self._candidate_start_sample = None

        return StreamingVADResult(
            contained_speech, self._speech_active, started, stopped, max_probability, start_ms, end_ms
        )

    def _score_frame(self, frame: np.ndarray) -> float:
        if self._frame_scorer is not None:
            return float(self._frame_scorer(frame))
        if self._model is None:
            try:
                import torch
                from silero_vad import load_silero_vad
            except ImportError as exc:
                raise ServerVADUnavailableError(
                    "server_vad requires the optional 'silero-vad' package; install vllm-omni[server-vad]"
                ) from exc
            self._model = load_silero_vad(onnx=False)
            self._frame_scorer = lambda value: float(
                self._model(torch.from_numpy(np.ascontiguousarray(value)), self._SAMPLE_RATE_HZ).item()
            )
        return float(self._frame_scorer(frame))
