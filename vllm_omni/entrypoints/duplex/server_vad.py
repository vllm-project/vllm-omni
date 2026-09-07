# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
from vllm.transformers_utils.repo_utils import try_get_local_file

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.audio_utils_mixin import StreamingAudioResampler

SILERO_VAD_REPO_ID = "istupakov/silero-vad-onnx"
SILERO_VAD_REVISION = "8b14476858ef240c50b3884bb38cc67290c1cc70"
SILERO_VAD_FILENAME = "silero_vad.onnx"
SILERO_VAD_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
SILERO_VAD_MIN_THRESHOLD = 0.15
SILERO_VAD_DEFAULT_MIN_SPEECH_DURATION_MS = 96

_SERVER_VAD_FIELDS = {
    "type",
    "threshold",
    "prefix_padding_ms",
    "silence_duration_ms",
    "create_response",
    "interrupt_response",
    "min_speech_duration_ms",
}


@dataclass(frozen=True, slots=True)
class ServerVADConfig:
    """Validated OpenAI-compatible ``server_vad`` session configuration."""

    type: Literal["server_vad"] = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    create_response: bool = True
    interrupt_response: bool = True
    min_speech_duration_ms: int | None = None

    @classmethod
    def from_value(cls, value: object) -> ServerVADConfig:
        if not isinstance(value, dict):
            raise ValueError("turn_detection must be null or an object")
        unknown = sorted(set(value) - _SERVER_VAD_FIELDS)
        if unknown:
            raise ValueError(f"Unknown server_vad field(s): {', '.join(unknown)}")
        vad_type = value.get("type")
        if vad_type != "server_vad":
            raise ValueError("turn_detection.type must be 'server_vad'")

        threshold = value.get("threshold", 0.5)
        if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not 0 <= threshold <= 1:
            raise ValueError("server_vad.threshold must be a number between 0 and 1")

        prefix_padding_ms = value.get("prefix_padding_ms", 300)
        if isinstance(prefix_padding_ms, bool) or not isinstance(prefix_padding_ms, int) or prefix_padding_ms < 0:
            raise ValueError("server_vad.prefix_padding_ms must be a non-negative integer")

        silence_duration_ms = value.get("silence_duration_ms", 500)
        if (
            isinstance(silence_duration_ms, bool)
            or not isinstance(silence_duration_ms, int)
            or silence_duration_ms <= 0
        ):
            raise ValueError("server_vad.silence_duration_ms must be a positive integer")

        create_response = value.get("create_response", True)
        if not isinstance(create_response, bool):
            raise ValueError("server_vad.create_response must be a boolean")

        interrupt_response = value.get("interrupt_response", True)
        if not isinstance(interrupt_response, bool):
            raise ValueError("server_vad.interrupt_response must be a boolean")

        min_speech_duration_ms = value.get("min_speech_duration_ms")
        if min_speech_duration_ms is not None and (
            isinstance(min_speech_duration_ms, bool)
            or not isinstance(min_speech_duration_ms, int | float)
            or not np.isfinite(float(min_speech_duration_ms))
            or min_speech_duration_ms < 0
        ):
            raise ValueError("server_vad.min_speech_duration_ms must be a non-negative number")

        return cls(
            type=vad_type,
            threshold=float(threshold),
            prefix_padding_ms=prefix_padding_ms,
            silence_duration_ms=silence_duration_ms,
            create_response=create_response,
            interrupt_response=interrupt_response,
            min_speech_duration_ms=(int(min_speech_duration_ms) if min_speech_duration_ms is not None else None),
        )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "type": self.type,
            "threshold": self.threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
            "create_response": self.create_response,
            "interrupt_response": self.interrupt_response,
        }
        if self.min_speech_duration_ms is not None:
            result["min_speech_duration_ms"] = self.min_speech_duration_ms
        return result


def parse_session_turn_detection(
    payload: Mapping[str, object],
) -> tuple[bool, ServerVADConfig | None]:
    """Extract and validate OpenAI-compatible turn detection aliases."""
    configured_values: list[tuple[str, object]] = []
    if "turn_detection" in payload:
        configured_values.append(("turn_detection", payload["turn_detection"]))
    audio = payload.get("audio")
    if isinstance(audio, Mapping):
        audio_input = audio.get("input")
        if isinstance(audio_input, Mapping) and "turn_detection" in audio_input:
            configured_values.append(("audio.input.turn_detection", audio_input["turn_detection"]))
    if not configured_values:
        return False, None

    first_path, first_value = configured_values[0]
    first_config = None if first_value is None else ServerVADConfig.from_value(first_value)
    for field_path, value in configured_values[1:]:
        config = None if value is None else ServerVADConfig.from_value(value)
        if config != first_config:
            raise ValueError(f"{first_path} and {field_path} must not specify conflicting values")
    return True, first_config


class SpeechDetectorBackend(Protocol):
    """Detector contract; per-stream state is owned by the pipeline."""

    frame_samples: int

    def new_state(self) -> object: ...

    def infer(self, frame: np.ndarray, state: object) -> tuple[float, object]: ...


class SpeechDetectorBackendProvider(Protocol):
    def get(self) -> SpeechDetectorBackend: ...


@dataclass(frozen=True, slots=True)
class SpeechEndpointDecision:
    speech_started: bool = False
    speech_stopped: bool = False
    audio_start_ms: int | None = None
    audio_end_ms: int | None = None
    endpoint_delay_ms: int | None = None


class SpeechEndpointPolicy(Protocol):
    @property
    def speech_active(self) -> bool: ...

    def update(
        self,
        probability: float,
        *,
        frame_start_sample: int,
        frame_samples: int,
    ) -> SpeechEndpointDecision: ...

    def reset(self) -> None: ...


class ThresholdEndpointPolicy:
    """Apply threshold and trailing-silence endpoint rules."""

    def __init__(self, config: ServerVADConfig, *, sample_rate_hz: int) -> None:
        self.config = config
        self.sample_rate_hz = sample_rate_hz
        self._speech_active = False
        self._candidate_samples = 0
        self._candidate_start_sample: int | None = None
        self._silence_samples = 0

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    def update(
        self,
        probability: float,
        *,
        frame_start_sample: int,
        frame_samples: int,
    ) -> SpeechEndpointDecision:
        if probability >= self.config.threshold:
            self._silence_samples = 0
            if not self._speech_active:
                if self._candidate_samples == 0:
                    self._candidate_start_sample = frame_start_sample
                self._candidate_samples += frame_samples
                min_speech_duration_ms = self.config.min_speech_duration_ms or 0
                min_speech_samples = max(1, round(min_speech_duration_ms * self.sample_rate_hz / 1000))
                if self._candidate_samples < min_speech_samples:
                    return SpeechEndpointDecision()
                self._speech_active = True
                detected_start_sample = (
                    self._candidate_start_sample if self._candidate_start_sample is not None else frame_start_sample
                )
                detected_start_ms = round(detected_start_sample * 1000 / self.sample_rate_hz)
                self._candidate_samples = 0
                self._candidate_start_sample = None
                return SpeechEndpointDecision(
                    speech_started=True,
                    audio_start_ms=max(0, detected_start_ms - self.config.prefix_padding_ms),
                )
            return SpeechEndpointDecision()

        if not self._speech_active:
            self._candidate_samples = 0
            self._candidate_start_sample = None
            return SpeechEndpointDecision()

        # Match Silero v6.2's streaming VAD hysteresis: activation uses the
        # configured threshold, while potential speech end starts only below
        # ``max(threshold - 0.15, 0.01)``. Once a silence candidate exists,
        # intermediate frames keep elapsed time moving but cannot themselves
        # close the turn.
        negative_threshold = max(self.config.threshold - SILERO_VAD_MIN_THRESHOLD, 0.01)
        below_negative_threshold = probability < negative_threshold
        if not below_negative_threshold and self._silence_samples == 0:
            return SpeechEndpointDecision()

        self._silence_samples += frame_samples
        if not below_negative_threshold:
            return SpeechEndpointDecision()
        silence_limit = max(1, round(self.config.silence_duration_ms * self.sample_rate_hz / 1000))
        if self._silence_samples < silence_limit:
            return SpeechEndpointDecision()

        # OpenAI defines ``audio_end_ms`` as the end of the audio sent to the
        # model, including the trailing silence used for endpoint detection.
        audio_end_sample = frame_start_sample + frame_samples
        endpoint_delay_ms = round(self._silence_samples * 1000 / self.sample_rate_hz)
        self.reset()
        return SpeechEndpointDecision(
            speech_stopped=True,
            audio_end_ms=max(0, round(audio_end_sample * 1000 / self.sample_rate_hz)),
            endpoint_delay_ms=endpoint_delay_ms,
        )

    def reset(self) -> None:
        self._speech_active = False
        self._candidate_samples = 0
        self._candidate_start_sample = None
        self._silence_samples = 0


@dataclass(frozen=True, slots=True)
class ServerVADFrame:
    samples: np.ndarray
    decision: SpeechEndpointDecision


@dataclass(frozen=True, slots=True)
class ServerVADBatch:
    frames: tuple[ServerVADFrame, ...]
    inference_ms: float


class ServerVADPipeline:
    """Frame continuous 16 kHz audio and apply acoustic endpointing.

    The pipeline retains only input-resampling residuals, a partial frame,
    detector state, and timing counters. Commit-eligible prefix and utterance
    audio remain session-owned.
    """

    sample_rate_hz = 16_000

    def __init__(
        self,
        backend: SpeechDetectorBackend,
        config: ServerVADConfig,
        *,
        endpoint_policy: SpeechEndpointPolicy | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.endpoint_policy = endpoint_policy or ThresholdEndpointPolicy(
            config,
            sample_rate_hz=self.sample_rate_hz,
        )
        self._input_resampler: StreamingAudioResampler | None = None
        self._source_sample_rate_hz: int | None = None
        self._scratch = np.empty(0, dtype=np.float32)
        self._detector_state = backend.new_state()
        self._processed_samples = 0

    @property
    def speech_active(self) -> bool:
        return self.endpoint_policy.speech_active

    @property
    def source_sample_rate_hz(self) -> int | None:
        return self._source_sample_rate_hz

    @property
    def scratch_bytes(self) -> int:
        resampler_bytes = self._input_resampler.scratch_bytes if self._input_resampler is not None else 0
        return int(self._scratch.nbytes) + resampler_bytes

    async def push_pcm16(
        self,
        samples: bytes | bytearray | memoryview | np.ndarray,
        *,
        source_sample_rate_hz: int,
    ) -> ServerVADBatch:
        """Normalize chunked mono PCM16 input and run endpoint detection."""
        if isinstance(source_sample_rate_hz, bool) or source_sample_rate_hz not in {16_000, 24_000}:
            raise ValueError("server_vad PCM16 input sample rate must be 16000 or 24000 Hz")

        if isinstance(samples, np.ndarray):
            if samples.ndim != 1 or samples.dtype.kind != "i" or samples.dtype.itemsize != 2:
                raise ValueError("server_vad PCM16 samples must be a one-dimensional int16 array")
            pcm16 = np.ascontiguousarray(samples, dtype="<i2")
        elif isinstance(samples, bytes | bytearray | memoryview):
            raw = bytes(samples)
            if len(raw) % np.dtype("<i2").itemsize:
                raise ValueError("server_vad PCM16 input contains an incomplete sample")
            pcm16 = np.frombuffer(raw, dtype="<i2")
        else:
            raise ValueError("server_vad PCM16 input must be bytes or an int16 array")

        if self._source_sample_rate_hz is not None and source_sample_rate_hz != self._source_sample_rate_hz:
            raise ValueError("server_vad input sample rate cannot change within a continuous audio stream")
        if pcm16.size and self._source_sample_rate_hz is None:
            from vllm_omni.entrypoints.openai.audio_utils_mixin import StreamingAudioResampler

            self._input_resampler = (
                None
                if source_sample_rate_hz == self.sample_rate_hz
                else StreamingAudioResampler(source_sample_rate_hz, self.sample_rate_hz)
            )
            self._source_sample_rate_hz = source_sample_rate_hz

        normalized = np.ascontiguousarray(pcm16, dtype=np.float32) * np.float32(1.0 / 32768.0)
        if self._input_resampler is not None:
            normalized = self._input_resampler.process(normalized)
        return await self.push(normalized)

    async def push(self, samples: np.ndarray) -> ServerVADBatch:
        normalized = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        return await asyncio.to_thread(self._push_sync, normalized)

    def _push_sync(self, samples: np.ndarray) -> ServerVADBatch:
        started_at = time.perf_counter()
        if self._scratch.size:
            samples = np.concatenate((self._scratch, samples))
        frame_samples = int(self.backend.frame_samples)
        complete_samples = samples.size - samples.size % frame_samples
        self._scratch = np.ascontiguousarray(samples[complete_samples:], dtype=np.float32)

        results: list[ServerVADFrame] = []
        for offset in range(0, complete_samples, frame_samples):
            frame = np.ascontiguousarray(samples[offset : offset + frame_samples], dtype=np.float32)
            frame_start = self._processed_samples
            probability, self._detector_state = self.backend.infer(frame, self._detector_state)
            probability = min(1.0, max(0.0, float(probability)))
            decision = self.endpoint_policy.update(
                probability,
                frame_start_sample=frame_start,
                frame_samples=frame_samples,
            )

            self._processed_samples += frame_samples
            results.append(
                ServerVADFrame(
                    samples=frame,
                    decision=decision,
                )
            )
        return ServerVADBatch(
            frames=tuple(results),
            inference_ms=(time.perf_counter() - started_at) * 1000,
        )

    def reset(self) -> None:
        self._input_resampler = None
        self._source_sample_rate_hz = None
        self._scratch = np.empty(0, dtype=np.float32)
        self._detector_state = self.backend.new_state()
        self.endpoint_policy.reset()


@dataclass(frozen=True, slots=True)
class _SileroVADState:
    model_state: np.ndarray
    context: np.ndarray


class SileroVADBackend:
    """Shared ONNX Runtime Silero v6.2 detector running on CPU."""

    sample_rate_hz = 16_000
    frame_samples = 512
    context_samples = 64
    model_state_shape = (2, 1, 128)

    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - platform packaging supplies ORT.
            raise RuntimeError("server_vad requires ONNX Runtime") from exc

        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 1

        self.model_path = Path(model_path)
        self._session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
            sess_options=session_options,
        )
        input_names = {item.name for item in self._session.get_inputs()}
        if "input" not in input_names or "sr" not in input_names or "state" not in input_names:
            raise RuntimeError(f"Unsupported Silero ONNX input contract: {sorted(input_names)}")
        self._warm_up()

    def _warm_up(self) -> None:
        self.infer(np.zeros(self.frame_samples, dtype=np.float32), self.new_state())

    def new_state(self) -> _SileroVADState:
        return _SileroVADState(
            model_state=np.zeros(self.model_state_shape, dtype=np.float32),
            context=np.zeros((1, self.context_samples), dtype=np.float32),
        )

    def infer(self, frame: np.ndarray, state: object) -> tuple[float, object]:
        if not isinstance(state, _SileroVADState):
            raise TypeError("Silero detector state must be created by SileroVADBackend.new_state()")
        model_state = np.ascontiguousarray(state.model_state, dtype=np.float32)
        context = np.ascontiguousarray(state.context, dtype=np.float32)
        if model_state.shape != self.model_state_shape:
            raise ValueError(f"Silero model state must have shape {self.model_state_shape}, got {model_state.shape}")
        expected_context_shape = (1, self.context_samples)
        if context.shape != expected_context_shape:
            raise ValueError(f"Silero context must have shape {expected_context_shape}, got {context.shape}")

        audio = np.ascontiguousarray(frame, dtype=np.float32).reshape(1, -1)
        if audio.shape[1] != self.frame_samples:
            raise ValueError(
                f"Silero detector frame must contain exactly {self.frame_samples} samples, got {audio.shape[1]}"
            )
        model_input = np.concatenate((context, audio), axis=1)
        inputs = {
            "input": model_input,
            "state": model_state,
            "sr": np.asarray(self.sample_rate_hz, dtype=np.int64),
        }
        # ONNX Runtime permits concurrent Run calls on one CPU session. Stream
        # state is explicit, so independent pipelines can safely share the session.
        output = self._session.run(None, inputs)
        if len(output) < 2:
            raise RuntimeError("Silero ONNX model did not return probability and model state")
        probability = float(np.asarray(output[0]).reshape(-1)[0])
        next_state = _SileroVADState(
            model_state=np.ascontiguousarray(output[1], dtype=np.float32),
            context=np.ascontiguousarray(model_input[:, -self.context_samples :], dtype=np.float32),
        )
        return probability, next_state


class SileroVADBackendProvider:
    """Resolve, verify, and load one Silero model instance per process."""

    def __init__(self, *, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._backend: SileroVADBackend | None = None
        self._lock = threading.Lock()

    def get(self) -> SileroVADBackend:
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is None:
                path = self._resolve_local_artifact()
                self._verify_checksum(path)
                self._backend = SileroVADBackend(path)
        return self._backend

    def _resolve_local_artifact(self) -> Path:
        if self.model_path:
            path = Path(self.model_path).expanduser()
            if path.is_file():
                return path
            raise RuntimeError(f"Configured Silero VAD model does not exist: {path}")
        cached = try_get_local_file(
            model=SILERO_VAD_REPO_ID,
            file_name=SILERO_VAD_FILENAME,
            revision=SILERO_VAD_REVISION,
        )
        if isinstance(cached, Path) and cached.is_file():
            return cached
        raise RuntimeError(
            "Silero VAD artifact is not available locally. Pre-download the pinned artifact "
            "or configure duplex_session.server_vad_model_path."
        )

    def _verify_checksum(self, path: Path) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SILERO_VAD_SHA256:
            raise RuntimeError(f"Silero VAD checksum mismatch for {path}: expected {SILERO_VAD_SHA256}, got {digest}")
