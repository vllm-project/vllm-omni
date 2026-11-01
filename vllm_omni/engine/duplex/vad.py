# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Silero server VAD for the engine-resident duplex session.

The detector keeps the split upstream's serving-side ``server_vad`` module
used, because the split is what makes one model serve many sessions: a
**backend** scores one 512-sample frame and is otherwise stateless, and the
per-stream state -- the partial frame, the model state, the endpoint counters --
belongs to :class:`SileroStreamingVAD`, one per session.

Two backends. :class:`SileroVADBackend` runs the pinned Silero v6.2 ONNX graph on
CPU and is shared process-wide, which is why its state is passed in and out
rather than held on the session object. :class:`TorchSileroBackend` loads the
same model through the ``silero-vad`` package for environments that have no
local ONNX artifact; torch keeps its state inside the module, so that one is per
session.

Endpointing follows Silero v6.2's streaming hysteresis: activation uses the
configured threshold, while a turn can only *end* on frames below
``max(threshold - 0.15, 0.01)``. Once a silence candidate exists, louder frames
keep the elapsed-silence clock running but cannot themselves close the turn.
"""

from __future__ import annotations

import binascii
import hashlib
import threading
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

import numpy as np
import pybase64 as base64
from vllm.logger import init_logger

logger = init_logger(__name__)

SILERO_VAD_REPO_ID = "istupakov/silero-vad-onnx"
SILERO_VAD_REVISION = "8b14476858ef240c50b3884bb38cc67290c1cc70"
SILERO_VAD_FILENAME = "silero_vad.onnx"
SILERO_VAD_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
SILERO_VAD_MIN_THRESHOLD = 0.15

_SAMPLE_RATE_HZ = 16_000
_FRAME_SAMPLES = 512


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
    # Actual start of the silence that produced speech_stopped. Hysteresis
    # can delay the stop beyond silence_duration_ms.
    silence_start_ms: int | None = None


_T = TypeVar("_T")

#: Endpointing written as a coroutine over Silero frames: it yields each 512-sample
#: frame that needs a score, receives the raw probability, and returns its result.
#: Whoever drives it decides how frames are scored -- one at a time
#: (:func:`run_frame_steps`) or batched across sessions (the duplex VAD batcher).
FrameSteps = Generator[np.ndarray, float, _T]


def run_frame_steps(steps: FrameSteps[_T], score: Callable[[np.ndarray], float]) -> _T:
    """Drive ``steps`` to completion, scoring each frame it asks for with ``score``."""
    try:
        frame = next(steps)
        while True:
            frame = steps.send(score(frame))
    except StopIteration as stop:
        return stop.value


# --------------------------------------------------------------------------- #
# Backends                                                                    #
# --------------------------------------------------------------------------- #


class SpeechDetectorBackend(Protocol):
    """Scores one frame. Per-stream state is created here but owned by the caller."""

    frame_samples: int

    def new_state(self) -> object: ...

    def infer(self, frame: np.ndarray, state: object) -> tuple[float, object]: ...


@dataclass(frozen=True, slots=True)
class _SileroVADState:
    model_state: np.ndarray
    context: np.ndarray


class SileroVADBackend:
    """Shared ONNX Runtime Silero v6.2 detector running on CPU."""

    sample_rate_hz = _SAMPLE_RATE_HZ
    frame_samples = _FRAME_SAMPLES
    context_samples = 64
    model_state_shape = (2, 1, 128)

    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - platform packaging supplies ORT.
            raise ServerVADUnavailableError("server_vad requires ONNX Runtime") from exc

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
        if not {"input", "sr", "state"} <= input_names:
            raise ServerVADUnavailableError(f"Unsupported Silero ONNX input contract: {sorted(input_names)}")
        self._warm_up()

    def _warm_up(self) -> None:
        self.infer(np.zeros(self.frame_samples, dtype=np.float32), self.new_state())

    def new_state(self) -> _SileroVADState:
        return _SileroVADState(
            model_state=np.zeros(self.model_state_shape, dtype=np.float32),
            context=np.zeros((1, self.context_samples), dtype=np.float32),
        )

    def infer(self, frame: np.ndarray, state: object) -> tuple[float, object]:
        probabilities, states = self.infer_batch([frame], [state])
        return probabilities[0], states[0]

    def infer_batch(
        self, frames: Sequence[np.ndarray], states: Sequence[object]
    ) -> tuple[list[float], list[_SileroVADState]]:
        """Score one frame from each of ``len(frames)`` independent streams in a single ONNX run.

        Rows never interact: the graph's batch axis carries one stream's frame,
        context and recurrent state each, so the result per row is the same as
        scoring that stream alone.
        """
        if len(frames) != len(states) or not frames:
            raise ValueError("Silero batch needs one state per frame and at least one frame")
        model_states, contexts, audio = [], [], []
        for frame, state in zip(frames, states):
            if not isinstance(state, _SileroVADState):
                raise TypeError("Silero detector state must be created by SileroVADBackend.new_state()")
            if state.model_state.shape != self.model_state_shape:
                raise ValueError(
                    f"Silero model state must have shape {self.model_state_shape}, got {state.model_state.shape}"
                )
            expected_context_shape = (1, self.context_samples)
            if state.context.shape != expected_context_shape:
                raise ValueError(f"Silero context must have shape {expected_context_shape}, got {state.context.shape}")
            row = np.asarray(frame, dtype=np.float32).reshape(-1)
            if row.size != self.frame_samples:
                raise ValueError(
                    f"Silero detector frame must contain exactly {self.frame_samples} samples, got {row.size}"
                )
            model_states.append(state.model_state)
            contexts.append(state.context)
            audio.append(row)
        model_input = np.concatenate((np.concatenate(contexts, axis=0), np.stack(audio)), axis=1).astype(
            np.float32, copy=False
        )
        # ONNX Runtime permits concurrent Run calls on one CPU session. Stream
        # state is explicit, so independent sessions can safely share this one.
        output = self._session.run(
            None,
            {
                "input": model_input,
                "state": np.ascontiguousarray(np.concatenate(model_states, axis=1), dtype=np.float32),
                "sr": np.asarray(self.sample_rate_hz, dtype=np.int64),
            },
        )
        if len(output) < 2:
            raise ServerVADUnavailableError("Silero ONNX model did not return probability and model state")
        probabilities = np.asarray(output[0], dtype=np.float32).reshape(-1)
        next_model_state = np.asarray(output[1], dtype=np.float32)
        if probabilities.size != len(frames) or next_model_state.shape[1:2] != (len(frames),):
            raise ServerVADUnavailableError("Silero ONNX model returned a batch of the wrong size")
        next_states = [
            _SileroVADState(
                model_state=np.ascontiguousarray(next_model_state[:, row : row + 1]),
                context=np.ascontiguousarray(model_input[row : row + 1, -self.context_samples :]),
            )
            for row in range(len(frames))
        ]
        return [float(p) for p in probabilities], next_states


class TorchSileroBackend:
    """The same model through the ``silero-vad`` package, for hosts with no ONNX artifact.

    Torch keeps the recurrent state inside the module, so unlike the ONNX
    backend this one cannot be shared: each session gets its own instance and
    ``new_state`` only resets it.
    """

    sample_rate_hz = _SAMPLE_RATE_HZ
    frame_samples = _FRAME_SAMPLES

    def __init__(self) -> None:
        try:
            import torch
            from silero_vad import load_silero_vad
        except ImportError as exc:
            raise ServerVADUnavailableError(
                "server_vad needs either a local Silero ONNX artifact (see "
                "duplex_session.server_vad_model_path) or the optional 'silero-vad' "
                "package; install vllm-omni[server-vad]"
            ) from exc
        self._torch = torch
        self._model = load_silero_vad(onnx=False)

    def new_state(self) -> object:
        reset_states = getattr(self._model, "reset_states", None)
        if callable(reset_states):
            reset_states()
        return None

    def infer(self, frame: np.ndarray, state: object) -> tuple[float, object]:
        tensor = self._torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
        return float(self._model(tensor, self.sample_rate_hz).item()), state


class SileroVADBackendProvider:
    """Resolve, verify and load one detector backend per engine process.

    ONNX is preferred and shared; the torch package is the fallback and is built
    per call because its state is internal.
    """

    def __init__(self, *, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._backend: SileroVADBackend | None = None
        self._lock = threading.Lock()

    def get(self) -> SpeechDetectorBackend:
        """The shared ONNX backend, or a per-session torch one when it cannot be built.

        An explicitly configured ``server_vad_model_path`` never falls back: the
        operator named that artifact, so a missing file or a missing ONNX Runtime
        is their error, not something to paper over with a different model.
        """
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is not None:
                return self._backend
            path = self._resolve_local_artifact()
            if path is None:
                # No pinned artifact: the torch package keeps state internally,
                # so this instance belongs to one session only.
                return TorchSileroBackend()
            self._verify_checksum(path)
            try:
                self._backend = SileroVADBackend(path)
            except ServerVADUnavailableError:
                if self.model_path:
                    raise
                logger.warning(
                    "Silero ONNX artifact %s found but ONNX Runtime is unavailable; "
                    "falling back to the torch 'silero-vad' package (one model per session)",
                    path,
                )
                return TorchSileroBackend()
        return self._backend

    def _resolve_local_artifact(self) -> Path | None:
        if self.model_path:
            path = Path(self.model_path).expanduser()
            if path.is_file():
                return path
            # Explicitly configured: a missing file is an error, not a fallback.
            raise ServerVADUnavailableError(f"Configured Silero VAD model does not exist: {path}")
        try:
            from vllm.transformers_utils.repo_utils import try_get_local_file

            cached = try_get_local_file(
                model=SILERO_VAD_REPO_ID,
                file_name=SILERO_VAD_FILENAME,
                revision=SILERO_VAD_REVISION,
            )
        except Exception:
            return None
        return cached if isinstance(cached, Path) and cached.is_file() else None

    def _verify_checksum(self, path: Path) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SILERO_VAD_SHA256:
            raise ServerVADUnavailableError(
                f"Silero VAD checksum mismatch for {path}: expected {SILERO_VAD_SHA256}, got {digest}"
            )


# --------------------------------------------------------------------------- #
# Per-session streaming detector                                              #
# --------------------------------------------------------------------------- #


class SileroStreamingVAD:
    _SAMPLE_RATE_HZ = _SAMPLE_RATE_HZ
    _WINDOW_SAMPLES = _FRAME_SAMPLES

    def __init__(
        self,
        config: SileroVADConfig,
        *,
        backend: SpeechDetectorBackend | None = None,
        backend_provider: SileroVADBackendProvider | None = None,
        frame_scorer: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        self.config = config
        self._explicit_backend = backend
        self._provider = backend_provider
        self._frame_scorer = frame_scorer
        self._backend: SpeechDetectorBackend | None = backend
        self._detector_state: object = None
        self._clear_stream_state()

    def _clear_stream_state(self) -> None:
        self._pending = np.empty(0, dtype=np.float32)
        self._speech_active = False
        self._candidate_samples = 0
        self._candidate_start_sample: int | None = None
        self._silence_samples = 0
        self._processed_samples = 0
        self._stream_start_ms = 0
        self._resample_rate_hz: int | None = None
        self._resample_remainder = 0

    @property
    def scratch_bytes(self) -> int:
        """Bytes this detector holds between chunks (the partial frame only)."""
        return int(self._pending.nbytes)

    @property
    def samples_until_frame(self) -> int:
        """Samples needed to finish the next scoring frame."""
        return self._WINDOW_SAMPLES - self._pending.size

    @property
    def speech_active(self) -> bool:
        return self._speech_active

    @property
    def speech_pending(self) -> bool:
        """An onset is being confirmed against ``min_speech_duration_ms``."""
        return self._candidate_samples > 0

    def reset(self) -> None:
        """Drop stream state while preserving the session-relative audio clock.

        The clock survives so ``speech_start_ms`` after a barge-in still refers
        to the session's timeline; ``_stream_start_ms`` then stops the prefix
        padding reaching back into audio that was discarded.
        """
        stream_start_ms = round(self._processed_samples * 1000 / self._SAMPLE_RATE_HZ)
        processed = self._processed_samples
        self._clear_stream_state()
        self._processed_samples = processed
        self._stream_start_ms = stream_start_ms
        if self._backend is not None:
            self._detector_state = self._backend.new_state()

    # ------------------------------------------------------------------ #
    # Input                                                              #
    # ------------------------------------------------------------------ #

    @property
    def processed_ms(self) -> int:
        """Session-relative end of the audio scored so far."""
        return round(self._processed_samples * 1000 / self._SAMPLE_RATE_HZ)

    def process_base64(
        self,
        audio: object,
        *,
        fmt: object,
        sample_rate_hz: object,
    ) -> StreamingVADResult:
        return self.process(self.decode_base64(audio, fmt=fmt, sample_rate_hz=sample_rate_hz))

    def decode_base64(self, audio: object, *, fmt: object, sample_rate_hz: object) -> np.ndarray:
        """Decode one appended chunk to 16 kHz float32 samples."""
        if not isinstance(audio, str):
            raise ValueError("Silero server VAD requires base64 audio")
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Silero server VAD received invalid base64 audio") from exc
        rate = int(sample_rate_hz) if isinstance(sample_rate_hz, int | float) else self._SAMPLE_RATE_HZ
        if rate <= 0:
            raise ValueError("Silero server VAD requires a positive sample rate")

        if fmt == "pcm_f32le":
            if len(raw) % 4:
                raise ValueError("Silero server VAD received an incomplete pcm_f32le frame")
            samples = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
        elif fmt == "pcm16":
            if len(raw) % 2:
                raise ValueError("Silero server VAD received an incomplete pcm16 sample")
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) * np.float32(1.0 / 32768.0)
        else:
            raise ValueError("Silero server VAD requires pcm_f32le or pcm16 audio")

        return self._to_16k(samples, rate)

    def _to_16k(self, samples: np.ndarray, rate: int) -> np.ndarray:
        """Resample to 16 kHz, carrying the fractional sample-count remainder.

        A fallback path: the session runner already normalises input to 16 kHz
        through ``convert_input_audio_with_rate`` before the detector sees it,
        so this runs only for callers that feed the VAD directly.

        Carrying the remainder keeps the *number* of output samples exact, so
        frame boundaries cannot drift over a long stream. The interpolation
        itself is per chunk and has no filter state, so sample values near a
        chunk edge differ slightly from resampling the same audio in one go --
        enough for the frame grid to stay put, not enough to call it a
        streaming resampler.
        """
        if rate == self._SAMPLE_RATE_HZ:
            self._resample_rate_hz = None
            self._resample_remainder = 0
            return samples
        if self._resample_rate_hz is not None and rate != self._resample_rate_hz:
            raise ValueError("server_vad input sample rate cannot change within a continuous audio stream")
        self._resample_rate_hz = rate
        numerator = samples.size * self._SAMPLE_RATE_HZ + self._resample_remainder
        target_size, self._resample_remainder = divmod(numerator, rate)
        if target_size == 0:
            return np.empty(0, dtype=np.float32)
        if samples.size == 1:
            return np.full(target_size, samples[0], dtype=np.float32)
        source_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=True)
        target_x = np.linspace(0.0, 1.0, num=target_size, endpoint=True)
        return np.interp(target_x, source_x, samples).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Endpointing                                                        #
    # ------------------------------------------------------------------ #

    def process(self, samples: np.ndarray) -> StreamingVADResult:
        return run_frame_steps(self.frame_steps(samples), self._score_frame)

    def frame_steps(self, samples: np.ndarray) -> FrameSteps[StreamingVADResult]:
        """:meth:`process` with frame scoring left to the driver (see :data:`FrameSteps`)."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size:
            self._pending = np.concatenate((self._pending, samples))

        started = False
        stopped = False
        start_ms: int | None = None
        end_ms: int | None = None
        silence_start_ms: int | None = None
        max_probability = 0.0
        contained_speech = self._speech_active
        min_speech_samples = max(1, round(self.config.min_speech_duration_ms * self._SAMPLE_RATE_HZ / 1000))
        min_silence_samples = max(1, round(self.config.silence_duration_ms * self._SAMPLE_RATE_HZ / 1000))
        negative_threshold = max(self.config.threshold - SILERO_VAD_MIN_THRESHOLD, 0.01)

        pending = self._pending
        offset = 0
        while pending.size - offset >= self._WINDOW_SAMPLES:
            frame = np.ascontiguousarray(pending[offset : offset + self._WINDOW_SAMPLES], dtype=np.float32)
            offset += self._WINDOW_SAMPLES
            frame_start = self._processed_samples
            self._processed_samples += self._WINDOW_SAMPLES
            probability = min(1.0, max(0.0, float((yield frame))))
            max_probability = max(max_probability, probability)

            if self._speech_active:
                contained_speech = True
                if probability >= self.config.threshold:
                    # Confirmed speech cancels a pending endpoint. Otherwise a
                    # brief pause can age through resumed speech and split the
                    # utterance at the next low-probability frame.
                    self._silence_samples = 0
                    continue
                below_negative_threshold = probability < negative_threshold
                if not below_negative_threshold and self._silence_samples == 0:
                    continue
                # Scores inside the hysteresis band keep the pending timer,
                # but cannot themselves close the turn.
                self._silence_samples += self._WINDOW_SAMPLES
                if not below_negative_threshold or self._silence_samples < min_silence_samples:
                    continue
                silence_start_ms = round(
                    (self._processed_samples - self._silence_samples) * 1000 / self._SAMPLE_RATE_HZ
                )
                self._speech_active = False
                self._silence_samples = 0
                stopped = True
                # OpenAI defines audio_end_ms as the end of the audio sent to
                # the model, trailing silence included.
                end_ms = max(0, round(self._processed_samples * 1000 / self._SAMPLE_RATE_HZ))
                continue

            if probability >= self.config.threshold:
                if self._candidate_samples == 0:
                    self._candidate_start_sample = frame_start
                self._candidate_samples += self._WINDOW_SAMPLES
                if self._candidate_samples >= min_speech_samples:
                    candidate_start = self._candidate_start_sample or 0
                    detected_start_ms = round(candidate_start * 1000 / self._SAMPLE_RATE_HZ)
                    self._speech_active = True
                    self._candidate_samples = self._silence_samples = 0
                    self._candidate_start_sample = None
                    contained_speech = True
                    started = True
                    # Prefix audio from before the last reset is gone.
                    start_ms = max(
                        self._stream_start_ms,
                        max(0, detected_start_ms - self.config.prefix_padding_ms),
                    )
            else:
                self._candidate_samples = 0
                self._candidate_start_sample = None

        # Copy the residual once, after the frames are scored, so a large input
        # chunk is neither pinned by a view nor re-copied for every frame.
        self._pending = pending[offset:].copy()

        return StreamingVADResult(
            contained_speech, self._speech_active, started, stopped, max_probability, start_ms, end_ms, silence_start_ms
        )

    def _score_frame(self, frame: np.ndarray) -> float:
        if self._frame_scorer is not None:
            return float(self._frame_scorer(frame))
        backend = self._require_backend()
        probability, self._detector_state = backend.infer(frame, self._detector_state)
        return float(probability)

    def _require_backend(self) -> SpeechDetectorBackend:
        if self._backend is None:
            provider = self._provider or SileroVADBackendProvider()
            self._backend = provider.get()
            self._detector_state = self._backend.new_state()
        return self._backend


def score_frames(requests: Sequence[tuple[SileroStreamingVAD, np.ndarray]]) -> list[float | BaseException]:
    """Score one pending frame for each of several streams, batching what can share a model run.

    Streams on the same batch-capable backend (the shared ONNX one) are scored in
    one ``infer_batch`` call; a stream with its own scorer or a per-session torch
    backend is scored alone. Each stream's recurrent state advances exactly as
    :meth:`SileroStreamingVAD.process` would advance it. An entry is the raw
    probability, or the exception that stream's scoring raised -- a failure stays
    with the streams it belongs to.
    """
    scores: list[float | BaseException] = [0.0] * len(requests)
    batches: dict[int, tuple[Callable[..., tuple[list[float], list[object]]], list[int]]] = {}
    for index, (vad, frame) in enumerate(requests):
        try:
            backend = None if vad._frame_scorer is not None else vad._require_backend()
            infer_batch = getattr(backend, "infer_batch", None)
            if infer_batch is not None:
                batches.setdefault(id(backend), (infer_batch, []))[1].append(index)
            else:
                scores[index] = vad._score_frame(frame)
        except Exception as exc:
            scores[index] = exc
    for infer_batch, indices in batches.values():
        streams = [requests[index][0] for index in indices]
        try:
            probabilities, states = infer_batch(
                [requests[index][1] for index in indices], [vad._detector_state for vad in streams]
            )
        except Exception as exc:
            for index in indices:
                scores[index] = exc
            continue
        for index, vad, probability, state in zip(indices, streams, probabilities, states):
            vad._detector_state = state
            scores[index] = probability
    return scores
