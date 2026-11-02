# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Server-side turn detection (``server_vad`` / ``semantic_vad``) for the session runner.

Moved from the Realtime input translator: the runner owns one detector per
session, feeds each appended chunk to it and emits
``input_audio_buffer.speech_started`` / ``speech_stopped`` from the result.
``server_vad`` (:class:`ServerTurnDetector`) ends the turn after a fixed
silence; ``semantic_vad`` (:class:`SemanticTurnDetector`) asks Smart Turn
whether the turn is complete when Silero hears a short pause.
The two-phase ``session.update`` commit (prepare / commit) that used
to live on the translator is expressed by :class:`PendingTurnDetectionUpdate`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pybase64 as base64
from vllm.logger import init_logger

from vllm_omni.engine.duplex.smart_turn import (
    SMART_TURN_WINDOW_SAMPLES,
    SemanticVADUnavailableError,
    SmartTurnBackendProvider,
)
from vllm_omni.engine.duplex.vad import (
    SILERO_VAD_MIN_THRESHOLD,
    ServerVADUnavailableError,
    SileroStreamingVAD,
    SileroVADBackendProvider,
    SileroVADConfig,
    StreamingVADResult,
)

logger = init_logger(__name__)

__all__ = [
    "PendingTurnDetectionUpdate",
    "SemanticTurnDetector",
    "SemanticVADUnavailableError",
    "SileroVADBackendProvider",
    "ServerTurnDetector",
    "ServerVADUnavailableError",
    "SmartTurnBackendProvider",
    "TurnDetectionConfig",
    "TurnDetectionResult",
    "TurnDetectionSegment",
    "TurnDetector",
    "apply_turn_detection_result",
    "configured_realtime_turn_detection",
    "normalize_turn_detection_session_payload",
    "validate_realtime_turn_detection",
]


def configured_realtime_turn_detection(session_payload: Mapping[str, object]) -> tuple[str | None, object]:
    """Return ``(field_path, value)`` for the turn detection setting present in the payload."""
    field = "turn_detection" if "turn_detection" in session_payload else None
    selected = session_payload.get("turn_detection")
    audio_config = session_payload.get("audio")
    if isinstance(audio_config, dict):
        audio_input = audio_config.get("input")
        if isinstance(audio_input, dict) and "turn_detection" in audio_input:
            field, selected = "audio.input.turn_detection", audio_input["turn_detection"]
    return field, selected


def _valid_vad_number(value: object, *, minimum: float, maximum: float | None = None, strict: bool = False) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and (float(value) > minimum if strict else float(value) >= minimum)
        and (maximum is None or float(value) <= maximum)
    )


#: Every field a ``server_vad`` object may carry. Unknown keys are refused
#: rather than ignored: a misspelled tuning knob that silently does nothing is
#: worse than a rejected session, because the endpointing still *looks* applied.
_SERVER_VAD_FIELDS = frozenset(
    {
        "type",
        "threshold",
        "prefix_padding_ms",
        "silence_duration_ms",
        "create_response",
        "interrupt_response",
        "min_speech_duration_ms",
    }
)

#: ``semantic_vad`` takes OpenAI's shape: eagerness instead of the Silero knobs.
_SEMANTIC_VAD_FIELDS = frozenset({"type", "eagerness", "create_response", "interrupt_response"})

#: Silence that makes a pause a candidate end of turn, by ``eagerness``
#: (``auto`` is ``medium``). The default favors keeping clause pauses inside
#: a turn: Smart Turn can confidently classify a short prefix as complete.
#: Waiting longer gives speech a chance to resume before that decision, at
#: the cost of the same additional latency on complete utterances. ``high``
#: remains the low-latency option. These delays do not guarantee semantic
#: correctness; the classifier can still mistake a longer thinking pause.
SEMANTIC_VAD_CANDIDATE_SILENCE_MS = {"high": 300, "medium": 1200, "auto": 1200, "low": 1500}
#: Longest silence ``semantic_vad`` waits through after Smart Turn calls the
#: turn incomplete, measured from the start of the pause.
SEMANTIC_VAD_MAX_SILENCE_MS = {"high": 2000, "medium": 4000, "auto": 4000, "low": 8000}
#: Smart Turn probability at or above which a candidate pause ends the turn.
SEMANTIC_VAD_COMPLETION_THRESHOLD = 0.5


def validate_realtime_turn_detection(session_payload: Mapping[str, object]) -> str | None:
    """Validate the ``turn_detection`` object of a Realtime session payload (None when valid)."""
    field, turn_detection = configured_realtime_turn_detection(session_payload)
    if field is None:
        if session_payload.get("overlap_policy") == "barge_in_on_speech":
            return (
                "overlap_policy='barge_in_on_speech' requires turn_detection.type='server_vad' or "
                "'semantic_vad' on the Realtime endpoint"
            )
        return None
    if turn_detection is not None and not isinstance(turn_detection, dict):
        return f"{field} must be null or an object with type='server_vad' or 'semantic_vad'"
    if isinstance(turn_detection, dict):
        detector_type = turn_detection.get("type")
        if detector_type not in ("server_vad", "semantic_vad"):
            return f"{field}.type must be 'server_vad' or 'semantic_vad'"
        allowed = _SERVER_VAD_FIELDS if detector_type == "server_vad" else _SEMANTIC_VAD_FIELDS
        unknown = sorted(set(turn_detection) - allowed)
        if unknown:
            return f"Unknown {field} field(s): {', '.join(unknown)}"
        if detector_type == "semantic_vad":
            eagerness = turn_detection.get("eagerness", "auto")
            if not isinstance(eagerness, str) or eagerness not in SEMANTIC_VAD_MAX_SILENCE_MS:
                return f"{field}.eagerness must be one of 'low', 'medium', 'high' or 'auto'"
        threshold = turn_detection.get("threshold", 0.5)
        if not _valid_vad_number(threshold, minimum=SILERO_VAD_MIN_THRESHOLD, strict=True, maximum=1):
            return f"{field}.threshold must be greater than {SILERO_VAD_MIN_THRESHOLD} and at most 1"
        for name in ("prefix_padding_ms", "silence_duration_ms", "min_speech_duration_ms"):
            value = turn_detection.get(name)
            if value is not None and not _valid_vad_number(value, minimum=0):
                return f"{field}.{name} must be a non-negative number"
        if turn_detection.get("interrupt_response", True) is not True:
            return (
                f"{field}.interrupt_response=false is unsupported; use turn_detection=null for model-owned listen/speak"
            )
    desired_policy = "barge_in_on_speech" if turn_detection is not None else "listen_only"
    overlap_policy = session_payload.get("overlap_policy")
    if isinstance(overlap_policy, str) and overlap_policy != desired_policy:
        return f"overlap_policy={overlap_policy!r} conflicts with turn_detection; expected {desired_policy!r}"
    return None


@dataclass(frozen=True, slots=True)
class TurnDetectionConfig:
    """Normalized ``server_vad`` / ``semantic_vad`` configuration (all fields filled with defaults).

    ``semantic_vad`` uses the Silero fields only internally: the wire object
    carries ``eagerness`` instead, and the defaults here apply.
    """

    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    min_speech_duration_ms: int = 96
    create_response: bool = True
    interrupt_response: bool = True
    type: str = "server_vad"
    eagerness: str = "auto"

    @classmethod
    def from_realtime(cls, turn_detection: Mapping[str, object]) -> TurnDetectionConfig:
        return cls(
            # Validated numbers (``validate_realtime_turn_detection``).
            threshold=float(cast("float", turn_detection.get("threshold", 0.5))),
            prefix_padding_ms=int(cast("float", turn_detection.get("prefix_padding_ms", 300))),
            silence_duration_ms=int(cast("float", turn_detection.get("silence_duration_ms", 500))),
            min_speech_duration_ms=int(cast("float", turn_detection.get("min_speech_duration_ms", 96))),
            create_response=bool(turn_detection.get("create_response", True)),
            interrupt_response=bool(turn_detection.get("interrupt_response", True)),
            type=str(turn_detection.get("type", "server_vad")),
            eagerness=str(turn_detection.get("eagerness", "auto")),
        )

    @property
    def overlap_policy(self) -> str:
        return "barge_in_on_speech"

    def build_detector(
        self,
        backend_provider: SileroVADBackendProvider | None = None,
        semantic_backend_provider: SmartTurnBackendProvider | None = None,
    ) -> ServerTurnDetector | SemanticTurnDetector:
        if self.type == "semantic_vad":
            return SemanticTurnDetector(
                self,
                backend_provider=backend_provider,
                smart_turn_provider=semantic_backend_provider,
            )
        return ServerTurnDetector(self, backend_provider=backend_provider)


def normalize_turn_detection_session_payload(
    session_payload: dict[str, object],
) -> tuple[bool, TurnDetectionConfig | None]:
    """Fill ``turn_detection`` defaults and derive ``overlap_policy`` in place.

    Returns ``(configured, config)`` where ``configured`` is False when the
    payload does not mention turn detection at all (nothing changes), and
    ``config`` is ``None`` for ``turn_detection: null`` (model-owned turns,
    ``overlap_policy`` forced to ``listen_only``).
    """
    field, configured = configured_realtime_turn_detection(session_payload)
    if field is None:
        return False, None
    if configured is None:
        session_payload["overlap_policy"] = "listen_only"
        return True, None
    assert isinstance(configured, dict)
    if configured.get("type") == "semantic_vad":
        merged: dict[str, object] = {
            "type": "semantic_vad",
            "eagerness": "auto",
            "create_response": True,
            "interrupt_response": True,
            **configured,
        }
    else:
        merged = {
            "type": "server_vad",
            "interrupt_response": True,
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
            "min_speech_duration_ms": 96,
            **configured,
        }
    config = TurnDetectionConfig.from_realtime(merged)
    session_payload["turn_detection"] = dict(merged)
    session_payload["overlap_policy"] = config.overlap_policy
    return True, config


SemanticDecision = Literal["smart_turn_complete", "smart_turn_incomplete", "smart_turn_error"]


@dataclass(frozen=True, slots=True)
class TurnDetectionResult:
    is_speech: bool
    speech_active: bool
    speech_started: bool = False
    speech_stopped: bool = False
    speech_probability: float = 0.0
    audio_start_ms: int | None = None
    audio_end_ms: int | None = None
    #: A ``speech_stopped`` under ``server_vad`` ends the user turn.
    should_commit: bool = False
    #: Whether the auto-commit should also request a response (``create_response``).
    create_response: bool = True
    backend: str = "silero"
    #: Smart Turn's P(turn complete) when this chunk ended a pause (``semantic_vad`` only).
    turn_complete_probability: float | None = None
    #: Actual Smart Turn scoring attempts, including failures, summed across frames.
    turn_evaluations: int = 0
    #: Internal diagnostics, emitted on candidate evaluation and final commit only.
    semantic_decision: SemanticDecision | None = None
    commit_reason: Literal["semantic_complete", "silence_timeout"] | None = None

    def as_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "backend": self.backend,
            "is_speech": self.is_speech,
            "speech_active": self.speech_active,
            "speech_started": self.speech_started,
            "speech_stopped": self.speech_stopped,
            "speech_probability": self.speech_probability,
        }
        if self.turn_complete_probability is not None:
            payload["turn_complete_probability"] = self.turn_complete_probability
        return payload


@dataclass(frozen=True, slots=True)
class TurnDetectionSegment:
    """Normalized PCM and its decision, ending no later than one turn boundary."""

    audio: str
    result: TurnDetectionResult


def _process_frames(detector: TurnDetector, samples: np.ndarray) -> list[tuple[int, TurnDetectionResult]]:
    """Advance detector state one Silero frame boundary at a time."""
    frames: list[tuple[int, TurnDetectionResult]] = []
    offset = 0
    while offset < samples.size:
        end = min(samples.size, offset + detector.vad.samples_until_frame)
        frames.append((end, detector._process_samples(samples[offset:end])))
        offset = end
    return frames


def _process_segments(
    detector: TurnDetector,
    base64_audio: str,
    *,
    fmt: str,
    sample_rate_hz: int | None,
) -> list[TurnDetectionSegment]:
    """Score on Silero's frame grid, retaining every boundary in a wire append.

    Only audio through the current frame may reach Smart Turn. Flush a segment
    at each commit so the runner can commit it before buffering the next turn.
    The final partial frame stays in the detector for the next append.
    """
    samples = detector.vad.decode_base64(base64_audio, fmt=fmt, sample_rate_hz=sample_rate_hz)
    segments: list[TurnDetectionSegment] = []
    start = 0
    accumulated: TurnDetectionResult | None = None
    for offset, result in _process_frames(detector, samples):
        if accumulated is not None:
            result = TurnDetectionResult(
                is_speech=accumulated.is_speech or result.is_speech,
                speech_active=result.speech_active,
                speech_started=accumulated.speech_started or result.speech_started,
                speech_stopped=result.speech_stopped,
                speech_probability=max(accumulated.speech_probability, result.speech_probability),
                audio_start_ms=accumulated.audio_start_ms if accumulated.speech_started else result.audio_start_ms,
                audio_end_ms=result.audio_end_ms,
                should_commit=result.should_commit,
                create_response=result.create_response,
                backend=result.backend,
                turn_complete_probability=(
                    result.turn_complete_probability
                    if result.turn_complete_probability is not None or result.semantic_decision == "smart_turn_error"
                    else accumulated.turn_complete_probability
                ),
                turn_evaluations=accumulated.turn_evaluations + result.turn_evaluations,
                semantic_decision=result.semantic_decision or accumulated.semantic_decision,
                commit_reason=result.commit_reason,
            )
        accumulated = result
        if result.should_commit or offset == samples.size:
            audio = base64.b64encode(samples[start:offset].astype("<f4", copy=False).tobytes()).decode("ascii")
            segments.append(TurnDetectionSegment(audio, result))
            start = offset
            accumulated = None
    return segments


class ServerTurnDetector:
    """One Silero-backed streaming detector per session."""

    def __init__(
        self,
        config: TurnDetectionConfig,
        *,
        backend_provider: SileroVADBackendProvider | None = None,
    ) -> None:
        self.config = config
        self._vad = SileroStreamingVAD(
            SileroVADConfig(
                threshold=float(config.threshold),
                prefix_padding_ms=int(config.prefix_padding_ms),
                silence_duration_ms=int(config.silence_duration_ms),
                min_speech_duration_ms=max(32, int(config.min_speech_duration_ms)),
            ),
            backend_provider=backend_provider,
        )

    @property
    def vad(self) -> SileroStreamingVAD:
        return self._vad

    def reconfigure(self, config: TurnDetectionConfig) -> None:
        """Apply updated settings without discarding the current audio stream state."""
        self.config = config
        self._vad.config = SileroVADConfig(
            threshold=float(config.threshold),
            prefix_padding_ms=int(config.prefix_padding_ms),
            silence_duration_ms=int(config.silence_duration_ms),
            min_speech_duration_ms=max(32, int(config.min_speech_duration_ms)),
        )

    def reset(self) -> None:
        self._vad.reset()

    def process(
        self,
        base64_audio: str,
        *,
        fmt: str,
        sample_rate_hz: int | None,
        audio_end_ms: int | None = None,
    ) -> TurnDetectionResult:
        """Return one aggregate result for the turn-based Realtime caller.

        That caller owns commit/reset and coalesces stop/restart within an
        append. Duplex uses ``process_segments`` to retain every turn boundary.
        """
        result: StreamingVADResult = self._vad.process_base64(base64_audio, fmt=fmt, sample_rate_hz=sample_rate_hz)
        return TurnDetectionResult(
            is_speech=result.is_speech,
            speech_active=result.speech_active,
            speech_started=result.speech_started,
            speech_stopped=result.speech_stopped,
            speech_probability=result.speech_probability,
            audio_start_ms=result.speech_start_ms,
            audio_end_ms=result.speech_end_ms if result.speech_end_ms is not None else audio_end_ms,
            should_commit=bool(result.speech_stopped and not result.speech_active),
            create_response=self.config.create_response,
        )

    def process_segments(
        self, base64_audio: str, *, fmt: str, sample_rate_hz: int | None
    ) -> list[TurnDetectionSegment]:
        return _process_segments(self, base64_audio, fmt=fmt, sample_rate_hz=sample_rate_hz)

    def _process_samples(self, samples: np.ndarray) -> TurnDetectionResult:
        result: StreamingVADResult = self._vad.process(samples)
        detected = TurnDetectionResult(
            is_speech=result.is_speech,
            speech_active=result.speech_active,
            speech_started=result.speech_started,
            speech_stopped=result.speech_stopped,
            speech_probability=result.speech_probability,
            audio_start_ms=result.speech_start_ms,
            audio_end_ms=result.speech_end_ms,
            should_commit=bool(result.speech_stopped and not result.speech_active),
            create_response=self.config.create_response,
        )
        if detected.should_commit:
            self.reset()
        return detected


class _TurnAudio:
    """Ring buffer of the most recent 8 s of 16 kHz input (Smart Turn's window)."""

    def __init__(self, capacity: int = SMART_TURN_WINDOW_SAMPLES) -> None:
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._write = 0
        self._filled = 0

    def append(self, samples: np.ndarray) -> None:
        capacity = self._buffer.size
        samples = samples[-capacity:]
        head = min(samples.size, capacity - self._write)
        self._buffer[self._write : self._write + head] = samples[:head]
        self._buffer[: samples.size - head] = samples[head:]
        self._write = (self._write + samples.size) % capacity
        self._filled = min(capacity, self._filled + samples.size)

    def last(self, count: int) -> np.ndarray:
        """The newest ``count`` samples (at most the window), oldest first."""
        count = min(max(0, count), self._filled)
        start = self._write - count
        if start >= 0:
            return self._buffer[start : self._write].copy()
        return np.concatenate((self._buffer[start:], self._buffer[: self._write]))


class SemanticTurnDetector:
    """``semantic_vad``: Silero hears pauses, Smart Turn decides whether they end the turn.

    Silero runs with a short, eagerness-dependent silence; its stop is only a *candidate*
    pause and never reaches the client. On each candidate Smart Turn scores the
    turn so far: a complete turn commits at once, an incomplete one keeps
    listening until speech resumes (same turn, no second ``speech_started``) or
    the silence reaches the eagerness limit. While paused the turn stays
    active, so the pause audio is buffered into the same utterance.

    Every call happens on the session's ordered mailbox, so a decision can never
    race a reset; the clock is samples received, so transport gaps are not silence.
    """

    backend_name = "silero+smart_turn"

    def __init__(
        self,
        config: TurnDetectionConfig,
        *,
        backend_provider: SileroVADBackendProvider | None = None,
        smart_turn_provider: SmartTurnBackendProvider | None = None,
        turn_scorer: Callable[[np.ndarray], float] | None = None,
        frame_scorer: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        self.config = config
        eagerness = config.eagerness if config.eagerness in SEMANTIC_VAD_MAX_SILENCE_MS else "auto"
        self._candidate_silence_ms = SEMANTIC_VAD_CANDIDATE_SILENCE_MS[eagerness]
        self._max_silence_ms = SEMANTIC_VAD_MAX_SILENCE_MS[eagerness]
        self._vad = SileroStreamingVAD(
            SileroVADConfig(
                threshold=float(config.threshold),
                prefix_padding_ms=int(config.prefix_padding_ms),
                silence_duration_ms=self._candidate_silence_ms,
                min_speech_duration_ms=max(32, int(config.min_speech_duration_ms)),
            ),
            backend_provider=backend_provider,
            frame_scorer=frame_scorer,
        )
        self._smart_turn_provider = smart_turn_provider
        self._turn_scorer = turn_scorer
        self._audio = _TurnAudio()
        self._clear_turn()

    def _clear_turn(self) -> None:
        self._turn_active = False
        self._turn_start_ms = 0
        #: Session ms where the current candidate pause began; None while speaking.
        self._pause_start_ms: int | None = None
        self._pause_decision: SemanticDecision | None = None

    @property
    def vad(self) -> SileroStreamingVAD:
        return self._vad

    @property
    def candidate_silence_ms(self) -> int:
        return self._candidate_silence_ms

    @property
    def max_silence_ms(self) -> int:
        return self._max_silence_ms

    def reset(self) -> None:
        self._vad.reset()
        self._clear_turn()

    def process_segments(
        self, base64_audio: str, *, fmt: str, sample_rate_hz: int | None
    ) -> list[TurnDetectionSegment]:
        return _process_segments(self, base64_audio, fmt=fmt, sample_rate_hz=sample_rate_hz)

    def process(
        self,
        base64_audio: str,
        *,
        fmt: str,
        sample_rate_hz: int | None,
        audio_end_ms: int | None = None,
    ) -> TurnDetectionResult:
        """Aggregate one appended chunk for the turn-based Realtime caller.

        Semantic state advances on Silero's frame grid so a large append cannot
        hide an earlier onset or candidate pause. The aggregate then coalesces a
        completed turn followed by new speech, matching the legacy Realtime API.
        """
        samples = self._vad.decode_base64(base64_audio, fmt=fmt, sample_rate_hz=sample_rate_hz)
        frames = _process_frames(self, samples)
        if not frames:
            return self._process_samples(samples)

        results = [result for _, result in frames]
        last = results[-1]
        committed = [result for result in results if result.should_commit]
        should_commit = bool(committed) and not last.speech_active
        selected = committed[-1] if should_commit else last
        first_started = next((result for result in results if result.speech_started), last)
        return TurnDetectionResult(
            is_speech=any(result.is_speech for result in results),
            speech_active=last.speech_active,
            speech_started=any(result.speech_started for result in results),
            speech_stopped=bool(committed),
            speech_probability=max(result.speech_probability for result in results),
            audio_start_ms=first_started.audio_start_ms,
            audio_end_ms=selected.audio_end_ms,
            should_commit=should_commit,
            create_response=last.create_response,
            backend=last.backend,
            turn_complete_probability=selected.turn_complete_probability,
            turn_evaluations=sum(result.turn_evaluations for result in results),
            semantic_decision=selected.semantic_decision,
            commit_reason=selected.commit_reason,
        )

    def reconfigure(self, config: TurnDetectionConfig) -> None:
        """Apply updated semantic settings while preserving the current turn."""
        self.config = config
        eagerness = config.eagerness if config.eagerness in SEMANTIC_VAD_MAX_SILENCE_MS else "auto"
        self._candidate_silence_ms = SEMANTIC_VAD_CANDIDATE_SILENCE_MS[eagerness]
        self._max_silence_ms = SEMANTIC_VAD_MAX_SILENCE_MS[eagerness]
        self._vad.config = SileroVADConfig(
            threshold=float(config.threshold),
            prefix_padding_ms=int(config.prefix_padding_ms),
            silence_duration_ms=self._candidate_silence_ms,
            min_speech_duration_ms=max(32, int(config.min_speech_duration_ms)),
        )

    def _process_samples(self, samples: np.ndarray) -> TurnDetectionResult:
        """Run the shared semantic state machine on decoded audio samples.

        Raises ``SemanticVADUnavailableError`` / ``ServerVADUnavailableError`` /
        ``ValueError`` like the server VAD. A Smart Turn *inference* failure does
        not raise: it is logged and the pause counts as incomplete, so the
        eagerness limit still ends the turn.
        """
        scorer = self._require_turn_scorer()
        self._audio.append(samples)
        inner = self._vad.process(samples)
        was_active = self._turn_active

        started = False
        start_ms: int | None = None
        if inner.speech_started and not self._turn_active:
            self._turn_active = True
            started = True
            start_ms = inner.speech_start_ms
            self._turn_start_ms = start_ms or 0

        probability: float | None = None
        evaluations = 0
        decision: SemanticDecision | None = None
        commit_reason: Literal["semantic_complete", "silence_timeout"] | None = None
        commit = False
        if self._turn_active:
            if inner.speech_active:
                # Speaking, or speech resumed inside a pause: still the same turn.
                self._pause_start_ms = None
                self._pause_decision = None
            elif inner.speech_stopped:
                # Silero saw the candidate silence. Decide on the whole turn so
                # far; a resumed-then-paused turn is re-scored with more context.
                # Use Silero's actual pause start. Its hysteresis band can
                # delay speech_stopped beyond the minimum candidate silence;
                # subtracting that minimum would move the timeout deadline.
                self._pause_start_ms = inner.silence_start_ms
                try:
                    window = self._turn_window()
                    evaluations = 1
                    probability = float(scorer(window))
                except Exception:
                    # Degrade to the eagerness silence limit rather than drop
                    # or prematurely commit the turn.
                    logger.warning("Smart Turn inference failed; treating the pause as incomplete", exc_info=True)
                commit = probability is not None and probability >= SEMANTIC_VAD_COMPLETION_THRESHOLD
                decision = (
                    "smart_turn_error"
                    if probability is None
                    else "smart_turn_complete"
                    if commit
                    else "smart_turn_incomplete"
                )
                self._pause_decision = decision
                if commit:
                    commit_reason = "semantic_complete"
            if (
                not commit
                and self._pause_start_ms is not None
                # Let an onset finish confirmation before timing out. If it
                # was only a short noise burst, Silero clears the candidate
                # and the original silence deadline applies immediately.
                and not self._vad.speech_pending
                and self._vad.processed_ms - self._pause_start_ms >= self._max_silence_ms
            ):
                commit = True
                decision = self._pause_decision
                commit_reason = "silence_timeout"

        end_ms = self._vad.processed_ms if commit else None
        result = TurnDetectionResult(
            # The pause belongs to the utterance: keep buffering it.
            is_speech=inner.is_speech or was_active or self._turn_active,
            speech_active=self._turn_active and not commit,
            speech_started=started,
            speech_stopped=commit,
            speech_probability=inner.speech_probability,
            audio_start_ms=start_ms,
            audio_end_ms=end_ms,
            should_commit=commit,
            create_response=self.config.create_response,
            backend=self.backend_name,
            turn_complete_probability=probability,
            turn_evaluations=evaluations,
            semantic_decision=decision,
            commit_reason=commit_reason,
        )
        if commit:
            self.reset()
        return result

    def _turn_window(self) -> np.ndarray:
        """The turn so far: from its (padded) start on Silero's clock to the newest sample.

        Measured on Silero's clock, plus the partial frame it has not scored
        yet, so a reset (which drops that partial frame) cannot make the window
        drift into the previous turn.
        """
        turn_samples = (self._vad.processed_ms - self._turn_start_ms) * 16
        return self._audio.last(turn_samples + self._vad.scratch_bytes // 4)

    def _require_turn_scorer(self) -> Callable[[np.ndarray], float]:
        """Load Smart Turn with the first chunk, off the endpoint path."""
        if self._turn_scorer is None:
            provider = self._smart_turn_provider or SmartTurnBackendProvider()
            self._turn_scorer = provider.get().predict
        return self._turn_scorer


TurnDetector = ServerTurnDetector | SemanticTurnDetector


def apply_turn_detection_result(payload: dict[str, object], result: TurnDetectionResult) -> None:
    """Merge a detector result onto an internal ``input_audio_buffer.append`` payload.

    Mirrors what the old translator attached: ``is_speech`` from the detector,
    the ``vad`` hint block, and ``force_listen`` while speech is active.
    """
    payload["is_speech"] = result.is_speech
    payload["vad"] = result.as_payload()
    if result.speech_active:
        payload["force_listen"] = True
    if result.audio_start_ms is not None and result.speech_started:
        payload["audio_start_ms"] = result.audio_start_ms
    if result.audio_end_ms is not None and result.speech_stopped:
        payload["audio_end_ms"] = result.audio_end_ms


@dataclass
class PendingTurnDetectionUpdate:
    """A turn-detection candidate adopted after ``session.update`` validation.

    Staged detectors have not processed audio or loaded a backend, so rejected
    candidates need no cleanup.
    """

    config: TurnDetectionConfig | None
    detector: TurnDetector | None

    @classmethod
    def prepare(
        cls,
        session_payload: dict[str, object],
        *,
        backend_provider: SileroVADBackendProvider | None = None,
        semantic_backend_provider: SmartTurnBackendProvider | None = None,
    ) -> PendingTurnDetectionUpdate | None:
        """Normalize the payload in place and stage the new detector; None when not configured."""
        configured, config = normalize_turn_detection_session_payload(session_payload)
        if not configured:
            return None
        detector = config.build_detector(backend_provider, semantic_backend_provider) if config is not None else None
        return cls(config=config, detector=detector)

    def commit(self, current: TurnDetector | None) -> tuple[TurnDetectionConfig | None, TurnDetector | None]:
        """Keep identical settings' live state; otherwise adopt the staged detector.

        Resetting the detector being replaced is best-effort. That detector is
        discarded either way, and a reset failure must not leave the new config
        live beside the old detector.
        """
        if current is not None and current.config == self.config:
            return self.config, current
        if current is not None:
            try:
                current.reset()
            except Exception:
                logger.exception("turn detection reset failed; adopting the staged detector")
        return self.config, self.detector
