# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Server-side turn detection (``turn_detection: server_vad``) for the session runner.

Moved from the Realtime input translator: the runner owns one
``ServerTurnDetector`` per session, feeds each appended chunk to it and emits
``input_audio_buffer.speech_started`` / ``speech_stopped`` from the result.
The two-phase ``session.update`` commit (prepare / commit / reject) that used
to live on the translator is expressed by :class:`PendingTurnDetectionUpdate`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from vllm_omni.engine.duplex.vad import (
    SILERO_VAD_MIN_THRESHOLD,
    ServerVADUnavailableError,
    SileroStreamingVAD,
    SileroVADConfig,
    StreamingVADResult,
)

__all__ = [
    "PendingTurnDetectionUpdate",
    "ServerTurnDetector",
    "ServerVADUnavailableError",
    "TurnDetectionConfig",
    "TurnDetectionResult",
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


def validate_realtime_turn_detection(session_payload: Mapping[str, object]) -> str | None:
    """Validate the ``turn_detection`` object of a Realtime session payload (None when valid)."""
    field, turn_detection = configured_realtime_turn_detection(session_payload)
    if field is None:
        if session_payload.get("overlap_policy") == "barge_in_on_speech":
            return (
                "overlap_policy='barge_in_on_speech' requires turn_detection.type='server_vad' on the Realtime endpoint"
            )
        return None
    if turn_detection is not None and not isinstance(turn_detection, dict):
        return f"{field} must be null or an object with type='server_vad'"
    if isinstance(turn_detection, dict):
        if turn_detection.get("type") != "server_vad":
            return f"{field}.type must be 'server_vad'"
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
    """Normalized ``server_vad`` configuration (all fields filled with defaults)."""

    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    min_speech_duration_ms: int = 96
    create_response: bool = True
    interrupt_response: bool = True

    @classmethod
    def from_realtime(cls, turn_detection: Mapping[str, object]) -> TurnDetectionConfig:
        return cls(
            threshold=float(turn_detection.get("threshold", 0.5)),
            prefix_padding_ms=int(turn_detection.get("prefix_padding_ms", 300)),
            silence_duration_ms=int(turn_detection.get("silence_duration_ms", 500)),
            min_speech_duration_ms=int(turn_detection.get("min_speech_duration_ms", 96)),
            create_response=bool(turn_detection.get("create_response", True)),
            interrupt_response=bool(turn_detection.get("interrupt_response", True)),
        )

    def as_realtime(self) -> dict[str, object]:
        return {
            "type": "server_vad",
            "interrupt_response": self.interrupt_response,
            "create_response": self.create_response,
            "threshold": self.threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
            "min_speech_duration_ms": self.min_speech_duration_ms,
        }

    @property
    def overlap_policy(self) -> str:
        return "barge_in_on_speech"

    def build_detector(self) -> ServerTurnDetector:
        return ServerTurnDetector(self)


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

    def as_payload(self) -> dict[str, object]:
        return {
            "backend": "silero",
            "is_speech": self.is_speech,
            "speech_active": self.speech_active,
            "speech_started": self.speech_started,
            "speech_stopped": self.speech_stopped,
            "speech_probability": self.speech_probability,
        }


class ServerTurnDetector:
    """One Silero-backed streaming detector per session."""

    def __init__(self, config: TurnDetectionConfig) -> None:
        self.config = config
        self._vad = SileroStreamingVAD(
            SileroVADConfig(
                threshold=float(config.threshold),
                prefix_padding_ms=int(config.prefix_padding_ms),
                silence_duration_ms=int(config.silence_duration_ms),
                min_speech_duration_ms=max(32, int(config.min_speech_duration_ms)),
            )
        )

    @property
    def vad(self) -> SileroStreamingVAD:
        return self._vad

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
        """Score one appended chunk. Raises ``ServerVADUnavailableError`` / ``ValueError`` like the VAD."""
        result: StreamingVADResult = self._vad.process_base64(base64_audio, fmt=fmt, sample_rate_hz=sample_rate_hz)
        stop_ms = result.speech_end_ms if result.speech_end_ms is not None else audio_end_ms
        return TurnDetectionResult(
            is_speech=result.is_speech,
            speech_active=result.speech_active,
            speech_started=result.speech_started,
            speech_stopped=result.speech_stopped,
            speech_probability=result.speech_probability,
            audio_start_ms=result.speech_start_ms,
            audio_end_ms=stop_ms,
            should_commit=bool(result.speech_stopped and not result.speech_active),
            create_response=self.config.create_response,
        )


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
    """A ``session.update`` turn-detection change awaiting the engine ACK (two-phase)."""

    config: TurnDetectionConfig | None
    detector: ServerTurnDetector | None

    @classmethod
    def prepare(cls, session_payload: dict[str, object]) -> PendingTurnDetectionUpdate | None:
        """Normalize the payload in place and stage the new detector; None when not configured."""
        configured, config = normalize_turn_detection_session_payload(session_payload)
        if not configured:
            return None
        return cls(config=config, detector=config.build_detector() if config is not None else None)

    def commit(
        self, current: ServerTurnDetector | None
    ) -> tuple[TurnDetectionConfig | None, ServerTurnDetector | None]:
        """Adopt the staged detector; the replaced one is reset. Returns the new (config, detector)."""
        if current is not None:
            current.reset()
        return self.config, self.detector

    def reject(self) -> None:
        if self.detector is not None:
            self.detector.reset()
