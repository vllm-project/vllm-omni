# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Model-agnostic ephemeral turn-commit helpers.

The buffer accumulates PCM and emits one utterance on commit. Session
state is the runner-owned per-session bag the plugin contract requires.
"""

from __future__ import annotations

import asyncio
import binascii
from dataclasses import dataclass, field

import pybase64 as base64

from vllm_omni.engine.duplex.plugin import (
    DuplexModelSessionState,
    PcmAppendBuffer,
    PcmAppendReservation,
)

_SAMPLE_BYTES = 4
_DEFAULT_SAMPLE_RATE_HZ = 16000


class TurnCommitPcmAppendReservation(PcmAppendReservation):
    __slots__ = ("_active", "_owner", "_raw", "operation_id", "payload")

    def __init__(
        self,
        *,
        owner: TurnCommitPcmAppendBuffer,
        operation_id: str,
        payload: dict[str, object] | None,
        raw: bytes,
    ) -> None:
        self._owner = owner
        self.operation_id = operation_id
        self.payload = payload
        self._raw = raw
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def byte_count(self) -> int:
        return len(self._raw)

    def commit(self) -> None:
        if not self._active:
            return
        self._active = False
        if self in self._owner._reservations:
            self._owner._reservations.remove(self)

    def rollback(self) -> None:
        if not self._active:
            return
        try:
            index = self._owner._reservations.index(self)
        except ValueError:
            self._active = False
            return
        restore = bytearray()
        for reservation in self._owner._reservations[index:]:
            if reservation._active:
                restore.extend(reservation._raw)
                reservation._active = False
        del self._owner._reservations[index:]
        self._owner._buffer[:0] = restore
        self._active = False


class TurnCommitPcmAppendBuffer(PcmAppendBuffer):
    """Accumulate PCM until commit; the framework's VAD owns turn boundaries."""

    def __init__(self, *, label: str = "turn-commit duplex") -> None:
        self._label = label
        self._buffer = bytearray()
        self._sample_rate_hz: int | None = None
        self._reservations: list[TurnCommitPcmAppendReservation] = []
        self._had_speech = False

    @property
    def pending_byte_count(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
        self._reservations.clear()
        self._had_speech = False

    def clear_force_listen(self) -> None:
        return

    def has_pending(self) -> bool:
        return bool(self._buffer)

    def has_reserved(self) -> bool:
        return any(reservation.active for reservation in self._reservations)

    def prepare_append(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        chunk_period_ms: int,
        allow_emit: bool,
    ) -> TurnCommitPcmAppendReservation | None:
        del chunk_period_ms, allow_emit, operation_id
        fmt = payload.get("format")
        sample_rate_hz = payload.get("sample_rate_hz")
        audio = payload.get("audio")
        if fmt != "pcm_f32le" or not isinstance(sample_rate_hz, int) or not isinstance(audio, str):
            raise ValueError(f"{self._label} append requires pcm_f32le audio and sample_rate_hz")

        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"{self._label} audio is not valid base64") from exc
        if len(raw) % _SAMPLE_BYTES:
            raise ValueError(f"{self._label} pcm_f32le payload has a partial sample")

        if self._sample_rate_hz is not None and self._sample_rate_hz != sample_rate_hz:
            raise ValueError(f"{self._label} audio append sample_rate_hz changed within a session")
        self._sample_rate_hz = sample_rate_hz
        self._buffer.extend(raw)
        self._had_speech = self._had_speech or bool(payload.get("is_speech", True))
        return None

    def prepare_commit(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> TurnCommitPcmAppendReservation:
        del chunk_period_ms
        if not self._buffer:
            reservation = TurnCommitPcmAppendReservation(
                owner=self,
                operation_id=operation_id,
                payload=None,
                raw=b"",
            )
            self._reservations.append(reservation)
            return reservation

        sample_rate_hz = self._sample_rate_hz or _DEFAULT_SAMPLE_RATE_HZ
        raw = bytes(self._buffer)
        self._buffer.clear()
        payload: dict[str, object] = {
            "type": "audio",
            "audio": base64.b64encode(raw).decode("ascii"),
            "format": "pcm_f32le",
            "sample_rate_hz": sample_rate_hz,
            "final": True,
            "is_speech": self._had_speech,
            "turn_commit": True,
        }
        self._had_speech = False
        reservation = TurnCommitPcmAppendReservation(
            owner=self,
            operation_id=operation_id,
            payload=payload,
            raw=raw,
        )
        self._reservations.append(reservation)
        return reservation

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        reservation = self.prepare_commit(
            operation_id="flush",
            chunk_period_ms=chunk_period_ms,
        )
        reservation.commit()
        return reservation.payload


@dataclass(slots=True)
class TurnCommitSessionState(DuplexModelSessionState):
    """Per-session turn-commit state (owned by SessionRunner)."""

    audio_buffer: TurnCommitPcmAppendBuffer = field(default_factory=TurnCommitPcmAppendBuffer)
    input_since_commit: bool = False
    speech_since_commit: bool = False
    context_locked: bool = False
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    continuation_owner_id: str | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    pending_silence_owner_id: str | None = None

    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None:
        self.committed_audio_payload = payload
        self.committed_audio_operation_id = operation_id
        self.committed_audio_reserved_bytes += max(0, int(reserved_bytes))

    def clear_committed_audio(self) -> int:
        reserved_bytes = self.committed_audio_reserved_bytes
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        self.deferred_response_create = False
        self.deferred_precreate_response = False
        return reserved_bytes

    def clear_continuation(self) -> None:
        self.continuation_owner_id = None
        self.continuation_units = 0
        self.pending_silence_task = None
        self.pending_silence_owner_id = None


__all__ = [
    "TurnCommitPcmAppendBuffer",
    "TurnCommitPcmAppendReservation",
    "TurnCommitSessionState",
]
