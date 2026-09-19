# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Commit-only PCM buffer for AURA duplex (emit whole utterance on commit)."""

from __future__ import annotations

import binascii

import pybase64 as base64

from vllm_omni.engine.duplex.plugin import PcmAppendBuffer, PcmAppendReservation

_SAMPLE_BYTES = 4
_DEFAULT_SAMPLE_RATE_HZ = 16000


class AuraPcmAppendReservation(PcmAppendReservation):
    __slots__ = ("_active", "_owner", "_raw", "operation_id", "payload")

    def __init__(
        self,
        *,
        owner: AuraPcmAppendBuffer,
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


class AuraPcmAppendBuffer(PcmAppendBuffer):
    """Accumulate PCM until commit; ignore 1s auto-flush unit semantics."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._sample_rate_hz: int | None = None
        self._frame_queue: list[str] = []
        self._reservations: list[AuraPcmAppendReservation] = []
        self._had_speech = False

    @property
    def pending_byte_count(self) -> int:
        # Include retained video frame wire sizes so admission/release accounting
        # stays balanced after the mailbox releases the full append reservation.
        return len(self._buffer) + sum(len(frame) for frame in self._frame_queue)

    def clear(self) -> None:
        self._buffer.clear()
        self._frame_queue.clear()
        self._reservations.clear()
        self._had_speech = False

    def clear_force_listen(self) -> None:
        return

    def has_pending(self) -> bool:
        return bool(self._buffer) or bool(self._frame_queue)

    def has_reserved(self) -> bool:
        return any(reservation.active for reservation in self._reservations)

    def prepare_append(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        chunk_period_ms: int,
        allow_emit: bool,
    ) -> AuraPcmAppendReservation | None:
        del chunk_period_ms, allow_emit, operation_id
        fmt = payload.get("format")
        sample_rate_hz = payload.get("sample_rate_hz")
        audio = payload.get("audio")
        frames_in = payload.get("video_frames")
        if isinstance(frames_in, list):
            latest = [frame for frame in frames_in if isinstance(frame, str) and frame]
            if latest:
                self._frame_queue = latest[-2:]

        if fmt != "pcm_f32le" or not isinstance(sample_rate_hz, int) or not isinstance(audio, str):
            self._had_speech = self._had_speech or bool(payload.get("is_speech", False))
            return None

        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
            return None
        if len(raw) % _SAMPLE_BYTES:
            return None

        if self._sample_rate_hz is not None and self._sample_rate_hz != sample_rate_hz:
            raise ValueError("AURA duplex audio append sample_rate_hz changed within a session")
        self._sample_rate_hz = sample_rate_hz
        self._buffer.extend(raw)
        self._had_speech = self._had_speech or bool(payload.get("is_speech", True))
        return None

    def prepare_commit(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> AuraPcmAppendReservation:
        del chunk_period_ms
        if not self._buffer and not self._frame_queue:
            reservation = AuraPcmAppendReservation(
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
            "audio": base64.b64encode(raw).decode("ascii") if raw else "",
            "format": "pcm_f32le",
            "sample_rate_hz": sample_rate_hz,
            "final": True,
            # Do not infer speech from non-empty PCM: vision-follow often sends
            # 10ms zeros with is_speech=False so Stage0/1 treat it as vision-only
            # rather than a new spoken utterance.
            "is_speech": self._had_speech,
            "aura_turn_commit": True,
        }
        if self._frame_queue:
            payload["video_frames"] = list(self._frame_queue)
            self._frame_queue.clear()
        self._had_speech = False
        reservation = AuraPcmAppendReservation(
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


__all__ = ["AuraPcmAppendBuffer", "AuraPcmAppendReservation"]
