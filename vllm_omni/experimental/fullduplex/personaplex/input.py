# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import binascii

import numpy as np

_SAMPLE_RATE_HZ = 24000
_FRAME_SAMPLES = 1920
_BYTES_PER_SAMPLE = 4


class PersonaPlexPcmAppendReservation:
    __slots__ = (
        "_active",
        "_owner",
        "_raw",
        "operation_id",
        "payload",
    )

    def __init__(
        self,
        *,
        owner: PersonaPlexPcmAppendBuffer,
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
        self._owner._commit_reservation(self)

    def rollback(self) -> None:
        self._owner._rollback_reservation(self)


class PersonaPlexPcmAppendBuffer:
    """Transactionally frame 24 kHz float PCM into PersonaPlex 80 ms units."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._sample_rate_hz: int | None = None
        self._reservations: list[PersonaPlexPcmAppendReservation] = []

    @property
    def pending_byte_count(self) -> int:
        return len(self._buffer)

    def has_pending(self) -> bool:
        return bool(self._buffer)

    def has_reserved(self) -> bool:
        return any(reservation.active for reservation in self._reservations)

    def clear_force_listen(self) -> None:
        return

    def clear(self) -> None:
        for reservation in self._reservations:
            reservation._active = False
        self._reservations.clear()
        self._buffer.clear()
        self._sample_rate_hz = None

    def prepare_append(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        chunk_period_ms: int,
        allow_emit: bool,
    ) -> PersonaPlexPcmAppendReservation | None:
        if chunk_period_ms != 80:
            raise ValueError("PersonaPlex chunk_period_ms must be 80")
        if any(reservation.active and reservation.operation_id == operation_id for reservation in self._reservations):
            raise ValueError(f"PersonaPlex duplicate active operation_id: {operation_id}")

        raw = self._decode_payload(payload)
        self._buffer.extend(raw)
        if not allow_emit or len(self._buffer) < _FRAME_SAMPLES * _BYTES_PER_SAMPLE:
            return None
        return self._reserve_frame(payload, operation_id=operation_id, flush=False)

    def prepare_commit(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> PersonaPlexPcmAppendReservation:
        if chunk_period_ms != 80:
            raise ValueError("PersonaPlex chunk_period_ms must be 80")
        if not self._buffer:
            reservation = PersonaPlexPcmAppendReservation(
                owner=self,
                operation_id=operation_id,
                payload=None,
                raw=b"",
            )
            self._reservations.append(reservation)
            return reservation
        payload: dict[str, object] = {
            "type": "audio",
            "format": "pcm_f32le",
            "sample_rate_hz": self._sample_rate_hz or _SAMPLE_RATE_HZ,
            "audio": "",
            "final": True,
        }
        return self._reserve_frame(payload, operation_id=operation_id, flush=True)

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        reservation = self.prepare_commit(
            operation_id="personaplex-flush",
            chunk_period_ms=chunk_period_ms,
        )
        payload = reservation.payload
        reservation.commit()
        return payload

    def _decode_payload(self, payload: dict[str, object]) -> bytes:
        if payload.get("format") != "pcm_f32le":
            raise ValueError("PersonaPlex input format must be pcm_f32le")
        sample_rate_hz = payload.get("sample_rate_hz")
        if sample_rate_hz != _SAMPLE_RATE_HZ:
            raise ValueError("PersonaPlex sample_rate_hz must be 24000")
        if self._sample_rate_hz is not None and self._sample_rate_hz != sample_rate_hz:
            raise ValueError("PersonaPlex sample_rate_hz changed within a session")
        audio = payload.get("audio")
        if not isinstance(audio, str):
            raise ValueError("PersonaPlex audio must be base64 pcm_f32le")
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("PersonaPlex audio is not valid base64") from exc
        if len(raw) % _BYTES_PER_SAMPLE:
            raise ValueError("PersonaPlex pcm_f32le byte length must be divisible by four")
        samples = np.frombuffer(raw, dtype="<f4")
        if not np.isfinite(samples).all():
            raise ValueError("PersonaPlex pcm_f32le samples must be finite")
        self._sample_rate_hz = int(sample_rate_hz)
        return raw

    def _reserve_frame(
        self,
        source_payload: dict[str, object],
        *,
        operation_id: str,
        flush: bool,
    ) -> PersonaPlexPcmAppendReservation:
        frame_bytes = _FRAME_SAMPLES * _BYTES_PER_SAMPLE
        if flush:
            consumed_bytes = min(len(self._buffer), frame_bytes)
        else:
            consumed_bytes = frame_bytes
        raw = bytes(self._buffer[:consumed_bytes])
        del self._buffer[:consumed_bytes]
        encoded_raw = raw + b"\x00" * (frame_bytes - consumed_bytes)
        payload = dict(source_payload)
        payload["type"] = "audio"
        payload["format"] = "pcm_f32le"
        payload["sample_rate_hz"] = _SAMPLE_RATE_HZ
        payload["audio"] = base64.b64encode(encoded_raw).decode("ascii")
        reservation = PersonaPlexPcmAppendReservation(
            owner=self,
            operation_id=operation_id,
            payload=payload,
            raw=raw,
        )
        self._reservations.append(reservation)
        return reservation

    def _commit_reservation(self, reservation: PersonaPlexPcmAppendReservation) -> None:
        if not reservation.active:
            return
        reservation._active = False
        self._reservations.remove(reservation)

    def _rollback_reservation(self, reservation: PersonaPlexPcmAppendReservation) -> None:
        if not reservation.active:
            return
        try:
            index = self._reservations.index(reservation)
        except ValueError:
            reservation._active = False
            return
        rolled_back = self._reservations[index:]
        restored = b"".join(item._raw for item in rolled_back if item.active)
        self._buffer[:0] = restored
        for item in rolled_back:
            item._active = False
        del self._reservations[index:]


__all__ = [
    "PersonaPlexPcmAppendBuffer",
    "PersonaPlexPcmAppendReservation",
]
