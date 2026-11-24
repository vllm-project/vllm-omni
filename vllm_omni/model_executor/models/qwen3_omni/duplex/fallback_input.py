# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Committed-turn PCM buffering for Qwen3-Omni's chat fallback."""

from __future__ import annotations

import binascii
import math
import struct

import pybase64 as base64

from vllm_omni.engine.duplex.plugin import PcmAppendBuffer, PcmAppendReservation

QWEN3_OMNI_SAMPLE_RATE_HZ = 16000
_BYTES_PER_SAMPLE = 4


def _decode_pcm_payload(payload: dict[str, object]) -> tuple[bytes, int]:
    if payload.get("format") != "pcm_f32le":
        raise ValueError("Qwen3-Omni duplex input format must be pcm_f32le")
    sample_rate_hz = payload.get("sample_rate_hz")
    if sample_rate_hz != QWEN3_OMNI_SAMPLE_RATE_HZ:
        raise ValueError("Qwen3-Omni duplex input sample_rate_hz must be 16000")
    audio = payload.get("audio", payload.get("data"))
    if not isinstance(audio, str):
        raise ValueError("Qwen3-Omni duplex audio must be base64 pcm_f32le")
    try:
        raw = base64.b64decode(audio, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Qwen3-Omni duplex audio is not valid base64") from exc
    if len(raw) % _BYTES_PER_SAMPLE:
        raise ValueError("Qwen3-Omni pcm_f32le byte length must be divisible by four")
    if raw and not all(math.isfinite(sample) for (sample,) in struct.iter_unpack("<f", raw)):
        raise ValueError("Qwen3-Omni pcm_f32le samples must be finite")
    return raw, sample_rate_hz


class Qwen3OmniPcmAppendReservation(PcmAppendReservation):
    """Transactional reservation for one committed Qwen3 input turn."""

    __slots__ = (
        "_active",
        "_force_listen",
        "_owner",
        "_sample_rate_hz",
        "_speech",
        "_raw",
        "operation_id",
        "payload",
    )

    def __init__(
        self,
        *,
        owner: Qwen3OmniPcmAppendBuffer,
        operation_id: str,
        payload: dict[str, object] | None,
        raw: bytes,
        sample_rate_hz: int | None,
        speech: bool,
        force_listen: bool,
    ) -> None:
        self._owner = owner
        self.operation_id = operation_id
        self.payload = payload
        self._raw = raw
        self._sample_rate_hz = sample_rate_hz
        self._speech = speech
        self._force_listen = force_listen
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


class Qwen3OmniPcmAppendBuffer(PcmAppendBuffer):
    """Accumulate all input audio until the client commits the turn.

    Qwen3-Omni uses the ordinary chat pipeline, so emitting scheduler-sized
    native chunks would create a second, unsupported generation path.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._sample_rate_hz: int | None = None
        self._turn_had_speech = False
        self._force_listen = False
        self._reservations: list[Qwen3OmniPcmAppendReservation] = []

    @property
    def pending_byte_count(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        for reservation in self._reservations:
            reservation._active = False
        self._reservations.clear()
        self._buffer.clear()
        self._sample_rate_hz = None
        self._turn_had_speech = False
        self._force_listen = False

    def clear_force_listen(self) -> None:
        self._force_listen = False

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
    ) -> Qwen3OmniPcmAppendReservation | None:
        del operation_id, chunk_period_ms, allow_emit
        raw, sample_rate_hz = _decode_pcm_payload(payload)
        if self._sample_rate_hz is not None and self._sample_rate_hz != sample_rate_hz:
            raise ValueError("Qwen3-Omni duplex audio sample_rate_hz changed within a session")
        self._sample_rate_hz = sample_rate_hz
        self._buffer.extend(raw)
        self._turn_had_speech = self._turn_had_speech or bool(payload.get("is_speech", False))
        self._force_listen = self._force_listen or bool(payload.get("force_listen", False))
        return None

    def prepare_commit(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> Qwen3OmniPcmAppendReservation:
        del chunk_period_ms
        if any(reservation.active for reservation in self._reservations):
            raise RuntimeError("Qwen3-Omni PCM commit has an unresolved reservation")

        raw = bytes(self._buffer)
        payload: dict[str, object] | None = None
        if raw:
            payload = {
                "type": "audio",
                "audio": base64.b64encode(raw).decode("ascii"),
                "format": "pcm_f32le",
                "sample_rate_hz": self._sample_rate_hz or QWEN3_OMNI_SAMPLE_RATE_HZ,
                "final": True,
                "is_speech": self._turn_had_speech,
                "force_listen": self._force_listen,
            }
        reservation = Qwen3OmniPcmAppendReservation(
            owner=self,
            operation_id=operation_id,
            payload=payload,
            raw=raw,
            sample_rate_hz=self._sample_rate_hz,
            speech=self._turn_had_speech,
            force_listen=self._force_listen,
        )
        self._reservations.append(reservation)
        self._buffer.clear()
        self._sample_rate_hz = None
        self._turn_had_speech = False
        self._force_listen = False
        return reservation

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        reservation = self.prepare_commit(
            operation_id="qwen3-omni-flush",
            chunk_period_ms=chunk_period_ms,
        )
        payload = reservation.payload
        reservation.commit()
        return payload

    def _commit_reservation(self, reservation: Qwen3OmniPcmAppendReservation) -> None:
        if not reservation._active:
            return
        try:
            index = self._reservations.index(reservation)
        except ValueError:
            reservation._active = False
            return
        if index != 0:
            raise RuntimeError("Qwen3-Omni PCM reservations must commit in wire order")
        self._reservations.pop(0)
        reservation._active = False

    def _rollback_reservation(self, reservation: Qwen3OmniPcmAppendReservation) -> None:
        if not reservation._active:
            return
        try:
            index = self._reservations.index(reservation)
        except ValueError:
            reservation._active = False
            return
        rolled_back = self._reservations[index:]
        self._buffer[:0] = b"".join(item._raw for item in rolled_back)
        self._sample_rate_hz = self._sample_rate_hz or next(
            (item._sample_rate_hz for item in rolled_back if item._sample_rate_hz is not None),
            None,
        )
        self._turn_had_speech = self._turn_had_speech or any(item._speech for item in rolled_back)
        self._force_listen = self._force_listen or any(item._force_listen for item in rolled_back)
        for item in rolled_back:
            item._active = False
        del self._reservations[index:]


__all__ = [
    "QWEN3_OMNI_SAMPLE_RATE_HZ",
    "Qwen3OmniPcmAppendBuffer",
    "Qwen3OmniPcmAppendReservation",
]
