# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transactional framing of client PCM into fixed model units.

A frame-locked model (PersonaPlex: 1920 samples every 80 ms at 24 kHz) takes
exactly one frame per append. Clients send arbitrary chunks, so the session
runner buffers them here and takes one whole frame out at a time as a
*reservation*: committed once the stage accepted the append, rolled back
(the bytes go back to the front of the buffer) when it did not.
"""

from __future__ import annotations

import pybase64 as base64

from vllm_omni.engine.duplex.plugin import PcmAppendBuffer, PcmAppendReservation
from vllm_omni.model_executor.common.audio.pcm import (
    PCM_F32LE_BYTES_PER_SAMPLE,
    decode_pcm_f32le_base64,
)


class FixedFramePcmAppendReservation(PcmAppendReservation):
    __slots__ = ("_active", "_owner", "_raw", "operation_id", "payload")

    def __init__(
        self,
        *,
        owner: FixedFramePcmAppendBuffer,
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


class FixedFramePcmAppendBuffer(PcmAppendBuffer):
    """Frame ``pcm_f32le`` at one sample rate into ``frame_samples`` units, transactionally."""

    def __init__(
        self,
        *,
        sample_rate_hz: int,
        frame_samples: int,
        chunk_period_ms: int,
        model: str = "duplex",
    ) -> None:
        if sample_rate_hz <= 0 or frame_samples <= 0 or chunk_period_ms <= 0:
            raise ValueError("FixedFramePcmAppendBuffer needs positive sample_rate_hz, frame_samples, chunk_period_ms")
        self.sample_rate_hz = int(sample_rate_hz)
        self.frame_samples = int(frame_samples)
        self.chunk_period_ms = int(chunk_period_ms)
        self.model = model
        self._buffer = bytearray()
        self._reservations: list[FixedFramePcmAppendReservation] = []

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * PCM_F32LE_BYTES_PER_SAMPLE

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

    def prepare_append(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        chunk_period_ms: int,
        allow_emit: bool,
    ) -> FixedFramePcmAppendReservation | None:
        self._require_chunk_period(chunk_period_ms)
        if any(reservation.active and reservation.operation_id == operation_id for reservation in self._reservations):
            raise ValueError(f"{self.model} duplicate active operation_id: {operation_id}")
        self._buffer.extend(self._decode_payload(payload))
        if not allow_emit or len(self._buffer) < self.frame_bytes:
            return None
        return self._reserve_frame(payload, operation_id=operation_id, flush=False)

    def prepare_commit(self, *, operation_id: str, chunk_period_ms: int) -> FixedFramePcmAppendReservation:
        self._require_chunk_period(chunk_period_ms)
        if not self._buffer:
            reservation = FixedFramePcmAppendReservation(owner=self, operation_id=operation_id, payload=None, raw=b"")
            self._reservations.append(reservation)
            return reservation
        payload: dict[str, object] = {
            "type": "audio",
            "format": "pcm_f32le",
            "sample_rate_hz": self.sample_rate_hz,
            "audio": "",
            "final": True,
        }
        return self._reserve_frame(payload, operation_id=operation_id, flush=True)

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        reservation = self.prepare_commit(operation_id=f"{self.model}-flush", chunk_period_ms=chunk_period_ms)
        payload = reservation.payload
        reservation.commit()
        return payload

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _require_chunk_period(self, chunk_period_ms: int) -> None:
        if chunk_period_ms != self.chunk_period_ms:
            raise ValueError(f"{self.model} chunk_period_ms must be {self.chunk_period_ms}")

    def _decode_payload(self, payload: dict[str, object]) -> bytes:
        if payload.get("format") != "pcm_f32le":
            raise ValueError(f"{self.model} input format must be pcm_f32le")
        if payload.get("sample_rate_hz") != self.sample_rate_hz:
            raise ValueError(f"{self.model} sample_rate_hz must be {self.sample_rate_hz}")
        return decode_pcm_f32le_base64(payload.get("audio"), model=self.model)

    def _reserve_frame(
        self,
        source_payload: dict[str, object],
        *,
        operation_id: str,
        flush: bool,
    ) -> FixedFramePcmAppendReservation:
        frame_bytes = self.frame_bytes
        consumed_bytes = min(len(self._buffer), frame_bytes) if flush else frame_bytes
        raw = bytes(self._buffer[:consumed_bytes])
        del self._buffer[:consumed_bytes]
        encoded_raw = raw + b"\x00" * (frame_bytes - consumed_bytes)
        payload = dict(source_payload)
        payload["type"] = "audio"
        payload["format"] = "pcm_f32le"
        payload["sample_rate_hz"] = self.sample_rate_hz
        payload["audio"] = base64.b64encode(encoded_raw).decode("ascii")
        reservation = FixedFramePcmAppendReservation(owner=self, operation_id=operation_id, payload=payload, raw=raw)
        self._reservations.append(reservation)
        return reservation

    def _commit_reservation(self, reservation: FixedFramePcmAppendReservation) -> None:
        if not reservation.active:
            return
        reservation._active = False
        self._reservations.remove(reservation)

    def _rollback_reservation(self, reservation: FixedFramePcmAppendReservation) -> None:
        if not reservation.active:
            return
        try:
            index = self._reservations.index(reservation)
        except ValueError:
            reservation._active = False
            return
        # Later reservations were taken after this one: they go back too, in order.
        rolled_back = self._reservations[index:]
        restored = b"".join(item._raw for item in rolled_back if item.active)
        self._buffer[:0] = restored
        for item in rolled_back:
            item._active = False
        del self._reservations[index:]


__all__ = ["FixedFramePcmAppendBuffer", "FixedFramePcmAppendReservation"]
