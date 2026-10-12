# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transactional 400 ms PCM input ledger for native Lychee duplex."""

from __future__ import annotations

import binascii

import pybase64 as base64

from vllm_omni.engine.duplex.pcm_reservation import (
    commit_ordered_reservation,
    rollback_ordered_reservation,
)
from vllm_omni.engine.duplex.plugin import PcmAppendBuffer, PcmAppendReservation
from vllm_omni.model_executor.models.lychee_fd.audio_features import (
    SAMPLE_RATE_HZ,
    WINDOW_MS,
    WINDOW_SAMPLES,
)

SAMPLES_PER_TICK = 640
TICKS_PER_WINDOW = WINDOW_SAMPLES // SAMPLES_PER_TICK


class LycheePcmAppendReservation(PcmAppendReservation):
    __slots__ = (
        "_active",
        "_owner",
        "_raw",
        "sample_end",
        "sample_start",
        "operation_id",
        "payload",
    )

    def __init__(
        self,
        *,
        owner: LycheePcmAppendBuffer,
        operation_id: str,
        payload: dict[str, object] | None,
        raw: bytes,
        sample_start: int,
        sample_end: int,
    ) -> None:
        self._owner = owner
        self.operation_id = operation_id
        self.payload = payload
        self._raw = raw
        self.sample_start = sample_start
        self.sample_end = sample_end
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


class LycheePcmAppendBuffer(PcmAppendBuffer):
    """Own received/consumable audio coordinates for one Lychee session.

    The buffer emits exactly one 400 ms window per normal append.  Short
    client chunks park without submitting a model request.  A commit pads the
    final tail to one whole window, but the ledger records both real and padded
    sample ends so rebuild logic never mistakes padding for received speech.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._reservations: list[LycheePcmAppendReservation] = []
        self._received_sample_end = 0
        self._next_window_sample = 0

    @property
    def pending_byte_count(self) -> int:
        return len(self._buffer)

    @property
    def received_sample_end(self) -> int:
        return self._received_sample_end

    @property
    def consumable_tick_end(self) -> int:
        return self._next_window_sample // SAMPLES_PER_TICK

    def clear(self) -> None:
        for reservation in self._reservations:
            reservation._active = False
        self._reservations.clear()
        self._buffer.clear()
        self._received_sample_end = 0
        self._next_window_sample = 0

    def clear_force_listen(self) -> None:
        return

    def has_pending(self) -> bool:
        return bool(self._buffer)

    def has_reserved(self) -> bool:
        return bool(self._reservations)

    @staticmethod
    def _decode(payload: dict[str, object]) -> bytes:
        if payload.get("format") != "pcm_f32le":
            raise ValueError("Lychee native duplex requires pcm_f32le audio")
        if payload.get("sample_rate_hz") != SAMPLE_RATE_HZ:
            raise ValueError(f"Lychee native duplex requires {SAMPLE_RATE_HZ} Hz audio")
        audio = payload.get("audio")
        if not isinstance(audio, str):
            raise ValueError("Lychee native duplex audio must be base64 encoded")
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Lychee native duplex audio is not valid base64") from exc
        if len(raw) % 4:
            raise ValueError("Lychee pcm_f32le audio byte length must be divisible by four")
        return raw

    def _reserve_window(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        flush: bool,
    ) -> LycheePcmAppendReservation | None:
        buffered_samples = len(self._buffer) // 4
        if not flush and buffered_samples < WINDOW_SAMPLES:
            return None
        if buffered_samples == 0:
            return None

        real_samples = min(buffered_samples, WINDOW_SAMPLES)
        raw_bytes = real_samples * 4
        reserved_raw = bytes(self._buffer[:raw_bytes])
        del self._buffer[:raw_bytes]
        padded_samples = WINDOW_SAMPLES - real_samples
        emitted_raw = reserved_raw + b"\x00" * (padded_samples * 4)
        sample_start = self._next_window_sample
        sample_end = sample_start + real_samples
        padded_sample_end = sample_start + WINDOW_SAMPLES
        self._next_window_sample = padded_sample_end

        out = dict(payload)
        out["audio"] = base64.b64encode(emitted_raw).decode("ascii")
        out["format"] = "pcm_f32le"
        out["sample_rate_hz"] = SAMPLE_RATE_HZ
        out["lychee_audio_ledger"] = {
            "received_sample_end": self._received_sample_end,
            "window_sample_start": sample_start,
            "window_sample_end": sample_end,
            "padded_sample_end": padded_sample_end,
            "consumable_tick_start": sample_start // SAMPLES_PER_TICK,
            "consumable_tick_end": padded_sample_end // SAMPLES_PER_TICK,
            "ticks_per_window": TICKS_PER_WINDOW,
        }
        reservation = LycheePcmAppendReservation(
            owner=self,
            operation_id=operation_id,
            payload=out,
            raw=reserved_raw,
            sample_start=sample_start,
            sample_end=sample_end,
        )
        self._reservations.append(reservation)
        return reservation

    def prepare_append(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        chunk_period_ms: int,
        allow_emit: bool,
    ) -> LycheePcmAppendReservation | None:
        if chunk_period_ms != WINDOW_MS:
            raise ValueError(f"Lychee native duplex chunk period must be {WINDOW_MS} ms")
        raw = self._decode(payload)
        self._buffer.extend(raw)
        self._received_sample_end += len(raw) // 4
        if not allow_emit:
            return None
        return self._reserve_window(payload, operation_id=operation_id, flush=False)

    def prepare_commit(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> LycheePcmAppendReservation:
        if chunk_period_ms != WINDOW_MS:
            raise ValueError(f"Lychee native duplex chunk period must be {WINDOW_MS} ms")
        reservation = self._reserve_window(
            {
                "type": "audio",
                "audio": "",
                "format": "pcm_f32le",
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "is_speech": True,
            },
            operation_id=operation_id,
            flush=True,
        )
        if reservation is not None:
            assert reservation.payload is not None
            reservation.payload["final"] = True
            return reservation
        reservation = LycheePcmAppendReservation(
            owner=self,
            operation_id=operation_id,
            payload=None,
            raw=b"",
            sample_start=self._next_window_sample,
            sample_end=self._next_window_sample,
        )
        self._reservations.append(reservation)
        return reservation

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        reservation = self.prepare_commit(operation_id="flush", chunk_period_ms=chunk_period_ms)
        reservation.commit()
        return reservation.payload

    def _commit_reservation(self, reservation: LycheePcmAppendReservation) -> None:
        commit_ordered_reservation(self._reservations, reservation, head_only=True)

    def _rollback_reservation(self, reservation: LycheePcmAppendReservation) -> None:
        rolled_back = rollback_ordered_reservation(
            self._reservations,
            reservation,
            self._buffer,
            active_only=False,
        )
        if rolled_back:
            self._next_window_sample = min(item.sample_start for item in rolled_back)


__all__ = [
    "SAMPLES_PER_TICK",
    "TICKS_PER_WINDOW",
    "LycheePcmAppendBuffer",
    "LycheePcmAppendReservation",
]
