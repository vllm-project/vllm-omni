# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transactional PCM buffering for committed Qwen speech turns."""

from __future__ import annotations

from dataclasses import dataclass, field

import pybase64 as base64

from vllm_omni.engine.duplex.plugin import PcmAppendBuffer, PcmAppendReservation


@dataclass
class Reservation(PcmAppendReservation):
    owner: QwenPcmBuffer
    operation_id: str
    payload: dict[str, object] | None
    raw: bytes
    _active: bool = True

    @property
    def active(self):
        return self._active

    @property
    def byte_count(self):
        return len(self.raw)

    def commit(self):
        if self._active:
            self._active = False
            self.owner.reservations.remove(self)

    def rollback(self):
        if self._active:
            self.owner.buffer[:0] = self.raw
            self.commit()


@dataclass
class QwenPcmBuffer(PcmAppendBuffer):
    buffer: bytearray = field(default_factory=bytearray)
    reservations: list[Reservation] = field(default_factory=list)

    @property
    def pending_byte_count(self):
        return len(self.buffer)

    def clear(self):
        for reservation in self.reservations:
            reservation._active = False
        self.reservations.clear()
        self.buffer.clear()

    def clear_force_listen(self):
        pass

    def has_pending(self):
        return bool(self.buffer)

    def has_reserved(self):
        return bool(self.reservations)

    def prepare_append(self, payload, *, operation_id, chunk_period_ms, allow_emit):
        if payload.get("format") != "pcm_f32le" or payload.get("sample_rate_hz") != 16000:
            raise ValueError("Qwen duplex requires normalized 16 kHz pcm_f32le")
        if payload.get("video_frames"):
            # Qwen answers whole turns, so it has no audio unit for a frame
            # track to interleave against. Images arrive the OpenAI way, as
            # input_image conversation items, and stay until deleted.
            raise ValueError(
                "Qwen duplex does not take video_frames on an audio append; "
                "send images as conversation.item.create items with input_image content"
            )
        raw = base64.b64decode(payload.get("audio", ""), validate=True)
        if len(raw) % 4:
            raise ValueError("Invalid float32 PCM length")
        self.buffer.extend(raw)
        # Qwen inference starts on commit, never on an individual audio chunk.
        return None

    def prepare_commit(self, *, operation_id, chunk_period_ms):
        raw = bytes(self.buffer)
        payload = (
            {
                "audio": base64.b64encode(raw).decode(),
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
                "is_speech": True,
            }
            if raw
            else None
        )
        self.buffer.clear()
        reservation = Reservation(self, operation_id, payload, raw)
        self.reservations.append(reservation)
        return reservation

    def flush(self, *, chunk_period_ms):
        reservation = self.prepare_commit(operation_id="flush", chunk_period_ms=chunk_period_ms)
        reservation.commit()
        return reservation.payload
