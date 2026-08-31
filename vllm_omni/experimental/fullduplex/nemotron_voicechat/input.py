# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""80 ms PCM packetization for Nemotron VoiceChat native duplex."""

from __future__ import annotations

import binascii

import numpy as np
import pybase64 as base64

NEMOTRON_VOICECHAT_SAMPLE_RATE = 16000
NEMOTRON_VOICECHAT_FRAME_SAMPLES = 1280
_SAMPLE_BYTES = 4
_FRAME_BYTES = NEMOTRON_VOICECHAT_FRAME_SAMPLES * _SAMPLE_BYTES


def _require_terminal_tail(buffered_bytes: int) -> None:
    if buffered_bytes > _FRAME_BYTES:
        raise ValueError(
            "Nemotron VoiceChat manual/deferred commit cannot contain more than one 80 ms frame; "
            "enable extra_body.auto_response for native full-duplex streaming"
        )


def decode_pcm_f32le(payload: object, *, exact_frame: bool = False) -> bytes:
    if not isinstance(payload, dict):
        raise ValueError("Nemotron VoiceChat duplex audio payload must be a mapping")
    if payload.get("format") != "pcm_f32le":
        raise ValueError("Nemotron VoiceChat duplex audio format must be pcm_f32le")
    if payload.get("sample_rate_hz") != NEMOTRON_VOICECHAT_SAMPLE_RATE:
        raise ValueError("Nemotron VoiceChat duplex audio sample_rate_hz must be 16000")
    encoded = payload.get("audio")
    if not isinstance(encoded, str):
        raise ValueError("Nemotron VoiceChat duplex audio must be base64 pcm_f32le")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Nemotron VoiceChat duplex audio is not valid base64") from exc
    if len(raw) % _SAMPLE_BYTES:
        raise ValueError("Nemotron VoiceChat pcm_f32le payload has a partial sample")
    values = np.frombuffer(raw, dtype="<f4")
    if values.size and not bool(np.isfinite(values).all()):
        raise ValueError("Nemotron VoiceChat duplex audio samples must be finite")
    if exact_frame and len(raw) != _FRAME_BYTES:
        raise ValueError(
            f"Nemotron VoiceChat native duplex append must contain exactly {NEMOTRON_VOICECHAT_FRAME_SAMPLES} samples"
        )
    return raw


def _frame_payload(raw: bytes, *, final: bool) -> dict[str, object]:
    return {
        "type": "audio",
        "audio": base64.b64encode(raw).decode("ascii"),
        "format": "pcm_f32le",
        "sample_rate_hz": NEMOTRON_VOICECHAT_SAMPLE_RATE,
        "final": final,
    }


class NemotronVoiceChatPcmAppendReservation:
    def __init__(
        self,
        owner: NemotronVoiceChatPcmAppendBuffer,
        *,
        operation_id: str,
        payload: dict[str, object] | None,
        raw: bytes,
    ) -> None:
        self._owner = owner
        self.operation_id = operation_id
        self.payload = payload
        self.raw = raw
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def byte_count(self) -> int:
        return len(self.raw)

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
                restore.extend(reservation.raw)
                reservation._active = False
        del self._owner._reservations[index:]
        self._owner._buffer[:0] = restore


class NemotronVoiceChatPcmAppendBuffer:
    """Adapt Realtime packets to the current full-duplex PCM buffer contract.

    The model consumes exactly one 1280-sample frame per scheduler append.
    Realtime packets may split a frame arbitrarily, but a single packet is
    capped at one frame so the serving contract never silently drops a second
    ready frame behind a single-reservation API.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._reservations: list[NemotronVoiceChatPcmAppendReservation] = []

    @property
    def pending_byte_count(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        for reservation in self._reservations:
            reservation._active = False
        self._reservations.clear()
        self._buffer.clear()

    def clear_force_listen(self) -> None:
        # Nemotron VoiceChat has no serving-side force-listen packet bit.
        return

    def has_pending(self) -> bool:
        return bool(self._buffer)

    def has_reserved(self) -> bool:
        return any(reservation.active for reservation in self._reservations)

    def _reserve_frame(
        self,
        *,
        operation_id: str,
        final: bool,
    ) -> NemotronVoiceChatPcmAppendReservation | None:
        if len(self._buffer) < _FRAME_BYTES:
            return None
        raw = bytes(self._buffer[:_FRAME_BYTES])
        del self._buffer[:_FRAME_BYTES]
        reservation = NemotronVoiceChatPcmAppendReservation(
            self,
            operation_id=operation_id,
            payload=_frame_payload(raw, final=final),
            raw=raw,
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
    ) -> NemotronVoiceChatPcmAppendReservation | None:
        if chunk_period_ms != 80:
            raise ValueError("Nemotron VoiceChat native duplex chunk_period_ms must be 80")
        raw = decode_pcm_f32le(payload)
        if len(raw) > _FRAME_BYTES:
            raise ValueError("Nemotron VoiceChat Realtime append must contain at most 1280 samples")
        self._buffer.extend(raw)
        if not allow_emit:
            return None
        return self._reserve_frame(operation_id=operation_id, final=False)

    def prepare_commit(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> NemotronVoiceChatPcmAppendReservation:
        if chunk_period_ms != 80:
            raise ValueError("Nemotron VoiceChat native duplex chunk_period_ms must be 80")
        _require_terminal_tail(len(self._buffer))
        if not self._buffer:
            reservation = NemotronVoiceChatPcmAppendReservation(
                self,
                operation_id=operation_id,
                payload=None,
                raw=b"",
            )
        else:
            raw = bytes(self._buffer)
            self._buffer.clear()
            reservation = NemotronVoiceChatPcmAppendReservation(
                self,
                operation_id=operation_id,
                payload=_frame_payload(raw + bytes(_FRAME_BYTES - len(raw)), final=True),
                raw=raw,
            )
        self._reservations.append(reservation)
        return reservation

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        if chunk_period_ms != 80:
            raise ValueError("Nemotron VoiceChat native duplex chunk_period_ms must be 80")
        _require_terminal_tail(len(self._buffer))
        if not self._buffer:
            return None
        raw = bytes(self._buffer)
        self._buffer.clear()
        return _frame_payload(raw + bytes(_FRAME_BYTES - len(raw)), final=True)


__all__ = [
    "NEMOTRON_VOICECHAT_FRAME_SAMPLES",
    "NEMOTRON_VOICECHAT_SAMPLE_RATE",
    "NemotronVoiceChatPcmAppendBuffer",
    "NemotronVoiceChatPcmAppendReservation",
    "decode_pcm_f32le",
]
