# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving-side PCM re-framing for Qwen3-Omni duplex appends.

Realtime clients push arbitrarily sized PCM frames; the engine reserves
scheduler slots from the payload size and the worker consumes whole chunks.
This buffer accumulates raw PCM and emits only whole ``chunk_period_ms``
units, with two-phase commit so a failed engine append restores the audio
instead of dropping it.

Satisfies ``vllm_omni.experimental.fullduplex.openai.runtime_adapter``'s
``PcmAppendBuffer`` / ``PcmAppendReservation`` protocols. Modelled on
``minicpmo45/input.py`` but audio-only: no video frame queue and no
``force_listen`` span tracking, because Qwen3-Omni has no learned
listen/speak control token for those to feed.
"""

from __future__ import annotations

import base64

from vllm_omni.experimental.fullduplex.qwen3omni.policy import Qwen3OmniDuplexPolicy

_BYTES_PER_SAMPLE = 4  # float32 little-endian


def decode_pcm_payload(payload: object) -> bytes:
    """Decode a Realtime audio payload to raw ``pcm_f32le`` bytes."""
    if not isinstance(payload, dict):
        raise ValueError("duplex audio payload must be a mapping")
    audio_format = payload.get("format", Qwen3OmniDuplexPolicy.PCM_FORMAT)
    if audio_format != Qwen3OmniDuplexPolicy.PCM_FORMAT:
        raise ValueError(
            f"unsupported duplex audio format: {audio_format!r} (expected {Qwen3OmniDuplexPolicy.PCM_FORMAT})"
        )
    data = payload.get("audio")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)
    raise ValueError("duplex audio payload is missing base64 'audio' data")


class Qwen3OmniPcmReservation:
    """A prepared-but-uncommitted slice of buffered PCM."""

    def __init__(
        self,
        buffer: Qwen3OmniPcmAppendBuffer,
        *,
        operation_id: str,
        payload: dict[str, object] | None,
        consumed: bytes,
    ) -> None:
        self.operation_id = operation_id
        self.payload = payload
        self._buffer = buffer
        self._consumed = consumed
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def byte_count(self) -> int:
        return len(self._consumed)

    def commit(self) -> None:
        """Accept the reservation; the consumed PCM is gone for good."""
        if not self._active:
            return
        self._active = False
        self._buffer._release_reservation(self)

    def rollback(self) -> None:
        """Return the consumed PCM to the front of the buffer."""
        if not self._active:
            return
        self._active = False
        self._buffer._restore_reservation(self, self._consumed)


class Qwen3OmniPcmAppendBuffer:
    """Chunk-aligned PCM accumulator with two-phase commit."""

    def __init__(self, *, sample_rate_hz: int | None = None) -> None:
        self._sample_rate_hz = sample_rate_hz or Qwen3OmniDuplexPolicy.SAMPLE_RATE_HZ
        self._pending = bytearray()
        self._reserved = 0

    # ---- protocol surface -------------------------------------------------

    @property
    def pending_byte_count(self) -> int:
        return len(self._pending)

    def clear(self) -> None:
        self._pending.clear()
        self._reserved = 0

    def clear_force_listen(self) -> None:
        """No-op.

        MiniCPM tracks per-span ``force_listen`` metadata so its model-owned
        listen/speak policy can be overridden. Qwen3-Omni has no such control
        token, so there is no force-listen state to clear. Implemented to
        satisfy the ``PcmAppendBuffer`` protocol.
        """
        return None

    def has_pending(self) -> bool:
        return bool(self._pending)

    def has_reserved(self) -> bool:
        return self._reserved > 0

    def prepare_append(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        chunk_period_ms: int,
        allow_emit: bool,
    ) -> Qwen3OmniPcmReservation | None:
        """Buffer incoming PCM, returning a reservation once a whole chunk exists."""
        self._pending.extend(decode_pcm_payload(payload))
        # Deliberately never emit mid-turn.
        #
        # The framework streams each emitted chunk to stage 0 as its own
        # append, and with auto_response the thinker generates off whatever it
        # has. For a model with learned listen/speak tokens (MiniCPM-o 4.5)
        # that is the point. Qwen3-Omni has no such token, so an intermediate
        # append asks it to continue a user turn that has no <|audio_end|> and
        # no assistant generation prompt yet -- observed output was
        # ' 1000000...' and ' a i \n\n\nuser\n...', i.e. the model completing a
        # truncated prompt. Only the first second of a 4 s utterance ever
        # reached it.
        #
        # Holding the audio until commit gives the thinker one well-formed
        # turn, identical in shape to what the working non-duplex path builds.
        # Cost: audio is not encoded incrementally, so time-to-first-token
        # starts at commit rather than during speech. Barge-in is unaffected --
        # it runs on the session epoch, not on append cadence.
        del allow_emit, operation_id, chunk_period_ms
        return None

    def prepare_commit(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> Qwen3OmniPcmReservation:
        """Reserve everything buffered, zero-padding the tail to a whole chunk.

        Used on an explicit client commit, where the remaining partial chunk
        must still reach the model. Padding rather than dropping keeps the
        reserved scheduler slots and the produced embedding count in
        agreement.
        """
        chunk_bytes = self._chunk_bytes(chunk_period_ms)
        if self._pending:
            remainder = len(self._pending) % chunk_bytes
            if remainder:
                self._pending.extend(b"\x00" * (chunk_bytes - remainder))
        reservation = self._reserve_whole_chunks(
            operation_id=operation_id,
            chunk_period_ms=chunk_period_ms,
        )
        if reservation is not None:
            # Tell the engine this append closes the user's turn, so the
            # prompt gets the assistant generation suffix.
            if reservation.payload is not None:
                reservation.payload[Qwen3OmniDuplexPolicy.TURN_FINAL_KEY] = True
            return reservation
        # Nothing buffered -- normally because `prepare_append` already
        # streamed every whole chunk out. The turn still has to be closed, or
        # the assistant generation prompt is never emitted and the model has
        # nothing telling it to reply. Hand back a zero-audio payload carrying
        # only the turn-final marker.
        return Qwen3OmniPcmReservation(
            self,
            operation_id=operation_id,
            payload={**self._payload(b""), Qwen3OmniDuplexPolicy.TURN_FINAL_KEY: True},
            consumed=b"",
        )

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        """Drain buffered PCM as one payload, zero-padded to a whole chunk."""
        if not self._pending:
            return None
        chunk_bytes = self._chunk_bytes(chunk_period_ms)
        remainder = len(self._pending) % chunk_bytes
        if remainder:
            self._pending.extend(b"\x00" * (chunk_bytes - remainder))
        drained = bytes(self._pending)
        self._pending.clear()
        return self._payload(drained)

    # ---- internals --------------------------------------------------------

    def _chunk_bytes(self, chunk_period_ms: int) -> int:
        period = chunk_period_ms if chunk_period_ms > 0 else Qwen3OmniDuplexPolicy.CHUNK_PERIOD_MS
        samples = self._sample_rate_hz * period // 1000
        return max(_BYTES_PER_SAMPLE, samples * _BYTES_PER_SAMPLE)

    def _reserve_whole_chunks(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> Qwen3OmniPcmReservation | None:
        chunk_bytes = self._chunk_bytes(chunk_period_ms)
        whole = (len(self._pending) // chunk_bytes) * chunk_bytes
        if whole <= 0:
            return None
        consumed = bytes(self._pending[:whole])
        del self._pending[:whole]
        self._reserved += whole
        return Qwen3OmniPcmReservation(
            self,
            operation_id=operation_id,
            payload=self._payload(consumed),
            consumed=consumed,
        )

    def _payload(self, data: bytes) -> dict[str, object]:
        return {
            "audio": base64.b64encode(data).decode("ascii"),
            "format": Qwen3OmniDuplexPolicy.PCM_FORMAT,
            "sample_rate_hz": self._sample_rate_hz,
            "num_samples": len(data) // _BYTES_PER_SAMPLE,
        }

    def _release_reservation(self, reservation: Qwen3OmniPcmReservation) -> None:
        self._reserved = max(0, self._reserved - reservation.byte_count)

    def _restore_reservation(self, reservation: Qwen3OmniPcmReservation, consumed: bytes) -> None:
        self._reserved = max(0, self._reserved - reservation.byte_count)
        # Restore to the FRONT: appends must stay in wire order.
        self._pending[:0] = consumed
