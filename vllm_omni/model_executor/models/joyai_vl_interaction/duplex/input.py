# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transactional audio and latest-frame input for committed JoyAI observations."""

from __future__ import annotations

import pybase64

from vllm_omni.engine.duplex.plugin import PcmAppendBuffer, PcmAppendReservation


class FrameReservation(PcmAppendReservation):
    def __init__(
        self,
        owner: JoyAIFrameBuffer,
        operation_id: str,
        frames: list[str],
        audio: bytes = b"",
        *,
        had_speech: bool = True,
    ) -> None:
        self.owner = owner
        self.operation_id = operation_id
        self.frames = frames
        self.audio = audio
        self.had_speech = had_speech
        self.version = owner.version
        self.payload = {"video_frames": frames, "is_speech": False} if frames else None
        if audio and had_speech:
            self.payload = {
                **(self.payload or {}),
                "audio": pybase64.b64encode(audio).decode("ascii"),
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
                "is_speech": True,
            }
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def byte_count(self) -> int:
        return len(self.audio) + sum(map(len, self.frames))

    def commit(self) -> None:
        if self._active:
            self._active = False
            self.owner.reservations.remove(self)

    def rollback(self) -> None:
        if self._active:
            self.owner.audio[:0] = self.audio
            self.owner._had_speech = self.owner._had_speech or self.had_speech
            # A newer camera update supersedes an observation that failed to submit.
            if not self.owner.frames and self.owner.version == self.version:
                self.owner.frames = self.frames
            self.commit()


class FrameAppendReservation(PcmAppendReservation):
    """Undo a buffered append if exact retained-byte accounting rejects it."""

    payload = None

    def __init__(self, owner: JoyAIFrameBuffer, operation_id: str) -> None:
        self.owner = owner
        self.operation_id = operation_id
        self.audio_size = len(owner.audio)
        self.dropped_audio = b""
        self.frames = owner.frames
        self.version = owner.version
        self.had_speech = owner._had_speech
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def byte_count(self) -> int:
        # All bytes remain pending in the owner; none are submitted yet.
        return 0

    def commit(self) -> None:
        self._active = False

    def rollback(self) -> None:
        if self._active:
            self.owner.audio[:0] = self.dropped_audio
            del self.owner.audio[self.audio_size :]
            self.owner.frames = self.frames
            self.owner.version = self.version
            self.owner._had_speech = self.had_speech
            self.commit()


class JoyAIFrameBuffer(PcmAppendBuffer):
    """Retain the latest camera frame until commit, with exact wire-byte accounting."""

    def __init__(self, *, accepts_audio: bool = False) -> None:
        self.accepts_audio = accepts_audio
        self.audio = bytearray()
        self._had_speech = False
        self.frames: list[str] = []
        self.reservations: list[FrameReservation] = []
        self.version = 0

    @property
    def pending_byte_count(self) -> int:
        return len(self.audio) + sum(map(len, self.frames))

    def clear(self) -> None:
        self.frames = []
        self.audio.clear()
        self._had_speech = False
        self.version += 1
        for reservation in self.reservations:
            reservation._active = False
        self.reservations.clear()

    def clear_force_listen(self) -> None:
        pass

    def has_pending(self) -> bool:
        return bool(self.audio or self.frames)

    def has_reserved(self) -> bool:
        return bool(self.reservations)

    def prepare_append(
        self, payload: dict[str, object], *, operation_id: str, chunk_period_ms: int, allow_emit: bool
    ) -> FrameAppendReservation:
        reservation = FrameAppendReservation(self, operation_id)
        if payload.get("audio"):
            if not self.accepts_audio:
                raise ValueError("JoyAI video profile accepts text and video; speech input requires the ASR profile")
            if payload.get("format") != "pcm_f32le" or payload.get("sample_rate_hz") != 16000:
                raise ValueError("JoyAI speech input requires normalized 16 kHz pcm_f32le")
            raw = pybase64.b64decode(payload["audio"], validate=True)
            if len(raw) % 4:
                raise ValueError("Invalid float32 PCM length")
            self.audio.extend(raw)
            self._had_speech = self._had_speech or payload.get("is_speech") is not False
            if not self._had_speech:
                # Keep one input chunk of pre-roll while watching an idle scene.
                # Once speech begins, retain the whole utterance until commit.
                preroll_bytes = 16000 * 4 * chunk_period_ms // 1000
                dropped_bytes = max(0, len(self.audio) - preroll_bytes)
                reservation.dropped_audio = bytes(self.audio[:dropped_bytes])
                del self.audio[:dropped_bytes]
        frames = payload.get("video_frames")
        if isinstance(frames, list) and frames:
            self.version += 1
            self.frames = [frames[-1]]
        return reservation

    def prepare_commit(self, *, operation_id: str, chunk_period_ms: int) -> FrameReservation:
        frames, self.frames = self.frames, []
        audio = bytes(self.audio)
        self.audio.clear()
        reservation = FrameReservation(self, operation_id, frames, audio, had_speech=self._had_speech)
        self._had_speech = False
        self.reservations.append(reservation)
        return reservation

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        reservation = self.prepare_commit(operation_id="flush", chunk_period_ms=chunk_period_ms)
        reservation.commit()
        return reservation.payload
