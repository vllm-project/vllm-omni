# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Latest-frame input for committed JoyAI observations."""

from __future__ import annotations

from vllm_omni.engine.duplex.plugin import PcmAppendBuffer, PcmAppendReservation


class FrameReservation(PcmAppendReservation):
    def __init__(self, owner: JoyAIFrameBuffer, operation_id: str, frames: list[str]) -> None:
        self.owner = owner
        self.operation_id = operation_id
        self.frames = frames
        self.version = owner.version
        self.payload = {"video_frames": frames, "is_speech": False} if frames else None
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def byte_count(self) -> int:
        return sum(map(len, self.frames))

    def commit(self) -> None:
        if self._active:
            self._active = False
            self.owner.reservations.remove(self)

    def rollback(self) -> None:
        if self._active:
            # A newer camera update supersedes an observation that failed to submit.
            if not self.owner.frames and self.owner.version == self.version:
                self.owner.frames = self.frames
            self.commit()


class JoyAIFrameBuffer(PcmAppendBuffer):
    """Retain the latest camera frame until commit, with exact wire-byte accounting."""

    def __init__(self) -> None:
        self.frames: list[str] = []
        self.reservations: list[FrameReservation] = []
        self.version = 0

    @property
    def pending_byte_count(self) -> int:
        return sum(map(len, self.frames))

    def clear(self) -> None:
        self.frames = []
        self.version += 1
        for reservation in self.reservations:
            reservation._active = False
        self.reservations.clear()

    def clear_force_listen(self) -> None:
        pass

    def has_pending(self) -> bool:
        return bool(self.frames)

    def has_reserved(self) -> bool:
        return bool(self.reservations)

    def prepare_append(
        self, payload: dict[str, object], *, operation_id: str, chunk_period_ms: int, allow_emit: bool
    ) -> None:
        if payload.get("audio"):
            raise ValueError("JoyAI video profile accepts text and video; speech input requires the ASR profile")
        frames = payload.get("video_frames")
        if isinstance(frames, list) and frames:
            self.version += 1
            self.frames = [frames[-1]]
        return None

    def prepare_commit(self, *, operation_id: str, chunk_period_ms: int) -> FrameReservation:
        frames, self.frames = self.frames, []
        reservation = FrameReservation(self, operation_id, frames)
        self.reservations.append(reservation)
        return reservation

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        reservation = self.prepare_commit(operation_id="flush", chunk_period_ms=chunk_period_ms)
        reservation.commit()
        return reservation.payload
