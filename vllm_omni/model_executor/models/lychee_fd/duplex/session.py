# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-session Lychee state owned by the Unified Duplex runner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from vllm_omni.engine.duplex.plugin import DuplexModelSessionState

from .input import LycheePcmAppendBuffer


@dataclass(slots=True)
class LycheeServingSessionState(DuplexModelSessionState):
    audio_buffer: LycheePcmAppendBuffer = field(default_factory=LycheePcmAppendBuffer)
    input_since_commit: bool = False
    speech_since_commit: bool = False
    context_locked: bool = False
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    continuation_owner_id: str | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    pending_silence_owner_id: str | None = None
    last_native_submit_monotonic: float | None = None
    silence_deadline_monotonic: float | None = None

    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None:
        self.committed_audio_payload = payload
        self.committed_audio_operation_id = operation_id
        self.committed_audio_reserved_bytes += max(0, int(reserved_bytes))

    def clear_committed_audio(self) -> int:
        reserved_bytes = self.committed_audio_reserved_bytes
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        self.deferred_response_create = False
        self.deferred_precreate_response = False
        return reserved_bytes

    def clear_continuation(self) -> None:
        self.continuation_owner_id = None
        self.continuation_units = 0
        self.pending_silence_task = None
        self.pending_silence_owner_id = None
        self.last_native_submit_monotonic = None
        self.silence_deadline_monotonic = None


__all__ = ["LycheeServingSessionState"]
