# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared handler protocol padding for Qwen3-Omni duplex sessions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from vllm_omni.entrypoints.duplex.runtime_adapter import PcmAppendReservation


class _NoopPcmAppendReservation:
    operation_id: str = ""
    payload: dict[str, object] | None = None
    active: bool = False
    byte_count: int = 0

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class _NoopPcmAppendBuffer:
    """Minimal PcmAppendBuffer for handler cleanup paths."""

    pending_byte_count: int = 0

    def clear(self) -> None: ...

    def clear_force_listen(self) -> None: ...

    def has_pending(self) -> bool:
        return False

    def has_reserved(self) -> bool:
        return False

    def prepare_append(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        chunk_period_ms: int,
        allow_emit: bool,
    ) -> PcmAppendReservation | None:
        del payload, operation_id, chunk_period_ms, allow_emit
        return None

    def prepare_commit(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> PcmAppendReservation:
        del operation_id, chunk_period_ms
        return _NoopPcmAppendReservation()

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        del chunk_period_ms
        return None


class Qwen3OmniServingSessionState:
    """Handler protocol padding for one Qwen3-Omni duplex session.

    The live conversation history is owned by ``DuplexSession.history``. The
    fields below exist for shared handler cleanup and adapter hooks; they are
    not a second production conversation ledger or audio input buffer.
    """

    def __init__(self) -> None:
        self.last_turn_interrupted = False
        self.audio_buffer = _NoopPcmAppendBuffer()
        self.input_since_commit = False
        self.speech_since_commit = False
        self.native_context_locked = False
        self.committed_audio_payload: dict[str, object] | None = None
        self.committed_audio_operation_id: str | None = None
        self.committed_audio_reserved_bytes = 0
        self.deferred_response_create = False
        self.deferred_precreate_response = False
        self.data_plane_task: asyncio.Task[None] | None = None
        self.data_plane_restart_requested = False
        self.continuation_owner_id: str | None = None
        self.continuation_units = 0
        self.pending_silence_task: asyncio.Task[bool] | None = None
        self.pending_silence_owner_id: str | None = None
        self.silence_continuation_scheduler: Callable[..., object] | None = None

    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None:
        self.committed_audio_payload = payload
        self.committed_audio_operation_id = operation_id
        self.committed_audio_reserved_bytes = reserved_bytes

    def clear_committed_audio(self) -> int:
        released = self.committed_audio_reserved_bytes
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        return released

    def clear_continuation(self) -> None:
        self.continuation_owner_id = None
        self.continuation_units = 0
