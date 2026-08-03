# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving-side session state for one Qwen3-Omni duplex session.

Satisfies the ``ServingRuntimeSessionState`` protocol in
``vllm_omni/experimental/fullduplex/openai/runtime_adapter.py``. Every field
below is read by the generic serving layer, so none of them can be dropped
even where Qwen3-Omni has no use for the concept.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from vllm_omni.experimental.fullduplex.qwen3omni.input import Qwen3OmniPcmAppendBuffer


@dataclass(slots=True)
class Qwen3OmniServingSessionState:
    """Mutable serving state owned by one Qwen3-Omni duplex session."""

    audio_buffer: Qwen3OmniPcmAppendBuffer = field(default_factory=Qwen3OmniPcmAppendBuffer)
    input_since_commit: bool = False
    speech_since_commit: bool = False
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    data_plane_task: asyncio.Task[None] | None = None
    data_plane_restart_requested: bool = False
    # Continuation / silence scheduling exists for MiniCPM's model-owned
    # listen policy, where the server re-issues a "keep listening" unit when
    # the model stays silent. Qwen3-Omni has no such policy, so these stay at
    # their defaults; they are retained because the generic session runner
    # reads them unconditionally.
    continuation_owner_id: str | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    pending_silence_owner_id: str | None = None
    silence_continuation_scheduler: Callable[..., Awaitable[bool]] | None = None

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
