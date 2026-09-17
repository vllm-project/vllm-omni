# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.model_executor.models.qwen3_omni.duplex.input import Qwen3OmniPcmAppendBuffer
from vllm_omni.model_executor.models.qwen3_omni.duplex.turn_commit import TurnCommitSessionState


@dataclass(slots=True)
class Qwen3OmniDuplexSessionState(TurnCommitSessionState):
    """Per-session Qwen3-Omni duplex state (owned by SessionRunner)."""

    audio_buffer: Qwen3OmniPcmAppendBuffer = field(default_factory=Qwen3OmniPcmAppendBuffer)


__all__ = ["Qwen3OmniDuplexSessionState"]
