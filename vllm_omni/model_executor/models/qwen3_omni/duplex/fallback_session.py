# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Model policy state for Qwen3's opt-in chat fallback."""

from dataclasses import dataclass, field

from vllm_omni.engine.duplex.plugin import DefaultDuplexModelSessionState
from vllm_omni.model_executor.models.qwen3_omni.duplex.fallback_input import Qwen3OmniPcmAppendBuffer


@dataclass(slots=True)
class Qwen3OmniFallbackSessionState(DefaultDuplexModelSessionState):
    """Runner protocol state; live history belongs to ``DuplexEngineSession``."""

    audio_buffer: Qwen3OmniPcmAppendBuffer = field(default_factory=Qwen3OmniPcmAppendBuffer)
    last_turn_interrupted: bool = False
