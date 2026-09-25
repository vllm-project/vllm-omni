# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.engine.duplex.plugin import DefaultDuplexModelSessionState
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.input import (
    MiniCPMO45PcmAppendBuffer,
)


@dataclass
class MiniCPMO45ServingSessionState(DefaultDuplexModelSessionState):
    """Mutable model-owned state of one MiniCPM duplex session (owned by the session runner).

    Only the audio buffer is MiniCPM's; the flags and their transitions are the
    framework default.
    """

    audio_buffer: MiniCPMO45PcmAppendBuffer = field(default_factory=MiniCPMO45PcmAppendBuffer)
