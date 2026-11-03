# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass, field

from vllm_omni.engine.duplex.plugin import DefaultDuplexModelSessionState
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.input import (
    MiniCPMO45PcmAppendBuffer,
)


@dataclass(slots=True)
class MiniCPMO45ServingSessionState(DefaultDuplexModelSessionState):
    """Mutable model-owned state of one MiniCPM duplex session (owned by the session runner).

    The runner-facing flag contract lives on ``DefaultDuplexModelSessionState``;
    MiniCPM-o only binds its ~1 s PCM unit buffer.
    """

    audio_buffer: MiniCPMO45PcmAppendBuffer = field(default_factory=MiniCPMO45PcmAppendBuffer)
