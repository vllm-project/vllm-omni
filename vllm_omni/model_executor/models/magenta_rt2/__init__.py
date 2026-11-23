# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Google Magenta RealTime 2 model exports."""

from vllm_omni.model_executor.models.magenta_rt2.magenta_rt2_code2wav import SpectroStreamCode2Wav
from vllm_omni.model_executor.models.magenta_rt2.magenta_rt2_talker import (
    MagentaRT2ForConditionalGeneration,
    MagentaRT2ModelState,
)

__all__ = [
    "MagentaRT2ForConditionalGeneration",
    "MagentaRT2ModelState",
    "SpectroStreamCode2Wav",
]
