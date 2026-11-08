# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA NemotronLabs VoiceChat-11B: offline speech-to-speech 3-stage pipeline."""

from vllm_omni.model_executor.models.nemotron_voicechat.configuration_nemotron_voicechat import (
    NemotronVoiceChatCode2WavConfig,
    NemotronVoiceChatConfig,
    NemotronVoiceChatTalkerConfig,
)
from vllm_omni.model_executor.models.nemotron_voicechat.pipeline import (
    NEMOTRON_VOICECHAT_PIPELINE,
)

__all__ = [
    "NemotronVoiceChatConfig",
    "NemotronVoiceChatTalkerConfig",
    "NemotronVoiceChatCode2WavConfig",
    "NEMOTRON_VOICECHAT_PIPELINE",
]
