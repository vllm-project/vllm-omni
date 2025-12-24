# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fun-Audio-Chat model for vLLM-Omni."""

from .audio_encoder import (
    FunAudioChatAudioEncoder,
    FunAudioChatDiscreteEncoder,
)
from .fun_audio_chat import FunAudioChatForConditionalGeneration

__all__ = [
    "FunAudioChatForConditionalGeneration",
    "FunAudioChatAudioEncoder",
    "FunAudioChatDiscreteEncoder",
]
