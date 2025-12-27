# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fun-Audio-Chat model for vLLM-Omni.

Supports two modes:
- Speech-to-Text (S2T): Single-stage audio understanding
- Speech-to-Speech (S2S): Three-stage pipeline:
  1. Main (FunAudioChatForConditionalGeneration): Audio → Text + Hidden States
  2. CRQ Decoder (FunAudioChatCRQDecoder): Hidden States → Speech Tokens
  3. CosyVoice (FunAudioChatCosyVoice): Speech Tokens → Audio Waveform
"""

from .audio_encoder import (
    FunAudioChatAudioEncoder,
    FunAudioChatDiscreteEncoder,
)
from .cosyvoice import FunAudioChatCosyVoice
from .crq_decoder import FunAudioChatCRQDecoder
from .fun_audio_chat import FunAudioChatForConditionalGeneration

__all__ = [
    # Main model (Stage 0)
    "FunAudioChatForConditionalGeneration",
    # Audio encoders
    "FunAudioChatAudioEncoder",
    "FunAudioChatDiscreteEncoder",
    # CRQ Decoder (Stage 1)
    "FunAudioChatCRQDecoder",
    # CosyVoice (Stage 2)
    "FunAudioChatCosyVoice",
]
