# SPDX-License-Identifier: Apache-2.0
"""Config shim for Fun-Audio-Chat-8B.

vllm >= 0.19 ships its own FunAudioChatConfig / FunAudioChatAudioEncoderConfig
at vllm.transformers_utils.configs.funaudiochat. When we registered our own
copies with AutoConfig they collided at worker init:

    TypeError: Invalid type of HuggingFace config.
    Expected: vllm_omni.transformers_utils.configs.fun_audio_chat.FunAudioChatConfig
    Found:    vllm.transformers_utils.configs.funaudiochat.FunAudioChatConfig

So we re-export vllm's classes under our existing names. Our ported modules
(encoder.py, crq_decoder.py, fun_audio_chat.py) only use `audio_config`
fields (codebook_size, bos_token_id, eos_token_id, pad_token_id,
continuous_features_mode, crq_transformer_config, group_size, n_window,
output_dim, etc.) — all of which vllm's config also exposes, verified via
the checkpoint config.json round-trip.
"""
from __future__ import annotations

from vllm.transformers_utils.configs.funaudiochat import (
    FunAudioChatAudioEncoderConfig,
    FunAudioChatConfig,
)

# Legacy alias kept so existing imports continue to work.
FunAudioChatAudioConfig = FunAudioChatAudioEncoderConfig

__all__ = [
    "FunAudioChatAudioEncoderConfig",
    "FunAudioChatConfig",
    "FunAudioChatAudioConfig",
]
