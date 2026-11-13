# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax Music 3 text-to-music.

Stage 0 is a Qwen3 backbone with an eight-codebook RVQ frame; stage 1 turns
the resulting conditioning frames into 32 kHz stereo audio through a
flow-matching transformer and a DAC decoder.
"""

from .constants import (
    AR_CHUNK_FRAMES,
    AR_CHUNK_HOP_FRAMES,
    AR_HIDDEN_SIZE,
    MAX_AUDIO_FRAMES,
    OUTPUT_SAMPLE_RATE,
)
from .prompt import (
    AUDIO_CODE_OFFSET,
    SPECIAL_TOKEN_IDS,
    build_cfg_null_token_ids,
    build_prompt,
)

__all__ = [
    "AR_CHUNK_FRAMES",
    "AR_CHUNK_HOP_FRAMES",
    "AR_HIDDEN_SIZE",
    "AUDIO_CODE_OFFSET",
    "MAX_AUDIO_FRAMES",
    "OUTPUT_SAMPLE_RATE",
    "SPECIAL_TOKEN_IDS",
    "build_cfg_null_token_ids",
    "build_prompt",
]
