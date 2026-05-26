# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-TTS model support for vLLM-Omni.

GLM-TTS is a two-stage text-to-speech system:
  - Stage 0 (LLM): Llama-based AR model generates speech tokens from text
  - Stage 1 (DiT): Flow matching model converts speech tokens to mel-spectrogram

Reference: https://github.com/zai-org/GLM-TTS
"""

from vllm_omni.model_executor.models.glm_tts.glm_tts import (
    GLMTTSForConditionalGeneration,
)

__all__ = [
    "GLMTTSForConditionalGeneration",
]
