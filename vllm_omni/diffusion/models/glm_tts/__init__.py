# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GLM-TTS model support for vLLM-Omni."""

from vllm_omni.diffusion.models.glm_tts.glm_tts_dit import (
    GLMTTSDiTModel,
)
from vllm_omni.diffusion.models.glm_tts.pipeline_glm_tts import (
    GLMTTSPipeline,
    get_glm_tts_post_process_func,
)

__all__ = [
    "GLMTTSDiTModel",
    "GLMTTSPipeline",
    "get_glm_tts_post_process_func",
]
