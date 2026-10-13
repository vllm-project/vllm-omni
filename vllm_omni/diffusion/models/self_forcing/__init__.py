# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Self-Forcing causal Wan2.1 text-to-video model."""

from vllm_omni.diffusion.models.self_forcing.pipeline_self_forcing import (
    SelfForcingPipeline,
    get_self_forcing_post_process_func,
)
from vllm_omni.diffusion.models.self_forcing.transformer import SelfForcingTransformer3DModel

__all__ = [
    "SelfForcingPipeline",
    "SelfForcingTransformer3DModel",
    "get_self_forcing_post_process_func",
]
