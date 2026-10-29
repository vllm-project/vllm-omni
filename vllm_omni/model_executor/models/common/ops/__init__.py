# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Common operators shared across models and hardware backends.

This module contains fused operators that can be reused across multiple models
and compiled for different hardware targets (CUDA, ROCm, etc.).
"""

from vllm_omni.model_executor.models.common.ops.alias_free_triton import (
    alias_free_snake,
)
from vllm_omni.model_executor.models.common.ops.fused_adaptive_group_norm_silu import (
    fused_adaptive_group_norm_silu,
)
from vllm_omni.model_executor.models.common.ops.fused_group_norm_silu import (
    fused_group_norm_silu,
)

__all__ = [
    "fused_group_norm_silu",
    "fused_adaptive_group_norm_silu",
    "alias_free_snake",
]
