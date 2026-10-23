# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Backward-compatible import path for shared merged LoRA layer wrappers."""

from vllm_omni.lora.layers import (
    DiffusionBaseLinearLayerWithLoRA,
    DiffusionColumnParallelLinearWithLoRA,
    DiffusionMergedColumnParallelLinearWithLoRA,
    DiffusionMergedQKVParallelLinearWithLoRA,
    DiffusionQKVParallelLinearWithLoRA,
    DiffusionReplicatedLinearWithLoRA,
    DiffusionRowParallelLinearWithLoRA,
)

from .row_parallel_linear import DiffusionMergedRowParallelLinearWithLoRA

__all__ = [
    "DiffusionBaseLinearLayerWithLoRA",
    "DiffusionColumnParallelLinearWithLoRA",
    "DiffusionMergedColumnParallelLinearWithLoRA",
    "DiffusionMergedQKVParallelLinearWithLoRA",
    "DiffusionMergedRowParallelLinearWithLoRA",
    "DiffusionQKVParallelLinearWithLoRA",
    "DiffusionReplicatedLinearWithLoRA",
    "DiffusionRowParallelLinearWithLoRA",
]
