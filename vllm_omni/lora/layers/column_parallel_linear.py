# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.lora.layers.column_parallel_linear import (
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
)

from .base_linear import OmniBaseLinearLayerWithLoRA


class OmniColumnParallelLinearWithLoRA(
    OmniBaseLinearLayerWithLoRA,
    ColumnParallelLinearWithLoRA,
):
    """
    ColumnParallelLinear with the shared Omni LoRA merge binding.
    """

    pass


class OmniMergedColumnParallelLinearWithLoRA(
    OmniBaseLinearLayerWithLoRA,
    MergedColumnParallelLinearWithLoRA,
):
    """
    MergedColumnParallelLinear (gate_up_proj) with the shared Omni LoRA merge binding.
    """

    pass


class OmniQKVParallelLinearWithLoRA(
    OmniBaseLinearLayerWithLoRA,
    QKVParallelLinearWithLoRA,
):
    """
    QKVParallelLinear with one LoRA slice.
    """

    pass


class OmniMergedQKVParallelLinearWithLoRA(
    OmniBaseLinearLayerWithLoRA,
    MergedQKVParallelLinearWithLoRA,
):
    """
    MergedQKVParallelLinear (to_qkv) with three LoRA slices.
    """

    pass


DiffusionColumnParallelLinearWithLoRA = OmniColumnParallelLinearWithLoRA
DiffusionMergedColumnParallelLinearWithLoRA = OmniMergedColumnParallelLinearWithLoRA
DiffusionQKVParallelLinearWithLoRA = OmniQKVParallelLinearWithLoRA
DiffusionMergedQKVParallelLinearWithLoRA = OmniMergedQKVParallelLinearWithLoRA
