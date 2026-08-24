# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.lora.layers.column_parallel_linear import (
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
)

from .base_linear import DiffusionBaseLinearLayerWithLoRA


class DiffusionColumnParallelLinearWithLoRA(
    DiffusionBaseLinearLayerWithLoRA,
    ColumnParallelLinearWithLoRA,
):
    """
    Diffusion ColumnParallelLinear with LoRA.
    Prioritize apply() in DiffusionBaseLinearLayerWithLoRA
    """

    pass


class DiffusionMergedColumnParallelLinearWithLoRA(
    DiffusionBaseLinearLayerWithLoRA,
    MergedColumnParallelLinearWithLoRA,
):
    """
    Diffusion MergedColumnParallelLinear (gate_up_proj) with LoRA.
    Prioritize apply() in DiffusionBaseLinearLayerWithLoRA
    """

    pass


class DiffusionQKVParallelLinearWithLoRA(
    DiffusionBaseLinearLayerWithLoRA,
    QKVParallelLinearWithLoRA,
):
    """
    Diffusion QKVParallelLinear with single LoRA.
    Prioritize apply() in DiffusionBaseLinearLayerWithLoRA
    """

    pass


class DiffusionMergedQKVParallelLinearWithLoRA(
    DiffusionBaseLinearLayerWithLoRA,
    MergedQKVParallelLinearWithLoRA,
):
    """
    Diffusion MergedQKVParallelLinear (to_qkv) with 3 LoRAs.
    Prioritize apply() in DiffusionBaseLinearLayerWithLoRA
    """

    pass
