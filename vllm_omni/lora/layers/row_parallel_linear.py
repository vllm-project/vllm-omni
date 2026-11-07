# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.lora.layers.row_parallel_linear import RowParallelLinearWithLoRA

from .base_linear import OmniBaseLinearLayerWithLoRA


class OmniRowParallelLinearWithLoRA(
    OmniBaseLinearLayerWithLoRA,
    RowParallelLinearWithLoRA,
):
    """
    RowParallelLinear with the shared Omni LoRA merge binding.
    """

    pass


DiffusionRowParallelLinearWithLoRA = OmniRowParallelLinearWithLoRA
