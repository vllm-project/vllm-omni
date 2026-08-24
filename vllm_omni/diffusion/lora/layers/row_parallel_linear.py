# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.lora.layers.row_parallel_linear import RowParallelLinearWithLoRA

from .base_linear import DiffusionBaseLinearLayerWithLoRA


class DiffusionRowParallelLinearWithLoRA(
    DiffusionBaseLinearLayerWithLoRA,
    RowParallelLinearWithLoRA,
):
    """
    Diffusion RowParallelLinear with LoRA.
    Prioritize apply() in DiffusionBaseLinearLayerWithLoRA
    """

    pass
