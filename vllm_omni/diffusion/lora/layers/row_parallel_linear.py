# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
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

    def set_additive_bias(
        self,
        bias: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        # Row-parallel outputs are all-reduced after apply(). Match the base
        # layer's bias semantics by contributing the replicated delta on rank
        # zero only; otherwise the collective would multiply it by TP size.
        if self.tp_size > 1 and self.tp_rank > 0:
            bias = None
        super().set_additive_bias(bias)
