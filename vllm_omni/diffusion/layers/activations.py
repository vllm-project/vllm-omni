# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.layers.linear import ColumnParallelLinear

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationConfig


class ColumnParallelGELU(nn.Module):
    """Column-parallel linear followed by GELU activation.

    Wraps a :class:`ColumnParallelLinear` that keeps its output partitioned
    across tensor-parallel ranks (``gather_output=False``).

    The linear and activation are bundled into a single module to preserve
    the ``net.0.proj`` weight path expected by diffusers checkpoints.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        approximate: str = "none",
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim_in,
            dim_out,
            bias=bias,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return F.gelu(x, approximate=self.approximate)
