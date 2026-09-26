# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in dynamic W8A8 video projections for SeedVR2."""

from typing import Literal

import torch
from torch import nn
from vllm import _custom_ops as ops

QuantizationMode = Literal["fp8", "int8"]


class SeedVR2W8A8Linear(nn.Module):
    def __init__(self, linear: nn.Linear, mode: QuantizationMode) -> None:
        super().__init__()
        self.mode = mode
        if mode == "fp8":
            weight, scale = ops.scaled_fp8_quant(linear.weight, use_per_token_if_dynamic=True)
        else:
            weight, scale, _ = ops.scaled_int8_quant(linear.weight)
        self.register_buffer("weight", weight.t())
        self.register_buffer("scale", scale.t())
        self.register_buffer("bias", linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_empty((0, self.weight.shape[1]))
        if self.mode == "fp8":
            quantized, scale = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        else:
            quantized, scale, _ = ops.scaled_int8_quant(x)
        return ops.cutlass_scaled_mm(quantized, self.weight, scale, self.scale, x.dtype, self.bias)


def quantize_video_projections(transformer: nn.Module, mode: QuantizationMode) -> None:
    # Keep independent text projections and modulation in FP16. Shared blocks
    # use the same quantized weights for both streams.
    for name, module in list(transformer.named_modules()):
        projection = ".attn.proj_" in name or ".mlp." in name
        if isinstance(module, nn.Linear) and projection and ".txt" not in name:
            parent_name, child_name = name.rsplit(".", 1)
            transformer.get_submodule(parent_name).add_module(child_name, SeedVR2W8A8Linear(module, mode))
