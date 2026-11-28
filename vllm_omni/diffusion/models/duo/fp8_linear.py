# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FP8 W8A8 body projections using vLLM CUTLASS; TP stays in native Linear."""

import torch
from torch import nn
from vllm import _custom_ops as ops


class RealizationFP8LinearMethod:
    def apply(self, layer, x, bias=None):
        rows = x.shape[:-1]
        q, scale = ops.scaled_fp8_quant(x.reshape(-1, x.shape[-1]), use_per_token_if_dynamic=True)
        out = ops.cutlass_scaled_mm(
            q, layer.weight.t(), scale, layer.realization_weight_scale.t(), torch.bfloat16, bias
        )
        return out.view(*rows, out.shape[-1])


@torch.no_grad()
def install_fp8_body(model):
    """Quantize after merging the fixed adapter, before capturing model graphs.

    Per-output-channel weights and dynamic per-token activations.
    """
    names = []
    for index, block in enumerate(model.blocks):
        for path in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            layer = block.get_submodule(path)
            w = layer.weight.detach()
            scale = w.float().abs().amax(1, keepdim=True).clamp_min(1e-12) / 448
            quant = (w.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            layer.register_buffer("realization_weight_scale", scale)
            layer.register_buffer("realization_fp8_weight", quant)
            layer.weight = nn.Parameter(quant, requires_grad=False)
            layer.quant_method = RealizationFP8LinearMethod()
            names.append(f"blocks.{index}.{path}")
    model._realization_fp8_body = tuple(names)
    return dict(
        projections=len(names),
        weight="e4m3_per_output_channel",
        activation="e4m3_dynamic_per_token",
        output="bf16",
        backend="vllm_cutlass_scaled_mm",
        reference_on_cpu=False,
    )
