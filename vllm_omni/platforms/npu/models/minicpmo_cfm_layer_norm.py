# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused gated residual and LayerNorm for MiniCPM-o FP32 CFM on Ascend."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gated_residual_layer_norm_kernel(
    x_ptr,
    y_ptr,
    gate_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    total,
    frames,
    xn,
    xt,
    yn,
    yt,
    eps: tl.constexpr,
    channels: tl.constexpr,
    tile_rows: tl.constexpr,
    has_gate: tl.constexpr,
):
    rows = tl.program_id(0) * tile_rows + tl.arange(0, tile_rows)
    cols = tl.arange(0, channels)
    n, t = rows // frames, rows % frames
    mask = rows[:, None] < total
    x = tl.load(x_ptr + n[:, None] * xn + t[:, None] * xt + cols[None, :], mask, 0)
    y = tl.load(y_ptr + n[:, None] * yn + t[:, None] * yt + cols[None, :], mask, 0)
    if has_gate:
        y = y * tl.load(gate_ptr + cols)[None, :]
    value = x + y
    tl.store(residual_ptr + rows[:, None] * channels + cols[None, :], value, mask)
    mean = tl.sum(value, 1) / channels
    centered = value - mean[:, None]
    variance = tl.sum(centered * centered, 1) / channels
    normalized = centered * tl.rsqrt(variance + eps)[:, None]
    normalized = normalized * tl.load(weight_ptr + cols)[None, :] + tl.load(bias_ptr + cols)[None, :]
    tl.store(out_ptr + rows[:, None] * channels + cols[None, :], normalized, mask)


def fused_residual_layer_norm(x, y, weight, bias, eps, *, gate=None, tile=None):
    assert x.dtype == y.dtype == torch.float32 and x.shape[-1] == 512
    assert x.stride(-1) == y.stride(-1) == 1
    residual = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    batch, frames, channels = x.shape
    if tile is None:
        tile = min(32, triton.next_power_of_2(max(1, (batch * frames + 39) // 40)))
    _gated_residual_layer_norm_kernel[(triton.cdiv(batch * frames, tile),)](
        x,
        y,
        x if gate is None else gate,
        weight,
        bias,
        residual,
        output,
        batch * frames,
        frames,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        eps=eps,
        channels=channels,
        tile_rows=tile,
        has_gate=gate is not None,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return residual, output
