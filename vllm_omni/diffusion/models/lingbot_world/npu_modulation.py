# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in LingBot NPU modulation kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _residual_camera(
    hidden,
    update,
    gate,
    scale,
    shift,
    output,
    tokens,
    width: tl.constexpr,
    tokens_per_frame: tl.constexpr,
    gate_stride: tl.constexpr,
    camera: tl.constexpr,
    tile: tl.constexpr,
):
    columns = tl.arange(0, tile)
    valid = columns < width
    for row in range(tl.program_id(0), tokens, tl.num_programs(0)):
        offsets = row * width + columns
        gate_offsets = row // tokens_per_frame * gate_stride + columns
        x = tl.load(hidden + offsets, valid, 0).to(tl.float32)
        y = tl.load(update + offsets, valid, 0).to(tl.float32)
        g = tl.load(gate + gate_offsets, valid, 0)
        residual = (x + y * g).to(tl.bfloat16)
        if camera:
            s = tl.load(scale + offsets, valid, 0).to(tl.float32)
            b = tl.load(shift + offsets, valid, 0).to(tl.float32)
            s = (1.0 + s).to(tl.bfloat16).to(tl.float32)
            product = (s * residual.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            residual = (product + b).to(tl.bfloat16)
        tl.store(output + offsets, residual, valid)


def residual_camera(hidden, update, gate, tokens_per_frame, *, scale=None, shift=None):
    """Fuse the FP32 gate and optional BF16 camera affine, or decline unsupported inputs."""
    if (
        hidden.ndim != 3
        or hidden.shape[0] != 1
        or hidden.shape[1] == 0
        or not 0 < hidden.shape[-1] <= 8192
        or hidden.device.type != "npu"
        or hidden.dtype != torch.bfloat16
        or update.shape != hidden.shape
        or update.dtype != hidden.dtype
        or gate.dtype != torch.float32
        or gate.shape != (1, hidden.shape[1] // tokens_per_frame, 1, hidden.shape[2])
        or gate.stride(-1) != 1
        or not hidden.is_contiguous()
        or not update.is_contiguous()
    ):
        return None
    camera = scale is not None
    if camera and any(
        x is None or x.shape != hidden.shape or x.dtype != hidden.dtype or not x.is_contiguous() for x in (scale, shift)
    ):
        return None
    output = torch.empty_like(hidden)
    tokens = hidden.shape[1]
    tile = triton.next_power_of_2(hidden.shape[-1])
    _residual_camera[(min(tokens, 56),)](
        hidden,
        update,
        gate,
        scale if camera else hidden,
        shift if camera else hidden,
        output,
        tokens,
        hidden.shape[-1],
        tokens_per_frame,
        gate.stride(1),
        camera,
        tile,
        enable_fp_fusion=False,
    )
    return output


def adaln(norm, hidden, scale, shift, tokens_per_frame):
    """Use the provider's fused LayerNorm and scale/shift with FP32 inputs."""
    if hidden.device.type != "npu" or hidden.dtype != torch.bfloat16:
        return None
    try:
        from mindiesd import layernorm_scale_shift
    except ImportError:
        return None
    x = hidden.float()
    if tokens_per_frame != 1:
        x = x.view(-1, tokens_per_frame, x.shape[-1])
        scale, shift = (value.reshape(-1, value.shape[-1]) for value in (scale, shift))
    else:
        scale, shift = scale.squeeze(2), shift.squeeze(2)
    return layernorm_scale_shift(norm, x, scale, shift, fused=True).reshape_as(hidden).to(hidden.dtype)
