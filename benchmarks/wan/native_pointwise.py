# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve the native BF16 rounding after every multiply and add."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _modulate(x_ptr, scale_ptr, shift_ptr, out_ptr, count: tl.constexpr, width: tl.constexpr, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    mask = i < count
    x = tl.load(x_ptr + i, mask, 0).to(tl.float32)
    scale = tl.load(scale_ptr + i % width).to(tl.float32)
    shift = tl.load(shift_ptr + i % width).to(tl.float32)
    factor = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
    product = (x * factor).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + i, product + shift, mask)


@triton.jit
def _residual(
    x_ptr,
    delta_ptr,
    gate_ptr,
    out_ptr,
    count: tl.constexpr,
    width: tl.constexpr,
    x_row_stride: tl.constexpr,
    x_col_stride: tl.constexpr,
    delta_row_stride: tl.constexpr,
    delta_col_stride: tl.constexpr,
    block: tl.constexpr,
):
    i = tl.program_id(0) * block + tl.arange(0, block)
    mask = i < count
    row, col = i // width, i % width
    x = tl.load(x_ptr + row * x_row_stride + col * x_col_stride, mask, 0).to(tl.float32)
    delta = tl.load(delta_ptr + row * delta_row_stride + col * delta_col_stride, mask, 0).to(tl.float32)
    gate = tl.load(gate_ptr + col).to(tl.float32)
    product = (delta * gate).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + i, x + product, mask)


def modulation(norm, scale, shift):
    assert norm.ndim == 3 and norm.shape[0] == 1 and norm.is_contiguous()
    assert norm.dtype == scale.dtype == shift.dtype == torch.bfloat16
    result = torch.empty_like(norm)
    _modulate[(triton.cdiv(norm.numel(), 256),)](
        norm, scale, shift, result, norm.numel(), norm.shape[-1], 256, enable_fp_fusion=False
    )
    return result


def residual(hidden, delta, gate):
    assert hidden.ndim == 3 and hidden.shape[0] == 1 and hidden.shape == delta.shape
    assert hidden.dtype == delta.dtype == gate.dtype == torch.bfloat16
    result = torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    _residual[(triton.cdiv(hidden.numel(), 256),)](
        hidden,
        delta,
        gate,
        result,
        hidden.numel(),
        hidden.shape[-1],
        *hidden.stride()[1:],
        *delta.stride()[1:],
        256,
        enable_fp_fusion=False,
    )
    return result
