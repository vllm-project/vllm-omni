# ruff: noqa: N803
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused DAC Snake activation: ``x + 1 / (alpha + 1e-9) * sin(alpha * x) ** 2``.

The eager expression launches about seven elementwise kernels per call, each
reading and writing the whole activation; in the codec's CUDA graphs they are
most of its GPU time. This kernel reads ``x`` once and rounds every
intermediate to the tensor dtype exactly where the eager ops do, so outputs are
bit-identical to the unfused expression. It can also absorb the preceding conv
bias add and residual add, which are otherwise two more full passes each.
"""

import torch
from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _mul_rn(a, b):
    # Explicit round-to-nearest ops are never contracted into FMA, so each
    # product/sum rounds exactly where the separate eager kernels round.
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _dac_snake_kernel(
    x_ptr,
    bias_ptr,
    residual_ptr,
    alpha_ptr,
    inverse_ptr,
    sum_ptr,
    out_ptr,
    stride_b,
    stride_c,
    t_len,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    WRITE_SUM: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    channel = tl.program_id(1).to(tl.int64)
    t = tl.program_id(2) * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = t < t_len
    offset = batch * stride_b + channel * stride_c + t
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    dtype = x.dtype
    xf = x.to(tl.float32)
    if HAS_BIAS:
        # The conv bias add that cuDNN convolutions run as a separate kernel.
        xf = _add_rn(xf, tl.load(bias_ptr + channel).to(tl.float32)).to(dtype).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(residual_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        xf = _add_rn(residual, xf).to(dtype).to(tl.float32)
    if WRITE_SUM:
        tl.store(sum_ptr + offset, xf.to(dtype), mask=mask)
    alpha = tl.load(alpha_ptr + channel).to(tl.float32)
    inverse = tl.load(inverse_ptr + channel).to(tl.float32)
    scaled = _mul_rn(alpha, xf).to(dtype).to(tl.float32)
    # libdevice sinf, as the eager CUDA kernel (tl.sin is the approximation).
    sine = tldevice.sin(scaled).to(dtype).to(tl.float32)
    square = _mul_rn(sine, sine).to(dtype).to(tl.float32)
    term = _mul_rn(inverse, square).to(dtype).to(tl.float32)
    tl.store(out_ptr + offset, _add_rn(xf, term).to(dtype), mask=mask)


def fused_snake1d(
    x: torch.Tensor,
    alpha: torch.Tensor,
    inverse: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    write_sum: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Snake of ``residual + (x + bias)``, rounded as the separate eager ops.

    x, residual: (B, C, T); alpha, inverse, bias: (C,), all in x's dtype.
    Returns ``(sum, snake(sum))``; ``sum`` is only materialized when
    ``write_sum`` is set (for the next residual connection), else None.
    """
    x = x.contiguous()
    if residual is not None:
        residual = residual.contiguous()
    batch, channels, t_len = x.shape
    out = torch.empty_like(x)
    total = torch.empty_like(x) if write_sum else None
    if x.numel() == 0:
        return total, out
    block = min(triton.next_power_of_2(t_len), 2048)
    _dac_snake_kernel[(batch, channels, triton.cdiv(t_len, block))](
        x,
        x if bias is None else bias,
        x if residual is None else residual,
        alpha,
        inverse,
        out if total is None else total,
        out,
        x.stride(0),
        x.stride(1),
        t_len,
        HAS_BIAS=bias is not None,
        HAS_RESIDUAL=residual is not None,
        WRITE_SUM=write_sum,
        BLOCK_T=block,
    )
    return total, out
