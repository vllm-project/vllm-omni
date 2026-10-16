# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-pass per-token FP8 E4M3 quantization for DiT activations.

Each program reads one row once, takes its absmax, and writes the FP8 row and its
dequantization scale ``max(absmax / 448, 1 / (448 * 512))`` (the same scale bounds as
vLLM's dynamic per-token quantizer). DiT rows are 3072-18432 wide, so a whole row fits
one program and the activation is read once instead of twice.
"""

from __future__ import annotations

import torch
from torch.library import Library
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_FP8_MIN_SCALE = 1.0 / (_FP8_MAX * 512.0)


@triton.jit
def _per_token_fp8_quant_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    x_row_stride,
    n_cols,
    fp8_max: tl.constexpr,
    min_scale: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, block_size)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0) / fp8_max, min_scale)
    q = tl.clamp(x / scale, -fp8_max, fp8_max)
    tl.store(out_ptr + row * n_cols + cols, q.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(scale_ptr + row, scale)


def _per_token_fp8_quant_impl(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = x.shape
    out = torch.empty((rows, cols), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    if rows:
        block = triton.next_power_of_2(cols)
        _per_token_fp8_quant_kernel[(rows,)](
            x,
            out,
            scale,
            x.stride(0),
            cols,
            fp8_max=_FP8_MAX,
            min_scale=_FP8_MIN_SCALE,
            block_size=block,
            num_warps=min(max(block // 512, 4), 16),
        )
    return out, scale


def _per_token_fp8_quant_fake(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn),
        torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32),
    )


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "per_token_fp8_quant"):
    direct_register_custom_op(
        op_name="per_token_fp8_quant",
        op_func=_per_token_fp8_quant_impl,
        fake_impl=_per_token_fp8_quant_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def per_token_fp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a ``[M, K]`` activation (unit stride in K) to FP8 with per-row scales ``[M, 1]``."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    return torch.ops.vllm_omni.per_token_fp8_quant(x)


__all__ = ["per_token_fp8_quant"]
