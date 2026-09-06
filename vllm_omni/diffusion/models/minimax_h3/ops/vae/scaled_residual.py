# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exact scaled residual update for the MiniMax H3 video VAE."""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _mul_rn_f32(x, y):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[x, y],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _scaled_residual_exact_kernel(
        output_ptr,
        residual_ptr,
        branch_ptr,
        scale_ptr,
        residual_stride_row,
        branch_stride_row,
        hidden_size: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, block_n)
        mask = columns < hidden_size
        residual = tl.load(
            residual_ptr + row * residual_stride_row + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        branch = tl.load(
            branch_ptr + row * branch_stride_row + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(scale_ptr + columns, mask=mask, other=0.0).to(tl.float32)
        # Prevent contraction into FMA and retain the eager rounding boundary.
        updated = residual + _mul_rn_f32(branch, scale)
        tl.store(output_ptr + row * hidden_size + columns, updated, mask=mask)


def try_scaled_residual_exact(
    residual: torch.Tensor,
    branch: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor | None:
    """Return the optimized result, or ``None`` outside its exact contract."""

    if not (
        HAS_TRITON
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and residual.is_cuda
        and branch.is_cuda
        and residual.device == branch.device == scale.device
        and residual.dtype == torch.float32
        and branch.dtype == torch.float16
        and scale.dtype == torch.float32
        and residual.shape == branch.shape
        and residual.ndim >= 1
        and residual.shape[-1] == 2048
        and scale.shape == residual.shape[-1:]
        and residual.numel() > 0
        and residual.is_contiguous()
        and branch.is_contiguous()
        and scale.is_contiguous()
    ):
        return None

    hidden_size = residual.shape[-1]
    residual_2d = residual.reshape(-1, hidden_size)
    branch_2d = branch.reshape(-1, hidden_size)
    output = torch.empty_like(residual_2d)
    rows = residual_2d.shape[0]
    _scaled_residual_exact_kernel[(rows,)](
        output,
        residual_2d,
        branch_2d,
        scale,
        residual_2d.stride(0),
        branch_2d.stride(0),
        hidden_size=hidden_size,
        block_n=triton.next_power_of_2(hidden_size),
        num_warps=8,
    )
    return output.reshape_as(residual)


__all__ = ["try_scaled_residual_exact"]
