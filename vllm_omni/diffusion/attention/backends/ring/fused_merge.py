# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Inference-only CUDA fusion of the Ring output and log-sum-exp update."""

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

if HAS_TRITON and torch.version.hip is None and current_platform.is_cuda():

    @triton.jit
    def _ring_merge_kernel(
        out_ptr,
        lse_ptr,
        block_out_ptr,
        block_lse_ptr,
        merged_out_ptr,
        merged_lse_ptr,
        rows,
        sequence: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        out_strides: tl.constexpr,
        lse_strides: tl.constexpr,
        block_out_strides: tl.constexpr,
        block_lse_strides: tl.constexpr,
        merged_out_strides: tl.constexpr,
        merged_lse_strides: tl.constexpr,
        block_rows: tl.constexpr,
        block_dim: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
        dim = tl.arange(0, block_dim).to(tl.int64)
        batch = row // (sequence * heads)
        token = (row // heads) % sequence
        head = row % heads
        valid_row = row < rows

        lse_offset = batch * lse_strides[0] + token * lse_strides[1] + head * lse_strides[2]
        block_lse_offset = batch * block_lse_strides[0] + token * block_lse_strides[1] + head * block_lse_strides[2]
        old_lse = tl.load(lse_ptr + lse_offset, valid_row, other=0.0)
        block_lse = tl.load(block_lse_ptr + block_lse_offset, valid_row, other=0.0)
        weight = tl.div_rn(1.0, 1.0 + tldevice.exp(-(block_lse - old_lse)))

        # Stable logsigmoid(x) = min(x, 0) - log1p(exp(-abs(x))). Keep the
        # reference subtraction order, including its nonfinite behavior.
        delta = old_lse - block_lse
        log_sigmoid = tl.minimum(delta, 0.0) - tldevice.log1p(tldevice.exp(-tl.abs(delta)))
        merged_lse = old_lse - log_sigmoid

        out_offset = batch * out_strides[0] + token * out_strides[1] + head * out_strides[2]
        block_out_offset = batch * block_out_strides[0] + token * block_out_strides[1] + head * block_out_strides[2]
        valid = valid_row[:, None] & (dim[None, :] < head_dim)
        old_out = tl.load(out_ptr + out_offset[:, None] + dim[None, :] * out_strides[3], valid, other=0.0)
        block_out = tl.load(
            block_out_ptr + block_out_offset[:, None] + dim[None, :] * block_out_strides[3], valid, other=0.0
        ).to(tl.float32)
        # enable_fp_fusion=False prevents contraction of this multiply and
        # subtraction. The accumulated output stays FP32 between Ring hops.
        merged_out = old_out - weight[:, None] * (old_out - block_out)

        merged_out_offset = batch * merged_out_strides[0] + token * merged_out_strides[1] + head * merged_out_strides[2]
        tl.store(merged_out_ptr + merged_out_offset[:, None] + dim[None, :] * merged_out_strides[3], merged_out, valid)
        merged_lse_offset = batch * merged_lse_strides[0] + token * merged_lse_strides[1] + head * merged_lse_strides[2]
        tl.store(merged_lse_ptr + merged_lse_offset, merged_lse, valid_row)


def try_fused_ring_merge(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Merge normalized ``(B, S, H, 1)`` LSE tensors, or request fallback.

    Outputs are newly allocated FP32 tensors; inputs are never mutated.
    Strides are respected, including padded/transposed FlashAttention LSEs.
    Autograd and compiler tracing retain the PyTorch implementation. The
    latter can already fuse the expression outside HSDP's eager boundary.
    A different current CUDA device retains PyTorch's device handling.
    This preserves the arithmetic order, not bitwise transcendental results.
    """
    if (
        not HAS_TRITON
        or torch.version.hip is not None
        or not current_platform.is_cuda()
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
    ):
        return None

    tensors = (out, lse, block_out, block_lse)
    if not all(t.is_cuda and t.device == out.device and t.layout == torch.strided for t in tensors):
        return None
    if out.device.index != torch.accelerator.current_device_index():
        return None
    if (
        out.dtype != torch.float32
        or lse.dtype != torch.float32
        or block_lse.dtype != torch.float32
        or block_out.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or out.ndim != 4
        or block_out.shape != out.shape
        or out.numel() == 0
        or not 0 < out.shape[-1] <= 256
        or lse.shape != (*out.shape[:3], 1)
        or block_lse.shape != lse.shape
        or any(stride < 0 for t in tensors for stride in t.stride())
    ):
        return None

    merged_out = torch.empty_like(out)
    merged_lse = torch.empty_like(lse)
    batch, sequence, heads, head_dim = out.shape
    rows = batch * sequence * heads
    block_rows = 4
    _ring_merge_kernel[(triton.cdiv(rows, block_rows),)](
        out,
        lse,
        block_out,
        block_lse,
        merged_out,
        merged_lse,
        rows,
        sequence,
        heads,
        head_dim,
        out.stride(),
        lse.stride(),
        block_out.stride(),
        block_lse.stride(),
        merged_out.stride(),
        merged_lse.stride(),
        block_rows=block_rows,
        block_dim=triton.next_power_of_2(head_dim),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return merged_out, merged_lse
