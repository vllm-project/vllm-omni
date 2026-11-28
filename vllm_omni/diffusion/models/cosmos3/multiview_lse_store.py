# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Indexed in-place LSE recurrence with the original FP32 rounding points.

The host planner guarantees unique query rows within a pass. Keep the pass
order unchanged; only the intermediate gathered/merged row buffers disappear.
"""

import torch
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

if HAS_TRITON:

    @triton.jit
    def _direct_store_kernel(
        ACC,
        ACC_LSE,
        OUT,
        LSE,
        INDEX,
        RESULT,
        SLOTS,
        HEADS: tl.constexpr,
        DIM: tl.constexpr,
        START: tl.constexpr,
        SEED: tl.constexpr,
        FINAL: tl.constexpr,
        BLOCK: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        slot = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = slot < SLOTS
        row, head = slot // HEADS, slot % HEADS
        if START >= 0:
            destination = row + START
        else:
            destination = tl.load(INDEX + row, valid, other=0)
        target = destination * HEADS + head
        dim = tl.arange(0, BLOCK_DIM)
        mask = valid[:, None] & (dim < DIM)[None, :]
        value = tl.load(OUT + slot[:, None] * DIM + dim[None, :], mask, other=0).to(tl.float32)
        lse = tl.load(LSE + slot, valid, other=0).to(tl.float32)
        if not SEED:
            prior_lse = tl.load(ACC_LSE + target, valid, other=0)
            prior = tl.load(ACC + target[:, None] * DIM + dim[None, :], mask, other=0).to(tl.float32)
            weight = tl.sigmoid(lse - prior_lse)
            value = prior - weight[:, None] * (prior - value)
            shift = prior_lse - lse
            log_sigmoid = tl.where(0.0 < shift, 0.0, shift) - tldevice.log1p(tldevice.exp(-tl.abs(shift)))
            lse = prior_lse - log_sigmoid
        # Preserve the accumulator rounding point before writing the final dtype.
        rounded = value.to(ACC.dtype.element_ty)
        tl.store(ACC + target[:, None] * DIM + dim[None, :], rounded, mask)
        tl.store(ACC_LSE + target, lse, valid)
        if FINAL:
            tl.store(RESULT + target[:, None] * DIM + dim[None, :], rounded, mask)


def direct_store_supported(
    acc: torch.Tensor,
    acc_lse: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    index: torch.Tensor,
    result: torch.Tensor | None = None,
) -> bool:
    tensors = (acc, acc_lse, out, lse, index) + (() if result is None else (result,))
    return (
        HAS_TRITON
        and acc.device.type == "cuda"
        and all(t.device == acc.device and t.is_contiguous() for t in tensors)
        and acc.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and out.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and acc_lse.dtype == lse.dtype == torch.float32
        and index.dtype == torch.int64
        and index.ndim == 1
        and acc.ndim == out.ndim == 3
        and acc.shape[1:] == out.shape[1:]
        and acc_lse.shape == acc.shape[:2]
        and lse.shape == out.shape[:2]
        and index.numel() == out.shape[0]
        and 0 < out.shape[-1] <= 256
        and (result is None or (result.shape == acc.shape and result.dtype == out.dtype))
    )


def direct_store(
    acc: torch.Tensor,
    acc_lse: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    index: torch.Tensor,
    *,
    start: int,
    seed: bool,
    result: torch.Tensor | None = None,
) -> bool:
    """Update unique plan rows; full final coverage also writes the result.

    A partial final pass retains the full copy after the update, preserving all
    rows last touched by earlier passes. Unsupported layouts use the caller's
    existing compiled path. No request, layer or tensor cache is retained.
    """
    if not direct_store_supported(acc, acc_lse, out, lse, index, result):
        return False
    rows, heads, dim = out.shape
    full_final = result is not None and rows == acc.shape[0]
    if rows:
        _direct_store_kernel[(triton.cdiv(rows * heads, 16),)](
            acc,
            acc_lse,
            out,
            lse,
            index,
            result if full_final else acc,
            rows * heads,
            heads,
            dim,
            start,
            seed,
            full_final,
            16,
            triton.next_power_of_2(dim),
            num_warps=4,
            enable_fp_fusion=True,
        )
    if result is not None and not full_final:
        result.copy_(acc)
    return True
