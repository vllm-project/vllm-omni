# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# ruff: noqa: N803

"""CUDA SigLIP softmax with BF16 score rounding and FP32 accumulation.

Keep each lane's additions in order before the warp reduction. Reassociating
this sum can change BF16 probabilities; those differences grow across the
vision tower. Scaling and softmax share a kernel to avoid writing scaled
scores and FP32 probabilities to global memory.
"""

import torch
from vllm.triton_utils import tl, triton

libdevice = tl.extra.cuda.libdevice


# Video resolutions and scheduler batches vary independently. Specializing on
# either dimension compiles a new, heavily unrolled kernel during inference.
# Only the reduction bucket needs to be static; keep strides and bounds dynamic.
@triton.jit(do_not_specialize=["N", "ROWS"])
def _scaled_softmax_kernel(X, Y, N, ROWS, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    lane = tl.arange(0, 32)
    xs = ()
    local_max = tl.full((4, 32), -float("inf"), tl.float32)
    for it in tl.static_range(BLOCK // 32):
        col = it * 32 + lane
        mask = rows[:, None] < ROWS
        if it == BLOCK // 32 - 1:
            mask = mask & (col[None, :] < N)
        x = tl.load(X + rows[:, None] * N + col[None, :], mask, other=0)
        x = (x.to(tl.float32) * SCALE).to(x.dtype).to(tl.float32)
        if it == BLOCK // 32 - 1:
            x = tl.where(col[None, :] < N, x, -float("inf"))
        xs += (x,)
        local_max = tl.maximum(local_max, x)
    row_max = tl.max(local_max, 1)
    es = ()
    local_sum = tl.full((4, 32), 0, tl.float32)
    for it in tl.static_range(BLOCK // 32):
        e = libdevice.exp(xs[it] - row_max[:, None])
        es += (e,)
        local_sum = local_sum + e
    row_sum = tl.sum(local_sum, 1)
    for it in tl.static_range(BLOCK // 32):
        col = it * 32 + lane
        p = libdevice.div_rn(es[it], row_sum[:, None])
        mask = rows[:, None] < ROWS
        if it == BLOCK // 32 - 1:
            mask = mask & (col[None, :] < N)
        tl.store(Y + rows[:, None] * N + col[None, :], p, mask)


def scaled_softmax(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Normalize contiguous, unmasked CUDA BF16 rows of at most 2048 scores."""
    assert x.is_contiguous() and x.dtype == torch.bfloat16 and x.is_cuda
    assert 0 < x.shape[-1] <= 2048
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    n = x.shape[-1]
    rows = x.numel() // n
    _scaled_softmax_kernel[(triton.cdiv(rows, 4),)](
        x, out, n, rows, scale, triton.cdiv(n, 32) * 32, enable_fp_fusion=False
    )
    return out
