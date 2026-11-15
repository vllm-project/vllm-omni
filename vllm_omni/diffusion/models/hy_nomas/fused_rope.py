# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Inference-only adjacent-pair RoPE with the eager FP32 rounding order."""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _rope_kernel(
        x,
        cos,
        sin,
        out,
        n: tl.constexpr,
        tokens: tl.constexpr,
        heads: tl.constexpr,
        dim: tl.constexpr,
        s0: tl.constexpr,
        s1: tl.constexpr,
        s2: tl.constexpr,
        s3: tl.constexpr,
        block: tl.constexpr,
    ):
        index = tl.program_id(0) * block + tl.arange(0, block)
        d = index % dim
        h = index // dim % heads
        t = index // (dim * heads) % tokens
        b = index // (dim * heads * tokens)
        base = b * s0 + t * s1 + h * s2
        value = tl.load(x + base + d * s3, index < n, other=0).to(tl.float32)
        pair = tl.load(x + base + (d ^ 1) * s3, index < n, other=0).to(tl.float32)
        rotated = tl.where(d % 2 == 0, -pair, pair)
        c = tl.load(cos + t * dim + d, index < n, other=0)
        s = tl.load(sin + t * dim + d, index < n, other=0)
        # Launch disables FMA so each multiply rounds before the addition.
        result = value * c + rotated * s
        tl.store(out + index, result, index < n)


def try_apply_rotary(x, tables):
    """Return None for layouts/devices requiring the original autograd path."""
    if (
        not HAS_TRITON
        or x.device.type != "cuda"
        or torch.is_grad_enabled()
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.ndim != 4
        or x.numel() == 0
        or x.shape[-1] % 2
        or len(tables) != 2
    ):
        return None
    if any(
        part.device != x.device
        or part.dtype != torch.float32
        or part.shape != (x.shape[1], x.shape[-1])
        or not part.is_contiguous()
        for part in tables
    ):
        return None
    result = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _rope_kernel[(triton.cdiv(x.numel(), 512),)](
        x,
        *tables,
        result,
        x.numel(),
        *x.shape[1:],
        *x.stride(),
        512,
        enable_fp_fusion=False,
    )
    return result
