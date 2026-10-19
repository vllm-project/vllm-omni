# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend SnakeBeta with eager BF16 rounding at each arithmetic boundary."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _snake(
    x_ptr,
    alpha_ptr,
    inverse_ptr,
    out_ptr,
    numel: tl.constexpr,
    channels: tl.constexpr,
    length: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    channel = (offsets // length) % channels
    x = tl.load(x_ptr + offsets, offsets < numel, other=0).to(tl.float32)
    alpha = tl.load(alpha_ptr + channel).to(tl.float32)
    inverse = tl.load(inverse_ptr + channel).to(tl.float32)
    phase = (x * alpha).to(tl.bfloat16).to(tl.float32)
    sine = tl.sin(phase).to(tl.bfloat16).to(tl.float32)
    square = (sine * sine).to(tl.bfloat16).to(tl.float32)
    scaled = (inverse * square).to(tl.bfloat16).to(tl.float32)
    result = (x + scaled).to(tl.bfloat16)
    tl.store(out_ptr + offsets, result, offsets < numel)


def snake_beta(hidden: torch.Tensor, alpha: torch.Tensor, inverse_beta: torch.Tensor) -> torch.Tensor:
    """Called only for nonempty contiguous NCT BF16 inference tensors."""
    output = torch.empty_like(hidden)
    block = 1024
    _snake[(triton.cdiv(hidden.numel(), block),)](
        hidden,
        alpha,
        inverse_beta,
        output,
        hidden.numel(),
        hidden.shape[1],
        hidden.shape[2],
        block,
        enable_fp_fusion=False,
    )
    return output
