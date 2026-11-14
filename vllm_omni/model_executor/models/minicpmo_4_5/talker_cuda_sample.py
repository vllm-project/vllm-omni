# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Experimental CUDA single-row sampler retaining invalid-probability checks."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _validate_argmax(probabilities, ratios, output, valid, n: tl.constexpr, block: tl.constexpr):
    offsets = tl.arange(0, block)
    p = tl.load(probabilities + offsets, offsets < n, other=0.0)
    r = tl.load(ratios + offsets, offsets < n, other=-float("inf"))
    finite_nonnegative = (p >= 0.0) & (p < float("inf"))
    legal = (tl.sum(finite_nonnegative.to(tl.int32), 0) == block) & (tl.sum(p, 0) > 0.0)
    token = tl.argmax(r, 0, tie_break_left=True)
    tl.store(output, token.to(tl.int64))
    tl.store(valid, legal)


def sample(probabilities: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    if (
        not probabilities.is_cuda
        or probabilities.dtype != torch.float32
        or probabilities.ndim != 2
        or probabilities.shape[0] != 1
        or not probabilities.is_contiguous()
        or not 0 < probabilities.shape[1] <= 8192
    ):
        return torch.multinomial(probabilities, 1, generator=generator).reshape(1)
    noise = torch.empty_like(probabilities).exponential_(1, generator=generator)
    ratios = probabilities / noise
    output = torch.empty(1, dtype=torch.long, device=probabilities.device)
    valid = torch.empty(1, dtype=torch.bool, device=probabilities.device)
    width = probabilities.shape[1]
    _validate_argmax[(1,)](probabilities, ratios, output, valid, width, triton.next_power_of_2(width))
    torch._assert_async(valid, "probability tensor contains inf, nan or a negative value, or has zero mass")
    return output
