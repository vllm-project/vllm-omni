# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Top-k Gumbel sampling with the outer MTP graph's supplied uniforms."""

import torch
from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _sample(
    logits,
    uniforms,
    output,
    ls: tl.constexpr,
    us: tl.constexpr,
    lc: tl.constexpr,
    uc: tl.constexpr,
    vocab: tl.constexpr,
    top_k: tl.constexpr,
    topk_block: tl.constexpr,
    inv_temperature: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.arange(0, block)
    value = tl.load(logits + row * ls + idx * lc, idx < vocab, other=-float("inf"))
    scaled = (value.to(tl.float32) * inv_temperature).to(value.dtype).to(tl.float32)
    if top_k > 0:
        if top_k == 1:
            threshold = tl.max(scaled, 0)
        else:
            ordered = tl.topk(scaled, topk_block)
            threshold = tl.sum(tl.where(tl.arange(0, topk_block) == top_k - 1, ordered, 0))
        scaled = tl.where(scaled < threshold, -float("inf"), scaled)
    u = tl.load(uniforms + row * us + idx * uc, idx < vocab, other=0.5)
    score = scaled - tldevice.log(-tldevice.log(u))
    score = tl.where(idx < vocab, score, -float("inf"))
    max_score = tl.max(score, 0)
    chosen = tl.min(tl.where(score == max_score, idx, block), 0)
    first_nan = tl.min(tl.where(score != score, idx, block), 0)
    tl.store(output + row, tl.where(first_nan < block, first_nan, chosen))


def sample_code_topk_gumbel(
    logits: torch.Tensor, uniforms: torch.Tensor, top_k: int, inv_temperature: float
) -> torch.Tensor:
    """Match BF16 scale rounding, top-k ties, and first-index Gumbel argmax.

    Random values are supplied by the outer graph, so RNG state and seeded
    per-request streams are unchanged. CUDA libdevice logs match ATen's logs.
    """
    if logits.shape != uniforms.shape:
        raise ValueError("sampling uniforms must match logits shape")
    if logits.device != uniforms.device or not logits.is_cuda:
        raise ValueError("logits and uniforms must be on the same CUDA device")
    if top_k > logits.shape[-1]:
        raise ValueError("top_k must not exceed the vocabulary size")
    b, v = logits.shape
    out = torch.empty((b, 1), device=logits.device, dtype=torch.int64)
    _sample[(b,)](
        logits,
        uniforms,
        out,
        logits.stride(0),
        uniforms.stride(0),
        logits.stride(1),
        uniforms.stride(1),
        v,
        top_k,
        triton.next_power_of_2(max(1, top_k)),
        inv_temperature,
        triton.next_power_of_2(v),
        num_warps=4,
    )
    return out
