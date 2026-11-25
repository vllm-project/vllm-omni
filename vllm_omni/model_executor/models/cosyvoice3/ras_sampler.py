# ruff: noqa: N803
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused CosyVoice3 repetition-aware sampling for Model Runner V2.

One program per row replaces the ~70 small kernels (softmax, full-vocabulary
sort, cumulative sum, masks, two Gumbel draws, history gather, rejection) of
the tensor implementation. The nucleus is the top-p prefix of the full
distribution capped at top-k, found by repeated argmax instead of a sort; a
draw that repeats within the last ``win_size`` generated tokens is replaced by
a draw from the full remaining distribution, from a disjoint noise stream.
"""

import torch
from vllm.triton_utils import tl, triton

MAX_FUSED_TOP_K = 64


@triton.jit
def _gumbel(seed, offset):
    u = tl.maximum(tl.rand(seed, offset), 4.6566127342e-10)
    return -tl.log(-tl.log(u))


@triton.jit
def _ras_sample_kernel(
    out_ptr,
    logits_ptr,
    logits_stride,
    vocab,
    idx_mapping_ptr,
    temperature_ptr,
    top_k_ptr,
    top_p_ptr,
    seeds_ptr,
    pos_ptr,
    tokens_ptr,
    tokens_stride,
    total_len_ptr,
    prompt_len_ptr,
    default_top_k,
    default_top_p,
    eps,
    threshold,
    win_size,
    HAS_TOP_K: tl.constexpr,
    HAS_TOP_P: tl.constexpr,
    WIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(idx_mapping_ptr + row).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < vocab
    x = tl.load(logits_ptr + row * logits_stride + cols, mask=mask, other=float("-inf")).to(tl.float32)
    temperature = tl.load(temperature_ptr + req).to(tl.float32)
    if temperature == 0.0:
        tl.store(out_ptr + row, tl.argmax(x, axis=0).to(tl.int64))
        return
    x = x / tl.maximum(temperature, eps)
    peak = tl.max(x, axis=0)
    weights = tl.where(mask, tl.exp(x - peak), 0.0)
    total = tl.sum(weights, axis=0)
    log_probs = x - peak - tl.log(total)
    top_k = tl.load(top_k_ptr + req) if HAS_TOP_K else default_top_k
    top_p = tl.load(top_p_ptr + req).to(tl.float32) if HAS_TOP_P else default_top_p
    seed = tl.load(seeds_ptr + req)
    pos = tl.load(pos_ptr + row)
    nucleus_seed = tl.randint(seed, pos)

    # Walk the distribution in descending order (ties: lowest id first, like a
    # stable descending sort) while the top-p prefix and the top-k cap allow.
    remaining = weights
    before = 0.0
    best = float("-inf")
    sampled = tl.argmax(x, axis=0).to(tl.int64)
    for _ in range(top_k):
        if before < top_p:
            value = tl.max(remaining, axis=0)
            token = tl.argmax(remaining, axis=0)
            prob = value / total
            score = tl.log(prob) + _gumbel(nucleus_seed, token)
            if score > best:
                best = score
                sampled = token.to(tl.int64)
            before += prob
            remaining = tl.where(cols == token, -1.0, remaining)

    # Repetition-aware rejection over the last WIN generated tokens.
    total_len = tl.load(total_len_ptr + req).to(tl.int64)
    prompt_len = tl.load(prompt_len_ptr + req).to(tl.int64)
    offsets = total_len - 1 - tl.arange(0, WIN)
    in_output = (offsets >= prompt_len) & (tl.arange(0, WIN) < win_size)
    history = tl.load(tokens_ptr + req * tokens_stride + tl.maximum(offsets, 0), mask=in_output, other=-1)
    repeats = tl.sum((in_output & (history == sampled)).to(tl.float32), axis=0)
    if (repeats >= threshold) & (tl.sum(in_output.to(tl.int32), axis=0) > 0):
        # Offset keeps the replacement noise disjoint from the nucleus draw's.
        replacement_seed = tl.randint(seed, pos + 1073741824)
        scores = tl.where(mask & (cols != sampled), log_probs + _gumbel(replacement_seed, cols), float("-inf"))
        if tl.max(scores, axis=0) > float("-inf"):
            sampled = tl.argmax(scores, axis=0).to(tl.int64)
    tl.store(out_ptr + row, sampled)


def fused_ras_sample(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    seeds: torch.Tensor,
    pos: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
    prompt_len: torch.Tensor,
    *,
    default_top_k: int,
    default_top_p: float,
    win_size: int,
    tau_r: float,
    eps: float,
) -> torch.Tensor:
    """Sample one token per logits row; per-request tensors are indexed by ``idx_mapping``."""
    rows, vocab = logits.shape
    out = torch.empty(rows, dtype=torch.int64, device=logits.device)
    if rows == 0:
        return out
    _ras_sample_kernel[(rows,)](
        out,
        logits,
        logits.stride(0),
        vocab,
        idx_mapping,
        temperature,
        temperature if top_k is None else top_k,
        temperature if top_p is None else top_p,
        seeds,
        pos,
        all_token_ids,
        all_token_ids.stride(0),
        total_len,
        prompt_len,
        int(default_top_k),
        float(default_top_p),
        float(eps),
        float(win_size * tau_r),
        int(win_size),
        HAS_TOP_K=top_k is not None,
        HAS_TOP_P=top_p is not None,
        WIN=triton.next_power_of_2(max(int(win_size), 1)),
        BLOCK=triton.next_power_of_2(vocab),
        num_warps=8,
    )
    return out
