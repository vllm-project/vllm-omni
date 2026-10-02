# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused Gumbel selection after native nucleus probability reductions.

Keep ATen's top-k threshold, sort, softmax and cumulative sum, including
boundary ties. Random numbers remain indexed by original token id. Only the
final mask, Gumbel transform and argmax are fused; no vocabulary scatter is
needed. This avoids changing which tokens survive a nucleus boundary through
different reduction orders.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _sorted_nucleus_gumbel(
    logits,
    indices,
    probabilities,
    cumulative,
    uniforms,
    output,
    vocab: tl.constexpr,
    logits_stride: tl.constexpr,
    token_stride: tl.constexpr,
    uniform_stride: tl.constexpr,
    top_p: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    valid = offsets < vocab
    values = tl.load(logits + row * logits_stride + offsets * token_stride, valid, other=-float("inf")).to(tl.float32)
    if top_p < 1.0:
        ids = tl.load(indices + row * vocab + offsets, valid, other=0)
        probs = tl.load(probabilities + row * vocab + offsets, valid, other=0)
        cdf = tl.load(cumulative + row * vocab + offsets, valid, other=0)
        values = tl.where((cdf - probs) >= top_p, -float("inf"), values)
    else:
        ids = offsets
    noise = tl.load(uniforms + row * uniform_stride + ids, valid, other=0.5)
    score = values - tl.extra.cuda.libdevice.log(-tl.extra.cuda.libdevice.log(noise))
    # torch.argmax returns the first original token for a tie or NaN.
    first_nan = tl.min(tl.where(valid & (score != score), ids, 2147483647), 0)
    maximum = tl.max(tl.where(valid, score, -float("inf")), 0)
    winner = tl.min(tl.where(valid & (score == maximum), ids, 2147483647), 0)
    tl.store(output + row, tl.where(first_nan < 2147483647, first_nan, winner))


def sample_top_k_top_p_gumbel(
    logits: torch.Tensor, uniforms: torch.Tensor, *, top_k: int, top_p: float
) -> torch.Tensor:
    """Sample CUDA rows without scattering sorted logits back to the vocabulary.

    Uniforms are caller-owned FP32 values with the same shape as logits; their
    generation and generator advancement are deliberately outside this kernel.
    Softmax/scan use the reference ATen operations. The Gumbel logarithms
    still use libdevice, so this is not a universal bitwise-equivalence claim.
    """
    if logits.ndim != 2 or not logits.is_cuda or uniforms.device != logits.device or torch.version.hip is not None:
        raise ValueError("fused sampling requires two-dimensional CUDA logits and uniforms")
    if uniforms.shape != logits.shape or uniforms.dtype != torch.float32 or uniforms.stride(1) != 1:
        raise ValueError("uniforms must be FP32 with the logits shape and contiguous rows")
    rows, vocab = logits.shape
    # The stored-mode caller disables top-k for both 0 and negative values
    # (vLLM commonly uses -1). Keep that behavior when fusion is enabled.
    top_k = max(top_k, 0)
    if vocab == 0 or top_k > vocab or not 0 < top_p <= 1:
        raise ValueError("invalid top-k/top-p sampling parameters")
    if top_k:
        threshold = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, -float("inf"))
    indices = probabilities = cumulative = None
    if top_p < 1:
        logits, indices = logits.sort(dim=-1, descending=True)
        probabilities = logits.softmax(dim=-1, dtype=torch.float32)
        cumulative = probabilities.cumsum(-1)
    result = torch.empty((rows, 1), dtype=torch.long, device=logits.device)
    if rows:
        _sorted_nucleus_gumbel[(rows,)](
            logits,
            indices,
            probabilities,
            cumulative,
            uniforms,
            result,
            vocab,
            logits.stride(0),
            logits.stride(1),
            uniforms.stride(0),
            top_p,
            triton.next_power_of_2(vocab),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return result
