# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared TTS sampling primitives: nucleus (top-p/top-k) and RAS.

Used by CosyVoice3 and GLM-TTS (and any future TTS model with RAS).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def multinomial_sample(
    probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one sample from a categorical distribution."""
    return torch.multinomial(probs, 1, replacement=True, generator=generator).reshape(())


def nucleus_sample_one(
    weighted_scores: torch.Tensor,
    *,
    top_p: float,
    top_k: int,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token using nucleus (top-p + top-k) filtering.

    ``weighted_scores`` should be log-softmax-ed logits (for RAS callers)
    or raw logits (softmax is applied internally).

    Vectorized via ``torch.cumsum`` — no Python loop over the vocabulary.
    """
    # Apply top-k before sort.  GLM-TTS and CosyVoice3 usually run with
    # top_k=25, so sorting the whole audio vocabulary on every AR step is
    # avoidable work in the hottest path.  Keep probabilities normalized over
    # the full vocabulary so the top-p cumulative mass matches the old path.
    if top_k > 0 and int(top_k) < int(weighted_scores.numel()):
        top_scores, sorted_idx = torch.topk(
            weighted_scores,
            k=int(top_k),
            dim=0,
            largest=True,
            sorted=True,
        )
        sorted_prob = (top_scores - torch.logsumexp(top_scores, dim=0)).exp()
    else:
        probs = weighted_scores.softmax(dim=0)
        sorted_prob, sorted_idx = probs.sort(descending=True, stable=True)

    # Apply top-p (nucleus) filtering: keep the smallest prefix whose
    # cumulative probability exceeds top_p, always including the first token.
    # Use float32 for cumulative sum to match the original Python float64
    # accumulation loop.  Low-precision (fp16/bf16) cumsum over ~32K vocab
    # causes rounding drift that shifts the top-p boundary, changing which
    # tokens are kept and ultimately producing different AR sequences.
    sorted_prob_f32 = sorted_prob.float()
    cum_prob = sorted_prob_f32.cumsum(dim=0)
    # Mask: include tokens where cumsum *before* this token is still < top_p
    mask = (cum_prob - sorted_prob_f32) < top_p
    # Safety: ensure top token is always kept.  With sorted descending
    # probabilities, mask[0] is always True when top_p > 0 (since the
    # cumsum-before-self of the first element is 0 < top_p).  Setting it
    # unconditionally avoids a GPU→CPU sync from the old `mask.any()` check.
    mask[0] = True

    kept_probs = torch.nan_to_num(sorted_prob[mask], nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
    kept_indices = sorted_idx[mask]
    # ``torch.multinomial`` accepts unnormalized weights, but not an all-zero
    # vector.  Handle the extreme underflow case on-device by forcing the top
    # retained token to be selectable without adding another CPU sync.
    kept_probs[0] = torch.where(
        kept_probs.sum() > 0,
        kept_probs[0],
        torch.ones_like(kept_probs[0]),
    )

    sample_pos = multinomial_sample(kept_probs, generator=generator)
    return int(kept_indices[sample_pos].item())


def ras_sample_one(
    weighted_scores: torch.Tensor,
    decoded_tokens: Sequence[int],
    *,
    top_p: float,
    top_k: int,
    win_size: int,
    tau_r: float,
    generator: torch.Generator | None = None,
) -> int:
    """Repetition-Aware Sampling following GLM-TTS/CosyVoice.

    If the nucleus-sampled token appears too often in the recent window,
    upstream samples once from the original full distribution instead of
    masking the repeated token.  This matters for GLM-TTS because masking the
    dominant repeated token can artificially raise EOA probability right after
    the minimum length guard is lifted.
    """

    def _random_sample_one() -> int:
        return int(multinomial_sample(weighted_scores.softmax(dim=0), generator=generator).item())

    top_id = nucleus_sample_one(
        weighted_scores,
        top_p=top_p,
        top_k=top_k,
        generator=generator,
    )
    if win_size > 0 and decoded_tokens:
        recent = decoded_tokens[-win_size:]
        rep_num = sum(1 for token in recent if int(token) == top_id)
        if rep_num >= win_size * tau_r:
            top_id = _random_sample_one()
    return top_id
