# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-owned sampling for YuE2-3B, ported from upstream ``sampling.py``.

The model claims ``prefer_model_sampler`` and reproduces the reference
request-local arithmetic exactly: phase masking (abc = text vocabulary + its
end; semantic = codec span + its end), windowed repetition penalty over the
request's own history, then temperature/top-k/top-p with a seeded multinomial.
Default CFG is off (guidance 1.0 for full/melody, 1.01 for off); the upstream
torch backend treats those the same, so Phase 1 samples a single row.
"""

from __future__ import annotations

import torch

from .constants import (
    ABC_END,
    CODEC_OFFSET,
    CODEC_SIZE,
    EOD,
    MUSIC_END,
)


def window_penalty(logits: torch.Tensor, recent_ids: list[int], penalty: float) -> torch.Tensor:
    """Upweight/downweight ids seen in the window, upstream arithmetic."""
    if penalty == 1.0 or not recent_ids:
        return logits
    recent = torch.as_tensor(recent_ids, dtype=torch.long, device=logits.device)
    if logits.dim() > 1:
        recent = recent.reshape(1, -1)
    freq = torch.zeros_like(logits)
    freq.scatter_add_(-1, recent, torch.ones_like(recent, dtype=logits.dtype))
    alpha = penalty**freq
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def distribution(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    penalty_window: int,
    history: list[int],
    step: int,
    min_tokens: int,
    phase: str,
) -> torch.Tensor:
    """Masked/penalized/shaped scores for one row (float32 path).

    Accepts either one row ``[vocab]`` (what the model's sampler sees) or a
    batch ``[rows, vocab]`` and returns the same shape it was given.
    """
    single = logits.dim() == 1
    logits = logits.unsqueeze(0) if single else logits
    scores = logits.float().clone()
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, float("-inf"))
    if phase == "abc":
        allowed[..., :EOD] = 0
    else:
        allowed[..., CODEC_OFFSET : CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., end] = 0
    scores = scores + allowed
    if step < min_tokens:
        scores[..., end] = -torch.inf
    scores = window_penalty(scores, history[-penalty_window:], repetition_penalty)
    if temperature == 0:
        return scores.squeeze(0) if single else scores
    if temperature != 1:
        scores = scores / temperature
    threshold = scores.topk(min(top_k, scores.shape[-1])).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    if top_p < 1:
        values, indices = scores.sort(descending=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > top_p
        removed[..., :1] = False
        values = values.masked_fill(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    return scores.squeeze(0) if single else scores


def sample_row(scores: torch.Tensor, generator: torch.Generator, *, greedy: bool = False) -> int:
    """Draw one token id from prepared scores with the request's generator."""
    if not torch.isfinite(scores).any():
        raise RuntimeError("sampling scores are all -inf; check phase masking")
    if greedy:
        return int(scores.argmax().item())
    probabilities = scores.softmax(-1)
    next_id = torch.multinomial(probabilities, 1, generator=generator)
    return int(next_id.item())


__all__ = ["distribution", "sample_row", "window_penalty"]
