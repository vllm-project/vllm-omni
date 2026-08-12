# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS sampling primitives, ported 1:1 from the reference checkpoint.

Filter (top-k/top-p on **raw** logits) -> divide surviving logits by
temperature -> Gumbel-max draw. This is the reverse of the usual filter order
and is audibly different at ``temperature != 1``, which is why this cannot
reuse ``models/common/nucleus_ras_sampling.py``. Per-row scalars are accepted
so mixed batches run in one shot without a GPU->CPU sync.
"""

from __future__ import annotations

import torch

#: Below this temperature a request is treated as greedy, matching vLLM.
SAMPLING_EPS = 1e-5


def _as_column(
    value: float | int | torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast a scalar or ``[B]`` tensor to a ``[B, 1]`` tensor."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype).reshape(-1, 1).expand(batch_size, 1)
    return torch.full((batch_size, 1), float(value), device=device, dtype=dtype)


def filter_top_k_top_p(
    logits: torch.Tensor,
    top_k: int | torch.Tensor,
    top_p: float | torch.Tensor,
) -> torch.Tensor:
    """Mask candidates outside the top-k / top-p set of the raw logits.

    ``top_k <= 0`` and ``top_p >= 1`` disable each filter. The highest-scoring
    candidate is always kept, so no row can become all ``-inf``.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D [B, V], got {tuple(logits.shape)}")
    batch_size, vocab = logits.shape
    scalar_k = not isinstance(top_k, torch.Tensor)
    scalar_p = not isinstance(top_p, torch.Tensor)
    if scalar_k and scalar_p and (int(top_k) <= 0 or int(top_k) >= vocab) and float(top_p) >= 1.0:
        return logits

    sorted_scores, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1)
    positions = torch.arange(vocab, device=logits.device).unsqueeze(0)

    keep_k = _as_column(top_k, batch_size, logits.device, torch.long)
    # top_k <= 0 means "no cut"; represent it as the full vocabulary.
    keep_k = torch.where(keep_k <= 0, torch.full_like(keep_k, vocab), keep_k)
    threshold_p = _as_column(top_p, batch_size, logits.device, cumulative.dtype)

    remove_sorted = (cumulative > threshold_p) | (positions >= keep_k)
    # Always keep the argmax so at least one candidate survives.
    remove_sorted[:, 0] = False
    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_indices, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def gumbel_argmax_sample(
    scores: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw one index per row via Gumbel-max over ``softmax(scores)``."""
    probabilities = torch.softmax(scores, dim=-1)
    uniform = torch.rand(
        probabilities.shape,
        dtype=probabilities.dtype,
        device=probabilities.device,
        generator=generator,
    )
    # -log(u) is Exp(1); argmax(p / Exp(1)) samples from p.
    noise = -torch.log(uniform.clamp_min(torch.finfo(probabilities.dtype).tiny))
    return torch.argmax(probabilities / noise, dim=-1)


def sample_scores(
    logits: torch.Tensor,
    *,
    temperature: float | torch.Tensor,
    top_k: int | torch.Tensor,
    top_p: float | torch.Tensor,
    do_sample: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Filter -> temper -> draw, in the reference order.

    Rows whose temperature is below :data:`SAMPLING_EPS` fall back to argmax.
    """
    filtered = filter_top_k_top_p(logits, top_k, top_p)
    greedy = filtered.argmax(dim=-1)
    if not do_sample:
        return greedy
    if not isinstance(temperature, torch.Tensor):
        if float(temperature) < SAMPLING_EPS:
            return greedy
        return gumbel_argmax_sample(filtered / float(temperature), generator=generator)

    temperature_col = _as_column(temperature, filtered.shape[0], filtered.device, filtered.dtype)
    safe_temperature = torch.where(
        temperature_col < SAMPLING_EPS,
        torch.ones_like(temperature_col),
        temperature_col,
    )
    sampled = gumbel_argmax_sample(filtered / safe_temperature, generator=generator)
    return torch.where(temperature_col.reshape(-1) < SAMPLING_EPS, greedy, sampled)


def ras_sample_semantic(
    logits: torch.Tensor,
    recent_ids: torch.Tensor | None,
    *,
    temperature: float | torch.Tensor,
    top_k: int | torch.Tensor,
    top_p: float | torch.Tensor,
    ras_temperature: float,
    ras_top_p: float,
    do_sample: bool = True,
    num_semantic_ids: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Repetition-Aware Sampling for the semantic (Slow AR) token.

    Draws with the request's ``temperature`` / ``top_p``; if the drawn id is
    already in ``recent_ids`` (pad with a value that cannot be sampled, e.g.
    ``-1``), redraws from the flatter RAS distribution instead. The
    ``num_semantic_ids`` slot and beyond (EOS) are never substituted, so a
    repeat EOS still ends the utterance.

    DualAR RAS *resamples* repeats rather than masking them — masking the
    dominant token inflates EOS probability and truncates the utterance.
    """
    normal = sample_scores(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        generator=generator,
    )
    if not do_sample or recent_ids is None or recent_ids.numel() == 0:
        return normal

    high = sample_scores(
        logits,
        temperature=ras_temperature,
        top_k=top_k,
        top_p=ras_top_p,
        do_sample=True,
        generator=generator,
    )
    repeated = (recent_ids == normal[:, None]).any(dim=1)
    if num_semantic_ids is not None:
        repeated &= normal < int(num_semantic_ids)
    return torch.where(repeated, high, normal)


def _row_value(value: object, index: int) -> object:
    """Slice a per-row value for request ``index``; scalars pass through."""
    return value[index : index + 1] if isinstance(value, torch.Tensor) else value


def ras_sample_batch(
    logits: torch.Tensor,
    recent_ids: torch.Tensor | None,
    *,
    temperature: float | torch.Tensor,
    top_k: int | torch.Tensor,
    top_p: float | torch.Tensor,
    ras_temperature: float,
    ras_top_p: float,
    num_semantic_ids: int | None,
    generators: dict,
    num_reqs: int,
) -> torch.Tensor:
    """Batched RAS that honors each request's own RNG generator.

    vLLM keys ``generators`` by batch slot, so a single shared generator would
    leak slot 0's RNG stream to every row and make a prompt's audio depend on
    its batch neighbours (undercutting per-request ``seed``). The single-request
    case keeps the batched fast path; larger batches loop so each row draws from
    its own (possibly unseeded) generator.
    """
    common = dict(
        ras_temperature=ras_temperature,
        ras_top_p=ras_top_p,
        num_semantic_ids=num_semantic_ids,
    )
    if num_reqs == 1:
        return ras_sample_semantic(
            logits,
            recent_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generators.get(0),
            **common,
        )
    rows = [
        ras_sample_semantic(
            logits[i : i + 1],
            None if recent_ids is None else recent_ids[i : i + 1],
            temperature=_row_value(temperature, i),
            top_k=_row_value(top_k, i),
            top_p=_row_value(top_p, i),
            generator=generators.get(i),
            **common,
        )
        for i in range(num_reqs)
    ]
    return torch.cat(rows, dim=0)


__all__ = [
    "SAMPLING_EPS",
    "filter_top_k_top_p",
    "gumbel_argmax_sample",
    "ras_sample_batch",
    "ras_sample_semantic",
    "sample_scores",
]
