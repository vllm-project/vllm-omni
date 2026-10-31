# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Seeded codec draw shared with the V1 Talker sampler.

V1 turns sampling params ``seed`` into a ``torch.Generator`` and draws with
``exponential_`` (``random_sample``). Model Runner V2's stock sampler draws
Triton Gumbel noise from ``(seed, position)`` instead. The two streams are not
the same, so a seeded MiniCPM-o Talker emitted a different codec sequence on
MRv2: shorter audio and higher WER at the same weights and seed.

This draw is the V1 tail only. The caller has already applied the 16-frame
codec penalty, the EOS mask, temperature and min-p.
"""

from __future__ import annotations

import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, random_sample

# Same cutoff vLLM's V1 sampler uses to treat a row as greedy.
_SAMPLING_EPS = 1e-5


def draw_v1_codec_ids(
    processed_logits: torch.Tensor,
    *,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    temperature: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Return one codec id per row, matching V1 ``Sampler.sample``.

    ``temperature`` is one value per row, on the same device as
    ``processed_logits``. An all-greedy batch returns the argmax and does not
    touch ``generators``. A mixed batch draws every row, then restores the
    argmax on greedy rows, which is the order V1 uses.
    """
    if processed_logits.ndim != 2:
        raise ValueError(f"codec logits must be 2D, got {tuple(processed_logits.shape)}")
    if temperature.shape[0] != processed_logits.shape[0]:
        raise ValueError(f"temperature rows {temperature.shape[0]} != logit rows {processed_logits.shape[0]}")
    greedy = temperature.reshape(-1) < _SAMPLING_EPS
    # Host read of a per-row flag. The Talker batch is the scheduled request
    # count, and skipping the draw on an all-greedy step keeps the generator
    # aligned with V1, which returns before random_sample in that case.
    if bool(torch.all(greedy).item()):
        return processed_logits.argmax(dim=-1)
    filtered = apply_top_k_top_p(processed_logits.clone(), top_k, top_p)
    sampled = random_sample(filtered.softmax(dim=-1, dtype=torch.float32), generators, False)
    if bool(torch.any(greedy).item()):
        sampled = torch.where(greedy, processed_logits.argmax(dim=-1), sampled)
    return sampled
