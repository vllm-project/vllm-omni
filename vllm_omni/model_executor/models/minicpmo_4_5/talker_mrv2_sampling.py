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

import numpy as np
import torch

from vllm_omni.worker.v1_compat.sample_ops.topk_topp_sampler import (
    apply_top_k_top_p,
    random_sample,
)

# Same cutoff vLLM's V1 sampler uses to treat a row as greedy.
_SAMPLING_EPS = 1e-5


def draw_v1_codec_ids(
    processed_logits: torch.Tensor,
    *,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    temperature: torch.Tensor,
    host_temperature: np.ndarray,
    generators: dict[int, torch.Generator],
    has_topp_only_rows: bool = True,
    use_fp64_gumbel: bool = False,
) -> torch.Tensor:
    """Return one codec id per row, matching V1 ``Sampler.sample``.

    ``temperature`` is one value per row, on the same device as
    ``processed_logits``; ``host_temperature`` holds the same values on the
    host, so branch decisions never read the device. An all-greedy batch
    returns the argmax and does not touch ``generators``. A mixed batch draws
    every row, then restores the argmax on greedy rows, which is the order V1
    uses.
    """
    if processed_logits.ndim != 2:
        raise ValueError(f"codec logits must be 2D, got {tuple(processed_logits.shape)}")
    num_rows = processed_logits.shape[0]
    if temperature.shape[0] != num_rows or host_temperature.shape[0] != num_rows:
        raise ValueError(
            f"temperature rows {temperature.shape[0]}/{host_temperature.shape[0]} != logit rows {num_rows}"
        )
    host_greedy = host_temperature < _SAMPLING_EPS
    # Skipping the draw on an all-greedy step keeps the generator aligned with
    # V1, which returns before random_sample in that case.
    if host_greedy.all():
        return processed_logits.argmax(dim=-1)
    filtered = apply_top_k_top_p(processed_logits.clone(), top_k, top_p, has_topp_only_rows=has_topp_only_rows)
    sampled = random_sample(filtered.softmax(dim=-1, dtype=torch.float32), generators, use_fp64_gumbel)
    if host_greedy.any():
        greedy = temperature.reshape(-1) < _SAMPLING_EPS
        sampled = torch.where(greedy, processed_logits.argmax(dim=-1), sampled)
    return sampled
