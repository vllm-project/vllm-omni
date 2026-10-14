# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The residual heads and the main-token sampler share one top-k/top-p filter."""

import math

import pytest
import torch

from vllm_omni.model_executor.models.fish_omni.codebook_tower import sample_rows, sample_top_k_top_p

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _frequencies(draws: torch.Tensor, width: int) -> torch.Tensor:
    return torch.bincount(draws, minlength=width).float() / draws.numel()


def test_top_p_uses_full_vocabulary_mass() -> None:
    """probs [0.4, 0.3, 0.2, 0.1], top_k 2, top_p 0.9: tokens 0 and 1 both stay."""
    rows = 10_000
    logits = torch.tensor([[math.log(p) for p in (0.4, 0.3, 0.2, 0.1)]]).expand(rows, -1)
    torch.manual_seed(0)
    scalar = _frequencies(sample_top_k_top_p(logits, 1.0, 2, 0.9), 4)
    torch.manual_seed(0)
    vector = _frequencies(
        sample_rows(
            logits,
            torch.full((rows,), 1.0),
            torch.full((rows,), 2, dtype=torch.long),
            torch.full((rows,), 0.9),
        ),
        4,
    )
    expected = torch.tensor([0.4 / 0.7, 0.3 / 0.7, 0.0, 0.0])
    assert torch.allclose(scalar, expected, atol=0.02)
    assert torch.allclose(vector, expected, atol=0.02)


def test_top_p_keeps_the_most_likely_token() -> None:
    logits = torch.tensor([[2.0, 0.0, 0.0]]).expand(1000, -1)
    torch.manual_seed(0)
    assert (sample_top_k_top_p(logits, 1.0, 0, 0.01) == 0).all()
