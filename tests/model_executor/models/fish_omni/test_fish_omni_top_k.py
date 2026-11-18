# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression for #9157: only the first main-token draw uses the wider top-k."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.fish_omni.sampling import FusedFrameSampler

pytestmark = [pytest.mark.core_model]


@pytest.fixture(
    params=[
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param(
            "cuda",
            marks=[
                *hardware_marks(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1),
                pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA or ROCm"),
            ],
        ),
    ],
)
def device(request):
    return torch.device(request.param)


def test_main_codebook_and_ras_candidate_ranges(device):
    width = 4096
    rows = 3
    sampler = FusedFrameSampler(max_top_k=width, num_codebooks=2, max_rows=rows, device=device)
    # Distinct, nearly uniform probabilities. Noise favors rank 100 in the
    # main draw, then rank 29 in RAS and the codebook. Ranks are zero-based.
    logits = (-torch.arange(width, device=device) / width).repeat(rows, 1)
    ids = torch.arange(width, device=device) + 1
    history = torch.full((rows, 16), -1, dtype=torch.long, device=device)
    temperature = torch.ones(rows, device=device)
    top_p = torch.full((rows,), 0.9, device=device)
    top_k = torch.tensor([width, 100, 7], device=device)
    noise = sampler.noise(rows, device).fill_(0.1)
    assert noise.shape == (rows, width + 30 + 30)
    noise[:, [100, width + 29, width + 30 + 29]] = 0.99999

    tokens, _ = sampler.main_tokens(logits, ids, history, temperature, top_k, top_p, noise, 0, width)
    assert tokens.tolist() == [101, 1, 1]
    codes = sampler.code(logits, temperature, top_k, top_p, noise, 0)
    assert codes.tolist() == [29, 29, 0]

    history[:, -1] = tokens
    redrawn, _ = sampler.main_tokens(logits, ids, history, temperature, top_k, top_p, noise, 0, width)
    assert redrawn.tolist() == [30, 30, 30]


def test_eos_at_top_p_boundary_can_end_first_or_repeated_draw(device):
    sampler = FusedFrameSampler(max_top_k=32, num_codebooks=2, max_rows=3, device=device)
    logits = torch.full((3, 32), -torch.inf, device=device)
    logits[:, :3] = torch.tensor([0.88, 0.09, 0.03], device=device).log()
    ids = torch.arange(11, 43, device=device)
    ids[1] = 0  # EOS crosses both the request's top_p=0.9 and RAS top_p=0.95.
    top_k = torch.tensor([32, 32, 1], device=device)
    top_p = torch.full((3,), 0.9, device=device)
    temperature = torch.ones(3, device=device)
    history = torch.full((3, 16), -1, dtype=torch.long, device=device)
    history[1, -1] = 11
    noise = sampler.noise(3, device).fill_(0.1)
    # Favor EOS, then favor the first excluded rank even more. Row 1 first
    # draws its repeated semantic token and gets a second chance to draw EOS.
    for start in (0, 32, 32 + 30):
        noise[:, start + 1] = 0.999
        noise[:, start + 2] = 0.99999
    noise[1, :3] = 0.1
    tokens, _ = sampler.main_tokens(logits, ids, history, temperature, top_k, top_p, noise, 10, 32)
    assert tokens.tolist() == [0, 0, 11]
    codes = sampler.code(logits, temperature, top_k, top_p, noise, 0)
    assert codes.tolist() == [0, 0, 0]
