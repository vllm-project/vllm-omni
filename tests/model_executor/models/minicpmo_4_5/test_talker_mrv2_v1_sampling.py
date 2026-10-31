# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 Talker codec ids match V1's seeded sampler on the same logits."""

from __future__ import annotations

import pytest
import torch
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.minicpmo_4_5.talker_mrv2_sampling import draw_v1_codec_ids

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_VOCAB = 64


def _generators(n: int, seed: int) -> dict[int, torch.Generator]:
    generators = {}
    for row in range(n):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + row)
        generators[row] = generator
    return generators


def _v1_sample(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """The V1 sampler tail: temperature, top-k/top-p, seeded exponential draw."""
    metadata = SamplingMetadata(
        temperature=temperature,
        all_greedy=bool(torch.all(temperature < 1e-5).item()),
        all_random=bool(torch.all(temperature >= 1e-5).item()),
        top_p=top_p,
        top_k=top_k,
        generators=generators,
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(logits.shape[0]),
        presence_penalties=torch.zeros(logits.shape[0]),
        repetition_penalties=torch.ones(logits.shape[0]),
        output_token_ids=[[] for _ in range(logits.shape[0])],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )
    sampled, _ = Sampler().sample(logits.clone(), metadata)
    return sampled


def _mrv2_sample(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Scale like the MRv2 logit pipeline, then the shared V1 draw."""
    scale = torch.where(temperature < 1e-5, torch.ones_like(temperature), temperature)
    processed = logits / scale.unsqueeze(1)
    return draw_v1_codec_ids(
        processed,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        generators=generators,
    )


def test_seeded_codec_ids_match_v1_across_steps():
    temperature = torch.tensor([0.8, 0.8])
    top_k = torch.tensor([8, 8])
    top_p = torch.tensor([0.85, 0.85])
    v1_generators = _generators(2, 42)
    mrv2_generators = _generators(2, 42)
    for step in range(6):
        logits = torch.arange(_VOCAB, dtype=torch.float32).repeat(2, 1)
        logits = logits + torch.randn(2, _VOCAB).mul_(step + 1)
        v1_ids = _v1_sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, generators=v1_generators)
        mrv2_ids = _mrv2_sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, generators=mrv2_generators)
        assert torch.equal(v1_ids, mrv2_ids), step


def test_mixed_greedy_rows_match_v1_and_keep_the_generator():
    temperature = torch.tensor([0.0, 0.8])
    top_k = torch.tensor([5, 5])
    top_p = torch.tensor([1.0, 0.9])
    logits = torch.randn(2, _VOCAB)
    v1_ids = _v1_sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, generators=_generators(2, 7))
    mrv2_ids = _mrv2_sample(logits, temperature=temperature, top_k=top_k, top_p=top_p, generators=_generators(2, 7))
    assert torch.equal(v1_ids, mrv2_ids)
    assert int(v1_ids[0]) == int(logits[0].argmax())

    # An all-greedy step must not consume the generator: the next seeded draw
    # matches a generator that never saw that step.
    greedy_logits = torch.randn(2, _VOCAB)
    held = _generators(2, 7)
    fresh = _generators(2, 7)
    _mrv2_sample(
        greedy_logits,
        temperature=torch.zeros(2),
        top_k=None,
        top_p=None,
        generators=held,
    )
    follow = torch.randn(2, _VOCAB)
    warm = torch.tensor([0.8, 0.8])
    assert torch.equal(
        _mrv2_sample(follow, temperature=warm, top_k=top_k, top_p=top_p, generators=held),
        _mrv2_sample(follow, temperature=warm, top_k=top_k, top_p=top_p, generators=fresh),
    )
