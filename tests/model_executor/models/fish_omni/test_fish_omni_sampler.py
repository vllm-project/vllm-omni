# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Drama 3 main-token sampler with repetition-aware re-draw.

The request parameters (top_k=1) make the first draw the arg-max token. The
re-draw uses fixed parameters (temperature 1.0, top_k 30, top_p 0.95), so over
many identical rows it spreads across the near-tied candidates it keeps. A row whose
first draw repeats its recent history must therefore often return another
token; a row without a repeat always keeps the arg-max token.
"""

import pytest
import torch
from torch import nn
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.fish_omni.fish_omni_model import FishOmniForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EOS = 5
SEMANTIC_BEGIN = 100
CODEBOOK_SIZE = 8
VOCAB = 128
ROWS = 2000
# Near-tied candidates: the arg-max semantic token, end of speech, and two more
# semantic tokens; top_p 0.95 drops the last of them.
ARGMAX, OTHER_A, OTHER_B = SEMANTIC_BEGIN + 1, SEMANTIC_BEGIN + 2, SEMANTIC_BEGIN + 3


class SemanticRange(nn.Module):
    def is_semantic(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens >= SEMANTIC_BEGIN) & (tokens < SEMANTIC_BEGIN + CODEBOOK_SIZE)


def _sampler_model() -> FishOmniForConditionalGeneration:
    """A model instance with only the state ``sample`` reads (no backbone)."""
    model = FishOmniForConditionalGeneration.__new__(FishOmniForConditionalGeneration)
    nn.Module.__init__(model)
    candidates = torch.tensor([EOS, *range(SEMANTIC_BEGIN, SEMANTIC_BEGIN + CODEBOOK_SIZE)])
    model.register_buffer("_main_token_candidates", candidates, persistent=False)
    model.tower = SemanticRange()
    return model


def _logits(top: int, second: int) -> torch.Tensor:
    logits = torch.full((ROWS, VOCAB), -30.0)
    logits[:, OTHER_A] = 0.0
    logits[:, OTHER_B] = 0.0
    logits[:, second] = 0.01
    logits[:, top] = 0.02
    return logits


def _metadata(history: list[int]) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=torch.full((ROWS,), 1.0),
        all_greedy=False,
        all_random=True,
        top_p=torch.full((ROWS,), 1.0),
        top_k=torch.full((ROWS,), 1, dtype=torch.long),
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(ROWS),
        presence_penalties=torch.zeros(ROWS),
        repetition_penalties=torch.ones(ROWS),
        output_token_ids=[list(history) for _ in range(ROWS)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _sample(logits: torch.Tensor, history: list[int]) -> torch.Tensor:
    torch.manual_seed(0)
    output = _sampler_model().sample(logits, _metadata(history))
    assert output is not None
    assert output.sampled_token_ids.shape == (ROWS, 1)
    return output.sampled_token_ids.squeeze(-1)


def test_non_repeated_token_keeps_first_draw():
    tokens = _sample(_logits(ARGMAX, OTHER_A), history=[OTHER_A, OTHER_B])
    assert bool((tokens == ARGMAX).all())


def test_repeated_semantic_token_is_redrawn():
    tokens = _sample(_logits(ARGMAX, OTHER_A), history=[OTHER_B, ARGMAX])
    kept = (tokens == ARGMAX).float().mean().item()
    assert 0.3 < kept < 0.7
    assert set(tokens.tolist()) <= {ARGMAX, OTHER_A, OTHER_B}


def test_redraw_may_end_speech():
    tokens = _sample(_logits(ARGMAX, EOS), history=[ARGMAX])
    assert bool((tokens == EOS).any())


def test_end_of_speech_is_never_redrawn():
    tokens = _sample(_logits(EOS, OTHER_A), history=[EOS, EOS])
    assert bool((tokens == EOS).all())


def test_repeat_outside_window_keeps_first_draw():
    tokens = _sample(_logits(ARGMAX, OTHER_A), history=[ARGMAX] + [OTHER_B] * 16)
    assert bool((tokens == ARGMAX).all())


def _metadata_with(
    rows: int,
    temperature: torch.Tensor | None,
    all_greedy: bool,
    generators: dict[int, torch.Generator] | None = None,
) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=not all_greedy,
        top_p=None,
        top_k=None,
        generators=generators or {},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(rows),
        presence_penalties=torch.zeros(rows),
        repetition_penalties=torch.ones(rows),
        output_token_ids=[[] for _ in range(rows)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _spread_logits(rows: int) -> torch.Tensor:
    """Several plausible semantic tokens, with OTHER_A the arg-max."""
    logits = torch.full((rows, VOCAB), -30.0)
    logits[:, ARGMAX] = 0.5
    logits[:, OTHER_B] = 0.5
    logits[:, OTHER_A] = 1.0
    return logits


def test_all_greedy_batch_takes_argmax() -> None:
    """vLLM passes temperature=None when every row is greedy."""
    rows = 256
    torch.manual_seed(0)
    output = _sampler_model().sample(_spread_logits(rows), _metadata_with(rows, None, all_greedy=True))
    assert output is not None
    assert (output.sampled_token_ids[:, 0] == OTHER_A).all()


def test_greedy_rows_in_mixed_batch_take_argmax() -> None:
    """A greedy row (temperature -1) does not depend on who shares its batch."""
    rows = 256
    temperature = torch.full((rows,), 1.0)
    temperature[::2] = -1.0
    torch.manual_seed(0)
    output = _sampler_model().sample(_spread_logits(rows), _metadata_with(rows, temperature, all_greedy=False))
    assert output is not None
    sampled = output.sampled_token_ids[:, 0]
    assert (sampled[::2] == OTHER_A).all()
    assert (sampled[1::2] != OTHER_A).any()


def test_request_generator_controls_its_rows() -> None:
    """A seeded row repeats under the same seed and ignores the global RNG."""
    rows = 64

    def run(global_seed: int) -> torch.Tensor:
        torch.manual_seed(global_seed)
        generators = {0: torch.Generator().manual_seed(123)}
        output = _sampler_model().sample(
            _spread_logits(rows), _metadata_with(rows, torch.full((rows,), 1.0), False, generators)
        )
        assert output is not None
        return output.sampled_token_ids[:, 0]

    draws_a = torch.stack([run(seed)[0] for seed in range(16)])
    assert (draws_a == draws_a[0]).all()
    unseeded = torch.stack([run(seed)[1] for seed in range(16)])
    assert (unseeded != unseeded[0]).any()
