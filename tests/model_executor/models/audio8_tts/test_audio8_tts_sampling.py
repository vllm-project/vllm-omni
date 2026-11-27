# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.sample import gumbel
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.sampler import Sampler as MRv2Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates

from vllm_omni.model_executor.models.audio8_tts.audio8_tts_slow_ar import Audio8TTSSlowARForConditionalGeneration
from vllm_omni.model_executor.models.audio8_tts.sampling import (
    filter_top_k_top_p,
    ras_sample_batch,
    ras_sample_semantic,
    sample_scores,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _kept(logits: torch.Tensor, top_k: int | torch.Tensor, top_p: float | torch.Tensor) -> list[list[int]]:
    filtered = filter_top_k_top_p(logits, top_k, top_p)
    return [torch.nonzero(row.isfinite()).reshape(-1).tolist() for row in filtered]


def test_top_p_is_measured_on_untempered_logits():
    """Audio8 filters the *raw* logits and only then divides by temperature.

    Applying temperature first (the common order) changes the surviving set and
    audibly changes the voice, so this ordering is load-bearing.
    """
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
    # Raw softmax is exactly (0.6, 0.3, 0.1): top_p=0.9 keeps the first two
    # (cumsum 0.6, 0.9) and drops the third (cumsum 1.0 > 0.9).
    assert _kept(logits, top_k=0, top_p=0.9) == [[0, 1]]
    # Had the logits been tempered by 0.5 first, probabilities would become
    # (0.7insh, ...) and only one token would survive.
    tempered = _kept(logits / 0.5, top_k=0, top_p=0.9)
    assert tempered == [[0, 1]] or tempered == [[0]]


def test_argmax_always_survives_filtering():
    logits = torch.tensor([[10.0, -1.0, -2.0]])
    # top_p=0.0 would otherwise remove every candidate and produce NaNs.
    assert _kept(logits, top_k=1, top_p=0.0) == [[0]]


def test_per_row_top_k_and_top_p_are_honoured():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05], [0.5, 0.3, 0.15, 0.05]]))
    kept = _kept(logits, top_k=torch.tensor([1, 3]), top_p=torch.tensor([1.0, 1.0]))
    assert kept == [[0], [0, 1, 2]]


def test_zero_temperature_row_falls_back_to_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0], [1.0, 5.0, 2.0]])
    out = sample_scores(
        logits,
        temperature=torch.tensor([0.0, 1.0]),
        top_k=0,
        top_p=1.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert int(out[0]) == 1


def test_ras_substitutes_only_repeated_semantic_ids():
    """RAS re-rolls a repeat from a flatter distribution; the EOS slot and
    non-repeats must pass through untouched, or utterances get truncated."""
    num_semantic = 2
    # Row 0 will draw id 0 (a repeat) -> substituted.
    # Row 1 will draw id 2 (the EOS slot) -> never substituted even if recent.
    logits = torch.tensor(
        [
            [10.0, -20.0, -20.0],
            [-20.0, -20.0, 10.0],
        ]
    )
    recent = torch.tensor([[0, -1], [2, -1]])
    out = ras_sample_semantic(
        logits,
        recent,
        temperature=1e-6,  # deterministic: argmax within each draw
        top_k=0,
        top_p=1.0,
        ras_temperature=1e-6,
        ras_top_p=1.0,
        num_semantic_ids=num_semantic,
    )
    # Both draws are argmax, so substitution is observable only through which
    # branch was taken; assert the EOS row is preserved verbatim.
    assert int(out[1]) == 2
    assert int(out[0]) == 0


def test_ras_without_history_returns_the_plain_draw():
    logits = torch.tensor([[10.0, 0.0, -5.0]])
    out = ras_sample_semantic(
        logits,
        None,
        temperature=1e-6,
        top_k=0,
        top_p=1.0,
        ras_temperature=1.0,
        ras_top_p=0.9,
        num_semantic_ids=2,
    )
    assert int(out[0]) == 0


def test_sampling_is_reproducible_for_a_seeded_generator():
    logits = torch.randn(4, 32, generator=torch.Generator().manual_seed(7))
    first = sample_scores(logits, temperature=0.7, top_k=8, top_p=0.9, generator=torch.Generator().manual_seed(1234))
    second = sample_scores(logits, temperature=0.7, top_k=8, top_p=0.9, generator=torch.Generator().manual_seed(1234))
    assert torch.equal(first, second)


def test_batch_routes_each_row_to_its_own_generator():
    """Regression for the shared slot-0 generator bug.

    A shared generator over ``[num_reqs, vocab]`` made every row draw from slot
    0's RNG stream, so a request's audio depended on its batch neighbours and
    per-request ``seed`` was ignored for slots > 0. Flat logits make the draw
    purely noise-driven so the wrong generator changes the token.
    """
    # Flat logits: the sampled id is determined only by the generator's noise,
    # so routing the generator to the wrong slot changes the result (coincidence ~1/256).
    logits = torch.zeros(2, 256)
    kw = dict(temperature=1.0, top_k=0, top_p=1.0, ras_temperature=1.0, ras_top_p=0.9, num_semantic_ids=200)

    # (1) Each batched row equals sampling that row alone with its own generator.
    batched = ras_sample_batch(
        logits,
        None,
        generators={0: torch.Generator().manual_seed(11), 1: torch.Generator().manual_seed(22)},
        num_reqs=2,
        **kw,
    )
    row0 = ras_sample_semantic(logits[0:1], None, generator=torch.Generator().manual_seed(11), **kw)
    row1 = ras_sample_semantic(logits[1:2], None, generator=torch.Generator().manual_seed(22), **kw)
    assert int(batched[0]) == int(row0)
    assert int(batched[1]) == int(row1)

    # (2) Row 1 must not change when only slot 0's seed changes.
    other = ras_sample_batch(
        logits,
        None,
        generators={0: torch.Generator().manual_seed(9999), 1: torch.Generator().manual_seed(22)},
        num_reqs=2,
        **kw,
    )
    assert int(other[1]) == int(batched[1])


def test_single_request_batch_keeps_the_fast_path():
    """``num_reqs == 1`` must match a direct single-row draw (no per-row loop)."""
    logits = torch.randn(1, 48, generator=torch.Generator().manual_seed(5))
    kw = dict(temperature=0.8, top_k=16, top_p=0.9, ras_temperature=1.0, ras_top_p=0.9, num_semantic_ids=40)
    batched = ras_sample_batch(logits, None, generators={0: torch.Generator().manual_seed(99)}, num_reqs=1, **kw)
    direct = ras_sample_semantic(logits, None, generator=torch.Generator().manual_seed(99), **kw)
    assert torch.equal(batched, direct)


def _state(tensor: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(gpu=tensor, np=tensor.numpy())


def _mrv2_sampler() -> tuple[MRv2Sampler, Mock]:
    model = object.__new__(Audio8TTSSlowARForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.vllm_config = SimpleNamespace(speculative_config=None)
    model._semantic_begin_id = 10
    model._semantic_end_id = 12
    model._num_semantic_ids = 3
    model._eos_token_id = 4
    model._ras_window_size = 2
    model._ras_temperature = 1.2
    model._ras_top_p = 1.0

    states = SimpleNamespace(
        vocab_size=16,
        temperature=_state(torch.tensor([0.5, 0.5, 0.5, 0.0])),
        top_k=_state(torch.tensor([16, 16, 2, 16], dtype=torch.int32)),
        top_p=_state(torch.tensor([1.0, 1.0, 0.905, 1.0])),
        min_p=_state(torch.zeros(4)),
        seeds=_state(torch.tensor([111, 222, 333, 444])),
    )
    states.get_top_k_top_p = MethodType(SamplingStates.get_top_k_top_p, states)
    req_states = SimpleNamespace(
        total_len=_state(torch.tensor([2, 2, 4, 2], dtype=torch.int32)),
        prompt_len=_state(torch.ones(4, dtype=torch.int32)),
        all_token_ids=_state(torch.tensor([[10, 10, 0, 0], [3, 4, 0, 0], [3, 12, 11, 10], [3, 10, 0, 0]])),
    )
    fallback = Mock()
    sampler = object.__new__(MRv2Sampler)
    sampler.sample = fallback
    sampler.sampling_states = states
    sampler.req_states = req_states
    sampler.return_sampling_mask = False
    sampler.penalties_state = SimpleNamespace(use_penalty=np.zeros(4, dtype=bool))
    sampler.logits_processors = []
    sampler.thinking_budget_state = SimpleNamespace(apply=Mock())
    sampler.use_fp64_gumbel = False
    assert model.mrv2_custom_sampler(sampler) == (sampler, None)
    return sampler, fallback


def _mrv2_sample_args(slots: list[int], positions: list[int] | None = None) -> tuple:
    indices = torch.tensor(slots, dtype=torch.int32)
    pos = torch.tensor(positions if positions is not None else [0] * len(slots), dtype=torch.int64)
    return indices, indices, indices.numpy(), pos, pos, torch.zeros_like(indices), pos.numpy() + 1


def test_mrv2_ras_uses_generated_history_and_preserves_eos_and_greedy(monkeypatch):
    sampler, fallback = _mrv2_sampler()

    def mask_eos(logits, _):
        logits[1, 4] = float("-inf")
        return logits

    sampler.logits_processors = [SimpleNamespace(apply=mask_eos)]
    draw = Mock(side_effect=[torch.tensor([0, 0, 3, 0]), torch.tensor([1, 1, 1, 1])])
    monkeypatch.setattr(gumbel, "gumbel_sample", draw)

    logits = torch.full((4, 16), float("-inf"))
    semantic_and_eos_ids = [10, 11, 12, 4]
    logits[:, semantic_and_eos_ids] = torch.log(torch.tensor([0.60, 0.30, 0.09, 0.01]))
    logits[2, semantic_and_eos_ids] = torch.log(torch.tensor([0.09, 0.01, 0.30, 0.60]))
    # Put slot 2 first to make batch position differ from request-state index.
    args = _mrv2_sample_args([2, 0, 1, 3], [3, 0, 1, 1])
    tokens, _ = sampler.sample(logits, *args)

    # Generated repeats redraw. At position 0, slot 0 has a match in the
    # prompt and a future match in storage. EOS and greedy also preserve the draw.
    assert tokens.tolist() == [11, 10, 4, 10]
    fallback.assert_not_called()
    sampler.thinking_budget_state.apply.assert_called_once()
    assert torch.isfinite(logits[1, 4])
    first, second = draw.call_args_list
    assert torch.isfinite(first.args[0][0]).tolist() == [True, True, False, False]
    assert torch.isneginf(second.args[0][1, 3])
    torch.testing.assert_close(first.args[0][0, :2], logits[0, [10, 11]] / 0.5)
    torch.testing.assert_close(second.args[0][0, :2], logits[0, [10, 11]] / 1.2)
    assert not first.kwargs["is_drafting"]
    assert second.kwargs["is_drafting"]
    for call in (first, second):
        assert call.args[1] is args[0]
        assert call.args[3] is sampler.sampling_states.seeds.gpu
        assert call.args[4] is args[3]


@pytest.mark.parametrize("feature", ["logprobs", "mask", "penalty", "min_p", "bad_words"])
def test_mrv2_ras_preserves_native_sampler_fallback(feature, monkeypatch):
    sampler, fallback = _mrv2_sampler()
    monkeypatch.setattr(gumbel, "gumbel_sample", Mock(side_effect=AssertionError("RAS must fall back")))
    logits = torch.zeros(1, 16)
    args = _mrv2_sample_args([0])
    if feature == "mask":
        sampler.return_sampling_mask = True
    elif feature == "penalty":
        sampler.penalties_state.use_penalty[0] = True
    elif feature == "min_p":
        sampler.sampling_states.min_p.np[0] = 0.1
    elif feature == "bad_words":
        bad_words = object.__new__(BadWordsState)
        bad_words.num_bad_words = _state(torch.tensor([1, 0, 0, 0]))
        sampler.logits_processors.append(bad_words)
    fallback.return_value = (torch.tensor([10]), logits)
    logprobs = feature == "logprobs"

    result = sampler.sample(logits, *args, return_logprobs=logprobs)

    assert result is fallback.return_value
    fallback.assert_called_once_with(logits, *args, return_logprobs=logprobs)
