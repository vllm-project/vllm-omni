# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""P3 sampler/state CPU tests with independent small probability/EOS oracles."""

import pytest
import torch

from vllm_omni.model_executor.models.zonos2.zonos2_sampler import (
    Zonos2RequestState,
    Zonos2SamplingParams,
    frame_probabilities,
    repetition_logits,
    sample_frame,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def params(**kwargs):
    return Zonos2SamplingParams(**kwargs)


def test_repetition_penalty_is_per_codebook_once_and_excludes_cb8_and_specials():
    logits = torch.ones(9, 1026)
    logits[0, 5] = 4
    logits[1, 7] = -4
    history = torch.tensor([[5, 7, 1024, 1025, 0, 0, 0, 0, 9], [5, 7, 1024, 1025, 0, 0, 0, 0, 9]], dtype=torch.int32)
    out = repetition_logits(logits, history, params(repetition_penalty=2))
    assert out[0, 5] == 2 and out[1, 7] == -8
    assert out[0, 7] == 1 and out[1, 5] == 1
    assert out[2, 1024] == 1 and out[3, 1025] == 1 and out[8, 9] == 1
    assert logits[0, 5] == 4


@pytest.mark.parametrize("kwargs", [{"repetition_window": 0}, {"repetition_penalty": 1}, {"repetition_codebooks": 0}])
def test_repetition_can_be_disabled(kwargs):
    x = torch.arange(9 * 1026).reshape(9, 1026).float()
    history = torch.zeros((4, 9), dtype=torch.int32)
    torch.testing.assert_close(repetition_logits(x, history, params(**kwargs)), x, rtol=0, atol=0)


def test_short_window_uses_only_recent_generated_codes():
    logits = torch.ones(9, 1026)
    history = torch.tensor([[4] * 9, [5] * 9], dtype=torch.int32)
    out = repetition_logits(logits, history, params(repetition_window=1, repetition_penalty=2))
    assert out[0, 4] == 1 and out[0, 5] == 0.5


def test_topk_keeps_ties_and_topp_uses_exclusive_cumulative_boundary():
    x = torch.tensor([[4.0, 3.0, 3.0, 1.0]]).repeat(9, 1)
    empty = torch.empty((0, 9), dtype=torch.int32)
    p = frame_probabilities(x, empty, params(temperature=1, top_k=2, top_p=1, min_p=0))
    assert p[0, 1] > 0 and p[0, 2] > 0 and p[0, 3] == 0
    x = torch.tensor([[4.0, 2.0, 1.0, 0.0]]).repeat(9, 1)
    p = frame_probabilities(x, empty, params(temperature=1, top_k=-1, top_p=0.7, min_p=0))
    torch.testing.assert_close(p, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(9, 1))


def test_minp_uses_each_codebook_own_peak():
    x = torch.log(torch.tensor([[0.6, 0.3, 0.1], [0.34, 0.33, 0.33]]))
    x = torch.cat((x, x[:1].repeat(7, 1)))
    p = frame_probabilities(
        x, torch.empty((0, 9), dtype=torch.int32), params(temperature=1, top_k=-1, top_p=1, min_p=0.6)
    )
    assert p[0].tolist() == [1.0, 0.0, 0.0]
    assert torch.all(p[1] > 0)


def test_greedy_bypasses_random_filters_and_seed():
    logits = torch.randn(9, 1026)
    empty = torch.empty((0, 9), dtype=torch.int32)
    options = params(temperature=0, top_k=1, top_p=0.01, min_p=1, repetition_penalty=1)
    assert torch.equal(sample_frame(logits, empty, options, 7, 0).long(), logits.argmax(-1))
    assert torch.equal(sample_frame(logits, empty, options, 777, 0), sample_frame(logits, empty, options, 7, 0))


def test_request_sampling_is_independent_of_batch_order_and_resume_position():
    logits = torch.linspace(-2, 2, 1026).repeat(9, 1)
    empty = torch.empty((0, 9), dtype=torch.int32)
    a = params(temperature=0.7, top_k=8, top_p=0.8, min_p=0.2)
    b = params(temperature=1.7, top_k=100, top_p=1, min_p=0)
    alone = [sample_frame(logits, empty, a, 42, i) for i in range(5)]
    interleaved = []
    for i in range(5):
        sample_frame(logits, empty, b, 99, i)
        interleaved.append(sample_frame(logits, empty, a, 42, i))
    assert all(torch.equal(x, y) for x, y in zip(alone, interleaved))
    assert torch.equal(alone[4], sample_frame(logits, empty, a, 42, 4))
    assert not torch.equal(alone[0], sample_frame(logits, empty, a, 43, 0))


@pytest.mark.parametrize("cb", list(range(9)))
def test_any_codebook_eoa_starts_ten_step_countdown(cb):
    state = Zonos2RequestState.rebuild("a", params(max_tokens=100), torch.zeros((12, 9), dtype=torch.int32))
    frame = torch.zeros(9, dtype=torch.int32)
    frame[cb] = 1024
    assert state.append(frame) == 0
    assert state.eos_frame == max(0, 12 - cb) and state.countdown == 9
    for remaining in range(8, 0, -1):
        assert state.append(torch.zeros(9, dtype=torch.int32)) == 0
        assert state.countdown == remaining
    assert state.append(torch.zeros(9, dtype=torch.int32)) == 1
    assert state.countdown == 0


def test_simultaneous_eoa_uses_highest_index_and_never_restarts_countdown():
    state = Zonos2RequestState.rebuild("a", params(), torch.empty((0, 9), dtype=torch.int32))
    frame = torch.zeros(9, dtype=torch.int32)
    frame[[0, 8]] = 1024
    state.append(frame)
    assert state.eos_frame == 0
    state.append(frame)
    assert state.countdown == 8 and state.eos_frame == 0


def test_max_tokens_wins_over_pending_countdown_and_ignore_eos_works():
    state = Zonos2RequestState.rebuild("a", params(max_tokens=2), torch.empty((0, 9), dtype=torch.int32))
    assert state.append(torch.full((9,), 1024, dtype=torch.int32)) == 0
    assert state.append(torch.zeros(9, dtype=torch.int32)) == 1
    ignored = Zonos2RequestState.rebuild(
        "b", params(max_tokens=2, ignore_eos=True), torch.empty((0, 9), dtype=torch.int32)
    )
    assert ignored.append(torch.full((9,), 1024, dtype=torch.int32)) == 0
    assert ignored.eos_frame == -1 and ignored.countdown == -1


def test_reconstruction_restores_eos_repetition_and_next_rng_from_full_history():
    options = params(max_tokens=100, seed=42)
    state = Zonos2RequestState.rebuild("a", options, torch.empty((0, 9), dtype=torch.int32))
    for i in range(7):
        row = torch.full((9,), i, dtype=torch.int32)
        if i == 3:
            row[2] = 1024
        state.append(row)
    resumed = Zonos2RequestState.rebuild("a", options, state.history, state.seed)
    assert resumed.eos_frame == state.eos_frame and resumed.countdown == state.countdown
    assert torch.equal(resumed.history, state.history)
    logits = torch.randn(9, 1026)
    assert torch.equal(
        sample_frame(logits, state.history, options, state.seed, len(state.history)),
        sample_frame(logits, resumed.history, options, resumed.seed, len(resumed.history)),
    )
    state.history[0, 0] = 999
    assert resumed.history[0, 0] == 0


def test_unseeded_request_base_seed_can_be_reconstructed_without_new_randomness():
    options = params(seed=None)
    state = Zonos2RequestState.rebuild("a", options, torch.empty((0, 9), dtype=torch.int32))
    resumed = Zonos2RequestState.rebuild("a", options, state.history, state.seed)
    assert state.seed == resumed.seed
    assert Zonos2SamplingParams.from_runtime({"seed": None}).seed is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -1},
        {"min_p": 1.1},
        {"top_p": -0.1},
        {"top_k": -2},
        {"repetition_penalty": 0.9},
        {"repetition_window": -1},
        {"repetition_codebooks": 9},
        {"max_tokens": 0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        params(**kwargs)


def test_standard_repetition_parameter_and_extra_override_are_request_local():
    assert Zonos2SamplingParams.from_runtime({"repetition_penalty": 1.0}).repetition_penalty == 1.0
    assert (
        Zonos2SamplingParams.from_runtime(
            {"repetition_penalty": 1.0, "extra_args": {"repetition_penalty": 2.0}}
        ).repetition_penalty
        == 2.0
    )


def test_uninterrupted_and_reconstructed_trajectory_remain_identical():
    options = params(temperature=0.9, top_k=50, repetition_window=3, repetition_penalty=1.5)
    empty = torch.empty((0, 9), dtype=torch.int32)
    state = Zonos2RequestState.rebuild("a", options, empty)
    logits = torch.randn(12, 9, 1026, generator=torch.Generator().manual_seed(91))
    for step in range(7):
        state.append(sample_frame(logits[step], state.history, options, state.seed, step))
    resumed = Zonos2RequestState.rebuild("a", options, state.history, state.seed)
    for step in range(7, 12):
        a = sample_frame(logits[step], state.history, options, state.seed, step)
        b = sample_frame(logits[step], resumed.history, options, resumed.seed, step)
        assert torch.equal(a, b)
        assert state.append(a) == resumed.append(b)
    assert torch.equal(state.history, resumed.history)
