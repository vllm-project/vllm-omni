# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU numerical and model-call contracts for sampler-owned integration."""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch

from vllm_omni.diffusion import sampler as sampler_module
from vllm_omni.diffusion.sampler import (
    KSAMPLER,
    Dpmpp2mHistory,
    MultistepHistory,
    ResMultistepHistory,
    Sampler,
    SamplingState,
    _commit,
    dpmpp_2m_coefficients,
    res_multistep_coefficients,
    sample_dpmpp_2m,
    sample_heun,
    sample_res_multistep,
    sampler_object,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _state(sigmas=(1.0, 0.5, 0.25, 0.0), dtype=torch.float32):
    return SamplingState((torch.ones(3, dtype=dtype),), (sigmas,))


def test_sampler_wraps_algorithm_function():
    sampler = sampler_object("res_multistep")
    assert isinstance(sampler, Sampler)
    assert isinstance(sampler, KSAMPLER)
    assert sampler.sampler_function is sample_res_multistep
    assert sampler_object("dpmpp_2m").sampler_function is sample_dpmpp_2m
    with pytest.raises(TypeError):
        Sampler()
    with pytest.raises(ValueError, match="Unsupported sampler"):
        sampler_object("unknown")
    assert sampler_object("heun").sampler_function is sample_heun


@pytest.mark.parametrize("name", ["euler", "res_multistep", "heun", "dpmpp_2m"])
def test_full_loop_matches_resumed_steps(name):
    full, stepped = _state(), _state()

    def model(es):
        return [tuple(x * 0.2 + sigma for x, sigma in zip(e.latents, e.sigmas)) for e in es]

    sampler_object(name).sample(model, [full])
    while not stepped.finished:
        sampler_object(name).sample(model, [stepped], num_steps=1)
    assert torch.equal(full.latents[0], stepped.latents[0])


@pytest.mark.parametrize("name", ["euler", "res_multistep", "heun", "dpmpp_2m"])
def test_failure_does_not_publish_any_state_or_history(name):
    states = [_state(), _state()]
    originals = [s.latents for s in states]
    histories = [s.history for s in states]

    def bad_model(es):
        return [(torch.zeros(3),), (torch.tensor([float("nan")]),)]

    with pytest.raises(ValueError):
        sampler_object(name).sample(bad_model, states, num_steps=1)
    for s, original, history in zip(states, originals, histories):
        assert s.step_index == 0
        assert s.latents is original
        assert s.history is history
        assert s.history is None
    sampler_object(name).sample(lambda es: [(torch.zeros(3),) for e in es], states, num_steps=1)
    assert all(s.step_index == 1 for s in states)


def test_context_wraps_prediction_and_callback_runs_once_per_step():
    events = []

    @contextmanager
    def context(states):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def model(es):
        events.append("predict")
        return [(torch.zeros(3),) for e in es]

    sampler_object("euler").sample(
        model, [_state()], num_steps=1, callback=lambda s: events.append(s.step_index), step_context=context
    )
    assert events == ["enter", "predict", 1, "exit"]


def test_res_history_is_owned_by_request_and_pass():
    first, second = _state(), _state()
    sampler_object("res_multistep").sample(lambda es: [(torch.full((3,), 2.0),)], [first], num_steps=1)
    assert isinstance(first.history, ResMultistepHistory)
    assert first.history.old_denoised[0][0] == 2.0
    assert second.history is None
    assert _state().history is None


@pytest.mark.parametrize("name", ["euler", "heun"])
def test_stateless_algorithms_do_not_create_res_history(name, monkeypatch):
    def unexpected_coefficients(*args, **kwargs):
        pytest.fail("Stateless sampling must not construct RES history")

    monkeypatch.setattr(sampler_module, "res_multistep_coefficients", unexpected_coefficients)
    state = _state()
    sampler_object(name).sample(lambda es: [(torch.zeros(3),) for e in es], [state])
    assert state.history is None


def test_res_stream_histories_are_independent():
    state = SamplingState((torch.ones(3), torch.ones(2)), ((1.0, 0.5, 0.0), (0.8, 0.3, 0.0)))
    sampler_object("res_multistep").sample(
        lambda es: [(torch.full((3,), 2.0), torch.full((2,), 4.0))], [state], num_steps=1
    )
    assert isinstance(state.history, ResMultistepHistory)
    video, audio = state.history.old_denoised
    assert video is not audio
    torch.testing.assert_close(video, torch.full((3,), 2.0))
    torch.testing.assert_close(audio, torch.full((2,), 4.0))


@pytest.mark.parametrize("completed_steps", [0, 1])
def test_res_invalid_update_preserves_batch_history_and_allows_retry(monkeypatch, completed_steps):
    states = [_state(), _state()]
    sampler = sampler_object("res_multistep")

    def model(es):
        return [(torch.full((3,), 0.4),) for e in es]

    if completed_steps:
        sampler.sample(model, states, num_steps=1)
    originals = [(s.latents, s.history) for s in states]
    previous = [s.history.old_denoised[0].clone() if s.history else None for s in states]
    step = sampler_module._multistep_step
    calls = 0

    def invalid_second_update(*args):
        nonlocal calls
        calls += 1
        out = step(*args)
        return out if calls == 1 else torch.full_like(out, float("nan"))

    with monkeypatch.context() as patch:
        patch.setattr(sampler_module, "_multistep_step", invalid_second_update)
        with pytest.raises(ValueError, match="Sampler output must be finite"):
            sampler.sample(model, states, num_steps=1)
    for state, (latents, history), prediction in zip(states, originals, previous):
        assert state.latents is latents
        assert state.history is history
        assert state.step_index == completed_steps
        if history is not None:
            assert torch.equal(history.old_denoised[0], prediction)
    sampler.sample(model, states, num_steps=1)
    assert all(s.step_index == completed_steps + 1 for s in states)


def test_custom_algorithm_can_keep_multiple_predictions_in_its_own_history():
    @dataclass
    class PredictionHistory:
        predictions: tuple[torch.Tensor, ...] = ()

    def algorithm(model, states, *, num_steps=None, callback=None, step_context=None):
        # A custom algorithm can use its own history type without changing SamplingState.
        from vllm_omni.diffusion.sampler import _current_evaluations, _evaluate

        predictions = _evaluate(model, _current_evaluations(states))
        histories = []
        for state, (prediction,) in zip(states, predictions):
            history = state.history or PredictionHistory()
            assert isinstance(history, PredictionHistory)
            histories.append(PredictionHistory((*history.predictions[-1:], prediction)))
        _commit(states, predictions, callback, histories=histories)

    first, second = _state(), _state()
    sampler = KSAMPLER(algorithm)
    observed = []
    for value in (2.0, 3.0, 4.0):
        sampler.sample(
            lambda es: [(torch.full((3,), value),)],
            [first],
            num_steps=1,
            callback=lambda s: observed.append((s.step_index, s.history)),
        )
    assert [step for step, _ in observed] == [1, 2, 3]
    assert [p[0].item() for p in first.history.predictions] == [3.0, 4.0]
    assert [p[0].item() for p in observed[0][1].predictions] == [2.0]
    assert second.history is None


_OTHER_HISTORY = {
    "res_multistep": Dpmpp2mHistory(((None, None, None),), (None,)),
    "dpmpp_2m": ResMultistepHistory(((None, None, None),), (None,)),
}


@pytest.mark.parametrize("name,label", [("res_multistep", "RES"), ("dpmpp_2m", r"DPM\+\+ 2M")])
@pytest.mark.parametrize("history,step_index", [(object(), 0), (None, 1), ("other", 0)])
def test_multistep_rejects_incompatible_or_missing_history_before_model(name, label, history, step_index):
    state = _state()
    state.history = _OTHER_HISTORY[name] if history == "other" else history
    state.step_index = step_index

    def model(es):
        pytest.fail("Invalid history must be rejected before model evaluation")

    with pytest.raises(ValueError, match=label):
        sampler_object(name).sample(model, [state], num_steps=1)


@pytest.mark.parametrize("sigmas", [(), (1.0,), (1.0, 1.0), (0.5, 1.0), (1.0, float("nan"))])
def test_invalid_schedule_fails_before_model(sigmas):
    with pytest.raises(ValueError):
        _state(sigmas)


# Expected values come from an independent float64 reference implementation.
# They pin the sampler math independently of this implementation, so a wrong
# coefficient formula or a dropped history term fails here.
_REF_VIDEO_SIGMAS = (1.0, 0.8, 0.55, 0.3, 0.1, 0.0)
_REF_AUDIO_SIGMAS = (1.0, 0.95, 0.7, 0.4, 0.05, 0.0)
_REF_COEFFS = {
    _REF_VIDEO_SIGMAS: [
        None,
        (0.5912149755174225, -0.2787149755174225),
        (0.8591171097234419, -0.4045716551779873),
        (1.3792885237771657, -0.7126218571104991),
        None,
    ],
    _REF_AUDIO_SIGMAS: [
        None,
        (1.086340638573002, -0.8231827438361596),
        (0.8576881078368309, -0.4291166792654025),
        (3.027265121259988, -2.152265121259988),
        None,
    ],
}
# Final latents of a full run with x0 = 0.5 * tanh(x) + 0.1 * sigma per stream.
_REF_FINAL = {
    "euler": (
        (0.1834079608727471, -0.03733660503628935, 0.08635930017295773),
        (-0.1410948311300223, 0.1362427051424114),
    ),
    "res_multistep": (
        (0.14573506828754482, -0.028911832422266616, 0.06471211385286005),
        (-0.08913371131450108, 0.05127898030681008),
    ),
    # DPM++ 2M reference, run per stream in float64.
    "dpmpp_2m": (
        (0.1512313510351679, -0.030122559589970732, 0.06780526634329226),
        (-0.10197562634476447, 0.07205982894020445),
    ),
}


@pytest.mark.parametrize("sigmas", list(_REF_COEFFS))
def test_res_coefficients_match_reference_implementation(sigmas):
    actual = res_multistep_coefficients(sigmas)
    expected = _REF_COEFFS[sigmas]
    assert [c is None for c in actual] == [c is None for c in expected]
    for got, want in zip(actual, expected, strict=True):
        if want is not None:
            assert got == pytest.approx(want, rel=1e-12, abs=0.0)
    # h*b1 + h*b2 == 1 - sigma_next/sigma_curr, so equal predictions reduce to Euler.
    for i, coeff in enumerate(actual):
        if coeff is not None:
            assert sum(coeff) == pytest.approx(1 - sigmas[i + 1] / sigmas[i], rel=1e-12)


@pytest.mark.parametrize("name", ["euler", "res_multistep", "dpmpp_2m"])
def test_full_coupled_run_matches_reference_implementation(name):
    """Coupled video/audio streams on different schedules, including history and the terminal Euler step."""
    state = SamplingState(
        (
            torch.tensor([0.9, -0.4, 0.25], dtype=torch.float64),
            torch.tensor([-1.2, 0.6], dtype=torch.float64),
        ),
        (_REF_VIDEO_SIGMAS, _REF_AUDIO_SIGMAS),
    )

    def model(evaluations):
        return [tuple(0.5 * torch.tanh(x) + 0.1 * sigma for x, sigma in zip(e.latents, e.sigmas)) for e in evaluations]

    sampler_object(name).sample(model, [state])
    assert state.finished
    for got, want in zip(state.latents, _REF_FINAL[name], strict=True):
        torch.testing.assert_close(got, torch.tensor(want, dtype=torch.float64), rtol=1e-12, atol=0.0)


def _reference_heun(latents, schedules, model):
    """Per-stream float64 Heun as in k-diffusion: Euler proposal, then average both slopes."""
    xs = [x.clone() for x in latents]
    calls = 0
    for i in range(len(schedules[0]) - 1):
        sigmas = [s[i] for s in schedules]
        nexts = [s[i + 1] for s in schedules]
        x0s = model(xs, sigmas)
        calls += 1
        ds = [(x - x0) / sigma for x, x0, sigma in zip(xs, x0s, sigmas)]
        proposals = [x + (nxt - sigma) * d for x, d, sigma, nxt in zip(xs, ds, sigmas, nexts)]
        if nexts[0] == 0:
            xs = proposals
            continue
        x0s_2 = model(proposals, nexts)
        calls += 1
        xs = [
            x + (nxt - sigma) * (d + (proposal - x0_2) / nxt) / 2
            for x, d, proposal, x0_2, sigma, nxt in zip(xs, ds, proposals, x0s_2, sigmas, nexts)
        ]
    return xs, calls


@pytest.mark.parametrize(
    "schedules",
    [(_REF_VIDEO_SIGMAS, _REF_AUDIO_SIGMAS), ((1.0, 0.8, 0.5, 0.2, 0.0), (1.0, 0.7, 0.3, 0.1, 0.0))],
)
def test_heun_full_coupled_run_matches_reference(schedules):
    initial = (
        torch.tensor([0.9, -0.4, 0.25], dtype=torch.float64),
        torch.tensor([-1.2, 0.6], dtype=torch.float64),
    )

    def x0(x, sigma):
        return 0.5 * torch.tanh(x) + 0.1 * sigma

    expected, expected_calls = _reference_heun(initial, schedules, lambda xs, ss: [x0(x, s) for x, s in zip(xs, ss)])
    state = SamplingState(tuple(x.clone() for x in initial), schedules)
    calls = []

    def model(evaluations):
        calls.append([e.sigmas for e in evaluations])
        return [tuple(x0(x, s) for x, s in zip(e.latents, e.sigmas)) for e in evaluations]

    sampler_object("heun").sample(model, [state])
    assert state.finished
    assert len(calls) == expected_calls == 2 * (len(schedules[0]) - 1) - 1
    for got, want in zip(state.latents, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-12, atol=0.0)


def test_heun_correction_evaluates_proposal_at_next_sigma_and_batches_only_non_terminal_requests():
    early, late = _state((1.0, 0.5, 0.0)), _state((1.0, 0.0))
    seen = []

    def model(evaluations):
        seen.append([(e.state, e.sigmas, e.latents[0].clone()) for e in evaluations])
        return [(torch.full((3,), 0.25),) for e in evaluations]

    sampler_object("heun").sample(model, [early, late], num_steps=1)
    prediction, correction = seen
    assert [(s, sig) for s, sig, _ in prediction] == [(early, (1.0,)), (late, (1.0,))]
    # Only the request whose next sigma is non-zero is corrected, at that next sigma,
    # from the Euler proposal x + (0.5 - 1.0) * (x - x0) / 1.0 = 0.625.
    ((state, sigmas, proposal),) = correction
    assert state is early and sigmas == (0.5,)
    torch.testing.assert_close(proposal, torch.full((3,), 0.625))
    # The terminal request keeps its Euler proposal, which lands on x0.
    torch.testing.assert_close(late.latents[0], torch.full((3,), 0.25))
    assert early.step_index == late.step_index == 1
    assert early.history is None and late.history is None


@pytest.mark.parametrize("sigmas", [_REF_VIDEO_SIGMAS, _REF_AUDIO_SIGMAS, (1.0, 0.6, 0.3, 0.1)])
def test_dpmpp_2m_coefficients_extrapolate_in_log_sigma(sigmas):
    """First and zero-sigma steps are Euler; others extrapolate x0 linearly in -log(sigma)."""
    coeffs = dpmpp_2m_coefficients(sigmas)
    for i, coeff in enumerate(coeffs):
        if i == 0 or sigmas[i + 1] == 0:
            assert coeff is None
            continue
        h = math.log(sigmas[i] / sigmas[i + 1])
        r = math.log(sigmas[i - 1] / sigmas[i]) / h
        weight = 1 - sigmas[i + 1] / sigmas[i]
        assert coeff == pytest.approx((weight * (1 + 1 / (2 * r)), -weight / (2 * r)), rel=1e-12)
        # Equal predictions reduce to Euler.
        assert sum(coeff) == pytest.approx(weight, rel=1e-12)
    assert coeffs != res_multistep_coefficients(sigmas)


def test_dpmpp_2m_history_is_per_request_and_released_when_finished():
    first, second = _state(), _state()
    sampler = sampler_object("dpmpp_2m")
    for value in (2.0, 3.0):
        sampler.sample(lambda es: [(torch.full((3,), value),)], [first], num_steps=1)
        assert type(first.history) is Dpmpp2mHistory
        assert first.history.old_denoised[0][0] == value
    assert second.history is None
    # Sibling history types: neither sampler accepts the other's history.
    assert isinstance(first.history, MultistepHistory)
    assert not isinstance(first.history, ResMultistepHistory)
    sampler.sample(lambda es: [(torch.full((3,), 4.0),)], [first], num_steps=1)
    assert first.finished
    assert first.history.old_denoised == (None,)
