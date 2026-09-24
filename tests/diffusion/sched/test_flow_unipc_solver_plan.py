# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Solver-plan parity, invalidation, and request-state isolation."""

import copy
from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def trajectory(scheduler, *, steps=8, begin=0, dtype=torch.float32, planned=True):
    scheduler.set_timesteps(steps)
    scheduler.set_begin_index(begin)
    sample = torch.arange(32, dtype=dtype).reshape(1, 2, 4, 4) / 32
    outputs = []
    with patch.object(
        scheduler, "_prepare_solver_plan", wraps=scheduler._prepare_solver_plan if planned else lambda _: None
    ):
        for timestep in scheduler.timesteps[begin:]:
            sample = scheduler.step(sample.sin(), timestep, sample).prev_sample
            outputs.append(sample.clone())
    return outputs


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("solver_type", ["bh1", "bh2"])
@pytest.mark.parametrize("predict_x0", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_planned_trajectory_matches_on_demand(order, solver_type, predict_x0, dtype):
    config = dict(solver_order=order, solver_type=solver_type, predict_x0=predict_x0)
    cached = FlowUniPCMultistepScheduler(**config)
    reference = FlowUniPCMultistepScheduler(**config)
    expected = trajectory(reference, dtype=dtype, planned=False)
    for _ in range(2):
        actual = trajectory(cached, dtype=dtype)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, rtol=0, atol=0, equal_nan=True)
    assert cached._solver_plan


def test_identical_schedule_reuses_coefficients_and_resets_history():
    scheduler = FlowUniPCMultistepScheduler()
    expected = trajectory(scheduler)
    with patch.object(scheduler, "_build_step_coefficients", side_effect=AssertionError("unexpected rebuild")):
        actual = trajectory(scheduler)
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("change", ["steps", "shift", "dtype", "begin", "order", "corrector", "final", "sigmas"])
def test_plan_invalidation(change):
    scheduler = FlowUniPCMultistepScheduler()
    trajectory(scheduler)
    old = scheduler._solver_plan_key
    kwargs = {}
    if change == "steps":
        kwargs["steps"] = 6
    elif change == "shift":
        scheduler.set_shift(3.0)
    elif change == "dtype":
        kwargs["dtype"] = torch.bfloat16
    elif change == "begin":
        kwargs["begin"] = 2
    elif change == "order":
        scheduler.register_to_config(solver_order=3)
    elif change == "corrector":
        scheduler.disable_corrector = [1, 3]
    elif change == "final":
        scheduler.register_to_config(lower_order_final=False, final_sigmas_type="sigma_min")
    elif change == "sigmas":
        scheduler.sigma_max = 0.8
    reference = copy.deepcopy(scheduler)
    reference._solver_plan.clear()
    reference._solver_plan_key = None
    actual = trajectory(scheduler, **kwargs)
    expected = trajectory(reference, planned=False, **kwargs)
    assert scheduler._solver_plan_key != old
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, rtol=0, atol=0, equal_nan=True)


def test_copied_schedulers_have_independent_history():
    scheduler = FlowUniPCMultistepScheduler()
    trajectory(scheduler)
    left, right = copy.deepcopy(scheduler), copy.deepcopy(scheduler)
    left.set_timesteps(8)
    right.set_timesteps(8)
    reference = FlowUniPCMultistepScheduler()
    reference.set_timesteps(8)
    a = torch.ones(1, 2, 4, 4)
    b = a * 3
    expected = b.clone()
    for timestep in left.timesteps:
        a = left.step(a.sin(), timestep, a).prev_sample
        b = right.step(b.cos(), timestep, b).prev_sample
        expected = reference.step(expected.cos(), timestep, expected).prev_sample
        torch.testing.assert_close(b, expected, rtol=0, atol=0)
        assert left.last_sample is not right.last_sample


@pytest.mark.parametrize("steps,order", [(129, 2), (8, 4)])
def test_unsupported_configuration_evaluates_on_demand(steps, order):
    scheduler = FlowUniPCMultistepScheduler(solver_order=order)
    expected = trajectory(FlowUniPCMultistepScheduler(solver_order=order), steps=steps, planned=False)
    actual = trajectory(scheduler, steps=steps)
    assert not scheduler._solver_plan
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, rtol=0, atol=0, equal_nan=True)


def test_mutated_sigma_schedule_does_not_reuse_old_plan():
    scheduler = FlowUniPCMultistepScheduler()
    trajectory(scheduler)
    scheduler.set_timesteps(8)
    old = scheduler._solver_plan_key
    scheduler.sigmas[1] *= 0.99
    sample = torch.ones(1, 2, 4, 4)
    scheduler.step(sample, scheduler.timesteps[0], sample)
    assert scheduler._solver_plan_key != old


@pytest.mark.parametrize("kind", ["duplicate", "nonfinite", "external"])
def test_unsupported_schedule_does_not_build_a_plan(kind):
    scheduler = FlowUniPCMultistepScheduler()
    scheduler.set_timesteps(8)
    scheduler._step_index = 0
    if kind == "duplicate":
        scheduler.sigmas[2] = scheduler.sigmas[1]
    elif kind == "nonfinite":
        scheduler.sigmas[2] = float("nan")
    else:
        scheduler.solver_p = FlowUniPCMultistepScheduler()
    with patch.object(scheduler, "_build_step_coefficients", side_effect=AssertionError("unexpected plan")):
        scheduler._prepare_solver_plan(torch.ones(1, 2, 4, 4))
    assert not scheduler._solver_plan


def test_failed_plan_build_does_not_publish_partial_coefficients():
    scheduler = FlowUniPCMultistepScheduler()
    scheduler.set_timesteps(8)
    scheduler._step_index = 0
    build = scheduler._build_step_coefficients

    def fail_on_second_interval(index, *args):
        if index == 1:
            raise RuntimeError("injected failure")
        return build(index, *args)

    with patch.object(scheduler, "_build_step_coefficients", side_effect=fail_on_second_interval):
        with pytest.raises(RuntimeError, match="injected failure"):
            scheduler._prepare_solver_plan(torch.ones(1, 2, 4, 4))
    assert not scheduler._solver_plan
    assert scheduler._solver_plan_key is None
