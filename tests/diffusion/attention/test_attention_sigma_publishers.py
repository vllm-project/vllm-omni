# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Normalized noise publication without loading model weights."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.attention.schedule import (
    normalized_sigma,
    parse_attention_sigma_schedule,
    select_attention_profile_by_sigma,
)
from vllm_omni.diffusion.forward_context import (
    DenoiseProgressMixin,
    begin_scheduled_denoise,
    bind_attention_sigma_schedule,
    get_forward_context,
    request_denoise_progress,
    set_forward_context,
    set_forward_context_denoise_sigma,
)
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_publish_denoise_progress

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("count", [4, 10, 30])
@pytest.mark.parametrize("shift", [1.0, 3.0, 7.0])
def test_switches_at_scheduler_noise_not_fraction_of_steps(count, shift):
    raw = [1.0 - i / count for i in range(count)]
    sigmas = [shift * s / (1.0 + (shift - 1.0) * s) for s in raw]
    publisher = DenoiseProgressMixin()
    publisher.scheduler = SimpleNamespace(sigmas=sigmas, config=SimpleNamespace(num_train_timesteps=1000))
    schedule = parse_attention_sigma_schedule([{"low": 0.0, "high": 0.3, "profile": "low_noise"}])
    with set_forward_context(), bind_attention_sigma_schedule(schedule):
        total = begin_scheduled_denoise(count)
        assert total == count
        for i, sigma in enumerate(sigmas):
            publisher.record_denoise_step(i, sigma * 1000, total_steps=total)
            ctx = get_forward_context()
            assert ctx.denoise_sigma == pytest.approx(sigma)
            assert ctx.denoise_timestep == pytest.approx(sigma)
            # The switch is bracketed by actual noise samples, not by i/count.
            expected = "low_noise" if sigma < 0.3 else None
            assert select_attention_profile_by_sigma(schedule, ctx.denoise_sigma) == expected
        publisher.record_denoise_step(None)
        assert ctx.denoise_sigma is None
        assert not ctx.attention_sigma_schedule_active


def test_scheduler_noise_is_independent_of_model_timestep():
    publisher = DenoiseProgressMixin()
    scheduler = SimpleNamespace(sigmas=[12.0, 6.0, 0.0], config=SimpleNamespace(num_train_timesteps=1000))
    with set_forward_context():
        publisher.record_denoise_step(1, 987.0, scheduler=scheduler, normalized_timestep=0.987)
        ctx = get_forward_context()
        assert ctx.denoise_sigma == 0.5
        assert ctx.denoise_timestep == 0.987
        publisher.record_denoise_step(1, 987.0, scheduler=SimpleNamespace())
        assert ctx.denoise_sigma is None  # no stale noise from the last scheduler


def test_explicit_flow_sigma_does_not_use_scheduler_index():
    with set_forward_context():
        DenoiseProgressMixin().record_denoise_step(2, 750.0, normalized_sigma=0.75)
        assert get_forward_context().denoise_sigma == 0.75
        minimax_h3_publish_denoise_progress(1, 0.25, 4)
        assert get_forward_context().denoise_sigma == 0.25
        assert get_forward_context().denoise_timestep == 0.25
        minimax_h3_publish_denoise_progress(None, None)
        assert get_forward_context().denoise_sigma is None


def test_request_progress_restores_sigma_on_exception():
    schedule = parse_attention_sigma_schedule([{"low": 0.0, "high": 1.0, "profile": "all"}])
    with set_forward_context(), bind_attention_sigma_schedule(schedule):
        ctx = get_forward_context()
        set_forward_context_denoise_sigma(0.9)
        with pytest.raises(RuntimeError, match="cancelled"):
            with request_denoise_progress(2, 4, 200.0, sigma=0.2):
                assert ctx.denoise_sigma == 0.2
                assert ctx.attention_sigma_schedule_active
                raise RuntimeError("cancelled")
        assert ctx.denoise_sigma == 0.9
        assert not ctx.attention_sigma_schedule_active


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_noise_publisher_rejects_invalid_normalized_values(value):
    with set_forward_context(), pytest.raises((ValueError, TypeError)):
        set_forward_context_denoise_sigma(value)


@pytest.mark.parametrize("reference", [0.0, -1.0, float("inf"), float("nan")])
def test_normalization_requires_positive_finite_reference(reference):
    with pytest.raises(ValueError, match="reference"):
        normalized_sigma(0.1, reference)
