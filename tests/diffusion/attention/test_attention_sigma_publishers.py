# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Scheduler sigma publication without loading model weights."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.schedule import (
    InvalidAttentionScheduleError,
    checked_sigma,
    parse_attention_sigma_schedule,
)
from vllm_omni.diffusion.forward_context import (
    DenoiseProgressMixin,
    begin_scheduled_denoise,
    bind_attention_schedule,
    check_scheduler_sigmas,
    denoise_sigma,
    get_attention_schedule_profile,
    get_forward_context,
    request_denoise_progress,
    set_forward_context,
    set_forward_context_denoise_sigma,
)
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import minimax_h3_publish_denoise_progress

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_ALL = parse_attention_sigma_schedule([{"low": 0.0, "high": 1.0, "profile": "all"}])


@pytest.mark.parametrize("count", [4, 10, 30])
@pytest.mark.parametrize("shift", [1.0, 3.0, 7.0])
def test_switches_at_scheduler_noise_not_fraction_of_steps(count, shift):
    raw = [1.0 - i / count for i in range(count)]
    sigmas = [shift * s / (1.0 + (shift - 1.0) * s) for s in raw]
    publisher = DenoiseProgressMixin()
    publisher.scheduler = SimpleNamespace(
        sigmas=torch.tensor(sigmas, dtype=torch.float64), config=SimpleNamespace(num_train_timesteps=1000)
    )
    schedule = parse_attention_sigma_schedule([{"low": 0.0, "high": 0.3, "profile": "low_noise"}])
    with set_forward_context(), bind_attention_schedule(None, schedule):
        total = begin_scheduled_denoise(count)
        assert total == count
        for i, sigma in enumerate(sigmas):
            publisher.record_denoise_step(i, sigma * 1000, total_steps=total)
            ctx = get_forward_context()
            assert ctx.denoise_sigma == pytest.approx(sigma)
            assert ctx.denoise_timestep == pytest.approx(sigma)
            # The switch is bracketed by actual noise samples, not by i/count.
            expected = "low_noise" if sigma < 0.3 else None
            assert get_attention_schedule_profile() == expected
        publisher.record_denoise_step(None)
        assert ctx.denoise_sigma is None
        assert get_attention_schedule_profile() is None


def test_scheduler_noise_is_independent_of_model_timestep():
    publisher = DenoiseProgressMixin()
    scheduler = SimpleNamespace(sigmas=torch.tensor([1.0, 0.5, 0.0]), config=SimpleNamespace(num_train_timesteps=1000))
    with set_forward_context(), bind_attention_schedule(None, _ALL):
        publisher.record_denoise_step(1, 987.0, scheduler=scheduler, normalized_timestep=0.987)
        ctx = get_forward_context()
        assert ctx.denoise_sigma == 0.5
        assert ctx.denoise_timestep == 0.987
        publisher.record_denoise_step(3, 987.0, scheduler=scheduler)
        assert ctx.denoise_sigma is None  # past the ladder: no stale noise from the last step


def test_sigma_is_published_as_is_not_relative_to_the_first_sigma():
    """Wan's UniPC ladder starts at 0.999, and step 0 publishes 0.999, not 1.0."""
    sigmas = torch.tensor([0.999, 0.5, 0.0])
    assert denoise_sigma(0, sigmas) == pytest.approx(0.999)
    assert denoise_sigma(1, sigmas) == 0.5
    assert denoise_sigma(3, sigmas) is None


@pytest.mark.parametrize("sigmas", [[1.0, 0.5, 0.0], (1.0, 0.5, 0.0), torch.ones(2, 2)])
def test_scheduler_sigmas_must_be_a_1d_tensor(sigmas):
    with pytest.raises(TypeError, match="1-D torch.Tensor"):
        denoise_sigma(0, sigmas)
    with set_forward_context(), bind_attention_schedule(None, _ALL), pytest.raises(TypeError, match="1-D torch.Tensor"):
        check_scheduler_sigmas(sigmas)


def test_explicit_flow_sigma_does_not_use_scheduler_index():
    with set_forward_context(), bind_attention_schedule(None, _ALL):
        DenoiseProgressMixin().record_denoise_step(2, 750.0, sigma=0.75)
        assert get_forward_context().denoise_sigma == 0.75
        minimax_h3_publish_denoise_progress(1, 0.25, 4)
        assert get_forward_context().denoise_sigma == 0.25
        assert get_forward_context().denoise_timestep == 0.25
        minimax_h3_publish_denoise_progress(None, None)
        assert get_forward_context().denoise_sigma is None


def test_request_progress_restores_sigma_on_exception():
    with set_forward_context(), bind_attention_schedule(None, _ALL):
        ctx = get_forward_context()
        set_forward_context_denoise_sigma(0.9)
        with pytest.raises(RuntimeError, match="cancelled"):
            with request_denoise_progress(2, 4, 200.0, sigma=0.2):
                assert ctx.denoise_sigma == 0.2
                assert get_attention_schedule_profile() == "all"
                raise RuntimeError("cancelled")
        assert ctx.denoise_sigma == 0.9
        assert get_attention_schedule_profile() is None


def test_scheduler_sigma_check_covers_the_whole_ladder_and_only_under_sigma_windows():
    """Pipelines call this right after set_timesteps, so unusable sigmas fail before the first forward."""
    unusable_last = torch.tensor([1.0, 0.5, 1.5])
    with set_forward_context():
        check_scheduler_sigmas(unusable_last)  # nothing reads sigmas without bound windows
    with set_forward_context(), bind_attention_schedule(None, _ALL):
        check_scheduler_sigmas(torch.tensor([1.0, 0.5, 0.0]))
        with pytest.raises(InvalidAttentionScheduleError, match="check flow_shift"):
            check_scheduler_sigmas(unusable_last)
        assert get_forward_context().denoise_sigma is None  # the check publishes nothing
    # Outside the forward context, as in step-mode prepare_encode, the caller says whether windows apply.
    check_scheduler_sigmas(unusable_last, sigma_scheduled=False)
    with pytest.raises(InvalidAttentionScheduleError, match="check flow_shift"):
        check_scheduler_sigmas(torch.tensor([float("nan"), 0.5, 0.0]), sigma_scheduled=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_noise_publisher_rejects_invalid_sigmas(value):
    with set_forward_context(), bind_attention_schedule(None, _ALL), pytest.raises((ValueError, TypeError)):
        set_forward_context_denoise_sigma(value)


def test_float32_rounding_at_the_end_points_publishes_inside_0_1():
    assert checked_sigma(1.0 + 1e-7) == 1.0
    assert checked_sigma(-1e-7) == 0.0


def test_h3_publishes_a_float32_sigma_one_ulp_above_1_as_1():
    """A float32 shifted ladder can start one ulp above 1 (H3 at flow_shift 2.14, for example)."""
    one_ulp_above = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)).item()
    assert one_ulp_above > 1.0
    with set_forward_context(), bind_attention_schedule(None, _ALL):
        minimax_h3_publish_denoise_progress(0, one_ulp_above, 8)
        assert get_forward_context().denoise_sigma == 1.0
        assert get_attention_schedule_profile() == "all"
