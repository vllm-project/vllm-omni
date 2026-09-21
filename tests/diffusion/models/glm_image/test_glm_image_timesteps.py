# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for GLM-Image's split timestep/sigma schedule.

GLM-Image integrates on the resolution-shifted ``sigmas`` (``mu``-shifted) but must
condition the DiT on the **unshifted** timesteps, because the model internalized the
``mu`` shift during training. ``FlowMatchEulerDiscreteScheduler.set_timesteps``
overwrites caller-supplied timesteps with ``sigmas * num_train_timesteps`` (diffusers
dropped the ``if not is_timesteps_provided:`` guard in 0.40.0), which conditioned the
DiT on ``t=929`` where it should see ``t=800`` and produced magenta, grainy images.

These tests need no model weights and no accelerator, and they are the regression
guard for that upstream behaviour: if a future ``diffusers`` bump changes it again,
``test_provided_timesteps_survive_set_timesteps`` fails rather than the output quality
silently degrading.
"""

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from vllm_omni.diffusion.models.glm_image.pipeline_glm_image import calculate_shift, retrieve_timesteps

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NUM_TRAIN_TIMESTEPS = 1000
NUM_INFERENCE_STEPS = 50
# 1024x1024 with vae_scale_factor 8 and patch size 2 -> (1024/8 * 1024/8) / 2**2.
IMAGE_SEQ_LEN_1024 = 4096


def _scheduler() -> FlowMatchEulerDiscreteScheduler:
    """A scheduler configured the way GLM-Image's scheduler_config.json is."""
    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        use_dynamic_shifting=True,
        time_shift_type="linear",
        base_shift=0.25,
        max_shift=0.75,
    )


def _schedules() -> tuple[np.ndarray, np.ndarray]:
    """The two schedules GlmImagePipeline builds before calling retrieve_timesteps."""
    timesteps = np.linspace(NUM_TRAIN_TIMESTEPS, 1.0, NUM_INFERENCE_STEPS + 1)[:-1]
    timesteps = timesteps.astype(np.int64).astype(np.float32)
    sigmas = timesteps / NUM_TRAIN_TIMESTEPS
    return timesteps, sigmas


def _retrieve(scheduler) -> tuple[torch.Tensor, int]:
    timesteps, sigmas = _schedules()
    mu = calculate_shift(IMAGE_SEQ_LEN_1024, 256, 0.25, 0.75)
    return retrieve_timesteps(
        scheduler,
        NUM_INFERENCE_STEPS,
        "cpu",
        timesteps.tolist(),
        sigmas.tolist(),
        mu=mu,
    )


def test_mu_matches_reference_shift():
    """mu is 3.25 at 1024x1024; the rest of the file depends on that shift being applied."""
    assert calculate_shift(IMAGE_SEQ_LEN_1024, 256, 0.25, 0.75) == pytest.approx(3.25)


def test_provided_timesteps_survive_set_timesteps():
    """The DiT must be conditioned on the unshifted timesteps we passed in."""
    expected, _ = _schedules()
    timesteps, num_inference_steps = _retrieve(_scheduler())

    assert num_inference_steps == NUM_INFERENCE_STEPS
    assert timesteps.shape == (NUM_INFERENCE_STEPS,)
    assert timesteps.dtype == torch.float32
    torch.testing.assert_close(timesteps, torch.as_tensor(expected))


def test_sigmas_are_still_resolution_shifted():
    """Restoring the timesteps must not disable the mu shift on the sigmas."""
    scheduler = _scheduler()
    _retrieve(scheduler)
    _, raw_sigmas = _schedules()

    # A terminal 0.0 sigma is appended, so sigmas has one more entry than timesteps.
    assert scheduler.sigmas.shape == (NUM_INFERENCE_STEPS + 1,)
    assert scheduler.sigmas[-1].item() == pytest.approx(0.0)
    assert not np.allclose(scheduler.sigmas[:-1].numpy(), raw_sigmas)
    # Shifting pulls the schedule toward 1.0: sigma[1] is 0.9938, not the raw 0.98.
    assert scheduler.sigmas[1].item() > raw_sigmas[1]


def test_timesteps_differ_from_shifted_product():
    """Guard the actual defect: timesteps must not be sigmas * num_train_timesteps.

    This is what diffusers >= 0.40.0 overwrites them with, and it is what the DiT was
    wrongly conditioned on. The deviation peaks mid-schedule rather than at the ends.
    """
    scheduler = _scheduler()
    timesteps, _ = _retrieve(scheduler)
    shifted_product = scheduler.sigmas[:-1] * NUM_TRAIN_TIMESTEPS

    deviation = (timesteps - shifted_product).abs()
    assert deviation.max().item() > 250.0
    assert timesteps[10].item() == pytest.approx(800.0)
    assert shifted_product[10].item() == pytest.approx(928.571, abs=1e-2)


def test_euler_step_uses_shifted_sigmas():
    """scheduler.step() must still integrate with the shifted sigmas.

    ``step()`` resolves its index by matching the timestep against
    ``scheduler.timesteps``, so restoring the unshifted timesteps must leave the
    Euler update itself untouched. Stepping mid-schedule rather than at index 0,
    where the shifted and unshifted schedules coincide at sigma 1.0 and the
    assertion would hold either way.
    """
    index = 10
    scheduler = _scheduler()
    timesteps, _ = _retrieve(scheduler)

    sample = torch.zeros(1, 4, 8, 8)
    model_output = torch.ones(1, 4, 8, 8)
    prev_sample = scheduler.step(model_output, timesteps[index], sample, return_dict=False)[0]

    assert scheduler.step_index == index + 1, "the unshifted timestep must still resolve to its own index"
    expected = sample + (scheduler.sigmas[index + 1] - scheduler.sigmas[index]) * model_output
    torch.testing.assert_close(prev_sample, expected)

    # And it is genuinely the shifted spacing, not the raw one.
    _, raw_sigmas = _schedules()
    raw_spacing = sample + (raw_sigmas[index + 1] - raw_sigmas[index]) * model_output
    assert not torch.allclose(prev_sample, raw_spacing)
