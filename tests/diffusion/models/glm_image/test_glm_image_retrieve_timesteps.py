# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for GLM-Image timestep handoff through retrieve_timesteps.

GLM-Image integrates with resolution-shifted sigmas but conditions the DiT on
unshifted timesteps. diffusers >= 0.40 FlowMatchEulerDiscreteScheduler.set_timesteps
overwrites caller-supplied timesteps with `sigmas * num_train_timesteps`;
retrieve_timesteps must restore the provided schedule.
"""

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from vllm_omni.diffusion.models.glm_image.pipeline_glm_image import (
    calculate_shift,
    retrieve_timesteps,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _glm_image_scheduler() -> FlowMatchEulerDiscreteScheduler:
    # Match scheduler/scheduler_config.json shipped with zai-org/GLM-Image.
    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        use_dynamic_shifting=True,
        time_shift_type="linear",
        base_shift=0.25,
        max_shift=0.75,
    )


def test_retrieve_timesteps_preserves_caller_unshifted_timesteps():
    """Caller timesteps must survive set_timesteps when both args are passed."""
    num_inference_steps = 50
    # 1024x1024 -> image_seq_len = ((1024/8)*(1024/8)) / (2**2) = 4096
    image_seq_len = 4096
    scheduler = _glm_image_scheduler()

    timesteps_array = np.linspace(scheduler.config.num_train_timesteps, 1.0, num_inference_steps + 1)[:-1]
    timesteps_array = timesteps_array.astype(np.int64).astype(np.float32)
    sigmas = timesteps_array / scheduler.config.num_train_timesteps
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("base_shift", 0.25),
        scheduler.config.get("max_shift", 0.75),
    )
    assert mu == pytest.approx(3.25)

    provided = timesteps_array.tolist()
    got, n_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        "cpu",
        provided,
        sigmas.tolist(),
        mu=mu,
    )

    expected = torch.as_tensor(provided, dtype=torch.float32)
    assert n_steps == num_inference_steps
    assert got.shape == expected.shape
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(scheduler.timesteps.cpu(), expected, rtol=0, atol=0)

    # Integration still uses the resolution-shifted sigma schedule.
    shifted = torch.as_tensor(sigmas, dtype=torch.float32)
    assert not torch.allclose(scheduler.sigmas[:-1].cpu(), shifted, rtol=0, atol=0)
