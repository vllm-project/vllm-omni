# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_refiner_startup_config_is_disabled_by_default_and_resolves_enabled_source():
    from vllm_omni.diffusion.models.lingbot_video import (
        normalize_lingbot_refiner_config,
    )

    disabled = normalize_lingbot_refiner_config(
        {},
        base_model="org/base",
        base_revision="base-revision",
    )
    assert disabled.enabled is False

    enabled = normalize_lingbot_refiner_config(
        {
            "lingbot_refiner": {
                "enabled": True,
                "model_dir": "org/refiner",
                "transformer_subfolder": "highres",
                "revision": "refiner-revision",
                "default_run": False,
                "offload_vae_during_denoise": False,
            }
        },
        base_model="org/base",
        base_revision="base-revision",
    )
    assert enabled.enabled is True
    assert enabled.model_dir == "org/refiner"
    assert enabled.transformer_subfolder == "highres"
    assert enabled.revision == "refiner-revision"
    assert enabled.default_run is False
    assert enabled.offload_vae_during_denoise is False


def test_refiner_frame_budget_uses_training_aligned_floor_and_optional_cap():
    from vllm_omni.diffusion.models.lingbot_video import (
        compute_refiner_frame_budget,
        compute_refiner_frame_indices,
    )

    assert compute_refiner_frame_budget(121, 24.0, sample_fps=24) == 121
    assert compute_refiner_frame_budget(121, 30.0, sample_fps=24) == 93
    assert compute_refiner_frame_budget(121, 24.0, sample_fps=24, max_frames=81) == 81
    assert compute_refiner_frame_budget(3, 60.0, sample_fps=24) == 1
    with pytest.raises(ValueError, match="aligned to the VAE temporal factor"):
        compute_refiner_frame_budget(121, 24.0, sample_fps=24, max_frames=80)

    indices = compute_refiner_frame_indices(5, 9)
    assert torch.equal(indices, torch.tensor([0, 1, 2, 3, 4, 4, 4, 4, 4]))


def test_refiner_sigma_schedule_and_initial_noise_match_official_formula():
    from vllm_omni.diffusion.models.lingbot_video import (
        compute_refiner_sigmas,
        prepare_refiner_latent,
    )

    sigmas = compute_refiner_sigmas(
        sigma_max=1.0,
        sigma_min=0.0,
        num_inference_steps=8,
        shift=3.0,
        t_thresh=0.85,
        tail_steps=2,
    )
    assert sigmas[0] == pytest.approx(0.85)
    assert np.all(np.diff(sigmas) < 0)
    assert sigmas[-1] >= 0.0

    x_up = torch.full((1, 1, 1, 1, 1), 2.0)
    noise = torch.full_like(x_up, 6.0)
    mixed = prepare_refiner_latent(x_up, noise, 0.25)
    assert torch.equal(mixed, torch.full_like(x_up, 3.0))


def test_refiner_resize_and_ti2v_first_frame_keep_canonical_layout():
    from vllm_omni.diffusion.models.lingbot_video import (
        align_refiner_first_frame,
        resize_refiner_video,
    )

    video = torch.linspace(0, 1, 1 * 3 * 5 * 8 * 12).reshape(1, 3, 5, 8, 12)
    resized = resize_refiner_video(video, height=16, width=32)
    assert resized.shape == (1, 3, 5, 16, 32)
    assert resized.min() >= 0
    assert resized.max() <= 1

    frame = align_refiner_first_frame(
        Image.new("RGB", (80, 60), color="blue"),
        target_height=16,
        target_width=32,
        source_height=8,
        source_width=12,
    )
    assert frame.shape == (1, 3, 1, 16, 32)
