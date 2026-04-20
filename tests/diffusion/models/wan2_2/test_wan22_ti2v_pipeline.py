# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from tests.diffusion.models.wan2_2.conftest import StubTransformer, StubVAE, noop_progress_bar
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_ti2v import (
    Wan22TI2VPipeline,
    get_wan22_ti2v_pre_process_func,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_ti2v_pipeline() -> Wan22TI2VPipeline:
    pipeline = object.__new__(Wan22TI2VPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = StubTransformer(in_channels=4, out_channels=4)
    pipeline.vae = StubVAE(z_dim=4)
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    pipeline.progress_bar = noop_progress_bar
    return pipeline


def test_ti2v_preprocess_uses_720p_area_for_image_condition() -> None:
    preprocess = get_wan22_ti2v_pre_process_func(SimpleNamespace())
    request = SimpleNamespace(
        prompts=[{"prompt": "p", "multi_modal_data": {"image": Image.new("RGB", (320, 160), "blue")}}],
        sampling_params=SimpleNamespace(height=None, width=None),
    )

    result = preprocess(request)

    assert result.sampling_params.height == 672
    assert result.sampling_params.width == 1344
    assert result.prompts[0]["multi_modal_data"]["image"].size == (1344, 672)
    assert result.prompts[0]["additional_information"]["preprocessed_image"].shape[-2:] == (672, 1344)


def test_ti2v_diffuse_without_image_condition_expands_patch_timesteps() -> None:
    pipeline = _make_ti2v_pipeline()
    latents = torch.zeros(1, 4, 2, 4, 4)
    calls = []

    def fake_predict_noise_maybe_with_cfg(**kwargs):
        calls.append(kwargs)
        return torch.ones_like(latents)

    pipeline.predict_noise_maybe_with_cfg = fake_predict_noise_maybe_with_cfg  # type: ignore[method-assign]
    pipeline.scheduler_step_maybe_with_cfg = lambda noise, t, current, cfg: current + noise  # type: ignore[method-assign]

    result = pipeline.diffuse(
        latents=latents,
        timesteps=torch.tensor([7]),
        prompt_embeds=torch.zeros(1, 2, 3),
        negative_prompt_embeds=torch.zeros(1, 2, 3),
        guidance_scale=3.0,
        dtype=torch.float32,
        attention_kwargs={"a": "b"},
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
    )

    positive = calls[0]["positive_kwargs"]
    assert calls[0]["do_true_cfg"] is True
    assert positive["timestep"].shape == (1, 8)
    torch.testing.assert_close(positive["timestep"], torch.full((1, 8), 7, dtype=positive["timestep"].dtype))
    torch.testing.assert_close(positive["hidden_states"], latents)
    torch.testing.assert_close(result, torch.ones_like(latents))


def test_ti2v_prepare_i2v_latents_encodes_condition_and_masks_first_frame() -> None:
    pipeline = _make_ti2v_pipeline()
    latents, latent_condition, first_frame_mask = pipeline.prepare_i2v_latents(
        image=torch.zeros(1, 3, 16, 16),
        batch_size=1,
        num_channels_latents=4,
        height=16,
        width=16,
        num_frames=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=None,
        latents=torch.zeros(1, 4, 2, 2, 2),
    )

    torch.testing.assert_close(latents, torch.zeros(1, 4, 2, 2, 2))
    assert latent_condition.shape == (1, 4, 1, 2, 2)
    assert first_frame_mask[:, :, 0].sum() == 0
    assert first_frame_mask[:, :, 1].sum() == 4
