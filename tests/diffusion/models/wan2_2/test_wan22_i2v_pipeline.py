# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from tests.diffusion.models.wan2_2.conftest import StubTransformer, StubVAE, noop_progress_bar
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import (
    Wan22I2VPipeline,
    get_wan22_i2v_pre_process_func,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_i2v_pipeline(*, expand_timesteps: bool) -> Wan22I2VPipeline:
    pipeline = object.__new__(Wan22I2VPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = StubTransformer(name="high", in_channels=8, out_channels=4)
    pipeline.transformer_2 = StubTransformer(name="low", in_channels=8, out_channels=4)
    pipeline.vae = StubVAE(z_dim=4)
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    pipeline.expand_timesteps = expand_timesteps
    pipeline.progress_bar = noop_progress_bar
    return pipeline


def test_i2v_preprocess_requires_image_and_resizes_to_480p_aspect() -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())
    request = SimpleNamespace(
        prompts=[{"prompt": "p", "multi_modal_data": {"image": Image.new("RGB", (320, 160), "red")}}],
        sampling_params=SimpleNamespace(height=None, width=None),
    )

    result = preprocess(request)
    prompt = result.prompts[0]

    assert result.sampling_params.height == 432
    assert result.sampling_params.width == 880
    assert prompt["multi_modal_data"]["image"].size == (880, 432)

    missing_image = SimpleNamespace(
        prompts=[{"prompt": "p", "multi_modal_data": {}}],
        sampling_params=SimpleNamespace(height=None, width=None),
    )
    with pytest.raises(ValueError, match="No image is provided"):
        preprocess(missing_image)


def test_i2v_diffuse_selects_stage_guidance_and_expands_timesteps() -> None:
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    latents = torch.zeros(1, 4, 2, 4, 4)
    condition = torch.ones_like(latents)
    first_frame_mask = torch.ones(1, 1, 2, 4, 4)
    first_frame_mask[:, :, 0] = 0
    timesteps = torch.tensor([900, 100])

    calls = []

    def fake_predict_noise_maybe_with_cfg(**kwargs):
        positive = kwargs["positive_kwargs"]
        calls.append(
            {
                "model": positive["current_model"].name,
                "scale": kwargs["true_cfg_scale"],
                "timestep_shape": tuple(positive["timestep"].shape),
                "timestep_values": positive["timestep"].clone(),
                "hidden_states": positive["hidden_states"].clone(),
            }
        )
        return torch.ones_like(latents)

    pipeline.predict_noise_maybe_with_cfg = fake_predict_noise_maybe_with_cfg  # type: ignore[method-assign]
    pipeline.scheduler_step_maybe_with_cfg = lambda noise, t, current, cfg: current + noise  # type: ignore[method-assign]

    result = pipeline.diffuse(
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=torch.zeros(1, 2, 3),
        negative_prompt_embeds=None,
        image_embeds=None,
        guidance_low=1.0,
        guidance_high=2.0,
        boundary_timestep=500.0,
        dtype=torch.float32,
        attention_kwargs={},
        condition=condition,
        first_frame_mask=first_frame_mask,
    )

    assert [call["model"] for call in calls] == ["high", "low"]
    assert [call["scale"] for call in calls] == [1.0, 2.0]
    assert calls[0]["timestep_shape"] == (1, 8)
    timestep_dtype = calls[0]["timestep_values"].dtype
    torch.testing.assert_close(calls[0]["timestep_values"][0, :4], torch.zeros(4, dtype=timestep_dtype))
    torch.testing.assert_close(calls[0]["timestep_values"][0, 4:], torch.full((4,), 900, dtype=timestep_dtype))
    torch.testing.assert_close(calls[0]["hidden_states"][:, :, 0], torch.ones(1, 4, 4, 4))
    torch.testing.assert_close(result, torch.full_like(latents, 2.0))


def test_i2v_prepare_latents_builds_expand_condition_and_first_frame_mask() -> None:
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    latents, condition, first_frame_mask = pipeline.prepare_latents(
        image=torch.zeros(1, 3, 16, 16),
        batch_size=1,
        num_channels_latents=4,
        height=16,
        width=16,
        num_frames=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator(device="cpu").manual_seed(0),
    )

    assert latents.shape == (1, 4, 2, 2, 2)
    assert condition.shape == (1, 4, 1, 2, 2)
    assert first_frame_mask.shape == (1, 1, 2, 2, 2)
    assert first_frame_mask[:, :, 0].sum() == 0
    assert first_frame_mask[:, :, 1].sum() == 4
