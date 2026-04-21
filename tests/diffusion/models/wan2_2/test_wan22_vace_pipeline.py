# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from tests.diffusion.models.wan2_2.conftest import StubTransformer, StubVAE, noop_progress_bar
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_vace import (
    Wan22VACEPipeline,
    create_vace_transformer_from_config,
    get_wan22_vace_pre_process_func,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_vace_pipeline() -> Wan22VACEPipeline:
    pipeline = object.__new__(Wan22VACEPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = StubTransformer(in_channels=4, out_channels=4)
    pipeline.transformer_config = pipeline.transformer.config
    pipeline.vae = StubVAE(z_dim=4)
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    pipeline.progress_bar = noop_progress_bar
    return pipeline


def test_vace_preprocess_collects_reference_video_and_mask_inputs() -> None:
    preprocess = get_wan22_vace_pre_process_func(SimpleNamespace())
    ref = Image.new("RGB", (320, 160), "green")
    frame = Image.new("RGB", (64, 64), "black")
    mask = Image.new("L", (64, 64), 255)
    request = SimpleNamespace(
        prompts=[
            {
                "prompt": "p",
                "multi_modal_data": {
                    "image": ref,
                    "video": [frame],
                    "mask": mask,
                },
            }
        ],
        sampling_params=SimpleNamespace(height=None, width=None),
    )

    result = preprocess(request)
    additional_info = result.prompts[0]["additional_information"]

    assert result.sampling_params.height == 432
    assert result.sampling_params.width == 880
    assert additional_info["reference_images"] == [ref]
    assert additional_info["source_video"] == [frame]
    assert additional_info["mask"] == [mask]


def test_create_vace_transformer_from_config_maps_vace_specific_keys(monkeypatch) -> None:
    captured = {}

    class FakeVACETransformer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_vace.WanVACETransformer3DModel",
        FakeVACETransformer,
    )

    transformer = create_vace_transformer_from_config(
        {
            "patch_size": [1, 2, 2],
            "in_channels": 96,
            "out_channels": 16,
            "vace_layers": [0, 1, 2],
            "vace_in_channels": 132,
            "unknown": "ignored",
        }
    )

    assert isinstance(transformer, FakeVACETransformer)
    assert captured == {
        "patch_size": (1, 2, 2),
        "in_channels": 96,
        "out_channels": 16,
        "vace_layers": [0, 1, 2],
        "vace_in_channels": 132,
    }


def test_vace_prepare_masks_encodes_spatial_stride_and_reference_padding() -> None:
    pipeline = _make_vace_pipeline()
    mask = torch.ones(1, 3, 5, 16, 16)
    reference_images = [[torch.zeros(3, 16, 16), torch.zeros(3, 16, 16)]]

    encoded = pipeline.prepare_masks(mask, reference_images)

    assert encoded.shape == (1, 64, 4, 2, 2)
    torch.testing.assert_close(encoded[:, :, :2], torch.zeros(1, 64, 2, 2, 2))
    torch.testing.assert_close(encoded[:, :, 2:], torch.ones(1, 64, 2, 2, 2))


def test_vace_diffuse_passes_context_and_scale_to_cfg_branches() -> None:
    pipeline = _make_vace_pipeline()
    latents = torch.zeros(1, 4, 1, 2, 2)
    vace_context = torch.ones(1, 12, 1, 2, 2)
    calls = []

    def fake_predict_noise_maybe_with_cfg(**kwargs):
        calls.append(kwargs)
        return torch.ones_like(latents)

    pipeline.predict_noise_maybe_with_cfg = fake_predict_noise_maybe_with_cfg  # type: ignore[method-assign]
    pipeline.scheduler_step_maybe_with_cfg = lambda noise, t, current, cfg: current + noise  # type: ignore[method-assign]

    result = pipeline.diffuse(
        latents=latents,
        timesteps=torch.tensor([5]),
        prompt_embeds=torch.zeros(1, 2, 3),
        negative_prompt_embeds=torch.zeros(1, 2, 3),
        guidance_scale=4.0,
        dtype=torch.float32,
        attention_kwargs={},
        vace_context=vace_context,
        vace_context_scale=0.75,
    )

    assert calls[0]["do_true_cfg"] is True
    assert calls[0]["true_cfg_scale"] == 4.0
    assert calls[0]["positive_kwargs"]["vace_context"] is vace_context
    assert calls[0]["negative_kwargs"]["vace_context_scale"] == 0.75
    torch.testing.assert_close(result, torch.ones_like(latents))
