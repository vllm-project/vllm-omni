# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.kandinsky6_sr.config import (
    default_steps,
    output_plan,
    transformer_config,
    upscaler_config,
    vae_config,
)
from vllm_omni.diffusion.models.kandinsky6_sr.pipeline_kandinsky6_sr import (
    Kandinsky6SRRefiner,
    normalize_frames,
    prepare_pixels,
)
from vllm_omni.diffusion.models.kandinsky6_sr.pipeline_output import Kandinsky6SRPipelineOutput
from vllm_omni.diffusion.models.kandinsky6_sr.scheduling_piflow import PiflowScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("target,height,width", [("hd", 720, 1280), ("fhd", 1080, 1920)])
@pytest.mark.parametrize("scale", [2, 2.25, 4])
def test_exact_delivery_geometry(target, height, width, scale):
    plan = output_plan(480, 864, None, None, {"target_resolution": target, "resolution_scale": scale})
    assert (plan.height, plan.width) == (height, width)
    assert plan.prepared_height % 16 == plan.prepared_width % 16 == 0
    assert plan.prepared_height * plan.tiling_scale >= height
    assert plan.prepared_width * plan.tiling_scale >= width
    assert plan.tiling_scale == (4 if scale == 4 else 2)


def test_native_scale_defaults():
    plan = output_plan(480, 864, None, None, {})
    assert (plan.height, plan.width) == (1080, 1944)


def test_three_frame_channels_last_batch():
    frames = torch.zeros((1, 3, 8, 16, 3), dtype=torch.uint8)
    assert normalize_frames(frames).shape == (3, 8, 16, 3)


def test_tiny_upscaler_bank_executes_both_routes():
    from vllm_omni.diffusion.models.kandinsky6_sr.latent_upscaler_kandinsky6_sr import Kandinsky6SRLatentUpscalerBank

    bank = Kandinsky6SRLatentUpscalerBank(
        in_channels=2,
        stage_channels=(16, 8, 4),
        num_pre_blocks=1,
        num_mid_blocks=1,
        num_post_blocks=1,
        num_x2_adapter_blocks=1,
    )
    latents = torch.randn(1, 2, 3, 4, 4)
    for scale in (2, 4):
        result = bank(latents, scale=scale).sample
        assert result.shape == (1, 2, 3, 4 * scale, 4 * scale)
        assert torch.isfinite(result).all()
    with pytest.raises(ValueError, match="No latent upscaler"):
        bank(latents, scale=3)


def test_tiling_covers_output_with_positive_weights():
    from vllm_omni.diffusion.models.kandinsky6_sr.sr_inference import _hann_window_2d, _tile_positions

    tops = _tile_positions(544, 256, 0.2, 16)
    lefts = _tile_positions(960, 384, 0.2, 16)
    weights = torch.zeros(1088, 1920)
    window = _hann_window_2d(512, 768, torch.device("cpu"))
    for top in tops:
        for left in lefts:
            assert top % 16 == left % 16 == 0
            weights[top * 2 : top * 2 + 512, left * 2 : left * 2 + 768] += window
    assert (weights > 0).all()


@pytest.mark.parametrize(
    "height,width,extra",
    [
        (None, 1280, {}),
        (721, 1280, {}),
        (0, 1280, {}),
        (720, 1280, {"target_resolution": "fhd"}),
        (None, None, {"target_resolution": "unknown"}),
        (None, None, {"resolution_scale": 3}),
    ],
)
def test_invalid_geometry(height, width, extra):
    with pytest.raises(ValueError):
        output_plan(480, 864, height, width, extra)


def test_pixel_preparation_crop_and_padding():
    frames = torch.full((4, 18, 32, 3), 128, dtype=torch.uint8)
    plan = output_plan(18, 32, 720, 1280, {})
    pixels = prepare_pixels(frames, plan, torch.device("cpu"))
    assert pixels.shape == (1, 3, 5, plan.prepared_height, plan.prepared_width)
    torch.testing.assert_close(pixels, torch.full_like(pixels, 128 / 255))


@pytest.mark.parametrize("layout", ["thwc", "tchw", "bcthw"])
def test_normalize_video_layouts(layout):
    expected = torch.full((5, 8, 16, 3), 255, dtype=torch.uint8)
    frames = expected.float() / 255
    if layout == "tchw":
        frames = frames.permute(0, 3, 1, 2)
    elif layout == "bcthw":
        frames = frames.permute(3, 0, 1, 2).unsqueeze(0)
    torch.testing.assert_close(normalize_frames(frames), expected)


@pytest.mark.parametrize(
    "frames",
    [np.zeros((0, 4, 4, 3)), np.full((1, 4, 4, 3), np.nan)],
)
def test_invalid_video_frames(frames):
    with pytest.raises(ValueError):
        normalize_frames(frames)


def test_legacy_config_normalization():
    dit = transformer_config(
        {
            "out_visual_dim": 640,
            "sr_params": {"scale_factor": {"512": [1, 2, 2]}, "visual_size": [512]},
            "attention_params": {"512": {"type": "nabla", "P": 0.8, "wT": 11, "wH": 7, "wW": 7}},
        }
    )
    assert dit == {"out_visual_dim": 640, "scale_factor": [1, 2, 2], "nabla_threshold": 0.8, "nabla_window": (11, 7, 7)}
    vae = vae_config(
        dict(
            ch=128,
            ch_mult=[0.125, 1, 2, 4, 8],
            decoder_ch=256,
            decoder_ch_mult=[0.0625, 1, 2, 4, 8],
            in_channels=3,
            out_channels=3,
            z_channels=64,
            num_res_blocks=2,
            temporal_compress_times=4,
            temporal_compress_start_level=1,
            scaling_factor=0.910344,
            norm_type="rms_norm",
            fix_pxs=True,
            downsample_version=2,
        )
    )
    assert vae["encoder_block_out_channels"] == (16, 128, 256, 512, 1024)
    assert vae["decoder_block_out_channels"] == (16, 256, 512, 1024, 2048)


def test_legacy_upscaler_tensor_order_overrides_config_order():
    model = dict(
        architecture="multi_scale",
        upsample_mode="pxs_v2",
        stage_channels=[16, 8, 4],
        in_channels=2,
        num_pre_blocks=1,
        num_mid_blocks=1,
        num_post_blocks=1,
    )
    config = {
        "models": [{"model": model, "target_scale": "4x"}, {"model": model, "target_scale": "2x"}],
        "scaling_factor": 0.910344,
    }
    keys = ["_models.0.x2_branch.adapter.0.conv1.weight", "_models.1.input_proj.0.weight"]
    assert upscaler_config(config, keys)["scales"] == (2, 4)
    with pytest.raises(ValueError, match="branches"):
        upscaler_config(config, ["_models.0.input_proj.0.weight"])


def test_checkpoint_scheduler_defaults_and_zero_policy():
    assert default_steps({"_class_name": "FlowMatchEulerDiscreteScheduler"}) == 4
    assert default_steps({"_class_name": "PiflowScheduler", "nfe": 2}) == 2
    with pytest.raises(ValueError):
        default_steps({"_class_name": "unknown"})
    scheduler = PiflowScheduler(shift=3.5)
    scheduler.set_timesteps(2)
    initial = torch.ones((1, 2, 3, 4, 2))
    sample = initial.clone()
    for timestep in scheduler.timesteps:
        sample = scheduler.step(torch.zeros((*sample.shape[:-1], 20)), timestep, sample).prev_sample
    torch.testing.assert_close(sample, initial)


def test_sr_options_belong_to_generation_pipeline():
    from vllm_omni.diffusion.registry import _DIFFUSION_MODELS
    from vllm_omni.model_extras import get_extra_body_params

    assert "Kandinsky6SRPipeline" not in _DIFFUSION_MODELS
    assert {"target_resolution", "sr_model", "sr_num_inference_steps"} <= get_extra_body_params(
        "Kandinsky6TI2VAPipeline"
    )


def test_internal_refiner_crops_padding():
    refiner = Kandinsky6SRRefiner.__new__(Kandinsky6SRRefiner)
    torch.nn.Module.__init__(refiner)
    refiner.device = torch.device("cpu")
    refiner.default_num_inference_steps = 2
    refiner.lq_noise_scale = 0.7
    calls = []

    def inference(**kwargs):
        calls.append(kwargs)
        _, frames, channels, height, width = kwargs["video"].shape
        assert channels == 3
        return Kandinsky6SRPipelineOutput(frames=np.full((1, frames, height * 2, width * 2, 3), 0.5, dtype=np.float32))

    refiner._inference = inference
    frames = torch.zeros((1, 4, 18, 32, 3), dtype=torch.uint8)
    result = refiner.refine(frames, height=720, width=1280, seed=42)
    assert calls[0]["num_inference_steps"] == 2
    assert result.shape == (1, 4, 720, 1280, 3)
    assert result.dtype == np.uint8
    refiner.refine(frames, height=720, width=1280, seed=42, num_inference_steps=3)
    assert calls[-1]["num_inference_steps"] == 3
