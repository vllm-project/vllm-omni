# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import create_autospec

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.models.kandinsky6.pipeline_kandinsky6 import LatentBundle
from vllm_omni.diffusion.models.kandinsky6.super_resolution import (
    DEFAULT_SR_MODEL,
    Kandinsky6SuperResolution,
    resolve_super_resolution,
    stage_generation_modules_on_cpu,
    validate_super_resolution_execution,
)
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("target,height,width", [("hd", 720, 1280), ("fhd", 1080, 1920)])
@pytest.mark.parametrize("explicit", [True, False])
def test_hd_fhd_resolves_at_generation_boundary(monkeypatch, target, height, width, explicit):
    from vllm_omni.diffusion.models.kandinsky6 import pipeline_kandinsky6 as module
    from vllm_omni.diffusion.models.kandinsky6.scheduling_kandinsky6 import KandinskyFlowMatchScheduler

    monkeypatch.setattr(module, "get_local_device", lambda: torch.device("cpu"))
    pipeline = module.Kandinsky6TI2VAPipeline(
        transformer=torch.nn.Identity(),
        vae=torch.nn.Identity(),
        text_encoder=torch.nn.Identity(),
        scheduler=KandinskyFlowMatchScheduler(),
    )
    sampling = OmniDiffusionSamplingParams(
        height=height if explicit else None,
        width=width if explicit else None,
        extra_args={} if explicit else {"target_resolution": target},
    )
    settings = resolve_super_resolution(sampling)
    assert settings is not None
    assert (settings.height, settings.width) == (height, width)
    assert settings.model == DEFAULT_SR_MODEL and settings.steps is None
    assert pipeline._resolve_generation_settings(sampling)[:2] == (480, 864)


@pytest.mark.parametrize(
    "extra",
    [
        {"target_resolution": "4k"},
        {"target_resolution": "hd", "sr_num_inference_steps": 0},
        {"target_resolution": "hd", "sr_tiles_batch_size": 0},
        {"target_resolution": "hd", "sr_min_overlap": 1},
        {"sr_model": DEFAULT_SR_MODEL},
        {"target_resolution": "fhd", "sr_model": ""},
    ],
)
def test_invalid_sr_options_fail_early(extra):
    with pytest.raises(ValueError):
        resolve_super_resolution(OmniDiffusionSamplingParams(extra_args=extra))


def test_conflicting_dimensions_and_latent_output():
    with pytest.raises(ValueError, match="conflict"):
        resolve_super_resolution(
            OmniDiffusionSamplingParams(height=720, width=1280, extra_args={"target_resolution": "fhd"})
        )
    with pytest.raises(ValueError, match="decoded video"):
        resolve_super_resolution(OmniDiffusionSamplingParams(height=1080, width=1920, output_type="latent"))
    assert resolve_super_resolution(OmniDiffusionSamplingParams(height=480, width=864)) is None


@pytest.mark.parametrize("world_size,hsdp,layerwise", [(2, False, False), (1, True, False), (1, False, True)])
def test_unsupported_execution_is_rejected_before_generation(world_size, hsdp, layerwise):
    config = OmniDiffusionConfig(
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=world_size, use_hsdp=hsdp, hsdp_shard_size=1),
        enable_layerwise_offload=layerwise,
    )
    with pytest.raises(ValueError, match="one GPU"):
        validate_super_resolution_execution(config)


def test_cpu_staging_restores_parameters_after_error(monkeypatch):
    from vllm_omni.diffusion.models.kandinsky6 import super_resolution as module

    monkeypatch.setattr(module.current_omni_platform, "empty_cache", lambda: None)
    model = torch.nn.Linear(2, 2)
    model.register_buffer("constant", torch.ones(2))
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(RuntimeError, match="refinement failed"):
        with stage_generation_modules_on_cpu([model]):
            raise RuntimeError("refinement failed")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_existing_image_example_leaves_resize_to_pipeline():
    from vllm_omni.model_extras import should_preserve_reference_image_size

    assert should_preserve_reference_image_size("Kandinsky6TI2VAPipeline", model="kandinskylab/test")


def test_api_full_hd_size_reaches_generation_boundary():
    from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest

    video = VideoGenerationRequest(prompt="A fox runs through snow", size="1920x1080").resolve_video_params()
    settings = resolve_super_resolution(OmniDiffusionSamplingParams(height=video.height, width=video.width))
    assert settings is not None and (settings.height, settings.width) == (1080, 1920)


@pytest.mark.parametrize("mode", ["t2va", "i2va"])
def test_step_decode_refines_video_and_keeps_generated_audio(monkeypatch, mode):
    from vllm_omni.diffusion.models.kandinsky6 import pipeline_kandinsky6 as module

    pipeline = module.Kandinsky6TI2VAPipeline.__new__(module.Kandinsky6TI2VAPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.vae = torch.nn.Identity()
    pipeline.audio_vae = torch.nn.Identity()
    pipeline.audio_sample_rate = 44100
    audio = [np.ones(100, dtype=np.int16)]
    refined = np.zeros((1, 5, 720, 1280, 3), dtype=np.uint8)
    calls = []

    def refine(video, sampling, settings):
        calls.append((video.shape, settings))
        return refined

    pipeline._super_resolution = create_autospec(Kandinsky6SuperResolution, instance=True)
    pipeline._super_resolution.refine.side_effect = refine
    monkeypatch.setattr(
        module, "postprocess_video", lambda *a, **kw: torch.zeros((1, 3, 5, 480, 864), dtype=torch.uint8)
    )
    monkeypatch.setattr(module, "postprocess_audio", lambda *a, **kw: audio)
    state = StepRequestState(
        request_id="sr-test",
        latents=torch.zeros(2, 1, 1, 2),
        sampling=OmniDiffusionSamplingParams(extra_args={"target_resolution": "hd"}),
        extra={
            "audio": None,
            "latent_frames": 2,
            "bundle": LatentBundle(video=None, audio=None, video_cu_seqlens=None, audio_cu_seqlens=None),
            "sample_audio": True,
            "audio_normalization": "normalize",
            "visual_cond_scheme": mode,
        },
    )
    result = pipeline.post_decode(state).output
    assert calls[0][0] == (1, 5, 480, 864, 3)
    assert result["video"] is refined
    assert result["audio"] is audio
    assert result["audio_sample_rate"] == 44100
