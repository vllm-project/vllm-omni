# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from tests.diffusion.models.wan2_2.conftest import StubScheduler, StubTransformer, StubVAE, noop_progress_bar
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_vace import (
    Wan22VACEPipeline,
    create_vace_transformer_from_config,
    get_wan22_vace_pre_process_func,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_vace_transformer import WanVACETransformer3DModel
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

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


def _make_vace_sampling(**overrides):
    values = {
        "height": 16,
        "width": 16,
        "num_frames": 5,
        "num_inference_steps": 1,
        "guidance_scale_provided": True,
        "guidance_scale": 1.0,
        "boundary_ratio": None,
        "generator": None,
        "seed": None,
        "num_outputs_per_prompt": 1,
        "max_sequence_length": 8,
        "latents": None,
        "output_type": "latent",
        "extra_args": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_vace_preprocess_collects_reference_video_and_mask_inputs() -> None:
    preprocess = get_wan22_vace_pre_process_func(SimpleNamespace())
    ref = Image.new("RGB", (320, 160), "green")
    frame = Image.new("RGB", (64, 64), "black")
    mask = Image.new("L", (64, 64), 255)
    request = SimpleNamespace(
        prompt={
            "prompt": "p",
            "multi_modal_data": {
                "image": ref,
                "video": [frame],
                "mask": mask,
            },
        },
        sampling_params=SimpleNamespace(height=None, width=None),
    )

    result = preprocess(request)
    additional_info = result.prompt["additional_information"]

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


def test_vace_transformer_layerwise_offloads_conditioning_blocks() -> None:
    assert WanVACETransformer3DModel._layerwise_offload_blocks_attrs == ["vace_blocks", "blocks"]


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


def test_vace_forward_batches_random_inputs_and_splits_outputs(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_vace.current_omni_platform",
        SimpleNamespace(is_available=lambda: False),
    )
    pipeline = _make_vace_pipeline()
    pipeline.transformer.vace_patch_embedding = None
    pipeline.transformer_2 = None
    pipeline.scheduler = StubScheduler([9])
    pipeline.od_config = SimpleNamespace(flow_shift=3.0)
    pipeline._sample_solver = "unipc"
    pipeline._flow_shift = 3.0
    pipeline.boundary_ratio = None
    pipeline._guidance_scale = None
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.check_inputs = lambda **kwargs: None
    prepare_call = {}

    def _fake_encode_prompt(**kwargs):
        batch_size = len(kwargs["prompt"])
        n = batch_size * kwargs["num_videos_per_prompt"]
        return torch.zeros(n, 2, 3), None

    def _fake_prepare_latents(**kwargs):
        prepare_call.update(kwargs)
        return kwargs["latents"]

    pipeline.encode_prompt = _fake_encode_prompt  # type: ignore[method-assign]
    pipeline.prepare_latents = _fake_prepare_latents  # type: ignore[method-assign]
    pipeline.diffuse = lambda **kwargs: kwargs["latents"]  # type: ignore[method-assign]
    gen_a = torch.Generator(device="cpu").manual_seed(1)
    gen_b = torch.Generator(device="cpu").manual_seed(2)
    latents_a = torch.zeros(2, 4, 2, 2, 2)
    latents_b = torch.ones(2, 4, 2, 2, 2)
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={"prompt": "first"},
                sampling_params=_make_vace_sampling(
                    generator=gen_a,
                    latents=latents_a,
                    num_outputs_per_prompt=2,
                ),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={"prompt": "second"},
                sampling_params=_make_vace_sampling(
                    generator=gen_b,
                    latents=latents_b,
                    num_outputs_per_prompt=2,
                ),
            ),
        ]
    )

    outputs = pipeline.forward(batch)

    assert prepare_call["batch_size"] == 4
    assert prepare_call["generator"] == [gen_a, gen_a, gen_b, gen_b]
    torch.testing.assert_close(prepare_call["latents"], torch.cat([latents_a, latents_b]))
    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0].output, latents_a)
    torch.testing.assert_close(outputs[1].output, latents_b)


def test_vace_forward_rejects_mismatched_condition_shapes(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_vace.current_omni_platform",
        SimpleNamespace(is_available=lambda: False),
    )
    pipeline = _make_vace_pipeline()
    pipeline.transformer.vace_patch_embedding = object()
    pipeline.transformer_2 = None
    pipeline.scheduler = StubScheduler([9])
    pipeline.od_config = SimpleNamespace(flow_shift=3.0)
    pipeline._sample_solver = "unipc"
    pipeline._flow_shift = 3.0
    pipeline.boundary_ratio = None
    pipeline._guidance_scale = None
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.check_inputs = lambda **kwargs: None
    pipeline.encode_prompt = lambda **kwargs: (torch.zeros(2, 2, 3), None)  # type: ignore[method-assign]

    def _fake_preprocess(video, **kwargs):
        del kwargs
        mask = torch.ones_like(video)
        return video, mask, [[]]

    pipeline.preprocess_conditions = _fake_preprocess  # type: ignore[method-assign]
    pipeline.prepare_video_latents = (  # type: ignore[method-assign]
        lambda video, mask, refs, generator, device: torch.zeros(1, 8, video.shape[2], 2, 2)
    )
    pipeline.prepare_masks = lambda mask, refs: torch.zeros(1, 4, mask.shape[2], 2, 2)  # type: ignore[method-assign]
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={
                    "prompt": "first",
                    "additional_information": {"source_video": torch.zeros(1, 3, 5, 16, 16)},
                },
                sampling_params=_make_vace_sampling(),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={
                    "prompt": "second",
                    "additional_information": {"source_video": torch.zeros(1, 3, 3, 16, 16)},
                },
                sampling_params=_make_vace_sampling(),
            ),
        ]
    )

    with pytest.raises(ValueError, match="VACE condition"):
        pipeline.forward(batch)


def test_vace_forward_rejects_mismatched_reference_image_counts(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_vace.current_omni_platform",
        SimpleNamespace(is_available=lambda: False),
    )
    pipeline = _make_vace_pipeline()
    pipeline.transformer.vace_patch_embedding = object()
    pipeline.transformer_2 = None
    pipeline.scheduler = StubScheduler([9])
    pipeline.od_config = SimpleNamespace(flow_shift=3.0)
    pipeline._sample_solver = "unipc"
    pipeline._flow_shift = 3.0
    pipeline.boundary_ratio = None
    pipeline._guidance_scale = None
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.check_inputs = lambda **kwargs: None
    pipeline.encode_prompt = lambda **kwargs: (torch.zeros(2, 2, 3), None)  # type: ignore[method-assign]

    def _fake_preprocess(video, mask, reference_images, **kwargs):
        del kwargs
        return video, mask, [reference_images]

    pipeline.preprocess_conditions = _fake_preprocess  # type: ignore[method-assign]
    pipeline.prepare_video_latents = (  # type: ignore[method-assign]
        lambda video, mask, refs, generator, device: torch.zeros(1, 8, 5, 2, 2)
    )
    pipeline.prepare_masks = lambda mask, refs: torch.zeros(1, 4, 5, 2, 2)  # type: ignore[method-assign]
    video = torch.zeros(1, 3, 5, 16, 16)
    mask = torch.ones_like(video)
    reference = torch.zeros(3, 16, 16)
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={
                    "prompt": "first",
                    "additional_information": {
                        "source_video": video,
                        "mask": mask,
                        "reference_images": [reference],
                    },
                },
                sampling_params=_make_vace_sampling(),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={
                    "prompt": "second",
                    "additional_information": {
                        "source_video": video,
                        "mask": mask,
                        "reference_images": [reference, reference],
                    },
                },
                sampling_params=_make_vace_sampling(),
            ),
        ]
    )

    with pytest.raises(ValueError, match="same number of reference images"):
        pipeline.forward(batch)
