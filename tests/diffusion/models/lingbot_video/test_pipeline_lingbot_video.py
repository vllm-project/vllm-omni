# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import weakref
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_pipeline():
    from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline

    pipeline = object.__new__(LingBotVideoPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.default_negative_prompt = "default negative"
    pipeline.default_image_negative_prompt = "default image negative"
    pipeline.img_prompt_template = "<image>"
    pipeline.od_config = SimpleNamespace(flow_shift=None)
    pipeline.enable_diffusion_pipeline_profiler = False
    return pipeline


def _make_request_batch(prompt, **sampling_overrides):
    sampling = OmniDiffusionSamplingParams(**sampling_overrides)
    request = OmniDiffusionRequest(prompt=prompt, sampling_params=sampling, request_id="lingbot-test")
    return DiffusionRequestBatch([request])


def test_lingbot_video_pipeline_import_and_registry():
    from vllm_omni.diffusion.models.lingbot_video import (
        LingBotVideoPipeline,
        LingBotVideoTransformer3DModel,
        get_lingbot_video_post_process_func,
        get_lingbot_video_pre_process_func,
    )
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
    )

    assert LingBotVideoPipeline is not None
    assert LingBotVideoTransformer3DModel is not None
    assert get_lingbot_video_pre_process_func is not None
    assert get_lingbot_video_post_process_func is not None
    assert _DIFFUSION_MODELS["LingBotVideoPipeline"] == (
        "lingbot_video",
        "pipeline_lingbot_video",
        "LingBotVideoPipeline",
    )
    assert _DIFFUSION_PRE_PROCESS_FUNCS["LingBotVideoPipeline"] == "get_lingbot_video_pre_process_func"
    assert _DIFFUSION_POST_PROCESS_FUNCS["LingBotVideoPipeline"] == "get_lingbot_video_post_process_func"


def test_component_discovery_declarations():
    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoPipeline

    assert LingBotVideoPipeline._dit_modules == ["transformer"]
    assert LingBotVideoPipeline._encoder_modules == ["text_encoder"]
    assert LingBotVideoPipeline._vae_modules == ["vae"]
    assert LingBotVideoPipeline.supports_step_execution is False
    assert LingBotVideoPipeline.max_outputs_per_prompt == 1


def test_extra_body_params_include_video_flow_shift_alias():
    from vllm_omni.model_extras import get_extra_body_params

    params = get_extra_body_params("LingBotVideoPipeline")
    assert "shift" in params
    assert "flow_shift" in params
    assert "batch_cfg" in params


def test_preprocess_preserves_geometry_and_converts_rgb(tmp_path):
    from vllm_omni.diffusion.models.lingbot_video import (
        get_lingbot_video_pre_process_func,
    )

    image_path = tmp_path / "reference.png"
    Image.new("L", (48, 32), color=127).save(image_path)
    request = OmniDiffusionRequest(
        prompt={
            "prompt": "a robot walks",
            "modalities": ["video"],
            "multi_modal_data": {"image": str(image_path)},
        },
        sampling_params=OmniDiffusionSamplingParams(),
        request_id="lingbot-preprocess-test",
    )

    result = get_lingbot_video_pre_process_func(SimpleNamespace())(request)

    image = result.prompt["multi_modal_data"]["image"]
    assert image.mode == "RGB"
    assert image.size == (48, 32)


def test_preprocess_marks_internal_image_warmup_as_video():
    from vllm_omni.diffusion.models.lingbot_video import (
        get_lingbot_video_pre_process_func,
    )

    request = OmniDiffusionRequest(
        prompt={
            "prompt": "dummy run",
            "multi_modal_data": {"image": Image.new("RGB", (32, 32))},
        },
        sampling_params=OmniDiffusionSamplingParams(),
        request_id=DUMMY_DIFFUSION_REQUEST_ID,
    )

    result = get_lingbot_video_pre_process_func(SimpleNamespace())(request)

    assert result.prompt["modalities"] == ["video"]


def test_check_inputs_accepts_t2v_shapes_and_rejects_invalid_values():
    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoPipeline

    LingBotVideoPipeline.check_inputs(height=192, width=320, num_frames=1)
    LingBotVideoPipeline.check_inputs(height=192, width=320, num_frames=9)

    with pytest.raises(ValueError, match="num_frames"):
        LingBotVideoPipeline.check_inputs(height=192, width=320, num_frames=8)
    with pytest.raises(ValueError, match="height"):
        LingBotVideoPipeline.check_inputs(height=190, width=320, num_frames=9)


def test_postprocess_keeps_torch_by_default_and_converts_np_when_requested():
    from vllm_omni.diffusion.models.lingbot_video import get_lingbot_video_post_process_func

    frames = torch.ones(2, 4, 4, 3)
    post = get_lingbot_video_post_process_func(SimpleNamespace())

    assert post(frames, SimpleNamespace(output_type="pt")) is frames
    array = post(frames, SimpleNamespace(output_type="np"))
    assert isinstance(array, np.ndarray)
    assert array.shape == (2, 4, 4, 3)
    video = post({"video": frames}, SimpleNamespace(output_type="pt"))
    assert set(video) == {"video"}
    assert video["video"] is frames
    image = post(
        {"image": torch.ones(4, 4, 3)},
        SimpleNamespace(output_type="pt"),
    )
    assert set(image) == {"image"}
    assert isinstance(image["image"], np.ndarray)
    assert image["image"].shape == (4, 4, 3)
    latent = torch.ones(1, 4, 1, 2, 2)
    latent_output = post({"image": latent}, SimpleNamespace(output_type="latent"))
    assert set(latent_output) == {"image"}
    assert latent_output["image"] is latent


def test_postprocess_preserves_image_contract_for_serving():
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.models.lingbot_video import get_lingbot_video_post_process_func
    from vllm_omni.diffusion.output_formatter import (
        format_diffusion_outputs,
        normalize_diffusion_postprocess_output,
    )
    from vllm_omni.entrypoints.openai.images.helpers import _extract_images_from_result

    sampling = OmniDiffusionSamplingParams(
        num_inference_steps=1,
        output_type="pt",
        resolution=64,
    )
    request = OmniDiffusionRequest(
        prompt={"prompt": "a robot", "modalities": ["image"]},
        sampling_params=sampling,
        request_id="lingbot-image-contract-test",
    )
    raw_output = {"image": torch.ones(8, 12, 3)}
    processed = get_lingbot_video_post_process_func(SimpleNamespace())(raw_output, sampling)
    normalized = normalize_diffusion_postprocess_output(processed)

    assert normalized.primary_key == "image"
    [result] = format_diffusion_outputs(
        request=request,
        od_config=SimpleNamespace(model_class_name="LingBotVideoPipeline"),
        diffusion_output=DiffusionOutput(output=raw_output),
        output_data=raw_output,
        postprocess_output=normalized,
    )

    assert len(result.images) == 1
    assert isinstance(result.images[0], np.ndarray)
    endpoint_images = _extract_images_from_result(result)
    assert len(endpoint_images) == 1
    assert endpoint_images[0].size == (12, 8)


def test_forward_resolves_t2v_sampling_and_flow_shift_alias():
    pipeline = _make_pipeline()
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return torch.zeros(9, 192, 320, 3)

    pipeline._generate = fake_generate
    req = _make_request_batch(
        {
            "prompt": "a robot arm",
            "negative_prompt": "bad hands",
            "modalities": ["video"],
        },
        height=192,
        width=320,
        num_frames=9,
        num_inference_steps=2,
        guidance_scale=3.0,
        seed=42,
        output_type="pt",
        extra_args={"flow_shift": 4.0, "batch_cfg": True},
    )

    out = pipeline.forward(req)

    assert torch.equal(out.output["video"], torch.zeros(9, 192, 320, 3))
    assert len(calls) == 1
    call = calls[0]
    assert call["prompt"] == "a robot arm"
    assert call["mode"].value == "t2v"
    assert call["input_image"] is None
    assert call["negative_prompt"] == "bad hands"
    assert call["height"] == 192
    assert call["width"] == 320
    assert call["num_frames"] == 9
    assert call["num_inference_steps"] == 2
    assert call["guidance_scale"] == 3.0
    assert call["shift"] == 4.0
    assert call["batch_cfg"] is True
    assert call["output_type"] == "pt"


def test_forward_uses_lingbot_guidance_default_when_omitted():
    pipeline = _make_pipeline()
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return torch.zeros(1)

    pipeline._generate = fake_generate
    req = _make_request_batch(
        "a robot arm",
        height=192,
        width=320,
        num_frames=81,
        num_inference_steps=2,
        seed=42,
    )

    pipeline.forward(req)

    assert calls[0]["num_frames"] == 81
    assert calls[0]["guidance_scale"] == 6.0


def test_forward_prefers_shift_over_flow_shift_and_defaults_negative_prompt():
    pipeline = _make_pipeline()
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return torch.zeros(5, 192, 192, 3)

    pipeline._generate = fake_generate
    req = _make_request_batch(
        "a robot arm",
        height=192,
        width=192,
        num_frames=5,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=42,
        extra_args={"shift": 5.0, "flow_shift": 4.0},
    )

    pipeline.forward(req)

    assert calls[0]["shift"] == 5.0
    assert calls[0]["negative_prompt"] == "default negative"


@pytest.mark.parametrize(
    ("prompt", "num_frames", "expected_mode", "output_key", "default_negative"),
    [
        (
            {"prompt": "a still robot", "modalities": ["image"]},
            1,
            "t2i",
            "image",
            "default image negative",
        ),
        (
            {
                "prompt": "a moving robot",
                "modalities": ["video"],
                "multi_modal_data": {"image": Image.new("RGB", (48, 32), color="blue")},
            },
            9,
            "ti2v",
            "video",
            "default negative",
        ),
    ],
)
def test_forward_dispatches_generation_mode_and_output(
    prompt,
    num_frames,
    expected_mode,
    output_key,
    default_negative,
):
    pipeline = _make_pipeline()
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return torch.zeros(192, 320, 3) if output_key == "image" else torch.zeros(num_frames, 192, 320, 3)

    pipeline._generate = fake_generate
    output = pipeline.forward(
        _make_request_batch(
            prompt,
            height=192,
            width=320,
            num_frames=num_frames,
            num_inference_steps=2,
            guidance_scale=3.0,
            seed=42,
            output_type="pt",
        )
    )

    assert set(output.output) == {output_key}
    assert calls[0]["mode"].value == expected_mode
    assert calls[0]["negative_prompt"] == default_negative
    if expected_mode == "ti2v":
        assert isinstance(calls[0]["input_image"], Image.Image)


def test_batched_cfg_pads_image_conditioned_embeddings_and_masks():
    from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import (
        _batch_cfg_prompt_inputs,
    )

    positive = torch.ones(1, 7, 4)
    positive_mask = torch.ones(1, 7, dtype=torch.long)
    negative = torch.full((1, 9, 4), 2.0)
    negative_mask = torch.ones(1, 9, dtype=torch.long)

    embeds, mask = _batch_cfg_prompt_inputs(
        positive,
        positive_mask,
        negative,
        negative_mask,
    )

    assert embeds.shape == (2, 9, 4)
    assert mask.shape == (2, 9)
    assert torch.equal(mask[0], torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0]))
    assert torch.equal(mask[1], torch.ones(9, dtype=torch.long))


@dataclass(eq=False)
class _PreparedInputObservation:
    latent_shape: torch.Size
    prompt_embeds: torch.Tensor
    attention_mask: torch.Tensor | None


class _RecordingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(in_channels=1)
        self.latent_inputs = []
        self.prepared_metadata = []
        self.forward_metadata = []

    def prepare_metadata(self, latents, prompt_embeds, encoder_attention_mask=None):
        metadata = _PreparedInputObservation(
            latent_shape=latents.shape,
            prompt_embeds=prompt_embeds,
            attention_mask=encoder_attention_mask,
        )
        self.prepared_metadata.append(metadata)
        return metadata

    def forward(
        self,
        latents,
        timestep,
        prompt_embeds,
        *,
        encoder_attention_mask,
        return_dict,
        metadata_cache=None,
    ):
        del timestep, return_dict
        assert metadata_cache is not None
        if metadata_cache.metadata is None:
            metadata_cache.metadata = self.prepare_metadata(latents, prompt_embeds, encoder_attention_mask)
        prepared_metadata = metadata_cache.metadata
        assert latents.shape == prepared_metadata.latent_shape
        assert prompt_embeds is prepared_metadata.prompt_embeds
        assert encoder_attention_mask is prepared_metadata.attention_mask
        self.forward_metadata.append(prepared_metadata)
        self.latent_inputs.append(latents.detach().clone())
        return (torch.zeros_like(latents),)


class _CorruptingScheduler:
    sigma_max = 1.0
    sigma_min = 0.0

    def set_timesteps(self, num_inference_steps, *, device, shift, **kwargs):
        del num_inference_steps, shift, kwargs
        self.timesteps = torch.tensor([1000.0, 500.0], device=device)

    def step(
        self,
        noise_pred,
        timestep,
        latents,
        *,
        return_dict,
        generator,
    ):
        del noise_pred, timestep, return_dict, generator
        return (latents + 7.0,)


@pytest.mark.parametrize(
    ("guidance_scale", "batch_cfg", "expected_prompts", "expected_transformer_calls"),
    [
        pytest.param(2.0, False, ["positive", "negative"], 4, id="sequential-cfg"),
        pytest.param(2.0, True, ["positive", "negative"], 2, id="batched-cfg"),
        pytest.param(1.0, False, ["positive"], 2, id="cfg-disabled"),
    ],
)
def test_ti2v_reinjects_clean_prefix_across_cfg_modes(
    mocker,
    guidance_scale,
    batch_cfg,
    expected_prompts,
    expected_transformer_calls,
):
    from vllm_omni.diffusion.models.lingbot_video import (
        LingBotGenerationMode,
        LingBotImageCondition,
    )
    from vllm_omni.diffusion.models.lingbot_video import pipeline_lingbot_video as module

    batch_inputs = mocker.spy(module, "_batch_cfg_prompt_inputs")
    pipeline = _make_pipeline()
    pipeline.transformer = _RecordingTransformer()
    pipeline.scheduler = _CorruptingScheduler()
    pipeline.progress_bar = lambda timesteps: timesteps
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(1, 1, 3, 1, 1)
    vlm_image = Image.new("RGB", (32, 32), color="red")
    condition = LingBotImageCondition(
        vlm_image=vlm_image,
        clean_latent=torch.full((1, 1, 1, 1, 1), 5.0),
    )
    pipeline.prepare_ti2v_image_condition = lambda *args, **kwargs: condition
    encode_calls = []

    def fake_encode(prompt, *, images, device):
        del device
        encode_calls.append((prompt, images))
        return torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.long)

    pipeline.encode_prompt = fake_encode

    latents = pipeline._generate(
        prompt="positive",
        mode=LingBotGenerationMode.TI2V,
        input_image=vlm_image,
        negative_prompt="negative",
        height=16,
        width=16,
        num_frames=9,
        num_inference_steps=2,
        guidance_scale=guidance_scale,
        shift=3.0,
        output_type="latent",
        batch_cfg=batch_cfg,
    )

    assert [call[0] for call in encode_calls] == expected_prompts
    assert all(call[1] == [vlm_image] for call in encode_calls)
    assert len(pipeline.transformer.latent_inputs) == expected_transformer_calls
    expected_preparations = 2 if guidance_scale > 1.0 and not batch_cfg else 1
    assert len(pipeline.transformer.prepared_metadata) == expected_preparations
    for metadata in pipeline.transformer.prepared_metadata:
        assert sum(item is metadata for item in pipeline.transformer.forward_metadata) == 2
    assert batch_inputs.call_count == int(guidance_scale > 1.0 and batch_cfg)
    for item in pipeline.transformer.latent_inputs:
        expected_prefix = condition.clean_latent.expand(item.shape[0], -1, -1, -1, -1)
        assert torch.equal(item[:, :, :1], expected_prefix)
    assert torch.equal(latents[:, :, :1], condition.clean_latent)
    assert torch.equal(latents[:, :, 1:], torch.full((1, 1, 2, 1, 1), 14.0))


@pytest.mark.parametrize("rank", [0, 1], ids=["positive-rank", "negative-rank"])
def test_cfg_parallel_prepares_only_local_branch_once(monkeypatch, rank):
    from vllm_omni.diffusion.models.lingbot_video import LingBotGenerationMode
    from vllm_omni.diffusion.models.lingbot_video import pipeline_lingbot_video as module

    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_rank", lambda group: rank)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(module.dist, "get_global_rank", lambda group, group_rank: group_rank)
    monkeypatch.setattr(module.dist, "broadcast", lambda tensor, **kwargs: tensor.zero_())
    monkeypatch.setattr(module.dist, "barrier", lambda **kwargs: None)
    pipeline = _make_pipeline()
    pipeline.transformer = _RecordingTransformer()
    pipeline.scheduler = _CorruptingScheduler()
    pipeline.progress_bar = lambda timesteps: timesteps
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(1, 1, 3, 1, 1)
    encoded_prompts = []

    def fake_encode(prompt, **kwargs):
        encoded_prompts.append(prompt)
        length = 2 if prompt == "positive" else 3
        return torch.full((1, length, 4), float(length)), torch.ones(1, length, dtype=torch.long)

    pipeline.encode_prompt = fake_encode
    pipeline._generate(
        prompt="positive",
        negative_prompt="negative",
        mode=LingBotGenerationMode.T2V,
        height=16,
        width=16,
        num_frames=9,
        num_inference_steps=2,
        guidance_scale=2.0,
        output_type="latent",
        cfg_parallel_group=object(),
    )

    assert encoded_prompts == (["positive"] if rank == 0 else ["negative"])
    assert len(pipeline.transformer.prepared_metadata) == 1
    [metadata] = pipeline.transformer.prepared_metadata
    assert metadata.attention_mask.shape == (1, 2 if rank == 0 else 3)
    assert len(pipeline.transformer.forward_metadata) == 2
    assert all(item is metadata for item in pipeline.transformer.forward_metadata)


def test_generate_prepares_fresh_metadata_for_next_request():
    from vllm_omni.diffusion.models.lingbot_video import LingBotGenerationMode

    pipeline = _make_pipeline()
    pipeline.transformer = _RecordingTransformer()
    pipeline.scheduler = _CorruptingScheduler()
    pipeline.progress_bar = lambda timesteps: timesteps
    pipeline.prepare_latents = lambda frames, height, width, *args: torch.zeros(1, 1, 3, height // 16, width // 16)

    def fake_encode(prompt, **kwargs):
        length = 2 if prompt == "first" else 5
        return torch.ones(1, length, 4), torch.ones(1, length, dtype=torch.long)

    pipeline.encode_prompt = fake_encode
    for prompt, width in [("first", 16), ("second", 32)]:
        pipeline._generate(
            prompt=prompt,
            mode=LingBotGenerationMode.T2V,
            height=16,
            width=width,
            num_frames=9,
            num_inference_steps=2,
            guidance_scale=1.0,
            output_type="latent",
        )

    first, second = pipeline.transformer.prepared_metadata
    assert first is not second
    assert first.latent_shape == (1, 1, 3, 1, 1)
    assert second.latent_shape == (1, 1, 3, 1, 2)
    assert first.attention_mask.shape == (1, 2)
    assert second.attention_mask.shape == (1, 5)
    assert all(item is first for item in pipeline.transformer.forward_metadata[:2])
    assert all(item is second for item in pipeline.transformer.forward_metadata[2:])


@pytest.mark.parametrize(
    ("guidance_scale", "batch_cfg", "cfg_parallel"),
    [
        pytest.param(1.0, False, False, id="cfg-disabled"),
        pytest.param(2.0, False, False, id="sequential-cfg"),
        pytest.param(2.0, True, False, id="batched-cfg"),
        pytest.param(2.0, False, True, id="parallel-cfg"),
    ],
)
def test_generate_releases_metadata_before_vae_decode(monkeypatch, guidance_scale, batch_cfg, cfg_parallel):
    from vllm_omni.diffusion.models.lingbot_video import LingBotGenerationMode
    from vllm_omni.diffusion.models.lingbot_video import pipeline_lingbot_video as module

    metadata_refs = []

    class Transformer(_RecordingTransformer):
        def prepare_metadata(self, latents, prompt_embeds, encoder_attention_mask=None):
            metadata = _PreparedInputObservation(latents.shape, prompt_embeds, encoder_attention_mask)
            metadata_refs.append(weakref.ref(metadata))
            return metadata

        def forward(self, latents, timestep, prompt_embeds, **kwargs):
            cache = kwargs["metadata_cache"]
            if cache.metadata is None:
                cache.metadata = self.prepare_metadata(latents, prompt_embeds, kwargs["encoder_attention_mask"])
            assert any(ref() is cache.metadata for ref in metadata_refs)
            return (torch.zeros_like(latents),)

    if cfg_parallel:
        monkeypatch.setattr(module.dist, "is_available", lambda: True)
        monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(module.dist, "get_rank", lambda group: 0)
        monkeypatch.setattr(module.dist, "get_world_size", lambda group: 2)
        monkeypatch.setattr(module.dist, "get_global_rank", lambda group, group_rank: group_rank)
        monkeypatch.setattr(module.dist, "broadcast", lambda tensor, **kwargs: tensor.zero_())
        monkeypatch.setattr(module.dist, "barrier", lambda **kwargs: None)
    pipeline = _make_pipeline()
    pipeline.transformer = Transformer()
    pipeline.scheduler = _CorruptingScheduler()
    pipeline.progress_bar = lambda timesteps: timesteps
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(1, 1, 3, 1, 1)
    pipeline.encode_prompt = lambda *args, **kwargs: (
        torch.ones(1, 2, 4),
        torch.ones(1, 2, dtype=torch.long),
    )
    decoded = torch.ones(3, 16, 16, 3)

    def decode(latents):
        expected_preparations = 2 if guidance_scale > 1.0 and not batch_cfg and not cfg_parallel else 1
        assert len(metadata_refs) == expected_preparations
        assert all(ref() is None for ref in metadata_refs)
        return decoded

    pipeline._decode_latents = decode
    actual = pipeline._generate(
        prompt="positive",
        mode=LingBotGenerationMode.T2V,
        height=16,
        width=16,
        num_frames=9,
        num_inference_steps=2,
        guidance_scale=guidance_scale,
        batch_cfg=batch_cfg,
        cfg_parallel_group=object() if cfg_parallel else None,
        output_type="pt",
    )

    assert actual is decoded


def test_t2i_decodes_and_returns_the_unique_frame():
    from vllm_omni.diffusion.models.lingbot_video import LingBotGenerationMode

    pipeline = _make_pipeline()
    pipeline.transformer = _RecordingTransformer()
    pipeline.scheduler = _CorruptingScheduler()
    pipeline.progress_bar = lambda timesteps: timesteps
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(1, 1, 1, 2, 2)
    pipeline.encode_prompt = lambda *args, **kwargs: (
        torch.ones(1, 2, 4),
        torch.ones(1, 2, dtype=torch.long),
    )
    pipeline._decode_latents = lambda latents: torch.ones(1, 16, 16, 3)

    image = pipeline._generate(
        prompt="a still robot",
        mode=LingBotGenerationMode.T2I,
        height=16,
        width=16,
        num_frames=1,
        num_inference_steps=2,
        guidance_scale=1.0,
        shift=3.0,
        output_type="pt",
    )

    assert image.shape == (16, 16, 3)


def test_forward_rejects_multi_request_batches():
    pipeline = _make_pipeline()
    first = OmniDiffusionRequest(
        prompt="first",
        sampling_params=OmniDiffusionSamplingParams(seed=1),
        request_id="lingbot-first",
    )
    second = OmniDiffusionRequest(
        prompt="second",
        sampling_params=OmniDiffusionSamplingParams(seed=2),
        request_id="lingbot-second",
    )

    with pytest.raises(ValueError, match="only supports one request"):
        pipeline.forward(DiffusionRequestBatch([first, second]))


def test_forward_marks_invalid_request_contract_as_client_error():
    from vllm_omni.errors import OmniClientError

    pipeline = _make_pipeline()
    request = _make_request_batch(
        {
            "prompt": "unsupported image edit",
            "modalities": ["image"],
            "multi_modal_data": {"image": Image.new("RGB", (32, 32), color="blue")},
        },
        height=192,
        width=320,
        num_frames=1,
    )

    with pytest.raises(OmniClientError, match="does not accept") as exc_info:
        pipeline.forward(request)

    assert exc_info.value.status_code == 400


def test_forward_marks_unsupported_output_count_as_client_error():
    from vllm_omni.errors import OmniClientError

    pipeline = _make_pipeline()
    request = _make_request_batch(
        {"prompt": "motion", "modalities": ["video"]},
        height=192,
        width=320,
        num_frames=9,
        num_outputs_per_prompt=2,
    )

    with pytest.raises(OmniClientError, match="one output per prompt") as exc_info:
        pipeline.forward(request)

    assert exc_info.value.status_code == 400


def test_load_weights_rejects_external_weight_stream():
    pipeline = _make_pipeline()

    assert pipeline.load_weights([]) == set()
    with pytest.raises(RuntimeError, match="components are loaded directly"):
        pipeline.load_weights([("transformer.weight", torch.zeros(1))])


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
def test_pipeline_profiler_records_executed_stages_per_request(monkeypatch, enabled):
    from vllm_omni.diffusion.models.lingbot_video import pipeline_lingbot_video as module
    from vllm_omni.diffusion.profiler import diffusion_pipeline_profiler as profiler_module
    from vllm_omni.diffusion.worker.utils import consume_pipeline_stage_durations

    class TextEncoder(nn.Module):
        def forward(self):
            return torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.long)

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(latents_mean=[0.0], latents_std=[1.0])

        def encode(self, value):
            return value

        def decode(self, latents):
            return (latents.repeat(1, 3, 1, 1, 1),)

    components = [
        (module.LingBotVideoTransformer3DModel, _RecordingTransformer()),
        (module.Qwen3VLForConditionalGeneration, TextEncoder()),
        (module.Qwen3VLProcessor, SimpleNamespace()),
        (module.AutoencoderKLWan, VAE()),
        (module.FlowUniPCMultistepScheduler, _CorruptingScheduler()),
    ]
    for component_class, component in components:
        monkeypatch.setattr(component_class, "from_pretrained", lambda *args, value=component, **kwargs: value)
    monkeypatch.setattr(module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(module.LingBotVideoPipeline, "encode_prompt", lambda self, *args, **kwargs: self.text_encoder())
    monkeypatch.setattr(profiler_module.current_omni_platform, "is_available", lambda: False)
    ticks = iter(range(100))
    monkeypatch.setattr(profiler_module.time, "perf_counter", lambda: float(next(ticks)))
    pipeline = module.LingBotVideoPipeline(
        od_config=SimpleNamespace(
            model="unused", dtype=torch.float32, enable_diffusion_pipeline_profiler=enabled, flow_shift=None
        )
    )
    request = _make_request_batch(
        {"prompt": "a robot", "modalities": ["video"]},
        height=16,
        width=16,
        num_frames=9,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=42,
    )
    prefix = "LingBotVideoPipeline."
    expected = {prefix + "text_encoder.forward": 1.0, prefix + "transformer.forward": 2.0, prefix + "vae.decode": 1.0}
    first = pipeline(request)
    assert first.stage_durations == (expected if enabled else {})
    assert first.output["video"].shape == (3, 2, 2, 3)
    # A latent-only request must not inherit the previous request's VAE timing.
    request.requests[0].sampling_params.output_type = "latent"
    second = pipeline(request)
    latent_expected = {key: value for key, value in expected.items() if not key.endswith("vae.decode")}
    assert second.stage_durations == (latent_expected if enabled else {})
    assert first.stage_durations == (expected if enabled else {})
    consumed = consume_pipeline_stage_durations(pipeline)
    if enabled:
        assert consumed == {**latent_expected, prefix + "forward": 7.0}
    else:
        assert consumed == {}
    assert consume_pipeline_stage_durations(pipeline) == {}
