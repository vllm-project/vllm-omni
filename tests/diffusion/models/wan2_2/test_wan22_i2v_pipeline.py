# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from tests.diffusion.models.wan2_2.conftest import StubScheduler, StubTransformer, StubVAE, noop_progress_bar
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import _WAN_TEXT_ENCODER_OFFLOAD_PLAN, build_wan_scheduler
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import (
    Wan22I2VPipeline,
    get_wan22_i2v_post_process_func,
    get_wan22_i2v_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_wan22_i2v_postprocess_honors_request_output_type() -> None:
    video = torch.zeros(1, 4, 1, 2, 2)

    output = get_wan22_i2v_post_process_func(SimpleNamespace())(
        video,
        sampling_params=SimpleNamespace(output_type="latent"),
    )

    assert output is video


def test_i2v_pipeline_declares_text_encoder_offload_blocks() -> None:
    assert Wan22I2VPipeline._offload_plan is _WAN_TEXT_ENCODER_OFFLOAD_PLAN


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


def _make_i2v_sampling(**overrides):
    values: dict[str, object] = {
        "height": 16,
        "width": 16,
        "num_frames": 5,
        "num_inference_steps": 1,
        "guidance_scale_provided": True,
        "guidance_scale": 1.0,
        "guidance_scale_2": None,
        "guidance_scale_2_provided": False,
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


def test_i2v_preprocess_requires_image_and_resizes_to_480p_aspect() -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())
    request = SimpleNamespace(
        prompt={"prompt": "p", "multi_modal_data": {"image": Image.new("RGB", (320, 160), "red")}},
        sampling_params=SimpleNamespace(height=None, width=None),
    )

    result = preprocess(request)
    prompt = result.prompt

    assert result.sampling_params.height == 432
    assert result.sampling_params.width == 880
    assert prompt["multi_modal_data"]["image"].size == (880, 432)

    missing_image = SimpleNamespace(
        prompt={"prompt": "p", "multi_modal_data": {}},
        sampling_params=SimpleNamespace(height=None, width=None),
    )
    with pytest.raises(ValueError, match="No image is provided"):
        preprocess(missing_image)


def _make_i2v_preprocess_request(last_image):
    return SimpleNamespace(
        prompt={
            "prompt": "p",
            "multi_modal_data": {
                "image": Image.new("RGB", (320, 160), "red"),
                "last_image": last_image,
            },
        },
        sampling_params=SimpleNamespace(height=16, width=16),
    )


def test_i2v_preprocess_treats_empty_last_image_list_as_absent() -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())

    result = preprocess(_make_i2v_preprocess_request([]))

    assert result.prompt["multi_modal_data"]["last_image"] is None
    assert result.batch_compatibility_key == ("wan22_i2v_last_image", False)


def test_i2v_preprocess_unwraps_single_last_image_list() -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())
    last_image = Image.new("RGB", (16, 16), "blue")

    result = preprocess(_make_i2v_preprocess_request([last_image]))

    assert result.prompt["multi_modal_data"]["last_image"] is last_image
    assert result.batch_compatibility_key == ("wan22_i2v_last_image", True)


@pytest.mark.parametrize(
    "last_image",
    [
        Image.new("RGB", (16, 16), "blue"),
        torch.zeros(3, 16, 16),
    ],
)
def test_i2v_preprocess_preserves_supported_last_image(last_image) -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())

    result = preprocess(_make_i2v_preprocess_request(last_image))

    assert result.prompt["multi_modal_data"]["last_image"] is last_image
    assert result.batch_compatibility_key == ("wan22_i2v_last_image", True)


def test_i2v_preprocess_loads_last_image_path(tmp_path) -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())
    path = tmp_path / "last.png"
    Image.new("RGB", (16, 16), "blue").save(path)

    result = preprocess(_make_i2v_preprocess_request(str(path)))

    last_image = result.prompt["multi_modal_data"]["last_image"]
    assert isinstance(last_image, Image.Image)
    assert last_image.mode == "RGB"
    assert result.batch_compatibility_key == ("wan22_i2v_last_image", True)


def test_i2v_preprocess_rejects_multiple_last_images() -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())

    with pytest.raises(ValueError, match="at most one last_image"):
        preprocess(
            _make_i2v_preprocess_request(
                [
                    Image.new("RGB", (16, 16), "blue"),
                    Image.new("RGB", (16, 16), "green"),
                ]
            )
        )


def test_i2v_preprocess_rejects_unsupported_last_image_type() -> None:
    preprocess = get_wan22_i2v_pre_process_func(SimpleNamespace())

    with pytest.raises(TypeError, match="Unsupported last_image format"):
        preprocess(_make_i2v_preprocess_request({"not": "an image"}))


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
    torch.testing.assert_close(
        calls[0]["hidden_states"][:, :, 0],
        torch.ones_like(calls[0]["hidden_states"][:, :, 0]),
    )
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


def test_i2v_prepare_latents_preserves_batched_image_conditions() -> None:
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    generators = [
        torch.Generator(device="cpu").manual_seed(1),
        torch.Generator(device="cpu").manual_seed(2),
    ]

    latents, condition, first_frame_mask = pipeline.prepare_latents(
        image=torch.zeros(2, 3, 16, 16),
        batch_size=2,
        num_channels_latents=4,
        height=16,
        width=16,
        num_frames=5,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=generators,
    )

    assert latents.shape == (2, 4, 2, 2, 2)
    assert condition.shape == (2, 4, 1, 2, 2)
    assert first_frame_mask.shape == (2, 1, 2, 2, 2)


@pytest.mark.parametrize("codec", ["libx264", "libx265"])
def test_i2v_forward_passes_codec_policy_to_chunked_decode(monkeypatch, codec) -> None:
    from vllm_omni.diffusion.models.wan2_2 import pipeline_wan2_2_i2v as module

    monkeypatch.setattr(module.current_omni_platform, "is_available", lambda: False)
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    pipeline.scheduler = StubScheduler([9])
    pipeline.od_config = OmniDiffusionConfig(model=None, flow_shift=5.0)
    pipeline._sample_solver = "unipc"
    pipeline._flow_shift = 5.0
    pipeline.boundary_ratio = 0.875
    pipeline.has_image_encoder = False
    pipeline.check_inputs = lambda **kwargs: None
    pipeline.encode_prompt = lambda **kwargs: (torch.zeros(1, 2, 3), None)
    pipeline.prepare_latents = lambda **kwargs: (
        torch.zeros(1, 4, 2, 2, 2),
        torch.zeros(1, 4, 2, 2, 2),
        torch.ones(1, 1, 2, 2, 2),
    )
    pipeline.diffuse = lambda **kwargs: kwargs["latents"]
    options = {"crf": "0"}
    captured = {}

    def decode(vae, latents, **kwargs):
        captured.update(kwargs)
        return [b"mp4"]

    monkeypatch.setattr(module, "decode_to_mp4", decode)
    sampling = OmniDiffusionSamplingParams(
        output_type="np",
        height=16,
        width=16,
        num_frames=5,
        num_inference_steps=1,
        guidance_scale=1.0,
        extra_args={"preencode_mp4": True, "video_codec": codec, "video_codec_options": options},
    )
    request = OmniDiffusionRequest(
        prompt={"prompt": "prompt", "multi_modal_data": {"image": torch.zeros(1, 3, 16, 16)}},
        sampling_params=sampling,
        request_id="codec",
    )
    outputs = pipeline.forward(DiffusionRequestBatch([request]))

    assert outputs[0].output == [b"mp4"]
    assert captured["video_codec"] == codec
    assert captured["video_codec_options"] == options


@pytest.mark.parametrize("solver", ["unipc", "euler"])
@pytest.mark.parametrize("shift", [3.0, 5.0, 12.0])
def test_i2v_forward_batches_conditions_random_inputs_and_outputs(monkeypatch, solver: str, shift: float) -> None:
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v.current_omni_platform",
        SimpleNamespace(is_available=lambda: False),
    )
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    pipeline.scheduler = build_wan_scheduler(solver, shift)
    pipeline.od_config = SimpleNamespace(flow_shift=shift)
    pipeline._sample_solver = solver
    pipeline._flow_shift = shift
    pipeline.boundary_ratio = 0.875
    pipeline.has_image_encoder = False
    pipeline._guidance_scale = None
    pipeline._guidance_scale_2 = None
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
        batch_size = kwargs["batch_size"]
        return (
            kwargs["latents"],
            torch.zeros(batch_size, 4, 1, 2, 2),
            torch.ones(batch_size, 1, 2, 2, 2),
        )

    pipeline.encode_prompt = _fake_encode_prompt  # type: ignore[method-assign]
    pipeline.prepare_latents = _fake_prepare_latents  # type: ignore[method-assign]
    pipeline.diffuse = lambda **kwargs: kwargs["latents"]  # type: ignore[method-assign]

    gen_a = torch.Generator(device="cpu").manual_seed(1)
    gen_b = torch.Generator(device="cpu").manual_seed(2)
    latents_a = torch.zeros(2, 4, 2, 2, 2)
    latents_b = torch.ones(2, 4, 2, 2, 2)
    image_a = torch.zeros(1, 3, 16, 16)
    image_b = torch.ones(1, 3, 16, 16)
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={"prompt": "first", "multi_modal_data": {"image": image_a}},
                sampling_params=_make_i2v_sampling(
                    generator=gen_a,
                    latents=latents_a,
                    num_outputs_per_prompt=2,
                    num_inference_steps=50,
                    extra_args={"sample_solver": solver, "flow_shift": shift},
                ),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={"prompt": "second", "multi_modal_data": {"image": image_b}},
                sampling_params=_make_i2v_sampling(
                    generator=gen_b,
                    latents=latents_b,
                    num_outputs_per_prompt=2,
                    num_inference_steps=50,
                    extra_args={"sample_solver": solver, "flow_shift": shift},
                ),
            ),
        ]
    )

    outputs = pipeline.forward(batch)
    if solver == "unipc":
        sigmas = np.linspace(float(np.float32(0.999)), 0.0, 51)[:-1]
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        torch.testing.assert_close(
            pipeline.scheduler.sigmas, torch.tensor(np.append(sigmas, 0.0), dtype=torch.float32), rtol=0, atol=0
        )
        assert pipeline.scheduler.timesteps.tolist() == (sigmas * 1000).astype(np.int64).tolist()
    else:
        reference = build_wan_scheduler("euler", shift)
        reference.set_timesteps(50, device="cpu")
        torch.testing.assert_close(pipeline.scheduler.sigmas, reference.sigmas, rtol=0, atol=0)

    assert prepare_call["batch_size"] == 4
    assert prepare_call["generator"] == [gen_a, gen_a, gen_b, gen_b]
    torch.testing.assert_close(prepare_call["latents"], torch.cat([latents_a, latents_b]))
    torch.testing.assert_close(
        prepare_call["image"],
        torch.cat([image_a, image_a, image_b, image_b]),
    )
    assert len(outputs) == 2
    torch.testing.assert_close(outputs[0].output, latents_a)
    torch.testing.assert_close(outputs[1].output, latents_b)


def test_i2v_forward_rejects_mismatched_tensor_condition_shapes(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v.current_omni_platform",
        SimpleNamespace(is_available=lambda: False),
    )
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    pipeline.scheduler = StubScheduler([9])
    pipeline.od_config = SimpleNamespace(flow_shift=5.0)
    pipeline._sample_solver = "unipc"
    pipeline._flow_shift = 5.0
    pipeline.boundary_ratio = 0.875
    pipeline.has_image_encoder = False
    pipeline._guidance_scale = None
    pipeline._guidance_scale_2 = None
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.check_inputs = lambda **kwargs: None
    pipeline.encode_prompt = lambda **kwargs: (torch.zeros(2, 2, 3), None)  # type: ignore[method-assign]
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={"prompt": "first", "multi_modal_data": {"image": torch.zeros(1, 3, 16, 16)}},
                sampling_params=_make_i2v_sampling(),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={"prompt": "second", "multi_modal_data": {"image": torch.zeros(1, 3, 16, 8)}},
                sampling_params=_make_i2v_sampling(),
            ),
        ]
    )

    with pytest.raises(ValueError, match="image condition"):
        pipeline.forward(batch)


def test_i2v_forward_rejects_mixed_last_image_presence() -> None:
    pipeline = _make_i2v_pipeline(expand_timesteps=True)
    image = torch.zeros(1, 3, 16, 16)
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={
                    "prompt": "first",
                    "multi_modal_data": {"image": image, "last_image": image},
                },
                sampling_params=_make_i2v_sampling(),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={"prompt": "second", "multi_modal_data": {"image": image}},
                sampling_params=_make_i2v_sampling(),
            ),
        ]
    )

    with pytest.raises(ValueError, match="mix of provided and missing last_image"):
        pipeline.forward(batch)
