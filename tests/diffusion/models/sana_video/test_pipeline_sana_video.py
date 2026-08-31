# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
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


class _DummyComponent:
    pass


def _make_request_batch(prompt, request_id="sana-video-test", **sampling_overrides):
    sampling = OmniDiffusionSamplingParams(**sampling_overrides)
    request = OmniDiffusionRequest(prompt=prompt, sampling_params=sampling, request_id=request_id)
    return DiffusionRequestBatch([request])


def test_sana_video_pipeline_import_and_registry():
    from vllm_omni.diffusion.models.sana_video import (
        SanaImageToVideoPipeline,
        SanaVideoPipeline,
        SanaVideoTransformer3DModel,
        get_sana_video_i2v_post_process_func,
        get_sana_video_i2v_pre_process_func,
        get_sana_video_post_process_func,
    )
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
    )

    assert SanaImageToVideoPipeline is not None
    assert SanaVideoPipeline is not None
    assert SanaVideoTransformer3DModel is not None
    assert get_sana_video_i2v_post_process_func is not None
    assert get_sana_video_i2v_pre_process_func is not None
    assert get_sana_video_post_process_func is not None
    assert _DIFFUSION_MODELS["SanaVideoPipeline"] == (
        "sana_video",
        "pipeline_sana_video",
        "SanaVideoPipeline",
    )
    assert _DIFFUSION_POST_PROCESS_FUNCS["SanaVideoPipeline"] == "get_sana_video_post_process_func"
    assert _DIFFUSION_MODELS["SanaImageToVideoPipeline"] == (
        "sana_video",
        "pipeline_sana_video_i2v",
        "SanaImageToVideoPipeline",
    )
    assert _DIFFUSION_POST_PROCESS_FUNCS["SanaImageToVideoPipeline"] == "get_sana_video_i2v_post_process_func"
    assert _DIFFUSION_PRE_PROCESS_FUNCS["SanaImageToVideoPipeline"] == "get_sana_video_i2v_pre_process_func"


def test_component_discovery_declarations():
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    assert SanaVideoPipeline._dit_modules == ["transformer"]
    assert SanaVideoPipeline._encoder_modules == ["text_encoder"]
    assert SanaVideoPipeline._vae_modules == ["vae"]
    assert SanaVideoPipeline.supports_step_execution is False


def test_sana_video_loads_concrete_gemma_tokenizer(monkeypatch):
    from vllm_omni.diffusion.models.sana_video import pipeline_sana_video

    captured: dict[str, object] = {}
    expected = object()

    def fake_from_pretrained_with_prefetch(loader, model, **kwargs):
        captured.update(loader=loader, model=model, kwargs=kwargs)
        return expected

    monkeypatch.setattr(
        pipeline_sana_video,
        "from_pretrained_with_prefetch",
        fake_from_pretrained_with_prefetch,
    )

    result = pipeline_sana_video._load_sana_tokenizer(
        "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
        ["tokenizer", "text_encoder"],
        local_files_only=False,
    )

    assert result is expected
    assert captured["loader"] == pipeline_sana_video.GemmaTokenizer.from_pretrained
    assert captured["model"] == "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
    assert captured["kwargs"] == {
        "subfolder": "tokenizer",
        "prefetch_list": ["tokenizer", "text_encoder"],
        "local_files_only": False,
    }


def test_sana_video_declares_extra_body_params():
    from vllm_omni.model_extras import get_extra_body_params

    assert get_extra_body_params("SanaVideoPipeline") == {
        "clean_caption",
        "motion_score",
        "use_resolution_binning",
    }
    assert get_extra_body_params("SanaImageToVideoPipeline") == {
        "clean_caption",
        "motion_score",
        "use_resolution_binning",
    }


def test_sana_video_i2v_preprocesses_image_and_preserves_aspect_ratio():
    from vllm_omni.diffusion.models.sana_video import get_sana_video_i2v_pre_process_func

    request = OmniDiffusionRequest(
        prompt={"prompt": "a robot walks", "multi_modal_data": {"image": Image.new("RGB", (640, 360))}},
        sampling_params=OmniDiffusionSamplingParams(),
        request_id="sana-video-i2v-preprocess",
    )
    processed = get_sana_video_i2v_pre_process_func(SimpleNamespace())(request)

    assert processed.sampling_params.height == 448
    assert processed.sampling_params.width == 832
    assert processed.prompt["multi_modal_data"]["image"].size == (832, 448)


def test_sana_video_i2v_remote_720p_uses_loaded_transformer_config():
    from vllm_omni.diffusion.models.sana_video import get_sana_video_i2v_pre_process_func

    od_config = SimpleNamespace(
        model="Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        tf_model_config={"sample_size": 22},
    )
    request = OmniDiffusionRequest(
        prompt={"prompt": "a robot walks", "multi_modal_data": {"image": Image.new("RGB", (1280, 704))}},
        sampling_params=OmniDiffusionSamplingParams(),
        request_id="sana-video-i2v-remote-720p",
    )

    processed = get_sana_video_i2v_pre_process_func(od_config)(request)

    assert processed.sampling_params.height == 704
    assert processed.sampling_params.width == 1280
    assert processed.prompt["multi_modal_data"]["image"].size == (1280, 704)


@pytest.mark.parametrize(
    "frames",
    [
        [[Image.new("RGB", (16, 16)) for _ in range(2)]],
        np.zeros((1, 2, 16, 16, 3), dtype=np.float32),
        [[np.zeros((16, 16, 3), dtype=np.float32) for _ in range(2)]],
    ],
)
@pytest.mark.parametrize(
    "factory_name",
    ["get_sana_video_post_process_func", "get_sana_video_i2v_post_process_func"],
)
def test_sana_video_diffusers_preserves_postprocessed_frames(frames, factory_name):
    from vllm_omni.diffusion.models import sana_video

    post_process = getattr(sana_video, factory_name)(SimpleNamespace(diffusion_load_format="diffusers"))

    result = post_process(frames)

    assert result["payload"]["video"] is frames


@pytest.mark.parametrize(
    "factory_name",
    ["get_sana_video_post_process_func", "get_sana_video_i2v_post_process_func"],
)
def test_sana_video_diffusers_still_postprocesses_native_tensor(factory_name):
    from vllm_omni.diffusion.models import sana_video

    decoded_video = torch.zeros(1, 3, 2, 16, 16)
    post_process = getattr(sana_video, factory_name)(SimpleNamespace(diffusion_load_format="diffusers"))

    result = post_process(decoded_video)

    assert isinstance(result["payload"]["video"], np.ndarray)
    assert result["payload"]["video"].shape == (1, 2, 16, 16, 3)


@pytest.mark.parametrize(
    "factory_name",
    ["get_sana_video_post_process_func", "get_sana_video_i2v_post_process_func"],
)
def test_sana_video_postprocess_preserves_requested_latents(factory_name):
    from vllm_omni.diffusion.models import sana_video

    latents = torch.randn(1, 4, 2, 2, 2)
    post_process = getattr(sana_video, factory_name)(SimpleNamespace(diffusion_load_format="native"))

    result = post_process(latents, sampling_params=SimpleNamespace(output_type="latent"))

    assert result is latents


def test_sana_video_720p_model_id_fallback(monkeypatch):
    from vllm_omni.diffusion.models.sana_video import pipeline_sana_video

    monkeypatch.setattr(pipeline_sana_video, "get_hf_file_to_dict", lambda *_args, **_kwargs: None)
    od_config = SimpleNamespace(
        model="Efficient-Large-Model/SANA-Video_2B_720p_diffusers",
        tf_model_config={},
    )

    assert pipeline_sana_video.resolve_sana_video_sample_size(od_config) == 22


@pytest.mark.parametrize(
    ("requested_output_type", "expected_internal_output_type"),
    [
        (None, "raw"),
        ("np", "raw"),
        ("latent", "latent"),
    ],
)
def test_sana_video_i2v_forward_maps_image_request(requested_output_type, expected_internal_output_type):
    from vllm_omni.diffusion.models.sana_video import SanaImageToVideoPipeline

    pipeline = object.__new__(SanaImageToVideoPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(sample_size=30))
    calls = []

    def fake_generate_i2v(**kwargs):
        calls.append(kwargs)
        return torch.zeros(kwargs["num_videos_per_prompt"], 3, 9, 192, 320)

    pipeline._generate_i2v = fake_generate_i2v
    image = Image.new("RGB", (320, 192))
    custom_timesteps = torch.tensor([900, 100])
    req = _make_request_batch(
        {"prompt": "a robot walks", "negative_prompt": "blurry", "multi_modal_data": {"image": image}},
        height=192,
        width=320,
        num_frames=9,
        num_inference_steps=2,
        timesteps=custom_timesteps,
        guidance_scale=4.5,
        num_outputs_per_prompt=2,
        eta=0.25,
        seed=42,
        output_type=requested_output_type,
        extra_args={"clean_caption": True, "motion_score": 30, "use_resolution_binning": False},
    )

    output = pipeline.forward(req)

    assert output.output.shape == (2, 3, 9, 192, 320)
    assert calls[0]["image"] is image
    assert calls[0]["prompt"] == "a robot walks motion score: 30."
    assert calls[0]["negative_prompt"] == "blurry"
    assert calls[0]["frames"] == 9
    assert calls[0]["timesteps"] is custom_timesteps
    assert calls[0]["sigmas"] is None
    assert calls[0]["num_videos_per_prompt"] == 2
    assert calls[0]["eta"] == 0.25
    assert calls[0]["clean_caption"] is True
    assert calls[0]["generator"].initial_seed() == 42
    assert calls[0]["output_type"] == expected_internal_output_type


def test_sana_video_i2v_forward_maps_custom_sigmas():
    from vllm_omni.diffusion.models.sana_video import SanaImageToVideoPipeline

    pipeline = object.__new__(SanaImageToVideoPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(sample_size=30))
    calls = []

    def fake_generate_i2v(**kwargs):
        calls.append(kwargs)
        return torch.zeros(1, 3, 9, 192, 320)

    pipeline._generate_i2v = fake_generate_i2v
    sigmas = [1.0, 0.5]
    output = pipeline.forward(
        _make_request_batch(
            {
                "prompt": "a robot walks",
                "multi_modal_data": {"image": Image.new("RGB", (320, 192))},
            },
            height=192,
            width=320,
            num_frames=9,
            sigmas=sigmas,
            generator_device="cpu",
            extra_args={"use_resolution_binning": False},
        )
    )

    assert output.output.shape == (1, 3, 9, 192, 320)
    assert calls[0]["timesteps"] is None
    assert calls[0]["sigmas"] is sigmas


def test_sana_video_i2v_omitted_num_frames_uses_model_default():
    from vllm_omni.diffusion.models.sana_video import SanaImageToVideoPipeline

    pipeline = object.__new__(SanaImageToVideoPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(sample_size=30))
    calls = []

    def fake_generate_i2v(**kwargs):
        calls.append(kwargs)
        return torch.zeros(1, 3, 81, 2, 2)

    pipeline._generate_i2v = fake_generate_i2v
    req = _make_request_batch(
        {
            "prompt": "a robot walks",
            "multi_modal_data": {"image": Image.new("RGB", (320, 192))},
        },
        height=192,
        width=320,
        num_inference_steps=1,
        generator_device="cpu",
        extra_args={"use_resolution_binning": False},
    )

    assert req.sampling_params.num_frames == 1
    output = pipeline.forward(req)

    assert output.output.shape == (1, 3, 81, 2, 2)
    assert calls[0]["frames"] == 81


def test_sana_video_i2v_clean_caption_request_is_explicitly_rejected():
    from vllm_omni.diffusion.models.sana_video import SanaImageToVideoPipeline

    pipeline = object.__new__(SanaImageToVideoPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(sample_size=30))
    pipeline.text_encoder = SimpleNamespace(dtype=torch.float32)
    pipeline.tokenizer = None
    pipeline.check_inputs = lambda **_kwargs: None

    req = _make_request_batch(
        {
            "prompt": "a robot walks",
            "multi_modal_data": {"image": Image.new("RGB", (320, 192))},
        },
        height=192,
        width=320,
        num_frames=9,
        generator_device="cpu",
        extra_args={"clean_caption": True, "use_resolution_binning": False},
    )

    with pytest.raises(ValueError, match="does not support `clean_caption=True`"):
        pipeline.forward(req)


@pytest.mark.parametrize(
    ("custom_timesteps", "custom_sigmas"),
    [
        (torch.tensor([4.0, 2.0]), None),
        (None, [1.0, 0.5]),
    ],
)
def test_sana_video_i2v_consumes_batch_schedule_and_step_kwargs(custom_timesteps, custom_sigmas):
    from vllm_omni.diffusion.models.sana_video import SanaImageToVideoPipeline

    calls = SimpleNamespace(
        encode_prompt=[],
        prepare_latents=[],
        set_timesteps=[],
        step=[],
        transformer=[],
    )

    class StubTransformer:
        dtype = torch.float32
        config = SimpleNamespace(
            sample_size=30,
            in_channels=4,
            out_channels=4,
            patch_size=(1, 1, 1),
        )

        def __call__(
            self,
            hidden_states,
            *,
            encoder_hidden_states,
            encoder_attention_mask,
            timestep,
            return_dict,
        ):
            calls.transformer.append(
                (
                    hidden_states.shape,
                    encoder_hidden_states.shape,
                    encoder_attention_mask.shape,
                    timestep.shape,
                    return_dict,
                )
            )
            return (torch.zeros_like(hidden_states),)

    class StubScheduler:
        order = 1

        def set_timesteps(
            self,
            num_inference_steps=None,
            *,
            device=None,
            timesteps=None,
            sigmas=None,
        ):
            calls.set_timesteps.append(
                {
                    "num_inference_steps": num_inference_steps,
                    "timesteps": timesteps,
                    "sigmas": sigmas,
                }
            )
            schedule_length = len(timesteps if timesteps is not None else sigmas)
            self.timesteps = torch.arange(schedule_length, 0, -1, dtype=torch.float32, device=device)

        def step(
            self,
            _noise_pred,
            _timestep,
            current_latents,
            *,
            eta,
            generator,
            return_dict,
        ):
            calls.step.append({"eta": eta, "generator": generator, "return_dict": return_dict})
            return (current_latents + 1,)

    class VaeMustNotBeUsed:
        @property
        def dtype(self):
            pytest.fail("latent output must return before accessing the VAE")

        def decode(self, *_args, **_kwargs):
            pytest.fail("latent output must not call VAE decode")

    pipeline = object.__new__(SanaImageToVideoPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = StubTransformer()
    pipeline.scheduler = StubScheduler()
    pipeline.vae = VaeMustNotBeUsed()
    pipeline.check_inputs = lambda **_kwargs: None
    pipeline.video_processor = SimpleNamespace(
        preprocess=lambda _image, height, width: torch.zeros(1, 3, height, width),
    )
    pipeline.progress_bar = lambda **_kwargs: nullcontext(SimpleNamespace(update=lambda: None))

    def capture_encode_prompt(*_args, **kwargs):
        calls.encode_prompt.append(kwargs)
        batch_size = kwargs["num_videos_per_prompt"]
        return (
            torch.zeros(batch_size, 1, 1),
            torch.ones(batch_size, 1, dtype=torch.bool),
            torch.zeros(batch_size, 1, 1),
            torch.ones(batch_size, 1, dtype=torch.bool),
        )

    pipeline.encode_prompt = capture_encode_prompt
    initial_latents = torch.zeros(2, 4, 2, 2, 2)

    def capture_prepare_latents(*_args, **kwargs):
        calls.prepare_latents.append(kwargs)
        return initial_latents

    pipeline._prepare_i2v_latents = capture_prepare_latents
    generator = torch.Generator(device="cpu").manual_seed(42)

    output = pipeline._generate_i2v(
        image=Image.new("RGB", (2, 2)),
        prompt="test",
        negative_prompt="",
        height=2,
        width=2,
        frames=2,
        num_inference_steps=50,
        timesteps=custom_timesteps,
        sigmas=custom_sigmas,
        guidance_scale=4.5,
        num_videos_per_prompt=2,
        eta=0.25,
        generator=generator,
        latents=None,
        clean_caption=True,
        use_resolution_binning=False,
        max_sequence_length=1,
        output_type="latent",
    )

    assert calls.encode_prompt[0]["num_videos_per_prompt"] == 2
    assert calls.encode_prompt[0]["clean_caption"] is True
    assert calls.prepare_latents[0]["batch_size"] == 2
    assert len(calls.set_timesteps) == 1
    assert calls.set_timesteps[0]["num_inference_steps"] is None
    assert calls.set_timesteps[0]["timesteps"] is custom_timesteps
    assert calls.set_timesteps[0]["sigmas"] is custom_sigmas
    assert len(calls.step) == 2
    assert all(call == {"eta": 0.25, "generator": generator, "return_dict": False} for call in calls.step)
    assert all(
        call
        == (
            torch.Size([4, 4, 2, 2, 2]),
            torch.Size([4, 1, 1]),
            torch.Size([4, 1]),
            torch.Size([4, 1, 2, 2, 2]),
            False,
        )
        for call in calls.transformer
    )
    expected = initial_latents.clone()
    expected[:, :, 1:] += 2
    torch.testing.assert_close(output, expected)


def test_sana_video_i2v_uses_diffusers_complex_instruction_default():
    import inspect

    from diffusers import SanaImageToVideoPipeline as DiffusersSanaImageToVideoPipeline

    from vllm_omni.diffusion.models.sana_video import SanaImageToVideoPipeline

    reference_default = (
        inspect.signature(DiffusersSanaImageToVideoPipeline.__call__).parameters["complex_human_instruction"].default
    )
    native_default = (
        inspect.signature(SanaImageToVideoPipeline._generate_i2v).parameters["complex_human_instruction"].default
    )
    assert native_default == reference_default

    class StopAfterPromptEncodingError(Exception):
        pass

    pipeline = object.__new__(SanaImageToVideoPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.check_inputs = lambda **_kwargs: None
    captured = {}

    def capture_encode_prompt(*_args, **kwargs):
        captured.update(kwargs)
        raise StopAfterPromptEncodingError

    pipeline.encode_prompt = capture_encode_prompt
    with pytest.raises(StopAfterPromptEncodingError):
        pipeline._generate_i2v(
            image=Image.new("RGB", (320, 192)),
            prompt="a robot walks",
            negative_prompt="blurry",
            height=192,
            width=320,
            frames=9,
            num_inference_steps=2,
            guidance_scale=4.5,
            generator=torch.Generator(device="cpu").manual_seed(42),
            latents=None,
            use_resolution_binning=False,
            max_sequence_length=300,
        )

    assert captured["complex_human_instruction"] == reference_default


def test_sana_video_t2v_complex_instruction_has_no_mutable_default():
    import inspect

    from vllm_omni.diffusion.models.sana_video.pipeline_sana_video import (
        SANA_VIDEO_COMPLEX_HUMAN_INSTRUCTION,
        SanaVideoPipeline,
    )

    parameter = inspect.signature(SanaVideoPipeline._generate).parameters["complex_human_instruction"]
    assert parameter.default is None
    assert isinstance(SANA_VIDEO_COMPLEX_HUMAN_INSTRUCTION, tuple)


def test_diffusers_adapter_t2v_omitted_num_frames_uses_sana_default():
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import DiffusersAdapterPipeline
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import SanaVideoPipelineUtils

    adapter = object.__new__(DiffusersAdapterPipeline)
    adapter._accept_call_kwargs = {"prompt", "negative_prompt", "frames", "generator"}
    adapter._pipeline_utils = SanaVideoPipelineUtils()
    adapter.od_config = SimpleNamespace(diffusers_call_kwargs={}, output_type=None)

    req = _make_request_batch(
        {"prompt": "a robot walks", "negative_prompt": "blurry"},
        seed=42,
        generator_device="cpu",
    )
    assert req.sampling_params.num_frames == 1

    kwargs = adapter._build_call_kwargs(req)

    assert kwargs["frames"] == 81
    assert "num_frames" not in kwargs
    assert kwargs["generator"].initial_seed() == 42


def test_diffusers_adapter_i2v_omitted_num_frames_uses_sana_default():
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import DiffusersAdapterPipeline
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import SanaVideoPipelineUtils

    adapter = object.__new__(DiffusersAdapterPipeline)
    adapter._accept_call_kwargs = {"prompt", "negative_prompt", "image", "frames", "generator"}
    adapter._pipeline_utils = SanaVideoPipelineUtils()
    adapter.od_config = SimpleNamespace(diffusers_call_kwargs={}, output_type=None)
    image = Image.new("RGB", (320, 192))

    req = _make_request_batch(
        {"prompt": "a robot walks", "negative_prompt": "blurry", "multi_modal_data": {"image": image}},
        seed=42,
        generator_device="cpu",
    )
    assert req.sampling_params.num_frames == 1

    kwargs = adapter._build_call_kwargs(req)

    assert kwargs["image"] is image
    assert kwargs["prompt"] == "a robot walks"
    assert kwargs["negative_prompt"] == "blurry"
    assert kwargs["frames"] == 81
    assert kwargs["generator"].initial_seed() == 42


@pytest.mark.parametrize(
    ("sampling_overrides", "expected_frames"),
    [
        ({}, 49),
        ({"num_frames": 9}, 9),
    ],
)
def test_diffusers_adapter_sana_resolves_frame_precedence(sampling_overrides, expected_frames):
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import DiffusersAdapterPipeline
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import SanaVideoPipelineUtils

    adapter = object.__new__(DiffusersAdapterPipeline)
    adapter._accept_call_kwargs = {"prompt", "frames", "generator"}
    adapter._pipeline_utils = SanaVideoPipelineUtils()
    adapter.od_config = SimpleNamespace(diffusers_call_kwargs={"frames": 49}, output_type=None)

    kwargs = adapter._build_call_kwargs(
        _make_request_batch(
            "a robot walks",
            seed=42,
            generator_device="cpu",
            **sampling_overrides,
        )
    )

    assert kwargs["frames"] == expected_frames


@pytest.mark.parametrize("pipeline_class_name", ["SanaVideoPipeline", "SanaImageToVideoPipeline"])
def test_diffusers_adapter_selects_sana_pipeline_utils(pipeline_class_name):
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import (
        SanaVideoPipelineUtils,
        get_pipeline_utils,
    )

    assert isinstance(get_pipeline_utils(pipeline_class_name), SanaVideoPipelineUtils)


def test_diffusers_adapter_resolves_requested_sana_i2v_pipeline():
    from diffusers import SanaImageToVideoPipeline, SanaVideoPipeline

    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import (
        SanaVideoPipelineUtils,
        get_pipeline_utils_for_config,
        resolve_diffusers_pipeline_class,
    )

    od_config = SimpleNamespace(
        model_class_name="SanaImageToVideoPipeline",
        diffusers_pipeline_cls=SanaVideoPipeline,
    )

    assert isinstance(get_pipeline_utils_for_config(od_config), SanaVideoPipelineUtils)
    assert resolve_diffusers_pipeline_class(od_config) is SanaImageToVideoPipeline


def test_diffusers_adapter_pipeline_override_is_sana_specific():
    from diffusers import DiffusionPipeline

    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import resolve_diffusers_pipeline_class

    class OtherPipeline(DiffusionPipeline):
        pass

    od_config = SimpleNamespace(
        model_class_name="DiffusersAdapterPipeline",
        diffusers_pipeline_cls=OtherPipeline,
    )

    assert resolve_diffusers_pipeline_class(od_config) is OtherPipeline


def test_diffusers_adapter_loads_resolved_pipeline_class(monkeypatch):
    from vllm_omni.diffusion.models.diffusers_adapter import pipeline_diffusers_adapter
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import DiffusersAdapterPipeline

    loaded = []
    expected_pipeline = object()

    class ResolvedPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            loaded.append((model_id, kwargs))
            return expected_pipeline

    monkeypatch.setattr(
        pipeline_diffusers_adapter,
        "resolve_diffusers_pipeline_class",
        lambda _od_config: ResolvedPipeline,
    )
    adapter = object.__new__(DiffusersAdapterPipeline)
    adapter.od_config = SimpleNamespace(diffusers_pipeline_cls=object)

    result = adapter._load_pipeline_from_pretrained("sana-checkpoint", {"torch_dtype": torch.bfloat16})

    assert result is expected_pipeline
    assert loaded == [("sana-checkpoint", {"torch_dtype": torch.bfloat16})]


def test_diffusers_adapter_sana_i2v_override_reports_image_capability():
    from diffusers import SanaVideoPipeline

    from vllm_omni.diffusion.io_support import supports_multimodal_input

    od_config = SimpleNamespace(
        diffusion_load_format="diffusers",
        model_class_name="SanaImageToVideoPipeline",
        diffusers_pipeline_cls=SanaVideoPipeline,
    )

    assert supports_multimodal_input(od_config) == (True, False)


def test_diffusers_adapter_sana_i2v_warmup_supplies_image():
    from diffusers import SanaVideoPipeline

    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(
        diffusion_load_format="diffusers",
        model_class_name="SanaImageToVideoPipeline",
        diffusers_pipeline_cls=SanaVideoPipeline,
    )
    captured_requests = []
    engine.pre_process_func = lambda request: request

    def capture_request(request):
        captured_requests.append(request)
        return SimpleNamespace(error=None)

    engine.add_req_and_wait_for_response = capture_request

    engine._dummy_run()

    assert len(captured_requests) == 1
    prompt = captured_requests[0].prompt
    assert isinstance(prompt, dict)
    assert isinstance(prompt["multi_modal_data"]["image"], Image.Image)


def test_diffusers_adapter_disables_resolution_binning_for_warmup():
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_diffusers_adapter import DiffusersAdapterPipeline
    from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import SanaVideoPipelineUtils

    adapter = object.__new__(DiffusersAdapterPipeline)
    adapter._accept_call_kwargs = {"prompt", "frames", "use_resolution_binning", "generator"}
    adapter._pipeline_utils = SanaVideoPipelineUtils()
    adapter.od_config = SimpleNamespace(diffusers_call_kwargs={}, output_type=None)

    kwargs = adapter._build_call_kwargs(
        _make_request_batch(
            "dummy run",
            request_id=DUMMY_DIFFUSION_REQUEST_ID,
            generator_device="cpu",
        )
    )

    assert kwargs["use_resolution_binning"] is False
    assert kwargs["frames"] == 1


def test_pipeline_is_torch_module_and_supports_eval():
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    pipeline = object.__new__(SanaVideoPipeline)
    nn.Module.__init__(pipeline)

    assert isinstance(pipeline, nn.Module)
    assert pipeline.eval() is pipeline
    assert pipeline.training is False


@pytest.mark.parametrize(
    ("vae_scale", "patch_size", "valid_size", "invalid_size", "alignment"),
    [
        (8, (1, 2, 2), 624, 632, 16),
        (32, (1, 1, 1), 704, 712, 32),
    ],
)
def test_check_inputs_uses_variant_spatial_alignment(vae_scale, patch_size, valid_size, invalid_size, alignment):
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    pipeline = object.__new__(SanaVideoPipeline)
    pipeline.vae_scale_factor_spatial = vae_scale
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(patch_size=patch_size))

    pipeline.check_inputs(prompt="test", height=valid_size, width=valid_size)
    with pytest.raises(ValueError, match=f"divisible by {alignment}"):
        pipeline.check_inputs(prompt="test", height=invalid_size, width=valid_size)


@pytest.mark.parametrize(
    ("requested_output_type", "expected_internal_output_type"),
    [
        (None, "raw"),
        ("np", "raw"),
        ("latent", "latent"),
    ],
)
def test_forward_maps_omni_request_to_sana_generation_args(requested_output_type, expected_internal_output_type):
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    pipeline = object.__new__(SanaVideoPipeline)
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(sample_size=30))
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(frames=torch.zeros(1, 3, 9, 192, 320))

    pipeline._generate = fake_generate
    req = _make_request_batch(
        {"prompt": "a robot walks", "negative_prompt": "blurry"},
        height=192,
        width=320,
        num_frames=9,
        num_inference_steps=2,
        guidance_scale=4.5,
        seed=42,
        output_type=requested_output_type,
        extra_args={"motion_score": 30, "use_resolution_binning": False},
    )

    output = pipeline.forward(req)

    assert output.output.shape == (1, 3, 9, 192, 320)
    assert calls[0]["prompt"] == "a robot walks motion score: 30."
    assert calls[0]["negative_prompt"] == "blurry"
    assert calls[0]["height"] == 192
    assert calls[0]["width"] == 320
    assert calls[0]["frames"] == 9
    assert calls[0]["num_inference_steps"] == 2
    assert calls[0]["guidance_scale"] == 4.5
    assert calls[0]["use_resolution_binning"] is False
    assert calls[0]["generator"].initial_seed() == 42
    assert calls[0]["output_type"] == expected_internal_output_type


def test_sana_video_t2v_omitted_num_frames_uses_model_default():
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    pipeline = object.__new__(SanaVideoPipeline)
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(sample_size=30))
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(frames=torch.zeros(1, 3, 81, 2, 2))

    pipeline._generate = fake_generate
    req = _make_request_batch(
        "a robot walks",
        height=192,
        width=320,
        num_inference_steps=1,
        generator_device="cpu",
    )

    assert req.sampling_params.num_frames == 1
    output = pipeline.forward(req)

    assert output.output.shape == (1, 3, 81, 2, 2)
    assert calls[0]["frames"] == 81


def test_sana_video_t2v_dummy_run_keeps_single_frame():
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    pipeline = object.__new__(SanaVideoPipeline)
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(sample_size=30))
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(frames=torch.zeros(1, 3, 1, 192, 320))

    pipeline._generate = fake_generate

    output = pipeline.forward(
        _make_request_batch(
            "dummy run",
            request_id=DUMMY_DIFFUSION_REQUEST_ID,
            height=192,
            width=320,
            num_frames=1,
            num_inference_steps=1,
            generator_device="cpu",
        )
    )

    assert output.output.shape == (1, 3, 1, 192, 320)
    assert calls[0]["frames"] == 1


@pytest.mark.parametrize(
    ("sample_size", "expected_height", "expected_width"),
    [
        (30, 480, 832),
        (22, 704, 1280),
    ],
)
def test_t2v_uses_variant_default_resolution(sample_size, expected_height, expected_width):
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    pipeline = object.__new__(SanaVideoPipeline)
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(sample_size=sample_size))
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(frames=torch.zeros(1, 3, 9, expected_height, expected_width))

    pipeline._generate = fake_generate
    output = pipeline.forward(
        _make_request_batch(
            "a robot walks",
            num_frames=9,
            num_inference_steps=1,
            guidance_scale=1.0,
            seed=42,
            generator_device="cpu",
        )
    )

    assert output.output.shape == (1, 3, 9, expected_height, expected_width)
    assert calls[0]["height"] == expected_height
    assert calls[0]["width"] == expected_width


def test_forward_requires_exactly_one_nonempty_prompt():
    from vllm_omni.diffusion.models.sana_video import SanaVideoPipeline

    pipeline = object.__new__(SanaVideoPipeline)

    with pytest.raises(ValueError, match="Prompt is required"):
        pipeline.forward(_make_request_batch(""))

    first = OmniDiffusionRequest(
        prompt="first",
        sampling_params=OmniDiffusionSamplingParams(),
        request_id="first",
    )
    second = OmniDiffusionRequest(
        prompt="second",
        sampling_params=OmniDiffusionSamplingParams(),
        request_id="second",
    )
    with pytest.raises(ValueError, match="exactly one prompt"):
        pipeline.forward(DiffusionRequestBatch([first, second]))
