# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.joy_image import (
    pipeline_joy_image_edit as joy_pipeline_module,
)
from vllm_omni.diffusion.models.joy_image.cfg_parallel import JoyImageEditCFGParallelMixin
from vllm_omni.diffusion.models.joy_image.joy_image_edit_transformer import (
    JoyImageAttention,
    JoyImageEditTransformer3DModel,
)
from vllm_omni.diffusion.models.joy_image.pipeline_joy_image_edit import (
    JOY_MAX_IMAGE_SEQ_LEN,
    JOY_PROMPT_TEMPLATE,
    JOY_VISION_TOKEN,
    JoyImageEditPipeline,
    _cast_floating_model_inputs,
    _format_qwen_multimodal_prompt,
    _get_transformer_config_kwargs_from_od_config,
    _raise_if_unsupported_hsdp,
    _resize_center_crop,
    _should_defer_component_device_placement,
    get_joy_image_edit_pre_process_func,
)
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _torch_sdpa_diffusion_config():
    return SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
        parallel_config=SimpleNamespace(ring_degree=1),
        diffusion_kv_cache_dtype=None,
        diffusion_kv_cache_skip_step_indices=None,
        diffusion_kv_cache_skip_layer_indices=None,
    )


def _make_joy_attention(*, dtype: torch.dtype = torch.float32) -> JoyImageAttention:
    with set_current_diffusion_config(_torch_sdpa_diffusion_config()):
        attention = JoyImageAttention(
            dim=32,
            num_attention_heads=4,
            attention_head_dim=8,
            prefix="double_blocks.0.attn",
        )
    return attention.to(dtype=dtype)


def _reference_joy_attention_output(
    attention: JoyImageAttention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    img_query, img_key, img_value = attention._stream_qkv(
        attention.img_attn_qkv,
        attention.img_attn_q_norm,
        attention.img_attn_k_norm,
        hidden_states,
    )
    txt_query, txt_key, txt_value = attention._stream_qkv(
        attention.txt_attn_qkv,
        attention.txt_attn_q_norm,
        attention.txt_attn_k_norm,
        encoder_hidden_states,
    )
    joint_query = torch.cat([img_query, txt_query], dim=1)
    joint_key = torch.cat([img_key, txt_key], dim=1)
    joint_value = torch.cat([img_value, txt_value], dim=1)

    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(torch.bool)[:, None, None, :]

    joint_hidden_states = F.scaled_dot_product_attention(
        joint_query.transpose(1, 2),
        joint_key.transpose(1, 2),
        joint_value.transpose(1, 2),
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / (attention.head_dim**0.5),
    )
    joint_hidden_states = joint_hidden_states.transpose(1, 2).flatten(2, 3).to(joint_query.dtype)

    image_seq_len = hidden_states.shape[1]
    return (
        attention.img_attn_proj(joint_hidden_states[:, :image_seq_len]),
        attention.txt_attn_proj(joint_hidden_states[:, image_seq_len:]),
    )


def _assert_attention_outputs_close(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
    *,
    dtype: torch.dtype,
) -> None:
    atol = rtol = 5e-3 if dtype is torch.bfloat16 else 1e-5
    max_abs = max(
        (actual_item - expected_item).abs().max().item()
        for actual_item, expected_item in zip(actual, expected)
    )
    assert max_abs <= atol
    for actual_item, expected_item in zip(actual, expected):
        torch.testing.assert_close(actual_item, expected_item, atol=atol, rtol=rtol)


def _write_model_configs(tmp_path):
    (tmp_path / "vae").mkdir()
    (tmp_path / "transformer").mkdir()
    (tmp_path / "vae" / "config.json").write_text(
        json.dumps(
            {
                "scale_factor_spatial": 8,
                "z_dim": 16,
                "latents_mean": [0.0] * 16,
                "latents_std": [1.0] * 16,
            }
        )
    )
    (tmp_path / "transformer" / "config.json").write_text(
        json.dumps(
            {
                "patch_size": [1, 2, 2],
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_layers": 1,
                "in_channels": 4,
                "out_channels": 4,
                "text_dim": 16,
            }
        )
    )
    return SimpleNamespace(
        model=str(tmp_path),
        model_class_name="JoyImageEditPipeline",
    )


def _make_params(**overrides):
    guidance_scale = overrides.pop("guidance_scale", 0.0)
    values = {
        "height": None,
        "width": None,
        "guidance_scale": guidance_scale or 1.0,
        "guidance_scale_provided": bool(guidance_scale),
        "true_cfg_scale": None,
        "num_outputs_per_prompt": 1,
        "num_inference_steps": None,
        "generator": None,
        "seed": None,
        "cfg_normalize": False,
        "guidance_scale_2": None,
        "do_classifier_free_guidance": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_request(*, prompt=None, params=None):
    prompt = prompt or {
        "prompt": "make it brighter",
        "multi_modal_data": {"image": Image.new("RGB", (2048, 1024), color="white")},
    }
    params = params or _make_params()
    return SimpleNamespace(
        prompts=[prompt],
        sampling_params=params,
        request_id="joy-test",
    )


def _make_single_omni_request(*, prompt=None, params=None):
    prompt = prompt or {
        "prompt": "make it brighter",
        "multi_modal_data": {"image": Image.new("RGB", (2048, 1024), color="white")},
    }
    params = params or _make_params()
    return OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=params,
        request_id="joy-test",
    )


def test_joy_registry_and_metadata_entries(tmp_path):
    from vllm_omni.diffusion.registry import (
        DiffusionModelRegistry,
        get_diffusion_post_process_func,
        get_diffusion_pre_process_func,
    )

    od_config = _write_model_configs(tmp_path)
    model_class = DiffusionModelRegistry._try_load_model_cls("JoyImageEditPipeline")
    assert model_class is JoyImageEditPipeline
    assert callable(get_diffusion_pre_process_func(od_config))
    postprocess = get_diffusion_post_process_func(od_config)
    assert callable(postprocess)
    assert len(postprocess(torch.zeros(1, 3, 2, 2))) == 1

    metadata = get_diffusion_model_metadata("JoyImageEditPipeline")
    assert metadata.supports_multimodal_inputs is True
    assert metadata.max_multimodal_image_inputs == 1


def test_get_model_path_downloads_only_runtime_configs(monkeypatch):
    captured = {}

    def fake_download(model_name, revision, allow_patterns, **kwargs):
        captured.update(
            {
                "model_name": model_name,
                "revision": revision,
                "allow_patterns": allow_patterns,
                "kwargs": kwargs,
            }
        )
        return "/tmp/joy-configs"

    monkeypatch.setattr(
        joy_pipeline_module,
        "download_weights_from_hf_specific",
        fake_download,
    )

    assert joy_pipeline_module._get_model_path("repo/joy") == "/tmp/joy-configs"
    assert captured == {
        "model_name": "repo/joy",
        "revision": None,
        "allow_patterns": ["vae/config.json", "transformer/config.json"],
        "kwargs": {"require_all": True},
    }


def test_transformer_config_kwargs_from_od_config_filters_metadata():
    class FakeTransformerConfig:
        def to_dict(self):
            return {
                "_class_name": "JoyImageEditTransformer3DModel",
                "_diffusers_version": "0.38.0",
                "hidden_size": 32,
                "num_layers": 1,
                "num_attention_heads": 4,
            }

    od_config = SimpleNamespace(tf_model_config=FakeTransformerConfig())

    assert _get_transformer_config_kwargs_from_od_config(od_config) == {
        "hidden_size": 32,
        "num_layers": 1,
        "num_attention_heads": 4,
    }


def test_defer_component_device_placement_for_offload_only():
    assert (
        _should_defer_component_device_placement(
            SimpleNamespace(
                enable_cpu_offload=False,
                enable_layerwise_offload=False,
                parallel_config=SimpleNamespace(use_hsdp=False),
            )
        )
        is False
    )
    assert _should_defer_component_device_placement(SimpleNamespace(enable_cpu_offload=True)) is True
    assert _should_defer_component_device_placement(SimpleNamespace(enable_layerwise_offload=True)) is True
    assert (
        _should_defer_component_device_placement(SimpleNamespace(parallel_config=SimpleNamespace(use_hsdp=True)))
        is False
    )


def test_joy_hsdp_is_explicitly_unsupported():
    with pytest.raises(ValueError, match="does not support HSDP"):
        _raise_if_unsupported_hsdp(SimpleNamespace(parallel_config=SimpleNamespace(use_hsdp=True)))

    _raise_if_unsupported_hsdp(SimpleNamespace(parallel_config=SimpleNamespace(use_hsdp=False)))
    _raise_if_unsupported_hsdp(SimpleNamespace())


def test_component_discovery_treats_vae_as_offload_peer():
    pipeline = object.__new__(JoyImageEditPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = torch.nn.Linear(1, 1)
    pipeline.text_encoder = torch.nn.Linear(1, 1)
    pipeline.vae = torch.nn.Linear(1, 1)

    modules = ModuleDiscovery.discover(pipeline)

    assert modules.dit_names == ["transformer"]
    assert modules.encoder_names == ["text_encoder", "vae"]
    assert modules.vaes == []


def test_preprocess_requires_exactly_one_image(tmp_path):
    od_config = _write_model_configs(tmp_path)
    preprocess = get_joy_image_edit_pre_process_func(od_config)
    no_image = _make_request(prompt={"prompt": "x", "multi_modal_data": {}})
    with pytest.raises(ValueError, match="exactly one input image"):
        preprocess(no_image)

    multi_image = _make_request(
        prompt={
            "prompt": "x",
            "multi_modal_data": {"image": [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]},
        }
    )
    with pytest.raises(ValueError, match="exactly one image"):
        preprocess(multi_image)


def test_preprocess_rejects_batched_prompts_and_partial_size(tmp_path):
    od_config = _write_model_configs(tmp_path)
    preprocess = get_joy_image_edit_pre_process_func(od_config)
    batched = _make_request()
    batched.prompts.append(batched.prompts[0])
    with pytest.raises(ValueError, match="exactly one prompt"):
        preprocess(batched)

    partial_size = _make_request(params=_make_params(height=512, width=None))
    with pytest.raises(ValueError, match="both `height` and `width`"):
        preprocess(partial_size)


def test_preprocess_uses_diffusers_bucket_for_default_image_size(tmp_path):
    od_config = _write_model_configs(tmp_path)
    preprocess = get_joy_image_edit_pre_process_func(od_config)

    request = preprocess(_make_request())
    prompt = request.prompts[0]
    info = prompt["additional_information"]
    height = info["height"]
    width = info["width"]

    assert (height, width) == (704, 1408)
    assert (height // 16) * (width // 16) <= JOY_MAX_IMAGE_SEQ_LEN
    assert info["image_tensor"].shape == (1, 3, 1, height, width)
    assert info["original_size"] == (2048, 1024)
    assert info["resized_size"] == (width, height)
    assert request.sampling_params.height == height
    assert request.sampling_params.width == width


def test_preprocess_accepts_single_omni_request_prompt(tmp_path):
    od_config = _write_model_configs(tmp_path)
    preprocess = get_joy_image_edit_pre_process_func(od_config)

    request = preprocess(_make_single_omni_request())
    prompt = request.prompt
    info = prompt["additional_information"]

    assert isinstance(prompt, dict)
    assert not hasattr(request, "prompts")
    assert info["image_tensor"].shape == (1, 3, 1, 704, 1408)
    assert request.sampling_params.height == 704
    assert request.sampling_params.width == 1408


def test_resize_center_crop_crops_instead_of_stretching():
    image = Image.new("RGB", (4, 2))
    pixels = image.load()
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 255),
    ]
    for x, color in enumerate(colors):
        for y in range(2):
            pixels[x, y] = color

    cropped = _resize_center_crop(image, height=2, width=2)
    cropped_pixels = cropped.load()

    assert cropped.size == (2, 2)
    assert [[cropped_pixels[x, y] for x in range(2)] for y in range(2)] == [
        [(0, 255, 0), (0, 0, 255)],
        [(0, 255, 0), (0, 0, 255)],
    ]


def test_joy_cfg_normalize_uses_channel_dimension():
    mixin = JoyImageEditCFGParallelMixin()
    noise_pred = torch.tensor([[[[[[3.0, 4.0]]], [[[4.0, 3.0]]]]]])
    comb_pred = torch.tensor([[[[[[6.0, 8.0]]], [[[8.0, 6.0]]]]]])

    normalized = mixin.cfg_normalize_function(noise_pred, comb_pred)
    expected = comb_pred * (
        torch.norm(noise_pred, dim=2, keepdim=True) / torch.norm(comb_pred, dim=2, keepdim=True).clamp_min(1e-6)
    )

    assert torch.equal(normalized, expected)


def test_joy_diffuse_preserves_scheduler_timestep_dtype_for_time_embedding():
    class FakeProgressBar:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def update(self):
            pass

    class FakeScheduler:
        def set_begin_index(self, index):
            self.begin_index = index

    class FakeMixin(JoyImageEditCFGParallelMixin):
        def __init__(self):
            self.scheduler = FakeScheduler()
            self.captured_timestep_dtype = None
            self._interrupt = False

        @property
        def interrupt(self):
            return self._interrupt

        def progress_bar(self, total):
            return FakeProgressBar()

        def predict_noise_maybe_with_cfg(self, **kwargs):
            self.captured_timestep_dtype = kwargs["positive_kwargs"]["timestep"].dtype
            return torch.zeros_like(kwargs["positive_kwargs"]["hidden_states"])

        def scheduler_step_maybe_with_cfg(self, noise_pred, timestep, latents, do_true_cfg):
            return latents

    mixin = FakeMixin()
    latents = torch.zeros(1, 2, 4, 1, 4, 4, dtype=torch.bfloat16)
    image_latents = torch.zeros(1, 1, 4, 1, 4, 4, dtype=torch.bfloat16)

    mixin.diffuse(
        latents=latents,
        image_latents=image_latents,
        prompt_embeds=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        prompt_embeds_mask=torch.ones(1, 2, dtype=torch.long),
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        timesteps=torch.tensor([999.125], dtype=torch.float32),
        do_true_cfg=False,
        true_cfg_scale=1.0,
    )

    assert mixin.captured_timestep_dtype == torch.float32


def test_format_qwen_multimodal_prompt_matches_diffusers_image_placeholder_replacement():
    formatted = _format_qwen_multimodal_prompt("make it brighter")

    assert formatted == (f"<|im_start|>user\n{JOY_VISION_TOKEN}make it brighter<|im_end|>\n")


def test_preprocess_maps_explicit_size_to_nearest_diffusers_bucket(tmp_path):
    od_config = _write_model_configs(tmp_path)
    preprocess = get_joy_image_edit_pre_process_func(od_config)
    params = _make_params(height=2048, width=2048)

    request = preprocess(_make_request(params=params))

    assert request.sampling_params.height == 1024
    assert request.sampling_params.width == 1024


def test_guidance_scale_alias_only_when_true_cfg_absent():
    request = _make_request(params=_make_params(guidance_scale=3.5))
    assert JoyImageEditPipeline.resolve_effective_true_cfg_scale(request) == 3.5

    canonical = _make_request(params=_make_params(true_cfg_scale=4.0))
    assert JoyImageEditPipeline.resolve_effective_true_cfg_scale(canonical) == 4.0

    default_guidance = _make_request(params=_make_params(guidance_scale=1.0, true_cfg_scale=4.0))
    assert JoyImageEditPipeline.resolve_effective_true_cfg_scale(default_guidance) == 4.0

    matching = _make_request(params=_make_params(guidance_scale=4.0, true_cfg_scale=4.0))
    assert JoyImageEditPipeline.resolve_effective_true_cfg_scale(matching) == 4.0

    conflict = _make_request(params=_make_params(guidance_scale=3.0, true_cfg_scale=4.0))
    with pytest.raises(ValueError, match="compatibility alias"):
        JoyImageEditPipeline.resolve_effective_true_cfg_scale(conflict)


def test_pad_prompt_embeds_keeps_last_tokens_and_builds_mask():
    first = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    second = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    prompt_embeds, prompt_mask = JoyImageEditPipeline._pad_prompt_embeds(
        [first, second],
        max_sequence_length=3,
    )

    assert prompt_embeds.shape == (2, 3, 4)
    assert torch.equal(prompt_embeds[0], first[-3:])
    assert torch.equal(prompt_embeds[1, :2], second)
    assert torch.equal(prompt_embeds[1, 2], torch.zeros(4))
    assert torch.equal(
        prompt_mask,
        torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="greater than 0"):
        JoyImageEditPipeline._pad_prompt_embeds([first], max_sequence_length=0)


@pytest.mark.parametrize(
    ("prompt_embeds", "prompt_embeds_mask", "match"),
    [
        (
            torch.zeros(2, 4),
            torch.ones(2, 4, dtype=torch.long),
            "prompt_embeds must be a 3D tensor",
        ),
        (
            torch.zeros(2, 4, 8),
            torch.ones(2, 4, 1, dtype=torch.long),
            "prompt_embeds_mask must be a 2D tensor",
        ),
        (
            torch.zeros(2, 4, 8),
            torch.ones(3, 4, dtype=torch.long),
            "same batch size",
        ),
        (
            torch.zeros(2, 4, 8),
            torch.ones(2, 5, dtype=torch.long),
            "same sequence length",
        ),
    ],
)
def test_encode_prompt_validates_precomputed_embeds_and_mask_shapes(
    prompt_embeds,
    prompt_embeds_mask,
    match,
):
    pipeline = object.__new__(JoyImageEditPipeline)

    with pytest.raises(ValueError, match=match):
        JoyImageEditPipeline.encode_prompt(
            pipeline,
            "prompt",
            Image.new("RGB", (16, 16)),
            num_images_per_prompt=1,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
        )


def test_encode_prompt_validation_uses_negative_parameter_names():
    pipeline = object.__new__(JoyImageEditPipeline)

    with pytest.raises(ValueError, match="negative_prompt_embeds_mask must be a 2D tensor"):
        JoyImageEditPipeline.encode_prompt(
            pipeline,
            "",
            Image.new("RGB", (16, 16)),
            num_images_per_prompt=1,
            prompt_embeds=torch.zeros(1, 4, 8),
            prompt_embeds_mask=torch.ones(1, 4, 1, dtype=torch.long),
            embeds_name="negative_prompt_embeds",
            mask_name="negative_prompt_embeds_mask",
        )


def test_last_layer_capture_passes_mm_token_type_ids():
    class FakeDecoderLayer(torch.nn.Module):
        def forward(self, hidden_states):
            return (hidden_states + 1.0,)

    class FakeLanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeDecoderLayer()])

    class FakeTextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(language_model=FakeLanguageModel())
            self.calls = []

        def forward(self, **kwargs):
            self.calls.append(kwargs)
            hidden_states = torch.zeros(1, 2, 4)
            self.model.language_model.layers[-1](hidden_states)

    pipeline = object.__new__(JoyImageEditPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.text_encoder = FakeTextEncoder()
    mm_token_type_ids = torch.tensor([[0, 1]], dtype=torch.long)
    model_inputs = SimpleNamespace(
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        pixel_values=torch.zeros(1, 3, 16, 16),
        image_grid_thw=torch.tensor([[1, 1, 1]], dtype=torch.long),
        mm_token_type_ids=mm_token_type_ids,
        extra_processor_field=torch.tensor([7], dtype=torch.long),
    )

    hidden_states = JoyImageEditPipeline._get_last_layer_pre_norm_hidden(
        pipeline,
        model_inputs,
    )

    assert torch.equal(hidden_states, torch.ones(1, 2, 4))
    assert pipeline.text_encoder.calls[0]["mm_token_type_ids"] is mm_token_type_ids
    assert torch.equal(pipeline.text_encoder.calls[0]["extra_processor_field"], torch.tensor([7], dtype=torch.long))
    assert pipeline.text_encoder.calls[0]["output_hidden_states"] is False


def test_cast_floating_model_inputs_only_casts_pixel_tensors():
    class FakeModelInputs(dict):
        def __getattr__(self, name):
            return self[name]

    model_inputs = FakeModelInputs(
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        pixel_values=torch.zeros(1, 3, dtype=torch.float32),
        image_grid_thw=torch.tensor([[1, 1, 1]], dtype=torch.long),
    )

    _cast_floating_model_inputs(model_inputs, torch.bfloat16)

    assert model_inputs.input_ids.dtype == torch.long
    assert model_inputs.attention_mask.dtype == torch.long
    assert model_inputs.pixel_values.dtype == torch.bfloat16
    assert model_inputs.image_grid_thw.dtype == torch.long


def test_qwen_prompt_embeds_casts_processor_pixel_values_to_encoder_dtype():
    class FakeModelInputs(dict):
        def __init__(self):
            super().__init__(
                input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
                attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.long),
                pixel_values=torch.zeros(1, 3, dtype=torch.float32),
                image_grid_thw=torch.tensor([[1, 1, 1]], dtype=torch.long),
            )

        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

        def to(self, device):
            for key, value in list(self.items()):
                if isinstance(value, torch.Tensor):
                    self[key] = value.to(device)
            return self

    class FakeProcessor:
        def __init__(self):
            self.model_inputs = FakeModelInputs()

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return self.model_inputs

    class FakeTextEncoder:
        dtype = torch.bfloat16

    pipeline = object.__new__(JoyImageEditPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.processor = FakeProcessor()
    pipeline.text_encoder = FakeTextEncoder()
    pipeline.prompt_template_encode = JOY_PROMPT_TEMPLATE
    pipeline.prompt_template_encode_start_idx = 0
    pipeline.tokenizer_max_length = 8
    captured = {}

    def fake_get_hidden(model_inputs):
        captured["pixel_dtype"] = model_inputs.pixel_values.dtype
        captured["attention_mask_dtype"] = model_inputs.attention_mask.dtype
        return torch.ones(1, 3, 4, dtype=torch.float32)

    pipeline._get_last_layer_pre_norm_hidden = fake_get_hidden

    prompt_embeds, prompt_mask = JoyImageEditPipeline._get_qwen_prompt_embeds(
        pipeline,
        "make it brighter",
        Image.new("RGB", (16, 16)),
    )

    assert captured == {
        "pixel_dtype": torch.bfloat16,
        "attention_mask_dtype": torch.long,
    }
    text = pipeline.processor.kwargs["text"][0]
    assert text.startswith("<|im_start|>system\n \\nDescribe")
    assert "Describe the image by detailing the color, shape, size, texture" in text
    assert (f"<|im_start|>user\n{JOY_VISION_TOKEN}make it brighter<|im_end|>\n") in text
    assert text.endswith("<|im_start|>assistant\n")
    assert prompt_embeds.dtype == torch.bfloat16
    assert prompt_mask.dtype == torch.long


def test_prepare_latents_casts_vae_encoded_latents_to_requested_dtype():
    pipeline = object.__new__(JoyImageEditPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.latent_channels = 4
    pipeline.vae_scale_factor = 8

    def fake_encode(image, generator):
        assert image.dtype == torch.bfloat16
        return torch.ones(1, 4, 1, 8, 8, dtype=torch.float32)

    pipeline._encode_vae_image = fake_encode

    latents, image_latents = JoyImageEditPipeline._prepare_latents(
        pipeline,
        image=torch.zeros(1, 3, 1, 64, 64, dtype=torch.float32),
        batch_size=1,
        num_channels_latents=4,
        height=64,
        width=64,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        generator=None,
    )

    assert latents.dtype == torch.bfloat16
    assert image_latents.dtype == torch.bfloat16


def test_encode_vae_image_does_not_forward_noise_generator_to_posterior_sample():
    class FakeLatentDist:
        def __init__(self):
            self.received_generator = object()

        def sample(self, generator=None):
            self.received_generator = generator
            return torch.zeros(1, 4, 1, 8, 8)

    class FakeVAE:
        def __init__(self):
            self.config = SimpleNamespace(
                latents_mean=[0.0, 0.0, 0.0, 0.0],
                latents_std=[1.0, 1.0, 1.0, 1.0],
            )
            self.latent_dist = FakeLatentDist()

        def encode(self, image):
            return SimpleNamespace(latent_dist=self.latent_dist)

    pipeline = object.__new__(JoyImageEditPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.vae = FakeVAE()
    pipeline.latent_channels = 4

    generator = torch.Generator(device="cpu").manual_seed(123)
    JoyImageEditPipeline._encode_vae_image(
        pipeline,
        torch.zeros(1, 3, 1, 64, 64),
        generator,
    )

    assert pipeline.vae.latent_dist.received_generator is None


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_joy_attention_native_sdpa_matches_torch_sdpa_without_mask(dtype):
    torch.manual_seed(123)
    device = torch.device("cuda")
    attention = _make_joy_attention(dtype=dtype).to(device=device)
    hidden_states = torch.randn(2, 3, 32, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(2, 5, 32, device=device, dtype=dtype)

    actual = attention(hidden_states, encoder_hidden_states)
    expected = _reference_joy_attention_output(attention, hidden_states, encoder_hidden_states)

    assert isinstance(attention.attn, Attention)
    assert attention.attn.role == "joy_image.joint"
    assert attention.attn.role_category == "self"
    assert attention.attn.qkv_layout == "BSND"
    _assert_attention_outputs_close(actual, expected, dtype=dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_joy_attention_native_sdpa_matches_torch_sdpa_with_padding_mask(dtype):
    torch.manual_seed(456)
    device = torch.device("cuda")
    attention = _make_joy_attention(dtype=dtype).to(device=device)
    hidden_states = torch.randn(2, 3, 32, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(2, 5, 32, device=device, dtype=dtype)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ],
        device=device,
        dtype=torch.bool,
    )

    actual = attention(hidden_states, encoder_hidden_states, attention_mask=attention_mask)
    expected = _reference_joy_attention_output(
        attention,
        hidden_states,
        encoder_hidden_states,
        attention_mask=attention_mask,
    )

    _assert_attention_outputs_close(actual, expected, dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_transformer_shape_and_masked_forward():
    device = torch.device("cuda")
    transformer = JoyImageEditTransformer3DModel(
        in_channels=4,
        out_channels=4,
        hidden_size=32,
        text_dim=16,
        num_layers=1,
        num_attention_heads=4,
        patch_size=(1, 2, 2),
    ).to(device=device)
    hidden_states = torch.randn(2, 2, 4, 1, 4, 4, device=device)
    encoder_hidden_states = torch.randn(2, 5, 16, device=device)
    encoder_hidden_states_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], device=device)

    output = transformer(
        hidden_states=hidden_states,
        timestep=torch.tensor([1.0, 2.0], device=device),
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        return_dict=False,
    )[0]

    assert output.shape == hidden_states.shape
    assert isinstance(transformer.double_blocks[0].attn.attn, Attention)


def test_transformer_from_config_file_loads_checkpoint_config(tmp_path):
    _write_model_configs(tmp_path)

    transformer = JoyImageEditTransformer3DModel.from_config_file(tmp_path / "transformer" / "config.json")

    assert transformer.patch_size == (1, 2, 2)
    assert transformer.hidden_size == 32
    assert transformer.num_attention_heads == 4
    assert transformer.num_layers == 1
    assert transformer.text_dim == 16
    assert transformer.theta == 256
    assert len(transformer.double_blocks) == 1


def test_transformer_load_weights_maps_diffusers_prefix():
    transformer = JoyImageEditTransformer3DModel(
        in_channels=4,
        out_channels=4,
        hidden_size=32,
        text_dim=16,
        num_layers=1,
        num_attention_heads=4,
        patch_size=(1, 2, 2),
    )
    weight = torch.randn_like(transformer.img_in.weight)

    loaded = transformer.load_weights([("transformer.patch_embedding.weight", weight)])

    assert loaded == {"img_in.weight"}
    assert torch.equal(transformer.img_in.weight, weight)

    text_weight = torch.randn_like(transformer.condition_embedder.text_embedder.linear_1.weight)
    loaded = transformer.load_weights([("transformer.condition_embedder.text_embedder.linear_1.weight", text_weight)])

    assert loaded == {"condition_embedder.text_embedder.linear_1.weight"}
    assert torch.equal(transformer.condition_embedder.text_embedder.linear_1.weight, text_weight)


def test_transformer_load_weights_warns_for_unknown_weight(caplog):
    transformer = JoyImageEditTransformer3DModel(
        in_channels=4,
        out_channels=4,
        hidden_size=32,
        text_dim=16,
        num_layers=1,
        num_attention_heads=4,
        patch_size=(1, 2, 2),
    )

    caplog.set_level(logging.WARNING)
    loaded = transformer.load_weights([("transformer.unmapped.weight", torch.empty(1))])

    assert loaded == set()
    assert "Skipping JoyAI-Image-Edit transformer weight transformer.unmapped.weight" in caplog.text
    assert "checkpoint mismatch" in caplog.text


def test_latent_normalization_round_trip():
    pipeline = object.__new__(JoyImageEditPipeline)
    pipeline.latent_channels = 4
    pipeline.vae = SimpleNamespace(
        config=SimpleNamespace(
            latents_mean=[0.1, -0.2, 0.3, -0.4],
            latents_std=[0.5, 0.75, 1.25, 2.0],
        )
    )
    latents = torch.randn(2, 4, 1, 2, 2)

    latents_mean, latents_std = JoyImageEditPipeline._latent_stats(
        pipeline,
        latents.device,
        latents.dtype,
    )
    normalized = (latents - latents_mean) / latents_std

    assert torch.allclose(normalized * latents_std + latents_mean, latents)


def test_decode_latents_unnormalizes_and_selects_target_slot():
    class FakeVAE:
        config = SimpleNamespace(
            latents_mean=[0.1, -0.2, 0.3, -0.4],
            latents_std=[0.5, 0.75, 1.25, 2.0],
        )

        def decode(self, latents, return_dict=False):
            return (latents,)

    pipeline = object.__new__(JoyImageEditPipeline)
    pipeline.latent_channels = 4
    pipeline.vae = FakeVAE()
    latents = torch.zeros(1, 2, 4, 1, 2, 2)
    latents[:, -1] = 2.0

    decoded = JoyImageEditPipeline._decode_latents(pipeline, latents)

    latents_mean, latents_std = JoyImageEditPipeline._latent_stats(
        pipeline,
        latents.device,
        latents.dtype,
    )
    assert decoded.shape == (1, 4, 2, 2)
    assert torch.equal(decoded, (latents[:, -1] * latents_std + latents_mean)[:, :, 0])


def test_prepare_latents_stacks_reference_first_target_last():
    pipeline = object.__new__(JoyImageEditPipeline)
    pipeline.vae_scale_factor = 8
    pipeline.latent_channels = 4
    image_latents = torch.ones(1, 4, 1, 2, 2)
    noise_latents = torch.zeros(1, 1, 4, 1, 2, 2)
    pipeline._encode_vae_image = lambda image, generator: image_latents

    latents, reference_latents = JoyImageEditPipeline._prepare_latents(
        pipeline,
        image=torch.zeros(1, 3, 1, 16, 16),
        batch_size=1,
        num_channels_latents=4,
        height=16,
        width=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=None,
        latents=noise_latents,
    )

    assert latents.shape == (1, 2, 4, 1, 2, 2)
    assert torch.equal(latents[:, :1], reference_latents)
    assert torch.equal(latents[:, -1:], noise_latents)


def test_prepare_latents_validates_generator_and_latent_shapes():
    pipeline = object.__new__(JoyImageEditPipeline)
    pipeline.vae_scale_factor = 8
    pipeline.latent_channels = 4
    image_latents = torch.ones(1, 4, 1, 2, 2)
    pipeline._encode_vae_image = lambda image, generator: image_latents

    with pytest.raises(ValueError, match="Generator list length"):
        JoyImageEditPipeline._prepare_latents(
            pipeline,
            image=torch.zeros(1, 3, 1, 16, 16),
            batch_size=2,
            num_channels_latents=4,
            height=16,
            width=16,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=[torch.Generator()],
        )

    with pytest.raises(ValueError, match="noise latents must have shape"):
        JoyImageEditPipeline._prepare_latents(
            pipeline,
            image=torch.zeros(1, 3, 1, 16, 16),
            batch_size=1,
            num_channels_latents=4,
            height=16,
            width=16,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=None,
            latents=torch.zeros(1, 1, 4, 1, 1, 2),
        )


def test_prepare_latents_validates_image_latent_shape():
    pipeline = object.__new__(JoyImageEditPipeline)
    pipeline.vae_scale_factor = 8
    pipeline.latent_channels = 4
    pipeline._encode_vae_image = lambda image, generator: torch.ones(1, 4, 2, 2)

    with pytest.raises(ValueError, match="image latents must have shape"):
        JoyImageEditPipeline._prepare_latents(
            pipeline,
            image=torch.zeros(1, 3, 1, 16, 16),
            batch_size=1,
            num_channels_latents=4,
            height=16,
            width=16,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=None,
        )


def test_diffuse_restores_reference_slots_each_step():
    class FakeScheduler:
        def set_begin_index(self, index):
            self.begin_index = index

        def step(self, noise_pred, timestep, latents, return_dict=False):
            return (latents + 1.0,)

    class FakeTransformer(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            return (torch.zeros_like(hidden_states),)

    pipeline = object.__new__(JoyImageEditPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.scheduler = FakeScheduler()
    pipeline.transformer = FakeTransformer()
    pipeline._interrupt = False
    pipeline._current_timestep = None

    latents = torch.zeros(1, 2, 4, 1, 2, 2)
    image_latents = torch.full((1, 1, 4, 1, 2, 2), 7.0)
    result = JoyImageEditPipeline.diffuse(
        pipeline,
        latents=latents,
        image_latents=image_latents,
        prompt_embeds=torch.zeros(1, 3, 8),
        prompt_embeds_mask=torch.ones(1, 3, dtype=torch.long),
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        timesteps=torch.tensor([2.0, 1.0]),
        do_true_cfg=False,
        true_cfg_scale=1.0,
    )

    assert torch.equal(result[:, :1], image_latents)
    assert torch.equal(result[:, -1:], torch.full((1, 1, 4, 1, 2, 2), 2.0))


def test_forward_synthesizes_empty_negative_prompt_for_cfg():
    class FakeScheduler:
        def set_timesteps(self, num_inference_steps, device):
            self.num_inference_steps = num_inference_steps
            self.timesteps = torch.tensor([1.0], device=device)

    pipeline = object.__new__(JoyImageEditPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.latent_channels = 4
    pipeline.scheduler = FakeScheduler()
    encode_calls = []
    diffuse_calls = []
    cfg_checks = []

    def encode_prompt(prompt, image, **kwargs):
        encode_calls.append(prompt)
        return torch.zeros(1, 3, 4), torch.ones(1, 3, dtype=torch.long)

    def fake_prepare_latents(**kwargs):
        latents = torch.zeros(1, 2, 4, 1, 2, 2)
        image_latents = torch.ones(1, 1, 4, 1, 2, 2)
        return latents, image_latents

    def diffuse(**kwargs):
        diffuse_calls.append(kwargs)
        return kwargs["latents"]

    pipeline.encode_prompt = encode_prompt
    pipeline._prepare_latents = fake_prepare_latents
    pipeline.diffuse = diffuse
    pipeline._decode_latents = lambda latents: torch.zeros(1, 3, 2, 2)
    pipeline.check_cfg_parallel_validity = (
        lambda scale, has_neg_prompt: cfg_checks.append((scale, has_neg_prompt)) or True
    )

    request = _make_request(
        prompt={
            "prompt": "make it brighter",
            "additional_information": {
                "image_tensor": torch.zeros(1, 3, 1, 16, 16),
                "prompt_image": Image.new("RGB", (16, 16)),
                "height": 16,
                "width": 16,
            },
        },
        params=_make_params(true_cfg_scale=2.0),
    )

    output = JoyImageEditPipeline.forward(pipeline, request)

    assert output.output.shape == (1, 3, 2, 2)
    assert cfg_checks == [(2.0, True)]
    assert encode_calls == ["make it brighter", ""]
    assert diffuse_calls[0]["do_true_cfg"] is True
    assert diffuse_calls[0]["true_cfg_scale"] == 2.0


def test_forward_skips_decode_for_dummy_warmup_request(monkeypatch):
    class FakeScheduler:
        def set_timesteps(self, num_inference_steps, device):
            self.timesteps = torch.tensor([1.0], device=device)

    pipeline = object.__new__(JoyImageEditPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=True,
        enable_layerwise_offload=False,
        pin_cpu_memory=False,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    pipeline.transformer = torch.nn.Linear(1, 1)
    pipeline.latent_channels = 4
    pipeline.scheduler = FakeScheduler()
    pipeline.encode_prompt = lambda *args, **kwargs: (
        torch.zeros(1, 3, 4),
        torch.ones(1, 3, dtype=torch.long),
    )
    pipeline._prepare_latents = lambda **kwargs: (
        torch.zeros(1, 2, 4, 1, 2, 2),
        torch.ones(1, 1, 4, 1, 2, 2),
    )
    pipeline.diffuse = lambda **kwargs: kwargs["latents"] + 1.0
    pipeline.check_cfg_parallel_validity = lambda scale, has_neg_prompt: True

    def decode_should_not_run(latents):
        raise AssertionError("dummy warmup should not decode latents")

    pipeline._decode_latents = decode_should_not_run
    move_calls = []

    def fake_move_params(module, target_device, **kwargs):
        move_calls.append((module, target_device, kwargs))

    monkeypatch.setattr(
        joy_pipeline_module.SequentialOffloadHook,
        "_move_params",
        fake_move_params,
    )
    monkeypatch.setattr(joy_pipeline_module.current_omni_platform, "empty_cache", lambda: None)

    request = _make_request(
        prompt={
            "prompt": "dummy run",
            "additional_information": {
                "image_tensor": torch.zeros(1, 3, 1, 16, 16),
                "prompt_image": Image.new("RGB", (16, 16)),
                "height": 16,
                "width": 16,
            },
        },
        params=_make_params(true_cfg_scale=1.0),
    )
    request.request_id = DUMMY_DIFFUSION_REQUEST_ID

    output = JoyImageEditPipeline.forward(pipeline, request)

    assert output.output.shape == (1, 4, 1, 2, 2)
    assert move_calls == [
        (
            pipeline.transformer,
            torch.device("cpu"),
            {"non_blocking": False, "pin_memory": False},
        )
    ]
