# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
    _build_mammoth_config,
    _root_weight_source,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _raw_config() -> dict:
    return {
        "model_type": "mammothmoda2",
        "llm_config": {
            "model_type": "mammothmoda2_qwen2_5_vl",
            "text_config": {
                "model_type": "mammothmoda2_qwen2_5_vl_text",
                "hidden_size": 8,
                "gen_vocab_start_index": 100,
            },
        },
        "gen_vae_config": {"block_out_channels": [8, 8]},
        "gen_dit_config": {"hidden_size": 8, "in_channels": 4},
    }


def _od_config() -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model="/models/MammothModa2-Preview",
        model_class_name="MammothModa2DiTPipeline",
        tf_model_config=TransformerConfig.from_dict(_raw_config()),
    )


def test_build_mammoth_config_uses_shared_transformer_projection() -> None:
    config = _build_mammoth_config(_od_config())
    assert config.model_type == "mammothmoda2"
    assert config.get_text_config().hidden_size == 8
    assert config.gen_dit_config == {"hidden_size": 8, "in_channels": 4}


def test_build_mammoth_config_rejects_empty_shared_projection() -> None:
    config = _od_config()
    config.tf_model_config = TransformerConfig()
    with pytest.raises(ValueError, match="root checkpoint config"):
        _build_mammoth_config(config)


def test_root_weight_source_loads_combined_checkpoint_once() -> None:
    source = _root_weight_source(_od_config())
    assert source.model_or_path == "/models/MammothModa2-Preview"
    assert source.subfolder is None
    assert source.prefix == ""
    assert source.fall_back_to_pt is True


def test_root_weight_source_forwards_revision() -> None:
    config = SimpleNamespace(model="/models/MammothModa2-Preview", revision="rev-7")
    assert _root_weight_source(config).revision == "rev-7"


def test_pipeline_declares_native_components_and_single_request_mode_only() -> None:
    assert MammothModa2DiTPipeline._dit_modules == ["gen_transformer"]
    assert MammothModa2DiTPipeline._encoder_modules == ["gen_image_condition_refiner"]
    assert MammothModa2DiTPipeline._vae_modules == ["gen_vae"]
    assert MammothModa2DiTPipeline.supports_request_batch is False
    assert MammothModa2DiTPipeline.supports_step_execution is False


def test_root_weight_source_rejects_missing_model_path() -> None:
    config = SimpleNamespace(model=None, revision=None)
    with pytest.raises(ValueError, match="model path"):
        _root_weight_source(config)


def _pipeline_shell() -> MammothModa2DiTPipeline:
    pipeline = object.__new__(MammothModa2DiTPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.config = SimpleNamespace(
        llm_config=SimpleNamespace(gen_vocab_start_index=100),
        image_token_id=900,
        video_token_id=901,
        vision_start_token_id=902,
        vision_end_token_id=903,
    )
    pipeline._llm_hidden_size = 8
    return pipeline


def _batch(
    *,
    request_id: str = "req-7",
    prompt: object | None = None,
    sampling: OmniDiffusionSamplingParams | None = None,
) -> DiffusionRequestBatch:
    if prompt is None:
        prompt = {
            "prompt": "",
            "height": 32,
            "width": 48,
            "additional_information": {
                "full_hidden_states": torch.arange(32, dtype=torch.float32).reshape(4, 8),
                "full_token_ids": [10, 11, 100, 101],
                "answer_start_index": 2,
            },
        }
    if sampling is None:
        sampling = OmniDiffusionSamplingParams(
            height=32,
            width=48,
            seed=42,
            guidance_scale=4.0,
            num_inference_steps=7,
            extra_args={"cfg_range": [0.2, 0.8]},
        )
    return DiffusionRequestBatch([OmniDiffusionRequest(prompt=prompt, sampling_params=sampling, request_id=request_id)])


def test_parse_request_uses_standard_sampling_fields() -> None:
    parsed = _pipeline_shell()._parse_request(_batch())
    assert parsed.request_id == "req-7"
    assert (parsed.height, parsed.width) == (32, 48)
    assert parsed.num_inference_steps == 7
    assert parsed.text_guidance_scale == 4.0
    assert parsed.cfg_range == (0.2, 0.8)
    assert parsed.seed == 42
    assert parsed.answer_start_index == 2


def test_parse_request_prefers_legacy_sampling_overrides() -> None:
    sampling = OmniDiffusionSamplingParams(
        guidance_scale=3.0,
        num_inference_steps=5,
        seed=9,
        extra_args={"text_guidance_scale": 6.0, "num_inference_steps": 11, "cfg_range": [0.0, 0.5]},
    )
    parsed = _pipeline_shell()._parse_request(_batch(sampling=sampling))
    assert parsed.text_guidance_scale == 6.0
    assert parsed.num_inference_steps == 11
    assert parsed.cfg_range == (0.0, 0.5)


def test_parse_request_rejects_multiple_requests_and_outputs() -> None:
    pipeline = _pipeline_shell()
    batch = _batch()
    batch.requests.append(_batch(request_id="req-8").requests[0])
    with pytest.raises(ValueError, match="exactly one request"):
        pipeline._parse_request(batch)
    with pytest.raises(ValueError, match="num_outputs_per_prompt=1"):
        pipeline._parse_request(_batch(sampling=OmniDiffusionSamplingParams(num_outputs_per_prompt=2)))


@pytest.mark.parametrize(
    ("cfg_range", "message"),
    [
        ([0.5], "two values"),
        ([-0.1, 0.5], "0 <= start <= end <= 1"),
        ([0.7, 0.2], "0 <= start <= end <= 1"),
        ([0.2, 1.1], "0 <= start <= end <= 1"),
    ],
)
def test_parse_request_rejects_invalid_cfg_range(cfg_range, message) -> None:
    batch = _batch(sampling=OmniDiffusionSamplingParams(extra_args={"cfg_range": cfg_range}))
    with pytest.raises(ValueError, match=message):
        _pipeline_shell()._parse_request(batch)


def test_parse_request_requires_real_ar_conditions() -> None:
    with pytest.raises(ValueError, match="req-missing"):
        _pipeline_shell()._parse_request(_batch(request_id="req-missing", prompt={"prompt": "draw a cat"}))


def test_parse_request_rejects_hidden_state_token_count_mismatch() -> None:
    batch = _batch()
    batch.prompts[0]["additional_information"]["full_hidden_states"] = torch.zeros(3, 8)
    with pytest.raises(ValueError, match="hidden-state/token-count mismatch"):
        _pipeline_shell()._parse_request(batch)


@pytest.mark.parametrize(
    ("height", "width", "message"),
    [
        (0, 32, "Invalid image size"),
        (32, -1, "Invalid image size"),
        (30, 32, "multiples of 16"),
        (32, 31, "multiples of 16"),
    ],
)
def test_parse_request_rejects_invalid_dimensions(height, width, message) -> None:
    batch = _batch()
    batch.prompts[0].update(height=height, width=width)
    with pytest.raises(ValueError, match=message):
        _pipeline_shell()._parse_request(batch)


def test_parse_request_rejects_explicit_zero_steps() -> None:
    with pytest.raises(ValueError, match="num_inference_steps must be positive"):
        _pipeline_shell()._parse_request(_batch(sampling=OmniDiffusionSamplingParams(num_inference_steps=0)))


@pytest.mark.parametrize(("standard_steps", "expected_steps"), [(7, 7), (None, 50)])
def test_parse_request_falls_through_null_legacy_step_count(standard_steps, expected_steps) -> None:
    sampling = OmniDiffusionSamplingParams(
        num_inference_steps=standard_steps, extra_args={"num_inference_steps": None}
    )
    parsed = _pipeline_shell()._parse_request(_batch(sampling=sampling))
    assert parsed.num_inference_steps == expected_steps


def test_parse_request_synthesizes_dummy_ar_conditions() -> None:
    batch = _batch(
        request_id="dummy_req_id",
        prompt={"prompt": "dummy run"},
        sampling=OmniDiffusionSamplingParams(
            height=512, width=512, seed=1, guidance_scale=0.0, num_inference_steps=2
        ),
    )
    parsed = _pipeline_shell()._parse_request(batch)
    assert parsed.full_hidden_states.shape == (2, 8)
    assert parsed.full_token_ids == [0, 100]
    assert parsed.answer_start_index == 1


class _FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(in_channels=4)
        self.time_caption_embed = SimpleNamespace(image_embedder=None)
        self.calls = 0

    def forward(self, *, hidden_states, **kwargs):
        self.calls += 1
        return torch.zeros_like(hidden_states)


class _FakeVae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(scaling_factor=None, shift_factor=None)

    def decode(self, latents, return_dict=False):
        assert return_dict is False
        return (torch.zeros(1, 3, 32, 48, dtype=latents.dtype),)


class _FakeScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([])
        self.requested_steps = None

    def set_timesteps(self, *, num_inference_steps, device, num_tokens):
        self.requested_steps = num_inference_steps
        self.timesteps = torch.arange(num_inference_steps, device=device, dtype=torch.float32)

    def step(self, model_pred, timestep, latents, return_dict=False):
        assert return_dict is False
        return (latents - model_pred,)


def test_forward_returns_diffusion_output_with_request_sampling() -> None:
    pipeline = _pipeline_shell()
    pipeline.gen_transformer = _FakeTransformer()
    pipeline.gen_image_condition_refiner = None
    pipeline.gen_vae = _FakeVae()
    pipeline.gen_freqs_cis = torch.zeros(1)
    scheduler = _FakeScheduler()
    captured = {}

    def fake_randn_tensor(shape, *, generator, device, dtype):
        captured["shape"] = shape
        captured["seed"] = generator.initial_seed()
        return torch.zeros(shape, device=device, dtype=dtype)

    module = "vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit"
    with (
        patch(f"{module}.FlowMatchEulerDiscreteScheduler", return_value=scheduler),
        patch(f"{module}.randn_tensor", side_effect=fake_randn_tensor),
    ):
        result = pipeline.forward(
            _batch(
                sampling=OmniDiffusionSamplingParams(
                    height=32, width=48, seed=42, guidance_scale=1.0, num_inference_steps=2
                )
            )
        )

    assert isinstance(result, DiffusionOutput)
    assert result.output.shape == (1, 3, 32, 48)
    assert captured["shape"] == (1, 4, 4, 6)
    assert captured["seed"] == 42
    assert scheduler.requested_steps == 2
    assert pipeline.gen_transformer.calls == 2


def test_forward_rejects_missing_visual_tokens_before_model_access() -> None:
    prompt = {
        "prompt": "",
        "additional_information": {
            "full_hidden_states": torch.zeros(3, 8),
            "full_token_ids": [10, 11, 12],
            "answer_start_index": 2,
        },
    }
    with pytest.raises(ValueError, match="no visual-token hidden states.*req-empty"):
        _pipeline_shell().forward(_batch(request_id="req-empty", prompt=prompt))
