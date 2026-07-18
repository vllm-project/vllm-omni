# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image as hidream_module
from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    PATCH_SIZE,
    HiDreamO1ImagePipeline,
    get_hidream_o1_image_post_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_IMAGE_LEN = 6
_PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE


def _make_init_config():
    return SimpleNamespace(
        model="dummy-hidream",
        dtype=torch.bfloat16,
        parallel_config=SimpleNamespace(cfg_parallel_size=1),
        enable_diffusion_pipeline_profiler=False,
    )


class _TestScheduler:
    def __init__(self, timesteps: torch.Tensor) -> None:
        self.timesteps = timesteps
        self.seen_model_output = None
        self.step_calls = 0

    def step(self, model_output, timestep, sample, return_dict):
        del timestep, return_dict
        self.seen_model_output = model_output.detach().clone()
        self.step_calls += 1
        return (sample,)


def _test_build_t2i_text_sample(*, prompt, height, width, tokenizer, processor, model_config):
    del prompt, tokenizer, processor, model_config
    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)
    text_len = 3
    all_len = text_len + image_len
    vinput_mask = torch.zeros((1, all_len), dtype=torch.bool)
    vinput_mask[0, text_len:] = True
    return {
        "input_ids": torch.zeros((1, text_len), dtype=torch.long),
        "position_ids": torch.zeros((3, 1, all_len), dtype=torch.long),
        "token_types": torch.zeros((1, all_len), dtype=torch.long),
        "vinput_mask": vinput_mask,
    }


def _make_test_pipeline_stub(
    x_pred_values: list[float],
    constant_z: torch.Tensor,
) -> HiDreamO1ImagePipeline:
    pipe = HiDreamO1ImagePipeline.__new__(HiDreamO1ImagePipeline)
    nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.dtype = torch.bfloat16
    pipe.tokenizer = None
    pipe.processor = None
    pipe.model = SimpleNamespace(config=None)

    call_idx = [0]
    pipe._forward_once_call_count = call_idx

    def _mock_prepare_noise(height, width, seed, dtype, device):
        del height, width, seed
        return constant_z.to(device=device, dtype=dtype)

    def _mock_forward_once(sample, z_in, t_pixeldit):
        del sample, t_pixeldit
        value = x_pred_values[call_idx[0]]
        call_idx[0] += 1
        return torch.full_like(z_in, value, dtype=torch.float32)

    pipe._prepare_noise_and_patchify = _mock_prepare_noise
    pipe._forward_once = _mock_forward_once
    return pipe


def _make_request(guidance_scale: float = 1.0) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=["a cat"],
        sampling_params=OmniDiffusionSamplingParams(
            height=64,
            width=96,
            num_inference_steps=1,
            seed=42,
            guidance_scale=guidance_scale,
        ),
        request_id="hidream-pipeline-test",
    )


def _run_forward_case(x_pred_values: list[float], guidance_scale: float):
    constant_z = torch.ones((1, _IMAGE_LEN, _PATCH_DIM), dtype=torch.bfloat16)
    pipe = _make_test_pipeline_stub(
        x_pred_values=x_pred_values,
        constant_z=constant_z,
    )
    test_scheduler = _TestScheduler(timesteps=torch.tensor([500.0]))

    with (
        patch.object(
            hidream_module,
            "build_hidream_o1_scheduler",
            lambda **_kwargs: test_scheduler,
        ),
        patch.object(
            hidream_module,
            "build_t2i_text_sample",
            _test_build_t2i_text_sample,
        ),
        patch.object(
            pipe,
            "_resolve_generation_params",
            lambda req: ("a cat", 64, 96, 1, 42, guidance_scale),
        ),
    ):
        out = pipe.forward(_make_request(guidance_scale=guidance_scale))

    return out, test_scheduler, pipe


def test_pipeline_registered_and_initializes_lightweight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        DiffusionModelRegistry,
    )

    validate_calls = {"count": 0}

    def _mock_init_processor_and_model(self) -> None:
        self.processor = SimpleNamespace(tokenizer=SimpleNamespace())
        self.model = nn.Linear(2, 3)
        HiDreamO1ImagePipeline._add_special_tokens(self.processor)
        self.tokenizer = self.processor.tokenizer

    def _mock_validate_tms(self) -> None:
        validate_calls["count"] += 1

    monkeypatch.setattr(
        hidream_module,
        "get_local_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        HiDreamO1ImagePipeline,
        "_init_processor_and_model",
        _mock_init_processor_and_model,
    )
    monkeypatch.setattr(
        HiDreamO1ImagePipeline,
        "_validate_tms_token_id",
        _mock_validate_tms,
    )

    pipeline = HiDreamO1ImagePipeline(od_config=_make_init_config())

    expected = (
        "hidream_o1_image",
        "pipeline_hidream_o1_image",
        "HiDreamO1ImagePipeline",
    )
    assert _DIFFUSION_MODELS["Qwen3VLForConditionalGeneration"] == expected
    assert _DIFFUSION_MODELS["HiDreamO1ImagePipeline"] == expected
    assert _DIFFUSION_POST_PROCESS_FUNCS["Qwen3VLForConditionalGeneration"] == (
        "get_hidream_o1_image_post_process_func"
    )
    assert DiffusionModelRegistry._try_load_model_cls("Qwen3VLForConditionalGeneration") is HiDreamO1ImagePipeline

    assert pipeline.dtype == torch.bfloat16
    assert pipeline.device.type == "cpu"
    assert pipeline.model is not None
    assert pipeline.tokenizer.boi_token == "<|boi_token|>"
    assert pipeline.tokenizer.tms_token == "<|tms_token|>"
    assert validate_calls["count"] == 1
    assert pipeline.load_weights(iter(())) == {name for name, _ in pipeline.named_parameters()}
    assert HiDreamO1ImagePipeline.dummy_run_num_frames == 0


def test_postprocess_returns_rgb_image() -> None:
    postprocess = get_hidream_o1_image_post_process_func(SimpleNamespace())
    z = torch.zeros((1, 6, _PATCH_DIM), dtype=torch.bfloat16)

    img = postprocess((z, 64, 96))

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (96, 64)
    arr = np.asarray(img)
    assert arr.shape == (64, 96, 3)
    assert np.all(arr == 128)


def test_forward_returns_patch_tensor_envelope() -> None:
    out, test_scheduler, pipe = _run_forward_case(
        x_pred_values=[3.0],
        guidance_scale=1.0,
    )
    z_out, height, width = out.output

    assert (height, width) == (64, 96)
    assert z_out.shape == (1, _IMAGE_LEN, _PATCH_DIM)
    assert z_out.dtype == torch.bfloat16
    assert test_scheduler.step_calls == 1
    assert pipe._forward_once_call_count[0] == 1

    expected = torch.full((1, _IMAGE_LEN, _PATCH_DIM), -4.0, dtype=torch.float32)
    torch.testing.assert_close(
        test_scheduler.seen_model_output,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_forward_guidance_scale_runs_cfg_branch() -> None:
    out, test_scheduler, pipe = _run_forward_case(
        x_pred_values=[3.0, 1.0],
        guidance_scale=5.0,
    )
    z_out, _, _ = out.output

    assert z_out.shape == (1, _IMAGE_LEN, _PATCH_DIM)
    assert test_scheduler.step_calls == 1
    assert pipe._forward_once_call_count[0] == 2

    expected = torch.full((1, _IMAGE_LEN, _PATCH_DIM), -20.0, dtype=torch.float32)
    torch.testing.assert_close(
        test_scheduler.seen_model_output,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )
