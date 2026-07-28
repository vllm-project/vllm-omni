# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import vllm_omni.diffusion.data as diffusion_data
import vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image as hidream_module
import vllm_omni.diffusion.utils.hf_utils as hf_utils
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    PATCH_SIZE,
    HiDreamO1ImagePipeline,
    get_hidream_o1_image_post_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_IMAGE_LEN = 6
_PATCH_DIM = 3 * PATCH_SIZE * PATCH_SIZE


def _make_init_config() -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        model="dummy-hidream",
        dtype=torch.bfloat16,
        enable_diffusion_pipeline_profiler=False,
        revision="test-revision",
    )


class _TestScheduler:
    def __init__(self, timesteps: torch.Tensor) -> None:
        self.timesteps = timesteps
        self.seen_model_output = None
        self.step_calls = 0

    def step(self, model_output, timestep, sample, return_dict):
        del timestep, return_dict
        self.seen_model_output = model_output.detach()
        self.step_calls += 1
        return (sample,)


class _ModelConfig:
    image_token_id = 1
    video_token_id = 2
    vision_start_token_id = 3


class _ModelStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _ModelConfig()


class _ForwardPipeline(HiDreamO1ImagePipeline):
    def __init__(
        self,
        *,
        x_pred_values: list[float],
        constant_z: torch.Tensor,
        guidance_scale: float,
    ) -> None:
        nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16
        self.tokenizer = None
        self.processor = None
        self.model = _ModelStub()
        self._progress_bar_config = {"disable": True}
        self.x_pred_values = x_pred_values
        self.constant_z = constant_z
        self.guidance_scale = guidance_scale
        self.forward_once_call_count = 0

    def _resolve_generation_params(
        self,
        req: DiffusionRequestBatch,
    ) -> tuple[str, int, int, int, int, float]:
        assert req.num_reqs == 1
        return "a cat", 64, 96, 1, 42, self.guidance_scale

    def _prepare_noise_and_patchify(
        self,
        height: int,
        width: int,
        seed: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        del height, width, seed
        return self.constant_z.to(device=device, dtype=dtype)

    def _forward_once(
        self,
        sample: hidream_module.HiDreamO1TextSample,
        z_in: torch.Tensor,
        t_pixeldit: torch.Tensor,
    ) -> torch.Tensor:
        del sample, t_pixeldit
        value = self.x_pred_values[self.forward_once_call_count]
        self.forward_once_call_count += 1
        return torch.full_like(z_in, value, dtype=torch.float32)


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


def _make_request_batch(guidance_scale: float = 1.0) -> DiffusionRequestBatch:
    return DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="a cat",
                sampling_params=OmniDiffusionSamplingParams(
                    height=64,
                    width=96,
                    num_inference_steps=1,
                    seed=42,
                    guidance_scale=guidance_scale,
                ),
                request_id="hidream-pipeline-test",
            )
        ]
    )


def _run_forward_case(x_pred_values: list[float], guidance_scale: float):
    constant_z = torch.ones((1, _IMAGE_LEN, _PATCH_DIM), dtype=torch.bfloat16)
    pipe = _ForwardPipeline(
        x_pred_values=x_pred_values,
        constant_z=constant_z,
        guidance_scale=guidance_scale,
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
    ):
        out = pipe.forward(_make_request_batch(guidance_scale=guidance_scale))

    return out, test_scheduler, pipe


def test_pipeline_registered_and_uses_standard_weight_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        DiffusionModelRegistry,
    )

    class _Tokenizer:
        tms_token = "<|tms_token|>"

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            assert text == self.tms_token
            return [7]

    class _Processor:
        def __init__(self) -> None:
            self.tokenizer = _Tokenizer()

    class _CoreModel:
        tms_token_id = 7

    class _TinyHiDreamModel(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.config = config
            self.model = _CoreModel()
            self.weight = nn.Parameter(torch.zeros(1))

    processor_calls = []
    config_calls = []

    def _load_processor(model, *, revision=None):
        processor_calls.append((model, {"revision": revision}))
        return _Processor()

    def _load_config(model, *, revision=None):
        config_calls.append((model, {"revision": revision}))
        return _ModelConfig()

    monkeypatch.setattr(
        hidream_module,
        "get_local_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        hidream_module.AutoProcessor,
        "from_pretrained",
        _load_processor,
    )
    monkeypatch.setattr(
        hidream_module.Qwen3VLConfig,
        "from_pretrained",
        _load_config,
    )
    monkeypatch.setattr(hidream_module, "HiDreamO1ImageTransformer", _TinyHiDreamModel)

    pipeline = HiDreamO1ImagePipeline(od_config=_make_init_config())

    expected = (
        "hidream_o1_image",
        "pipeline_hidream_o1_image",
        "HiDreamO1ImagePipeline",
    )
    assert _DIFFUSION_MODELS["HiDreamO1ImagePipeline"] == expected
    assert "Qwen3VLForConditionalGeneration" not in _DIFFUSION_MODELS
    assert "Qwen3VLForConditionalGeneration" not in _DIFFUSION_POST_PROCESS_FUNCS
    assert DiffusionModelRegistry._try_load_model_cls("HiDreamO1ImagePipeline") is HiDreamO1ImagePipeline

    assert pipeline.dtype == torch.bfloat16
    assert pipeline.device.type == "cpu"
    assert pipeline.tokenizer.boi_token == "<|boi_token|>"
    assert pipeline.tokenizer.tms_token == "<|tms_token|>"
    assert processor_calls == [("dummy-hidream", {"revision": "test-revision"})]
    assert config_calls == [("dummy-hidream", {"revision": "test-revision"})]
    assert len(pipeline.weights_sources) == 1
    assert pipeline.weights_sources[0].prefix == "model."
    assert pipeline.weights_sources[0].revision == "test-revision"

    loaded = pipeline.load_weights(iter([("model.weight", torch.ones(1))]))
    assert loaded == {"model.weight"}
    torch.testing.assert_close(pipeline.model.weight, torch.ones(1))
    assert not HiDreamO1ImagePipeline.supports_request_batch
    assert HiDreamO1ImagePipeline.dummy_run_num_frames == 0


def test_hidream_o1_checkpoint_detection_does_not_hijack_qwen3vl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qwen_config = {
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
    }

    def _get_hf_file(filename: str, model: str):
        del model
        if filename == "model_index.json":
            raise OSError("not a diffusers checkpoint")
        if filename == "config.json":
            return qwen_config
        if filename == "model.safetensors.index.json":
            return {"weight_map": {"model.final_layer2.linear.weight": "model-00001-of-00002.safetensors"}}
        raise AssertionError(filename)

    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_hf_file_to_dict",
        _get_hf_file,
    )
    monkeypatch.setattr(hf_utils, "get_hf_file_to_dict", _get_hf_file)

    assert diffusion_data.resolve_model_class_name("hidream-o1") == "HiDreamO1ImagePipeline"

    def _reject_diffusers_config(model: str):
        raise ValueError(model)

    monkeypatch.setattr(hf_utils, "load_diffusers_config", _reject_diffusers_config)
    hf_utils.is_diffusion_model.cache_clear()
    assert hf_utils.is_diffusion_model("hidream-o1")

    config = _make_init_config()
    config.model = "hidream-o1"
    config.model_class_name = None
    config.enrich_config()
    assert config.model_class_name == "HiDreamO1ImagePipeline"

    def _get_plain_qwen_file(filename: str, model: str):
        del model
        if filename == "model_index.json":
            raise OSError("not a diffusers checkpoint")
        if filename == "config.json":
            return qwen_config
        if filename == "model.safetensors.index.json":
            return {"weight_map": {"model.language_model.embed_tokens.weight": "model.safetensors"}}
        raise AssertionError(filename)

    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_hf_file_to_dict",
        _get_plain_qwen_file,
    )
    monkeypatch.setattr(hf_utils, "get_hf_file_to_dict", _get_plain_qwen_file)

    assert diffusion_data.resolve_model_class_name("plain-qwen3-vl") == "Qwen3VLForConditionalGeneration"
    hf_utils.is_diffusion_model.cache_clear()
    assert not hf_utils.is_diffusion_model("plain-qwen3-vl")


def test_postprocess_returns_rgb_image() -> None:
    postprocess = get_hidream_o1_image_post_process_func(_make_init_config())
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
    assert pipe.forward_once_call_count == 1

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
    assert pipe.forward_once_call_count == 2

    expected = torch.full((1, _IMAGE_LEN, _PATCH_DIM), -20.0, dtype=torch.float32)
    torch.testing.assert_close(
        test_scheduler.seen_model_output,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )
