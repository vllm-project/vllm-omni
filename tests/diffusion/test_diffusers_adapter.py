# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch
from diffusers import DiffusionPipeline
from PIL import Image

from vllm_omni.diffusion.data import (
    DiffusionOutput,
    DiffusionParallelConfig,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.models.diffusers_adapter import DiffusersAdapterPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_od_config(**overrides) -> OmniDiffusionConfig:
    od_config = OmniDiffusionConfig(
        model="test/model",
        model_class_name="DiffusersAdapterPipeline",
        dtype=torch.float16,
        diffusion_load_format="diffusers",
        diffusers_load_kwargs={},
        diffusers_call_kwargs={},
        output_type="pil",
        parallel_config=DiffusionParallelConfig(cfg_parallel_size=1, sequence_parallel_size=1),
        cache_backend="none",
    )
    for key, value in overrides.items():
        setattr(od_config, key, value)
    return od_config


def _make_request(**overrides) -> OmniDiffusionRequest:
    prompt = overrides.pop("prompt", "a test prompt")
    negative_prompt = overrides.pop("negative_prompt", None)
    prompt_obj: dict[str, str] = {"prompt": prompt}
    if negative_prompt is not None:
        prompt_obj["negative_prompt"] = negative_prompt

    defaults = {
        "prompts": [prompt_obj],
        "sampling_params": OmniDiffusionSamplingParams(
            num_inference_steps=20,
            guidance_scale=7.5,
            height=16,
            width=16,
            num_frames=1,
            num_outputs_per_prompt=1,
            seed=42,
            output_type="pil",
            generator_device="cpu",
        ),
    }
    defaults.update(overrides)
    return OmniDiffusionRequest(**defaults)


class TestDiffusersAdapterPipeline:
    def test_adapter_forward_returns_output(self, mocker):
        od_config = _make_od_config()
        request = _make_request()
        stub_image = Image.new("RGB", (request.sampling_params.width, request.sampling_params.height))  # pyright: ignore[reportArgumentType]

        adapter = DiffusersAdapterPipeline(od_config=od_config)
        MockPipelineOutput = namedtuple("MockPipelineOutput", ["image"])
        MockPipeline = type("MockPipeline", (DiffusionPipeline,), {})
        adapter._pipeline = MockPipeline()

        mocker.patch.object(
            MockPipeline,
            "__call__",
            return_value=MockPipelineOutput(image=stub_image),
        )
        output = adapter.forward(request)

        assert isinstance(output, DiffusionOutput)
        assert isinstance(output.output, MockPipelineOutput)
        assert output.output.image is stub_image

    @pytest.mark.parametrize(
        "feature_id",
        ["cfg_parallel", "ulysses", "ring", "teacache", "cache_dit", "enforce_eager", "quantization"],
    )
    def test_adapter_guard_unsupported_feature(self, feature_id):
        if feature_id == "cfg_parallel":
            od_config = _make_od_config(
                parallel_config=DiffusionParallelConfig(cfg_parallel_size=2, sequence_parallel_size=1),
                cache_backend="none",
            )
        elif feature_id == "ulysses":
            od_config = _make_od_config(
                parallel_config=DiffusionParallelConfig(cfg_parallel_size=1, ulysses_degree=2),
                cache_backend="none",
            )
        elif feature_id == "ring":
            od_config = _make_od_config(
                parallel_config=DiffusionParallelConfig(cfg_parallel_size=1, ring_degree=2),
                cache_backend="none",
            )
        elif feature_id == "teacache":
            od_config = _make_od_config(
                parallel_config=DiffusionParallelConfig(cfg_parallel_size=1, sequence_parallel_size=1),
                cache_backend="tea_cache",
            )
        elif feature_id == "cache_dit":
            od_config = _make_od_config(
                parallel_config=DiffusionParallelConfig(cfg_parallel_size=1, sequence_parallel_size=1),
                cache_backend="cache_dit",
            )
        elif feature_id == "enforce_eager":
            od_config = _make_od_config(enforce_eager=True)
        elif feature_id == "quantization":
            od_config = _make_od_config(quantization_config=SimpleNamespace(quant_method="fp8"))
        else:
            raise ValueError(f"Unknown feature ID: {feature_id}")

        with pytest.raises(NotImplementedError):
            DiffusersAdapterPipeline(od_config=od_config)

    def test_adapter_guard_unknown_output_type(self, mocker):
        """Test that the adapter wraps an unknown output type as-is.
        This is useful when `return_dict=True` and the diffusers pipeline returns an OrderedDict subclass."""

        adapter = DiffusersAdapterPipeline(od_config=_make_od_config())
        raw_output = {"unexpected": "dict-output"}

        MockPipeline = type("MockPipeline", (DiffusionPipeline,), {})
        adapter._pipeline = MockPipeline()

        mocker.patch.object(
            MockPipeline,
            "__call__",
            return_value=raw_output,
        )
        output = adapter.forward(_make_request())

        assert isinstance(output, DiffusionOutput)
        assert output.output == raw_output

    def test_adapter_build_call_kwargs(self):
        adapter = DiffusersAdapterPipeline(
            od_config=_make_od_config(
                diffusers_call_kwargs={
                    "guidance_scale": 1.25,
                    "eta": 0.3,
                    "output_type": "np",
                }
            )
        )
        req = _make_request(
            prompt="a cat on mars",
            negative_prompt="low quality",
            sampling_params=OmniDiffusionSamplingParams(
                num_inference_steps=9,
                guidance_scale=8.0,
                height=320,
                width=640,
                num_frames=8,
                num_outputs_per_prompt=2,
                seed=123,
                output_type="pil",
            ),
        )

        kwargs = adapter._build_call_kwargs(req)

        assert kwargs["prompt"] == "a cat on mars"
        assert kwargs["negative_prompt"] == "low quality"
        assert kwargs["num_inference_steps"] == 9
        assert kwargs["guidance_scale"] == 8.0
        assert kwargs["height"] == 320
        assert kwargs["width"] == 640
        assert kwargs["num_frames"] == 8
        assert kwargs["num_images_per_prompt"] == 2
        assert kwargs["output_type"] == "pil"
        assert isinstance(kwargs["generator"], torch.Generator)
        assert kwargs["generator"].device.type == "cpu"
        assert kwargs["generator"].initial_seed() == 123
