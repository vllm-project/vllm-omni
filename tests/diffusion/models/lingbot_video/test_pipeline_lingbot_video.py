# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_pipeline():
    from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline

    pipeline = object.__new__(LingBotVideoPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.default_negative_prompt = "default negative"
    pipeline.od_config = SimpleNamespace(flow_shift=None)
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
    )
    from vllm_omni.diffusion.registry import _DIFFUSION_MODELS, _DIFFUSION_POST_PROCESS_FUNCS

    assert LingBotVideoPipeline is not None
    assert LingBotVideoTransformer3DModel is not None
    assert get_lingbot_video_post_process_func is not None
    assert _DIFFUSION_MODELS["LingBotVideoPipeline"] == (
        "lingbot_video",
        "pipeline_lingbot_video",
        "LingBotVideoPipeline",
    )
    assert _DIFFUSION_POST_PROCESS_FUNCS["LingBotVideoPipeline"] == "get_lingbot_video_post_process_func"


def test_component_discovery_declarations():
    from vllm_omni.diffusion.models.lingbot_video import LingBotVideoPipeline

    assert LingBotVideoPipeline._dit_modules == ["transformer"]
    assert LingBotVideoPipeline._encoder_modules == ["text_encoder"]
    assert LingBotVideoPipeline._vae_modules == ["vae"]
    assert LingBotVideoPipeline.supports_step_execution is False


def test_extra_body_params_include_video_flow_shift_alias():
    from vllm_omni.model_extras import get_extra_body_params

    params = get_extra_body_params("LingBotVideoPipeline")
    assert "shift" in params
    assert "flow_shift" in params
    assert "batch_cfg" in params


def test_extract_prompt_string_and_mapping():
    from vllm_omni.diffusion.models.lingbot_video.pipeline_lingbot_video import _extract_prompt

    string_req = SimpleNamespace(prompt="a robot waves")
    assert _extract_prompt(string_req) == ("a robot waves", None)

    mapping_req = SimpleNamespace(prompt={"prompt": "a robot walks", "negative_prompt": "low quality"})
    assert _extract_prompt(mapping_req) == ("a robot walks", "low quality")

    with pytest.raises(TypeError, match="Unsupported LingBot prompt type"):
        _extract_prompt(SimpleNamespace(prompt=["not", "supported"]))


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


def test_forward_resolves_t2v_sampling_and_flow_shift_alias():
    pipeline = _make_pipeline()
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return torch.zeros(9, 192, 320, 3)

    pipeline._generate = fake_generate
    req = _make_request_batch(
        {"prompt": "a robot arm", "negative_prompt": "bad hands"},
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

    assert torch.equal(out.output, torch.zeros(9, 192, 320, 3))
    assert len(calls) == 1
    call = calls[0]
    assert call["prompt"] == "a robot arm"
    assert call["negative_prompt"] == "bad hands"
    assert call["height"] == 192
    assert call["width"] == 320
    assert call["num_frames"] == 9
    assert call["num_inference_steps"] == 2
    assert call["guidance_scale"] == 3.0
    assert call["shift"] == 4.0
    assert call["batch_cfg"] is True
    assert call["output_type"] == "pt"


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


def test_load_weights_rejects_external_weight_stream():
    pipeline = _make_pipeline()

    assert pipeline.load_weights([]) == set()
    with pytest.raises(RuntimeError, match="components are loaded directly"):
        pipeline.load_weights([("transformer.weight", torch.zeros(1))])
