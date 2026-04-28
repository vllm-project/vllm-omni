# SPDX-License-Identifier: Apache-2.0
"""Regression tests for multistage diffusion generation input construction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image
from vllm.sampling_params import SamplingParams

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def serving_chat():
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    return object.__new__(OmniOpenAIServingChat)


def test_build_multistage_generation_inputs_applies_stage_specific_overrides(serving_chat):
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    engine = SimpleNamespace(
        stage_configs=[
            SimpleNamespace(stage_type="llm", is_comprehension=True),
            SimpleNamespace(stage_type="diffusion", is_comprehension=False),
            SimpleNamespace(stage_type="diffusion", is_comprehension=False),
        ],
        default_sampling_params_list=[
            SamplingParams(temperature=0.2, seed=11),
            OmniDiffusionSamplingParams(),
            OmniDiffusionSamplingParams(),
        ],
    )
    reference_image = Image.new("RGB", (24, 24), color="green")
    extra_body = {
        "negative_prompt": "blurry",
        "num_inference_steps": 28,
        "guidance_scale": 7.5,
        "true_cfg_scale": 5.0,
        "guidance_scale_2": 1.25,
        "layers": 6,
        "resolution": 1024,
        "lora": {"name": "adapter-a", "path": "/tmp/adapter-a", "scale": 0.6},
    }
    gen_params = OmniDiffusionSamplingParams(height=768, width=1024, seed=0, num_outputs_per_prompt=2)

    engine_prompt, sampling_params_list = OmniOpenAIServingChat._build_multistage_generation_inputs(
        serving_chat,
        engine=engine,
        prompt="draw a robot",
        extra_body=extra_body,
        reference_images=[reference_image],
        gen_params=gen_params,
    )

    assert engine_prompt["prompt"] == "draw a robot"
    assert engine_prompt["modalities"] == ["img2img"]
    assert engine_prompt["negative_prompt"] == "blurry"
    assert engine_prompt["mm_processor_kwargs"] == {"target_h": 768, "target_w": 1024}
    assert engine_prompt["multi_modal_data"]["img2img"].size == (24, 24)

    assert len(sampling_params_list) == 3
    assert sampling_params_list[0].temperature == 0.2
    assert sampling_params_list[0].seed == 0
    assert sampling_params_list[0].extra_args == {"target_h": 768, "target_w": 1024}
    assert sampling_params_list[1] is not gen_params
    assert sampling_params_list[2] is not gen_params
    assert sampling_params_list[1] is not sampling_params_list[2]
    assert sampling_params_list[1].height == 768
    assert sampling_params_list[1].width == 1024
    assert sampling_params_list[1].seed == 0
    assert sampling_params_list[1].num_inference_steps == 28
    assert sampling_params_list[1].guidance_scale == 7.5
    assert sampling_params_list[1].num_outputs_per_prompt == 2
    assert sampling_params_list[1].true_cfg_scale == 5.0
    assert sampling_params_list[1].lora_request.name == "adapter-a"
    assert sampling_params_list[1].lora_scale == 0.6
    assert sampling_params_list[2].height == 768
    assert sampling_params_list[2].width == 1024
    assert sampling_params_list[2].seed == 0
    assert sampling_params_list[2].num_inference_steps == 28
    assert sampling_params_list[2].lora_request.name == "adapter-a"
    assert sampling_params_list[2].lora_scale == 0.6
    assert gen_params.lora_request is None
    assert engine.default_sampling_params_list[1].height is None
    assert engine.default_sampling_params_list[1].lora_request is None
    assert engine.default_sampling_params_list[2].resolution == 640
    assert engine.default_sampling_params_list[2].lora_request is None
