# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for InternVL-U's shared-example prompt adapters."""

import pytest
from PIL import Image

from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import (
    build_image_to_image_prompt,
    build_text_to_image_prompt,
    get_extra_body_params,
    get_extra_output_params,
    should_init_extra_args_for_non_diffusion_stages,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_internvlu_extra_registry() -> None:
    assert get_extra_body_params("InternVLUPipeline") == frozenset(
        {
            "all_cfg_scale",
            "flow_shift",
            "part_cfg_scale",
            "timestep_trunc",
        }
    )
    assert get_extra_output_params("InternVLUPipeline") == frozenset()
    assert should_init_extra_args_for_non_diffusion_stages("InternVLUPipeline") is False


def test_internvlu_extra_registry_preserves_checkpoint_flow_shift_default() -> None:
    params = OmniDiffusionSamplingParams()
    apply_declared_extra_args(
        params,
        get_extra_body_params("InternVLUPipeline"),
        {
            "timestep_shift": 1.0,
            "flow_shift": 3.5,
        },
    )
    assert params.extra_args == {"flow_shift": 3.5}


def test_internvlu_text_to_image_prompt_marks_generation_mode() -> None:
    assert build_text_to_image_prompt(
        "InternVLUPipeline",
        prompt="Draw a red panda.",
        negative_prompt=None,
        height=512,
        width=768,
    ) == {
        "prompt": "Draw a red panda.",
        "modalities": ["image"],
        "mm_processor_kwargs": {
            "modalities": ["image"],
            "target_h": 512,
            "target_w": 768,
        },
    }


def test_internvlu_image_to_image_prompt_preserves_all_references() -> None:
    references = [Image.new("RGB", (32, 32)), Image.new("RGB", (64, 64))]
    result = build_image_to_image_prompt(
        "InternVLUPipeline",
        prompt="Combine these references.",
        negative_prompt=None,
        input_image=references,
        height=512,
        width=512,
    )

    assert result["prompt"] == "Combine these references."
    assert result["modalities"] == ["image"]
    assert result["multi_modal_data"]["image"] is references
    assert result["mm_processor_kwargs"] == {
        "modalities": ["img2img"],
        "target_h": 512,
        "target_w": 512,
    }


def test_internvlu_prompt_builders_reject_negative_prompts() -> None:
    with pytest.raises(ValueError, match="negative prompts are not supported"):
        build_text_to_image_prompt(
            "InternVLUPipeline",
            prompt="Draw a red panda.",
            negative_prompt="blurry",
        )


def test_internvlu_builders_do_not_pin_a_generation_mode() -> None:
    """Think mode is deployment-wide (mm_processor_kwargs.think); the shared
    builders must not pin internvlu_generation_mode, or think deployments
    could never apply it to builder-made prompts."""
    t2i = build_text_to_image_prompt("InternVLUPipeline", prompt="Draw a red panda.", negative_prompt=None)
    i2i = build_image_to_image_prompt(
        "InternVLUPipeline",
        prompt="Restyle it.",
        negative_prompt=None,
        input_image=Image.new("RGB", (32, 32)),
    )
    for built in (t2i, i2i):
        assert "internvlu_generation_mode" not in built["mm_processor_kwargs"]
        assert "think" not in built["mm_processor_kwargs"]
