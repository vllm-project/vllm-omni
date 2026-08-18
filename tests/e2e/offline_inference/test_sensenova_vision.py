# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


"""
End-to-end tests for SenseNova-Vision text2text, img2text, and text2img tasks.

These tests validate that the SenseNova-Vision multistage pipeline correctly
generates text output for understanding tasks and image output for generation
tasks, matching the BAGEL-fork lineage of the model.

Equivalent to running:
    python examples/offline_inference/sensenova_vision/end2end.py \
        --modality text2text \
        --prompts "What is the capital of France?"

    python examples/offline_inference/sensenova_vision/end2end.py \
        --modality img2text \
        --image-path <image> \
        --prompts "What are the main objects in this scene and their relationships?"

    python examples/offline_inference/sensenova_vision/end2end.py \
        --modality text2img \
        --prompts "A cute corgi astronaut on the moon, cinematic"
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
from vllm.assets.image import ImageAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

MODEL_NAME = "sensenova/SenseNova-Vision-7B-MoT"
STAGE_CONFIG = get_deploy_config_path("ci/sensenova_vision.yaml")

REFERENCE_TEXT_TEXT2TEXT = "Paris"
REFERENCE_TEXT_IMG2TEXT = "This is a photo of a wooden boardwalk or pathway that leads through tall green grass."


# (model, stage_config_path, extra_omni_kwargs) for ``@pytest.mark.parametrize("omni_runner", ..., indirect=True)``
_OMNI_RUNNER_PARAM = (MODEL_NAME, STAGE_CONFIG)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.diffusion,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


def _extract_text(omni_outputs: list) -> str:
    """Extract generated text from OmniRequestOutput list."""
    for req_output in omni_outputs:
        ro = req_output
        if ro and getattr(ro, "outputs", None):
            return "".join(getattr(o, "text", "") or "" for o in ro.outputs)
    return ""


def _extract_image(omni_outputs: list):
    """Extract the first generated image from OmniRequestOutput list."""
    for req_output in omni_outputs:
        images = getattr(req_output, "images", None)
        if images:
            return images[0]
    return None


@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_sensenova_vision_text2text(run_level, omni_runner: OmniRunner) -> None:
    """Test SenseNovaVision text2text produces correct text output."""
    omni = omni_runner.omni
    prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    params_list = omni.default_sampling_params_list
    omni_outputs = list(
        omni.generate(
            prompts=[{"prompt": prompt, "modalities": ["text"], "mode": "understanding"}],
            sampling_params_list=params_list,
        )
    )

    assert len(omni_outputs) > 0, "No outputs returned"
    text = _extract_text(omni_outputs)
    assert len(text) > 0, "Generated text is empty"

    if run_level == "advanced_model":
        assert "paris" in text.lower(), f"Text mismatch: expected 'Paris' in {text!r}"


@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_sensenova_vision_img2text(run_level, omni_runner: OmniRunner) -> None:
    """Test SenseNovaVision img2text produces correct text output."""
    input_image = ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")
    omni = omni_runner.omni
    prompt = (
        "<|im_start|>user\n<|image_pad|>\n"
        "What are the main objects in this scene and their relationships?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    params_list = omni.default_sampling_params_list
    omni_outputs = list(
        omni.generate(
            prompts=[
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": input_image},
                    "modalities": ["text"],
                    "mode": "understanding",
                }
            ],
            sampling_params_list=params_list,
        )
    )

    assert len(omni_outputs) > 0, "No outputs returned"
    text = _extract_text(omni_outputs)
    assert len(text) > 0, "Generated text is empty"

    if run_level in ["advanced_model", "full_model"]:
        assert "wooden boardwalk" in text.lower(), f"Text mismatch: expected 'wooden boardwalk' in {text!r}"


@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_sensenova_vision_text2img(omni_runner: OmniRunner) -> None:
    """Test SenseNovaVision text2img produces an image output."""
    omni = omni_runner.omni
    prompt = "<|im_start|>A cute corgi astronaut on the moon, cinematic<|im_end|>"
    params_list = omni.default_sampling_params_list
    omni_outputs = list(
        omni.generate(
            prompts=[{"prompt": prompt, "modalities": ["image"], "mode": "generate"}],
            sampling_params_list=params_list,
        )
    )

    assert len(omni_outputs) > 0, "No outputs returned"
    img = _extract_image(omni_outputs)
    assert img is not None, "No image generated"
