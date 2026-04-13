# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end tests for Bagel text2text and img2text (understanding) tasks.

These tests validate that the Bagel multistage pipeline correctly generates
text output for understanding tasks, matching reference results.

Equivalent to running:
    python3 examples/offline_inference/bagel/end2end.py \
        --modality text2text \
        --prompts "Where is the capital of France?"

    python3 examples/offline_inference/bagel/end2end.py \
        --modality img2text \
        --prompts "Please describe this image" \
        --image-path 2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from pathlib import Path

import pytest
from vllm.assets.image import ImageAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import modify_stage_config

MODEL_NAME = "ByteDance-Seed/BAGEL-7B-MoT"
STAGE_CONFIG = str(Path(__file__).parent / "stage_configs" / "bagel_sharedmemory_ci.yaml")

REFERENCE_TEXT_TEXT2TEXT = "The capital of France is Paris."

REFERENCE_TEXT_IMG2TEXT = (
    "This is a photo of a wooden boardwalk or pathway that leads through "
    "tall green grass. The path appears to be in a natural setting, possibly "
    "a wetland or marsh area. The sky above is blue with some scattered "
    "clouds, suggesting it might be a sunny day. The overall scene looks "
    "peaceful and serene."
)


def _resolve_stage_config(config_path: str, run_level: str) -> str:
    """Strip load_format: dummy for advanced_model (real weights)."""
    if run_level == "advanced_model":
        return modify_stage_config(
            config_path,
            deletes={
                "stage_args": {
                    0: ["engine_args.load_format"],
                    1: ["engine_args.load_format"],
                }
            },
        )
    return config_path


def _extract_text(omni_outputs: list) -> str:
    """Extract generated text from OmniRequestOutput list."""
    for req_output in omni_outputs:
        ro = getattr(req_output, "request_output", None)
        if ro and getattr(ro, "outputs", None):
            return "".join(getattr(o, "text", "") or "" for o in ro.outputs)
    return ""


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_bagel_text2text(run_level):
    """Test Bagel text2text produces correct text output."""
    config_path = _resolve_stage_config(STAGE_CONFIG, run_level)
    with OmniRunner(
        MODEL_NAME,
        stage_configs_path=config_path,
    ) as runner:
        omni = runner.omni
        prompt = "<|im_start|>user\nWhere is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
        params_list = omni.default_sampling_params_list
        omni_outputs = list(
            omni.generate(
                prompts=[{"prompt": prompt, "modalities": ["text"]}],
                sampling_params_list=params_list,
            )
        )

        assert len(omni_outputs) > 0, "No outputs returned"
        text = _extract_text(omni_outputs)
        assert len(text) > 0, "Generated text is empty"

        if run_level == "advanced_model":
            assert text == REFERENCE_TEXT_TEXT2TEXT, (
                f"Text mismatch: expected {REFERENCE_TEXT_TEXT2TEXT!r}, got {text!r}"
            )


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
def test_bagel_img2text(run_level):
    """Test Bagel img2text produces correct text output."""
    input_image = ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")
    config_path = _resolve_stage_config(STAGE_CONFIG, run_level)
    with OmniRunner(
        MODEL_NAME,
        stage_configs_path=config_path,
        stage_init_timeout=300,
    ) as runner:
        omni = runner.omni
        prompt = "<|im_start|>user\n<|image_pad|>\nPlease describe this image<|im_end|>\n<|im_start|>assistant\n"
        params_list = omni.default_sampling_params_list
        omni_outputs = list(
            omni.generate(
                prompts=[
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": input_image},
                        "modalities": ["text"],
                    }
                ],
                sampling_params_list=params_list,
            )
        )

        assert len(omni_outputs) > 0, "No outputs returned"
        text = _extract_text(omni_outputs)
        assert len(text) > 0, "Generated text is empty"

        if run_level == "advanced_model":
            assert text == REFERENCE_TEXT_IMG2TEXT, f"Text mismatch: expected {REFERENCE_TEXT_IMG2TEXT!r}, got {text!r}"
