# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for Qwen2.5-Omni AutoRound W4A16 quantized inference.

These tests cover text, audio, image, video, and mixed-modality inputs
to verify multimodal understanding with quantized weights.

Requirements:
  - CUDA GPUs (4x L4 / 24 GB or equivalent)
  - The quantized model checkpoint (Intel/Qwen2.5-Omni-7B-int4-AutoRound)
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import (
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
)
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

QUANTIZED_MODEL = "Intel/Qwen2.5-Omni-7B-int4-AutoRound"
BASELINE_MODEL = "Qwen/Qwen2.5-Omni-7B"

# Allow overriding via environment for local testing
QUANTIZED_MODEL = os.environ.get("QWEN2_5_OMNI_AUTOROUND_MODEL", QUANTIZED_MODEL)
BASELINE_MODEL = os.environ.get("QWEN2_5_OMNI_BASELINE_MODEL", BASELINE_MODEL)

_CI_DEPLOY = get_deploy_config_path("ci/qwen2_5_omni.yaml")


def _get_stage_config():
    """Build a CI-friendly stage config with eager mode."""
    return modify_stage_config(
        _CI_DEPLOY,
        updates={
            "stages": {
                0: {"enforce_eager": True},
                1: {"enforce_eager": True},
            },
        },
    )


stage_config = _get_stage_config()

# Parametrise: (model, stage_config)
quant_params = [(QUANTIZED_MODEL, stage_config)]


# ------------------------------------------------------------------
# Test: text-only input → text output
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_text_to_text(omni_runner, omni_runner_handler):
    """Text input → text output with W4A16 quantized Qwen2.5-Omni."""
    request_config = {
        "prompts": "What is the capital of China?",
        "modalities": ["text"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
    assert response.text_content and len(response.text_content.strip()) > 0


# ------------------------------------------------------------------
# Test: audio input → text output
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_audio_to_text(omni_runner, omni_runner_handler):
    """Audio input → text output with W4A16 quantized Qwen2.5-Omni."""
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()

    request_config = {
        "prompts": "What is the content of this audio?",
        "audios": (audio, 16000),
        "modalities": ["text"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
    assert response.text_content and len(response.text_content.strip()) > 0


# ------------------------------------------------------------------
# Test: image input → text output
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_image_to_text(omni_runner, omni_runner_handler):
    """Image input → text output with W4A16 quantized Qwen2.5-Omni."""
    image = generate_synthetic_image(16, 16)["np_array"]

    request_config = {
        "prompts": "Describe what you see in this image.",
        "images": image,
        "modalities": ["text"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
    assert response.text_content and len(response.text_content.strip()) > 0


# ------------------------------------------------------------------
# Test: video input → text output
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_video_to_text(omni_runner, omni_runner_handler):
    """Video input → text output with W4A16 quantized Qwen2.5-Omni."""
    video = generate_synthetic_video(16, 16, 30)["np_array"]

    request_config = {
        "prompts": "Describe the video briefly.",
        "videos": video,
        "modalities": ["text"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
    assert response.text_content and len(response.text_content.strip()) > 0


# ------------------------------------------------------------------
# Test: mixed modality (audio + image + video) → audio output
# ------------------------------------------------------------------


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_mix_to_audio(omni_runner, omni_runner_handler):
    """Mixed-modality input → audio output with W4A16 quantized Qwen2.5-Omni."""
    video = generate_synthetic_video(16, 16, 30)["np_array"]
    image = generate_synthetic_image(16, 16)["np_array"]
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()

    request_config = {
        "prompts": "What is recited in the audio? What is in this image? Describe the video briefly.",
        "videos": video,
        "images": image,
        "audios": (audio, 16000),
        "modalities": ["audio"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
