# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for Qwen3-Omni AutoRound W4A16 quantized inference.

These tests cover text, audio, image, video, and mixed-modality inputs
to verify multimodal understanding with quantized weights.

Requirements:
  - CUDA GPUs (2x H100-80G or equivalent)
  - The quantized model checkpoint (Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound)
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

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.omni,
]

_SKIP_ISSUE_3195 = pytest.mark.skip(
    reason="https://github.com/vllm-project/vllm-omni/issues/3195",
)

QUANTIZED_MODEL = "Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound"
BASELINE_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Allow overriding via environment for local testing
QUANTIZED_MODEL = os.environ.get("QWEN3_OMNI_AUTOROUND_MODEL", QUANTIZED_MODEL)
BASELINE_MODEL = os.environ.get("QWEN3_OMNI_BASELINE_MODEL", BASELINE_MODEL)

_CI_DEPLOY = get_deploy_config_path("ci/qwen3_omni_moe.yaml")


@pytest.fixture(scope="module", autouse=True)
def _qwen3_omni_env():
    """Set env vars required by multi-stage worker spawning.

    Must run before CUDA context init.  Reverted after every test module
    so that values do not leak into unrelated test files.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        mp.setenv("VLLM_TEST_CLEAN_GPU_MEMORY", "0")
        yield


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


@_SKIP_ISSUE_3195
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_text_to_text(omni_runner, omni_runner_handler):
    """Text input → text output with W4A16 quantized Qwen3-Omni."""
    request_config = {
        "prompts": "What is the capital of France?",
        "modalities": ["text"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
    assert response.text_content and len(response.text_content.strip()) > 0


# ------------------------------------------------------------------
# Test: audio input → text output
# ------------------------------------------------------------------


@_SKIP_ISSUE_3195
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_audio_to_text(omni_runner, omni_runner_handler):
    """Audio input → text output with W4A16 quantized Qwen3-Omni."""
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


@_SKIP_ISSUE_3195
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_image_to_text(omni_runner, omni_runner_handler):
    """Image input → text output with W4A16 quantized Qwen3-Omni."""
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


@_SKIP_ISSUE_3195
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_video_to_text(omni_runner, omni_runner_handler):
    """Video input → text output with W4A16 quantized Qwen3-Omni."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]

    request_config = {
        "prompts": "Describe the video briefly.",
        "videos": video,
        "modalities": ["text"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"
    assert response.text_content and len(response.text_content.strip()) > 0


# ------------------------------------------------------------------
# Test: video input → audio output
# ------------------------------------------------------------------


@_SKIP_ISSUE_3195
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_video_to_audio(omni_runner, omni_runner_handler):
    """Video input → audio output with W4A16 quantized Qwen3-Omni."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]

    request_config = {
        "prompts": "Describe the video briefly.",
        "videos": video,
        "modalities": ["audio"],
    }
    response = omni_runner_handler.send_request(request_config)
    assert response.success, f"Request failed: {response.error_message}"


# ------------------------------------------------------------------
# Test: mixed modality (audio + image + video) → audio output
# ------------------------------------------------------------------


@_SKIP_ISSUE_3195
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", quant_params, indirect=True)
def test_mix_to_audio(omni_runner, omni_runner_handler):
    """Mixed-modality input → audio output with W4A16 quantized Qwen3-Omni."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]
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
