# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
E2E offline tests for Omni model with video input and audio output.

``test_video_to_audio`` loads production ``qwen3_omni_moe.yaml``. Deploy
projection into ``VllmConfig`` and ``SamplingParams`` is covered by
``tests/engine/test_qwen3_omni_deploy_projection.py`` and does not launch
the 30B model.

Abort / sleep-admission lives in ``test_qwen3_omni_colocate_async.py`` so it
does not overlap this module's OmniRunners.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import QWEN3_OMNI_COLD_STAGE_INIT_TIMEOUT_S
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.platforms import current_omni_platform

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]
thinker_only_models = ["Qwen/Qwen3-Omni-30B-A3B-Captioner", "Qwen/Qwen3-Omni-30B-A3B-Thinking"]

# Packaged production deploy for the Instruct video case. Its ``platforms:``
# section selects rocm/xpu/cuda deltas. CUDA leaves ``enforce_eager`` unset
# (cudagraphs on). Thinker-only stays on the CI deploy, so it does not reuse
# this runner.
_PRODUCTION_DEPLOY = get_deploy_config_path("qwen3_omni_moe.yaml")

# Thinker-only cases stay on the CI deploy. Eager mode there is the existing
# smoke bound, not a production requirement. XPU's CI overlay already sets it.
_CI_DEPLOY = get_deploy_config_path("ci/qwen3_omni_moe.yaml")


def get_cuda_graph_config():
    return modify_stage_config(
        _CI_DEPLOY,
        updates={
            "stages": {
                0: {"enforce_eager": True},
                1: {"enforce_eager": True},
            },
        },
    )


if current_omni_platform.is_xpu():
    thinker_stage_configs = [_CI_DEPLOY]
else:
    thinker_stage_configs = [get_cuda_graph_config()]

# The third item is OmniRunner kwargs. Only ROCm receives the bounded per-stage
# cold-NFS allowance; the runner's existing 1800s overall deadline is retained.
_STAGE_INIT = {"stage_init_timeout": QWEN3_OMNI_COLD_STAGE_INIT_TIMEOUT_S} if current_omni_platform.is_rocm() else {}
test_params = [(model, _PRODUCTION_DEPLOY, _STAGE_INIT) for model in models]
# The resolver drops audio stages when the HF config has no audio output, so
# thinker-only models can load this CI deploy without the full pipeline.
thinker_test_params = [
    (model, stage_config, _STAGE_INIT) for model in thinker_only_models for stage_config in thinker_stage_configs
]


def get_prompt(prompt_type: str):
    prompts = {
        "video": "Describe the video briefly.",
        "image": "Describe the image.",
        "audio": "Describe the audio.",
        "text": "What is the capital of the United Kingdom?",
    }
    return prompts[prompt_type]


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text(omni_runner, offline_client) -> None:
    """Test generating text from text input"""
    request_config = {"prompts": get_prompt("text"), "modalities": ["text"]}

    # Test single completion
    offline_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_audio(omni_runner, offline_client) -> None:
    """Test processing audio from text input"""
    request_config = {"prompts": get_prompt("text"), "modalities": ["audio"]}

    # Test single completion
    offline_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_image_to_text(omni_runner, offline_client) -> None:
    """Test processing image, generating text output."""
    image = generate_synthetic_image(224, 224)["np_array"]

    request_config = {"prompts": get_prompt("image"), "images": image, "modalities": ["text"]}

    # Test single completion
    offline_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_image_to_audio(omni_runner, offline_client) -> None:
    """Test processing image, generating audio output."""
    image = generate_synthetic_image(224, 224)["np_array"]

    request_config = {"prompts": get_prompt("image"), "images": image, "modalities": ["audio"]}

    # Test single completion
    offline_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_text(omni_runner, offline_client) -> None:
    """Test processing video, generating text output."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]

    request_config = {"prompts": get_prompt("video"), "videos": video, "modalities": ["text"]}

    # Test single completion
    offline_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_audio(omni_runner, offline_client) -> None:
    """Test processing video, generating audio output."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]

    request_config = {"prompts": get_prompt("video"), "videos": video, "modalities": ["audio"]}

    # Test single completion
    offline_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_audio_to_text(omni_runner, offline_client) -> None:
    """Test processing audio, generating text output"""
    audio = generate_synthetic_audio(2, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()

    request_config = {"prompts": get_prompt("audio"), "audios": audio, "modalities": ["text"]}

    # Test single completion
    offline_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_audio_to_audio(omni_runner, offline_client) -> None:
    """Test audio processing, producing audio output"""
    audio = generate_synthetic_audio(2, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()

    request_config = {"prompts": get_prompt("audio"), "audios": audio, "modalities": ["audio"]}

    # Test single completion
    offline_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", thinker_test_params, indirect=True)
def test_thinker_only_model_request(omni_runner, offline_client) -> None:
    """Test that we can load and run a request through a model that only has the thinker stage."""
    request_config = {"prompts": "what color is the sky?", "modalities": ["text"]}

    # Test single completion
    offline_client.send_omni_request(request_config)
