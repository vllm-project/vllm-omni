# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import (
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
)
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

models = ["Jonathan1909/Ming-flash-omni-2.0"]

# Ming-specific
SYSTEM_PROMPT = "你是一个友好的AI助手。\n\ndetailed thinking off"
EOS_TOKEN = "<|role_end|>"
IMAGE_TOKEN = "<IMAGE>"
VIDEO_TOKEN = "<VIDEO>"
AUDIO_TOKEN = "<AUDIO>"


def build_prompt(user_text: str) -> str:
    """Build a Ming chat prompt."""
    return (
        f"<role>SYSTEM</role>{SYSTEM_PROMPT}{EOS_TOKEN}<role>HUMAN</role>{user_text}{EOS_TOKEN}<role>ASSISTANT</role>"
    )


_CI_DEPLOY_THINKER_ONLY = get_deploy_config_path("ci/ming_flash_omni_thinker_only.yaml")
_CI_DEPLOY = get_deploy_config_path("ci/ming_flash_omni.yaml")


def get_eager_config_thinker():
    return modify_stage_config(_CI_DEPLOY_THINKER_ONLY, updates={"stages": {0: {"enforce_eager": True}}})


def get_eager_config():
    """Thinker+talker CI config with enforce_eager on the thinker stage."""
    return modify_stage_config(_CI_DEPLOY, updates={"stages": {0: {"enforce_eager": True}}})


test_params_thinker = [(m, c) for m in models for c in [get_eager_config_thinker()]]
test_params = [(m, c) for m in models for c in [get_eager_config()]]


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params_thinker, indirect=True)
def test_text_to_text(omni_runner, omni_runner_handler) -> None:
    """
    Test text-only input processing and text output generation.
    Input Modal: text
    Output Modal: text
    """
    prompt = build_prompt("请详细介绍鹦鹉的生活习性。")
    request_config = {"prompts": prompt, "modalities": ["text"]}

    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params_thinker, indirect=True)
def test_image_to_text(omni_runner, omni_runner_handler) -> None:
    """
    Test image understanding with text output.
    Input Modal: image + text
    Output Modal: text
    """
    image = generate_synthetic_image(224, 224)["np_array"]
    prompt = build_prompt(f"{IMAGE_TOKEN}Describe this image briefly.")
    request_config = {"prompts": prompt, "images": image, "modalities": ["text"]}

    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params_thinker, indirect=True)
def test_audio_to_text(omni_runner, omni_runner_handler) -> None:
    """
    Test audio understanding with text output.
    Input Modal: audio + text
    Output Modal: text
    """
    audio = generate_synthetic_audio(2, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    prompt = build_prompt(f"{AUDIO_TOKEN}Please recognize the language of this speech and transcribe it. Format: oral.")
    request_config = {"prompts": prompt, "audios": audio, "modalities": ["text"]}

    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params_thinker, indirect=True)
def test_video_to_text(omni_runner, omni_runner_handler) -> None:
    """
    Test video understanding with text output.
    Input Modal: video + text
    Output Modal: text
    """
    video = generate_synthetic_video(224, 224, 30)["np_array"]
    prompt = build_prompt(f"{VIDEO_TOKEN}Describe what is happening in this video.")
    request_config = {"prompts": prompt, "videos": video, "modalities": ["text"]}

    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params_thinker, indirect=True)
def test_mixed_to_text(omni_runner, omni_runner_handler) -> None:
    """
    Test mixed modality input (image + audio) with text output.
    Input Modal: image + audio + text
    Output Modal: text
    """
    image = generate_synthetic_image(224, 224)["np_array"]
    audio = generate_synthetic_audio(2, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    prompt = build_prompt(f"{IMAGE_TOKEN}{AUDIO_TOKEN}Describe the image and transcribe the audio.")
    request_config = {"prompts": prompt, "images": image, "audios": audio, "modalities": ["text"]}

    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_audio(omni_runner, omni_runner_handler) -> None:
    """
    Test text input with audio output via the thinker+talker pipeline.
    Input Modal: text
    Output Modal: audio
    """
    prompt = build_prompt("请简单介绍一下北京。")
    request_config = {"prompts": prompt, "modalities": ["audio"]}

    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_image_to_audio(omni_runner, omni_runner_handler) -> None:
    """
    Test image + text input with audio output via the thinker+talker pipeline.
    Input Modal: image + text
    Output Modal: audio
    """
    image = generate_synthetic_image(224, 224)["np_array"]
    prompt = build_prompt(f"{IMAGE_TOKEN}Describe this image briefly.")
    request_config = {"prompts": prompt, "images": image, "modalities": ["audio"]}

    omni_runner_handler.send_omni_request(request_config)
