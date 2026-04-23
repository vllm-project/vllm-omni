# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import (
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
)
from tests.helpers.stage_config import modify_stage_config

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


def get_eager_config():
    path = modify_stage_config(
        str(Path(__file__).parent.parent / "stage_configs" / "bailingmm_moe_v2_lite_thinker_only_ci.yaml"),
        updates={
            "stage_args": {
                0: {
                    "engine_args.enforce_eager": "true",
                },
            },
        },
    )
    return path


def get_eager_tts_config():
    """Thinker+talker CI config with enforce_eager on the thinker stage."""
    path = modify_stage_config(
        str(Path(__file__).parent.parent / "stage_configs" / "bailingmm_moe_v2_lite_ci.yaml"),
        updates={
            "stage_args": {
                0: {
                    "engine_args.enforce_eager": "true",
                },
            },
        },
    )
    return path


# Thinker-only config — used by text-output tests.
stage_configs = [get_eager_config()]
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]

# Thinker+talker config — used by audio-output tests.
stage_configs_tts = [get_eager_tts_config()]
test_params_tts = [(model, stage_config) for model in models for stage_config in stage_configs_tts]


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text(omni_runner, omni_runner_handler) -> None:
    """
    Test text-only input processing and text output generation.
    Input Modal: text
    Output Modal: text
    """
    prompt = build_prompt("请详细介绍鹦鹉的生活习性。")
    request_config = {"prompts": prompt, "modalities": ["text"]}

    omni_runner_handler.send_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_image_to_text(omni_runner, omni_runner_handler) -> None:
    """
    Test image understanding with text output.
    Input Modal: image + text
    Output Modal: text
    """
    image = generate_synthetic_image(224, 224)["np_array"]
    prompt = build_prompt(f"{IMAGE_TOKEN}Describe this image briefly.")
    request_config = {"prompts": prompt, "images": image, "modalities": ["text"]}

    omni_runner_handler.send_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
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

    omni_runner_handler.send_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_text(omni_runner, omni_runner_handler) -> None:
    """
    Test video understanding with text output.
    Input Modal: video + text
    Output Modal: text
    """
    video = generate_synthetic_video(224, 224, 30)["np_array"]
    prompt = build_prompt(f"{VIDEO_TOKEN}Describe what is happening in this video.")
    request_config = {"prompts": prompt, "videos": video, "modalities": ["text"]}

    omni_runner_handler.send_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
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

    omni_runner_handler.send_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params_tts, indirect=True)
def test_text_to_audio(omni_runner, omni_runner_handler) -> None:
    """
    Test text input with audio output via the thinker+talker pipeline.
    Input Modal: text
    Output Modal: audio
    """
    prompt = build_prompt("请简单介绍一下北京。")
    request_config = {"prompts": prompt, "modalities": ["audio"]}

    omni_runner_handler.send_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_runner", test_params_tts, indirect=True)
def test_image_to_audio(omni_runner, omni_runner_handler) -> None:
    """
    Test image + text input with audio output via the thinker+talker pipeline.
    Input Modal: image + text
    Output Modal: audio
    """
    image = generate_synthetic_image(224, 224)["np_array"]
    prompt = build_prompt(f"{IMAGE_TOKEN}Describe this image briefly.")
    request_config = {"prompts": prompt, "images": image, "modalities": ["audio"]}

    omni_runner_handler.send_request(request_config)
