# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
E2E tests for Qwen2.5-Omni model with mixed modality inputs, audio and text output.
"""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import (
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
)
from tests.helpers.stage_config import get_deploy_config_path

models = ["Qwen/Qwen2.5-Omni-7B"]

# Single CI deploy YAML; rocm/xpu deltas are picked automatically via the
# platforms: section. NPU still uses the legacy per-platform YAML until it
# also migrates to the new schema.
_CI_DEPLOY = get_deploy_config_path("ci/qwen2_5_omni.yaml")


# The bundled deploy YAML runs the talker with CUDA graphs on CUDA and keeps
# eager on ROCm, on the NPU talker and on XPU through its ``platforms:``
# section, so every platform runs this test on the same resolved config.
stage_config = _CI_DEPLOY

# Create parameter combinations for model and stage config
# Qwen2.5-Omni with TP=3 needs longer init timeout
test_params = [(model, stage_config, {"stage_init_timeout": 1200, "init_timeout": 1800}) for model in models]


def get_question(prompt_type="mix"):
    prompts = {
        "mix": "What is recited in the audio? What is in this image? Describe the video briefly.",
        "text_only": "What is the capital of China?",
    }
    return prompts.get(prompt_type, prompts["mix"])


@pytest.mark.slow
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards={"cuda": 4, "rocm": 2, "xpu": 3})
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_mix_to_audio(omni_runner, offline_client) -> None:
    """
    Test multi-modal input processing and text/audio output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text + audio + video + image
    Output Modal: audio
    Input Setting: stream=False
    Datasets: single request
    """
    video = generate_synthetic_video(16, 16, 30)["np_array"]
    image = generate_synthetic_image(16, 16)["np_array"]
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()

    request_config = {
        "prompts": get_question(),
        "videos": video,
        "images": image,
        "audios": (audio, 16000),
        "modalities": ["audio"],
    }

    # Test single completion
    offline_client.send_omni_request(request_config)


@pytest.mark.slow
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards={"cuda": 4, "rocm": 2, "xpu": 3})
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text(omni_runner, offline_client) -> None:
    """
    Test text input processing and text output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: text
    Input Setting: stream=False
    Datasets: single request
    """

    request_config = {"prompts": get_question("text_only"), "modalities": ["text"]}

    # Test single completion
    offline_client.send_omni_request(request_config)
