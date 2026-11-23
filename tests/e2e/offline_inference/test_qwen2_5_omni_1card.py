# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen2.5-Omni (3B):  thinker + talker + code2wav on a single XPU card."""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import (
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
)
from tests.helpers.stage_config import get_deploy_config_path

models = ["Qwen/Qwen2.5-Omni-3B"]
stage_config = get_deploy_config_path("ci/qwen2_5_omni_1card.yaml")

test_params = [(model, stage_config, {"stage_init_timeout": 1200, "init_timeout": 1800}) for model in models]


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "xpu": "B60"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text_1card(omni_runner, offline_client) -> None:
    request_config = {"prompts": "What is the capital of China?", "modalities": ["text"]}
    offline_client.send_omni_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "xpu": "B60"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_mix_to_audio_1card(omni_runner, offline_client) -> None:
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
    offline_client.send_omni_request(request_config)
