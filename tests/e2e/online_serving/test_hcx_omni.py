# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E online serving tests for HyperCLOVAX-SEED-Omni-8B.

Tests the OpenAI-compatible HTTP API for Speech-to-Speech and
Text-to-Vision generation.
"""

import os
from pathlib import Path

import pytest

from tests.conftest import (
    OmniServerParams,
    generate_synthetic_audio,
    generate_synthetic_image,
)
from tests.utils import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B"
_CI_YAML = str(Path(__file__).parent.parent / "stage_configs" / "hcx_omni_ci.yaml")

test_params = [OmniServerParams(model=MODEL, stage_config_path=_CI_YAML)]

SYSTEM_PROMPT = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": (
                "당신은 CLOVA X입니다. 네이버가 만든 AI 어시스턴트로서 "
                "오디오와 이미지를 인식하고 텍스트, 음성, 이미지를 생성할 수 있습니다."
            ),
        }
    ],
}


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards={"cuda": 3})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_speech_to_speech(omni_server, omni_server_handler) -> None:
    """Speech-to-Speech: audio input → text + audio response."""
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()

    messages = [
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio, "format": "wav"},
                },
                {"type": "text", "text": "이 오디오에서 무슨 내용이 들리나요?"},
            ],
        },
    ]
    request_config = {
        "messages": messages,
        "modalities": ["text", "audio"],
        "stream": False,
    }
    response = omni_server.chat(request_config)
    assert response is not None


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards={"cuda": 3})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_vision(omni_server, omni_server_handler) -> None:
    """Text-to-Vision: text prompt → text + image response."""
    messages = [
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "고양이 한 마리가 소파에 앉아 있는 그림을 그려줘."},
            ],
        },
    ]
    request_config = {
        "messages": messages,
        "modalities": ["text", "image"],
        "stream": False,
    }
    response = omni_server.chat(request_config)
    assert response is not None


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards={"cuda": 3})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_image_understanding(omni_server, omni_server_handler) -> None:
    """Image understanding: image input → text description."""
    image = generate_synthetic_image(224, 224)["np_array"]
    messages = [
        SYSTEM_PROMPT,
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"},
                },
                {"type": "text", "text": "이 이미지에 무엇이 있나요?"},
            ],
        },
    ]
    request_config = {
        "messages": messages,
        "modalities": ["text"],
        "stream": False,
    }
    response = omni_server.chat(request_config)
    assert response is not None
