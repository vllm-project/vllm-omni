# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E tests for Voxtral TTS online serving.

Smoke tests (core_model + advanced_model) and expansion scenarios (slow)
for the /v1/audio/speech endpoint.
"""

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "mistralai/Voxtral-4B-TTS-2603"
STAGE_CONFIG = get_deploy_config_path("voxtral_tts.yaml")
EXTRA_ARGS = ["--trust-remote-code", "--enforce-eager", "--disable-log-stats"]
TEST_PARAMS = [OmniServerParams(model=MODEL, stage_config_path=STAGE_CONFIG, server_args=EXTRA_ARGS)]
_MIN_AUDIO_BYTES_BASIC = 10000

pytestmark = [
    pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True),
    pytest.mark.slow,
    pytest.mark.tts,
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_speech_english_basic(omni_server, openai_client) -> None:
    """Test basic English TTS generation."""
    openai_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "how are you",
            "voice": "casual_female",
            "language": "English",
            "response_format": "wav",
            "timeout": 120.0,
            "min_audio_bytes": _MIN_AUDIO_BYTES_BASIC,
        }
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_speech_english_streaming(omni_server, openai_client) -> None:
    """Test basic streaming English TTS generation (PCM via streaming API)."""
    openai_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "Hello, how are you?",
            "voice": "casual_female",
            "language": "English",
            "stream": True,
            "response_format": "pcm",
            "timeout": 120.0,
            "min_audio_bytes": _MIN_AUDIO_BYTES_BASIC,
        }
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_speech_different_voices(omni_server, openai_client) -> None:
    """Test TTS with different voice presets."""
    voices = ["casual_female", "neutral_male"]
    for voice in voices:
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Testing voice selection.",
                "voice": voice,
                "language": "English",
                "response_format": "wav",
                "timeout": 120.0,
            }
        )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_speech_speed(omni_server, openai_client) -> None:
    """Request with speed parameters."""
    speeds = [0.5, 1, 1.5, 2, 2.5]
    for speed in speeds:
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "The boy was there when the sun rose.",
                "voice": "casual_female",
                "language": "English",
                "response_format": "wav",
                "timeout": 120.0,
                "speed": speed,
            }
        )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_speech_instructions(omni_server, openai_client) -> None:
    """Request with instructions parameters."""
    instructions = [
        "Speak formally",
        "Speak angrily",
        "Deliver with a sad voice",
        "Speak with a chirpy happy voice",
    ]
    for instruction in instructions:
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "The boy was there when the sun rose.",
                "voice": "casual_female",
                "language": "English",
                "response_format": "wav",
                "timeout": 120.0,
                "instructions": instruction,
            }
        )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_speech_response_formats(omni_server, openai_client) -> None:
    """Test TTS with different response formats."""
    response_formats = ["wav", "mp3"]
    for response_format in response_formats:
        openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "Testing various response formats.",
                "voice": "casual_male",
                "language": "English",
                "response_format": response_format,
                "timeout": 120.0,
            }
        )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_speech_batches(omni_server, openai_client) -> None:
    """Test TTS batches."""
    items = [
        {"input": "The birch canoe slid on the smooth planks."},
        {"input": "Glue the sheet to the dark blue background."},
        {"input": "It's easy to tell the depth of a well."},
        {"input": "These days a chicken leg is a rare dish."},
        {"input": "Rice is often served in round bowls."},
    ]

    openai_client.send_audio_speech_batch_http_request(
        {
            "json": {
                "model": omni_server.model,
                "items": items,
                "voice": "casual_male",
                "language": "English",
                "response_format": "wav",
            },
            "timeout": 120.0,
        }
    )
