# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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


@hardware_test(res={"cuda": "L4", "xpu": "B60"}, num_cards=1)
def test_speech_english_basic(omni_server, online_client) -> None:
    """Test basic English TTS generation."""
    online_client.send_audio_speech_request(
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


@hardware_test(res={"cuda": "L4", "xpu": "B60"}, num_cards=1)
def test_speech_english_streaming(omni_server, online_client) -> None:
    """Test basic streaming English TTS generation (PCM via streaming API)."""
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "Hello, how are you?",
            "voice": "casual_female",
            "language": "English",
            "stream": True,
            "stream_format": "audio",
            "response_format": "pcm",
            "timeout": 120.0,
            "min_audio_bytes": _MIN_AUDIO_BYTES_BASIC,
        }
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_speech_different_voices(omni_server, online_client) -> None:
    """Test TTS with different voice presets."""
    voices = ["casual_female", "neutral_male"]
    for voice in voices:
        online_client.send_audio_speech_request(
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
def test_speech_speed(omni_server, online_client) -> None:
    """Request with speed parameters."""
    speeds = [0.5, 1, 1.5, 2, 2.5]
    for speed in speeds:
        online_client.send_audio_speech_request(
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
def test_speech_instructions(omni_server, online_client) -> None:
    """Request with instructions parameters."""
    instructions = [
        "Speak formally",
        "Speak angrily",
        "Deliver with a sad voice",
        "Speak with a chirpy happy voice",
    ]
    for instruction in instructions:
        online_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                "input": "The boy was there when the sun rose.",
                "voice": "casual_female",
                "language": "English",
                "response_format": "wav",
                "timeout": 120.0,
                "instructions": instruction,
                # Re-grade a below-threshold result with whisper large-v3 before
                # failing, so a miss is attributable rather than ambiguous. Same
                # opt-in as test_higgs_audio_v3.py.
                #
                # Note: for this input it does NOT rescue the check — small and
                # large-v3 both transcribe "The boy was there when sun rose."
                # (sim=0.87 each) against an input of "...when the sun rose.",
                # i.e. the TTS really does drop the article. That is the value
                # here: two independent ASR models agreeing rules out a grader
                # flake and points at the model.
                "transcript_escalation_model": "large-v3",
            }
        )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_speech_response_formats(omni_server, online_client) -> None:
    """Test TTS with different response formats."""
    response_formats = ["wav", "mp3"]
    for response_format in response_formats:
        online_client.send_audio_speech_request(
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
def test_speech_batches(omni_server, online_client) -> None:
    """Test TTS batches."""
    items = [
        {"input": "The birch canoe slid on the smooth planks."},
        {"input": "Glue the sheet to the dark blue background."},
        {"input": "It's easy to tell the depth of a well."},
        {"input": "These days a chicken leg is a rare dish."},
        {"input": "Rice is often served in round bowls."},
    ]

    online_client.send_audio_speech_batch_http_request(
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
