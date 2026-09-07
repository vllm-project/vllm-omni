# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
E2E Online tests for F5-TTS (flow-matching diffusion TTS).

Tests verify the /v1/audio/speech endpoint with the F5-TTS diffusion
pipeline, covering voice cloning, custom sampling parameters, and the
recommended Cache-DiT acceleration.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams

MODEL = "SWivid/F5-TTS/F5TTS_v1_Base"

# F5-TTS official example reference audio
REF_AUDIO_URL = (
    "https://raw.githubusercontent.com/SWivid/F5-TTS/main/"
    "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
)
REF_TEXT = "Some call me nature, others call me mother nature."


def get_prompt(prompt_type="en"):
    prompts = {
        "en": (
            "I don't really care what you call me. "
            "I've been a silent spectator, watching species evolve, "
            "empires rise and fall. But always, I am here."
        ),
        "zh": "今天天气真好，我们一起去公园散步吧。",
    }
    return prompts.get(prompt_type, prompts["en"])


# F5-TTS is an unregistered single-stage diffusion model: serving builds
# the default diffusion stage config (async_omni_engine fallback), so no
# deploy config or stage overrides are needed.
tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=[
                "--trust-remote-code",
                "--enforce-eager",
                "--omni",
                "--disable-log-stats",
            ],
        ),
        id="f5_tts_v1_base",
    )
]

tts_cache_dit_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=[
                "--trust-remote-code",
                "--enforce-eager",
                "--omni",
                "--disable-log-stats",
                "--cache-backend",
                "cache_dit",
            ],
        ),
        id="f5_tts_cache_dit",
    )
]


@pytest.mark.core_model
@pytest.mark.tts
@pytest.mark.omni
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_f5_tts_voice_clone(omni_server, openai_client) -> None:
    """F5-TTS voice cloning with reference audio and text."""
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config, request_num=3)


@pytest.mark.core_model
@pytest.mark.tts
@pytest.mark.omni
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_f5_tts_custom_params(omni_server, openai_client) -> None:
    """F5-TTS with custom inference steps, guidance scale, and seed."""
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "num_inference_steps": 16,
        "guidance_scale": 2.0,
        "seed": 42,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@pytest.mark.omni
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_cache_dit_server_params, indirect=True)
def test_f5_tts_cache_dit(omni_server, openai_client) -> None:
    """F5-TTS with Cache-DiT acceleration (recommended)."""
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("en"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)
