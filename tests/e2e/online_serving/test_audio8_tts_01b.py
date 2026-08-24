# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E online tests for Audio8 TTS Preview 0.1b via /v1/audio/speech.

Real model inference (no mocks): text-only synthesis, streaming, and zero-shot
voice cloning through the Falcon-H1 hybrid Slow AR. The serving path is shared
with the 0.6b variant (same adapter); only the model and deploy config differ.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("AUDIO8_TTS_01B_MODEL_PATH", "Audio8/Audio8-TTS-Preview-0.1b")
DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 300.0
# The codec decodes at 44.1 kHz; raw PCM carries no header, so rate-dependent
# assertions (HNR pitch search) must be told the rate explicitly.
SAMPLE_RATE = 44100
MAX_CONCURRENT = 4
_MIN_AUDIO_BYTES = 40_000

REF_AUDIO_URL = get_asset_path("cosyvoice3/zero_shot_prompt.wav", as_data_url=True)
REF_TEXT = "希望你以后能够做的比我还好呦。"


def get_prompt(prompt_type: str = "en") -> str:
    prompts = {
        "en": "The weather is nice today, perfect for a walk in the park.",
        "zh": "今天天气很好，非常适合去公园散步。",
    }
    return prompts.get(prompt_type, prompts["en"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("audio8_tts_01b.yaml"),
            server_args=["--disable-log-stats"],
        ),
        id="audio8_tts_01b",
    )
]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client) -> None:
    """Baseline smoke: default deploy, non-streaming WAV."""
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    openai_client.send_audio_speech_request(request_config, request_num=MAX_CONCURRENT)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_streaming_002(omni_server, openai_client) -> None:
    """Streaming PCM: exercises the async_chunk path end to end."""
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": True,
        "stream_format": "audio",
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "pcm",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
        "expected_sample_rate": SAMPLE_RATE,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_003(omni_server, openai_client) -> None:
    """Zero-shot voice cloning: reference audio encoded model-side and spliced
    into the prompt."""
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("zh"),
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    openai_client.send_audio_speech_request(request_config)
