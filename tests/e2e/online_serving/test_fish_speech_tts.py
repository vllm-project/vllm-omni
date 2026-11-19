# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E online tests for Fish Speech S2 Pro (`fishaudio/s2-pro`).

Covers `/v1/audio/speech` with the bundled two-stage SlowAR → DAC pipeline.
Zero-shot TTS needs `input` and placeholder `voice=default`. Voice clone
also needs `ref_audio` and `ref_text`.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("FISH_SPEECH_MODEL", "fishaudio/s2-pro")
DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 300.0

# ~0.5 s of 44.1 kHz mono PCM_16 in WAV (~44k payload + header).
_MIN_AUDIO_BYTES = 40_000

# Vendored under tests/assets/qwen3_tts/clone_2.wav so the server does not
# fetch reference audio over HTTPS at request time (see issue #3263).
REF_AUDIO_URL = get_asset_path("qwen3_tts/clone_2.wav", as_data_url=True)
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."


def get_prompt(prompt_type: str = "text") -> str:
    """English prompt for zero-shot TTS."""
    prompts = {
        "text": "The weather is nice today, perfect for a walk in the park.",
    }
    return prompts.get(prompt_type, prompts["text"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("fish_qwen3_omni.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="fish_speech",
    )
]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, online_client) -> None:
    """Zero-shot non-streaming TTS via `/v1/audio/speech`."""
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "voice": "default",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    online_client.send_audio_speech_request(request_config)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_002(omni_server, online_client) -> None:
    """Zero-shot streaming TTS via `/v1/audio/speech`."""
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": True,
        "stream_format": "audio",
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "voice": "default",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    online_client.send_audio_speech_request(request_config)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_001(omni_server, online_client) -> None:
    """Voice-clone TTS via `/v1/audio/speech` (`ref_audio` + `ref_text`)."""
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "voice": "default",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    online_client.send_audio_speech_request(request_config)
