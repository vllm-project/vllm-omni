# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E online tests for Fish Audio Drama 3 via /v1/audio/speech.

Real model inference (no mocks): text-only synthesis, streaming, voice cloning
and concurrent requests. The reference audio is inlined as a base64 data URL
because CI hosts cannot fetch external URLs.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "fishaudio/drama-3"
DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 300.0
# The DAC codec decodes at 44.1 kHz; raw PCM carries no header.
SAMPLE_RATE = 44100
MAX_CONCURRENT = 4

# ~0.5 s of 44.1 kHz mono PCM_16 WAV.
_MIN_AUDIO_BYTES = 40_000

REF_AUDIO_URL = get_asset_path("qwen3_tts/clone_2.wav", as_data_url=True)
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."


def get_prompt(prompt_type: str = "en") -> str:
    prompts = {
        "en": "The weather is nice today, perfect for a walk in the park.",
        "clone": "The quarterly report is due on Thursday, so please finish it by noon.",
    }
    return prompts.get(prompt_type, prompts["en"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("fish_omni.yaml"),
            # No --trust-remote-code: vllm-omni owns the `fish_omni` config.
            server_args=["--disable-log-stats"],
            stage_init_timeout=900,
        ),
        id="drama_3",
    )
]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client) -> None:
    """Baseline smoke: default deploy, non-streaming WAV.

    Deploy Setting: fish_omni.yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_streaming_002(omni_server, openai_client) -> None:
    """Voice cloning over the async_chunk streaming path.

    Deploy Setting: fish_omni.yaml (async_chunk: true)
    Input Modal: text + ref_audio + ref_text
    Input Setting: stream=True, response_format=pcm
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("clone"),
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "stream": True,
        "stream_format": "audio",
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "pcm",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
        "expected_sample_rate": SAMPLE_RATE,
        "transcript_pcm_sample_rate": SAMPLE_RATE,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_concurrent_003(omni_server, openai_client) -> None:
    """Concurrent voice-clone requests keep their own codebook-tower state.

    Input Modal: text + ref_audio + ref_text
    Datasets: MAX_CONCURRENT identical requests (max_num_seqs=8)
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("clone"),
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    openai_client.send_audio_speech_request(request_config, request_num=MAX_CONCURRENT)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_requires_ref_text_004(omni_server, openai_client) -> None:
    """``ref_audio`` without ``ref_text`` is rejected."""
    openai_client.send_audio_speech_http_request(
        {
            "json": {
                "model": omni_server.model,
                "input": get_prompt(),
                "ref_audio": REF_AUDIO_URL,
                "response_format": "wav",
            },
            "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        },
        err_code=400,
        err_message="ref_text",
    )
