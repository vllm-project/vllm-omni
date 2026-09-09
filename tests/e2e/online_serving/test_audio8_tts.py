# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E online tests for Audio8 TTS Preview via /v1/audio/speech.

Real model inference (no mocks): text-only synthesis, streaming, and zero-shot
voice cloning. The reference audio is inlined as a base64 data URL because CI
hosts cannot fetch external URLs.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("AUDIO8_TTS_MODEL_PATH", "Audio8/Audio8-TTS-Preview-0.6b")
DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 300.0
# The codec decodes at 44.1 kHz; raw PCM carries no header, so assertions that
# depend on the rate (HNR pitch search) have to be told explicitly.
SAMPLE_RATE = 44100
MAX_CONCURRENT = 4

# ~0.5 s of 44.1 kHz mono PCM_16 WAV.
_MIN_AUDIO_BYTES = 40_000

# Reused vendored asset (3.5 s, Chinese) plus its exact transcript: Audio8 TTS
# requires the reference transcript to match the recording.
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
            stage_config_path=get_deploy_config_path("audio8_tts.yaml"),
            # No --trust-remote-code: vllm-omni owns the `arktts` config.
            server_args=["--disable-log-stats"],
        ),
        id="audio8_tts",
    )
]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client) -> None:
    """Baseline smoke: default deploy, non-streaming WAV.

    Deploy Setting: audio8_tts.yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=False
    Datasets: few requests (max_num_seqs=4)
    """
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
    """Streaming PCM: exercises the async_chunk path end to end.

    Deploy Setting: audio8_tts.yaml (async_chunk: true)
    Input Setting: stream=True, response_format=pcm
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": True,
        "stream_format": "audio",
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "pcm",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
        # Raw PCM has no header: without this the harness assumes 24 kHz and the
        # HNR pitch search lands on 147-735 Hz instead of 80-400 Hz, scoring
        # clean 44.1 kHz speech ~1.9 dB too low.
        "expected_sample_rate": SAMPLE_RATE,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_003(omni_server, openai_client) -> None:
    """Zero-shot voice cloning: the reference audio is encoded model-side and
    its codec codes are spliced into the prompt.

    Input Modal: text + ref_audio + ref_text
    """
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


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_requires_ref_text_004(omni_server, openai_client) -> None:
    """``ref_audio`` without ``ref_text`` must be rejected, not silently
    synthesized with an untethered voice."""
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "ref_audio": REF_AUDIO_URL,
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "status_code": (400, 422),
        "err_message": "ref_text",
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_empty_input_is_rejected_005(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "input": "   ",
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "status_code": (400, 422),
        "err_message": "Input text cannot be empty",
    }
    openai_client.send_audio_speech_request(request_config)
