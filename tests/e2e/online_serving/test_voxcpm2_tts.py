# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
E2E Online tests for VoxCPM2 with text input and audio output.

These tests verify the /v1/audio/speech endpoint works correctly with
actual model inference, not mocks. VoxCPM2 zero-shot needs only ``input``
and a placeholder ``voice`` (``default``); no ``task_type`` / ``ref_text``.
Voice clone uses ``ref_audio`` only. The reference clip is vendored under
tests/assets/qwen3_tts/clone_2.wav.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("VOXCPM2_MODEL", "openbmb/VoxCPM2")
DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 300.0
MAX_CONCURRENT = 4

# ~0.5 s of 48 kHz mono PCM_16 in WAV (~48k payload + header).
_MIN_AUDIO_BYTES = 40_000
REF_AUDIO_URL = get_asset_path("qwen3_tts/clone_2.wav", as_data_url=True)


def get_prompt(prompt_type="text"):
    """English prompt for zero-shot TTS (Whisper-friendly for advanced_model accuracy)."""
    prompts = {
        "text": "The weather is nice today, perfect for a walk in the park.",
    }
    return prompts.get(prompt_type, prompts["text"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("ci/voxcpm2.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="voxcpm2",
    )
]


# Weekly CUDA and ROCm collect these four vanilla cases in one pytest process.
# ``omni_server`` is module-scoped and is reused only while the next item's
# OmniServerParams compare equal. Keep this block contiguous. Startup LoRA
# stays in test_voxcpm2_tts_lora.py so this server is torn down before that
# deploy and before Voxtral.
@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, online_client) -> None:
    """
    Test zero-shot text-to-audio via OpenAI API.
    Deploy Setting: voxcpm2.yaml
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
        "voice": "default",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    online_client.send_audio_speech_request(request_config, request_num=MAX_CONCURRENT)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_002(omni_server, online_client) -> None:
    """
    Test zero-shot streaming text-to-audio via OpenAI API.
    Deploy Setting: voxcpm2.yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=True
    Datasets: single request
    """
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


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_streaming_001(omni_server, online_client) -> None:
    """
    Test voice-clone text-to-audio via OpenAI API.
    Deploy Setting: voxcpm2.yaml
    Input Modal: text + ref_audio
    Output Modal: audio
    Input Setting: stream=True
    Datasets: few requests (max_num_seqs=4)
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": True,
        "stream_format": "audio",
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "voice": "default",
        "ref_audio": REF_AUDIO_URL,
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }
    online_client.send_audio_speech_request(request_config, request_num=MAX_CONCURRENT)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "npu": "A2"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_response_format_001(omni_server, online_client) -> None:
    """
    Test voice-clone non-stream PCM output via OpenAI API.
    Deploy Setting: voxcpm2.yaml
    Input Modal: text + ref_audio
    Output Modal: audio (pcm)
    Input Setting: stream=False
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "response_format": "pcm",
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "voice": "default",
        "ref_audio": REF_AUDIO_URL,
        "min_audio_bytes": _MIN_AUDIO_BYTES,
        "min_hnr_db": -2.0,
    }
    online_client.send_audio_speech_request(request_config)
