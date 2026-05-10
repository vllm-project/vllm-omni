# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-TTS model with text input and audio output.

These tests verify the /v1/audio/speech endpoint works correctly with
actual model inference, not mocks.
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import load_test_audio_data_url
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.full_model, pytest.mark.tts]

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# See tests/e2e/online_serving/test_qwen3_tts_base.py for the vendored-asset rationale.
REF_AUDIO_URL = load_test_audio_data_url("qwen3_tts/clone_2.wav")
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."


def get_prompt(prompt_type="text"):
    """Text prompt for text-to-audio tests (same as test_qwen3_omni - beijing test case)."""
    prompts = {
        "text": "The weather is nice today, perfect for a walk in the park.",
    }
    return prompts.get(prompt_type, prompts["text"])


def get_max_batch_size(size_type="few"):
    """Batch size for concurrent requests (same as test_qwen3_omni)."""
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("qwen3_tts.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="async_chunk",
    ),
    # Synchronous (no async-chunk) variant — ``--no-async-chunk`` alone
    # flips the deploy yaml's bool and the pipeline dispatches to the
    # end-to-end codec processor. No variant yaml / pipeline needed.
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("qwen3_tts.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats", "--no-async-chunk"],
        ),
        id="no_async_chunk",
    ),
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_voice_clone_streaming_001(omni_server, openai_client) -> None:
    """
    Test text input processing and audio output via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=True
    Datasets: few requests
    """

    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": True,
        "response_format": "wav",
        "task_type": "Base",
        "voice": "clone",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config, request_num=get_max_batch_size("few"))


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_response_format_001(omni_server, openai_client) -> None:
    """
    Test text input processing and audio output via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: audio
    Input Setting: non-stream PCM
    Datasets: few requests
    """

    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "response_format": "pcm",
        "task_type": "Base",
        "voice": "clone",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)
