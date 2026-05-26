# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for GLM-TTS model with text input and audio output.

These tests verify the /v1/audio/speech endpoint works correctly with
the GLM-TTS two-stage pipeline (AR + DiT).
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import load_test_audio_data_url
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("GLM_TTS_MODEL_PATH", "zai-org/GLM-TTS")
REF_TEXT = "他当时还跟线下其他的站姐吵架，然后，打架进局子了。"

DEPLOY_CONFIG = get_deploy_config_path("glm_tts.yaml")

# Official GLM-TTS reference audio and transcript. Vendored under tests/assets/
# so Buildkite does not depend on raw.githubusercontent.com during E2E.
REF_AUDIO_URL = load_test_audio_data_url("glm_tts/jiayan_zh.wav")


SYNC_EXTRA_ARGS = [
    "--trust-remote-code",
    "--disable-log-stats",
    "--no-async-chunk",
]

ASYNC_EXTRA_ARGS = [
    "--trust-remote-code",
    "--disable-log-stats",
]

tts_sync_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            server_args=SYNC_EXTRA_ARGS,
            stage_init_timeout=600,
        ),
        id="glm_tts_sync",
    )
]

tts_async_chunk_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            server_args=ASYNC_EXTRA_ARGS,
            stage_init_timeout=600,
        ),
        id="glm_tts_async_chunk",
    )
]


# GLM-TTS is a zero-shot voice cloning model (always requires ref_audio).
# There is no text-only / non-clone path — the official implementation
# asserts prompt_feat.shape[1] != 0 in token2wav_with_cache and every
# example prompt includes a reference audio.  CosyVoice3 follows the same
# pattern: all tests are voice_clone_*.


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_sync_server_params, indirect=True)
def test_voice_clone_zh_sync(omni_server, openai_client) -> None:
    """
    Test voice cloning TTS with Chinese text.
    Deploy Setting: glm_tts.yaml with ``--no-async-chunk`` (sync two-stage)
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio
    Input Setting: stream=False

    Uses the official jiayan_zh.wav reference audio from the upstream
    GLM-TTS repository (real human speech matching REF_TEXT).
    """
    request_config = {
        "model": omni_server.model,
        "input": "今天天气真不错，适合出去散散步。",
        "stream": False,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_async_chunk_server_params, indirect=True)
def test_voice_clone_zh_async_chunk(omni_server, openai_client) -> None:
    """
    Test voice cloning TTS with Chinese text via async_chunk streaming.
    Deploy Setting: glm_tts.yaml default ``async_chunk: true``
    Input Modal: text + ref_audio + ref_text
    Output Modal: audio
    Input Setting: stream=True
    """
    request_config = {
        "model": omni_server.model,
        "input": "今天天气真不错，适合出去散散步。",
        "stream": True,
        "response_format": "wav",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_sync_server_params, indirect=True)
def test_models_endpoint(omni_server, openai_client) -> None:
    """Test the /v1/models endpoint returns loaded model."""
    models = openai_client.client.models.list()
    assert len(models.data) > 0
