# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for MOSS-TTS-Nano model with text input and audio output.

These tests verify the /v1/audio/speech endpoint works correctly with
actual model inference, not mocks. The server uses upstream's
``voice_clone`` mode (no transcript needed); ``ref_text`` and ``voice``
are accepted in the OpenAI schema but ignored. One case includes a
non-empty ``ref_text`` to verify it does not break the pipeline. The
reference clip is fetched from the upstream repo (assets/audio/zh_1.wav,
~50 KB).
"""

import base64
import os
import urllib.request

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

# ``omni`` for all tests; 001/002 also get ``full_model`` via ``TestMossTtsNanoFull`` (003 is core+advanced only, no full_model).
pytestmark = [pytest.mark.full_model, pytest.mark.tts]

_SKIP_ISSUE_3168 = pytest.mark.skip(
    reason="https://github.com/vllm-project/vllm-omni/issues/3168",
)

MODEL = "OpenMOSS-Team/MOSS-TTS-Nano"
REF_AUDIO_URL = "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS-Nano/main/assets/audio/zh_1.wav"
REF_AUDIO_TRANSCRIPT = "欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。"


@pytest.fixture(scope="session")
def ref_audio_data_url() -> str:
    """Fetch the upstream sample clip and return it as a base64 data URL.

    The fetch failure is escalated to a hard failure (not pytest.skip) so that
    a broken network path does not silently mask regressions in
    /v1/audio/speech. Set ``MOSS_TTS_NANO_SKIP_ON_NET_FAIL=1`` to opt into
    skipping in air-gapped environments.
    """
    try:
        with urllib.request.urlopen(REF_AUDIO_URL, timeout=30) as resp:
            data = resp.read()
    except Exception as e:
        msg = f"Cannot fetch upstream reference clip {REF_AUDIO_URL}: {e}"
        if os.environ.get("MOSS_TTS_NANO_SKIP_ON_NET_FAIL"):
            pytest.skip(msg)
        pytest.fail(msg)
    if not data:
        pytest.fail(f"Reference clip empty: {REF_AUDIO_URL}")
    return f"data:audio/wav;base64,{base64.b64encode(data).decode('ascii')}"


def get_prompt(prompt_type="text"):
    """Text prompt for text-to-audio tests.

    Avoid the model's own name ("MOSS-TTS-Nano") in the test input — the
    codec consistently mispronounces it ("MOS's TTS Nano", "Was TTS Nano",
    etc.), which trips the transcript-similarity assertion in CI without
    indicating a real regression. Use plain natural-sounding sentences.
    """
    prompts = {
        "text": "Hello, this is a short voice cloning demo for testing.",
        "chinese": "你好，这是一段简单的语音合成测试。",
    }
    return prompts.get(prompt_type, prompts["text"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("moss_tts_nano.yaml"),
            server_args=["--disable-log-stats"],
        ),
        id="moss_tts_nano",
    )
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client, ref_audio_data_url) -> None:
    """
    Test voice_clone mode (no ref_text) via /v1/audio/speech.
    Deploy Setting: default yaml
    Input Modal: text + reference audio (no transcript)
    Output Modal: audio (48 kHz, WAV)
    Input Setting: stream=False
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "response_format": "wav",
        "ref_audio": ref_audio_data_url,
    }

    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_002(omni_server, openai_client, ref_audio_data_url) -> None:
    """
    Test streaming voice_clone via /v1/audio/speech.
    Deploy Setting: default yaml
    Input Modal: text + reference audio (no transcript)
    Output Modal: audio (48 kHz, PCM stream)
    Input Setting: stream=True
    Datasets: single request

    NOTE: ``min_hnr_db=-5.0`` overrides the conftest default of 1.0 dB.
    MOSS-TTS-Nano's voice_clone output is intrinsically noisy by that
    metric — verified on H20, both streaming and non-streaming clips
    measure HNR around -2 dB even when the audio sounds correct. We
    keep the PCM-stream check in place to catch catastrophic decoder
    failures (which produce HNR much further below 0), just with a
    threshold MOSS can clear in normal operation.
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": True,
        "response_format": "pcm",
        "ref_audio": ref_audio_data_url,
        "min_hnr_db": -5.0,
    }

    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_003(omni_server, openai_client, ref_audio_data_url) -> None:
    """
    Test Chinese voice_clone via /v1/audio/speech.
    Deploy Setting: default yaml
    Input Modal: text (Chinese) + reference audio (no transcript)
    Output Modal: audio (48 kHz, WAV)
    Input Setting: stream=False
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("chinese"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": ref_audio_data_url,
    }

    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_004_ref_text_ignored(omni_server, openai_client, ref_audio_data_url) -> None:
    """
    Sending ``ref_text`` alongside ``ref_audio`` is accepted but ignored.

    The OpenAI ``ref_text`` field is part of the schema, so clients may
    send it. The server uses upstream's voice_clone mode regardless and
    drops the transcript; this case verifies the request still completes
    and produces audio.
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("chinese"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": ref_audio_data_url,
        "ref_text": REF_AUDIO_TRANSCRIPT,
    }

    openai_client.send_audio_speech_request(request_config)
