# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online tests for the full MOSS-TTS family via /v1/audio/speech.

The offline suite (``tests/e2e/offline_inference/test_moss_tts_*_expansion.py``)
covers the engine path; this file covers the
*serving* path (``serving_speech.py`` → ``_detect_moss_variant`` → the
delay/realtime prompt builders), which the Nano online test does not exercise.

We use MOSS-TTS-Realtime (1.7B, ``MossTTSRealtime``) because it is the
smallest full-family variant and fits a single L4 — the 8B delay variants
(MOSS-TTS / v1.5) are H100-gated and stay in the offline suite. Realtime uses
the local-transformer generation path and, like the ``tts`` delay variant,
requires ``ref_audio`` for voice cloning.

The reference clip is fetched from the upstream repo
(``assets/audio/reference_zh_1.wav``, ~1.5 MB) and passed as a base64 data URL.
"""

import base64
import os
import urllib.request

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/4700"),
]

MODEL = "OpenMOSS-Team/MOSS-TTS-Realtime"
REF_AUDIO_URL = "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS/HEAD/assets/audio/reference_zh_1.wav"
# Voice-clone output is not reliably transcribed by the Whisper check used at
# full_model run_level; assert non-trivial WAV payload size instead.
_MIN_AUDIO_BYTES = 10_000


@pytest.fixture(scope="session")
def ref_audio_data_url() -> str:
    """Fetch the upstream sample clip and return it as a base64 data URL.

    The fetch failure is escalated to a hard failure (not pytest.skip) so a
    broken network path does not silently mask regressions in
    /v1/audio/speech. Set ``MOSS_TTS_SKIP_ON_NET_FAIL=1`` to opt into skipping
    in air-gapped environments.
    """
    try:
        with urllib.request.urlopen(REF_AUDIO_URL, timeout=30) as resp:
            data = resp.read()
    except Exception as e:  # noqa: BLE001
        msg = f"Cannot fetch upstream reference clip {REF_AUDIO_URL}: {e}"
        if os.environ.get("MOSS_TTS_SKIP_ON_NET_FAIL"):
            pytest.skip(msg)
        pytest.fail(msg)
    if not data:
        pytest.fail(f"Reference clip empty: {REF_AUDIO_URL}")
    return f"data:audio/wav;base64,{base64.b64encode(data).decode('ascii')}"


def get_prompt(prompt_type: str = "text") -> str:
    """Plain natural-sounding sentences for text-to-audio tests.

    Avoid the model's own name in the input — the codec mispronounces it,
    which can trip transcript-similarity checks without a real regression.
    """
    prompts = {
        "text": "Hello, this is a real-time voice cloning demo for testing.",
        "chinese": "你好，这是一段实时语音合成测试。",
    }
    return prompts.get(prompt_type, prompts["text"])


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("moss_tts_realtime.yaml"),
            server_args=["--disable-log-stats"],
        ),
        id="moss_tts_realtime",
    )
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client, ref_audio_data_url) -> None:
    """
    Realtime voice_clone via /v1/audio/speech, non-streaming.
    Deploy Setting: moss_tts_realtime.yaml
    Input Modal: text + reference audio (voice clone)
    Output Modal: audio (24 kHz, WAV)
    Input Setting: stream=False
    Datasets: single request

    NOTE: ``min_audio_bytes`` skips Whisper transcript similarity — cloned
    voice output is often empty or mismatched under ASR without indicating a
    real serving regression.
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt(),
        "stream": False,
        "response_format": "wav",
        "ref_audio": ref_audio_data_url,
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }

    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_002_streaming(omni_server, openai_client, ref_audio_data_url) -> None:
    """
    Realtime voice_clone via /v1/audio/speech, streaming PCM.
    Deploy Setting: moss_tts_realtime.yaml
    Input Modal: text + reference audio (voice clone)
    Output Modal: audio (24 kHz, PCM stream)
    Input Setting: stream=True
    Datasets: single request

    NOTE: ``min_hnr_db=-5.0`` mirrors the Nano streaming case — MOSS
    voice_clone output is intrinsically noisy by the HNR metric even when it
    sounds correct; the relaxed threshold still catches catastrophic decoder
    failures (HNR far below 0) without flaking on normal output.
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
def test_text_to_audio_003_chinese(omni_server, openai_client, ref_audio_data_url) -> None:
    """
    Realtime voice_clone via /v1/audio/speech, Chinese input.
    Deploy Setting: moss_tts_realtime.yaml
    Input Modal: text (Chinese) + reference audio (voice clone)
    Output Modal: audio (24 kHz, WAV)
    Input Setting: stream=False
    Datasets: single request

    NOTE: same ``min_audio_bytes`` rationale as test_text_to_audio_001.
    """
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("chinese"),
        "stream": False,
        "response_format": "wav",
        "ref_audio": ref_audio_data_url,
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }

    openai_client.send_audio_speech_request(request_config)
