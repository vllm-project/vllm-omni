# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E online tests for Audex (Nemotron-Labs-Audex-2B) TTS.

Covers /v1/audio/speech non-streaming and audio-byte streaming, plus the
Audex request policies: single built-in voice (missing/empty/"default" only),
no reference audio, and cfg_scale reserved at 1.0.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.slow, pytest.mark.tts]

MODEL = "nvidia/Nemotron-Labs-Audex-2B"
PROMPT = "The weather is so good, and I want to enjoy the beautiful morning in the park."

audex_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("nemotron_labs_audex.yaml"),
            server_args=["--trust-remote-code"],
        ),
        id="audex",
    )
]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", audex_server_params, indirect=True)
def test_audex_tts_nonstream(omni_server, openai_client) -> None:
    """Plain English TTS, non-streaming WAV response."""
    request_config = {
        "model": omni_server.model,
        "input": PROMPT,
        "stream": False,
        "response_format": "wav",
    }
    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", audex_server_params, indirect=True)
def test_audex_tts_stream_audio(omni_server, openai_client) -> None:
    """Plain English TTS, raw audio-byte streaming."""
    request_config = {
        "model": omni_server.model,
        "input": PROMPT,
        "stream": True,
        "stream_format": "audio",
        "response_format": "wav",
    }
    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", audex_server_params, indirect=True)
def test_audex_tts_request_policies(omni_server, openai_client) -> None:
    """Unsupported voice / CFG / empty input must be rejected with 400."""
    import requests

    url = f"{openai_client.base_url.rstrip('/')}/v1/audio/speech"

    ok = requests.post(url, json={"model": omni_server.model, "input": PROMPT, "voice": "default"})
    assert ok.status_code == 200, ok.text

    for bad_payload, needle in (
        ({"input": PROMPT, "voice": "alloy"}, "voice"),
        ({"input": "   "}, "non-empty"),
        ({"input": PROMPT, "extra_params": {"cfg_scale": 1.5}}, "cfg_scale"),
        ({"input": PROMPT, "ref_audio": "https://example.com/ref.wav"}, "reference audio"),
    ):
        resp = requests.post(url, json={"model": omni_server.model, **bad_payload})
        assert resp.status_code == 400, f"{bad_payload} -> {resp.status_code}: {resp.text[:200]}"
        assert needle in resp.text, f"{bad_payload} error missing {needle!r}: {resp.text[:200]}"
