# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CUDA MRV2 integration coverage for Voxtral TTS."""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("VLLM_TEST_VOXTRAL_MODEL", "mistralai/Voxtral-4B-TTS-2603")
STAGE_CONFIG = get_deploy_config_path("voxtral_tts_mrv2.yaml")
SERVER_PARAMS = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=STAGE_CONFIG,
        server_args=["--trust-remote-code", "--disable-log-stats"],
    )
]

pytestmark = [
    pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True),
    pytest.mark.slow,
    pytest.mark.tts,
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_voxtral_tts_mrv2_long_stream_completes(omni_server, online_client) -> None:
    text = " ".join(["Voxtral should read every sentence clearly and naturally."] * 12)
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": text,
            "voice": "casual_female",
            "language": "English",
            "stream": True,
            "stream_format": "audio",
            "response_format": "pcm",
            "timeout": 300.0,
            "min_audio_bytes": 10000,
        }
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_voxtral_tts_mrv2_abort_then_followup_succeeds(omni_server, openai_client) -> None:
    import requests

    url = f"{openai_client.base_url.rstrip('/')}/v1/audio/speech"
    long_text = " ".join(["Please keep speaking until this streamed request is cancelled."] * 20)
    with requests.post(
        url,
        json={
            "model": omni_server.model,
            "input": long_text,
            "voice": "casual_female",
            "language": "English",
            "stream": True,
            "stream_format": "audio",
            "response_format": "pcm",
        },
        stream=True,
        timeout=300,
    ) as response:
        assert response.status_code == 200, response.text
        next(response.iter_content(chunk_size=4096))

    followup = requests.post(
        url,
        json={
            "model": omni_server.model,
            "input": "The server should still answer after a client abort.",
            "voice": "casual_female",
            "language": "English",
            "response_format": "wav",
        },
        timeout=300,
    )
    assert followup.status_code == 200, followup.text[:200]
    assert followup.content[:4] == b"RIFF" and len(followup.content) > 44


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_voxtral_tts_mrv2_concurrent_requests_keep_audio(omni_server, openai_client) -> None:
    import concurrent.futures

    import requests

    url = f"{openai_client.base_url.rstrip('/')}/v1/audio/speech"
    payloads = [
        {
            "model": omni_server.model,
            "input": f"Concurrent request number {index} should keep its own voice frame.",
            "voice": "casual_female",
            "language": "English",
            "response_format": "wav",
        }
        for index in range(4)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda body: requests.post(url, json=body, timeout=300), payloads))

    for response in responses:
        assert response.status_code == 200, response.text[:200]
        assert response.content[:4] == b"RIFF" and len(response.content) > 44
