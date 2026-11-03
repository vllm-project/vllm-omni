# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("QWEN3_TTS_NPU_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
REF_AUDIO_URL = get_asset_path("qwen3_tts/clone_2.wav", as_data_url=True)
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("qwen3_tts_mrv2.yaml"),
            server_args=["--trust-remote-code"],
        ),
        False,
        id="mrv2-full-payload",
    ),
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("qwen3_tts_mrv2.yaml"),
            server_args=["--trust-remote-code"],
        ),
        True,
        id="mrv2-streaming-chunks",
    ),
]


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server,stream", SERVER_PARAMS, indirect=["omni_server"])
def test_qwen3_tts_npu_mrv2_serves_audio(omni_server, online_client, stream):
    request_config = {
        "model": omni_server.model,
        "input": "The weather is nice today, perfect for a walk in the park.",
        "stream": stream,
        "timeout": 900,
        "response_format": "wav",
        "task_type": "Base",
        "voice": "clone",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }
    if stream:
        request_config["stream_format"] = "audio"
    online_client.send_audio_speech_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"npu": "A3"}, num_cards=1)
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                stage_config_path=get_deploy_config_path("qwen3_tts_mrv2.yaml"),
                server_args=["--trust-remote-code"],
            ),
            id="mrv2-abort-and-long-input",
        )
    ],
    indirect=True,
)
def test_qwen3_tts_npu_mrv2_stream_abort_and_long_input(omni_server, openai_client):
    """Disconnect a long streaming request, then verify long synthesis still works."""
    import requests

    url = f"{openai_client.base_url.rstrip(chr(47))}/v1/audio/speech"
    long_text = " ".join(
        [
            "The weather is nice today, and the warm sunlight makes the garden feel peaceful.",
            "We can take a walk beside the river, listen to the birds, and talk about the books we enjoy.",
        ]
    )
    payload = {
        "model": omni_server.model,
        "input": long_text,
        "stream": True,
        "stream_format": "audio",
        "response_format": "wav",
        "task_type": "Base",
        "voice": "clone",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    }

    with requests.post(url, json=payload, stream=True, timeout=900) as response:
        assert response.status_code == 200, response.text[:200]
        first_chunk = next((chunk for chunk in response.iter_content(chunk_size=None) if chunk), b"")
        assert first_chunk, "stream returned no audio before disconnect"

    full_response = requests.post(
        url,
        json={**{k: v for k, v in payload.items() if k != "stream_format"}, "stream": False},
        timeout=900,
    )
    assert full_response.status_code == 200, full_response.text[:200]
    assert full_response.content[:4] == b"RIFF" and len(full_response.content) > 44
