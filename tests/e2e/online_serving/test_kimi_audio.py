# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real speech requests against one async-chunk server; pretrained weights required."""

import base64
import io
import json
import os

import numpy as np
import pytest
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

MODEL = os.environ.get("KIMI_AUDIO_MODEL_PATH", "moonshotai/Kimi-Audio-7B-Instruct")
SERVER_ARGS = ["--trust-remote-code"]
if glm_path := os.environ.get("KIMI_AUDIO_GLM_TOKENIZER_PATH"):
    SERVER_ARGS += ["--additional-config", json.dumps({"kimi_audio": {"glm_tokenizer_path": glm_path}})]

# Permit two active requests in both stages instead of only testing HTTP queuing.
# The shipped single-request deployment remains unchanged.
STAGE_CONFIG = modify_stage_config(
    get_deploy_config_path("kimi_audio_async_chunk.yaml"),
    updates={"stages": {0: {"max_num_seqs": 2}, 1: {"max_num_seqs": 2}}},
)
pytestmark = [
    pytest.mark.core_model,
    pytest.mark.tts,
    pytest.mark.parametrize(
        "omni_server",
        [
            pytest.param(
                OmniServerParams(
                    model=MODEL,
                    stage_config_path=STAGE_CONFIG,
                    server_args=SERVER_ARGS,
                    env_dict={"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
                ),
                id="kimi_audio_async_chunk",
            )
        ],
        indirect=True,
    ),
]

# Long enough to require more than one acoustic block during ordinary reading.
TEXT = (
    "The morning sun shines through the window. I open a book and read a story about a quiet village. "
    "Outside, children are playing in the garden, and a gentle breeze moves the leaves on the trees."
)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("request_num", [1, 2], ids=["single", "concurrent"])
def test_speech_requests_complete(omni_server, online_client, request_num):
    responses = online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": TEXT,
            "voice": "default",
            "response_format": "wav",
            "max_new_tokens": 512,
            "timeout": 300.0,
        },
        request_num=request_num,
    )
    assert len(responses) == request_num
    for response in responses:
        audio, sample_rate = sf.read(io.BytesIO(response.audio_bytes), dtype="float32", always_2d=True)
        assert sample_rate == 24000
        assert audio.shape[0] > 0 and audio.shape[1] == 1
        assert np.isfinite(audio).all() and np.any(audio != 0)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_speech_stream_delivers_audio_deltas(omni_server, online_client):
    chunks = []
    finished = False
    # Read semantic SSE events: TCP/HTTP byte chunks do not identify model chunks.
    with online_client.client.audio.speech.with_streaming_response.create(
        model=omni_server.model,
        input=TEXT,
        voice="default",
        response_format="pcm",
        extra_body={"stream": True, "stream_format": "sse", "max_new_tokens": 512},
        timeout=300.0,
    ) as response:
        assert "text/event-stream" in response.headers["content-type"]
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line.removeprefix("data:"))
            assert not finished, "Received an event after speech.audio.done"
            assert event["type"] in {"speech.audio.delta", "speech.audio.done"}, event
            if event["type"] == "speech.audio.done":
                finished = True
                continue
            assert event["response_format"] == "pcm"
            chunk = base64.b64decode(event["audio"], validate=True)
            assert len(chunk) > 0 and len(chunk) % 2 == 0
            chunks.append(chunk)
    assert finished
    assert len(chunks) >= 2, "Expected incremental audio before completion"
    # PCM has no sample-rate header; the WAV case checks the native 24 kHz rate.
    audio = np.frombuffer(b"".join(chunks), dtype="<i2")
    assert np.any(audio != 0)
