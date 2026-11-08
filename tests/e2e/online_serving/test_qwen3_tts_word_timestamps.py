# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
import json

import pytest
from websockets.sync.client import connect

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

pytestmark = [pytest.mark.slow, pytest.mark.tts]

_DEPLOY = modify_stage_config(
    get_deploy_config_path("qwen3_tts.yaml"),
    updates={
        "stages": {
            0: {
                "max_num_seqs": 4,
                "max_num_batched_tokens": 2048,
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.45,
                "compilation_config.cudagraph_capture_sizes": [1, 2, 4],
                "default_sampling_params.max_tokens": 256,
            },
            1: {
                "max_num_seqs": 4,
                "max_num_batched_tokens": 4096,
                "max_model_len": 8192,
                "gpu_memory_utilization": 0.15,
            },
        }
    },
)
_PARAMS = [
    OmniServerParams(
        model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        stage_config_path=_DEPLOY,
        server_args=[
            "--trust-remote-code",
            "--forced-aligner",
            "Qwen/Qwen3-ForcedAligner-0.6B",
            "--kv-cache-memory-bytes",
            str(512 * 1024**2),
            "--max-num-seqs",
            "4",
            "--max-num-batched-tokens",
            "2048",
        ],
    )
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_async_chunk_word_timestamps(omni_server, online_client):
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "The weather is nice today, perfect for a walk in the park.",
            "voice": "vivian",
            "language": "English",
            "word_timestamps": True,
            "response_format": "wav",
            "min_audio_bytes": 40_000,
            "timeout": 120,
        },
        request_num=2,
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_async_chunk_speech_without_timestamps(omni_server, online_client):
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "The weather is nice today, perfect for a walk in the park.",
            "voice": "vivian",
            "language": "English",
            "stream": True,
            "stream_format": "audio",
            "response_format": "wav",
            "min_audio_bytes": 40_000,
            "timeout": 120,
        }
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_async_chunk_streaming_word_timestamps(omni_server):
    text = "The weather is nice today, perfect for a walk in the park."
    chunks = []
    timestamps = None
    with connect(f"ws://{omni_server.host}:{omni_server.port}/v1/audio/speech/stream") as ws:
        ws.send(
            json.dumps(
                {
                    "type": "session.config",
                    "voice": "vivian",
                    "language": "English",
                    "stream_audio": True,
                    "response_format": "pcm",
                    "word_timestamps": True,
                }
            )
        )
        ws.send(json.dumps({"type": "input.text", "text": text}))
        ws.send(json.dumps({"type": "input.done"}))
        while True:
            message = json.loads(ws.recv(timeout=120))
            assert message["type"] != "error", message
            if message["type"] == "audio.chunk":
                if message["audio_b64"]:
                    chunks.append(base64.b64decode(message["audio_b64"]))
                if message["timestamps"] is not None:
                    timestamps = message["timestamps"]
                    sample_rate = message["sample_rate"]
            elif message["type"] == "audio.done":
                assert not message["error"], message
                assert message["total_bytes"] == sum(map(len, chunks))
            elif message["type"] == "session.done":
                break

    assert len(chunks) > 1
    assert timestamps
    assert len(timestamps) == len(text.split())
    duration_ms = sum(map(len, chunks)) / (2 * sample_rate) * 1000
    final_chunk_ms = len(chunks[-1]) / (2 * sample_rate) * 1000
    assert timestamps[-1]["end_ms"] > max(duration_ms / 2, final_chunk_ms)
    previous_end = 0
    for timestamp in timestamps:
        assert previous_end <= timestamp["start_ms"] <= timestamp["end_ms"] <= duration_ms + 1
        previous_end = timestamp["end_ms"]
