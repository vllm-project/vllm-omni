# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

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
