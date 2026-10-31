# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native Soprano PCM/SSE deltas through the shared speech client."""

import os
from contextlib import nullcontext
from io import BytesIO
from urllib.parse import urlsplit

import numpy as np
import pytest
import soundfile as sf

from tests.helpers.client import OnlineOmniClient
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.tts]


@pytest.fixture(scope="module")
def speech_client(request):
    attached = os.environ.get("SOPRANO_STREAM_TEST_BASE_URL")
    context = (
        nullcontext()
        if attached
        else OmniServer(
            os.environ.get("SOPRANO_MODEL_PATH", "ekwek/Soprano-1.1-80M"),
            ["--deploy-config", get_deploy_config_path("soprano_streaming.yaml"), "--served-model-name", "soprano"],
            env_dict={"TRITON_F32_DEFAULT": "ieee", "VLLM_USE_V2_MODEL_RUNNER": "0"},
        )
    )
    with context as server:
        endpoint = urlsplit(attached) if attached else urlsplit(f"http://{server.host}:{server.port}")
        assert endpoint.hostname is not None
        yield OnlineOmniClient(
            host=endpoint.hostname, port=endpoint.port, run_level=request.config.getoption("run_level")
        )


def config(**changes):
    return {
        "model": "soprano",
        "input": "Hello, this is a simple speech synthesis test.",
        "voice": "default",
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
        "seed": 44,
        "sample_rate": 32000,
        "transcript_pcm_sample_rate": 32000,
        "min_audio_bytes": 4096,
        **changes,
    }


@pytest.mark.core_model
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_raw_pcm_sse_and_complete_concatenate_identically(speech_client):
    raw = speech_client.send_audio_speech_request(config())[0]
    assert len(raw.audio_chunk_sizes) > 1
    assert raw.audio_chunk_arrivals_s[0] < raw.e2e_latency
    sse = speech_client.send_audio_speech_request(config(stream_format="sse"))[0]
    assert sse.audio_bytes == raw.audio_bytes
    complete = speech_client.send_audio_speech_request(config(stream=False, stream_format=None, response_format="wav"))[
        0
    ]
    wave, rate = sf.read(BytesIO(complete.audio_bytes), dtype="int16")
    assert rate == 32000 and wave.ndim == 1
    assert np.array_equal(wave, np.frombuffer(raw.audio_bytes, dtype="<i2"))


@pytest.mark.core_model
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_short_stream_error_and_final_partial_block(speech_client):
    speech_client.send_audio_speech_http_request(
        {
            "json": {
                k: v
                for k, v in config(max_new_tokens=1).items()
                if k not in {"min_audio_bytes", "transcript_pcm_sample_rate"}
            }
        },
        err_code=500,
        err_message="Soprano",
    )
    result = speech_client.send_audio_speech_request(config(max_new_tokens=9))[0]
    assert len(result.audio_bytes) == 2 * 2048 * 8
