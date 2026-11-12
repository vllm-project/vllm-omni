# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real native speech through the standard module-scoped runtime fixtures.

L2 has one complete-response and one streaming protocol smoke. L3 additionally
checks length flush and cancellation/slot reuse. Hardware variants are collected
separately; selecting RTX4090Laptop never executes the L4-labelled variant.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.helpers.assertions import (
    assert_audio_speech_equal,
    assert_audio_speech_samples,
    assert_audio_stream_progress,
)
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.tts]
MODEL = os.environ.get("SOPRANO_MODEL_PATH", "ekwek/Soprano-1.1-80M")


def server_params(deploy):
    return [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                stage_config_path=get_deploy_config_path(deploy),
                server_args=["--disable-log-stats"],
                env_dict={"TRITON_F32_DEFAULT": "ieee", "VLLM_USE_V2_MODEL_RUNNER": "0"},
            ),
            id=sku,
            marks=hardware_marks(res={"cuda": sku}, num_cards=1),
        )
        for sku in ["L4", "RTX4090Laptop"]
    ]


_COMPLETE = server_params("soprano.yaml")
_STREAM = server_params("soprano_streaming.yaml")


def config(server, **changes):
    return {
        "model": server.model,
        "input": "Hello, this is a simple speech synthesis test.",
        "response_format": "wav",
        "seed": 44,
        "sample_rate": 32000,
        "expected_sample_rate": 32000,
        "min_audio_bytes": 4096,
        "extra_params": {"temperature": 0.3, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.2},
        **changes,
    }


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.parametrize("omni_server", _COMPLETE, indirect=True)
def test_complete_wav_and_fixed_default_voice(omni_server, online_client):
    omitted = online_client.send_audio_speech_request(config(omni_server))[0]
    explicit = online_client.send_audio_speech_request(config(omni_server, voice="default", language="English"))[0]
    assert_audio_speech_equal(omitted, explicit, sample_rate=32000)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.parametrize("omni_server", _STREAM, indirect=True)
def test_raw_pcm_sse_and_complete_concatenate_identically(omni_server, online_client):
    # This fixed text/seed: official FP32 HNR=0.55369 dB (BF16=0.67694),
    # native=0.55378, all at 32 kHz. Round the independent reference down
    # to 0.1 dB; keep the shared 1 dB default for all other scenarios.
    # ASR and exact raw/SSE/WAV equality remain mandatory.
    raw = online_client.send_audio_speech_request(
        config(
            omni_server,
            stream=True,
            stream_format="audio",
            response_format="pcm",
            transcript_pcm_sample_rate=32000,
            min_hnr_db=0.5,
        )
    )[0]
    assert_audio_stream_progress(raw)
    sse = online_client.send_audio_speech_request(
        config(
            omni_server,
            stream=True,
            stream_format="sse",
            response_format="pcm",
            transcript_pcm_sample_rate=32000,
            min_hnr_db=0.5,
        )
    )[0]
    complete = online_client.send_audio_speech_request(config(omni_server))[0]
    assert_audio_speech_equal(sse, raw, sample_rate=32000)
    assert_audio_speech_equal(complete, raw, sample_rate=32000)


@pytest.mark.advanced_model
@pytest.mark.parametrize("omni_server", _STREAM, indirect=True)
def test_length_stop_flushes_partial_block(omni_server, online_client):
    # Forced short output is a shape/tail contract, not a full-text ASR test.
    result = online_client.send_audio_speech_request(
        config(
            omni_server,
            stream=True,
            stream_format="audio",
            response_format="wav",
            max_new_tokens=9,
            transcript_expected_text=None,
        )
    )[0]
    assert_audio_speech_samples(result, samples=2048 * 8, sample_rate=32000)


@pytest.mark.advanced_model
@pytest.mark.parametrize("omni_server", _STREAM, indirect=True)
def test_cancel_peer_completion_and_slot_reuse(omni_server, online_client):
    first = config(
        omni_server,
        stream=True,
        stream_format="audio",
        response_format="pcm",
        input="The patient traveler walks beside the river and describes the bright colors of the trees to the children.",
    )
    second = config(
        omni_server,
        stream=True,
        stream_format="audio",
        response_format="pcm",
        input="A silver bell rings in the quiet room.",
    )
    first_baseline = online_client.send_audio_speech_request(first)[0]
    second_baseline = online_client.send_audio_speech_request(second)[0]
    payload = {k: v for k, v in first.items() if k not in {"min_audio_bytes", "expected_sample_rate"}}
    with ThreadPoolExecutor(max_workers=2) as executor:
        cancelled = executor.submit(online_client.send_audio_speech_cancel_request, {"json": payload})
        peer = executor.submit(online_client.send_audio_speech_request, second)
        prefix, completed = cancelled.result(), peer.result()[0]
    assert_audio_speech_equal(prefix, first_baseline, sample_rate=32000, prefix=True)
    assert_audio_speech_equal(completed, second_baseline, sample_rate=32000)
    # The same server's only sequence slot serves a new request after cancellation.
    recovered = online_client.send_audio_speech_request(first)[0]
    assert_audio_speech_equal(recovered, first_baseline, sample_rate=32000)
