# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E online tests for MOSS-TTS Local 1.5 on Ascend NPU with Stage0 on MRV2.

Uses the opt-in ``moss_tts_local_npu_mrv2.yaml`` profile (Talker on MRV2, codec
on V1, eager) through /v1/audio/speech. The reference audio is inlined as a base64
data URL because CI hosts cannot fetch external URLs.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.advanced_model, pytest.mark.tts]

MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 600.0
# Reject a response that contains only a header or a very short audio payload.
_MIN_AUDIO_BYTES = 10_000
REF_AUDIO_URL = get_asset_path("qwen3_tts/clone_2.wav", as_data_url=True)

tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("moss_tts_local_npu_mrv2.yaml"),
            server_args=["--disable-log-stats"],
            stage_init_timeout=1200,
        ),
        id="moss_tts_local_npu_mrv2",
    )
]


def _request_config(omni_server, **overrides) -> dict:
    config = {
        "model": omni_server.model,
        "input": "The weather is nice today, perfect for a walk in the park.",
        "ref_audio": REF_AUDIO_URL,
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
        # whisper-small occasionally mishears the cloned voice; rescue with a stronger ASR.
        "transcript_escalation_model": "large-v3",
    }
    config.update(overrides)
    return config


@pytest.mark.core_model
@hardware_test(res={"npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client) -> None:
    """Non-streaming WAV with Stage0 on MRV2.

    Deploy Setting: moss_tts_local_npu_mrv2.yaml
    Input Modal: text + reference audio
    Output Modal: audio (48 kHz, WAV)
    Input Setting: stream=False
    """
    openai_client.send_audio_speech_request(_request_config(omni_server))


@pytest.mark.core_model
@hardware_test(res={"npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_002_streaming(omni_server, openai_client) -> None:
    """Streaming PCM over the async_chunk path.

    Input Setting: stream=True, response_format=pcm
    """
    openai_client.send_audio_speech_request(
        _request_config(
            omni_server,
            stream=True,
            stream_format="audio",
            response_format="pcm",
            expected_sample_rate=48000,
            transcript_pcm_sample_rate=48000,
            # Local 1.5 streams interleaved stereo PCM.
            transcript_pcm_channels=2,
            # Local 1.5 output is intrinsically around -4 dB on this metric (the
            # same on V1); keep the catastrophic-failure check only.
            min_hnr_db=-10.0,
        )
    )


@hardware_test(res={"npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_003_concurrent(omni_server, openai_client) -> None:
    """Concurrent requests keep their own slot state."""
    openai_client.send_audio_speech_request(_request_config(omni_server), request_num=4)
