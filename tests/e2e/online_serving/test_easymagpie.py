# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in serving checks for a locally converted EasyMagpie checkpoint.

Set EASYMAGPIE_MODEL, EASYMAGPIE_VOICE and EASYMAGPIE_CONTEXT_TEXT to the
checkpoint's assets. EASYMAGPIE_DEPLOY_CONFIG may select a local deployment
override (for example, to disable optional Perth watermarking).
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("EASYMAGPIE_MODEL", "")
VOICE = os.environ.get("EASYMAGPIE_VOICE", "eng")
CONTEXT = os.environ.get("EASYMAGPIE_CONTEXT_TEXT", "[EN]")
TEXT = "The weather is nice today, perfect for a walk in the park."

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.advanced_model,
    pytest.mark.tts,
    pytest.mark.skipif(not MODEL, reason="EASYMAGPIE_MODEL must point to a converted checkpoint"),
]

SERVER = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=os.environ.get("EASYMAGPIE_DEPLOY_CONFIG")
            or get_deploy_config_path("easymagpie_a4500.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
            env_dict={"VLLM_USE_FLASHINFER_SAMPLER": "0", "OMP_NUM_THREADS": "1"},
        ),
        id="easymagpie",
    )
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER, indirect=True)
def test_complete_wav(omni_server, online_client):
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": TEXT,
            "voice": VOICE,
            "extra_params": {"context_text": CONTEXT},
            "response_format": "wav",
            "stream": False,
            "max_new_tokens": 256,
            "min_audio_bytes": 22050,
            "timeout": 300,
        }
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER, indirect=True)
def test_concurrent_streams(omni_server, online_client):
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": TEXT,
            "voice": VOICE,
            "extra_params": {"context_text": CONTEXT},
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
            "transcript_pcm_sample_rate": 22050,
            "max_new_tokens": 256,
            "min_audio_bytes": 22050,
            "timeout": 300,
        },
        request_num=4,
    )
