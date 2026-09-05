# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Breeze voice design through the shared speech API test client."""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "BreezeBlue/Breeze-TTS-2"

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.advanced_model,
    pytest.mark.tts,
    pytest.mark.parametrize(
        "omni_server",
        [OmniServerParams(model=MODEL, stage_config_path=get_deploy_config_path("breeze_tts.yaml"))],
        indirect=True,
    ),
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_chinese_voice_design(omni_server, online_client, run_level) -> None:
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "你好，欢迎使用语音合成系统。今天的天气很好。",
            "instructions": "年轻女性，使用标准普通话，语速适中。",
            "transcript_language": "zh",
            "response_format": "wav",
            "sample_rate": 24000,
            "min_audio_bytes": 24000,
            "seed": 42,
            "max_new_tokens": 8 if run_level == "core_model" else 250,
        }
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_english_streaming(omni_server, online_client, run_level) -> None:
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "Hello, this is a demonstration of natural speech synthesis.",
            "instructions": "A warm, clear female voice speaking English at a moderate pace.",
            "transcript_language": "en",
            "response_format": "pcm",
            "sample_rate": 24000,
            "min_audio_bytes": 24000,
            "stream": True,
            "stream_format": "audio",
            # The reference eager runtime scores 0.56 dB on this same prompt
            # (seed 42, CFG=1); the shared 1 dB floor rejects it as well.
            "min_hnr_db": 0.0,
            "seed": 42,
            "max_new_tokens": 8 if run_level == "core_model" else 250,
        }
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_two_concurrent_requests(omni_server, online_client, run_level) -> None:
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "Each request keeps its own audio generation state.",
            "instructions": "A clear English voice.",
            "transcript_language": "en",
            "response_format": "wav",
            "sample_rate": 24000,
            "min_audio_bytes": 24000,
            "seed": 123,
            "max_new_tokens": 8 if run_level == "core_model" else 250,
        },
        request_num=2,
    )
