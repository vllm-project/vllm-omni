# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sesame CSM-1B through the shared speech client: text, streaming, concurrency and a reference turn."""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "sesame/csm-1b"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.parametrize(
        "omni_server",
        [OmniServerParams(model=MODEL, stage_config_path=get_deploy_config_path("csm.yaml"))],
        indirect=True,
    ),
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_text_to_speech(omni_server, online_client) -> None:
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "The quick brown fox jumps over the lazy dog near the river.",
            "transcript_language": "en",
            "transcript_escalation_model": "large-v3",
            "response_format": "wav",
            "min_audio_bytes": 24000,
            "seed": 42,
            "max_new_tokens": 250,
        }
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_streaming_pcm(omni_server, online_client) -> None:
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "We walked along the beach and watched the sun go down.",
            "transcript_language": "en",
            "transcript_pcm_sample_rate": 24000,
            "transcript_escalation_model": "large-v3",
            "response_format": "pcm",
            "min_audio_bytes": 24000,
            "stream": True,
            "stream_format": "audio",
            "seed": 42,
            "max_new_tokens": 250,
        }
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_two_concurrent_requests(omni_server, online_client) -> None:
    # No seed: a seeded request runs alone, and these two must share the engine.
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "Each request keeps its own audio generation state.",
            "transcript_language": "en",
            "transcript_escalation_model": "large-v3",
            "response_format": "wav",
            "min_audio_bytes": 24000,
            "max_new_tokens": 250,
        },
        request_num=2,
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_second_speaker_id(omni_server, online_client) -> None:
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "A different speaker label still produces clear speech.",
            "extra_params": {"speaker_id": 1},
            "transcript_language": "en",
            "transcript_escalation_model": "large-v3",
            "response_format": "wav",
            "min_audio_bytes": 24000,
            "seed": 42,
            "max_new_tokens": 250,
        }
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_reference_turn_is_continued(omni_server, online_client) -> None:
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "Please read this sentence clearly and naturally.",
            "ref_audio": get_asset_path("qwen3_tts/clone_2.wav", as_data_url=True),
            "ref_text": "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
            "transcript_language": "en",
            "transcript_escalation_model": "large-v3",
            "response_format": "wav",
            "min_audio_bytes": 24000,
            "seed": 42,
            "max_new_tokens": 250,
        }
    )
