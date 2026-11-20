# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-weight LFM3-Omni serving: streamed and full-payload speech, and text-only replies."""

import base64
import os
from pathlib import Path

import pytest

from tests.helpers.assertions import assert_wav_sample_rate
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("LFM3_OMNI_MODEL_PATH", "LiquidAI/LFM3-Omni-2B")
# Mandarin speech: "不用帮我叫车了，我自己坐地铁回去。"
ASR_CLIP = Path(__file__).resolve().parents[2] / "assets" / "semantic_vad" / "complete.wav"
NO_MEDIA = ["--limit-mm-per-prompt", '{"image":0,"audio":0,"video":0}']
ASYNC_CHUNK_SERVER = OmniServerParams(
    model=MODEL,
    stage_config_path=get_deploy_config_path("lfm3_omni.yaml"),
    server_args=["--trust-remote-code", "--async-chunk", *NO_MEDIA],
)
# Without --async-chunk, LiquidCodec receives one payload after the temporal stage finishes.
FULL_PAYLOAD_SERVER = OmniServerParams(
    model=MODEL,
    stage_config_path=get_deploy_config_path("lfm3_omni.yaml"),
    server_args=["--trust-remote-code", *NO_MEDIA],
)
# One audio input keeps the audio encoder in startup profiling and in the request path.
TEXT_SERVER = OmniServerParams(
    model=MODEL,
    stage_config_path=get_deploy_config_path("lfm3_omni_text.yaml"),
    server_args=["--trust-remote-code", "--limit-mm-per-prompt", '{"image":0,"audio":1,"video":0}'],
)


def speech_request(model, stream):
    request = {
        "model": model,
        "input": "The quick brown fox jumps over the lazy dog.",
        "seed": 42,
        "extra_params": {"audio_temperature": 0.0},
        "max_new_tokens": 96,
        "response_format": "pcm" if stream else "wav",
        "stream": stream,
        "expected_sample_rate": 24000,
        "min_audio_bytes": 24000,
        "timeout": 240,
        "min_hnr_db": 0.0,
    }
    if stream:
        request["stream_format"] = "audio"
    return request


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [ASYNC_CHUNK_SERVER], indirect=True)
@pytest.mark.parametrize("stream", [False, True])
def test_speech_response_and_concurrent_streaming(omni_server, openai_client, stream):
    responses = openai_client.send_audio_speech_request(
        speech_request(omni_server.model, stream), request_num=2 if stream else 1
    )
    if not stream:
        assert_wav_sample_rate(responses[0].audio_bytes, 24000)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [FULL_PAYLOAD_SERVER], indirect=True)
def test_full_payload_speech_response(omni_server, openai_client):
    responses = openai_client.send_audio_speech_request(speech_request(omni_server.model, False), request_num=2)
    for response in responses:
        assert_wav_sample_rate(response.audio_bytes, 24000)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [TEXT_SERVER], indirect=True)
def test_text_only_profile_answers_in_text(omni_server, openai_client):
    request = {
        "model": omni_server.model,
        "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        "modalities": ["text"],
        "key_words": {"text": ["Paris"]},
    }
    openai_client.send_omni_request(request)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [TEXT_SERVER], indirect=True)
def test_text_only_profile_transcribes_audio_input(omni_server, openai_client):
    clip = base64.b64encode(ASR_CLIP.read_bytes()).decode()
    request = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(
            system_prompt={"role": "system", "content": "Perform ASR in Chinese."},
            audio_data_url=f"data:audio/wav;base64,{clip}",
        ),
        "modalities": ["text"],
        "key_words": {"text": ["坐地铁回去"]},
    }
    openai_client.send_omni_request(request)
