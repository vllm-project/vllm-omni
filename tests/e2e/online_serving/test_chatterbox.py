# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox speech serving through the shared TTS test client."""

import json

import pytest
from websockets.sync.client import connect

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

TEXT = "The weather is nice today, perfect for a walk in the park."
SERVERS = [
    pytest.param(
        OmniServerParams(
            model=model,
            stage_config_path=modify_stage_config(get_deploy_config_path(deploy), {"model_runner": runner}),
        ),
        id=f"{variant}-{runner}",
    )
    for variant, model, deploy in (
        ("turbo", "ResembleAI/chatterbox-turbo", "chatterbox_turbo.yaml"),
        ("original", "ResembleAI/chatterbox", "chatterbox.yaml"),
    )
    for runner in ("v2", "v1")
]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVERS, indirect=True)
def test_text_to_audio_001(omni_server, online_client):
    """The built-in voice works without reference encoders or a transcript."""
    online_client.send_audio_speech_request(
        {"model": omni_server.model, "input": TEXT, "voice": "default", "response_format": "wav", "timeout": 300.0}
    )


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVERS, indirect=True)
@pytest.mark.parametrize("response_format", ["wav", "pcm"])
def test_streaming_concurrent_audio(omni_server, online_client, response_format):
    """Four overlapping streams finish independently through the public API."""
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": TEXT,
            "stream": True,
            "stream_format": "audio",
            "response_format": response_format,
            # With this metric on this sentence, upstream Turbo measures 0.67
            # to 2.32 dB (30 seeds, mean 1.66) and upstream Original -1.20 to
            # -0.25 dB (seeds 0/42/123); all four streams must clear the floor,
            # so it sits below each range. PCM content is checked separately.
            "min_hnr_db": -3.0 if omni_server.model == "ResembleAI/chatterbox" else -1.0,
            "transcript_pcm_sample_rate": 24000,
            "timeout": 300.0,
        },
        request_num=4,
    )


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVERS, indirect=True)
def test_reference_voice_without_transcript(omni_server, online_client):
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": TEXT,
            "ref_audio": get_asset_path("qwen3_tts/clone_2.wav", as_data_url=True),
            "response_format": "wav",
            "timeout": 300.0,
        },
        request_num=4,
    )


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVERS, indirect=True)
def test_sentence_streaming_websocket(omni_server):
    """The shared WebSocket frontend splits the text into sentences and streams PCM for each."""
    sentences: dict[int, int] = {}
    with connect(f"ws://{omni_server.host}:{omni_server.port}/v1/audio/speech/stream") as ws:
        ws.send(
            json.dumps(
                {
                    "type": "session.config",
                    "voice": "default",
                    "response_format": "pcm",
                    "stream_audio": True,
                    "split_granularity": "sentence",
                }
            )
        )
        ws.send(json.dumps({"type": "input.text", "text": f"{TEXT} Please leave your name after the tone."}))
        ws.send(json.dumps({"type": "input.done"}))
        current = -1
        while True:
            frame = ws.recv(timeout=300)
            if isinstance(frame, bytes):
                sentences[current] += len(frame)
                continue
            message = json.loads(frame)
            assert message["type"] != "error", message
            if message["type"] == "audio.start":
                assert message["sample_rate"] == 24000
                current = message["sentence_index"]
                sentences[current] = 0
            elif message["type"] == "session.done":
                assert message["total_sentences"] == 2
                break

    # Mono 16-bit PCM: more than a second of audio per sentence.
    assert sorted(sentences) == [0, 1]
    assert all(size > 2 * 24000 for size in sentences.values())
