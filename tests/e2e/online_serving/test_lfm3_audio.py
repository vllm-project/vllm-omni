# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-weight LFM3-Audio serving: speech output with concurrent streams, and ASR from audio input."""

import base64
import os
from pathlib import Path

import pytest

from tests.helpers.assertions import assert_wav_sample_rate
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("LFM3_AUDIO_MODEL_PATH", "LiquidAI/LFM3-Audio-2B")
# Mandarin speech: "不用帮我叫车了，我自己坐地铁回去。" (transcribed verbatim by the reference).
ASR_CLIP = Path(__file__).resolve().parents[2] / "assets" / "semantic_vad" / "complete.wav"
# One audio input keeps the speech encoder in startup profiling.
SERVER = OmniServerParams(
    model=MODEL,
    stage_config_path=get_deploy_config_path("lfm3_audio.yaml"),
    server_args=["--trust-remote-code", "--async-chunk", "--limit-mm-per-prompt", '{"audio":1}'],
)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [SERVER], indirect=True)
@pytest.mark.parametrize("stream", [False, True])
def test_speech_response_and_concurrent_streaming(omni_server, openai_client, stream):
    request = {
        "model": omni_server.model,
        "input": "The quick brown fox jumps over the lazy dog.",
        "seed": 42,
        "extra_params": {"audio_temperature": 0.0},
        "max_new_tokens": 96,
        "response_format": "pcm" if stream else "wav",
        "stream": stream,
        "expected_sample_rate": 24000,
        "min_audio_bytes": 24000,
        "timeout": 240,
        # Transformers' own outputs for this checkpoint score 0.4-3.0 dB on
        # this HNR proxy; zero still rejects silence and noise.
        "min_hnr_db": 0.0,
    }
    if stream:
        request["stream_format"] = "audio"
    responses = openai_client.send_audio_speech_request(request, request_num=2 if stream else 1)
    if not stream:
        # LiquidCodec's native rate; the request asks for none, so nothing resamples it.
        assert_wav_sample_rate(responses[0].audio_bytes, 24000)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [SERVER], indirect=True)
def test_asr_text_from_audio_input(omni_server, openai_client):
    clip = base64.b64encode(ASR_CLIP.read_bytes()).decode()
    request = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(
            system_prompt={"role": "system", "content": "Perform ASR in Chinese."},
            audio_data_url=f"data:audio/wav;base64,{clip}",
        ),
        "modalities": ["text"],
        # Keep the reply in text, like the reference's suppress_mixed/suppress_audio_out.
        "extra_body": {"vllm_xargs": {"generation_mode": "text"}},
        "key_words": {"text": ["坐地铁回去"]},
    }
    openai_client.send_omni_request(request)
