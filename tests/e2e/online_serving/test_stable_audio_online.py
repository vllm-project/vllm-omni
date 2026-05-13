# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import io

import pytest
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler, dummy_messages_from_mix_data

MODEL = "linyueqian/stable_audio_random"
SAMPLE_RATE = 44100


TEST_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=["--disable-log-stats"],
        ),
        id="single_card_001",
    )
]


def _collect_audio_url_items(openai_client: OpenAIClientHandler, request_config: dict):
    chat_completion = openai_client.client.chat.completions.create(
        model=request_config["model"],
        messages=request_config["messages"],
        extra_body=request_config.get("extra_body"),
    )

    audio_items = []
    for choice in chat_completion.choices:
        content = getattr(choice.message, "content", None)
        assert content is not None, "API response content is None"
        if isinstance(content, str):
            pytest.fail(f"Expected multimodal content list, got plain string: {content[:200]}")
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "audio_url" and item.get("audio_url") is not None:
                    audio_items.append(item)
            elif hasattr(item, "audio_url") and item.audio_url is not None:
                audio_items.append(item)
    return audio_items


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "xpu": "B60"}, num_cards=1)
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_stable_audio_online(omni_server: OmniServer, openai_client: OpenAIClientHandler):
    messages = dummy_messages_from_mix_data(
        content_text="The sound of a dog barking",
    )
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "num_inference_steps": 2,
            "guidance_scale": 7.0,
            "audio_start_in_s": 0.0,
            "audio_end_in_s": 2.0,
            "seed": 42,
        },
    }

    audio_items = _collect_audio_url_items(openai_client, request_config)

    assert len(audio_items) > 0, "Expected at least one audio_url item in response"
    for item in audio_items:
        url = item["audio_url"]["url"] if isinstance(item, dict) else item.audio_url.url
        assert url.startswith("data:audio/wav;base64,"), f"Unexpected audio URL prefix: {url[:64]}"
        wav_bytes = base64.b64decode(url.split(",", 1)[1])
        audio, sample_rate = sf.read(io.BytesIO(wav_bytes))
        assert sample_rate == SAMPLE_RATE, f"Expected sample rate {SAMPLE_RATE}, got {sample_rate}"
        assert getattr(audio, "size", 0) > 0, "Audio data is empty"
        # Stable Audio produces stereo audio (2 channels)
        if audio.ndim > 1:
            assert audio.shape[-1] == 2, f"Expected stereo audio (2 channels), got {audio.shape[-1]}"
