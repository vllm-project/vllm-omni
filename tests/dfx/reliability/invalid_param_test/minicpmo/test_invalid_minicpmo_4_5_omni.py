# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Invalid audio format rejection on MiniCPM-o 4.5: ``POST /v1/chat/completions``."""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OpenAIClientHandler, dummy_messages_from_mix_data

pytestmark = [pytest.mark.slow, pytest.mark.omni, pytest.mark.full_model]


def _system_prompt() -> dict[str, object]:
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are MiniCPM-o 4.5, a virtual human capable of perceiving "
                    "auditory and visual inputs, as well as generating text and speech."
                ),
            }
        ],
    }


def _prompt() -> str:
    return "What is the capital of China? Answer in 20 words."


@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=1)
def test_invalid_audio_format_rejected(omni_server, openai_client: OpenAIClientHandler) -> None:
    """
    Test that invalid audio format is properly rejected.
    Deploy Setting: default 2GPU
    Input Modal: text + invalid audio format
    Output Modal: error
    """
    fake_base64 = "AAAA"  # not a real audio file

    responses = openai_client.send_chat_completions_http_request(
        {
            "json": {
                "model": omni_server.model,
                "messages": dummy_messages_from_mix_data(
                    system_prompt=_system_prompt(),
                    audio_data_url=f"data:audio/invalid;base64,{fake_base64}",
                    content_text=_prompt(),
                ),
                "stream": True,
            },
        },
        err_code=400,
        err_message="Invalid or unsupported audio file",
    )
    assert not responses[0].success
