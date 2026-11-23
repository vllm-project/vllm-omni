# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Reasoning output can satisfy an explicitly opted-in text smoke check."""

from contextlib import nullcontext

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from tests.helpers.client import OnlineOmniClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("run_level", ["advanced_model", "full_model"])
@pytest.mark.parametrize(
    ("content", "reasoning", "allow_reasoning", "passes"),
    [
        (None, "The circles are moving.", True, True),
        ("", "The circles are moving.", True, True),
        (" \n", "The circles are moving.", True, True),
        ("Moving circles.", None, True, True),
        ("Moving circles.", "The circles are moving.", True, True),
        ("Moving circles.", None, False, True),
        (None, "The circles are moving.", False, False),
        (None, None, True, False),
        ("", "", True, False),
        (" \n", "\t", True, False),
    ],
)
def test_text_request_accepts_reasoning_only_when_enabled(
    monkeypatch, run_level, content, reasoning, allow_reasoning, passes
):
    completion = ChatCompletion.model_validate(
        {
            "id": "chatcmpl-reasoning",
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": content, "reasoning": reasoning},
                }
            ],
        }
    )
    client = OnlineOmniClient(port=8000, run_level=run_level, log_stats=False)
    monkeypatch.setattr(client.client.chat.completions, "create", lambda **kwargs: completion)
    request_config = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Describe the video."}],
        "modalities": ["text"],
        "allow_reasoning": allow_reasoning,
    }

    with client.client:
        with nullcontext() if passes else pytest.raises(AssertionError, match="No text output is generated"):
            responses = client.send_omni_request(request_config)
            assert responses[0].text_content == content
            assert responses[0].reasoning_content == reasoning


@pytest.mark.parametrize("content", ["", "Moving circles."])
def test_streaming_text_request_collects_reasoning(monkeypatch, content):
    chunks = [
        ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-reasoning",
                "created": 0,
                "model": "test-model",
                "object": "chat.completion.chunk",
                "modality": "text",
                "choices": [{"index": 0, "delta": delta}],
            }
        )
        for delta in ({"reasoning": "The circles "}, {"reasoning": "are moving."}, {"content": content})
    ]
    client = OnlineOmniClient(port=8000, run_level="advanced_model", log_stats=False)
    monkeypatch.setattr(client.client.chat.completions, "create", lambda **kwargs: iter(chunks))

    with client.client:
        responses = client.send_omni_request(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Describe the video."}],
                "modalities": ["text"],
                "stream": True,
                "allow_reasoning": True,
            }
        )

    assert responses[0].text_content == content
    assert responses[0].reasoning_content == "The circles are moving."
