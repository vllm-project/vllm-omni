# SPDX-License-Identifier: Apache-2.0
"""Tests for non-streaming choice merging in chat_completion_full_generator.

Verifies that /v1/chat/completions with stream=False produces the correct
number of choices and merges text+audio into a single choice per the
OpenAI API spec.
"""

from unittest.mock import MagicMock

import pytest
from openai.types.chat.chat_completion_audio import (
    ChatCompletionAudio as OpenAIChatCompletionAudio,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import UsageInfo

from tests.helpers.serving_chat import (
    build_serving_chat,
    make_audio_omni_output,
    make_text_omni_output,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FAKE_AUDIO = OpenAIChatCompletionAudio(
    id="audio-test",
    data="dGVzdA==",
    expires_at=9999999999,
    transcript="",
)


def _mock_full_audio_choices(role="assistant"):
    return [
        ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role=role, audio=FAKE_AUDIO),
            logprobs=None,
            finish_reason="stop",
            stop_reason=None,
        )
    ]


def _mock_text_choice_result(choices):
    return (
        choices,
        UsageInfo(prompt_tokens=3, completion_tokens=3, total_tokens=6),
        None,
        [1, 2, 3],
        None,
    )


async def _async_iter(items):
    for item in items:
        yield item


async def _call_full_generator(serving, outputs, modalities):
    from tests.helpers.serving_chat import make_request

    request = make_request(modalities, stream=False)
    result = await serving.chat_completion_full_generator(
        request=request,
        result_generator=_async_iter(outputs),
        request_id="chatcmpl-test",
        model_name="test-model",
        conversation=[],
        tokenizer=MagicMock(),
        request_metadata=MagicMock(),
    )
    return result


@pytest.mark.asyncio
async def test_text_only():
    serving = build_serving_chat()
    text_output = make_text_omni_output(text="hello world", finish_reason="stop")

    text_choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content="hello world"),
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    serving._create_text_choice = MagicMock(return_value=_mock_text_choice_result([text_choice]))

    response = await _call_full_generator(serving, [text_output], ["text"])

    assert len(response.choices) == 1
    assert response.choices[0].message.content == "hello world"
    assert response.choices[0].message.audio is None


@pytest.mark.asyncio
async def test_audio_only():
    serving = build_serving_chat()
    serving._create_audio_choice = MagicMock(side_effect=lambda *a, **kw: _mock_full_audio_choices())
    audio_output = make_audio_omni_output()

    response = await _call_full_generator(serving, [audio_output], ["audio"])

    assert len(response.choices) == 1
    assert response.choices[0].message.audio is FAKE_AUDIO


@pytest.mark.asyncio
async def test_text_and_audio_merged():
    serving = build_serving_chat()
    serving._create_audio_choice = MagicMock(side_effect=lambda *a, **kw: _mock_full_audio_choices())
    text_output = make_text_omni_output(text="hello", finish_reason="stop")
    audio_output = make_audio_omni_output()

    text_choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content="hello"),
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    serving._create_text_choice = MagicMock(return_value=_mock_text_choice_result([text_choice]))

    response = await _call_full_generator(serving, [text_output, audio_output], ["text", "audio"])

    assert len(response.choices) == 1
    assert response.choices[0].message.content == "hello"
    assert response.choices[0].message.audio is FAKE_AUDIO
