# SPDX-License-Identifier: Apache-2.0
"""Tests for non-streaming choice merging in chat_completion_full_generator.

Verifies that /v1/chat/completions with stream=False produces the correct
number of choices and merges text+audio into a single choice per the
OpenAI API spec.
"""

import enum
from unittest.mock import MagicMock

import pytest

if not hasattr(enum, "StrEnum"):

    class _StrEnum(str, enum.Enum):
        pass

    enum.StrEnum = _StrEnum  # type: ignore[attr-defined]

from openai.types.chat.chat_completion_audio import (
    ChatCompletionAudio as OpenAIChatCompletionAudio,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.outputs import CompletionOutput, RequestOutput

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FAKE_AUDIO = OpenAIChatCompletionAudio(
    id="audio-test",
    data="dGVzdA==",
    expires_at=9999999999,
    transcript="",
)


def _make_text_omni_output(
    text="hello",
    finish_reason="stop",
    index=0,
):
    res = RequestOutput(
        request_id="req",
        prompt="test",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=index,
                text=text,
                token_ids=[10, 11, 12],
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason=finish_reason,
                stop_reason=None,
            )
        ],
        finished=True,
    )
    return OmniRequestOutput(
        request_id="req",
        final_output_type="text",
        request_output=res,
        finished=True,
    )


def _make_audio_omni_output():
    res = RequestOutput(
        request_id="req",
        prompt="test",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=[],
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason="stop",
                stop_reason=None,
            )
        ],
        finished=True,
    )
    return OmniRequestOutput(
        request_id="req",
        final_output_type="audio",
        request_output=res,
        finished=True,
    )


def _mock_audio_choices(role="assistant"):
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


def _build_serving_chat():
    mock_engine = MagicMock()
    mock_engine.errored = False

    models = OpenAIServingModels(
        engine_client=mock_engine,
        base_model_paths=[],
    )

    instance = OmniOpenAIServingChat(
        engine_client=mock_engine,
        models=models,
        response_role="assistant",
        openai_serving_render=MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    instance._create_audio_choice = MagicMock(
        side_effect=lambda *a, **kw: _mock_audio_choices()
    )
    return instance


def _make_request(modalities, stream=False):
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        stream=stream,
    )
    req.modalities = modalities
    return req


async def _async_iter(items):
    for item in items:
        yield item


async def _call_full_generator(serving, outputs, modalities):
    request = _make_request(modalities)
    tokenizer = MagicMock()
    metadata = MagicMock()
    result = await serving.chat_completion_full_generator(
        request=request,
        result_generator=_async_iter(outputs),
        request_id="chatcmpl-test",
        model_name="test-model",
        conversation=[],
        tokenizer=tokenizer,
        request_metadata=metadata,
    )
    return result


@pytest.mark.asyncio
async def test_text_only():
    serving = _build_serving_chat()
    text_output = _make_text_omni_output(text="hello world")

    text_choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content="hello world"),
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    serving._create_text_choice = MagicMock(
        return_value=_mock_text_choice_result([text_choice])
    )

    response = await _call_full_generator(serving, [text_output], ["text"])

    assert len(response.choices) == 1
    assert response.choices[0].message.content == "hello world"
    assert response.choices[0].message.audio is None


@pytest.mark.asyncio
async def test_audio_only():
    serving = _build_serving_chat()
    audio_output = _make_audio_omni_output()

    response = await _call_full_generator(serving, [audio_output], ["audio"])

    assert len(response.choices) == 1
    assert response.choices[0].message.audio is FAKE_AUDIO


@pytest.mark.asyncio
async def test_text_and_audio_merged():
    serving = _build_serving_chat()
    text_output = _make_text_omni_output(text="hello")
    audio_output = _make_audio_omni_output()

    text_choice = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content="hello"),
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    serving._create_text_choice = MagicMock(
        return_value=_mock_text_choice_result([text_choice])
    )

    response = await _call_full_generator(
        serving, [text_output, audio_output], ["text", "audio"]
    )

    assert len(response.choices) == 1
    assert response.choices[0].message.content == "hello"
    assert response.choices[0].message.audio is FAKE_AUDIO


@pytest.mark.asyncio
async def test_text_and_audio_merged_n2():
    serving = _build_serving_chat()
    text_output = _make_text_omni_output(text="first")
    audio_output = _make_audio_omni_output()

    choice_0 = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content="first"),
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    choice_1 = ChatCompletionResponseChoice(
        index=1,
        message=ChatMessage(role="assistant", content="second"),
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    serving._create_text_choice = MagicMock(
        return_value=_mock_text_choice_result([choice_0, choice_1])
    )

    response = await _call_full_generator(
        serving, [text_output, audio_output], ["text", "audio"]
    )

    assert len(response.choices) == 2
    assert response.choices[0].index == 0
    assert response.choices[0].message.content == "first"
    assert response.choices[0].message.audio is FAKE_AUDIO
    assert response.choices[1].index == 1
    assert response.choices[1].message.content == "second"
    assert response.choices[1].message.audio is FAKE_AUDIO
