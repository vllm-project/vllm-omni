# SPDX-License-Identifier: Apache-2.0
"""Regression tests for merging audio-only choices into text choices.

AR pipelines finish one output per modality (stage 0 text, then a later
audio stage). Before the merge, a non-streaming text+audio request returned
two choices that both claimed index 0: the text choice carried content and
no audio, the audio-only choice carried audio and no content. Spec-shaped
clients reading choices[0] silently lost the audio.
"""

from __future__ import annotations

import pytest
from openai.types.chat.chat_completion_audio import ChatCompletionAudio

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponseChoice,
    ChatMessage,
)

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _audio(audio_id: str = "audio-abc123") -> ChatCompletionAudio:
    return ChatCompletionAudio(
        id=audio_id,
        data="UklGRiQ=",
        expires_at=1789122547,
        transcript="",
    )


def _text_choice(index: int = 0, content: str = "hello") -> ChatCompletionResponseChoice:
    return ChatCompletionResponseChoice(
        index=index,
        message=ChatMessage(role="assistant", content=content),
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )


def _audio_only_choice(index: int = 0, audio_id: str = "audio-abc123") -> ChatCompletionResponseChoice:
    return ChatCompletionResponseChoice(
        index=index,
        message=ChatMessage(role="assistant", audio=_audio(audio_id)),
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )


def _merge(choices):
    return OmniOpenAIServingChat._merge_audio_choices_into_text(choices)


def test_text_and_audio_choices_merge_into_one():
    merged = _merge([_text_choice(), _audio_only_choice()])

    assert len(merged) == 1
    message = merged[0].message
    assert message.content == "hello"
    assert message.audio is not None
    assert message.audio.id == "audio-abc123"
    assert merged[0].index == 0


def test_audio_only_request_keeps_standalone_choice():
    merged = _merge([_audio_only_choice()])

    assert len(merged) == 1
    assert merged[0].message.content is None
    assert merged[0].message.audio is not None


def test_audio_choice_before_text_choice_merges():
    merged = _merge([_audio_only_choice(), _text_choice()])

    assert len(merged) == 1
    assert merged[0].message.content == "hello"
    assert merged[0].message.audio is not None


def test_n_greater_than_one_pairs_by_index():
    merged = _merge(
        [
            _text_choice(index=0, content="first"),
            _text_choice(index=1, content="second"),
            _audio_only_choice(index=0, audio_id="audio-zero"),
            _audio_only_choice(index=1, audio_id="audio-one"),
        ]
    )

    assert len(merged) == 2
    by_index = {choice.index: choice for choice in merged}
    assert by_index[0].message.content == "first"
    assert by_index[0].message.audio.id == "audio-zero"
    assert by_index[1].message.content == "second"
    assert by_index[1].message.audio.id == "audio-one"


def test_choices_without_audio_are_untouched():
    text = _text_choice(index=0)
    # Diffusion image choices carry list content via model_construct, the
    # same bypass used by _create_image_choice (ChatMessage.content is str
    # only for validated construction).
    image_message = ChatMessage.model_construct(role="assistant")
    object.__setattr__(image_message, "content", [{"type": "image_url"}])
    if hasattr(image_message, "__pydantic_fields_set__"):
        image_message.__pydantic_fields_set__.add("content")
    image_style = ChatCompletionResponseChoice(
        index=1,
        message=image_message,
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )

    merged = _merge([text, image_style])

    assert merged == [text, image_style]


def test_empty_choices_passthrough():
    assert _merge([]) == []
