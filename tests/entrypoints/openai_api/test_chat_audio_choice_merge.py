# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for merging audio final outputs into text choices.

Covers #7376: a non-streaming chat request with
``modalities=["text", "audio"]`` used to return two choices with duplicate
``index=0`` — one carrying the text and one carrying the audio — so clients
reading ``choices[0]`` silently lost the audio.
"""

from __future__ import annotations

import json

import pytest
from openai.types.chat.chat_completion_audio import ChatCompletionAudio as OpenAIChatCompletionAudio
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.serve.engine.protocol import UsageInfo

from vllm_omni.entrypoints.openai.protocol.audio import AudioChunkMetadata
from vllm_omni.entrypoints.openai.protocol.chat_completion import OmniChatCompletionResponseChoice
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _text_choice(index: int, content: str) -> ChatCompletionResponseChoice:
    return ChatCompletionResponseChoice(
        index=index,
        message=ChatMessage(role="assistant", content=content),
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )


def _audio_choice(index: int, data: str = "Zm9v") -> OmniChatCompletionResponseChoice:
    return OmniChatCompletionResponseChoice(
        index=index,
        message=ChatMessage(
            role="assistant",
            audio=OpenAIChatCompletionAudio(id=f"audio-{index}", data=data, expires_at=0, transcript=""),
        ),
        audio_metadata=AudioChunkMetadata(format="wav", sample_rate_hz=24000, frame_count=17, channels=1),
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )


@pytest.fixture
def serving_chat():
    return object.__new__(OmniOpenAIServingChat)


class TestMergeAudioChoices:
    def test_text_and_audio_merge_into_single_choice(self, serving_chat):
        merged = serving_chat._merge_audio_choices([_text_choice(0, "hello")], [_audio_choice(0)])

        assert len(merged) == 1
        assert merged[0].index == 0
        assert merged[0].message.content == "hello"
        assert merged[0].message.audio is not None
        assert merged[0].message.audio.data == "Zm9v"
        assert isinstance(merged[0], OmniChatCompletionResponseChoice)
        assert merged[0].audio_metadata is not None
        assert merged[0].audio_metadata.format == "wav"

    def test_no_duplicate_indexes_in_serialized_response(self, serving_chat):
        """The exact #7376 symptom: two choices with index=0 in one response."""
        merged = serving_chat._merge_audio_choices([_text_choice(0, "hello")], [_audio_choice(0)])

        response = ChatCompletionResponse(
            id="chatcmpl-7376",
            created=0,
            model="test-model",
            choices=merged,
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        serialized = json.loads(response.model_dump_json())
        indexes = [c["index"] for c in serialized["choices"]]
        assert indexes == [0]
        assert serialized["choices"][0]["message"]["content"] == "hello"
        assert serialized["choices"][0]["message"]["audio"]["data"] == "Zm9v"

    def test_audio_only_request_keeps_standalone_choice(self, serving_chat):
        merged = serving_chat._merge_audio_choices([], [_audio_choice(0)])

        assert len(merged) == 1
        assert merged[0].message.audio is not None
        assert merged[0].message.content is None

    def test_n_gt_1_merges_by_index(self, serving_chat):
        text_choices = [_text_choice(0, "first"), _text_choice(1, "second")]
        audio_choices = [_audio_choice(0, "AAAA"), _audio_choice(1, "BBBB")]

        merged = serving_chat._merge_audio_choices(text_choices, audio_choices)

        assert len(merged) == 2
        assert [c.index for c in merged] == [0, 1]
        assert merged[0].message.content == "first"
        assert merged[0].message.audio.data == "AAAA"
        assert merged[1].message.content == "second"
        assert merged[1].message.audio.data == "BBBB"

    def test_audio_without_matching_text_choice_is_appended(self, serving_chat):
        # n=1 text but the audio stage reported a second index (defensive):
        # the unmatched audio choice must not be dropped.
        merged = serving_chat._merge_audio_choices([_text_choice(0, "only")], [_audio_choice(1)])

        assert len(merged) == 2
        assert merged[0].message.audio is None
        assert merged[1].message.audio is not None

    def test_text_choice_fields_survive_merge(self, serving_chat):
        text = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content="kept"),
            logprobs=None,
            finish_reason="length",
            stop_reason="max_tokens",
        )

        merged = serving_chat._merge_audio_choices([text], [_audio_choice(0)])

        assert merged[0].finish_reason == "length"
        assert merged[0].stop_reason == "max_tokens"
        assert merged[0].message.content == "kept"
