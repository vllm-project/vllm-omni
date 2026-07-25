# SPDX-License-Identifier: Apache-2.0
"""Unit tests for non-stream ChatCompletionAudio.transcript population."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def serving_chat():
    chat = object.__new__(OmniOpenAIServingChat)
    chat.create_audio = MagicMock(return_value=SimpleNamespace(audio_data="YmFzZTY0YXVkaW8=", media_type="audio/wav"))
    return chat


def _audio_omni_output(text_in_mm: str | None = None):
    mm_output = {"audio": torch.zeros(8, dtype=torch.float32), "sr": 24000}
    if text_in_mm is not None:
        mm_output["transcript"] = text_in_mm
    completion = SimpleNamespace(
        index=0,
        multimodal_output=mm_output,
        finish_reason="stop",
        stop_reason=None,
        token_ids=[],
    )
    request_output = SimpleNamespace(outputs=[completion])
    return SimpleNamespace(request_output=request_output, final_output_type="audio")


class _FakeReasoningParser:
    def extract_reasoning(self, model_output: str, request=None):
        marker = "</think>"
        if marker in model_output:
            reasoning, content = model_output.split(marker, 1)
            return reasoning, content
        return model_output, ""


def test_resolve_audio_transcript_prefers_index_map():
    assert (
        OmniOpenAIServingChat._resolve_audio_transcript({0: "hello from thinker"}, 0, {"transcript": "mm"})
        == "hello from thinker"
    )


def test_resolve_audio_transcript_falls_back_to_mm_output():
    assert OmniOpenAIServingChat._resolve_audio_transcript({}, 0, {"transcript": "from mm"}) == "from mm"
    assert OmniOpenAIServingChat._resolve_audio_transcript(None, 0, {"text": "from text"}) == "from text"
    assert OmniOpenAIServingChat._resolve_audio_transcript(None, 0, None) == ""


def test_visible_content_for_transcript_strips_reasoning():
    request = SimpleNamespace()
    raw = "<think>hidden plan</think>Hello world"
    assert OmniOpenAIServingChat._visible_content_for_transcript(raw, request, _FakeReasoningParser()) == "Hello world"
    assert OmniOpenAIServingChat._visible_content_for_transcript(raw, request, None) == raw


def test_create_audio_choice_fills_nonstream_transcript(serving_chat):
    request = SimpleNamespace(return_token_ids=False)
    choices = OmniOpenAIServingChat._create_audio_choice(
        serving_chat,
        _audio_omni_output(),
        role="assistant",
        request=request,
        stream=False,
        transcripts={0: "Spoken reply from thinker."},
    )
    assert len(choices) == 1
    assert choices[0].message.audio is not None
    assert choices[0].message.audio.transcript == "Spoken reply from thinker."
    assert choices[0].message.audio.data == "YmFzZTY0YXVkaW8="


def test_create_audio_choice_uses_mm_transcript_fallback(serving_chat):
    request = SimpleNamespace(return_token_ids=False)
    choices = OmniOpenAIServingChat._create_audio_choice(
        serving_chat,
        _audio_omni_output(text_in_mm="mm transcript"),
        role="assistant",
        request=request,
        stream=False,
        transcripts=None,
    )
    assert choices[0].message.audio.transcript == "mm transcript"
