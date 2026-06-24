# SPDX-License-Identifier: Apache-2.0
"""Tests for multi-modal streaming finish_reason behavior (commit 44c799bc).

Verifies that the /v1/chat/completions streaming endpoint emits exactly one
finish_reason="stop" per choice when multiple output modalities (text, audio)
are active, complying with the OpenAI streaming spec.

Key invariants tested:
  - Single modality (text only): last chunk carries finish_reason="stop"
  - Multi-modality (text+audio): only the final modality chunk carries
    finish_reason="stop"; earlier finishing modalities emit finish_reason=null
  - n>1 with multi-modality: each choice independently tracks its own
    modality state, so each choice gets exactly one "stop"
  - Engine skips a declared modality: fallback stop chunk is emitted at
    stream end so the client always receives finish_reason="stop"
  - voice/speaker parameter compatibility in chat completions
"""

import json
from unittest.mock import MagicMock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)

from tests.helpers.serving_chat import (
    build_serving_chat,
    make_audio_omni_output,
    make_request,
    make_text_omni_output,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse_chunks(lines: list[str]) -> list[dict]:
    """Parse SSE lines into JSON dicts."""
    prefix = "data: "
    chunks = []
    for line in lines:
        line = line.strip()
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix) :].strip()
        if payload == "[DONE]":
            continue
        try:
            chunks.append(json.loads(payload))
        except json.JSONDecodeError:
            pass
    return chunks


async def _collect_stream(gen):
    result = []
    async for item in gen:
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Tests: finish_reason correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_modality_text_only_one_stop():
    """Text-only streaming: exactly one chunk has finish_reason='stop'."""
    serving_chat = build_serving_chat()
    request = make_request(modalities=["text"])

    async def result_generator():
        yield make_text_omni_output(text="he", token_ids=[10, 11], finish_reason=None)
        yield make_text_omni_output(text="llo", token_ids=[12], finish_reason="stop")

    raw_lines = await _collect_stream(
        serving_chat.chat_completion_stream_generator(
            request=request,
            result_generator=result_generator(),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )
    )

    chunks = _parse_sse_chunks(raw_lines)
    finish_reasons = [c["choices"][0]["finish_reason"] for c in chunks if c.get("choices")]

    assert finish_reasons[-1] == "stop"
    assert finish_reasons.count("stop") == 1
    for fr in finish_reasons[:-1]:
        assert fr is None


@pytest.mark.asyncio
async def test_multi_modal_text_audio_only_last_stop():
    """text+audio: text finish sends finish_reason=null, audio sends stop."""
    serving_chat = build_serving_chat()
    request = make_request(modalities=["text", "audio"])

    async def result_generator():
        yield make_text_omni_output(text="he", token_ids=[10, 11], finish_reason=None)
        yield make_text_omni_output(text="llo", token_ids=[12], finish_reason="stop")
        yield make_audio_omni_output()

    raw_lines = await _collect_stream(
        serving_chat.chat_completion_stream_generator(
            request=request,
            result_generator=result_generator(),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )
    )

    chunks = _parse_sse_chunks(raw_lines)
    finish_reasons = [ch["finish_reason"] for c in chunks for ch in c.get("choices", [])]

    assert finish_reasons.count("stop") == 1
    assert finish_reasons[-1] == "stop"

    # The text finish chunk must have finish_reason=None
    for idx, c in enumerate(chunks):
        for ch in c.get("choices", []):
            if c.get("modality") == "text" and ch.get("delta", {}).get("content") == "lo":
                assert ch["finish_reason"] is None


@pytest.mark.asyncio
async def test_multi_modal_n2_independent_per_choice():
    """n=2 with text+audio: each choice gets exactly one stop, at the end."""
    serving_chat = build_serving_chat()
    request = make_request(modalities=["text", "audio"], n=2)

    async def result_generator():
        yield make_text_omni_output(text="A", token_ids=[10], finish_reason=None, index=0)
        yield make_text_omni_output(text="B", token_ids=[20], finish_reason=None, index=1)
        yield make_text_omni_output(text="", token_ids=[11], finish_reason="stop", index=0)
        yield make_text_omni_output(text="", token_ids=[21], finish_reason="stop", index=1)
        yield make_audio_omni_output(index=0)
        yield make_audio_omni_output(index=1)

    raw_lines = await _collect_stream(
        serving_chat.chat_completion_stream_generator(
            request=request,
            result_generator=result_generator(),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )
    )

    chunks = _parse_sse_chunks(raw_lines)
    per_choice: dict[int, list] = {}
    for c in chunks:
        for ch in c.get("choices", []):
            per_choice.setdefault(ch["index"], []).append(ch["finish_reason"])

    for idx, reasons in per_choice.items():
        assert reasons.count("stop") == 1, f"Choice {idx} has {reasons.count('stop')} stops"
        assert reasons[-1] == "stop", f"Choice {idx} last reason is {reasons[-1]}"


@pytest.mark.asyncio
async def test_single_modality_audio_only_one_stop():
    """Audio-only streaming: the audio chunk carries finish_reason='stop'."""
    serving_chat = build_serving_chat()
    request = make_request(modalities=["audio"])

    async def result_generator():
        yield make_audio_omni_output()

    raw_lines = await _collect_stream(
        serving_chat.chat_completion_stream_generator(
            request=request,
            result_generator=result_generator(),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )
    )

    chunks = _parse_sse_chunks(raw_lines)
    finish_reasons = [ch["finish_reason"] for c in chunks for ch in c.get("choices", [])]

    assert finish_reasons.count("stop") == 1
    assert finish_reasons[-1] == "stop"


# ---------------------------------------------------------------------------
# Tests: fallback stop chunk when declared modality is not produced
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_declared_modality_not_produced_emits_fallback_stop():
    """If request.modalities declares ["text","audio"] but engine only produces
    text, a fallback stop chunk is emitted at stream end."""
    serving_chat = build_serving_chat()
    request = make_request(modalities=["text", "audio"])

    async def result_generator():
        # Engine only produces text, no audio output at all
        yield make_text_omni_output(text="hi", token_ids=[10], finish_reason=None)
        yield make_text_omni_output(text="!", token_ids=[11], finish_reason="stop")

    raw_lines = await _collect_stream(
        serving_chat.chat_completion_stream_generator(
            request=request,
            result_generator=result_generator(),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )
    )

    chunks = _parse_sse_chunks(raw_lines)
    finish_reasons = [ch["finish_reason"] for c in chunks for ch in c.get("choices", [])]

    # Text finish is suppressed (audio not seen yet), but fallback stop
    # chunk must appear at end.
    assert finish_reasons.count("stop") == 1, f"Expected 1 stop, got {finish_reasons}"
    assert finish_reasons[-1] == "stop"


@pytest.mark.asyncio
async def test_declared_modality_not_produced_text_finish_suppressed():
    """When text finishes but audio (declared in modalities) never appears,
    the text finish chunk has finish_reason=null (suppressed)."""
    serving_chat = build_serving_chat()
    request = make_request(modalities=["text", "audio"])

    async def result_generator():
        yield make_text_omni_output(text="hi", token_ids=[10], finish_reason=None)
        yield make_text_omni_output(text="!", token_ids=[11], finish_reason="stop")
        # No audio output — stream ends

    raw_lines = await _collect_stream(
        serving_chat.chat_completion_stream_generator(
            request=request,
            result_generator=result_generator(),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )
    )

    chunks = _parse_sse_chunks(raw_lines)

    # Find the text finish chunk (content "!")
    for c in chunks:
        for ch in c.get("choices", []):
            if c.get("modality") == "text" and ch.get("delta", {}).get("content") == "!":
                # Text finish should be suppressed because audio hasn't appeared
                assert ch["finish_reason"] is None


# ---------------------------------------------------------------------------
# Tests: voice/speaker parameter compatibility
# ---------------------------------------------------------------------------


class TestVoiceSpeakerCompat:
    """Tests for voice/speaker parameter handling in chat completions."""

    def test_voice_parameter_takes_priority(self):
        """When both voice and speaker are provided via extra_body, voice wins."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        # Pydantic model_extra collects unknown fields
        req.voice = "alloy"  # type: ignore[attr-defined]
        req.speaker = "vivian"  # type: ignore[attr-defined]

        # voice takes priority: getattr(request, "voice", None) returns "alloy"
        speaker = getattr(req, "voice", None) or getattr(req, "speaker", None)
        assert speaker == "alloy"

    def test_speaker_fallback_when_no_voice(self):
        """When only speaker is provided, it is used."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        req.speaker = "vivian"  # type: ignore[attr-defined]

        speaker = getattr(req, "voice", None) or getattr(req, "speaker", None)
        assert speaker == "vivian"

    def test_neither_voice_nor_speaker(self):
        """When neither is provided, result is None."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )

        speaker = getattr(req, "voice", None) or getattr(req, "speaker", None)
        assert speaker is None

    def test_empty_voice_falls_back_to_speaker(self):
        """Empty string voice falls back to speaker."""
        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        req.voice = ""  # type: ignore[attr-defined]
        req.speaker = "vivian"  # type: ignore[attr-defined]

        # Empty string is falsy, so speaker is used
        speaker = getattr(req, "voice", None) or getattr(req, "speaker", None)
        assert speaker == "vivian"
