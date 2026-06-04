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

import enum
import json
from unittest.mock import MagicMock

import pytest

# Python 3.10 compat: StrEnum was added in 3.11
if not hasattr(enum, "StrEnum"):

    class _StrEnum(str, enum.Enum):
        """Minimal StrEnum backport for Python 3.10."""

    enum.StrEnum = _StrEnum  # type: ignore[attr-defined]

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.outputs import CompletionOutput, RequestOutput

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_omni_output(
    request_id: str = "test-req",
    text: str = "hello",
    token_ids: list[int] | None = None,
    finish_reason: str | None = None,
    index: int = 0,
    num_prompt_tokens: int = 3,
) -> OmniRequestOutput:
    """Build an OmniRequestOutput wrapping a text RequestOutput."""
    if token_ids is None:
        token_ids = [10, 11, 12]
    res = RequestOutput(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=list(range(num_prompt_tokens)),
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=index,
                text=text,
                token_ids=token_ids,
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason=finish_reason,
                stop_reason=None,
            )
        ],
        finished=finish_reason is not None,
    )
    return OmniRequestOutput(
        request_id=request_id,
        final_output_type="text",
        request_output=res,
        finished=finish_reason is not None,
    )


def _make_audio_omni_output(
    request_id: str = "test-req",
    index: int = 0,
    num_prompt_tokens: int = 3,
) -> OmniRequestOutput:
    """Build an OmniRequestOutput for audio (no torch dependency)."""
    res = RequestOutput(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=list(range(num_prompt_tokens)),
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=index,
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
        request_id=request_id,
        final_output_type="audio",
        request_output=res,
        finished=True,
    )


def _mock_audio_choices(index: int = 0, role: str = "assistant"):
    return [
        ChatCompletionResponseStreamChoice(
            index=index,
            delta=DeltaMessage(role=role, content="dGVzdA=="),
            logprobs=None,
            finish_reason="stop",
        )
    ]


def _build_serving_chat():
    """Create a minimal OmniOpenAIServingChat for testing."""
    mock_engine = MagicMock()
    mock_engine.errored = False

    models = OpenAIServingModels(
        engine_client=mock_engine,
        base_model_paths=[],
    )
    mock_render = MagicMock()

    instance = OmniOpenAIServingChat(
        engine_client=mock_engine,
        models=models,
        response_role="assistant",
        openai_serving_render=mock_render,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    instance._create_audio_choice = MagicMock(
        side_effect=lambda omni_res, role, request, stream=False: _mock_audio_choices(
            index=omni_res.request_output.outputs[0].index,
            role=role,
        )
    )
    return instance


def _make_request(modalities: list[str], n: int = 1) -> ChatCompletionRequest:
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        n=n,
        stream=True,
    )
    req.modalities = modalities  # type: ignore[attr-defined]
    return req


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
    serving_chat = _build_serving_chat()
    request = _make_request(modalities=["text"])

    async def result_generator():
        yield _make_text_omni_output(text="he", token_ids=[10, 11], finish_reason=None)
        yield _make_text_omni_output(text="llo", token_ids=[12], finish_reason="stop")

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
    serving_chat = _build_serving_chat()
    request = _make_request(modalities=["text", "audio"])

    async def result_generator():
        yield _make_text_omni_output(text="he", token_ids=[10, 11], finish_reason=None)
        yield _make_text_omni_output(text="llo", token_ids=[12], finish_reason="stop")
        yield _make_audio_omni_output()

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
    serving_chat = _build_serving_chat()
    request = _make_request(modalities=["text", "audio"], n=2)

    async def result_generator():
        yield _make_text_omni_output(text="A", token_ids=[10], finish_reason=None, index=0)
        yield _make_text_omni_output(text="B", token_ids=[20], finish_reason=None, index=1)
        yield _make_text_omni_output(text="", token_ids=[11], finish_reason="stop", index=0)
        yield _make_text_omni_output(text="", token_ids=[21], finish_reason="stop", index=1)
        yield _make_audio_omni_output(index=0)
        yield _make_audio_omni_output(index=1)

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
    serving_chat = _build_serving_chat()
    request = _make_request(modalities=["audio"])

    async def result_generator():
        yield _make_audio_omni_output()

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
    serving_chat = _build_serving_chat()
    request = _make_request(modalities=["text", "audio"])

    async def result_generator():
        # Engine only produces text, no audio output at all
        yield _make_text_omni_output(text="hi", token_ids=[10], finish_reason=None)
        yield _make_text_omni_output(text="!", token_ids=[11], finish_reason="stop")

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
    serving_chat = _build_serving_chat()
    request = _make_request(modalities=["text", "audio"])

    async def result_generator():
        yield _make_text_omni_output(text="hi", token_ids=[10], finish_reason=None)
        yield _make_text_omni_output(text="!", token_ids=[11], finish_reason="stop")
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
