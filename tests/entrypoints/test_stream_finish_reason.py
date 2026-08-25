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

import time
from unittest.mock import MagicMock

import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.outputs import CompletionOutput, RequestOutput

from tests.helpers.serving_chat import (
    build_serving_chat,
    collect_stream,
    make_request,
    make_text_omni_output,
    parse_sse_chunks,
)
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio_omni_output(
    request_id: str = "test-req",
    index: int = 0,
    num_prompt_tokens: int = 3,
    stage_id: int | None = None,
    replica_id: int | None = None,
    audio_samples: int = 0,
) -> OmniRequestOutput:
    """Build an OmniRequestOutput for audio (no torch dependency)."""
    completion = CompletionOutput(
        index=index,
        text="",
        token_ids=[],
        cumulative_logprob=0.0,
        logprobs=None,
        finish_reason="stop",
        stop_reason=None,
    )
    if audio_samples > 0:
        completion.multimodal_output = {
            "audio": [MagicMock(numel=MagicMock(return_value=audio_samples))],
            "sr": 24000,
        }
    res = RequestOutput(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=list(range(num_prompt_tokens)),
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
    )
    return OmniRequestOutput.from_stage_output(
        res,
        request_id=request_id,
        stage_id=stage_id,
        replica_id=replica_id,
        final_output_type="audio",
        finished=True,
    )


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

    raw_lines = await collect_stream(
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

    chunks = parse_sse_chunks(raw_lines)
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
        yield _make_audio_omni_output()

    raw_lines = await collect_stream(
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

    chunks = parse_sse_chunks(raw_lines)
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
        yield _make_audio_omni_output(index=0)
        yield _make_audio_omni_output(index=1)

    raw_lines = await collect_stream(
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

    chunks = parse_sse_chunks(raw_lines)
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
        yield _make_audio_omni_output()

    raw_lines = await collect_stream(
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

    chunks = parse_sse_chunks(raw_lines)
    finish_reasons = [ch["finish_reason"] for c in chunks for ch in c.get("choices", [])]

    assert finish_reasons.count("stop") == 1
    assert finish_reasons[-1] == "stop"


@pytest.mark.parametrize(
    ("output_replica_id", "pool_replica_id", "expected_replica_id", "pool_lookup"),
    [
        (7, None, 7, False),
        (None, 5, 5, True),
    ],
)
@pytest.mark.asyncio
async def test_streaming_audio_metrics_resolve_replica_id(
    monkeypatch,
    output_replica_id,
    pool_replica_id,
    expected_replica_id,
    pool_lookup,
):
    import vllm_omni.entrypoints.openai.serving_chat as serving_chat_mod
    from vllm_omni.entrypoints.client_request_state import ClientRequestState

    serving_chat = build_serving_chat()
    request = make_request(modalities=["audio"])
    req_state = ClientRequestState(
        request_id="internal-req",
        external_request_id="test-req",
    )
    req_state.request_arrival_ts = time.time() - 1.0
    serving_chat.engine_client.request_states = {"internal-req": req_state}
    serving_chat.engine_client.mod_metrics = object()
    serving_chat.engine_client.engine = MagicMock()
    serving_chat.engine_client.engine.stage_pools = [MagicMock(), MagicMock(), MagicMock()]
    serving_chat.engine_client.engine.stage_pools[2].get_bound_replica_id.return_value = pool_replica_id

    first_packet_calls = []
    finalize_calls = []

    def fake_observe_audio_first_packet(*args, **kwargs):
        first_packet_calls.append(kwargs)

    def fake_observe_audio_streaming_finalize(*args, **kwargs):
        finalize_calls.append(kwargs)

    monkeypatch.setattr(serving_chat_mod, "observe_audio_first_packet", fake_observe_audio_first_packet)
    monkeypatch.setattr(serving_chat_mod, "observe_audio_streaming_finalize", fake_observe_audio_streaming_finalize)

    async def result_generator():
        yield _make_audio_omni_output(stage_id=2, replica_id=output_replica_id, audio_samples=2400)

    await collect_stream(
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

    if pool_lookup:
        serving_chat.engine_client.engine.stage_pools[2].get_bound_replica_id.assert_called_once_with("internal-req")
    else:
        serving_chat.engine_client.engine.stage_pools[2].get_bound_replica_id.assert_not_called()
    assert first_packet_calls[0]["stage_id"] == 2
    assert first_packet_calls[0]["replica_id"] == expected_replica_id
    assert finalize_calls[0]["stage_id"] == 2
    assert finalize_calls[0]["replica_id"] == expected_replica_id


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

    raw_lines = await collect_stream(
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

    chunks = parse_sse_chunks(raw_lines)
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

    raw_lines = await collect_stream(
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

    chunks = parse_sse_chunks(raw_lines)

    # Find the text finish chunk (content "!")
    for c in chunks:
        for ch in c.get("choices", []):
            if c.get("modality") == "text" and ch.get("delta", {}).get("content") == "!":
                # Text finish should be suppressed because audio hasn't appeared
                assert ch["finish_reason"] is None


# ---------------------------------------------------------------------------
# Tests: audio chunks that carry no waveform
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_chunk_without_waveform_keeps_stream_alive():
    """An audio message with no PCM must not break the stream.

    ``_create_audio_choice`` used to return an ErrorResponse here, which the
    stream generator then iterated as a list of choices; the pydantic model
    yields (field, value) tuples, so the loop raised
    ``AttributeError: 'tuple' object has no attribute 'finish_reason'``.
    """
    serving_chat = build_serving_chat()
    request = make_request(modalities=["text", "audio"])

    empty_audio = _make_audio_omni_output()
    empty_audio.outputs[0].multimodal_output = {"audio": []}

    def create_audio_choice(omni_res, role, request, stream=False):
        return OmniOpenAIServingChat._create_audio_choice(serving_chat, omni_res, role, request, stream=stream)

    serving_chat._create_audio_choice = create_audio_choice

    async def result_generator():
        yield make_text_omni_output(text="hi", token_ids=[10], finish_reason=None)
        yield make_text_omni_output(text="!", token_ids=[11], finish_reason="stop")
        yield empty_audio

    raw_lines = await collect_stream(
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

    assert not any("Error in chat completion stream generator" in line for line in raw_lines)
    chunks = parse_sse_chunks(raw_lines)
    finish_reasons = [ch["finish_reason"] for c in chunks for ch in c.get("choices", [])]
    assert finish_reasons.count("stop") == 1, f"Expected 1 stop, got {finish_reasons}"
    assert finish_reasons[-1] == "stop"


@pytest.mark.asyncio
async def test_audio_choice_error_response_is_not_iterated_as_choices():
    """Any ErrorResponse from the audio path is skipped, not iterated."""
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse

    serving_chat = build_serving_chat()
    request = make_request(modalities=["text", "audio"])

    serving_chat._create_audio_choice = MagicMock(
        side_effect=lambda omni_res, role, request, stream=False: serving_chat._create_error_response("boom")
    )

    async def result_generator():
        yield make_text_omni_output(text="hi", token_ids=[10], finish_reason="stop")
        yield _make_audio_omni_output()

    raw_lines = await collect_stream(
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

    assert isinstance(serving_chat._create_error_response("boom"), ErrorResponse)
    assert not any("AttributeError" in line for line in raw_lines)
    chunks = parse_sse_chunks(raw_lines)
    finish_reasons = [ch["finish_reason"] for c in chunks for ch in c.get("choices", [])]
    assert finish_reasons.count("stop") == 1, f"Expected 1 stop, got {finish_reasons}"


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
