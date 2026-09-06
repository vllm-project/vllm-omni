# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OmniChatCompletionResponse/StreamResponse metrics field."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.helpers.serving_chat import (
    build_serving_chat,
    collect_stream,
    make_request,
    make_text_omni_output,
    parse_sse_chunks,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_omni_chat_completion_response_metrics():
    """Test OmniChatCompletionResponse metrics field works correctly."""
    from vllm.entrypoints.openai.engine.protocol import UsageInfo

    from vllm_omni.entrypoints.openai.protocol.chat_completion import (
        OmniChatCompletionResponse,
    )

    usage = UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    # Default is None
    response = OmniChatCompletionResponse(id="test-id", created=1234567890, model="test-model", choices=[], usage=usage)
    assert response.metrics is None

    # Can set metrics and serialize
    test_metrics = {"thinker_ttft": 0.123, "talker_ttft": 0.456}
    response = OmniChatCompletionResponse(
        id="test-id",
        created=1234567890,
        model="test-model",
        choices=[],
        usage=usage,
        metrics=test_metrics,
    )
    assert response.metrics == test_metrics
    assert "thinker_ttft" in response.model_dump_json()


def test_omni_chat_completion_stream_response_metrics():
    """Test OmniChatCompletionStreamResponse metrics and modality fields."""
    from vllm_omni.entrypoints.openai.protocol.chat_completion import (
        OmniChatCompletionStreamResponse,
    )

    response = OmniChatCompletionStreamResponse(
        id="test-id",
        created=1234567890,
        model="test-model",
        choices=[],
        modality="audio",
        metrics={"stage_latency": 0.5},
    )
    assert response.modality == "audio"
    assert response.metrics == {"stage_latency": 0.5}


@pytest.mark.asyncio
async def test_stream_prompt_token_details_include_zero_cached_and_multimodal_tokens():
    serving_chat = build_serving_chat()
    serving_chat.enable_prompt_tokens_details = True
    request = make_request(modalities=["text"], include_usage=True)

    async def result_generator():
        yield make_text_omni_output(
            token_ids=[10],
            finish_reason="stop",
            num_cached_tokens=0,
        )

    raw_lines = await collect_stream(
        serving_chat.chat_completion_stream_generator(
            request=request,
            result_generator=result_generator(),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
            mm_token_counts={"image": 7},
        )
    )

    chunks = parse_sse_chunks(raw_lines)
    usage = next(chunk["usage"] for chunk in chunks if chunk.get("usage"))

    assert usage["prompt_tokens_details"] == {
        "cached_tokens": 0,
        "multimodal_tokens": {"image": 7},
    }


def test_non_stream_prompt_token_details_include_zero_cached_and_multimodal_tokens():
    serving_chat = build_serving_chat()
    serving_chat.enable_prompt_tokens_details = True
    request = make_request(modalities=["text"], stream=False)
    omni_output = make_text_omni_output(
        token_ids=[10],
        finish_reason="stop",
        num_cached_tokens=0,
    )

    _, usage, _, _, _ = serving_chat._create_text_choice(
        request=request,
        omni_outputs=omni_output,
        tokenizer=MagicMock(),
        conversation=[],
        role="assistant",
        mm_token_counts={"audio": 5},
    )

    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 0
    assert usage.prompt_tokens_details.multimodal_tokens == {"audio": 5}


def test_create_image_choice_exposes_diffusion_metrics():
    """Ensure image chat content exposes profiler metrics for clients."""
    from PIL import Image

    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    stage_durations = {"prefill": 0.12, "diffusion": 1.23}
    peak_memory_mb = 3210.5
    omni_outputs = SimpleNamespace(
        request_output=None,
        stage_durations=stage_durations,
        peak_memory_mb=peak_memory_mb,
        images=[Image.new("RGB", (2, 2), color=(255, 0, 0))],
    )

    choices = OmniOpenAIServingChat._create_image_choice(  # type: ignore[misc]
        None,
        omni_outputs=omni_outputs,
        role="assistant",
        request=SimpleNamespace(return_token_ids=False),
    )

    assert len(choices) == 1
    content = choices[0].message.content
    assert isinstance(content, list)
    assert len(content) == 1
    first_item = content[0]
    assert first_item["type"] == "image_url"
    assert first_item["image_url"]["url"].startswith("data:image/png;base64,")
    assert first_item["stage_durations"] == stage_durations
    assert first_item["peak_memory_mb"] == peak_memory_mb
