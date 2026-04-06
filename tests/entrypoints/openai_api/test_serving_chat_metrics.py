# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OmniChatCompletionResponse/StreamResponse metrics field."""

import pytest

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
