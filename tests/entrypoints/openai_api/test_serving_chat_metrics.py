# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OmniChatCompletionResponse/StreamResponse metrics field."""

from types import SimpleNamespace

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
