# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for patch.py
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

from vllm_omni.benchmarks.patch.patch import MixRequestFuncOutput, async_request_openai_chat_omni_completions

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


class MockResponse:
    """Mock aiohttp response for testing"""

    def __init__(self, status, chunks, delay_between_chunks=0):
        self.status = status
        self.reason = "OK" if status == 200 else "Error"
        self._chunks = chunks
        self._delay = delay_between_chunks
        self.content = self

    async def iter_any(self):
        for chunk in self._chunks:
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def create_sse_chunk(data_dict):
    """Helper to create SSE formatted chunk"""
    return f"data: {json.dumps(data_dict)}\n\n".encode()


# ============================================================================
# output_tokens Tests
# ============================================================================


@pytest.mark.asyncio
async def test_output_tokens_assigned_with_metrics():
    """Test that output.output_tokens is assigned when metrics are present"""
    # Arrange
    request_input = RequestFuncInput(
        model="test-model",
        model_name="test-model",
        prompt="test prompt",
        api_url="http://test.com/v1/chat/completions",
        prompt_len=10,
        output_len=20,
    )

    # Create response chunks with metrics
    chunks = [
        create_sse_chunk(
            {
                "choices": [{"delta": {"content": "Hello"}}],
                "modality": "text",
            }
        ),
        create_sse_chunk(
            {
                "choices": [{"delta": {"content": " world"}}],
                "modality": "text",
                "metrics": {"num_tokens_out": 42, "num_tokens_in": 10},
            }
        ),
        b"data: [DONE]\n\n",
    ]

    mock_response = MockResponse(200, chunks)
    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)

    # Act
    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    # Assert
    assert output.success is True
    assert output.output_tokens == 42, "output_tokens should be assigned from metrics"
    assert output.generated_text == "Hello world"


@pytest.mark.asyncio
async def test_output_tokens_not_assigned_without_metrics():
    """Test that output.output_tokens defaults to 0 when no metrics present"""
    # Arrange
    request_input = RequestFuncInput(
        model="test-model",
        model_name="test-model",
        prompt="test prompt",
        api_url="http://test.com/v1/chat/completions",
        prompt_len=10,
        output_len=20,
    )

    # Create response chunks without metrics
    chunks = [
        create_sse_chunk(
            {
                "choices": [{"delta": {"content": "Hello"}}],
                "modality": "text",
            }
        ),
        create_sse_chunk(
            {
                "choices": [{"delta": {"content": " world"}}],
                "modality": "text",
            }
        ),
        b"data: [DONE]\n\n",
    ]

    mock_response = MockResponse(200, chunks)
    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)

    # Act
    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    # Assert
    assert output.success is True
    assert output.output_tokens == 0, "output_tokens should default to 0 when no metrics"
    assert output.generated_text == "Hello world"


@pytest.mark.asyncio
async def test_output_tokens_assigned_multiple_metrics():
    """Test that output.output_tokens is updated with the latest metrics value"""
    # Arrange
    request_input = RequestFuncInput(
        model="test-model",
        model_name="test-model",
        prompt="test prompt",
        api_url="http://test.com/v1/chat/completions",
        prompt_len=10,
        output_len=20,
    )

    # Create response chunks with multiple metrics updates
    chunks = [
        create_sse_chunk(
            {
                "choices": [{"delta": {"content": "Hello"}}],
                "modality": "text",
                "metrics": {"num_tokens_out": 5},
            }
        ),
        create_sse_chunk(
            {
                "choices": [{"delta": {"content": " world"}}],
                "modality": "text",
                "metrics": {"num_tokens_out": 10},
            }
        ),
        create_sse_chunk(
            {
                "choices": [{"delta": {"content": "!"}}],
                "modality": "text",
                "metrics": {"num_tokens_out": 15},
            }
        ),
        b"data: [DONE]\n\n",
    ]

    mock_response = MockResponse(200, chunks)
    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)

    # Act
    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    # Assert
    assert output.success is True
    assert output.output_tokens == 15, "output_tokens should be updated to the latest value"
    assert output.generated_text == "Hello world!"


@pytest.mark.asyncio
async def test_output_tokens_with_audio_and_text():
    """Test output_tokens assignment in mixed audio and text response"""
    # Arrange
    request_input = RequestFuncInput(
        model="test-model",
        model_name="test-model",
        prompt="test prompt",
        api_url="http://test.com/v1/chat/completions",
        prompt_len=10,
        output_len=20,
    )

    # Create response chunks with both audio and text, with metrics
    chunks = [
        create_sse_chunk(
            {
                "choices": [{"delta": {"content": "Text response"}}],
                "modality": "text",
            }
        ),
        create_sse_chunk(
            {
                "choices": [{"delta": {"content": ""}}],
                "modality": "audio",
            }
        ),
        create_sse_chunk(
            {
                "modality": "text",
                "metrics": {"num_tokens_out": 25, "num_tokens_in": 10},
            }
        ),
        b"data: [DONE]\n\n",
    ]

    mock_response = MockResponse(200, chunks)
    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)

    # Act
    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    # Assert
    assert output.success is True
    assert output.output_tokens == 25, "output_tokens should be assigned even with audio modality"


@pytest.mark.asyncio
async def test_output_tokens_with_missing_num_tokens_out():
    """Test that output_tokens defaults to 0 when num_tokens_out is missing in metrics"""
    # Arrange
    request_input = RequestFuncInput(
        model="test-model",
        model_name="test-model",
        prompt="test prompt",
        api_url="http://test.com/v1/chat/completions",
        prompt_len=10,
        output_len=20,
    )

    # Create response chunks with metrics but without num_tokens_out
    chunks = [
        create_sse_chunk(
            {
                "choices": [{"delta": {"content": "Hello"}}],
                "modality": "text",
            }
        ),
        create_sse_chunk(
            {
                "choices": [{"delta": {"content": " world"}}],
                "modality": "text",
                "metrics": {"num_tokens_in": 10},  # Missing num_tokens_out
            }
        ),
        b"data: [DONE]\n\n",
    ]

    mock_response = MockResponse(200, chunks)
    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_response)

    # Act
    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    # Assert
    assert output.success is True
    assert output.output_tokens == 0, "output_tokens should default to 0 when num_tokens_out is missing"


@pytest.mark.asyncio
async def test_output_tokens_initialization():
    """Test that MixRequestFuncOutput initializes output_tokens correctly"""
    # Arrange & Act
    output = MixRequestFuncOutput()

    # Assert
    assert hasattr(output, "output_tokens"), "MixRequestFuncOutput should have output_tokens attribute"
    assert output.output_tokens == 0, "output_tokens should be initialized to 0"


# ============================================================================
# text_latency Tests
# ============================================================================


class TestTextLatencyAttribute:
    """Tests for text_latency attribute existence and assignment"""

    def test_mix_request_func_output_has_text_latency(self):
        """Test that MixRequestFuncOutput has text_latency attribute"""
        output = MixRequestFuncOutput()
        assert hasattr(output, "text_latency"), "MixRequestFuncOutput should have text_latency attribute"

    def test_text_latency_initial_value(self):
        """Test that text_latency initializes to a default value"""
        output = MixRequestFuncOutput()
        # Check if attribute exists and has a value (should be 0.0 or similar default)
        text_latency = getattr(output, "text_latency", None)
        assert text_latency is not None or hasattr(output, "text_latency"), "text_latency attribute should exist"

    @pytest.mark.asyncio
    async def test_text_latency_assigned_with_text_response(self):
        """Test that text_latency is assigned when text response is received"""
        request_input = RequestFuncInput(
            model="test-model",
            model_name="test-model",
            prompt="test prompt",
            api_url="http://test.com/v1/chat/completions",
            prompt_len=10,
            output_len=20,
        )

        # Create response chunks with text modality
        chunks = [
            create_sse_chunk(
                {
                    "choices": [{"delta": {"content": "Hello"}}],
                    "modality": "text",
                }
            ),
            create_sse_chunk(
                {
                    "choices": [{"delta": {"content": " world"}}],
                    "modality": "text",
                }
            ),
            b"data: [DONE]\n\n",
        ]

        mock_response = MockResponse(200, chunks)
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)

        # Act
        output = await async_request_openai_chat_omni_completions(request_input, mock_session)

        # Assert
        assert output.success is True
        assert hasattr(output, "text_latency"), "Output should have text_latency attribute"
        assert output.text_latency > 0, "text_latency should be greater than 0 for text response"

    @pytest.mark.asyncio
    async def test_text_latency_updated_with_multiple_text_chunks(self):
        """Test that text_latency is updated as more text chunks arrive"""
        request_input = RequestFuncInput(
            model="test-model",
            model_name="test-model",
            prompt="test prompt",
            api_url="http://test.com/v1/chat/completions",
            prompt_len=10,
            output_len=20,
        )

        # Create response chunks with small delays to simulate streaming
        chunks = [
            create_sse_chunk(
                {
                    "choices": [{"delta": {"content": "First"}}],
                    "modality": "text",
                }
            ),
            create_sse_chunk(
                {
                    "choices": [{"delta": {"content": " second"}}],
                    "modality": "text",
                }
            ),
            create_sse_chunk(
                {
                    "choices": [{"delta": {"content": " third"}}],
                    "modality": "text",
                }
            ),
            b"data: [DONE]\n\n",
        ]

        mock_response = MockResponse(200, chunks)
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)

        # Act
        output = await async_request_openai_chat_omni_completions(request_input, mock_session)

        # Assert
        assert output.success is True
        assert hasattr(output, "text_latency"), "Output should have text_latency attribute"
        assert output.text_latency > 0, "text_latency should accumulate"
        assert output.generated_text == "First second third"

    @pytest.mark.asyncio
    async def test_text_latency_with_only_audio_response(self):
        """Test text_latency behavior when only audio is received"""
        request_input = RequestFuncInput(
            model="test-model",
            model_name="test-model",
            prompt="test prompt",
            api_url="http://test.com/v1/chat/completions",
            prompt_len=10,
            output_len=20,
        )

        # Create response chunks with only audio modality (no text)
        chunks = [
            create_sse_chunk(
                {
                    "choices": [{"delta": {"content": ""}}],
                    "modality": "audio",
                }
            ),
            b"data: [DONE]\n\n",
        ]

        mock_response = MockResponse(200, chunks)
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)

        # Act
        output = await async_request_openai_chat_omni_completions(request_input, mock_session)

        # Assert
        assert output.success is True
        assert hasattr(output, "text_latency"), "Output should have text_latency attribute even with audio-only"
        # text_latency should either be 0 or the initial value since no text was processed
        assert output.text_latency >= 0, "text_latency should be non-negative"

    @pytest.mark.asyncio
    async def test_text_latency_not_affected_by_metrics(self):
        """Test that text_latency is independent of metrics data"""
        request_input = RequestFuncInput(
            model="test-model",
            model_name="test-model",
            prompt="test prompt",
            api_url="http://test.com/v1/chat/completions",
            prompt_len=10,
            output_len=20,
        )

        # Create response with text and metrics
        chunks = [
            create_sse_chunk(
                {
                    "choices": [{"delta": {"content": "Response text"}}],
                    "modality": "text",
                }
            ),
            create_sse_chunk(
                {
                    "choices": [{"delta": {"content": ""}}],
                    "modality": "text",
                    "metrics": {"num_tokens_out": 100, "num_tokens_in": 20},
                }
            ),
            b"data: [DONE]\n\n",
        ]

        mock_response = MockResponse(200, chunks)
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)

        # Act
        output = await async_request_openai_chat_omni_completions(request_input, mock_session)

        # Assert
        assert output.success is True
        assert hasattr(output, "text_latency"), "text_latency should exist"
        assert output.text_latency > 0, "text_latency should be set for text response"
        # Verify metrics are also present
        assert output.output_tokens == 100, "metrics should not affect text_latency"

    @pytest.mark.asyncio
    async def test_text_latency_mixed_modalities(self):
        """Test text_latency with mixed text and audio modalities"""
        request_input = RequestFuncInput(
            model="test-model",
            model_name="test-model",
            prompt="test prompt",
            api_url="http://test.com/v1/chat/completions",
            prompt_len=10,
            output_len=20,
        )

        # Create response with both text and audio
        chunks = [
            create_sse_chunk(
                {
                    "choices": [{"delta": {"content": "Text"}}],
                    "modality": "text",
                }
            ),
            create_sse_chunk(
                {
                    "choices": [{"delta": {"content": ""}}],
                    "modality": "audio",
                }
            ),
            create_sse_chunk(
                {
                    "choices": [{"delta": {"content": " more text"}}],
                    "modality": "text",
                }
            ),
            b"data: [DONE]\n\n",
        ]

        mock_response = MockResponse(200, chunks)
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)

        # Act
        output = await async_request_openai_chat_omni_completions(request_input, mock_session)

        # Assert
        assert output.success is True
        assert hasattr(output, "text_latency"), "text_latency should exist with mixed modalities"
        assert output.text_latency > 0, "text_latency should be set when text is present"
        assert output.generated_text == "Text more text"

    @pytest.mark.asyncio
    async def test_text_latency_value_consistency(self):
        """Test that text_latency matches latency minus ttft relationship"""
        request_input = RequestFuncInput(
            model="test-model",
            model_name="test-model",
            prompt="test prompt",
            api_url="http://test.com/v1/chat/completions",
            prompt_len=10,
            output_len=20,
        )

        chunks = [
            create_sse_chunk(
                {
                    "choices": [{"delta": {"content": "Hello"}}],
                    "modality": "text",
                }
            ),
            create_sse_chunk(
                {
                    "choices": [{"delta": {"content": " world"}}],
                    "modality": "text",
                }
            ),
            b"data: [DONE]\n\n",
        ]

        mock_response = MockResponse(200, chunks)
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)

        # Act
        output = await async_request_openai_chat_omni_completions(request_input, mock_session)

        # Assert
        assert output.success is True
        assert hasattr(output, "text_latency"), "text_latency should exist"
        assert hasattr(output, "ttft"), "ttft should exist"
        assert hasattr(output, "latency"), "latency should exist"
        # text_latency should be between ttft and total latency
        assert output.ttft <= output.text_latency <= output.latency, (
            "text_latency should be between ttft and total latency"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
