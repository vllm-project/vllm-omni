# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Unit tests for patch.py
"""

import asyncio
import base64
import json
import time
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

from vllm_omni.benchmarks.patch.patch import (
    MixRequestFuncOutput,
    _apply_stage0_token_timings,
    async_request_openai_chat_omni_completions,
    async_request_openai_realtime_duplex,
)
from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector

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


@pytest.mark.asyncio
async def test_seed_tts_realtime_duplex_exports_per_request_metrics(monkeypatch):
    class FakeRealtimeClient:
        last_instance = None

        def __init__(self, url):
            assert url == "ws://localhost:8000/v1/realtime?duplex=1"
            self.events = RealtimeEventCollector()
            self.configure_kwargs = None
            self.sent = []
            self.response_count = 0
            self.ack_count = 0
            type(self).last_instance = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def configure(self, model, **kwargs):
            assert model == "openbmb/MiniCPM-o-4_5"
            self.configure_kwargs = kwargs

        async def send(self, event):
            self.sent.append(event)
            if event["type"] != "response.create":
                return
            self.response_count += 1
            response_id = f"resp-{self.response_count}"
            now = time.monotonic()
            self.events.add(
                {"type": "response.created", "response": {"id": response_id}},
                received_at_s=now + 0.01,
            )
            self.events.add(
                {
                    "type": "response.audio_transcript.delta",
                    "response_id": response_id,
                    "delta": f"turn {self.response_count}",
                },
                received_at_s=now + 0.02,
            )
            self.events.add(
                {
                    "type": "response.audio.delta",
                    "response_id": response_id,
                    "delta": base64.b64encode(b"\x00\x00" * 2400).decode(),
                    "sample_rate_hz": 24_000,
                    "metadata": {"audio_duration_ms": 100},
                },
                received_at_s=now + 0.03,
            )
            self.events.add(
                {
                    "type": "response.done",
                    "response": {
                        "id": response_id,
                        "metadata": {
                            "vllm_omni": {
                                "stage_metrics": {
                                    "0": {
                                        "num_tokens_out": 3,
                                        "vllm_ttft_ms": 20.0,
                                        "vllm_tpot_ms": 10.0,
                                        "vllm_itls_ms": [10.0, 10.0],
                                    }
                                }
                            }
                        },
                    },
                },
                received_at_s=now + 0.04,
            )

        async def acknowledge_playback(self):
            self.ack_count += 1

        async def close_session(self, **_kwargs):
            return None

    monkeypatch.setattr(
        "vllm_omni.benchmarks.patch.patch.RealtimeDuplexClient",
        FakeRealtimeClient,
    )
    request_input = RequestFuncInput(
        model="openbmb/MiniCPM-o-4_5",
        model_name="openbmb/MiniCPM-o-4_5",
        prompt="hello",
        api_url="http://localhost:8000/v1/realtime",
        prompt_len=1,
        output_len=20,
        logprobs=None,
        multi_modal_content=None,
        ignore_eos=False,
        extra_body={"save_duplex_request_metrics": True},
    )
    request_input.seed_tts_speech_extra = {"ref_audio": "data:audio/wav;base64,AAAA"}
    request_input.seed_tts_system_prompt = "Speak exactly."
    request_input.seed_tts_turns = tuple(
        SimpleNamespace(utterance_id=f"utt-{index}", target_text=f"text {index}") for index in range(4)
    )

    output = await async_request_openai_realtime_duplex(
        request_input,
        session=None,
    )

    client = FakeRealtimeClient.last_instance
    assert client.configure_kwargs["native_duplex"] is False
    assert client.configure_kwargs["extra_body"] == {
        "ref_audio": "data:audio/wav;base64,AAAA",
        "return_stage_metrics": True,
    }
    assert client.sent == [
        event
        for index in range(4)
        for event in (
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"text {index}"}],
                },
            },
            {"type": "response.create"},
        )
    ]
    assert client.ack_count == 4
    assert output.success is True
    assert output.generated_text == "turn 1 turn 2 turn 3 turn 4"
    assert output.audio_duration == pytest.approx(0.4)
    assert output.ttft > 0
    assert output.audio_ttfp > output.ttft
    assert output.audio_rtf > 0
    assert output.latency > 0
    assert output.output_tokens == 12
    assert output.itl == [0.01, 0.01] * 4
    assert output.text_latency == pytest.approx(output.ttft + 0.08)
    assert output.tpot_measured is True
    assert output.tts_turn_pcm_bytes == [b"\x00\x00" * 2400] * 4
    assert output.tts_output_pcm_bytes == b"\x00\x00" * 9600
    session_id = output.duplex_request_metrics[0]["session_id"]
    assert len(output.duplex_request_metrics) == 4
    for index, metrics in enumerate(output.duplex_request_metrics):
        assert metrics == {
            "session_id": session_id,
            "request_index": index,
            "utterance_id": f"utt-{index}",
            "response_id": f"resp-{index + 1}",
            "source": "client_monotonic_receive",
            "measurement_origin": {
                "ttft": "conversation.item.create client send to first non-empty text delta",
                "ttfp": "conversation.item.create client send to first audio packet",
                "rtf": "request-start-to-last-audio receive time divided by emitted audio duration",
            },
            "ttft_ms": pytest.approx(20.0, abs=2.0),
            "ttfp_ms": pytest.approx(30.0, abs=2.0),
            "rtf": pytest.approx(0.3, abs=0.03),
            "audio_generation_ms": pytest.approx(30.0, abs=2.0),
            "audio_duration_ms": 100.0,
        }
    assert output.duplex_session_metrics == {
        "session_id": session_id,
        "audio_turn_count": 4,
        "mean_ttft_ms": pytest.approx(20.0, abs=2.0),
        "mean_ttfp_ms": pytest.approx(30.0, abs=2.0),
        "mean_rtf": pytest.approx(0.3, abs=0.03),
    }


def test_stage0_token_timings_use_weighted_tpot_when_itls_are_incomplete():
    output = MixRequestFuncOutput(ttft=0.1)

    measured = _apply_stage0_token_timings(
        output,
        [
            {"output_token_count": 3, "itls_ms": [], "tpot_ms": 10.0},
            {"output_token_count": 2, "itls_ms": [], "tpot_ms": 40.0},
        ],
        expected_output_tokens=5,
    )

    assert measured is True
    assert output.itl == []
    assert output.tpot_measured is True
    assert (output.text_latency - output.ttft) / 4 == pytest.approx(0.02)


def test_stage0_zero_itls_fall_back_to_positive_tpot():
    output = MixRequestFuncOutput(ttft=0.1)

    measured = _apply_stage0_token_timings(
        output,
        [{"output_token_count": 3, "itls_ms": [0.0, 0.0], "tpot_ms": 25.0}],
        expected_output_tokens=3,
    )

    assert measured is True
    assert output.itl == []
    assert output.tpot_measured is True
    assert (output.text_latency - output.ttft) / 2 == pytest.approx(0.025)


def test_stage0_token_count_without_timing_is_not_measured():
    output = MixRequestFuncOutput(ttft=0.1)

    measured = _apply_stage0_token_timings(
        output,
        [{"output_token_count": 3, "itls_ms": [], "tpot_ms": None}],
        expected_output_tokens=3,
    )

    assert measured is False
    assert output.itl == []
    assert output.text_latency == output.ttft
    assert output.tpot_measured is False


def create_sse_chunk(data_dict):
    """Helper to create SSE formatted chunk"""
    return f"data: {json.dumps(data_dict)}\n\n".encode()


# ============================================================================
# output_tokens Tests
# ============================================================================


@pytest.mark.asyncio
async def test_output_tokens_assigned_with_metrics(mocker: MockerFixture):
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
    mock_session = mocker.AsyncMock()
    mock_session.post = mocker.MagicMock(return_value=mock_response)

    # Act
    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    # Assert
    assert output.success is True
    assert output.output_tokens == 42, "output_tokens should be assigned from metrics"
    assert output.generated_text == "Hello world"


@pytest.mark.asyncio
async def test_output_tokens_not_assigned_without_metrics(mocker: MockerFixture):
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
    mock_session = mocker.AsyncMock()
    mock_session.post = mocker.MagicMock(return_value=mock_response)

    # Act
    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    # Assert
    assert output.success is True
    # Without server metrics we no longer infer count from SSE chunks because
    # one chunk may carry multiple tokens. output_tokens stays 0 here.
    assert output.output_tokens == 0, "output_tokens should be 0 when no server metrics are present"
    assert output.generated_text == "Hello world"


@pytest.mark.asyncio
async def test_output_tokens_assigned_multiple_metrics(mocker: MockerFixture):
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
    mock_session = mocker.AsyncMock()
    mock_session.post = mocker.MagicMock(return_value=mock_response)

    # Act
    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    # Assert
    assert output.success is True
    assert output.output_tokens == 15, "output_tokens should be updated to the latest value"
    assert output.generated_text == "Hello world!"


@pytest.mark.asyncio
async def test_output_tokens_with_audio_and_text(mocker: MockerFixture):
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
    mock_session = mocker.AsyncMock()
    mock_session.post = mocker.MagicMock(return_value=mock_response)

    # Act
    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    # Assert
    assert output.success is True
    assert output.output_tokens == 25, "output_tokens should be assigned even with audio modality"


@pytest.mark.asyncio
async def test_output_tokens_with_missing_num_tokens_out(mocker: MockerFixture):
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
    mock_session = mocker.AsyncMock()
    mock_session.post = mocker.MagicMock(return_value=mock_response)

    # Act
    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    # Assert
    assert output.success is True
    assert output.output_tokens == 0, "output_tokens should be 0 when num_tokens_out is missing"


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
    async def test_text_latency_assigned_with_text_response(self, mocker: MockerFixture):
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
        mock_session = mocker.AsyncMock()
        mock_session.post = mocker.MagicMock(return_value=mock_response)

        # Act
        output = await async_request_openai_chat_omni_completions(request_input, mock_session)

        # Assert
        assert output.success is True
        assert hasattr(output, "text_latency"), "Output should have text_latency attribute"
        assert output.text_latency > 0, "text_latency should be greater than 0 for text response"

    @pytest.mark.asyncio
    async def test_text_latency_updated_with_multiple_text_chunks(self, mocker: MockerFixture):
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
        mock_session = mocker.AsyncMock()
        mock_session.post = mocker.MagicMock(return_value=mock_response)

        # Act
        output = await async_request_openai_chat_omni_completions(request_input, mock_session)

        # Assert
        assert output.success is True
        assert hasattr(output, "text_latency"), "Output should have text_latency attribute"
        assert output.text_latency > 0, "text_latency should accumulate"
        assert output.generated_text == "First second third"

    @pytest.mark.asyncio
    async def test_text_latency_with_only_audio_response(self, mocker: MockerFixture):
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
        mock_session = mocker.AsyncMock()
        mock_session.post = mocker.MagicMock(return_value=mock_response)

        # Act
        output = await async_request_openai_chat_omni_completions(request_input, mock_session)

        # Assert
        assert output.success is True
        assert hasattr(output, "text_latency"), "Output should have text_latency attribute even with audio-only"
        # text_latency should either be 0 or the initial value since no text was processed
        assert output.text_latency >= 0, "text_latency should be non-negative"

    @pytest.mark.asyncio
    async def test_text_latency_not_affected_by_metrics(self, mocker: MockerFixture):
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
        mock_session = mocker.AsyncMock()
        mock_session.post = mocker.MagicMock(return_value=mock_response)

        # Act
        output = await async_request_openai_chat_omni_completions(request_input, mock_session)

        # Assert
        assert output.success is True
        assert hasattr(output, "text_latency"), "text_latency should exist"
        assert output.text_latency > 0, "text_latency should be set for text response"
        # Verify metrics are also present
        assert output.output_tokens == 100, "metrics should not affect text_latency"

    @pytest.mark.asyncio
    async def test_text_latency_mixed_modalities(self, mocker: MockerFixture):
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
        mock_session = mocker.AsyncMock()
        mock_session.post = mocker.MagicMock(return_value=mock_response)

        # Act
        output = await async_request_openai_chat_omni_completions(request_input, mock_session)

        # Assert
        assert output.success is True
        assert hasattr(output, "text_latency"), "text_latency should exist with mixed modalities"
        assert output.text_latency > 0, "text_latency should be set when text is present"
        assert output.generated_text == "Text more text"

    @pytest.mark.asyncio
    async def test_text_latency_value_consistency(self, mocker: MockerFixture):
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
        mock_session = mocker.AsyncMock()
        mock_session.post = mocker.MagicMock(return_value=mock_response)

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


# ============================================================================
# prompt_len Tests
# ============================================================================


@pytest.mark.asyncio
async def test_skips_empty_role_chunk_for_ttft_and_itl(mocker: MockerFixture):
    """Role/empty text chunks must not count toward TTFT or ITL."""
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
                "choices": [{"delta": {"role": "assistant", "content": ""}}],
                "modality": "text",
            }
        ),
        create_sse_chunk(
            {
                "choices": [{"delta": {"content": "Hello"}}],
                "modality": "text",
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            }
        ),
        create_sse_chunk(
            {
                "choices": [{"delta": {"content": " world"}}],
                "modality": "text",
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        ),
        b"data: [DONE]\n\n",
    ]

    mock_response = MockResponse(200, chunks, delay_between_chunks=0.01)
    mock_session = mocker.AsyncMock()
    mock_session.post = mocker.MagicMock(return_value=mock_response)

    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    assert output.success is True
    assert output.output_tokens == 2
    assert len(output.itl) == 1
    assert output.ttft > 0
    assert output.text_latency - output.ttft == pytest.approx(sum(output.itl), rel=1e-6, abs=1e-6)


@pytest.mark.asyncio
async def test_bundled_tokens_split_itl_from_usage(mocker: MockerFixture):
    """Multiple tokens in one SSE chunk should expand ITL via usage deltas."""
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
                "choices": [{"delta": {"content": "A"}}],
                "modality": "text",
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            }
        ),
        create_sse_chunk(
            {
                "choices": [{"delta": {"content": "BCD"}}],
                "modality": "text",
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            }
        ),
        b"data: [DONE]\n\n",
    ]

    mock_response = MockResponse(200, chunks, delay_between_chunks=0.02)
    mock_session = mocker.AsyncMock()
    mock_session.post = mocker.MagicMock(return_value=mock_response)

    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    assert output.success is True
    assert output.output_tokens == 4
    assert len(output.itl) == 3
    assert output.text_latency - output.ttft == pytest.approx(sum(output.itl), rel=1e-6, abs=1e-6)


@pytest.mark.asyncio
async def test_output_tokens_prefers_usage_over_metrics(mocker: MockerFixture):
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
                "metrics": {"num_tokens_out": 5},
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            }
        ),
        b"data: [DONE]\n\n",
    ]

    mock_response = MockResponse(200, chunks)
    mock_session = mocker.AsyncMock()
    mock_session.post = mocker.MagicMock(return_value=mock_response)

    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    assert output.success is True
    assert output.output_tokens == 8


@pytest.mark.asyncio
async def test_metrics_only_chunk_updates_output_tokens(mocker: MockerFixture):
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
                "choices": [{"delta": {"content": "Text response"}}],
                "modality": "text",
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
    mock_session = mocker.AsyncMock()
    mock_session.post = mocker.MagicMock(return_value=mock_response)

    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    assert output.success is True
    assert output.output_tokens == 25


@pytest.mark.asyncio
async def test_prompt_len_assigned_from_usage(mocker: MockerFixture):
    # Arrange: request claims prompt_len=100, but server reports 4992 (multimodal).
    request_input = RequestFuncInput(
        model="test-model",
        model_name="test-model",
        prompt="test prompt",
        api_url="http://test.com/v1/chat/completions",
        prompt_len=100,
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
        # Final usage chunk emitted because stream_options.include_usage=True.
        create_sse_chunk(
            {
                "choices": [],
                "usage": {"prompt_tokens": 4992, "completion_tokens": 2, "total_tokens": 4994},
            }
        ),
        b"data: [DONE]\n\n",
    ]

    mock_response = MockResponse(200, chunks)
    mock_session = mocker.AsyncMock()
    mock_session.post = mocker.MagicMock(return_value=mock_response)

    # Act
    output = await async_request_openai_chat_omni_completions(request_input, mock_session)

    # Assert
    assert output.success is True
    assert output.prompt_len == 4992, (
        "prompt_len should be overridden by usage.prompt_tokens to reflect the true multimodal input token count"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
