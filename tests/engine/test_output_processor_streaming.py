"""Unit tests for MultimodalOutputProcessor streaming input handling."""

from collections import deque
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import (
    OutputProcessorOutput,
    RequestOutputCollector,
    StreamingUpdate,
)

from vllm_omni.engine.output_processor import (
    MultimodalOutputProcessor,
    OmniRequestState,
)


def _make_tokenizer():
    """Create a minimal mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    tokenizer.convert_ids_to_tokens.return_value = ["<test>"]
    return tokenizer


def _make_engine_core_request(
    request_id: str = "req-0",
    prompt_token_ids: list[int] | None = None,
    resumable: bool = False,
) -> EngineCoreRequest:
    """Create a minimal EngineCoreRequest."""
    req = MagicMock(spec=EngineCoreRequest)
    req.request_id = request_id
    req.external_req_id = request_id
    req.prompt_token_ids = prompt_token_ids or [1, 2, 3]
    req.prompt_embeds = None
    req.mm_features = None
    req.sampling_params = SamplingParams(max_tokens=10)
    req.pooling_params = None
    req.eos_token_id = 2
    req.arrival_time = 1000.0
    req.lora_request = None
    req.cache_salt = None
    req.data_parallel_rank = None
    req.resumable = resumable
    req.trace_headers = None
    return req


def _make_engine_core_output(
    request_id: str = "req-0",
    new_token_ids: list[int] | None = None,
    finish_reason: FinishReason | None = None,
    pooling_output: torch.Tensor | None = None,
) -> EngineCoreOutput:
    """Create a minimal EngineCoreOutput."""
    eco = EngineCoreOutput(
        request_id=request_id,
        new_token_ids=new_token_ids or [10],
        finish_reason=finish_reason,
    )
    eco.pooling_output = pooling_output
    eco.stop_reason = None
    eco.kv_transfer_params = None
    eco.num_cached_tokens = 0
    eco.routed_experts = None
    return eco


@pytest.fixture
def processor():
    """Create a MultimodalOutputProcessor with a mock tokenizer."""
    tokenizer = _make_tokenizer()
    proc = MultimodalOutputProcessor(
        tokenizer=tokenizer,
        log_stats=False,
        engine_core_output_type=None,
    )
    return proc


class TestAddRequestStreamingSession:
    """Tests for add_request handling of streaming input sessions."""

    def test_add_request_new_creates_state(self, processor):
        """First add_request creates OmniRequestState."""
        request = _make_engine_core_request("req-new", resumable=True)
        processor.add_request(request, prompt="Hello")

        assert "req-new" in processor.request_states
        state = processor.request_states["req-new"]
        assert isinstance(state, OmniRequestState)
        assert state.streaming_input is True
        assert state.input_chunk_queue is not None

    def test_add_request_duplicate_routes_to_streaming_update(self, processor):
        """Second add_request with same ID routes to _update_streaming_request_state."""
        request1 = _make_engine_core_request("req-dup", resumable=True)
        processor.add_request(request1, prompt="Hello")

        state_before = processor.request_states["req-dup"]
        assert isinstance(state_before, OmniRequestState)

        # Second call with same ID should update, not raise
        request2 = _make_engine_core_request("req-dup", resumable=True)
        request2.prompt_token_ids = [1, 2, 3, 4, 5]
        processor.add_request(request2, prompt=" world")

        # State should still exist (not replaced)
        assert "req-dup" in processor.request_states

    def test_add_request_non_resumable_creates_normal_state(self, processor):
        """Non-resumable request creates normal state without streaming fields."""
        request = _make_engine_core_request("req-normal", resumable=False)
        processor.add_request(request, prompt="Hello")

        state = processor.request_states["req-normal"]
        assert isinstance(state, OmniRequestState)
        assert state.streaming_input is False
        assert state.input_chunk_queue is None


class TestFinishRequest:
    """Tests for _finish_request omni-specific cleanup."""

    def test_finish_cleans_mm_state(self, processor):
        """_finish_request clears mm_accumulated and mm_type."""
        request = _make_engine_core_request("req-finish")
        processor.add_request(request, prompt="Hello")

        state = processor.request_states["req-finish"]
        state.mm_accumulated = {"audio": torch.randn(1, 4)}
        state.mm_type = "audio"

        processor._finish_request(state)

        assert state.mm_accumulated is None
        assert state.mm_type is None
        assert "req-finish" not in processor.request_states


class TestProcessOutputsStreaming:
    """Tests for process_outputs handling of streaming input requests."""

    def test_streaming_output_marked_not_finished(self, processor):
        """When streaming_input is True, output.finished should be False."""
        request = _make_engine_core_request("req-stream", resumable=True)
        processor.add_request(request, prompt="Hello")

        state = processor.request_states["req-stream"]
        assert state.streaming_input is True

        eco = _make_engine_core_output(
            "req-stream",
            new_token_ids=[10],
            finish_reason=FinishReason.LENGTH,
        )

        result = processor.process_outputs([eco])
        # Since we have no queue, outputs go to request_outputs list.
        # The streaming request should remain in request_states (not freed).
        assert "req-stream" in processor.request_states

    def test_non_streaming_output_freed(self, processor):
        """Non-streaming finished request should be freed."""
        request = _make_engine_core_request("req-normal", resumable=False)
        processor.add_request(request, prompt="Hello")

        eco = _make_engine_core_output(
            "req-normal",
            new_token_ids=[10],
            finish_reason=FinishReason.LENGTH,
        )

        result = processor.process_outputs([eco])
        assert "req-normal" not in processor.request_states

    def test_streaming_with_queued_update_applies_update(self, processor):
        """When streaming request finishes and has queued update, it's applied."""
        request = _make_engine_core_request("req-queued", resumable=True)
        processor.add_request(request, prompt="Hello")

        state = processor.request_states["req-queued"]
        assert state.streaming_input is True

        update = StreamingUpdate(
            prompt=" world",
            prompt_token_ids=[4, 5, 6],
            arrival_time=2000.0,
        )
        state.input_chunk_queue.append(update)

        eco = _make_engine_core_output(
            "req-queued",
            new_token_ids=[10],
            finish_reason=FinishReason.LENGTH,
        )

        processor.process_outputs([eco])

        # Request should still be in states (streaming continues)
        assert "req-queued" in processor.request_states
        # The update should have been applied
        assert state.is_prefilling is True

    def test_streaming_empty_queue_clears_queue(self, processor):
        """When streaming request finishes and queue is empty, queue is set to None."""
        request = _make_engine_core_request("req-empty-q", resumable=True)
        processor.add_request(request, prompt="Hello")

        state = processor.request_states["req-empty-q"]
        assert state.streaming_input is True
        # Queue is empty (no pending updates)
        assert len(state.input_chunk_queue) == 0

        eco = _make_engine_core_output(
            "req-empty-q",
            new_token_ids=[10],
            finish_reason=FinishReason.LENGTH,
        )

        processor.process_outputs([eco])

        # Queue should now be None (waiting for next streaming input)
        assert state.input_chunk_queue is None
        # Request should still exist
        assert "req-empty-q" in processor.request_states


class TestMultimodalAccumulation:
    """Tests for multimodal tensor accumulation in OmniRequestState."""

    def test_add_multimodal_tensor_single(self):
        """Single tensor addition to empty state."""
        state = OmniRequestState.__new__(OmniRequestState)
        state.mm_type = None
        state.mm_accumulated = None

        tensor = torch.randn(2, 3)
        state.add_multimodal_tensor(tensor, "audio")

        assert state.mm_type == "audio"
        assert "audio" in state.mm_accumulated
        assert torch.equal(state.mm_accumulated["audio"], tensor)

    def test_add_multimodal_tensor_accumulate(self):
        """Multiple tensors accumulated into a list for deferred concat."""
        state = OmniRequestState.__new__(OmniRequestState)
        state.mm_type = None
        state.mm_accumulated = None

        t1 = torch.randn(2, 3)
        t2 = torch.randn(2, 3)
        state.add_multimodal_tensor(t1, "audio")
        state.add_multimodal_tensor(t2, "audio")

        assert isinstance(state.mm_accumulated["audio"], list)
        assert len(state.mm_accumulated["audio"]) == 2

    def test_consolidate_tensors(self):
        """Consolidation concatenates accumulated tensor lists."""
        state = OmniRequestState.__new__(OmniRequestState)
        state.mm_type = "audio"
        state.mm_accumulated = {
            "audio": [torch.randn(1, 4), torch.randn(1, 4)]
        }

        state._consolidate_multimodal_tensors()

        assert isinstance(state.mm_accumulated["audio"], torch.Tensor)
        assert state.mm_accumulated["audio"].shape == (1, 8)  # cat on dim=-1 for audio

    def test_add_multimodal_tensor_none_is_noop(self):
        """Adding None payload should not change state."""
        state = OmniRequestState.__new__(OmniRequestState)
        state.mm_type = None
        state.mm_accumulated = None

        state.add_multimodal_tensor(None, "audio")

        assert state.mm_accumulated is None

    def test_add_multimodal_dict_payload(self):
        """Dict payload is normalized correctly."""
        state = OmniRequestState.__new__(OmniRequestState)
        state.mm_type = None
        state.mm_accumulated = None

        payload = {"model_outputs": torch.randn(2, 4)}
        state.add_multimodal_tensor(payload, "audio")

        # "model_outputs" should be renamed to the mm_type ("audio")
        assert "audio" in state.mm_accumulated
        assert state.mm_accumulated["audio"].shape == (2, 4)
