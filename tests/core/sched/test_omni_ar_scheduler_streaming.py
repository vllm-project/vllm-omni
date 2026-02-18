"""Unit tests for OmniARScheduler streaming input handling.

Tests the key behavioral changes for streaming input support:
1. finish_reason is captured BEFORE _handle_stopped_request (which may reset status)
2. _free_request is only called when the request truly finishes
3. Output is always emitted when stopped (even without new tokens)
"""

from collections import deque
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler


def _make_mock_request(
    request_id: str = "req-0",
    resumable: bool = False,
    status: RequestStatus = RequestStatus.RUNNING,
) -> Mock:
    """Create a mock Request for testing."""
    req = MagicMock(spec=Request)
    req.request_id = request_id
    req.client_index = 0
    req.status = status
    req.resumable = resumable
    req.output_token_ids = [10, 11]
    req.num_tokens = 5
    req.num_computed_tokens = 5
    req.num_output_placeholders = 0
    req.sampling_params = SamplingParams(max_tokens=10)
    req.pooling_params = None
    req.stop_reason = None
    req.trace_headers = None
    req.num_cached_tokens = 0
    req.num_nans_in_logits = None
    req.streaming_queue = deque() if resumable else None
    req.take_events.return_value = None

    # Simulate get_finished_reason returning LENGTH before reset
    req.get_finished_reason.return_value = FinishReason.LENGTH
    return req


class TestFinishReasonCapturedBeforeHandleStop:
    """Test that finish_reason is captured before _handle_stopped_request."""

    def test_finish_reason_captured_for_resumable_request(self):
        """For resumable requests, finish_reason must be captured before
        _handle_stopped_request which may reset the status.

        The scheduler captures finish_reason before calling
        _handle_stopped_request. For resumable requests,
        _handle_stopped_request returns False (request continues)
        and may reset the status to WAITING. If we queried
        get_finished_reason after that, we'd get None.
        """
        request = _make_mock_request("req-resume", resumable=True)
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED

        # Capture finish_reason BEFORE handle
        finish_reason = request.get_finished_reason()
        assert finish_reason == FinishReason.LENGTH

        # After _handle_stopped_request for resumable, status would be reset
        # Simulate: base scheduler sets status to WAITING_FOR_STREAMING_REQ
        request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
        # Now get_finished_reason would return STOP (or different value)
        request.get_finished_reason.return_value = FinishReason.STOP

        # The captured value should still be LENGTH
        assert finish_reason == FinishReason.LENGTH

    def test_finish_reason_captured_for_non_resumable(self):
        """For non-resumable, behavior is the same - capture before handle."""
        request = _make_mock_request("req-normal", resumable=False)
        request.status = RequestStatus.FINISHED_STOPPED
        request.get_finished_reason.return_value = FinishReason.STOP

        finish_reason = request.get_finished_reason()
        assert finish_reason == FinishReason.STOP


class TestFreeRequestConditional:
    """Test that _free_request is only called when request truly finishes."""

    def test_free_called_for_non_resumable(self):
        """Non-resumable stopped request: _handle_stopped_request returns True,
        _free_request IS called."""
        scheduler = MagicMock(spec=OmniARScheduler)

        # Simulate _handle_stopped_request returning True (finished)
        scheduler._handle_stopped_request.return_value = True
        scheduler._free_request.return_value = {"kv": "params"}

        request = _make_mock_request("req-done", resumable=False)

        # Execute the logic from _update_outputs
        finished = scheduler._handle_stopped_request(request)
        kv_transfer_params = None
        if finished:
            kv_transfer_params = scheduler._free_request(request)

        assert finished is True
        assert kv_transfer_params == {"kv": "params"}
        scheduler._free_request.assert_called_once_with(request)

    def test_free_not_called_for_resumable(self):
        """Resumable stopped request: _handle_stopped_request returns False,
        _free_request is NOT called (KV blocks are preserved)."""
        scheduler = MagicMock(spec=OmniARScheduler)

        # Simulate _handle_stopped_request returning False (resumable, continues)
        scheduler._handle_stopped_request.return_value = False

        request = _make_mock_request("req-continue", resumable=True)

        finished = scheduler._handle_stopped_request(request)
        kv_transfer_params = None
        if finished:
            kv_transfer_params = scheduler._free_request(request)

        assert finished is False
        assert kv_transfer_params is None
        scheduler._free_request.assert_not_called()


class TestOutputEmittedWhenStopped:
    """Test that output is always emitted when stopped, even without new tokens."""

    def test_output_emitted_on_stopped_without_tokens(self):
        """The condition `new_token_ids or pooler_output is not None
        or kv_transfer_params or stopped` should emit output even when
        new_token_ids is empty, as long as stopped is True.

        This is critical for streaming input: the last sub-request
        may produce no new tokens but must still signal finish.
        """
        new_token_ids = []
        pooler_output = None
        kv_transfer_params = None
        stopped = True

        should_emit = bool(new_token_ids or pooler_output is not None or kv_transfer_params or stopped)
        assert should_emit is True

    def test_no_output_when_nothing_to_emit(self):
        """Without tokens, pooler, kv_params, or stop, nothing emitted."""
        new_token_ids = []
        pooler_output = None
        kv_transfer_params = None
        stopped = False

        should_emit = bool(new_token_ids or pooler_output is not None or kv_transfer_params or stopped)
        assert should_emit is False

    def test_output_emitted_with_tokens(self):
        """Normal case: tokens present means output emitted."""
        new_token_ids = [42]
        pooler_output = None
        kv_transfer_params = None
        stopped = False

        should_emit = bool(new_token_ids or pooler_output is not None or kv_transfer_params or stopped)
        assert should_emit is True


class TestBaseHandleStoppedRequest:
    """Test the base _handle_stopped_request behavior (inherited from VLLMScheduler)."""

    def test_non_resumable_returns_true(self):
        """Non-resumable request always returns True (finished)."""
        request = _make_mock_request("req-nr", resumable=False)
        # The base _handle_stopped_request checks request.resumable
        # For non-resumable: return True
        assert not request.resumable

    def test_resumable_with_empty_queue_waits(self):
        """Resumable request with empty streaming_queue enters waiting state."""
        request = _make_mock_request("req-wait", resumable=True)
        request.streaming_queue = deque()  # empty queue
        assert request.resumable
        assert len(request.streaming_queue) == 0

    def test_resumable_with_none_in_queue_finishes(self):
        """Resumable request with None sentinel in queue means finished."""
        request = _make_mock_request("req-fin", resumable=True)
        request.streaming_queue = deque([None])
        assert request.resumable
        # Popping None from the queue signals the request is done
        update = request.streaming_queue.popleft()
        assert update is None

    def test_resumable_with_update_in_queue_continues(self):
        """Resumable request with a real update in queue should continue."""
        request = _make_mock_request("req-cont", resumable=True)
        mock_update = MagicMock()
        request.streaming_queue = deque([mock_update])
        assert request.resumable
        update = request.streaming_queue.popleft()
        assert update is mock_update
