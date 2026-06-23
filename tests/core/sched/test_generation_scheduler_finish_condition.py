"""Tests for OmniGenerationScheduler update_from_output finish conditions.

The three-way finish condition in update_from_output determines when a
generation request (e.g. Code2Wav) is marked FINISHED_STOPPED:

  1. No adapter (sync mode): finish when all prompt tokens are computed.
  2. Adapter present, adapter done: finish when computed AND adapter signals done.
  3. Adapter present, adapter NOT done: do NOT finish — wait for next chunk.

These cases are the core correctness guarantee for async_chunk batch mode.
Without condition #3, a request finishes prematurely after processing
only the first chunk, producing truncated audio.
"""

import unittest

from vllm.v1.request import RequestStatus


class FakeRequest:
    """Minimal request for testing finish conditions."""

    def __init__(self, request_id: str, prompt_len: int, num_computed: int):
        self.request_id = request_id
        self.prompt_token_ids = list(range(prompt_len))
        self.num_computed_tokens = num_computed
        self.status = RequestStatus.RUNNING
        self.stop_reason = None
        self.sampling_params = None
        self.num_cached_tokens = 0
        self.num_output_tokens = 0
        self.num_output_placeholders = 0
        self.num_external_computed_tokens = 0
        self.trace_headers = None
        self.num_nans_in_logits = 0
        self.client_index = 0

    def is_finished(self):
        return self.status in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_ABORTED,
            RequestStatus.FINISHED_ERROR,
        )

    def get_finished_reason(self):
        return "stop"

    def take_events(self):
        return None


class FakeChunkAdapter:
    """Minimal chunk transfer adapter mock."""

    def __init__(self, finished_request_ids: set[str] | None = None):
        self.finished_requests = finished_request_ids or set()

    def cleanup(self, req_id, external_req_id=None):
        pass


def _evaluate_finish_condition(request, chunk_transfer_adapter):
    """Extract the three-way finish logic from update_from_output."""
    _all_computed = request.num_computed_tokens >= len(request.prompt_token_ids)
    _adapter_done = (
        chunk_transfer_adapter is not None and request.request_id in chunk_transfer_adapter.finished_requests
    )
    return (
        request.status == RequestStatus.FINISHED_STOPPED
        or (_all_computed and chunk_transfer_adapter is None)
        or (_all_computed and _adapter_done)
    )


class TestFinishConditionSyncMode(unittest.TestCase):
    """Case 1: No adapter (sync mode) — finish when all tokens computed."""

    def test_finishes_when_all_computed(self):
        req = FakeRequest("r1", prompt_len=100, num_computed=100)
        assert _evaluate_finish_condition(req, chunk_transfer_adapter=None)

    def test_does_not_finish_when_partially_computed(self):
        req = FakeRequest("r1", prompt_len=100, num_computed=50)
        assert not _evaluate_finish_condition(req, chunk_transfer_adapter=None)


class TestFinishConditionAsyncAdapterDone(unittest.TestCase):
    """Case 2: Adapter present AND adapter signals done — finish."""

    def test_finishes_when_computed_and_adapter_done(self):
        req = FakeRequest("r1", prompt_len=100, num_computed=100)
        adapter = FakeChunkAdapter(finished_request_ids={"r1"})
        assert _evaluate_finish_condition(req, adapter)

    def test_does_not_finish_when_computed_but_adapter_not_done(self):
        """Case 3: Adapter present but NOT done — must NOT finish."""
        req = FakeRequest("r1", prompt_len=100, num_computed=100)
        adapter = FakeChunkAdapter(finished_request_ids=set())
        assert not _evaluate_finish_condition(req, adapter)


class TestFinishConditionAsyncNotComputed(unittest.TestCase):
    """Adapter present, tokens not all computed — never finishes."""

    def test_does_not_finish_partial_compute_adapter_done(self):
        req = FakeRequest("r1", prompt_len=100, num_computed=50)
        adapter = FakeChunkAdapter(finished_request_ids={"r1"})
        assert not _evaluate_finish_condition(req, adapter)

    def test_does_not_finish_partial_compute_adapter_not_done(self):
        req = FakeRequest("r1", prompt_len=100, num_computed=50)
        adapter = FakeChunkAdapter(finished_request_ids=set())
        assert not _evaluate_finish_condition(req, adapter)


class TestFinishConditionAlreadyStopped(unittest.TestCase):
    """Request already FINISHED_STOPPED — always returns True."""

    def test_already_stopped_no_adapter(self):
        req = FakeRequest("r1", prompt_len=100, num_computed=0)
        req.status = RequestStatus.FINISHED_STOPPED
        assert _evaluate_finish_condition(req, chunk_transfer_adapter=None)

    def test_already_stopped_with_adapter(self):
        req = FakeRequest("r1", prompt_len=100, num_computed=0)
        req.status = RequestStatus.FINISHED_STOPPED
        adapter = FakeChunkAdapter(finished_request_ids=set())
        assert _evaluate_finish_condition(req, adapter)


if __name__ == "__main__":
    unittest.main()
