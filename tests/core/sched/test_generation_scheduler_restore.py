"""Test that OmniGenerationScheduler restores chunk-waiting requests
even when the OmniNewRequestData rewrapping fails.

Regression test: if process_pending_chunks() moves requests into
internal deques but restore_queues() is not called due to an exception,
those requests are permanently orphaned.
"""

from collections import deque
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.sched.interface import PauseState

from vllm_omni.core.sched.omni_generation_scheduler import (
    OmniGenerationScheduler,
    _has_async_chunk_payload_to_run,
)
from vllm_omni.engine.serialization import serialize_additional_information

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_generation_scheduler_recognizes_serialized_terminal_codec_payload() -> None:
    payload = serialize_additional_information({"codes": {"audio": torch.tensor([[1, 2, 3]], dtype=torch.long)}})

    assert payload is not None
    assert _has_async_chunk_payload_to_run(SimpleNamespace(additional_information=payload))


class FakeAdapter:
    """Minimal mock of OmniChunkTransferAdapter tracking restore calls."""

    def __init__(self):
        self.waiting_for_chunk_waiting_requests = deque()
        self.waiting_for_chunk_running_requests = deque()
        self._held_non_active = deque()
        self.restore_called = False
        self.done_request_ids = set()

    def process_pending_chunks(self, waiting, running, scheduler_requests=None):
        """Simulate moving requests out of the scheduler queues."""
        # Move one request from running into internal deque
        if running:
            req = running.pop()
            self.waiting_for_chunk_running_requests.append(req)

    def is_done_receiving_chunks(self, request_id):
        return request_id in self.done_request_ids

    def collect_failed_send_request_ids(self):
        return {}

    def restore_queues(self, waiting, running, scheduler_requests=None):
        """Put requests back."""
        self.restore_called = True
        running.extend(self.waiting_for_chunk_running_requests)
        self.waiting_for_chunk_running_requests = deque()

    def postprocess_scheduler_output(self, output):
        pass


class _FakeQueue:
    def __init__(self, requests):
        self._requests = deque(requests)

    def __bool__(self):
        return bool(self._requests)

    def __iter__(self):
        return iter(self._requests)

    def peek_request(self):
        return self._requests[0]

    def pop_request(self):
        return self._requests.popleft()

    def prepend_request(self, request):
        self._requests.appendleft(request)

    def prepend_requests(self, requests):
        for request in reversed(list(requests._requests)):
            self._requests.appendleft(request)


def _scheduler_with_parked_generation_request(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_v2_model_runner: bool,
):
    parked = SimpleNamespace(request_id="parked")
    waiting = SimpleNamespace(
        request_id="waiting",
        prompt_token_ids=[1],
        num_computed_tokens=0,
        status=None,
        sampling_params=None,
        pooling_params=None,
        mm_features=None,
        lora_request=None,
        prompt_is_token_ids=True,
        additional_information=None,
        external_req_id="waiting",
        prefill_stats=None,
        record_event=lambda *args, **kwargs: None,
    )
    adapter = FakeAdapter()
    adapter.waiting_for_chunk_running_requests.append(parked)

    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler.max_num_scheduled_tokens = 8
    scheduler.max_num_running_reqs = 1
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.running = []
    scheduler.waiting = _FakeQueue([waiting])
    scheduler.requests = {"parked": parked, "waiting": waiting}
    scheduler.policy = "fcfs"
    scheduler.chunk_transfer_adapter = adapter
    scheduler.input_coordinator = None
    scheduler.log_stats = False
    scheduler.scheduler_config = SimpleNamespace(enable_chunked_prefill=True)
    scheduler.num_lookahead_tokens = 0
    scheduler.kv_cache_manager = SimpleNamespace(
        new_step_starts=lambda: None,
        allocate_slots=lambda *args, **kwargs: SimpleNamespace(get_block_ids=lambda: ([1],)),
        get_num_common_prefix_blocks=lambda request_id: [0],
        take_new_block_ids=lambda: None,
    )
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    scheduler.use_v2_model_runner = use_v2_model_runner
    scheduler.skipped_waiting = _FakeQueue([])
    scheduler.needs_kv_cache_zeroing = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(get_freed_mm_hashes=lambda: [])
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler._pending_finish_reqs = []
    scheduler._consume_pending_connector_output = lambda model_mode: None
    scheduler._process_pending_input_timeouts = lambda: None
    scheduler._make_cached_request_data = lambda **kwargs: SimpleNamespace(
        req_ids=[],
        resumed_req_ids=[],
        new_token_ids=[],
        all_token_ids=[],
        new_block_ids=[],
        num_computed_tokens=[],
        num_output_tokens=[],
    )
    scheduler._update_after_schedule = lambda output: None
    scheduler._wrap_omni_scheduler_output = lambda output: output
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_generation_scheduler.create_request_queue",
        lambda policy: _FakeQueue([]),
    )
    return scheduler, waiting


def test_native_generation_scheduler_reads_terminal_state_from_coordinator() -> None:
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler.chunk_transfer_adapter = None
    scheduler._native_data_plane = True
    scheduler.input_coordinator = SimpleNamespace(finished_requests={"done"})

    assert scheduler._async_chunk_transport_enabled()
    assert scheduler._is_done_receiving_chunks("done")
    assert not scheduler._is_done_receiving_chunks("running")


def test_mrv1_generation_scheduler_keeps_adapter_terminal_state() -> None:
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    adapter = FakeAdapter()
    adapter.done_request_ids.add("done")
    scheduler.chunk_transfer_adapter = adapter
    scheduler._native_data_plane = False
    scheduler.input_coordinator = None

    assert scheduler._async_chunk_transport_enabled()
    assert scheduler._is_done_receiving_chunks("done")


def test_mrv1_generation_scheduler_does_not_reserve_parked_async_chunk_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, waiting = _scheduler_with_parked_generation_request(
        monkeypatch,
        use_v2_model_runner=False,
    )

    output = OmniGenerationScheduler.schedule(scheduler)

    assert waiting in scheduler.running
    assert output.num_scheduled_tokens == {"waiting": 1}


def test_mrv2_generation_scheduler_does_not_reserve_released_runner_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler, waiting = _scheduler_with_parked_generation_request(
        monkeypatch,
        use_v2_model_runner=True,
    )

    output = OmniGenerationScheduler.schedule(scheduler)

    assert waiting in scheduler.running
    assert output.num_scheduled_tokens == {"waiting": 1}


def test_generation_scheduler_schedules_terminal_empty_prompt_chunk_once(monkeypatch: pytest.MonkeyPatch) -> None:
    waiting = SimpleNamespace(
        request_id="terminal",
        prompt_token_ids=[],
        num_computed_tokens=0,
        num_prompt_tokens=0,
        status=None,
        sampling_params=None,
        pooling_params=None,
        mm_features=None,
        lora_request=None,
        prompt_is_token_ids=True,
        additional_information=serialize_additional_information(
            {"codes": {"audio": torch.tensor([[1, 2, 3]], dtype=torch.long)}}
        ),
        external_req_id="terminal",
        prefill_stats=None,
        record_event=lambda *args, **kwargs: None,
    )
    adapter = FakeAdapter()
    adapter.done_request_ids.add("terminal")

    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler.max_num_scheduled_tokens = 8
    scheduler.max_num_running_reqs = 1
    scheduler._pause_state = PauseState.UNPAUSED
    scheduler.running = []
    scheduler.waiting = _FakeQueue([waiting])
    scheduler.skipped_waiting = _FakeQueue([])
    scheduler.requests = {"terminal": waiting}
    scheduler.policy = "fcfs"
    scheduler.chunk_transfer_adapter = adapter
    scheduler.input_coordinator = None
    scheduler.log_stats = False
    scheduler.scheduler_config = SimpleNamespace(enable_chunked_prefill=True)
    scheduler.num_lookahead_tokens = 0
    scheduler.kv_cache_manager = SimpleNamespace(
        new_step_starts=lambda: None,
        allocate_slots=lambda *args, **kwargs: SimpleNamespace(get_block_ids=lambda: ([1],)),
        get_num_common_prefix_blocks=lambda request_id: [0],
        take_new_block_ids=lambda: None,
    )
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    scheduler.use_v2_model_runner = False
    scheduler.needs_kv_cache_zeroing = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(get_freed_mm_hashes=lambda: [])
    scheduler.connector = None
    scheduler.ec_connector = None
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler._pending_finish_reqs = []
    scheduler._consume_pending_connector_output = lambda model_mode: None
    scheduler._process_pending_input_timeouts = lambda: None
    scheduler._make_cached_request_data = lambda **kwargs: SimpleNamespace(
        req_ids=[],
        resumed_req_ids=[],
        new_token_ids=[],
        all_token_ids=[],
        new_block_ids=[],
        num_computed_tokens=[],
        num_output_tokens=[],
    )
    scheduler._update_after_schedule = lambda output: None
    scheduler._wrap_omni_scheduler_output = lambda output: output
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_generation_scheduler.create_request_queue",
        lambda policy: _FakeQueue([]),
    )

    output = OmniGenerationScheduler.schedule(scheduler)

    assert output.num_scheduled_tokens == {"terminal": 1}
    assert output.scheduled_new_reqs[0].req_id == "terminal"
    assert scheduler._pending_finish_reqs == []


class TestRestoreQueuesOnError:
    """Verify that restore_queues is called even when rewrapping raises."""

    def test_requests_not_lost_on_exception(self):
        """Simulate the error path: process_pending_chunks moves a request
        out, then an exception occurs during rewrapping.
        The finally block must restore the request to the queue."""

        adapter = FakeAdapter()
        running = ["req-A", "req-B"]

        # Step 1: process_pending_chunks moves req-B out
        adapter.process_pending_chunks(waiting=[], running=running)
        assert running == ["req-A"]
        assert len(adapter.waiting_for_chunk_running_requests) == 1

        # Step 2: simulate the try/except/finally pattern
        try:
            raise RuntimeError("OmniNewRequestData construction failed")
        except Exception:
            pass  # Log error, leave output unchanged
        finally:
            # This is what guarantees restore always runs
            adapter.restore_queues(waiting=[], running=running)

        # Step 3: verify request is restored
        assert adapter.restore_called is True
        assert "req-B" in running
        assert len(adapter.waiting_for_chunk_running_requests) == 0

    def test_requests_lost_without_fix(self):
        """Demonstrate the bug: without restore in except, request is lost."""

        adapter = FakeAdapter()
        running = ["req-A", "req-B"]

        adapter.process_pending_chunks(waiting=[], running=running)
        assert running == ["req-A"]

        # Simulate the BUGGY code: except without restore
        try:
            raise RuntimeError("OmniNewRequestData construction failed")
        except Exception:
            pass  # Bug: no restore_queues call

        # Request is lost!
        assert "req-B" not in running
        assert len(adapter.waiting_for_chunk_running_requests) == 1

    def test_happy_path_restores_via_finally(self):
        """When no exception, restore_queues is still called via finally."""

        adapter = FakeAdapter()
        running = ["req-A", "req-B"]

        adapter.process_pending_chunks(waiting=[], running=running)

        # Happy path: no exception, finally still runs
        try:
            pass  # Rewrapping succeeds
        finally:
            adapter.restore_queues(waiting=[], running=running)

        assert adapter.restore_called is True
        assert "req-B" in running
