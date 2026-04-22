# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniSchedulingCoordinator (formerly ChunkSchedulingCoordinator).

These tests use mock request objects and mock queues.  They do not require
GPU, vLLM runtime, or any connector.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import vllm_omni.core.sched.omni_scheduling_coordinator as coord_mod
from vllm_omni.core.sched.omni_scheduling_coordinator import (
    ChunkSchedulingCoordinator,
    OmniSchedulingCoordinator,
)

# ------------------------------------------------------------------ #
#  Mock helpers
# ------------------------------------------------------------------ #


class _RequestStatus:
    WAITING = "waiting"
    RUNNING = "running"
    WAITING_FOR_CHUNK = "waiting_for_chunk"
    WAITING_FOR_INPUT = "waiting_for_input"
    FINISHED_STOPPED = "finished_stopped"


# Patch RequestStatus for tests that don't import vllm
try:
    from vllm.v1.request import RequestStatus
except ImportError:
    RequestStatus = _RequestStatus  # type: ignore[misc,assignment]

if not hasattr(RequestStatus, "WAITING_FOR_INPUT"):
    coord_mod.RequestStatus = _RequestStatus  # type: ignore[assignment]
    RequestStatus = _RequestStatus  # type: ignore[misc,assignment]


def _make_request(req_id: str, status: str = "waiting") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=req_id,
        external_req_id=req_id,
        status=status,
        additional_information=None,
        prompt_token_ids=[],
        num_prompt_tokens=0,
        num_computed_tokens=0,
        _all_token_ids=[],
        _output_token_ids=[],
    )


class MockQueue:
    """Simplified queue that mimics the Scheduler waiting queue interface."""

    def __init__(self, items: list | None = None):
        self._items: list = list(items or [])

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __contains__(self, item):
        return item in self._items

    def add_request(self, request):
        self._items.append(request)

    def prepend_requests(self, requests):
        self._items = list(requests) + self._items

    def remove(self, request):
        self._items.remove(request)

    def remove_requests(self, requests):
        remove_set = set(id(r) for r in requests)
        self._items = [r for r in self._items if id(r) not in remove_set]


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #


class TestChunkCoordinatorStateTransition(unittest.TestCase):
    """Test 5: process_pending_chunks transitions WAITING_FOR_CHUNK → target."""

    def test_ready_request_transitions_to_waiting(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r1"},
            chunk_finished_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING)
        self.assertIn("r1", coord.requests_with_ready_chunks)

    def test_non_ready_stays_waiting_for_chunk(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids=set(),
            chunk_finished_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)

    def test_stage_0_is_noop(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=0)
        req = _make_request("r1")
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r1"},
            chunk_finished_req_ids=set(),
        )
        self.assertNotEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)


class TestChunkCoordinatorRestoreQueues(unittest.TestCase):
    """Test 6: restore_queues returns waiting-for-chunk requests."""

    def test_restore(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        r1 = _make_request("r1")
        r2 = _make_request("r2")
        coord._waiting_for_chunk_waiting.append(r1)
        coord._waiting_for_chunk_running.append(r2)

        waiting = MockQueue()
        running: list = []

        coord.restore_queues(waiting, running)

        self.assertIn(r1, waiting)
        self.assertIn(r2, running)
        self.assertEqual(len(coord._waiting_for_chunk_waiting), 0)
        self.assertEqual(len(coord._waiting_for_chunk_running), 0)


class TestChunkCoordinatorFinishedSignal(unittest.TestCase):
    """Test 8: chunk_finished_req_ids → finished_requests."""

    def test_finished_signal(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r1"},
            chunk_finished_req_ids={"r1"},
        )

        self.assertIn("r1", coord.finished_requests)


class TestChunkCoordinatorUpdateRequestMetadata(unittest.TestCase):
    """Test update_request_metadata applies scheduling metadata to requests."""

    def test_ar_mode_no_longer_sets_additional_information(self):
        """AR mode only processes scheduling metadata, not full payloads."""
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1")
        requests = {"r1": req}

        # Only scheduling metadata is passed now (full payload stays in model runner)
        request_metadata = {"r1": {"next_stage_prompt_len": 50}}

        coord.update_request_metadata(requests, request_metadata, model_mode="ar")

        # next_stage_prompt_len should update prompt_token_ids
        self.assertEqual(len(req.prompt_token_ids), 50)
        self.assertEqual(req.num_prompt_tokens, 50)
        # additional_information should NOT be set
        self.assertIsNone(getattr(req, "additional_information", None))

    def test_generation_mode(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1")
        req.prompt_token_ids = [0, 0, 0]
        requests = {"r1": req}

        request_metadata = {
            "r1": {
                "code_predictor_codes": [10, 20, 30],
                "left_context_size": 25,
            }
        }

        coord.update_request_metadata(requests, request_metadata, model_mode="generation")

        self.assertEqual(req.prompt_token_ids, [10, 20, 30])
        self.assertEqual(req.num_computed_tokens, 0)
        self.assertIsNone(req.additional_information)
        self.assertEqual(req._omni_initial_model_buffer, {"left_context_size": 25})


class TestChunkCoordinatorPostprocess(unittest.TestCase):
    """Test postprocess_scheduler_output clears ready chunks."""

    def test_clear_ready(self):
        coord = ChunkSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)
        coord.requests_with_ready_chunks = {"r1", "r2"}

        new_req = SimpleNamespace(req_id="r1")
        cached_reqs = SimpleNamespace(req_ids=["r2"])
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[new_req],
            scheduled_cached_reqs=cached_reqs,
        )

        coord.postprocess_scheduler_output(scheduler_output)

        self.assertEqual(coord.requests_with_ready_chunks, set())


class TestWaitingForInputTransition(unittest.TestCase):
    """Test B8: process_pending_full_payload_inputs transitions WAITING_FOR_INPUT."""

    def test_transition_on_recv(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_full_payload_inputs(
            waiting,
            running,
            stage_recv_req_ids={"r1"},
        )

        self.assertEqual(req.status, RequestStatus.WAITING)

    def test_stays_waiting_for_input_if_not_received(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_full_payload_inputs(
            waiting,
            running,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_INPUT)
        self.assertEqual(len(coord._waiting_for_input), 1)

    def test_stage_0_is_noop(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=0)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_full_payload_inputs(
            waiting,
            running,
            stage_recv_req_ids={"r1"},
        )
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_INPUT)

    def test_restore_queues_includes_waiting_for_input(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        r1 = _make_request("r1")
        coord._waiting_for_input.append(r1)

        waiting = MockQueue()
        running: list = []

        coord.restore_queues(waiting, running)

        self.assertIn(r1, waiting)
        self.assertEqual(len(coord._waiting_for_input), 0)

    def test_full_payload_mode_auto_transitions_waiting_to_waiting_for_input(self):
        """In full_payload_mode (async_chunk=False), fresh WAITING requests on
        non-Stage-0 should be transitioned to WAITING_FOR_INPUT."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=False,
        )

        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_full_payload_inputs(
            waiting,
            running,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_INPUT)
        self.assertEqual(len(coord._waiting_for_input), 1)
        self.assertEqual(len(coord.pending_input_registrations), 1)

    def test_async_chunk_mode_does_not_auto_transition(self):
        """In async_chunk mode, fresh WAITING requests should NOT be
        transitioned to WAITING_FOR_INPUT."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=True,
        )

        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_full_payload_inputs(
            waiting,
            running,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING)

    def test_pending_input_registrations(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_full_payload_inputs(
            waiting,
            running,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(len(coord.pending_input_registrations), 1)
        self.assertEqual(coord.pending_input_registrations[0].request_id, "r1")


class TestTimeoutDetection(unittest.TestCase):
    """Regression tests for orphaned pending-recv timeout detection.

    Covers the full lifecycle:
      1. Request enters WAITING_FOR_CHUNK from either waiting or running queue
      2. restore_queues() moves it back to the scheduler queue
      3. Timeout fires via collect_timed_out_request_ids()
      4. Scheduler removes from both queues and calls _free_request()
    """

    def test_waiting_since_recorded_on_chunk_wait(self):
        """_waiting_since is set when a request enters WAITING_FOR_CHUNK."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=True,
        )
        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])

        coord.process_pending_chunks(
            waiting,
            [],
            chunk_ready_req_ids=set(),
            chunk_finished_req_ids=set(),
        )

        self.assertIn("r1", coord._waiting_since)
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)

    def test_waiting_since_cleared_on_chunk_arrival(self):
        """_waiting_since is cleared when a chunk arrives."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=True,
        )
        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])

        coord.process_pending_chunks(
            waiting,
            [],
            chunk_ready_req_ids={"r1"},
            chunk_finished_req_ids=set(),
        )

        self.assertNotIn("r1", coord._waiting_since)

    def test_waiting_since_recorded_on_input_wait(self):
        """_waiting_since is set when a request enters WAITING_FOR_INPUT."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=False,
        )
        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])

        coord.process_pending_full_payload_inputs(
            waiting,
            [],
            stage_recv_req_ids=set(),
        )

        self.assertIn("r1", coord._waiting_since)

    def test_waiting_since_cleared_on_input_arrival(self):
        """_waiting_since is cleared when input data arrives."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=False,
        )
        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        coord._waiting_for_input.append(req)
        coord._waiting_since["r1"] = 0.0

        waiting = MockQueue()
        coord.process_pending_full_payload_inputs(
            waiting,
            [],
            stage_recv_req_ids={"r1"},
        )

        self.assertNotIn("r1", coord._waiting_since)
        self.assertEqual(req.status, RequestStatus.WAITING)

    def test_collect_timed_out_request_ids_no_timeout(self):
        """No IDs returned when nothing has timed out."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
        )
        import time

        coord._waiting_since["r1"] = time.monotonic()

        result = coord.collect_timed_out_request_ids(timeout_s=300.0)
        self.assertEqual(result, set())

    def test_collect_timed_out_request_ids_expired(self):
        """Timed-out IDs are returned and _waiting_since is cleared."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
        )
        coord._waiting_since["r1"] = 0.0  # epoch → definitely expired
        coord._waiting_since["r2"] = 0.0

        import time

        coord._waiting_since["r3"] = time.monotonic() + 9999  # far future

        result = coord.collect_timed_out_request_ids(timeout_s=1.0)

        self.assertEqual(result, {"r1", "r2"})
        self.assertNotIn("r1", coord._waiting_since)
        self.assertNotIn("r2", coord._waiting_since)
        self.assertIn("r3", coord._waiting_since)

    def test_collect_removes_from_coordinator_queues(self):
        """Timed-out requests are defensively removed from internal queues."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
        )
        r1 = _make_request("r1")
        r2 = _make_request("r2")
        coord._waiting_for_chunk_waiting.append(r1)
        coord._waiting_for_input.append(r2)
        coord._waiting_since["r1"] = 0.0
        coord._waiting_since["r2"] = 0.0

        result = coord.collect_timed_out_request_ids(timeout_s=1.0)

        self.assertEqual(result, {"r1", "r2"})
        self.assertEqual(len(coord._waiting_for_chunk_waiting), 0)
        self.assertEqual(len(coord._waiting_for_input), 0)

    def test_free_finished_request_clears_waiting_since(self):
        """free_finished_request clears _waiting_since."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
        )
        coord._waiting_since["r1"] = 0.0
        coord.free_finished_request("r1")
        self.assertNotIn("r1", coord._waiting_since)

    def test_timeout_from_running_queue_full_lifecycle(self):
        """End-to-end: request from running → WAITING_FOR_CHUNK → restore →
        timeout → removed from running list.

        This is the critical regression case: WAITING_FOR_CHUNK requests
        that originated from self.running are placed back into self.running
        by restore_queues(), but their status remains WAITING_FOR_CHUNK.
        The scheduler must remove from BOTH queues unconditionally.
        """
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=True,
        )

        # 1) Request starts in running queue with WAITING status
        req = _make_request("r1", status=RequestStatus.WAITING)
        running = [req]
        waiting = MockQueue()

        # 2) process_pending_chunks: moves to WAITING_FOR_CHUNK
        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids=set(),
            chunk_finished_req_ids=set(),
        )
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)
        self.assertIn("r1", coord._waiting_since)
        self.assertEqual(len(coord._waiting_for_chunk_running), 1)

        # 3) restore_queues: back to running (status stays WAITING_FOR_CHUNK)
        coord.restore_queues(waiting, running)
        self.assertIn(req, running)
        self.assertEqual(len(coord._waiting_for_chunk_running), 0)
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)

        # 4) Force timeout by setting _waiting_since to epoch
        coord._waiting_since["r1"] = 0.0

        timed_out_ids = coord.collect_timed_out_request_ids(timeout_s=1.0)
        self.assertEqual(timed_out_ids, {"r1"})

        # 5) Scheduler removes from both queues (simulating the scheduler path)
        timed_out_id_set = {id(req)}
        running = [r for r in running if id(r) not in timed_out_id_set]
        waiting.remove_requests([req])

        self.assertNotIn(req, running)
        self.assertEqual(len(waiting), 0)

    def test_timeout_from_waiting_queue_full_lifecycle(self):
        """End-to-end: request from waiting → WAITING_FOR_CHUNK → restore →
        timeout → removed from waiting queue."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=True,
        )

        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids=set(),
            chunk_finished_req_ids=set(),
        )
        self.assertEqual(len(coord._waiting_for_chunk_waiting), 1)

        coord.restore_queues(waiting, running)
        self.assertIn(req, waiting)

        coord._waiting_since["r1"] = 0.0
        timed_out_ids = coord.collect_timed_out_request_ids(timeout_s=1.0)
        self.assertEqual(timed_out_ids, {"r1"})

        waiting.remove_requests([req])
        self.assertEqual(len(waiting), 0)


class TestOverflowPreemption(unittest.TestCase):
    """Tests for P1-1: overflow requests must get WAITING status.

    Overflow happens when multiple WAITING_FOR_CHUNK requests in
    ``_waiting_for_chunk_running`` receive their chunk in the same cycle.
    ``_process_chunk_queue`` restores them to RUNNING (``continue``
    path) while RUNNING requests without chunks are moved out.  If the
    net result exceeds ``scheduler_max_num_seqs``, the tail is pushed
    to ``waiting_queue`` and must have status == WAITING.
    """

    def test_overflow_sets_waiting_status(self):
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=1,
            stage_id=1,
            async_chunk=True,
        )

        # r1 is currently RUNNING in the queue.
        # r2, r3 were previously moved to _waiting_for_chunk_running.
        r1 = _make_request("r1", status=RequestStatus.RUNNING)
        r2 = _make_request("r2", status=RequestStatus.WAITING_FOR_CHUNK)
        r3 = _make_request("r3", status=RequestStatus.WAITING_FOR_CHUNK)

        running = [r1]
        waiting = MockQueue([])
        coord._waiting_for_chunk_running.extend([r2, r3])

        # restore_queues puts r2, r3 back into running
        coord.restore_queues(waiting, running)
        self.assertEqual(len(running), 3)

        # Now process_pending_chunks with r2, r3 chunks ready:
        # _process_chunk_queue will:
        #   r1 (RUNNING) → no chunk → move to _waiting_for_chunk_running
        #   r2 (WAITING_FOR_CHUNK, chunk ready) → set RUNNING, stay in running
        #   r3 (WAITING_FOR_CHUNK, chunk ready) → set RUNNING, stay in running
        # running = [r2, r3], len=2 > max=1 → overflow
        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r2", "r3"},
            chunk_finished_req_ids=set(),
        )

        self.assertEqual(len(running), 1)
        self.assertEqual(len(waiting), 1)
        overflow_req = list(waiting)[0]
        self.assertEqual(
            overflow_req.status,
            RequestStatus.WAITING,
            f"Overflowed request should have WAITING status, got {overflow_req.status}",
        )

    def test_overflow_does_not_strand_request(self):
        """Without the fix, the overflowed request would keep its
        RUNNING status in the waiting queue and never be re-scheduled."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=1,
            stage_id=1,
            async_chunk=True,
        )

        r1 = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        r2 = _make_request("r2", status=RequestStatus.WAITING_FOR_CHUNK)
        coord._waiting_for_chunk_running.extend([r1, r2])

        running: list = []
        waiting = MockQueue([])

        coord.restore_queues(waiting, running)
        self.assertEqual(len(running), 2)

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r1", "r2"},
            chunk_finished_req_ids=set(),
        )

        self.assertEqual(len(running), 1)
        self.assertEqual(len(waiting), 1)
        for req in waiting:
            self.assertNotEqual(req.status, RequestStatus.RUNNING, "Overflowed request must not keep RUNNING status")


if __name__ == "__main__":
    unittest.main()
