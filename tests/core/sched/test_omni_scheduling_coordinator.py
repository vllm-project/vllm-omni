# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for OmniSchedulingCoordinator.

These tests use mock request objects and mock queues.  They do not require
GPU, vLLM runtime, or any connector.

Chunk waiting (WAITING_FOR_CHUNK / process_pending_chunks) lives on
OmniChunkTransferAdapter — see tests/distributed/omni_connectors/.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import pytest

import vllm_omni.core.sched.omni_scheduling_coordinator as coord_mod
from vllm_omni.core.sched.omni_scheduling_coordinator import (
    OmniSchedulingCoordinator,
    uses_native_mrv2_data_plane,
)
from vllm_omni.core.sched.output import OmniChunkRecvHandle
from vllm_omni.outputs import SchedulingMetadataUpdate

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# ------------------------------------------------------------------ #
#  Mock helpers
# ------------------------------------------------------------------ #


class _RequestStatus:
    WAITING = "waiting"
    RUNNING = "running"
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
        num_output_placeholders=0,
        _all_token_ids=[],
        _output_token_ids=[],
        payload_sender_info=None,
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

    def remove_request(self, request):
        self._items.remove(request)

    def remove_requests(self, requests):
        remove_set = set(id(r) for r in requests)
        self._items = [r for r in self._items if id(r) not in remove_set]


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #


class TestNativeMRV2DataPlaneSelection(unittest.TestCase):
    def test_native_plane_requires_v2_and_async_chunk_capability(self):
        native = SimpleNamespace(async_chunk=True, supports_native_mrv2_data_plane=True)

        self.assertTrue(uses_native_mrv2_data_plane(native, use_v2_model_runner=True))
        self.assertFalse(uses_native_mrv2_data_plane(native, use_v2_model_runner=False))
        self.assertFalse(uses_native_mrv2_data_plane(SimpleNamespace(async_chunk=False), use_v2_model_runner=True))


def test_chunk_registration_ready_and_terminal_lifecycle():
    coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)
    req = _make_request("internal", status=RequestStatus.WAITING)
    req.external_req_id = "external"
    waiting = MockQueue([req])
    coord.process_pending_chunks(waiting, [], set(), set())
    assert req.status == RequestStatus.WAITING_FOR_CHUNK
    [handle] = coord.pending_chunk_registrations
    assert isinstance(handle, OmniChunkRecvHandle)
    assert (handle.request_id, handle.external_req_id) == ("internal", "external")
    coord.restore_queues(waiting, [])
    coord.process_pending_chunks(waiting, [], {"internal"}, set())
    assert req.status == RequestStatus.WAITING
    assert "internal" in coord.requests_with_ready_chunks
    req.status = RequestStatus.WAITING_FOR_CHUNK
    coord.process_pending_chunks(waiting, [], set(), {"internal"})
    assert "internal" in coord.finished_requests
    coord.update_request_metadata(
        {"internal": req},
        {"internal": SchedulingMetadataUpdate(input_terminal=True)},
    )
    unscheduled = SimpleNamespace(num_scheduled_tokens={})
    assert coord.get_scheduled_input_terminal_req_ids(unscheduled) == set()
    coord.postprocess_scheduler_output(SimpleNamespace(scheduled_new_reqs=[], scheduled_cached_reqs=None))
    assert coord.input_terminal_req_ids == {"internal"}
    scheduled = SimpleNamespace(num_scheduled_tokens={"internal": 1}, scheduled_new_reqs=[], scheduled_cached_reqs=None)
    assert coord.get_scheduled_input_terminal_req_ids(scheduled) == {"internal"}
    coord.postprocess_scheduler_output(scheduled)
    assert coord.input_terminal_req_ids == set()


def test_chunk_waiting_removes_request_from_running_list():
    coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)
    req = _make_request("running", status=RequestStatus.RUNNING)
    running = [req]

    coord.process_pending_chunks(MockQueue(), running, set(), set())

    assert running == []
    assert req.status == RequestStatus.WAITING_FOR_CHUNK
    assert list(coord._waiting_for_chunk_running) == [req]


class TestChunkCoordinatorUpdateRequestMetadata(unittest.TestCase):
    """Test update_request_metadata applies scheduling metadata to requests."""

    def test_ar_mode_no_longer_sets_additional_information(self):
        """AR mode only processes scheduling metadata, not full payloads."""
        coord = OmniSchedulingCoordinator(stage_id=1)

        req = _make_request("r1")
        requests = {"r1": req}

        # Only scheduling metadata is passed now (full payload stays in model runner)
        request_metadata = {"r1": SchedulingMetadataUpdate(resize_prompt_to=50)}

        coord.update_request_metadata(requests, request_metadata)

        # next_stage_prompt_len should update prompt_token_ids
        self.assertEqual(len(req.prompt_token_ids), 50)
        self.assertEqual(req.num_prompt_tokens, 50)
        # additional_information should NOT be set
        self.assertIsNone(getattr(req, "additional_information", None))

    def test_generation_mode(self):
        coord = OmniSchedulingCoordinator(stage_id=1)

        req = _make_request("r1")
        req.prompt_token_ids = [0, 0, 0]
        req.num_prompt_tokens = 3
        req.num_computed_tokens = 3
        req._all_token_ids = [0, 0, 0, 99]
        req._output_token_ids = [99]
        requests = {"r1": req}

        request_metadata = {"r1": SchedulingMetadataUpdate(prompt_token_ids=(10, 20, 30))}

        coord.update_request_metadata(requests, request_metadata)

        self.assertEqual(req.prompt_token_ids, [10, 20, 30])
        self.assertEqual(req.num_prompt_tokens, 3)
        self.assertEqual(req.num_computed_tokens, 0)
        self.assertEqual(req._all_token_ids, [10, 20, 30])
        self.assertEqual(req._output_token_ids, [])
        self.assertIsNone(req.additional_information)

    def test_typed_prompt_update_replaces_existing_prompt(self):
        coord = OmniSchedulingCoordinator(stage_id=1)

        req = _make_request("r1")
        req.prompt_token_ids = [9]
        req.num_prompt_tokens = 1
        req._all_token_ids = [9, 8]
        req._output_token_ids = [8]
        requests = {"r1": req}

        coord.update_request_metadata(
            requests,
            {"r1": SchedulingMetadataUpdate(prompt_token_ids=(1, 2, 3))},
        )

        self.assertEqual(req.prompt_token_ids, [1, 2, 3])
        self.assertEqual(req.num_prompt_tokens, 3)
        self.assertEqual(req._all_token_ids, [1, 2, 3])
        self.assertEqual(req._output_token_ids, [])

    def test_typed_prompt_update_preserves_all_token_invariants(self):
        coord = OmniSchedulingCoordinator(stage_id=1)

        req = _make_request("r1")
        req.prompt_token_ids = [9]
        req.num_prompt_tokens = 1
        req._all_token_ids = [9, 8]
        req._output_token_ids = [8]
        requests = {"r1": req}

        coord.update_request_metadata(
            requests,
            {"r1": SchedulingMetadataUpdate(prompt_token_ids=(1, 2, 3, 4))},
        )

        self.assertEqual(req.prompt_token_ids, [1, 2, 3, 4])
        self.assertEqual(req.num_prompt_tokens, 4)
        self.assertEqual(req._all_token_ids, [1, 2, 3, 4])
        self.assertEqual(req._output_token_ids, [])


class TestWaitingForInputTransition(unittest.TestCase):
    """Test process_pending_full_payload_inputs transitions WAITING_FOR_INPUT."""

    def test_transition_on_recv(self):
        coord = OmniSchedulingCoordinator(stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])

        coord.process_pending_full_payload_inputs(
            waiting,
            stage_recv_req_ids={"r1"},
        )

        self.assertEqual(req.status, RequestStatus.WAITING)

    def test_stays_waiting_for_input_if_not_received(self):
        coord = OmniSchedulingCoordinator(stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])

        coord.process_pending_full_payload_inputs(
            waiting,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_INPUT)
        self.assertEqual(len(coord._waiting_for_input), 1)

    def test_stage_0_is_noop(self):
        coord = OmniSchedulingCoordinator(stage_id=0)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])

        coord.process_pending_full_payload_inputs(
            waiting,
            stage_recv_req_ids={"r1"},
        )
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_INPUT)

    def test_restore_queues_includes_waiting_for_input(self):
        coord = OmniSchedulingCoordinator(stage_id=1)

        r1 = _make_request("r1")
        coord._waiting_for_input.append(r1)

        waiting = MockQueue()

        coord.restore_queues(waiting)

        self.assertIn(r1, waiting)
        self.assertEqual(len(coord._waiting_for_input), 0)

    def test_full_payload_mode_auto_transitions_waiting_to_waiting_for_input(self):
        """Fresh downstream WAITING requests enter WAITING_FOR_INPUT."""
        coord = OmniSchedulingCoordinator(stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])

        coord.process_pending_full_payload_inputs(
            waiting,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_INPUT)
        self.assertEqual(len(coord._waiting_for_input), 1)
        self.assertEqual(len(coord.pending_input_registrations), 1)

    def test_pending_input_registrations(self):
        coord = OmniSchedulingCoordinator(stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])

        coord.process_pending_full_payload_inputs(
            waiting,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(len(coord.pending_input_registrations), 1)
        self.assertEqual(coord.pending_input_registrations[0].request_id, "r1")

    def test_pending_input_registration_carries_payload_sender_info(self):
        coord = OmniSchedulingCoordinator(stage_id=1)
        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        req.payload_sender_info = {"host": "10.0.0.1", "zmq_port": 50051}
        waiting = MockQueue([req])

        coord.process_pending_full_payload_inputs(waiting, set())

        self.assertEqual(
            coord.pending_input_registrations[0].payload_sender_info,
            {"host": "10.0.0.1", "zmq_port": 50051},
        )

    def test_idle_cycles_retain_received_marker_before_request_appears(self):
        coord = OmniSchedulingCoordinator(stage_id=1)
        coord._full_payload_input_received.add("late")
        coord.finished_requests.add("late")

        waiting = MockQueue()

        coord.process_pending_full_payload_inputs(waiting, stage_recv_req_ids=set())

        self.assertIn("late", coord._full_payload_input_received)
        self.assertIn("late", coord.finished_requests)

        late_req = _make_request("late", status=RequestStatus.WAITING)
        waiting.add_request(late_req)

        coord.process_pending_full_payload_inputs(waiting, stage_recv_req_ids=set())

        self.assertEqual(late_req.status, RequestStatus.WAITING)
        self.assertEqual(coord.pending_input_registrations, [])
        self.assertIn("late", coord._full_payload_input_received)
        self.assertIn("late", coord.finished_requests)


class TestTimeoutDetection(unittest.TestCase):
    """Regression tests for orphaned pending-recv timeout detection.

    Covers WAITING_FOR_INPUT lifecycle timeouts. Chunk waiting timeouts are
    covered by OmniChunkTransferAdapter tests.
    """

    def test_waiting_since_recorded_on_input_wait(self):
        """_waiting_since is set when a request enters WAITING_FOR_INPUT."""
        coord = OmniSchedulingCoordinator(stage_id=1)
        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])

        coord.process_pending_full_payload_inputs(
            waiting,
            stage_recv_req_ids=set(),
        )

        self.assertIn("r1", coord._waiting_since)

    def test_waiting_since_cleared_on_input_arrival(self):
        """_waiting_since is cleared when input data arrives."""
        coord = OmniSchedulingCoordinator(stage_id=1)
        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        coord._waiting_for_input.append(req)
        coord._waiting_since["r1"] = 0.0

        waiting = MockQueue()
        coord.process_pending_full_payload_inputs(
            waiting,
            stage_recv_req_ids={"r1"},
        )

        self.assertNotIn("r1", coord._waiting_since)
        self.assertEqual(req.status, RequestStatus.WAITING)

    def test_collect_timed_out_request_ids_no_timeout(self):
        """No IDs returned when nothing has timed out."""
        coord = OmniSchedulingCoordinator(stage_id=1)
        import time

        coord._waiting_since["r1"] = time.monotonic()

        result = coord.collect_timed_out_request_ids(timeout_s=300.0)
        self.assertEqual(result, set())

    def test_collect_timed_out_request_ids_expired(self):
        """Timed-out IDs are returned and _waiting_since is cleared."""
        coord = OmniSchedulingCoordinator(stage_id=1)
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
        coord = OmniSchedulingCoordinator(stage_id=1)
        r1 = _make_request("r1")
        coord._waiting_for_input.append(r1)
        coord._waiting_since["r1"] = 0.0

        result = coord.collect_timed_out_request_ids(timeout_s=1.0)

        self.assertEqual(result, {"r1"})
        self.assertEqual(len(coord._waiting_for_input), 0)

    def test_free_finished_request_clears_all_lifecycle_state(self):
        """free_finished_request makes stale connector events harmless."""
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)
        r1, r2 = _make_request("r1"), _make_request("r2")
        coord.finished_requests.add("r1")
        coord.requests_with_ready_chunks.add("r1")
        coord._waiting_for_chunk_running.extend([r1, r2])
        coord.pending_chunk_registrations = [OmniChunkRecvHandle(request_id="r1"), OmniChunkRecvHandle(request_id="r2")]

        coord.free_finished_request("r1")

        self.assertNotIn("r1", coord.finished_requests)
        self.assertNotIn("r1", coord.requests_with_ready_chunks)
        self.assertEqual([r.request_id for r in coord._waiting_for_chunk_running], ["r2"])
        self.assertEqual([h.request_id for h in coord.pending_chunk_registrations], ["r2"])
