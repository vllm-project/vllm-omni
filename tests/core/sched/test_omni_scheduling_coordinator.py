# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for OmniSchedulingCoordinator.

These tests use mock request objects and mock queues.  They do not require
GPU, vLLM runtime, or any connector.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import vllm_omni.core.sched.omni_scheduling_coordinator as coord_mod
from vllm_omni.core.sched.omni_scheduling_coordinator import (
    OmniSchedulingCoordinator,
    uses_async_chunk_coordinator,
    uses_async_chunk_model_runner_transport,
    uses_full_payload_input_coordinator,
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


class TestFullPayloadCoordinatorSelection(unittest.TestCase):
    """Tests for the (model_arch, model_stage) whitelist gate.

    The init_omni_connectors arch allowlist is keyed by ``model_arch`` and
    is a superset of the stages registered here -- consumer-wait stages
    must be registered explicitly in ``_FULL_PAYLOAD_INPUT_STAGES``, while
    the init allowlist covers both producer- and consumer-side runners.
    These tests pin which ``(arch, stage)`` pairs the gate fires for today.
    """

    # Expected whitelist (model_arch, model_stage).  Hardcoded to avoid the
    # tautology of importing _FULL_PAYLOAD_INPUT_STAGES and asserting it
    # against itself; any drift between this matrix and the whitelist will
    # fail loudly here.
    EXPECTED_FULL_PAYLOAD_INPUT_STAGES: frozenset[tuple[str, str]] = frozenset(
        {
            ("Qwen3OmniMoeForConditionalGeneration", "talker"),
            ("Qwen3OmniMoeForConditionalGeneration", "code2wav"),
            ("Qwen2_5OmniForConditionalGeneration", "talker"),
            ("Qwen2_5OmniForConditionalGeneration", "code2wav"),
            ("CovoAudioForConditionalGeneration", "code2wav"),
            ("MiMoAudioModel", "code2wav"),
            ("Qwen3TTSCode2Wav", "code2wav"),
            ("CosyVoice3Model", "cosyvoice3_code2wav"),
            ("IndexTTS2S2MelDecoder", "indextts2_s2mel_decoder"),
            ("DyninOmniForConditionalGeneration", "token2image"),
            ("DyninOmniForConditionalGeneration", "token2audio"),
        }
    )

    def test_whitelist_matches_expected_matrix(self):
        """_FULL_PAYLOAD_INPUT_STAGES must equal the hardcoded expected matrix.

        Catches both accidental additions (which would silently enable the
        consumer-wait gate for a new arch) and accidental removals (which
        would silently disable an enabled arch).
        """
        from vllm_omni.core.sched.omni_scheduling_coordinator import _FULL_PAYLOAD_INPUT_STAGES

        self.assertEqual(
            frozenset(_FULL_PAYLOAD_INPUT_STAGES),
            self.EXPECTED_FULL_PAYLOAD_INPUT_STAGES,
            msg="_FULL_PAYLOAD_INPUT_STAGES drifted from the expected matrix; "
            "update EXPECTED_FULL_PAYLOAD_INPUT_STAGES if intentional.",
        )

    def test_all_whitelisted_arch_stage_pairs_fire_gate(self):
        """Every (arch, stage) pair in the expected matrix must fire
        the gate when stage_id > 0 and async_chunk=False.
        """
        for arch, stage in self.EXPECTED_FULL_PAYLOAD_INPUT_STAGES:
            model_config = SimpleNamespace(
                stage_id=1,
                async_chunk=False,
                model_arch=arch,
                model_stage=stage,
            )
            self.assertTrue(
                uses_full_payload_input_coordinator(model_config),
                msg=f"expected gate to fire for {arch}/{stage}",
            )

    def test_other_arch_or_stage_or_mode_does_not_fire(self):
        cases = [
            SimpleNamespace(
                stage_id=1, async_chunk=True, model_arch="Qwen3OmniMoeForConditionalGeneration", model_stage="talker"
            ),
            SimpleNamespace(
                stage_id=0, async_chunk=False, model_arch="Qwen3OmniMoeForConditionalGeneration", model_stage="thinker"
            ),
            SimpleNamespace(
                stage_id=1,
                async_chunk=False,
                model_arch="Qwen3OmniMoeForConditionalGeneration",
                model_stage="some_future_stage",
            ),
            SimpleNamespace(
                stage_id=1, async_chunk=False, model_arch="Qwen3TTSForConditionalGeneration", model_stage="code2wav"
            ),
            SimpleNamespace(
                stage_id=1, async_chunk=False, model_arch="MingFlashOmniForConditionalGeneration", model_stage="talker"
            ),
            SimpleNamespace(stage_id=1, async_chunk=False, model_arch=None, model_stage="talker"),
            SimpleNamespace(
                stage_id=1, async_chunk=False, model_arch="Qwen3OmniMoeForConditionalGeneration", model_stage=None
            ),
        ]
        for model_config in cases:
            self.assertFalse(
                uses_full_payload_input_coordinator(model_config),
                msg=f"expected gate OFF for {model_config}",
            )


class TestChunkCoordinatorStateTransition(unittest.TestCase):
    """Test 5: process_pending_chunks transitions WAITING_FOR_CHUNK → target."""

    def test_ready_request_transitions_to_waiting(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)

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

    def test_late_ready_before_queue_insertion_is_retained(self):
        # A chunk can arrive before the request is surfaced into a
        # queue.  The readiness must be retained (not lost when the connector
        # output is cleared) so a later cycle still transitions the request.
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)

        # Cycle 1: ready for "r1" arrives while no queue holds it yet.
        coord.process_pending_chunks(MockQueue([]), [], chunk_ready_req_ids={"r1"}, chunk_finished_req_ids=set())
        self.assertIn("r1", coord.requests_with_ready_chunks, "late ready must be retained")

        # Cycle 2: r1 now appears as a fresh WAITING request, but chunk_ready is
        # already empty (the connector output was consumed last cycle).  Because
        # retention recorded r1, it must NOT be parked into WAITING_FOR_CHUNK --
        # it stays schedulable.  Without the retain it would be wrongly parked.
        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])
        coord.process_pending_chunks(waiting, [], chunk_ready_req_ids=set(), chunk_finished_req_ids=set())
        self.assertEqual(req.status, RequestStatus.WAITING, "ready-before-insertion must not be parked")
        self.assertIn(req, waiting, "request must remain schedulable in the waiting queue")

    def test_non_ready_stays_waiting_for_chunk(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)

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

    def test_finish_only_request_does_not_become_ready(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids=set(),
            chunk_finished_req_ids={"r1"},
        )

        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)
        self.assertNotIn("r1", coord.requests_with_ready_chunks)
        self.assertIn("r1", coord.finished_requests)

    def test_stage_0_is_noop(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=0)
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
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

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
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r1"},
            chunk_finished_req_ids={"r1"},
        )

        self.assertNotIn("r1", coord.finished_requests)
        self.assertIn("r1", coord._completed_chunk_streams)
        self.assertTrue(coord.chunk_stream_completed("r1"))

    def test_terminal_ready_request_does_not_wait_for_more_chunks(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_CHUNK)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(
            waiting,
            running,
            chunk_ready_req_ids={"r1"},
            chunk_finished_req_ids={"r1"},
        )
        coord.postprocess_scheduler_output(
            SimpleNamespace(scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(req_ids=["r1"]))
        )

        req.status = RequestStatus.RUNNING
        running = [req]

        coord.process_pending_chunks(
            MockQueue([]),
            running,
            chunk_ready_req_ids=set(),
            chunk_finished_req_ids=set(),
        )

        self.assertEqual(req.status, RequestStatus.RUNNING)
        self.assertIn(req, running)
        self.assertEqual(coord.pending_connector_registrations, [])


class TestChunkCoordinatorUpdateRequestMetadata(unittest.TestCase):
    """Test update_request_metadata applies scheduling metadata to requests."""

    def test_ar_mode_no_longer_sets_additional_information(self):
        """AR mode only processes scheduling metadata, not full payloads."""
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

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
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1")
        req.prompt_token_ids = [0, 0, 0]
        req.num_prompt_tokens = 3
        req.num_computed_tokens = 3
        req._all_token_ids = [0, 0, 0, 99]
        req._output_token_ids = [99]
        requests = {"r1": req}

        request_metadata = {
            "r1": {
                "code_predictor_codes": [10, 20, 30],
                "left_context_size": 25,
            }
        }

        coord.update_request_metadata(requests, request_metadata, model_mode="generation")

        self.assertEqual(req.prompt_token_ids, [10, 20, 30])
        self.assertEqual(req.num_prompt_tokens, 3)
        self.assertEqual(req.num_computed_tokens, 0)
        self.assertEqual(req._all_token_ids, [10, 20, 30])
        self.assertEqual(req._output_token_ids, [])
        self.assertIsNone(req.additional_information)
        self.assertEqual(req._omni_initial_model_buffer, {"meta": {"left_context_size": 25}})

    def test_generation_mode_flattens_tensor_code_predictor_codes(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1")
        req.prompt_token_ids = [9]
        req.num_prompt_tokens = 1
        req._all_token_ids = [9, 8]
        req._output_token_ids = [8]
        requests = {"r1": req}

        coord.update_request_metadata(
            requests,
            {"r1": {"code_predictor_codes": torch.tensor([[1, 2, 3]], dtype=torch.long)}},
            model_mode="generation",
        )

        self.assertEqual(req.prompt_token_ids, [1, 2, 3])
        self.assertEqual(req.num_prompt_tokens, 3)
        self.assertEqual(req._all_token_ids, [1, 2, 3])
        self.assertEqual(req._output_token_ids, [])

    def test_generation_mode_flattens_nested_code_predictor_codes(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1")
        req.prompt_token_ids = [9]
        req.num_prompt_tokens = 1
        req._all_token_ids = [9, 8]
        req._output_token_ids = [8]
        requests = {"r1": req}

        coord.update_request_metadata(
            requests,
            {"r1": {"code_predictor_codes": [[1, 2], [3, 4]]}},
            model_mode="generation",
        )

        self.assertEqual(req.prompt_token_ids, [1, 2, 3, 4])
        self.assertEqual(req.num_prompt_tokens, 4)
        self.assertEqual(req._all_token_ids, [1, 2, 3, 4])
        self.assertEqual(req._output_token_ids, [])


class TestChunkCoordinatorPostprocess(unittest.TestCase):
    """Test postprocess_scheduler_output clears ready chunks."""

    def test_clear_ready(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)
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
        self.assertEqual(len(coord.pending_connector_registrations), 1)

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

    def test_pending_connector_registrations(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1)

        req = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_full_payload_inputs(
            waiting,
            running,
            stage_recv_req_ids=set(),
        )

        self.assertEqual(len(coord.pending_connector_registrations), 1)
        self.assertEqual(coord.pending_connector_registrations[0].request_id, "r1")

    def test_idle_cycles_retain_received_marker_before_request_appears(self):
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
            async_chunk=False,
        )
        coord._full_payload_input_received.add("late")
        coord.finished_requests.add("late")

        waiting = MockQueue()
        running: list = []

        coord.process_pending_full_payload_inputs(waiting, running, stage_recv_req_ids=set())

        self.assertIn("late", coord._full_payload_input_received)
        self.assertIn("late", coord.finished_requests)

        late_req = _make_request("late", status=RequestStatus.WAITING)
        waiting.add_request(late_req)

        coord.process_pending_full_payload_inputs(waiting, running, stage_recv_req_ids=set())

        self.assertEqual(late_req.status, RequestStatus.WAITING)
        self.assertEqual(coord.pending_connector_registrations, [])
        self.assertIn("late", coord._full_payload_input_received)
        self.assertIn("late", coord.finished_requests)


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
        """Timed-out full-payload input waits are returned and cleared."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
        )
        coord._waiting_since["r1"] = 0.0  # epoch -> definitely expired
        coord._waiting_since["r2"] = 0.0
        coord._waiting_for_input_req_ids.update({"r1", "r2"})

        import time

        coord._waiting_since["r3"] = time.monotonic() + 9999  # far future
        coord._waiting_for_input_req_ids.add("r3")

        result = coord.collect_timed_out_request_ids(timeout_s=1.0)

        self.assertEqual(result, {"r1", "r2"})
        self.assertNotIn("r1", coord._waiting_since)
        self.assertNotIn("r2", coord._waiting_since)
        self.assertIn("r3", coord._waiting_since)
        self.assertNotIn("r1", coord._waiting_for_input_req_ids)
        self.assertNotIn("r2", coord._waiting_for_input_req_ids)
        self.assertIn("r3", coord._waiting_for_input_req_ids)

    def test_collect_removes_from_coordinator_queues(self):
        """Timed-out full-payload waits are removed from input queues only."""
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
        coord._waiting_for_input_req_ids.add("r2")

        result = coord.collect_timed_out_request_ids(timeout_s=1.0)

        self.assertEqual(result, {"r2"})
        self.assertEqual(len(coord._waiting_for_chunk_waiting), 1)
        self.assertEqual(len(coord._waiting_for_input), 0)
        self.assertIn("r1", coord._waiting_since)
        self.assertNotIn("r2", coord._waiting_since)

    def test_free_finished_request_clears_waiting_since(self):
        """free_finished_request clears coordinator lifecycle markers."""
        coord = OmniSchedulingCoordinator(
            scheduler_max_num_seqs=10,
            stage_id=1,
        )
        coord._waiting_since["r1"] = 0.0
        coord._full_payload_input_received.add("r1")
        coord.finished_requests.add("r1")
        coord._completed_chunk_streams.add("r1")
        coord.free_finished_request("r1")
        self.assertNotIn("r1", coord._waiting_since)
        self.assertNotIn("r1", coord._full_payload_input_received)
        self.assertNotIn("r1", coord.finished_requests)
        self.assertNotIn("r1", coord._completed_chunk_streams)

    def test_async_chunk_running_queue_wait_does_not_timeout(self):
        """Async chunk waits may pause after restore without being timed out."""
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

        # 4) Async chunk waits may pause without being failed by the coordinator
        coord._waiting_since["r1"] = 0.0

        timed_out_ids = coord.collect_timed_out_request_ids(timeout_s=1.0)
        self.assertEqual(timed_out_ids, set())
        self.assertIn("r1", coord._waiting_since)
        self.assertIn(req, running)
        self.assertEqual(len(waiting), 0)

    def test_async_chunk_waiting_queue_wait_does_not_timeout(self):
        """Async chunk waits in the waiting queue are not timeout-failed."""
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
        self.assertEqual(timed_out_ids, set())
        self.assertIn("r1", coord._waiting_since)
        self.assertIn(req, waiting)


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


class TestAsyncChunkCoordinatorGate(unittest.TestCase):
    """Async-chunk transport is broader than scheduler receive coordination.

    Qwen3 thinker is a producer-only stage: it sends through the model-runner
    connector, but it must not enter coordinator WAITING_FOR_CHUNK state.
    """

    _SM = {"name": "SharedMemoryConnector"}
    _MOONCAKE = {"name": "MooncakeStoreConnector"}
    _DEFAULT_CONNECTOR = object()

    def _qwen3(self, stage: str, *, async_chunk: bool = True, connector=_DEFAULT_CONNECTOR):
        return SimpleNamespace(
            async_chunk=async_chunk,
            model_arch="Qwen3OmniMoeForConditionalGeneration",
            model_stage=stage,
            stage_connector_config=self._SM if connector is self._DEFAULT_CONNECTOR else connector,
        )

    def test_qwen3_sharedmemory_transport_matrix(self):
        with mock.patch.object(
            coord_mod,
            "_supports_async_chunk_model_runner_transport_platform",
            return_value=True,
        ):
            thinker = self._qwen3("thinker")
            talker = self._qwen3("talker")
            code2wav = self._qwen3("code2wav")

            self.assertTrue(uses_async_chunk_model_runner_transport(thinker))
            self.assertFalse(uses_async_chunk_coordinator(thinker))
            self.assertTrue(uses_async_chunk_model_runner_transport(talker))
            self.assertTrue(uses_async_chunk_coordinator(talker))
            self.assertTrue(uses_async_chunk_model_runner_transport(code2wav))
            self.assertTrue(uses_async_chunk_coordinator(code2wav))

            # Missing connector config uses the SharedMemory default on the runner path.
            self.assertTrue(uses_async_chunk_model_runner_transport(self._qwen3("talker", connector=None)))
            self.assertTrue(uses_async_chunk_model_runner_transport(self._qwen3("talker", connector={})))
            self.assertFalse(uses_async_chunk_model_runner_transport(self._qwen3("talker", connector={"name": ""})))

    def test_npu_stays_on_adapter(self):
        npu_platform = SimpleNamespace(is_npu=lambda: True)
        with mock.patch.object(coord_mod.omni_platforms, "_current_omni_platform", npu_platform):
            mc = self._qwen3("talker")
            self.assertFalse(uses_async_chunk_model_runner_transport(mc))
            self.assertFalse(uses_async_chunk_coordinator(mc))

    def test_mooncake_stays_on_adapter(self):
        mc = self._qwen3("talker", connector=self._MOONCAKE)
        self.assertFalse(uses_async_chunk_model_runner_transport(mc))
        self.assertFalse(uses_async_chunk_coordinator(mc))

    def test_sync_or_non_allowlisted_does_not_fire(self):
        sync_talker = self._qwen3("talker", async_chunk=False)
        self.assertFalse(uses_async_chunk_model_runner_transport(sync_talker))
        self.assertFalse(uses_async_chunk_coordinator(sync_talker))

        non_allowlisted_arch = SimpleNamespace(
            async_chunk=True,
            model_arch="MiMoAudioModel",
            model_stage="code2wav",
            stage_connector_config=self._SM,
        )
        self.assertFalse(uses_async_chunk_model_runner_transport(non_allowlisted_arch))
        self.assertFalse(uses_async_chunk_coordinator(non_allowlisted_arch))

        non_allowlisted_stage = self._qwen3("not-a-stage")
        self.assertFalse(uses_async_chunk_model_runner_transport(non_allowlisted_stage))
        self.assertFalse(uses_async_chunk_coordinator(non_allowlisted_stage))


class TestAsyncChunkRecvRegistration(unittest.TestCase):
    """Regression coverage: a parked async-chunk
    request must be registered for bg-thread recv via SchedulerOutput
    ``pending_connector_registrations``; otherwise the runner never calls
    register_chunk_recv and the request can wait until timeout. The
    full-payload pass runs after process_pending_chunks each cycle in
    async-chunk mode, so it must NOT re-clear the chunk registrations.
    """

    def test_parked_chunk_request_registered_and_survives_full_payload_pass(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)
        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])
        running: list = []

        # No chunk ready yet -> park WAITING_FOR_CHUNK AND register for recv.
        coord.process_pending_chunks(waiting, running, chunk_ready_req_ids=set(), chunk_finished_req_ids=set())
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)
        regs = [h.request_id for h in coord.pending_connector_registrations]
        self.assertIn("r1", regs, "parked async-chunk request must be registered for bg recv polling")

        # The full-payload pass (runs after, every cycle) must not wipe it.
        coord.process_pending_full_payload_inputs(waiting, running, stage_recv_req_ids=set())
        regs_after = [h.request_id for h in coord.pending_connector_registrations]
        self.assertIn("r1", regs_after, "full-payload pass must not drop async-chunk recv registrations")

    def test_free_finished_request_prunes_parked_requests(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)
        keep_waiting = _make_request("keep-w", status=RequestStatus.WAITING_FOR_CHUNK)
        keep_running = _make_request("keep-r", status=RequestStatus.WAITING_FOR_CHUNK)
        keep_input = _make_request("keep-i", status=RequestStatus.WAITING_FOR_INPUT)
        stale_waiting = _make_request("stale", status=RequestStatus.WAITING_FOR_CHUNK)
        stale_running = _make_request("stale", status=RequestStatus.WAITING_FOR_CHUNK)
        stale_input = _make_request("stale", status=RequestStatus.WAITING_FOR_INPUT)

        coord._waiting_for_chunk_waiting.extend([stale_waiting, keep_waiting])
        coord._waiting_for_chunk_running.extend([stale_running, keep_running])
        coord._waiting_for_input.extend([stale_input, keep_input])
        coord._waiting_since["stale"] = 1.0
        coord._waiting_for_input_req_ids.add("stale")

        coord.free_finished_request("stale")

        waiting = MockQueue()
        running: list = []
        coord.restore_queues(waiting, running)

        self.assertEqual([request.request_id for request in waiting._items], ["keep-w", "keep-i"])
        self.assertEqual([request.request_id for request in running], ["keep-r"])
        self.assertNotIn("stale", coord._waiting_since)
        self.assertNotIn("stale", coord._waiting_for_input_req_ids)

    def test_input_timeout_ignores_async_chunk_waits(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=True)
        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_chunks(waiting, running, chunk_ready_req_ids=set(), chunk_finished_req_ids=set())
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_CHUNK)
        coord._waiting_since["r1"] = 0.0

        with mock.patch.object(coord_mod.time, "monotonic", return_value=10.0):
            timed_out = coord.collect_timed_out_request_ids(timeout_s=1.0)

        self.assertEqual(timed_out, set())
        self.assertIn("r1", coord._waiting_since)

    def test_input_timeout_still_collects_full_payload_waits(self):
        coord = OmniSchedulingCoordinator(scheduler_max_num_seqs=10, stage_id=1, async_chunk=False)
        req = _make_request("r1", status=RequestStatus.WAITING)
        waiting = MockQueue([req])
        running: list = []

        coord.process_pending_full_payload_inputs(waiting, running, stage_recv_req_ids=set())
        self.assertEqual(req.status, RequestStatus.WAITING_FOR_INPUT)
        coord._waiting_since["r1"] = 0.0

        with mock.patch.object(coord_mod.time, "monotonic", return_value=10.0):
            timed_out = coord.collect_timed_out_request_ids(timeout_s=1.0)

        self.assertEqual(timed_out, {"r1"})
        self.assertNotIn("r1", coord._waiting_since)
        self.assertNotIn("r1", coord._waiting_for_input_req_ids)


if __name__ == "__main__":
    unittest.main()
