# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
import torch

import vllm_omni.core.sched.omni_scheduling_coordinator as coord_mod
from vllm_omni.core.sched.omni_scheduling_coordinator import (
    OmniSchedulingCoordinator,
    uses_full_payload_input_coordinator,
)

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


class TestCoordinatorUpdateRequestMetadata(unittest.TestCase):
    """Test update_request_metadata applies scheduling metadata to requests."""

    def test_ar_mode_no_longer_sets_additional_information(self):
        """AR mode only processes scheduling metadata, not full payloads."""
        coord = OmniSchedulingCoordinator(stage_id=1)

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
        coord = OmniSchedulingCoordinator(stage_id=1)

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
            }
        }

        coord.update_request_metadata(requests, request_metadata, model_mode="generation")

        self.assertEqual(req.prompt_token_ids, [10, 20, 30])
        self.assertEqual(req.num_prompt_tokens, 3)
        self.assertEqual(req.num_computed_tokens, 0)
        self.assertEqual(req._all_token_ids, [10, 20, 30])
        self.assertEqual(req._output_token_ids, [])
        self.assertIsNone(req.additional_information)

    def test_generation_mode_flattens_tensor_code_predictor_codes(self):
        coord = OmniSchedulingCoordinator(stage_id=1)

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
        coord = OmniSchedulingCoordinator(stage_id=1)

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

    def test_free_finished_request_clears_waiting_since(self):
        """free_finished_request clears coordinator lifecycle markers."""
        coord = OmniSchedulingCoordinator(stage_id=1)
        coord._waiting_since["r1"] = 0.0
        coord._full_payload_input_received.add("r1")
        coord.finished_requests.add("r1")
        coord.free_finished_request("r1")
        self.assertNotIn("r1", coord._waiting_since)
        self.assertNotIn("r1", coord._full_payload_input_received)
        self.assertNotIn("r1", coord.finished_requests)


if __name__ == "__main__":
    unittest.main()
