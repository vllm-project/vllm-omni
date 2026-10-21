# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for OmniSchedulingCoordinator.

These tests use real request objects and mock queues. They do not require
a GPU, a model, or any connector.

Chunk waiting (WAITING_FOR_CHUNK / process_pending_chunks) lives on
OmniChunkTransferAdapter — see tests/distributed/omni_connectors/.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.request import Request

import vllm_omni.core.sched.omni_scheduling_coordinator as coord_mod
from vllm_omni.core.sched.omni_scheduling_coordinator import (
    OmniSchedulingCoordinator,
    uses_native_mrv2_data_plane,
)
from vllm_omni.core.sched.output import OmniChunkRecvHandle
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import _LoadEntry
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.request import OmniRequest

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


def _make_request(req_id: str, status: str = "waiting") -> Request:
    request = Request(
        request_id=req_id,
        prompt_token_ids=[],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
    )
    request.external_req_id = req_id
    request.status = status
    request.additional_information = None
    request.payload_sender_info = None
    return request


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


@pytest.mark.parametrize("codes", [[], torch.empty((0, 16), dtype=torch.long)])
@pytest.mark.parametrize("prompt_len", [None, 4])
def test_empty_generation_snapshot_clears_previous_input(codes, prompt_len, mocker):
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

    init_none_hash(sha256)
    request = Request(
        request_id="r1",
        prompt_token_ids=[1] * 8,
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        block_hasher=get_request_block_hasher(4, sha256),
    )
    request.append_output_token_ids([99])
    request.num_computed_tokens = 8
    old_ids = request.all_token_ids
    prepare = mocker.spy(coord_mod, "prepare_request_input")
    coordinator = OmniSchedulingCoordinator(stage_id=2)

    coordinator.update_request_metadata(
        {"r1": request},
        {"r1": {"code_predictor_codes": codes, "next_stage_prompt_len": prompt_len}},
        model_mode="generation",
    )

    assert prepare.call_count == 1
    assert prepare.call_args.kwargs["prompt_token_ids"] == []
    assert request.prompt_token_ids == []
    assert request.num_prompt_tokens == request.num_computed_tokens == 0
    assert list(request.all_token_ids) == list(request.output_token_ids) == []
    assert request.block_hashes == []
    assert list(old_ids) == [1] * 8 + [99]


@pytest.mark.parametrize("metadata", [{}, {"code_predictor_codes": None}])
def test_missing_generation_snapshot_preserves_input(metadata, mocker):
    request = _make_request("r1")
    request.append_output_token_ids([99])
    request.num_computed_tokens = 1
    old_ids = request.all_token_ids
    prepare = mocker.spy(coord_mod, "prepare_request_input")

    OmniSchedulingCoordinator(stage_id=2).update_request_metadata(
        {"r1": request}, {"r1": metadata}, model_mode="generation"
    )

    prepare.assert_not_called()
    assert request.all_token_ids is old_ids
    assert list(request.output_token_ids) == [99]
    assert request.num_computed_tokens == 1


@pytest.mark.parametrize("prompt_len", [None, 4, 12])
@pytest.mark.parametrize("codes", [[1, 2, 3, 4], [[1, 2], [3, 4]], torch.tensor([[1, 2], [3, 4]])])
def test_generation_notice_hashes_actual_codec_input_once(prompt_len, codes, mocker):
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

    init_none_hash(sha256)
    hasher = get_request_block_hasher(4, sha256)
    request = Request(
        request_id="r1",
        prompt_token_ids=[0] * 8,
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        block_hasher=hasher,
    )
    old_ids = request.all_token_ids
    prepare = mocker.spy(coord_mod, "prepare_request_input")
    coordinator = OmniSchedulingCoordinator(stage_id=2)
    coordinator.update_request_metadata(
        {"r1": request},
        {
            "r1": {
                "next_stage_prompt_len": prompt_len,
                "code_predictor_codes": codes,
                "left_context_size": 2,
                "input_terminal": True,
            }
        },
        model_mode="generation",
    )
    fresh = Request(
        request_id="fresh",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=request.sampling_params,
        pooling_params=None,
        block_hasher=hasher,
    )
    assert prepare.call_count == 1
    assert prepare.call_args.kwargs["prompt_token_ids"] == [1, 2, 3, 4]
    assert list(request.all_token_ids) == [1, 2, 3, 4]
    assert list(old_ids) == [0] * 8
    assert request.block_hashes == fresh.block_hashes
    assert request._omni_initial_model_buffer == {"meta": {"left_context_size": 2}}
    assert coordinator.input_terminal_req_ids == {"r1"}


@pytest.mark.parametrize(
    "codes",
    [[-1], [1.5], [True], ["1"], [float("inf")], [{}], [[[1]]], torch.tensor([[1.5]])],
    ids=["negative", "float", "bool", "string", "infinity", "mapping", "extra-dimension", "float-tensor"],
)
def test_invalid_generation_codes_do_not_install_length_only_input(codes):
    request = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
    coordinator = OmniSchedulingCoordinator(stage_id=2)
    before = dict(request.__dict__)
    with pytest.raises(ValueError, match="non-negative integer"):
        coordinator.update_request_metadata(
            {"r1": request},
            {"r1": {"next_stage_prompt_len": 4, "code_predictor_codes": codes, "input_terminal": True}},
            model_mode="generation",
        )
    assert request.__dict__ == before
    assert not coordinator.input_terminal_req_ids


@pytest.mark.parametrize("new_length", [4, 12])
def test_initial_length_finalization_rebuilds_real_block_hashes(new_length):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256)
    hasher = get_request_block_hasher(4, sha256)
    request = Request(
        request_id="r1",
        prompt_token_ids=[0] * 8,
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        block_hasher=hasher,
        cache_salt="caller",
    )
    old_ids = request.all_token_ids
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    coordinator.update_request_metadata({"r1": request}, {"r1": {"next_stage_prompt_len": new_length}})
    fresh = Request(
        request_id="fresh",
        prompt_token_ids=[0] * new_length,
        sampling_params=request.sampling_params,
        pooling_params=None,
        block_hasher=hasher,
        cache_salt="caller",
    )
    assert request.block_hashes == fresh.block_hashes
    assert list(request.all_token_ids) == [0] * new_length
    assert list(old_ids) == [0] * 8
    assert getattr(request, "_omni_segment_generation", 0) == 0
    finalized_hashes = request.block_hashes
    coordinator.update_request_metadata({"r1": request}, {"r1": {"next_stage_prompt_len": new_length}})
    assert request.block_hashes is finalized_hashes
    with pytest.raises(ValueError, match="finalized prompt length"):
        coordinator.update_request_metadata({"r1": request}, {"r1": {"next_stage_prompt_len": new_length + 4}})
    assert request.block_hashes is finalized_hashes
    assert request.num_prompt_tokens == new_length


@pytest.mark.parametrize("new_ids", [[11, 12, 13, 14], list(range(12))])
def test_received_prompt_ids_finalize_hashes_once_and_reject_late_changes(new_ids):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    init_none_hash(sha256)
    hasher = get_request_block_hasher(4, sha256)
    request = Request(
        request_id="r1",
        prompt_token_ids=[0] * len(new_ids),
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        block_hasher=hasher,
        cache_salt="caller",
    )
    old_ids, old_hashes = request.all_token_ids, request.block_hashes
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    metadata = {"next_stage_prompt_ids": new_ids, "next_stage_prompt_len": len(new_ids)}
    coordinator.update_request_metadata({"r1": request}, {"r1": metadata})
    fresh = Request(
        request_id="fresh",
        prompt_token_ids=new_ids,
        sampling_params=request.sampling_params,
        pooling_params=None,
        block_hasher=hasher,
        cache_salt="caller",
    )
    assert list(request.all_token_ids) == new_ids
    assert request.block_hashes == fresh.block_hashes != old_hashes
    assert list(old_ids) == [0] * len(new_ids)
    hashes = request.block_hashes
    request.num_computed_tokens = 3
    coordinator.update_request_metadata({"r1": request}, {"r1": metadata})
    assert request.block_hashes is hashes
    assert request.num_computed_tokens == 3
    with pytest.raises(ValueError, match="conflicting finalized prompt IDs"):
        coordinator.update_request_metadata({"r1": request}, {"r1": {"next_stage_prompt_ids": [999] * len(new_ids)}})
    assert request.block_hashes is hashes
    assert request.num_computed_tokens == 3


@pytest.mark.parametrize("bad_ids", [[], [True], [-1], [1.5], "1", [None], [1]])
def test_invalid_received_prompt_ids_do_not_resize_or_release_input(bad_ids):
    request = _make_request("r1", status=RequestStatus.WAITING_FOR_INPUT)
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    old_ids, old_hashes = request.all_token_ids, request.block_hashes
    with pytest.raises(ValueError):
        coordinator.update_request_metadata(
            {"r1": request},
            {
                "r1": {
                    "next_stage_prompt_ids": bad_ids,
                    "next_stage_prompt_len": 3,
                    "input_terminal": True,
                }
            },
        )
    assert request.all_token_ids is old_ids
    assert request.block_hashes is old_hashes
    assert "r1" not in coordinator.input_terminal_req_ids


def test_conditioning_notice_installs_salt_once_and_rejects_late_conflict():
    from tests.core.sched.test_input_finalization import _request
    from vllm_omni.core.sched.input_finalization import compose_conditioning_cache_salt

    request = _request(None, [0] * 4, None)
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    metadata = {
        "next_stage_prompt_ids": [11, 12, 13, 14],
        "next_stage_prompt_len": 4,
        "next_stage_conditioning_digest": "ab" * 32,
    }
    old_ids, old_hashes = request.all_token_ids, request.block_hashes
    coordinator.update_request_metadata({"r1": request}, {"r1": metadata})
    expected_salt = compose_conditioning_cache_salt("caller", "ab" * 32)
    assert request.cache_salt == expected_salt
    assert request._omni_original_cache_salt == "caller"
    assert request._omni_conditioning_digest == "ab" * 32
    assert request.block_hashes is not old_hashes
    assert list(old_ids) == [0] * 4
    final_hashes = request.block_hashes
    request.num_computed_tokens = 2
    coordinator.update_request_metadata({"r1": request}, {"r1": metadata})
    coordinator.update_request_metadata({"r1": request}, {"r1": {"next_stage_prompt_len": 4}})
    assert request.cache_salt == expected_salt
    assert request.block_hashes is final_hashes
    assert request.num_computed_tokens == 2
    with pytest.raises(ValueError, match="conflicting finalized conditioning"):
        coordinator.update_request_metadata(
            {"r1": request},
            {
                "r1": {
                    **metadata,
                    "next_stage_conditioning_digest": "cd" * 32,
                }
            },
        )
    assert request.cache_salt == expected_salt
    assert request.block_hashes is final_hashes
    assert request.num_computed_tokens == 2


@pytest.mark.parametrize(
    "metadata",
    [
        {"next_stage_conditioning_digest": "ab" * 32},
        {"next_stage_conditioning_digest": "ab" * 32, "next_stage_prompt_len": 4},
        {"next_stage_conditioning_digest": "ab" * 32, "next_stage_prompt_ids": [1, 2, 3, 4]},
        {"next_stage_conditioning_digest": "bad", "next_stage_prompt_ids": [1, 2, 3, 4], "next_stage_prompt_len": 4},
    ],
)
def test_partial_or_invalid_conditioning_notice_has_no_visible_effect(metadata):
    from tests.core.sched.test_input_finalization import _request

    request = _request(None, [0] * 8, None)
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    before = dict(request.__dict__)
    with pytest.raises(ValueError):
        coordinator.update_request_metadata({"r1": request}, {"r1": {**metadata, "input_terminal": True}})
    assert request.__dict__ == before
    assert not coordinator.input_terminal_req_ids


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


def test_sender_address_on_the_engine_request_reaches_both_receive_paths():
    """Ensure the sender address the orchestrator puts on an engine request reaches the receivers."""
    sender = {"host": "10.0.0.2", "zmq_port": 50071}
    engine_request = build_engine_core_request_from_tokens(
        "req-1", {"prompt_token_ids": [0]}, SamplingParams(max_tokens=1)
    )
    engine_request.payload_sender_info = sender
    request = OmniRequest.from_engine_core_request(engine_request, block_hasher=None)
    # Build the coordinator and process the request with sender info
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    coordinator.process_pending_full_payload_inputs(MockQueue([request]), stage_recv_req_ids=set())

    # Ensure that the payload send info is accessible on both pending input registrations and a wrapped load entry
    assert [handle.payload_sender_info for handle in coordinator.pending_input_registrations] == [sender]
    assert _LoadEntry(request).source_metadata == {"source_host": "10.0.0.2", "source_port": 50071}
