# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import time
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

# Bind Request and RequestStatus only after Omni installs its vLLM patches.
# isort: off
import vllm_omni  # noqa: F401 - import for request patches
from vllm import SamplingParams
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.engine import FinishReason
from vllm.v1.request import Request, RequestStatus
# isort: on

import vllm_omni.core.sched.omni_ar_scheduler as ar_sched_mod
import vllm_omni.core.sched.omni_generation_scheduler as gen_sched_mod
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator
from vllm_omni.core.sched.output import OmniChunkRecvHandle
from vllm_omni.outputs import OmniModelRunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeInputCoordinator:
    def __init__(self) -> None:
        self.freed: list[str] = []

    def free_finished_request(self, request_id: str) -> None:
        self.freed.append(request_id)


class DummyRequest:
    request_id = "req-free"
    client_index = 0
    additional_information = None

    def is_finished(self) -> bool:
        return True


@pytest.mark.parametrize(
    ("scheduler_cls", "scheduler_mod"),
    [
        (OmniARScheduler, ar_sched_mod),
        (OmniGenerationScheduler, gen_sched_mod),
    ],
)
def test_finish_requests_cleans_input_coordinator_for_finished_ids(
    monkeypatch: pytest.MonkeyPatch,
    scheduler_cls,
    scheduler_mod,
) -> None:
    coordinator = FakeInputCoordinator()
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = coordinator
    scheduler.requests = {}
    scheduler.running = []
    scheduler.waiting = []
    scheduler.kv_holding_waiting = []
    scheduler.deferred_waiting = set()
    scheduler._outputs_awaiting_stage_payload = {}

    def fake_finish_requests(self, request_ids, finished_status):
        assert request_ids == ["req-a", "req-b"]
        assert finished_status == RequestStatus.FINISHED_STOPPED
        # vLLM 0.26 returns the finished Request objects, not (req_id, client_index) tuples.
        return [
            SimpleNamespace(request_id="req-a", client_index=0),
            SimpleNamespace(request_id="req-b", client_index=1),
        ]

    monkeypatch.setattr(scheduler_mod.VLLMScheduler, "finish_requests", fake_finish_requests)

    result = scheduler_cls.finish_requests(
        scheduler,
        ["req-a", "req-b"],
        RequestStatus.FINISHED_STOPPED,
    )

    assert [(r.request_id, r.client_index) for r in result] == [("req-a", 0), ("req-b", 1)]
    assert coordinator.freed == ["req-a", "req-b"]


def test_ar_free_request_cleans_input_coordinator_on_normal_free() -> None:
    coordinator = FakeInputCoordinator()
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.input_coordinator = coordinator
    scheduler.chunk_transfer_adapter = None
    scheduler._omits_kv_transfer_cache = {"req-free": (1, True, False)}
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda request: None)
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    scheduler._new_prompt_len_snapshot = {"req-free": 3}
    scheduler._connector_finished = lambda request: (False, None)
    scheduler._should_transfer_kv_for_request = lambda request_id: False
    scheduler._free_blocks = lambda request: None

    result = OmniARScheduler._free_request(scheduler, DummyRequest())

    assert result == (None, None)
    assert coordinator.freed == ["req-free"]
    assert scheduler._omits_kv_transfer_cache == {}
    assert scheduler._new_prompt_len_snapshot == {}


def test_generation_free_request_cleans_input_coordinator(monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator = FakeInputCoordinator()
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler.input_coordinator = coordinator

    def fake_free_request(self, request, delay_free_blocks=False):
        assert delay_free_blocks is True


@pytest.mark.parametrize("queue_name", ["waiting", "kv_holding_waiting", "running"])
@pytest.mark.parametrize("in_flight_tokens", [0, 8])
def test_native_oversized_chunk_uses_real_finish_cleanup_and_emits_one_error(
    mocker, queue_name: str, in_flight_tokens: int
) -> None:
    """Exercise actual scheduler/coordinator cleanup, stubbing only cache I/O."""
    request = Request("oversized-native", [0] * 9, SamplingParams(max_tokens=4), pooling_params=None)
    request.client_index = 4
    request.resumable = True
    request.num_in_flight_tokens = in_flight_tokens
    request.status = RequestStatus.RUNNING if queue_name == "running" else RequestStatus.WAITING
    healthy = Request("healthy-native", [0], SamplingParams(max_tokens=4), pooling_params=None)

    coordinator = OmniSchedulingCoordinator(scheduler_max_num_seqs=4, stage_id=1, async_chunk=True)
    for request_ids in (
        coordinator.requests_with_ready_chunks,
        coordinator.finished_requests,
        coordinator.input_terminal_req_ids,
    ):
        request_ids.update((request.request_id, healthy.request_id))
    # Cleanup must also remove an old parked/registration entry, not just ready.
    parked = (
        coordinator._waiting_for_chunk_running if queue_name == "running" else coordinator._waiting_for_chunk_waiting
    )
    parked.append(request)
    coordinator.pending_chunk_registrations = [OmniChunkRecvHandle(request_id=request.request_id)]
    coordinator._waiting_since[request.request_id] = 0.0

    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler._native_data_plane = True
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = coordinator
    scheduler.max_num_scheduled_tokens = 8
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.kv_holding_waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.running = []
    scheduler.deferred_waiting = set()
    scheduler._outputs_awaiting_stage_payload = {}
    scheduler.requests = {request.request_id: request, healthy.request_id: healthy}
    if queue_name == "running":
        scheduler.running.append(request)
    else:
        getattr(scheduler, queue_name).add_request(request)
    scheduler.waiting.add_request(healthy)
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = defaultdict(set)
    scheduler.num_waiting_for_streaming_input = 0
    scheduler._inflight_prefills = set()
    scheduler.connector = None
    scheduler.aux_output_connector = None
    scheduler.ec_connector = None
    scheduler.defer_block_free = False
    kv_cache = mocker.Mock(spec=KVCacheManager)
    kv_cache.take_events.return_value = []
    kv_cache.estimate_cached_tokens.return_value = 0
    scheduler.kv_cache_manager = kv_cache
    encoder_cache = mocker.Mock(spec=EncoderCacheManager)
    scheduler.encoder_cache_manager = encoder_cache
    scheduler._pending_request_prewarms = {request.request_id: {}, healthy.request_id: {}}
    scheduler._pending_chunk_error_outputs = []
    scheduler._pending_finish_reqs = []
    scheduler._pooling_output_decoder = None
    scheduler.log_stats = False
    # Match vLLM 0.32 initialization with KV-fetch tracking disabled.
    scheduler._kv_fetch_stages = None
    scheduler.perf_metrics = None
    scheduler._last_stats_time = time.monotonic()

    if in_flight_tokens:
        scheduler._reject_oversized_ready_chunks()

        assert scheduler.requests[request.request_id] is request
        assert request in getattr(scheduler, queue_name)
        assert request.request_id in coordinator.requests_with_ready_chunks
        assert not request.is_finished()
        assert request.stop_reason is None
        assert scheduler.finished_req_ids == set()
        assert scheduler._pending_chunk_error_outputs == []
        kv_cache.free.assert_not_called()
        encoder_cache.free.assert_not_called()

        # Return the prior valid audio through the actual update path, rather
        # than manually clearing the in-flight counter before rejection.
        audio_payload = {"audio": torch.tensor([0.25, -0.5])}
        previous_step = SchedulerOutput.make_empty()
        previous_step.num_scheduled_tokens = {request.request_id: in_flight_tokens}
        previous_step.total_num_scheduled_tokens = in_flight_tokens
        previous_output = OmniModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[]],
            pooler_output=[],
            multimodal_outputs=[audio_payload],
        )

        previous_result = scheduler.update_from_output(previous_step, previous_output)

        [audio_output] = previous_result[request.client_index].outputs
        assert audio_output.request_id == request.request_id
        assert audio_output.multimodal_output is audio_payload
        assert audio_output.finish_reason is None
        assert not previous_result[request.client_index].finished_requests
        assert request.num_in_flight_tokens == 0
        assert scheduler.requests[request.request_id] is request
        assert request.request_id in coordinator.requests_with_ready_chunks
        assert scheduler._pending_chunk_error_outputs == []
        kv_cache.free.assert_not_called()
        encoder_cache.free.assert_not_called()

    scheduler._reject_oversized_ready_chunks()

    assert request.status == RequestStatus.FINISHED_ERROR
    assert request.resumable is False
    assert scheduler.requests == {healthy.request_id: healthy}
    assert scheduler.running == []
    assert list(scheduler.waiting) == [healthy]
    assert list(scheduler.kv_holding_waiting) == []
    assert scheduler.deferred_waiting == set()
    assert scheduler.finished_req_ids == {request.request_id}
    assert scheduler.finished_req_ids_dict == {request.client_index: {request.request_id}}
    kv_cache.free.assert_called_once_with(request)
    encoder_cache.free.assert_called_once_with(request)
    assert coordinator.requests_with_ready_chunks == {healthy.request_id}
    assert coordinator.finished_requests == {healthy.request_id}
    assert coordinator.input_terminal_req_ids == {healthy.request_id}
    assert not coordinator._waiting_for_chunk_waiting
    assert not coordinator._waiting_for_chunk_running
    assert coordinator.pending_chunk_registrations == []
    assert coordinator._waiting_since == {}
    assert scheduler._pending_request_prewarms == {healthy.request_id: {}}

    scheduler._reject_oversized_ready_chunks()
    assert len(scheduler._pending_chunk_error_outputs) == 1
    runner_output = OmniModelRunnerOutput(
        req_ids=[],
        req_id_to_index={},
        sampled_token_ids=[],
        pooler_output=[],
    )
    empty_step = SchedulerOutput.make_empty()

    first = scheduler.update_from_output(empty_step, runner_output)

    assert set(first) == {request.client_index}
    assert first[request.client_index].finished_requests == {request.request_id}
    assert len(first[request.client_index].outputs) == 1
    error_output = first[request.client_index].outputs[0]
    assert error_output.request_id == request.request_id
    assert error_output.finish_reason == FinishReason.ERROR
    assert error_output.new_token_ids == []
    assert error_output.stop_reason == "async chunk requires 9 tokens, but max_num_scheduled_tokens is 8"
    assert error_output.is_segment_finished is False
    assert scheduler._pending_chunk_error_outputs == []
    assert scheduler.finished_req_ids_dict == {}
    assert scheduler.update_from_output(empty_step, runner_output) == {}
    kv_cache.free.assert_called_once_with(request)
    encoder_cache.free.assert_called_once_with(request)
