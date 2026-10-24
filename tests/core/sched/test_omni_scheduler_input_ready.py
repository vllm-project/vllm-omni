# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Input-ready time of downstream-stage requests, which splits vllm_queued_ms."""

import time

import pytest
from vllm import SamplingParams
from vllm.config import SchedulerConfig, VllmConfig
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.engine import EngineCoreEvent, EngineCoreEventType
from vllm.v1.request import Request, RequestStatus

import vllm_omni.core.sched.omni_ar_scheduler as ar_sched_mod
from vllm_omni.config.model import OmniModelConfig
from vllm_omni.core.sched import omni_scheduler_mixin
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator
from vllm_omni.outputs import OmniConnectorOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _gated_scheduler(*, async_chunk: bool, log_stats: bool = True, gates_stage_input: bool = True) -> OmniARScheduler:
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.kv_holding_waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.deferred_waiting = set()
    scheduler.running = []
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = OmniSchedulingCoordinator(
        stage_id=1, scheduler_max_num_seqs=2, async_chunk=async_chunk
    )
    scheduler._init_omni_connector_output_inbox()
    scheduler.log_stats = log_stats
    scheduler._gates_stage_input = gates_stage_input
    return scheduler


def _add_request(scheduler: OmniARScheduler, request_id: str = "downstream") -> Request:
    request = Request(request_id, [1], SamplingParams(max_tokens=1), pooling_params=None)
    scheduler.requests = {request.request_id: request}
    scheduler._enqueue_waiting_request(request)
    return request


def _ready_output(async_chunk: bool, request_id: str = "downstream") -> OmniConnectorOutput:
    if async_chunk:
        return OmniConnectorOutput(chunk_ready_req_ids={request_id})
    return OmniConnectorOutput(stage_recv_req_ids={request_id})


def _step(scheduler: OmniARScheduler, now: float, ready: OmniConnectorOutput | None = None) -> None:
    """Input handling of one scheduler step: gates, stamp, restore."""
    if ready is not None:
        scheduler.enqueue_omni_connector_output(ready)
    scheduler._consume_pending_connector_output(model_mode="ar")
    scheduler._stamp_input_ready(now)
    scheduler._restore_omni_wait_queues()


@pytest.mark.parametrize("async_chunk", [False, True], ids=["full_payload", "async_chunk"])
def test_input_gate_records_the_first_step_that_sees_input_ready(async_chunk):
    scheduler = _gated_scheduler(async_chunk=async_chunk)
    request = _add_request(scheduler)

    # Queued before its upstream input exists: the gate parks it.
    _step(scheduler, 1.0)
    _step(scheduler, 2.0)
    assert getattr(request, "omni_input_ready_ts", None) is None

    _step(scheduler, 3.0, _ready_output(async_chunk))
    assert request.status == RequestStatus.WAITING
    assert request.omni_input_ready_ts == 3.0

    # Only the first input is recorded.
    _step(scheduler, 4.0, _ready_output(async_chunk))
    assert request.omni_input_ready_ts == 3.0


@pytest.mark.parametrize(
    ("log_stats", "gates_stage_input"),
    [(True, False), (False, True)],
    ids=["no_input_gate", "stats_off"],
)
def test_input_ready_time_needs_an_input_gate_and_stats(log_stats, gates_stage_input):
    scheduler = _gated_scheduler(async_chunk=False, log_stats=log_stats, gates_stage_input=gates_stage_input)
    request = _add_request(scheduler)

    _step(scheduler, 1.0, _ready_output(async_chunk=False))

    assert request.status == RequestStatus.WAITING
    assert getattr(request, "omni_input_ready_ts", None) is None


@pytest.mark.parametrize(
    "status",
    [RequestStatus.WAITING_FOR_REMOTE_KVS, RequestStatus.PREEMPTED],
)
def test_input_ready_time_skips_requests_not_waiting_for_admission(status):
    scheduler = _gated_scheduler(async_chunk=False)
    scheduler.input_coordinator = None
    request = _add_request(scheduler)
    request.status = status

    scheduler._stamp_input_ready(1.0)

    assert getattr(request, "omni_input_ready_ts", None) is None


def test_input_ready_time_covers_requests_holding_kv_blocks():
    scheduler = _gated_scheduler(async_chunk=False)
    scheduler.input_coordinator = None
    request = Request("held", [1, 2], SamplingParams(max_tokens=1), pooling_params=None)
    request.num_computed_tokens = 1
    scheduler.requests = {request.request_id: request}
    scheduler._enqueue_waiting_request(request)
    assert request in scheduler.kv_holding_waiting

    scheduler._stamp_input_ready(1.0)

    assert request.omni_input_ready_ts == 1.0


def _init_gate_flag(**model_config_fields) -> bool:
    class _Scheduler(omni_scheduler_mixin.OmniSchedulerMixin):
        pass

    scheduler = _Scheduler()
    model_config = object.__new__(OmniModelConfig)
    for name, value in model_config_fields.items():
        setattr(model_config, name, value)
    scheduler.vllm_config = object.__new__(VllmConfig)
    scheduler.vllm_config.model_config = model_config
    scheduler._init_omni_io_scheduling_state()
    return scheduler._gates_stage_input


@pytest.mark.parametrize(
    ("stage_id", "async_chunk", "required", "role", "expected"),
    [
        pytest.param(0, True, False, None, False, id="async_chunk_stage0"),
        pytest.param(1, True, False, None, True, id="async_chunk_receiver"),
        pytest.param(1, True, False, "sender", False, id="async_chunk_sender_only"),
        pytest.param(0, False, True, None, False, id="full_payload_stage0"),
        pytest.param(1, False, True, None, True, id="full_payload_receiver"),
        pytest.param(1, False, False, None, False, id="orchestrator_fed"),
    ],
)
def test_gates_stage_input_follows_the_input_gates(monkeypatch, stage_id, async_chunk, required, role, expected):
    adapter_cls = omni_scheduler_mixin.OmniChunkTransferAdapter
    monkeypatch.setattr(omni_scheduler_mixin, "OmniChunkTransferAdapter", lambda _config: object.__new__(adapter_cls))

    flag = _init_gate_flag(
        stage_id=stage_id,
        async_chunk=async_chunk,
        requires_full_payload_input=required,
        stage_connector_config={"name": "SharedMemoryConnector", "extra": {"role": role} if role else {}},
    )

    assert flag is expected


@pytest.mark.parametrize(("role", "expected"), [("receiver", True), ("sender", False)])
def test_gates_stage_input_on_the_native_data_plane(role, expected):
    class _Scheduler(omni_scheduler_mixin.OmniSchedulerMixin):
        pass

    scheduler = _Scheduler()
    model_config = object.__new__(OmniModelConfig)
    model_config.stage_id = 1
    model_config.async_chunk = True
    model_config.requires_full_payload_input = False
    model_config.supports_native_mrv2_data_plane = True
    model_config.use_v2_model_runner = True
    model_config.stage_connector_config = {"name": "SharedMemoryConnector", "extra": {"role": role}}
    scheduler.vllm_config = object.__new__(VllmConfig)
    scheduler.vllm_config.model_config = model_config
    scheduler.vllm_config.scheduler_config = object.__new__(SchedulerConfig)
    scheduler.vllm_config.scheduler_config.max_num_seqs = 1
    scheduler._init_omni_io_scheduling_state()

    assert scheduler._native_data_plane
    assert scheduler._gates_stage_input is expected


@pytest.mark.parametrize(
    ("events", "ready_ts", "expected"),
    [
        pytest.param([EngineCoreEvent(EngineCoreEventType.SCHEDULED, 5.0)], 4.5, 4.5, id="with_events"),
        # Decode outputs carry no events and do not resend the time.
        pytest.param(None, 4.5, None, id="without_events"),
        # Stage 0 or a stage without an input gate never records one.
        pytest.param([EngineCoreEvent(EngineCoreEventType.SCHEDULED, 5.0)], None, None, id="not_recorded"),
    ],
)
def test_engine_output_sends_input_ready_time_with_engine_core_events(events, ready_ts, expected):
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    request = Request("r", [1], SamplingParams(max_tokens=1), pooling_params=None)
    request.events = list(events or [])
    if ready_ts is not None:
        request.omni_input_ready_ts = ready_ts

    output = scheduler._make_omni_engine_output(request, new_token_ids=[])

    assert output.events == events
    assert output.input_ready_ts == expected


def test_ar_schedule_stamps_after_the_input_gates_and_before_upstream(monkeypatch):
    calls: list[str] = []
    times: dict[str, float] = {}
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.kv_holding_waiting = create_request_queue(SchedulingPolicy.FCFS)
    scheduler.deferred_waiting = set()
    scheduler.running = []
    scheduler.chunk_transfer_adapter = None
    scheduler.max_num_active_reqs = 8
    scheduler.max_num_running_reqs = 8
    scheduler.input_coordinator = None
    scheduler.num_waiting_for_streaming_input = 0
    scheduler._process_pending_omni_inputs = lambda model_mode: calls.append("inputs")

    def stamp_input_ready(now: float) -> None:
        calls.append("stamp")
        times["stamp"] = now

    scheduler._stamp_input_ready = stamp_input_ready
    scheduler._resync_streaming_input_counter = lambda: None
    scheduler._should_defer_waiting_admission = lambda: False
    scheduler._restore_omni_wait_queues = lambda: None
    scheduler._postprocess_omni_schedule_output = lambda *args, **kwargs: None
    scheduler.get_finished_requests_needing_kv_transfer = lambda: {}
    scheduler._wrap_omni_scheduler_output = lambda output, **kwargs: output

    def upstream_schedule(_self, _throttle_prefills: bool = False):
        # Upstream takes its SCHEDULED timestamp here, after the stamp.
        calls.append("upstream")
        times["upstream"] = time.monotonic()
        return object()

    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "schedule", upstream_schedule)

    OmniARScheduler.schedule(scheduler)

    assert calls == ["inputs", "stamp", "upstream"]
    assert times["stamp"] <= times["upstream"]
