# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.config import SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import FinishReason
from vllm.v1.request import Request

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.core.sched import omni_scheduler_mixin
from vllm_omni.core.sched.omni_ar_scheduler import _segment_generation_of
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.core.sched.omni_scheduler_mixin import (
    OmniSchedulerMixin,
    _initialize_resumable_segment_generation,
)
from vllm_omni.core.sched.output import OmniChunkRecvHandle
from vllm_omni.data_entry_keys import RETURN_STAGE_PAYLOAD_KEY
from vllm_omni.outputs import OmniConnectorOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Scheduler(OmniSchedulerMixin):
    _is_blocked_waiting_status = staticmethod(Scheduler._is_blocked_waiting_status)


@dataclass
class _OutputRequest:
    request_id: str = "req-output"
    trace_headers: object | None = None
    _omni_segment_generation: int | None = None

    def take_events(self) -> list[object]:
        return []


@dataclass
class _FreshRequest:
    request_id: str = "req-fresh"
    resumable: bool = True
    additional_information: object | None = None
    model_intermediate_buffer: object | None = None
    trace_headers: object | None = None

    def take_events(self) -> list[object]:
        return []


def test_async_chunk_adapter_initializes_for_stage_zero_sender_and_stage_one_receiver(monkeypatch):
    created_adapters = []

    def adapter_factory(config):
        adapter = SimpleNamespace(vllm_config=config)
        created_adapters.append(adapter)
        return adapter

    monkeypatch.setattr(omni_scheduler_mixin, "OmniChunkTransferAdapter", adapter_factory)

    def init_scheduler(**model_config):
        scheduler = _Scheduler()
        scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(**model_config))
        scheduler._init_omni_io_scheduling_state()
        return scheduler

    producer = init_scheduler(
        stage_id=0,
        async_chunk=True,
        requires_full_payload_input=False,
        custom_process_next_stage_input_func="test.pipeline.produce_async_chunk",
    )
    receiver = init_scheduler(
        stage_id=1,
        async_chunk=True,
        requires_full_payload_input=True,
        custom_process_next_stage_input_func=None,
    )
    terminal_stage = init_scheduler(
        stage_id=2,
        async_chunk=True,
        requires_full_payload_input=False,
        custom_process_next_stage_input_func=None,
    )

    assert producer.chunk_transfer_adapter is not None
    assert receiver.chunk_transfer_adapter is not None
    assert terminal_stage.chunk_transfer_adapter is not None
    assert receiver.input_coordinator is None
    assert len(created_adapters) == 3


@pytest.mark.parametrize(
    ("stage_id", "async_chunk", "required", "enabled"),
    [
        (0, False, True, False),
        (1, True, True, False),
        (1, False, False, False),
        (1, False, True, True),
    ],
)
def test_full_payload_coordinator_matches_legacy_gate(monkeypatch, stage_id, async_chunk, required, enabled):
    monkeypatch.setattr(
        omni_scheduler_mixin,
        "OmniChunkTransferAdapter",
        lambda _config: SimpleNamespace(),
    )
    scheduler = _Scheduler()
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            stage_id=stage_id,
            async_chunk=async_chunk,
            requires_full_payload_input=required,
        )
    )

    scheduler._init_omni_io_scheduling_state()

    assert (scheduler.input_coordinator is not None) is enabled


@pytest.mark.parametrize("async_chunk", [False, True])
@pytest.mark.parametrize(("stage_id", "required"), [(0, False), (1, False), (1, True)])
def test_native_data_plane_uses_the_selected_input_protocol(async_chunk, stage_id, required):
    scheduler = _Scheduler()
    model_config = object.__new__(OmniModelConfig)
    model_config.stage_id = stage_id
    model_config.async_chunk = async_chunk
    model_config.requires_full_payload_input = required
    model_config.supports_native_mrv2_data_plane = True
    model_config.use_v2_model_runner = True
    scheduler.vllm_config = object.__new__(VllmConfig)
    scheduler.vllm_config.model_config = model_config
    scheduler.vllm_config.scheduler_config = object.__new__(SchedulerConfig)
    scheduler.vllm_config.scheduler_config.max_num_seqs = 1
    scheduler._init_omni_io_scheduling_state()

    assert scheduler._native_data_plane
    if async_chunk or (stage_id > 0 and required):
        assert scheduler.input_coordinator._async_chunk is async_chunk
        assert scheduler.input_coordinator.receives_chunks is (async_chunk and stage_id > 0)
    else:
        assert scheduler.input_coordinator is None
    assert scheduler._async_chunk_transport_enabled() is async_chunk
    assert scheduler.chunk_transfer_adapter is None
    # Stage zero can own an async coordinator for bookkeeping without
    # receiving connector chunks. Full-payload inputs are not atomic chunks.
    assert OmniGenerationScheduler._requires_atomic_chunk(scheduler) is (async_chunk and stage_id > 0)


def test_schedule_lifecycle_helpers_process_and_restore_both_input_paths():
    calls: list[tuple[Any, ...]] = []
    scheduler = _Scheduler()
    waiting = Request("waiting", [1], SamplingParams(max_tokens=1), pooling_params=None)
    holding = Request("holding", [1, 2], SamplingParams(max_tokens=1), pooling_params=None)
    holding.num_computed_tokens = 1
    scheduler.waiting = [waiting]
    scheduler.kv_holding_waiting = [holding]
    scheduler.running = ["running"]
    scheduler.requests = {"request": object()}
    scheduler._consume_pending_connector_output = lambda mode: calls.append(("consume", mode))
    scheduler._process_pending_input_timeouts = lambda: calls.append(("timeouts",))

    def _collect_timed_out(timeout_s):
        calls.append(("chunk-timeouts", timeout_s))
        return set()

    def _collect_failed_sends():
        calls.append(("failed-sends",))
        return {}

    scheduler.chunk_transfer_adapter = SimpleNamespace(
        receives_chunks=True,
        process_pending_chunks=lambda waiting, running, scheduler_requests: calls.append(
            ("process", list(waiting), running, scheduler_requests)
        ),
        restore_queues=lambda waiting, running, scheduler_requests: calls.append(
            ("restore-chunks", list(waiting), running, scheduler_requests)
        ),
        collect_timed_out_request_ids=_collect_timed_out,
        collect_failed_send_request_ids=_collect_failed_sends,
    )
    scheduler.input_coordinator = SimpleNamespace(
        restore_queues=lambda waiting, running: calls.append(("restore-full", list(waiting), running))
    )

    scheduler._process_pending_omni_inputs("ar")
    scheduler._restore_omni_wait_queues()

    assert calls == [
        ("consume", "ar"),
        ("timeouts",),
        ("process", [holding, waiting], scheduler.running, scheduler.requests),
        # The chunk deadline runs after chunks are applied, so a chunk that
        # arrived this cycle resets the clock before it is measured (R1.1).
        ("chunk-timeouts", omni_scheduler_mixin.DEFAULT_INPUT_WAIT_TIMEOUT_S),
        ("failed-sends",),
        ("restore-chunks", [holding, waiting], scheduler.running, scheduler.requests),
        ("restore-full", [holding, waiting], scheduler.running),
    ]


@pytest.mark.parametrize(
    ("synthesize_abort_outputs", "expected_finish_reason"),
    [(False, None), (True, FinishReason.ABORT)],
)
def test_finished_request_attachment_keeps_ar_abort_policy_explicit(
    synthesize_abort_outputs,
    expected_finish_reason,
):
    scheduler = _Scheduler()
    scheduler.finished_req_ids_dict = defaultdict(set, {2: {"req-finished"}})
    scheduler._outputs_awaiting_stage_payload = {}
    outputs: dict = {}

    scheduler._attach_finished_request_sets(
        outputs,
        synthesize_abort_outputs=synthesize_abort_outputs,
    )

    assert outputs[2].finished_requests == {"req-finished"}
    if expected_finish_reason is None:
        assert outputs[2].outputs == []
    else:
        assert outputs[2].outputs[0].finish_reason == expected_finish_reason
    assert scheduler.finished_req_ids_dict == {}


def test_chunk_receive_handle_carries_minimal_registration_fields():
    handle = OmniChunkRecvHandle(request_id="req", external_req_id="external")
    assert handle.request_id == "req"
    assert handle.external_req_id == "external"


def test_output_helper_preserves_required_nan_counter_default():
    scheduler = _Scheduler()
    request = SimpleNamespace(
        request_id="req-output",
        trace_headers=None,
        take_events=lambda: [],
    )

    output = scheduler._make_omni_engine_output(request, new_token_ids=[])

    assert output.num_nans_in_logits == 0
    assert output.segment_generation is None


def test_output_helper_copies_request_segment_generation():
    scheduler = _Scheduler()
    request = _OutputRequest(_omni_segment_generation=4)

    output = scheduler._make_omni_engine_output(request, new_token_ids=[2150], is_segment_finished=True)

    assert output.is_segment_finished is True
    assert output.segment_generation == 4


def test_output_helper_keeps_explicit_segment_generation_over_request():
    scheduler = _Scheduler()
    request = _OutputRequest(_omni_segment_generation=4)

    output = scheduler._make_omni_engine_output(request, new_token_ids=[2150], segment_generation=0)

    assert output.segment_generation == 0


def test_initialize_resumable_generation_seeds_missing_counter_only():
    fresh = _FreshRequest()
    assert not hasattr(fresh, "_omni_segment_generation")
    _initialize_resumable_segment_generation(fresh)
    assert fresh._omni_segment_generation == 0
    _initialize_resumable_segment_generation(fresh)
    assert fresh._omni_segment_generation == 0

    one_shot = _FreshRequest(request_id="req-oneshot", resumable=False)
    _initialize_resumable_segment_generation(one_shot)
    assert not hasattr(one_shot, "_omni_segment_generation")


def test_add_request_seeds_generation_zero_on_fresh_resumable(monkeypatch):
    added: list[object] = []

    def fake_add_request(self, request: object) -> None:
        added.append(request)
        self.requests[request.request_id] = request

    monkeypatch.setattr(Scheduler, "add_request", fake_add_request)
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler.requests = {}
    scheduler.chunk_transfer_adapter = None
    scheduler._pending_request_prewarms = {}
    scheduler._outputs_awaiting_stage_payload = {}
    scheduler._omni_talker_kstep_cache = False

    request = _FreshRequest()
    assert not hasattr(request, "_omni_segment_generation")
    scheduler.add_request(request)
    assert request._omni_segment_generation == 0
    assert added == [request]

    one_shot = _FreshRequest(request_id="req-oneshot", resumable=False)
    scheduler.add_request(one_shot)
    assert not hasattr(one_shot, "_omni_segment_generation")


def test_park_output_keeps_pre_stop_generation_after_queued_replacement():
    """First park must keep 0 when stop handling already incremented to 1."""
    scheduler = _Scheduler()
    request = _FreshRequest()
    assert not hasattr(request, "_omni_segment_generation")
    _initialize_resumable_segment_generation(request)
    snapshot = _segment_generation_of(request)
    assert snapshot == 0

    request._omni_segment_generation = int(getattr(request, "_omni_segment_generation", 0) or 0) + 1
    assert request._omni_segment_generation == 1
    output = scheduler._make_omni_engine_output(
        request,
        new_token_ids=[2150],
        is_segment_finished=True,
        segment_generation=snapshot,
    )
    assert output.segment_generation == 0
    assert _segment_generation_of(request) == 1


@pytest.mark.parametrize(("role", "coordinated"), [("sender", False), ("receiver", True), (None, True)])
def test_native_downstream_sender_stage_does_not_wait_for_chunks(role, coordinated):
    """An orchestrator-fed downstream stage (MiniCPM-o's Talker) owns only an
    outgoing connector; the native plane must not park it for input chunks."""
    scheduler = _Scheduler()
    model_config = object.__new__(OmniModelConfig)
    model_config.stage_id = 1
    model_config.async_chunk = True
    model_config.requires_full_payload_input = False
    model_config.supports_native_mrv2_data_plane = True
    model_config.use_v2_model_runner = True
    model_config.stage_connector_config = {"name": "SharedMemoryConnector", "extra": {"role": role} if role else {}}
    scheduler.vllm_config = object.__new__(VllmConfig)
    scheduler.vllm_config.model_config = model_config
    scheduler.vllm_config.scheduler_config = object.__new__(SchedulerConfig)
    scheduler.vllm_config.scheduler_config.max_num_seqs = 1
    scheduler._init_omni_io_scheduling_state()

    assert scheduler._native_data_plane
    assert (scheduler.input_coordinator is not None) is coordinated
    if coordinated:
        assert scheduler.input_coordinator.receives_chunks
    assert scheduler._async_chunk_transport_enabled() is coordinated
    assert scheduler.chunk_transfer_adapter is None
    assert OmniGenerationScheduler._requires_atomic_chunk(scheduler) is coordinated


def test_only_run_requests_wait_for_their_stage_payload():
    """Ensure a run request's finished output is held, not reported as aborted, then sent once with its payload."""
    scheduler = _Scheduler()
    # NOTE: run requests don't currently support async chunk
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=0, async_chunk=False))
    scheduler._init_omni_io_scheduling_state()
    scheduler.finished_req_ids_dict = defaultdict(set, {0: {"run"}})
    outputs: dict = defaultdict(list)

    for request_id, tags in (("normal", None), ("run", {RETURN_STAGE_PAYLOAD_KEY: True})):
        request = SimpleNamespace(
            request_id=request_id,
            client_index=0,
            trace_headers=None,
            take_events=lambda: [],
            additional_information=tags,
        )
        scheduler._append_request_output(outputs, request, new_token_ids=[], finish_reason=FinishReason.STOP)

    # Ensure that attaching the finished request sets emits nothing for the held run request
    # while the normal request's output is already in this step's outputs.
    held_step: dict = {}
    scheduler._attach_finished_request_sets(held_step, synthesize_abort_outputs=True)
    assert [output.request_id for output in outputs[0]] == ["normal"]
    assert held_step[0].outputs == []

    # Ensure that once the worker's stage payload arrives, the held
    # run request's finished is emitted once with the payload,
    scheduler.enqueue_omni_connector_output(OmniConnectorOutput(stage_payloads={"run": b"payload"}))
    scheduler._consume_pending_connector_output("ar")
    released_step: dict = {}
    scheduler._attach_finished_request_sets(released_step, synthesize_abort_outputs=True)

    [output] = released_step[0].outputs
    assert (output.request_id, output.finish_reason, output.stage_payload) == ("run", FinishReason.STOP, b"payload")
