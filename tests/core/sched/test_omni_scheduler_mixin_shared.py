from collections import defaultdict
from types import SimpleNamespace

import pytest
from vllm.v1.engine import FinishReason

from vllm_omni.core.sched import omni_scheduler_mixin
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.output import OmniChunkRecvHandle

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Scheduler(OmniSchedulerMixin):
    pass


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


def test_schedule_lifecycle_helpers_process_and_restore_both_input_paths():
    calls = []
    scheduler = _Scheduler()
    scheduler.waiting = ["waiting"]
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
            ("process", waiting, running, scheduler_requests)
        ),
        restore_queues=lambda waiting, running, scheduler_requests: calls.append(
            ("restore-chunks", waiting, running, scheduler_requests)
        ),
        collect_timed_out_request_ids=_collect_timed_out,
        collect_failed_send_request_ids=_collect_failed_sends,
    )
    scheduler.input_coordinator = SimpleNamespace(
        restore_queues=lambda waiting: calls.append(("restore-full", waiting))
    )

    scheduler._process_pending_omni_inputs("ar")
    scheduler._restore_omni_wait_queues()

    assert calls == [
        ("consume", "ar"),
        ("timeouts",),
        ("process", scheduler.waiting, scheduler.running, scheduler.requests),
        # The chunk deadline runs after chunks are applied, so a chunk that
        # arrived this cycle resets the clock before it is measured (R1.1).
        ("chunk-timeouts", omni_scheduler_mixin.DEFAULT_INPUT_WAIT_TIMEOUT_S),
        ("failed-sends",),
        ("restore-chunks", scheduler.waiting, scheduler.running, scheduler.requests),
        ("restore-full", scheduler.waiting),
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
    outputs = {}

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
