from collections import defaultdict
from types import SimpleNamespace

import pytest
from vllm.v1.engine import FinishReason

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.output import OmniChunkRecvHandle

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Scheduler(OmniSchedulerMixin):
    pass


def test_schedule_lifecycle_helpers_process_and_restore_both_input_paths():
    calls = []
    scheduler = _Scheduler()
    scheduler.waiting = ["waiting"]
    scheduler.running = ["running"]
    scheduler.requests = {"request": object()}
    scheduler._consume_pending_connector_output = lambda mode: calls.append(("consume", mode))
    scheduler._process_pending_input_timeouts = lambda: calls.append(("timeouts",))
    scheduler.chunk_transfer_adapter = SimpleNamespace(
        process_pending_chunks=lambda waiting, running, scheduler_requests: calls.append(
            ("process", waiting, running, scheduler_requests)
        ),
        restore_queues=lambda waiting, running, scheduler_requests: calls.append(
            ("restore-chunks", waiting, running, scheduler_requests)
        ),
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
