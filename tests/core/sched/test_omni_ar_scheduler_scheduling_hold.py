# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A stage processor can ask the AR scheduler to skip its sole running request."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler, VLLMScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler(running, held_ids):
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.running = list(running)
    scheduler._scheduling_hold_hook = lambda now: set(held_ids)
    return scheduler


def test_request_is_not_held_while_other_rows_decode():
    held, other = SimpleNamespace(request_id="a"), SimpleNamespace(request_id="b")
    scheduler = _scheduler([held, other], {"a"})
    assert scheduler._take_held_running_requests() == []
    assert scheduler.running == [held, other]


@pytest.mark.parametrize("raises", [False, True])
def test_held_request_keeps_admission_slot_and_is_restored(monkeypatch, raises):
    held = SimpleNamespace(request_id="held")
    admitted = SimpleNamespace(request_id="new")
    scheduler = _scheduler([held], {"held"})
    scheduler.max_num_active_reqs = 2
    scheduler._drop_aborted_queued_requests = lambda: None
    scheduler._process_pending_omni_inputs = lambda **kwargs: None
    scheduler._resync_streaming_input_counter = lambda: None
    scheduler._should_defer_waiting_admission = lambda: False
    scheduler._async_chunk_transport_enabled = lambda: False
    scheduler._restore_omni_wait_queues = lambda: None
    scheduler._postprocess_omni_schedule_output = lambda *args, **kwargs: None
    scheduler.get_finished_requests_needing_kv_transfer = lambda: {}
    scheduler._wrap_omni_scheduler_output = lambda output, **kwargs: output
    output = object()

    def upstream(owner, throttle_prefills=False):
        # The retained slot is excluded from the upstream admission limit.
        assert owner.max_num_active_reqs == 1
        owner.running.append(admitted)
        if raises:
            raise RuntimeError("schedule failed")
        return output

    monkeypatch.setattr(VLLMScheduler, "schedule", upstream)
    if raises:
        with pytest.raises(RuntimeError, match="schedule failed"):
            scheduler.schedule()
    else:
        assert scheduler.schedule() is output
    assert scheduler.running == [admitted, held]
    assert len(scheduler.running) == scheduler.max_num_active_reqs == 2
