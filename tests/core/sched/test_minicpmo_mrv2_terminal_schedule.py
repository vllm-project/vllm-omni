# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The terminal-row hold through the production schedule() and KV allocator.

Only the hold selection is replaced, so the test pins the upstream contract the
hold relies on: a running decode whose next_decode_eligible_step is ahead is
not scheduled, and the field is restored after the call.
"""

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus

import vllm_omni.core.sched.omni_ar_scheduler as ar_sched_mod
from tests.core.sched.test_omni_recompute_preemption import _output
from tests.core.sched.test_omni_recompute_preemption import _scheduler as _real_scheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_schedule_skips_a_held_decode_and_restores_its_step(monkeypatch):
    scheduler = _real_scheduler(num_blocks=8)
    request = Request("held", list(range(16)), SamplingParams(max_tokens=8), None, client_index=0)
    scheduler.add_request(request)
    scheduler.update_from_output(scheduler.schedule(), _output(request))
    assert request.status == RequestStatus.RUNNING
    eligible = request.next_decode_eligible_step

    def hold():
        held = [(request, request.next_decode_eligible_step)]
        request.next_decode_eligible_step = ar_sched_mod._HELD_DECODE_STEP
        return held

    monkeypatch.setattr(scheduler, "_hold_minicpmo45_terminal_steps", hold)
    assert request.request_id not in scheduler.schedule().num_scheduled_tokens
    assert request.next_decode_eligible_step == eligible

    monkeypatch.setattr(scheduler, "_hold_minicpmo45_terminal_steps", lambda: [])
    assert scheduler.schedule().num_scheduled_tokens == {request.request_id: 1}
