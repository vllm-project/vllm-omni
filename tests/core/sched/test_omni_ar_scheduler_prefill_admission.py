# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prefill admission batching of OmniARScheduler (stage connector extra or additional_config knobs)."""

import time
from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

defer = OmniARScheduler._should_defer_waiting_admission


def _scheduler(min_batch=4, max_wait_s=5.0, min_running=0, running=2, waiting=1, waited_s=0.0):
    return SimpleNamespace(
        _prefill_admission_min_batch=min_batch,
        _prefill_admission_max_wait_s=max_wait_s,
        _prefill_admission_min_running=min_running,
        running=list(range(running)),
        waiting=[SimpleNamespace(arrival_time=time.time() - waited_s) for _ in range(waiting)],
    )


def test_waits_until_a_batch_queues_or_the_oldest_waited_long_enough():
    assert defer(_scheduler(waiting=3))
    assert not defer(_scheduler(waiting=4))
    assert not defer(_scheduler(waiting=2, max_wait_s=0.05, waited_s=1.0))


def test_admits_at_once_below_min_running_or_when_idle():
    assert not defer(_scheduler(min_running=4, running=3))
    assert defer(_scheduler(min_running=4, running=4))
    assert not defer(_scheduler(running=0))


def test_off_without_a_batch_size():
    assert not defer(_scheduler(min_batch=0))
    assert not defer(_scheduler(min_batch=1))


def test_a_batch_size_needs_a_wait_bound():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            stage_connector_config={"extra": {"ar_prefill_admission_min_batch": 6}},
        )
    )
    scheduler = SimpleNamespace(vllm_config=config)
    with pytest.raises(ValueError, match="max_wait_ms"):
        OmniARScheduler._init_prefill_admission_policy(scheduler)


def test_a_stage_without_connector_knobs_reads_additional_config():
    knobs = {
        "ar_prefill_admission_min_batch": 8,
        "ar_prefill_admission_max_wait_ms": 100,
        "ar_prefill_admission_min_running": 4,
    }
    config = SimpleNamespace(
        model_config=SimpleNamespace(stage_connector_config={"extra": {"stage_id": 0}}),
        additional_config=knobs,
    )
    scheduler = SimpleNamespace(vllm_config=config)
    OmniARScheduler._init_prefill_admission_policy(scheduler)
    assert scheduler._prefill_admission_min_batch == 8
    assert scheduler._prefill_admission_max_wait_s == pytest.approx(0.1)
    assert scheduler._prefill_admission_min_running == 4

    config.model_config.stage_connector_config["extra"]["ar_prefill_admission_min_batch"] = 6
    OmniARScheduler._init_prefill_admission_policy(scheduler)
    assert scheduler._prefill_admission_min_batch == 6


@pytest.mark.parametrize("connector_batch", [None, 3])
def test_single_stage_admission_settings_and_connector_precedence(connector_batch):
    scheduler = _scheduler(waiting=3, running=32)
    scheduler.vllm_config = SimpleNamespace(
        additional_config={
            "ar_prefill_admission_min_batch": 8,
            "ar_prefill_admission_max_wait_ms": 150,
            "ar_prefill_admission_min_running": 32,
        },
        model_config=SimpleNamespace(
            stage_connector_config=(
                None if connector_batch is None else {"extra": {"ar_prefill_admission_min_batch": connector_batch}}
            )
        ),
    )
    OmniARScheduler._init_prefill_admission_policy(scheduler)
    assert defer(scheduler) is (connector_batch is None)
    # The wait bound applies even if the single-stage batch has not filled.
    scheduler.waiting[0].arrival_time = time.time() - 1.0
    assert not defer(scheduler)


def test_deferred_waiting_prefill_keeps_talker_kstep_drafts(monkeypatch):
    from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
    from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler

    sched = object.__new__(OmniARScheduler)
    sched.policy = SchedulingPolicy.FCFS
    decode = SimpleNamespace(
        request_id="d0",
        spec_token_ids=[1, 2, 3],
        num_computed_tokens=10,
        prompt_token_ids=[0] * 10,
        status=None,
    )
    prefill = SimpleNamespace(request_id="p0", arrival_time=time.time(), status=None)
    sched.running = [decode]
    sched.waiting = create_request_queue(sched.policy)
    sched.waiting.add_request(prefill)
    sched.kv_holding_waiting = create_request_queue(sched.policy)
    sched.deferred_waiting = set()
    sched.max_num_running_reqs = sched.max_num_active_reqs = 8
    sched.max_model_len = 4096
    sched.num_spec_tokens = 3
    sched._prefill_admission_min_batch = 4
    sched._prefill_admission_max_wait_s = 5.0
    sched._prefill_admission_min_running = 1
    sched._talker_kstep_armed = lambda: True
    sched.use_v2_model_runner = False
    sched.chunk_transfer_adapter = sched.input_coordinator = None
    sched._process_pending_omni_inputs = lambda **kw: None
    sched._postprocess_omni_schedule_output = lambda *a, **kw: None
    sched.get_finished_requests_needing_kv_transfer = lambda: {}
    sched._wrap_omni_scheduler_output = lambda out, **kw: out
    monkeypatch.setattr(
        VLLMScheduler,
        "schedule",
        lambda self, *a: SimpleNamespace(
            num_scheduled_tokens={"d0": 1 + len(decode.spec_token_ids)},
            scheduled_spec_decode_tokens={"d0": list(decode.spec_token_ids)} if decode.spec_token_ids else {},
        ),
    )

    out = sched.schedule()
    assert decode.spec_token_ids == [1, 2, 3]
    assert out.num_scheduled_tokens == {"d0": 4}
    assert list(sched.waiting) == [prefill]
