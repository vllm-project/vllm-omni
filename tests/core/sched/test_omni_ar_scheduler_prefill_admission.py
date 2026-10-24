# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prefill admission batching of OmniARScheduler (stage connector extra knobs)."""

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
