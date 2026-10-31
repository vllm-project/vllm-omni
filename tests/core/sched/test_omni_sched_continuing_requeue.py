# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Continuing resumable stops are admitted once.

Upstream ``update_from_output`` re-enqueues a request when
``_handle_stopped_request`` returns False. Omni queue cleanup already keeps
resumable rows that a receiver admitted, so the requeue must not admit those
again.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SCHEDULERS = pytest.mark.parametrize(
    "scheduler_cls",
    [OmniARScheduler, OmniGenerationScheduler],
)


class _Req:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id


def _scheduler(scheduler_cls):
    sched = scheduler_cls.__new__(scheduler_cls)
    sched.waiting = []
    sched.kv_holding_waiting = []
    sched.deferred_waiting = set()
    sched.enqueued = []

    def _enqueue(request) -> None:
        sched.enqueued.append(request)
        sched.waiting.append(request)

    sched._enqueue_waiting_request = _enqueue
    return sched


@_SCHEDULERS
def test_requeue_admits_unqueued_request_once(scheduler_cls) -> None:
    sched = _scheduler(scheduler_cls)
    request = _Req("cont")

    sched._requeue_continuing_requests([request])
    sched._requeue_continuing_requests([request])

    assert sched.enqueued == [request]
    assert sched.waiting == [request]


@_SCHEDULERS
@pytest.mark.parametrize("container", ["waiting", "kv_holding_waiting", "deferred_waiting"])
def test_requeue_skips_request_already_queued(scheduler_cls, container: str) -> None:
    sched = _scheduler(scheduler_cls)
    request = _Req("held")
    queue = getattr(sched, container)
    if isinstance(queue, set):
        queue.add(request)
    else:
        queue.append(request)

    sched._requeue_continuing_requests([request])

    assert sched.enqueued == []


def test_requeue_does_not_iterate_magicmock_queues() -> None:
    sched = _scheduler(OmniARScheduler)
    sched.waiting = MagicMock()
    sched.kv_holding_waiting = MagicMock()
    request = _Req("mock-queue")

    sched._requeue_continuing_requests([request])

    assert sched.enqueued == [request]
