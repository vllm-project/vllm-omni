"""Integration tests for the post-finish defensive purge of
``self.running`` placed in
``OmniARScheduler.finish_requests`` /
``OmniGenerationScheduler.finish_requests``.

Pins *placement* — that ``_purge_finished_from_running`` runs AFTER
``super().finish_requests`` so it catches entries upstream's removal
missed (e.g. status mid-transition, connector cleanup popping
``self.requests`` without unwinding ``self.running``). Direct helper
unit tests live in ``test_omni_scheduler_mixin.py``; these tests
exercise the full ``finish_requests`` path end-to-end.

Regression for the residual self.running slot leak surface paired
with ``_realign_request_status_to_queues`` (vllm-omni#3774).
"""

from __future__ import annotations

import pytest
from vllm.v1.request import RequestStatus

import vllm_omni.core.sched.omni_ar_scheduler as ar_sched_mod
import vllm_omni.core.sched.omni_generation_scheduler as gen_sched_mod
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _StubRequest:
    """Minimal Request stub with the surface the helpers exercise."""

    def __init__(self, request_id: str, status: RequestStatus) -> None:
        self.request_id = request_id
        self.status = status

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)


_SCHEDULER_PARAMS = [
    pytest.param(OmniARScheduler, ar_sched_mod, id="ar"),
    pytest.param(OmniGenerationScheduler, gen_sched_mod, id="generation"),
]


def _make_scheduler(scheduler_cls, *, requests, running, waiting):
    """Bare scheduler with just the surface ``finish_requests`` reads."""
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = None
    scheduler.requests = requests
    scheduler.running = running
    scheduler.waiting = waiting
    return scheduler


@pytest.mark.parametrize(("scheduler_cls", "scheduler_mod"), _SCHEDULER_PARAMS)
def test_finish_requests_purges_finished_request_from_running(
    monkeypatch: pytest.MonkeyPatch,
    scheduler_cls,
    scheduler_mod,
) -> None:
    """A request whose ``status`` is a finished value but still lives
    in ``self.running`` after ``super().finish_requests`` returns must
    be swept by the post-finish purge. Mocks ``super()`` to simulate
    the corner case where upstream's status-driven removal missed the
    entry.
    """
    finished = _StubRequest("req-finished", RequestStatus.FINISHED_STOPPED)
    scheduler = _make_scheduler(
        scheduler_cls,
        requests={finished.request_id: finished},
        running=[finished],
        waiting=[],
    )

    def fake_finish_requests(self, request_ids, finished_status):
        # super() returning the abort tuple but leaving self.running untouched
        return [(finished.request_id, 0)]

    monkeypatch.setattr(scheduler_mod.VLLMScheduler, "finish_requests", fake_finish_requests)

    result = scheduler_cls.finish_requests(
        scheduler,
        [finished.request_id],
        RequestStatus.FINISHED_STOPPED,
    )

    assert scheduler.running == []
    assert result == [(finished.request_id, 0)]


@pytest.mark.parametrize(("scheduler_cls", "scheduler_mod"), _SCHEDULER_PARAMS)
def test_finish_requests_purges_untracked_request_from_running(
    monkeypatch: pytest.MonkeyPatch,
    scheduler_cls,
    scheduler_mod,
) -> None:
    """A request that has been deleted from ``self.requests`` (e.g.
    after upstream's ``_free_request`` ran but a status mismatch
    caused the ``self.running`` removal branch to no-op) must be
    swept by the post-finish purge.
    """
    untracked = _StubRequest("req-untracked", RequestStatus.RUNNING)
    scheduler = _make_scheduler(
        scheduler_cls,
        requests={},  # already deleted from self.requests
        running=[untracked],
        waiting=[],
    )

    def fake_finish_requests(self, request_ids, finished_status):
        return [(untracked.request_id, 0)]

    monkeypatch.setattr(scheduler_mod.VLLMScheduler, "finish_requests", fake_finish_requests)

    result = scheduler_cls.finish_requests(
        scheduler,
        [untracked.request_id],
        RequestStatus.FINISHED_STOPPED,
    )

    assert scheduler.running == []
    assert result == [(untracked.request_id, 0)]


@pytest.mark.parametrize(("scheduler_cls", "scheduler_mod"), _SCHEDULER_PARAMS)
def test_finish_requests_leaves_healthy_running_intact(
    monkeypatch: pytest.MonkeyPatch,
    scheduler_cls,
    scheduler_mod,
) -> None:
    """A live ``RUNNING`` request that is not in the finish set must
    not be swept by the defensive purge -- this pins that the predicate
    is conservative and won't aggressively drop healthy entries.
    """
    alive = _StubRequest("req-alive", RequestStatus.RUNNING)
    leaving_id = "req-leaving"
    scheduler = _make_scheduler(
        scheduler_cls,
        requests={alive.request_id: alive},
        running=[alive],  # leaving already removed by super() (typical happy path)
        waiting=[],
    )

    def fake_finish_requests(self, request_ids, finished_status):
        return [(leaving_id, 0)]

    monkeypatch.setattr(scheduler_mod.VLLMScheduler, "finish_requests", fake_finish_requests)

    scheduler_cls.finish_requests(
        scheduler,
        [leaving_id],
        RequestStatus.FINISHED_STOPPED,
    )

    assert scheduler.running == [alive]
