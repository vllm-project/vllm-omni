# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Capacity guard on the Talker K-step draft drop (PR #7929 review).

A busy stream keeps a request queued on most steps. Treating "something is
waiting" as "a prefill row joins this step" cleared the Talker's continuation
drafts on nearly every decode step, so a large share of steps ran the
single-frame path. vLLM only admits a waiting request while
``len(running) + num_waiting_for_streaming_input < max_num_running_reqs``
(vllm v1/core/sched/scheduler.py:864), so a full running batch admits
nothing and its uniform decode spans may keep the drafts.
"""

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _req(spec_token_ids: list[int], *, num_computed_tokens: int = 100, num_tokens: int = 101) -> SimpleNamespace:
    return SimpleNamespace(
        spec_token_ids=list(spec_token_ids),
        num_computed_tokens=num_computed_tokens,
        num_tokens=num_tokens,
        prompt_token_ids=[0] * 50,
    )


def _sched(*, running: list, waiting: list, max_num_running_reqs: int | None, num_spec_tokens: int = 7):
    sched = SimpleNamespace(
        _talker_kstep_armed=lambda: True,
        num_spec_tokens=num_spec_tokens,
        max_model_len=4096,
        num_waiting_for_streaming_input=0,
        running=running,
        waiting=waiting,
    )
    if max_num_running_reqs is not None:
        sched.max_num_running_reqs = max_num_running_reqs
    sched._talker_waiting_prefill_may_run = lambda: OmniARScheduler._talker_waiting_prefill_may_run(sched)
    return sched


def test_waiting_admitted_drops_the_drafts():
    """Capacity available: the waiting prefill joins the step, spans would be
    uneven, so the K-step drafts must be dropped."""
    running = [_req([1, 2, 3, 4, 5, 6, 7]), _req([1, 2, 3, 4, 5, 6, 7])]
    sched = _sched(running=running, waiting=[object()], max_num_running_reqs=8)
    assert sched._talker_waiting_prefill_may_run() is True
    OmniARScheduler._drop_talker_drafts_if_prefill_pending(sched)
    assert all(req.spec_token_ids == [] for req in running)


def test_full_running_batch_keeps_the_drafts():
    """Capacity full: nothing new is admitted, the step is a pure decode batch
    with uniform spans, and the K-step drafts survive."""
    running = [_req([1, 2, 3, 4, 5, 6, 7]) for _ in range(4)]
    sched = _sched(running=running, waiting=[object()], max_num_running_reqs=4)
    assert sched._talker_waiting_prefill_may_run() is False
    OmniARScheduler._drop_talker_drafts_if_prefill_pending(sched)
    assert all(req.spec_token_ids == [1, 2, 3, 4, 5, 6, 7] for req in running)


def test_waiting_for_streaming_input_counts_towards_capacity():
    """vLLM counts the streaming-waiting requests in the same guard, so they
    can fill the batch on their own."""
    running = [_req([1, 2, 3]) for _ in range(3)]
    sched = _sched(running=running, waiting=[object()], max_num_running_reqs=4)
    sched.num_waiting_for_streaming_input = 1
    assert sched._talker_waiting_prefill_may_run() is False
    OmniARScheduler._drop_talker_drafts_if_prefill_pending(sched)
    assert all(req.spec_token_ids == [1, 2, 3] for req in running)


def test_no_waiting_keeps_the_drafts():
    running = [_req([1, 2, 3, 4, 5, 6, 7])]
    sched = _sched(running=running, waiting=[], max_num_running_reqs=8)
    assert sched._talker_waiting_prefill_may_run() is False
    OmniARScheduler._drop_talker_drafts_if_prefill_pending(sched)
    assert running[0].spec_token_ids == [1, 2, 3, 4, 5, 6, 7]


def test_chunked_prefill_row_in_running_still_drops_the_drafts():
    """A running request that is still consuming its prompt (a streaming or
    chunked input) puts a foreign row width in the batch regardless of the
    waiting queue, so the drafts go."""
    running = [
        _req([1, 2, 3, 4, 5, 6, 7]),
        _req([1, 2, 3, 4, 5, 6, 7], num_computed_tokens=10, num_tokens=60),
    ]
    sched = _sched(running=running, waiting=[], max_num_running_reqs=8)
    OmniARScheduler._drop_talker_drafts_if_prefill_pending(sched)
    assert all(req.spec_token_ids == [] for req in running)


def test_unknown_capacity_keeps_the_pre_drop_behaviour():
    """No capacity information: assume the waiting request joins, which is
    the direction that cannot feed the runner a mixed span."""
    running = [_req([1, 2, 3, 4, 5, 6, 7])]
    sched = _sched(running=running, waiting=[object()], max_num_running_reqs=None)
    assert sched._talker_waiting_prefill_may_run() is True
    OmniARScheduler._drop_talker_drafts_if_prefill_pending(sched)
    assert running[0].spec_token_ids == []
