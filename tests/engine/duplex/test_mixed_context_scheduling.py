# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Both memory policies share one real runner slot and one finite allowance."""

import asyncio
from collections import Counter
from types import SimpleNamespace

import pytest
from vllm import SamplingParams

from tests.engine.duplex.test_context_auxiliary_resources import second_session
from tests.engine.duplex.test_session_auxiliary import wait_until
from tests.engine.duplex.test_session_manager import FakeClock
from tests.engine.duplex.test_session_runner import close_harness, open_harness
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.commands import Heartbeat
from vllm_omni.engine.duplex.contracts import duplex_is_auxiliary_request
from vllm_omni.engine.duplex.memory_types import DuplexContextAuxiliaryPolicy, DuplexMemoryProposalBatch
from vllm_omni.engine.duplex.messages import CloseDuplexSessionMessage, ResumeDuplexSessionMessage
from vllm_omni.engine.duplex.plugin import DuplexAuxiliaryPolicy, DuplexAuxiliaryRequest, duplex_history_digest
from vllm_omni.engine.duplex.session.auxiliary_budget import DuplexAuxiliaryBudgetLimits

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]


def _submissions(h):
    return [
        s
        for s in h.port.submissions
        if duplex_is_auxiliary_request(s.context.request_id) and s.context.session_id == h.session.session_id
    ]


def _kind(submission):
    return "summary" if submission.prompt["prompt_token_ids"][0] == 11 else "card"


def _active(h):
    pool = h.manager._auxiliary_permits
    return pool.active_count if pool is not None else 0


def _install(h, mocker, *, summary=True, card=True, retries=(), timeout_s=30.0, budget=None):
    plugin = h.runner.plugin
    work = SimpleNamespace(summary=summary, card=card, summary_version=0, prepared=[], applied=[], fail_prepare=set())
    limits = budget or DuplexAuxiliaryBudgetLimits(
        max_input_tokens=1000, max_output_tokens=1000, idle_s=0, cooldown_s=0
    )
    mocker.patch.object(plugin, "supports_auxiliary_requests", True)
    mocker.patch.object(plugin, "supports_context_auxiliary_requests", True)
    mocker.patch.object(plugin, "auxiliary_budget_limits", return_value=limits)
    mocker.patch.object(
        plugin,
        "auxiliary_policy",
        side_effect=lambda **kwargs: DuplexAuxiliaryPolicy(("summary", work.summary_version), True)
        if work.summary
        else None,
    )
    mocker.patch.object(
        plugin,
        "context_auxiliary_policy",
        side_effect=lambda **kwargs: DuplexContextAuxiliaryPolicy(allow_during_response=True, compute_budget=limits)
        if work.card
        else None,
    )

    def request(kind):
        work.prepared.append(kind)
        if kind in work.fail_prepare:
            raise RuntimeError("injected_prepare_failure")
        return DuplexAuxiliaryRequest(
            stage_id=0,
            prompt={"prompt_token_ids": [11 if kind == "summary" else 22] * 10},
            sampling_params=SamplingParams(max_tokens=5),
            history_length=0,
            history_digest=duplex_history_digest(()),
            retry_delays_s=retries,
            timeout_s=timeout_s,
        )

    async def summary_request(**kwargs):
        return request("summary")

    async def card_request(**kwargs):
        return request("card")

    def apply(*, output, **kwargs):
        work.applied.append("summary")
        if output.accepted:
            work.summary = False
        return output.accepted

    def decode(*, output, **kwargs):
        work.applied.append("card")
        return DuplexMemoryProposalBatch(()) if output.accepted else None

    mocker.patch.object(plugin, "prepare_auxiliary_request", side_effect=summary_request)
    mocker.patch.object(plugin, "prepare_context_auxiliary_request", side_effect=card_request)
    mocker.patch.object(plugin, "auxiliary_source_matches", return_value=True)
    mocker.patch.object(plugin, "apply_auxiliary_result", side_effect=apply)
    mocker.patch.object(plugin, "decode_context_auxiliary_result", side_effect=decode)
    h.session.register_history_item("fact", {"role": "user", "content": "The user likes tea."})
    return work, limits


def _finish(h, submission, *, accepted=True):
    h.deliver(
        SimpleNamespace(request_id=submission.context.request_id, accepted=accepted, finished=True),
        stage_id=0,
        segment_finished=True,
    )


@pytest.mark.parametrize("summary,card", [(False, False), (True, False), (False, True), (True, True)])
async def test_ready_policies_share_one_slot_without_starvation(mocker, summary, card):
    h = await open_harness(auto_response=False)
    try:
        work, limits = _install(h, mocker, summary=summary, card=card)
        h.submit(Heartbeat())
        expected = int(summary) + int(card)
        for index in range(expected):
            await wait_until(lambda: len(_submissions(h)) == index + 1)
            assert _active(h) == 1
            _finish(h, _submissions(h)[index])
        await h.settle()
        kinds = [_kind(s) for s in _submissions(h)]
        assert Counter(kinds) == Counter({kind: 1 for kind, ready in (("summary", summary), ("card", card)) if ready})
        assert _active(h) == 0
        if expected:
            assert h.session.context_compute_budget(limits).usage(now=h.manager._clock()) == (
                10 * expected,
                5 * expected,
            )
            assert Counter(work.applied) == Counter(kinds)
    finally:
        await close_harness(h)


@pytest.mark.parametrize("failure", ["output", "submit", "timeout"])
async def test_kind_switch_and_epoch_do_not_reset_exhausted_work(mocker, failure):
    h = await open_harness(auto_response=False)
    try:
        work, limits = _install(h, mocker, retries=(0,), timeout_s=0.05 if failure == "timeout" else 30.0)
        if failure == "submit":
            original = h.port.submit

            async def fail_after_submission(submission):
                await original(submission)
                raise RuntimeError("injected_submit_failure")

            mocker.patch.object(h.port, "submit", side_effect=fail_after_submission)
        h.submit(Heartbeat())
        if failure == "output":
            for index in range(4):
                await wait_until(lambda: len(_submissions(h)) == index + 1)
                _finish(h, _submissions(h)[index], accepted=False)
        else:
            await wait_until(lambda: len(_submissions(h)) >= 4)
        await h.settle()
        assert [_kind(s) for s in _submissions(h)] == ["summary", "card", "summary", "card"]
        assert h.session.context_compute_budget(limits).usage(now=h.manager._clock()) == (40, 20)

        work.summary = work.card = False
        h.submit(Heartbeat())
        await h.settle()
        work.summary = work.card = True
        h.session.epoch += 1
        h.submit(Heartbeat())
        await h.settle()
        assert len(_submissions(h)) == 4

        work.summary_version += 1
        h.submit(Heartbeat())
        expected = 5 if failure == "output" else 6
        await wait_until(lambda: len(_submissions(h)) >= expected)
        await h.settle()
        assert len(_submissions(h)) == expected
        assert _kind(_submissions(h)[-1]) == "summary"
    finally:
        await close_harness(h)


async def test_other_kind_waits_for_cleanup_before_reusing_slot(mocker):
    h = await open_harness(auto_response=False)
    entered, release = asyncio.Event(), asyncio.Event()
    original = h.port.cleanup

    async def cleanup(request_ids, *, abort=False):
        entered.set()
        await release.wait()
        await original(request_ids, abort=abort)

    try:
        _install(h, mocker)
        mocker.patch.object(h.port, "cleanup", side_effect=cleanup)
        h.submit(Heartbeat())
        await wait_until(lambda: len(_submissions(h)) == 1)
        first = _submissions(h)[0]
        _finish(h, first)
        await asyncio.wait_for(entered.wait(), 2)
        await h.settle()
        assert len(_submissions(h)) == 1 and _active(h) == 1
        release.set()
        await wait_until(lambda: len(_submissions(h)) == 2)
        assert _kind(_submissions(h)[-1]) != _kind(first)
        assert h.port.cleanups.count(([first.context.request_id], True)) == 1
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.parametrize("failed_kind", ["summary", "card"])
async def test_cleanup_failure_blocks_both_kinds_and_keeps_charge(mocker, failed_kind):
    h = await open_harness(auto_response=False)
    try:
        work, limits = _install(h, mocker, summary=failed_kind == "summary", card=failed_kind == "card")
        h.port.cleanup_failures_remaining = 1
        h.submit(Heartbeat())
        await wait_until(lambda: len(_submissions(h)) == 1)
        submission = _submissions(h)[0]
        _finish(h, submission)
        await wait_until(lambda: h.runner.memory._task.done())
        work.summary = work.card = True
        h.submit(Heartbeat())
        await h.settle()
        assert len(_submissions(h)) == 1
        assert _active(h) == 1
        assert not work.applied
        assert submission.context.request_id in h.session.resource_request_ids()
        assert h.session.context_compute_budget(limits).usage(now=h.manager._clock()) == (10, 5)
    finally:
        await close_harness(h)


async def test_pending_foreground_preempts_and_pauses_both_kinds(mocker):
    h = await open_harness(auto_response=False)
    try:
        work, limits = _install(h, mocker)
        mocker.patch.object(h.runner.plugin, "auxiliary_preempted_by_input", return_value=True)
        h.submit(Heartbeat())
        await wait_until(lambda: len(_submissions(h)) == 1)
        first = _submissions(h)[0]
        task = h.runner.memory._task
        h.runner.memory.note_committed_input({}, has_new_user_input=True)
        h.runner.memory.preempt_for_submission()
        await wait_until(task.done)
        h.submit(Heartbeat())
        await h.settle()
        assert len(_submissions(h)) == 1
        assert not work.applied and _active(h) == 0
        assert h.session.context_compute_budget(limits).usage(now=h.manager._clock()) == (10, 5)

        h.session.epoch += 1
        h.runner.memory.forget_committed_input()
        h.submit(Heartbeat())
        await wait_until(lambda: len(_submissions(h)) == 2)
        assert _kind(_submissions(h)[-1]) != _kind(first)
        assert h.session.context_compute_budget(limits).usage(now=h.manager._clock()) == (20, 10)
    finally:
        await close_harness(h)


@pytest.mark.parametrize("failed_kind", ["summary", "card"])
async def test_prepare_failure_parks_only_its_kind_without_charging(mocker, failed_kind):
    h = await open_harness(auto_response=False)
    try:
        work, limits = _install(h, mocker)
        work.fail_prepare.add(failed_kind)
        h.submit(Heartbeat())
        await wait_until(lambda: len(_submissions(h)) == 1)
        assert _kind(_submissions(h)[0]) != failed_kind
        _finish(h, _submissions(h)[0])
        await h.settle()
        h.submit(Heartbeat())
        await h.settle()
        assert work.prepared.count(failed_kind) == 1
        assert len(_submissions(h)) == 1
        assert h.session.context_compute_budget(limits).usage(now=h.manager._clock()) == (10, 5)
    finally:
        await close_harness(h)


async def test_sessions_keep_separate_failure_budgets_while_sharing_permit(mocker):
    h = await open_harness(auto_response=False, runtime_config=DuplexSessionRuntimeConfig(max_sessions=2))
    other = None
    try:
        _, limits = _install(h, mocker, card=False)
        other = await second_session(h)
        h.submit(Heartbeat())
        await wait_until(lambda: len(_submissions(h)) == 1)
        other.submit(Heartbeat())
        await wait_until(lambda: other.runner.memory._resource_waiting)
        assert not _submissions(other)
        assert other.session.context_compute_budget(limits).usage(now=h.manager._clock()) == (0, 0)
        _finish(h, _submissions(h)[0], accepted=False)
        await wait_until(lambda: len(_submissions(other)) == 1)
        assert len(_submissions(h)) == 1 and _active(h) == 1
        assert other.session.context_compute_budget(limits).usage(now=h.manager._clock()) == (10, 5)
        assert h.session.context_compute_budget(limits) is not other.session.context_compute_budget(limits)
    finally:
        await close_harness(h)


async def test_deferred_kind_keeps_next_window_priority_under_continuous_work(mocker):
    clock = FakeClock()
    h = await open_harness(auto_response=False, clock=clock)
    try:
        limits = DuplexAuxiliaryBudgetLimits(
            max_input_tokens=15, max_output_tokens=10, window_s=1, cooldown_s=0, idle_s=0
        )
        work, _ = _install(h, mocker, budget=limits)

        def keep_summary_ready(**kwargs):
            work.summary_version += 1
            return True

        mocker.patch.object(h.runner.plugin, "apply_auxiliary_result", side_effect=keep_summary_ready)
        h.submit(Heartbeat())
        await wait_until(lambda: len(_submissions(h)) == 1)
        assert _kind(_submissions(h)[0]) == "summary"
        _finish(h, _submissions(h)[0])
        await wait_until(lambda: h.runner.memory._resource_waiting)
        assert work.prepared == ["summary", "card"]
        clock.advance(1)
        h.runner.memory.resources_available()
        h.submit(Heartbeat())
        await wait_until(lambda: len(_submissions(h)) == 2)
        assert _kind(_submissions(h)[1]) == "card"
        assert h.session.context_compute_budget(limits).usage(now=clock()) == (10, 5)
    finally:
        await close_harness(h)


async def test_live_resume_preserves_account_and_exhausted_kind(mocker):
    h = await open_harness(auto_response=False)
    try:
        _, limits = _install(h, mocker, card=False)
        h.submit(Heartbeat())
        await wait_until(lambda: len(_submissions(h)) == 1)
        _finish(h, _submissions(h)[0], accepted=False)
        await h.settle()
        original, budget = h.session, h.session.context_compute_budget(limits)
        original.detach_lease()
        await h.manager.handle(
            ResumeDuplexSessionMessage(
                control_id="resume-mixed",
                session_id=original.session_id,
                expected_lease_generation=original.lease_generation,
            )
        )
        assert (await asyncio.wait_for(h.results.get(), 2)).ok
        assert h.manager.runners[original.session_id].session is original
        assert h.session.context_compute_budget(limits) is budget
        h.submit(Heartbeat())
        await h.settle()
        assert len(_submissions(h)) == 1
        assert budget.usage(now=h.manager._clock()) == (10, 5)
    finally:
        await close_harness(h)


async def test_close_cancels_pending_retries_for_both_kinds(mocker):
    h = await open_harness(auto_response=False)
    try:
        _, limits = _install(h, mocker, retries=(1.0,))
        h.submit(Heartbeat())
        for index in range(2):
            await wait_until(lambda: len(_submissions(h)) == index + 1)
            _finish(h, _submissions(h)[index], accepted=False)
        await wait_until(lambda: h.runner.memory._retry_timer is not None)
        budget = h.session.context_compute_budget(limits)
        await h.manager.handle(CloseDuplexSessionMessage(control_id="close-mixed", session_id=h.session.session_id))
        await asyncio.sleep(1.05)
        assert len(_submissions(h)) == 2 and _active(h) == 0
        assert h.runner.memory._retry_timer is None
        assert budget.usage(now=h.manager._clock()) == (20, 10)
    finally:
        await close_harness(h)


async def test_mixed_retry_timer_wakes_without_new_input(mocker):
    h = await open_harness(auto_response=False)
    try:
        _install(h, mocker, retries=(0.02,))
        h.submit(Heartbeat())
        for index in range(4):
            await wait_until(lambda: len(_submissions(h)) == index + 1)
            _finish(h, _submissions(h)[index], accepted=False)
        await h.settle()
        assert [_kind(s) for s in _submissions(h)] == ["summary", "card", "summary", "card"]
        assert _active(h) == 0
    finally:
        await close_harness(h)


@pytest.mark.parametrize("kind", ["summary", "card"])
async def test_input_between_reservation_and_dispatch_does_not_park_unsubmitted_work(mocker, kind):
    h = await open_harness(auto_response=False)
    try:
        _, limits = _install(h, mocker, summary=kind == "summary", card=kind == "card")
        mocker.patch.object(h.runner.plugin, "auxiliary_preempted_by_input", return_value=True)
        original, entered = h.port.ensure_request, asyncio.Event()

        def foreground_arrives(context):
            original(context)
            if duplex_is_auxiliary_request(context.request_id) and not entered.is_set():
                entered.set()
                h.runner.memory.note_committed_input({}, has_new_user_input=True)

        mocker.patch.object(h.port, "ensure_request", side_effect=foreground_arrives)
        h.submit(Heartbeat())
        await asyncio.wait_for(entered.wait(), 2)
        await wait_until(lambda: h.runner.memory._task.done())
        assert not _submissions(h) and _active(h) == 0
        assert h.session.context_compute_budget(limits).usage(now=h.manager._clock()) == (0, 0)
        h.runner.memory.forget_committed_input()
        h.submit(Heartbeat())
        await wait_until(lambda: len(_submissions(h)) == 1)
        assert _kind(_submissions(h)[0]) == kind
        assert h.session.context_compute_budget(limits).usage(now=h.manager._clock()) == (10, 5)
    finally:
        await close_harness(h)
