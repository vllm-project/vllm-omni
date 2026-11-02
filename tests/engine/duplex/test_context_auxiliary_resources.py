# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise actual session/manager/runtime ownership with a recording stage."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams

from tests.engine.duplex.test_session_auxiliary import wait_until
from tests.engine.duplex.test_session_manager import FakeClock
from tests.engine.duplex.test_session_runner import Harness, close_harness, open_harness
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.commands import CreateItem, DeleteItem, Heartbeat
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, duplex_is_auxiliary_request
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.memory_types import DuplexContextAuxiliaryPolicy, DuplexMemoryProposalBatch
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexSessionCommandMessage,
    OpenDuplexSessionMessage,
    ResumeDuplexSessionMessage,
)
from vllm_omni.engine.duplex.plugin import DuplexAuxiliaryRequest, duplex_history_digest
from vllm_omni.engine.duplex.session.auxiliary_budget import DuplexAuxiliaryBudgetLimits

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]


class ScopedHarness(Harness):
    def submit(self, command):
        self.manager.dispatch(DuplexSessionCommandMessage(session_id=self.session.session_id, command=command))


def limits(**changes):
    return DuplexAuxiliaryBudgetLimits(
        **dict({"max_input_tokens": 100, "max_output_tokens": 10, "idle_s": 0, "cooldown_s": 0}, **changes)
    )


def submissions(h):
    return [
        s
        for s in h.port.submissions
        if duplex_is_auxiliary_request(s.context.request_id) and s.context.session_id == h.session.session_id
    ]


def finish(h, *, accepted=True):
    request_id = submissions(h)[-1].context.request_id
    h.deliver(
        SimpleNamespace(request_id=request_id, accepted=accepted, finished=True), stage_id=0, segment_finished=True
    )
    return request_id


def account(h, policy):
    return h.session.context_compute_budget(policy)


def active(h):
    pool = h.manager._auxiliary_permits
    return 0 if pool is None else pool.active_count


def install(h, mocker, policy, *, retries=()):
    plugin = h.runner.plugin
    work = SimpleNamespace(
        contexts=[],
        decodes=[],
        entered=asyncio.Event(),
        gate=None,
        prompt={"prompt_token_ids": list(range(100))},
        enabled=True,
    )
    mocker.patch.object(plugin, "supports_context_auxiliary_requests", True)
    mocker.patch.object(
        plugin,
        "context_auxiliary_policy",
        side_effect=lambda *, runtime_config: DuplexContextAuxiliaryPolicy(
            max_sources=1,
            max_chars=512,
            allow_during_response=True,
            compute_budget=policy,
        )
        if work.enabled
        else None,
    )

    async def prepare(*, context, runtime_config):
        work.contexts.append(context)
        work.entered.set()
        if work.gate is not None:
            await work.gate.wait()
        return DuplexAuxiliaryRequest(
            stage_id=0,
            prompt=work.prompt,
            sampling_params=SamplingParams(n=1, max_tokens=10, temperature=0),
            history_length=0,
            history_digest=duplex_history_digest(()),
            retry_delays_s=retries,
        )

    def decode(*, request, output):
        work.decodes.append(request)
        return DuplexMemoryProposalBatch(()) if output.accepted else None

    mocker.patch.object(plugin, "prepare_context_auxiliary_request", side_effect=prepare)
    mocker.patch.object(plugin, "decode_context_auxiliary_result", side_effect=decode)
    h.session.register_history_item("old", {"role": "user", "content": "I live in Hangzhou."})
    return work


async def second_session(h):
    settings = h.manager.runtime_config
    buffer = DuplexOutputBuffer(
        max_bytes=settings.max_pending_output_bytes_per_session,
        max_events=settings.max_pending_output_events_per_session,
    )
    await h.manager.handle(
        OpenDuplexSessionMessage(
            control_id="open-second",
            session_id="second",
            session_config=h.session.config,
            output_buffer=buffer,
        )
    )
    assert (await asyncio.wait_for(h.results.get(), 2)).ok
    other = ScopedHarness(h.manager, h.port, h.output, buffer, h.results, h.manager.runners["second"])
    other.session.register_history_item("old", {"role": "user", "content": "I live in Shanghai."})
    return other


async def start(h):
    h.submit(Heartbeat())
    await wait_until(lambda: len(submissions(h)) == 1)


async def test_actual_dispatch_charges_once_and_live_resume_keeps_account(mocker):
    clock, policy = FakeClock(), limits()
    h = await open_harness(auto_response=False, clock=clock)
    try:
        work = install(h, mocker, policy)
        await start(h)
        original, budget = h.session, account(h, policy)
        assert budget.usage(now=clock()) == (100, 10)
        finish(h)
        await wait_until(lambda: len(work.decodes) == 1)
        await h.settle()
        original.detach_lease()
        await h.manager.handle(
            ResumeDuplexSessionMessage(
                control_id="resume-budget",
                session_id=original.session_id,
                expected_lease_generation=original.lease_generation,
            )
        )
        assert (await asyncio.wait_for(h.results.get(), 2)).ok
        assert h.manager.runners[original.session_id].session is original
        assert account(h, policy) is budget and budget.usage(now=clock()) == (100, 10)
        h.session.register_history_item("next", {"role": "user", "content": "Another fact."})
        h.submit(Heartbeat())
        await h.settle(timeout_s=0.1)
        assert len(submissions(h)) == 1 and h.runner.memory._failures == 0
    finally:
        await close_harness(h)


async def test_idle_cooldown_and_window_defer_without_preparation_or_retry(mocker):
    clock, policy = FakeClock(), limits(idle_s=1, cooldown_s=5)
    h = await open_harness(auto_response=False, clock=clock)
    try:
        work = install(h, mocker, policy)
        h.submit(Heartbeat())
        await h.settle(timeout_s=0.1)
        assert not work.contexts and not submissions(h)
        clock.advance(1)
        await start(h)
        finish(h)
        await wait_until(lambda: len(work.decodes) == 1)
        h.session.register_history_item("next", {"role": "user", "content": "A later fact."})
        h.submit(Heartbeat())
        await h.settle(timeout_s=0.1)
        assert len(work.contexts) == 1 and h.runner.memory._failures == 0
        clock.advance(60)
        h.submit(Heartbeat())
        await wait_until(lambda: len(submissions(h)) == 2)
        assert account(h, policy).usage(now=clock()) == (100, 10)
    finally:
        await close_harness(h)


async def test_real_resource_timer_wakes_without_new_user_command(mocker):
    policy = limits(idle_s=0.03)
    h = await open_harness(auto_response=False)
    try:
        install(h, mocker, policy)
        await start(h)
        assert h.runner.memory._failures == 0
        assert active(h) == 1
    finally:
        await close_harness(h)


async def test_ensure_failure_refunds_but_submit_failure_remains_charged(mocker):
    clock, policy = FakeClock(), limits()
    h = await open_harness(auto_response=False, clock=clock)
    try:
        work = install(h, mocker, policy)
        ensure = mocker.patch.object(h.port, "ensure_request", side_effect=RuntimeError("ensure_failed"))
        h.submit(Heartbeat())
        await asyncio.wait_for(work.entered.wait(), 2)
        await wait_until(lambda: h.runner.memory._task.done())
        assert account(h, policy).usage(now=clock()) == (0, 0)
        assert active(h) == 0 and h.runner.memory._failures == 0
        ensure.side_effect = None
        h.port.fail_submit = RuntimeError("submit_failed")
        h.session.register_history_item("old", {"role": "user", "content": "An edited fact."})
        h.submit(Heartbeat())
        await wait_until(lambda: len(work.contexts) == 2 and h.runner.memory._task.done())
        assert account(h, policy).usage(now=clock()) == (100, 10)
        await wait_until(lambda: h.runner.memory._failures == 1)
        assert active(h) == 0
    finally:
        await close_harness(h)


async def test_retry_uses_same_session_allowance_and_finite_failure_budget(mocker):
    policy = limits(max_input_tokens=200, max_output_tokens=20)
    h = await open_harness(auto_response=False, clock=FakeClock())
    try:
        work = install(h, mocker, policy, retries=(0,))
        await start(h)
        finish(h, accepted=False)
        await wait_until(lambda: len(submissions(h)) == 2)
        finish(h, accepted=False)
        await wait_until(lambda: len(work.decodes) == 2)
        await h.settle()
        assert account(h, policy).usage(now=0) == (200, 20)
        assert h.runner.memory._failures == 2 and h.runner.memory._retry_exhausted
        assert len(submissions(h)) == 2
        h.manager._clock.advance(61)
        h.submit(Heartbeat())
        await h.settle(timeout_s=0.1)
        assert len(submissions(h)) == 2 and h.runner.memory._resource_timer is None
    finally:
        await close_harness(h)


async def test_second_session_refunds_while_waiting_and_wakes_after_successful_cleanup(mocker):
    policy = limits()
    h = await open_harness(
        auto_response=False, clock=FakeClock(), runtime_config=DuplexSessionRuntimeConfig(max_sessions=2)
    )
    try:
        work = install(h, mocker, policy)
        other = await second_session(h)
        await start(h)
        other.submit(Heartbeat())
        await wait_until(lambda: len(work.contexts) == 2)
        await other.settle(timeout_s=0.1)
        assert not submissions(other) and account(other, policy).usage(now=0) == (0, 0)
        assert other.runner.memory._failures == 0 and active(h) == 1
        finish(h)
        await wait_until(lambda: len(submissions(other)) == 1)
        assert account(h, policy).usage(now=0) == account(other, policy).usage(now=0) == (100, 10)
        assert active(h) == 1
    finally:
        await close_harness(h)


@pytest.mark.parametrize("extra_failure", [False, True])
async def test_failed_cleanup_retains_permit_until_close_or_reaper_succeeds(mocker, extra_failure):
    clock, policy = FakeClock(), limits()
    h = await open_harness(auto_response=False, clock=clock, runtime_config=DuplexSessionRuntimeConfig(max_sessions=2))
    try:
        work = install(h, mocker, policy)
        other = await second_session(h)
        await start(h)
        h.port.cleanup_failures_remaining = 2 if extra_failure else 1
        request_id = finish(h)
        await wait_until(lambda: h.runner.memory._task.done())
        assert active(h) == 1 and request_id in h.session.resource_request_ids()
        assert not work.decodes
        other.submit(Heartbeat())
        await other.settle(timeout_s=0.1)
        assert not submissions(other)
        await h.manager.close(CloseDuplexSessionMessage(control_id="close-first", session_id=h.session.session_id))
        if extra_failure:
            assert not submissions(other) and active(h) == 1
            clock.advance(1)
            await h.manager.reap_expired()
        await wait_until(lambda: len(submissions(other)) == 1)
        assert request_id not in h.manager._auxiliary_request_permits
        assert account(h, policy).usage(now=clock()) == (100, 10)
    finally:
        await close_harness(h)


async def test_close_during_submit_keeps_charge_and_releases_after_abort(mocker):
    policy, clock = limits(), FakeClock()
    h = await open_harness(auto_response=False, clock=clock)
    try:
        install(h, mocker, policy)
        h.port.submit_gate = asyncio.Event()
        h.submit(Heartbeat())
        await asyncio.wait_for(h.port.submit_started.wait(), 2)
        budget = account(h, policy)
        assert budget.usage(now=0) == (100, 10) and active(h) == 1
        await h.manager.close(CloseDuplexSessionMessage(control_id="close-blocked", session_id=h.session.session_id))
        assert budget.usage(now=0) == (100, 10)
        assert active(h) == 0 and not h.manager._auxiliary_request_permits
        assert budget.try_reserve(1, 1, now=0) is None
        assert h.runner.memory._resource_timer is None
    finally:
        await close_harness(h)


async def test_foreground_preempts_preparation_without_spending_or_waiting_for_permit(mocker):
    policy = limits()
    h = await open_harness(auto_response=False, clock=FakeClock())
    try:
        work = install(h, mocker, policy)
        work.gate = asyncio.Event()
        mocker.patch.object(h.runner.plugin, "auxiliary_preempted_by_input", return_value=True)
        mocker.patch.object(
            h.runner.plugin, "prepare_append_plan", return_value=DuplexAppendPlan(prompt={"prompt_token_ids": [1]})
        )
        h.submit(Heartbeat())
        await asyncio.wait_for(work.entered.wait(), 2)
        await h.run(
            CreateItem(
                item={
                    "id": "question",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "A question."}],
                }
            )
        )
        append = await h.runner._start_append({"video_frames": [b"frame"]}, final=True, precreate_response=True)
        assert await append
        await h.settle(timeout_s=0.1)
        assert not submissions(h) and account(h, policy).usage(now=0) == (0, 0)
        assert active(h) == 0 and h.runner.memory._resource_timer is None
        assert h.port.submissions  # The foreground request did reach the stage.
    finally:
        await close_harness(h)


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "unmeasured text"},
        {"prompt_token_ids": [True]},
        {"prompt_token_ids": [1], "multi_modal_data": {}},
        {"prompt_token_ids": []},
    ],
)
async def test_unmeasured_or_invalid_token_payload_never_submits_or_spends(mocker, payload):
    policy = limits()
    h = await open_harness(auto_response=False, clock=FakeClock())
    try:
        work = install(h, mocker, policy)
        work.prompt = payload
        h.submit(Heartbeat())
        await asyncio.wait_for(work.entered.wait(), 2)
        await wait_until(lambda: h.runner.memory._task.done())
        assert not submissions(h) and account(h, policy).usage(now=0) == (0, 0)
        assert active(h) == 0 and h.runner.memory._failures == 0
    finally:
        await close_harness(h)


async def test_limit_changes_do_not_reset_session_spending(mocker):
    policy = limits()
    h = await open_harness(auto_response=False, clock=FakeClock())
    try:
        install(h, mocker, policy)
        await start(h)
        budget = account(h, policy)
        with pytest.raises(ValueError, match="auxiliary_budget_limits_are_session_fixed"):
            account(h, replace(policy, max_input_tokens=200))
        assert account(h, policy) is budget and budget.usage(now=0) == (100, 10)
    finally:
        await close_harness(h)


async def test_no_compute_policy_preserves_existing_context_execution(mocker):
    h = await open_harness(auto_response=False)
    try:
        from tests.engine.duplex.test_context_auxiliary_runtime import install_context_work

        install_context_work(h, mocker, empty=True)
        await start(h)
        assert h.session._memory_compute_budget is None
        assert h.manager._auxiliary_permits is None
        assert h.runner.memory._resource_timer is None
    finally:
        await close_harness(h)


async def test_session_only_finalization_cannot_release_an_uncovered_forward(mocker):
    """A legacy late callback carries no proof that this request was cleaned."""
    h = await open_harness(auto_response=False, clock=FakeClock())
    try:
        install(h, mocker, limits())
        await start(h)
        request_id = submissions(h)[0].context.request_id
        h.manager.finalize_closed_sessions([h.session.session_id])
        assert active(h) == 1
        assert request_id in h.manager._auxiliary_request_permits
        assert not any(request_id in ids for ids, _ in h.port.cleanups)
    finally:
        await close_harness(h)


async def test_orchestrator_cleanup_rpc_releases_only_after_success(mocker):
    from tests.engine.test_duplex_orchestrator import _build
    from vllm_omni.engine.duplex.contracts import DuplexFence, duplex_resource_request_id
    from vllm_omni.engine.orchestrator import Orchestrator

    orchestrator, _, _, _ = _build()
    manager = orchestrator.session_manager
    request_id = duplex_resource_request_id(DuplexFence("scope", epoch=0), "auxiliary-cleanup")
    try:
        assert manager.try_acquire_auxiliary_permit(request_id)
        assert manager.mark_auxiliary_submitted(request_id)
        cleanup = mocker.patch.object(Orchestrator, "_cleanup_request_ids", new_callable=mocker.AsyncMock)
        cleanup.side_effect = RuntimeError("rpc_failed")
        with pytest.raises(RuntimeError, match="rpc_failed"):
            await orchestrator._cleanup_request_ids([request_id], abort=True)
        assert manager._auxiliary_permits.active_count == 1
        cleanup.side_effect = None
        cleanup.return_value = []
        assert await orchestrator._cleanup_request_ids([request_id], abort=True) == []
        assert manager._auxiliary_permits.active_count == 0
        assert request_id not in manager._auxiliary_request_permits
    finally:
        await manager.shutdown()


async def test_context_only_epoch_cancellation_restarts_after_cleanup_without_another_command(mocker):
    policy = limits(max_input_tokens=200, max_output_tokens=20)
    h = await open_harness(auto_response=False, clock=FakeClock())
    try:
        install(h, mocker, policy)
        assert not h.runner.plugin.supports_auxiliary_requests
        await start(h)
        first = submissions(h)[0].context.request_id
        h.session.epoch += 1
        h.submit(Heartbeat())
        await wait_until(lambda: len(submissions(h)) == 2)
        assert ([first], True) in h.port.cleanups
        assert submissions(h)[1].context.request_id != first
        assert account(h, policy).usage(now=0) == (200, 20)
        assert h.runner.memory._failures == 0
    finally:
        await close_harness(h)


async def test_waiting_session_makes_progress_while_first_has_more_batches(mocker):
    policy = limits(max_input_tokens=1000, max_output_tokens=100)
    h = await open_harness(
        auto_response=False, clock=FakeClock(), runtime_config=DuplexSessionRuntimeConfig(max_sessions=2)
    )
    try:
        work = install(h, mocker, policy)
        for index in range(5):
            h.session.register_history_item(str(index), {"role": "user", "content": f"Fact {index}."})
        other = await second_session(h)
        await start(h)
        other.submit(Heartbeat())
        await wait_until(lambda: len(work.contexts) == 2)
        finish(h)
        for index in range(4):
            await wait_until(lambda: len(submissions(h)) + len(submissions(other)) >= index + 2)
            if submissions(other):
                break
            finish(h)
        assert submissions(other)
        assert active(h) == 1
    finally:
        await close_harness(h)


async def test_other_session_foreground_does_not_wait_for_private_permit(mocker):
    h = await open_harness(
        auto_response=False, clock=FakeClock(), runtime_config=DuplexSessionRuntimeConfig(max_sessions=2)
    )
    try:
        install(h, mocker, limits())
        other = await second_session(h)
        await start(h)
        private_id = submissions(h)[0].context.request_id
        mocker.patch.object(h.runner.plugin, "auxiliary_preempted_by_input", return_value=True)
        mocker.patch.object(
            h.runner.plugin, "prepare_append_plan", return_value=DuplexAppendPlan(prompt={"prompt_token_ids": [1]})
        )
        await other.run(
            CreateItem(
                item={
                    "id": "question",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "A foreground question."}],
                }
            )
        )
        append = await other.runner._start_append({"video_frames": [b"frame"]}, final=True, precreate_response=True)
        assert await asyncio.wait_for(append, 2)
        assert any(
            s.context.session_id == other.session.session_id and not duplex_is_auxiliary_request(s.context.request_id)
            for s in h.port.submissions
        )
        assert active(h) == 1 and private_id in h.manager._auxiliary_request_permits
    finally:
        await close_harness(h)


async def test_close_cancels_deferred_resource_timer_and_cannot_revive_work(mocker):
    h = await open_harness(auto_response=False)
    try:
        work = install(h, mocker, limits(idle_s=0.3))
        h.submit(Heartbeat())
        await h.settle(timeout_s=0.1)
        assert not work.contexts and h.runner.memory._resource_timer is not None
        await h.manager.close(CloseDuplexSessionMessage(control_id="close-deferred", session_id=h.session.session_id))
        await asyncio.sleep(0.35)
        assert not work.contexts and not submissions(h)
        assert h.runner.memory._resource_timer is None
        assert not h.manager._auxiliary_request_permits
    finally:
        await close_harness(h)


@pytest.mark.parametrize("foreground", ["pending", "preparing"])
async def test_foreground_arriving_during_auxiliary_prepare_defers_dispatch_and_keeps_progress(mocker, foreground):
    from vllm_omni.engine.duplex.plugin import DuplexInputCapacityError

    h = await open_harness(auto_response=False, clock=FakeClock())
    gate = asyncio.Event()
    try:
        policy = limits()
        work = install(h, mocker, policy)
        work.gate = asyncio.Event()
        mocker.patch.object(h.runner.plugin, "auxiliary_preempted_by_input", return_value=True)

        async def rejected_foreground(*args, **kwargs):
            await gate.wait()
            raise DuplexInputCapacityError("foreground_capacity")

        mocker.patch.object(h.runner.plugin, "prepare_append_plan", side_effect=rejected_foreground)
        h.submit(Heartbeat())
        await asyncio.wait_for(work.entered.wait(), 2)
        h.submit(
            CreateItem(
                item={
                    "id": "pending",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "A new foreground question."}],
                }
            )
        )
        await wait_until(lambda: h.runner.memory._interaction_pending)
        operation = h.runner.memory._generation
        append = None
        if foreground == "preparing":
            append = await h.runner._start_append({"video_frames": [b"frame"]}, final=True, precreate_response=True)
            await wait_until(lambda: h.runner.memory._interaction_active)
        assert h.runner.memory._generation == operation  # No successful append preemption yet.
        work.gate.set()
        await h.settle(timeout_s=0.2)
        assert not submissions(h) and account(h, policy).usage(now=0) == (0, 0)
        assert active(h) == 0 and h.runner.memory._failures == 0
        if append is not None:
            gate.set()
            assert not await append
        else:
            h.submit(DeleteItem(item_id="pending"))
        await wait_until(lambda: len(submissions(h)) == 1)
        assert account(h, policy).usage(now=0) == (100, 10)
        assert h.runner.memory._failures == 0
    finally:
        gate.set()
        await close_harness(h)


async def test_activity_after_reservation_rechecks_idle_before_dispatch(mocker):
    clock, policy = FakeClock(), limits(idle_s=1)
    h = await open_harness(auto_response=False, clock=clock)
    try:
        work = install(h, mocker, policy)
        h.submit(Heartbeat())
        await h.settle(timeout_s=0.1)
        original = h.port.ensure_request
        noted = False

        def activity_between_reservation_and_submit(context):
            nonlocal noted
            original(context)
            if duplex_is_auxiliary_request(context.request_id) and not noted:
                noted = True
                h.session.note_auxiliary_activity()

        mocker.patch.object(h.port, "ensure_request", side_effect=activity_between_reservation_and_submit)
        clock.advance(1)
        h.submit(Heartbeat())
        await wait_until(lambda: len(work.contexts) == 1)
        await h.settle(timeout_s=0.2)
        assert noted and not submissions(h)
        assert account(h, policy).usage(now=clock()) == (0, 0)
        assert active(h) == 0 and h.runner.memory._failures == 0
        assert h.runner.memory._resource_timer is not None
        clock.advance(1)
        h.submit(Heartbeat())
        await wait_until(lambda: len(submissions(h)) == 1)
        assert account(h, policy).usage(now=clock()) == (100, 10)
    finally:
        await close_harness(h)
