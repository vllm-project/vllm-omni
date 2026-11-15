# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Versioned context work uses the existing runner, mailbox and resource slot."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.engine.duplex.test_session_auxiliary import auxiliary_submissions, wait_until
from tests.engine.duplex.test_session_runner import close_harness, open_harness
from vllm_omni.engine.duplex.commands import CreateItem, Heartbeat
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan, duplex_is_auxiliary_request
from vllm_omni.engine.duplex.plugin import DuplexAuxiliaryRequest, duplex_history_digest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def install_context_work(h, mocker, *, retry_delays_s=(), retry_cooldown_s=None, empty=False):
    from vllm_omni.engine.duplex.memory_types import (
        DuplexContextAuxiliaryPolicy,
        DuplexMemoryEvidenceSelection,
        DuplexMemoryProposal,
        DuplexMemoryProposalBatch,
    )

    plugin = h.runner.plugin
    work = SimpleNamespace(contexts=[], decodes=[], entered=asyncio.Event(), release=None, enabled=True)
    mocker.patch.object(plugin, "supports_context_auxiliary_requests", True)
    mocker.patch.object(plugin, "supports_auxiliary_requests", True)
    for name in ("prepare_auxiliary_request", "apply_auxiliary_result", "auxiliary_policy", "auxiliary_source_matches"):
        mocker.patch.object(plugin, name, side_effect=AssertionError("legacy hook called"))
    mocker.patch.object(
        plugin,
        "context_auxiliary_policy",
        side_effect=lambda *, runtime_config: DuplexContextAuxiliaryPolicy(allow_during_response=True)
        if work.enabled
        else None,
    )

    async def prepare(*, context, runtime_config):
        work.contexts.append(context)
        work.entered.set()
        if work.release is not None:
            await work.release.wait()
        return DuplexAuxiliaryRequest(
            stage_id=0,
            prompt={"prompt": "bounded private context work"},
            sampling_params=h.manager.sampling_params_for(h.session)[0],
            history_length=0,
            history_digest=duplex_history_digest(()),
            retry_delays_s=retry_delays_s,
            retry_cooldown_s=retry_cooldown_s,
        )

    def decode(*, request, output):
        work.decodes.append((request, tuple(h.port.cleanups), tuple(h.session.request_resources)))
        if not output.accepted:
            return None
        if empty:
            return DuplexMemoryProposalBatch(())
        return DuplexMemoryProposalBatch(
            (
                DuplexMemoryProposal(
                    "user/self/city",
                    "Hangzhou",
                    (DuplexMemoryEvidenceSelection(0, 0, len(work.contexts[-1].evidence[0].text)),),
                ),
            )
        )

    mocker.patch.object(plugin, "prepare_context_auxiliary_request", side_effect=prepare)
    mocker.patch.object(plugin, "decode_context_auxiliary_result", side_effect=decode)
    h.session.register_history_item("old", {"role": "user", "content": "I live in Hangzhou."})
    return work


def finish_context(h, *, accepted=True):
    request_id = auxiliary_submissions(h)[-1].context.request_id
    h.deliver(
        SimpleNamespace(request_id=request_id, accepted=accepted, finished=True), stage_id=0, segment_finished=True
    )
    return request_id


@pytest.mark.asyncio
async def test_context_result_is_decoded_after_cleanup_and_admitted_by_session(mocker):
    h = await open_harness()
    try:
        work = install_context_work(h, mocker)
        before = h.session.history_snapshot()
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        task = h.runner.memory._task
        assert task in h.runner._background_tasks
        request_id = finish_context(h)
        await wait_until(lambda: len(h.session.context_ledger.card_candidates()) == 1)
        request, cleanups, resources = work.decodes[0]
        assert ([request_id], True) in cleanups
        assert not any(duplex_is_auxiliary_request(request_id) for _, request_id in resources)
        assert request.context_work == work.contexts[0].work
        assert h.session.history_snapshot() == before
        # A newly discovered key adds one explicit catalogue coverage page.
        await wait_until(lambda: len(auxiliary_submissions(h)) == 2)
        finish_context(h)
        await wait_until(lambda: len(work.decodes) == 2)
        await h.settle()
        assert len(auxiliary_submissions(h)) == 2
        assert not h.runner.plugin.apply_auxiliary_result.called
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_context_valid_empty_result_consumes_window_without_retry(mocker):
    h = await open_harness()
    try:
        work = install_context_work(h, mocker, empty=True, retry_delays_s=(0,))
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        finish_context(h)
        await wait_until(lambda: len(work.decodes) == 1)
        await h.settle()
        h.submit(Heartbeat())
        await h.settle()
        assert len(auxiliary_submissions(h)) == 1
        assert h.session.context_ledger.card_catalog() == ()
        assert h.session.capture_memory_context() is None
        assert h.runner.memory._failures == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["edit", "delete", "recreate", "epoch", "settings"])
async def test_context_stale_result_is_never_decoded_or_admitted(mocker, change):
    h = await open_harness()
    try:
        work = install_context_work(h, mocker)
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        work.enabled = False
        if change in {"delete", "recreate"}:
            h.session.delete_history_item("old")
        if change in {"edit", "recreate"}:
            h.session.register_history_item("old", {"role": "user", "content": "I live in Shanghai."})
        if change == "epoch":
            h.session.epoch += 1
        if change == "settings":
            h.session.replace_config(replace(h.session.config, voice="another"))
        finish_context(h)
        await h.settle()
        assert not work.decodes
        assert h.session.context_ledger.card_candidates() == ()
        assert not any(duplex_is_auxiliary_request(request_id) for _, request_id in h.session.request_resources)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_context_internal_generation_and_unrelated_append_do_not_fence(mocker):
    h = await open_harness()
    try:
        work = install_context_work(h, mocker)
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        work.enabled = False
        h.session.config_generation += 1
        h.session.register_history_item("tail", {"role": "user", "content": "An unrelated observation."})
        finish_context(h)
        await wait_until(lambda: len(h.session.context_ledger.card_candidates()) == 1)
        assert len(work.decodes) == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_context_failed_source_keeps_retry_budget_across_tail_appends(mocker):
    h = await open_harness()
    try:
        work = install_context_work(h, mocker, retry_delays_s=(0,))
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        for attempt in range(2):
            h.session.register_history_item(f"tail-{attempt}", {"role": "user", "content": "New observation."})
            finish_context(h, accepted=False)
            if attempt == 0:
                await wait_until(lambda: len(auxiliary_submissions(h)) == 2)
        await h.settle()
        for _ in range(3):
            h.submit(Heartbeat())
        await h.settle()
        assert len(auxiliary_submissions(h)) == len(work.decodes) == 2
        assert work.contexts[0].work_key == work.contexts[1].work_key
        assert h.runner.memory._failures == 2
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_context_pending_input_blocks_new_work_without_preempting_existing(mocker):
    h = await open_harness()
    try:
        work = install_context_work(h, mocker, empty=True)
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        request_id = auxiliary_submissions(h)[0].context.request_id
        await h.run(
            CreateItem(
                item={
                    "id": "pending",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Question?"}],
                }
            )
        )
        assert h.runner.memory._interaction_pending
        assert ([request_id], True) not in h.port.cleanups
        finish_context(h)
        await h.settle()
        assert len(work.contexts) == len(auxiliary_submissions(h)) == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_context_prepare_stale_does_not_submit_private_request(mocker):
    h = await open_harness()
    try:
        work = install_context_work(h, mocker)
        work.release = asyncio.Event()
        h.submit(Heartbeat())
        await asyncio.wait_for(work.entered.wait(), timeout=2)
        work.enabled = False
        h.session.delete_history_item("old")
        work.release.set()
        await h.settle()
        assert not auxiliary_submissions(h)
        assert not work.decodes
        assert h.runner.memory._failures == 0
        assert not any(duplex_is_auxiliary_request(request_id) for _, request_id in h.session.request_resources)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_context_opt_out_skips_capture_and_has_no_legacy_fallback(mocker):
    h = await open_harness()
    try:
        work = install_context_work(h, mocker)
        work.enabled = False
        capture = mocker.spy(h.session, "capture_memory_context")
        h.submit(Heartbeat())
        await h.settle()
        assert not capture.called
        assert not work.contexts
        assert not auxiliary_submissions(h)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_context_close_during_cleanup_never_admits_result(mocker):
    h = await open_harness()
    entered, release = asyncio.Event(), asyncio.Event()
    original_cleanup = h.port.cleanup

    async def cleanup(request_ids, *, abort=False):
        entered.set()
        await release.wait()
        await original_cleanup(request_ids, abort=abort)

    try:
        work = install_context_work(h, mocker)
        mocker.patch.object(h.port, "cleanup", side_effect=cleanup)
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        request_id = finish_context(h)
        await asyncio.wait_for(entered.wait(), timeout=2)
        h.runner.memory.close()
        h.runner.memory.close()
        assert request_id in h.session.resource_request_ids()
        release.set()
        await wait_until(lambda: h.runner.memory._task.done())
        assert not work.decodes
        assert not h.session.context_ledger.card_candidates()
        assert h.port.cleanups.count(([request_id], True)) == 1
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_context_failed_cleanup_keeps_resource_and_prevents_admission(mocker):
    h = await open_harness()
    try:
        work = install_context_work(h, mocker)
        h.port.cleanup_failures_remaining = 1
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        request_id = finish_context(h)
        await wait_until(lambda: h.runner.memory._task.done())
        await h.settle()
        assert request_id in h.session.resource_request_ids()
        assert not work.decodes
        assert not h.session.context_ledger.card_candidates()
        h.submit(Heartbeat())
        await h.settle()
        assert len(auxiliary_submissions(h)) == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_delays", [(), (0,)])
async def test_foreground_turn_does_not_replenish_exhausted_context_work(mocker, retry_delays):
    h = await open_harness(auto_response=False)
    try:
        work = install_context_work(h, mocker, retry_delays_s=retry_delays)
        mocker.patch.object(
            h.runner.plugin,
            "auxiliary_preempted_by_input",
            side_effect=lambda *, payload, state, has_new_user_input: has_new_user_input,
        )
        mocker.patch.object(
            h.runner.plugin, "prepare_append_plan", return_value=DuplexAppendPlan(prompt={"prompt_token_ids": [1]})
        )
        h.submit(Heartbeat())
        attempts = len(retry_delays) + 1
        for index in range(attempts):
            await wait_until(lambda: len(auxiliary_submissions(h)) == index + 1)
            finish_context(h, accepted=False)
            await wait_until(lambda: len(work.decodes) == index + 1)
        await h.settle()
        original_key = work.contexts[0].work_key
        await h.run(
            CreateItem(
                item={
                    "id": "question",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "A new question."}],
                }
            )
        )
        question = await h.runner._start_append({"video_frames": [b"frame"]}, final=True, precreate_response=True)
        assert await question
        assert h.runner.memory._interaction_active
        turn = h.session.active_response_turn_id
        assert turn is not None
        h.session.end_response()
        h.session.complete_model_turn(turn)
        observation = await h.runner._start_append({"video_frames": [b"next"]}, final=True, precreate_response=True)
        assert await observation
        await h.settle(idle_s=0.02, timeout_s=0.2)
        assert not h.runner.memory._interaction_active
        assert work.contexts[-1].work_key == original_key
        assert len(auxiliary_submissions(h)) == attempts
        assert h.runner.memory._failures == attempts
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_context_cooldown_cannot_restart_exhausted_work(mocker):
    h = await open_harness()
    try:
        work = install_context_work(h, mocker, retry_delays_s=(0,), retry_cooldown_s=0.01)
        h.submit(Heartbeat())
        for index in range(2):
            await wait_until(lambda: len(auxiliary_submissions(h)) == index + 1)
            finish_context(h, accepted=False)
            await wait_until(lambda: len(work.decodes) == index + 1)
        await h.settle(idle_s=0.04, timeout_s=0.3)
        assert len(auxiliary_submissions(h)) == 2
        assert work.contexts[0].work_key == work.contexts[-1].work_key
        assert h.runner.memory._retry_timer is None
        assert h.runner.memory._failures == 2
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["epoch", "source", "settings"])
async def test_context_retry_identity_separates_execution_fences_from_new_work(mocker, change):
    h = await open_harness()
    try:
        work = install_context_work(h, mocker)
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        finish_context(h, accepted=False)
        await wait_until(lambda: len(work.decodes) == 1)
        await h.settle()
        if change == "epoch":
            h.session.epoch += 1
        elif change == "source":
            h.session.register_history_item("old", {"role": "user", "content": "A changed fact."})
        else:
            h.session.replace_config(replace(h.session.config, voice="new"))
        h.submit(Heartbeat())
        await h.settle(idle_s=0.02, timeout_s=0.2)
        assert len(auxiliary_submissions(h)) == (1 if change == "epoch" else 2)
        if change != "epoch":
            assert work.contexts[0].work_key != work.contexts[-1].work_key
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_temporary_context_policy_idle_does_not_replenish_failure_budget(mocker):
    h = await open_harness()
    try:
        work = install_context_work(h, mocker)
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        finish_context(h, accepted=False)
        await wait_until(lambda: len(work.decodes) == 1)
        await h.settle()
        work.enabled = False
        await h.run(Heartbeat())
        work.enabled = True
        h.submit(Heartbeat())
        await h.settle(idle_s=0.02, timeout_s=0.2)
        assert len(auxiliary_submissions(h)) == 1
        assert h.runner.memory._failures == 1
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_epoch_cancelled_context_attempt_can_resume_without_false_exhaustion(mocker):
    h = await open_harness()
    try:
        work = install_context_work(h, mocker, empty=True)
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        first = auxiliary_submissions(h)[0].context.request_id
        h.session.epoch += 1
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 2)
        assert ([first], True) in h.port.cleanups
        finish_context(h)
        await wait_until(lambda: len(work.decodes) == 1)
        assert h.runner.memory._failures == 0
    finally:
        await close_harness(h)
