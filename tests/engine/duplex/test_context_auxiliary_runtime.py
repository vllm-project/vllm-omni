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
from vllm_omni.engine.duplex.contracts import duplex_is_auxiliary_request
from vllm_omni.engine.duplex.plugin import DuplexAuxiliaryRequest, duplex_history_digest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def install_context_work(h, mocker, *, retry_delays_s=(), empty=False):
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
