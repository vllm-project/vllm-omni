# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Late raw request errors must obey the same resident owner fence as outputs."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.engine.duplex.test_session_runner import close_harness, open_harness
from vllm_omni.engine.duplex.config import DuplexSessionState
from vllm_omni.engine.duplex.contracts import duplex_ephemeral_stage_request_id
from vllm_omni.engine.duplex.messages import CloseDuplexSessionMessage

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


async def _close_session(harness):
    await harness.manager.handle(
        CloseDuplexSessionMessage(
            control_id="owner-fence-close", session_id=harness.session.session_id, reason="owner_fence_test"
        )
    )
    result = await asyncio.wait_for(harness.results.get(), timeout=1)
    assert result.ok


def _bind_turn(harness, turn_id):
    session = harness.session
    session.turn_id = turn_id
    fence = session.fence
    request_id = duplex_ephemeral_stage_request_id(fence, stage_id=1)
    session.bind_stage_request(1, request_id, fence=fence)
    session.bind_request(request_id)
    harness.manager.register_request(request_id, session.session_id)
    response_id = session.begin_response(turn_id=turn_id)
    harness.runner.emit(dict(type="response.created", response_id=response_id, epoch=session.epoch))
    return request_id, response_id, fence


async def _retire_then_bind_new(harness, old_request_id, *, terminal):
    session = harness.session
    session.end_response(commit_text=False)
    if terminal:
        harness.manager.plugin.data_plane.mark_terminal(old_request_id)
    # RecordingStagePort.cleanup does not suspend. Thus the old mailbox item is
    # already queued but has not been consumed when the new turn binds.
    await harness.runner.model._release_ephemeral_request(old_request_id, whole_turn=True)
    assert old_request_id not in session.resource_request_ids()
    new_request_id, new_response_id, _ = _bind_turn(harness, 2)
    return new_request_id, new_response_id


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [True, False], ids=["marked-terminal", "retired-resource-only"])
async def test_queued_retired_same_epoch_raw_error_cannot_fail_new_turn(terminal):
    harness = await open_harness()
    try:
        harness.session.capabilities = replace(harness.session.capabilities, supports_core_resumable_request=False)
        old_request, _, old_fence = _bind_turn(harness, 1)
        await harness.settle()
        harness.runner.on_stage_request_error(
            1,
            "original ephemeral request failed after retirement",
            request_id=old_request,
            expected_epoch=old_fence.epoch,
        )
        assert not harness.runner._mailbox.empty()
        new_request, new_response = await _retire_then_bind_new(harness, old_request, terminal=terminal)
        assert harness.session.epoch == old_fence.epoch
        assert harness.session.active_request_id == new_request
        events = await harness.settle()
        observed = dict(
            session_state=harness.session.state.value,
            active_response=harness.session.active_response_id,
            events=[
                dict(
                    type=event.type,
                    response_id=getattr(event, "response_id", None),
                    status=getattr(event, "status", None),
                )
                for event in events
            ],
        )
        assert harness.session.active_response_id == new_response, observed
        assert harness.session.state == DuplexSessionState.OPEN, observed
        assert not any(event.type == "error" for event in events), observed
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [True, False], ids=["marked-terminal", "retired-resource-only"])
async def test_existing_stage_output_guard_drops_the_same_queued_retired_owner(monkeypatch, terminal):
    harness = await open_harness()
    try:
        harness.session.capabilities = replace(harness.session.capabilities, supports_core_resumable_request=False)
        old_request, _, _ = _bind_turn(harness, 1)
        await harness.settle()
        build = Mock(side_effect=AssertionError("A retired output reached model projection"))
        monkeypatch.setattr(harness.runner.model, "_build_stage_output", build)
        harness.deliver(SimpleNamespace(request_id=old_request), stage_id=1)
        assert not harness.runner._mailbox.empty()
        _, new_response = await _retire_then_bind_new(harness, old_request, terminal=terminal)
        events = await harness.settle()
        build.assert_not_called()
        assert harness.session.active_response_id == new_response
        assert harness.session.state == DuplexSessionState.OPEN
        assert not any(event.type == "error" for event in events)
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_current_owner_raw_error_is_still_reported_and_failed():
    harness = await open_harness()
    try:
        request, response, fence = _bind_turn(harness, 1)
        await harness.settle()
        harness.runner.on_stage_request_error(
            1, "current request failed", request_id=request, expected_epoch=fence.epoch
        )
        events = await harness.settle()
        assert any(event.type == "error" for event in events)
        assert any(
            event.type == "response.done" and event.response_id == response and event.status == "failed"
            for event in events
        )
        assert harness.session.state == DuplexSessionState.CLOSED
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_old_epoch_raw_error_is_already_ignored_for_a_new_owner():
    harness = await open_harness()
    try:
        old_request, _, fence = _bind_turn(harness, 1)
        await harness.settle()
        harness.runner.on_stage_request_error(1, "old epoch failed", request_id=old_request, expected_epoch=fence.epoch)
        harness.session.epoch += 1
        _, new_response, _ = _bind_turn(harness, 2)
        events = await harness.settle()
        assert harness.session.active_response_id == new_response
        assert harness.session.state == DuplexSessionState.OPEN
        assert not any(event.type == "error" for event in events)
    finally:
        await close_harness(harness)


def _bind_draining_response(harness):
    session = harness.session
    session.capabilities = replace(session.capabilities, supports_concurrent_turn_requests=True)
    old_request, old_response, old_fence = _bind_turn(harness, 1)
    sibling = duplex_ephemeral_stage_request_id(old_fence, stage_id=2)
    session.bind_stage_request(2, sibling, fence=old_fence)
    harness.manager.register_request(sibling, session.session_id)
    session.end_response(commit_text=False)
    new_request, new_response, _ = _bind_turn(harness, 2)
    session.bind_draining_request(old_request, old_response)
    session.bind_draining_request(sibling, old_response)
    return old_request, sibling, old_response, old_fence, new_request, new_response


@pytest.mark.asyncio
async def test_draining_error_fails_only_its_response_and_aborts_its_stage_owners():
    harness = await open_harness(stage_count=3)
    try:
        old_request, sibling, old_response, fence, new_request, new_response = _bind_draining_response(harness)
        await harness.settle()
        harness.runner.on_stage_request_error(
            1, "old synthesis failed", request_id=old_request, expected_epoch=fence.epoch
        )
        events = await harness.settle()
        assert harness.session.state == DuplexSessionState.OPEN
        assert harness.session.active_request_id == new_request
        assert harness.session.active_response_id == new_response
        assert [(event.response_id, event.status) for event in events if event.type == "response.done"] == [
            (old_response, "failed")
        ]
        assert not any(event.type == "session.closed" for event in events)
        assert harness.port.cleanups == [([old_request, sibling], True)]
        assert old_request not in harness.session.resource_request_ids()
        assert sibling not in harness.session.resource_request_ids()
        assert new_request in harness.session.resource_request_ids()
        for request_id in (old_request, sibling):
            assert harness.manager.plugin.data_plane.is_terminal(request_id)
            harness.runner.on_stage_request_error(
                1, "duplicate late error", request_id=request_id, expected_epoch=fence.epoch
            )
        assert await harness.settle() == []
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_failed_draining_abort_retains_owners_until_retry_ack_without_closing_new_response(monkeypatch):
    harness = await open_harness(stage_count=3)
    retry_started = asyncio.Event()
    retry_ack = asyncio.Event()
    try:
        old_request, sibling, old_response, fence, new_request, new_response = _bind_draining_response(harness)
        await harness.settle()
        calls = []
        original_cleanup = harness.port.cleanup

        async def cleanup(request_ids, *, abort=False):
            calls.append((list(request_ids), abort))
            assert set(request_ids) == {old_request, sibling}
            assert abort
            assert {old_request, sibling, new_request}.issubset(harness.session.resource_request_ids())
            if len(calls) == 1:
                raise RuntimeError("first abort acknowledgement failed")
            retry_started.set()
            await retry_ack.wait()
            await original_cleanup(request_ids, abort=abort)

        monkeypatch.setattr(harness.port, "cleanup", cleanup)
        harness.runner.on_stage_request_error(
            1, "old synthesis failed", request_id=old_request, expected_epoch=fence.epoch
        )
        await asyncio.wait_for(retry_started.wait(), timeout=1)
        assert harness.session.active_response_id == new_response
        assert harness.session.state == DuplexSessionState.OPEN
        assert len(harness.runner._background_tasks) == 1
        retry_ack.set()
        events = await harness.settle()
        assert calls == [([old_request, sibling], True)] * 2
        assert [(event.response_id, event.status) for event in events if event.type == "response.done"] == [
            (old_response, "failed")
        ]
        assert harness.session.active_response_id == new_response
        assert new_request in harness.session.resource_request_ids()
        assert old_request not in harness.session.resource_request_ids()
        assert not harness.runner._background_tasks
        monkeypatch.setattr(harness.port, "cleanup", original_cleanup)
    finally:
        retry_ack.set()
        await close_harness(harness)


@pytest.mark.asyncio
async def test_exhausted_draining_abort_keeps_resources_for_manager_close(monkeypatch):
    harness = await open_harness(stage_count=3)
    old_request, sibling, _, fence, new_request, new_response = _bind_draining_response(harness)
    calls = []
    original_cleanup = harness.port.cleanup

    async def cleanup(request_ids, *, abort=False):
        calls.append((list(request_ids), abort))
        if len(calls) <= 3:
            raise RuntimeError("abort transport unavailable")
        await original_cleanup(request_ids, abort=abort)

    monkeypatch.setattr(harness.port, "cleanup", cleanup)
    try:
        await harness.settle()
        harness.runner.on_stage_request_error(
            1, "old synthesis failed", request_id=old_request, expected_epoch=fence.epoch
        )
        await harness.settle()
        assert calls == [([old_request, sibling], True)] * 3
        assert not harness.runner._background_tasks
        assert {old_request, sibling, new_request}.issubset(harness.session.resource_request_ids(submitted=True))
        assert harness.session.active_response_id == new_response
        assert harness.session.state == DuplexSessionState.OPEN
    finally:
        await _close_session(harness)
        await close_harness(harness)
    submitted_cleanups = [ids for ids, abort in calls if abort]
    assert len(submitted_cleanups) == 4
    assert {old_request, sibling, new_request}.issubset(submitted_cleanups[-1])
    assert not harness.session.request_resources


@pytest.mark.asyncio
async def test_session_close_cancels_draining_cleanup_and_manager_aborts_retained_owners(monkeypatch):
    harness = await open_harness(stage_count=3)
    old_request, sibling, _, fence, new_request, _ = _bind_draining_response(harness)
    cleanup_started = asyncio.Event()
    cleanup_cancelled = asyncio.Event()
    original_cleanup = harness.port.cleanup
    calls = []

    async def cleanup(request_ids, *, abort=False):
        calls.append((list(request_ids), abort))
        if len(calls) == 1:
            cleanup_started.set()
            try:
                await asyncio.Future()
            finally:
                cleanup_cancelled.set()
        else:
            await original_cleanup(request_ids, abort=abort)

    monkeypatch.setattr(harness.port, "cleanup", cleanup)
    try:
        await harness.settle()
        harness.runner.on_stage_request_error(
            1, "old synthesis failed", request_id=old_request, expected_epoch=fence.epoch
        )
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        assert {old_request, sibling, new_request}.issubset(harness.session.resource_request_ids(submitted=True))
    finally:
        await _close_session(harness)
        await close_harness(harness)
    assert cleanup_cancelled.is_set()
    assert not harness.runner._background_tasks
    submitted_cleanups = [ids for ids, abort in calls if abort]
    assert len(submitted_cleanups) == 2
    assert {old_request, sibling, new_request}.issubset(submitted_cleanups[-1])
    assert not harness.session.request_resources


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_owner", ["unknown", "wrong-stage", "wrong-epoch", "wrong-session", "terminal", "dead-lease"]
)
async def test_invalid_current_epoch_owner_error_is_ignored(invalid_owner):
    harness = await open_harness()
    try:
        request, response, fence = _bind_turn(harness, 1)
        await harness.settle()
        stage_id = 1
        if invalid_owner == "unknown":
            harness.session.request_resources.pop((1, request))
        elif invalid_owner == "wrong-stage":
            stage_id = 0
        elif invalid_owner == "wrong-epoch":
            harness.session.request_resources[(1, request)].fence = replace(fence, epoch=fence.epoch + 1)
        elif invalid_owner == "wrong-session":
            harness.session.request_resources[(1, request)].fence = replace(fence, session_id="another-session")
        elif invalid_owner == "terminal":
            harness.manager.plugin.data_plane.mark_terminal(request)
        else:
            harness.session.lease.terminal_reason = "expired"
        harness.runner.on_stage_request_error(
            stage_id, "stale owner error", request_id=request, expected_epoch=fence.epoch
        )
        assert await harness.settle() == []
        assert harness.session.active_response_id == response
        assert harness.session.state == DuplexSessionState.OPEN
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_shutdown_cancels_tracked_draining_abort_without_erasing_worker_ownership(monkeypatch):
    harness = await open_harness(stage_count=3)
    old_request, sibling, _, fence, new_request, _ = _bind_draining_response(harness)
    cleanup_started = asyncio.Event()
    cleanup_cancelled = asyncio.Event()

    async def cleanup(request_ids, *, abort=False):
        assert set(request_ids) == {old_request, sibling} and abort
        cleanup_started.set()
        try:
            await asyncio.Future()
        finally:
            cleanup_cancelled.set()

    monkeypatch.setattr(harness.port, "cleanup", cleanup)
    try:
        await harness.settle()
        harness.runner.on_stage_request_error(
            1, "old synthesis failed", request_id=old_request, expected_epoch=fence.epoch
        )
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    finally:
        await close_harness(harness)
    assert cleanup_cancelled.is_set()
    assert not harness.runner._background_tasks
    # Manager shutdown delegates worker teardown to the outer orchestrator;
    # it must not invent a successful abort ACK or erase durable ownership.
    assert {old_request, sibling, new_request}.issubset(harness.session.resource_request_ids(submitted=True))


@pytest.mark.asyncio
async def test_draining_owner_requires_explicit_concurrent_turn_capability():
    harness = await open_harness(stage_count=3)
    try:
        old_request, sibling, _, fence, new_request, new_response = _bind_draining_response(harness)
        harness.session.capabilities = replace(harness.session.capabilities, supports_concurrent_turn_requests=False)
        await harness.settle()
        harness.runner.on_stage_request_error(
            1, "unadmitted drain failed", request_id=old_request, expected_epoch=fence.epoch
        )
        assert await harness.settle() == []
        assert harness.port.cleanups == []
        assert harness.session.active_response_id == new_response
        assert {old_request, sibling, new_request}.issubset(harness.session.resource_request_ids(submitted=True))
    finally:
        await close_harness(harness)
