# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Adversarial settlement orderings over the ledger and real session mailbox.

These tests inject clock movement and timer delivery, not sleeps or replacement
packages. The recording stage port does not establish model-side recovery.
"""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest

from tests.engine.duplex.test_session_manager import FakeClock
from tests.engine.duplex.test_session_runner import types
from tests.engine.duplex.test_session_runner_nemotron_voicechat import (
    TIMEOUT_S,
    answer,
    batches,
    error_codes,
    errors,
    expose_function_call,
    generation,
    open_timed_harness,
    output_item,
    prepare,
)
from tests.engine.duplex.test_tool_call_ledger import FakeTimerScheduler
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionState
from vllm_omni.engine.duplex.messages import CloseDuplexSessionMessage
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.engine.duplex.session.tool_calls import (
    ToolCallAcceptance,
    ToolCallLedger,
    ToolCallOrigin,
    ToolCallRejection,
    ToolCallState,
)

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.session.tool_calls import ToolCallSettlement

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def origin(call_id="call_a", *, epoch=3):
    return ToolCallOrigin(
        call_id=call_id,
        epoch=epoch,
        turn_id=7,
        response_id="response_a",
        item_id=f"item_{call_id}",
    )


@pytest.mark.parametrize("entry", ["begin_accept", "revalidate", "expire"])
def test_every_expiry_entry_notifies_the_same_committed_fact_once(entry):
    clock = FakeClock(100.0)
    scheduler = FakeTimerScheduler()
    queued_expiries: list[str] = []
    settlements: list[ToolCallSettlement] = []

    def observe(settlement: ToolCallSettlement) -> None:
        call = ledger.record(settlement.origin.call_id)
        assert call is not None
        assert call.state is settlement.state
        assert call.settled_epoch == settlement.settled_epoch
        assert call.timer is None
        settlements.append(settlement)

    ledger = ToolCallLedger(
        max_calls=4, clock=clock, timeout_s=5.0, on_expire=queued_expiries.append, on_settle=observe
    )
    ledger.set_scheduler(scheduler)
    call = ledger.register(origin())
    acceptance = ledger.begin_accept("call_a")
    assert isinstance(acceptance, ToolCallAcceptance)
    late_callback = scheduler.timers[0][2]
    assert callable(late_callback)
    clock.advance(5.0)

    if entry == "begin_accept":
        assert ledger.begin_accept("call_a") is ToolCallRejection.EXPIRED
    elif entry == "revalidate":
        assert ledger.revalidate(acceptance, epoch=3) is ToolCallRejection.EXPIRED
    else:
        assert ledger.expire("call_a") is True

    assert len(settlements) == 1
    settlement = settlements[0]
    assert settlement.origin == call.origin == origin()
    assert settlement.state is ToolCallState.TIMED_OUT
    assert settlement.settled_epoch == 3
    with pytest.raises(FrozenInstanceError):
        settlement.state = ToolCallState.ACCEPTED  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        settlement.origin.call_id = "call_b"  # type: ignore[misc]

    # A callback already dequeued by the event loop can still run after cancel.
    late_callback()
    late_callback()
    assert queued_expiries == ["call_a", "call_a"]
    for call_id in queued_expiries:
        assert ledger.expire(call_id) is False
    assert ledger.begin_accept("call_a") is ToolCallRejection.NOT_PENDING
    assert ledger.revalidate(acceptance, epoch=3) is ToolCallRejection.NOT_PENDING
    assert ledger.commit_accept(acceptance, epoch=3) is False
    assert ledger.invalidate_epoch(4) == ()
    assert settlements == [settlement]
    assert scheduler.pending_delays() == []


@pytest.mark.parametrize("terminal", [ToolCallState.ACCEPTED, ToolCallState.CANCELLED])
def test_other_terminal_decisions_notify_once_and_ignore_a_late_timer(terminal):
    clock = FakeClock(100.0)
    settlements: list[ToolCallSettlement] = []
    ledger = ToolCallLedger(max_calls=4, clock=clock, timeout_s=5.0, on_settle=settlements.append)
    call = ledger.register(origin())
    acceptance = ledger.begin_accept("call_a")
    assert isinstance(acceptance, ToolCallAcceptance)

    if terminal is ToolCallState.ACCEPTED:
        assert ledger.commit_accept(acceptance, epoch=3) is True
        expected_epoch = 3
    else:
        assert ledger.invalidate_epoch(4) == (call.origin,)
        expected_epoch = 4

    clock.advance(10.0)
    assert ledger.expire("call_a") is False
    assert ledger.commit_accept(acceptance, epoch=4) is False
    assert ledger.invalidate_epoch(5) == ()
    assert len(settlements) == 1
    assert settlements[0].origin == call.origin
    assert settlements[0].state is terminal
    assert settlements[0].settled_epoch == expected_epoch


@pytest.mark.parametrize("elapsed", [5.0, 50.0])
def test_commit_without_revalidate_cannot_cross_the_deadline(elapsed):
    clock = FakeClock(100.0)
    scheduler = FakeTimerScheduler()
    ledger = ToolCallLedger(max_calls=4, clock=clock, timeout_s=5.0)
    ledger.set_scheduler(scheduler)
    call = ledger.register(origin())
    acceptance = ledger.begin_accept("call_a")
    assert isinstance(acceptance, ToolCallAcceptance)
    clock.advance(elapsed)

    assert ledger.commit_accept(acceptance, epoch=3) is False
    assert call.state is ToolCallState.TIMED_OUT
    assert call.settled_epoch == 3
    assert scheduler.pending_delays() == []


def test_commit_without_revalidate_cannot_cross_an_epoch():
    ledger = ToolCallLedger(max_calls=4, clock=FakeClock(100.0))
    call = ledger.register(origin())
    acceptance = ledger.begin_accept("call_a")
    assert isinstance(acceptance, ToolCallAcceptance)

    assert ledger.commit_accept(acceptance, epoch=4) is False
    assert call.state is ToolCallState.CANCELLED
    assert call.settled_epoch == 4


def test_an_early_timer_rearms_only_the_remaining_deadline():
    clock = FakeClock(100.0)
    scheduler = FakeTimerScheduler()
    queued: list[str] = []
    ledger = ToolCallLedger(max_calls=4, clock=clock, timeout_s=5.0, on_expire=queued.append)
    ledger.set_scheduler(scheduler)
    call = ledger.register(origin())
    deadline = call.deadline

    clock.advance(1.25)
    scheduler.fire_all()
    assert queued == ["call_a"]
    assert ledger.expire(queued.pop()) is False
    assert call.state is ToolCallState.PENDING
    assert call.deadline == deadline
    assert scheduler.pending_delays() == [3.75]

    clock.advance(3.75)
    scheduler.fire_all()
    assert queued == ["call_a"]
    assert ledger.expire(queued.pop()) is True
    assert scheduler.pending_delays() == []


def install_recorder(monkeypatch, h):
    observed = []

    def record(config, current, settlement):
        assert asyncio.current_task() is h.runner._worker
        call = h.runner.tool_calls.record(settlement.origin.call_id)
        assert call is not None and call.state is settlement.state
        observed.append((settlement, deepcopy(current)))
        return None

    # Absence on the baseline must produce a behavioral failure, not prevent
    # collection of the old-API deadline and timer regressions above.
    monkeypatch.setattr(h.runner.plugin, "runtime_config_for_tool_call_timeout", record, raising=False)
    return observed


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["begin_accept", "revalidate", "expire"])
async def test_runner_emits_one_timeout_notice_for_every_expiry_entry(monkeypatch, entry):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        observed = install_recorder(monkeypatch, h)
        call = h.runner.tool_calls.record(call_id)
        late_callback = h.scheduler.timers[0][2]

        if entry == "expire":
            h.clock.advance(TIMEOUT_S)
            h.scheduler.fire_all()
            assert call.state is ToolCallState.PENDING and observed == []
            events = await h.settle()
            assert error_codes(events) == ["tool_result_timeout"]
        else:
            if entry == "begin_accept":
                h.clock.advance(TIMEOUT_S)
            else:

                async def slow_tail():
                    h.clock.advance(TIMEOUT_S)
                    return True

                monkeypatch.setattr(h.runner.control, "_wait_for_append_tail", slow_tail)
            events = await h.run(commands.CreateItem(item=output_item(call_id), event_id="result-at-deadline"))
            assert error_codes(events) == ["invalid_function_call_output", "tool_result_timeout"]
            assert errors(events)[0].param == "item.call_id"
            assert errors(events)[0].related_event_id == "result-at-deadline"

        assert len(observed) == 1
        settlement, current = observed[0]
        assert settlement.origin == call.origin
        assert settlement.state is ToolCallState.TIMED_OUT
        assert settlement.settled_epoch == h.session.epoch
        assert generation(h) == 0 and batches(h) == []
        assert current == h.session.runtime_config
        late_callback()
        late_callback()
        assert error_codes(await h.settle()) == []
        assert error_codes(await answer(h, call_id)) == ["invalid_function_call_output"]
        assert len(observed) == 1


@pytest.mark.asyncio
async def test_timeout_of_a_preserves_b_and_its_independent_timer(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_a = await prepare(h)
        h.clock.advance(1.0)
        call_b, _ = await expose_function_call(h)
        observed = install_recorder(monkeypatch, h)
        record_b = h.runner.tool_calls.record(call_b)
        timer_b = record_b.timer
        deadline_b = record_b.deadline
        runtime_before = deepcopy(dict(h.session.runtime_config))
        epoch_before = h.session.epoch
        h.clock.advance(TIMEOUT_S - 1.0)
        h.scheduler.timers[0][2]()

        assert error_codes(await h.settle()) == ["tool_result_timeout"]
        assert [entry[0].origin.call_id for entry in observed] == [call_a]
        assert h.runner.tool_calls.record(call_b) is record_b
        assert record_b.state is ToolCallState.PENDING
        assert record_b.timer is timer_b and record_b.deadline == deadline_b
        assert h.scheduler.timers[1][1] is False
        assert h.session.runtime_config == runtime_before
        assert h.session.epoch == epoch_before

        assert error_codes(await answer(h, call_b)) == []
        assert record_b.state is ToolCallState.ACCEPTED
        assert [entry[0].origin.call_id for entry in observed] == [call_a]
        assert generation(h) == 1 and len(batches(h)) == 1
        assert h.scheduler.pending_delays() == []


@pytest.mark.asyncio
async def test_runner_early_timer_retains_a_future_wakeup(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        observed = install_recorder(monkeypatch, h)
        h.clock.advance(2.0)
        h.scheduler.fire_all()
        assert error_codes(await h.settle()) == []
        assert observed == []
        assert h.scheduler.pending_delays() == [TIMEOUT_S - 2.0]

        h.clock.advance(TIMEOUT_S - 2.0)
        h.scheduler.fire_all()
        assert error_codes(await h.settle()) == ["tool_result_timeout"]
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.TIMED_OUT
        assert len(observed) == 1


@pytest.mark.asyncio
async def test_conversion_candidate_is_discarded_when_commit_returns_false(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        observed = install_recorder(monkeypatch, h)
        runtime_before = deepcopy(dict(h.session.runtime_config))
        submissions_before = len(h.port.submissions)
        converted = []
        real_convert = h.runner.plugin.runtime_config_for_function_output

        def convert(config, current, item, **kwargs):
            candidate = real_convert(config, current, item, **kwargs)
            converted.append(deepcopy(candidate))
            return candidate

        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_function_output", convert)
        real_commit = h.runner.tool_calls.commit_accept
        monkeypatch.setattr(h.runner.tool_calls, "commit_accept", lambda acceptance, *, epoch: False)
        events = await h.run(commands.CreateItem(item=output_item(call_id), event_id="commit-refused"))

        assert len(converted) == 1 and converted[0]["nvc_function_response_generation"] == 1
        assert error_codes(events) == ["commit_aborted"]
        assert errors(events)[0].related_event_id == "commit-refused"
        assert "conversation.item.created" not in types(events)
        assert h.session.runtime_config == runtime_before
        assert len(h.port.submissions) == submissions_before
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.PENDING
        assert observed == []
        monkeypatch.setattr(h.runner.tool_calls, "commit_accept", real_commit)
        assert "conversation.item.created" in types(await answer(h, call_id))


@pytest.mark.asyncio
async def test_settlement_queued_before_cancel_cannot_reconfigure_the_new_epoch(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        observed = install_recorder(monkeypatch, h)
        cleanups = []
        real_cleanup = h.runner.plugin.on_tool_call_settled

        def cleanup(settlement):
            assert asyncio.current_task() is h.runner._worker
            cleanups.append(settlement)
            real_cleanup(settlement)

        monkeypatch.setattr(h.runner.plugin, "on_tool_call_settled", cleanup)
        old_epoch = h.session.epoch
        h.clock.advance(TIMEOUT_S)
        # Expiry queues its settlement behind this already queued cancellation.
        h.scheduler.fire_all()
        h.submit(commands.CancelInput())
        await h.settle()

        assert h.session.epoch == old_epoch + 1
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.TIMED_OUT
        assert len(cleanups) == 1
        assert cleanups[0].origin.call_id == call_id
        assert cleanups[0].settled_epoch == old_epoch
        assert observed == []
        h.scheduler.timers[0][2]()
        assert error_codes(await h.settle()) == []
        assert len(cleanups) == 1
        next_call = await prepare(h)
        assert error_codes(await answer(h, next_call)) == []
        assert observed == []
        assert [settlement.origin.call_id for settlement in cleanups] == [call_id, next_call]


@pytest.mark.asyncio
async def test_settlement_hook_failure_closes_without_leaking_candidate_mutation(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        h.session.replace_runtime_config({**h.session.runtime_config, "settlement_probe": {"values": [1]}})
        observed = []

        def fail(config, current, settlement):
            observed.append(settlement)
            current["settlement_probe"]["values"].append(2)
            raise RuntimeError("settlement adapter failed")

        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_tool_call_timeout", fail, raising=False)
        h.clock.advance(TIMEOUT_S)
        h.scheduler.fire_all()
        events = await h.settle()

        assert len(observed) == 1 and observed[0].origin.call_id == call_id
        assert error_codes(events) == ["tool_result_timeout", "tool_call_settlement_failed"]
        assert h.runner.closing is True
        assert h.session.state is DuplexSessionState.CLOSED
        assert types(events).count("session.closed") == 1
        assert h.session.runtime_config["settlement_probe"] == {"values": [1]}
        assert len(h.runner.tool_calls) == 0
        assert h.scheduler.pending_delays() == []
        assert error_codes(await answer(h, call_id)) == ["unknown_session"]


@pytest.mark.asyncio
async def test_the_ledger_notification_only_enqueues_cleanup_and_model_effects(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        observed = install_recorder(monkeypatch, h)
        cleanups = []

        def cleanup(settlement):
            assert asyncio.current_task() is h.runner._worker
            cleanups.append(settlement)

        monkeypatch.setattr(h.runner.plugin, "on_tool_call_settled", cleanup)
        h.clock.advance(TIMEOUT_S)
        # Drive the ledger seam directly so a synchronous plugin callback
        # cannot accidentally pass just because the caller is already a worker.
        assert h.runner.tool_calls.expire(call_id) is True
        assert cleanups == [] and observed == []
        await h.settle()
        assert len(cleanups) == len(observed) == 1
        assert cleanups[0] is observed[0][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_before_hook", [False, True])
async def test_cleanup_failure_closes_even_when_its_settlement_is_superseded(monkeypatch, cancel_before_hook):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        observed = install_recorder(monkeypatch, h)
        cleanups = []

        def fail(settlement):
            cleanups.append(settlement)
            raise RuntimeError("terminal metadata cleanup failed")

        monkeypatch.setattr(h.runner.plugin, "on_tool_call_settled", fail)
        h.clock.advance(TIMEOUT_S)
        h.scheduler.fire_all()
        if cancel_before_hook:
            h.submit(commands.CancelInput())
        events = await h.settle()

        assert len(cleanups) == 1 and cleanups[0].origin.call_id == call_id
        assert observed == []
        assert error_codes(events) == ["tool_result_timeout", "tool_call_settlement_failed"]
        assert h.runner.closing is True
        assert h.session.state is DuplexSessionState.CLOSED
        assert types(events).count("session.closed") == 1
        assert len(h.runner.tool_calls) == 0
        assert h.scheduler.pending_delays() == []
        assert error_codes(await answer(h, call_id)) == ["unknown_session"]


@pytest.mark.asyncio
async def test_a_rejected_conversion_cannot_mutate_nested_live_runtime_config(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        h.session.replace_runtime_config({**h.session.runtime_config, "conversion_probe": {"values": [1]}})
        observed = install_recorder(monkeypatch, h)
        runtime_before = deepcopy(dict(h.session.runtime_config))
        submissions_before = len(h.port.submissions)
        real_convert = h.runner.plugin.runtime_config_for_function_output

        def reject(config, current, item, **kwargs):
            del item, kwargs
            current["conversion_probe"]["values"].append(2)
            raise DuplexRuntimeConfigError("the candidate cannot be rendered", code="bad_function_result")

        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_function_output", reject)
        events = await answer(h, call_id)

        assert error_codes(events) == ["bad_function_result"]
        assert "conversation.item.created" not in types(events)
        assert h.session.runtime_config == runtime_before
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.PENDING
        assert h.scheduler.pending_delays() == [TIMEOUT_S]
        assert len(h.port.submissions) == submissions_before
        assert observed == []

        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_function_output", real_convert)
        assert error_codes(await answer(h, call_id)) == []
        assert generation(h) == 1 and len(batches(h)) == 1
        assert h.session.runtime_config["conversion_probe"] == {"values": [1]}


@pytest.mark.asyncio
async def test_recognized_calls_are_registered_before_emission_and_invalid_calls_stay_hidden(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        exposed = []
        real_emit = h.runner.out.emit

        def observe(payload):
            if payload.get("type") == "function_call.done":
                call = h.runner.tool_calls.record(payload["call_id"])
                assert call is not None and call.state is ToolCallState.PENDING
                assert call.origin.item_id == payload["item_id"]
                assert call.origin.response_id == payload["response_id"]
                assert call.origin.name == payload["name"]
                assert call.origin.arguments == payload["arguments"]
                exposed.append(call.origin)
            real_emit(payload)

        monkeypatch.setattr(h.runner.out, "emit", observe)
        call_id = await prepare(h)
        assert len(exposed) == 1 and exposed[0].call_id == call_id
        for malformed in ({"call_id": "bad", "name": ""}, {"name": "lookup", "arguments": "{}"}):
            await h.runner.model._expose_function_call(malformed)
        events = await h.settle()
        assert error_codes(events) == ["invalid_function_call", "invalid_function_call"]
        assert all(error.to_realtime()["error"]["type"] == "invalid_request_error" for error in errors(events))
        assert len(exposed) == len(h.runner.tool_calls) == 1
        assert h.runner.tool_calls.record("bad") is None


@pytest.mark.asyncio
async def test_accepted_cleanup_failure_installs_no_candidate_ack_or_continuation(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        call = h.runner.tool_calls.record(call_id)
        runtime_before = deepcopy(dict(h.session.runtime_config))
        submissions_before = len(h.port.submissions)
        observed = install_recorder(monkeypatch, h)
        cleanups, converted, emitted, continuations, disposed = [], [], [], [], []
        real_convert = h.runner.plugin.runtime_config_for_function_output
        real_emit = h.runner.out.emit
        real_continue = h.runner.model.maybe_continue_response
        real_dispose = h.runner.plugin.on_tool_calls_closed

        def convert(config, current, item, **kwargs):
            candidate = real_convert(config, current, item, **kwargs)
            converted.append(deepcopy(candidate))
            return candidate

        def emit(payload):
            emitted.append(payload["type"])
            real_emit(payload)

        def continue_response(**kwargs):
            continuations.append(kwargs)
            return real_continue(**kwargs)

        def fail_cleanup(settlement):
            assert asyncio.current_task() is h.runner._worker
            assert call.state is ToolCallState.ACCEPTED
            assert h.session.runtime_config == runtime_before
            cleanups.append(settlement)
            raise RuntimeError("accepted call metadata could not be finalized")

        def dispose(origins):
            disposed.append(origins)
            real_dispose(origins)

        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_function_output", convert)
        monkeypatch.setattr(h.runner.plugin, "on_tool_call_settled", fail_cleanup)
        monkeypatch.setattr(h.runner.plugin, "on_tool_calls_closed", dispose)
        monkeypatch.setattr(h.runner.out, "emit", emit)
        monkeypatch.setattr(h.runner.model, "maybe_continue_response", continue_response)
        events = await answer(h, call_id)

        assert len(converted) == 1 and converted[0]["nvc_function_response_generation"] == 1
        assert len(cleanups) == 1 and cleanups[0].state is ToolCallState.ACCEPTED
        assert error_codes(events) == ["tool_call_settlement_failed"]
        assert errors(events)[0].to_realtime()["error"]["type"] == "server_error"
        assert "conversation.item.created" not in emitted
        assert "conversation.item.created" not in types(events)
        assert continuations == [] and observed == []
        assert len(h.port.submissions) == submissions_before
        assert h.session.runtime_config == runtime_before
        assert h.runner.closing is True
        assert call.state is ToolCallState.ACCEPTED  # failure never reopens admission
        assert h.session.state is DuplexSessionState.CLOSED
        assert types(events).count("session.closed") == 1
        assert disposed == [(call.origin,)]
        assert len(h.runner.tool_calls) == 0
        assert h.scheduler.pending_delays() == []
        assert error_codes(await answer(h, call_id)) == ["unknown_session"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["close", "shutdown"])
@pytest.mark.parametrize("disposal_raises", [False, True])
async def test_teardown_disposes_pending_and_queued_terminal_calls_once(monkeypatch, operation, disposal_raises):
    async with open_timed_harness(monkeypatch) as h:
        call_a = await prepare(h)
        h.clock.advance(1.0)
        call_b, _ = await expose_function_call(h)
        record_a = h.runner.tool_calls.record(call_a)
        record_b = h.runner.tool_calls.record(call_b)
        expected_origins = (record_a.origin, record_b.origin)
        observed = install_recorder(monkeypatch, h)
        cleanups: list[ToolCallSettlement] = []
        disposed: list[tuple[ToolCallOrigin, ...]] = []
        real_dispose = h.runner.plugin.on_tool_calls_closed

        def dispose(origins):
            assert isinstance(origins, tuple)
            assert h.runner.closing is True
            assert h.runner.tool_calls.record(call_a).state is ToolCallState.TIMED_OUT
            assert h.runner.tool_calls.record(call_b).state is ToolCallState.PENDING
            disposed.append(origins)
            if disposal_raises:
                raise RuntimeError("close-time metadata disposal failed")
            real_dispose(origins)

        monkeypatch.setattr(h.runner.plugin, "on_tool_call_settled", cleanups.append)
        monkeypatch.setattr(h.runner.plugin, "on_tool_calls_closed", dispose)
        h.clock.advance(TIMEOUT_S - 1.0)
        # Commit the timeout and queue its fact, then let the control-plane
        # teardown overtake that fact before the mailbox has another turn.
        assert h.runner.tool_calls.expire(call_a) is True
        assert cleanups == [] and observed == []
        if operation == "close":
            await h.manager.close(
                CloseDuplexSessionMessage(control_id="close-before-settlement", session_id=h.session.session_id)
            )
        else:
            await h.manager.shutdown()

        assert disposed == [expected_origins]
        assert cleanups == [] and observed == []
        assert h.session.state is DuplexSessionState.CLOSED
        assert h.session.session_id not in h.manager.runners
        assert len(h.runner.tool_calls) == 0
        assert h.scheduler.pending_delays() == []
        # Teardown callers may overlap. Even a failed disposal is never retried
        # against records already cleared by the first irreversible close.
        await h.runner.close("duplicate-close")
        await h.runner.shutdown()
        assert disposed == [expected_origins]


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [ToolCallState.ACCEPTED, ToolCallState.TIMED_OUT, ToolCallState.CANCELLED])
async def test_replayed_internal_settlements_repeat_neither_cleanup_nor_model_effect(monkeypatch, terminal):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        cleanups = []
        real_cleanup = h.runner.plugin.on_tool_call_settled

        def cleanup(settlement):
            cleanups.append(settlement)
            real_cleanup(settlement)

        monkeypatch.setattr(h.runner.plugin, "on_tool_call_settled", cleanup)
        observed = install_recorder(monkeypatch, h)
        if terminal is ToolCallState.ACCEPTED:
            assert error_codes(await answer(h, call_id)) == []
        elif terminal is ToolCallState.TIMED_OUT:
            h.clock.advance(TIMEOUT_S)
            h.scheduler.fire_all()
            assert error_codes(await h.settle()) == ["tool_result_timeout"]
        else:
            assert error_codes(await h.run(commands.CancelInput())) == []

        expected_recoveries = 1 if terminal is ToolCallState.TIMED_OUT else 0
        assert len(cleanups) == 1 and len(observed) == expected_recoveries
        original = cleanups[0]
        assert original.state is terminal
        runtime_before = deepcopy(dict(h.session.runtime_config))
        # A fresh value-equal notification must also deduplicate; object
        # identity is not the identity of the ledger's one terminal decision.
        replay = type(original)(original.origin, original.state, original.settled_epoch)
        h.runner._enqueue_tool_call_settlement(replay)
        h.runner._enqueue_tool_call_settlement(original)
        assert error_codes(await h.settle()) == []
        assert len(cleanups) == 1 and len(observed) == expected_recoveries
        assert h.session.runtime_config == runtime_before
        assert h.runner.tool_calls.record(call_id).state is terminal


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", [{"x": 1}, ["x", 1], None])
async def test_non_string_arguments_are_rejected_without_publication(monkeypatch, arguments):
    async with open_timed_harness(monkeypatch) as h:
        await prepare(h)
        before = len(h.runner.tool_calls)
        await h.runner.model._expose_function_call(
            {"call_id": "structured-call", "name": "lookup", "arguments": arguments}
        )
        events = await h.settle()
        assert error_codes(events) == ["invalid_function_call"]
        assert h.runner.tool_calls.record("structured-call") is None
        assert len(h.runner.tool_calls) == before


@pytest.mark.asyncio
async def test_string_arguments_are_identical_in_origin_and_wire(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        await prepare(h)
        captured = []
        real_emit = h.runner.out.emit
        arguments = '{"x": 1}'

        def emit(payload):
            if payload.get("type") == "function_call.done":
                captured.append(dict(payload))
            return real_emit(payload)

        monkeypatch.setattr(h.runner.out, "emit", emit)
        await h.runner.model._expose_function_call(
            {"call_id": "structured-call", "name": "lookup", "arguments": arguments}
        )
        assert len(captured) == 1
        call = h.runner.tool_calls.record("structured-call")
        assert call is not None
        assert call.origin.arguments == captured[0]["arguments"] == arguments
        assert json.loads(call.origin.arguments) == {"x": 1}


@pytest.mark.asyncio
async def test_candidate_is_detached_before_terminal_cleanup_mutates_plugin_cache(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        cache = {"values": [1]}
        real_convert = h.runner.plugin.runtime_config_for_function_output

        def convert(config, current, item, **kwargs):
            candidate = real_convert(config, current, item, **kwargs)
            candidate["cache_probe"] = cache
            return candidate

        def cleanup(settlement):
            cache["values"].append(2)

        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_function_output", convert)
        monkeypatch.setattr(h.runner.plugin, "on_tool_call_settled", cleanup)
        assert error_codes(await answer(h, call_id)) == []
        assert cache == {"values": [1, 2]}
        assert h.session.runtime_config["cache_probe"] == {"values": [1]}
        cache["values"].append(3)
        assert h.session.runtime_config["cache_probe"] == {"values": [1]}


@pytest.mark.asyncio
async def test_timeout_candidate_does_not_alias_plugin_owned_values(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        await prepare(h)
        cache = {"values": [1]}

        def recover(config, current, settlement):
            return {**current, "cache_probe": cache}

        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_tool_call_timeout", recover)
        h.clock.advance(TIMEOUT_S)
        h.scheduler.fire_all()
        assert error_codes(await h.settle()) == ["tool_result_timeout"]
        cache["values"].append(2)
        assert h.session.runtime_config["cache_probe"] == {"values": [1]}


@pytest.mark.asyncio
@pytest.mark.parametrize("copy_failure", ["input", "candidate"])
async def test_unexpected_preparation_failure_is_correlated_and_retryable(monkeypatch, copy_failure):
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise TypeError("plugin-owned object cannot be copied")

    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        before = dict(h.session.runtime_config)
        real_convert = h.runner.plugin.runtime_config_for_function_output
        if copy_failure == "input":
            h.session.replace_runtime_config({**before, "copy_probe": Uncopyable()})
        else:

            def convert(config, current, item, **kwargs):
                return {**real_convert(config, current, item, **kwargs), "copy_probe": Uncopyable()}

            monkeypatch.setattr(h.runner.plugin, "runtime_config_for_function_output", convert)
        events = await h.run(commands.CreateItem(item=output_item(call_id), event_id="prepare-failed"))
        assert error_codes(events) == ["commit_aborted"]
        assert errors(events)[0].related_event_id == "prepare-failed"
        assert "conversation.item.created" not in types(events)
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.PENDING
        assert generation(h) == 0 and batches(h) == []
        h.session.replace_runtime_config(before)
        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_function_output", real_convert)
        assert error_codes(await answer(h, call_id)) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [ToolCallState.ACCEPTED, ToolCallState.CANCELLED])
async def test_non_timeout_decisions_cannot_enter_the_model_recovery_hook(monkeypatch, terminal):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        recoveries = []

        def poison(config, current, settlement):
            recoveries.append(settlement)
            return {"unexpected_reconfiguration": True}

        monkeypatch.setattr(h.runner.plugin, "runtime_config_for_tool_call_timeout", poison)
        if terminal is ToolCallState.ACCEPTED:
            assert error_codes(await answer(h, call_id)) == []
            assert generation(h) == 1 and len(batches(h)) == 1
        else:
            assert error_codes(await h.run(commands.CancelInput())) == []
        assert recoveries == []
        assert "unexpected_reconfiguration" not in h.session.runtime_config
