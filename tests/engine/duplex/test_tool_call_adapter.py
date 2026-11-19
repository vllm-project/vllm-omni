# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""What happens to a tool call that did not complete.

Ledger cases enumerate the decisions by hand. Runner cases drive them through
the real mailbox and the real Nemotron plugin over a recording stage port, so
they show what reaches the client and the model's runtime config. They do not
show how a model responds to a notice; that needs a real checkpoint.
"""

from __future__ import annotations

import json

import pytest

from tests.engine.duplex.test_session_manager import FakeClock
from tests.engine.duplex.test_session_runner import append_audio
from tests.engine.duplex.test_session_runner_nemotron_voicechat import (
    TIMEOUT_S,
    answer,
    batches,
    decode_a_different_call,
    deliver,
    error_codes,
    open_nemotron_harness,
    open_timed_harness,
    prepare,
)
from tests.engine.duplex.test_tool_call_ledger import FakeTimerScheduler
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import (
    DuplexCapabilities,
    DuplexConfigError,
    DuplexSessionConfig,
    effective_function_call_retry_policy,
    resolve_function_call_policy,
)
from vllm_omni.engine.duplex.contracts import ToolCallAbandonReason, abandoned_tool_call_notice
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError, validate_session_tools
from vllm_omni.engine.duplex.realtime_commands import translate_realtime_command
from vllm_omni.engine.duplex.session.tool_calls import (
    ToolCallAcceptance,
    ToolCallLedger,
    ToolCallOrigin,
    ToolCallRejection,
    ToolCallSettlement,
    ToolCallState,
    tool_call_fingerprint,
)
from vllm_omni.protocol.duplex import RealtimeProtocolError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def origin(call_id="call_a", *, epoch=3, name="get_weather", arguments='{"city":"Taipei","unit":"c"}'):
    return ToolCallOrigin(
        call_id=call_id,
        epoch=epoch,
        turn_id=7,
        response_id="response_a",
        item_id=f"item_{call_id}",
        name=name,
        arguments=arguments,
    )


def ledger_with(**kwargs) -> tuple[ToolCallLedger, list[ToolCallSettlement]]:
    settlements: list[ToolCallSettlement] = []
    ledger = ToolCallLedger(max_calls=8, clock=FakeClock(100.0), on_settle=settlements.append, **kwargs)
    return ledger, settlements


# --------------------------------------------------------------------------- #
# Fingerprints and the repeat guard                                           #
# --------------------------------------------------------------------------- #


def test_fingerprint_ignores_key_order_but_not_values_or_names():
    same = tool_call_fingerprint("f", '{"a":1,"b":2}')
    assert tool_call_fingerprint("f", '{"b": 2, "a": 1}') == same
    assert tool_call_fingerprint("f", '{"a":1,"b":3}') != same
    assert tool_call_fingerprint("g", '{"a":1,"b":2}') != same
    assert tool_call_fingerprint("f", "not json") == tool_call_fingerprint("f", "not json")
    assert tool_call_fingerprint("f", None) == tool_call_fingerprint("f", "")


def test_a_cancelled_call_blocks_its_repeat_for_the_session_by_default():
    ledger, _ = ledger_with()
    ledger.register(origin())

    assert ledger.cancel("call_a", reason=ToolCallAbandonReason.CLIENT_CANCELLED, epoch=3) is None

    assert ledger.blocked_by("get_weather", '{"unit":"c","city":"Taipei"}') == "call_a"
    assert ledger.blocked_by("get_weather", '{"city":"Paris","unit":"c"}') is None


def test_a_pending_call_blocks_its_repeat_until_it_settles():
    user_turns = [4]
    ledger, _ = ledger_with(user_turn_seq=lambda: user_turns[0])
    ledger.register(origin())
    arguments = '{"unit":"c","city":"Taipei"}'

    assert ledger.blocked_by("get_weather", arguments) == "call_a"
    # A user turn lifts a block on an abandoned call, not on a pending one.
    user_turns[0] = 5
    assert ledger.blocked_by("get_weather", arguments) == "call_a"
    assert ledger.blocked_by("get_weather", '{"city":"Paris","unit":"c"}') is None

    acceptance = ledger.begin_accept("call_a")
    assert isinstance(acceptance, ToolCallAcceptance)
    assert ledger.commit_accept(acceptance, epoch=3) is True
    assert ledger.blocked_by("get_weather", arguments) is None


def test_an_accepted_call_does_not_block_its_repeat():
    ledger, _ = ledger_with()
    ledger.register(origin())
    acceptance = ledger.begin_accept("call_a")
    assert isinstance(acceptance, ToolCallAcceptance)
    assert ledger.commit_accept(acceptance, epoch=3) is True
    assert ledger.blocked_by("get_weather", '{"city":"Taipei","unit":"c"}') is None


def test_only_a_later_user_turn_lifts_the_block():
    user_turns = [4]
    ledger, _ = ledger_with(user_turn_seq=lambda: user_turns[0])
    ledger.register(origin())
    ledger.cancel("call_a", reason=ToolCallAbandonReason.CLIENT_FAILED, epoch=3)
    arguments = '{"city":"Taipei","unit":"c"}'

    assert ledger.blocked_by("get_weather", arguments) == "call_a"
    user_turns[0] = 5
    assert ledger.blocked_by("get_weather", arguments) is None
    # The lift is consumed, so a later failure blocks from its own turn.
    ledger.register(origin("call_b"))
    ledger.cancel("call_b", reason=ToolCallAbandonReason.CLIENT_FAILED, epoch=3)
    assert ledger.blocked_by("get_weather", arguments) == "call_b"


def test_a_turn_before_the_failure_does_not_lift_the_block():
    user_turns = [4]
    ledger, _ = ledger_with(user_turn_seq=lambda: user_turns[0])
    ledger.register(origin())
    user_turns[0] = 5
    ledger.cancel("call_a", reason=ToolCallAbandonReason.CLIENT_CANCELLED, epoch=3)
    assert ledger.blocked_by("get_weather", '{"city":"Taipei","unit":"c"}') == "call_a"


def test_close_forgets_abandoned_calls():
    ledger, _ = ledger_with()
    ledger.register(origin())
    ledger.cancel("call_a", reason=ToolCallAbandonReason.CLIENT_CANCELLED, epoch=3)
    ledger.register(origin("call_b", arguments="{}"))
    ledger.close()
    assert ledger.blocked_by("get_weather", '{"city":"Taipei","unit":"c"}') is None
    assert ledger.blocked_by("get_weather", "{}") is None


# --------------------------------------------------------------------------- #
# Per-call cancel                                                             #
# --------------------------------------------------------------------------- #


def test_cancel_settles_once_with_its_reason():
    ledger, settlements = ledger_with()
    ledger.register(origin())

    assert ledger.cancel("missing", reason=ToolCallAbandonReason.CLIENT_FAILED, epoch=3) is ToolCallRejection.UNKNOWN
    assert ledger.cancel("call_a", reason=ToolCallAbandonReason.CLIENT_FAILED, epoch=4) is None
    assert ledger.cancel("call_a", reason=ToolCallAbandonReason.CLIENT_FAILED, epoch=4) is ToolCallRejection.NOT_PENDING

    assert [(item.state, item.reason, item.settled_epoch) for item in settlements] == [
        (ToolCallState.CANCELLED, ToolCallAbandonReason.CLIENT_FAILED, 4)
    ]
    assert ledger.begin_accept("call_a") is ToolCallRejection.NOT_PENDING


def test_a_cancel_racing_an_acceptance_voids_it():
    ledger, _ = ledger_with()
    ledger.register(origin())
    acceptance = ledger.begin_accept("call_a")
    assert isinstance(acceptance, ToolCallAcceptance)
    ledger.cancel("call_a", reason=ToolCallAbandonReason.CLIENT_CANCELLED, epoch=3)
    assert ledger.commit_accept(acceptance, epoch=3) is False


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"type": "function_call.cancel", "call_id": "call_a"}, "cancelled"),
        ({"type": "function_call.cancel", "call_id": "call_a", "reason": "failed"}, "failed"),
    ],
)
def test_cancel_command_decodes(payload, expected):
    command = translate_realtime_command(payload)
    assert isinstance(command, commands.CancelFunctionCall)
    assert (command.call_id, command.reason) == ("call_a", expected)


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "function_call.cancel"},
        {"type": "function_call.cancel", "call_id": ""},
        {"type": "function_call.cancel", "call_id": "call_a", "reason": "because"},
    ],
)
def test_a_malformed_cancel_is_refused(payload):
    with pytest.raises(RealtimeProtocolError) as refused:
        translate_realtime_command(payload)
    assert refused.value.code == "invalid_function_call_cancel"


# --------------------------------------------------------------------------- #
# Keeping calls across an epoch advance                                       #
# --------------------------------------------------------------------------- #


def test_default_policy_still_cancels_on_an_epoch_advance():
    ledger, settlements = ledger_with()
    ledger.register(origin())
    assert [item.call_id for item in ledger.invalidate_epoch(4)] == ["call_a"]
    assert settlements[0].reason is ToolCallAbandonReason.EPOCH_CANCELLED


def test_a_kept_call_is_admitted_in_the_new_epoch():
    ledger, settlements = ledger_with(keep_on_epoch_advance=True)
    ledger.register(origin())

    assert ledger.invalidate_epoch(4) == ()
    assert settlements == []
    acceptance = ledger.begin_accept("call_a")
    assert isinstance(acceptance, ToolCallAcceptance)
    assert acceptance.epoch == 4
    assert ledger.commit_accept(acceptance, epoch=4) is True


def test_an_epoch_advance_during_the_await_does_not_void_a_kept_acceptance():
    ledger, _ = ledger_with(keep_on_epoch_advance=True)
    ledger.register(origin())
    acceptance = ledger.begin_accept("call_a")
    assert isinstance(acceptance, ToolCallAcceptance)

    ledger.invalidate_epoch(4)

    assert ledger.revalidate(acceptance, epoch=4) is None
    assert ledger.commit_accept(acceptance, epoch=4) is True


def test_a_kept_call_times_out_in_the_current_epoch():
    clock = FakeClock(100.0)
    settlements: list[ToolCallSettlement] = []
    ledger = ToolCallLedger(
        max_calls=8, clock=clock, timeout_s=5.0, on_settle=settlements.append, keep_on_epoch_advance=True
    )
    ledger.set_scheduler(FakeTimerScheduler())
    ledger.register(origin())
    ledger.invalidate_epoch(4)
    clock.advance(5.0)

    assert ledger.expire("call_a") is True

    assert settlements[0].settled_epoch == 4


# --------------------------------------------------------------------------- #
# Configuration and capabilities                                              #
# --------------------------------------------------------------------------- #


def test_policies_default_and_validate():
    assert resolve_function_call_policy({}, "function_call_retry_policy") == "user_turn"
    assert resolve_function_call_policy(None, "function_call_interrupt_policy") == "cancel"
    assert (
        resolve_function_call_policy({"function_call_retry_policy": "never"}, "function_call_retry_policy") == "never"
    )
    with pytest.raises(DuplexConfigError) as refused:
        DuplexSessionConfig(model="m", extra_body={"function_call_interrupt_policy": "later"})
    assert refused.value.code == "invalid_function_call_policy"


def test_an_invalid_policy_in_an_update_is_refused_as_invalid():
    config = DuplexSessionConfig(model="m")
    with pytest.raises(DuplexConfigError) as refused:
        config.apply_realtime_update({"extra_body": {"function_call_retry_policy": "sometimes"}}, session_id="s")
    assert refused.value.code == "invalid_function_call_policy"


def test_the_effective_retry_policy_needs_a_user_turn_boundary():
    with_boundary = DuplexCapabilities(clears_function_call_block_on_user_commit=True)
    assert effective_function_call_retry_policy(with_boundary, {}) == "user_turn"
    assert effective_function_call_retry_policy(with_boundary, {"function_call_retry_policy": "never"}) == "never"
    assert effective_function_call_retry_policy(DuplexCapabilities(), {}) == "never"


def test_a_policy_cannot_change_after_the_session_opens():
    config = DuplexSessionConfig(model="m", extra_body={"function_call_interrupt_policy": "keep"})
    with pytest.raises(DuplexConfigError) as refused:
        config.apply_realtime_update({"extra_body": {"function_call_interrupt_policy": "cancel"}}, session_id="s")
    assert refused.value.code == "function_call_policy_update_unsupported"
    # A client that resends the whole session is not changing anything.
    config.apply_realtime_update({"extra_body": {"function_call_interrupt_policy": "keep"}}, session_id="s")


def test_tools_are_refused_where_a_model_would_ignore_them():
    tools = [{"type": "function", "name": "f"}]
    with pytest.raises(DuplexRuntimeConfigError) as refused:
        validate_session_tools(DuplexCapabilities(), {"realtime_tools": tools})
    assert refused.value.code == "tools_unsupported"
    # An empty declaration asks for nothing.
    validate_session_tools(DuplexCapabilities(), {"realtime_tools": []})
    validate_session_tools(DuplexCapabilities(), {})


@pytest.mark.parametrize("tool_choice", ["none", "required", {"type": "function", "name": "f"}])
def test_tool_choice_other_than_auto_is_refused(tool_choice):
    capabilities = DuplexCapabilities(tool_mode="declared", max_tools=None)
    extra_body = {"realtime_tools": [{"name": "f"}], "realtime_tool_choice": tool_choice}
    with pytest.raises(DuplexRuntimeConfigError) as refused:
        validate_session_tools(capabilities, extra_body)
    assert refused.value.code == "unsupported_tool_choice"


def test_more_tools_than_a_model_supports_are_refused():
    capabilities = DuplexCapabilities(tool_mode="declared", max_tools=1)
    tools = [{"name": "f"}, {"name": "g"}]
    with pytest.raises(DuplexRuntimeConfigError) as refused:
        validate_session_tools(capabilities, {"realtime_tools": tools})
    assert refused.value.code == "too_many_tools"
    validate_session_tools(capabilities, {"realtime_tools": tools[:1]})
    unlimited = DuplexCapabilities(tool_mode="declared", max_tools=None)
    validate_session_tools(unlimited, {"realtime_tools": tools})


def test_the_tool_limit_covers_any_sequence():
    capabilities = DuplexCapabilities(tool_mode="declared", max_tools=1)
    with pytest.raises(DuplexRuntimeConfigError) as refused:
        validate_session_tools(capabilities, {"realtime_tools": ({"name": "f"}, {"name": "g"})})
    assert refused.value.code == "too_many_tools"


def test_tool_choice_auto_is_accepted():
    capabilities = DuplexCapabilities(tool_mode="declared", max_tools=None)
    validate_session_tools(capabilities, {"realtime_tools": [{"name": "f"}], "realtime_tool_choice": "auto"})


def test_capabilities_reach_the_client_payload():
    payload = DuplexCapabilities(
        tool_mode="declared",
        function_call_output_continuation="automatic",
        max_tools=5,
        clears_function_call_block_on_user_commit=True,
    ).as_dict()
    assert payload["tool_mode"] == "declared"
    assert payload["function_call_output_continuation"] == "automatic"
    assert payload["max_tools"] == 5
    assert payload["clears_function_call_block_on_user_commit"] is True


# --------------------------------------------------------------------------- #
# Through the session                                                         #
# --------------------------------------------------------------------------- #


def wire(events, event_type: str) -> list[dict[str, object]]:
    """The ``event`` body of each ``duplex.*`` event of this type on the wire."""
    found = [event.to_realtime() for event in events]
    return [event["event"] for event in found if event.get("type") == event_type]


async def repeat_call(h):
    events: list = []
    for token_id in (20, 7, 21):
        events.extend(await deliver(h, nvc_function_token=[token_id]))
    return events


@pytest.mark.asyncio
async def test_a_timeout_reaches_the_client_and_the_model_and_refuses_the_late_result(monkeypatch):
    async with open_timed_harness(monkeypatch) as h:
        call_id = await prepare(h)
        before = len(batches(h))

        h.clock.advance(TIMEOUT_S)
        h.scheduler.fire_all()
        events = await h.settle()

        settled = wire(events, "duplex.function_call.settled")
        assert [(item["call_id"], item["state"], item["reason"]) for item in settled] == [
            (call_id, "timed_out", "timeout")
        ]
        assert len(batches(h)) == before + 1

        late = await answer(h, call_id)
        assert "invalid_function_call_output" in error_codes(late)


@pytest.mark.asyncio
async def test_a_failed_tool_never_reaches_the_model_as_a_result(monkeypatch):
    async with open_nemotron_harness(monkeypatch) as h:
        call_id = await prepare(h)

        events = await h.run(commands.CancelFunctionCall(call_id=call_id, reason="failed"))

        assert [item["reason"] for item in wire(events, "duplex.function_call.settled")] == ["client_failed"]
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.CANCELLED
        assert len(batches(h)) == 1

        again = await h.run(commands.CancelFunctionCall(call_id=call_id))
        assert error_codes(again) == ["invalid_function_call_cancel"]
        assert len(wire(again, "duplex.function_call.settled")) == 0


@pytest.mark.asyncio
async def test_a_repeat_is_suppressed_and_not_shown_to_the_client(monkeypatch):
    async with open_nemotron_harness(monkeypatch) as h:
        call_id = await prepare(h)
        await h.run(commands.CancelFunctionCall(call_id=call_id))
        records = len(h.runner.tool_calls)

        events = await repeat_call(h)

        suppressed = wire(events, "duplex.function_call.suppressed")
        assert [item["blocked_by_call_id"] for item in suppressed] == [call_id]
        assert not [event for event in events if event.type == "response.function_call_arguments.done"]
        assert len(h.runner.tool_calls) == records


@pytest.mark.asyncio
async def test_a_repeat_of_a_pending_call_is_suppressed_without_a_notice(monkeypatch):
    async with open_nemotron_harness(monkeypatch) as h:
        call_id = await prepare(h)

        events = await repeat_call(h)

        suppressed = wire(events, "duplex.function_call.suppressed")
        assert [(item["blocked_by_call_id"], item["blocked_by_state"]) for item in suppressed] == [(call_id, "pending")]
        assert not [event for event in events if event.type == "response.function_call_arguments.done"]
        assert batches(h) == []
        assert error_codes(await answer(h, call_id)) == []
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.ACCEPTED


@pytest.mark.asyncio
async def test_a_tool_result_is_not_a_user_turn_and_lifts_no_block(monkeypatch):
    async with open_nemotron_harness(monkeypatch) as h:
        h.runner.tool_calls._user_turn_seq = h.session.user_turn_seq
        call_a = await prepare(h)
        await h.run(commands.CancelFunctionCall(call_id=call_a))
        original_decode = h.tokenizer.decode
        decode_a_different_call(h)
        call_b = await prepare(h)
        turns = h.session.user_turn_seq()

        assert error_codes(await answer(h, call_b)) == []

        assert h.session.user_turn_seq() == turns
        h.tokenizer.decode = original_decode
        suppressed = wire(await repeat_call(h), "duplex.function_call.suppressed")
        assert [item["blocked_by_call_id"] for item in suppressed] == [call_a]


@pytest.mark.asyncio
async def test_a_config_error_from_a_plugin_check_is_reported_on_update(monkeypatch):
    async with open_nemotron_harness(monkeypatch) as h:
        seen: list[object] = []

        def refuse(extra_body):
            seen.append(extra_body)
            if isinstance(extra_body, dict) and extra_body.get("realtime_tools"):
                raise DuplexConfigError("refused", code="unsupported_tool_choice")

        monkeypatch.setattr(h.runner.plugin, "validate_client_extra_body", refuse)
        events = await h.run(commands.UpdateSession(patch={"tools": [{"type": "function", "name": "f"}]}))

        assert error_codes(events) == ["unsupported_tool_choice"]
        assert not h.runner.closing
        # The hook gets the whole extra_body, as it does at session open.
        assert isinstance(seen[-1], dict) and "auto_response" in seen[-1]


@pytest.mark.asyncio
async def test_the_session_reports_the_policy_it_applies(monkeypatch):
    async with open_nemotron_harness(monkeypatch, extra_body={"function_call_retry_policy": "user_turn"}) as h:
        assert h.session.as_public_dict()["function_call_retry_policy"] == "never"


@pytest.mark.asyncio
async def test_nemotron_has_no_user_turn_so_a_repeat_stays_blocked(monkeypatch):
    async with open_nemotron_harness(monkeypatch) as h:
        call_id = await prepare(h)
        await h.run(commands.CancelFunctionCall(call_id=call_id))
        await h.run(append_audio(samples=1280))

        events = await repeat_call(h)

        assert len(wire(events, "duplex.function_call.suppressed")) == 1


@pytest.mark.asyncio
async def test_an_effective_cancel_is_reported_and_keeps_nemotron_queue_clean(monkeypatch):
    async with open_nemotron_harness(monkeypatch) as h:
        call_id = await prepare(h)

        events = await h.run(commands.CancelInput())

        settled = wire(events, "duplex.function_call.settled")
        assert [(item["call_id"], item["reason"]) for item in settled] == [(call_id, "epoch_cancelled")]
        assert batches(h) == []


@pytest.mark.asyncio
async def test_keep_policy_admits_a_result_after_an_effective_cancel(monkeypatch):
    async with open_nemotron_harness(monkeypatch, extra_body={"function_call_interrupt_policy": "keep"}) as h:
        call_id = await prepare(h)
        epoch = h.session.epoch

        cancelled = await h.run(commands.CancelInput())

        assert h.session.epoch == epoch + 1
        assert wire(cancelled, "duplex.function_call.settled") == []
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.PENDING
        result = await answer(h, call_id)
        assert error_codes(result) == []
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.ACCEPTED


@pytest.mark.asyncio
async def test_a_tool_declared_on_update_is_refused_on_a_tool_less_model(monkeypatch):
    async with open_nemotron_harness(monkeypatch) as h:
        monkeypatch.setattr(h.session.capabilities, "tool_mode", "none")
        events = await h.run(commands.UpdateSession(patch={"tools": [{"type": "function", "name": "f"}]}))
        assert error_codes(events) == ["tools_unsupported"]


def test_notice_text_is_fixed_and_names_only_the_reason():
    notice = json.loads(abandoned_tool_call_notice(ToolCallAbandonReason.TIMEOUT))
    assert notice["status"] == "abandoned"
    assert notice["reason"] == "timeout"
    assert set(notice) == {"status", "reason", "instruction"}
