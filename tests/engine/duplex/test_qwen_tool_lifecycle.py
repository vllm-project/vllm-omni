# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen tool recovery through the real runner, with scripted output and time."""

import ast
import json
from copy import deepcopy

import pytest
import pytest_asyncio

from tests.engine.duplex.test_session_manager import FakeClock
from tests.engine.duplex.test_session_runner import tts_output
from tests.engine.duplex.test_tool_call_ledger import FakeTimerScheduler
from tests.model_executor.models.qwen3_omni.test_duplex_plugin import _publish_call, open_qwen
from vllm_omni.engine.duplex.commands import CancelFunctionCall, CreateItem, CreateResponse
from vllm_omni.engine.duplex.session.tool_calls import ToolCallState
from vllm_omni.protocol.realtime.items import normalize_conversation_item

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


async def _user_item(h, text: str) -> None:
    await h.run(
        CreateItem(
            item=normalize_conversation_item(
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
            )
        )
    )


@pytest_asyncio.fixture
async def qwen_tool_call():
    """Publish one real session call; virtual timers avoid sleeping for expiry."""
    clock = FakeClock(100.0)
    timers = FakeTimerScheduler()
    h = await open_qwen(
        clock=clock,
        modalities=["text"],
        extra_body={"tool_result_timeout_s": 5.0, "function_call_retry_policy": "never"},
    )
    try:
        h.runner.tool_calls.set_scheduler(timers)
        await _user_item(h, "original input")
        await _publish_call(h, "call_original", name="lookup", arguments='{"key":"original"}')
        yield h, clock, timers
    finally:
        await h.manager.shutdown()


async def _abandon(h, clock, timers, reason: str):
    if reason == "timeout":
        clock.advance(5.0)
        timers.fire_all()
        return await h.settle()
    return await h.run(CancelFunctionCall(call_id="call_original", reason="failed"))


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["timeout", "failed"])
async def test_abandoned_result_is_refused_and_its_notice_reaches_the_next_prompt(qwen_tool_call, reason):
    h, clock, timers = qwen_tool_call
    events = await _abandon(h, clock, timers, reason)
    expected_state = ToolCallState.TIMED_OUT if reason == "timeout" else ToolCallState.CANCELLED
    expected_reason = "timeout" if reason == "timeout" else "client_failed"
    record = h.runner.tool_calls.record("call_original")
    assert record is not None and record.state is expected_state
    settled = [event.to_realtime()["event"] for event in events if event.type == "duplex.function_call.settled"]
    assert [(event["call_id"], event["state"], event["reason"]) for event in settled] == [
        ("call_original", expected_state.value, expected_reason)
    ]
    notices = deepcopy(h.session.runtime_config["qwen_tool_turns"])
    assert len(notices) == 1 and notices[0]["notice"] is True
    notice = notices[0]["messages"][-1]
    assert notice["tool_call_id"] == "call_original"
    payload = json.loads(notice["content"])
    assert (payload["status"], payload["reason"]) == ("abandoned", expected_reason)
    assert not h.port.submissions, "A notice waits for new user input"

    rejected_output = '{"marker":"rejected-result"}'
    rejected = await h.run(
        CreateItem(
            item=normalize_conversation_item(
                {"type": "function_call_output", "call_id": "call_original", "output": rejected_output}
            )
        )
    )
    assert [event.code for event in rejected if event.type == "error"] == ["invalid_function_call_output"]
    assert not any(event.type == "conversation.item.created" for event in rejected)
    assert record.state is expected_state
    assert h.session.runtime_config["qwen_tool_turns"] == notices

    await _user_item(h, "next input")
    started = await h.run(CreateResponse())
    assert not any(event.type == "error" for event in started)
    assert len(h.port.submissions) == 1
    submission = h.port.submissions[-1]
    messages = ast.literal_eval(submission.prompt["prompt"])
    assert [message for message in messages if message["role"] == "tool"] == [notice]
    assert rejected_output not in submission.prompt["prompt"]
    response_id = h.session.active_response_id
    # This text is supplied at the model boundary; its meaning is irrelevant.
    completed = await h.deliver_and_settle(
        tts_output(submission.context.request_id, samples=0, text="93", finished=True), stage_id=0
    )
    completed += await h.deliver_and_settle(
        tts_output(submission.context.request_id, samples=0, text="", finished=True), stage_id=2
    )
    assert any(
        event.type == "response.done" and event.response_id == response_id and event.status == "completed"
        for event in completed
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "timeout", "failed"])
async def test_scripted_repeat_is_suppressed_without_republishing_or_rebinding(qwen_tool_call, state):
    h, clock, timers = qwen_tool_call
    if state != "pending":
        await _abandon(h, clock, timers, state)
    original = h.runner.tool_calls.record("call_original")
    assert original is not None
    original_state, origin = original.state, original.origin
    await _user_item(h, "next input")
    await h.run(CreateResponse())
    assert len(h.port.submissions) == 1
    request_id = h.port.submissions[-1].context.request_id
    # Force a retry through Qwen's real parser; no model choice can skip the guard.
    events = await h.deliver_and_settle(
        tts_output(
            request_id,
            samples=0,
            text='<tool_call>{"name":"lookup","arguments":{"key":"original"}}</tool_call>',
            finished=True,
        ),
        stage_id=0,
    )
    suppressed = [event.to_realtime()["event"] for event in events if event.type == "duplex.function_call.suppressed"]
    assert [(event["blocked_by_call_id"], event["blocked_by_state"]) for event in suppressed] == [
        ("call_original", original_state.value)
    ]
    assert not any(event.type == "response.function_call_arguments.done" for event in events)
    assert len(h.runner.tool_calls) == 1
    assert h.runner.tool_calls.record("call_original") is original
    assert original.state is original_state and original.origin == origin
