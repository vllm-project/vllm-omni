# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The client tool API and ``DuplexToolRunner`` against a fake socket."""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.clients.test_duplex_client import SESSION_CLOSED, SESSION_CREATED, FakeSocket, make_client
from vllm_omni.clients.tool_calls import DuplexToolRunner, tool_session_defaults

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def call_event(call_id="call_a", name="get_weather", arguments='{"city":"Taipei"}'):
    return {
        "type": "response.output_item.done",
        "response_id": "resp_1",
        "item": {"type": "function_call", "id": "item_1", "call_id": call_id, "name": name, "arguments": arguments},
    }


def settled_event(call_id="call_a"):
    return {
        "type": "duplex.tool_call.settled",
        "event": {"call_id": call_id, "state": "timed_out", "reason": "timeout"},
    }


def sent(sock: FakeSocket, event_type: str) -> list[dict[str, object]]:
    return [event for event in sock.sent if event.get("type") == event_type]


def session_with(continuation: str) -> dict[str, object]:
    created = json.loads(json.dumps(SESSION_CREATED))
    created["session"]["capabilities"] = {"tool_result_continuation": continuation}
    return created


async def settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def run_with(sock: FakeSocket, handlers, body) -> None:
    client, _ = make_client(sock)
    async with client:
        task = asyncio.create_task(DuplexToolRunner(client, handlers).run())
        await settle()
        await body()
        await settle()
        sock.feed(SESSION_CLOSED)
        await asyncio.wait_for(task, 2)


def test_defaults_keep_pending_tools_across_an_interruption():
    assert tool_session_defaults() == {"tool_interrupt_policy": "keep"}


@pytest.mark.asyncio
async def test_client_helpers_send_documented_events():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        await client.send_tool_result("call_a", "20")
        await client.cancel_tool_call("call_a", reason="failed")
        items = sent(sock, "conversation.item.create")
        cancels = sent(sock, "tool_call.cancel")
        sock.feed(SESSION_CLOSED)
    assert items[0]["item"] == {"type": "function_call_output", "call_id": "call_a", "output": "20"}
    assert (cancels[0]["call_id"], cancels[0]["reason"]) == ("call_a", "failed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("continuation", "creates"),
    [("client_response_create", 1), ("automatic", 0), ("next_observation", 0)],
)
async def test_result_is_sent_and_continued_only_where_the_model_needs_it(continuation, creates):
    sock = FakeSocket()
    sock.feed(session_with(continuation))
    seen: list[dict[str, object]] = []

    async def get_weather(arguments):
        seen.append(arguments)
        return "25C"

    async def body():
        sock.feed(call_event())

    await run_with(sock, {"get_weather": get_weather}, body)
    assert seen == [{"city": "Taipei"}]
    assert sent(sock, "conversation.item.create")[0]["item"]["output"] == "25C"
    assert len(sent(sock, "response.create")) == creates


@pytest.mark.asyncio
async def test_a_failed_handler_reports_failure_without_its_text():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)

    async def broken(arguments):
        raise RuntimeError("secret detail")

    async def body():
        sock.feed(call_event())

    await run_with(sock, {"get_weather": broken}, body)
    assert sent(sock, "conversation.item.create") == []
    assert sent(sock, "tool_call.cancel")[0]["reason"] == "failed"
    assert "secret detail" not in json.dumps(sent(sock, "tool_call.cancel"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event",
    [call_event(name="unknown"), call_event(arguments="not json"), call_event(arguments="[1]")],
)
async def test_an_unusable_call_is_reported_as_failed(event):
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)

    async def get_weather(arguments):
        return "25C"

    async def body():
        sock.feed(event)

    await run_with(sock, {"get_weather": get_weather}, body)
    assert sent(sock, "tool_call.cancel")[0]["reason"] == "failed"


@pytest.mark.asyncio
async def test_a_settled_call_stops_its_tool_and_drops_the_result():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    cancelled = asyncio.Event()

    async def slow(arguments):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "late"

    async def body():
        sock.feed(call_event())
        await settle()
        sock.feed(settled_event())

    await run_with(sock, {"get_weather": slow}, body)
    assert cancelled.is_set()
    assert sent(sock, "conversation.item.create") == []


@pytest.mark.asyncio
async def test_a_slow_tool_does_not_block_the_next_one():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    release = asyncio.Event()
    order: list[str] = []

    async def slow(arguments):
        await release.wait()
        order.append("slow")
        return "slow"

    async def fast(arguments):
        order.append("fast")
        release.set()
        return "fast"

    async def body():
        sock.feed(call_event("call_a", name="slow"))
        await settle()
        sock.feed(call_event("call_b", name="fast"))

    await run_with(sock, {"slow": slow, "fast": fast}, body)
    assert order == ["fast", "slow"]


@pytest.mark.asyncio
async def test_a_tool_error_does_not_fail_a_consumer_waiting_for_an_event():
    sock = FakeSocket()
    sock.feed(SESSION_CREATED)
    client, _ = make_client(sock)
    async with client:
        waiter = asyncio.create_task(client.wait_for("response.done", timeout_s=2))
        await settle()
        sock.feed({"type": "error", "error": {"code": "tool_result_timeout", "message": "expired"}})
        sock.feed({"type": "response.done", "response": {"id": "resp_1"}})
        event = await asyncio.wait_for(waiter, 2)
        assert event.type == "response.done"
        sock.feed(SESSION_CLOSED)
