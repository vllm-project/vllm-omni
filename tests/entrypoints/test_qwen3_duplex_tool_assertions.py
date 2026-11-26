# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the live tool-test assertions with a real client and scripted socket."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from tests.clients.test_duplex_client import FakeSocket, make_client
from tests.entrypoints.openai_api import test_qwen3_omni_duplex_playback as playback
from vllm_omni.clients.duplex import EventCollector

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def run_tool_case(monkeypatch):
    """Run an unchanged GPU test body against controlled protocol events."""
    monkeypatch.setattr(playback, "TURN_TIMEOUT_S", 1.0)

    def run(
        test_name: str, answer: str | None, *, ack_call_id: str | None = "call_weather", response_id="continuation"
    ):
        async def in_session(config, body):
            sock = FakeSocket()
            sock.feed(
                {
                    "type": "session.created",
                    "session": {
                        "id": "session_test",
                        "capabilities": {"function_call_output_continuation": "client_response_create"},
                    },
                }
            )
            client, _ = make_client(sock, config=config)
            async with client:
                collector = EventCollector()
                consumer = asyncio.create_task(collector.consume(client))

                def complete(rid: str, text: str) -> None:
                    sock.feed({"type": "response.created", "response": {"id": rid}})
                    sock.feed({"type": "response.output_text.delta", "response_id": rid, "delta": text})
                    sock.feed({"type": "response.done", "response": {"id": rid, "status": "completed"}})

                async def serve():
                    cursor = 0
                    initial = True
                    while True:
                        for event in sock.sent[cursor:]:
                            cursor += 1
                            item = event.get("item")
                            if (
                                event["type"] == "conversation.item.create"
                                and item.get("type") == "function_call_output"
                            ):
                                if ack_call_id is not None:
                                    sock.feed(
                                        {
                                            "type": "conversation.item.created",
                                            "item": {**item, "call_id": ack_call_id},
                                        }
                                    )
                            elif event["type"] == "response.create":
                                if initial:
                                    initial = False
                                    sock.feed({"type": "response.created", "response": {"id": "initial"}})
                                    sock.feed(
                                        {
                                            "type": "response.output_text.delta",
                                            "response_id": "initial",
                                            "delta": "17:00",
                                        }
                                    )
                                    sock.feed(
                                        {
                                            "type": "response.output_item.done",
                                            "response_id": "initial",
                                            "item": {
                                                "type": "function_call",
                                                "call_id": "call_weather",
                                                "name": "get_weather",
                                                "arguments": '{"city":"Paris"}',
                                            },
                                        }
                                    )
                                    sock.feed(
                                        {"type": "response.done", "response": {"id": "initial", "status": "completed"}}
                                    )
                                elif answer is not None:
                                    complete(response_id, answer)
                        await asyncio.sleep(0)

                server = asyncio.create_task(serve())
                try:
                    await body(playback._ToolSession(client, collector))
                finally:
                    sock.feed({"type": "session.closed", "session_id": "session_test", "reason": "client_close"})
                    server.cancel()
                    consumer.cancel()
                    await asyncio.gather(server, consumer, return_exceptions=True)

        monkeypatch.setattr(
            playback, "_run_tool_session", lambda server, config, body: asyncio.run(in_session(config, body))
        )
        test = getattr(playback.TestQwen3OmniDuplexPlayback(), test_name)
        test(Mock(spec=playback._DuplexTestRuntime, server=object()))

    return run


@pytest.mark.parametrize(("answer", "accepted"), [("17", True), (" 17\n", True), ("117", False), ("17:00", False)])
def test_weather_answer_matches_the_entire_temperature(run_tool_case, answer, accepted):
    name = "test_a_tool_result_reaches_the_next_answer"
    if accepted:
        run_tool_case(name, answer)
    else:
        with pytest.raises(AssertionError):
            run_tool_case(name, answer)


@pytest.mark.parametrize(
    ("answer", "ack_call_id", "response_id", "accepted"),
    [
        ("17", "call_weather", "continuation", True),
        (None, "call_weather", "continuation", False),
        ("17", None, "continuation", False),
        ("17", "wrong_call", "continuation", False),
        ("17", "call_weather", "initial", False),
        ("117", "call_weather", "continuation", False),
    ],
)
def test_runner_requires_its_acknowledged_new_response(run_tool_case, answer, ack_call_id, response_id, accepted):
    name = "test_immediate_runner_handler_continues_without_another_client_action"
    if accepted:
        run_tool_case(name, answer, ack_call_id=ack_call_id, response_id=response_id)
    else:
        with pytest.raises(AssertionError):
            run_tool_case(name, answer, ack_call_id=ack_call_id, response_id=response_id)
