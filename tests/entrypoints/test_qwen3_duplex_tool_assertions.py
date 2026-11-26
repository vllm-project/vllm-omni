# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check tool lifecycle events independently of the model's chosen answer text."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from tests.clients.test_duplex_client import FakeSocket, make_client
from tests.entrypoints.openai_api import test_qwen3_omni_duplex_playback as playback
from vllm_omni.clients.duplex import EventCollector

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def run_tool_round_trip(monkeypatch):
    """Run the live smoke's event checks against a real client and scripted socket."""
    monkeypatch.setattr(playback, "TURN_TIMEOUT_S", 1.0)

    def run(
        answer: str | None,
        *,
        ack_call_id: str | None = "call_weather",
        response_id="continuation",
        corrupt_result=False,
        status="completed",
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
                    sock.feed({"type": "response.done", "response": {"id": rid, "status": status}})

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
                                            "item": {
                                                **item,
                                                "call_id": ack_call_id,
                                                "output": "wrong-result" if corrupt_result else item["output"],
                                            },
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
        test = playback.TestQwen3OmniDuplexTools().test_live_tool_runner_completes_a_tool_round_trip
        test(Mock(spec=playback._DuplexTestRuntime, server=object()))

    return run


@pytest.mark.parametrize(
    ("answer", "ack_call_id", "response_id", "corrupt_result", "status", "accepted"),
    [
        ("An arbitrary reply.", "call_weather", "continuation", False, "completed", True),
        ("17", "call_weather", "continuation", False, "completed", True),
        ("117", "call_weather", "continuation", False, "completed", True),
        ("17:00", "call_weather", "continuation", False, "completed", True),
        ("", "call_weather", "continuation", False, "completed", True),
        (None, "call_weather", "continuation", False, "completed", False),
        ("reply", None, "continuation", False, "completed", False),
        ("reply", "wrong_call", "continuation", False, "completed", False),
        ("reply", "call_weather", "initial", False, "completed", False),
        ("reply", "call_weather", "continuation", True, "completed", False),
        ("reply", "call_weather", "continuation", False, "failed", False),
    ],
)
def test_tool_round_trip_checks_events_not_answer_text(
    run_tool_round_trip, answer, ack_call_id, response_id, corrupt_result, status, accepted
):
    if accepted:
        run_tool_round_trip(
            answer, ack_call_id=ack_call_id, response_id=response_id, corrupt_result=corrupt_result, status=status
        )
    else:
        with pytest.raises(AssertionError):
            run_tool_round_trip(
                answer, ack_call_id=ack_call_id, response_id=response_id, corrupt_result=corrupt_result, status=status
            )
