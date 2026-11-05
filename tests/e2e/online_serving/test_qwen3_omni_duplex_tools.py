# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tool calling on a real Qwen3-Omni duplex session.

The engine tests cover the session's decisions with fake model output. Only a
real model can show what happens around them: that it emits a call, that the
result reaches its answer, and whether it stays away from a call the session
abandoned. Each test holds one tool call pending, ending it differently.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.clients.duplex import DuplexClient, EventCollector, SessionConfig

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni]

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(model=MODEL, stage_config_path=get_deploy_config_path("qwen3_omni_duplex.yaml")),
        id="qwen3-omni-duplex",
    )
]

TURN_TIMEOUT_S = 180.0
RESULT_TIMEOUT_S = 3.0
WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "The city name."}},
        "required": ["city"],
    },
}
INSTRUCTIONS = (
    "You are a helpful assistant. For any question about the weather you must call the get_weather tool "
    "instead of guessing. For anything else, answer directly in one short sentence without calling a tool."
)
WEATHER_QUESTION = "What is the weather in Paris right now?"
WEATHER_RESULT = json.dumps({"city": "Paris", "temperature_c": 17})


def session_config(**extra_body: object) -> SessionConfig:
    return SessionConfig(
        modalities=("text",),
        auto_response=False,
        temperature=0.0,
        instructions=INSTRUCTIONS,
        max_output_tokens=128,
        extra_body=extra_body,
    )


class ToolSession:
    """One connected client, with its events kept for the assertions."""

    def __init__(self, client: DuplexClient, collector: EventCollector) -> None:
        self.client = client
        self.collector = collector

    def events_since(self, start: int) -> list[dict[str, object]]:
        return self.collector.events[start:]

    def calls(self) -> list[dict[str, object]]:
        """Every function call the model published to the client."""
        items = [
            event.get("item") for event in self.collector.events if event.get("type") == "response.output_item.done"
        ]
        return [item for item in items if isinstance(item, dict) and item.get("type") == "function_call"]

    def duplex_events(self, event_type: str) -> list[dict[str, object]]:
        bodies = [event.get("event") for event in self.collector.events if event.get("type") == event_type]
        return [body for body in bodies if isinstance(body, dict)]

    def error_codes(self) -> list[object]:
        codes = []
        for event in self.collector.errors():
            error = event.get("error")
            codes.append(error.get("code") if isinstance(error, dict) else None)
        return codes

    async def finish_response(self, start: int) -> str:
        """Wait for the turn's own response, and return its text.

        The turn's response opens first. A function call inside it is projected
        as a separate response, which must not be mistaken for the end.
        """
        deadline = asyncio.get_running_loop().time() + TURN_TIMEOUT_S
        while asyncio.get_running_loop().time() < deadline:
            events = self.events_since(start)
            created = next((event for event in events if event.get("type") == "response.created"), None)
            response_id = self.collector.response_id(created) if created is not None else None
            if response_id is not None and self.collector.response_is_done(response_id):
                text = self.collector.response_text(response_id)
                print(f"[qwen-tools] answer {response_id}: {text!r}")
                return text
            await asyncio.sleep(0.1)
        raise AssertionError(f"the turn did not finish; events: {[event.get('type') for event in events][-12:]}")

    async def ask(self, text: str) -> str:
        start = len(self.collector.events)
        await self.client.send(
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
            }
        )
        await self.client.send({"type": "response.create"})
        return await self.finish_response(start)

    async def ask_for_weather(self) -> str:
        """Ask the weather question and return the call_id the model published."""
        await self.ask(WEATHER_QUESTION)
        calls = self.calls()
        assert calls, f"Qwen published no function call: {self.collector.errors()}"
        assert calls[-1]["name"] == "get_weather"
        assert "paris" in str(calls[-1]["arguments"]).lower()
        return str(calls[-1]["call_id"])

    async def wait_for_settled(self, call_id: str) -> dict[str, object]:
        deadline = asyncio.get_running_loop().time() + TURN_TIMEOUT_S
        while asyncio.get_running_loop().time() < deadline:
            for event in self.duplex_events("duplex.function_call.settled"):
                if event.get("call_id") == call_id:
                    return event
            await asyncio.sleep(0.05)
        raise AssertionError(f"call {call_id} never settled; errors: {self.collector.errors()}")

    async def continue_with_result(self, call_id: str, output: str) -> str:
        start = len(self.collector.events)
        await self.client.send_function_call_output(call_id, output)
        await self.client.send({"type": "response.create"})
        return await self.finish_response(start)

    def report(self) -> None:
        """Keep what the model did in the test output, pass or fail."""
        print(
            f"[qwen-tools] calls={len(self.calls())} "
            f"suppressed={len(self.duplex_events('duplex.function_call.suppressed'))} errors={self.error_codes()}"
        )
        for event in self.collector.events:
            event_type = str(event.get("type"))
            if event_type in {"response.created", "response.done", "duplex.function_call.suppressed"}:
                print(f"[qwen-tools] {event_type} {self.collector.response_id(event)}")
            elif event_type == "response.output_item.done":
                print(f"[qwen-tools] output_item {json.dumps(event.get('item'))[:200]}")


async def in_session(url: str, config: SessionConfig, body) -> None:
    collector = EventCollector()
    async with DuplexClient(url, model=MODEL, config=config, reconnect=None) as client:
        consumer = asyncio.create_task(collector.consume(client))
        await asyncio.sleep(0)
        session = ToolSession(client, collector)
        try:
            await client.send({"type": "session.update", "session": {"tools": [WEATHER_TOOL]}})
            capabilities = client.session_info.get("capabilities")
            assert isinstance(capabilities, dict)
            assert capabilities["tool_mode"] == "declared"
            assert capabilities["function_call_output_continuation"] == "client_response_create"
            await body(session)
        finally:
            session.report()
            await client.close()
            await asyncio.wait_for(consumer, timeout=10)


def run(omni_server, config: SessionConfig, body) -> None:
    url = f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1"
    asyncio.run(in_session(url, config, body))


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_a_tool_result_reaches_the_next_answer(omni_server):
    async def body(session: ToolSession) -> None:
        call_id = await session.ask_for_weather()
        answer = await session.continue_with_result(call_id, WEATHER_RESULT)
        assert "17" in answer, answer
        assert session.error_codes() == []

    run(omni_server, session_config(), body)


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_the_client_can_keep_talking_while_a_tool_is_pending(omni_server):
    async def body(session: ToolSession) -> None:
        call_id = await session.ask_for_weather()

        reply = await session.ask("Before that, what is the capital of Japan?")
        assert "tokyo" in reply.lower(), reply
        assert session.duplex_events("duplex.function_call.settled") == []
        assert len(session.calls()) == 1, session.calls()

        await session.continue_with_result(call_id, WEATHER_RESULT)
        assert session.error_codes() == []

    run(omni_server, session_config(function_call_interrupt_policy="keep"), body)


@pytest.mark.xfail(
    strict=True,
    reason="#8857 splices a result before later user turns, so the model answers the newest question instead",
)
@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_a_result_after_a_later_turn_reaches_the_answer(omni_server):
    async def body(session: ToolSession) -> None:
        call_id = await session.ask_for_weather()
        await session.ask("Before that, what is the capital of Japan?")

        answer = await session.continue_with_result(call_id, WEATHER_RESULT)

        assert "17" in answer, answer

    run(omni_server, session_config(function_call_interrupt_policy="keep"), body)


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_a_timed_out_call_is_not_repeated_on_the_next_turn(omni_server):
    """Nothing blocks a repeat here: the user turn lifts the block, so this measures the notice."""

    async def body(session: ToolSession) -> None:
        call_id = await session.ask_for_weather()
        settled = await session.wait_for_settled(call_id)
        assert (settled["state"], settled["reason"]) == ("timed_out", "timeout")

        await session.ask("Thanks. What is the capital of Japan?")

        assert len(session.calls()) == 1, session.calls()

    run(omni_server, session_config(tool_result_timeout_s=RESULT_TIMEOUT_S), body)


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_a_late_result_never_reaches_the_answer(omni_server):
    async def body(session: ToolSession) -> None:
        call_id = await session.ask_for_weather()
        await session.wait_for_settled(call_id)

        await session.client.send_function_call_output(call_id, json.dumps({"city": "Paris", "temperature_c": 93}))
        answer = await session.ask("Thanks. What is the capital of Japan?")

        assert "invalid_function_call_output" in session.error_codes()
        assert "93" not in answer, answer

    run(omni_server, session_config(tool_result_timeout_s=RESULT_TIMEOUT_S), body)


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_a_repeated_question_cannot_republish_an_abandoned_call(omni_server):
    """The same question invites the same call, so the guard has to hold under ``never``."""

    async def body(session: ToolSession) -> None:
        call_id = await session.ask_for_weather()
        await session.wait_for_settled(call_id)

        await session.ask(WEATHER_QUESTION)

        assert len(session.calls()) == 1, session.calls()
        for suppressed in session.duplex_events("duplex.function_call.suppressed"):
            assert suppressed["blocked_by_call_id"] == call_id

    config = session_config(tool_result_timeout_s=RESULT_TIMEOUT_S, function_call_retry_policy="never")
    run(omni_server, config, body)


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=2)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_a_failed_tool_ends_the_call_and_the_session_carries_on(omni_server):
    async def body(session: ToolSession) -> None:
        call_id = await session.ask_for_weather()

        await session.client.cancel_function_call(call_id, reason="failed")
        settled = await session.wait_for_settled(call_id)
        assert (settled["state"], settled["reason"]) == ("cancelled", "client_failed")

        reply = await session.ask("Thanks. What is the capital of Japan?")
        assert "tokyo" in reply.lower(), reply
        assert len(session.calls()) == 1, session.calls()
        assert session.error_codes() == []

    run(omni_server, session_config(), body)
