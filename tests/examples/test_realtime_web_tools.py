# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import json

import pytest

from examples.online_serving.realtime_web import server, tools

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _run(executor: tools.RealtimeToolExecutor, name: str, arguments: str) -> dict:
    return json.loads(asyncio.run(executor.run(name, arguments)))


def test_realtime_tools_use_the_flat_session_shape() -> None:
    names = [tool["name"] for tool in tools.RealtimeToolExecutor().realtime_tools]
    assert names == [
        "calculator",
        "get_current_datetime",
        "get_city_weather",
        "convert_currency",
        "DeepSeek",
        "get_current_location",
        "WebSearch",
    ]
    tool = tools.RealtimeToolExecutor().realtime_tools[0]
    assert tool["type"] == "function"
    assert tool["parameters"]["required"] == ["expression"]


def test_calculator_returns_the_result_and_rejects_code() -> None:
    executor = tools.RealtimeToolExecutor()
    assert _run(executor, "calculator", '{"expression": "17 * 23"}') == {
        "ok": True,
        "result": {"expression": "17 * 23", "result": 391, "summary": "17 * 23 = 391"},
    }
    assert _run(executor, "calculator", '{"expression": "__import__(\'os\')"}')["error"]["code"] == (
        "calculator_unsupported_function"
    )


def test_bad_calls_become_error_outputs() -> None:
    executor = tools.RealtimeToolExecutor()
    assert _run(executor, "rm_rf", "{}")["error"]["code"] == "unknown_tool"
    assert _run(executor, "calculator", "not json")["error"]["code"] == "invalid_tool_arguments"
    assert _run(executor, "calculator", '{"expr": "1"}')["error"]["code"] == "invalid_tool_arguments"


def test_weather_aliases_map_to_the_registered_tool() -> None:
    assert tools._canonical_tool_name("get_weather") == "get_city_weather"
    assert tools._canonical_tool_name("Get City Weather") == "get_city_weather"
    assert tools._canonical_tool_name("get_current_location") == "get_current_location"


def test_optional_tools_need_their_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_REALTIME_WEB_TOOL_DDG", "1")
    names = [tool["name"] for tool in tools.RealtimeToolExecutor().realtime_tools]
    assert "duckduckgo_search" in names
    assert "WebFetch" not in names


def test_webfetch_blocks_private_hosts() -> None:
    with pytest.raises(ValueError, match="webfetch_ssrf_blocked"):
        tools._check_ssrf("http://127.0.0.1:8091/")


class _Tools:
    realtime_tools = [{"type": "function", "name": "calculator"}]

    async def run(self, name: str, arguments: str) -> str:
        return json.dumps({"name": name, "arguments": arguments})


def test_session_update_gets_tools_only_when_it_has_none() -> None:
    update = json.dumps({"type": "session.update", "session": {"instructions": "x"}})
    assert json.loads(server._with_session_tools(update, _Tools()))["session"] == {
        "instructions": "x",
        "tools": _Tools.realtime_tools,
    }
    own = json.dumps({"type": "session.update", "session": {"tools": []}})
    assert server._with_session_tools(own, _Tools()) == own
    append = json.dumps({"type": "input_audio_buffer.append", "audio": "AA"})
    assert server._with_session_tools(append, _Tools()) == append


def test_function_call_is_answered_then_followed_up() -> None:
    sent: list[dict] = []

    class _Backend:
        async def send(self, message: str) -> None:
            sent.append(json.loads(message))

    event = {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "call_id": "call_1",
            "name": "calculator",
            "arguments": '{"expression": "1+1"}',
        },
    }
    call = server._finished_function_call(event)
    assert call is not None
    assert server._finished_function_call({"type": "response.function_call_arguments.done", "call_id": "x"}) is None
    assert server._finished_function_call({"type": "response.output_item.done", "item": {"type": "message"}}) is None
    asyncio.run(server._answer_function_call(_Backend(), _Tools(), call))
    assert [message["type"] for message in sent] == ["conversation.item.create", "response.create"]
    sent.clear()
    asyncio.run(server._answer_function_call(_Backend(), _Tools(), call, follow_up=False))
    assert [message["type"] for message in sent] == ["conversation.item.create"]
    assert server._is_user_commit('{"type":"input_audio_buffer.commit"}')


def test_follow_up_waits_until_the_tool_response_is_done() -> None:
    sent: list[dict] = []

    class _Backend:
        async def send(self, message: str) -> None:
            sent.append(json.loads(message))

    call = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "calculator",
        "arguments": '{"expression": "1+1"}',
    }
    state = {"count": 0, "gen": 0, "pending_gen": 0, "awaiting_done": False, "ready": False}
    server._hold_follow_up_for_response_done(state)
    asyncio.run(server._answer_function_call(_Backend(), _Tools(), call, follow_state=state))
    assert [message["type"] for message in sent] == ["conversation.item.create"]
    assert state["ready"] is True
    assert server._take_deferred_follow_up(state) is True
    assert server._take_deferred_follow_up(state) is False

    state["ready"] = True
    state["pending_gen"] = 0
    state["gen"] = 1
    assert server._take_deferred_follow_up(state) is False

    class _CommitDuringTool(_Tools):
        async def run(self, name: str, arguments: str) -> str:
            state["gen"] += 1
            state["awaiting_done"] = False
            state["ready"] = False
            return await super().run(name, arguments)

    state.update({"awaiting_done": True, "ready": False, "gen": 0, "pending_gen": 0})
    sent.clear()
    asyncio.run(server._answer_function_call(_Backend(), _CommitDuringTool(), call, follow_state=state))
    assert [message["type"] for message in sent] == ["conversation.item.create"]
    assert state["ready"] is False

    state.update({"awaiting_done": False, "ready": False, "gen": 2, "pending_gen": 0})
    sent.clear()
    asyncio.run(server._answer_function_call(_Backend(), _Tools(), call, follow_state=state))
    assert [message["type"] for message in sent] == ["conversation.item.create", "response.create"]
    assert not server._is_user_commit('{"type":"input_audio_buffer.append"}')
    item = sent[0]["item"]
    assert item["type"] == "function_call_output"
    assert item["call_id"] == "call_1"
    assert json.loads(item["output"]) == {"name": "calculator", "arguments": '{"expression": "1+1"}'}


def test_latest_frame_is_kept_and_passed_only_when_the_executor_asks() -> None:
    state = {"gen": 0, "count": 0, "awaiting_done": False, "ready": False, "last_frame": "", "turn_frame": ""}
    server._note_client_event(state, {"type": "input_audio_buffer.append", "video_frames": ["frame-a", "frame-b"]})
    server._note_client_event(state, {"type": "input_audio_buffer.append", "audio": "pcm"})
    assert state["last_frame"] == "frame-b"
    server._note_client_event(state, {"type": "input_audio_buffer.commit"})
    assert state["turn_frame"] == "frame-b"

    seen: dict[str, str] = {}

    class _FrameTools(_Tools):
        uses_latest_frame = True

        async def run(self, name: str, arguments: str, frame_b64: str = "") -> str:
            seen["frame"] = frame_b64
            return await super().run(name, arguments)

    sent: list[dict] = []

    class _Backend:
        async def send(self, message: str) -> None:
            sent.append(json.loads(message))

    call = {"type": "function_call", "call_id": "call_1", "name": "calculator", "arguments": "{}"}
    asyncio.run(
        server._answer_function_call(
            _Backend(), _FrameTools(), call, follow_up=False, follow_state=state, frame_b64=state["turn_frame"]
        )
    )
    assert seen["frame"] == "frame-b"
    asyncio.run(server._answer_function_call(_Backend(), _Tools(), call, follow_up=False, follow_state=state))
    assert sent[-1]["item"]["output"]


def test_two_calls_in_one_turn_follow_up_once() -> None:
    sent: list[dict] = []

    class _Backend:
        async def send(self, message: str) -> None:
            sent.append(json.loads(message))

    state = {"count": 0, "gen": 0, "pending_gen": 0, "awaiting_done": False, "ready": False}
    call = {"type": "function_call", "call_id": "call_1", "name": "calculator", "arguments": "{}"}
    server._hold_follow_up_for_response_done(state)
    server._begin_tool_call(state)
    server._begin_tool_call(state)
    asyncio.run(server._answer_function_call(_Backend(), _Tools(), call, follow_state=state))
    asyncio.run(server._answer_function_call(_Backend(), _Tools(), {**call, "call_id": "call_2"}, follow_state=state))
    assert [message["type"] for message in sent] == ["conversation.item.create", "conversation.item.create"]
    assert server._take_deferred_follow_up(state) is True


def test_tool_followups_allow_five_rounds_then_one_final_answer() -> None:
    assert server._MAX_TOOL_FOLLOWUPS == 5
    state = {
        "count": 5,
        "gen": 1,
        "pending_gen": 0,
        "awaiting_done": False,
        "ready": False,
        "final_answer": False,
        "inflight": 0,
        "inflight_gen": None,
    }
    # At the cap, the proxy still opens one answer create (AURA injects the nudge).
    follow_up = state["count"] < server._MAX_TOOL_FOLLOWUPS
    final_answer = not follow_up and not state.get("final_answer")
    assert follow_up is False and final_answer is True
    state["final_answer"] = True
    assert not (not follow_up and not state.get("final_answer"))
    server._note_client_event(state, {"type": "input_audio_buffer.commit"})
    assert state["count"] == 0 and state["final_answer"] is False


def test_vision_tools_hide_search_and_report_unknown_names() -> None:
    executor = tools.VisionToolExecutor()
    names = [tool["name"] for tool in executor.realtime_tools]
    assert names == [
        "deepseek_vlm_search",
        "get_current_datetime",
        "DeepSeek",
        "get_current_location",
        "get_city_weather",
        "visual_web_search",
    ]
    assert executor.uses_latest_frame is True
    assert asyncio.run(executor.run("draw", "{}")) == "tool not found: draw"
    assert "no video frame" in asyncio.run(executor.run("visual_web_search", '{"query": "logo"}'))


@pytest.mark.asyncio
async def test_backend_close_cancels_outstanding_tool_tasks() -> None:
    cancelled = asyncio.Event()

    class _Tools:
        async def run(self, name, arguments, frame_b64=""):
            del name, arguments, frame_b64
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return "{}"

    class _Backend:
        def __init__(self) -> None:
            self._sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._sent:
                await asyncio.sleep(30)
            self._sent = True
            return json.dumps(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_x",
                        "name": "calculator",
                        "arguments": "{}",
                    },
                }
            )

        async def send(self, message):
            del message

    class _Client:
        async def send_text(self, message):
            del message

        async def send_bytes(self, message):
            del message

    task = asyncio.create_task(
        server._pump_backend_to_client(
            _Client(),
            _Backend(),
            _Tools(),
            {"count": 0, "gen": 0, "final_answer": False},
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


class _QueueBackend:
    """Backend stub: yields pushed events and records what the proxy sends."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[str] = asyncio.Queue()
        self.sent: list[dict] = []

    def push(self, event: dict) -> None:
        self.inbox.put_nowait(json.dumps(event))

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        return await self.inbox.get()

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    def types(self) -> list[str]:
        return [message["type"] for message in self.sent]


class _NullClient:
    async def send_text(self, message: str) -> None:
        del message

    async def send_bytes(self, message: bytes) -> None:
        del message


def _call_event(call_id: str, response_id: str | None) -> dict:
    event: dict = {
        "type": "response.output_item.done",
        "item": {"type": "function_call", "call_id": call_id, "name": "calculator", "arguments": "{}"},
    }
    if response_id is not None:
        event["response_id"] = response_id
    return event


def _proxy_state() -> dict:
    return {
        "count": 0,
        "gen": 0,
        "pending_gen": 0,
        "awaiting_done": False,
        "ready": False,
        "final_answer": False,
        "inflight": 0,
        "inflight_gen": None,
        "last_frame": "",
        "turn_frame": "",
        "pending_response_id": None,
    }


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("result_before_other_done", [False, True])
async def test_follow_up_waits_for_the_response_that_owns_the_call(result_before_other_done: bool) -> None:
    # A is still draining when B publishes a call. A finishes first.
    gate = asyncio.Event()

    class _GatedTools(_Tools):
        async def run(self, name: str, arguments: str) -> str:
            await gate.wait()
            return await super().run(name, arguments)

    backend = _QueueBackend()
    state = _proxy_state()
    server._note_client_event(state, {"type": "input_audio_buffer.commit"})
    pump = asyncio.create_task(server._pump_backend_to_client(_NullClient(), backend, _GatedTools(), state))
    try:
        backend.push(_call_event("call_b", "resp_B"))
        await _settle()
        if result_before_other_done:
            gate.set()
            await _settle()
            assert backend.types() == ["conversation.item.create"]
        backend.push({"type": "response.done", "response_id": "resp_A"})
        await _settle()
        if not result_before_other_done:
            gate.set()
            await _settle()
        # A's completion must not release B's follow-up: B is still active.
        assert "response.create" not in backend.types()
        backend.push({"type": "response.done", "response_id": "resp_B"})
        await _settle()
        assert backend.types().count("response.create") == 1
        assert backend.types()[-1] == "response.create"
    finally:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)


@pytest.mark.asyncio
async def test_follow_up_keeps_the_old_release_when_the_response_id_is_missing() -> None:
    backend = _QueueBackend()
    state = _proxy_state()
    server._note_client_event(state, {"type": "input_audio_buffer.commit"})
    pump = asyncio.create_task(server._pump_backend_to_client(_NullClient(), backend, _Tools(), state))
    try:
        backend.push(_call_event("call_x", None))
        await _settle()
        backend.push({"type": "response.done"})
        await _settle()
        assert backend.types() == ["conversation.item.create", "response.create"]
    finally:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)


@pytest.mark.asyncio
async def test_tool_frame_belongs_to_the_turn_that_made_the_call() -> None:
    seen: list[str] = []

    class _FrameTools(_Tools):
        uses_latest_frame = True

        async def run(self, name: str, arguments: str, frame_b64: str = "") -> str:
            seen.append(frame_b64)
            return await super().run(name, arguments)

    backend = _QueueBackend()
    state = _proxy_state()
    pump = asyncio.create_task(server._pump_backend_to_client(_NullClient(), backend, _FrameTools(), state))
    try:
        # Turn 1: camera on.
        server._note_client_event(state, {"type": "input_audio_buffer.append", "video_frames": ["frame-1"]})
        server._note_client_event(state, {"type": "input_audio_buffer.commit"})
        backend.push(_call_event("call_1", "resp_1"))
        await _settle()
        # A second call in the same turn keeps that turn's frame.
        backend.push(_call_event("call_1b", "resp_1"))
        await _settle()
        # Turn 2: camera turned off, audio only.
        server._note_client_event(state, {"type": "input_audio_buffer.append", "audio": "pcm"})
        server._note_client_event(state, {"type": "input_audio_buffer.commit"})
        backend.push(_call_event("call_2", "resp_2"))
        await _settle()
    finally:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)
    assert seen == ["frame-1", "frame-1", ""]
