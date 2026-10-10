# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin, QwenDataPlane
from vllm_omni.model_executor.models.qwen3_omni.duplex.tools import ThinkerToolCursor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_CALL = '<tool_call>{"name": "get_current_time", "arguments": {}}</tool_call>'


def test_partial_tool_text_is_not_a_call_or_speech():
    cursor = ThinkerToolCursor()
    speech, call = cursor.absorb('<tool_call>{"name": "get_current')
    assert speech == ""
    assert call is None
    assert cursor.holding_open_call()


def test_complete_call_is_recognized_once():
    cursor = ThinkerToolCursor()
    speech, call = cursor.absorb(f"The time is {_CALL}")
    assert speech == "The time is "
    assert call is not None
    assert call["name"] == "get_current_time"
    assert json.loads(call["arguments"]) == {}
    again_speech, again_call = cursor.absorb(cursor.text)
    assert again_speech == ""
    assert again_call is None


def test_unclosed_call_is_dropped_at_end_of_turn():
    cursor = ThinkerToolCursor()
    cursor.absorb("Hello <tool_call>")
    assert cursor.finish() == ""
    assert cursor.tool_turn


def _output(text: str, *, request_id: str = "req", stage_id: int = 0, finished: bool = False, audio: bool = False):
    completion = SimpleNamespace(text=text, multimodal_output={})
    mm = {"audio": np.ones(16, dtype=np.float32), "sr": 16000} if audio else {}
    return {
        "data_plane_outputs": [
            SimpleNamespace(
                request_id=request_id,
                stage_id=stage_id,
                finished=finished,
                outputs=(completion,),
                multimodal_output=mm,
            )
        ]
    }


def test_data_plane_holds_markup_and_emits_one_call():
    plane = QwenDataPlane(lambda *args: "AAAA")
    context = {"modalities": ("text", "audio"), "response_format": "pcm16", "turn_id": 1}
    first = list(plane.project(_output(f"Hi {_CALL}", audio=True), context=context))
    assert [event.get("text") for event in first] == ["Hi ", ""]
    assert first[0]["audio_data"] == ""
    assert first[1]["qwen_function_call"]["name"] == "get_current_time"
    assert "qwen_function_call" not in first[0]
    second = list(plane.project(_output(f"Hi {_CALL}", audio=True), context=context))
    assert all("qwen_function_call" not in event for event in second)
    audio = list(plane.project(_output("", stage_id=2, audio=True), context=context))
    assert all(event.get("audio_data", "") == "" for event in audio)


def test_plain_text_still_passes_through():
    plane = QwenDataPlane(lambda *args: "AAAA")
    events = list(
        plane.project(
            _output("hello", audio=True, finished=True),
            context={"modalities": ("text", "audio"), "turn_id": 0},
        )
    )
    assert len(events) == 1
    assert events[0]["text"] == "hello"
    assert events[0]["audio_data"] == "AAAA"
    assert events[0]["end_of_turn"] is True
    assert "qwen_function_call" not in events[0]


def test_tools_reach_the_chat_template_and_results_join_the_next_prompt():
    seen: dict[str, object] = {}

    def apply_chat_template(messages, **kwargs):
        seen["messages"] = messages
        seen["kwargs"] = kwargs
        return "prompt"

    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    plugin.processor = SimpleNamespace(apply_chat_template=apply_chat_template)
    plugin.validate_client_extra_body({"realtime_tools": [{"type": "function", "name": "get_current_time"}]})
    runtime = plugin.runtime_config_for_update(
        DuplexSessionConfig(
            model="qwen",
            extra_body={"realtime_tools": [{"type": "function", "name": "get_current_time", "description": "Now"}]},
        ),
        {},
    )
    call = plugin.parse_function_call(
        {
            "qwen_function_call": {
                "call_id": "call_1",
                "name": "get_current_time",
                "arguments": "{}",
            }
        }
    )
    assert call == {"call_id": "call_1", "name": "get_current_time", "arguments": "{}"}
    updated = plugin.runtime_config_for_function_output(
        DuplexSessionConfig(model="qwen"),
        runtime,
        {"type": "function_call_output", "call_id": "call_1", "output": "2026-10-05T00:00:00+00:00"},
    )
    assert updated["tool_followup_ready"] is True
    plugin.plan_append(
        request_id="follow-up",
        fence=None,
        session_config={"qwen_messages": [{"role": "user", "content": "what time is it"}]},
        runtime_config=updated,
        seq=1,
        turn_seq=1,
        payload={"type": "conversation"},
        final=True,
        sampling_params=None,
    )
    messages = seen["messages"]
    assert messages[-2]["role"] == "assistant"
    assert messages[-2]["tool_calls"][0]["function"]["name"] == "get_current_time"
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "2026-10-05T00:00:00+00:00",
    }
    assert seen["kwargs"]["tools"] == [
        {"type": "function", "function": {"name": "get_current_time", "description": "Now"}}
    ]


def test_prompt_without_tools_does_not_pass_a_tools_argument():
    seen: dict[str, object] = {}

    def apply_chat_template(messages, **kwargs):
        seen["kwargs"] = kwargs
        return repr(messages)

    plugin = Qwen3OmniDuplexPlugin(lambda *args: None)
    plugin.processor = SimpleNamespace(apply_chat_template=apply_chat_template)
    plan = plugin.plan_append(
        request_id="plain",
        fence=None,
        session_config={"qwen_messages": [{"role": "user", "content": "hello"}]},
        runtime_config={},
        seq=1,
        turn_seq=1,
        payload={"type": "conversation"},
        final=True,
        sampling_params=None,
    )
    assert "tools" not in seen["kwargs"]
    assert plan.prompt["prompt"] == repr([{"role": "user", "content": [{"type": "text", "text": "hello"}]}])


def test_example_local_tool_returns_utc_time():
    path = Path(__file__).resolve().parents[4] / "examples/online_serving/qwen3_omni/duplex_tool_client.py"
    spec = importlib.util.spec_from_file_location("qwen_duplex_tool_client", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = module.execute_local_tool("get_current_time", "{}")
    assert output.endswith("+00:00")
    assert json.loads(module.execute_local_tool("lookup", "{}"))["error"] == "unknown tool lookup"
