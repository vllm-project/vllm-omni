# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for AURA_v2 duplex tool-calling: parsing, plugin hooks, history."""

from __future__ import annotations

import json

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.model_executor.models.aura_omni.duplex.history import (
    drop_session_history,
    get_or_create_session_history,
)
from vllm_omni.model_executor.models.aura_omni.duplex.plugin import AuraDuplexPlugin
from vllm_omni.model_executor.models.aura_omni.duplex.tools_utils import (
    build_tools_block,
    chat_template_tools,
    extract_finished_tool_calls,
    parse_completed_tool_call,
    safe_streaming_prefix,
    strip_tool_call_markup,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _encode_audio(audio: object, sample_rate: int, fmt: str, speed: float | None) -> str | None:
    del audio, sample_rate, fmt, speed
    return "ZmFrZQ=="


# -- tools_utils: parsing AURA_v2's own <function=...><parameter=...> body --


def test_extract_finished_tool_calls_single_parameter() -> None:
    text = (
        "<|respond|>先幫你查一下。"
        "<tool_call>\n<function=calculator>\n<parameter=expression>\n17 * 23\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    [call] = extract_finished_tool_calls(text)
    assert call is not None
    assert call["name"] == "calculator"
    assert json.loads(call["arguments"]) == {"expression": "17 * 23"}
    assert call["call_id"].startswith("call_")


def test_extract_finished_tool_calls_multiple_parameters_and_json_coercion() -> None:
    text = (
        "<tool_call>\n<function=convert_currency>\n"
        "<parameter=amount>\n100\n</parameter>\n"
        "<parameter=from_currency>\nUSD\n</parameter>\n"
        "<parameter=to_currency>\nTWD\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    [call] = extract_finished_tool_calls(text)
    assert call is not None
    assert json.loads(call["arguments"]) == {"amount": 100, "from_currency": "USD", "to_currency": "TWD"}


def test_extract_finished_tool_calls_rejects_nan_literal() -> None:
    text = "<tool_call>\n<function=calculator>\n<parameter=expression>\nNaN\n</parameter>\n</function>\n</tool_call>"
    [call] = extract_finished_tool_calls(text)
    assert call is not None
    # "NaN" is not valid JSON by strict rules; kept as the raw string, not
    # coerced into a float that ``json.dumps`` could not even re-encode.
    assert json.loads(call["arguments"]) == {"expression": "NaN"}


def test_extract_finished_tool_calls_empty_without_a_closed_tag() -> None:
    assert extract_finished_tool_calls("just a normal spoken reply") == []
    assert extract_finished_tool_calls("<tool_call>still open, no close tag") == []


def test_extract_finished_tool_calls_returns_every_call_in_order() -> None:
    text = (
        "<|respond|><tool_call>\n<function=get_current_datetime>\n</function>\n</tool_call>\n"
        "<tool_call>\nno function tag\n</tool_call>\n"
        "<tool_call>\n<function=get_current_location>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=still_open>"
    )
    calls = extract_finished_tool_calls(text)
    assert [call["name"] for call in calls] == ["get_current_datetime", "get_current_location"]
    assert calls[0]["call_id"] != calls[1]["call_id"]


def test_parse_completed_tool_call_missing_function_name_is_none() -> None:
    assert parse_completed_tool_call("no function tag here") is None


def test_strip_tool_call_markup_drops_tag_and_suffix() -> None:
    text = "先幫你查一下。<tool_call>\n<function=calculator>\n</function>\n</tool_call>"
    assert strip_tool_call_markup(text) == "先幫你查一下。"
    assert strip_tool_call_markup("no tag at all") == "no tag at all"


def test_safe_streaming_prefix_holds_back_partial_and_complete_tags() -> None:
    assert safe_streaming_prefix("先幫你查一下。") == "先幫你查一下。"
    assert safe_streaming_prefix("先幫你查一下。<tool_c") == "先幫你查一下。"
    assert safe_streaming_prefix("先幫你查一下。<tool_call>\n<function=x>") == "先幫你查一下。"
    # A tail that stops matching the tag prefix releases what was held back.
    assert safe_streaming_prefix("先幫你查一下。<not_a_tag") == "先幫你查一下。<not_a_tag"


def test_chat_template_tools_converts_realtime_flat_shape() -> None:
    converted = chat_template_tools(
        [
            {"type": "function", "name": "calculator", "description": "d", "parameters": {"type": "object"}},
            {"type": "function", "function": {"name": "already_nested"}},
            {"type": "function", "name": ""},
            "not-a-mapping",
        ]
    )
    assert converted == [
        {
            "type": "function",
            "function": {"name": "calculator", "description": "d", "parameters": {"type": "object"}},
        },
        {"type": "function", "function": {"name": "already_nested"}},
    ]
    assert chat_template_tools("not-a-list") == []


def test_build_tools_block_empty_vs_nonempty() -> None:
    assert build_tools_block([]) == ""
    block = build_tools_block([{"type": "function", "function": {"name": "calculator"}}])
    assert block.startswith("# Tools")
    assert '"name": "calculator"' in block
    assert "<function=example_function_name>" in block


# -- plugin hooks --


def test_capabilities_support_text_only_turn_for_tool_followup() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    assert plugin.capabilities(max_sessions=1).supports_text_only_turn is True


def test_parse_function_call_reads_aura_function_call_key() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    call = plugin.parse_function_call(
        {"aura_function_call": {"call_id": "call_1", "name": "calculator", "arguments": '{"expression": "1+1"}'}}
    )
    assert call == {"call_id": "call_1", "name": "calculator", "arguments": '{"expression": "1+1"}'}
    assert plugin.parse_function_call({"text": "no call here"}) is None
    assert plugin.parse_function_call({"aura_function_call": {"name": "missing_call_id"}}) is None


def test_function_call_releases_turn_only_when_nothing_left_to_speak() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    assert plugin.function_call_releases_turn({"aura_tool_call_ends_turn": True}) is True
    assert plugin.function_call_releases_turn({"aura_tool_call_ends_turn": False}) is False
    assert plugin.function_call_releases_turn({}) is False


def test_has_pending_tool_followup() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    assert plugin.has_pending_tool_followup({}) is False
    assert plugin.has_pending_tool_followup({"aura_tool_pending": []}) is False
    assert plugin.has_pending_tool_followup({"aura_tool_pending": [{"output": "18C"}]}) is True
    assert plugin.has_pending_tool_followup({"aura_tool_pending": [{"output": "18C"}, {"output": 123}]}) is False


def test_runtime_config_for_function_output_accumulates_pending_results() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    config = DuplexSessionConfig(model="aurateam/AURA")
    runtime = {"aura_system_prompt": "sys"}
    for call_id in ("call_1", "call_2"):
        runtime = plugin.note_published_function_call(runtime, call_id=call_id, history_len=0)
    assert runtime["aura_tool_batch"] == ["call_1", "call_2"]
    updated = plugin.runtime_config_for_function_output(
        config,
        runtime,
        {"call_id": "call_1", "name": "get_city_weather", "arguments": '{"city": "台北"}', "output": "18C sunny"},
    )
    assert updated["aura_system_prompt"] == "sys"  # existing keys preserved
    first = {"call_id": "call_1", "name": "get_city_weather", "arguments": '{"city": "台北"}', "output": "18C sunny"}
    assert updated["aura_tool_pending"] == [first]
    updated = plugin.runtime_config_for_function_output(
        config,
        updated,
        {"call_id": "call_2", "name": "get_current_datetime", "arguments": "{}", "output": "10:00"},
    )
    assert updated["aura_tool_pending"] == [
        first,
        {"call_id": "call_2", "name": "get_current_datetime", "arguments": "{}", "output": "10:00"},
    ]


def test_runtime_config_for_function_output_ignores_call_outside_current_batch() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    runtime = {"aura_tool_batch": ["call_new"]}
    updated = plugin.runtime_config_for_function_output(
        DuplexSessionConfig(model="aurateam/AURA"),
        runtime,
        {"call_id": "call_old", "name": "get_current_datetime", "arguments": "{}", "output": "10:00"},
    )
    assert updated == runtime
    assert plugin.has_pending_tool_followup(updated) is False


@pytest.mark.parametrize(
    "item",
    [
        {"name": "n", "arguments": "{}", "output": "ok"},  # missing call_id
        {"call_id": "c", "name": "n", "arguments": "{}", "output": 1},  # output not a string
        {"call_id": "c", "name": "n", "output": "ok"},  # missing arguments
    ],
)
def test_runtime_config_for_function_output_rejects_malformed_item(item: dict[str, object]) -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    with pytest.raises(DuplexRuntimeConfigError):
        plugin.runtime_config_for_function_output(DuplexSessionConfig(model="aurateam/AURA"), {}, item)


def test_runtime_patch_for_submitted_prompt_clears_pending_once_consumed() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    assert plugin.runtime_patch_for_submitted_prompt({"text": "x"}) is None
    prompt = {"_aura_tool_pending_consumed": True, "other": 1}
    patch = plugin.runtime_patch_for_submitted_prompt(prompt)
    assert patch == {"aura_tool_pending": None, "aura_tool_batch": None}
    assert "_aura_tool_pending_consumed" not in prompt


@pytest.mark.asyncio
async def test_prepare_runtime_config_and_update_read_realtime_tools() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    config = DuplexSessionConfig(
        model="aurateam/AURA",
        extra_body={"realtime_tools": [{"type": "function", "name": "calculator"}]},
    )
    runtime = await plugin.prepare_runtime_config(config, model_config=None)
    assert runtime["aura_tools"] == [{"type": "function", "function": {"name": "calculator"}}]

    updated = plugin.runtime_config_for_update(
        DuplexSessionConfig(model="aurateam/AURA", extra_body={"realtime_tools": []}),
        runtime,
    )
    assert "aura_tools" not in updated


def test_plan_append_passes_aura_tools_through() -> None:
    plugin = AuraDuplexPlugin(_encode_audio)
    fence = DuplexFence("sess-tools", epoch=0, turn_id=1)
    payload = {
        "type": "audio",
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "audio": "",
        "final": True,
        "aura_turn_commit": True,
        "is_speech": False,
        "video_frames": [],
    }
    tools = [{"type": "function", "function": {"name": "calculator"}}]
    import base64

    import numpy as np

    payload["audio"] = base64.b64encode(np.zeros(800, dtype="<f4").tobytes()).decode("ascii")
    plan = plugin.plan_append(
        request_id="req",
        fence=fence,
        session_config={},
        runtime_config={"aura_tools": tools},
        seq=1,
        turn_seq=1,
        payload=payload,
        final=True,
        sampling_params=SamplingParams(max_tokens=8),
    )
    assert plan.prompt["additional_information"]["aura_tools"] == tools


def test_plan_append_tool_followup_builds_pad_and_appends_history() -> None:
    drop_session_history("sess-followup")
    history = get_or_create_session_history("sess-followup")
    history.begin_user_turn("台北天氣如何？")
    history.commit_turn("<|respond|><tool_call>...</tool_call>")

    plugin = AuraDuplexPlugin(_encode_audio)
    fence = DuplexFence("sess-followup", epoch=0, turn_id=2)
    plan = plugin.plan_append(
        request_id="req",
        fence=fence,
        session_config={},
        runtime_config={
            "aura_system_prompt": "sys",
            "aura_tool_pending": [
                {
                    "call_id": "call_1",
                    "name": "get_city_weather",
                    "arguments": '{"city": "台北"}',
                    "output": "18C sunny",
                }
            ],
        },
        seq=1,
        turn_seq=1,
        payload={"type": "conversation"},
        final=True,
        sampling_params=SamplingParams(max_tokens=8),
    )
    info = plan.prompt["additional_information"]
    assert info["aura_tool_followup"] is True
    assert info["is_speech"] is False
    assert "audio" in plan.prompt["multi_modal_data"]
    assert plan.prompt["_aura_tool_pending_consumed"] is True
    assert "<tool_response>\n18C sunny\n</tool_response>" in history.render_prefix()
    drop_session_history("sess-followup")


def test_plan_append_tool_followup_nudges_after_native_round_cap() -> None:
    drop_session_history("sess-nudge")
    history = get_or_create_session_history("sess-nudge")
    history.begin_user_turn("查一下")
    history.commit_turn("<|respond|><tool_call>...</tool_call>")
    plugin = AuraDuplexPlugin(_encode_audio)
    fence = DuplexFence("sess-nudge", epoch=0, turn_id=2)
    pending = [
        {
            "call_id": "call_1",
            "name": "get_city_weather",
            "arguments": "{}",
            "output": "18C",
        }
    ]
    plan = plugin.plan_append(
        request_id="req",
        fence=fence,
        session_config={},
        runtime_config={"aura_tool_pending": pending, "aura_tool_rounds": 5},
        seq=1,
        turn_seq=1,
        payload={"type": "conversation"},
        final=True,
        sampling_params=SamplingParams(max_tokens=8),
    )
    assert plan.prompt["_aura_tool_rounds"] == 6
    rendered = history.render_prefix()
    assert "<|im_start|>assistant\n<|silent|><|im_end|>\n" in rendered
    assert "请根据当前工具结果，直接回答用户的问题。" in rendered
    assert plugin.runtime_patch_for_submitted_prompt(plan.prompt) == {
        "aura_tool_pending": None,
        "aura_tool_batch": None,
        "aura_tool_rounds": 6,
    }
    drop_session_history("sess-nudge")


def test_plan_append_tool_followup_without_pending_raises() -> None:
    drop_session_history("sess-no-pending")
    get_or_create_session_history("sess-no-pending")
    plugin = AuraDuplexPlugin(_encode_audio)
    fence = DuplexFence("sess-no-pending", epoch=0, turn_id=1)
    with pytest.raises(ValueError):
        plugin.plan_append(
            request_id="req",
            fence=fence,
            session_config={},
            runtime_config={},
            seq=1,
            turn_seq=1,
            payload={"type": "conversation"},
            final=True,
            sampling_params=SamplingParams(max_tokens=8),
        )
    drop_session_history("sess-no-pending")


# -- history --


def test_append_tool_response_renders_as_user_turn_with_tool_response_tag() -> None:
    drop_session_history("sess-hist-tool")
    history = get_or_create_session_history("sess-hist-tool")
    history.begin_user_turn("問題")
    history.commit_turn("<|respond|>查一下")
    history.append_tool_response("18C sunny")
    rendered = history.render_prefix()
    assert "<|im_start|>user\n<tool_response>\n18C sunny\n</tool_response><|im_end|>\n" in rendered
    # Empty tool output renders nothing (no bare <tool_response></tool_response>).
    history.append_tool_response("")
    assert rendered == history.render_prefix()
    drop_session_history("sess-hist-tool")
