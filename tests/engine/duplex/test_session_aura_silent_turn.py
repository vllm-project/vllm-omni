# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AURA terminal silence must leave the next turn free to speak."""

from __future__ import annotations

import pytest
from vllm.outputs import CompletionOutput

from tests.engine.duplex.test_session_manager import FakeClock
from tests.engine.duplex.test_session_runner import close_harness, open_harness, types
from tests.engine.duplex.test_session_runner_nemotron_voicechat import error_codes, output_item
from tests.engine.duplex.test_tool_call_ledger import FakeTimerScheduler
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.contracts import ToolCallAbandonReason, abandoned_tool_call_notice
from vllm_omni.engine.duplex.session import helpers
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.model_executor.models.aura_omni.duplex.history import (
    drop_session_history,
    get_or_create_session_history,
)
from vllm_omni.model_executor.models.aura_omni.duplex.plugin import AuraDuplexPlugin
from vllm_omni.model_executor.stage_input_processors.aura_omni import SILENT_TEXT
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_aura_silence_lets_the_next_turn_of_the_session_speak() -> None:
    plugin = AuraDuplexPlugin(lambda audio, sample_rate, fmt, speed: "ZmFrZQ==")
    h = await open_harness(plugin=plugin, model="aura", modalities=("text", "audio"), stage_count=4)
    try:
        request_ids: list[str] = []
        for text, finished in [(SILENT_TEXT, True), ("上海明天多云。", False)]:
            fence = helpers.append_fence(h.session, None)
            request_id = DuplexSessionManager.stage_request_id(fence, stage_id=1, resumable=False)
            request_ids.append(request_id)
            plugin.data_plane.begin_request(request_id)
            h.session.bind_request(request_id)
            output = OmniRequestOutput(
                request_id=request_id,
                finished=finished,
                stage_id=1,
                outputs=[CompletionOutput(index=0, text=text, token_ids=[], cumulative_logprob=None, logprobs=None)],
            )
            for event in plugin.data_plane.project({"data_plane_outputs": [output]}):
                await h.runner.model._send_one_model_output_event(event)
            events = await h.settle()
            if finished:
                assert types(events) == ["response.listen"]
            else:
                assert "response.listen" not in types(events) and h.session.active_response_id is not None
        assert request_ids[0] != request_ids[1]
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_box_plus_tool_completes_the_response_and_admits_the_followup() -> None:
    plugin = AuraDuplexPlugin(lambda audio, sample_rate, fmt, speed: "ZmFrZQ==")
    h = await open_harness(plugin=plugin, model="aura", modalities=("text", "audio"), stage_count=4)
    try:
        fence = helpers.append_fence(h.session, None)
        request_id = DuplexSessionManager.stage_request_id(fence, stage_id=1, resumable=False)
        plugin.data_plane.begin_request(request_id)
        h.session.bind_request(request_id)
        box = "<|box_start|>(712.0,531.0),(888.0,662.0)<|box_end|>"
        text = f"<|respond|>{box}<tool_call>\n<function=get_city_weather>\n</function>\n</tool_call>"
        output = OmniRequestOutput(
            request_id=request_id,
            finished=True,
            stage_id=1,
            outputs=[CompletionOutput(index=0, text=text, token_ids=[], cumulative_logprob=None, logprobs=None)],
        )
        for event in plugin.data_plane.project({"data_plane_outputs": [output]}):
            await h.runner.model._send_one_model_output_event(event)
        events = await h.settle()
        assert any(getattr(event, "text", "") == box for event in events)
        done = [event for event in events if event.type == "response.function_call_arguments.done"]
        assert len(done) == 1
        assert any(event.type == "response.done" and event.response_id == done[0].response_id for event in events)
        before = len(h.port.submissions)
        assert error_codes(await h.run(commands.CreateItem(item=output_item(done[0].call_id, "18C")))) == []
        follow = await h.run(commands.CreateResponse())
        assert error_codes(follow) == []
        assert len(h.port.submissions) == before + 1
        assert h.port.submissions[-1].prompt["additional_information"]["aura_tool_followup"] is True
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_early_response_create_keeps_the_rest_of_the_tool_batch() -> None:
    """A result for call A must not drop call B that has not arrived yet."""
    plugin = AuraDuplexPlugin(lambda audio, sample_rate, fmt, speed: "ZmFrZQ==")
    h = await open_harness(plugin=plugin, model="aura", modalities=("text", "audio"), stage_count=4)
    try:
        fence = helpers.append_fence(h.session, None)
        request_id = DuplexSessionManager.stage_request_id(fence, stage_id=1, resumable=False)
        plugin.data_plane.begin_request(request_id)
        h.session.bind_request(request_id)
        text = (
            "<|respond|>"
            "<tool_call>\n<function=get_current_datetime>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=get_current_location>\n</function>\n</tool_call>"
        )
        output = OmniRequestOutput(
            request_id=request_id,
            finished=True,
            stage_id=1,
            outputs=[CompletionOutput(index=0, text=text, token_ids=[], cumulative_logprob=None, logprobs=None)],
        )
        for event in plugin.data_plane.project({"data_plane_outputs": [output]}):
            await h.runner.model._send_one_model_output_event(event)
        events = await h.settle()
        done = [event for event in events if event.type == "response.function_call_arguments.done"]
        assert len(done) == 2
        assert len({event.call_id for event in done}) == 2
        assert h.session.runtime_config["aura_tool_batch"] == [event.call_id for event in done]

        before = len(h.port.submissions)
        assert error_codes(await h.run(commands.CreateItem(item=output_item(done[0].call_id, "10:00")))) == []
        early = await h.run(commands.CreateResponse())
        assert error_codes(early) == ["response_create_without_input"]
        assert len(h.port.submissions) == before
        assert h.session.runtime_config["aura_tool_batch"] == [event.call_id for event in done]
        assert [entry["call_id"] for entry in h.session.runtime_config["aura_tool_pending"]] == [done[0].call_id]

        assert error_codes(await h.run(commands.CreateItem(item=output_item(done[1].call_id, "台北")))) == []
        follow = await h.run(commands.CreateResponse())
        assert error_codes(follow) == []
        assert len(h.port.submissions) == before + 1
        assert h.port.submissions[-1].prompt["additional_information"]["aura_tool_followup"] is True
        history = get_or_create_session_history(h.session.session_id)
        assert [message["content"] for message in history.messages if message["role"] == "tool_response"] == [
            "10:00",
            "台北",
        ]
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_timed_out_tool_call_settles_so_the_accepted_result_can_follow_up() -> None:
    """A expires in the ledger; B's result must still be able to start one follow-up."""
    plugin = AuraDuplexPlugin(lambda audio, sample_rate, fmt, speed: "ZmFrZQ==")
    clock = FakeClock(100.0)
    scheduler = FakeTimerScheduler()
    h = await open_harness(
        plugin=plugin,
        model="aura",
        modalities=("text", "audio"),
        stage_count=4,
        clock=clock,
        extra_body={"tool_result_timeout_s": 5.0},
    )
    h.runner.tool_calls.set_scheduler(scheduler)
    drop_session_history(h.session.session_id)
    try:
        fence = helpers.append_fence(h.session, None)
        request_id = DuplexSessionManager.stage_request_id(fence, stage_id=1, resumable=False)
        plugin.data_plane.begin_request(request_id)
        h.session.bind_request(request_id)
        text = (
            "<|respond|>"
            "<tool_call>\n<function=get_current_datetime>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=get_current_location>\n</function>\n</tool_call>"
        )
        output = OmniRequestOutput(
            request_id=request_id,
            finished=True,
            stage_id=1,
            outputs=[CompletionOutput(index=0, text=text, token_ids=[], cumulative_logprob=None, logprobs=None)],
        )
        for event in plugin.data_plane.project({"data_plane_outputs": [output]}):
            await h.runner.model._send_one_model_output_event(event)
        events = await h.settle()
        done = [event for event in events if event.type == "response.function_call_arguments.done"]
        assert len(done) == 2
        accepted, expired = done

        before = len(h.port.submissions)
        assert error_codes(await h.run(commands.CreateItem(item=output_item(accepted.call_id, "10:00")))) == []
        clock.advance(5.0)
        scheduler.fire_all()
        timed_out = await h.settle()
        assert error_codes(timed_out) == ["tool_result_timeout"]
        late = await h.run(commands.CreateItem(item=output_item(expired.call_id, "台北")))
        assert error_codes(late) == ["invalid_function_call_output"]

        follow = await h.run(commands.CreateResponse())
        assert error_codes(follow) == []
        assert len(h.port.submissions) == before + 1
        history = get_or_create_session_history(h.session.session_id)
        assert [message["content"] for message in history.messages if message["role"] == "tool_response"] == [
            "10:00",
            abandoned_tool_call_notice(ToolCallAbandonReason.TIMEOUT),
        ]
    finally:
        drop_session_history(h.session.session_id)
        await close_harness(h)
