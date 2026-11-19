# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AURA terminal silence must leave the next turn free to speak."""

from __future__ import annotations

import pytest
from vllm.outputs import CompletionOutput

from tests.engine.duplex.test_session_runner import close_harness, open_harness, types
from tests.engine.duplex.test_session_runner_nemotron_voicechat import error_codes, output_item
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.session import helpers
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
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
