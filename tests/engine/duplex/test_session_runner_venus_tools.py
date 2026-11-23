# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Venus delegation through the real mailbox, ledger, and recording stage port."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from tests.engine.duplex.test_session_runner import append_audio, close_harness, open_harness, tts_output, types
from tests.engine.duplex.test_session_runner_nemotron_voicechat import answer, error_codes
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.session.tool_calls import ToolCallState
from vllm_omni.model_executor.models.minicpmo_4_5.handoff import SPEECH_MUTED_KEY
from vllm_omni.model_executor.models.realtime_venus.duplex import RealtimeVenusDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@asynccontextmanager
async def venus_session(**kwargs):
    plugin = RealtimeVenusDuplexPlugin(lambda *args: "")
    h = await open_harness(plugin=plugin, **kwargs)
    plugin._tokenizers[""] = SimpleNamespace(encode=lambda text, **kw: list(text.encode()))
    try:
        yield h
    finally:
        await close_harness(h)


async def delegate(h, question="12 + 30"):
    assert error_codes(await h.run(append_audio(samples=32000))) == []
    output = tts_output(
        h.stage0_request_id(),
        text=f"<delegate>{question}</delegate>",
        samples=0,
        tts_is_last_chunk=True,
        epoch=h.session.epoch,
        turn_id=h.session.turn_id,
    )
    output.multimodal_output[SPEECH_MUTED_KEY] = True
    events = await h.deliver_and_settle(output)
    calls = [event for event in events if event.type == "response.function_call_arguments.done"]
    assert len(calls) == 1, types(events)
    return calls[0].call_id


def submitted_results(h):
    return [
        submission.prompt["model_intermediate_buffer"]["duplex"]["runtime_config"].get(
            "duplex_append_text_token_ids", []
        )
        for submission in h.port.submissions
    ]


@pytest.mark.asyncio
async def test_result_reaches_one_append_and_duplicate_cannot_replay_it():
    async with venus_session() as h:
        call_id = await delegate(h)
        count = len(h.port.submissions)
        await h.run(append_audio())
        assert len(h.port.submissions) > count  # Input continues during the call.
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.PENDING
        assert error_codes(await answer(h, call_id, "42")) == []
        await h.run(append_audio())
        assert submitted_results(h).count(list(b"<backend>42</backend>")) == 1
        assert h.session.runtime_config["venus_function_results"] == []
        assert error_codes(await answer(h, call_id, "WRONG")) == ["invalid_function_call_output"]
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.ACCEPTED
        assert not any(ord("W") in ids for ids in submitted_results(h))


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [False, True])
async def test_interrupted_call_obeys_session_policy(keep):
    async with venus_session(extra_body={"function_call_interrupt_policy": "keep" if keep else "cancel"}) as h:
        call_id = await delegate(h)
        await h.run(commands.CancelInput())
        events = await answer(h, call_id, "42")
        assert error_codes(events) == ([] if keep else ["invalid_function_call_output"])
        await h.run(append_audio(samples=32000))
        assert submitted_results(h).count(list(b"<backend>42</backend>")) == int(keep)


@pytest.mark.asyncio
async def test_rejected_result_can_be_corrected():
    async with venus_session() as h:
        call_id = await delegate(h)
        assert error_codes(await answer(h, call_id, "x" * 4001)) == ["invalid_function_call_output"]
        assert h.runner.tool_calls.record(call_id).state is ToolCallState.PENDING
        assert error_codes(await answer(h, call_id, "42")) == []
        await h.run(append_audio())
        assert submitted_results(h).count(list(b"<backend>42</backend>")) == 1
