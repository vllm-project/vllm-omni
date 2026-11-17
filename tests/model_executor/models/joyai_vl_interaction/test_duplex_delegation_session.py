# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""JoyAI delegation through the real session: publication and follow-up delivery.

The model-side rules live in ``test_duplex_delegation.py``. This module drives
them through the real JoyAI plugin, session manager and mailbox: a delegated
observation publishes one shared function call, an accepted
``function_call_output`` reaches the next JoyAI prompt, and the session lifecycle
the shared ledger owns (duplicate, unknown, cancelled) reaches the client
unchanged. Admission itself is covered by the engine tests; these cases prove
the JoyAI plugin is wired to it.
"""

import json

import pytest

from tests.engine.duplex.test_session_runner import tts_output
from tests.model_executor.models.joyai_vl_interaction.test_duplex_plugin import observe, open_joyai
from vllm_omni.engine.duplex.commands import AckPlayback, CancelResponse, CreateItem, UpdateSession
from vllm_omni.engine.duplex.session.tool_calls import ToolCallState
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.delegation import (
    DELEGATION_TOOL_NAME,
    delegation_tool_definition,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DELEGATED = "</response> Checking the pot. </delegation> Is the pot overheated?"
#: A delegation without a spoken note: the model hands the question over silently.
DELEGATED_NO_NOTE = "</delegation> Is the pot overheated?"
DIGEST = "The pot has been simmering for 20 minutes."


def realtime_events(h, events=None) -> list[dict[str, object]]:
    """Wire view of ``events`` (default: everything ``h.run`` collected).

    ``deliver_and_settle`` returns the events it produced without appending
    them to ``h.events``, so a caller that delivered a stage output must pass
    them here.
    """
    source = h.events if events is None else events
    return [event.to_realtime() for event in source]


def function_calls(h, events=None) -> list[dict[str, object]]:
    return [
        event for event in realtime_events(h, events) if event.get("type") == "response.function_call_arguments.done"
    ]


def error_codes(events) -> list[str]:
    return [event.code for event in events if getattr(event, "code", None)]


def result_item(call_id: str, output: str) -> CreateItem:
    return CreateItem(
        item={
            "id": f"result-{call_id}",
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }
    )


async def delegating_session(*, modalities=("text", "audio"), asr: bool = False):
    """A JoyAI session that declares the delegate tool."""
    h = await open_joyai(modalities=modalities, asr=asr)
    await h.run(UpdateSession(patch={"tools": [delegation_tool_definition()]}))
    return h


async def publish_delegation(h) -> str:
    """Run one delegating observation and return the published call id.

    The observation's response owns its terminal and its playback: the note's
    audio is only released once the client acknowledges it, so a later camera
    commit is not deferred behind an unheard reply.
    """
    request_id = await observe(h)
    events = await h.deliver_and_settle(
        tts_output(request_id, text=DELEGATED, finished=True), stage_id=0, segment_finished=True
    )
    calls = function_calls(h, events)
    assert len(calls) == 1, realtime_events(h, events)
    if h.session.active_response_id is not None:
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(request_id, samples=24000, finished=True), stage_id=2, segment_finished=True
        )
        await h.run(AckPlayback(response_id=response_id, played_ms=1000))
    return calls[0]["call_id"]


@pytest.mark.asyncio
async def test_delegation_publishes_one_function_call_with_question_and_note():
    h = await delegating_session()
    try:
        request_id = await observe(h)
        events = await h.deliver_and_settle(
            tts_output(request_id, text=DELEGATED, finished=True), stage_id=0, segment_finished=True
        )
        [call] = function_calls(h, events)
        # ``response.function_call_arguments.done`` carries the identity and the
        # arguments; the function name lives on the projected output item.
        item = next(
            event["item"]
            for event in realtime_events(h, events)
            if event.get("type") == "response.output_item.added"
            and event.get("item", {}).get("type") == "function_call"
        )
        assert item["name"] == DELEGATION_TOOL_NAME
        assert item["call_id"] == call["call_id"]
        arguments = json.loads(call["arguments"])
        assert arguments == {"note": "Checking the pot.", "question": "Is the pot overheated?"}
        # The session ledger owns the call, so a result can be admitted for it.
        assert h.runner.tool_calls.record(call["call_id"]) is not None
        # The note is spoken: it is the response's transcript, and the delegated
        # question never is. Stage hand-off itself belongs to the orchestrator,
        # which this CPU harness replaces with a recording port.
        transcript = "".join(
            event["delta"]
            for event in realtime_events(h, events)
            if event.get("type") == "response.output_audio_transcript.delta"
        )
        assert transcript == "Checking the pot."
        assert "Is the pot overheated?" not in transcript
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_accepted_result_reaches_the_next_prompt():
    h = await delegating_session()
    try:
        call_id = await publish_delegation(h)
        accepted = await h.run(result_item(call_id, DIGEST))
        assert error_codes(accepted) == []
        # The next camera observation carries the accepted result.
        await observe(h)
        prompt = h.port.submissions[-1].prompt["prompt"]
        assert DIGEST in prompt
        assert "Is the pot overheated?" in prompt
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_duplicate_result_is_refused_and_does_not_reach_the_prompt():
    h = await delegating_session()
    try:
        call_id = await publish_delegation(h)
        first = await h.run(result_item(call_id, DIGEST))
        assert error_codes(first) == []
        duplicate = await h.run(result_item(call_id, "second answer"))
        assert error_codes(duplicate) == ["invalid_function_call_output"]
        await observe(h)
        prompt = h.port.submissions[-1].prompt["prompt"]
        assert DIGEST in prompt
        assert "second answer" not in prompt
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_unknown_call_id_is_refused():
    h = await delegating_session()
    try:
        rejected = await h.run(result_item("call_never_published", DIGEST))
        assert error_codes(rejected) == ["invalid_function_call_output"]
        await observe(h)
        assert DIGEST not in h.port.submissions[-1].prompt["prompt"]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_cancelled_call_refuses_its_late_result():
    """A cancelled epoch settles the pending call, and its late result is refused.

    In this CPU harness a later commit does not advance the epoch, so the
    cancellation is driven the way the runner does it: the epoch advances and
    the ledger invalidates the pending calls registered in the old one.
    """
    h = await delegating_session()
    try:
        request_id = await observe(h)
        events = await h.deliver_and_settle(
            tts_output(request_id, text=DELEGATED, finished=True), stage_id=0, segment_finished=True
        )
        [call] = function_calls(h, events)
        # The runner's epoch advance is the single cancellation authority.
        h.session.barge_in()
        h.runner.tool_calls.invalidate_epoch(h.session.epoch)
        assert h.runner.tool_calls.record(call["call_id"]).state is ToolCallState.CANCELLED
        await h.settle()
        late = await h.run(result_item(call["call_id"], DIGEST))
        assert error_codes(late) == ["invalid_function_call_output"]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_delegation_without_declared_tool_fails_fast():
    h = await open_joyai()
    try:
        request_id = await observe(h)
        events = await h.deliver_and_settle(
            tts_output(request_id, text=DELEGATED, finished=True), stage_id=0, segment_finished=True
        )
        assert function_calls(h, events) == []
        failures = [event for event in events if event.type == "response.done"]
        assert failures and failures[0].to_realtime()["response"]["status"] == "failed"
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_note_free_delegation_releases_its_observation():
    """A silent delegation still ends its observation, so the next one can start.

    The call is published, but the shared call handler returns before the response
    lifecycle runs. Without the empty terminal the observation would keep its
    active request and the next camera commit could not open a response.
    """
    h = await delegating_session()
    try:
        request_id = await observe(h)
        events = await h.deliver_and_settle(
            tts_output(request_id, text=DELEGATED_NO_NOTE, finished=True), stage_id=0, segment_finished=True
        )
        # Exactly one call, and it carries the question with no note.
        [call] = function_calls(h, events)
        assert json.loads(call["arguments"]) == {"question": "Is the pot overheated?"}
        # The observation ended, so no response is left owning the request.
        assert h.session.active_response_id is None, realtime_events(h, events)
        # Nothing was spoken, so nothing is stored as an assistant turn.
        assert not [message for message in h.session.history if message.get("role") == "assistant"]
        # The session is free to take the next observation.
        accepted = await h.run(result_item(call["call_id"], DIGEST))
        assert error_codes(accepted) == []
        await observe(h)
        assert h.port.submissions, "the follow-up observation must reach the stage"
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_cancelled_note_audio_is_not_stored_as_spoken_history():
    """A note whose audio never completed must not enter history as spoken.

    The note carries ``text_requires_complete_audio``, so cancelling between the
    projection and the first Code2Wav output leaves its text uncommitted. An
    empty assistant turn is still recorded as the audio reply's boundary; what
    must not appear is the note's text, which a later prompt would otherwise
    treat as something the user already heard.
    """
    h = await delegating_session()
    try:
        request_id = await observe(h)
        events = await h.deliver_and_settle(
            tts_output(request_id, text=DELEGATED, finished=True), stage_id=0, segment_finished=True
        )
        [call] = function_calls(h, events)
        assert call["call_id"]
        # Cancel before any Code2Wav audio reaches the session.
        cancelled = await h.run(CancelResponse())
        assert not [event for event in cancelled if getattr(event, "code", None)], cancelled
        spoken = [message.get("content") for message in h.session.history if message.get("role") == "assistant"]
        assert "Checking the pot." not in spoken, h.session.history
        assert all(not content for content in spoken), h.session.history
    finally:
        await h.manager.shutdown()
