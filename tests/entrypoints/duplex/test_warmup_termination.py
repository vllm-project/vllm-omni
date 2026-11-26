# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The warmup must end on the right outcome and never confuse them."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

import pytest

from vllm_omni.entrypoints.duplex import warmup

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _recv(events: Sequence[Mapping[str, object]]):
    """A recv callable that returns queued events, then blocks until timeout."""
    pending: list[Mapping[str, object]] = list(events)

    async def recv() -> dict[str, object]:
        if pending:
            return dict(pending.pop(0))
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    return recv


@pytest.mark.asyncio
async def test_silent_probe_listen_ends_the_wait():
    """A listen-first model answering the silent unit closes the wait cleanly."""
    outcome = await warmup.wait_for_warmup_outcome(_recv([{"type": "response.listen"}]), 5.0, kind="silent_frames")
    assert outcome == warmup.WARMUP_OUTCOME_LISTEN


@pytest.mark.asyncio
async def test_video_probe_ignores_listen_and_waits_for_audio():
    """The video probe needs speech, so a listen there is not success."""
    events = [{"type": "response.listen"}, {"type": "response.output_audio.delta"}]
    outcome = await warmup.wait_for_warmup_outcome(_recv(events), 5.0, kind="video_turn")
    assert outcome == warmup.WARMUP_OUTCOME_AUDIO


@pytest.mark.asyncio
async def test_audio_ends_the_wait_for_every_probe():
    outcome = await warmup.wait_for_warmup_outcome(
        _recv([{"type": "response.output_audio.delta"}]), 5.0, kind="silent_frames"
    )
    assert outcome == warmup.WARMUP_OUTCOME_AUDIO


@pytest.mark.asyncio
async def test_timeout_is_never_reported_as_listen():
    """Unrelated events must not be read as a listen when the budget expires."""
    outcome = await warmup.wait_for_warmup_outcome(
        _recv([{"type": "response.created"}, {"type": "session.created"}]),
        0.2,
        kind="silent_frames",
    )
    assert outcome == warmup.WARMUP_OUTCOME_TIMEOUT


def test_outcomes_are_distinct():
    assert (
        len(
            {
                warmup.WARMUP_OUTCOME_AUDIO,
                warmup.WARMUP_OUTCOME_TEXT,
                warmup.WARMUP_OUTCOME_LISTEN,
                warmup.WARMUP_OUTCOME_TIMEOUT,
            }
        )
        == 4
    )


@pytest.mark.asyncio
async def test_text_probe_completed_response_ends_the_wait():
    response = {
        "status": "completed",
        "output": [{"content": [{"type": "output_text", "text": "ready"}]}],
    }
    outcome = await warmup.wait_for_warmup_outcome(
        _recv([{"type": "response.done", "response": response}]),
        5.0,
        kind="silent_frames",
        text_only=True,
    )
    assert outcome == warmup.WARMUP_OUTCOME_TEXT


_COMPLETED_TEXT = {"status": "completed", "output": [{"content": [{"type": "output_text", "text": "ready"}]}]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"status": "completed", "output": []},
        {"status": "completed", "output": [{"content": [{"type": "output_text", "text": ""}]}]},
        {"status": "incomplete", "output": [{"content": [{"type": "output_text", "text": "ready"}]}]},
    ],
)
async def test_text_probe_ignores_responses_without_completed_text(response):
    events = [{"type": "response.done", "response": response}]
    outcome = await warmup.wait_for_warmup_outcome(_recv(events), 0.2, kind="silent_frames", text_only=True)
    assert outcome == warmup.WARMUP_OUTCOME_TIMEOUT


@pytest.mark.asyncio
async def test_text_probe_waits_past_a_listen_for_the_text_reply():
    """A listen unit can precede the seeded reply; only completed text ends the wait."""
    events: list[Mapping[str, object]] = [
        {"type": "response.listen"},
        {"type": "response.done", "response": _COMPLETED_TEXT},
    ]
    outcome = await warmup.wait_for_warmup_outcome(_recv(events), 5.0, kind="silent_frames", text_only=True)
    assert outcome == warmup.WARMUP_OUTCOME_TEXT


@pytest.mark.asyncio
async def test_audio_probe_ignores_completed_text_response():
    events: list[Mapping[str, object]] = [
        {
            "type": "response.done",
            "response": {
                "status": "completed",
                "output": [{"content": [{"type": "output_text", "text": "ready"}]}],
            },
        },
        {"type": "response.output_audio.delta"},
    ]
    outcome = await warmup.wait_for_warmup_outcome(_recv(events), 5.0, kind="silent_frames", text_only=False)
    assert outcome == warmup.WARMUP_OUTCOME_AUDIO
