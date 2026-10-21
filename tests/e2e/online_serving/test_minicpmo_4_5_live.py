# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU coverage for MiniCPM-o 4.5 on the OpenAI Live endpoint (``/v1/live/sessions``).

One server for the module: ``minicpmo_4_5_live.yaml`` with Stage 0 at
``max_model_len: 1024`` (the first context trim comes about a minute into a
session, so the short scenarios stay below it) and ``max_sessions: 2``.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.e2e.online_serving.helpers.live_sessions import (
    LiveSession,
    Utterance,
    context_trims,
    live_url,
    load_speech,
    metrics_url,
    start_over_limit,
    watch_context_trims,
)
from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import MODEL
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, write_deploy_overlay

pytestmark = pytest.mark.omni

VOICE = "default"
MAX_SESSIONS = 2
LIVE_DEPLOY = write_deploy_overlay(
    get_deploy_config_path("minicpmo_4_5_live.yaml"),
    {
        "live_session_config": {"session_lifetime_s": 1800, "max_sessions": MAX_SESSIONS},
        # The base limits (64 images and audios, a 128-frame video) cannot fit
        # 1024 tokens; Live prompts carry one audio item, the reference voice.
        "stages": [{"stage_id": 0, "max_model_len": 1024, "limit_mm_per_prompt": {"image": 0, "audio": 1, "video": 0}}],
    },
)
SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=LIVE_DEPLOY,
            use_stage_cli=False,
            server_args=["--trust-remote-code"],
        ),
        id="live-1024",
    )
]


def _assert_healthy(session: LiveSession) -> None:
    assert not session.errors(), session.errors()
    assert not session.closed_by_server, f"{session.name}: closed by the server ({session.close_code})"


def _assert_reply(session: LiveSession, utterance: Utterance) -> None:
    assert session.answered(utterance), f"{session.name}: no spoken reply to {utterance}"
    assert session.transcript(utterance.start, utterance.window_end).strip(), (
        f"{session.name}: no transcript for {utterance}"
    )


def _assert_answered(session: LiveSession) -> None:
    _assert_healthy(session)
    for utterance in session.utterances:
        _assert_reply(session, utterance)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_minicpmo_live_conversation(omni_server) -> None:
    """Three turns in one session: a spoken, transcribed reply to each, then an orderly close."""
    speech = load_speech()

    async def run() -> LiveSession:
        async with LiveSession(live_url(omni_server), MODEL, VOICE) as session:
            await session.silence(1.0)
            for _ in range(3):
                await session.say(speech, gap_s=6.0)
            await session.close()
        return session

    session = asyncio.run(run())

    _assert_answered(session)
    assert session.closed_reason() == "close_requested"
    assert session.close_code == 1000
    assert any(event["type"] == "session.usage.updated" for event in session.events)


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_minicpmo_live_context_trim(omni_server) -> None:
    """A session past the 1024-token context is trimmed, resubmitted and keeps answering."""
    speech = load_speech()
    url = metrics_url(omni_server)
    before = context_trims(url)

    async def run() -> tuple[LiveSession, list[tuple[float, float]]]:
        async with LiveSession(live_url(omni_server), MODEL, VOICE) as session:
            stop = asyncio.Event()
            watcher = asyncio.create_task(watch_context_trims(url, session, stop))
            await session.silence(1.0)
            for _ in range(8):
                await session.say(speech, gap_s=7.0)
            stop.set()
            changes = await watcher
            await session.close()
        return session, changes

    session, changes = asyncio.run(run())

    trimmed = [t for t, value in changes if value > before]
    assert trimmed, f"no context trim in {session.now():.0f} s at max_model_len 1024: {changes}"
    _assert_healthy(session)
    after_trim = [u for u in session.utterances if u.start > trimmed[0]]
    assert after_trim, f"no utterance after the first trim at {trimmed[0]:.0f} s"
    assert any(session.answered(u) for u in after_trim), f"no spoken reply after the trim at {trimmed[0]:.0f} s"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_minicpmo_live_concurrent_sessions(omni_server) -> None:
    """Two sessions at once; a third start is refused; a closed session's slot is reused."""
    speech = load_speech()
    url = live_url(omni_server)

    async def converse(session: LiveSession, lead_s: float) -> None:
        await session.silence(lead_s)
        for _ in range(2):
            await session.say(speech, gap_s=6.0)

    async def run() -> tuple[list[LiveSession], tuple[list[dict], int | None]]:
        async with (
            LiveSession(url, MODEL, VOICE, name="a") as a,
            LiveSession(url, MODEL, VOICE, name="b") as b,
        ):
            await asyncio.gather(converse(a, 1.0), converse(b, 2.5))
            refused = await start_over_limit(url, MODEL, VOICE, "start-over-limit")
            await a.close()
            async with LiveSession(url, MODEL, VOICE, name="c") as c:
                await converse(c, 1.0)
                await c.close()
            await b.close()
        return [a, b, c], refused

    sessions, (events, close_code) = asyncio.run(run())

    errors = [event for event in events if event.get("type") == "error"]
    assert errors, f"over-limit session.start got no error: {events}"
    assert errors[0]["error"]["type"] == "server_error"
    assert errors[0]["error"]["code"] == "session_limit_reached"
    assert errors[0].get("client_event_id") == "start-over-limit"
    assert not any(event.get("type") in ("session.started", "session.closed") for event in events), events
    assert close_code == 1013
    for session in sessions:
        _assert_answered(session)
