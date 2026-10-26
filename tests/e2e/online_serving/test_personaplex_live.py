# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU coverage for PersonaPlex on the OpenAI Live endpoint (``/v1/live/sessions``).

Two servers, both ``personaplex_live.yaml`` (``max_sessions: 2``, which also
sizes the Stage 0 Mimi encoder rows and the Code2Wav decoders) with Stage 0 at
``max_model_len: 600`` frames, so a session overflows after about 41 s of audio
(the voice and persona prefill takes the first 89 frames). The second also caps
the Stage 0 KV cache at one full context, so two growing sessions run it out
and stall. PersonaPlex streams audio for every 80 ms frame, so replies are
checked for voiced output, not for audio events. The model is
``PERSONAPLEX_MODEL_PATH`` when set, else the gated ``nvidia/personaplex-7b-v1``
from Hugging Face.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from tests.e2e.online_serving.helpers.live_sessions import (
    LiveSession,
    assert_ended_by_server,
    context_trims,
    converse_until_server_ends_one,
    live_errors,
    live_url,
    load_speech,
    metrics_url,
    start_over_limit,
    watch_context_trims,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, write_deploy_overlay

pytestmark = pytest.mark.omni

MODEL = os.environ.get("PERSONAPLEX_MODEL_PATH") or "nvidia/personaplex-7b-v1"
VOICE = "NATF2"
STARVE_WINDOW_S = 120.0
LIVE_DEPLOY = write_deploy_overlay(
    get_deploy_config_path("personaplex_live.yaml"),
    {"stages": [{"stage_id": 0, "max_model_len": 600}]},
)
# 40 blocks of 16 frames: one full 600-frame context (vLLM keeps two blocks
# back here, so 38 would admit only 576 frames). Two growing sessions
# run it out and stall (waiting Live requests keep their blocks and are not
# preempted) until the backlog limit ends them.
STARVE_DEPLOY = write_deploy_overlay(
    get_deploy_config_path("personaplex_live.yaml"),
    {
        "live_session_config": {"session_lifetime_s": 1800, "max_sessions": 2, "max_input_backlog_s": 10},
        "stages": [{"stage_id": 0, "max_model_len": 600, "block_size": 16, "num_gpu_blocks_override": 40}],
    },
)


def _server(deploy: str, server_id: str):
    return pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=deploy,
            use_stage_cli=False,
            server_args=["--trust-remote-code"],
        ),
        id=server_id,
    )


SERVER_PARAMS = [_server(LIVE_DEPLOY, "live-600")]
STARVE_SERVER_PARAMS = [_server(STARVE_DEPLOY, "live-600-kv-640")]


def _assert_healthy(session: LiveSession) -> None:
    assert not session.errors(), session.errors()
    assert not session.closed_by_server, f"{session.name}: closed by the server ({session.close_code})"


def _assert_answered(session: LiveSession) -> None:
    _assert_healthy(session)
    for utterance in session.utterances:
        assert session.answered(utterance), f"{session.name}: no voiced reply to {utterance}"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_personaplex_live_conversation(omni_server) -> None:
    """Three turns in one session: a voiced reply to each, then an orderly close."""
    speech = load_speech()

    async def run() -> LiveSession:
        async with LiveSession(live_url(omni_server), MODEL, VOICE) as session:
            await session.silence(1.0)
            for _ in range(3):
                await session.say(speech, gap_s=5.0)
            await session.close()
        return session

    session = asyncio.run(run())

    _assert_answered(session)
    assert session.closed_reason() == "close_requested"
    assert session.close_code == 1000


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_personaplex_live_context_trim(omni_server) -> None:
    """A session past the 600-frame context is trimmed, resubmitted and keeps answering."""
    speech = load_speech()
    url = metrics_url(omni_server)
    before = context_trims(url)

    async def run() -> tuple[LiveSession, list[tuple[float, float]]]:
        async with LiveSession(live_url(omni_server), MODEL, VOICE) as session:
            stop = asyncio.Event()
            watcher = asyncio.create_task(watch_context_trims(url, session, stop))
            await session.silence(1.0)
            for _ in range(6):
                await session.say(speech, gap_s=7.0)
            stop.set()
            changes = await watcher
            await session.close()
        return session, changes

    session, changes = asyncio.run(run())

    trimmed = [t for t, value in changes if value > before]
    assert trimmed, f"no context trim in {session.now():.0f} s at max_model_len 600: {changes}"
    _assert_healthy(session)
    after_trim = [u for u in session.utterances if u.start > trimmed[0]]
    assert after_trim, f"no utterance after the first trim at {trimmed[0]:.0f} s"
    assert any(session.answered(u) for u in after_trim), f"no voiced reply after the trim at {trimmed[0]:.0f} s"


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_personaplex_live_concurrent_sessions(omni_server) -> None:
    """Two sessions at once; a third start is refused; a closed session's Mimi slots are reused."""
    speech = load_speech()
    url = live_url(omni_server)

    async def converse(session: LiveSession, lead_s: float) -> None:
        await session.silence(lead_s)
        for _ in range(2):
            await session.say(speech, gap_s=5.0)

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


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", STARVE_SERVER_PARAMS, indirect=True)
def test_personaplex_live_kv_exhaustion_ends_stalled_sessions(omni_server) -> None:
    """Sessions stalled on a full Stage 0 KV cache are ended by the backlog limit; the server recovers."""
    speech = load_speech()
    url = live_url(omni_server)
    metrics = metrics_url(omni_server)
    before = live_errors(metrics, "backlog")

    async def run() -> tuple[list[LiveSession], LiveSession]:
        sessions = await converse_until_server_ends_one(url, MODEL, VOICE, speech, gap_s=5.0, window_s=STARVE_WINDOW_S)
        async with LiveSession(url, MODEL, VOICE, name="after") as after:
            await after.silence(1.0)
            await after.say(speech, gap_s=5.0)
            await after.close()
        return sessions, after

    sessions, after = asyncio.run(run())

    assert_ended_by_server(sessions, STARVE_WINDOW_S)
    assert live_errors(metrics, "backlog") > before, "the backlog limit did not end the stalled sessions"
    _assert_answered(after)
