# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A failed duplex live exercise must settle its microphone and event reader."""

import asyncio
from pathlib import Path

import pytest

from tests.helpers.runtime import run_duplex_client_session, run_duplex_seeded_text_to_audio
from vllm_omni.clients import duplex

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["input", "output", "input_tail"])
async def test_live_session_cleans_up_after_input_or_output_failure(monkeypatch, failure):
    settled = set()
    blocked = asyncio.Event()
    input_chunks = []

    class Client:
        session_id = "session"
        resume_token = "token"

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            settled.add("client")

        async def stream_pcm(self, pcm, **kwargs):
            input_chunks.append(pcm)
            try:
                if failure == "input_tail":
                    if len(input_chunks) == 1:
                        return 0
                    raise RuntimeError("input_tail failed")
                if failure == "input":
                    raise RuntimeError("input failed")
                await blocked.wait()
            finally:
                settled.add("microphone")

        async def responses(self):
            if failure == "output":
                raise RuntimeError("output failed")
            await blocked.wait()
            yield None

    class Collector:
        async def consume(self, client):
            try:
                await blocked.wait()
            finally:
                settled.add("collector")

    monkeypatch.setattr(duplex, "DuplexClient", Client)
    monkeypatch.setattr(duplex, "EventCollector", Collector)
    monkeypatch.setattr(duplex, "audio_data_url", lambda path: "data:audio/wav;base64,AA==")
    monkeypatch.setattr(duplex, "read_pcm16_wav", lambda path: bytes(32000))
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        await asyncio.wait_for(
            run_duplex_client_session(url="ws://example", model="model", ref_audio=Path("ref"), input_wav=Path("mic")),
            timeout=1,
        )
    assert settled == {"client", "microphone", "collector"}
    if failure == "input_tail":
        assert input_chunks[1] == bytes(6400), "ongoing native input must be microphone silence"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["reader", "input", "cancel"])
async def test_seeded_session_failure_is_observed_and_settled(monkeypatch, failure):
    import json

    import websockets

    tasks_before = asyncio.all_tasks()
    closed = []
    sent = []
    blocked = asyncio.Event()

    class Socket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(True)

        async def send(self, data):
            event = json.loads(data)
            sent.append(event)
            if failure == "input" and event["type"] == "input_audio_buffer.append":
                raise RuntimeError("injected input failure")

        async def recv(self):
            if failure == "reader":
                return '{"type":"error","message":"injected failure"}'
            await blocked.wait()

    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: Socket())
    exception, message = {
        "reader": (AssertionError, "injected failure"),
        "input": (RuntimeError, "injected input failure"),
        "cancel": (TimeoutError, None),
    }[failure]
    with pytest.raises(exception, match=message):
        await asyncio.wait_for(
            run_duplex_seeded_text_to_audio(url="ws://example", model="model", ref_audio=None, text="question"),
            timeout=1,
        )
    assert closed == [True]
    assert sent[-1]["type"] == "session.close"
    assert asyncio.all_tasks() == tasks_before
