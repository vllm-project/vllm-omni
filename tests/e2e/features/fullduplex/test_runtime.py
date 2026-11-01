# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import AsyncIterator

import pytest

from vllm_omni.experimental.fullduplex.core import protocol as ev
from vllm_omni.experimental.fullduplex.core.adapter import DuplexCapability, OutputChunk
from vllm_omni.experimental.fullduplex.core.runtime import DuplexRuntime
from vllm_omni.experimental.fullduplex.core.session import DuplexSession, DuplexSessionConfig
from vllm_omni.experimental.fullduplex.joyvl.adapter import JoyVLDuplexAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


async def _feed(events):
    for e in events:
        yield e


def _collector():
    out: list[dict] = []

    async def emit(event: dict) -> None:
        out.append(event)

    return out, emit


class _FakeAdapter:
    def __init__(self, chunks, barge_after=None):
        self._chunks = chunks
        self._barge_after = barge_after

    def capabilities(self):
        return DuplexCapability(frozenset({"text"}), frozenset({"text"}), proactive=False)

    async def on_input(self, session, modality, data):
        pass

    def should_respond(self, session):
        return True

    async def respond(self, session) -> AsyncIterator[OutputChunk]:
        for i, c in enumerate(self._chunks):
            if self._barge_after is not None and i == self._barge_after:
                session.barge_in()
            yield OutputChunk("text", c)

    async def on_barge_in(self, session):
        pass

    async def on_playback_ack(self, session, cursor):
        pass


@pytest.mark.asyncio
async def test_runtime_basic_response():
    session = DuplexSession("s", DuplexSessionConfig(output_modalities=("text",)))
    rt = DuplexRuntime(session, _FakeAdapter(["a", "b"]))
    out, emit = _collector()
    await rt.run(_feed([{"type": ev.INPUT_COMMIT}, {"type": ev.CLOSE}]), emit)
    types = [e["type"] for e in out]
    assert types == [ev.RESPONSE_CREATED, ev.RESPONSE_DELTA, ev.RESPONSE_DELTA, ev.RESPONSE_DONE]
    assert [e["data"] for e in out if e["type"] == ev.RESPONSE_DELTA] == ["a", "b"]


@pytest.mark.asyncio
async def test_runtime_barge_in_drops_stale_output():
    session = DuplexSession("s")
    rt = DuplexRuntime(session, _FakeAdapter(["a", "b", "c"], barge_after=1))
    out, emit = _collector()
    await rt.run(_feed([{"type": ev.INPUT_COMMIT}, {"type": ev.CLOSE}]), emit)
    data = [e["data"] for e in out if e["type"] == ev.RESPONSE_DELTA]
    assert data == ["a"]
    assert ev.RESPONSE_DONE not in [e["type"] for e in out]


class _SlowAdapter:
    def capabilities(self):
        return DuplexCapability(frozenset({"text"}), frozenset({"text"}), proactive=False)

    async def on_input(self, session, modality, data):
        pass

    def should_respond(self, session):
        return True

    async def respond(self, session) -> AsyncIterator[OutputChunk]:
        for c in ["a", "b", "c", "d", "e"]:
            await asyncio.sleep(0.02)
            yield OutputChunk("text", c)

    async def on_barge_in(self, session):
        pass

    async def on_playback_ack(self, session, cursor):
        pass


@pytest.mark.asyncio
async def test_runtime_cancel_event_interrupts_active_response():
    session = DuplexSession("s")
    rt = DuplexRuntime(session, _SlowAdapter())
    out, emit = _collector()

    async def feed():
        yield {"type": ev.INPUT_COMMIT}
        await asyncio.sleep(0.03)
        yield {"type": ev.RESPONSE_CANCEL}
        yield {"type": ev.CLOSE}

    await rt.run(feed(), emit)
    deltas = [e["data"] for e in out if e["type"] == ev.RESPONSE_DELTA]
    types = [e["type"] for e in out]
    assert len(deltas) < 5
    assert ev.RESPONSE_CANCELLED in types
    assert ev.RESPONSE_DONE not in types


@pytest.mark.asyncio
async def test_runtime_new_response_supersedes_inflight_without_blocking():
    # A second response trigger arriving while the first is still streaming must
    # supersede it (barge-in) rather than block the input loop until it finishes.
    session = DuplexSession("s")
    rt = DuplexRuntime(session, _SlowAdapter())
    out, emit = _collector()

    async def feed():
        yield {"type": ev.INPUT_COMMIT}
        await asyncio.sleep(0.03)  # let the first response emit ~one chunk
        yield {"type": ev.INPUT_COMMIT}  # supersedes the in-flight first response
        await asyncio.sleep(0.2)  # let the second response finish
        yield {"type": ev.CLOSE}

    await rt.run(feed(), emit)
    created = [e for e in out if e["type"] == ev.RESPONSE_CREATED]
    done = [e for e in out if e["type"] == ev.RESPONSE_DONE]
    # two responses created; only the latest one completes (the first was cancelled)
    assert len(created) == 2
    assert len(done) == 1
    assert done[0]["response_index"] == created[-1]["response_index"]


@pytest.mark.asyncio
async def test_joyvl_adapter_speaks_then_silences():
    replies = iter(["</response> a fire is breaking out", "</silence>"])

    async def fake_generate(messages):
        return next(replies)

    adapter = JoyVLDuplexAdapter(fake_generate, num_frames=4)
    cfg = DuplexSessionConfig(input_modalities=("video", "text"), output_modalities=("text",), proactive=True)
    rt = DuplexRuntime(DuplexSession("s", cfg), adapter)
    out, emit = _collector()
    await rt.run(
        _feed(
            [
                {"type": ev.INPUT_APPEND, "modality": "text", "data": "alert me if a fire breaks out"},
                {"type": ev.INPUT_APPEND, "modality": "video", "data": "data:image/jpeg;base64,AAA"},
                {"type": ev.INPUT_APPEND, "modality": "video", "data": "data:image/jpeg;base64,BBB"},
                {"type": ev.CLOSE},
            ]
        ),
        emit,
    )
    deltas = [e["data"] for e in out if e["type"] == ev.RESPONSE_DELTA]
    assert "a fire is breaking out" in deltas

    assert deltas.count("a fire is breaking out") == 1
