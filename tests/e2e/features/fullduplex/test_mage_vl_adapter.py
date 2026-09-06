# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest

from vllm_omni.experimental.fullduplex.core import protocol as ev
from vllm_omni.experimental.fullduplex.core.session import DuplexSession, DuplexSessionConfig
from vllm_omni.experimental.fullduplex.mage_vl import (
    MageVLCodecWindow,
    MageVLDuplexAdapter,
    MageVLDuplexRuntime,
    MageVLGateDecision,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


async def _feed(events):
    for event in events:
        yield event


def _collector():
    out: list[dict] = []

    async def emit(event: dict) -> None:
        out.append(event)

    return out, emit


def test_mage_vl_capabilities_match_backend_contract():
    capabilities = MageVLDuplexAdapter().capabilities()

    assert capabilities.input_modalities == frozenset({"text", "video"})
    assert capabilities.output_modalities == frozenset({"text"})
    assert capabilities.proactive


@pytest.mark.asyncio
async def test_mage_vl_proactive_gate_speaks_once_for_event():
    async def gate(session, windows):
        del session
        if windows[-1].segment_id == "goal":
            return MageVLGateDecision(True, text="Goal scored.", event_id="goal-1", score=0.91)
        return MageVLGateDecision(False)

    adapter = MageVLDuplexAdapter(gate=gate, window_size=1)
    session = DuplexSession(
        "mage",
        DuplexSessionConfig(
            input_modalities=("video", "text"),
            output_modalities=("text",),
            proactive=True,
        ),
    )
    rt = MageVLDuplexRuntime(session, adapter)
    out, emit = _collector()

    await rt.run(
        _feed(
            [
                {"type": ev.INPUT_APPEND, "modality": "video", "data": {"segment_id": "warmup", "frames": ["f0"]}},
                {"type": ev.INPUT_APPEND, "modality": "video", "data": {"segment_id": "goal", "frames": ["f1"]}},
                {"type": ev.INPUT_APPEND, "modality": "video", "data": {"segment_id": "goal", "frames": ["f2"]}},
                {"type": ev.CLOSE},
            ]
        ),
        emit,
    )

    deltas = [event["data"] for event in out if event["type"] == ev.RESPONSE_DELTA]
    assert deltas == ["Goal scored."]


@pytest.mark.asyncio
async def test_mage_vl_query_bypasses_silent_gate_and_preserves_codec_metadata():
    seen: dict[str, object] = {}

    async def gate(session, windows):
        del session, windows
        return {"should_respond": False}

    async def generate(session, windows, query, gate_decision):
        del session, gate_decision
        seen["query"] = query
        seen["window"] = windows[-1]
        yield "The player is moving downfield."

    adapter = MageVLDuplexAdapter(gate=gate, generate=generate, window_size=1)
    session = DuplexSession(
        "mage",
        DuplexSessionConfig(
            input_modalities=("video", "codec_window", "text"),
            output_modalities=("text",),
            proactive=True,
        ),
    )
    rt = MageVLDuplexRuntime(session, adapter)
    out, emit = _collector()

    await rt.run(
        _feed(
            [
                {"type": ev.INPUT_APPEND, "modality": "text", "data": "What changed?"},
                {
                    "type": ev.INPUT_APPEND,
                    "modality": "video",
                    "data": {
                        "kind": "h264",
                        "segment_id": "seg-7",
                        "pts_ms": 7000,
                        "duration_ms": 1000,
                        "codec": {"motion_vectors": "mv", "residual_energy": "res"},
                        "metadata": {"gop": "P"},
                    },
                },
                {"type": ev.CLOSE},
            ]
        ),
        emit,
    )

    window = seen["window"]
    assert seen["query"] == "What changed?"
    assert isinstance(window, MageVLCodecWindow)
    assert window.kind == "h264"
    assert window.segment_id == "seg-7"
    assert window.pts_ms == 7000
    assert window.metadata == {"gop": "P"}
    assert [event["data"] for event in out if event["type"] == ev.RESPONSE_DELTA] == ["The player is moving downfield."]


@pytest.mark.asyncio
async def test_mage_vl_session_state_is_isolated():
    seen: dict[str, tuple[MageVLCodecWindow, ...]] = {}

    async def generate(session, windows, query, gate_decision):
        del query, gate_decision
        seen[session.session_id] = tuple(windows)
        yield session.session_id

    adapter = MageVLDuplexAdapter(generate=generate, window_size=1)
    session_a = DuplexSession(
        "mage-a",
        DuplexSessionConfig(input_modalities=("video", "text"), output_modalities=("text",), proactive=True),
    )
    session_b = DuplexSession(
        "mage-b",
        DuplexSessionConfig(input_modalities=("video", "text"), output_modalities=("text",), proactive=True),
    )

    await adapter.on_input(session_a, "video", {"segment_id": "a", "frames": ["a0"]})
    await adapter.on_input(session_b, "video", {"segment_id": "b", "frames": ["b0"]})
    await adapter.on_input(session_a, "text", "describe a")
    await adapter.on_input(session_b, "text", "describe b")
    assert adapter.should_respond(session_a)
    assert adapter.should_respond(session_b)

    async for _ in adapter.respond(session_a):
        pass
    async for _ in adapter.respond(session_b):
        pass
    assert [window.segment_id for window in seen["mage-a"]] == ["a"]
    assert [window.segment_id for window in seen["mage-b"]] == ["b"]
