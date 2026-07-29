# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.experimental.fullduplex.joyvl.serving.config import InteractionConfig
from vllm_omni.experimental.fullduplex.joyvl.serving.server import (
    _extract_frames_and_query,
    _incoming_time_ranges,
    _normalize_time_range,
)
from vllm_omni.experimental.fullduplex.joyvl.serving.session import InteractionSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FRAME = "data:image/jpeg;base64,AAA"


class _FakeBackend:
    def __init__(self):
        self.requests = []

    async def generate(self, messages, **kwargs):
        self.requests.append(messages)
        return "</silence>", None

    async def aclose(self):
        pass


class _FakeRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}


def _payload(parts):
    return {"messages": [{"role": "user", "content": parts}]}


def test_normalize_time_range_variants():
    assert _normalize_time_range("3.0 seconds") == "3.0 seconds"
    assert _normalize_time_range("3s") == "3.0 seconds"
    assert _normalize_time_range("<12.34 seconds>") == "12.3 seconds"
    assert _normalize_time_range("1s~2s") == "1.0 seconds ~ 2.0 seconds"
    assert _normalize_time_range("1.0 seconds-2.0 seconds") == "1.0 seconds-2.0 seconds"
    assert _normalize_time_range("not a time") is None
    assert _normalize_time_range(None) is None


def test_time_markers_stripped_from_query():
    frames, query, markers = _extract_frames_and_query(
        _payload(
            [
                {"type": "text", "text": "<3.0 seconds>"},
                {"type": "text", "text": "Alert me if a fire breaks out"},
                {"type": "image_url", "image_url": {"url": FRAME}},
            ]
        )
    )
    assert frames == [FRAME]
    assert query == "Alert me if a fire breaks out"
    assert markers == ["3.0 seconds"]


def test_incoming_time_ranges_priority():
    payload = _payload([{"type": "image_url", "image_url": {"url": FRAME}}])
    payload["frame_time_ranges"] = ["5s", "6s"]
    payload["frame_time_range"] = "9s"
    assert _incoming_time_ranges(_FakeRequest(), payload, []) == ["5.0 seconds", "6.0 seconds"]

    del payload["frame_time_ranges"]
    assert _incoming_time_ranges(_FakeRequest(), payload, []) == ["9.0 seconds"]

    del payload["frame_time_range"]
    request = _FakeRequest({"x-frame-time-range": "7s"})
    assert _incoming_time_ranges(request, payload, []) == ["7.0 seconds"]

    assert _incoming_time_ranges(_FakeRequest(), payload, ["3.0 seconds"]) == ["3.0 seconds"]


@pytest.mark.asyncio
async def test_step_emits_one_time_marker_per_turn():
    config = InteractionConfig(enable_memory=False, force_silence_before_query=False)
    session = InteractionSession("s", config, _FakeBackend())
    await session.step([FRAME, FRAME, FRAME], time_ranges=["4.0 seconds"])
    content = session.chunk.messages[0]["content"]
    text_parts = [p["text"] for p in content if p["type"] == "text"]
    assert text_parts == ["<4.0 seconds>"]
    assert sum(1 for p in content if p["type"] == "image_url") == 3


@pytest.mark.asyncio
async def test_step_counts_turns_for_chunk_window():
    config = InteractionConfig(enable_memory=False, force_silence_before_query=False, chunk_frames=2)
    session = InteractionSession("s", config, _FakeBackend())
    result = await session.step([FRAME, FRAME, FRAME])  # one turn, three frames
    assert result.turn_index == 1
    assert session._policy.needs_flush() is False
    result = await session.step([FRAME])
    assert result.turn_index == 2
    assert session._policy.needs_flush() is True
