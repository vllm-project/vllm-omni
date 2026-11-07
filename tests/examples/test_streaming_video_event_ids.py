# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scheduled controls must not reuse IDs when timestamps round to the same value."""

import asyncio
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.example]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    example_dir = Path(__file__).resolve().parents[2] / "examples/online_serving/streaming_video_generation"
    monkeypatch.syspath_prepend(str(example_dir))
    return importlib.import_module("streaming_video_client")


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def perf_counter(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.now += delay


class _CaptureWebSocket:
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.messages: list[dict[str, Any]] = []
        self.sent_at: list[float] = []

    async def send(self, message: str) -> None:
        self.messages.append(json.loads(message))
        self.sent_at.append(self.clock.now - 100.0)


@pytest.fixture
def clock(client: ModuleType, monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(client.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(client.asyncio, "sleep", clock.sleep)
    return clock


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["prompt", "camera"])
@pytest.mark.parametrize(
    ("times", "order", "sent_at"),
    [
        ([0.0001, 0.0002], [0, 1], [0.0001, 0.0002]),
        ([1.0, 1.0], [0, 1], [1.0, 1.0]),
        ([0.0002, 0.0001, 0.0002], [1, 0, 2], [0.0001, 0.0002, 0.0002]),
    ],
    ids=["sub-millisecond", "same-time", "stable-sort"],
)
async def test_scheduled_updates_have_unique_ids(
    client: ModuleType, clock: _Clock, kind: str, times: list[float], order: list[int], sent_at: list[float]
) -> None:
    entries: list[dict[str, object]]
    expected: list[dict[str, object]]
    if kind == "prompt":
        entries = [
            {"prompt": "first scene"},
            {"prompt": "second scene", "transition_chunks": 0},
            {"prompt": "third scene", "transition_chunks": 2},
        ]
        expected = [
            {"event": {"prompt": "first scene"}, "transition_chunks": 3},
            {"event": {"prompt": "second scene"}, "transition_chunks": 0},
            {"event": {"prompt": "third scene"}, "transition_chunks": 2},
        ]
    else:
        velocity = {"mode": "velocity", "data": {"translation": [0, 0, 0.05], "rotation": [0, 0, 0, 1]}}
        target = {"mode": "target", "data": {"translation": [0, 0, 1], "rotation": [0, 0, 0, 1]}}
        entries = [
            {"camera": velocity},
            {"camera": target, "transition_chunks": 0},
            {"camera": target},
        ]
        expected = [
            {"event": {"multi_modal_data": {"camera": velocity}}},
            {"event": {"multi_modal_data": {"camera": target}}, "transition_chunks": 0},
            {"event": {"multi_modal_data": {"camera": target}}, "transition_chunks": 3},
        ]

    updates = getattr(client, f"_parse_{kind}_updates")(
        json.dumps([dict(entry, at=at) for entry, at in zip(entries, times)])
    )
    websocket = _CaptureWebSocket(clock)
    await getattr(client, f"_run_{kind}_update_scheduler")(
        websocket, updates, video_started_at=100.0, send_lock=asyncio.Lock()
    )

    assert len(websocket.messages) == len(times)
    assert all(message["type"] == "session.interaction" for message in websocket.messages)
    interactions = [message["interaction"] for message in websocket.messages]
    ids = [interaction.pop("event_id") for interaction in interactions]
    assert all(isinstance(event_id, str) and 0 < len(event_id) <= 128 for event_id in ids)
    assert len(set(ids)) == len(times)
    assert interactions == [expected[index] for index in order]
    assert websocket.sent_at == pytest.approx(sent_at)


@pytest.mark.asyncio
async def test_prompt_and_camera_schedules_share_one_id_namespace(client: ModuleType, clock: _Clock) -> None:
    prompts = client._parse_prompt_updates('[{"at": 0, "prompt": "first"}, {"at": 0, "prompt": "second"}]')
    cameras = client._parse_camera_updates('[{"at": 0, "actions": ["w"]}, {"at": 0, "actions": ["a"]}]')
    websocket = _CaptureWebSocket(clock)
    send_lock = asyncio.Lock()
    await asyncio.gather(
        client._run_prompt_update_scheduler(websocket, prompts, video_started_at=100.0, send_lock=send_lock),
        client._run_camera_update_scheduler(websocket, cameras, video_started_at=100.0, send_lock=send_lock),
    )

    assert len(websocket.messages) == 4
    ids = [message["interaction"]["event_id"] for message in websocket.messages]
    assert len(set(ids)) == 4
