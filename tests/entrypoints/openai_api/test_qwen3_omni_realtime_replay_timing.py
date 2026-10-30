# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Client-VAD replay must not let the capture's teardown truncate an answer.

Regression for issue #9069: ``test_livekit_client_vad_replay`` reported an empty
third answer in Merge CI. The capture ends with ``response.cancel`` events sent
5.93 s after its last ``response.create``. Replaying them at the capture's own
pace gives the third turn a fixed wall-clock deadline, so a reply that is only
slower than the recording is cancelled before its first transcript delta and
the test reports ``''`` -- exactly the observed failure signature.

These tests drive the real replay helper against a scripted local WebSocket, so
they need no accelerator and no model.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import websockets

from tests.entrypoints.openai_api.test_qwen3_omni_realtime_websocket import (
    CLIENT_VAD_REPLAY_PATH,
    _run_client_vad_replay,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TURN_ANSWERS = [
    "Hello! I'm Qwen-Omni, a multimodal large-scale language model. How can I assist you?",
    "The capital of France is Paris.",
    "The capital of France is Paris.",
]


class _ScriptedRealtimeServer:
    """Answers every ``response.create`` and honours ``response.cancel``."""

    def __init__(self, first_delta_delay_s: Callable[[int], float]) -> None:
        self._first_delta_delay_s = first_delta_delay_s
        self._turn = 0
        self._active: str | None = None
        self.received_cancels = 0

    async def _send_done(self, ws, response_id: str, text: str, status: str) -> None:
        content = [{"type": "audio", "transcript": text}] if text else []
        await ws.send(
            json.dumps(
                {
                    "type": "response.done",
                    "response": {
                        "id": response_id,
                        "status": status,
                        "output": [{"type": "message", "content": content}],
                    },
                }
            )
        )
        if self._active == response_id:
            self._active = None

    async def _generate(self, ws, response_id: str, turn: int) -> None:
        delay = self._first_delta_delay_s(turn)
        if delay:
            await asyncio.sleep(delay)
        if self._active != response_id:
            # Cancelled while generating: the server finishes it with no text.
            await self._send_done(ws, response_id, "", "cancelled")
            return
        await ws.send(
            json.dumps(
                {
                    "type": "response.output_audio_transcript.delta",
                    "response_id": response_id,
                    "delta": TURN_ANSWERS[turn - 1],
                }
            )
        )
        await self._send_done(ws, response_id, TURN_ANSWERS[turn - 1], "completed")

    async def handle(self, ws) -> None:
        tasks: list[asyncio.Task] = []
        try:
            async for raw in ws:
                event = json.loads(raw)
                event_type = event.get("type")
                if event_type == "response.cancel":
                    self.received_cancels += 1
                    self._active = None
                    continue
                if event_type != "response.create":
                    continue
                self._turn += 1
                response_id = f"resp-{self._turn}"
                self._active = response_id
                await ws.send(
                    json.dumps(
                        {
                            "type": "response.created",
                            "response": {"id": response_id, "status": "in_progress", "output": []},
                        }
                    )
                )
                tasks.append(asyncio.create_task(self._generate(ws, response_id, self._turn)))
        finally:
            for task in tasks:
                task.cancel()


async def _replay_against(server: _ScriptedRealtimeServer) -> list[str]:
    async with websockets.serve(server.handle, "127.0.0.1", 0) as served:
        port = served.sockets[0].getsockname()[1]
        return await _run_client_vad_replay(
            "127.0.0.1",
            port,
            "scripted-model",
            CLIENT_VAD_REPLAY_PATH,
            wait_s=0.5,
            answer_timeout_s=30.0,
        )


@pytest.mark.asyncio
async def test_replay_waits_out_a_slow_turn_after_the_last_create() -> None:
    """A third turn slower than the capture's teardown gap still keeps its text."""
    server = _ScriptedRealtimeServer(lambda turn: 8.0 if turn == 3 else 0.0)
    answers = await asyncio.wait_for(_replay_against(server), timeout=180)

    assert len(answers) == 3, answers
    assert "qwen" in answers[0].lower(), answers
    assert "paris" in answers[1].lower(), answers
    assert "paris" in answers[2].lower(), answers


@pytest.mark.asyncio
async def test_replay_still_sends_the_captured_teardown() -> None:
    """Deferring the trailing cancels must not drop them from the replay."""
    server = _ScriptedRealtimeServer(lambda turn: 0.0)
    answers = await asyncio.wait_for(_replay_against(server), timeout=180)

    captured_cancels = [
        json.loads(line)["data"]
        for line in CLIENT_VAD_REPLAY_PATH.read_text().splitlines()
        if json.loads(line)["data"].get("type") == "response.cancel"
    ]
    assert server.received_cancels == len(captured_cancels)
    assert len(answers) == 3, answers


def test_replay_capture_ends_with_cancels_after_the_last_create() -> None:
    """Guard the capture shape this fix depends on."""
    records = [json.loads(line) for line in Path(CLIENT_VAD_REPLAY_PATH).read_text().splitlines()]
    types = [record["data"].get("type") for record in records]
    last_create = len(types) - 1 - types[::-1].index("response.create")
    trailing = [
        record["data"] for record in records[last_create + 1 :] if record["data"].get("type") == "response.cancel"
    ]
    assert trailing, "capture no longer ends with response.cancel events; revisit the replay fix"
