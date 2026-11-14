# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared OpenAI API test helpers for realtime sessions and video frames."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import wave
from contextlib import suppress
from pathlib import Path

import numpy as np
import websockets

CLIENT_VAD_REPLAY_PATH = Path(__file__).resolve().parents[2] / "assets" / "livekit" / "client_vad_replay.jsonl"
SEMANTIC_VAD_MODEL_PATH = os.environ.get("VLLM_OMNI_TEST_SMART_TURN_MODEL_PATH")
SERVER_VAD_MODEL_PATH = os.environ.get("VLLM_OMNI_TEST_SILERO_VAD_MODEL_PATH")
MODEL = os.environ.get("VLLM_OMNI_TEST_QWEN3_OMNI_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")


def _pcm16_mono_from_wav_bytes(wav_bytes: bytes, sample_rate_hz: int = 16000) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono WAV, got {wf.getnchannels()} channels")
        if wf.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM, sampwidth={wf.getsampwidth()}")
        if wf.getframerate() != sample_rate_hz:
            raise ValueError(f"Expected {sample_rate_hz} Hz input for /v1/realtime, got {wf.getframerate()} Hz")
        if wf.getcomptype() != "NONE":
            raise ValueError(f"Expected uncompressed PCM, comptype={wf.getcomptype()!r}")
        return wf.readframes(wf.getnframes())


async def _append_pcm16_chunks(ws, pcm16: bytes, chunk_bytes: int) -> None:
    for offset in range(0, len(pcm16), chunk_bytes):
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm16[offset : offset + chunk_bytes]).decode("utf-8"),
                }
            )
        )


async def _receive_server_vad_turn(ws) -> list[dict]:
    events: list[dict] = []
    while True:
        message = await asyncio.wait_for(ws.recv(), timeout=600)
        if isinstance(message, bytes):
            continue
        event = json.loads(message)
        events.append(event)
        if event.get("type") == "error":
            raise AssertionError(f"WebSocket error: {event}")
        if event.get("type") == "response.done":
            return events


async def _run_server_vad_audio_roundtrips(
    host: str,
    port: int,
    model: str,
    pcm16_24k: bytes,
    *,
    chunk_ms: int = 100,
    turns: int = 2,
    turn_detection: dict | None = None,
    silence_ms: int = 1500,
    duplex: bool = False,
) -> list[list[dict]]:
    chunk_bytes = max(24_000 * 2 // 1000 * chunk_ms, 2)
    turn_events: list[list[dict]] = []

    query = "?duplex=1" if duplex else ""
    async with websockets.connect(f"ws://{host}:{port}/v1/realtime{query}", max_size=64 * 1024 * 1024) as ws:
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "model": model,
                            "audio": {
                                "input": {
                                    "format": {"type": "audio/pcm", "rate": 24_000},
                                    "turn_detection": turn_detection or {"type": "server_vad"},
                                }
                            },
                        },
                    }
                )
            )

            silence = bytes(24_000 * 2 * silence_ms // 1000)
            for _ in range(turns):
                await _append_pcm16_chunks(ws, pcm16_24k, chunk_bytes)
                await _append_pcm16_chunks(ws, silence, chunk_bytes)
                turn_events.append(await _receive_server_vad_turn(ws))
        finally:
            if duplex:
                # Disconnect leaves a resumable duplex session holding the
                # admission slot. Wait for its release before the next test.
                await ws.send(json.dumps({"type": "session.close"}))
                closing_events: list[dict] = []
                await _duplex_receive_until(
                    ws,
                    closing_events,
                    lambda: any(e["type"] == "session.closed" for e in closing_events),
                    timeout_s=15,
                )

    return turn_events


def _assert_auto_committed_turns(turns: list[list[dict]]) -> None:
    """Each turn: one speech start/stop, one commit and one terminal audio response, in order."""
    required_sequence = [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "response.created",
        "response.output_audio.delta",
        "response.output_audio.done",
        "response.done",
    ]
    input_item_ids: list[str] = []
    response_ids: list[str] = []
    for events in turns:
        event_types = [event["type"] for event in events]
        for event_type in required_sequence:
            if event_type != "response.output_audio.delta":
                assert event_types.count(event_type) == 1, event_types
        positions = [event_types.index(event_type) for event_type in required_sequence]
        assert positions == sorted(positions)

        started = next(event for event in events if event["type"] == "input_audio_buffer.speech_started")
        stopped = next(event for event in events if event["type"] == "input_audio_buffer.speech_stopped")
        committed = next(event for event in events if event["type"] == "input_audio_buffer.committed")
        created = next(event for event in events if event["type"] == "response.created")["response"]
        done = next(event for event in events if event["type"] == "response.done")["response"]

        assert started["item_id"] == stopped["item_id"] == committed["item_id"]
        input_item_id = committed["item_id"]
        history_events = [
            event
            for event in events
            if event["type"] in {"conversation.item.added", "conversation.item.done"}
            and event["item"]["id"] == input_item_id
        ]
        assert [event["type"] for event in history_events] == [
            "conversation.item.added",
            "conversation.item.done",
        ]
        assert all(event["item"]["role"] == "user" for event in history_events)
        output_pcm = b"".join(
            base64.b64decode(event["delta"])
            for event in events
            if event["type"] == "response.output_audio.delta" and event.get("delta")
        )
        assert output_pcm
        assert created["id"] == done["id"]
        # CI caps stage-0 text generation at 150 tokens, so a valid VAD turn can
        # finish with an incomplete response and reason=max_output_tokens.
        assert done["status"] in {"completed", "incomplete"}, done
        if done["status"] == "incomplete":
            assert done["status_details"]["reason"] == "max_output_tokens", done
        input_item_ids.append(input_item_id)
        response_ids.append(created["id"])

    assert len(set(input_item_ids)) == 2
    assert len(set(response_ids)) == 2


def _output_text(response: dict) -> str:
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            text = content.get("transcript") or content.get("text")
            if text:
                parts.append(text)
    return "".join(parts)


async def _run_client_vad_replay(
    host: str,
    port: int,
    model: str,
    replay_path: Path,
    *,
    wait_s: float = 10.0,
    answer_timeout_s: float = 60.0,
    stall_s: list[float] | None = None,
) -> list[str]:
    """Replay a captured LiveKit client-VAD session at its original speed.

    The capture ends with the recorder's own teardown: two ``response.cancel``
    events sent 5.93 s after its last ``response.create``. Replayed at their
    captured offsets they are not part of the conversation but a wall-clock
    deadline for the last turn -- a reply that is merely slower than the
    recording is cancelled before its first transcript delta and comes back as
    an empty answer (issue #9069). They are held back and replayed once the
    answers are collected, so the checks below compare model output rather than
    the recording's teardown. The cancels that precede a ``response.create`` are
    part of the interaction and still go out on time.

    When ``stall_s`` is given, it receives one entry per response: seconds from
    ``response.created`` to its first transcript delta (or to ``response.done``
    if no delta arrived).
    """
    with replay_path.open() as replay_file:
        records = [json.loads(line) for line in replay_file]
    assert records

    answers_by_response_id: dict[str, str] = {}
    response_order: list[str] = []
    completed_answers: dict[str, str] = {}
    pending_responses: dict[str, asyncio.Future] = {}
    errors: list[dict] = []
    created_at: dict[str, float] = {}
    first_output_at: dict[str, float] = {}

    last_create = max(
        (index for index, record in enumerate(records) if record["data"].get("type") == "response.create"),
        default=-1,
    )
    trailing_cancels = [
        record["data"]
        for index, record in enumerate(records)
        if index > last_create and record["data"].get("type") == "response.cancel"
    ]

    async def _await_last_turn() -> None:
        """Wait for the last started response to finish, without cancelling it."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + answer_timeout_s
        while not response_order and loop.time() < deadline:
            await asyncio.sleep(0.01)
        if not response_order:
            return
        pending = pending_responses[response_order[-1]]
        with suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(pending), timeout=max(deadline - loop.time(), 0.0))

    async with websockets.connect(
        f"ws://{host}:{port}/v1/realtime?model={model}",
        max_size=64 * 1024 * 1024,
    ) as ws:

        async def receive_responses() -> None:
            async for message in ws:
                if isinstance(message, bytes):
                    continue
                event = json.loads(message)
                event_type = event.get("type")
                if event_type == "error":
                    errors.append(event)
                elif event_type == "response.created":
                    response_id = event["response"]["id"]
                    response_order.append(response_id)
                    answers_by_response_id[response_id] = ""
                    pending_responses[response_id] = asyncio.get_running_loop().create_future()
                    created_at[response_id] = asyncio.get_running_loop().time()
                elif event_type in {
                    "response.audio_transcript.delta",
                    "response.output_audio_transcript.delta",
                    "response.output_text.delta",
                    "transcription.delta",
                }:
                    response_id = event.get("response_id")
                    if response_id in answers_by_response_id:
                        answers_by_response_id[response_id] += event.get("delta", "")
                        if event.get("delta"):
                            first_output_at.setdefault(response_id, asyncio.get_running_loop().time())
                elif event_type == "response.done":
                    response = event["response"]
                    response_id = response["id"]
                    text = answers_by_response_id.get(response_id, "")
                    completed_answers[response_id] = text or _output_text(response)
                    first_output_at.setdefault(response_id, asyncio.get_running_loop().time())
                    future = pending_responses.get(response_id)
                    if future is not None and not future.done():
                        future.set_result(None)

        receiver = asyncio.create_task(receive_responses())
        capture_start = records[0]["ts"]
        replay_start = asyncio.get_running_loop().time()

        try:
            for index, record in enumerate(records):
                target = replay_start + (record["ts"] - capture_start)
                delay = target - asyncio.get_running_loop().time()
                if delay > 0:
                    await asyncio.sleep(delay)

                message = record["data"]
                # The capture's reference-voice update is only valid when the
                # replay client uploads its optional reference-audio asset.
                if "__VOICE__" in json.dumps(message):
                    continue
                if index > last_create and message.get("type") == "response.cancel":
                    continue
                await ws.send(json.dumps(message))

            await _await_last_turn()
            # Replay the recorded teardown now that every answer is in.
            for message in trailing_cancels:
                await ws.send(json.dumps(message))
            await asyncio.sleep(wait_s)
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)

    assert not [error for error in errors if error.get("error", {}).get("type") == "server_error"]
    if stall_s is not None:
        stall_s.extend(
            first_output_at[response_id] - created_at[response_id]
            for response_id in response_order
            if response_id in first_output_at
        )
    return [completed_answers[response_id] for response_id in response_order if response_id in completed_answers]


def _duplex_fixture_pcm(path: Path, *, duration_s: float | None = None) -> bytes:
    """Replay checked-in speech at the web client's negotiated 24 kHz PCM16 rate."""
    pcm = np.frombuffer(_pcm16_mono_from_wav_bytes(path.read_bytes()), dtype="<i2")
    if duration_s is not None:
        pcm = pcm[: int(duration_s * 16000)]
    positions = np.arange(len(pcm) * 3 // 2) * (2.0 / 3.0)
    return np.interp(positions, np.arange(len(pcm)), pcm).astype("<i2").tobytes()


async def _duplex_receive_until(ws, events: list[dict], predicate, *, timeout_s: float = 180) -> None:
    # One deadline covers the whole operation, including an endless stream that
    # would otherwise reset a per-recv timeout forever.
    async def receive() -> None:
        while not predicate():
            raw = await ws.recv()
            assert isinstance(raw, str), "Expected JSON Realtime events"
            event = json.loads(raw)
            events.append(event)
            assert event.get("type") != "error", event

    await asyncio.wait_for(receive(), timeout=timeout_s)


def make_jpeg(r: int = 128, g: int = 128, b: int = 128, size: int = 64) -> bytes:
    """Create a solid-colour JPEG image."""
    import pytest

    Image = pytest.importorskip("PIL.Image", reason="Pillow required for video stream tests")
    img = Image.new("RGB", (size, size), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def make_gradient_jpeg(seed: int, size: int = 64) -> bytes:
    """Create a random-gradient JPEG that varies based on *seed*."""
    import pytest

    Image = pytest.importorskip("PIL.Image", reason="Pillow required for video stream tests")
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()
