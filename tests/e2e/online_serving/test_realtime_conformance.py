# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E conformance tests for ``WS /v1/realtime``.

Validates WebSocket lifecycle, audio streaming, transcription output,
error handling, and recovery for the OpenAI-compatible realtime endpoint.

From ``tests/``::

    pytest -s -v e2e/online_serving/test_realtime_conformance.py
"""

import asyncio
import json
import os
import warnings

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pybase64 as base64
import pytest
import websockets
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio

from tests.helpers.mark import hardware_marks, hardware_test
from tests.helpers.runtime import OmniServerParams

pytestmark = [pytest.mark.slow, pytest.mark.omni]

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=["--trust-remote-code"],
            env_dict={"VLLM_ENGINE_ITERATION_TIMEOUT_S": "600"},
            use_omni=True,
        ),
        id="qwen3_omni_realtime",
        marks=hardware_marks(res={"cuda": "H100"}),
    ),
]


# ---- Helpers ----


def _get_audio_chunks() -> list[str]:
    """Split mary_had_lamb into ~0.1s PCM16 chunks (base64-encoded)."""
    path = AudioAsset("mary_had_lamb").get_local_path()
    audio, _ = load_audio(str(path), sr=16000, mono=True)
    chunk_size = 1600
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        chunk_int16 = (chunk * 32767).astype(np.int16)
        chunks.append(base64.b64encode(chunk_int16.tobytes()).decode("utf-8"))
    return chunks


async def _recv(ws, timeout: float = 60.0) -> dict:
    msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(msg)


async def _send(ws, event: dict) -> None:
    await ws.send(json.dumps(event))


def _ws_url(openai_client) -> str:
    base = openai_client.base_url.rstrip("/")
    if base.startswith("http://"):
        return "ws://" + base.removeprefix("http://") + "/v1/realtime"
    if base.startswith("https://"):
        return "wss://" + base.removeprefix("https://") + "/v1/realtime"
    raise ValueError(f"Unsupported base_url: {base!r}")


async def _setup_session(ws, model: str) -> None:
    """Connect handshake: wait for session.created, send session.update, wait for session.updated."""
    event = await _recv(ws, timeout=30.0)
    assert event["type"] == "session.created", f"Expected session.created, got {event}"

    await _send(ws, {"type": "session.update", "model": model})

    try:
        while True:
            event = await _recv(ws, timeout=5.0)
            if event["type"] == "session.updated":
                break
    except (TimeoutError, asyncio.TimeoutError):
        warnings.warn(
            "session.updated not received within 5s — server may not implement this event.",
            stacklevel=3,
        )


async def _warmup(ws, audio_chunks: list[str]) -> None:
    """Send a small warmup commit to trigger JIT compilation."""
    await _send(ws, {"type": "input_audio_buffer.commit"})
    await _send(ws, {"type": "input_audio_buffer.append", "audio": audio_chunks[0]})
    await _send(ws, {"type": "input_audio_buffer.commit", "final": True})

    while True:
        event = await _recv(ws, timeout=600.0)
        if event["type"] in ("transcription.done", "error"):
            break


async def _stream_and_collect(ws, audio_chunks: list[str]) -> tuple[str, dict]:
    """Send all audio chunks, commit, collect transcription deltas, return (full_text, done_event)."""
    await _send(ws, {"type": "input_audio_buffer.commit"})
    for chunk in audio_chunks:
        await _send(ws, {"type": "input_audio_buffer.append", "audio": chunk})
    await _send(ws, {"type": "input_audio_buffer.commit", "final": True})

    full_text = ""
    while True:
        event = await _recv(ws, timeout=60.0)
        if event["type"] == "transcription.delta":
            full_text += event["delta"]
        elif event["type"] == "transcription.done":
            return full_text, event
        elif event["type"] == "error":
            pytest.fail(f"Received error during streaming: {event}")


# ---- Session lifecycle ----


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_realtime_session_created(omni_server, openai_client) -> None:
    """Connecting to ``/v1/realtime`` returns a ``session.created`` event."""
    ws_url = _ws_url(openai_client)

    async def _run():
        async with websockets.connect(ws_url) as ws:
            event = await _recv(ws, timeout=30.0)
            assert event["type"] == "session.created", f"Expected session.created, got {event}"
            assert isinstance(event, dict)

    asyncio.run(_run())


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_realtime_session_update(omni_server, openai_client) -> None:
    """``session.update`` with valid model returns ``session.updated`` event per OpenAI Realtime spec."""
    ws_url = _ws_url(openai_client)

    async def _run():
        async with websockets.connect(ws_url) as ws:
            event = await _recv(ws, timeout=30.0)
            assert event["type"] == "session.created"

            await _send(ws, {"type": "session.update", "model": omni_server.model})
            event = await _recv(ws, timeout=10.0)
            assert event["type"] == "session.updated", f"Expected session.updated per OpenAI spec, got {event}"

    asyncio.run(_run())


# ---- Audio streaming + transcription ----


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_realtime_multi_chunk_streaming(omni_server, openai_client) -> None:
    """Full lifecycle: session → stream audio chunks → transcription.done with accumulated text."""
    ws_url = _ws_url(openai_client)
    audio_chunks = _get_audio_chunks()

    async def _run():
        async with websockets.connect(ws_url) as ws:
            await _setup_session(ws, omni_server.model)
            await _warmup(ws, audio_chunks)
            full_text, done_event = await _stream_and_collect(ws, audio_chunks)

            assert full_text.strip(), "Streaming produced empty transcription"
            assert done_event["text"] == full_text, (
                f"done.text mismatch: {done_event['text']!r} vs accumulated {full_text!r}"
            )
            assert "mary" in full_text.lower(), f"Expected 'mary' in transcription, got: {full_text!r}"

    asyncio.run(_run())


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_realtime_transcription_delta_schema(omni_server, openai_client) -> None:
    """``transcription.delta`` has ``delta`` (str); ``transcription.done`` has ``text`` (str)."""
    ws_url = _ws_url(openai_client)
    audio_chunks = _get_audio_chunks()

    async def _run():
        async with websockets.connect(ws_url) as ws:
            await _setup_session(ws, omni_server.model)
            await _warmup(ws, audio_chunks)

            await _send(ws, {"type": "input_audio_buffer.commit"})
            for chunk in audio_chunks:
                await _send(ws, {"type": "input_audio_buffer.append", "audio": chunk})
            await _send(ws, {"type": "input_audio_buffer.commit", "final": True})

            delta_seen = False
            while True:
                event = await _recv(ws, timeout=60.0)
                if event["type"] == "transcription.delta":
                    delta_seen = True
                    assert "delta" in event, f"Missing 'delta' key: {event}"
                    assert isinstance(event["delta"], str), f"delta is not a string: {type(event['delta'])}"
                elif event["type"] == "transcription.done":
                    assert "text" in event, f"Missing 'text' in done event: {event}"
                    assert isinstance(event["text"], str)
                    break
                elif event["type"] == "error":
                    pytest.fail(f"Error: {event}")

            assert delta_seen, "No transcription.delta events received"

    asyncio.run(_run())


# ---- Error handling ----


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_realtime_invalid_model_error(omni_server, openai_client) -> None:
    """``session.update`` with a nonexistent model returns an ``error`` event."""
    ws_url = _ws_url(openai_client)

    async def _run():
        async with websockets.connect(ws_url) as ws:
            event = await _recv(ws, timeout=30.0)
            assert event["type"] == "session.created"

            await _send(ws, {"type": "session.update", "model": "nonexistent-model"})
            event = await _recv(ws, timeout=10.0)
            assert event["type"] == "error", f"Expected error, got {event}"
            assert "nonexistent-model" in event.get("error", ""), f"Error should mention bad model name: {event}"

    asyncio.run(_run())


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_realtime_commit_without_session_update(omni_server, openai_client) -> None:
    """Committing before ``session.update`` returns an error with ``model_not_validated``."""
    ws_url = _ws_url(openai_client)

    async def _run():
        async with websockets.connect(ws_url) as ws:
            event = await _recv(ws, timeout=30.0)
            assert event["type"] == "session.created"

            await _send(ws, {"type": "input_audio_buffer.commit", "final": True})
            event = await _recv(ws, timeout=10.0)
            assert event["type"] == "error", f"Expected error, got {event}"
            assert "model_not_validated" in event.get("code", ""), f"Expected code 'model_not_validated', got: {event}"

    asyncio.run(_run())


# ---- Recovery ----


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_realtime_empty_commit_recovery(omni_server, openai_client) -> None:
    """Empty commit (no audio) does not crash the engine; a second connection succeeds."""
    ws_url = _ws_url(openai_client)
    audio_chunks = _get_audio_chunks()

    async def _run():
        # First connection: empty commit
        async with websockets.connect(ws_url) as ws:
            await _setup_session(ws, omni_server.model)
            await _send(ws, {"type": "input_audio_buffer.commit"})
            await _send(ws, {"type": "input_audio_buffer.commit", "final": True})
            event = await _recv(ws, timeout=360.0)
            assert event["type"] in ("error", "transcription.done", "transcription.delta")

        # Second connection: normal transcription to verify engine is alive
        async with websockets.connect(ws_url) as ws:
            await _setup_session(ws, omni_server.model)
            await _warmup(ws, audio_chunks)
            full_text, _ = await _stream_and_collect(ws, audio_chunks)
            assert full_text.strip(), "Engine failed after empty commit"

    asyncio.run(_run())


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_realtime_graceful_close(omni_server, openai_client) -> None:
    """Clean WebSocket close after session setup does not crash the server."""
    ws_url = _ws_url(openai_client)

    async def _run():
        # Open and close cleanly
        async with websockets.connect(ws_url) as ws:
            await _setup_session(ws, omni_server.model)
            # Close without sending any audio

        # Verify server is still healthy with a new connection
        async with websockets.connect(ws_url) as ws:
            event = await _recv(ws, timeout=30.0)
            assert event["type"] == "session.created", f"Server unhealthy after graceful close: {event}"

    asyncio.run(_run())


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_realtime_abnormal_disconnect(omni_server, openai_client) -> None:
    """Mid-stream disconnect without final commit does not crash the engine."""
    ws_url = _ws_url(openai_client)
    audio_chunks = _get_audio_chunks()

    async def _run():
        # Connect, start streaming, then disconnect abruptly
        ws = await websockets.connect(ws_url)
        try:
            await _setup_session(ws, omni_server.model)
            await _warmup(ws, audio_chunks)
            await _send(ws, {"type": "input_audio_buffer.commit"})
            for chunk in audio_chunks[:5]:
                await _send(ws, {"type": "input_audio_buffer.append", "audio": chunk})
        finally:
            await ws.close()

        # Verify engine is still alive
        async with websockets.connect(ws_url) as ws2:
            event = await _recv(ws2, timeout=30.0)
            assert event["type"] == "session.created", f"Server unhealthy after abnormal disconnect: {event}"

    asyncio.run(_run())
