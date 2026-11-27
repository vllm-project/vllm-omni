# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""#11074 e2e: realtime transcription sessions on /v1/realtime.

The conversation (non-duplex) realtime path serves a transcription session:
``session.type: "transcription"`` switches committed audio turns into ASR —
one engine request per turn, streaming
``conversation.item.input_audio_transcription.delta`` and finishing with
``.completed``. Both turn sources drive it: server VAD commits and explicit
``input_audio_buffer.commit``.
"""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest
import websockets

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    SERVER_PARAMS,
    validated_input_wav,
)
from tests.helpers.mark import hardware_test
from vllm_omni.clients.duplex import read_pcm16_wav

pytestmark = [pytest.mark.advanced_model]

SAMPLE_RATE_HZ = 24000
_INPUT_RATE_HZ = 16000


def conversation_url(omni_server) -> str:
    """The non-duplex realtime endpoint: ?duplex=0 bypasses the duplex
    handler even on a duplex-capable server, reaching the conversation
    connection that transcription sessions extend."""
    return f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=0"


def _pcm16_24k(wav_path) -> bytes:
    """Upsample the 16 kHz fixture to the 24 kHz the session expects."""
    samples = np.frombuffer(read_pcm16_wav(wav_path), dtype=np.int16).astype(np.float64)
    upsampled = np.interp(
        np.arange(len(samples) * 3 // 2) * 2.0 / 3.0,
        np.arange(len(samples)),
        samples,
    )
    return upsampled.astype(np.int16).tobytes()


async def _run_transcription_session(url: str, wav_path, *, vad: bool) -> list[dict]:
    events: list[dict] = []
    async with websockets.connect(url, ping_interval=None) as ws:
        session: dict = {
            "type": "transcription",
            "input_audio_transcription": {"language": "en"},
        }
        if vad:
            session["audio"] = {"input": {"turn_detection": {"type": "server_vad"}}}
        await ws.send(json.dumps({"type": "session.update", "session": session}))

        # The update must be accepted before any audio streams.
        while True:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=30.0))
            events.append(event)
            if event.get("type") == "session.updated":
                break
            assert event.get("type") != "error", f"session.update rejected: {event}"

        pcm = _pcm16_24k(wav_path)
        # One second of trailing silence: VAD needs it to detect end of
        # speech, and it is harmless padding for the manual commit.
        pcm += b"\x00" * (SAMPLE_RATE_HZ * 2)
        chunk_bytes = SAMPLE_RATE_HZ * 2 // 5  # 200 ms per append
        for i in range(0, len(pcm), chunk_bytes):
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm[i : i + chunk_bytes]).decode("ascii"),
                    }
                )
            )
        if not vad:
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Collect until the turn's transcription finishes.
        while True:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=60.0))
            events.append(event)
            if event.get("type") in (
                "conversation.item.input_audio_transcription.completed",
                "conversation.item.input_audio_transcription.failed",
                # Fail fast on the first error (e.g. server_vad_unavailable)
                # instead of waiting out the timeout.
                "error",
            ):
                break
    return events


def _assert_transcribed(events: list[dict]) -> None:
    errors = [e for e in events if e.get("type") == "error"]
    assert not errors, f"Session errors: {errors}"
    deltas = [e for e in events if e.get("type") == "conversation.item.input_audio_transcription.delta"]
    completed = [e for e in events if e.get("type") == "conversation.item.input_audio_transcription.completed"]
    assert completed, f"No transcription.completed event; got: {[e.get('type') for e in events]}"
    assert deltas, "No transcription.delta events"
    assert completed[0].get("transcript"), "Transcription completed with empty transcript"


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
@pytest.mark.parametrize("vad", [True, False])
def test_realtime_transcription_session(omni_server, vad) -> None:
    """A transcription session turns each committed turn into ASR events."""
    events = asyncio.run(_run_transcription_session(conversation_url(omni_server), validated_input_wav(), vad=vad))
    _assert_transcribed(events)
