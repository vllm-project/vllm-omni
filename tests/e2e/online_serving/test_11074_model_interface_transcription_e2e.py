# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""#11074 follow-up e2e: model-interface realtime transcription sessions.

A stage-0 model implementing the vLLM speech-to-text interface drives the
transcription turns through its own protocol: ``get_generation_prompt``
builds the turn (the committed PCM is resampled from the session's 24 kHz to
the model's 16 kHz) and the model's streaming cleaner strips the raw output
framing from the deltas, so the client-visible transcript is the clean text
one-shot transcription would produce — not the raw model output.
"""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest
import websockets

from tests.e2e.online_serving.helpers.ming3_audio_asr import SERVER_PARAMS
from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import validated_input_wav
from tests.helpers.mark import hardware_test
from vllm_omni.clients.duplex import read_pcm16_wav

pytestmark = [pytest.mark.advanced_model]

SAMPLE_RATE_HZ = 24000


def conversation_url(omni_server) -> str:
    """The non-duplex realtime endpoint: ``?duplex=0`` reaches the
    conversation connection that transcription sessions extend."""
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

        while True:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=30.0))
            events.append(event)
            if event.get("type") == "session.updated":
                break
            assert event.get("type") != "error", f"session.update rejected: {event}"

        pcm = _pcm16_24k(wav_path)
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

        while True:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=60.0))
            events.append(event)
            if event.get("type") in (
                "conversation.item.input_audio_transcription.completed",
                "conversation.item.input_audio_transcription.failed",
                "error",
            ):
                break
    return events


def _assert_clean_transcription(events: list[dict]) -> None:
    errors = [e for e in events if e.get("type") == "error"]
    assert not errors, f"Session errors: {errors}"
    deltas = [e for e in events if e.get("type") == "conversation.item.input_audio_transcription.delta"]
    completed = [e for e in events if e.get("type") == "conversation.item.input_audio_transcription.completed"]
    assert completed, f"No transcription.completed event; got: {[e.get('type') for e in events]}"
    assert deltas, "No transcription.delta events"
    transcript = completed[0].get("transcript")
    assert transcript, "Transcription completed with empty transcript"
    # The model's raw output carries a leading language field separated by a
    # tab before the transcript text; the streaming cleaner must keep every
    # raw framing token out of the client-visible stream.
    assert "\t" not in transcript, f"Raw framing leaked into transcript: {transcript!r}"
    for delta in deltas:
        assert "\t" not in delta["delta"], f"Raw framing leaked into delta: {delta['delta']!r}"


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
@pytest.mark.parametrize("vad", [True, False])
def test_model_interface_transcription_session(omni_server, vad) -> None:
    """A speech-to-text stage-0 model streams clean transcripts per turn."""
    events = asyncio.run(_run_transcription_session(conversation_url(omni_server), validated_input_wav(), vad=vad))
    _assert_clean_transcription(events)
