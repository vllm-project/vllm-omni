"""Single-turn response-required test for Qwen3-Omni native-duplex Realtime API."""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from pathlib import Path

import pytest
import websockets

from tests.helpers.mark import hardware_test
from vllm_omni.experimental.fullduplex.client import (
    build_realtime_url,
    read_pcm16_wav,
)

pytestmark = pytest.mark.omni

RESPONSE_REQUIRED_WAV = Path(__file__).resolve().parents[3] / "assets" / "minicpmo_4_5" / "response_required_16k.wav"
CHUNK_MS = 200
PCM16_SAMPLE_RATE = 16_000
PCM16_BYTES_PER_SAMPLE = 2


async def _run_single_turn_response_required(
    *,
    url: str,
    model: str,
    input_wav: Path,
    timeout_s: float = 30.0,
) -> dict:
    session_id = f"qwen3-omni-duplex-ci-{uuid.uuid4().hex}"
    ws_url = build_realtime_url(url, model, autostart=False, session_id=session_id)
    pcm16 = read_pcm16_wav(input_wav)

    events: list[dict] = []
    audio_deltas: list[bytes] = []
    transcript_chunks: list[str] = []
    done_count = 0
    error_count = 0
    output_sample_rate_hz = 24_000

    chunk_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * CHUNK_MS // 1000

    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "session_id": session_id,
                        "model": model,
                        "modalities": ["audio", "text"],
                        "input_audio_format": "pcm16",
                        "output_audio_format": "pcm16",
                        "turn_detection": {"type": "server_vad"},
                        "temperature": 0.0,
                        "extra_body": {},
                    },
                }
            )
        )

        created = await asyncio.wait_for(ws.recv(), timeout=30)
        event = json.loads(created)
        events.append(event)
        assert event.get("type") == "session.created", f"expected session.created, got {event}"

        for offset in range(0, len(pcm16), chunk_bytes):
            chunk = pcm16[offset : offset + chunk_bytes]
            duration_ms = len(chunk) * 1000 // (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                        "input_audio_format": "pcm16",
                        "sample_rate_hz": PCM16_SAMPLE_RATE,
                        "duration_ms": duration_ms,
                    }
                )
            )

        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        deadline = asyncio.get_event_loop().time() + timeout_s
        while asyncio.get_event_loop().time() < deadline:
            remaining = deadline - asyncio.get_event_loop().time()
            raw = await asyncio.wait_for(ws.recv(), timeout=max(1.0, remaining))
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            events.append(event)
            event_type = event.get("type")

            if event_type == "response.output_audio.delta":
                delta = event.get("delta") or event.get("audio")
                if isinstance(delta, str) and delta:
                    audio_deltas.append(base64.b64decode(delta))
                sr = event.get("sample_rate_hz")
                if isinstance(sr, int) and sr > 0:
                    output_sample_rate_hz = sr

            elif event_type == "response.output_audio_transcript.delta":
                d = event.get("delta", "")
                if d:
                    transcript_chunks.append(d)

            elif event_type == "response.done":
                done_count += 1
                break

            elif event_type == "error":
                error_count += 1
                break

    event_types = [e.get("type") for e in events]
    transcript = "".join(transcript_chunks).strip()

    return {
        "ok": done_count == 1 and error_count == 0 and len(audio_deltas) > 0,
        "audio_delta_count": len(audio_deltas),
        "done_count": done_count,
        "error_count": error_count,
        "transcript": transcript,
        "has_transcript": bool(transcript),
        "output_sample_rate_hz": output_sample_rate_hz,
        "total_audio_bytes": sum(len(d) for d in audio_deltas),
        "event_types": event_types,
    }


@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_duplex_single_turn_response_required(omni_server) -> None:
    assert RESPONSE_REQUIRED_WAV.is_file(), f"missing test asset: {RESPONSE_REQUIRED_WAV}"
    url = f"ws://{omni_server.host}:{omni_server.port}/v1/realtime"
    result = asyncio.run(
        _run_single_turn_response_required(
            url=url,
            model=omni_server.model,
            input_wav=RESPONSE_REQUIRED_WAV,
        )
    )
    assert result["ok"], json.dumps(result, ensure_ascii=False, indent=2)
    assert result["audio_delta_count"] > 0
    assert result["done_count"] == 1
    assert result["error_count"] == 0
    assert result["has_transcript"]
