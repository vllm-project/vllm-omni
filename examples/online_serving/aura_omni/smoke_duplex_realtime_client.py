#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal AURA duplex Realtime smoke via DuplexClient."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni.clients.duplex import (  # noqa: E402
    DuplexClient,
    EventCollector,
    SessionConfig,
    read_pcm16_wav,
)

PROGRESS_TYPES = {
    "response.created",
    "response.done",
    "response.audio.delta",
    "response.audio_transcript.delta",
    "response.output_text.delta",
    "response.text.delta",
    "response.listen",
    "response.model_listen",
    "error",
}


def _tiny_jpeg_b64() -> str:
    img = Image.new("RGB", (64, 64), color=(200, 40, 40))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def run(url: str, model: str, wav: Path, timeout_s: float) -> int:
    pcm16 = read_pcm16_wav(wav)
    if not pcm16:
        print("empty wav", file=sys.stderr)
        return 2

    frame = _tiny_jpeg_b64()
    collector = EventCollector()
    config = SessionConfig(
        modalities=("text", "audio"),
        instructions="You are AURA. Reply briefly in Chinese.",
        extra_body={
            "aura_system_prompt": "You are AURA. Reply briefly in Chinese.",
            "tts_task_type": "CustomVoice",
            "tts_language": "Chinese",
            "tts_speaker": "Vivian",
        },
    )

    started = time.time()
    client = DuplexClient(
        url,
        model=model,
        config=config,
        reconnect=None,
        heartbeat_interval_s=None,
        handshake_timeout_s=min(60.0, timeout_s),
    )

    async with client:
        consume_task = asyncio.create_task(collector.consume(client))
        collector.add({"type": "session.created", "session": getattr(client, "session_info", {})})

        await client.append_audio(pcm16, is_speech=True, video_frames=[frame])
        await client.commit(create_response=True)

        progress = False
        got_done = False
        while time.time() - started < timeout_s:
            types = [e.get("type") for e in collector.events]
            if any(t in PROGRESS_TYPES for t in types):
                progress = True
            if "response.done" in types or "response.completed" in types:
                got_done = True
                progress = True
                break
            for e in collector.events:
                meta = e.get("metadata") if isinstance(e.get("metadata"), dict) else {}
                if meta.get("model_listen") or meta.get("listen_source") == "aura_silent":
                    progress = True
                    break
            if collector.count("error") > 0 or collector.count("session.closed") > 0:
                break
            # Keep draining until response.done so TTS is not truncated.
            await asyncio.sleep(0.2)
        # expose for result
        collector._smoke_got_done = got_done  # type: ignore[attr-defined]

        try:
            await client.close(timeout_s=30.0)
        except Exception as exc:  # noqa: BLE001
            print(f"close_warn={exc}", file=sys.stderr)
        try:
            await asyncio.wait_for(consume_task, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            consume_task.cancel()

    event_types = [e.get("type") for e in collector.events]
    errors = [json.dumps(e, ensure_ascii=False)[:500] for e in collector.errors()]
    result = {
        "ok": progress and bool(getattr(collector, "_smoke_got_done", False) or collector.count("response.done")),
        "got_response_progress": progress,
        "got_response_done": bool(getattr(collector, "_smoke_got_done", False) or collector.count("response.done")),
        "errors": errors,
        "event_types": event_types,
        "n_events": len(event_types),
        "audio_bytes": len(collector.audio_bytes()),
        "llm_text": collector.response_text(collector.response_ids[0]) if collector.response_ids else "",
        "elapsed_s": round(time.time() - started, 2),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    log_path = Path("/tmp/aura_duplex_smoke/client_events.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in collector.events),
        encoding="utf-8",
    )
    print(f"events_log={log_path}", file=sys.stderr)
    return 0 if progress else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8099/v1/realtime")
    p.add_argument("--model", default="/workspace/models/AURA")
    p.add_argument(
        "--wav",
        default=str(REPO_ROOT / "tests/assets/minicpmo_4_5/response_required_16k.wav"),
    )
    p.add_argument("--timeout-s", type=float, default=300.0)
    args = p.parse_args()
    return asyncio.run(run(args.url, args.model, Path(args.wav), args.timeout_s))


if __name__ == "__main__":
    raise SystemExit(main())
