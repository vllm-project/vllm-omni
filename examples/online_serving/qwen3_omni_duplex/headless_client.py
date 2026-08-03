#!/usr/bin/env python3
"""Drive this demo's duplex session from the command line, no browser.

Mirrors app.html on the wire -- same query string, same ``session.update``,
same 200 ms ``is_speech`` appends read from ``sample_16k.wav``, same commit,
same ``session.close``. One "session" here is one Connect click.

Useful because the browser needs a microphone and a tunnel, and because a
reproduction you can run N times in a row is what turns "it goes silent
sometimes" into a number:

    python headless_client.py --sessions 5 --turns 3

Exits non-zero unless every turn of every session produced audio.

Turns are framed by silence on the wire, not by a terminator event:
``response.audio.done`` and ``response.done`` both arrive, and a trailing one
left queued reads as the *next* turn's terminator, which reports a working turn
as empty. That artifact cost a wrong conclusion once already.
"""

import argparse
import asyncio
import base64
import json
import pathlib
import sys
import time

import websockets

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SR_IN = 16000
CHUNK_MS = 200
CLIP = pathlib.Path(__file__).parent / "sample_16k.wav"

URL = (
    "ws://127.0.0.1:8099/v1/realtime"
    f"?duplex=1&model={MODEL.replace('/', '%2F')}"
    "&qwen3_omni_native_duplex=1&autostart=0"
)

SESSION_UPDATE = {
    "type": "session.update",
    "session": {
        "model": MODEL,
        "modalities": ["audio", "text"],
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "turn_detection": None,
        "overlap_policy": "listen_only",
        "playback_commit_policy": "ack_only",
        "extra_body": {
            "auto_response": True,
            "qwen3_omni_native_duplex": True,
            "force_listen_count": 0,
        },
    },
}


def stamp() -> str:
    return time.strftime("%H:%M:%S")


class Turn:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.text = ""
        self.audio_deltas = 0
        self.audio_bytes = 0
        self.response_created = False
        self.errors: list[str] = []


async def run_session(idx: int, turns: int, timeout: float, pcm: bytes) -> list[Turn]:
    results: list[Turn] = []
    print(f"[{stamp()}] session {idx}: connecting", flush=True)
    async with websockets.connect(URL, max_size=None, open_timeout=30) as ws:
        await ws.send(json.dumps(SESSION_UPDATE))

        # Wait for session.created / session.updated before talking, like the
        # browser enabling its buttons.
        ready = False
        deadline = time.monotonic() + 30
        while not ready and time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                continue
            msg = json.loads(raw)
            if msg.get("type") in ("session.created", "session.updated"):
                sid = (msg.get("session") or {}).get("id") or msg.get("session_id")
                print(f"[{stamp()}] session {idx}: {msg['type']} sid={sid}", flush=True)
                ready = True
        if not ready:
            print(f"[{stamp()}] session {idx}: NO session.created -- handshake hung", flush=True)
            return results

        bytes_per_chunk = SR_IN * 2 * CHUNK_MS // 1000
        for turn_no in range(1, turns + 1):
            turn = Turn()
            results.append(turn)
            sent_ms = 0
            for off in range(0, len(pcm) - bytes_per_chunk + 1, bytes_per_chunk):
                sent_ms += CHUNK_MS
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(pcm[off : off + bytes_per_chunk]).decode(),
                            "input_audio_format": "pcm16",
                            "sample_rate_hz": SR_IN,
                            "duration_ms": CHUNK_MS,
                            "audio_end_ms": sent_ms,
                            "is_speech": True,
                        }
                    )
                )
                await asyncio.sleep(0.01)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
            t_commit = time.monotonic()
            print(f"[{stamp()}] session {idx} turn {turn_no}: committed {sent_ms} ms", flush=True)

            # Drain everything this turn produces. Terminators are not a
            # reliable frame boundary here (`response.audio.done` and
            # `response.done` both arrive, and a trailing one left queued gets
            # misread as the *next* turn's terminator), so read until the wire
            # goes quiet for `idle_s` instead.
            idle_s = 4.0
            while True:
                left = timeout - (time.monotonic() - t_commit)
                if left <= 0:
                    print(
                        f"[{stamp()}] session {idx} turn {turn_no}: TIMEOUT after {timeout:.0f}s"
                        f" (events: {turn.events or 'NONE'})",
                        flush=True,
                    )
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(left, idle_s))
                except asyncio.TimeoutError:
                    if turn.events:
                        break  # quiet after real activity: turn is over
                    continue
                except websockets.ConnectionClosed as exc:
                    turn.errors.append(f"closed: {exc}")
                    print(f"[{stamp()}] session {idx} turn {turn_no}: SOCKET CLOSED {exc}", flush=True)
                    return results
                msg = json.loads(raw)
                t = msg.get("type", "?")
                turn.events.append(t)
                if t == "response.created":
                    turn.response_created = True
                elif t in ("response.audio_transcript.delta", "response.output_text.delta"):
                    turn.text += msg.get("delta") or ""
                elif t == "response.audio_transcript.done" and msg.get("transcript"):
                    turn.text = msg["transcript"]
                elif t == "response.audio.delta":
                    data = msg.get("delta") or msg.get("audio") or ""
                    turn.audio_deltas += 1
                    turn.audio_bytes += len(base64.b64decode(data)) if data else 0
                elif t == "error":
                    turn.errors.append(json.dumps(msg))
                    print(f"[{stamp()}] session {idx} turn {turn_no}: ERROR {msg}", flush=True)
            dt = time.monotonic() - t_commit
            print(
                f"[{stamp()}] session {idx} turn {turn_no}: text={turn.text!r} "
                f"audio_deltas={turn.audio_deltas} bytes={turn.audio_bytes} in {dt:.1f}s",
                flush=True,
            )

        await ws.send(json.dumps({"type": "session.close"}))
        await asyncio.sleep(0.5)
    print(f"[{stamp()}] session {idx}: closed", flush=True)
    return results


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--turns", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=90)
    args = ap.parse_args()

    pcm = CLIP.read_bytes()[44:]
    print(f"clip: {len(pcm)} bytes = {len(pcm) / 2 / SR_IN:.1f}s", flush=True)

    ok = 0
    for idx in range(1, args.sessions + 1):
        try:
            turns = await run_session(idx, args.turns, args.timeout, pcm)
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            print(f"[{stamp()}] session {idx}: EXCEPTION {type(exc).__name__}: {exc}", flush=True)
            continue
        if turns and all(t.audio_deltas > 0 for t in turns):
            ok += 1
        await asyncio.sleep(2)

    print(f"\n{ok}/{args.sessions} sessions produced audio on every turn", flush=True)
    return 0 if ok == args.sessions else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
