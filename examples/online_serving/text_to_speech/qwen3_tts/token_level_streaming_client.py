# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-level streaming text input client for Qwen3-TTS.

Demonstrates ``streaming_mode="token_level"`` over the WebSocket speech
endpoint ``/v1/audio/speech/stream``: a single TTS request is started from an
initial text fragment and extended with later ``input.text`` chunks through
the engine's native resumable-input path. Audio for the start of an utterance
can begin before the rest of the text has been sent (e.g. while an upstream
LLM is still producing tokens), without rebuilding the request.

Usage:
    python token_level_streaming_client.py \
        --api-base http://localhost:8091 \
        --voice vivian \
        --out token_level.wav

Requirements:
    pip install websockets
"""

import argparse
import asyncio
import json
import time
import wave

try:
    import websockets
except ImportError:
    raise SystemExit("Please install websockets: pip install websockets")

# Default text, split into small chunks to simulate an upstream LLM streaming
# tokens. A real caller would forward tokens as they are produced.
DEFAULT_CHUNKS = [
    "Hello there, ",
    "this is a token level ",
    "streaming text input test ",
    "for Qwen3 TTS. ",
    "The audio starts playing ",
    "before all the text has been sent.",
]


def save_pcm_wav(path: str, pcm: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # s16le
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


async def run(args: argparse.Namespace) -> None:
    url = args.api_base.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    url += "/v1/audio/speech/stream"

    config = {
        "type": "session.config",
        "voice": args.voice,
        "streaming_mode": "token_level",
        "response_format": "pcm",
    }
    if args.model:
        config["model"] = args.model

    audio = bytearray()
    sample_rate = 24000
    t0 = time.perf_counter()
    t_first_audio: float | None = None
    t_input_done: float | None = None

    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps(config))

        async def feed_text() -> None:
            nonlocal t_input_done
            for chunk in DEFAULT_CHUNKS:
                await ws.send(json.dumps({"type": "input.text", "text": chunk}))
                print(f"[{time.perf_counter() - t0:5.2f}s] sent input.text ({len(chunk)} chars)")
                await asyncio.sleep(args.chunk_delay)
            await ws.send(json.dumps({"type": "input.done"}))
            t_input_done = time.perf_counter() - t0
            print(f"[{t_input_done:5.2f}s] sent input.done")

        feeder = asyncio.create_task(feed_text())
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
                now = time.perf_counter() - t0
                if isinstance(msg, bytes):
                    if t_first_audio is None:
                        t_first_audio = now
                        print(f"[{now:5.2f}s] first audio frame ({len(msg)} bytes)")
                    audio.extend(msg)
                    continue
                evt = json.loads(msg)
                etype = evt.get("type")
                if etype == "audio.start":
                    sample_rate = evt.get("sample_rate", sample_rate)
                    print(f"[{now:5.2f}s] audio.start sample_rate={sample_rate}")
                elif etype == "audio.done":
                    print(f"[{now:5.2f}s] audio.done total_bytes={evt.get('total_bytes')} error={evt.get('error')}")
                elif etype == "error":
                    print(f"[{now:5.2f}s] ERROR: {evt.get('message')}")
                elif etype == "session.done":
                    print(f"[{now:5.2f}s] session.done")
                    break
        finally:
            if not feeder.done():
                feeder.cancel()

    duration = len(audio) / 2 / sample_rate
    print(f"\nreceived {len(audio)} PCM bytes ({duration:.2f}s @ {sample_rate} Hz)")
    if t_first_audio is not None and t_input_done is not None:
        if t_first_audio < t_input_done:
            print(f"streaming confirmed: first audio at {t_first_audio:.2f}s, input.done at {t_input_done:.2f}s")
    if audio:
        save_pcm_wav(args.out, bytes(audio), sample_rate)
        print(f"wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://localhost:8091", help="Server base URL")
    parser.add_argument("--model", default=None, help="Optional model name to validate against the server")
    parser.add_argument("--voice", default="vivian", help="Voice preset")
    parser.add_argument("--chunk-delay", type=float, default=0.1, help="Seconds between input.text chunks")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-message receive timeout")
    parser.add_argument("--out", default="token_level.wav", help="Output WAV path")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
