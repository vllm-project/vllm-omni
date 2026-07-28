# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent sentence-commit streaming client for Qwen3-TTS.

Each sentence is committed as soon as it is available, while one WebSocket
and one engine request preserve model state for the full utterance.
"""

import argparse
import asyncio
import json
import wave

try:
    import websockets
except ImportError:
    raise SystemExit("Please install websockets: pip install websockets")


DEFAULT_SENTENCES = [
    "Hello there, this first sentence can begin playing immediately.",
    " The second sentence resumes the same request without resetting the voice.",
    " The final sentence is followed by input.done to release the session.",
]


def save_pcm_wav(path: str, pcm: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


async def run(args: argparse.Namespace) -> None:
    url = args.api_base.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    url += "/v1/audio/speech/stream"
    config = {
        "type": "session.config",
        "voice": args.voice,
        "streaming_mode": "sentence_commit",
        "response_format": "pcm",
    }
    if args.model:
        config["model"] = args.model

    audio = bytearray()
    sample_rate = 24000
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps(config))

        async def feed_sentences() -> None:
            for sentence_index, sentence in enumerate(DEFAULT_SENTENCES):
                await ws.send(json.dumps({"type": "input.text", "text": sentence}))
                await ws.send(
                    json.dumps(
                        {
                            "type": "input.commit",
                            "commit_id": f"sentence-{sentence_index}",
                        }
                    )
                )
                await asyncio.sleep(args.sentence_delay)
            await ws.send(json.dumps({"type": "input.done"}))

        feeder = asyncio.create_task(feed_sentences())
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
                if isinstance(msg, bytes):
                    audio.extend(msg)
                    continue
                event = json.loads(msg)
                event_type = event.get("type")
                if event_type == "audio.start":
                    sample_rate = event.get("sample_rate", sample_rate)
                elif event_type == "input.committed":
                    print(
                        "committed",
                        event.get("commit_id"),
                        f"as sentence {event.get('sentence_index')}",
                    )
                elif event_type == "error":
                    print("ERROR:", event.get("message"))
                elif event_type == "session.done":
                    break
        finally:
            if not feeder.done():
                feeder.cancel()

    if audio:
        save_pcm_wav(args.out, bytes(audio), sample_rate)
        print(f"wrote {len(audio)} PCM bytes to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://localhost:8091")
    parser.add_argument("--model", default=None)
    parser.add_argument("--voice", default="vivian")
    parser.add_argument("--sentence-delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", default="sentence_commit.wav")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
