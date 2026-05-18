"""WebSocket client for CosyVoice3 streaming text-input TTS.

Connects to /v1/audio/speech/stream, sends text incrementally, and saves one
audio file per generated sentence. Use --stream-audio with --response-format pcm
to receive progressive PCM frames while each sentence is decoded.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import inspect
import json
import os
from pathlib import Path
from typing import Any

try:
    import websockets
except ImportError:
    websockets = None

DEFAULT_WS_URL = "ws://localhost:8091/v1/audio/speech/stream"
DEFAULT_MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

MIME_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
}


def encode_audio_to_base64(audio_path: str) -> str:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    mime_type = MIME_BY_SUFFIX.get(path.suffix.lower(), "audio/wav")
    audio_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{audio_b64}"


def normalize_ref_audio(ref_audio: str) -> str:
    if ref_audio.startswith(("http://", "https://", "data:", "file:")):
        return ref_audio
    return encode_audio_to_base64(ref_audio)


def build_session_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": args.model,
        "response_format": args.response_format,
        "ref_audio": normalize_ref_audio(args.ref_audio),
        "ref_text": args.ref_text,
        "split_granularity": args.split_granularity,
    }
    if args.stream_audio:
        config["stream_audio"] = True
    if args.max_new_tokens is not None:
        config["max_new_tokens"] = args.max_new_tokens
    return config


async def send_text(ws: Any, text: str, simulate_stt: bool, stt_delay: float) -> None:
    if simulate_stt:
        words = text.split(" ")
        for index, word in enumerate(words):
            chunk = word + (" " if index < len(words) - 1 else "")
            await ws.send(json.dumps({"type": "input.text", "text": chunk}))
            print(f"sent: {chunk!r}")
            await asyncio.sleep(stt_delay)
    else:
        await ws.send(json.dumps({"type": "input.text", "text": text}))
        print(f"sent text: {text!r}")
    await ws.send(json.dumps({"type": "input.done"}))
    print("sent input.done")


async def stream_tts(args: argparse.Namespace) -> None:
    if websockets is None:
        raise SystemExit("Please install websockets: pip install websockets")

    os.makedirs(args.output_dir, exist_ok=True)
    config = build_session_config(args)

    connect_kwargs: dict[str, Any] = {"max_size": 64 * 1024 * 1024}
    if "proxy" in inspect.signature(websockets.connect).parameters:
        connect_kwargs["proxy"] = None

    async with websockets.connect(args.url, **connect_kwargs) as ws:
        await ws.send(json.dumps({"type": "session.config", **config}))
        print(f"sent session.config: response_format={args.response_format}, stream_audio={args.stream_audio}")
        sender = asyncio.create_task(send_text(ws, args.text, args.simulate_stt, args.stt_delay))

        current_sentence_index = 0
        current_chunks: list[bytes] = []

        try:
            while True:
                message = await ws.recv()
                if isinstance(message, bytes):
                    current_chunks.append(message)
                    print(f"received audio chunk: {len(message)} bytes")
                    continue

                msg = json.loads(message)
                msg_type = msg.get("type")
                if msg_type == "audio.start":
                    current_sentence_index = int(msg["sentence_index"])
                    current_chunks = []
                    print(f"[sentence {current_sentence_index}] {msg['sentence_text']!r}")
                elif msg_type == "audio.done":
                    filename = Path(args.output_dir) / f"sentence_{current_sentence_index:03d}.{args.response_format}"
                    filename.write_bytes(b"".join(current_chunks))
                    print(f"saved {filename} ({msg.get('total_bytes', filename.stat().st_size)} bytes)")
                    current_chunks = []
                elif msg_type == "session.done":
                    print(f"complete: {msg['total_sentences']} sentence(s)")
                    break
                elif msg_type == "error":
                    print(f"error: {msg['message']}")
                else:
                    print(f"unknown message: {msg}")
        finally:
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CosyVoice3 WebSocket streaming speech client")
    parser.add_argument("--url", default=DEFAULT_WS_URL, help="WebSocket endpoint URL")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--ref-audio", required=True, help="Reference audio path, URL, data URL, or file URI")
    parser.add_argument("--ref-text", required=True, help="Transcript of the reference audio")
    parser.add_argument("--output-dir", default="cosyvoice3_streaming_output")
    parser.add_argument("--response-format", default="wav", choices=["wav", "pcm", "flac", "mp3", "aac", "opus"])
    parser.add_argument("--stream-audio", action="store_true", help="Receive progressive PCM frames")
    parser.add_argument("--split-granularity", default="sentence", choices=["sentence", "clause"])
    parser.add_argument("--simulate-stt", action="store_true", help="Send words incrementally")
    parser.add_argument("--stt-delay", type=float, default=0.1, help="Delay between simulated STT words")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stream_audio and args.response_format != "pcm":
        raise SystemExit("--stream-audio requires --response-format pcm")
    asyncio.run(stream_tts(args))


if __name__ == "__main__":
    main()
