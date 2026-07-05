#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example WebSocket client for AURA streaming video (/v1/video/chat/stream).

Unlike the Qwen-Omni client, AURA auto-triggers observation turns once enough
frames are buffered. Manual ``video.query`` is optional.

Usage:
    python streaming_video_client.py --synthetic-frames 8
    python streaming_video_client.py --video clip.mp4 --audio mic.pcm
    python streaming_video_client.py --text-only
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
import wave

try:
    import websockets
except ImportError:
    print("Please install websockets:  pip install websockets")
    sys.exit(1)

from PIL import Image


def _generate_synthetic_frame(index: int, width: int = 320, height: int = 240) -> bytes:
    r = (index * 37) % 256
    g = (index * 73) % 256
    b = (index * 113) % 256
    img = Image.new("RGB", (width, height), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _load_video_frames(path: str, max_frames: int = 64, fps: int = 2) -> list[bytes]:
    try:
        import cv2
    except ImportError:
        print("opencv-python is required to read video files: pip install opencv-python")
        sys.exit(1)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Cannot open video: {path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(video_fps / fps))
    frames: list[bytes] = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frames.append(buf.tobytes())
        idx += 1
    cap.release()
    print(f"Loaded {len(frames)} frames from {path}")
    return frames


def _pcm_from_wav_bytes(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, sample_rate


def _write_wav_pcm16(path: str, pcm: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


async def run(args: argparse.Namespace) -> None:
    uri = args.url or f"ws://{args.host}:{args.port}/v1/video/chat/stream"

    if args.video:
        frames = _load_video_frames(args.video, max_frames=args.max_frames, fps=args.fps)
    else:
        frames = [_generate_synthetic_frame(i) for i in range(args.synthetic_frames)]
        print(f"Generated {len(frames)} synthetic frames")

    audio_data: bytes | None = None
    if args.audio:
        with open(args.audio, "rb") as f:
            audio_data = f.read()
        print(f"Loaded audio: {len(audio_data)} bytes")

    modalities = ["text"] if args.text_only else ["text", "audio"]

    async with websockets.connect(uri, max_size=16 * 1024 * 1024) as ws:
        config = {
            "type": "session.config",
            "model": args.model,
            "modalities": modalities,
            "max_frames": args.max_frames,
            "auto_trigger": True,
            "auto_trigger_min_frames": args.auto_trigger_min_frames,
            "max_frames_per_round": args.max_frames_per_round,
            "enable_frame_filter": args.evs,
            "frame_filter_threshold": args.evs_threshold,
        }
        if args.cross_turn_penalty > 0:
            config["cross_turn_penalty"] = args.cross_turn_penalty
            config["cross_turn_lookback"] = args.cross_turn_lookback
        await ws.send(json.dumps(config))
        print(f"Sent session.config (modalities={modalities})")

        for i, frame in enumerate(frames):
            await ws.send(
                json.dumps(
                    {
                        "type": "video.frame",
                        "data": base64.b64encode(frame).decode(),
                    }
                )
            )
            if (i + 1) % 4 == 0:
                print(f"  Sent {i + 1}/{len(frames)} frames")
            await asyncio.sleep(args.frame_interval_ms / 1000.0)
        print(f"Sent all {len(frames)} frames")

        if audio_data:
            chunk_size = 16000 * 2
            for offset in range(0, len(audio_data), chunk_size):
                chunk = audio_data[offset : offset + chunk_size]
                await ws.send(
                    json.dumps(
                        {
                            "type": "audio.chunk",
                            "data": base64.b64encode(chunk).decode(),
                        }
                    )
                )
            print("Sent audio chunks")

        if args.query:
            await ws.send(json.dumps({"type": "video.query", "text": args.query}))
            print(f"Manual query: {args.query}")

        await ws.send(json.dumps({"type": "video.done"}))

        recv_timeout = 120
        audio_pcm_parts: list[bytes] = []
        audio_sample_rate = 24000
        audio_delta_count = 0
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "response.text.delta":
                print(data.get("delta", ""), end="", flush=True)
            elif msg_type == "response.text.done":
                text = data.get("text", "")
                if text.strip():
                    print(f"\n[text.done] {text}")
                else:
                    print()
            elif msg_type == "response.audio.delta":
                audio_delta_count += 1
                wav_b64 = data.get("data", "")
                if wav_b64:
                    pcm, sr = _pcm_from_wav_bytes(base64.b64decode(wav_b64))
                    audio_pcm_parts.append(pcm)
                    audio_sample_rate = sr
                    print(f"[audio.delta #{audio_delta_count}] {len(pcm)} pcm bytes @ {sr} Hz")
            elif msg_type == "response.audio.done":
                print(f"[audio.done] received {audio_delta_count} delta(s)")
                if audio_pcm_parts and args.output_wav:
                    all_pcm = b"".join(audio_pcm_parts)
                    _write_wav_pcm16(args.output_wav, all_pcm, audio_sample_rate)
                    print(f"Saved audio to {args.output_wav}")
            elif msg_type == "response.start":
                print("\n[response.start]")
            elif msg_type == "session.done":
                print("Session complete.")
                break
            elif msg_type == "error":
                print(f"\nError: {data.get('message')}")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="AURA streaming video client")
    parser.add_argument("--url", help="Full WebSocket URL (overrides host/port)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="aurateam/AURA")
    parser.add_argument("--video", help="Path to video file")
    parser.add_argument("--audio", help="Path to raw PCM16 16 kHz mono audio")
    parser.add_argument("--query", default="", help="Optional manual video.query text")
    parser.add_argument("--synthetic-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-frames-per-round", type=int, default=16)
    parser.add_argument("--auto-trigger-min-frames", type=int, default=2)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--frame-interval-ms", type=int, default=100)
    parser.add_argument("--text-only", action="store_true", help="Request text-only modalities")
    parser.add_argument("--output-wav", default="aura_stream_output.wav", help="Save concatenated TTS audio")
    parser.add_argument("--cross-turn-penalty", type=float, default=0.0)
    parser.add_argument("--cross-turn-lookback", type=int, default=2)
    parser.add_argument("--no-evs", dest="evs", action="store_false")
    parser.set_defaults(evs=True)
    parser.add_argument("--evs-threshold", type=float, default=0.95)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
