#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example WebSocket client for the /v1/video/chat/stream endpoint.

Sends video frames from a local file (or generates synthetic ones), submits a
query, and prints the streamed text response.

Requirements:
    pip install websockets pillow

Usage:
    # With a video file (requires opencv-python):
    python streaming_video_client.py --video my_clip.mp4 \\
        --query "What is happening in this video?"

    # Synthetic frames (no extra deps):
    python streaming_video_client.py \\
        --query "Describe what you see." \\
        --synthetic-frames 10

    # With audio (Phase 3):
    python streaming_video_client.py --video my_clip.mp4 \\
        --audio my_audio.pcm \\
        --query "What is the person saying and doing?"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys

try:
    import websockets
except ImportError:
    print("Please install websockets:  pip install websockets")
    sys.exit(1)

from PIL import Image


def _generate_synthetic_frame(index: int, width: int = 320, height: int = 240) -> bytes:
    """Generate a simple synthetic JPEG frame with a colour gradient."""
    r = (index * 37) % 256
    g = (index * 73) % 256
    b = (index * 113) % 256
    img = Image.new("RGB", (width, height), (r, g, b))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _load_video_frames(path: str, max_frames: int = 64, fps: int = 2) -> list[bytes]:
    """Extract frames from a video file using OpenCV."""
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
    print(f"Loaded {len(frames)} frames from {path} (interval={frame_interval})")
    return frames


async def run(args: argparse.Namespace) -> None:
    uri = f"ws://{args.host}:{args.port}/v1/video/chat/stream"

    # Prepare frames
    if args.video:
        frames = _load_video_frames(args.video, max_frames=args.max_frames, fps=args.fps)
    else:
        frames = [_generate_synthetic_frame(i) for i in range(args.synthetic_frames)]
        print(f"Generated {len(frames)} synthetic frames")

    # Prepare audio (optional, Phase 3)
    audio_data: bytes | None = None
    if args.audio:
        with open(args.audio, "rb") as f:
            audio_data = f.read()
        print(f"Loaded audio: {len(audio_data)} bytes")

    async with websockets.connect(uri, max_size=16 * 1024 * 1024) as ws:
        # 1. Send session.config
        config = {
            "type": "session.config",
            "model": args.model,
            "modalities": ["text", "audio"] if audio_data else ["text"],
            "max_frames": args.max_frames,
            "num_frames": args.num_sample_frames,
            "enable_frame_filter": args.evs,
            "frame_filter_threshold": args.evs_threshold,
            "use_audio_in_video": bool(audio_data),
        }
        await ws.send(json.dumps(config))
        print(f"Sent session.config: model={args.model} evs={args.evs}")

        # 2. Send frames
        for i, frame in enumerate(frames):
            msg = {
                "type": "video.frame",
                "data": base64.b64encode(frame).decode(),
            }
            await ws.send(json.dumps(msg))
            if (i + 1) % 10 == 0:
                print(f"  Sent {i + 1}/{len(frames)} frames")
        print(f"Sent all {len(frames)} frames")

        # 3. Send audio chunks (Phase 3)
        if audio_data:
            chunk_size = 16000 * 2  # 1 second of 16 kHz 16-bit PCM
            for offset in range(0, len(audio_data), chunk_size):
                chunk = audio_data[offset : offset + chunk_size]
                msg = {
                    "type": "audio.chunk",
                    "data": base64.b64encode(chunk).decode(),
                }
                await ws.send(json.dumps(msg))
            print(f"Sent audio in {(len(audio_data) + chunk_size - 1) // chunk_size} chunks")

        # 4. Send query, then immediately send video.done so the server
        #    knows the session is complete (avoids deadlock where client
        #    waits for session.done while server waits for video.done).
        await ws.send(json.dumps({"type": "video.query", "text": args.query}))
        print(f"\nQuery: {args.query}")
        print("Response: ", end="", flush=True)

        # Signal end of session right after the query.  The server will
        # process the query first (it's already queued), then handle
        # video.done and reply with session.done.
        await ws.send(json.dumps({"type": "video.done"}))

        # 5. Receive response until session.done
        recv_timeout = 120  # seconds — avoid infinite hang if server stalls
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "response.text.delta":
                print(data.get("delta", ""), end="", flush=True)
            elif msg_type == "response.text.done":
                print()  # newline
            elif msg_type == "response.evs_stats":
                retained = data.get("retained_count", 0)
                dropped = data.get("dropped_count", 0)
                rate = data.get("drop_rate", 0)
                print(f"\nEVS stats: retained={retained} dropped={dropped} drop_rate={rate:.1%}")
            elif msg_type == "session.done":
                print("Session complete.")
                break
            elif msg_type == "error":
                print(f"\nError: {data.get('message')}")
                break
            elif msg_type == "response.start":
                pass  # expected
            else:
                print(f"\n[unknown message] {data}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming video chat client")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-MoE")
    parser.add_argument("--video", help="Path to video file (requires opencv-python)")
    parser.add_argument("--audio", help="Path to raw PCM 16kHz audio file (Phase 3)")
    parser.add_argument("--query", default="What do you see in this video?")
    parser.add_argument(
        "--synthetic-frames", type=int, default=10, help="Number of synthetic frames if --video is not set"
    )
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--num-sample-frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=2, help="Frame extraction rate from video")
    parser.add_argument(
        "--no-evs", dest="evs", action="store_false", help="Disable EVS frame filtering (enabled by default)"
    )
    parser.set_defaults(evs=True)
    parser.add_argument("--evs-threshold", type=float, default=0.95)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
