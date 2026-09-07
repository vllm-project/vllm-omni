# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Live client: vLLM-Omni ``WS /v1/realtime/video`` to an LL-HLS playlist.

Connects to a running vLLM-Omni server, streams generated fragments straight into
the packager, and writes a latency report covering the terms this process can see:
session start, first byte, first playable part.

Protocol is the one documented in
``examples/online_serving/streaming_video_generation/README.md``:

    -> {"type": "session.start", "model": ..., "prompt": ..., "format": "m4s", ...}
    <- {"type": "video.start", "request_id": ..., "config": {...}, "format": "m4s"}
    <- {"type": "video.chunk_metadata", ...}      # JSON, precedes each binary frame
    <- <binary frame: fragmented MP4 bytes>
    <- {"type": "session.done", "chunks": N, "stopped": false}

Requires ``websockets`` (the upstream example already depends on it):

    pip install websockets

Usage:

    python3 src/client.py --host 127.0.0.1 --port 8000 \\
        --model BestWishYsh/Helios-Distilled \\
        --prompt "A serene lakeside sunrise with mist over the water." \\
        --width 640 --height 384 --fps 16 --num-frames 99 \\
        --out out/live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fmp4 import FragmentedMP4Splitter  # noqa: E402
from llhls import LLHLSPackager  # noqa: E402

# The Helios-Distilled preset the upstream example uses by default. Streaming-rate
# generation depends on the low step counts; without them chunks arrive far slower
# than playback and the whole premise collapses.
HELIOS_PRESET = {
    "is_enable_stage2": True,
    "pyramid_num_stages": 3,
    "pyramid_num_inference_steps_list": [1, 1, 1],
    "is_amplify_first_chunk": True,
}


def build_session_start(a: argparse.Namespace) -> dict:
    msg: dict = {
        "type": "session.start",
        "model": a.model,
        "prompt": a.prompt,
        "format": "m4s",
        "width": a.width,
        "height": a.height,
        "fps": a.fps,
        "num_frames": a.num_frames,
    }
    if a.seed is not None:
        msg["seed"] = a.seed
    if a.guidance_scale is not None:
        msg["guidance_scale"] = a.guidance_scale
    extra = dict(HELIOS_PRESET) if a.helios_preset else {}
    if a.extra_params:
        extra.update(json.loads(a.extra_params))
    if extra:
        msg["extra_params"] = extra
    return msg


async def run(a: argparse.Namespace) -> int:
    try:
        import websockets
    except ImportError:
        print("websockets is required:  pip install websockets", file=sys.stderr)
        return 2

    url = f"ws://{a.host}:{a.port}/v1/realtime/video"
    out = Path(a.out)
    splitter = FragmentedMP4Splitter()
    packager: LLHLSPackager | None = None
    part_duration: float | None = None

    marks: dict[str, float] = {}
    chunks = 0
    started = False

    # max_size=None: fragments can exceed the default 1MiB frame cap.
    async with websockets.connect(url, max_size=None, open_timeout=a.timeout) as ws:
        marks["connected"] = time.perf_counter()
        await ws.send(json.dumps(build_session_start(a)))
        marks["session_start_sent"] = time.perf_counter()
        print(f"connected {url}")

        async for message in ws:
            now = time.perf_counter()

            if isinstance(message, (bytes, bytearray)):
                marks.setdefault("first_binary", now)
                for frag in splitter.feed(bytes(message)):
                    if not started:
                        if splitter.init_segment is None:
                            print("fragment before init segment; cannot package", file=sys.stderr)
                            return 1
                        packager = LLHLSPackager(out_dir=out, part_duration=part_duration or 9 / 16)
                        packager.start(splitter.init_segment)
                        marks["init_written"] = time.perf_counter()
                        started = True
                    assert packager is not None
                    rec = packager.add_fragment(frag, received_at=now)
                    marks.setdefault("first_part_published", rec.published_at)
                    chunks += 1
                    print(f"  part {rec.index:>3}  {rec.bytes:>8}B  publish {rec.publish_latency * 1000:.2f}ms")
                continue

            msg = json.loads(message)
            kind = msg.get("type")

            if kind == "video.start":
                marks["video_start"] = now
                cfg = msg.get("config") or {}
                fps = cfg.get("fps") or a.fps
                # Derive part duration from the first metadata rather than assuming:
                # frames-per-chunk is a model/config property, not a constant.
                print(f"video.start request_id={msg.get('request_id')} fps={fps}")
            elif kind == "video.chunk_metadata":
                nf = msg.get("num_frames")
                if part_duration is None and nf:
                    part_duration = nf / (a.fps or 16)
                    print(f"  chunk shape: {nf} frames -> part duration {part_duration:.5f}s")
            elif kind == "session.done":
                marks["session_done"] = now
                print(f"session.done chunks={msg.get('chunks')} stopped={msg.get('stopped')}")
                break
            elif kind == "error":
                print(f"server error: {msg.get('message')}", file=sys.stderr)
                return 1

    if packager is None:
        print("no fragments received", file=sys.stderr)
        return 1

    packager.finish()
    report = packager.report()

    t0 = marks["session_start_sent"]
    report["live"] = {
        "connect_to_video_start_s": round(marks.get("video_start", t0) - t0, 4),
        "prompt_to_first_byte_s": round(marks.get("first_binary", t0) - t0, 4),
        "prompt_to_first_playable_part_s": round(marks.get("first_part_published", t0) - t0, 4),
        "total_session_s": round(marks.get("session_done", t0) - t0, 4),
        "chunks": chunks,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\nlatency budget")
    for k, v in report["live"].items():
        print(f"  {k}: {v}")
    print(f"\nplaylist: {out / 'stream.m3u8'}")
    print(f"serve it:  python3 -m http.server 8080 --directory {out}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="BestWishYsh/Helios-Distilled")
    p.add_argument("--prompt", required=True)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--num-frames", type=int, default=99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--extra-params", default=None, help="JSON merged over the Helios preset")
    p.add_argument("--no-helios-preset", dest="helios_preset", action="store_false")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--out", default="out/live")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
