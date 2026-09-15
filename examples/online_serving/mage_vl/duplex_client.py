#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import websockets


async def run(args: argparse.Namespace) -> None:
    headers = {"Authorization": f"Bearer {args.auth_token}"} if args.auth_token else None
    started = time.perf_counter()
    async with websockets.connect(
        f"{args.url}?session_id={args.session_id}",
        additional_headers=headers,
        max_size=args.max_message_bytes,
        open_timeout=args.timeout,
    ) as websocket:
        created = json.loads(await asyncio.wait_for(websocket.recv(), args.timeout))
        print(json.dumps(created, ensure_ascii=False))
        await websocket.send(
            json.dumps(
                {
                    "type": "input.append",
                    "modality": "text",
                    "data": args.prompt,
                }
            )
        )
        encoded = base64.b64encode(Path(args.video).read_bytes()).decode("ascii")
        await websocket.send(
            json.dumps(
                {
                    "type": "input.append",
                    "modality": "video",
                    "data": {
                        "kind": "video/mp4",
                        "segment_id": args.segment_id,
                        "pts_ms": args.pts_ms,
                        "duration_ms": args.duration_ms,
                        "data": {"video_base64": encoded},
                    },
                }
            )
        )
        while True:
            event = json.loads(await asyncio.wait_for(websocket.recv(), args.timeout))
            print(json.dumps(event, ensure_ascii=False))
            if event.get("type") in {"response.done", "error"}:
                break
        await websocket.send(json.dumps({"type": "close"}))
    print(f"elapsed_s={time.perf_counter() - started:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one Mage-VL video window over the duplex WebSocket.")
    parser.add_argument("--url", default="ws://127.0.0.1:8090/v1/mage-vl/duplex")
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", default="Describe this video segment in detail.")
    parser.add_argument("--session-id", default="mage-vl-example")
    parser.add_argument("--segment-id", default="segment-0")
    parser.add_argument("--pts-ms", type=int, default=0)
    parser.add_argument("--duration-ms", type=int, default=4000)
    parser.add_argument("--auth-token")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-message-bytes", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
