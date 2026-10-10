# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Replay deterministic burst/staggered OpenPI waves; write per-request JSONL.

Run against one externally started server. Repeat for each backend using the
same arguments and stagger interval. Connection setup is excluded from latency;
packing, send, inference, receive and unpack are included. No profiler is used.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def make_observation(seed: int, views: int, steps: int = 10) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "prompt": "Pick up the red block and place it in the bin",
        "state": rng.uniform(-0.1, 0.1, 32).astype(np.float32),
        "images": {
            f"observation.images.{camera}": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            for camera in CAMERAS[:views]
        },
        "seed": seed,
        "sampling_params": {"num_inference_steps": steps},
    }


def summarize(rows: list[dict], duration_s: float) -> dict:
    good = [r["latency_ms"] for r in rows if r["error"] is None]
    return {
        "completed": len(good),
        "failed": len(rows) - len(good),
        "req_per_s": len(good) / duration_s if duration_s else 0,
        **{f"p{q}_ms": float(np.percentile(good, q)) if good else None for q in (50, 95, 99)},
    }


async def run_wave(connections, observations, *, stagger_ms: float, timeout: float, save_actions: bool):
    from openpi_client import msgpack_numpy

    epoch = time.perf_counter()

    async def infer(index, conn, obs):
        scheduled = index * stagger_ms / 1000
        await asyncio.sleep(max(0, epoch + scheduled - time.perf_counter()))
        start = time.perf_counter()
        row = {"seed": obs["seed"], "scheduled_ms": scheduled * 1000, "sent_ms": (start - epoch) * 1000, "error": None}
        try:
            async with asyncio.timeout(timeout):
                await conn.send(msgpack_numpy.Packer().pack(obs))
                actions = msgpack_numpy.unpackb(await conn.recv())
            if isinstance(actions, dict):
                raise ValueError(f"Server returned an error instead of actions: {actions}")
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape != (50, 32) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid actions: {actions.shape}")
            row["latency_ms"] = (time.perf_counter() - start) * 1000
            row["action_sha256"] = hashlib.sha256(actions.tobytes()).hexdigest()
            if save_actions:
                row["actions"] = actions.tolist()
        except Exception as exc:
            row["latency_ms"] = (time.perf_counter() - start) * 1000
            row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    rows = await asyncio.gather(*(infer(i, c, o) for i, (c, o) in enumerate(zip(connections, observations))))
    return rows, time.perf_counter() - epoch


async def benchmark(args):
    from contextlib import AsyncExitStack

    from openpi_client import msgpack_numpy
    from websockets.asyncio.client import connect

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    # Exclusive creation prevents silently replacing an earlier A/B run.
    with args.output.open("x") as raw:
        raw.write(json.dumps({"metadata": {**vars(args), "output": str(args.output)}}) + "\n")
        for views in args.views:
            for count in args.concurrency:
                for arrival in args.arrivals:
                    async with AsyncExitStack() as stack:
                        connections = []
                        for _ in range(count):
                            conn = await stack.enter_async_context(
                                connect(args.url, compression=None, max_size=8 << 20)
                            )
                            async with asyncio.timeout(args.timeout):
                                metadata = msgpack_numpy.unpackb(await conn.recv())
                            if metadata.get("action_horizon") != 50 or metadata.get("action_dim") != 32:
                                raise ValueError(f"Unexpected server contract: {metadata}")
                            connections.append(conn)
                        for wave in range(math.ceil(args.warmup / count)):
                            obs = [make_observation(args.seed + i, views) for i in range(count)]
                            rows, _ = await run_wave(
                                connections, obs, stagger_ms=0, timeout=args.timeout, save_actions=False
                            )
                            if any(r["error"] for r in rows):
                                raise RuntimeError(f"Warmup failed: {rows}")
                        rows_all, duration = [], 0.0
                        for wave in range(math.ceil(args.requests / count)):
                            obs = [make_observation(args.seed + wave * count + i, views) for i in range(count)]
                            rows, elapsed = await run_wave(
                                connections,
                                obs,
                                stagger_ms=args.stagger_ms if arrival == "staggered" else 0,
                                timeout=args.timeout,
                                save_actions=args.save_actions,
                            )
                            duration += elapsed
                            for row in rows:
                                row.update(
                                    views=views, concurrency=count, arrival=arrival, wave=wave, backend=args.backend
                                )
                                raw.write(json.dumps(row, allow_nan=False) + "\n")
                            raw.flush()
                            rows_all.extend(rows)
                            if any(r["error"] for r in rows):
                                raise RuntimeError(
                                    "Request failed; raw results saved. Stop before increasing capacity."
                                )
                        summary = {
                            "views": views,
                            "concurrency": count,
                            "arrival": arrival,
                            "backend": args.backend,
                            **summarize(rows_all, duration),
                        }
                        summaries.append(summary)
                        print(json.dumps(summary), flush=True)
        raw.write(json.dumps({"summaries": summaries}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8093/v1/realtime/robot/openpi")
    parser.add_argument(
        "--backend", required=True, help="Evidence label, e.g. f1-full or d0-step (not a server switch)"
    )
    parser.add_argument("--server-command", required=True, help="Exact startup command, without credentials")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument(
        "--concurrency", type=int, nargs="+", choices=(1, 2, 4, 8, 16, 32), default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--arrivals", nargs="+", choices=("burst", "staggered"), default=["burst", "staggered"])
    parser.add_argument("--stagger-ms", type=float, help="Fixed arrival gap calibrated on the serial baseline")
    parser.add_argument("--requests", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-actions", action="store_true", help="Save numerical outputs for paired correctness analysis"
    )
    args = parser.parse_args()
    if "staggered" in args.arrivals and (args.stagger_ms is None or args.stagger_ms <= 0):
        parser.error("staggered runs require a positive --stagger-ms, identical across backends")
    if args.requests < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("requests/timeout must be positive and warmup must be nonnegative")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
