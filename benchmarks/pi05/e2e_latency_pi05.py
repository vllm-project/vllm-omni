#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Client-side E2E latency for a running π0.5 OpenPI policy server.

Times the full round trip a robot actually sees: msgpack pack, websocket send,
engine + scheduler + IPC, denoise, msgpack unpack. This is the metric the other
recipes in this repo report as "Client E2E"; the model-layer sample_actions
number from bench_pi05.py measures a strictly smaller thing.

The server must already be serving. Start it with the recipe's command, wait for
the port, then run this against it.

Usage:
  python e2e_latency_pi05.py --host 127.0.0.1 --port 8000 --views 3 \
      --warmup 3 --iters 30 --out e2e_fp32.json

Reports p50/p95/min/max over --iters round trips, plus the device-level memory
high-water mark sampled from nvidia-smi during the run.

For engine-path attribution (see bench_engine_overhead_pi05.py), --requests-out
appends one row per request with the client's ``time.perf_counter_ns()`` stamps,
the clock the server's ``VLLM_OMNI_LATENCY_TRACE_DIR`` trace uses on the same
host, and --request-seed with --save-actions makes two servers comparable
output for output.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
import uuid

import numpy as np

OPENPI_PATH = "/v1/realtime/robot/openpi"
CAMERA_KEYS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)
IMAGE_SIZE = 224
STATE_DIM = 32


def make_obs(*, views: int, prompt: str, session_id: str, rng: np.random.Generator) -> dict:
    """One observation with ``views`` cameras. Random pixels, so no codec path is
    accidentally measured on constant data."""
    obs: dict = {cam: rng.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) for cam in CAMERA_KEYS[:views]}
    obs["state"] = rng.standard_normal(STATE_DIM).astype(np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


class MemorySampler(threading.Thread):
    """Poll nvidia-smi at 5 Hz; report the high-water mark in MiB."""

    def __init__(self, gpu_index: int = 0) -> None:
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.peak_mib = 0
        self._stop_evt = threading.Event()

    def run(self) -> None:
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop_evt.is_set():
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                self.peak_mib = max(self.peak_mib, int(out.stdout.strip().splitlines()[0]))
            except Exception:  # noqa: BLE001 - sampling is best effort
                pass
            self._stop_evt.wait(0.2)

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--views", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--prompt", default="pick up the red block and place it in the bin")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--num-inference-steps", type=int, default=None)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--request-seed", type=int, default=None, help="Noise seed sent with every request.")
    ap.add_argument(
        "--requests-out", default=None, help="Append per-request client timestamps (JSONL), warmup included."
    )
    ap.add_argument("--round", type=int, default=0, help="Label written to every --requests-out row.")
    ap.add_argument("--save-actions", default=None, help="Save the last timed action chunk (.npz).")
    ap.add_argument(
        "--no-memory-sampler",
        action="store_true",
        help="Do not poll nvidia-smi during the run (device_peak_mib is then 0). On the server's host the "
        "sampler adds about 0.2 ms to every round trip.",
    )
    args = ap.parse_args()

    import websockets.sync.client as ws_client
    from openpi_client import msgpack_numpy

    rng = np.random.default_rng(args.seed)
    packer = msgpack_numpy.Packer()
    session_id = str(uuid.uuid4())
    uri = f"ws://{args.host}:{args.port}{OPENPI_PATH}"

    conn = ws_client.connect(uri, compression=None, max_size=None, ping_interval=300, ping_timeout=3600)
    try:
        metadata = msgpack_numpy.unpackb(conn.recv())
        print("handshake:", json.dumps({k: str(v) for k, v in dict(metadata).items()}, indent=1))

        request_rows: list[dict] = []

        def one_round_trip(warmup: bool = False) -> tuple[float, np.ndarray]:
            payload = make_obs(views=args.views, prompt=args.prompt, session_id=session_id, rng=rng)
            if args.num_inference_steps is not None:
                payload["sampling_params"] = {"num_inference_steps": args.num_inference_steps}
            if args.request_seed is not None:
                payload["seed"] = args.request_seed
            payload["endpoint"] = "infer"
            # The timed region is everything the robot pays for: pack, send,
            # server, receive, unpack. Starting after pack or stopping before
            # unpack measures the wire and the server, not the client.
            t0 = time.perf_counter_ns()
            blob = packer.pack(payload)
            conn.send(blob)
            t_sent = time.perf_counter_ns()
            raw = conn.recv()
            t_received = time.perf_counter_ns()
            if isinstance(raw, str):
                raise RuntimeError(f"Inference failed: {raw}")
            reply = msgpack_numpy.unpackb(raw)
            # Same decode contract as tests/helpers/runtime.py::_pi0_decode_action_response
            if isinstance(reply, dict) and reply.get("type") == "error":
                raise RuntimeError(f"Inference failed: {reply.get('message', reply)}")
            actions = np.asarray(reply, dtype=np.float32)
            t_done = time.perf_counter_ns()
            request_rows.append(
                {
                    "round": args.round,
                    "views": args.views,
                    "session_id": session_id,
                    "index": len(request_rows),
                    "warmup": warmup,
                    "cli_start": t0,
                    "cli_sent": t_sent,
                    "cli_received": t_received,
                    "cli_done": t_done,
                }
            )
            return (t_done - t0) / 1e6, actions

        for _ in range(args.warmup):
            one_round_trip(warmup=True)

        sampler = MemorySampler(args.gpu_index)
        if not args.no_memory_sampler:
            sampler.start()
        samples: list[float] = []
        shapes = set()
        for _ in range(args.iters):
            dt, actions = one_round_trip()
            samples.append(dt)
            shapes.add(actions.shape)
        if not args.no_memory_sampler:
            sampler.stop()
    finally:
        conn.close()

    issue_order = list(samples)
    samples.sort()
    result = {
        "all_ms_in_issue_order": [round(x, 3) for x in issue_order],
        "views": args.views,
        "iters": args.iters,
        "warmup": args.warmup,
        "num_inference_steps": args.num_inference_steps,
        "action_shapes": sorted(str(s) for s in shapes),
        "p50_ms": round(statistics.median(samples), 2),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 2),
        "min_ms": round(samples[0], 2),
        "max_ms": round(samples[-1], 2),
        "mean_ms": round(statistics.fmean(samples), 2),
        "device_peak_mib": sampler.peak_mib,
        "device_peak_gib": round(sampler.peak_mib / 1024.0, 3),
        "metadata": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(metadata).items()},
    }
    print(json.dumps(result, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        print("wrote", args.out)
    if args.requests_out:
        with open(args.requests_out, "a", encoding="utf-8") as fh:
            fh.writelines(json.dumps(row) + "\n" for row in request_rows)
    if args.save_actions:
        np.savez(args.save_actions, **{f"views{args.views}": actions})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
