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
high-water mark sampled from nvidia-smi during the run. nvidia-smi reads the GPU
of the machine this client runs on, so it measures the server only when both
share that GPU. When no sample succeeds (no nvidia-smi, an NVML error, a remote
server's GPU), the peak is reported as null with the last sampling error, never
as 0.
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
    """Poll nvidia-smi at 5 Hz; report the high-water mark in MiB.

    ``peak_mib`` stays ``None`` until a sample succeeds, so a run in which every
    sample failed reports no measurement rather than 0. Failed samples are
    counted, and the last failure is kept to explain a missing peak.
    """

    SAMPLE_TIMEOUT_S = 5.0
    POLL_INTERVAL_S = 0.2

    def __init__(self, gpu_index: int = 0) -> None:
        super().__init__(daemon=True)
        self.gpu_index = gpu_index
        self.peak_mib: int | None = None
        self.samples_ok = 0
        self.samples_failed = 0
        self.last_error: str | None = None
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
                # check=True: a nonzero exit (an NVML error) is a failed sample.
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=self.SAMPLE_TIMEOUT_S, check=True)
                used_mib = int(out.stdout.strip().splitlines()[0])
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
                # Missing binary, nonzero exit or timeout, unparsable or empty output.
                self.samples_failed += 1
                stderr = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else ""
                self.last_error = f"{type(exc).__name__}: {exc}" + (f" ({stderr})" if stderr else "")
            else:
                self.samples_ok += 1
                self.peak_mib = used_mib if self.peak_mib is None else max(self.peak_mib, used_mib)
            self._stop_evt.wait(self.POLL_INTERVAL_S)

    def stop(self) -> None:
        self._stop_evt.set()
        # Longer than one sample's timeout, so a sample in flight can finish.
        self.join(timeout=2 * self.SAMPLE_TIMEOUT_S)


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
    args = ap.parse_args()

    import websockets.sync.client as ws_client
    from openpi_client import msgpack_numpy

    rng = np.random.default_rng(args.seed)
    packer = msgpack_numpy.Packer()
    session_id = str(uuid.uuid4())
    uri = f"ws://{args.host}:{args.port}{OPENPI_PATH}"

    sampler: MemorySampler | None = None
    conn = ws_client.connect(uri, compression=None, max_size=None, ping_interval=300, ping_timeout=3600)
    try:
        metadata = msgpack_numpy.unpackb(conn.recv())
        print("handshake:", json.dumps({k: str(v) for k, v in dict(metadata).items()}, indent=1))

        def one_round_trip() -> tuple[float, tuple]:
            payload = make_obs(views=args.views, prompt=args.prompt, session_id=session_id, rng=rng)
            if args.num_inference_steps is not None:
                payload["sampling_params"] = {"num_inference_steps": args.num_inference_steps}
            payload["endpoint"] = "infer"
            # The timed region is everything the robot pays for: pack, send,
            # server, receive, unpack. Starting after pack or stopping before
            # unpack measures the wire and the server, not the client.
            t0 = time.perf_counter()
            blob = packer.pack(payload)
            conn.send(blob)
            raw = conn.recv()
            if isinstance(raw, str):
                raise RuntimeError(f"Inference failed: {raw}")
            reply = msgpack_numpy.unpackb(raw)
            # Same decode contract as tests/helpers/runtime.py::_pi0_decode_action_response
            if isinstance(reply, dict) and reply.get("type") == "error":
                raise RuntimeError(f"Inference failed: {reply.get('message', reply)}")
            actions = np.asarray(reply, dtype=np.float32)
            dt = (time.perf_counter() - t0) * 1000.0
            return dt, actions.shape

        for _ in range(args.warmup):
            one_round_trip()

        sampler = MemorySampler(args.gpu_index)
        sampler.start()
        samples: list[float] = []
        shapes = set()
        for _ in range(args.iters):
            dt, shape = one_round_trip()
            samples.append(dt)
            shapes.add(shape)
    finally:
        # Also on a failed request, so the sampling thread never outlives the run.
        if sampler is not None:
            sampler.stop()
        conn.close()

    peak_mib = sampler.peak_mib
    memory_error = None if peak_mib is not None else (sampler.last_error or "no nvidia-smi sample was taken")
    if memory_error is not None:
        print(f"WARNING: device memory peak unavailable, no nvidia-smi sample succeeded: {memory_error}")
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
        # null when no nvidia-smi sample succeeded: unavailable, not 0.
        "device_peak_mib": peak_mib,
        "device_peak_gib": None if peak_mib is None else round(peak_mib / 1024.0, 3),
        "device_memory_samples": {"ok": sampler.samples_ok, "failed": sampler.samples_failed},
        "device_memory_error": memory_error,
        "metadata": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(metadata).items()},
    }
    print(json.dumps(result, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
