# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure base64 and same-host shared-memory video delivery in a consumer process.

Example:

    python benchmarks/diffusion/bench_video_output_sinks.py --frames 48
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pybase64 as base64

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _frames(count: int, height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((count, height, width, 3)) * 255).astype(np.uint8)


def _rss_mib() -> float:
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS not found")


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def _run_consumer(mode: str, payload_path: str, expected_digest: str) -> dict[str, object]:
    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--role",
            "consumer",
            "--mode",
            mode,
            "--payload",
            payload_path,
            "--expected",
            expected_digest,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    for line in process.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line.removeprefix("RESULT "))
    raise RuntimeError(f"consumer failed ({mode}):\n{process.stdout}\n{process.stderr}")


def _consumer(mode: str, payload_path: str, expected_digest: str) -> None:
    if mode == "shared_memory":
        from vllm_omni.entrypoints.openai.video_output_shm import borrowed_video_frames
    else:
        import av

    with open(payload_path) as payload_file:
        payload = json.load(payload_file)

    baseline = _rss_mib()
    if mode == "shared_memory":
        with borrowed_video_frames(payload["handle"]) as frames:
            lossless = _digest(frames) == expected_digest
            rss = _rss_mib()
    else:
        video_bytes = base64.b64decode(payload["b64_json"])
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output:
            output.write(video_bytes)
            output_path = output.name
        try:
            with av.open(output_path) as container:
                stream = container.streams.video[0]
                decoded = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(stream)])
        finally:
            os.unlink(output_path)
        lossless = _digest(decoded) == expected_digest
        rss = _rss_mib()

    print(
        "RESULT "
        + json.dumps(
            {
                "mode": mode,
                "baseline_mib": round(baseline, 1),
                "rss_mib": round(rss, 1),
                "delta_mib": round(rss - baseline, 1),
                "boundary_bytes": os.path.getsize(payload_path),
                "lossless": lossless,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["driver", "consumer"], default="driver")
    parser.add_argument("--mode", choices=["base64", "shared_memory"], default="base64")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--payload")
    parser.add_argument("--expected")
    args = parser.parse_args()

    if args.role == "consumer":
        _consumer(args.mode, args.payload, args.expected)
        return
    if args.rounds < 1:
        parser.error("--rounds must be positive")

    frames = _frames(args.frames, args.height, args.width)
    payload_mib = frames.nbytes / 1024**2
    print(f"video: {frames.shape} uint8 = {payload_mib:.1f} MiB\n")

    with tempfile.TemporaryDirectory() as directory:
        expected_digest = _digest(frames)
        rows = []
        for mode in ("base64", "shared_memory"):
            measurements = []
            for round_index in range(args.rounds):
                payload_path = os.path.join(directory, f"payload_{mode}_{round_index}.json")
                if mode == "shared_memory":
                    from multiprocessing import shared_memory

                    from vllm_omni.entrypoints.openai.video_output_shm import export_video_frames_to_shm

                    handle = export_video_frames_to_shm(frames, ttl_seconds=300)
                    payload = {"handle": handle.model_dump(mode="json")}
                else:
                    from vllm_omni.entrypoints.openai.video_api_utils import encode_video_base64

                    payload = {"b64_json": encode_video_base64(frames, fps=24)}
                with open(payload_path, "w") as output:
                    json.dump(payload, output)
                measurements.append(_run_consumer(mode, payload_path, expected_digest))
                if mode == "shared_memory":
                    try:
                        leaked = shared_memory.SharedMemory(name=handle.name)
                    except FileNotFoundError:
                        pass
                    else:
                        leaked.close()
                        leaked.unlink()
                        raise RuntimeError(f"shared-memory consumer leaked segment {handle.name!r}")

            deltas = [float(measurement["delta_mib"]) for measurement in measurements]
            boundaries = [int(measurement["boundary_bytes"]) for measurement in measurements]
            rows.append(
                {
                    "mode": mode,
                    "boundary_bytes": sum(boundaries) / len(boundaries),
                    "delta_mean_mib": sum(deltas) / len(deltas),
                    "delta_min_mib": min(deltas),
                    "delta_max_mib": max(deltas),
                    "lossless": all(bool(measurement["lossless"]) for measurement in measurements),
                }
            )

    print(f"{'sink':<15}{'boundary':>12}{'consumer RSS mean [range]':>31}{'lossless':>10}")
    for row in rows:
        boundary_mib = float(row["boundary_bytes"]) / 1024**2
        rss = (
            f"{float(row['delta_mean_mib']):.1f} "
            f"[{float(row['delta_min_mib']):.1f}, {float(row['delta_max_mib']):.1f}] MiB"
        )
        print(f"{str(row['mode']):<15}{boundary_mib:>10.2f} MiB{rss:>29}{str(row['lossless']):>10}")
    print(f"\npayload = {payload_mib:.1f} MiB")


if __name__ == "__main__":
    main()
