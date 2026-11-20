# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One-GPU job controller with idle guard, sampled memory and owned cleanup."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("Supply a command after --")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain exactly the requested single GPU")

    def query(fields, apps=False):
        option = "--query-compute-apps=" if apps else "--query-gpu="
        return subprocess.check_output(
            ["nvidia-smi", "-i", args.gpu, option + fields, "--format=csv,noheader,nounits"], text=True
        ).strip()

    used, util = map(int, query("memory.used,utilization.gpu").split(","))
    if used > 200 or util > 5 or query("pid", True):
        raise RuntimeError("Requested GPU is busy; wait for it to be idle")
    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        ZONOS2_TTS_NORM="0",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        ZONOS2_BENCH_RECORD_DIR=str(args.out / "record"),
        HF_MODULES_CACHE=str(args.out.parent / "hf-modules"),
        TORCHINDUCTOR_CACHE_DIR=str(args.out / "inductor-cache"),
        TRITON_CACHE_DIR=str(args.out / "triton-cache"),
    )
    if args.profile:
        env["ZONOS2_BENCH_PROFILE"] = "1"
    stop = threading.Event()
    measurements = []
    process = None

    def monitor():
        while not stop.is_set():
            memory, load = map(int, query("memory.used,utilization.gpu").split(","))
            pids = query("pid", True).splitlines()
            unexpected: list[int] = []
            for pid in pids:
                if process is None:
                    continue
                try:
                    instance = psutil.Process(int(pid))
                    if instance.pid != process.pid and process.pid not in [parent.pid for parent in instance.parents()]:
                        unexpected.append(int(pid))
                except psutil.NoSuchProcess:
                    pass
            measurements.append(
                {
                    "time": time.perf_counter(),
                    "memory_mib": memory,
                    "gpu_util_percent": load,
                    "compute_pids": pids,
                    "unexpected_pids": unexpected,
                }
            )
            stop.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    with (args.out / "run.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        thread.start()
        try:
            status = process.wait(timeout=args.timeout)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            stop.set()
            thread.join()
    (args.out / "gpu_samples.json").write_text(json.dumps(measurements))
    contended = any(row["unexpected_pids"] for row in measurements)
    (args.out / "controller.json").write_text(
        json.dumps(
            {"command": command, "exit_code": status, "gpu": args.gpu, "idle_before": True, "contended": contended},
            indent=2,
        )
    )
    if contended:
        raise RuntimeError("GPU became contended; exclude this job from the comparison")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
