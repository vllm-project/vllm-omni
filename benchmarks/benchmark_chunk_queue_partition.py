# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measure list-backed chunk queue processing without a model or GPU."""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import platform
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--queue-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 100, 128])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=3)
    return parser.parse_args()


def cpu_model() -> str:
    with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
        for line in cpuinfo:
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.source_root))
    request_status = importlib.import_module("vllm.v1.request").RequestStatus
    coordinator_type = importlib.import_module(
        "vllm_omni.core.sched.omni_scheduling_coordinator"
    ).OmniSchedulingCoordinator

    print(f"python={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"cpu={cpu_model()}")
    print(f"cpus={os.cpu_count()}")
    print(f"warmups={args.warmups} samples={args.samples} iterations={args.iterations}")
    print("size,mean_ms,stdev_ms,min_ms,max_ms")

    for size in args.queue_sizes:

        def prepare() -> tuple[list[SimpleNamespace], object]:
            requests = [
                SimpleNamespace(
                    request_id=f"request-{index}",
                    external_req_id=f"request-{index}",
                    status=request_status.RUNNING,
                )
                for index in range(size)
            ]
            return (
                list(requests),
                coordinator_type(
                    scheduler_max_num_seqs=size,
                    stage_id=1,
                    async_chunk=True,
                ),
            )

        def run(running: list[SimpleNamespace], coordinator: object) -> None:
            coordinator._process_chunk_queue(
                running,
                deque(),
                request_status.RUNNING,
                set(),
            )

        for _ in range(args.warmups):
            run(*prepare())

        samples_ns: list[float] = []
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(args.samples):
                prepared_runs = [prepare() for _ in range(args.iterations)]
                started = time.perf_counter_ns()
                for prepared_run in prepared_runs:
                    run(*prepared_run)
                samples_ns.append((time.perf_counter_ns() - started) / args.iterations)
        finally:
            if gc_was_enabled:
                gc.enable()

        samples_ms = [sample / 1_000_000 for sample in samples_ns]
        stdev_ms = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
        print(f"{size},{statistics.mean(samples_ms):.4f},{stdev_ms:.4f},{min(samples_ms):.4f},{max(samples_ms):.4f}")


if __name__ == "__main__":
    main()
