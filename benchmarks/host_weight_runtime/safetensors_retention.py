# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Linux CPU diagnostic: repeated safetensors materialization versus view reuse."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import tempfile
import time
import weakref
from pathlib import Path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def sample(phase: str, iterations: int, loop_seconds: float) -> dict[str, object]:
    gc.collect()
    counters = {}
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2] == "kB":
            counters[fields[0].removesuffix(":")] = int(fields[1]) * 1024
    return {
        "phase": phase,
        "iterations": iterations,
        "loop_seconds": loop_seconds,
        "rss_bytes": counters["Rss"],
        "private_bytes": counters["Private_Clean"] + counters["Private_Dirty"],
        "anonymous_bytes": counters["Anonymous"],
        "fd_count": len(tuple(Path("/proc/self/fd").iterdir())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("get_tensor", "reuse"), required=True)
    parser.add_argument("--iterations", type=positive_int, default=1_000_000)
    parser.add_argument("--sample-every", type=positive_int, default=100_000)
    parser.add_argument("--warmup", type=positive_int, default=1000)
    args = parser.parse_args()
    if not Path("/proc/self/smaps_rollup").is_file():
        parser.error("requires Linux /proc/self/smaps_rollup")

    preparation_start = time.monotonic()
    import safetensors
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    samples = [sample("before_file", 0, 0.0)]
    with tempfile.TemporaryDirectory(prefix="safetensors-retention-") as directory:
        path = Path(directory) / "synthetic.safetensors"
        save_file({"weight": torch.arange(16, dtype=torch.float32)}, str(path))
        with safe_open(path, framework="pt", device="cpu") as reader:
            cached = reader.get_tensor("weight")
            assert torch.equal(cached, torch.arange(16, dtype=torch.float32))
            for _ in range(args.warmup):
                tensor = reader.get_tensor("weight") if args.mode == "get_tensor" else cached
                del tensor
            samples.append(sample("baseline", 0, 0.0))
            preparation_seconds = time.monotonic() - preparation_start
            completed = 0
            loop_seconds = 0.0
            while completed < args.iterations:
                count = min(args.sample_every, args.iterations - completed)
                start = time.monotonic()
                for _ in range(count):
                    tensor = reader.get_tensor("weight") if args.mode == "get_tensor" else cached
                last_tensor = weakref.ref(tensor)
                del tensor
                loop_seconds += time.monotonic() - start
                completed += count
                snapshot = sample("loop", completed, loop_seconds)
                snapshot["last_tensor_alive"] = last_tensor() is not None
                if args.mode == "get_tensor":
                    assert last_tensor() is None, "diagnostic retained a transient tensor"
                samples.append(snapshot)
            del cached
        del reader
        samples.append(sample("closed", completed, loop_seconds))
        assert last_tensor() is None, "last tensor survived reader/view teardown"

    print(
        json.dumps(
            {
                "schema_version": 1,
                "arguments": vars(args),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "safetensors": safetensors.__version__,
                "kernel": platform.release(),
                "hostname": platform.node(),
                "pid": os.getpid(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "torch_threads": torch.get_num_threads(),
                "pythonmalloc": os.environ.get("PYTHONMALLOC", "default"),
                "preparation_seconds": preparation_seconds,
                "samples": samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
