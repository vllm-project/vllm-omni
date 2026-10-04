# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare forced-aligner CPU output processing with a local Git revision.

Run from a development checkout. Model inference and transfers are excluded.
Example: python benchmarks/kernels/benchmark_forced_aligner.py --base-ref c8a1419
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import astuple
from functools import partial
from pathlib import Path
from types import ModuleType

import numpy as np

from vllm_omni.utils import forced_aligner, qwen3_force_align_processor

ROOT = Path(__file__).resolve().parents[2]


def load_baseline(name: str, filename: str, ref: str) -> ModuleType:
    path = f"vllm_omni/utils/{filename}.py"
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True)
    module = ModuleType(name)
    module.__file__ = str(ROOT / path)
    sys.modules[name] = module
    exec(compile(source, str(ROOT / path), "exec"), module.__dict__)
    return module


def measure(operations: dict[str, Callable], repeats: int, reverse: bool) -> dict:
    for operation in operations.values():
        for _ in range(3):
            operation()
    samples = {name: [] for name in operations}
    names = list(operations)
    for repeat in range(repeats):
        for name in names if (repeat + reverse) % 2 == 0 else names[::-1]:
            start = time.perf_counter_ns()
            operations[name]()
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    results = {}
    for name, operation in operations.items():
        tracemalloc.start()
        operation()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[name] = {
            "median_ms": statistics.median(samples[name]),
            "stdev_ms": statistics.pstdev(samples[name]),
            "samples_ms": samples[name],
            "temporary_peak_bytes": peak,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True, help="Local Git revision to compare.")
    parser.add_argument("--markers", nargs="+", type=int, default=[40, 156, 512, 1024, 2048])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--reverse", action="store_true", help="Reverse the first comparison order.")
    args = parser.parse_args()
    if args.repeats < 1 or any(count < 2 or count % 2 for count in args.markers):
        parser.error("Use positive repeats and even marker counts of at least two.")

    old_processor = load_baseline("aligner_old_processor", "qwen3_force_align_processor", args.base_ref)
    old_decoder = load_baseline("aligner_old_decoder", "forced_aligner", args.base_ref)
    old_decoder._processor = old_processor
    repair_only = load_baseline("aligner_repair_only", "forced_aligner", args.base_ref)
    repair_only._processor = qwen3_force_align_processor
    decoders = {
        "main": old_decoder._decode_timestamps,
        "repair_only": repair_only._decode_timestamps,
        "combined": forced_aligner._decode_timestamps,
    }
    print(json.dumps({"base_ref": args.base_ref, "python": sys.version, "numpy": np.__version__, "warmup": 3}))
    rng = np.random.default_rng(20260928)
    for count in args.markers:
        words = ["word"] * (count // 2)
        prefix = len(words) * 3 + 64
        positions = [prefix + 3 * (i // 2) + 1 + i % 2 for i in range(count)]
        logits = rng.random((prefix + len(words) * 3, 5000), dtype=np.float32)
        bins = np.arange(count) * 3 // 2 % 5000
        bins[20::50] = np.maximum(0, bins[20::50] - 12)
        logits[positions, bins] = 4.0
        operations = {
            name: partial(
                decode,
                logits=logits,
                words=words,
                timestamp_positions=positions,
                classify_num=5000,
                timestamp_segment_time_ms=80,
                audio_duration_ms=400_000,
            )
            for name, decode in decoders.items()
        }
        reference = [astuple(item) for item in operations["main"]()]
        for operation in operations.values():
            assert [astuple(item) for item in operation()] == reference
        print(
            json.dumps(
                {
                    "markers": count,
                    "logits_shape": logits.shape,
                    "results": measure(operations, args.repeats, args.reverse),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
