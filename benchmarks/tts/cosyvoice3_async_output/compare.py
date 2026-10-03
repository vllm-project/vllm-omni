# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import gzip
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


def profile_summary(directory):
    records = []
    for path in directory.rglob("*.json*"):
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as handle:
                data = json.load(handle)
        else:
            data = json.loads(path.read_text())
        events = data.get("traceEvents", [])
        cpu_scopes = [event for event in events if event.get("ph") == "X" and event.get("cat") == "user_annotation"]
        counts = Counter(event.get("name") for event in cpu_scopes)
        starts = sorted(event["ts"] for event in cpu_scopes if event.get("name") == "gpu_model_runner: bookkeep")
        intervals = [(end - start) / 1000 for start, end in zip(starts[1:], starts[2:])]
        steady = [value for value in intervals if value < 15]
        records.append(
            {
                "file": str(path),
                "bookkeep_events": len(starts),
                "mean_step_interval_ms": statistics.mean(intervals) if intervals else None,
                "median_step_interval_ms": statistics.median(intervals) if intervals else None,
                "steady_step_interval_ms": statistics.mean(steady) if steady else None,
                "steady_intervals": len(steady),
                "total_intervals": len(intervals),
                "async_snapshot_events": counts["omni_async_output:snapshot_cpu_payload"],
                "background_build_events": counts["omni_async_output:get_output/build_model_runner_output"],
                "output_builder_events": counts["omni_output_builder:total"],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("patched")
    parser.add_argument("--results-dir", type=Path, default=Path("/validation/results"))
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    root = args.results_dir.resolve()
    runs = {}
    for label in [args.baseline, args.patched]:
        data = json.loads((root / label / "metrics.json").read_text())
        rows = data["requests"]
        assert rows and all(len(row["chunk_samples"]) > 1 for row in rows), f"{label}: rerun with streaming output"
        concurrency = data["concurrency"]
        batch_seconds = sum(row["batch_e2e_ms"] for row in rows if row["index"] % concurrency == 0) / 1000
        if all(row.get("ar_itls_ms") for row in rows):
            # Current main returns native timing arrays in output.metrics.
            # Reading them avoids enabling DEBUG tables on the request path.
            itls = [(sum(row["ar_itls_ms"]) / len(row["ar_itls_ms"]), len(row["ar_itls_ms"])) for row in rows]
            timing_source = "native_output_metrics"
        else:
            log = (root / f"{label}.log").read_text(errors="replace")
            # The second BENCHMARK line begins the separate profiler process.
            unprofiled = log.split(f"BENCHMARK {label}")[1]
            itls = [
                (float(value.replace(",", "")), int(count))
                for value, count in re.findall(r"\| vllm_itls_ms\s*\|\s*([\d,.]+) \(n=(\d+)\)", unprofiled)
            ]
            itls = itls[args.warmups :]
            timing_source = "legacy_log_tables"
        assert len(itls) == len(rows), (label, "missing per-request AR timing", len(itls), len(rows))
        runs[label] = {
            "concurrency": concurrency,
            "requests": len(rows),
            "median_ttfa_ms": statistics.median(row["ttfa_ms"] for row in rows),
            "mean_last_audio_ms": statistics.mean(row["last_audio_ms"] for row in rows),
            "audio_seconds_per_second": sum(row["duration_s"] for row in rows) / batch_seconds,
            "unprofiled_ar_itl_ms": sum(value * count for value, count in itls) / sum(count for _, count in itls),
            "ar_token_intervals": sum(count for _, count in itls),
            "ar_timing_source": timing_source,
            "profiles": profile_summary(root / label / "traces"),
        }
    comparisons = []
    assert len(list((root / args.baseline).glob("audio_*.npy"))) == runs[args.baseline]["requests"]
    for path in sorted((root / args.baseline).glob("audio_*.npy")):
        before = np.load(path)
        after = np.load(root / args.patched / path.name)
        same_shape = before.shape == after.shape
        comparisons.append(
            {
                "audio": path.name,
                "baseline_samples": int(before.size),
                "patched_samples": int(after.size),
                "bitwise_equal": bool(np.array_equal(before, after)),
                "max_absolute_error": float(np.max(np.abs(before - after))) if same_shape else None,
            }
        )
    report = {"runs": runs, "audio_comparison": comparisons}
    (root / f"compare-{args.baseline}-{args.patched}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
