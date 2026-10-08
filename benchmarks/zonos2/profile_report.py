# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Count actual kernels and busy intervals; do not double-count CPU op GPU attribution."""

import argparse
import json
from pathlib import Path


def union_us(intervals):
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for a, b in ordered[1:]:
        if a <= end:
            end = max(end, b)
        else:
            total += end - start
            start, end = a, b
    return total + end - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for role in ("ar", "dac"):
        events = json.loads((args.record / f"{role}.trace.json").read_text())["traceEvents"]
        kernels = [row for row in events if row.get("cat") == "kernel" and row.get("ph") == "X"]
        intervals = [(row["ts"], row["ts"] + row["dur"]) for row in kernels]
        window = max(b for a, b in intervals) - min(a for a, b in intervals)
        busy = union_us(intervals)
        stats = json.loads((args.record / f"{role}.profile.json").read_text())
        top_cpu = sorted(stats, key=lambda row: row["self_cpu_us"], reverse=True)[:15]
        top_device = sorted(stats, key=lambda row: row["self_device_us"], reverse=True)[:15]
        result[role] = {
            "kernel_count": len(kernels),
            "kernel_busy_ms": busy / 1000,
            "kernel_window_ms": window / 1000,
            "kernel_busy_fraction": busy / window,
            "sum_kernel_ms": sum(row["dur"] for row in kernels) / 1000,
            "top_cpu_ops": top_cpu,
            "top_device_attribution_ops": top_device,
        }
    result["interpretation"] = {
        "AR": "Many small launches and host dispatch dominate; AR runtime graph needs separate integration.",
        "DAC": "Component A/B measures decoder contribution; it does not replace AR optimization.",
        "guard": "AR runtime graph is explicitly unsupported by existing request-state guard; it is not bypassed.",
        "scope": "32 warmed AR forwards/four DAC calls; trace excluded from performance runs.",
    }
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
