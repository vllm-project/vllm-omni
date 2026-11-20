# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Export a standalone measurement figure; no dashboard/network dependency."""

import argparse
import json
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.performance.read_text())["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for backend, color in (("vllm-omni", "#2457a7"), ("official", "#dc752b")):
        subset = sorted(
            (row for row in rows if row["backend"] == backend and not row["sync"]), key=lambda row: row["concurrency"]
        )
        x = [row["concurrency"] for row in subset]
        for axis, metric, title in (
            (axes[0, 0], "latency_s", "E2E latency P50 (s)"),
            (axes[0, 1], "ttfp_s", "First PCM P50 (s)"),
            (axes[1, 0], "rtf", "Per-request RTF P50"),
        ):
            stats = [row["request_metrics"][metric] for row in subset]
            y = [row["p50"] for row in stats]
            lower = [row["p50"] - row["p50_ci95"][0] for row in stats]
            upper = [row["p50_ci95"][1] - row["p50"] for row in stats]
            axis.errorbar(x, y, yerr=[lower, upper], marker="o", color=color, label=backend, capsize=3)
            axis.set_title(title)
            axis.set_xticks([1, 4, 8])
            axis.grid(alpha=0.2)
        axes[1, 1].plot(
            x, [row["throughput"]["requests_per_s"] for row in subset], marker="o", color=color, label=backend
        )
    axes[1, 1].set_title("Throughput (requests/s)")
    axes[1, 1].set_xticks([1, 4, 8])
    axes[1, 1].grid(alpha=0.2)
    for axis in axes.flat:
        axis.set_xlabel("Concurrency")
        axis.legend()
    fig.suptitle(
        "ZONOS2: fixed 10-case corpus, 3 repeats, single A40 GPU2\n"
        "Warmups excluded; offline drivers; 95% request bootstrap CI"
    )
    fig.savefig(args.out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
