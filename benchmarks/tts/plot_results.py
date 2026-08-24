"""Plot universal TTS benchmark results.

Reads JSON files saved by ``bench_tts.py`` (via ``vllm bench serve --omni``)
and generates comparison bar charts grouped by task type.

Metrics plotted:
- AUDIO_TTFP  (mean audio time-to-first-packet, ms)
- E2EL        (mean end-to-end latency, ms)
- Audio RTF   (mean real-time factor)
- Audio throughput (audio-seconds / wall-second)

Quality metrics (WER / SIM / UTMOS) are printed in a table when present.

Usage::

    # Single run — one JSON per task, all in results/
    python benchmarks/tts/plot_results.py \\
        --results results/bench_tts_*.json \\
        --output results/tts_benchmark.png

    # Compare two runs (e.g. async_chunk on vs off)
    python benchmarks/tts/plot_results.py \\
        --results run_a/bench_tts_*.json \\
        --results run_b/bench_tts_*.json \\
        --labels "async_chunk_on" "async_chunk_off" \\
        --output results/comparison.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------


def load_run(paths: list[str]) -> list[dict]:
    """Load and merge all JSON files for one run into a flat list of records.

    Each record is expected to have at least ``_concurrency`` (int) and
    ``_task`` (str) keys injected by ``bench_tts.py``.  Records that come
    from a file that contains a list are flattened.
    """
    records: list[dict] = []
    for p in paths:
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            records.extend(raw)
        elif isinstance(raw, dict):
            records.append(raw)
    return records


def _get(record: dict, key: str) -> float:
    v = record.get(key, float("nan"))
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _bar_group(
    ax: plt.Axes,
    x: np.ndarray,
    data_per_label: dict[str, list[float]],
    width: float,
    colors: list[str],
    ylabel: str,
    title: str,
    concurrency_labels: list[str],
    fmt: str = ".1f",
) -> None:
    n = len(data_per_label)
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n) if n > 1 else [0.0]

    for i, (label, values) in enumerate(data_per_label.items()):
        plot_vals = [0.0 if math.isnan(v) else v for v in values]
        bar = ax.bar(x + offsets[i], plot_vals, width, label=label, color=colors[i % len(colors)], alpha=0.85)
        max_val = max((v for v in values if not math.isnan(v)), default=1.0)
        for rect, val in zip(bar, values):
            if not math.isnan(val) and val > 0:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + max_val * 0.02,
                    f"{val:{fmt}}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_xlabel("Concurrency", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(concurrency_labels)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#FFC107", "#9C27B0"]


# ---------------------------------------------------------------------------
# Comparison plot (multiple labels / runs)
# ---------------------------------------------------------------------------


def plot_comparison(
    all_runs: list[list[dict]],
    labels: list[str],
    output_path: str,
    task_filter: str | None = None,
    title_prefix: str = "TTS",
) -> None:
    """One 2×2 subplot per task found in the data."""
    # Determine tasks to plot
    tasks: list[str] = []
    for run in all_runs:
        for r in run:
            t = r.get("_task", "unknown")
            if t not in tasks:
                tasks.append(t)
    if task_filter:
        tasks = [t for t in tasks if t == task_filter]

    n_tasks = len(tasks)
    if n_tasks == 0:
        print("[plot_results] No tasks found in data.")
        return

    fig, axes_grid = plt.subplots(n_tasks, 4, figsize=(18, 4.5 * n_tasks))
    fig.suptitle(f"{title_prefix} Benchmark", fontsize=15, fontweight="bold")

    # Ensure axes_grid is always 2D
    if n_tasks == 1:
        axes_grid = [axes_grid]

    for row_idx, task in enumerate(tasks):
        # Collect concurrencies across all runs for this task
        all_concs: set[int] = set()
        for run in all_runs:
            for r in run:
                if r.get("_task") == task:
                    c = r.get("_concurrency")
                    if c is not None:
                        all_concs.add(int(c))
        concurrencies = sorted(all_concs)
        x = np.arange(len(concurrencies))
        conc_labels = [str(c) for c in concurrencies]

        def _series(run: list[dict], metric_key: str) -> list[float]:
            conc_map = {int(r["_concurrency"]): r for r in run if r.get("_task") == task and "_concurrency" in r}
            return [_get(conc_map.get(c, {}), metric_key) for c in concurrencies]

        metrics = [
            ("mean_audio_ttfp_ms", "TTFP (ms)", "Time-to-First-Packet", ".0f"),
            ("mean_e2el_ms", "E2E Latency (ms)", "End-to-End Latency", ".0f"),
            ("mean_audio_rtf", "RTF", "Real-Time Factor (RTF)", ".3f"),
            ("audio_throughput", "audio-s / wall-s", "Audio Throughput", ".2f"),
        ]

        axes_row = axes_grid[row_idx]
        for col_idx, (key, ylabel, subtitle, fmt) in enumerate(metrics):
            data_per_label = {lbl: _series(run, key) for lbl, run in zip(labels, all_runs)}
            _bar_group(
                axes_row[col_idx],
                x,
                data_per_label,
                width=0.3 if len(labels) > 1 else 0.5,
                colors=COLORS,
                ylabel=ylabel,
                title=f"{task} — {subtitle}",
                concurrency_labels=conc_labels,
                fmt=fmt,
            )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Markdown comparison table
# ---------------------------------------------------------------------------


def print_comparison_table(all_runs: list[list[dict]], labels: list[str]) -> None:
    tasks: list[str] = []
    for run in all_runs:
        for r in run:
            t = r.get("_task", "unknown")
            if t not in tasks:
                tasks.append(t)

    perf_metrics = [
        ("TTFP (ms)", "mean_audio_ttfp_ms", ".1f"),
        ("E2E (ms)", "mean_e2el_ms", ".1f"),
        ("RTF", "mean_audio_rtf", ".3f"),
        ("Throughput (a-s/s)", "audio_throughput", ".2f"),
    ]
    quality_metrics = [
        ("WER (%)", "seed_tts_mean_wer", ".1f"),
        ("SIM", "seed_tts_mean_sim", ".3f"),
        ("UTMOS", "seed_tts_mean_utmos", ".2f"),
    ]

    for task in tasks:
        all_concs: set[int] = set()
        for run in all_runs:
            for r in run:
                if r.get("_task") == task:
                    c = r.get("_concurrency")
                    if c is not None:
                        all_concs.add(int(c))
        concurrencies = sorted(all_concs)

        print(f"\n## {task}\n")
        col_header = "| Metric | Concurrency |" + "".join(f" {lbl} |" for lbl in labels)
        sep = "| --- | --- |" + " --- |" * len(labels)
        print(col_header)
        print(sep)

        for metric, key, fmt in perf_metrics + quality_metrics:
            for c in concurrencies:
                row = f"| {metric} | {c} |"
                for run in all_runs:
                    conc_map = {
                        int(r["_concurrency"]): r for r in run if r.get("_task") == task and "_concurrency" in r
                    }
                    val = _get(conc_map.get(c, {}), key)
                    row += f" {val:{fmt}} |" if not math.isnan(val) else " n/a |"
                print(row)

        # Improvement column (2-run comparison only)
        if len(all_runs) == 2:
            print(f"\n### Improvement ({labels[0]} vs {labels[1]})\n")
            print("| Metric | Concurrency | Change |")
            print("| --- | --- | --- |")
            for metric, key, _ in perf_metrics:
                for c in concurrencies:
                    conc_map0 = {
                        int(r["_concurrency"]): r for r in all_runs[0] if r.get("_task") == task and "_concurrency" in r
                    }
                    conc_map1 = {
                        int(r["_concurrency"]): r for r in all_runs[1] if r.get("_task") == task and "_concurrency" in r
                    }
                    v0 = _get(conc_map0.get(c, {}), key)
                    v1 = _get(conc_map1.get(c, {}), key)
                    if not math.isnan(v0) and not math.isnan(v1) and v1 > 0:
                        pct = (v1 - v0) / v1 * 100
                        print(f"| {metric} | {c} | {pct:+.1f}% |")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        action="append",
        required=True,
        metavar="FILE",
        help="JSON result file(s) for one run. Repeat --results for multiple runs to compare.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Label for each --results group (must match the number of --results groups).",
    )
    parser.add_argument("--output", type=str, default="results/tts_benchmark.png", help="Output image path.")
    parser.add_argument("--title", type=str, default="TTS", help="Title prefix for the plot.")
    parser.add_argument("--task", type=str, default=None, help="Filter to a single task (e.g. voice_clone).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # args.results is a list-of-lists due to action="append"
    all_runs: list[list[dict]] = [load_run(group) for group in args.results]
    n_runs = len(all_runs)

    labels: list[str]
    if args.labels:
        if len(args.labels) != n_runs:
            raise SystemExit(f"--labels count ({len(args.labels)}) must match --results groups ({n_runs})")
        labels = args.labels
    else:
        labels = [f"run{i + 1}" for i in range(n_runs)]

    print_comparison_table(all_runs, labels)
    plot_comparison(all_runs, labels, args.output, task_filter=args.task, title_prefix=args.title)


if __name__ == "__main__":
    main()
