"""Plot TTFP comparison: async_chunk off vs on.

Generates a bar chart with improvement arrows, matching the Qwen3-Omni
async_chunk benchmark figure style.

Usage:
    python plot_async_chunk.py \
        --off results/bench_async_chunk_off_*.json \
        --on results/bench_async_chunk_on_*.json \
        --output results/qwen3_tts_async_chunk_ttfp.png

    # Also supports E2E and RTF metrics:
    python plot_async_chunk.py \
        --off results/bench_async_chunk_off_*.json \
        --on results/bench_async_chunk_on_*.json \
        --metric e2e \
        --output results/qwen3_tts_async_chunk_e2e.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRIC_CONFIG = {
    "ttfp": {
        "key": "mean_ttfp_ms",
        "ylabel": "TTFP (s)",
        "title": "TTFP (Time to First Audio Packet) - Qwen3-TTS, by concurrency",
        "to_seconds": True,
    },
    "e2e": {
        "key": "mean_e2e_ms",
        "ylabel": "E2E (s)",
        "title": "E2E Latency - Qwen3-TTS, by concurrency",
        "to_seconds": True,
    },
    "rtf": {
        "key": "mean_rtf",
        "ylabel": "RTF",
        "title": "Real-Time Factor - Qwen3-TTS, by concurrency",
        "to_seconds": False,
    },
}


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def plot_ttfp_comparison(
    off_results: list[dict],
    on_results: list[dict],
    metric: str,
    output_path: str,
    title_override: str | None = None,
):
    cfg = METRIC_CONFIG[metric]
    key = cfg["key"]
    to_seconds = cfg["to_seconds"]

    off_map = {r["concurrency"]: r for r in off_results}
    on_map = {r["concurrency"]: r for r in on_results}
    concurrencies = sorted(set(off_map.keys()) & set(on_map.keys()))

    off_vals = []
    on_vals = []
    for c in concurrencies:
        v_off = off_map[c][key]
        v_on = on_map[c][key]
        if to_seconds:
            v_off /= 1000.0
            v_on /= 1000.0
        off_vals.append(v_off)
        on_vals.append(v_on)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(concurrencies))
    width = 0.3

    ax.bar(x - width / 2, off_vals, width, label="async_chunk off", color="#87CEEB", edgecolor="none")
    ax.bar(x + width / 2, on_vals, width, label="async_chunk on", color="#FFF8DC", edgecolor="#DDD8B8")

    # Draw improvement arrows and labels
    for i in range(len(concurrencies)):
        v_off = off_vals[i]
        v_on = on_vals[i]
        if v_on > 0:
            improvement = v_off / v_on
        else:
            improvement = float("inf")

        # Arrow from top of off-bar to top of on-bar
        arrow_start_x = x[i] - width / 2
        arrow_start_y = v_off * 0.95
        arrow_end_x = x[i] + width / 2
        arrow_end_y = v_on * 1.05

        ax.annotate(
            "",
            xy=(arrow_end_x, arrow_end_y),
            xytext=(arrow_start_x, arrow_start_y),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        )

        # Improvement label
        label_x = (arrow_start_x + arrow_end_x) / 2
        label_y = arrow_start_y + (v_off - v_on) * 0.15
        ax.text(
            label_x,
            label_y,
            f"{improvement:.1f}x improvement",
            ha="center",
            va="bottom",
            fontsize=10,
            color="red",
            fontweight="bold",
        )

    title = title_override or cfg["title"]
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(cfg["ylabel"], fontsize=12)
    ax.set_xlabel("Max concurrency", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in concurrencies])
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_all_metrics(off_results: list[dict], on_results: list[dict], output_path: str):
    """Generate a 1x3 subplot with TTFP, E2E, and RTF comparisons."""
    off_map = {r["concurrency"]: r for r in off_results}
    on_map = {r["concurrency"]: r for r in on_results}
    concurrencies = sorted(set(off_map.keys()) & set(on_map.keys()))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Qwen3-TTS: async_chunk on vs off", fontsize=15, fontweight="bold")

    for ax, metric in zip(axes, ["ttfp", "e2e", "rtf"]):
        cfg = METRIC_CONFIG[metric]
        key = cfg["key"]
        to_seconds = cfg["to_seconds"]

        off_vals = []
        on_vals = []
        for c in concurrencies:
            v_off = off_map[c][key]
            v_on = on_map[c][key]
            if to_seconds:
                v_off /= 1000.0
                v_on /= 1000.0
            off_vals.append(v_off)
            on_vals.append(v_on)

        x = np.arange(len(concurrencies))
        width = 0.3
        ax.bar(x - width / 2, off_vals, width, label="async_chunk off", color="#87CEEB")
        ax.bar(x + width / 2, on_vals, width, label="async_chunk on", color="#FFF8DC", edgecolor="#DDD8B8")

        for i in range(len(concurrencies)):
            if on_vals[i] > 0:
                improvement = off_vals[i] / on_vals[i]
                ax.annotate(
                    "",
                    xy=(x[i] + width / 2, on_vals[i] * 1.05),
                    xytext=(x[i] - width / 2, off_vals[i] * 0.95),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                )
                label_y = off_vals[i] * 0.85
                ax.text(x[i], label_y, f"{improvement:.1f}x", ha="center", fontsize=10, color="red", fontweight="bold")

        ax.set_title(cfg["title"].split(" - ")[0], fontsize=12, fontweight="bold")
        ax.set_ylabel(cfg["ylabel"], fontsize=11)
        ax.set_xlabel("Max concurrency", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in concurrencies])
        if metric != "rtf":
            ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def print_table(off_results: list[dict], on_results: list[dict]):
    off_map = {r["concurrency"]: r for r in off_results}
    on_map = {r["concurrency"]: r for r in on_results}
    concurrencies = sorted(set(off_map.keys()) & set(on_map.keys()))

    print("\n## Benchmark Results: async_chunk off vs on\n")
    print("| Metric | Concurrency | async_chunk off | async_chunk on | Improvement |")
    print("| --- | --- | --- | --- | --- |")

    for name, key, fmt in [
        ("TTFP (ms)", "mean_ttfp_ms", ".1f"),
        ("E2E (ms)", "mean_e2e_ms", ".1f"),
        ("RTF", "mean_rtf", ".3f"),
        ("Throughput", "audio_throughput", ".2f"),
    ]:
        for c in concurrencies:
            v_off = off_map[c].get(key, 0)
            v_on = on_map[c].get(key, 0)
            if v_on > 0 and key != "audio_throughput":
                ratio = f"{v_off / v_on:.1f}x"
            elif v_off > 0 and key == "audio_throughput":
                ratio = f"{v_on / v_off:.1f}x"
            else:
                ratio = "N/A"
            print(f"| {name} | {c} | {v_off:{fmt}} | {v_on:{fmt}} | {ratio} |")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot async_chunk comparison for Qwen3-TTS")
    parser.add_argument("--off", type=str, required=True, help="JSON results for async_chunk off")
    parser.add_argument("--on", type=str, required=True, help="JSON results for async_chunk on")
    parser.add_argument("--metric", type=str, default="ttfp", choices=["ttfp", "e2e", "rtf", "all"])
    parser.add_argument("--output", type=str, default="results/qwen3_tts_async_chunk.png")
    parser.add_argument("--title", type=str, default=None, help="Custom title override")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    off_results = load_results(args.off)
    on_results = load_results(args.on)

    print_table(off_results, on_results)

    if args.metric == "all":
        plot_all_metrics(off_results, on_results, args.output)
    else:
        plot_ttfp_comparison(off_results, on_results, args.metric, args.output, args.title)
