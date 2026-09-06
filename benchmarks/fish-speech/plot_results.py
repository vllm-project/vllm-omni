"""Plot Fish Speech benchmark results.

Generates comparison bar charts for:
- TTFP (Time-to-First-Packet)
- E2E latency
- RTF (Real-Time Factor)

Usage:
    # Compare two configs
    python benchmarks/fish-speech/plot_results.py \
        --results \
            benchmarks/fish-speech/results/bench_vllm_omni_*.json \
            benchmarks/fish-speech/results/bench_sglang_omni_*.json \
        --labels "vllm-omni" "sglang-omni" \
        --output benchmarks/fish-speech/results/comparison.png

    # Single config
    python benchmarks/fish-speech/plot_results.py \
        --results benchmarks/fish-speech/results/bench_vllm_omni_*.json \
        --labels "vllm-omni" \
        --output benchmarks/fish-speech/results/vllm_omni.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(result_files: list[str]) -> list[list[dict]]:
    """Load benchmark results from JSON files."""
    all_results = []
    for file_path in result_files:
        with open(file_path) as fh:
            data = json.load(fh)
        all_results.append(data)
    return all_results


def plot_comparison(
    all_results: list[list[dict]],
    labels: list[str],
    output_path: str,
    title_prefix: str = "Fish Speech S2 Pro",
) -> None:
    """Generate comparison bar charts."""
    n_configs = len(all_results)

    all_concurrencies = [set(result["concurrency"] for result in results) for results in all_results]
    concurrencies = sorted(set.union(*all_concurrencies))

    ttfp_data = {label: [] for label in labels}
    e2e_data = {label: [] for label in labels}
    rtf_data = {label: [] for label in labels}
    throughput_data = {label: [] for label in labels}

    for results, label in zip(all_results, labels):
        conc_map = {result["concurrency"]: result for result in results}
        for concurrency in concurrencies:
            result = conc_map.get(concurrency)
            ttfp_data[label].append(result["mean_ttfp_ms"] if result else None)
            e2e_data[label].append(result["mean_e2e_ms"] if result else None)
            rtf_data[label].append(result["mean_rtf"] if result else None)
            throughput_data[label].append(result["audio_throughput"] if result else None)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title_prefix} Performance Benchmark", fontsize=16, fontweight="bold")

    x = np.arange(len(concurrencies))
    width = 0.35 if n_configs == 2 else 0.5
    if n_configs > 1:
        offsets = np.linspace(-width / 2 * (n_configs - 1), width / 2 * (n_configs - 1), n_configs)
    else:
        offsets = [0]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FFC107"]

    def plot_metric(ax, data_dict, ylabel, title, fmt=".1f"):
        for index, (label, values) in enumerate(data_dict.items()):
            plot_values = [value if value is not None else 0 for value in values]
            color = colors[index % len(colors)]
            bars = ax.bar(x + offsets[index], plot_values, width, label=label, color=color, alpha=0.85)
            max_val = max((value for value in values if value is not None), default=1)
            for rect, value in zip(bars, values):
                if value is not None and value > 0:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + max_val * 0.02,
                        f"{value:{fmt}}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )
        ax.set_xlabel("Concurrency", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(concurrency) for concurrency in concurrencies])
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    plot_metric(axes[0, 0], ttfp_data, "TTFP (ms)", "Time to First Audio Packet (TTFP)")
    plot_metric(axes[0, 1], e2e_data, "E2E Latency (ms)", "End-to-End Latency (E2E)")
    plot_metric(axes[1, 0], rtf_data, "RTF", "Real-Time Factor (RTF)", fmt=".3f")
    plot_metric(axes[1, 1], throughput_data, "Audio-sec / Wall-sec", "Audio Throughput", fmt=".2f")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_single_summary(
    results: list[dict],
    label: str,
    output_path: str,
    title_prefix: str = "Fish Speech S2 Pro",
) -> None:
    """Generate a single-config summary with percentile breakdown."""
    concurrencies = [result["concurrency"] for result in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{title_prefix} Benchmark - {label}", fontsize=15, fontweight="bold")

    x = np.arange(len(concurrencies))
    width = 0.2

    ax = axes[0]
    means = [result["mean_ttfp_ms"] for result in results]
    medians = [result["median_ttfp_ms"] for result in results]
    p90s = [result["p90_ttfp_ms"] for result in results]
    p99s = [result["p99_ttfp_ms"] for result in results]
    ax.bar(x - 1.5 * width, means, width, label="mean", color="#2196F3")
    ax.bar(x - 0.5 * width, medians, width, label="median", color="#4CAF50")
    ax.bar(x + 0.5 * width, p90s, width, label="p90", color="#FF9800")
    ax.bar(x + 1.5 * width, p99s, width, label="p99", color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels([str(concurrency) for concurrency in concurrencies])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("TTFP (ms)")
    ax.set_title("Time to First Audio Packet")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    means = [result["mean_e2e_ms"] for result in results]
    medians = [result["median_e2e_ms"] for result in results]
    p90s = [result["p90_e2e_ms"] for result in results]
    p99s = [result["p99_e2e_ms"] for result in results]
    ax.bar(x - 1.5 * width, means, width, label="mean", color="#2196F3")
    ax.bar(x - 0.5 * width, medians, width, label="median", color="#4CAF50")
    ax.bar(x + 0.5 * width, p90s, width, label="p90", color="#FF9800")
    ax.bar(x + 1.5 * width, p99s, width, label="p99", color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels([str(concurrency) for concurrency in concurrencies])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("E2E Latency (ms)")
    ax.set_title("End-to-End Latency")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    means = [result["mean_rtf"] for result in results]
    medians = [result["median_rtf"] for result in results]
    ax.bar(x - 0.15, means, 0.3, label="mean", color="#2196F3")
    ax.bar(x + 0.15, medians, 0.3, label="median", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels([str(concurrency) for concurrency in concurrencies])
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("RTF")
    ax.set_title("Real-Time Factor")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def print_comparison_table(all_results: list[list[dict]], labels: list[str]) -> None:
    """Print a markdown-formatted comparison table."""
    all_concurrencies = [set(result["concurrency"] for result in results) for results in all_results]
    concurrencies = sorted(set.union(*all_concurrencies))

    print("\n## Benchmark Results\n")
    header = "| Metric | Concurrency |"
    separator = "| --- | --- |"
    for label in labels:
        header += f" {label} |"
        separator += " --- |"
    print(header)
    print(separator)

    for metric, key, fmt in [
        ("TTFP (ms)", "mean_ttfp_ms", ".1f"),
        ("E2E (ms)", "mean_e2e_ms", ".1f"),
        ("RTF", "mean_rtf", ".3f"),
        ("Throughput (audio-s/s)", "audio_throughput", ".2f"),
    ]:
        for concurrency in concurrencies:
            row = f"| {metric} | {concurrency} |"
            for results in all_results:
                conc_map = {result["concurrency"]: result for result in results}
                value = conc_map.get(concurrency, {}).get(key, 0)
                row += f" {value:{fmt}} |"
            print(row)

    if len(all_results) == 2:
        print(f"\n## Improvement ({labels[0]} vs {labels[1]})\n")
        print("| Metric | Concurrency | Improvement |")
        print("| --- | --- | --- |")
        for metric, key in [("TTFP", "mean_ttfp_ms"), ("E2E", "mean_e2e_ms"), ("RTF", "mean_rtf")]:
            for concurrency in concurrencies:
                left_map = {result["concurrency"]: result for result in all_results[0]}
                right_map = {result["concurrency"]: result for result in all_results[1]}
                left_value = left_map.get(concurrency, {}).get(key, 0)
                right_value = right_map.get(concurrency, {}).get(key, 0)
                if right_value > 0:
                    pct = (right_value - left_value) / right_value * 100
                    print(f"| {metric} | {concurrency} | {pct:+.1f}% |")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Fish Speech benchmark results")
    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to result JSON files (one per config)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="Labels for each config (must match --results count)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/fish-speech/results/fish_speech_benchmark.png",
        help="Output image path",
    )
    parser.add_argument("--title", type=str, default="Fish Speech S2 Pro", help="Title prefix for the plot")
    args = parser.parse_args()
    if len(args.results) != len(args.labels):
        parser.error("--results and --labels must have the same count")
    return args


if __name__ == "__main__":
    args = parse_args()
    all_results = load_results(args.results)
    print_comparison_table(all_results, args.labels)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if len(all_results) == 1:
        plot_single_summary(all_results[0], args.labels[0], args.output, title_prefix=args.title)
    else:
        plot_comparison(all_results, args.labels, args.output, title_prefix=args.title)
