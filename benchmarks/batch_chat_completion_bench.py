# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Benchmark batch chat completion: optimized vs legacy path.

Measures wall-clock latency and throughput at varying batch sizes.
Requires a running vLLM-Omni server (``vllm-omni serve <model>``).

Usage:
    python benchmarks/batch_chat_completion_bench.py \
        --base-url http://localhost:8000 \
        --batch-sizes 1 2 4 8 16 32
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import requests


def build_batch_payload(
    model: str,
    batch_size: int,
    max_tokens: int = 32,
) -> dict:
    messages = [[{"role": "user", "content": f"Say hello in exactly one word. (item {i})"}] for i in range(batch_size)]
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }


def build_single_payload(
    model: str,
    max_tokens: int = 32,
    item_idx: int = 0,
) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": f"Say hello in exactly one word. (item {item_idx})"}],
        "max_tokens": max_tokens,
    }


def bench_batch(
    base_url: str,
    model: str,
    batch_size: int,
    max_tokens: int,
    warmup: int,
    repeats: int,
) -> dict:
    url = f"{base_url}/v1/chat/completions/batch"
    payload = build_batch_payload(model, batch_size, max_tokens)

    for _ in range(warmup):
        requests.post(url, json=payload, timeout=120)

    latencies = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=120)
        t1 = time.perf_counter()
        resp.raise_for_status()
        latencies.append(t1 - t0)

    return {
        "path": "batch",
        "batch_size": batch_size,
        "latencies_s": latencies,
        "mean_s": statistics.mean(latencies),
        "median_s": statistics.median(latencies),
        "throughput_req_per_s": batch_size / statistics.mean(latencies),
    }


def bench_sequential(
    base_url: str,
    model: str,
    batch_size: int,
    max_tokens: int,
    warmup: int,
    repeats: int,
) -> dict:
    url = f"{base_url}/v1/chat/completions"

    for _ in range(warmup):
        payload = build_single_payload(model, max_tokens)
        requests.post(url, json=payload, timeout=120)

    latencies = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(batch_size):
            payload = build_single_payload(model, max_tokens, item_idx=i)
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    return {
        "path": "sequential",
        "batch_size": batch_size,
        "latencies_s": latencies,
        "mean_s": statistics.mean(latencies),
        "median_s": statistics.median(latencies),
        "throughput_req_per_s": batch_size / statistics.mean(latencies),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark batch chat completion")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default=None, help="Model name (auto-detected from server if omitted)")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    if args.model is None:
        resp = requests.get(f"{args.base_url}/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json()["data"]
        if not models:
            raise SystemExit("No models found on server")
        args.model = models[0]["id"]
        print(f"Auto-detected model: {args.model}")

    print(f"Model: {args.model}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Max tokens: {args.max_tokens}, Warmup: {args.warmup}, Repeats: {args.repeats}")
    print()

    cols = [
        f"{'Batch Size':>10}",
        f"{'Path':>12}",
        f"{'Mean (s)':>10}",
        f"{'Median (s)':>10}",
        f"{'Throughput':>12}",
        f"{'Speedup':>8}",
    ]
    header = " | ".join(cols)
    print(header)
    print("-" * len(header))

    all_results = []

    for bs in args.batch_sizes:
        seq_result = bench_sequential(
            args.base_url,
            args.model,
            bs,
            args.max_tokens,
            args.warmup,
            args.repeats,
        )
        batch_result = bench_batch(
            args.base_url,
            args.model,
            bs,
            args.max_tokens,
            args.warmup,
            args.repeats,
        )

        speedup = seq_result["mean_s"] / batch_result["mean_s"] if batch_result["mean_s"] > 0 else float("inf")

        seq_tp = seq_result["throughput_req_per_s"]
        bat_tp = batch_result["throughput_req_per_s"]
        print(
            f"{bs:>10} | {'sequential':>12} | "
            f"{seq_result['mean_s']:>10.3f} | "
            f"{seq_result['median_s']:>10.3f} | "
            f"{seq_tp:>12.2f} |"
        )
        print(
            f"{bs:>10} | {'batch':>12} | "
            f"{batch_result['mean_s']:>10.3f} | "
            f"{batch_result['median_s']:>10.3f} | "
            f"{bat_tp:>12.2f} | {speedup:>7.2f}x"
        )

        all_results.append({"sequential": seq_result, "batch": batch_result, "speedup": speedup})

    if args.output:
        for r in all_results:
            for key in ("sequential", "batch"):
                r[key].pop("latencies_s", None)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
