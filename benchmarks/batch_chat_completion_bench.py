# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Benchmark: batch chat completion endpoint vs N individual requests.

Requires a running vLLM-Omni server.  Measures wall-clock latency of:
  - Batch endpoint:  POST /v1/chat/completions/batch  (N messages)
  - Individual:      N concurrent POST /v1/chat/completions

Usage:
    # Start server first, then:
    python benchmarks/batch_chat_completion_bench.py

    # Custom settings:
    python benchmarks/batch_chat_completion_bench.py \\
        --base-url http://localhost:8000 \\
        --batch-sizes 1 2 4 8 16 32 64 \\
        --max-tokens 32 --warmup 5 --repeats 10

    # Save results for A/B comparison:
    python benchmarks/batch_chat_completion_bench.py --output head.json

    # Compare two saved results:
    python benchmarks/batch_chat_completion_bench.py \\
        --compare base.json head.json
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import statistics
import time

import httpx
from PIL import Image

PROMPTS = [
    "Explain what a neural network is in one sentence.",
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "What is 2 + 2?",
    "Name three primary colors.",
    "What year did the moon landing happen?",
    "Define the word 'serendipity'.",
    "What is the speed of light?",
    "Explain gravity in simple terms.",
    "What is the largest planet in our solar system?",
    "How many continents are there?",
    "What is photosynthesis?",
    "Name the four seasons.",
    "What is the boiling point of water in Celsius?",
    "What does DNA stand for?",
    "Explain what an algorithm is.",
    "What is the smallest prime number?",
    "Name a mammal that can fly.",
    "What is the chemical symbol for gold?",
    "How many sides does a hexagon have?",
    "What is the speed of sound approximately?",
    "Name the closest star to Earth.",
    "What is the freezing point of water in Fahrenheit?",
    "Define the word 'ubiquitous'.",
    "What is the square root of 144?",
    "Name the longest river in the world.",
    "What is an atom?",
    "How many bones are in the adult human body?",
    "What is the chemical formula for water?",
    "Name the inventor of the telephone.",
    "What is the largest ocean on Earth?",
    "Define the word 'ephemeral'.",
    "What is Pi approximately equal to?",
    "Name the first element on the periodic table.",
    "What causes rainbows?",
    "How many legs does a spider have?",
    "What is the capital of Japan?",
    "Define the word 'pragmatic'.",
    "What is absolute zero in Celsius?",
    "Name the largest desert on Earth.",
    "What is a black hole?",
    "How many minutes are in a day?",
    "What is the hardest natural substance?",
    "Name the author of Romeo and Juliet.",
    "What is the tallest mountain on Earth?",
    "Define the word 'paradigm'.",
    "What is the distance from Earth to the Moon approximately?",
    "Name three types of rocks.",
    "What is an eclipse?",
    "How many teeth does an adult human have?",
    "What is the chemical symbol for silver?",
    "Name the smallest country in the world.",
    "What is entropy?",
    "How many strings does a standard guitar have?",
    "What is the capital of Australia?",
    "Define the word 'dichotomy'.",
    "What is the speed of Earth's rotation?",
    "Name the largest animal ever known to exist.",
    "What is a supernova?",
    "How many chambers does the human heart have?",
    "What is the chemical formula for salt?",
    "Name the deepest point in the ocean.",
    "What is a quasar?",
    "How many chromosomes do humans have?",
]


async def detect_model(client: httpx.AsyncClient, base_url: str) -> str:
    resp = await client.get(f"{base_url}/v1/models")
    resp.raise_for_status()
    models = resp.json()["data"]
    if not models:
        raise RuntimeError("No models found on the server")
    return models[0]["id"]


def make_image_urls(count: int, size: int) -> list[str]:
    """Generate distinct JPEG data URLs.

    Distinct images reduce (but do not eliminate) server-side multimodal
    cache hits: images are cycled across warmup, repeats, and both endpoints,
    so raise ``--num-images`` to cover the full request count if absolute
    preprocess timings are needed.  Batch-vs-individual speedup ratios are
    unaffected because both modes cycle the same set.
    """
    urls: list[str] = []
    for i in range(count):
        channels = [Image.effect_noise((size, size), sigma=40.0 + 10.0 * ((i + k) % 7)) for k in range(3)]
        img = Image.merge("RGB", channels)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        urls.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
    return urls


def build_message(prompt: str, image_url: str | None) -> list[dict]:
    if image_url is None:
        return [{"role": "user", "content": prompt}]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


async def bench_batch(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    batch_size: int,
    max_tokens: int,
    image_urls: list[str] | None = None,
) -> float:
    """Send one batch request with batch_size messages. Return latency in seconds."""
    messages = [
        build_message(
            PROMPTS[i % len(PROMPTS)],
            image_urls[i % len(image_urls)] if image_urls else None,
        )
        for i in range(batch_size)
    ]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    resp = await client.post(
        f"{base_url}/v1/chat/completions/batch",
        json=payload,
        timeout=300.0,
    )
    t1 = time.perf_counter()
    resp.raise_for_status()
    data = resp.json()
    assert "choices" in data, f"Unexpected response: {data}"
    assert len(data["choices"]) == batch_size, f"Expected {batch_size} choices, got {len(data['choices'])}"
    return t1 - t0


async def bench_individual(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    batch_size: int,
    max_tokens: int,
    image_urls: list[str] | None = None,
) -> float:
    """Send batch_size individual requests concurrently. Return total latency."""
    payloads = [
        {
            "model": model,
            "messages": build_message(
                PROMPTS[i % len(PROMPTS)],
                image_urls[i % len(image_urls)] if image_urls else None,
            ),
            "max_tokens": max_tokens,
        }
        for i in range(batch_size)
    ]

    async def _single(payload):
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=300.0,
        )
        resp.raise_for_status()
        return resp.json()

    t0 = time.perf_counter()
    results = await asyncio.gather(*[_single(p) for p in payloads])
    t1 = time.perf_counter()
    assert len(results) == batch_size
    return t1 - t0


async def run_benchmark(args):
    image_urls: list[str] | None = None
    if args.image_mode:
        image_urls = make_image_urls(args.num_images, args.image_size)
        print(f"Image mode: {args.num_images} distinct {args.image_size}x{args.image_size} JPEGs")

    async with httpx.AsyncClient() as client:
        model = args.model
        if not model:
            model = await detect_model(client, args.base_url)
        print(f"Model: {model}")
        print(f"Base URL: {args.base_url}")
        print(f"Max tokens: {args.max_tokens}")
        print(f"Batch sizes: {args.batch_sizes}")
        print(f"Warmup: {args.warmup}, Repeats: {args.repeats}")
        print()

        # Warmup
        print("Warming up...", end="", flush=True)
        for _ in range(args.warmup):
            await bench_batch(
                client,
                args.base_url,
                model,
                2,
                args.max_tokens,
                image_urls,
            )
            await bench_individual(
                client,
                args.base_url,
                model,
                2,
                args.max_tokens,
                image_urls,
            )
        print(" done\n")

        header = (
            f"{'Batch':>5} | "
            f"{'Batch EP (ms)':>13} | "
            f"{'Individual (ms)':>15} | "
            f"{'Speedup':>8} | "
            f"{'Batch req/s':>11} | "
            f"{'Indiv req/s':>11}"
        )
        print(header)
        print("-" * len(header))

        all_results = []

        for bs in args.batch_sizes:
            batch_latencies = []
            indiv_latencies = []

            for _ in range(args.repeats):
                bl = await bench_batch(
                    client,
                    args.base_url,
                    model,
                    bs,
                    args.max_tokens,
                    image_urls,
                )
                batch_latencies.append(bl)

                il = await bench_individual(
                    client,
                    args.base_url,
                    model,
                    bs,
                    args.max_tokens,
                    image_urls,
                )
                indiv_latencies.append(il)

            batch_median = statistics.median(batch_latencies)
            indiv_median = statistics.median(indiv_latencies)
            speedup = indiv_median / batch_median if batch_median > 0 else 0
            batch_rps = bs / batch_median if batch_median > 0 else 0
            indiv_rps = bs / indiv_median if indiv_median > 0 else 0

            print(
                f"{bs:>5} | "
                f"{batch_median * 1000:>13.1f} | "
                f"{indiv_median * 1000:>15.1f} | "
                f"{speedup:>7.2f}x | "
                f"{batch_rps:>11.1f} | "
                f"{indiv_rps:>11.1f}"
            )

            all_results.append(
                {
                    "batch_size": bs,
                    "batch_latency_ms": batch_median * 1000,
                    "individual_latency_ms": indiv_median * 1000,
                    "speedup": speedup,
                    "batch_req_per_sec": batch_rps,
                    "individual_req_per_sec": indiv_rps,
                    "batch_all_ms": [x * 1000 for x in batch_latencies],
                    "individual_all_ms": [x * 1000 for x in indiv_latencies],
                }
            )

        if args.output:
            out = {
                "model": model,
                "base_url": args.base_url,
                "max_tokens": args.max_tokens,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "image_mode": args.image_mode,
                "results": all_results,
            }
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\nResults saved to {args.output}")


def compare(file_a: str, file_b: str):
    """Compare two saved benchmark results (base vs head)."""
    with open(file_a) as f:
        a = json.load(f)
    with open(file_b) as f:
        b = json.load(f)

    print(f"A (base): {file_a}  model={a['model']}")
    print(f"B (head): {file_b}  model={b['model']}")
    print()

    header = (
        f"{'Batch':>5} | "
        f"{'Base (ms)':>10} | "
        f"{'Head (ms)':>10} | "
        f"{'Change':>8} | "
        f"{'Base req/s':>10} | "
        f"{'Head req/s':>10}"
    )
    print(header)
    print("-" * len(header))

    a_results = {r["batch_size"]: r for r in a["results"]}
    b_results = {r["batch_size"]: r for r in b["results"]}

    for bs in sorted(set(a_results) | set(b_results)):
        ar = a_results.get(bs)
        br = b_results.get(bs)
        if not ar or not br:
            continue

        a_ms = ar["batch_latency_ms"]
        b_ms = br["batch_latency_ms"]
        speedup = a_ms / b_ms if b_ms > 0 else 0
        a_rps = ar["batch_req_per_sec"]
        b_rps = br["batch_req_per_sec"]

        marker = "faster" if speedup > 1.05 else "slower" if speedup < 0.95 else "~same"
        print(
            f"{bs:>5} | "
            f"{a_ms:>10.1f} | "
            f"{b_ms:>10.1f} | "
            f"{speedup:>6.2f}x {marker} | "  # noqa: E501
            f"{a_rps:>10.1f} | "
            f"{b_rps:>10.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark batch chat completion endpoint")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="vLLM server base URL",
    )
    parser.add_argument("--model", default=None, help="Model name (auto-detected if omitted)")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64],
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--image-mode",
        action="store_true",
        help="Send image+text multimodal messages (distinct generated JPEGs) instead of text-only",
    )
    parser.add_argument("--image-size", type=int, default=512, help="Generated image size (image mode)")
    parser.add_argument(
        "--num-images",
        type=int,
        default=8,
        help="Number of distinct images to cycle through (image mode); raise to reduce MM-cache hits across iterations",
    )
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASE", "HEAD"),
        help="Compare two saved result files",
    )
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
        return

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
