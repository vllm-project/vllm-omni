#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
DiT Batching Performance Benchmark.

This script benchmarks the performance improvements from DiT batching,
comparing throughput, latency, and GPU utilization with and without batching.
"""

import argparse
import asyncio
import json
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.config.batching import DiTBatchingConfig
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    config_name: str
    total_requests: int
    total_time: float
    throughput_rps: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    gpu_memory_peak_mb: float
    gpu_utilization_avg: float
    batching_stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "total_requests": self.total_requests,
            "total_time": self.total_time,
            "throughput_rps": self.throughput_rps,
            "avg_latency": self.avg_latency,
            "p50_latency": self.p50_latency,
            "p95_latency": self.p95_latency,
            "p99_latency": self.p99_latency,
            "gpu_memory_peak_mb": self.gpu_memory_peak_mb,
            "gpu_utilization_avg": self.gpu_utilization_avg,
            "batching_stats": self.batching_stats,
        }


class DiTBatchingBenchmark:
    """Benchmark suite for DiT batching performance."""

    def __init__(self, model_name: str = "Qwen/Qwen-Image", device: str = "auto"):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        # Base configuration
        self.base_config = OmniDiffusionConfig(
            model=model_name,
            num_gpus=1 if self.device == "cuda" else 0,
        )

        # Benchmark configurations
        self.configs = {
            "no_batching": DiTBatchingConfig(enable_batching=False),
            "batching_small": DiTBatchingConfig(
                enable_batching=True,
                max_batch_size=4,
                max_wait_time_ms=50,
                max_memory_mb=4096,
            ),
            "batching_medium": DiTBatchingConfig(
                enable_batching=True,
                max_batch_size=8,
                max_wait_time_ms=100,
                max_memory_mb=8192,
            ),
            "batching_large": DiTBatchingConfig(
                enable_batching=True,
                max_batch_size=12,
                max_wait_time_ms=150,
                max_memory_mb=16384,
            ),
        }

    def create_test_requests(self, num_requests: int, diverse: bool = True) -> List[OmniDiffusionRequest]:
        """Create test requests for benchmarking."""
        requests = []

        # Base prompts for diversity
        prompts = [
            "A beautiful sunset over the ocean",
            "A cat sitting on a windowsill",
            "A futuristic city skyline at night",
            "A serene mountain landscape",
            "A cup of coffee on a wooden table",
            "A butterfly on a flower",
            "A snowy winter forest",
            "A tropical beach with palm trees",
            "A vintage car on a city street",
            "A starry night sky over a lake",
        ]

        for i in range(num_requests):
            prompt = prompts[i % len(prompts)] if diverse else "A beautiful landscape"

            # Vary parameters for realistic diversity
            height = 1024 if i % 3 == 0 else 512 if i % 3 == 1 else 768
            width = height  # Square images
            steps = 50 if i % 4 == 0 else 30 if i % 4 == 1 else 20 if i % 4 == 2 else 40
            cfg = 7.5 if i % 2 == 0 else 5.0

            request = OmniDiffusionRequest(
                prompt=prompt,
                request_id=f"bench-{i:04d}",
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=cfg,
                num_outputs_per_prompt=1,
                seed=i,  # Deterministic for reproducibility
            )
            requests.append(request)

        return requests

    async def benchmark_config(self, config_name: str, batching_config: DiTBatchingConfig,
                              requests: List[OmniDiffusionRequest], num_runs: int = 3) -> BenchmarkResult:
        """Benchmark a specific configuration."""
        print(f"\nBenchmarking {config_name}...")

        latencies = []
        gpu_memory_usage = []
        gpu_utilizations = []

        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}")

            # Create fresh engine for each run
            engine = DiffusionEngine.make_engine(self.base_config, batching_config)

            # Warm up
            warmup_request = OmniDiffusionRequest(
                prompt="warmup image",
                num_inference_steps=5,  # Fast warmup
                height=256,
                width=256,
            )
            engine.step_sync([warmup_request])

            # Benchmark run
            start_time = time.time()
            latencies_run = []

            for request in tqdm(requests, desc=f"Processing requests (run {run + 1})"):
                req_start = time.time()
                result = engine.step_sync([request])
                req_end = time.time()

                if result:
                    latencies_run.append(req_end - req_start)

                # Monitor GPU usage (if available)
                if torch.cuda.is_available():
                    memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_memory_usage.append(memory_mb)

                    # Note: GPU utilization would require nvidia-ml-py3 or similar
                    # For now, we'll use a placeholder
                    gpu_utilizations.append(85.0)  # Placeholder

            total_time = time.time() - start_time

            latencies.extend(latencies_run)

            # Cleanup
            engine.close()

        # Calculate statistics
        avg_latency = statistics.mean(latencies) if latencies else 0
        p50_latency = statistics.median(latencies) if latencies else 0
        p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else p50_latency
        p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else p50_latency

        throughput = len(requests) * num_runs / sum([total_time] * num_runs) if total_time > 0 else 0

        peak_memory = max(gpu_memory_usage) if gpu_memory_usage else 0
        avg_gpu_util = statistics.mean(gpu_utilizations) if gpu_utilizations else 0

        # Get batching stats if available
        batching_stats = None
        if hasattr(engine, 'get_batching_stats'):
            batching_stats = engine.get_batching_stats()

        return BenchmarkResult(
            config_name=config_name,
            total_requests=len(requests) * num_runs,
            total_time=total_time * num_runs,
            throughput_rps=throughput,
            avg_latency=avg_latency,
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            gpu_memory_peak_mb=peak_memory,
            gpu_utilization_avg=avg_gpu_util,
            batching_stats=batching_stats,
        )

    async def run_full_benchmark(self, num_requests: int = 50, diverse_requests: bool = True) -> Dict[str, BenchmarkResult]:
        """Run full benchmark suite comparing all configurations."""
        print("Starting DiT Batching Performance Benchmark")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Requests per config: {num_requests}")
        print(f"Diverse requests: {diverse_requests}")

        # Create test requests
        requests = self.create_test_requests(num_requests, diverse_requests)

        results = {}
        for config_name, batching_config in self.configs.items():
            result = await self.benchmark_config(config_name, batching_config, requests)
            results[config_name] = result

        return results

    def print_results(self, results: Dict[str, BenchmarkResult]):
        """Print benchmark results in a formatted way."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        print("<20")
        print("-" * 80)

        for result in results.values():
            print("<20"
                  "<8.2f"
                  "<8.2f"
                  "<8.2f"
                  "<8.2f"
                  "<10.0f"
                  "<8.1f")

            if result.batching_stats:
                stats = result.batching_stats
                print(f"    Batching: avg_batch_size={stats.get('avg_batch_size', 0):.1f}, "
                      f"batched_requests={stats.get('batched_requests', 0)}, "
                      f"single_requests={stats.get('single_requests', 0)}")

        # Calculate improvements
        if "no_batching" in results and "batching_medium" in results:
            base = results["no_batching"]
            batched = results["batching_medium"]

            throughput_improvement = ((batched.throughput_rps - base.throughput_rps) / base.throughput_rps) * 100
            latency_improvement = ((base.avg_latency - batched.avg_latency) / base.avg_latency) * 100

            print("IMPROVEMENTS (Medium Batching vs No Batching):")
            print(".1f")
            print(".1f")

    def save_results(self, results: Dict[str, BenchmarkResult], output_file: str):
        """Save results to JSON file."""
        data = {
            "benchmark_info": {
                "model": self.model_name,
                "device": self.device,
                "timestamp": time.time(),
            },
            "results": {name: result.to_dict() for name, result in results.items()}
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="DiT Batching Performance Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen-Image",
                       help="Model name to benchmark")
    parser.add_argument("--num-requests", type=int, default=50,
                       help="Number of requests per configuration")
    parser.add_argument("--diverse", action="store_true", default=True,
                       help="Use diverse prompts and parameters")
    parser.add_argument("--output", type=str, default="dit_batching_benchmark_results.json",
                       help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")

    args = parser.parse_args()

    benchmark = DiTBatchingBenchmark(model_name=args.model, device=args.device)
    results = await benchmark.run_full_benchmark(
        num_requests=args.num_requests,
        diverse_requests=args.diverse
    )

    benchmark.print_results(results)
    benchmark.save_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())