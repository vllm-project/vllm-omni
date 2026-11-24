# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Omni compatible implementation of vLLM's startup benchmarking.

Usage: `vllm bench startup <opts> --omni`
"""

import argparse
import json
import multiprocessing
import time
from typing import Any

from tqdm import tqdm
from vllm.benchmarks.startup import (
    MetricDesc,
    _compute_metric,
    _metric_to_json,
    _print_phase,
    add_cli_args,
    cold_startup,
)

from vllm_omni.entrypoints.cli.benchmark.base import OmniBenchmarkSubcommandBase
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.utils.tracking_parser import TrackingNamespace


def run_startup_in_subprocess(args: TrackingNamespace, result_queue: multiprocessing.Queue) -> None:
    """Run Omni startup in a subprocess and return timing metrics via a queue.
    Analogous to the run_startup_in_subprocess in vllm.benchmarks.startup, although
    we don't need to import Omni lazily for now since we do not support fork yet.
    """
    # Measure total startup time
    start_time = time.perf_counter()

    omni = Omni.from_cli_args(args)

    total_startup_time = time.perf_counter() - start_time
    omni.close()

    # TODO: it would be nice to report compile time metrics here, but this is nontrivial
    # since stages run in separate processes; currently upstream vLLM gets around this with
    # VLLM_ENABLE_V1_MULTIPROCESSING=0 and setting it on the config, but we can't do that very
    # easily here.
    result_queue.put({"total_startup_time": total_startup_time})


def create_omni_and_measure_startup(args: TrackingNamespace) -> dict[str, float]:
    """Create an Omni instance in a subprocess and measure startup time."""
    result_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_startup_in_subprocess, args=(args, result_queue))
    process.start()
    process.join()
    if result_queue.empty():
        raise RuntimeError(f"Startup subprocess failed (exit code {process.exitcode}); see its traceback above")
    return result_queue.get()


class OmniBenchmarkStartupSubcommand(OmniBenchmarkSubcommandBase):
    """The `startup` subcommand for vllm bench."""

    name = "startup"
    help = "Benchmark the cold and warm startup time of vLLM-Omni pipelines."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)
        parser.add_argument("--deploy-config", type=str, default=None, help="Model specific deploy config override.")

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # Omni.from_cli_args needs the explicit-arg tracking; fail here rather than in every subprocess.
        if not isinstance(args, TrackingNamespace):
            raise TypeError(f"expected a TrackingNamespace, got {type(args)}")

        # Set multiprocessing start method to 'spawn' for clean process isolation
        # This ensures each subprocess starts fresh without inheriting state
        multiprocessing.set_start_method("spawn", force=True)

        # Collect cold startup iterations
        print("Measuring cold startup time...\n")
        cold_iterations = []
        for _ in tqdm(range(args.num_iters_cold), desc="Cold startup iterations"):
            with cold_startup():
                cold_iterations.append(create_omni_and_measure_startup(args))

        # Warmup for warm startup
        print("\nWarming up for warm startup measurement...\n")
        for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
            create_omni_and_measure_startup(args)

        # Collect warm startup iterations
        print("\nMeasuring warm startup time...\n")
        warm_iterations = []
        for _ in tqdm(range(args.num_iters_warm), desc="Warm startup iterations"):
            warm_iterations.append(create_omni_and_measure_startup(args))

        # Skip phases with no iterations (e.g. --num-iters-warm 0); vLLM's metric helpers need >= 1.
        startup = MetricDesc("total_startup_time", "startup", "Startup time")
        cold_metrics = [_compute_metric("cold", startup, cold_iterations)] if cold_iterations else []
        warm_metrics = [_compute_metric("warm", startup, warm_iterations)] if warm_iterations else []
        all_metrics = cold_metrics + warm_metrics

        # Print results
        print("\n" + "=" * 60)
        print("STARTUP TIME BENCHMARK RESULTS")
        print("=" * 60)
        if cold_metrics:
            _print_phase("COLD STARTUP", cold_metrics)
        if warm_metrics:
            _print_phase("WARM STARTUP", warm_metrics)
        print("=" * 60)

        # Output JSON results if specified
        if args.output_json:
            results: dict[str, Any] = {}
            for m in all_metrics:
                results.update(_metric_to_json(m))
            with open(args.output_json, "w") as f:
                json.dump(results, f, indent=4)
