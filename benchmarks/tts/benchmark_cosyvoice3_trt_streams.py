"""Benchmark CosyVoice3-style CUDA stream handoff strategies."""

import argparse
import json
import statistics
import time

import torch

from vllm_omni.platforms import current_omni_platform


def run_case(
    *,
    device: torch.device,
    steps: int,
    warmup_steps: int,
    producer_cycles: int,
    estimator_cycles: int,
    use_stream_dependencies: bool,
) -> dict[str, float | int | str]:
    caller_stream = torch.cuda.Stream(device=device)
    estimator_stream = torch.cuda.Stream(device=device)
    input_tensor = torch.zeros(1024, device=device)
    output_tensor = torch.empty_like(input_tensor)
    submission_ms = []

    def step(measure: bool) -> None:
        with torch.cuda.stream(caller_stream):
            torch.cuda._sleep(producer_cycles)
            input_tensor.add_(1)

        start = time.perf_counter_ns()
        if use_stream_dependencies:
            estimator_stream.wait_stream(caller_stream)
        else:
            caller_stream.synchronize()

        with torch.cuda.stream(estimator_stream):
            torch.cuda._sleep(estimator_cycles)
            output_tensor.copy_(input_tensor)

        if use_stream_dependencies:
            caller_stream.wait_stream(estimator_stream)
        else:
            estimator_stream.synchronize()

        if measure:
            submission_ms.append((time.perf_counter_ns() - start) / 1e6)

    for _ in range(warmup_steps):
        step(measure=False)
    caller_stream.synchronize()
    estimator_stream.synchronize()
    input_tensor.zero_()
    torch.accelerator.reset_peak_memory_stats(device)
    initial_allocated = torch.accelerator.memory_allocated(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    with torch.cuda.stream(caller_stream):
        start_event.record()
    for _ in range(steps):
        step(measure=True)
    with torch.cuda.stream(caller_stream):
        end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1e3
    peak_delta_bytes = torch.accelerator.max_memory_allocated(device) - initial_allocated

    expected = torch.full_like(output_tensor, steps)
    torch.testing.assert_close(output_tensor, expected, rtol=0, atol=0)
    return {
        "mode": "wait_stream" if use_stream_dependencies else "synchronize",
        "steps": steps,
        "submission_median_ms": statistics.median(submission_ms),
        "submission_p95_ms": statistics.quantiles(submission_ms, n=20)[18],
        "submission_mean_ms": statistics.mean(submission_ms),
        "gpu_elapsed_ms": start_event.elapsed_time(end_event),
        "wall_elapsed_ms": wall_ms,
        "peak_allocation_delta_bytes": peak_delta_bytes,
        "final_value": int(output_tensor[0].item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--producer-cycles", type=int, default=1_000_000)
    parser.add_argument("--estimator-cycles", type=int, default=2_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    current_omni_platform.set_device(device)
    torch.manual_seed(0)

    old_results = []
    new_results = []
    for repeat in range(args.repeats):
        common_args = {
            "device": device,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "producer_cycles": args.producer_cycles,
            "estimator_cycles": args.estimator_cycles,
        }
        if repeat % 2 == 0:
            old_results.append(run_case(**common_args, use_stream_dependencies=False))
            new_results.append(run_case(**common_args, use_stream_dependencies=True))
        else:
            new_results.append(run_case(**common_args, use_stream_dependencies=True))
            old_results.append(run_case(**common_args, use_stream_dependencies=False))

    old_submission_ms = statistics.median(result["submission_median_ms"] for result in old_results)
    new_submission_ms = statistics.median(result["submission_median_ms"] for result in new_results)
    old_gpu_ms = statistics.median(result["gpu_elapsed_ms"] for result in old_results)
    new_gpu_ms = statistics.median(result["gpu_elapsed_ms"] for result in new_results)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "repeats": args.repeats,
                "warmup_steps": args.warmup_steps,
                "producer_cycles": args.producer_cycles,
                "estimator_cycles": args.estimator_cycles,
                "old_submission_median_ms": old_submission_ms,
                "new_submission_median_ms": new_submission_ms,
                "submission_median_speedup": old_submission_ms / new_submission_ms,
                "old_gpu_elapsed_median_ms": old_gpu_ms,
                "new_gpu_elapsed_median_ms": new_gpu_ms,
                "gpu_elapsed_change_percent": (new_gpu_ms - old_gpu_ms) / old_gpu_ms * 100,
                "parity": all(result["final_value"] == args.steps for result in old_results + new_results),
                "old_runs": old_results,
                "new_runs": new_results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
