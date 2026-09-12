# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reproduce MammothModa2 AR prefix-cache A/B1/B2 measurements.

A disables prefix caching. B1 enables it but clears the cache after warmup and
before every sample. B2 enables it and measures verified cache hits.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pynvml
import torch
import vllm
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_extras import build_text_to_image_prompt, get_model_class_name

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "bytedance-research/MammothModa2-Preview"
DEFAULT_PROMPT = "A red cube on a white table"
DEFAULT_BLOCK_SIZE = 16
SCENARIO_CONFIG = {
    "a": "vllm_omni/deploy/mammoth_moda2_ar.yaml",
    "b1": "vllm_omni/deploy/mammoth_moda2_ar_prefix_cache.yaml",
    "b2": "vllm_omni/deploy/mammoth_moda2_ar_prefix_cache.yaml",
}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


class MemoryMonitor:
    """Poll device-wide NVML memory while one isolated benchmark is running."""

    def __init__(self, device_handle: Any) -> None:
        self.handle = device_handle
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            used = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1024**2
            self.peak_mib = max(self.peak_mib, float(used))
            self._stop.wait(0.02)

    def __enter__(self) -> MemoryMonitor:
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        self._thread.join()
        used = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 1024**2
        self.peak_mib = max(self.peak_mib, float(used))


def resolve_nvml_device(logical_index: int) -> tuple[Any, str]:
    """Resolve a logical CUDA ordinal to its physical NVML device."""
    if logical_index < 0:
        raise ValueError("--device-index must be non-negative")

    visible_devices = [
        device.strip() for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if device.strip()
    ]
    if visible_devices:
        if logical_index >= len(visible_devices):
            raise ValueError(
                f"--device-index {logical_index} is outside CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}"
            )
        physical_device = visible_devices[logical_index]
        try:
            physical_index = int(physical_device)
        except ValueError:
            handle = pynvml.nvmlDeviceGetHandleByUUID(physical_device)
        else:
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
    else:
        device_count = pynvml.nvmlDeviceGetCount()
        if logical_index >= device_count:
            raise ValueError(f"--device-index {logical_index} is invalid for {device_count} physical GPUs")
        handle = pynvml.nvmlDeviceGetHandleByIndex(logical_index)

    return handle, str(pynvml.nvmlDeviceGetUUID(handle))


def build_request(
    omni: Omni,
    prompt_text: str,
    height: int,
    width: int,
    seed: int,
) -> tuple[dict[str, Any], SamplingParams]:
    request = build_text_to_image_prompt(
        model_class_name=get_model_class_name(omni),
        prompt={"prompt": prompt_text, "modalities": ["image"]},
        height=height,
        width=width,
    )
    info = request["additional_information"]
    ar_width = int(info["ar_width"][0])
    ar_height = int(info["ar_height"][0])
    params = SamplingParams(
        temperature=0.0,
        top_k=1,
        seed=seed,
        max_tokens=ar_height * (ar_width + 1) + 1,
        detokenize=False,
    )
    return request, params


def run_once(
    omni: Omni,
    request: dict[str, Any],
    params: SamplingParams,
    device_handle: Any,
) -> dict[str, Any]:
    with MemoryMonitor(device_handle) as memory:
        started = time.perf_counter()
        outputs = omni.generate(
            request,
            sampling_params_list=[params],
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - started
    output = outputs[0]
    completion = output.outputs[0]
    token_ids = list(getattr(completion, "cumulative_token_ids", None) or completion.token_ids)
    stage_metrics = output.metrics["stage_metrics"]["0"]
    return {
        "latency_ms": elapsed * 1000,
        "ttft_ms": float(stage_metrics["vllm_ttft_ms"]),
        "prompt_tokens": len(output.prompt_token_ids),
        "generated_tokens": len(token_ids),
        "tokens_per_second": len(token_ids) / elapsed,
        "cached_tokens": int(output.num_cached_tokens or 0),
        "cache_creation_tokens": int(output.num_cache_creation_tokens or 0),
        "peak_gpu_memory_mib": memory.peak_mib,
    }


def clear_prefix_cache(omni: Omni) -> None:
    """Reset AR prefix state without unloading the warmed model or kernels."""
    pause_results = omni.engine.collective_rpc(
        method="pause_scheduler",
        kwargs={"mode": "wait", "clear_cache": True},
        stage_ids=[0],
    )
    if any(isinstance(result, dict) and result.get("error") for result in pause_results):
        raise RuntimeError(f"failed to clear prefix cache: {pause_results}")
    resume_results = omni.engine.collective_rpc(
        method="resume_scheduler",
        stage_ids=[0],
    )
    if any(isinstance(result, dict) and result.get("error") for result in resume_results):
        raise RuntimeError(f"failed to resume scheduler: {resume_results}")


def validate_samples(
    scenario: str,
    samples: list[dict[str, Any]],
    block_size: int,
) -> None:
    if scenario == "a":
        valid = all(sample["cached_tokens"] == 0 and sample["cache_creation_tokens"] == 0 for sample in samples)
    elif scenario == "b1":
        valid = all(sample["cached_tokens"] == 0 and sample["cache_creation_tokens"] > 0 for sample in samples)
    else:
        valid = all(
            sample["cached_tokens"] == (sample["prompt_tokens"] - 1) // block_size * block_size
            and sample["cached_tokens"] > 0
            and sample["cache_creation_tokens"]
            == sample["prompt_tokens"] // block_size * block_size - sample["cached_tokens"]
            for sample in samples
        )
    if not valid:
        states = [(sample["cached_tokens"], sample["cache_creation_tokens"]) for sample in samples]
        raise RuntimeError(f"scenario {scenario} cache-state verification failed: {states}")


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [sample["latency_ms"] for sample in samples]
    ttfts = [sample["ttft_ms"] for sample in samples]
    throughputs = [sample["tokens_per_second"] for sample in samples]
    return {
        "count": len(samples),
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_p95": percentile(latencies, 0.95),
        "latency_ms_mean": statistics.mean(latencies),
        "ttft_ms_p50": statistics.median(ttfts),
        "ttft_ms_p95": percentile(ttfts, 0.95),
        "ttft_ms_mean": statistics.mean(ttfts),
        "throughput_tokens_per_second_mean": statistics.mean(throughputs),
        "peak_gpu_memory_mib_max": max(sample["peak_gpu_memory_mib"] for sample in samples),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIO_CONFIG, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Logical CUDA device ordinal, resolved through CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.prompt_repeat < 1:
        raise ValueError("--prompt-repeat must be at least 1")
    if args.block_size < 1:
        raise ValueError("--block-size must be at least 1")

    pynvml.nvmlInit()
    device_handle, device_uuid = resolve_nvml_device(args.device_index)
    deploy_config = REPO_ROOT / SCENARIO_CONFIG[args.scenario]
    stage_overrides = {"0": {"block_size": args.block_size}}
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        stage_overrides["0"]["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profile_dir),
            "torch_profiler_use_gzip": False,
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True,
        }

    omni = Omni(
        model=args.model,
        deploy_config=str(deploy_config),
        mode="text-to-image",
        log_stats=True,
        stage_overrides=stage_overrides,
    )
    try:
        prompt_text = " ".join([args.prompt] * args.prompt_repeat)
        request, params = build_request(
            omni,
            prompt_text,
            args.height,
            args.width,
            args.seed,
        )
        warmup = run_once(omni, request, params, device_handle)

        if args.profile_dir is not None:
            omni.start_profile(profile_prefix=f"mammoth_ar_{args.scenario}")
        try:
            samples = []
            for _ in range(args.iterations):
                if args.scenario == "b1":
                    clear_prefix_cache(omni)
                samples.append(
                    run_once(
                        omni,
                        request,
                        params,
                        device_handle,
                    )
                )
        finally:
            if args.profile_dir is not None:
                omni.stop_profile()

        validate_samples(args.scenario, samples, args.block_size)
        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(sys.argv),
            "scenario": args.scenario,
            "model": args.model,
            "deploy_config": str(deploy_config.relative_to(REPO_ROOT)),
            "prompt": args.prompt,
            "prompt_repeat": args.prompt_repeat,
            "image_size": [args.height, args.width],
            "seed": args.seed,
            "block_size": args.block_size,
            "iterations": args.iterations,
            "warmup": warmup,
            "samples": samples,
            "summary": summarize(samples),
            "environment": {
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=REPO_ROOT,
                    text=True,
                ).strip(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "vllm": vllm.__version__,
                "cuda_logical_device": args.device_index,
                "gpu_uuid": device_uuid,
                "gpu": pynvml.nvmlDeviceGetName(device_handle),
            },
            "profile_dir": (str(args.profile_dir) if args.profile_dir is not None else None),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        omni.shutdown()
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
