# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import statistics
import time
from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LingBot dense generation after model load.")
    parser.add_argument("--model", default="/home/models/lingbot-video-dense-1.3b")
    parser.add_argument("--prompt", default="a robotic arm picks up a red block")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=5)
    return parser.parse_args()


def peak_memory_mb(output: Any) -> float:
    result = output[0] if isinstance(output, list) and output else output
    return float(getattr(result, "peak_memory_mb", 0.0) or 0.0)


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1.")

    load_start = time.perf_counter()
    omni = Omni(
        model=args.model,
        model_class_name="LingBotVideoPipeline",
        flow_shift=args.shift,
        parallel_config=DiffusionParallelConfig(),
    )
    load_seconds = time.perf_counter() - load_start
    print(f"load_seconds={load_seconds:.4f}", flush=True)

    latencies: list[float] = []
    peak_mb = 0.0
    prompt = {"prompt": args.prompt}

    for idx in range(args.runs):
        generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
        sampling_params = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        start = time.perf_counter()
        output = omni.generate(prompt, sampling_params)
        latency = time.perf_counter() - start
        latencies.append(latency)
        peak_mb = max(peak_mb, peak_memory_mb(output))
        print(f"run_{idx + 1}_seconds={latency:.4f}", flush=True)
        del output

    print(f"mean_seconds={statistics.mean(latencies):.4f}", flush=True)
    print(f"median_seconds={statistics.median(latencies):.4f}", flush=True)
    print(f"min_seconds={min(latencies):.4f}", flush=True)
    print(f"max_seconds={max(latencies):.4f}", flush=True)
    print(f"peak_mb={peak_mb:.2f}", flush=True)
    omni.close()


if __name__ == "__main__":
    main()
