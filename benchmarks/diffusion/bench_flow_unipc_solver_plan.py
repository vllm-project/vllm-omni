# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare FlowUniPC against a trusted scheduler source file from the base revision."""

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Trusted base scheduler Python source")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 2:
        parser.error("steps must be positive and repeats must be at least 2")
    spec = importlib.util.spec_from_file_location("reference_unipc", args.reference)
    if spec is None or spec.loader is None:
        parser.error("reference must name an importable Python source file")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    schedulers = {
        "base": reference.FlowUniPCMultistepScheduler(),
        "plan": FlowUniPCMultistepScheduler(),
    }
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(42)
    initial = torch.randn((1, 16, 5, 32, 32), device=device, dtype=torch.bfloat16, generator=generator)
    timings = {name: [] for name in schedulers}
    cold = {}
    for repeat in range(args.repeats + 2):
        results = {}
        for name in list(schedulers) if repeat % 2 == 0 else list(reversed(schedulers)):
            scheduler = schedulers[name]
            sample = initial.clone()
            if device.type == "cuda":
                torch.accelerator.synchronize(device)
            start = time.perf_counter()
            scheduler.set_timesteps(args.steps, device=device)
            with torch.inference_mode():
                for timestep in scheduler.timesteps:
                    sample = scheduler.step(initial, timestep, sample).prev_sample
            if device.type == "cuda":
                torch.accelerator.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if repeat == 0:
                cold[name] = elapsed_ms
            if repeat >= 2:
                timings[name].append(elapsed_ms)
            results[name] = sample
        torch.testing.assert_close(results["base"], results["plan"], rtol=0, atol=0)
    print(
        json.dumps(
            {
                "device": str(device),
                "torch": torch.__version__,
                "steps": args.steps,
                "shape": list(initial.shape),
                "dtype": str(initial.dtype),
                "warmup": 2,
                "cold_ms": cold,
                "samples_ms": timings,
                "median_ms": {k: statistics.median(v) for k, v in timings.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
