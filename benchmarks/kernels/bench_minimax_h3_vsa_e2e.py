# SPDX-License-Identifier: Apache-2.0
"""Benchmark FastH3 VSA end to end on the 8-GPU TP2 x Ulysses4 layout."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from vllm_omni.entrypoints.async_omni import AsyncOmni


PROMPT = (
    "At night, three cats march into a bedroom playing tiny brass instruments, "
    "then abruptly file out, with synchronized room ambience."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict[str, Any]:
    import flashinfer
    import torch
    import vllm_omni

    flashinfer_root = Path(flashinfer.__file__).resolve().parent.parent
    vllm_omni_root = Path(vllm_omni.__file__).resolve().parent.parent
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "flashinfer": flashinfer.__version__,
        "flashinfer_source": str(flashinfer_root),
        "flashinfer_commit": _git_revision(flashinfer_root),
        "vllm_omni_source": str(vllm_omni_root),
        "vllm_omni_commit": _git_revision(vllm_omni_root),
    }


def engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "trust_remote_code": True,
        "task_type": "fl2va",
        "num_gpus": 8,
        "tensor_parallel_size": 2,
        "ulysses_degree": 4,
        "ring_degree": 1,
        "data_parallel_size": 1,
        "text_encoder_tp_size": 8,
        "vae_patch_parallel_size": 8,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": True,
        "diffusion_attention_config": {
            "default": {
                "backend": "FASTVIDEO_VSA",
                "fastvideo_vsa_topk": 64,
                "fastvideo_vsa_h3_kernel_backend": "flashinfer",
            }
        },
        "lora_path": args.adapter,
        "request_batch_max_wait_ms": 0.0,
        "enable_diffusion_pipeline_profiler": True,
        "stage_init_timeout": 1800.0,
        "init_timeout": 1800.0,
    }


def sampling_params(engine: AsyncOmni, duration: float) -> list[Any]:
    params = copy.deepcopy(engine.default_sampling_params_list)
    diffusion = params[0]
    diffusion.width = 1344
    diffusion.height = 768
    diffusion.fps = 24
    diffusion.num_inference_steps = 4
    diffusion.seed = 1101
    diffusion.extra_args = {
        "task": "t2va",
        "duration": duration,
        "aspect_ratio": "16:9",
    }
    return params


async def generate(engine: AsyncOmni, duration: float, request_id: str) -> dict[str, Any]:
    final_output = None
    started = time.perf_counter()
    async for output in engine.generate(
        prompt=PROMPT,
        request_id=request_id,
        sampling_params_list=sampling_params(engine, duration),
    ):
        if output.finished:
            final_output = output
    elapsed = time.perf_counter() - started
    if final_output is None or not final_output.images:
        raise RuntimeError(f"{request_id} finished without video output")

    frames = np.asarray(final_output.images[0])
    payload = final_output.multimodal_output or {}
    audio = np.asarray(payload.get("audio"))
    return {
        "request_id": request_id,
        "wall_time_s": elapsed,
        "stage_durations": final_output.stage_durations,
        "peak_memory_mb": float(final_output.peak_memory_mb),
        "frames_shape": list(frames.shape),
        "frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
        "audio_shape": list(audio.shape),
        "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = engine_kwargs(args)
    engine = AsyncOmni(**kwargs)
    try:
        for index in range(args.warmups):
            await generate(engine, args.duration, f"warmup-{index + 1}")
        measured = [
            await generate(engine, args.duration, f"measured-{index + 1}")
            for index in range(args.repeats)
        ]
    finally:
        engine.close()

    wall_times = [sample["wall_time_s"] for sample in measured]
    return {
        "environment": environment(),
        "engine_kwargs": kwargs,
        "workload": {
            "duration_s": args.duration,
            "width": 1344,
            "height": 768,
            "fps": 24,
            "steps": 4,
            "seed": 1101,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "measured": measured,
        "summary": {
            "mean_wall_time_s": sum(wall_times) / len(wall_times),
            "min_wall_time_s": min(wall_times),
            "max_wall_time_s": max(wall_times),
        },
    }


def main() -> None:
    args = parse_args()
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    result = asyncio.run(run(args))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print("MINIMAX_H3_VSA_E2E: PASS")
    print(rendered)


if __name__ == "__main__":
    main()
