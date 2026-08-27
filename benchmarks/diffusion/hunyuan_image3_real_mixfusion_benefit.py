# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark real HunyuanImage-3.0 MixFusion benefit with the loaded model.

This is intentionally different from the synthetic block benchmark:

* It loads the real HunyuanImage-3.0 DiT pipeline through DiffusionEngine.
* It runs the real tokenizer/RoPE/CFG/scheduler/VAE path.
* It compares:
  - independent: same prompts run one by one
  - mixfusion_batch: the same prompts in one batched diffusion request

The batched path creates one OmniDiffusionRequest per prompt and runs them
concurrently, reaching HunyuanImage3Pipeline.forward() with a multi-request
batch so the current MixFusion implementation can build a mixed-resolution plan.

Example:
    python benchmarks/diffusion/hunyuan_image3_real_mixfusion_benefit.py \
      --model tencent/HunyuanImage-3.0-Instruct \
      --image-sizes 1024x1024,512x512 \
      --steps 20 --iters 3 --tensor-parallel-size 4 --quantization fp8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt


def parse_image_sizes(raw: str) -> list[tuple[int, int]]:
    sizes = []
    for item in raw.split(","):
        h_raw, w_raw = item.lower().split("x", maxsplit=1)
        sizes.append((int(h_raw), int(w_raw)))
    if not sizes:
        raise ValueError("At least one image size is required.")
    return sizes


def parse_prompts(raw: str | None, count: int) -> list[str]:
    if raw is None:
        defaults = [
            "A cinematic photo of a glass observatory on Mars at sunrise",
            "A watercolor painting of a quiet mountain lake",
            "A detailed product photo of a transparent mechanical keyboard",
            "A cozy reading room with warm sunlight and plants",
        ]
        prompts = defaults[:count]
        while len(prompts) < count:
            prompts.append(f"A high quality image sample {len(prompts)}")
        return prompts

    prompts = [item.strip() for item in raw.split("||") if item.strip()]
    if len(prompts) != count:
        raise ValueError(f"Expected {count} prompts separated by '||', got {len(prompts)}.")
    return prompts


def build_prompt_dicts(prompts: list[str], sizes: list[tuple[int, int]]) -> list[OmniTextPrompt]:
    result: list[OmniTextPrompt] = []
    for prompt, (height, width) in zip(prompts, sizes, strict=True):
        result.append(
            {
                "prompt": prompt,
                "height": height,
                "width": width,
                "modalities": ["image"],
            }
        )
    return result


def make_config(args: argparse.Namespace) -> OmniDiffusionConfig:
    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        sequence_parallel_size=args.ulysses_degree * args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
    )
    config = OmniDiffusionConfig(
        model=args.model,
        model_class_name=args.model_class_name,
        trust_remote_code=True,
        dtype=getattr(torch, args.dtype),
        distributed_executor_backend=args.distributed_executor_backend,
        enforce_eager=args.enforce_eager,
        parallel_config=parallel_config,
        quantization_config=args.quantization,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        output_type="pil",
        max_num_seqs=1,
    )
    config.enrich_config()
    return config


def make_sampling_params(
    args: argparse.Namespace,
    *,
    height: int,
    width: int,
    seed: int,
    enable_mixfusion: bool,
) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=height,
        width=width,
        seed=seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_provided=True,
        num_outputs_per_prompt=1,
        extra_args={
            "enable_mixfusion": enable_mixfusion,
            "use_system_prompt": args.use_system_prompt,
            "system_prompt": args.system_prompt,
        },
    )


async def run_request(
    engine: DiffusionEngine,
    prompts: list[OmniTextPrompt],
    sampling_params: OmniDiffusionSamplingParams,
    request_prefix: str,
) -> tuple[float, list[Any]]:
    request_id = f"{request_prefix}-{uuid.uuid4()}"
    requests = [
        OmniDiffusionRequest(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=f"{request_id}-{idx}",
        )
        for idx, prompt in enumerate(prompts)
    ]
    start = time.perf_counter()
    outputs = await asyncio.gather(*(engine.step(request) for request in requests))
    elapsed = time.perf_counter() - start
    return elapsed, [output for batch in outputs for output in batch]


async def run_independent(
    engine: DiffusionEngine,
    prompts: list[OmniTextPrompt],
    args: argparse.Namespace,
) -> tuple[float, list[Any]]:
    elapsed_total = 0.0
    all_outputs = []
    for idx, prompt in enumerate(prompts):
        height = int(prompt["height"])
        width = int(prompt["width"])
        params = make_sampling_params(
            args,
            height=height,
            width=width,
            seed=args.seed + idx,
            enable_mixfusion=False,
        )
        elapsed, outputs = await run_request(engine, [prompt], params, f"independent-{idx}")
        elapsed_total += elapsed
        all_outputs.extend(outputs)
    return elapsed_total, all_outputs


def validate_args(args: argparse.Namespace) -> None:
    sizes = parse_image_sizes(args.image_sizes)
    if len(sizes) < 2:
        raise ValueError("--image-sizes must contain at least two entries for the MixFusion batch path.")
    if args.iters < 1:
        raise ValueError("--iters must be >= 1.")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1.")
    if args.cfg_parallel_size != 1:
        raise ValueError("This MixFusion benchmark requires --cfg-parallel-size 1.")
    if args.ulysses_degree != 1 or args.ring_degree != 1:
        raise ValueError(
            "This MixFusion benchmark requires sequence parallel disabled: --ulysses-degree 1 --ring-degree 1."
        )
    if len(set(sizes)) == 1:
        raise ValueError("All requested image sizes are identical; use at least two resolutions to exercise MixFusion.")


async def run_mixfusion_batch(
    engine: DiffusionEngine,
    prompts: list[OmniTextPrompt],
    sizes: list[tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[float, list[Any]]:
    first_h, first_w = sizes[0]
    params = make_sampling_params(
        args,
        height=first_h,
        width=first_w,
        seed=args.seed,
        enable_mixfusion=True,
    )
    return await run_request(engine, prompts, params, "mixfusion-batch")


def maybe_save_outputs(outputs: list[Any], output_dir: str | None, prefix: str) -> None:
    if output_dir is None:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_idx = 0
    for output in outputs:
        images = getattr(output, "images", None) or []
        for image in images:
            image.save(out_dir / f"{prefix}_{image_idx}.png")
            image_idx += 1


def summarize_outputs(outputs: list[Any]) -> dict[str, Any]:
    stage_durations: list[dict[str, float]] = []
    peak_memory_mb = 0.0
    image_count = 0
    for output in outputs:
        durations = getattr(output, "stage_durations", None)
        if durations:
            stage_durations.append(dict(durations))
        peak_memory_mb = max(peak_memory_mb, float(getattr(output, "peak_memory_mb", 0.0) or 0.0))
        image_count += len(getattr(output, "images", None) or [])
    return {
        "num_outputs": len(outputs),
        "num_images": image_count,
        "peak_memory_mb": peak_memory_mb,
        "stage_durations": stage_durations,
    }


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    sizes = parse_image_sizes(args.image_sizes)
    prompts_text = parse_prompts(args.prompts, len(sizes))
    prompts = build_prompt_dicts(prompts_text, sizes)

    config = make_config(args)
    engine = DiffusionEngine.make_engine(config)
    try:
        if args.warmup:
            await run_independent(engine, prompts[:1], args)

        independent_times = []
        mixfusion_times = []
        independent_outputs = []
        mixfusion_outputs = []

        for _ in range(args.iters):
            elapsed, outputs = await run_independent(engine, prompts, args)
            independent_times.append(elapsed)
            independent_outputs = outputs

            elapsed, outputs = await run_mixfusion_batch(engine, prompts, sizes, args)
            mixfusion_times.append(elapsed)
            mixfusion_outputs = outputs

        maybe_save_outputs(independent_outputs, args.output_dir, "independent")
        maybe_save_outputs(mixfusion_outputs, args.output_dir, "mixfusion")

        independent_mean = sum(independent_times) / len(independent_times)
        mixfusion_mean = sum(mixfusion_times) / len(mixfusion_times)

        return {
            "config": {
                "model": args.model,
                "image_sizes": args.image_sizes,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "iters": args.iters,
                "tensor_parallel_size": args.tensor_parallel_size,
                "enable_expert_parallel": args.enable_expert_parallel,
                "quantization": args.quantization,
                "dtype": args.dtype,
                "model_class_name": args.model_class_name,
                "distributed_executor_backend": args.distributed_executor_backend,
                "enforce_eager": args.enforce_eager,
            },
            "time_s": {
                "independent_each_iter": independent_times,
                "mixfusion_batch_each_iter": mixfusion_times,
                "independent_mean": independent_mean,
                "mixfusion_batch_mean": mixfusion_mean,
            },
            "speedup": {
                "mixfusion_batch_vs_independent": independent_mean / mixfusion_mean,
            },
            "outputs": {
                "independent": summarize_outputs(independent_outputs),
                "mixfusion_batch": summarize_outputs(mixfusion_outputs),
            },
            "notes": [
                "independent runs each prompt as a separate diffusion request with seed+i.",
                "mixfusion_batch runs all prompts in one OmniDiffusionRequest with seed as the base seed.",
                "This measures the real loaded Hunyuan DiT path, including tokenizer, RoPE, scheduler, DiT, and VAE.",
                "It intentionally bypasses online stepwise scheduling; online mixed-resolution request batching still "
                "needs Hunyuan stepwise support.",
            ],
        }
    finally:
        engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark real HunyuanImage-3.0 MixFusion benefit.")
    parser.add_argument("--model", default="tencent/HunyuanImage-3.0-Instruct")
    parser.add_argument("--model-class-name", default="HunyuanImage3ForCausalMM")
    parser.add_argument("--image-sizes", default="1024x1024,512x512")
    parser.add_argument("--prompts", default=None, help="Prompts separated by '||'. Must match --image-sizes count.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--quantization", default="fp8")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--enable-expert-parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--cfg-parallel-size", type=int, default=1)
    parser.add_argument("--distributed-executor-backend", default="mp")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae-use-slicing", action="store_true")
    parser.add_argument("--vae-use-tiling", action="store_true")
    parser.add_argument("--enable-diffusion-pipeline-profiler", action="store_true")
    parser.add_argument("--use-system-prompt", default=None)
    parser.add_argument("--system-prompt", default=None)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    result = asyncio.run(benchmark(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
