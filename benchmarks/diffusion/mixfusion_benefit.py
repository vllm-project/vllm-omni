# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Benchmark the real-model MixFusion benefit for Qwen-Image and HunyuanImage-3.0.

Unlike ``mixfusion_kernel_benchmark.py``, which times a synthetic DiT-like stack,
this loads the real pipeline through ``DiffusionEngine`` and runs the whole
tokenizer/RoPE/CFG/scheduler/DiT/VAE path. For one resolution set it compares two
ways of serving the same prompts: ``independent`` (one request at a time, so the
DiT stage never batches) and ``mixfusion_batch`` (the same prompts admitted
concurrently, so the DiT stage receives one mixed-resolution batch).

``--family`` selects the model family. Qwen-Image carries a request's resolution
in the sampling params, so stepwise scheduling batches mixed resolutions under the
MixFusion-relaxed key. HunyuanImage-3.0 reads it per prompt, so the whole batch
shares one set of sampling params. Small-GCD resolution sets explode the chunk
count, so the Qwen-Image path can reject them before spending GPU time.

Example:
    DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \\
    python benchmarks/diffusion/mixfusion_benefit.py \\
      --family qwen --model Qwen/Qwen-Image \\
      --image-sizes 1024x1024,1024x768 --steps 8 --iters 3

    python benchmarks/diffusion/mixfusion_benefit.py \\
      --family hunyuan --model tencent/HunyuanImage-3.0-Instruct \\
      --image-sizes 1024x1024,512x512 --steps 20 --iters 3 \\
      --tensor-parallel-size 4 --quantization fp8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

DEFAULT_PROMPTS = [
    "A cinematic photo of a glass observatory on Mars at sunrise",
    "A watercolor painting of a quiet mountain lake",
    "A detailed product photo of a transparent mechanical keyboard",
    "A cozy reading room with warm sunlight and plants",
]

# Reported verbatim from the parsed args.
CONFIG_KEYS = (
    "model",
    "model_class_name",
    "steps",
    "iters",
    "guidance_scale",
    "seed",
    "dtype",
    "quantization",
    "tensor_parallel_size",
    "enable_expert_parallel",
    "distributed_executor_backend",
    "enforce_eager",
    "output_type",
    "enable_mixfusion",
)


def parse_image_sizes(raw: str) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        height_raw, width_raw = item.lower().split("x", maxsplit=1)
        sizes.append((int(height_raw), int(width_raw)))
    if not sizes:
        raise ValueError("--image-sizes must contain at least one HxW entry.")
    return sizes


def parse_prompts(raw: str | None, count: int) -> list[str]:
    if raw is None:
        prompts = list(DEFAULT_PROMPTS[:count])
        while len(prompts) < count:
            prompts.append(f"A high quality image sample {len(prompts)}")
        return prompts
    prompts = [item.strip() for item in raw.split("||") if item.strip()]
    if len(prompts) != count:
        raise ValueError(f"Expected {count} prompts separated by '||', got {len(prompts)}.")
    return prompts


def qwen_image_token_len(height: int, width: int, vae_scale_factor: int) -> int:
    alignment = vae_scale_factor * 2
    height = (height // alignment) * alignment
    width = (width // alignment) * alignment
    return (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2)


def predict_qwen_plan(
    sizes: list[tuple[int, int]],
    *,
    vae_scale_factor: int,
    min_chunk_tokens: int,
    max_chunks: int,
) -> dict[str, Any]:
    """Predict the MixFusion plan for a resolution set without loading a model."""
    seq_lens = [qwen_image_token_len(height, width, vae_scale_factor) for height, width in sizes]
    chunk_size = math.gcd(*seq_lens)
    chunk_count = sum(seq_len // chunk_size for seq_len in seq_lens)

    accepted, reason = True, "ok"
    if chunk_size < min_chunk_tokens:
        accepted, reason = False, f"chunk_size={chunk_size} < min_chunk_tokens={min_chunk_tokens}"
    elif chunk_count > max_chunks:
        accepted, reason = False, f"chunk_count={chunk_count} > max_chunks={max_chunks}"
    return {
        "accepted": accepted,
        "reason": reason,
        "seq_lens": seq_lens,
        "chunk_size": chunk_size,
        "chunk_count": chunk_count,
    }


def build_prompt_dicts(
    prompts: list[str],
    sizes: list[tuple[int, int]],
    family: Family,
) -> list[OmniTextPrompt]:
    if not family.resolution_in_prompt:
        return [{"prompt": prompt, "modalities": ["image"]} for prompt in prompts]
    return [
        {"prompt": prompt, "height": height, "width": width, "modalities": ["image"]}
        for prompt, (height, width) in zip(prompts, sizes, strict=True)
    ]


def make_config(args: argparse.Namespace, family: Family) -> OmniDiffusionConfig:
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
        output_type=args.output_type,
        **family.config_fields(args),
    )
    config.enrich_config()
    return config


def make_sampling_params(
    args: argparse.Namespace,
    family: Family,
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
        extra_args=family.extra_args(args, enable_mixfusion),
        **family.sampling_fields,
    )


def build_requests(
    args: argparse.Namespace,
    family: Family,
    prompts: list[OmniTextPrompt],
    sizes: list[tuple[int, int]],
    request_prefix: str,
    enable_mixfusion: bool,
    seed_offset: int = 0,
) -> list[OmniDiffusionRequest]:
    """Build one request per prompt, each carrying its own resolution."""
    if family.resolution_in_prompt:
        # The resolution travels in the prompt dict, so the whole batch shares a
        # single set of sampling params, anchored on the first resolution.
        height, width = sizes[0]
        params = make_sampling_params(
            args,
            family,
            height=height,
            width=width,
            seed=args.seed + seed_offset,
            enable_mixfusion=enable_mixfusion,
        )
        return [
            OmniDiffusionRequest(
                prompt=prompt,
                sampling_params=params,
                request_id=f"{request_prefix}-{idx}-{uuid.uuid4()}",
            )
            for idx, prompt in enumerate(prompts)
        ]

    return [
        OmniDiffusionRequest(
            prompt=prompt,
            sampling_params=make_sampling_params(
                args,
                family,
                height=height,
                width=width,
                seed=args.seed + seed_offset + idx,
                enable_mixfusion=enable_mixfusion,
            ),
            request_id=f"{request_prefix}-{idx}-{uuid.uuid4()}",
        )
        for idx, (prompt, (height, width)) in enumerate(zip(prompts, sizes, strict=True))
    ]


async def run_requests(
    engine: DiffusionEngine,
    requests: list[OmniDiffusionRequest],
) -> tuple[float, list[Any]]:
    start = time.perf_counter()
    grouped_outputs = await asyncio.gather(*(engine.step(request) for request in requests))
    elapsed = time.perf_counter() - start
    return elapsed, [output for outputs in grouped_outputs for output in outputs]


async def run_independent(
    engine: DiffusionEngine,
    prompts: list[OmniTextPrompt],
    sizes: list[tuple[int, int]],
    args: argparse.Namespace,
    family: Family,
) -> tuple[float, list[Any]]:
    """Serve the prompts one at a time, so no two requests share a DiT batch."""
    elapsed_total = 0.0
    all_outputs: list[Any] = []
    for idx in range(len(prompts)):
        requests = build_requests(
            args,
            family,
            prompts[idx : idx + 1],
            sizes[idx : idx + 1],
            f"independent-{idx}",
            enable_mixfusion=family.independent_mixfusion,
            seed_offset=idx,
        )
        elapsed, outputs = await run_requests(engine, requests)
        elapsed_total += elapsed
        all_outputs.extend(outputs)
    return elapsed_total, all_outputs


async def run_mixfusion_batch(
    engine: DiffusionEngine,
    prompts: list[OmniTextPrompt],
    sizes: list[tuple[int, int]],
    args: argparse.Namespace,
    family: Family,
) -> tuple[float, list[Any]]:
    """Serve every prompt concurrently so the DiT stage sees one mixed batch."""
    requests = build_requests(
        args,
        family,
        prompts,
        sizes,
        "mixfusion-batch",
        enable_mixfusion=args.enable_mixfusion,
    )
    return await run_requests(engine, requests)


def summarize_outputs(outputs: list[Any]) -> dict[str, Any]:
    peak_memory_mb = 0.0
    image_count = 0
    for output in outputs:
        peak_memory_mb = max(peak_memory_mb, float(getattr(output, "peak_memory_mb", 0.0) or 0.0))
        image_count += len(getattr(output, "images", None) or [])
    return {
        "num_outputs": len(outputs),
        "num_images": image_count,
        "peak_memory_mb": peak_memory_mb,
        "stage_durations": [
            dict(durations) for output in outputs if (durations := getattr(output, "stage_durations", None))
        ],
    }


def validate_args(args: argparse.Namespace, family: Family, sizes: list[tuple[int, int]]) -> None:
    if args.iters < 1:
        raise ValueError("--iters must be >= 1.")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1.")
    if args.cfg_parallel_size != 1:
        raise ValueError("This benchmark requires --cfg-parallel-size 1.")
    if args.ulysses_degree != 1 or args.ring_degree != 1:
        raise ValueError("This benchmark requires --ulysses-degree 1 --ring-degree 1.")
    family.validate(args, sizes)


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    family = FAMILIES[args.family]
    sizes = parse_image_sizes(args.image_sizes)
    validate_args(args, family, sizes)
    prompts = build_prompt_dicts(parse_prompts(args.prompts, len(sizes)), sizes, family)
    candidate = family.candidate(args, sizes)
    backend = os.environ.get("DIFFUSION_ATTENTION_BACKEND", "")

    if candidate is not None and not candidate["accepted"] and args.skip_rejected_mixfusion_cases:
        return {
            "skipped": True,
            "skip_reason": candidate["reason"],
            "family": args.family,
            "backend": backend,
            "image_sizes": sizes,
            "candidate": candidate,
        }

    config = make_config(args, family)
    engine = DiffusionEngine.make_engine(config)
    try:
        for _ in range(args.warmup):
            await run_independent(engine, prompts[:1], sizes[:1], args, family)

        independent_times: list[float] = []
        mixfusion_times: list[float] = []
        independent_outputs: list[Any] = []
        mixfusion_outputs: list[Any] = []
        for _ in range(args.iters):
            elapsed, independent_outputs = await run_independent(engine, prompts, sizes, args, family)
            independent_times.append(elapsed)
            elapsed, mixfusion_outputs = await run_mixfusion_batch(engine, prompts, sizes, args, family)
            mixfusion_times.append(elapsed)

        independent_mean = sum(independent_times) / len(independent_times)
        mixfusion_mean = sum(mixfusion_times) / len(mixfusion_times)
        return {
            "skipped": False,
            "family": args.family,
            "backend": backend,
            "config": {key: getattr(args, key) for key in CONFIG_KEYS}
            | {
                "image_sizes": sizes,
                "max_num_seqs": family.config_fields(args)["max_num_seqs"],
            },
            "candidate": candidate,
            "time_s": {
                "independent_each_iter": independent_times,
                "mixfusion_batch_each_iter": mixfusion_times,
                "independent_mean": independent_mean,
                "mixfusion_batch_mean": mixfusion_mean,
            },
            "speedup": {
                "mixfusion_batch_vs_independent": (independent_mean / mixfusion_mean if mixfusion_mean > 0 else None),
            },
            "outputs": {
                "independent": summarize_outputs(independent_outputs),
                "mixfusion_batch": summarize_outputs(mixfusion_outputs),
            },
            "notes": [
                "independent runs each prompt as its own request, one after another, with seed+i.",
                "mixfusion_batch runs all prompts concurrently, so the DiT stage sees one mixed batch.",
                "This measures the real loaded pipeline: tokenizer, RoPE, CFG, scheduler, DiT and VAE.",
                *family.notes,
            ],
        }
    finally:
        engine.close()


def _qwen_extra_args(args: argparse.Namespace, enable_mixfusion: bool) -> dict[str, Any]:
    if not enable_mixfusion:
        return {}
    return {
        "enable_mixfusion": True,
        "mixfusion_min_chunk_tokens": args.mixfusion_min_chunk_tokens,
        "mixfusion_max_chunks": args.mixfusion_max_chunks,
    }


def _qwen_config_fields(args: argparse.Namespace) -> dict[str, Any]:
    return {"step_execution": True, "max_num_seqs": args.max_num_seqs}


def _qwen_validate(args: argparse.Namespace, sizes: list[tuple[int, int]]) -> None:
    if len(sizes) < 2:
        raise ValueError("--image-sizes must contain at least two entries.")
    if args.max_num_seqs < len(sizes):
        raise ValueError("--max-num-seqs must be >= the number of requested image sizes.")


def _qwen_candidate(args: argparse.Namespace, sizes: list[tuple[int, int]]) -> dict[str, Any]:
    return predict_qwen_plan(
        sizes,
        vae_scale_factor=args.vae_scale_factor,
        min_chunk_tokens=args.mixfusion_min_chunk_tokens,
        max_chunks=args.mixfusion_max_chunks,
    )


def _hunyuan_extra_args(args: argparse.Namespace, enable_mixfusion: bool) -> dict[str, Any]:
    return {
        "enable_mixfusion": enable_mixfusion,
        "use_system_prompt": args.use_system_prompt,
        "system_prompt": args.system_prompt,
    }


def _hunyuan_config_fields(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "vae_use_slicing": args.vae_use_slicing,
        "vae_use_tiling": args.vae_use_tiling,
        "max_num_seqs": 1,
    }


def _hunyuan_validate(args: argparse.Namespace, sizes: list[tuple[int, int]]) -> None:
    if len(sizes) < 2:
        raise ValueError("--image-sizes must contain at least two entries.")
    if len(set(sizes)) == 1:
        raise ValueError("All requested image sizes are identical; use two or more resolutions.")


@dataclass(frozen=True)
class Family:
    """Per-family knobs; the benchmark loop itself is shared."""

    default_model: str
    default_model_class_name: str
    defaults: dict[str, Any]
    # Where a request's resolution lives: the sampling params, or the prompt dict.
    resolution_in_prompt: bool
    # Whether the one-request-at-a-time baseline also runs the MixFusion path.
    # Qwen-Image keeps it on, so the speedup isolates the batching gain; the
    # HunyuanImage-3.0 single-request path does not use it.
    independent_mixfusion: bool
    extra_args: Callable[[argparse.Namespace, bool], dict[str, Any]]
    sampling_fields: dict[str, Any]
    config_fields: Callable[[argparse.Namespace], dict[str, Any]]
    validate: Callable[[argparse.Namespace, list[tuple[int, int]]], None]
    candidate: Callable[[argparse.Namespace, list[tuple[int, int]]], dict[str, Any] | None]
    notes: list[str]


FAMILIES: dict[str, Family] = {
    "qwen": Family(
        default_model="Qwen/Qwen-Image",
        default_model_class_name="QwenImagePipeline",
        defaults={
            "image_sizes": "1024x1024,1024x768",
            "steps": 8,
            "guidance_scale": 1.0,
            "tensor_parallel_size": 1,
            "max_num_seqs": 2,
            "quantization": "",
            "enforce_eager": False,
            "enable_expert_parallel": False,
        },
        resolution_in_prompt=False,
        independent_mixfusion=True,
        extra_args=_qwen_extra_args,
        sampling_fields={"true_cfg_scale": 1.0, "max_sequence_length": 1024},
        config_fields=_qwen_config_fields,
        validate=_qwen_validate,
        candidate=_qwen_candidate,
        notes=[
            "Qwen-Image carries the resolution in the sampling params, so stepwise scheduling "
            "batches mixed resolutions under the MixFusion-relaxed sampling-params key.",
        ],
    ),
    "hunyuan": Family(
        default_model="tencent/HunyuanImage-3.0-Instruct",
        default_model_class_name="HunyuanImage3ForCausalMM",
        defaults={
            "image_sizes": "1024x1024,512x512",
            "steps": 20,
            "guidance_scale": 5.0,
            "tensor_parallel_size": 4,
            "max_num_seqs": 1,
            "quantization": "fp8",
            "enforce_eager": True,
            "enable_expert_parallel": True,
        },
        resolution_in_prompt=True,
        independent_mixfusion=False,
        extra_args=_hunyuan_extra_args,
        sampling_fields={},
        config_fields=_hunyuan_config_fields,
        validate=_hunyuan_validate,
        candidate=lambda args, sizes: None,
        notes=[
            "HunyuanImage-3.0 reads the resolution from each prompt dict, so the batch shares one "
            "set of sampling params and bypasses online stepwise mixed-resolution scheduling.",
        ],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILIES), default="qwen")
    parser.add_argument("--model", help="Defaults to the model of the selected --family.")
    parser.add_argument("--model-class-name", help="Defaults to the model class of the selected --family.")
    parser.add_argument("--image-sizes", help="Comma-separated HxW list, e.g. '1024x1024,1024x768'.")
    parser.add_argument("--prompts", help="Prompts separated by '||'. Must match --image-sizes count.")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--output-type", default="pil", choices=["pil", "latent"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--quantization")
    parser.add_argument("--distributed-executor-backend", default="mp")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--cfg-parallel-size", type=int, default=1)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--enable-expert-parallel", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-diffusion-pipeline-profiler", action="store_true")
    parser.add_argument(
        "--enable-mixfusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MixFusion for the batched path. Disable it for a batching-only control run.",
    )
    parser.add_argument("--json-output", help="Also write the result JSON to this path.")
    # Qwen-Image only.
    parser.add_argument("--vae-scale-factor", type=int, default=8)
    parser.add_argument("--mixfusion-min-chunk-tokens", type=int, default=256)
    parser.add_argument("--mixfusion-max-chunks", type=int, default=128)
    parser.add_argument(
        "--skip-rejected-mixfusion-cases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip resolution sets whose predicted MixFusion plan is rejected (Qwen-Image only).",
    )
    # HunyuanImage-3.0 only.
    parser.add_argument("--vae-use-slicing", action="store_true")
    parser.add_argument("--vae-use-tiling", action="store_true")
    parser.add_argument("--use-system-prompt", default=None)
    parser.add_argument("--system-prompt", default=None)

    args = parser.parse_args()
    family = FAMILIES[args.family]
    if args.model is None:
        args.model = family.default_model
    if args.model_class_name is None:
        args.model_class_name = family.default_model_class_name
    for key, value in family.defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    result = asyncio.run(benchmark(args))
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
