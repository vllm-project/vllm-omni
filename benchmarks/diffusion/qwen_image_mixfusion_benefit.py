# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Qwen-Image mixed-resolution batching benefit.

This script is meant for measuring whether a MixFusion-style mixed-resolution
batch is worth pursuing on Qwen-Image without paying HunyuanImage-3.0's model
size cost. It uses the real Qwen-Image DiffusionEngine stepwise path:

* prompts are encoded once per independent request in prepare_encode();
* denoising requests are admitted concurrently so the DiT stage can batch them;
* the selected attention backend is exercised through the normal model path;
* small-GCD cases are filtered before running to avoid chunk explosion.

Example:
    DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
    python benchmarks/diffusion/qwen_image_mixfusion_benefit.py \
      --model Qwen/Qwen-Image \
      --image-sizes 1024x1024,1024x768 \
      --steps 8 --iters 3 --tensor-parallel-size 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
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
    sizes: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        h_raw, w_raw = item.lower().split("x", maxsplit=1)
        sizes.append((int(h_raw), int(w_raw)))
    if len(sizes) < 2:
        raise ValueError("--image-sizes must contain at least two entries.")
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


def normalize_qwen_size(height: int, width: int, vae_scale_factor: int) -> tuple[int, int]:
    alignment = vae_scale_factor * 2
    return (height // alignment) * alignment, (width // alignment) * alignment


def qwen_image_token_len(height: int, width: int, vae_scale_factor: int) -> int:
    height, width = normalize_qwen_size(height, width, vae_scale_factor)
    return (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2)


def evaluate_mixfusion_candidate(
    sizes: list[tuple[int, int]],
    *,
    vae_scale_factor: int,
    min_chunk_tokens: int,
    max_chunks: int,
) -> dict[str, Any]:
    seq_lens = [qwen_image_token_len(height, width, vae_scale_factor) for height, width in sizes]
    chunk_size = seq_lens[0]
    for seq_len in seq_lens[1:]:
        chunk_size = math.gcd(chunk_size, seq_len)
    chunk_count = sum(seq_len // chunk_size for seq_len in seq_lens)

    max_seq_len = max(seq_lens)
    padded_token_work = len(seq_lens) * max_seq_len
    packed_token_work = sum(seq_lens)
    padded_attention_work = len(seq_lens) * max_seq_len * max_seq_len
    packed_attention_work = sum(seq_len * seq_len for seq_len in seq_lens)
    token_saving = 1.0 - packed_token_work / padded_token_work
    attention_saving = 1.0 - packed_attention_work / padded_attention_work

    accepted = True
    reason = "ok"
    if chunk_size < min_chunk_tokens:
        accepted = False
        reason = f"chunk_size={chunk_size} < min_chunk_tokens={min_chunk_tokens}"
    elif chunk_count > max_chunks:
        accepted = False
        reason = f"chunk_count={chunk_count} > max_chunks={max_chunks}"
    elif token_saving <= 0.0 and attention_saving <= 0.0:
        accepted = False
        reason = "no positive token or attention padding saving"

    return {
        "accepted": accepted,
        "reason": reason,
        "seq_lens": seq_lens,
        "chunk_size": chunk_size,
        "chunk_count": chunk_count,
        "token_saving": token_saving,
        "attention_saving": attention_saving,
        "normalized_sizes": [normalize_qwen_size(h, w, vae_scale_factor) for h, w in sizes],
    }


def build_prompt_dicts(prompts: list[str]) -> list[OmniTextPrompt]:
    return [{"prompt": prompt, "modalities": ["image"]} for prompt in prompts]


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
        output_type=args.output_type,
        step_execution=True,
        max_num_seqs=args.max_num_seqs,
    )
    config.enrich_config()
    return config


def make_sampling_params(
    args: argparse.Namespace,
    *,
    height: int,
    width: int,
    seed: int,
) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=height,
        width=width,
        seed=seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_provided=True,
        true_cfg_scale=args.true_cfg_scale,
        num_outputs_per_prompt=1,
        max_sequence_length=args.max_sequence_length,
        output_type=args.output_type,
        extra_args={
            "enable_mixfusion": True,
            "mixfusion_min_chunk_tokens": args.mixfusion_min_chunk_tokens,
            "mixfusion_max_chunks": args.mixfusion_max_chunks,
        },
    )


def make_request(
    prompt: OmniTextPrompt,
    sampling_params: OmniDiffusionSamplingParams,
    request_prefix: str,
) -> OmniDiffusionRequest:
    request_id = f"{request_prefix}-{uuid.uuid4()}"
    return OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    )


async def run_serial(
    engine: DiffusionEngine,
    prompts: list[OmniTextPrompt],
    sizes: list[tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[float, list[Any]]:
    elapsed_total = 0.0
    all_outputs: list[Any] = []
    for idx, (prompt, (height, width)) in enumerate(zip(prompts, sizes, strict=True)):
        params = make_sampling_params(args, height=height, width=width, seed=args.seed + idx)
        request = make_request(prompt, params, f"qwen-serial-{idx}")
        start = time.perf_counter()
        outputs = await engine.step(request)
        elapsed_total += time.perf_counter() - start
        all_outputs.extend(outputs)
    return elapsed_total, all_outputs


async def run_concurrent_batch(
    engine: DiffusionEngine,
    prompts: list[OmniTextPrompt],
    sizes: list[tuple[int, int]],
    args: argparse.Namespace,
) -> tuple[float, list[Any]]:
    requests: list[OmniDiffusionRequest] = []
    for idx, (prompt, (height, width)) in enumerate(zip(prompts, sizes, strict=True)):
        params = make_sampling_params(args, height=height, width=width, seed=args.seed + idx)
        requests.append(make_request(prompt, params, f"qwen-batched-{idx}"))

    start = time.perf_counter()
    grouped_outputs = await asyncio.gather(*(engine.step(request) for request in requests))
    elapsed = time.perf_counter() - start
    return elapsed, [output for outputs in grouped_outputs for output in outputs]


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
    }


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


def validate_args(args: argparse.Namespace) -> None:
    sizes = parse_image_sizes(args.image_sizes)
    if args.iters < 1:
        raise ValueError("--iters must be >= 1.")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1.")
    if args.max_num_seqs < len(sizes):
        raise ValueError("--max-num-seqs must be >= the number of requested image sizes.")
    if args.cfg_parallel_size != 1:
        raise ValueError("This benchmark requires --cfg-parallel-size 1.")
    if args.ulysses_degree != 1 or args.ring_degree != 1:
        raise ValueError("This benchmark requires sequence parallel disabled.")


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    sizes = parse_image_sizes(args.image_sizes)
    prompts = build_prompt_dicts(parse_prompts(args.prompts, len(sizes)))
    candidate = evaluate_mixfusion_candidate(
        sizes,
        vae_scale_factor=args.vae_scale_factor,
        min_chunk_tokens=args.mixfusion_min_chunk_tokens,
        max_chunks=args.mixfusion_max_chunks,
    )

    if args.skip_rejected_mixfusion_cases and not candidate["accepted"]:
        return {
            "skipped": True,
            "skip_reason": candidate["reason"],
            "backend": os.environ.get("DIFFUSION_ATTENTION_BACKEND", ""),
            "candidate": candidate,
            "image_sizes": sizes,
        }

    config = make_config(args)
    engine = DiffusionEngine.make_engine(config)
    try:
        if args.warmup > 0:
            for _ in range(args.warmup):
                await run_serial(engine, prompts[:1], sizes[:1], args)

        serial_times: list[float] = []
        batched_times: list[float] = []
        serial_outputs: list[Any] = []
        batched_outputs: list[Any] = []
        for _ in range(args.iters):
            elapsed, outputs = await run_serial(engine, prompts, sizes, args)
            serial_times.append(elapsed)
            serial_outputs = outputs

            elapsed, outputs = await run_concurrent_batch(engine, prompts, sizes, args)
            batched_times.append(elapsed)
            batched_outputs = outputs

        maybe_save_outputs(serial_outputs, args.output_dir, "serial")
        maybe_save_outputs(batched_outputs, args.output_dir, "batched")

        serial_avg = sum(serial_times) / len(serial_times)
        batched_avg = sum(batched_times) / len(batched_times)
        return {
            "skipped": False,
            "backend": os.environ.get("DIFFUSION_ATTENTION_BACKEND", ""),
            "model": args.model,
            "model_class_name": args.model_class_name,
            "image_sizes": sizes,
            "candidate": candidate,
            "steps": args.steps,
            "iters": args.iters,
            "max_num_seqs": args.max_num_seqs,
            "tensor_parallel_size": args.tensor_parallel_size,
            "serial_times_s": serial_times,
            "batched_times_s": batched_times,
            "serial_avg_s": serial_avg,
            "batched_avg_s": batched_avg,
            "speedup": serial_avg / batched_avg if batched_avg > 0 else None,
            "serial_outputs": summarize_outputs(serial_outputs),
            "batched_outputs": summarize_outputs(batched_outputs),
        }
    finally:
        engine.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen-Image")
    parser.add_argument("--model-class-name", default="QwenImagePipeline")
    parser.add_argument("--image-sizes", default="1024x1024,1024x768")
    parser.add_argument("--prompts")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--output-type", default="pil", choices=["pil", "latent"])
    parser.add_argument("--output-dir")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--quantization")
    parser.add_argument("--distributed-executor-backend", default="mp")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cfg-parallel-size", type=int, default=1)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--ring-degree", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--vae-scale-factor", type=int, default=8)
    parser.add_argument("--mixfusion-min-chunk-tokens", type=int, default=256)
    parser.add_argument("--mixfusion-max-chunks", type=int, default=128)
    parser.add_argument("--skip-rejected-mixfusion-cases", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-expert-parallel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--json-output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(benchmark(args))
    text = json.dumps(result, indent=2)
    print(text)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
