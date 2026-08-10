# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce MiniMax-H3 request-mode and DLO/DP2 lifecycle behavior.

Examples:

    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
    VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=600 \
    CUDA_VISIBLE_DEVICES=0,1 \
    python examples/offline_inference/minimax_h3/dlo_dp2_lifecycle.py \
        --model /path/to/MiniMax-H3/FL2VA --mode dlo-dp2

    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
    CUDA_VISIBLE_DEVICES=0,1 \
    python examples/offline_inference/minimax_h3/dlo_dp2_lifecycle.py \
        --model /path/to/MiniMax-H3/FL2VA --mode request
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import numpy as np

from vllm_omni.entrypoints.async_omni import AsyncOmni

DEFAULT_PROMPTS = (
    "At night, three cats march into a bedroom playing tiny brass instruments, "
    "then abruptly file out, with synchronized room ambience.",
    "A paper boat crosses a rain-filled street while distant traffic and water sounds remain synchronized.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to MiniMax-H3/FL2VA")
    parser.add_argument(
        "--mode",
        choices=("dlo-dp2", "request"),
        required=True,
        help="DLO AllGather DP2 or ordinary non-DLO TP2 request mode",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--batch-wait-ms", type=float, default=500.0)
    parser.add_argument("--init-timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    common: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "num_gpus": 2,
        "usp": 1,
        "ring": 1,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": True,
        "diffusion_attention_backend": "CUDNN_ATTN",
        "request_batch_max_wait_ms": args.batch_wait_ms,
        "enforce_eager": True,
        "stage_init_timeout": args.init_timeout,
        "init_timeout": args.init_timeout,
    }
    if args.mode == "dlo-dp2":
        common.update(
            tensor_parallel_size=1,
            data_parallel_size=2,
            text_encoder_tp_size=1,
            vae_patch_parallel_size=1,
            enable_distributed_layerwise_offload=True,
            dlo_use_allgather=True,
            dlo_resident_layers=0,
        )
    else:
        common.update(
            tensor_parallel_size=2,
            data_parallel_size=1,
            text_encoder_tp_size=2,
            vae_patch_parallel_size=2,
            enable_distributed_layerwise_offload=False,
        )
    return common


def sampling_params(
    engine: AsyncOmni,
    args: argparse.Namespace,
    seed: int,
) -> list[Any]:
    params = copy.deepcopy(engine.default_sampling_params_list)
    diffusion = params[0]
    diffusion.width = args.width
    diffusion.height = args.height
    diffusion.fps = 24
    diffusion.num_inference_steps = args.steps
    diffusion.seed = seed
    diffusion.extra_args = {
        "task": "t2va",
        "duration": args.duration,
        "aspect_ratio": "16:9",
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
    }
    return params


async def generate_one(
    engine: AsyncOmni,
    args: argparse.Namespace,
    *,
    request_id: str,
    prompt: str,
    seed: int,
) -> Any:
    final_output = None
    async for output in engine.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=sampling_params(engine, args, seed),
    ):
        if output.finished:
            final_output = output
    if final_output is None:
        raise RuntimeError(f"{request_id} finished without an output")
    return final_output


def output_summary(output: Any, args: argparse.Namespace) -> dict[str, Any]:
    if not output.images:
        raise RuntimeError(f"{output.request_id} returned no video")
    frames = np.asarray(output.images[0])
    audio = np.asarray(output.multimodal_output.get("audio"))
    if frames.ndim != 4 or tuple(frames.shape[1:]) != (
        args.height,
        args.width,
        3,
    ):
        raise RuntimeError(f"{output.request_id} returned invalid video shape {tuple(frames.shape)}")
    if audio.ndim != 3 or tuple(audio.shape[:2]) != (1, 2):
        raise RuntimeError(f"{output.request_id} returned invalid audio shape {tuple(audio.shape)}")
    if args.duration == 5.0 and args.height == 768 and args.width == 1344:
        if tuple(frames.shape) != (124, 768, 1344, 3) or tuple(audio.shape) != (1, 2, 165600):
            raise RuntimeError(
                f"{output.request_id} default shape mismatch: video={tuple(frames.shape)}, audio={tuple(audio.shape)}"
            )
    return {
        "request_id": output.request_id,
        "frames_shape": list(frames.shape),
        "audio_shape": list(audio.shape),
        "peak_memory_mb": output.peak_memory_mb,
        "stage_durations": output.stage_durations,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    engine = AsyncOmni(**engine_kwargs(args))
    summary: dict[str, Any] = {
        "mode": args.mode,
        "engine_kwargs": engine_kwargs(args),
    }
    try:
        if args.mode == "dlo-dp2":
            started = time.perf_counter()
            asymmetric = await asyncio.gather(
                generate_one(
                    engine,
                    args,
                    request_id="invalid-empty",
                    prompt="",
                    seed=1001,
                ),
                generate_one(
                    engine,
                    args,
                    request_id="invalid-peer",
                    prompt=DEFAULT_PROMPTS[0],
                    seed=1002,
                ),
                return_exceptions=True,
            )
            summary["asymmetric_wave_s"] = time.perf_counter() - started
            summary["asymmetric_errors"] = [
                f"{type(result).__name__}: {result}" for result in asymmetric if isinstance(result, BaseException)
            ]
            if len(summary["asymmetric_errors"]) != 2:
                raise RuntimeError("Both requests in the asymmetric DP wave must fail before dispatch")

        started = time.perf_counter()
        request_count = 2 if args.mode == "dlo-dp2" else 1
        outputs = await asyncio.gather(
            *(
                generate_one(
                    engine,
                    args,
                    request_id=f"recovery-{index}",
                    prompt=DEFAULT_PROMPTS[index],
                    seed=2000 + index,
                )
                for index in range(request_count)
            )
        )
        summary["valid_wave_s"] = time.perf_counter() - started
        summary["outputs"] = [output_summary(output, args) for output in outputs]
    finally:
        started = time.perf_counter()
        engine.close()
        summary["shutdown_s"] = time.perf_counter() - started

    children = multiprocessing.active_children()
    summary["active_children"] = [{"name": child.name, "pid": child.pid} for child in children]
    if children:
        raise RuntimeError(f"Worker processes remain after shutdown: {children}")
    return summary


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run(args))
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(f"E2E_RESULT {rendered}", flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
