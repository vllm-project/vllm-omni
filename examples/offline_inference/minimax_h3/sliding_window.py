# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax-H3 sliding-window generation for videos longer than 15 seconds.

Each window stays inside the native 4-15 s contract and is decoded on its own.
The frame the next window starts on is taken from the decoded previous window
and conditions it as a first-frame keyframe, and the first latents of the
shared span (video and audio) are held on the previous window's tail while it
denoises. On concatenation the previous window is kept through those held
frames, video and audio cross-fade to the new window for up to half a second,
the rest of the span is the new window's, and its audio onset is lifted towards
the previous level. This runs in request mode (no --step-execution).

Example:

    CUDA_VISIBLE_DEVICES=0,1 \
    python examples/offline_inference/minimax_h3/sliding_window.py \
        --model /path/to/MiniMax-H3/FL2VA --tp-size 2 \
        --duration 30 --output cliff_30s.mp4
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import time
from pathlib import Path
from typing import Any

import numpy as np

from vllm_omni.entrypoints.async_omni import AsyncOmni

DEFAULT_PROMPT = (
    "A drone shot gliding over a coastal cliff at sunrise, waves crashing below, with wind and seabird ambience."
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def num_segments_value(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an int or 'auto'") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to MiniMax-H3/FL2VA")
    parser.add_argument("--tp-size", type=positive_int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--duration", type=float, default=30.0, help="Total seconds (>15)")
    parser.add_argument(
        "--num-segments",
        type=num_segments_value,
        default=None,
        help="int >= 2 or 'auto'; defaults to 'auto' when duration > 15",
    )
    parser.add_argument(
        "--overlap-frames",
        type=int,
        default=None,
        help="Overlap request in frames; unset uses the server default (58)",
    )
    parser.add_argument("--window-duration", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-wait-ms", type=float, default=500.0)
    parser.add_argument("--init-timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "trust_remote_code": True,
        "num_gpus": args.tp_size,
        "tensor_parallel_size": args.tp_size,
        "data_parallel_size": 1,
        "text_encoder_tp_size": args.tp_size,
        "vae_patch_parallel_size": args.tp_size,
        "usp": 1,
        "ring": 1,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": True,
        "diffusion_attention_backend": "CUDNN_ATTN",
        "request_batch_max_wait_ms": args.batch_wait_ms,
        "enforce_eager": True,
        "enable_distributed_layerwise_offload": False,
        "stage_init_timeout": args.init_timeout,
        "init_timeout": args.init_timeout,
    }


def sampling_params(engine: AsyncOmni, args: argparse.Namespace) -> list[Any]:
    params = copy.deepcopy(engine.default_sampling_params_list)
    diffusion = params[0]
    diffusion.width = args.width
    diffusion.height = args.height
    diffusion.fps = 24
    diffusion.num_inference_steps = args.steps
    diffusion.seed = args.seed
    extra_args: dict[str, Any] = {
        "task": "t2va",
        "duration": args.duration,
        "aspect_ratio": "16:9",
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "window_duration": args.window_duration,
    }
    if args.num_segments is not None:
        extra_args["num_segments"] = args.num_segments
    if args.overlap_frames is not None:
        extra_args["overlap_frames"] = args.overlap_frames
    diffusion.extra_args = extra_args
    return params


async def generate(engine: AsyncOmni, args: argparse.Namespace) -> Any:
    final_output = None
    async for output in engine.generate(
        prompt=DEFAULT_PROMPT,
        request_id="minimax-h3-sliding-window",
        sampling_params_list=sampling_params(engine, args),
    ):
        if output.finished:
            final_output = output
    if final_output is None:
        raise RuntimeError("request finished without an output")
    return final_output


async def run(args: argparse.Namespace) -> dict[str, Any]:
    engine = AsyncOmni(**engine_kwargs(args))
    started = time.perf_counter()
    try:
        output = await generate(engine, args)
    finally:
        engine.close()
    frames = np.asarray(output.images[0])
    audio = np.asarray(output.multimodal_output.get("audio"))
    summary = {
        "duration_seconds": args.duration,
        "frames_shape": list(frames.shape),
        "audio_shape": list(audio.shape),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_mb": output.peak_memory_mb,
        "stage_durations": output.stage_durations,
    }
    if args.output is not None:
        # Frames arrive as THWC float32 in [0, 1]; audio as (1, 2, samples)
        # float32 at 32 kHz. Encode H.264 from uint8 and mux with ffmpeg.
        import imageio.v3 as iio

        video_path = args.output.with_suffix(".tmp.mp4")
        frames_u8 = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
        iio.imwrite(video_path, frames_u8, fps=24, codec="libx264", quality=8)
        audio_path = args.output.with_suffix(".tmp.f32")
        np.ascontiguousarray(audio[0].T, dtype=np.float32).tofile(audio_path)
        import subprocess

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-f",
                "f32le",
                "-ar",
                "32000",
                "-ac",
                "2",
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                str(args.output),
            ],
            check=True,
        )
        Path(video_path).unlink(missing_ok=True)
        Path(audio_path).unlink(missing_ok=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.duration <= 15.0:
        raise SystemExit("duration must be > 15 to trigger sliding-window generation")
    summary = asyncio.run(run(args))
    print(summary)


if __name__ == "__main__":
    main()
