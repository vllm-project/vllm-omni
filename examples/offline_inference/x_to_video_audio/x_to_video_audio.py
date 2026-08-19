# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import time
from typing import Any

import numpy as np

from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline inference for X -> video+audio models.")
    parser.add_argument("--model", required=True, help="Model ckpt root directory.")
    parser.add_argument(
        "--model-type",
        default="magi-human",
        choices=["magi-human"],
        help="Model type.",
    )
    parser.add_argument("--prompt", default=None, help="Text prompt.")
    parser.add_argument(
        "--extra-body",
        type=str,
        default=None,
        help="JSON dict of model-specific extra params (declared in vllm_omni/model_extras/), "
        'merged into sampling extra_args. Example: \'{"image_path": "/path/to/img.jpg", "seconds": 5}\'.',
    )

    parser.add_argument("--height", type=int, default=256, help="Video height.")
    parser.add_argument("--width", type=int, default=448, help="Video width.")
    parser.add_argument("--num-inference-steps", type=int, default=8, help="Sampling steps.")
    parser.add_argument("--seed", type=int, default=52, help="Random seed for reproducible generation.")
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Number of GPUs used for classifier free guidance parallel size.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for tensor parallelism (TP) inside the DiT.",
    )
    parser.add_argument("--output", default="output_magihuman.mp4", help="Output video path.")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["fp8", "int8"],
        help="Online (dynamic) quantization method for the transformer.",
    )
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        default=False,
        help="Enable CPU offloading for diffusion models.",
    )
    parser.add_argument(
        "--enable-layerwise-offload",
        action="store_true",
        help="Enable layerwise (blockwise) offloading on DiT modules.",
    )
    parser.add_argument("--cache-backend", type=str, default=None, choices=["cache_dit"], help="Cache backend.")
    parser.add_argument(
        "--use-hsdp",
        action="store_true",
        help="Enable HSDP for DiT weight sharding.",
    )
    parser.add_argument(
        "--hsdp-shard-size",
        type=int,
        default=1,
        help="Number of GPUs used for HSDP sharding when HSDP is enabled.",
    )
    parser.add_argument(
        "--hsdp-replicate-size",
        type=int,
        default=1,
        help="Number of HSDP replica groups. Default 1 means pure sharding.",
    )
    return parser.parse_args()


def _extract_peak_memory_mb(result: Any) -> float:
    """Pull worker-reported peak VRAM (MiB) generation result.

    Mirrors vllm_omni/entrypoints/openai/serving_video.py:_extract_peak_memory_mb.
    """
    if isinstance(result, list):
        result = result[0] if result else None
    if result is None:
        return 0.0
    val = getattr(result, "peak_memory_mb", 0.0)
    if not val:
        inner = result
        if isinstance(inner, list):
            inner = inner[0] if inner else None
        val = getattr(inner, "peak_memory_mb", 0.0)
    try:
        return float(val or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    args = parse_args()
    if args.prompt is None:
        raise ValueError("--prompt must be provided.")

    extra_args = json.loads(args.extra_body) if args.extra_body else {}
    sampling_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        extra_args=extra_args,
    )

    cache_config = None
    if args.cache_backend == "cache_dit":
        cache_config = {
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "max_warmup_steps": 4,
            "max_cached_steps": 20,
            "residual_diff_threshold": 0.24,
            "max_continuous_cached_steps": 3,
            "enable_taylorseer": False,
            "taylorseer_order": 1,
            "scm_steps_mask_policy": None,
            "scm_steps_policy": "dynamic",
        }

    omni_kwargs = dict(
        model=args.model,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
        model_type=args.model_type,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
    )
    if args.quantization is not None:
        omni_kwargs["quantization"] = args.quantization
    omni = Omni(**omni_kwargs)
    start = time.perf_counter()
    outputs = omni.generate(args.prompt, sampling_params)
    elapsed = time.perf_counter() - start

    if not outputs:
        raise RuntimeError("No output returned from the model.")
    result = outputs[0]
    if not result.images:
        raise RuntimeError("No video frames found in model output.")
    generated_video = result.images[0]
    mm = result.multimodal_output or {}
    generated_audio = mm.get("audio")

    fps = float(mm.get("fps", 25))
    sample_rate = int(mm.get("audio_sample_rate", 24000))
    frames = generated_video

    audio_np = None
    if generated_audio is not None:
        audio_np = np.squeeze(np.asarray(generated_audio)).astype(np.float32)

    video_bytes = mux_video_audio_bytes(
        frames,
        audio_np,
        fps=float(fps),
        audio_sample_rate=sample_rate,
    )
    with open(args.output, "wb") as f:
        f.write(video_bytes)
    print(f"Saved generated video to {args.output}")
    print(f"Total time: {elapsed:.2f}s")
    peak_mb = _extract_peak_memory_mb(outputs)
    if peak_mb:
        print(f"Worker peak GPU memory (reserved): {peak_mb:.2f} MiB ({peak_mb / 1024:.2f} GiB)")


if __name__ == "__main__":
    main()
