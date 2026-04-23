# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Speech-to-Video generation example using Wan2.2 S2V.

Generates talking-head videos from a reference image and an audio clip
using the Wan2.2 S2V pipeline with multi-clip autoregressive generation.

Usage:
    python speech_to_video.py \
        --model /path/to/Wan2.2-S2V-14B \
        --image reference.jpg \
        --audio speech.wav \
        --prompt "A person speaking naturally"
"""

import argparse
import os
import time
from pathlib import Path

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a talking-head video from a reference image and audio (Wan2.2 S2V)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to Wan2.2 S2V model (local path or HuggingFace ID).",
    )
    parser.add_argument("--image", required=True, help="Path to reference image.")
    parser.add_argument("--audio", required=True, help="Path to audio file (wav/mp3).")
    parser.add_argument(
        "--prompt",
        default="A person speaking naturally",
        help="Text prompt describing the scene.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt (uses S2V default if not set).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.5,
        help="CFG scale (default: 4.5).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Video height (auto-calculated from reference image if not set).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Video width (auto-calculated from reference image if not set).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=80,
        help="Frames per clip (should be divisible by 4, default: 80).",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=40,
        help="Number of denoising steps (default: 40).",
    )
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=3.0,
        help="Scheduler flow shift (default: 3.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="s2v_output.mp4",
        help="Path to save the output video.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frames per second for the output video (default: 16).",
    )
    parser.add_argument(
        "--init-first-frame",
        action="store_true",
        help="Use the reference image as the first frame of the video.",
    )
    parser.add_argument(
        "--vae-use-slicing",
        action="store_true",
        help="Enable VAE slicing for memory optimization.",
    )
    parser.add_argument(
        "--vae-use-tiling",
        action="store_true",
        help="Enable VAE tiling for memory optimization.",
    )
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offloading for diffusion models.",
    )
    parser.add_argument(
        "--enable-layerwise-offload",
        action="store_true",
        help="Enable layerwise (blockwise) offloading on DiT modules.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for tensor parallelism (TP) inside the DiT.",
    )
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of GPUs used for classifier free guidance parallel size.",
    )
    parser.add_argument(
        "--vae-patch-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for VAE patch/tile parallelism (decode).",
    )
    parser.add_argument(
        "--ring-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ring sequence parallelism.",
    )
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ulysses sequence parallelism.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to display stage durations.",
    )
    parser.add_argument(
        "--cache-backend",
        type=str,
        default=None,
        choices=["cache_dit"],
        help="Cache backend for acceleration. Default: None.",
    )
    parser.add_argument(
        "--enable-cache-dit-summary",
        action="store_true",
        help="Enable cache-dit summary logging after diffusion forward passes.",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Enable torch profiler and save traces to this directory.",
    )
    parser.add_argument(
        "--profile-record-shapes",
        action="store_true",
        help="Record tensor shapes in profiler (increases trace size).",
    )
    parser.add_argument(
        "--profile-with-stack",
        action="store_true",
        help="Record stack traces in profiler (increases overhead).",
    )
    parser.add_argument(
        "--profile-with-memory",
        action="store_true",
        help="Profile memory usage.",
    )
    parser.add_argument(
        "--profile-with-flops",
        action="store_true",
        help="Estimate FLOPs for operations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    # Load reference image
    import PIL.Image

    image = PIL.Image.open(args.image).convert("RGB")

    # Cache-dit config
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

    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=args.tensor_parallel_size,
        cfg_parallel_size=args.cfg_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        ring_degree=args.ring_degree,
        ulysses_degree=args.ulysses_degree,
    )

    # Check if profiling is requested via CLI args or environment variable
    profile_dir = args.profile_dir or os.getenv("VLLM_TORCH_PROFILER_DIR")
    profiler_enabled = bool(profile_dir)
    profiler_config = None
    if profiler_enabled:
        from vllm.config import ProfilerConfig

        profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=profile_dir,
            torch_profiler_record_shapes=args.profile_record_shapes,
            torch_profiler_with_stack=args.profile_with_stack,
            torch_profiler_with_memory=args.profile_with_memory,
            torch_profiler_with_flops=args.profile_with_flops,
        )

    omni = Omni(
        model=args.model,
        model_class_name="WanS2VPipeline",
        flow_shift=args.flow_shift,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        enable_cache_dit_summary=args.enable_cache_dit_summary,
        profiler_config=profiler_config,
    )

    # Print generation configuration
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Reference image: {args.image}")
    print(f"  Audio: {args.audio}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Frames per clip: {args.num_frames}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Flow shift: {args.flow_shift}")
    print(f"  Init first frame: {args.init_first_frame}")
    if args.height and args.width:
        print(f"  Video size: {args.width}x{args.height}")
    else:
        print("  Video size: auto (from reference image)")
    print(f"{'=' * 60}\n")

    # Start profiling if enabled
    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

    generation_start = time.perf_counter()

    result = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "multi_modal_data": {
                "image": image,
                "audio": args.audio,
                "init_first_frame": args.init_first_frame,
            },
        },
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            generator=generator,
        ),
    )

    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    # Stop profiling if enabled
    if profiler_enabled:
        print("\n[Profiler] Stopping profiler and collecting results...")
        profile_results = omni.stop_profile()
        if profile_results and isinstance(profile_results, list):
            print("\n" + "=" * 60)
            print("PROFILING RESULTS:")
            for i, result in enumerate(profile_results):
                print(f"\nStage {i}:")
                if result:
                    print(f"  {result}")
            print("=" * 60)
        else:
            print("[Profiler] No valid profiling data returned.")

    # Extract output from result
    output = OmniRequestOutput.unwrap_result(result)

    if not output.images:
        raise ValueError("No video frames found in OmniRequestOutput.")

    # Extract audio from multimodal_output (set by pipeline post-processor)
    mm = output.multimodal_output or {}
    audio_waveform = mm.get("audio")
    output_fps = float(mm.get("fps", args.fps))
    output_sr = int(mm.get("audio_sample_rate", 16000))

    if audio_waveform is not None:
        print(f"Audio waveform: shape={audio_waveform.shape}, sr={output_sr}")
    else:
        print("Warning: no audio waveform in pipeline output")

    # Normalize frames to (T, H, W, C) uint8 numpy array
    import numpy as np

    def _flatten_to_array(data):
        """Unwrap to a single (T, H, W, C) numpy array."""
        if isinstance(data, np.ndarray):
            if data.ndim == 5:
                return data[0]  # (B, T, H, W, C) → (T, H, W, C)
            if data.ndim == 4:
                return data  # already (T, H, W, C)
        if isinstance(data, list) and data:
            first_elem = data[0]
            if isinstance(first_elem, np.ndarray) and first_elem.ndim >= 4:
                return _flatten_to_array(first_elem)
            if isinstance(first_elem, np.ndarray) and first_elem.ndim == 3:
                return np.stack(data)  # list of (H, W, C) → (T, H, W, C)
        return data

    video_frames = _flatten_to_array(output.images)
    # postprocess_video returns float32 in [0,1]; mux_video_audio_bytes needs uint8
    if isinstance(video_frames, np.ndarray) and video_frames.dtype != np.uint8:
        video_frames = (np.clip(video_frames, 0, 1) * 255).astype(np.uint8)
    num_frames = video_frames.shape[0] if isinstance(video_frames, np.ndarray) else len(video_frames)

    # Save output video with audio
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

    video_bytes = mux_video_audio_bytes(
        video_frames,
        audio_waveform,
        fps=output_fps,
        audio_sample_rate=output_sr,
    )
    with open(output_path, "wb") as f:
        f.write(video_bytes)

    print(f"Saved generated video to {output_path}")
    print(f"Video has {num_frames} frames at {output_fps} fps ({num_frames / output_fps:.1f}s)")
    if audio_waveform is not None:
        print(f"Audio: {len(audio_waveform) / output_sr:.1f}s at {output_sr} Hz")


if __name__ == "__main__":
    main()
