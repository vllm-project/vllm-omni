# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

_MODEL_PRESETS = {
    "wan": {
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "fps": 24,
        "output": "wan22_output.mp4",
    },
    "hunyuan": {
        "height": 480,
        "width": 832,
        "num_frames": 121,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "fps": 24,
        "output": "hunyuan_video_15_output.mp4",
    },
}


def _detect_preset(model: str) -> dict:
    model_lower = model.lower()
    if "hunyuan" in model_lower:
        return _MODEL_PRESETS["hunyuan"]
    return _MODEL_PRESETS["wan"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a video from a text prompt. "
        "Supports Wan2.2, HunyuanVideo-1.5, and other text-to-video models."
    )
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Diffusers model ID or local path. "
        "Examples: Wan-AI/Wan2.2-T2V-A14B-Diffusers, "
        "hunyuanvideo-community/HunyuanVideo-1.5-480p_t2v",
    )
    parser.add_argument(
        "--model-class-name",
        default=None,
        help="Override model class name (e.g., LTX2TwoStagesVideoPipeline).",
    )
    parser.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.", help="Text prompt.")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="CFG scale. Default: model-specific.")
    parser.add_argument(
        "--guidance-scale-high", type=float, default=None, help="Separate CFG for high-noise stage (Wan2.2 only)."
    )
    parser.add_argument("--height", type=int, default=None, help="Video height. Default: model-specific.")
    parser.add_argument("--width", type=int, default=None, help="Video width. Default: model-specific.")
    parser.add_argument("--num-frames", type=int, default=None, help="Number of frames. Default: model-specific.")
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="Optional generation frame rate (used by models like LTX2). Defaults to --fps.",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=None, help="Sampling steps. Default: model-specific."
    )
    parser.add_argument(
        "--boundary-ratio",
        type=float,
        default=None,
        help="(Wan2.2) Boundary split ratio for low/high DiT. Default 0.875.",
    )
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=None,
        help="Scheduler flow_shift. Wan2.2: 5.0(720p)/12.0(480p). HunyuanVideo-1.5: 5.0(480p)/9.0(720p).",
    )
    parser.add_argument(
        "--cache-backend",
        type=str,
        default=None,
        choices=["cache_dit"],
        help="Cache backend for acceleration (Wan2.2). Default: None.",
    )
    parser.add_argument(
        "--enable-cache-dit-summary",
        action="store_true",
        help="Enable cache-dit summary logging after diffusion forward passes.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output path (mp4). Default: model-specific.")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second for the output video.")
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
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
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
        "--ulysses-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ulysses sequence parallelism.",
    )
    parser.add_argument(
        "--ring-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ring sequence parallelism.",
    )
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of GPUs used for classifier free guidance parallel size.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for tensor parallelism (TP) inside the DiT.",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=24000,
        help="Sample rate for audio output when saved (default: 24000).",
    )
    parser.add_argument(
        "--vae-patch-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for VAE patch/tile parallelism (decode).",
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="Enable expert parallelism for MoE layers.",
    )
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to display stage durations.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["fp8", "gguf"],
        help="Quantization method for the transformer (fp8 for online FP8 quantization).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_class_name = args.model_class_name

    preset = _detect_preset(args.model)
    for key, default_val in preset.items():
        if getattr(args, key.replace("-", "_"), None) is None:
            setattr(args, key.replace("-", "_"), default_val)

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    # Cache-dit config (Wan2.2 only)
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

    # Configure parallel settings
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
    )

    # Check if profiling is requested via environment variable
    profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))

    omni_kwargs = dict(
        model=args.model,
        enable_layerwise_offload=args.enable_layerwise_offload,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        enable_cpu_offload=args.enable_cpu_offload,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        model_class_name=model_class_name,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
    )
    if args.boundary_ratio is not None:
        omni_kwargs["boundary_ratio"] = args.boundary_ratio
    if args.flow_shift is not None:
        omni_kwargs["flow_shift"] = args.flow_shift
    if args.quantization is not None:
        omni_kwargs["quantization"] = args.quantization
    if args.cache_backend is not None:
        omni_kwargs["cache_backend"] = args.cache_backend
        omni_kwargs["cache_config"] = cache_config
        omni_kwargs["enable_cache_dit_summary"] = args.enable_cache_dit_summary

    omni = Omni(**omni_kwargs)

    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

    # Print generation configuration
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Frames: {args.num_frames}")
    print(
        f"  Parallel configuration: ulysses_degree={args.ulysses_degree}, ring_degree={args.ring_degree},"
        f" cfg_parallel_size={args.cfg_parallel_size}, tensor_parallel_size={args.tensor_parallel_size},"
        f" vae_patch_parallel_size={args.vae_patch_parallel_size}, enable_expert_parallel={args.enable_expert_parallel}"
    )
    print(f"  Video size: {args.width}x{args.height}")
    print(f"{'=' * 60}\n")

    prompt_dict = {"prompt": args.prompt}
    if args.negative_prompt:
        prompt_dict["negative_prompt"] = args.negative_prompt

    sampling_kwargs = dict(
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
    )
    if args.guidance_scale_high is not None:
        sampling_kwargs["guidance_scale_2"] = args.guidance_scale_high

    generation_start = time.perf_counter()
    frames = omni.generate(
        prompt_dict,
        OmniDiffusionSamplingParams(**sampling_kwargs),
    )
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    # Print profiling results
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    audio = None
    if isinstance(frames, list):
        frames = frames[0] if frames else None

    if isinstance(frames, OmniRequestOutput):
        if frames.final_output_type != "image":
            raise ValueError(
                f"Unexpected output type '{frames.final_output_type}', expected 'image' for video generation."
            )
        if frames.multimodal_output and "audio" in frames.multimodal_output:
            audio = frames.multimodal_output["audio"]
        if frames.is_pipeline_output and frames.request_output is not None:
            inner_output = frames.request_output
            if isinstance(inner_output, OmniRequestOutput):
                if inner_output.multimodal_output and "audio" in inner_output.multimodal_output:
                    audio = inner_output.multimodal_output["audio"]
                frames = inner_output
        if isinstance(frames, OmniRequestOutput):
            if frames.images:
                if len(frames.images) == 1 and isinstance(frames.images[0], tuple) and len(frames.images[0]) == 2:
                    frames, audio = frames.images[0]
                elif len(frames.images) == 1 and isinstance(frames.images[0], dict):
                    audio = frames.images[0].get("audio")
                    frames = frames.images[0].get("frames") or frames.images[0].get("video")
                else:
                    frames = frames.images
            else:
                raise ValueError("No video frames found in OmniRequestOutput.")

    if isinstance(frames, list) and frames:
        first_item = frames[0]
        if isinstance(first_item, tuple) and len(first_item) == 2:
            frames, audio = first_item
        elif isinstance(first_item, dict):
            audio = first_item.get("audio")
            frames = first_item.get("frames") or first_item.get("video")
        elif isinstance(first_item, list):
            frames = first_item

    if isinstance(frames, tuple) and len(frames) == 2:
        frames, audio = frames
    elif isinstance(frames, dict):
        audio = frames.get("audio")
        frames = frames.get("frames") or frames.get("video")

    if frames is None:
        raise ValueError("No video frames found in output.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.utils import export_to_video
    except ImportError:
        raise ImportError("diffusers is required for export_to_video.")

    def _normalize_frame(frame):
        if isinstance(frame, torch.Tensor):
            frame_tensor = frame.detach().cpu()
            if frame_tensor.dim() == 4 and frame_tensor.shape[0] == 1:
                frame_tensor = frame_tensor[0]
            if frame_tensor.dim() == 3 and frame_tensor.shape[0] in (3, 4):
                frame_tensor = frame_tensor.permute(1, 2, 0)
            if frame_tensor.is_floating_point():
                frame_tensor = frame_tensor.clamp(-1, 1) * 0.5 + 0.5
            return frame_tensor.float().numpy()
        if isinstance(frame, np.ndarray):
            frame_array = frame
            if frame_array.ndim == 4 and frame_array.shape[0] == 1:
                frame_array = frame_array[0]
            if np.issubdtype(frame_array.dtype, np.integer):
                frame_array = frame_array.astype(np.float32) / 255.0
            return frame_array
        try:
            from PIL import Image
        except ImportError:
            Image = None
        if Image is not None and isinstance(frame, Image.Image):
            return np.asarray(frame).astype(np.float32) / 255.0
        return frame

    def _ensure_frame_list(video_array):
        if isinstance(video_array, list):
            if len(video_array) == 0:
                return video_array
            first_item = video_array[0]
            if isinstance(first_item, np.ndarray):
                if first_item.ndim == 5:
                    return list(first_item[0])
                if first_item.ndim == 4:
                    if len(video_array) == 1:
                        return list(first_item)
                    return list(first_item)
                if first_item.ndim == 3:
                    return video_array
            return video_array
        if isinstance(video_array, np.ndarray):
            if video_array.ndim == 5:
                return list(video_array[0])
            if video_array.ndim == 4:
                return list(video_array)
            if video_array.ndim == 3:
                return [video_array]
        return video_array

    # frames may be np.ndarray, torch.Tensor, or list of tensors/arrays/images
    # export_to_video expects a list of frames with values in [0, 1]
    if isinstance(frames, torch.Tensor):
        video_tensor = frames.detach().cpu()
        if video_tensor.dim() == 5:
            if video_tensor.shape[1] in (3, 4):
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                video_tensor = video_tensor[0]
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        video_array = video_tensor.float().numpy()
    elif isinstance(frames, np.ndarray):
        video_array = frames
        if video_array.ndim == 5:
            video_array = video_array[0]
        if np.issubdtype(video_array.dtype, np.integer):
            video_array = video_array.astype(np.float32) / 255.0
    elif isinstance(frames, list):
        if len(frames) == 0:
            raise ValueError("No video frames found in output.")
        video_array = [_normalize_frame(frame) for frame in frames]
    else:
        video_array = frames

    video_array = _ensure_frame_list(video_array)

    if audio is not None:
        from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

        if isinstance(video_array, list):
            frames_np = np.stack(video_array, axis=0)
        elif isinstance(video_array, np.ndarray):
            frames_np = video_array
        else:
            frames_np = np.asarray(video_array)

        frames_u8 = (np.clip(frames_np, 0.0, 1.0) * 255).round().clip(0, 255).astype("uint8")

        audio_np = audio
        if isinstance(audio_np, list):
            audio_np = audio_np[0] if audio_np else None
        if isinstance(audio_np, torch.Tensor):
            audio_np = audio_np.detach().cpu().float().numpy()
        if isinstance(audio_np, np.ndarray):
            audio_np = np.squeeze(audio_np).astype(np.float32)

        video_bytes = mux_video_audio_bytes(
            frames_u8,
            audio_np,
            fps=float(args.fps),
            audio_sample_rate=args.audio_sample_rate,
        )
        with open(str(output_path), "wb") as f:
            f.write(video_bytes)
    else:
        export_to_video(video_array, str(output_path), fps=args.fps)
    print(f"Saved generated video to {output_path}")

    if profiler_enabled:
        print("\n[Profiler] Stopping profiler and collecting results...")
        profile_results = omni.stop_profile()
        if profile_results and isinstance(profile_results, dict):
            traces = profile_results.get("traces", [])
            print("\n" + "=" * 60)
            print("PROFILING RESULTS:")
            for rank, trace in enumerate(traces):
                print(f"\nRank {rank}:")
                if trace:
                    print(f"  • Trace: {trace}")
            if not traces:
                print("  No traces collected.")
            print("=" * 60)
        else:
            print("[Profiler] No valid profiling data returned.")


if __name__ == "__main__":
    main()
