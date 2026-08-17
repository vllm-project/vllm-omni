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
import json
import time
from pathlib import Path
from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_profiler_config(value: str) -> dict[str, Any]:
    try:
        config = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--profiler-config must be valid JSON: {e}") from e
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError("--profiler-config must be a JSON object")
    return config


def validate_media_inputs(
    model_type: str,
    *,
    image: str | None,
    audio: str | None,
    extra_body: dict[str, Any],
) -> None:
    """Fail early when the media a pipeline needs was not supplied.

    An official LongCat Avatar JSON case carries its own reference image and
    speaker tracks, so both flags become optional -- but only in that mode.
    ``input_json`` means nothing to Wan2.2 S2V, so it must not relax the checks
    there.
    """
    is_longcat_avatar = model_type == "longcat-video-avatar"
    has_input_json = is_longcat_avatar and bool(extra_body.get("input_json"))
    if audio is None and not has_input_json:
        raise ValueError("--audio is required (only a LongCat Avatar --extra-body input_json can supply it).")
    if image is None and not has_input_json:
        if not is_longcat_avatar:
            raise ValueError(f"--image is required for --model-type {model_type}.")
        # LongCat Avatar AT2V is driven by audio alone; AI2V needs the image.
        if str(extra_body.get("stage", "")).lower() == "ai2v":
            raise ValueError("--image is required for LongCat Avatar AI2V.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a talking-head video from a reference image and audio (Wan2.2 S2V)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to Wan2.2 S2V model (local path or HuggingFace ID).",
    )
    parser.add_argument(
        "--model-type",
        default="wan-s2v",
        choices=["wan-s2v", "longcat-video-avatar"],
        help="Model type.",
    )
    parser.add_argument(
        "--image",
        help="Path to reference image. Required for Wan2.2 S2V and for LongCat Avatar AI2V; "
        "LongCat Avatar AT2V is driven by audio alone, and an --extra-body input_json brings its own.",
    )
    parser.add_argument(
        "--audio",
        help="Path to audio file (wav/mp3). Required unless a LongCat Avatar JSON case supplies the tracks.",
    )
    parser.add_argument(
        "--extra-body",
        type=str,
        default=None,
        help="[longcat-video-avatar] JSON dict of model-specific extra params (declared in "
        "vllm_omni/model_extras/), merged into sampling extra_args. "
        'Example: \'{"stage": "ai2v", "num_segments": "auto"}\'.',
    )
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
        default=None,
        help="Frames per clip (should be divisible by 4). Default: 80 for Wan2.2 S2V, 93 for LongCat Avatar.",
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
        "--boundary-ratio",
        type=float,
        default=None,
        help="Boundary split ratio for low/high DiT stages (S2V). Default varies by model.",
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
        default=None,
        help="Frames per second for the output video. Default: 16 for Wan2.2 S2V, 25 for LongCat Avatar.",
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
        "--use-hsdp",
        action="store_true",
        help="Enable Hybrid Sharded Data Parallel to shard model weights across GPUs.",
    )
    parser.add_argument(
        "--hsdp-shard-size",
        type=int,
        default=-1,
        help=(
            "Number of GPUs to shard model weights across within each replica group. "
            "-1 (default) auto-calculates as world_size / replicate_size."
        ),
    )
    parser.add_argument(
        "--hsdp-replicate-size",
        type=int,
        default=1,
        help=(
            "Number of replica groups for HSDP. Each replica holds a full sharded copy. "
            "Default 1 means pure sharding (no replication)."
        ),
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
        "--profiler-config",
        type=parse_profiler_config,
        default=None,
        help='JSON profiler config for torch/cuda profiling, e.g. \'{"profiler":"torch","torch_profiler_dir":"./perf"}\'.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    is_longcat_avatar = args.model_type == "longcat-video-avatar"
    # Per-model defaults, so each pipeline keeps the clip length and frame rate
    # it was tuned for.
    num_frames = args.num_frames if args.num_frames is not None else (93 if is_longcat_avatar else 80)
    fps = args.fps if args.fps is not None else (25 if is_longcat_avatar else 16)
    extra_body = json.loads(args.extra_body) if args.extra_body else {}
    validate_media_inputs(args.model_type, image=args.image, audio=args.audio, extra_body=extra_body)

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    # Load reference image
    import PIL.Image

    image = PIL.Image.open(args.image).convert("RGB") if args.image else None

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
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
    )

    profiler_enabled = args.profiler_config is not None

    omni_kwargs = dict(
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
        profiler_config=args.profiler_config,
    )
    if args.boundary_ratio is not None:
        omni_kwargs["boundary_ratio"] = args.boundary_ratio

    if is_longcat_avatar:
        from vllm_omni.diffusion.models.longcat_video.pipeline_longcat_video_avatar import (
            prepare_longcat_video_avatar_model_for_omni,
        )

        # Avatar loads its own component set, so drop the Wan-only engine knobs
        # and keep the shared ones (parallelism, offload, cache, profiling).
        for key in ("flow_shift", "boundary_ratio", "vae_use_slicing", "vae_use_tiling"):
            omni_kwargs.pop(key, None)
        # The pipeline defaults every engine-level knob and reads the rest from
        # sampling extra_args, so only the load-time ones are forwarded here.
        additional_config = {
            key: extra_body.pop(key)
            for key in ("use_int8", "build_components_on_gpu", "base_model_dir")
            if key in extra_body
        }
        omni_kwargs["model"] = prepare_longcat_video_avatar_model_for_omni(
            args.model, additional_config.get("use_int8", True)
        )
        omni_kwargs["model_class_name"] = "LongCatVideoAvatarPipeline"
        omni_kwargs["dtype"] = "bfloat16"
        if additional_config:
            omni_kwargs["additional_config"] = additional_config

    omni = Omni(**omni_kwargs)

    # Print generation configuration
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Reference image: {args.image}")
    print(f"  Audio: {args.audio}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Frames per clip: {num_frames}")
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

    if is_longcat_avatar:
        # The pipeline resolves the stage, the speaker layout and the official
        # JSON cases itself, so the request only carries the raw media inputs.
        multi_modal_data: dict[str, Any] = {}
        if args.audio is not None:
            multi_modal_data["audio"] = args.audio
        if image is not None:
            multi_modal_data["image"] = image
        prompt = {"prompt": args.prompt, "multi_modal_data": multi_modal_data}
        if args.negative_prompt is not None:
            prompt["negative_prompt"] = args.negative_prompt
        sampling_params = OmniDiffusionSamplingParams(
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            generator=generator,
            extra_args=extra_body,
        )
    else:
        prompt = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "multi_modal_data": {
                "image": image,
                "audio": args.audio,
                "init_first_frame": args.init_first_frame,
            },
        }
        sampling_params = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            generator=generator,
        )

    result = omni.generate(prompt, sampling_params)

    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    # Stop profiling if enabled
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

    # omni.generate() returns a list for sync calls; unwrap single-result list.
    output = result[0] if isinstance(result, list) else result

    if not output.images:
        raise ValueError("No video frames found in OmniRequestOutput.")

    # Extract audio from multimodal_output (set by pipeline post-processor)
    mm = output.multimodal_output or {}
    audio_waveform = mm.get("audio")
    output_fps = float(mm.get("fps", fps))
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
            if isinstance(first_elem, list):
                return _flatten_to_array(first_elem)  # [[frame, ...]] → [frame, ...]
            if isinstance(first_elem, PIL.Image.Image):
                return np.stack([np.asarray(frame) for frame in data])
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
        audio_samples = audio_waveform.shape[-1] if audio_waveform.ndim > 1 else len(audio_waveform)
        print(f"Audio: {audio_samples / output_sr:.1f}s at {output_sr} Hz")


if __name__ == "__main__":
    main()
