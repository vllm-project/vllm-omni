# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script for text/video-to-audio generation with diffusion audio models.

This script supports:
  * Stable Audio Open (`stabilityai/stable-audio-open-1.0`) — text-to-audio.
  * AudioX (`zhangj1an/AudioX`) — six tasks: t2a / t2m / v2a / v2m / tv2a / tv2m.

Model-specific generation knobs are declared in `vllm_omni/model_extras` and
routed into `sampling_params.extra_args` via `apply_declared_extra_args`, so
the same script serves both models without per-model branching in the engine.

Usage:
    # Stable Audio Open
    python text_to_audio.py --prompt "The sound of a dog barking"
    python text_to_audio.py --prompt "A piano playing a gentle melody" --audio-length 10.0
    python text_to_audio.py --prompt "Thunder and rain sounds" --negative-prompt "Low quality"
    python text_to_audio.py --prompt "A soft synth pad" --cache-backend tea_cache

    # AudioX (text-to-audio / text-to-music)
    python text_to_audio.py --model zhangj1an/AudioX --task t2a \
        --prompt "Fireworks burst twice, then a clock ticking." \
        --num-inference-steps 250 --guidance-scale 6.0 --audio-length 10.0
    # AudioX video-conditioned (v2*/tv2*) — pass a video file/URL
    python text_to_audio.py --model zhangj1an/AudioX --task tv2a \
        --prompt "drum beating sound and human talking" --video clip.mp4 \
        --extra-body '{"sigma_min": 0.03, "sigma_max": 1000.0}'
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import get_extra_body_params, get_model_class_name
from vllm_omni.platforms import current_omni_platform

AUDIOX_TASKS = ("t2a", "t2m", "v2a", "v2m", "tv2a", "tv2m")
AUDIOX_VIDEO_TASKS = frozenset({"v2a", "v2m", "tv2a", "tv2m"})


def parse_extra_body(value: str) -> dict[str, Any]:
    """Parse a JSON object string for --extra-body."""
    try:
        obj = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--extra-body must be valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise argparse.ArgumentTypeError("--extra-body must be a JSON object")
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate audio with supported diffusion audio models.")
    parser.add_argument(
        "--model",
        default="stabilityai/stable-audio-open-1.0",
        help="Audio diffusion model name or local path. Supported: stabilityai/stable-audio-open-1.0, zhangj1an/AudioX.",
    )
    parser.add_argument(
        "--prompt",
        default="The sound of a hammer hitting a wooden surface.",
        help="Text prompt for audio generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help=(
            "Negative prompt for classifier-free guidance. Off by default; "
            'recommended for Stable Audio, e.g. --negative-prompt "Low quality".'
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic results.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--audio-start",
        type=float,
        default=0.0,
        help=(
            "Audio start offset in seconds. Applies to both models "
            "(audio_start_in_s for Stable Audio, seconds_start for AudioX)."
        ),
    )
    parser.add_argument(
        "--audio-length",
        type=float,
        default=10.0,
        help=(
            "Audio duration in seconds. Maps to audio length for Stable Audio "
            "(max ~47s for stable-audio-open-1.0) and seconds_total for AudioX."
        ),
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=100,
        help="Number of denoising steps for the diffusion sampler.",
    )
    parser.add_argument(
        "--num-waveforms",
        type=int,
        default=1,
        help="Number of audio waveforms to generate for the given prompt.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=list(AUDIOX_TASKS),
        help=(
            "[AudioX only] Generation task: t2a/t2m/v2a/v2m/tv2a/tv2m. "
            "v2*/tv2* require --video. Ignored for Stable Audio Open."
        ),
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="[AudioX v2*/tv2*] Video file path or URL used for conditioning.",
    )
    parser.add_argument(
        "--extra-body",
        type=parse_extra_body,
        default=None,
        help=(
            "Model-specific generation params as a JSON object, e.g. "
            '\'{"sigma_min": 0.03, "sigma_max": 1000.0}\'. Each key is filtered '
            "against the model's declared extra_body_params (see vllm_omni/model_extras), "
            "so unknown keys are silently dropped. Values here override the equivalent flags."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path to save the generated audio (WAV format). Defaults to "
            "audiox_output.wav for AudioX and stable_audio_output.wav otherwise."
        ),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate for output audio (Stable Audio uses 44100 Hz).",
    )
    parser.add_argument(
        "--cache-backend",
        type=str,
        default=None,
        choices=["tea_cache"],
        help=(
            "Cache backend to use for acceleration. "
            "Stable Audio currently supports 'tea_cache'. "
            "Default: None (no cache acceleration)."
        ),
    )
    parser.add_argument(
        "--tea-cache-rel-l1-thresh",
        type=float,
        default=0.2,
        help="[tea_cache] Threshold for accumulated relative L1 distance.",
    )
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to display stage durations.",
    )
    parser.add_argument(
        "--enable-cpu-offload", action="store_true", help="Enable ModelWise cpu offloading to save gpu memory"
    )
    parser.add_argument(
        "--enable-layerwise-offload", action="store_true", help="Enable Layerwise cpu offloading to save gpu memory"
    )
    parser.add_argument(
        "--use-hsdp",
        action="store_true",
        help="Enable HSDP for Stable Audio DiT weight sharding.",
    )
    parser.add_argument(
        "--hsdp-shard-size",
        type=int,
        default=1,
        help="Number of GPUs to shard Stable Audio DiT weights across when HSDP is enabled.",
    )
    parser.add_argument(
        "--hsdp-replicate-size",
        type=int,
        default=1,
        help="Number of HSDP replica groups. Default 1 means pure sharding.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for tensor parallelism (TP) inside the DiT.",
    )
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ulysses sequence parallelism.",
    )
    parser.add_argument(
        "--ulysses-mode",
        type=str,
        default="strict",
        choices=["strict", "advanced_uaa"],
        help="Ulysses sequence-parallel mode: 'strict' (divisibility required) or 'advanced_uaa' (UAA).",
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
        "--vae-patch-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for VAE patch/tile parallelism (decode).",
    )
    return parser.parse_args()


def save_audio(audio_data: np.ndarray, output_path: str, sample_rate: int = 44100):
    """Save audio data to a WAV file."""
    try:
        import soundfile as sf

        sf.write(output_path, audio_data, sample_rate)
    except ImportError:
        try:
            import scipy.io.wavfile as wav

            # Ensure audio is in the correct format for scipy
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                # Normalize to int16 range
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_data = (audio_data * 32767).astype(np.int16)
            wav.write(output_path, sample_rate, audio_data)
        except ImportError:
            raise ImportError(
                "Either 'soundfile' or 'scipy' is required to save audio files. "
                "Install with: pip install soundfile or pip install scipy"
            )


def main():
    args = parse_args()
    model_name = str(args.model).lower() if args.model else ""
    is_audiox = "audiox" in model_name
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    cache_config = None
    if args.cache_backend == "tea_cache":
        cache_config = {
            "rel_l1_thresh": args.tea_cache_rel_l1_thresh,
        }

    print(f"\n{'=' * 60}")
    print("Text/Video-to-Audio Generation")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model}")
    if args.task is not None:
        print(f"  Task: {args.task}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Negative prompt: {args.negative_prompt}")
    print(f"  Duration: {args.audio_length}s")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Cache backend: {args.cache_backend if args.cache_backend else 'None (no acceleration)'}")
    print(f"  ModelWise Offload: {'Enabled' if args.enable_cpu_offload else 'None'}")
    print(f"  LayerWise Offload: {'Enabled' if args.enable_layerwise_offload else 'None'}")
    if args.use_hsdp:
        print(f"  HSDP: enabled (shard_size={args.hsdp_shard_size}, replicate_size={args.hsdp_replicate_size})")
    else:
        print("  HSDP: disabled")
    print(f"  Seed: {args.seed}")
    print(f"{'=' * 60}\n")

    parallel_config = DiffusionParallelConfig(
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
    )

    # Initialize Omni. AudioX requires an explicit pipeline class; Stable Audio
    # is resolved from its config.
    omni_kwargs: dict[str, Any] = dict(
        model=args.model,
        parallel_config=parallel_config,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
    )
    if is_audiox:
        omni_kwargs["model_class_name"] = "AudioXPipeline"
    omni = Omni(**omni_kwargs)
    model_class_name = get_model_class_name(omni)
    declared_extra_body_params = get_extra_body_params(model_class_name)

    diffusion_params = OmniDiffusionSamplingParams(
        generator=generator,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_outputs_per_prompt=args.num_waveforms,
        seed=args.seed,
    )

    # Negative prompt is opt-in for both models (off by default). Including a
    # non-empty negative prompt enables the separate CFG conditioning branch.
    prompt: dict[str, Any] = {"prompt": args.prompt}
    if args.negative_prompt:
        prompt["negative_prompt"] = args.negative_prompt

    if model_class_name == "AudioXPipeline":
        if args.task is None:
            raise ValueError("AudioX requires --task (one of t2a/t2m/v2a/v2m/tv2a/tv2m).")
        if args.task in AUDIOX_VIDEO_TASKS and not args.video:
            raise ValueError(f"AudioX task {args.task!r} requires --video.")
        # AudioX core knobs are surfaced as shared flags; sampler/reference knobs
        # (sigma_min/sigma_max/cfg_rescale/audio_path) arrive via --extra-body and
        # are filtered against the declared extra_body contract.
        user_extra: dict[str, Any] = {
            "audiox_task": args.task,
            "seconds_start": args.audio_start,
            "seconds_total": args.audio_length,
            "video_path": args.video,
        }
        if args.extra_body:
            user_extra.update(args.extra_body)
        apply_declared_extra_args(diffusion_params, declared_extra_body_params, user_extra)
    else:
        # Stable Audio Open path (backward-compatible).
        audio_end_in_s = args.audio_start + args.audio_length
        diffusion_params.extra_args = {
            "audio_start_in_s": args.audio_start,
            "audio_end_in_s": audio_end_in_s,
        }
        if args.extra_body:
            diffusion_params.extra_args.update(args.extra_body)

    generation_start = time.perf_counter()

    outputs = omni.generate(prompt, diffusion_params)

    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    print(f"Total generation time: {generation_time:.2f} seconds")

    output = args.output or ("audiox_output.wav" if model_class_name == "AudioXPipeline" else "stable_audio_output.wav")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".wav"
    stem = output_path.stem or "stable_audio_output"

    if not outputs:
        raise ValueError("No output generated from omni.generate()")

    output = outputs[0]
    if not hasattr(output, "request_output") or not output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")
    request_output = output.request_output
    if not hasattr(request_output, "multimodal_output"):
        raise ValueError("No multimodal_output found in request_output")

    audio = request_output.multimodal_output.get("audio")
    if audio is None:
        raise ValueError("No audio output found in request_output")

    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().float().numpy()

    # Audio shape is typically [batch, channels, samples] or [channels, samples]
    if audio.ndim == 3:
        # [batch, channels, samples]
        if args.num_waveforms <= 1:
            audio_data = audio[0].T  # [samples, channels]
            save_audio(audio_data, str(output_path), args.sample_rate)
            print(f"Saved generated audio to {output_path}")
        else:
            for idx in range(audio.shape[0]):
                audio_data = audio[idx].T  # [samples, channels]
                save_path = output_path.parent / f"{stem}_{idx}{suffix}"
                save_audio(audio_data, str(save_path), args.sample_rate)
                print(f"Saved generated audio to {save_path}")
    elif audio.ndim == 2:
        # [channels, samples]
        audio_data = audio.T  # [samples, channels]
        save_audio(audio_data, str(output_path), args.sample_rate)
        print(f"Saved generated audio to {output_path}")
    else:
        # [samples] - mono audio
        save_audio(audio, str(output_path), args.sample_rate)
        print(f"Saved generated audio to {output_path}")

    print(f"\nGenerated ~{args.audio_length}s of audio at {args.sample_rate} Hz")


if __name__ == "__main__":
    main()
