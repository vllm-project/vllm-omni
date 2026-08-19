# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from vllm.multimodal.media.audio import load_audio

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.openai.video_api_utils import encode_video_bytes
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import (
    build_x_to_video_audio_prompt as build_model_x_to_video_audio_prompt,
)
from vllm_omni.model_extras import (
    get_extra_body_params,
    get_input_audio_sample_rate,
    get_model_class_name,
    get_output_tensor_range,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


@dataclass(frozen=True)
class XToVideoAudioOutput:
    """Canonical generated video+audio payload and its required metadata."""

    video: Any
    audio: Any | None
    fps: float
    audio_sample_rate: int | None
    output_tensor_range: str = "negative_one_to_one"


def parse_json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--extra-body must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--extra-body must be a JSON object")
    return parsed


def build_x_to_video_audio_prompt(
    prompt: str,
    media_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical request envelope for image/audio-to-video models."""
    result: dict[str, Any] = {"prompt": prompt, "modalities": ["video"]}
    if media_inputs:
        result["multi_modal_data"] = media_inputs
    return result


def _clean_official_prompt(prompt: str) -> str:
    """Remove metadata tags used by the official DreamID prompt fixtures."""
    prompt = re.sub(
        r"\[SPEAKER_TIMESTAMPS_START\].*?\[SPEAKER_TIMESTAMPS_END\]",
        "",
        prompt,
        flags=re.DOTALL,
    ).strip()
    prompt = re.sub(
        r"\[AUDIO_DESCRIPTION_START].*?\[AUDIO_DESCRIPTION_END]",
        "",
        prompt,
        flags=re.DOTALL,
    ).strip()
    prompt = re.sub(r"\[[A-Z_]+\]", "", prompt)
    return re.sub(r"\n\s*\n", "\n", prompt).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline inference for X -> video+audio models.")
    parser.add_argument("--model", required=True, help="Model ckpt root directory.")
    parser.add_argument(
        "--model-type",
        default=None,
        help="Optional legacy model type override (for example, magi-human).",
    )
    parser.add_argument("--model-class-name", default=None, help="Optional diffusion pipeline class override.")
    parser.add_argument("--prompt", default=None, help="Text prompt.")

    parser.add_argument("--image-path", type=str, nargs="+", help="list of image-path")
    parser.add_argument("--audio-path", type=str, nargs="+", help="list of audio-path")
    parser.add_argument("--prompt-file", type=str, default=None, help="Text prompt in json format.")
    parser.add_argument(
        "--extra-body",
        type=parse_json_object,
        default=None,
        help="JSON dict of model-specific extra params (declared in vllm_omni/model_extras/), "
        'merged into sampling extra_args. Example: \'{"image_path": "/path/to/img.jpg", "seconds": 5}\'.',
    )

    parser.add_argument("--height", type=int, default=704, help="Video height.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--num-inference-steps", type=int, default=45, help="Sampling steps.")
    parser.add_argument("--solver-name", default="unipc", help="Solver name: unipc|dpm++|euler.")
    parser.add_argument("--shift", type=float, default=5.0, help="Scheduler shift.")
    parser.add_argument("--seed", type=int, default=103, help="Random seed for reproducible generation.")
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Number of GPUs used for classifier free guidance parallel size (max 4 branches).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help=("Number of GPUs used for tensor parallelism (TP) inside the DiT."),
    )
    parser.add_argument(
        "--video-negative-prompt",
        default="jitter, bad hands, blur, distortion",
        help="Negative prompt for video.",
    )
    parser.add_argument(
        "--audio-negative-prompt",
        default="robotic, muffled, echo, distorted",
        help="Negative prompt for audio.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS metadata.")
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=None,
        help="Override output audio sample-rate metadata.",
    )
    parser.add_argument("--output", default="dreamid_output.mp4", help="Output video path.")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["fp8", "int8"],
        help="Online (dynamic) quantization method for the model transformer.",
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
        help="Enable HSDP for supported transformer blocks.",
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


def load_image_and_audio(
    image_paths: list[str] | None,
    audio_paths: list[str] | None,
    *,
    audio_sample_rate: int | None = None,
) -> tuple[list[Image.Image], list[tuple[Any, int]]]:
    """Load complete media inputs into the canonical multimodal representation."""
    images = []
    audios = []

    for path in image_paths or []:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))

    for path in audio_paths or []:
        waveform, sample_rate = load_audio(path, sr=audio_sample_rate)
        audios.append((waveform, sample_rate))
    return images, audios


def extract_x_to_video_audio_output(
    outputs: Any,
    *,
    fps: float | None = None,
    audio_sample_rate: int | None = None,
    output_tensor_range: str = "negative_one_to_one",
) -> XToVideoAudioOutput:
    """Normalize engine output without depending on a model's tensor layout."""
    if isinstance(outputs, list):
        if not outputs:
            raise RuntimeError("No output returned from the model.")
        result = outputs[0]
    else:
        result = outputs
    if result is None:
        raise RuntimeError("No output returned from the model.")

    multimodal_output = getattr(result, "multimodal_output", None) or {}
    if not isinstance(multimodal_output, Mapping):
        raise RuntimeError("Model multimodal output must be a mapping.")
    images = getattr(result, "images", None)
    video = images[0] if images else multimodal_output.get("video")
    if video is None:
        raise RuntimeError("No video payload found in model output.")

    resolved_fps = fps if fps is not None else multimodal_output.get("fps")
    try:
        resolved_fps = float(resolved_fps)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Model output must declare a valid 'fps', or pass --fps.") from exc
    if not math.isfinite(resolved_fps) or resolved_fps <= 0:
        raise RuntimeError("Output FPS must be finite and positive.")

    audio = multimodal_output.get("audio")
    resolved_sample_rate = audio_sample_rate
    if resolved_sample_rate is None:
        resolved_sample_rate = multimodal_output.get("audio_sample_rate")
    if audio is not None:
        try:
            resolved_sample_rate = int(resolved_sample_rate)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Model output with audio must declare a valid 'audio_sample_rate', or pass --audio-sample-rate."
            ) from exc
        if resolved_sample_rate <= 0:
            raise RuntimeError("Output audio sample rate must be positive.")
    elif resolved_sample_rate is not None:
        resolved_sample_rate = int(resolved_sample_rate)

    return XToVideoAudioOutput(video, audio, resolved_fps, resolved_sample_rate, output_tensor_range)


def _normalize_output_tensor_range(video: Any, source_range: str) -> Any:
    """Normalize floating-point output according to the pipeline contract."""
    if isinstance(video, torch.Tensor):
        if not video.is_floating_point():
            return video
        video = video.detach().cpu().float().numpy()
    if isinstance(video, np.ndarray):
        if not np.issubdtype(video.dtype, np.floating):
            return video
        if source_range == "negative_one_to_one":
            return np.clip(video, -1.0, 1.0) * 0.5 + 0.5
        if source_range == "zero_to_one":
            return np.clip(video, 0.0, 1.0)
        raise ValueError(f"Unsupported floating-point tensor range: {source_range!r}")
    if isinstance(video, list):
        return [_normalize_output_tensor_range(frame, source_range) for frame in video]
    return video


def encode_x_to_video_audio_output(output: XToVideoAudioOutput) -> bytes:
    """Encode any video layout supported by the shared video API utility."""
    return encode_video_bytes(
        _normalize_output_tensor_range(output.video, output.output_tensor_range),
        output.fps,
        output.audio,
        output.audio_sample_rate,
    )


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
    if args.prompt is None and args.prompt_file is None:
        raise ValueError("Either --prompt or --prompt-file must be provided.")

    text_prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            text_prompt = json.load(f)
            if isinstance(text_prompt, str):
                text_prompt = _clean_official_prompt(text_prompt)
    if not isinstance(text_prompt, str):
        raise ValueError("Prompt content must be a JSON string or --prompt text.")

    parallel_config = DiffusionParallelConfig(
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
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

    omni_kwargs: dict[str, Any] = dict(
        model=args.model,
        parallel_config=parallel_config,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_layerwise_offload=args.enable_layerwise_offload,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
    )
    if args.model_type is not None:
        omni_kwargs["model_type"] = args.model_type
    if args.model_class_name is not None:
        omni_kwargs["model_class_name"] = args.model_class_name
    if args.quantization is not None:
        omni_kwargs["quantization"] = args.quantization
    omni = Omni(**omni_kwargs)
    try:
        model_class_name = args.model_class_name or get_model_class_name(omni)
        input_sample_rate = get_input_audio_sample_rate(model_class_name)
        images, audios = load_image_and_audio(
            args.image_path,
            args.audio_path,
            audio_sample_rate=input_sample_rate,
        )
        media_inputs: dict[str, Any] = {}
        if images:
            media_inputs["image"] = images
        if audios:
            media_inputs["audio"] = audios

        canonical_prompt = build_x_to_video_audio_prompt(text_prompt, media_inputs)
        prompt = build_model_x_to_video_audio_prompt(
            model_class_name,
            canonical_prompt,
            {
                "video_negative_prompt": args.video_negative_prompt,
                "audio_negative_prompt": args.audio_negative_prompt,
            },
        )

        sampling_params = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            extra_args={},
        )
        declared_extra_body_params = get_extra_body_params(model_class_name)
        extra_body = dict(args.extra_body or {})
        legacy_extra_body = {"solver_name": args.solver_name, "shift": args.shift}
        for key, value in legacy_extra_body.items():
            if key in declared_extra_body_params:
                extra_body.setdefault(key, value)
        if declared_extra_body_params:
            apply_declared_extra_args(sampling_params, declared_extra_body_params, extra_body)
        elif extra_body:
            sampling_params.extra_args.update({key: value for key, value in extra_body.items() if value is not None})

        start = time.perf_counter()
        outputs = omni.generate(prompt, sampling_params)
        elapsed = time.perf_counter() - start
        output = extract_x_to_video_audio_output(
            outputs,
            fps=args.fps,
            audio_sample_rate=args.audio_sample_rate,
            output_tensor_range=get_output_tensor_range(model_class_name),
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(encode_x_to_video_audio_output(output))
        print(f"Saved generated video to {output_path}")
        print(f"Total time: {elapsed:.2f}s")
        peak_mb = _extract_peak_memory_mb(outputs)
        if peak_mb:
            print(f"Worker peak GPU memory (reserved): {peak_mb:.2f} MiB ({peak_mb / 1024:.2f} GiB)")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
