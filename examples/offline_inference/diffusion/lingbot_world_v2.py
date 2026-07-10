# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate a LingBot-World v2 video from an image and camera trajectory.

The official checkpoint is licensed separately under CC BY-NC-SA and is
restricted to non-commercial use. This vLLM-Omni integration remains
Apache-2.0 licensed.

Example:
    python examples/offline_inference/diffusion/lingbot_world_v2.py \
        --prompt "The camera moves slowly forward through the scene." \
        --image /path/to/first_frame.png \
        --action-dir /path/to/actions/forward \
        --output lingbot_world_v2.mp4

``--action-dir`` must contain ``poses.npy`` and ``intrinsics.npy``. The
example configures its parent as the trusted action root and sends the
contained directory through ``extra_args["action_path"]``.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MODEL = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"
_NUM_INFERENCE_STEPS = 4
_MAX_SEQUENCE_LENGTH = 512
_MAX_PIXEL_AREA = 480 * 832
_MAX_RAW_FRAMES = 117
_TEMPORAL_COMPRESSION = 4
_LATENT_FRAMES_PER_BLOCK = 3
_SPATIAL_ALIGNMENT = 16


@dataclass(frozen=True)
class LingBotPaths:
    image: Path
    action_dir: Path
    action_root: Path
    action_relative: Path
    output: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LingBot-World v2 video from a first-frame image and camera trajectory."
    )
    parser.add_argument("--model", default=_MODEL, help="Official Hugging Face model ID or local checkpoint path.")
    parser.add_argument("--prompt", required=True, help="Text description of the scene and desired motion.")
    parser.add_argument("--image", required=True, help="Path to the first-frame image.")
    parser.add_argument(
        "--action-dir",
        required=True,
        help="Camera action directory containing poses.npy and intrinsics.npy.",
    )
    parser.add_argument("--height", type=int, default=480, help="Output height; must be divisible by 16.")
    parser.add_argument("--width", type=int, default=832, help="Output width; must be divisible by 16.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=9,
        help="Raw output frame count. Nine frames are one three-latent-frame causal block.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for tensor parallelism inside the DiT.",
    )
    parser.add_argument("--flow-shift", type=float, default=5.0, help="Positive FlowUniPC scheduler shift.")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second in the exported MP4.")
    parser.add_argument("--output", default="lingbot_world_v2.mp4", help="Output MP4 path.")
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable compilation and force eager execution.",
    )
    return parser.parse_args(argv)


def resolve_cli_paths(args: argparse.Namespace) -> LingBotPaths:
    """Canonicalize local inputs and construct a safe trusted-root mapping."""

    image = Path(args.image).expanduser().resolve()
    if not image.is_file():
        raise ValueError("--image must point to an existing file.")

    action_dir = Path(args.action_dir).expanduser().resolve()
    if not action_dir.is_dir():
        raise ValueError("--action-dir must point to an existing directory.")
    if not all((action_dir / name).is_file() for name in ("poses.npy", "intrinsics.npy")):
        raise ValueError("--action-dir must contain poses.npy and intrinsics.npy.")

    action_root = action_dir.parent.resolve()
    try:
        action_relative = action_dir.relative_to(action_root)
    except ValueError:
        raise ValueError("--action-dir must be contained by its canonical trusted action root.") from None
    if not action_relative.parts:
        raise ValueError("--action-dir must name a directory contained by a trusted action root.")

    output = Path(args.output).expanduser().resolve()
    return LingBotPaths(
        image=image,
        action_dir=action_dir,
        action_root=action_root,
        action_relative=action_relative,
        output=output,
    )


def _positive_finite(value: float, flag: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{flag} must be a positive finite number.")
    return result


def build_omni_kwargs(
    args: argparse.Namespace,
    paths: LingBotPaths,
    *,
    parallel_config: Any,
) -> dict[str, Any]:
    """Build Omni configuration without overriding checkpoint class discovery."""

    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be a positive integer.")
    flow_shift = _positive_finite(args.flow_shift, "--flow-shift")
    model_path = Path(args.model).expanduser()
    model = str(model_path.resolve()) if model_path.exists() else args.model
    return {
        "model": model,
        "flow_shift": flow_shift,
        "parallel_config": parallel_config,
        "enforce_eager": args.enforce_eager,
        "model_config": {"lingbot_action_root": str(paths.action_root)},
    }


def build_request(
    args: argparse.Namespace,
    paths: LingBotPaths,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the prompt and fixed v1 causal-DMD sampling contract."""

    prompt_text = args.prompt.strip()
    if not prompt_text:
        raise ValueError("--prompt must contain non-whitespace text.")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive integers.")
    if args.height % _SPATIAL_ALIGNMENT or args.width % _SPATIAL_ALIGNMENT:
        raise ValueError("--height and --width must each be divisible by 16.")
    if args.height * args.width > _MAX_PIXEL_AREA:
        raise ValueError("--height * --width must not exceed 480 * 832 pixels.")
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be a positive integer.")
    if args.num_frames > _MAX_RAW_FRAMES:
        raise ValueError("--num-frames must not exceed 117 raw frames.")
    if (args.num_frames - 1) % _TEMPORAL_COMPRESSION:
        raise ValueError("--num-frames must satisfy (num_frames - 1) divisible by 4.")
    latent_frames = (args.num_frames - 1) // _TEMPORAL_COMPRESSION + 1
    if latent_frames % _LATENT_FRAMES_PER_BLOCK:
        raise ValueError("--num-frames must map to whole three-frame latent blocks.")
    if args.fps <= 0:
        raise ValueError("--fps must be a positive integer.")
    flow_shift = _positive_finite(args.flow_shift, "--flow-shift")

    prompt = {
        "prompt": prompt_text,
        "multi_modal_data": {"image": str(paths.image)},
    }
    sampling_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": _NUM_INFERENCE_STEPS,
        "max_sequence_length": _MAX_SEQUENCE_LENGTH,
        "seed": args.seed,
        "fps": args.fps,
        "extra_args": {
            "action_path": paths.action_relative.as_posix(),
            "flow_shift": flow_shift,
        },
    }
    return prompt, sampling_kwargs


def extract_video_array(outputs: Any) -> np.ndarray:
    """Unwrap one standard Omni diffusion video into ``[frames, H, W, C]``."""

    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("Expected exactly one Omni output for the single LingBot request.")
    output = outputs[0]
    request_output = getattr(output, "request_output", None)
    if request_output is not None and getattr(request_output, "images", None):
        output = request_output

    images = getattr(output, "images", None)
    if not images:
        multimodal_output = getattr(output, "multimodal_output", None) or {}
        images = multimodal_output.get("video")
    if images is None:
        raise ValueError("No video frames found in Omni output.")
    if isinstance(images, list) and len(images) == 1:
        images = images[0]

    video = np.asarray(images)
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"Expected one generated video, got batch shape {video.shape}.")
        video = video[0]
    if video.ndim != 4 or video.shape[-1] not in (3, 4):
        raise ValueError(f"Expected video shape [frames, height, width, channels], got {video.shape}.")
    return video


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    paths = resolve_cli_paths(args)
    prompt, sampling_kwargs = build_request(args, paths)
    paths.output.parent.mkdir(parents=True, exist_ok=True)

    # Keep vLLM imports below pure CLI/path validation so ``--help`` and the
    # helper tests work on machines without the native vLLM runtime.
    from diffusers.utils import export_to_video

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    parallel_config = DiffusionParallelConfig(tensor_parallel_size=args.tensor_parallel_size)
    omni_kwargs = build_omni_kwargs(args, paths, parallel_config=parallel_config)
    omni = Omni(**omni_kwargs)
    try:
        outputs = omni.generate(
            prompt,
            OmniDiffusionSamplingParams(**sampling_kwargs),
            use_tqdm=False,
        )
    finally:
        omni.close()

    frames = extract_video_array(outputs)
    export_to_video(frames, str(paths.output), fps=args.fps)
    print(f"Saved {frames.shape[0]} frames to {paths.output}")
    return paths.output


if __name__ == "__main__":
    main()
