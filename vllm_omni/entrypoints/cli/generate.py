# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate subcommand for vLLM-Omni."""

import argparse
from pathlib import Path
from typing import Any

from PIL import Image
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.cli.diffusion_args import (
    add_diffusion_cfg_parallel_arg,
    add_diffusion_cpu_offload_arg,
    add_diffusion_sequence_parallel_args,
    add_diffusion_vae_memory_args,
    add_diffusion_vae_patch_parallel_arg,
    add_diffusion_weight_loading_args,
    add_stage_configs_path_arg,
    add_tensor_parallel_size_arg,
)

logger = init_logger(__name__)

DESCRIPTION = """Generate images or videos using diffusion models.
Diffusion parallel and memory optimization knobs are shared with
`vllm serve --omni` where they apply to offline generation.

Examples:
  vllm generate --model Qwen/Qwen-Image \\
    --prompt "a cup of coffee on the table" --output output.png

  vllm generate --task i2i --model Qwen/Qwen-Image-Edit \\
    --prompt "make it watercolor" --input-image input.png --output edited.png

  vllm generate --task t2v --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \\
    --prompt "a sunset over the ocean" --num-frames 81 --output output.mp4

  vllm generate --task i2v --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \\
    --prompt "make the scene cinematic" --input-image input.png --output output.mp4

  vllm generate --model Qwen/Qwen-Image \\
    --prompt "a sunset over the ocean" --height 512 --width 512 \\
    --num-inference-steps 30 --seed 123

  vllm generate --model Qwen/Qwen-Image \\
    --prompt "a landscape" --tensor-parallel-size 2 --vae-use-tiling
"""

_IMAGE_TASKS = {"t2i", "i2i"}
_VIDEO_TASKS = {"t2v", "i2v"}
_TASK_ALIASES = {
    "text-to-image": "t2i",
    "image-to-image": "i2i",
    "text-to-video": "t2v",
    "image-to-video": "i2v",
}
_TASK_TO_MODE = {
    "t2i": "text-to-image",
    "i2i": "image-to-image",
    "t2v": "text-to-video",
    "i2v": "image-to-video",
}


def _normalize_task(task: str) -> str:
    return _TASK_ALIASES.get(task, task)


def _load_input_images(paths: list[str]) -> Image.Image | list[Image.Image] | None:
    if not paths:
        return None
    images = [Image.open(path).convert("RGB") for path in paths]
    return images[0] if len(images) == 1 else images


def _build_prompt(args: argparse.Namespace) -> dict[str, Any]:
    prompt: dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
    }
    input_images = _load_input_images(getattr(args, "input_image", []))
    if input_images is not None:
        prompt["multi_modal_data"] = {"image": input_images}
    return prompt


def _first_output(outputs: list[Any]) -> Any:
    if not outputs:
        raise RuntimeError("No output generated.")
    output = outputs[0]
    seen: set[int] = set()
    while hasattr(output, "request_output") and getattr(output, "request_output") is not None:
        if id(output) in seen:
            break
        seen.add(id(output))
        output = output.request_output
    return output


def _extract_images(outputs: list[Any]) -> list[Image.Image]:
    output = _first_output(outputs)
    images = getattr(output, "images", None)
    if images is None and hasattr(output, "multimodal_output"):
        images = output.multimodal_output.get("image")
    if not images:
        raise RuntimeError("No images in output.")
    return images


def _save_images(images: list[Image.Image], output: str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    if len(images) == 1:
        save_path = output_path if output_path.suffix else output_path.with_suffix(suffix)
        images[0].save(save_path)
        print(f"Saved image to {save_path}")
        return

    stem = output_path.stem
    for i, img in enumerate(images):
        save_path = output_path.parent / f"{stem}_{i}{suffix}"
        img.save(save_path)
        print(f"Saved image to {save_path}")


def _normalize_video_outputs(videos: Any) -> list[Any]:
    if videos is None:
        return []
    if hasattr(videos, "ndim") and videos.ndim == 5:
        return [videos[i] for i in range(videos.shape[0])]
    if isinstance(videos, list):
        if not videos:
            return []
        first = videos[0]
        if isinstance(first, tuple) and len(first) == 2:
            return [first[0]]
        if isinstance(first, dict):
            return [item.get("frames") or item.get("video") for item in videos]
        if isinstance(first, list):
            return videos
        if hasattr(first, "ndim") and first.ndim == 3:
            return [videos]
        if isinstance(first, Image.Image):
            return [videos]
        return videos
    if isinstance(videos, tuple) and len(videos) == 2:
        return [videos[0]]
    if isinstance(videos, dict):
        video = videos.get("frames") or videos.get("video")
        return [] if video is None else [video]
    return [videos]


def _extract_videos(outputs: list[Any]) -> list[Any]:
    output = _first_output(outputs)
    videos = getattr(output, "images", None)
    if videos is None and hasattr(output, "multimodal_output"):
        videos = output.multimodal_output.get("video")
    videos = _normalize_video_outputs(videos)
    if not videos:
        raise RuntimeError("No videos in output.")
    return videos


def _normalize_frame(frame: Any) -> Any:
    import numpy as np
    import torch

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
    if isinstance(frame, Image.Image):
        return np.asarray(frame).astype(np.float32) / 255.0
    return frame


def _prepare_video_for_export(video: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(video, torch.Tensor):
        video_tensor = video.detach().cpu()
        if video_tensor.dim() == 5:
            if video_tensor.shape[1] in (3, 4):
                video_tensor = video_tensor[0].permute(1, 2, 3, 0)
            else:
                video_tensor = video_tensor[0]
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        return video_tensor.float().numpy()
    if isinstance(video, np.ndarray):
        video_array = video[0] if video.ndim == 5 else video
        if np.issubdtype(video_array.dtype, np.integer):
            video_array = video_array.astype(np.float32) / 255.0
        return video_array
    if isinstance(video, list):
        return [_normalize_frame(frame) for frame in video]
    return video


def _save_videos(videos: list[Any], output: str, fps: int) -> None:
    try:
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise ImportError("diffusers is required for video export.") from exc

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".mp4"
    if len(videos) == 1:
        save_path = output_path if output_path.suffix else output_path.with_suffix(suffix)
        export_to_video(_prepare_video_for_export(videos[0]), str(save_path), fps=fps)
        print(f"Saved video to {save_path}")
        return

    stem = output_path.stem
    for i, video in enumerate(videos):
        save_path = output_path.parent / f"{stem}_{i}{suffix}"
        export_to_video(_prepare_video_for_export(video), str(save_path), fps=fps)
        print(f"Saved video to {save_path}")


class OmniGenerateCommand(CLISubcommand):
    """The `generate` subcommand for the vLLM-Omni CLI."""

    name = "generate"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        import torch

        from vllm_omni.diffusion.data import DiffusionParallelConfig
        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        from vllm_omni.platforms import current_omni_platform

        args.task = _normalize_task(getattr(args, "task", "t2i"))

        # 1. torch.Generator for reproducible generation
        device_type = current_omni_platform.device_type
        generator = torch.Generator(device=device_type).manual_seed(args.seed)

        # 2. Parallel config
        parallel_config = DiffusionParallelConfig(
            tensor_parallel_size=args.tensor_parallel_size,
            ulysses_degree=getattr(args, "ulysses_degree", None) or 1,
            ring_degree=getattr(args, "ring_degree", None) or 1,
            ulysses_mode=getattr(args, "ulysses_mode", "strict"),
            cfg_parallel_size=getattr(args, "cfg_parallel_size", 1),
            vae_patch_parallel_size=getattr(args, "vae_patch_parallel_size", 1),
        )

        # 3. Initialize Omni
        omni_kwargs: dict = {
            "model": args.model,
            "parallel_config": parallel_config,
            "enforce_eager": args.enforce_eager,
            "enable_cpu_offload": args.enable_cpu_offload,
            "vae_use_slicing": getattr(args, "vae_use_slicing", False),
            "vae_use_tiling": getattr(args, "vae_use_tiling", False),
            "enable_multithread_weight_load": getattr(args, "enable_multithread_weight_load", True),
            "num_weight_load_threads": getattr(args, "num_weight_load_threads", 4),
            "mode": _TASK_TO_MODE[args.task],
        }
        if args.stage_configs_path:
            omni_kwargs["stage_configs_path"] = args.stage_configs_path
        omni = Omni(**omni_kwargs)

        # 4. Generate
        sampling_params = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            true_cfg_scale=args.cfg_scale,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_2,
            num_inference_steps=args.num_inference_steps,
            num_outputs_per_prompt=args.num_images,
        )
        if args.task in _VIDEO_TASKS:
            sampling_params.num_frames = args.num_frames
            sampling_params.frame_rate = args.fps
            sampling_params.fps = args.fps

        outputs = omni.generate(_build_prompt(args), sampling_params)

        # 5. Save outputs
        if args.task in _VIDEO_TASKS:
            _save_videos(_extract_videos(outputs), args.output, args.fps)
        else:
            _save_images(_extract_images(outputs), args.output)

    def validate(self, args: argparse.Namespace) -> None:
        from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model

        if not is_diffusion_model(args.model):
            raise ValueError(
                f"'{args.model}' is not a diffusion model. 'vllm generate' currently only supports diffusion models."
            )
        args.task = _normalize_task(args.task)
        if args.task in {"i2i", "i2v"} and not args.input_image:
            raise ValueError(f"'--input-image' is required for task '{args.task}'.")
        if args.task in {"t2i", "t2v"} and args.input_image:
            raise ValueError(f"'--input-image' is only supported for image-conditioned tasks, got '{args.task}'.")

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            description=DESCRIPTION,
            usage="vllm generate --model MODEL --prompt PROMPT [--task TASK] [options]",
        )

        # Keep accepting the old form: vllm generate --omni ...
        parser.add_argument(
            "--omni",
            action="store_true",
            help=argparse.SUPPRESS,
        )

        # Required
        parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Diffusion model name or path.",
        )
        parser.add_argument(
            "--prompt",
            type=str,
            required=True,
            help="Text prompt for generation.",
        )
        parser.add_argument(
            "--task",
            type=str,
            default="t2i",
            choices=["t2i", "i2i", "t2v", "i2v", "text-to-image", "image-to-image", "text-to-video", "image-to-video"],
            help="Generation task (default: t2i).",
        )
        parser.add_argument(
            "--input-image",
            action="append",
            default=[],
            help="Input image path for image-conditioned tasks. Can be passed multiple times.",
        )

        # Output
        parser.add_argument(
            "--output",
            type=str,
            default="output",
            help="Output path. Defaults to output.png for image tasks and output.mp4 for video tasks.",
        )

        # Generation parameters
        parser.add_argument(
            "--height",
            type=int,
            default=1024,
            help="Image height (default: 1024).",
        )
        parser.add_argument(
            "--width",
            type=int,
            default=1024,
            help="Image width (default: 1024).",
        )
        parser.add_argument(
            "--num-inference-steps",
            type=int,
            default=50,
            help="Number of denoising steps (default: 50).",
        )
        parser.add_argument(
            "--guidance-scale",
            type=float,
            default=4.0,
            help="CFG guidance scale (default: 4.0).",
        )
        parser.add_argument(
            "--guidance-scale-2",
            type=float,
            default=None,
            help="Secondary CFG guidance scale used by some video/image-edit pipelines.",
        )
        parser.add_argument(
            "--cfg-scale",
            type=float,
            default=4.0,
            help="True CFG scale for Qwen-Image (default: 4.0).",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed (default: 42).",
        )
        parser.add_argument(
            "--num-images",
            type=int,
            default=1,
            help="Number of images/videos to generate (default: 1).",
        )
        parser.add_argument(
            "--num-frames",
            type=int,
            default=81,
            help="Number of frames for video tasks (default: 81).",
        )
        parser.add_argument(
            "--fps",
            type=int,
            default=24,
            help="Output video frame rate (default: 24).",
        )
        parser.add_argument(
            "--negative-prompt",
            type=str,
            default=None,
            help="Negative prompt for CFG.",
        )

        # Hardware / loading
        add_tensor_parallel_size_arg(parser)
        add_diffusion_sequence_parallel_args(parser)
        add_diffusion_cfg_parallel_arg(parser)
        add_diffusion_vae_patch_parallel_arg(parser)
        add_stage_configs_path_arg(parser)
        parser.add_argument(
            "--enforce-eager",
            action="store_true",
            help="Disable torch.compile.",
        )
        add_diffusion_vae_memory_args(parser)
        add_diffusion_weight_loading_args(parser)
        add_diffusion_cpu_offload_arg(parser)

        return parser


def cmd_init() -> list[CLISubcommand]:
    return [OmniGenerateCommand()]
