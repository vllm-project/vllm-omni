from __future__ import annotations

import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import requests
import torch
from diffusers import UniPCMultistepScheduler, WanImageToVideoPipeline
from diffusers.pipelines.wan import pipeline_wan_i2v as wan_i2v_module
from diffusers.utils import export_to_video, load_image
from PIL import Image

from tests.e2e.accuracy.wan22_i2v.wan22_i2v_video_similarity_common import BOUNDARY_RATIO


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wan2.2 I2V diffusers offline generation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-source", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--guidance-scale-2", type=float, required=True)
    parser.add_argument("--flow-shift", type=float, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    return parser.parse_args()


def _parse_size(size: str) -> tuple[int, int]:
    width_str, height_str = size.lower().split("x", 1)
    return int(width_str), int(height_str)


class _IdentityFtfy:
    @staticmethod
    def fix_text(text: str) -> str:
        return text


def _ensure_wan_ftfy_fallback() -> None:
    if not hasattr(wan_i2v_module, "ftfy"):
        wan_i2v_module.ftfy = _IdentityFtfy()


def _offline_cuda_device() -> torch.device:
    return torch.device("cuda:0")


def _load_input_image(source: str) -> Image.Image:
    if source.startswith("data:image"):
        _, encoded = source.split(",", 1)
        image = Image.open(BytesIO(base64.b64decode(encoded)))
        image.load()
        return image.convert("RGB")

    source_path = Path(source)
    if source_path.exists():
        image = Image.open(source_path)
        image.load()
        return image.convert("RGB")

    image = load_image(source)
    if isinstance(image, Image.Image):
        image.load()
        return image.convert("RGB")

    response = requests.get(source, timeout=60)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    image.load()
    return image.convert("RGB")


def _resize_to_target(image: Image.Image, *, width: int, height: int) -> Image.Image:
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _configure_scheduler(pipe: WanImageToVideoPipeline, *, flow_shift: float) -> None:
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=flow_shift,
    )


def _write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    width: int,
    height: int,
    frame_count: int,
) -> None:
    payload = {
        "model": args.model,
        "image_source": args.image_source,
        "size": args.size,
        "width": width,
        "height": height,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "actual_frame_count": frame_count,
        "guidance_scale": args.guidance_scale,
        "guidance_scale_2": args.guidance_scale_2,
        "boundary_ratio": BOUNDARY_RATIO,
        "flow_shift": args.flow_shift,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "world_size": 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    device = _offline_cuda_device()
    torch.cuda.set_device(device)
    _ensure_wan_ftfy_fallback()

    pipe = WanImageToVideoPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.register_to_config(boundary_ratio=BOUNDARY_RATIO)
    _configure_scheduler(pipe, flow_shift=args.flow_shift)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    input_image = _load_input_image(args.image_source)
    width, height = _parse_size(args.size)
    resized_image = _resize_to_target(input_image, width=width, height=height)

    generator = torch.Generator(device=device.type).manual_seed(args.seed)
    frames = pipe(
        image=resized_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=height,
        width=width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    ).frames[0]

    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=args.fps)
    _write_metadata(metadata_path, args=args, width=width, height=height, frame_count=len(frames))

    if hasattr(pipe, "maybe_free_model_hooks"):
        pipe.maybe_free_model_hooks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
