# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Helios video generation example supporting T2V, I2V, and V2V.

Usage (T2V, Helios-Base, Stage 1 only):
    python end2end.py \
        --model /path/to/Helios-Base --sample-type t2v \
        --prompt "A serene lakeside sunrise with mist over the water." \
        --height 384 --width 640 --num-frames 99 \
        --num-inference-steps 50 --guidance-scale 5.0

Usage (I2V, Helios-Base):
    python end2end.py \
        --model /path/to/Helios-Base --sample-type i2v \
        --image-path /path/to/image.jpg \
        --prompt "Description of desired animation." \
        --guidance-scale 5.0

Usage (V2V, Helios-Base):
    python end2end.py \
        --model /path/to/Helios-Base --sample-type v2v \
        --video-path /path/to/video.mp4 \
        --prompt "Description of desired transformation." \
        --guidance-scale 5.0

Usage (Helios-Mid, Stage 2 + CFG-Zero*):
    python end2end.py \
        --model /path/to/Helios-Mid --sample-type t2v \
        --prompt "A serene lakeside sunrise with mist over the water." \
        --guidance-scale 5.0 \
        --is-enable-stage2 \
        --pyramid-num-inference-steps-list 20 20 20 \
        --use-cfg-zero-star --use-zero-init --zero-steps 1

Usage (Helios-Distilled, Stage 2 pyramid + DMD):
    python end2end.py \
        --model /path/to/Helios-Distilled --sample-type t2v \
        --prompt "A serene lakeside sunrise with mist over the water." \
        --guidance-scale 1.0 \
        --is-enable-stage2 \
        --pyramid-num-inference-steps-list 2 2 2 \
        --is-amplify-first-chunk
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def load_image_as_tensor(image_path: str, height: int, width: int) -> torch.Tensor:
    """Load an image and return a (1, C, H, W) tensor normalized to [-1, 1]."""
    from PIL import Image
    from torchvision import transforms

    image = Image.open(image_path).convert("RGB").resize((width, height))
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(image).unsqueeze(0)


def load_video_as_tensor(video_path: str, height: int, width: int) -> torch.Tensor:
    """Load a video and return a (1, C, T, H, W) tensor normalized to [-1, 1]."""
    from torchvision import transforms
    from torchvision.io import read_video

    video_frames, _, _ = read_video(video_path, output_format="TCHW")
    video_frames = video_frames.float() / 255.0

    transform = transforms.Compose(
        [
            transforms.Resize((height, width), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    video_frames = torch.stack([transform(f) for f in video_frames])
    return video_frames.permute(1, 0, 2, 3).unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a video with Helios (T2V / I2V / V2V).")
    parser.add_argument(
        "--model",
        default="BestWishYsh/Helios-Base",
        help="Helios model ID or local path (e.g. Helios-Base, Helios-Mid, Helios-Distilled).",
    )
    parser.add_argument(
        "--sample-type",
        choices=["t2v", "i2v", "v2v"],
        default="t2v",
        help="Generation mode: t2v (text-to-video), i2v (image-to-video), v2v (video-to-video).",
    )
    parser.add_argument("--prompt", default="A serene lakeside sunrise with mist over the water.", help="Text prompt.")
    parser.add_argument(
        "--negative-prompt",
        default=(
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
            "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
            "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
            "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
            "in the background, walking backwards"
        ),
        help="Negative prompt.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG scale.")
    parser.add_argument("--height", type=int, default=384, help="Video height.")
    parser.add_argument("--width", type=int, default=640, help="Video width.")
    parser.add_argument("--num-frames", type=int, default=99, help="Number of video frames.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Sampling steps (Stage 1 only).")
    parser.add_argument("--output", type=str, default="helios_output.mp4", help="Output video path.")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for the output video.")

    # I2V / V2V
    parser.add_argument("--image-path", type=str, default=None, help="Input image path for I2V mode.")
    parser.add_argument("--video-path", type=str, default=None, help="Input video path for V2V mode.")

    # Stage 2 (pyramid multi-stage denoising)
    parser.add_argument(
        "--is-enable-stage2",
        action="store_true",
        help="Enable pyramid multi-stage denoising (Stage 2). Required for Helios-Distilled.",
    )
    parser.add_argument(
        "--pyramid-num-stages",
        type=int,
        default=3,
        help="Number of pyramid stages for Stage 2.",
    )
    parser.add_argument(
        "--pyramid-num-inference-steps-list",
        type=int,
        nargs="+",
        default=[10, 10, 10],
        help="Inference steps per pyramid stage.",
    )

    # DMD
    parser.add_argument(
        "--is-amplify-first-chunk",
        action="store_true",
        help="Enable DMD amplification for the first chunk (Helios-Distilled).",
    )

    # CFG Zero Star
    parser.add_argument(
        "--use-cfg-zero-star",
        action="store_true",
        help="Enable CFG Zero Star guidance (recommended for Helios-Mid).",
    )
    parser.add_argument(
        "--use-zero-init",
        action="store_true",
        help="Use zero initialization for the first denoising steps with CFG-Zero*.",
    )
    parser.add_argument(
        "--zero-steps",
        type=int,
        default=1,
        help="Number of initial denoising steps using zero prediction (default: 1).",
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
        help="Quantization method (e.g. 'fp8').",
    )

    # Memory & parallelism
    parser.add_argument("--vae-use-slicing", action="store_true", help="Enable VAE slicing.")
    parser.add_argument("--vae-use-tiling", action="store_true", help="Enable VAE tiling.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--enable-cpu-offload", action="store_true", help="Enable CPU offloading.")
    parser.add_argument("--enable-layerwise-offload", action="store_true", help="Enable layerwise offloading.")
    parser.add_argument("--ulysses-degree", type=int, default=1, help="Ulysses SP degree.")
    parser.add_argument("--ring-degree", type=int, default=1, help="Ring SP degree.")
    parser.add_argument("--cfg-parallel-size", type=int, default=1, choices=[1, 2], help="CFG parallel size.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism size.")

    return parser.parse_args()


def main():
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    omni = Omni(
        model=args.model,
        enable_layerwise_offload=args.enable_layerwise_offload,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        enable_cpu_offload=args.enable_cpu_offload,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
        quantization=args.quantization,
    )

    # Validate I2V / V2V arguments
    if args.sample_type == "i2v" and not args.image_path:
        raise ValueError("--image-path is required for I2V mode.")
    if args.sample_type == "v2v" and not args.video_path:
        raise ValueError("--video-path is required for V2V mode.")

    # Build extra_args for Helios-specific parameters
    extra_args = {}
    if args.is_enable_stage2:
        extra_args["is_enable_stage2"] = True
        extra_args["pyramid_num_stages"] = args.pyramid_num_stages
        extra_args["pyramid_num_inference_steps_list"] = args.pyramid_num_inference_steps_list
    if args.is_amplify_first_chunk:
        extra_args["is_amplify_first_chunk"] = True
    if args.use_cfg_zero_star:
        extra_args["use_cfg_zero_star"] = True
    if args.use_zero_init:
        extra_args["use_zero_init"] = True
        extra_args["zero_steps"] = args.zero_steps

    if args.sample_type == "i2v":
        image_tensor = load_image_as_tensor(args.image_path, args.height, args.width)
        extra_args["image"] = image_tensor
    elif args.sample_type == "v2v":
        video_tensor = load_video_as_tensor(args.video_path, args.height, args.width)
        extra_args["video"] = video_tensor

    # Print generation configuration
    print(f"\n{'=' * 60}")
    print("Helios Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Sample type: {args.sample_type.upper()}")
    if args.sample_type == "i2v":
        print(f"  Image: {args.image_path}")
    elif args.sample_type == "v2v":
        print(f"  Video: {args.video_path}")
    print(f"  Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"  Video size: {args.width}x{args.height}, {args.num_frames} frames")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Guidance scale: {args.guidance_scale}")
    if args.is_enable_stage2:
        print(f"  Stage 2: enabled (stages={args.pyramid_num_stages}, steps={args.pyramid_num_inference_steps_list})")
        if args.is_amplify_first_chunk:
            print("  DMD amplify first chunk: enabled")
        if args.use_cfg_zero_star:
            print(f"  CFG Zero Star: enabled (zero_init={args.use_zero_init}, zero_steps={args.zero_steps})")
    else:
        if args.use_cfg_zero_star:
            print(f"  CFG Zero Star: enabled (zero_init={args.use_zero_init}, zero_steps={args.zero_steps})")
        print("  Stage 2: disabled (Stage 1 only)")
    print(f"{'=' * 60}\n")

    generation_start = time.perf_counter()
    frames = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
        },
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            extra_args=extra_args,
        ),
    )
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    # Extract video frames from OmniRequestOutput
    if isinstance(frames, list) and len(frames) > 0:
        first_item = frames[0]

        if hasattr(first_item, "final_output_type"):
            if first_item.final_output_type != "image":
                raise ValueError(
                    f"Unexpected output type '{first_item.final_output_type}', expected 'image' for video generation."
                )

            if hasattr(first_item, "is_pipeline_output") and first_item.is_pipeline_output:
                inner_output = first_item.request_output
                if isinstance(inner_output, OmniRequestOutput) and hasattr(inner_output, "images"):
                    frames = inner_output.images[0] if inner_output.images else None
                    if frames is None:
                        raise ValueError("No video frames found in output.")
            elif hasattr(first_item, "images") and first_item.images:
                frames = first_item.images
            else:
                raise ValueError("No video frames found in OmniRequestOutput.")

    # Unwrap batch list from postprocess_video: [numpy(T,H,W,C)] -> numpy(T,H,W,C)
    if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], np.ndarray):
        frames = frames[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from diffusers.utils import export_to_video
    except ImportError:
        raise ImportError("diffusers is required for export_to_video.")

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
    else:
        video_array = frames

    if isinstance(video_array, np.ndarray) and video_array.ndim == 5:
        video_array = video_array[0]

    export_to_video(video_array, str(output_path), fps=args.fps)
    print(f"Saved generated video to {output_path}")


if __name__ == "__main__":
    main()
