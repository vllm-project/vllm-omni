# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script for image editing with Qwen-Image-Edit.

Usage:
    python image_edit.py \
        --image input.png \
        --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
        --output output_image_edit.png \
        --num_inference_steps 50 \
        --cfg_scale 4.0

For more options, run:
    python image_edit.py --help
"""

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.utils.platform_utils import detect_device_type, is_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit an image with Qwen-Image-Edit.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-Edit",
        help="Diffusion model name or local path.",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file (PNG, JPG, etc.).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the edit to make to the image.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=" ",
        required=False,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic results.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        help="True classifier-free guidance scale specific to Qwen-Image-Edit.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_image_edit.png",
        help="Path to save the edited image (PNG).",
    )
    parser.add_argument(
        "--num_outputs_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for the given prompt.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps for the diffusion sampler.",
    )
    parser.add_argument(
        "--ulysses_degree",
        type=int,
        default=1,
        help="Number of GPUs used for ulysses sequence parallelism.",
    )
    parser.add_argument(
        "--ring_degree",
        type=int,
        default=1,
        help="Number of GPUs used for ring sequence parallelism.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input image exists
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")

    # Load input image
    input_image = Image.open(args.image).convert("RGB")
    print(f"Loaded input image from {args.image} (size: {input_image.size})")

    device = detect_device_type()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Enable VAE memory optimizations on NPU
    vae_use_slicing = is_npu()
    vae_use_tiling = is_npu()

    assert args.ring_degree == 1, "Ring attention is not supported yet"
    parallel_config = DiffusionParallelConfig(ulysses_degree=args.ulysses_degree, ring_degree=args.ring_degree)
    omni = Omni(
        model=args.model,
        model_class_name="QwenImageEditPipeline",
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
        parallel_config=parallel_config,
    )

    print("Pipeline loaded")
    # Time profiling for generation
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Parallel configuration: ulysses_degree={args.ulysses_degree}, ring_degree={args.ring_degree}")
    print(f"{'=' * 60}\n")

    generation_start = time.perf_counter()
    # Generate edited image
    images = omni.generate(
        prompt=args.prompt,
        pil_image=input_image,
        negative_prompt=args.negative_prompt,
        generator=generator,
        true_cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        num_outputs_per_prompt=args.num_outputs_per_prompt,
    )
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    # Print profiling results
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    # Save output image(s)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem or "output_image_edit"

    if args.num_outputs_per_prompt <= 1:
        images[0].save(output_path)
        print(f"Saved edited image to {os.path.abspath(output_path)}")
    else:
        for idx, img in enumerate(images):
            save_path = output_path.parent / f"{stem}_{idx}{suffix}"
            img.save(save_path)
            print(f"Saved edited image to {os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()
