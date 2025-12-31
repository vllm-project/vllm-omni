# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script for image editing with Qwen-Image-Edit.

Usage (single image):
    python image_edit.py \
        --image input.png \
        --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
        --output output_image_edit.png \
        --num_inference_steps 50 \
        --cfg_scale 4.0 \
        --guidance_scale 1.0

Usage (multiple images):
    python image_edit.py \
        --image input1.png input2.png input3.png \
        --prompt "Combine these images into a single scene" \
        --output output_image_edit.png \
        --num_inference_steps 50 \
        --cfg_scale 4.0 \
        --guidance_scale 1.0

Usage (with cache-dit acceleration):
    python image_edit.py \
        --image input.png \
        --prompt "Edit description" \
        --cache_backend cache_dit \
        --cache_dit_max_continuous_cached_steps 3 \
        --cache_dit_residual_diff_threshold 0.24 \
        --cache_dit_enable_taylorseer

Usage (with tea_cache acceleration):
    python image_edit.py \
        --image input.png \
        --prompt "Edit description" \
        --cache_backend tea_cache \
        --tea_cache_rel_l1_thresh 0.25

Usage (layered):
    python image_edit.py \
        --model "Qwen/Qwen-Image-Layered" \
        --image input.png \
        --prompt "" \
        --output "layered" \
        --num_inference_steps 50 \
        --cfg_scale 4.0 \
        --layers 4 \
        --color-format "RGBA"

For more options, run:
    python image_edit.py --help
"""

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image

from vllm_omni.diffusion.data import DiffusionParallelConfig, logger
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.utils.platform_utils import detect_device_type, is_npu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit an image with Qwen-Image-Edit.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-Edit",
        help=(
            "Diffusion model name or local path. "
            "For multiple image inputs, use Qwen/Qwen-Image-Edit-2509 or Qwen/Qwen-Image-Edit-2511"
            "which supports QwenImageEditPlusPipeline."
        ),
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to input image file(s) (PNG, JPG, etc.). Can specify multiple images.",
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
        help=(
            "True classifier-free guidance scale (default: 4.0). Guidance scale as defined in Classifier-Free "
            "Diffusion Guidance. Classifier-free guidance is enabled by setting cfg_scale > 1 and providing "
            "a negative_prompt. Higher guidance scale encourages images closely linked to the text prompt, "
            "usually at the expense of lower image quality."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help=(
            "Guidance scale for guidance-distilled models (default: 1.0, disabled). "
            "Unlike classifier-free guidance (--cfg_scale), guidance-distilled models take the guidance scale "
            "directly as an input parameter. Enabled when guidance_scale > 1. Ignored when not using guidance-distilled models."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_image_edit.png",
        help=("Path to save the edited image (PNG). Or prefix for Qwen-Image-Layered model save images(PNG)."),
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
        "--cache_backend",
        type=str,
        default=None,
        choices=["cache_dit", "tea_cache"],
        help=(
            "Cache backend to use for acceleration. "
            "Options: 'cache_dit' (DBCache + SCM + TaylorSeer), 'tea_cache' (Timestep Embedding Aware Cache). "
            "Default: None (no cache acceleration)."
        ),
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
    parser.add_argument("--layers", type=int, default=4, help="Number of layers to decompose the input image into.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        help="Bucket in (640, 1024) to determine the condition and output resolution",
    )

    parser.add_argument(
        "--color-format",
        type=str,
        default="RGB",
        help="For Qwen-Image-Layered, set to RGBA.",
    )

    # Cache-DiT specific parameters
    parser.add_argument(
        "--cache_dit_fn_compute_blocks",
        type=int,
        default=1,
        help="[cache-dit] Number of forward compute blocks. Optimized for single-transformer models.",
    )
    parser.add_argument(
        "--cache_dit_bn_compute_blocks",
        type=int,
        default=0,
        help="[cache-dit] Number of backward compute blocks.",
    )
    parser.add_argument(
        "--cache_dit_max_warmup_steps",
        type=int,
        default=4,
        help="[cache-dit] Maximum warmup steps (works for few-step models).",
    )
    parser.add_argument(
        "--cache_dit_residual_diff_threshold",
        type=float,
        default=0.24,
        help="[cache-dit] Residual diff threshold. Higher values enable more aggressive caching.",
    )
    parser.add_argument(
        "--cache_dit_max_continuous_cached_steps",
        type=int,
        default=3,
        help="[cache-dit] Maximum continuous cached steps to prevent precision degradation.",
    )
    parser.add_argument(
        "--cache_dit_enable_taylorseer",
        action="store_true",
        default=False,
        help="[cache-dit] Enable TaylorSeer acceleration (not suitable for few-step models).",
    )
    parser.add_argument(
        "--cache_dit_taylorseer_order",
        type=int,
        default=1,
        help="[cache-dit] TaylorSeer polynomial order.",
    )
    parser.add_argument(
        "--cache_dit_scm_steps_mask_policy",
        type=str,
        default=None,
        choices=[None, "slow", "medium", "fast", "ultra"],
        help="[cache-dit] SCM mask policy: None (disabled), slow, medium, fast, ultra.",
    )
    parser.add_argument(
        "--cache_dit_scm_steps_policy",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="[cache-dit] SCM steps policy: dynamic or static.",
    )

    # TeaCache specific parameters
    parser.add_argument(
        "--tea_cache_rel_l1_thresh",
        type=float,
        default=0.2,
        help="[tea_cache] Threshold for accumulated relative L1 distance.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input images exist and load them
    input_images = []
    for image_path in args.image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        img = Image.open(image_path).convert(args.color_format)
        input_images.append(img)

    # Use single image or list based on number of inputs
    if len(input_images) == 1:
        input_image = input_images[0]
    else:
        input_image = input_images

    device = detect_device_type()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Enable VAE memory optimizations on NPU
    vae_use_slicing = is_npu()
    vae_use_tiling = is_npu()
    parallel_config = DiffusionParallelConfig(ulysses_degree=args.ulysses_degree, ring_degree=args.ring_degree)
    # Configure cache based on backend type
    cache_config = None
    if args.cache_backend == "cache_dit":
        # cache-dit configuration: Hybrid DBCache + SCM + TaylorSeer
        cache_config = {
            "Fn_compute_blocks": args.cache_dit_fn_compute_blocks,
            "Bn_compute_blocks": args.cache_dit_bn_compute_blocks,
            "max_warmup_steps": args.cache_dit_max_warmup_steps,
            "residual_diff_threshold": args.cache_dit_residual_diff_threshold,
            "max_continuous_cached_steps": args.cache_dit_max_continuous_cached_steps,
            "enable_taylorseer": args.cache_dit_enable_taylorseer,
            "taylorseer_order": args.cache_dit_taylorseer_order,
            "scm_steps_mask_policy": args.cache_dit_scm_steps_mask_policy,
            "scm_steps_policy": args.cache_dit_scm_steps_policy,
        }
    elif args.cache_backend == "tea_cache":
        # TeaCache configuration
        cache_config = {
            "rel_l1_thresh": args.tea_cache_rel_l1_thresh,
            # Note: coefficients will use model-specific defaults based on model_type
        }

    # Initialize Omni with appropriate pipeline
    omni = Omni(
        model=args.model,
        vae_use_slicing=vae_use_slicing,
        vae_use_tiling=vae_use_tiling,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        parallel_config=parallel_config,
    )
    print("Pipeline loaded")

    # Time profiling for generation
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Cache backend: {args.cache_backend if args.cache_backend else 'None (no acceleration)'}")
    if isinstance(input_image, list):
        print(f"  Number of input images: {len(input_image)}")
        for idx, img in enumerate(input_image):
            print(f"    Image {idx + 1} size: {img.size}")
    else:
        print(f"  Input image size: {input_image.size}")
    print(f"  Parallel configuration: ulysses_degree={args.ulysses_degree}, ring_degree={args.ring_degree}")
    print(f"{'=' * 60}\n")

    try:
        generation_start = time.perf_counter()
        # Generate edited image
        generate_kwargs = {
            "prompt": args.prompt,
            "pil_image": input_image,
            "negative_prompt": args.negative_prompt,
            "generator": generator,
            "true_cfg_scale": args.cfg_scale,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "num_outputs_per_prompt": args.num_outputs_per_prompt,
            "layers": args.layers,
            "resolution": args.resolution,
        }

        outputs = omni.generate(**generate_kwargs)
        generation_end = time.perf_counter()
        generation_time = generation_end - generation_start

        # Print profiling results
        print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

        # omni.generate() returns Generator[OmniRequestOutput, None, None], convert to list
        outputs = list(outputs)
        if not outputs:
            raise ValueError("No output generated from omni.generate()")
        logger.info("Outputs: %s", outputs)

        # Extract images from OmniRequestOutput
        # omni.generate() returns Generator[OmniRequestOutput, None, None], extract images from request_output[0].images
        first_output = outputs[0]
        if not hasattr(first_output, "request_output") or not first_output.request_output:
            raise ValueError("No request_output found in OmniRequestOutput")

        req_out = first_output.request_output[0]
        if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
            raise ValueError("Invalid request_output structure or missing 'images' key")

        images = req_out.images
        if not images:
            raise ValueError("No images found in request_output")

        # Save output image(s)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix or ".png"
        stem = output_path.stem or "output_image_edit"

        # Handle layered output (each image may be a list of layers)
        if args.num_outputs_per_prompt <= 1:
            img = images[0]
            # Check if this is a layered output (list of images)
            if isinstance(img, list):
                for sub_idx, sub_img in enumerate(img):
                    save_path = output_path.parent / f"{stem}_{sub_idx}{suffix}"
                    sub_img.save(save_path)
                    print(f"Saved edited image to {os.path.abspath(save_path)}")
            else:
                img.save(output_path)
                print(f"Saved edited image to {os.path.abspath(output_path)}")
        else:
            for idx, img in enumerate(images):
                # Check if this is a layered output (list of images)
                if isinstance(img, list):
                    for sub_idx, sub_img in enumerate(img):
                        save_path = output_path.parent / f"{stem}_{idx}_{sub_idx}{suffix}"
                        sub_img.save(save_path)
                        print(f"Saved edited image to {os.path.abspath(save_path)}")
                else:
                    save_path = output_path.parent / f"{stem}_{idx}{suffix}"
                    img.save(save_path)
                    print(f"Saved edited image to {os.path.abspath(save_path)}")
    finally:
        omni.close()


if __name__ == "__main__":
    main()
