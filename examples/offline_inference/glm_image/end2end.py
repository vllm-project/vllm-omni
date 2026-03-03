# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end offline inference example for GLM-Image with multistage architecture.

This script tests the multistage pipeline where:
- Stage 0 (AR): vLLM-optimized GlmImageForConditionalGeneration generates prior_token_ids
- Stage 1 (Diffusion): GlmImagePipeline performs DiT denoising + VAE decode

Usage (text-to-image):
    python end2end.py \
        --model-path /path/to/glm-image \
        --config-path /path/to/glm_image.yaml \
        --prompt "A beautiful sunset over the ocean" \
        --output output_t2i.png

Usage (image-to-image / image edit):
    python end2end.py \
        --model-path /path/to/glm-image \
        --config-path /path/to/glm_image.yaml \
        --prompt "Make it look like winter" \
        --image input.png \
        --output output_i2i.png

Usage (with custom parameters):
    python end2end.py \
        --model-path /path/to/glm-image \
        --config-path /path/to/glm_image.yaml \
        --prompt "A cat sitting on a window sill" \
        --height 1024 \
        --width 1024 \
        --num-inference-steps 50 \
        --guidance-scale 1.5 \
        --seed 42

For more options, run:
    python end2end.py --help
"""

import argparse
import os
import time
from pathlib import Path

from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Default stage config path (relative to vllm_omni package)
DEFAULT_CONFIG_PATH = "vllm_omni/model_executor/stage_configs/glm_image.yaml"

SEED = 42

# GLM-Image special tokens
GLM_IMAGE_EOS_TOKEN_ID = 16385  # eos_token_id from generation_config.json
GLM_IMAGE_VISION_VOCAB_SIZE = 16512  # top_k should be vision_vocab_size


def compute_max_tokens(height: int, width: int, factor: int = 32) -> int:
    """
    Compute max_new_tokens for GLM-Image AR generation.

    GLM-Image generates tokens in this order for text-to-image:
    1. Small preview image (half resolution in each dimension)
    2. Large target image (full resolution)
    3. EOS token

    Args:
        height: Target image height in pixels
        width: Target image width in pixels
        factor: Downsampling factor (32 for GLM-Image AR output)

    Returns:
        Total number of tokens to generate (small + large + EOS)
    """
    # Large image tokens (target resolution)
    token_h = height // factor
    token_w = width // factor
    large_tokens = token_h * token_w

    # Small preview tokens (half resolution in each dimension)
    small_h = token_h // 2
    small_w = token_w // 2
    small_tokens = small_h * small_w

    # Total: small + large + EOS
    return small_tokens + large_tokens + 1


def load_image(image_path: str) -> Image.Image:
    """Load an image from file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def save_image(image: Image.Image, output_path: str) -> None:
    """Save an image to file path."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    image.save(output_path)
    print(f"Image saved to: {output_path}")


def build_prompt_for_t2i(
    prompt: str,
    height: int = 1024,
    width: int = 1024,
) -> dict:
    """
    Build prompt dict for text-to-image generation.

    Args:
        prompt: Text description for image generation
        height: Target image height
        width: Target image width

    Returns:
        Dict containing prompt and generation parameters
    """
    return {
        "prompt": prompt,
        "height": height,
        "width": width,
        # Pass target dimensions to AR processor for proper grid token generation
        "mm_processor_kwargs": {
            "target_h": height,
            "target_w": width,
        },
    }


def build_prompt_for_i2i(
    prompt: str,
    image: Image.Image,
    height: int | None = None,
    width: int | None = None,
) -> dict:
    """
    Build prompt dict for image-to-image generation.

    Args:
        prompt: Text description for image editing
        image: Source image for editing
        height: Target image height (default: use source image size)
        width: Target image width (default: use source image size)

    Returns:
        Dict containing prompt, image, and generation parameters
    """
    # Use source image dimensions if not specified
    if height is None:
        height = image.height
    if width is None:
        width = image.width

    return {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image,
        },
        "height": height,
        "width": width,
        # Pass target dimensions to AR processor for proper grid token generation
        "mm_processor_kwargs": {
            "target_h": height,
            "target_w": width,
        },
    }


def main(args: argparse.Namespace) -> None:
    """Main entry point for GLM-Image end-to-end inference."""
    print("=" * 60)
    print("GLM-Image Multistage End-to-End Inference")
    print("=" * 60)

    # Validate arguments
    if not args.model_path:
        raise ValueError("--model-path is required")

    if not args.prompt:
        raise ValueError("--prompt is required")

    # Determine config path
    config_path = args.config_path
    if config_path is None:
        # Try to find default config
        if os.path.exists(DEFAULT_CONFIG_PATH):
            config_path = DEFAULT_CONFIG_PATH
        else:
            # Try relative to script location
            script_dir = Path(__file__).parent.parent.parent.parent
            config_path = script_dir / "vllm_omni/model_executor/stage_configs/glm_image.yaml"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Stage config not found. Please specify --config-path. Tried: {DEFAULT_CONFIG_PATH}"
                )
            config_path = str(config_path)

    print(f"Model path: {args.model_path}")
    print(f"Config path: {config_path}")
    print(f"Prompt: {args.prompt}")

    # Load source image for image-to-image mode
    source_image = None
    if args.image:
        print(f"Source image: {args.image}")
        source_image = load_image(args.image)
        print(f"  Image size: {source_image.width}x{source_image.height}")

    # Build prompt based on mode
    if source_image is not None:
        # Image-to-image mode
        prompt_dict = build_prompt_for_i2i(
            prompt=args.prompt,
            image=source_image,
            height=args.height,
            width=args.width,
        )
        mode = "image-to-image"
    else:
        # Text-to-image mode
        prompt_dict = build_prompt_for_t2i(
            prompt=args.prompt,
            height=args.height or 1024,
            width=args.width or 1024,
        )
        mode = "text-to-image"

    print(f"Mode: {mode}")
    print(f"Target size: {prompt_dict.get('height', 1024)}x{prompt_dict.get('width', 1024)}")

    # Add generation parameters to prompt
    prompt_dict["seed"] = args.seed
    prompt_dict["num_inference_steps"] = args.num_inference_steps
    prompt_dict["guidance_scale"] = args.guidance_scale

    if args.negative_prompt:
        prompt_dict["negative_prompt"] = args.negative_prompt

    # Initialize Omni with multistage config
    print("\nInitializing Omni with multistage pipeline...")
    start_time = time.time()

    omni = Omni(
        model=args.model_path,
        stage_configs_path=config_path,
        log_stats=args.enable_stats,
        stage_init_timeout=args.stage_init_timeout,
    )

    init_time = time.time() - start_time
    print(f"Initialization completed in {init_time:.2f}s")

    # Prepare prompts (support batch generation)
    prompts = [prompt_dict for _ in range(args.num_prompts)]

    # No explicit sampling_params for diffusion - parameters are in prompt_dict
    # For multistage, the AR stage may need sampling params
    from vllm import SamplingParams

    # Compute max_tokens dynamically based on target image size
    target_height = prompt_dict.get("height", 1024)
    target_width = prompt_dict.get("width", 1024)
    calculated_max_tokens = compute_max_tokens(target_height, target_width)

    # Use calculated value unless user explicitly specified a different value
    # Default args.max_tokens is 16384 (very large), so prefer calculated value
    effective_max_tokens = calculated_max_tokens if args.max_tokens == 16384 else args.max_tokens

    if args.verbose:
        print(f"AR max_tokens: {effective_max_tokens} (calculated: {calculated_max_tokens}, arg: {args.max_tokens})")

    # IMPORTANT: GLM-Image AR model requires these exact sampling parameters
    # from generation_config.json for proper image token generation.
    # - temperature=0.9, top_p=0.75, top_k=16512 (vision_vocab_size)
    # - stop_token_ids=[16385] (eos_token_id) is CRITICAL to stop generation
    ar_sampling_params = SamplingParams(
        temperature=0.9,  # From generation_config.json
        top_p=0.75,  # From generation_config.json
        top_k=GLM_IMAGE_VISION_VOCAB_SIZE,  # 16512, vision vocabulary size
        max_tokens=effective_max_tokens,
        stop_token_ids=[GLM_IMAGE_EOS_TOKEN_ID],  # 16385, CRITICAL for stopping
        seed=args.seed,
        detokenize=False,
    )

    # For diffusion stage, sampling_params contains diffusion-specific parameters
    # These are passed as kwargs to the diffusion engine
    diffusion_sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=prompt_dict.get("height", 1024),
        width=prompt_dict.get("width", 1024),
        seed=args.seed,
    )

    # For multistage, we need sampling_params for each stage
    # Stage 0 (AR): SamplingParams for vLLM
    # Stage 1 (Diffusion): dict with diffusion kwargs
    sampling_params_list = [ar_sampling_params, diffusion_sampling_params]

    # Run generation
    print(f"\nGenerating {args.num_prompts} image(s)...")
    gen_start_time = time.time()

    output_dir = os.path.dirname(args.output) if args.output else "outputs"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_count = 0
    for stage_outputs in omni.generate(prompts, sampling_params_list, py_generator=True):
        if stage_outputs.final_output_type == "image":
            for output in stage_outputs.request_output:
                request_id = output.request_id

                # Get generated images
                images = output.images if hasattr(output, "images") else []
                if not images and hasattr(output, "multimodal_output"):
                    images = output.multimodal_output.get("images", [])

                # Save each generated image
                for idx, img in enumerate(images):
                    if args.num_prompts == 1 and len(images) == 1:
                        output_path = args.output
                    else:
                        base, ext = os.path.splitext(args.output)
                        output_path = f"{base}_{request_id}_{idx}{ext}"

                    if isinstance(img, Image.Image):
                        save_image(img, output_path)
                    else:
                        print(f"Warning: Unexpected image type for request {request_id}: {type(img)}")

                    output_count += 1

        elif stage_outputs.final_output_type == "text":
            # AR stage output (intermediate, for debugging)
            if args.verbose:
                for output in stage_outputs.request_output:
                    print(f"AR output for request {output.request_id}:")
                    print(f"  Token count: {len(output.outputs[0].token_ids)}")

    gen_time = time.time() - gen_start_time
    print(f"\nGeneration completed in {gen_time:.2f}s")
    print(f"Generated {output_count} image(s)")

    # Cleanup
    omni.close()
    print("\nDone!")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GLM-Image Multistage End-to-End Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to GLM-Image model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )

    # Optional arguments
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to stage config YAML file (default: auto-detect)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to source image for image-to-image mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_glm_image.png",
        help="Output image path (default: output_glm_image.png)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt for classifier-free guidance",
    )

    # Generation parameters
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output image height (default: 1024 for t2i, source size for i2i)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output image width (default: 1024 for t2i, source size for i2i)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of diffusion denoising steps (default: 50)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale (default: 1.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens for AR generation (default: 16384)",
    )

    # Batch processing
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)",
    )

    # Runtime options
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        default=False,
        help="Enable statistics logging",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=600,
        help="Timeout for stage initialization in seconds (default: 600)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose output",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
