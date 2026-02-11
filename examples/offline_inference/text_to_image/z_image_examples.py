#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Z-Image Base vs Turbo Comparison Examples

This script demonstrates the differences between Z-Image Base and Z-Image Turbo models.

Key Differences:
- Z-Image Base: Foundation model with full CFG support, fine-tunable, 28-50 steps
- Z-Image Turbo: Distilled model optimized for speed, 8 steps, guidance_scale must be 0.0
"""

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def z_image_base_example():
    """
    Z-Image Base - High Quality Generation

    Features:
    - Full CFG support (guidance_scale 3.0-5.0)
    - Negative prompts work
    - Fine-tunable
    - 28-50 inference steps (default 50)
    - Scheduler shift: 6.0
    """
    print("\n=== Z-Image Base (High Quality) ===")

    omni_base = Omni(model="Tongyi-MAI/Z-Image")

    outputs_base = omni_base.generate(
        {
            "prompt": "a majestic mountain landscape at sunset, detailed, photorealistic",
            "negative_prompt": "blurry, low quality, distorted, oversaturated",
        },
        OmniDiffusionSamplingParams(
            height=1280,
            width=720,
            num_inference_steps=50,
            guidance_scale=4.0,
            seed=42,
        ),
    )

    images = outputs_base[0].request_output[0].images
    images[0].save("z_image_base_output.png")
    print("Saved to: z_image_base_output.png")
    print(f"Generated {len(images)} image(s) with 50 steps and CFG=4.0")


def z_image_turbo_example():
    """
    Z-Image Turbo - Fast Inference

    Features:
    - Optimized for speed
    - guidance_scale MUST be 0.0 (no CFG)
    - Negative prompts not supported
    - 8 inference steps
    - Scheduler shift: 3.0
    """
    print("\n=== Z-Image Turbo (Fast) ===")

    omni_turbo = Omni(model="Tongyi-MAI/Z-Image-Turbo")

    outputs_turbo = omni_turbo.generate(
        "a majestic mountain landscape at sunset, detailed, photorealistic",
        OmniDiffusionSamplingParams(
            height=1024,
            width=1024,
            num_inference_steps=8,
            guidance_scale=0.0,  # MUST be 0.0 for Turbo!
            seed=42,
        ),
    )

    images = outputs_turbo[0].request_output[0].images
    images[0].save("z_image_turbo_output.png")
    print("Saved to: z_image_turbo_output.png")
    print(f"Generated {len(images)} image(s) with 8 steps (no CFG)")


def batch_inference_example():
    """
    Batch inference with Z-Image Base

    Note: Batch processing depends on max_batch_size in stage configs.
    By default, diffusion models process one prompt at a time.
    """
    print("\n=== Batch Inference Example ===")

    omni = Omni(model="Tongyi-MAI/Z-Image")

    prompts = [
        {"prompt": "a cup of coffee on a wooden table", "negative_prompt": "blurry, low quality"},
        {"prompt": "a cat sleeping on a cozy blanket", "negative_prompt": "blurry, low quality"},
        {"prompt": "a futuristic city skyline at night", "negative_prompt": "blurry, low quality"},
    ]

    # Note: These will be processed sequentially unless max_batch_size > 1
    outputs = omni.generate(
        prompts,
        OmniDiffusionSamplingParams(
            height=1024,
            width=1024,
            num_inference_steps=40,
            guidance_scale=4.0,
            seed=42,
        ),
    )

    for i, output in enumerate(outputs):
        image = output.request_output[0].images[0]
        image.save(f"batch_output_{i}.png")
        print(f"Saved to: batch_output_{i}.png")


def recommended_settings():
    """
    Print recommended settings for both models
    """
    print("\n=== Recommended Settings ===\n")

    print("Z-Image Base (Tongyi-MAI/Z-Image):")
    print("  - num_inference_steps: 28-50 (default: 50)")
    print("  - guidance_scale: 3.0-5.0 (default: 4.0)")
    print("  - negative_prompt: Supported and recommended")
    print("  - resolution: 1280x720 or 720x1280")
    print("  - cfg_normalization: False (default)")
    print("  - Use when: Quality is priority, fine-tuning needed")

    print("\nZ-Image Turbo (Tongyi-MAI/Z-Image-Turbo):")
    print("  - num_inference_steps: 8")
    print("  - guidance_scale: 0.0 (REQUIRED)")
    print("  - negative_prompt: Not supported")
    print("  - resolution: 1024x1024")
    print("  - Use when: Speed is priority, quick iterations")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Z-Image Base vs Turbo comparison examples")
    parser.add_argument(
        "--example",
        choices=["base", "turbo", "batch", "all"],
        default="all",
        help="Which example to run (default: all)",
    )

    args = parser.parse_args()

    recommended_settings()

    if args.example in ("base", "all"):
        z_image_base_example()

    if args.example in ("turbo", "all"):
        z_image_turbo_example()

    if args.example in ("batch", "all"):
        batch_inference_example()

    print("\nDone!")
