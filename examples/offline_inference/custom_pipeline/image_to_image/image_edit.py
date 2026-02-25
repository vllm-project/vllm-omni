# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
======================================================
 Qwen Image Editor (via vLLM-Omni)
======================================================

Example CLI for Qwen-Image-Edit and compatible models.

This script edits, combines, or layers images according to a text prompt
using the vLLM Omni diffusion backend.

Examples
---------
Single image edit:
    python image_edit.py \
        --image input.png \
        --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles saying 'Be Kind'" \
        --output output.png

Multiple image composition:
    python image_edit.py \
        --image input1.png input2.png \
        --prompt "Combine these into a single magical composition" \
        --output combined.png

Accelerated with cache_dit:
    python image_edit.py \
        --image input.png \
        --prompt "Edit description" \
        --cache_backend cache_dit \
        --cache_dit_enable_taylorseer

Layered RGBA output:
    python image_edit.py \
        --model "Qwen/Qwen-Image-Layered" \
        --image input.png \
        --prompt "" \
        --output layered \
        --layers 4 \
        --color-format RGBA
"""

import argparse
import asyncio
import os
import time
from pathlib import Path

import torch
from PIL import Image

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


# ===========================
# Argument Parser
# ===========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit or generate images using Qwen-Image-Edit.")
    parser.add_argument("--model", default="Qwen/Qwen-Image-Edit", help="Model name or local path.")
    parser.add_argument("--image", type=str, nargs="+", required=True, help="Input image file(s).")
    parser.add_argument("--prompt", type=str, required=True, help="Edit description prompt.")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--num_outputs_per_prompt", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cache_backend", type=str, default=None, choices=["cache_dit", "tea_cache"])
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--color-format", type=str, default="RGB")

    # Acceleration + Optimization Options
    parser.add_argument("--cache_dit_fn_compute_blocks", type=int, default=1)
    parser.add_argument("--cache_dit_bn_compute_blocks", type=int, default=0)
    parser.add_argument("--cache_dit_max_warmup_steps", type=int, default=4)
    parser.add_argument("--cache_dit_residual_diff_threshold", type=float, default=0.24)
    parser.add_argument("--cache_dit_max_continuous_cached_steps", type=int, default=3)
    parser.add_argument("--cache_dit_enable_taylorseer", action="store_true", default=False)
    parser.add_argument("--cache_dit_taylorseer_order", type=int, default=1)
    parser.add_argument(
        "--cache_dit_scm_steps_mask_policy", type=str, default=None, choices=[None, "slow", "medium", "fast", "ultra"]
    )
    parser.add_argument("--cache_dit_scm_steps_policy", type=str, default="dynamic", choices=["dynamic", "static"])
    parser.add_argument("--tea_cache_rel_l1_thresh", type=float, default=0.2)
    parser.add_argument("--cfg_parallel_size", type=int, default=1, choices=[1, 2])
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--vae_use_slicing", action="store_true")
    parser.add_argument("--vae_use_tiling", action="store_true")
    parser.add_argument("--enable-cpu-offload", action="store_true")

    return parser.parse_args()


# ===========================
# Core Logic
# ===========================
async def main():
    args = parse_args()

    # ---- Load input images ----
    input_images: list[Image.Image] = []
    for image_path in args.image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
        input_images.append(Image.open(image_path).convert(args.color_format))

    # Single or multi-image input
    input_image: Image.Image | list[Image.Image] = input_images[0] if len(input_images) == 1 else input_images

    # ---- Torch setup ----
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # ---- Cache Config ----
    if args.cache_backend == "cache_dit":
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
        cache_config = {"rel_l1_thresh": args.tea_cache_rel_l1_thresh}
    else:
        cache_config = None

    # ---- Initialize Omni ----
    omni = Omni(
        model=args.model,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
        diffusion_load_format="dummy",
        custom_pipeline_args={"pipeline_class": "custom_pipeline.CustomPipeline"},
    )

    print(">>> Pipeline loaded successfully")

    # ---- Profiling + Info ----
    profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))
    print(f"\n{'=' * 60}")
    print("Generation Configuration")
    print(f"Model: {args.model}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Cache: {args.cache_backend or 'None'}")
    print(f"Input(s): {len(input_images)} image(s)")
    for idx, img in enumerate(input_images):
        print(f"  • Image {idx + 1}: size={img.size}")
    print(
        f"Parallel config: Ulysses={args.ulysses_degree}, Ring={args.ring_degree}, "
        f"CFG Parallel={args.cfg_parallel_size}, Tensor Parallel={args.tensor_parallel_size}"
    )
    print(f"{'=' * 60}\n")

    if profiler_enabled:
        omni.start_profile()

    # ---- Generation ----
    t0 = time.perf_counter()
    outputs = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "multi_modal_data": {"image": input_image},
        },
        OmniDiffusionSamplingParams(
            generator=generator,
            true_cfg_scale=args.cfg_scale,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_outputs_per_prompt=args.num_outputs_per_prompt,
            layers=args.layers,
            resolution=args.resolution,
        ),
    )
    t1 = time.perf_counter()
    print(f"Generation completed in {t1 - t0:.2f}s")

    if profiler_enabled:
        omni.stop_profile()

    # ---- Output Validation ----
    if not outputs:
        raise ValueError("No output produced from omni.generate()")

    first_out = outputs[0].request_output[0]
    req_out: OmniRequestOutput = first_out

    # Verify trajectory data (from custom pipeline)
    print(f"\n{'=' * 60}")
    print("TRAJECTORY VERIFICATION:")
    assert hasattr(req_out, "latents") and req_out.latents is not None
    print(f"  ✓ trajectory_latents shape: {req_out.latents.shape}")
    print(f"  ✓ trajectory_latents dtype: {req_out.latents.dtype}")
    print(f"  ✓ trajectory_latents mean: {req_out.latents.mean().item():.6f}")
    print(f"  ✓ trajectory_latents std: {req_out.latents.std().item():.6f}")
    assert req_out.latents.shape[0] == args.num_inference_steps, (
        f"Expected {args.num_inference_steps} latent snapshots, got {req_out.latents.shape[0]}"
    )
    print(f"{'=' * 60}\n")

    # ---- Save images ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem or "output"

    images = req_out.images
    if args.num_outputs_per_prompt <= 1:
        img = images[0]
        if isinstance(img, list):  # Layered
            for i, sub_img in enumerate(img):
                save_path = output_path.parent / f"{stem}_{i}{suffix}"
                sub_img.save(save_path)
                print(f"Saved layer {i}: {save_path.resolve()}")
        else:
            img.save(output_path)
            print(f"Saved: {output_path.resolve()}")
    else:
        for idx, img in enumerate(images):
            if isinstance(img, list):
                for sub_idx, sub_img in enumerate(img):
                    save_path = output_path.parent / f"{stem}_{idx}_{sub_idx}{suffix}"
                    sub_img.save(save_path)
                    print(f"Saved composite: {save_path.resolve()}")
            else:
                save_path = output_path.parent / f"{stem}_{idx}{suffix}"
                img.save(save_path)
                print(f"Saved: {save_path.resolve()}")


# ===========================
# Entrypoint
# ===========================
if __name__ == "__main__":
    asyncio.run(main())
