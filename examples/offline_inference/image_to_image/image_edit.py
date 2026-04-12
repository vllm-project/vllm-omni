# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script for image editing with OmniGen2.

    python image_edit.py \
        --image input.png \
        --model "OmniGen2/OmniGen2" \
        --prompt "Change the background to classroom." \
        --negative-prompt "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar" \
        --num-inference-steps 50 \
        --seed 0 \
        --guidance-scale 5.0 \
        --guidance-scale-2 2.0 \
        --output outputs/image_edit.png \
        --num-outputs-per-prompt 2

    Note: For OmniGen2, `guidance_scale` works as `text_guidance_scale`,
    and `guidance_scale_2` works as `image_guidance_scale`.

Example script for image editing with FLUX.2-klein.

Usage:
    python image_edit.py \
        --model "black-forest-labs/FLUX.2-klein-4B" \
        --image input.png \
        --prompt "Change the background to a beach" \
        --output output_image_edit.png \
        --num-inference-steps 50 \
        --cfg-scale 4.0 \
        --guidance-scale 1.0

    FLUX.2-klein is also available as a 9B variant:
        --model "black-forest-labs/FLUX.2-klein-9B"

Example script for image editing with Qwen-Image-Edit.

Usage (single image):
    python image_edit.py \
        --image input.png \
        --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
        --output output_image_edit.png \
        --num-inference-steps 50 \
        --cfg-scale 4.0 \
        --guidance-scale 1.0



Usage (multiple images):
    python image_edit.py \
        --image input1.png input2.png input3.png \
        --prompt "Combine these images into a single scene" \
        --output output_image_edit.png \
        --num-inference-steps 50 \
        --cfg-scale 4.0 \
        --guidance-scale 1.0

Usage (with cache-dit acceleration):
    python image_edit.py \
        --image input.png \
        --prompt "Edit description" \
        --cache-backend cache_dit \
        --cache-dit-max-continuous-cached-steps 3 \
        --cache-dit-residual-diff-threshold 0.24 \
        --cache-dit-enable-taylorseer

Usage (with tea_cache acceleration):
    python image_edit.py \
        --image input.png \
        --prompt "Edit description" \
        --cache-backend tea_cache \
        --tea-cache-rel-l1-thresh 0.25

Usage (layered):
    python image_edit.py \
        --model "Qwen/Qwen-Image-Layered" \
        --image input.png \
        --prompt "" \
        --output "layered" \
        --num-inference-steps 50 \
        --cfg-scale 4.0 \
        --layers 4 \
        --color-format "RGBA"

Usage (with CFG Parallel):
    python image_edit.py \
        --image input.png \
        --prompt "Edit description" \
        --cfg-parallel-size 2 \
        --num-inference-steps 50 \
        --cfg-scale 4.0

Usage (disable torch.compile):
    python image_edit.py \
        --image input.png \
        --prompt "Edit description" \
        --enforce-eager \
        --num-inference-steps 50 \
        --cfg-scale 4.0

For more options, run:
    python image_edit.py --help
"""

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform


def is_nextstep_model(model_name: str) -> bool:
    """Check if the model is a NextStep model by reading its config."""
    from vllm.transformers_utils.config import get_hf_file_to_dict

    try:
        cfg = get_hf_file_to_dict("config.json", model_name)
        if cfg and cfg.get("model_type") == "nextstep":
            return True
    except Exception:
        pass
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit an image with supported diffusion models.")
    # --- Shared args (same order as text_to_image.py) ---
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-Edit",
        help=(
            "Diffusion model name or local path. "
            "Supported models: Qwen/Qwen-Image-Edit (default), "
            "Qwen/Qwen-Image-Edit-2509, Qwen/Qwen-Image-Edit-2511 (multi-image), "
            "Qwen/Qwen-Image-Layered (layered output), "
            "black-forest-labs/FLUX.2-klein-4B, black-forest-labs/FLUX.2-klein-9B, "
            "OmniGen2/OmniGen2, meituan-longcat/LongCat-Image-Edit, "
            "zai-org/GLM-Image, and NextStep-1.1 models."
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
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt for classifier-free conditional guidance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic results.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="True classifier-free guidance scale specific to Qwen-Image.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of generated image. If not set, the pipeline auto-sizes from the input image.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width of generated image. If not set, the pipeline auto-sizes from the input image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_image_edit.png",
        help="Path to save the edited image (PNG). Or prefix for Qwen-Image-Layered model save images(PNG).",
    )
    parser.add_argument(
        "--num-outputs-per-prompt",
        type=int,
        default=1,
        help="Number of images to generate for the given prompt.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps for the diffusion sampler.",
    )
    parser.add_argument(
        "--cache-backend",
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
        "--enable-cache-dit-summary",
        action="store_true",
        help="Enable cache-dit summary logging after diffusion forward passes.",
    )
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ulysses sequence parallelism.",
    )
    parser.add_argument(
        "--ring-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ring sequence parallelism.",
    )
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of GPUs used for classifier free guidance parallel size.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offloading for diffusion models.",
    )
    parser.add_argument(
        "--enable-layerwise-offload",
        action="store_true",
        help="Enable layerwise (blockwise) offloading on DiT modules.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["fp8", "gguf"],
        help=(
            "Quantization method for the transformer. "
            "Options: 'fp8' (FP8 W8A8), 'gguf' (GGUF quantized weights). "
            "Default: None (no quantization, uses BF16)."
        ),
    )
    parser.add_argument(
        "--gguf-model",
        type=str,
        default=None,
        help="GGUF file path or HF reference for transformer weights. Required when --quantization gguf is set.",
    )
    parser.add_argument(
        "--ignored-layers",
        type=str,
        default=None,
        help="Comma-separated list of layer name patterns to skip quantization. "
        "Only used when --quantization is set. "
        "Available layers: to_qkv, to_out, add_kv_proj, to_add_out, img_mlp, txt_mlp, proj_out. "
        "Example: --ignored-layers 'add_kv_proj,to_add_out'",
    )
    parser.add_argument(
        "--vae-use-slicing",
        action="store_true",
        help="Enable VAE slicing for memory optimization.",
    )
    parser.add_argument(
        "--vae-use-tiling",
        action="store_true",
        help="Enable VAE tiling for memory optimization.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for tensor parallelism (TP) inside the DiT.",
    )
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="Enable expert parallelism for MoE layers.",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA adapter folder (PEFT format). Loaded at initialization and used for generation.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Scale factor for LoRA weights (default: 1.0).",
    )
    parser.add_argument(
        "--vae-patch-parallel-size",
        type=int,
        default=1,
        help="Number of ranks used for VAE patch/tile parallelism (decode/encode).",
    )
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to display stage durations.",
    )
    # NextStep-1.1 specific arguments
    parser.add_argument(
        "--guidance-scale-2",
        type=float,
        default=None,
        help="Secondary guidance scale (e.g. image-level CFG for NextStep-1.1 or OmniGen2).",
    )
    parser.add_argument(
        "--timesteps-shift",
        type=float,
        default=1.0,
        help="[NextStep-1.1 only] Timesteps shift parameter for sampling.",
    )
    parser.add_argument(
        "--cfg-schedule",
        type=str,
        default="constant",
        choices=["constant", "linear"],
        help="[NextStep-1.1 only] CFG schedule type.",
    )
    parser.add_argument(
        "--use-norm",
        action="store_true",
        help="[NextStep-1.1 only] Apply layer normalization to sampled tokens.",
    )
    # --- Image-edit-specific args ---
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Enable logging of statistics.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=4,
        help="[Qwen-Image-Layered] Number of layers to decompose the input image into.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        help="[Qwen-Image-Layered] Bucket in (640, 1024) to determine the condition and output resolution.",
    )
    parser.add_argument(
        "--color-format",
        type=str,
        default="RGB",
        help="[Qwen-Image-Layered] Color format. Set to RGBA for layered output.",
    )
    # Cache-DiT specific parameters
    parser.add_argument(
        "--cache-dit-fn-compute-blocks",
        type=int,
        default=1,
        help="[cache-dit] Number of forward compute blocks. Optimized for single-transformer models.",
    )
    parser.add_argument(
        "--cache-dit-bn-compute-blocks",
        type=int,
        default=0,
        help="[cache-dit] Number of backward compute blocks.",
    )
    parser.add_argument(
        "--cache-dit-max-warmup-steps",
        type=int,
        default=4,
        help="[cache-dit] Maximum warmup steps (works for few-step models).",
    )
    parser.add_argument(
        "--cache-dit-residual-diff-threshold",
        type=float,
        default=0.24,
        help="[cache-dit] Residual diff threshold. Higher values enable more aggressive caching.",
    )
    parser.add_argument(
        "--cache-dit-max-continuous-cached-steps",
        type=int,
        default=3,
        help="[cache-dit] Maximum continuous cached steps to prevent precision degradation.",
    )
    parser.add_argument(
        "--cache-dit-enable-taylorseer",
        action="store_true",
        default=False,
        help="[cache-dit] Enable TaylorSeer acceleration (not suitable for few-step models).",
    )
    parser.add_argument(
        "--cache-dit-taylorseer-order",
        type=int,
        default=1,
        help="[cache-dit] TaylorSeer polynomial order.",
    )
    parser.add_argument(
        "--cache-dit-scm-steps-mask-policy",
        type=str,
        default=None,
        choices=[None, "slow", "medium", "fast", "ultra"],
        help="[cache-dit] SCM mask policy: None (disabled), slow, medium, fast, ultra.",
    )
    parser.add_argument(
        "--cache-dit-scm-steps-policy",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="[cache-dit] SCM steps policy: dynamic or static.",
    )
    # TeaCache specific parameters
    parser.add_argument(
        "--tea-cache-rel-l1-thresh",
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

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
    )

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

    # Prepare LoRA kwargs for Omni initialization
    lora_args: dict[str, Any] = {}
    if args.lora_path:
        lora_args["lora_path"] = args.lora_path
        print(f"Using LoRA from: {args.lora_path}")

    # Build quantization kwargs: use quantization_config dict when
    # ignored_layers is specified so the list flows through OmniDiffusionConfig
    quant_kwargs: dict[str, Any] = {}
    ignored_layers = [s.strip() for s in args.ignored_layers.split(",") if s.strip()] if args.ignored_layers else None
    if args.quantization == "gguf":
        if not args.gguf_model:
            raise ValueError("--gguf-model is required when --quantization gguf is set.")
        quant_kwargs["quantization_config"] = {
            "method": "gguf",
            "gguf_model": args.gguf_model,
        }
    elif args.quantization and ignored_layers:
        quant_kwargs["quantization_config"] = {
            "method": args.quantization,
            "ignored_layers": ignored_layers,
        }
    elif args.quantization:
        quant_kwargs["quantization"] = args.quantization

    # Initialize Omni with appropriate pipeline

    omni = Omni(
        model=args.model,
        enable_layerwise_offload=args.enable_layerwise_offload,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        enable_cache_dit_summary=args.enable_cache_dit_summary,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
        log_stats: args.log_stats,
        **lora_args,
        **quant_kwargs,
    )
    print("Pipeline loaded")

    # Check if profiling is requested via environment variable
    profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))

    # Time profiling for generation
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Cache backend: {args.cache_backend if args.cache_backend else 'None (no acceleration)'}")
    print(f"  Quantization: {args.quantization if args.quantization else 'None (BF16)'}")
    if ignored_layers:
        print(f"  Ignored layers: {ignored_layers}")
    if isinstance(input_image, list):
        print(f"  Number of input images: {len(input_image)}")
        for idx, img in enumerate(input_image):
            print(f"    Image {idx + 1} size: {img.size}")
    else:
        print(f"  Input image size: {input_image.size}")
    print(
        f"  Parallel configuration: ulysses_degree={args.ulysses_degree}, ring_degree={args.ring_degree}, cfg_parallel_size={args.cfg_parallel_size}, tensor_parallel_size={args.tensor_parallel_size}, vae_patch_parallel_size={args.vae_patch_parallel_size}, enable_expert_parallel: {args.enable_expert_parallel}"
    )
    print(f"  CPU offload: {args.enable_cpu_offload}")
    if args.lora_path:
        print(f"  LoRA: scale={args.lora_scale}")
    print(f"{'=' * 60}\n")

    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

    # Build LoRA request when --lora-path is set
    lora_request = None
    if args.lora_path:
        lora_request_id = stable_lora_int_id(args.lora_path)
        lora_request = LoRARequest(
            lora_name=Path(args.lora_path).stem,
            lora_int_id=lora_request_id,
            lora_path=args.lora_path,
        )

    generation_start = time.perf_counter()

    extra_args = {
        "timesteps_shift": args.timesteps_shift,
        "cfg_schedule": args.cfg_schedule,
        "use_norm": args.use_norm,
    }

    if lora_request:
        extra_args["lora_request"] = lora_request
        extra_args["lora_scale"] = args.lora_scale

    # Generate edited image
    outputs = omni.generate(
        {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "multi_modal_data": {"image": input_image},
        },
        OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=generator,
            true_cfg_scale=args.cfg_scale,
            guidance_scale=args.guidance_scale,
            guidance_scale_2=args.guidance_scale_2,
            num_inference_steps=args.num_inference_steps,
            num_outputs_per_prompt=args.num_outputs_per_prompt,
            layers=args.layers,
            resolution=args.resolution,
            extra_args=extra_args,
        ),
    )
    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    # Print profiling results
    print(f"Total generation time: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)")

    if profiler_enabled:
        print("\n[Profiler] Stopping profiler and collecting results...")
        profile_results = omni.stop_profile()
        if profile_results and isinstance(profile_results, dict):
            traces = profile_results.get("traces", [])
            print("\n" + "=" * 60)
            print("PROFILING RESULTS:")
            for rank, trace in enumerate(traces):
                print(f"\nRank {rank}:")
                if trace:
                    print(f"  • Trace: {trace}")
            if not traces:
                print("  No traces collected.")
            print("=" * 60)
        else:
            print("[Profiler] No valid profiling data returned.")

    if not outputs:
        raise ValueError("No output generated from omni.generate()")

    # Extract images from OmniRequestOutput
    # omni.generate() returns list[OmniRequestOutput], extract images from request_output.images
    first_output = outputs[0]
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output
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


if __name__ == "__main__":
    main()
