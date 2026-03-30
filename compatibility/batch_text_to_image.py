# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, logger
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch generate images from a prompt file.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image",
        help="Diffusion model name or local path. Supported models: "
        "Qwen/Qwen-Image, Tongyi-MAI/Z-Image-Turbo, Qwen/Qwen-Image-2512",
    )
    parser.add_argument(
        "--prompt-file",
        required=True,
        help="Path to a text file containing prompts (one prompt per line).",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="negative prompt for classifier-free conditional guidance.",
    )
    parser.add_argument("--seed", type=int, default=142, help="Random seed for deterministic results.")
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
    parser.add_argument("--height", type=int, default=1024, help="Height of generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of generated image.")
    parser.add_argument(
        "--output",
        type=str,
        default="batch_outputs",
        help="Output directory to save generated images.",
    )
    parser.add_argument(
        "--num-images-per-prompt",
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
            "'fp8': FP8 W8A8 on Ada/Hopper, weight-only on older GPUs. "
            "'gguf': Native GGUF transformer-only weights (Q4, Q8, etc.). "
            "Default: None (no quantization, uses BF16)."
        ),
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
        "--vae-patch-parallel-size",
        type=int,
        default=1,
        help="Number of ranks used for VAE patch/tile parallelism (decode/encode).",
    )
    # ── HSDP ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--use-hsdp",
        action="store_true",
        help=(
            "Enable Hybrid Sharded Data Parallel (HSDP) for weight sharding via FSDP2. "
            "Not compatible with --tensor-parallel-size > 1 or data parallelism."
        ),
    )
    parser.add_argument(
        "--hsdp-shard-size",
        type=int,
        default=-1,
        help=(
            "Number of GPUs to shard weights across within each HSDP replica group. "
            "-1 means auto-calculate from world size (requires other parallelism). "
            "Only used when --use-hsdp is set."
        ),
    )
    parser.add_argument(
        "--hsdp-replicate-size",
        type=int,
        default=1,
        help=(
            "Number of HSDP replica groups. Each replica holds a full sharded copy. "
            "Total GPUs for HSDP = hsdp-shard-size × hsdp-replicate-size. "
            "Only used when --use-hsdp is set."
        ),
    )
    # ── Expert Parallelism ────────────────────────────────────────────────────
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help=(
            "Enable expert parallelism for MoE diffusion models (e.g. HunyuanImage-3.0). "
            "TP is still used for non-MoE layers; requires --tensor-parallel-size ≥ 2."
        ),
    )
    # ── LoRA Inference ────────────────────────────────────────────────────────
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help=(
            "Path to a LoRA adapter directory containing adapter_model.safetensors "
            "and adapter_config.json.  When set, LoRA weights are applied to every "
            "generation request in the batch."
        ),
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Scaling factor for the LoRA adapter weights.  Only used when --lora-path is set.",
    )
    return parser.parse_args()


def read_prompts_from_file(filepath: str) -> list[str]:
    """Read prompts from a text file, one prompt per line."""
    prompts = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def main():
    args = parse_args()

    # Read prompts from file
    prompt_file_path = Path(args.prompt_file)
    if not prompt_file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")

    prompts = read_prompts_from_file(args.prompt_file)
    if not prompts:
        raise ValueError(f"No prompts found in file: {args.prompt_file}")

    print(f"\n{'=' * 60}")
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    print(f"{'=' * 60}\n")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)

    # Configure cache based on backend type
    cache_config = None
    if args.cache_backend == "cache_dit":
        cache_config = {
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "max_warmup_steps": 4,
            "residual_diff_threshold": 0.24,
            "max_continuous_cached_steps": 3,
            "enable_taylorseer": False,
            "taylorseer_order": 1,
            "scm_steps_mask_policy": None,
            "scm_steps_policy": "dynamic",
        }
    elif args.cache_backend == "tea_cache":
        cache_config = {
            "rel_l1_thresh": 0.2,
        }

    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
    )

    # Check if profiling is requested via environment variable
    profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))

    # Build quantization kwargs
    quant_kwargs: dict[str, Any] = {}
    ignored_layers = [s.strip() for s in args.ignored_layers.split(",") if s.strip()] if args.ignored_layers else None
    if args.quantization and ignored_layers:
        quant_kwargs["quantization_config"] = {
            "method": args.quantization,
            "ignored_layers": ignored_layers,
        }
    elif args.quantization:
        quant_kwargs["quantization"] = args.quantization

    # Build LoRA request if --lora-path is provided
    lora_request = None
    if args.lora_path:
        from vllm_omni.lora.request import LoRARequest
        from vllm_omni.lora.utils import stable_lora_int_id

        lora_request = LoRARequest(
            lora_name="batch_lora",
            lora_int_id=stable_lora_int_id(args.lora_path),
            lora_path=args.lora_path,
        )
        logger.info("LoRA adapter loaded from: %s (scale=%.2f)", args.lora_path, args.lora_scale)

    # Initialize Omni model
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
        **quant_kwargs,
    )

    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

    # Print configuration
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Cache backend: {args.cache_backend if args.cache_backend else 'None (no acceleration)'}")
    print(f"  Quantization: {args.quantization if args.quantization else 'None (BF16)'}")
    if ignored_layers:
        print(f"  Ignored layers: {ignored_layers}")
    print(
        f"  Parallel: tp={args.tensor_parallel_size}, ulysses={args.ulysses_degree}, "
        f"ring={args.ring_degree}, cfg={args.cfg_parallel_size}, "
        f"vae_patch={args.vae_patch_parallel_size}, "
        f"expert_parallel={args.enable_expert_parallel}"
    )
    if args.use_hsdp:
        print(f"  HSDP: shard_size={args.hsdp_shard_size}, replicate_size={args.hsdp_replicate_size}")
    print(f"  CPU offload (module): {args.enable_cpu_offload}")
    print(f"  CPU offload (layerwise): {args.enable_layerwise_offload}")
    if lora_request is not None:
        print(f"  LoRA adapter: {args.lora_path} (scale={args.lora_scale})")
    print(f"  Image size: {args.width}x{args.height}")
    print(f"{'=' * 60}\n")

    # Process each prompt sequentially
    generation_times = []
    image_counter = 0

    for prompt_idx, prompt in enumerate(prompts, start=1):
        print(f"\n{'=' * 60}")
        print(f"Processing prompt {prompt_idx}/{len(prompts)}")
        print(f"Prompt: {prompt}")
        print(f"{'=' * 60}\n")

        sampling_params_kwargs: dict[str, Any] = dict(
            height=args.height,
            width=args.width,
            generator=generator,
            true_cfg_scale=args.cfg_scale,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            num_outputs_per_prompt=args.num_images_per_prompt,
        )
        if lora_request is not None:
            sampling_params_kwargs["lora_request"] = lora_request
            sampling_params_kwargs["lora_scale"] = args.lora_scale

        generation_start = time.perf_counter()
        outputs = omni.generate(
            {
                "prompt": prompt,
                "negative_prompt": args.negative_prompt,
            },
            OmniDiffusionSamplingParams(**sampling_params_kwargs),
        )
        generation_end = time.perf_counter()
        generation_time = generation_end - generation_start
        generation_times.append(generation_time)

        print(
            f"Generation time for prompt {prompt_idx}: {generation_time:.4f} seconds ({generation_time * 1000:.2f} ms)"
        )

        # Extract and save images
        if not outputs or len(outputs) == 0:
            logger.warning(f"No output generated for prompt {prompt_idx}")
            continue

        first_output = outputs[0]
        if not hasattr(first_output, "request_output") or not first_output.request_output:
            logger.warning(f"No request_output found for prompt {prompt_idx}")
            continue

        req_out = first_output.request_output
        if not hasattr(req_out, "images"):
            logger.warning(f"Invalid request_output structure for prompt {prompt_idx}")
            continue

        images = req_out.images
        if not images:
            logger.warning(f"No images found for prompt {prompt_idx}")
            continue

        # Save images with sequential numbering
        for img in images:
            save_path = output_dir / f"image_{image_counter:04d}.png"
            img.save(save_path)
            print(f"Saved image to {save_path}")
            image_counter += 1

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print("BATCH GENERATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total prompts processed: {len(prompts)}")
    print(f"Total images generated: {image_counter}")
    print("\nGeneration times:")
    for idx, gen_time in enumerate(generation_times, start=1):
        print(f"  Prompt {idx}: {gen_time:.4f} seconds ({gen_time * 1000:.2f} ms)")

    if generation_times:
        avg_time = sum(generation_times) / len(generation_times)
        total_time = sum(generation_times)
        print(f"\nAverage generation time: {avg_time:.4f} seconds ({avg_time * 1000:.2f} ms)")
        print(f"Total generation time: {total_time:.4f} seconds ({total_time * 1000:.2f} ms)")
    print(f"{'=' * 60}\n")

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


if __name__ == "__main__":
    main()
