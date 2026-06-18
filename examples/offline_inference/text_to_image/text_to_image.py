# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from vllm_omni.diffusion.data import logger
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.openai.stage_params import clone_sampling_params
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id
from vllm_omni.model_extras import (
    build_text_to_image_prompt,
    get_extra_body_params,
    get_model_class_name,
    should_init_extra_args_for_non_diffusion_stages,
)
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


def parse_profiler_config(value: str) -> dict[str, Any]:
    try:
        config = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--profiler-config must be valid JSON: {e}") from e
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError("--profiler-config must be a JSON object")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with supported diffusion models.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image",
        help="Diffusion model name or local path. Supported models: "
        "Qwen/Qwen-Image, Tongyi-MAI/Z-Image-Turbo, Qwen/Qwen-Image-2512, stepfun-ai/NextStep-1.1, "
        "black-forest-labs/FLUX.1-dev, black-forest-labs/FLUX.2-klein-9B, "
        "black-forest-labs/FLUX.2-dev, tencent/HunyuanImage-3.0-Instruct, "
        "meituan-longcat/LongCat-Image, OvisAI/Ovis-Image, "
        "stabilityai/stable-diffusion-3.5-medium, Tongyi-MAI/Z-Image-Turbo and etc.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to a YAML file containing stage configurations for Omni.",
    )
    parser.add_argument("--prompt", default="a cup of coffee on the table", help="Text prompt for image generation.")
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
        default=4.0,
        help="Classifier-free guidance scale. HunyuanImage3 recommends 4.0-5.0.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of generated image.")
    parser.add_argument(
        "--output",
        type=str,
        default="qwen_image_output.png",
        help="Path to save the generated image (PNG).",
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
        "--ulysses-mode",
        type=str,
        default="strict",
        choices=["strict", "advanced_uaa"],
        help="Ulysses sequence-parallel mode: 'strict' (divisibility required) or 'advanced_uaa' (UAA).",
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
        "--use-hsdp",
        action="store_true",
        help="Enable HSDP (Hybrid Sharded Data Parallel) for diffusion models.",
    )
    parser.add_argument(
        "--hsdp-shard-size",
        type=int,
        default=1,
        help="Number of GPUs to shard weights across for HSDP.",
    )
    parser.add_argument(
        "--hsdp-replicate-size",
        type=int,
        default=1,
        help="Number of HSDP replica groups.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["fp8", "int8", "gguf"],
        help="Quantization method for the transformer. "
        "Options: 'fp8' (FP8 W8A8 on Ada/Hopper, weight-only on older GPUs), 'int8' (Int8 W8A8), 'gguf' (GGUF quantized weights). "
        "Default: None (no quantization, uses BF16).",
    )
    parser.add_argument(
        "--gguf-model",
        type=str,
        default=None,
        help=("GGUF file path or HF reference for transformer weights. Required when --quantization gguf is set."),
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
    # NextStep-1.1 specific arguments
    parser.add_argument(
        "--guidance-scale-2",
        type=float,
        default=1.0,
        help="Secondary guidance scale (e.g. image-level CFG for NextStep-1.1).",
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
    parser.add_argument(
        "--extra-body",
        type=parse_profiler_config,
        default=None,
        help=(
            "Model-specific generation params as a JSON object, e.g. "
            '\'{"timestep_shift": 3.0, "cfg_text_scale": 4.0, "cfg_interval": [0.4, 1.0]}\'. '
            "Each key is filtered against the model's declared extra_body_params "
            "(see vllm_omni/model_extras), so unknown keys for the chosen model are "
            "silently dropped. Values here take precedence over the equivalent "
            "model-specific flags above."
        ),
    )
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to display stage durations.",
    )
    parser.add_argument(
        "--profiler-config",
        type=parse_profiler_config,
        default=None,
        help='JSON profiler config for torch/cuda profiling, e.g. \'{"profiler":"torch","torch_profiler_dir":"./perf"}\'.',
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        help="Enable logging of diffusion pipeline stats.",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=600,
        help="Timeout for initializing a single stage in seconds (default: 600s)",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=600,
        help="Timeout for initializing a single stage in seconds (default: 600s)",
    )
    parser.add_argument(
        "--use-system-prompt",
        type=str,
        default=None,
        choices=["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "en_unified", "custom"],
        help="System prompt preset for generation. Recommended: en_unified.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help=("Custom system prompt. Used when --use-system-prompt is custom. "),
    )
    parser.add_argument(
        "--auxiliary-text-encoder",
        type=str,
        default=None,
        help="Supplementary auxiliary text encoder parameters model name or path (especially for Hidream-l1-full).",
    )
    current_omni_platform.pre_register_and_update(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)
    use_nextstep = is_nextstep_model(args.model)

    cache_config = None
    cache_backend = args.cache_backend

    if cache_backend == "cache_dit":
        # cache-dit configuration: Hybrid DBCache + SCM + TaylorSeer
        # All parameters marked with [cache-dit only] in DiffusionCacheConfig
        cache_config = {
            # DBCache parameters [cache-dit only]
            "Fn_compute_blocks": 1,  # Optimized for single-transformer models
            "Bn_compute_blocks": 0,  # Number of backward compute blocks
            "max_warmup_steps": 4,  # Maximum warmup steps (works for few-step models)
            "residual_diff_threshold": 0.24,  # Higher threshold for more aggressive caching
            "max_continuous_cached_steps": 3,  # Limit to prevent precision degradation
            # TaylorSeer parameters [cache-dit only]
            "enable_taylorseer": False,  # Disabled by default (not suitable for few-step models)
            "taylorseer_order": 1,  # TaylorSeer polynomial order
            # SCM (Step Computation Masking) parameters [cache-dit only]
            "scm_steps_mask_policy": None,  # SCM mask policy: None (disabled), "slow", "medium", "fast", "ultra"
            "scm_steps_policy": "dynamic",  # SCM steps policy: "dynamic" or "static"
        }
    elif cache_backend == "tea_cache":
        # TeaCache configuration
        # All parameters marked with [tea_cache only] in DiffusionCacheConfig
        cache_config = {
            # TeaCache parameters [tea_cache only]
            "rel_l1_thresh": 0.2,  # Threshold for accumulated relative L1 distance
            # Note: coefficients will use model-specific defaults based on model_type
            #       (e.g., QwenImagePipeline or FluxPipeline)
        }

    profiler_enabled = args.profiler_config is not None

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

    omni_kwargs = {
        "model": args.model,
        "enable_layerwise_offload": args.enable_layerwise_offload,
        "vae_use_slicing": args.vae_use_slicing,
        "vae_use_tiling": args.vae_use_tiling,
        "cache_backend": args.cache_backend,
        "cache_config": cache_config,
        "enable_cache_dit_summary": args.enable_cache_dit_summary,
        "ulysses_degree": args.ulysses_degree,
        "ring_degree": args.ring_degree,
        "ulysses_mode": args.ulysses_mode,
        "cfg_parallel_size": args.cfg_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "vae_patch_parallel_size": args.vae_patch_parallel_size,
        "enable_expert_parallel": args.enable_expert_parallel,
        "enforce_eager": args.enforce_eager,
        "enable_cpu_offload": args.enable_cpu_offload,
        "mode": "text-to-image",
        "log_stats": args.log_stats,
        "enable_diffusion_pipeline_profiler": args.enable_diffusion_pipeline_profiler,
        "profiler_config": args.profiler_config,
        "init_timeout": args.init_timeout,
        "stage_init_timeout": args.stage_init_timeout,
        "auxiliary_text_encoder": args.auxiliary_text_encoder,
        **lora_args,
        **quant_kwargs,
    }
    if args.stage_configs_path:
        omni_kwargs["stage_configs_path"] = args.stage_configs_path
    if use_nextstep:
        # NextStep-1.1 requires explicit pipeline class
        omni_kwargs["model_class_name"] = "NextStep11Pipeline"
    omni = Omni(**omni_kwargs)
    model_class_name = get_model_class_name(omni)
    declared_extra_body_params = get_extra_body_params(model_class_name)

    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

    # Time profiling for generation
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Cache backend: {cache_backend if cache_backend else 'None (no acceleration)'}")
    print(f"  Quantization: {args.quantization if args.quantization else 'None (BF16)'}")
    if ignored_layers:
        print(f"  Ignored layers: {ignored_layers}")
    print(
        f"  Parallel configuration: tensor_parallel_size={args.tensor_parallel_size}, "
        f"ulysses_degree={args.ulysses_degree}, ulysses_mode={args.ulysses_mode}, "
        f"ring_degree={args.ring_degree}, cfg_parallel_size={args.cfg_parallel_size}, "
        f"vae_patch_parallel_size={args.vae_patch_parallel_size}, "
        f"enable_expert_parallel={args.enable_expert_parallel}."
    )
    print(f"  CPU offload: {args.enable_cpu_offload}; CPU Layerwise Offload: {args.enable_layerwise_offload}")
    print(f"  Image size: {args.width}x{args.height}")
    if args.lora_path:
        print(f"  LoRA: scale={args.lora_scale}")
    if args.stage_configs_path:
        print(f"  stage-configs-path: {args.stage_configs_path}")
    print(f"{'=' * 60}\n")

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

    prompt_dict = build_text_to_image_prompt(
        model_class_name=model_class_name,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
    )

    diffusion_params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=args.seed,
        generator=generator,
        true_cfg_scale=args.cfg_scale,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        num_inference_steps=args.num_inference_steps,
        num_outputs_per_prompt=args.num_images_per_prompt,
    )

    # Base layer: backward-compatible model-specific flags. New model params should
    # instead be declared in vllm_omni/model_extras and passed via --extra-body, so
    # this dict does not need to grow per-model.
    user_extra = {
        "cfg_scale": args.cfg_scale,
        "cfg_text_scale": args.cfg_scale,
        "negative_prompt": args.negative_prompt,
        "timestep_shift": args.timesteps_shift,
        "timesteps_shift": args.timesteps_shift,
        "cfg_schedule": args.cfg_schedule,
        "use_norm": args.use_norm,
        "use_system_prompt": args.use_system_prompt,
        "system_prompt": args.system_prompt,
    }
    # Override layer: generic JSON passthrough wins over the flags above. Keys are
    # still filtered against the model's declared extra_body_params downstream.
    if args.extra_body:
        user_extra.update(args.extra_body)
    if declared_extra_body_params:
        apply_declared_extra_args(diffusion_params, declared_extra_body_params, user_extra)
    else:
        diffusion_params.extra_args.update({k: v for k, v in user_extra.items() if v is not None})

    if lora_request:
        diffusion_params.extra_args["lora_request"] = lora_request
        diffusion_params.extra_args["lora_scale"] = args.lora_scale

    # Build per-stage sampling params for multi-stage models (e.g. BAGEL),
    # or wrap single diffusion params for single-stage models.
    init_non_diffusion = should_init_extra_args_for_non_diffusion_stages(
        model_class_name,
    )
    defaults = list(omni.default_sampling_params_list or [])
    sampling_params_list = [clone_sampling_params(p) for p in defaults]
    if not sampling_params_list:
        sampling_params_list = [diffusion_params]

    diffusion_replaced = False
    for idx, params in enumerate(sampling_params_list):
        if isinstance(params, OmniDiffusionSamplingParams):
            sampling_params_list[idx] = diffusion_params
            diffusion_replaced = True
        elif init_non_diffusion and hasattr(params, "extra_args"):
            if params.extra_args is None:
                params.extra_args = {}
            if args.seed is not None and hasattr(params, "seed"):
                params.seed = args.seed

    if not diffusion_replaced and len(sampling_params_list) == 1:
        sampling_params_list = [diffusion_params]

    outputs = omni.generate(prompt_dict, sampling_params_list=sampling_params_list)

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

    # omni.generate() returns list[OmniRequestOutput]
    if not outputs or len(outputs) == 0:
        raise ValueError("No output generated from omni.generate()")
    logger.info(f"Outputs: {outputs}")

    images = None
    for output in outputs:
        images = getattr(output, "images", None)
        if images:
            break
        req_out = getattr(output, "request_output", None)
        images = getattr(req_out, "images", None) if req_out is not None else None
        if images:
            break

    if not images:
        raise ValueError("No images found in request_output")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem or "qwen_image_output"
    if len(images) <= 1:
        images[0].save(output_path)
        print(f"Saved generated image to {output_path}")
    else:
        for idx, img in enumerate(images):
            save_path = output_path.parent / f"{stem}_{idx}{suffix}"
            img.save(save_path)
            print(f"Saved generated image to {save_path}")


if __name__ == "__main__":
    main()
