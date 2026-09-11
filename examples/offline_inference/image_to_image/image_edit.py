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
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vllm_omni.diffusion.data import logger
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.openai.stage_params import clone_sampling_params
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import (
    build_image_to_image_prompt,
    get_ar_input_builder,
    get_ar_tokenizer_validator,
    get_extra_body_params,
    get_model_class_name,
    should_init_extra_args_for_non_diffusion_stages,
)
from vllm_omni.platforms import current_omni_platform


def _apply_ar_stage_inputs(
    ar_input_builder: Any,
    *,
    model: str,
    prompt_text: str,
    extra_body: dict[str, Any],
    num_images: int,
    height: int | None,
    width: int | None,
    prompt_dict: dict[str, Any],
    sampling_params_list: list[Any],
    text_output: bool = False,
    trust_remote_code: bool = False,
    validate_tokenizer: Any | None = None,
    allow_tokenizer_fallback: bool = False,
) -> None:
    """Apply a model's declared AR-stage inputs to the request in place.

    Mirrors the helper in ``text_to_image.py``: loads the tokenizer for
    byte-for-byte HF-parity AR tokenization, asks the model's
    ``ar_input_builder`` for the AR prefill + stop tokens, and writes them
    onto ``prompt_dict`` and the non-diffusion (AR) stage params.
    Model-agnostic -- only the declared-hook contract is assumed.

    ``trust_remote_code`` must be threaded in from the caller's own resolved
    ``--trust-remote-code`` flag -- this is a separate tokenizer load outside
    the main engine, so it needs the same explicit user opt-in rather than
    defaulting to trusting whatever ``--model`` was passed.

    ``validate_tokenizer``, when the model declares one via
    ``get_ar_tokenizer_validator``, is called on a successfully-loaded real
    tokenizer -- outside the load's try/except, so a validation failure (a
    model/tokenizer revision drifting from hardcoded special-token ids)
    raises instead of being swallowed into the string-prompt fallback.

    ``allow_tokenizer_fallback`` (default ``False``, i.e. fail fast): a
    tokenizer load failure normally raises, since silently degrading to the
    string-prompt form can produce a request that completes with subtly
    wrong stop tokens instead of surfacing the real problem (missing
    ``--trust-remote-code``, network/cache issue, etc.). Set ``True`` only
    for explicit unit tests or a deliberate offline-compat run where the
    caller has already decided a degraded prompt is acceptable.
    """
    try:
        from transformers import AutoTokenizer

        ar_tokenizer: Any | None = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    except Exception as exc:  # noqa: BLE001 - re-raised unless the caller opts into the fallback
        if not allow_tokenizer_fallback:
            raise
        logger.warning(f"AR tokenizer load failed ({exc}); falling back to string prompt (no BPE parity).")
        ar_tokenizer = None

    if ar_tokenizer is not None and validate_tokenizer is not None:
        validate_tokenizer(ar_tokenizer)

    ar_inputs = ar_input_builder(
        prompt=prompt_text,
        tokenizer=ar_tokenizer,
        extra_body=extra_body,
        num_images=num_images,
        height=height,
        width=width,
        text_output=text_output,
    )

    if ar_inputs.prompt_token_ids is not None:
        prompt_dict["prompt_token_ids"] = ar_inputs.prompt_token_ids
    elif ar_inputs.prompt is not None:
        prompt_dict["prompt"] = ar_inputs.prompt
    if ar_inputs.use_system_prompt:
        prompt_dict["use_system_prompt"] = ar_inputs.use_system_prompt
    prompt_dict["modalities"] = ar_inputs.modalities

    # Apply stop_token_ids to exactly the stage(s) the builder names -- not
    # "whichever stage isn't a diffusion stage," which would misfire on any
    # future topology with more than one non-diffusion stage.
    for stage_index in ar_inputs.stage_indices:
        if stage_index >= len(sampling_params_list):
            continue
        stage_params = sampling_params_list[stage_index]
        if not isinstance(stage_params, OmniDiffusionSamplingParams) and hasattr(stage_params, "stop_token_ids"):
            stage_params.stop_token_ids = ar_inputs.stop_token_ids


def parse_profiler_config(value: str) -> dict[str, Any]:
    try:
        config = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--profiler-config must be valid JSON: {e}") from e
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError("--profiler-config must be a JSON object")
    return config


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
        "--deploy-config",
        type=str,
        default=None,
        help=(
            "Path to a deploy YAML. Required for multi-stage image-edit pipelines "
            "whose deploy config is not auto-loaded."
        ),
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
        required=False,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        metavar="W",
        help="Output image width in pixels. Default: None (pipeline's default).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        metavar="H",
        help="Output image height in pixels. Default: None (pipeline's default).",
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
        help=(
            "True classifier-free guidance scale (default: 4.0). Guidance scale as defined in Classifier-Free "
            "Diffusion Guidance. Classifier-free guidance is enabled by setting cfg_scale > 1 and providing "
            "a negative_prompt. Higher guidance scale encourages images closely linked to the text prompt, "
            "usually at the expense of lower image quality."
        ),
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help=(
            "Guidance scale for guidance-distilled models (default: 1.0, disabled). "
            "Unlike classifier-free guidance (--cfg-scale), guidance-distilled models take the guidance scale "
            "directly as an input parameter. Enabled when guidance_scale > 1. Ignored when not using guidance-distilled models."
        ),
    )
    parser.add_argument(
        "--guidance-scale-2", type=float, default=None, help="image guidance scale for image-to-image generation."
    )
    parser.add_argument(
        "--extra-args",
        type=parse_profiler_config,
        default=None,
        help="JSON object copied to OmniDiffusionSamplingParams.extra_args, e.g. '{\"cfg_text_scale\": 4.0}'.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_image_edit.png",
        help=("Path to save the edited image (PNG). Or prefix for Qwen-Image-Layered model save images(PNG)."),
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
    parser.add_argument("--layers", type=int, default=4, help="Number of layers to decompose the input image into.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Bucket in (640, 1024) to determine the condition and output resolution. If width and height are not provided, this will be set to default 640.",
    )

    parser.add_argument(
        "--color-format",
        type=str,
        default="RGB",
        help="For Qwen-Image-Layered, set to RGBA.",
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
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Number of GPUs used for classifier free guidance parallel size (max 3 branches).",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=None,
        help=(
            "Disable torch.compile and force eager execution. Left unset (None) "
            "so it is only forwarded when explicitly given; "
            "otherwise the per-stage deploy YAML value wins."
        ),
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
        "--vae-patch-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for VAE patch/tile parallelism (decode).",
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
        "--trust-remote-code",
        action="store_true",
        help="Trust and execute custom modeling code from the model repo (required by e.g. HunyuanImage-3.0).",
    )
    parser.add_argument(
        "--allow-tokenizer-fallback",
        action="store_true",
        help=(
            "For models with a declared ar_input_builder (e.g. HunyuanImage-3.0): if loading "
            "the AR tokenizer fails, degrade to the string-prompt form (no BPE parity) instead "
            "of failing the run. Off by default -- a tokenizer load failure usually means a "
            "missing --trust-remote-code or a network/cache issue that's worth surfacing, not "
            "silently masking with a possibly-wrong prompt."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.resolution and (args.width or args.height):
        raise ValueError("--resolution and --width/--height cannot be specified together")
    if args.width is not None and args.width <= 0:
        raise ValueError("--width must be a positive integer")
    if args.height is not None and args.height <= 0:
        raise ValueError("--height must be a positive integer")
    if not args.width and not args.height and not args.resolution:
        args.resolution = 640

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
    omni_kwargs: dict[str, Any] = dict(
        model=args.model,
        enable_layerwise_offload=args.enable_layerwise_offload,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        enable_cpu_offload=args.enable_cpu_offload,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
        profiler_config=args.profiler_config,
    )
    if args.enforce_eager is not None:
        omni_kwargs["enforce_eager"] = args.enforce_eager
    if args.trust_remote_code:
        omni_kwargs["trust_remote_code"] = True
    if args.deploy_config:
        omni_kwargs["deploy_config"] = args.deploy_config
    omni = Omni(**omni_kwargs)
    model_class_name = get_model_class_name(omni)
    declared_extra_body_params = get_extra_body_params(model_class_name)
    print("Pipeline loaded")

    profiler_enabled = args.profiler_config is not None

    # Time profiling for generation
    print(f"\n{'=' * 60}")
    print("Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  Cache backend: {args.cache_backend if args.cache_backend else 'None (no acceleration)'}")
    if args.height is not None or args.width is not None:
        print(f"  Output size: {args.width or 'auto'}x{args.height or 'auto'}")
    if isinstance(input_image, list):
        print(f"  Number of input images: {len(input_image)}")
        for idx, img in enumerate(input_image):
            print(f"    Image {idx + 1} size: {img.size}")
    else:
        print(f"  Input image size: {input_image.size}")
    print(
        f"  Parallel configuration: ulysses_degree={args.ulysses_degree}, ring_degree={args.ring_degree}, cfg_parallel_size={args.cfg_parallel_size}, tensor_parallel_size={args.tensor_parallel_size}, enable_expert_parallel: {args.enable_expert_parallel}"
    )
    print(f"{'=' * 60}\n")

    generation_start = time.perf_counter()

    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

    prompt_dict = build_image_to_image_prompt(
        model_class_name=model_class_name,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=input_image,
        height=args.height,
        width=args.width,
    )

    extra_args_from_cli = dict(args.extra_args or {})
    if args.negative_prompt is not None:
        extra_args_from_cli.setdefault("negative_prompt", args.negative_prompt)

    diffusion_params = OmniDiffusionSamplingParams(
        generator=generator,
        true_cfg_scale=args.cfg_scale,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        num_inference_steps=args.num_inference_steps,
        num_outputs_per_prompt=args.num_outputs_per_prompt,
        layers=args.layers,
        resolution=args.resolution,
        height=args.height,
        width=args.width,
    )
    if declared_extra_body_params:
        apply_declared_extra_args(
            diffusion_params,
            declared_extra_body_params,
            extra_args_from_cli,
        )
    else:
        diffusion_params.extra_args.update({k: v for k, v in extra_args_from_cli.items() if v is not None})

    # Build per-stage sampling params for multi-stage models
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
            merged_extra = dict(getattr(params, "extra_args", {}) or {})
            merged_extra.update(diffusion_params.extra_args)
            diffusion_params.extra_args = merged_extra
            sampling_params_list[idx] = diffusion_params
            diffusion_replaced = True
        elif init_non_diffusion and hasattr(params, "extra_args"):
            if params.extra_args is None:
                params.extra_args = {}

    if not diffusion_replaced and len(sampling_params_list) == 1:
        sampling_params_list = [diffusion_params]

    # Models with an AR text stage (e.g. HunyuanImage3 image-editing) declare an
    # ar_input_builder; build the AR prefill + stop tokens declaratively from the
    # plain prompt + reference image count + extra_args. Others are untouched.
    ar_input_builder = get_ar_input_builder(model_class_name)
    if ar_input_builder is not None:
        _apply_ar_stage_inputs(
            ar_input_builder,
            model=args.model,
            prompt_text=args.prompt,
            extra_body=extra_args_from_cli,
            num_images=len(input_images),
            height=args.height,
            width=args.width,
            prompt_dict=prompt_dict,
            sampling_params_list=sampling_params_list,
            trust_remote_code=args.trust_remote_code,
            validate_tokenizer=get_ar_tokenizer_validator(model_class_name),
            allow_tokenizer_fallback=args.allow_tokenizer_fallback,
        )

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

    if not outputs:
        raise ValueError("No output generated from omni.generate()")

    images = None
    for output in outputs:
        images = getattr(output, "images", None)
        if images:
            break
        req_out = output
        images = getattr(req_out, "images", None) if req_out is not None else None
        if images:
            break

    if not images:
        images = extract_images_from_outputs(outputs)

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
