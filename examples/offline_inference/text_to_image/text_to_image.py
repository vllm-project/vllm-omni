# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import re
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm_omni.diffusion.data import logger
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id
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
        "--prompts",
        nargs="+",
        default=None,
        help="Multiple prompts for batched generation. Overrides --prompt when set. "
        "Each prompt is dispatched as part of a single omni.generate() batch call.",
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
        default=4.0,
        help="Classifier-free guidance scale. HunyuanImage3 recommends 4.0-5.0.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Height of generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of generated image.")
    parser.add_argument(
        "--output",
        type=str,
        default="qwen_image_output.png",
        help="Path to save the generated image (PNG). Used only in single-output mode "
        "(one prompt, one LoRA combo, one image). Ignored when --output-dir is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for batch/XYZ output. Required when there are multiple prompts "
        "or --axis is set. Files are saved as cell_x{x}_y{y}_z{z}.png; with --axis a "
        "grid.png (or grid_z{k}.png per Z value) is also written.",
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
        help="Path to LoRA adapter folder (PEFT format). Init-time static load: the adapter is "
        "pre-loaded into the engine cache and applied to every request. Mutually exclusive with --lora-paths.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Scale factor for --lora-path (default: 1.0).",
    )
    parser.add_argument(
        "--lora-paths",
        nargs="+",
        default=None,
        help="Multiple LoRA adapter folders (PEFT format) for per-request composition. "
        "Each request applies all listed adapters with the matching --lora-scales. "
        "Mutually exclusive with --lora-path.",
    )
    parser.add_argument(
        "--lora-scales",
        nargs="+",
        type=float,
        default=None,
        help="Per-adapter scales for --lora-paths. Length must match --lora-paths; "
        "defaults to 1.0 per adapter when omitted.",
    )
    parser.add_argument(
        "--max-loras",
        type=int,
        default=None,
        help="Maximum number of LoRA slots active simultaneously. Defaults to max(len(--lora-paths), 1).",
    )
    parser.add_argument(
        "--axis",
        action="append",
        default=None,
        metavar="SPEC",
        help="XYZ axis. Repeat up to 3 times. Spec form: NAME=TYPE:v1|v2|v3 where "
        "NAME ∈ {x,y,z} and TYPE ∈ {prompt, lora_scale[i], guidance_scale, "
        "num_inference_steps, seed}. The Cartesian product of X×Y×Z defines cells: "
        "X is columns, Y is rows, Z produces one grid per value (grid_z{k}.png). "
        'Example: --axis "x=lora_scale[0]:0|1" --axis "y=lora_scale[1]:0|1" '
        '--axis "z=prompt:a girl|a cat" yields a 2×2 grid per prompt.',
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
    current_omni_platform.pre_register_and_update(parser)
    from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults

    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


def _resolve_prompts(args: argparse.Namespace) -> list[str]:
    """Return the list of prompts to run. Prefers --prompts; falls back to --prompt."""
    if args.prompts:
        return list(args.prompts)
    return [args.prompt]


def _build_lora_request(path: str) -> LoRARequest:
    return LoRARequest(
        lora_name=Path(path).stem,
        lora_int_id=stable_lora_int_id(path),
        lora_path=path,
    )


def _resolve_lora(
    args: argparse.Namespace,
) -> tuple[list[LoRARequest], list[float], bool]:
    """Return (lora_requests, lora_scales, is_per_request) for the default cell.

    ``is_per_request`` is True when --lora-paths is given (per-request LoRA),
    False when --lora-path (init-time) or no LoRA is used.
    """
    if args.lora_path and args.lora_paths:
        raise ValueError("--lora-path and --lora-paths are mutually exclusive.")

    if not args.lora_paths:
        return [], [], False

    lora_paths = list(args.lora_paths)
    lora_scales = list(args.lora_scales) if args.lora_scales is not None else [1.0] * len(lora_paths)
    if len(lora_paths) != len(lora_scales):
        raise ValueError(
            f"--lora-paths ({len(lora_paths)}) and --lora-scales ({len(lora_scales)}) must have the same length."
        )
    requests = [_build_lora_request(p) for p in lora_paths]
    return requests, lora_scales, True


_LORA_SCALE_TYPE_RE = re.compile(r"^lora_scale\[(\d+)\]$")
_AXIS_TYPES = {"prompt", "guidance_scale", "num_inference_steps", "seed"}


@dataclass
class _Axis:
    name: str  # 'x' | 'y' | 'z'
    type: str
    values: list[str]  # raw strings; converted per type when applied


def _parse_axes(specs: list[str] | None) -> dict[str, _Axis]:
    """Parse repeated --axis specs into a dict keyed by axis name."""
    if not specs:
        return {}
    axes: dict[str, _Axis] = {}
    for spec in specs:
        name_part, sep, rest = spec.partition("=")
        if not sep:
            raise ValueError(f"--axis spec missing '=': {spec!r}")
        type_part, sep, values_part = rest.partition(":")
        if not sep:
            raise ValueError(f"--axis spec missing ':' between type and values: {spec!r}")
        name = name_part.strip().lower()
        if name not in ("x", "y", "z"):
            raise ValueError(f"--axis name must be x, y, or z; got {name!r}")
        if name in axes:
            raise ValueError(f"--axis {name} specified twice")
        atype = type_part.strip()
        if atype not in _AXIS_TYPES and not _LORA_SCALE_TYPE_RE.match(atype):
            raise ValueError(
                f"--axis type {atype!r} unknown. Supported: prompt, lora_scale[i], "
                f"guidance_scale, num_inference_steps, seed."
            )
        values = [v.strip() for v in values_part.split("|") if v.strip()]
        if not values:
            raise ValueError(f"--axis {name} has no values: {spec!r}")
        axes[name] = _Axis(name=name, type=atype, values=values)
    return axes


def _axis_label(axis: _Axis, value: str, lora_names: list[str]) -> str:
    """Render a short cell-header label. Embeds a newline between name and value
    so wide labels (e.g. ``lora_chardesign=1.00``) wrap cleanly in the grid
    margin strips; the grid composer honors explicit newlines verbatim.
    """
    if axis.type == "prompt":
        s = value if len(value) <= 40 else value[:37] + "..."
        return s
    m = _LORA_SCALE_TYPE_RE.match(axis.type)
    if m:
        idx = int(m.group(1))
        name = lora_names[idx] if idx < len(lora_names) else f"lora[{idx}]"
        return f"{name}\n{float(value):.2f}"
    return f"{axis.type}\n{value}"


def _apply_axis(
    axis: _Axis,
    raw_value: str,
    cell: dict,
    lora_count: int,
) -> None:
    """Mutate cell in place by applying a single axis value."""
    t = axis.type
    if t == "prompt":
        cell["prompt"] = raw_value
        return
    m = _LORA_SCALE_TYPE_RE.match(t)
    if m:
        idx = int(m.group(1))
        if idx >= lora_count:
            raise ValueError(f"axis lora_scale[{idx}] but only {lora_count} LoRA(s) provided via --lora-paths")
        cell["lora_scales"] = list(cell["lora_scales"])
        cell["lora_scales"][idx] = float(raw_value)
        return
    if t == "guidance_scale":
        cell["guidance_scale"] = float(raw_value)
        return
    if t == "num_inference_steps":
        cell["num_inference_steps"] = int(raw_value)
        return
    if t == "seed":
        cell["seed"] = int(raw_value)
        return
    raise ValueError(f"axis type {t!r} not implemented")


def _compose_grid(
    results: dict[tuple[int, int], list[Any]],
    num_rows: int,
    num_cols: int,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    title: str | None = None,
) -> Any:
    """Stitch the first image of each (row, col) cell into a single grid PIL image.

    Optional ``row_labels`` / ``col_labels`` reserve left/top strips with per-row
    and per-column text. Optional ``title`` reserves a narrow banner on top.
    """
    from PIL import Image, ImageDraw

    sample = next(iter(results.values()))[0]
    cell_w, cell_h = sample.width, sample.height

    col_strip = max(64, cell_h // 10) if col_labels else 0
    row_strip = max(220, cell_w // 4) if row_labels else 0
    title_strip = max(48, cell_h // 14) if title else 0

    top = title_strip + col_strip
    left = row_strip

    grid = Image.new("RGB", (left + cell_w * num_cols, top + cell_h * num_rows), color="white")
    draw = ImageDraw.Draw(grid)

    font = _load_label_font(max(18, (col_strip or cell_h // 10) // 3))
    font_row = _load_label_font(max(16, (col_strip or cell_h // 10) // 4))
    font_title = _load_label_font(max(22, title_strip // 2)) if title else font

    if title:
        draw.text(
            (grid.size[0] // 2, title_strip // 2),
            title,
            fill="black",
            font=font_title,
            anchor="mm",
        )
    if col_labels:
        for c_idx, lbl in enumerate(col_labels):
            x = left + c_idx * cell_w + cell_w // 2
            y = title_strip + col_strip // 2
            draw.text((x, y), lbl, fill="black", font=font, anchor="mm", align="center")
    if row_labels:
        for r_idx, lbl in enumerate(row_labels):
            y = top + r_idx * cell_h + cell_h // 2
            # Honor explicit newlines from axis labels; otherwise soft-wrap long text.
            rendered = lbl if "\n" in lbl else ("\n".join(textwrap.wrap(lbl, width=18)) or lbl)
            draw.text((row_strip // 2, y), rendered, fill="black", font=font_row, anchor="mm", align="center")

    for (r, c), imgs in results.items():
        grid.paste(imgs[0], (left + c * cell_w, top + r * cell_h))
    return grid


def _load_label_font(size: int):
    """Return a readable TrueType font if available, otherwise PIL's default bitmap font."""
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    args = parse_args()
    use_nextstep = is_nextstep_model(args.model)

    prompts = _resolve_prompts(args)
    lora_requests, lora_scales, lora_is_per_request = _resolve_lora(args)
    axes = _parse_axes(args.axis)

    if axes and len(prompts) > 1:
        raise ValueError(
            "--axis cannot be combined with multi-prompt input; put prompts on the prompt axis instead, "
            'e.g. --axis "z=prompt:a|b".'
        )
    requires_output_dir = len(prompts) > 1 or args.num_images_per_prompt > 1 or bool(axes)
    if requires_output_dir and not args.output_dir:
        raise ValueError(
            "--output-dir is required when running multiple prompts, multiple images per prompt, "
            "or --axis. Single --output is only valid for one image."
        )
    if args.max_loras is not None and args.lora_paths and args.max_loras < len(args.lora_paths):
        raise ValueError(
            f"--max-loras ({args.max_loras}) is smaller than len(--lora-paths) ({len(args.lora_paths)}). "
            "Composition needs one slot per adapter — raise --max-loras or remove it to auto-size."
        )

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
        print(f"Using init-time LoRA from: {args.lora_path}")

    # max_loras sizes the adapter cache at init. Per-request combos may load
    # several adapters simultaneously, so default to max(len(lora_paths), 1).
    if args.max_loras is not None:
        lora_args["max_loras"] = args.max_loras
    elif args.lora_paths:
        lora_args["max_loras"] = max(len(args.lora_paths), 1)

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
        **lora_args,
        **quant_kwargs,
    }
    if args.stage_configs_path:
        omni_kwargs["stage_configs_path"] = args.stage_configs_path
    if use_nextstep:
        # NextStep-1.1 requires explicit pipeline class
        omni_kwargs["model_class_name"] = "NextStep11Pipeline"
    omni = Omni(**omni_kwargs)

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
        print(f"  Init-time LoRA: scale={args.lora_scale}")
    if lora_is_per_request:
        print(f"  Per-request LoRA ({len(lora_requests)}):")
        for idx, (req, scale) in enumerate(zip(lora_requests, lora_scales)):
            print(f"    [{idx}] {req.lora_name} scale={scale}")
    print(f"  Prompts: {len(prompts)}")
    if axes:
        print(f"  Axes: {', '.join(f'{a.name}={a.type}:{len(a.values)} values' for a in axes.values())}")
    if args.stage_configs_path:
        print(f"  stage-configs-path: {args.stage_configs_path}")
    print(f"{'=' * 60}\n")

    extra_args = {
        "timesteps_shift": args.timesteps_shift,
        "cfg_schedule": args.cfg_schedule,
        "use_norm": args.use_norm,
        "use_system_prompt": args.use_system_prompt,
        "system_prompt": args.system_prompt,
    }

    def _run_cell(prompt: str, cell: dict) -> list[Any]:
        gen = torch.Generator(device=current_omni_platform.device_type).manual_seed(cell["seed"])
        sp = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            generator=gen,
            true_cfg_scale=args.cfg_scale,
            guidance_scale=cell["guidance_scale"],
            guidance_scale_2=args.guidance_scale_2,
            num_inference_steps=cell["num_inference_steps"],
            num_outputs_per_prompt=args.num_images_per_prompt,
            lora_requests=lora_requests if lora_is_per_request else [],
            lora_scales=cell["lora_scales"] if lora_is_per_request else [],
            extra_args=extra_args,
        )
        outs = omni.generate([{"prompt": prompt, "negative_prompt": args.negative_prompt}], sp)
        if not outs or not getattr(outs[0], "request_output", None):
            raise ValueError("Generate returned no request_output")
        imgs = outs[0].request_output.images
        if not imgs:
            raise ValueError("Empty image list from generate")
        return imgs

    defaults = {
        "prompt": prompts[0],
        "lora_scales": list(lora_scales),
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
    }

    # (z_idx, y_idx, x_idx) -> images for one cell. Unused axes collapse to idx 0.
    cell_images: dict[tuple[int, int, int], list[Any]] = {}

    generation_start = time.perf_counter()

    if axes:
        x_axis, y_axis, z_axis = axes.get("x"), axes.get("y"), axes.get("z")
        z_values = z_axis.values if z_axis else [None]
        y_values = y_axis.values if y_axis else [None]
        x_values = x_axis.values if x_axis else [None]

        total = len(z_values) * len(y_values) * len(x_values)
        counter = 0
        for z_idx, z_val in enumerate(z_values):
            for y_idx, y_val in enumerate(y_values):
                for x_idx, x_val in enumerate(x_values):
                    cell = dict(defaults)
                    for ax, raw in ((x_axis, x_val), (y_axis, y_val), (z_axis, z_val)):
                        if ax is not None:
                            _apply_axis(ax, raw, cell, len(lora_requests))
                    counter += 1
                    label = " ".join(
                        f"{ax.name}={raw}"
                        for ax, raw in ((x_axis, x_val), (y_axis, y_val), (z_axis, z_val))
                        if ax is not None
                    )
                    print(f"[cell {counter}/{total}] {label}")
                    cell_images[(z_idx, y_idx, x_idx)] = _run_cell(cell["prompt"], cell)
    else:
        # No axes: generate one image per prompt; cell key (0, p_idx, 0).
        cell = dict(defaults)
        for p_idx, prompt in enumerate(prompts):
            cell["prompt"] = prompt
            print(f"[cell {p_idx + 1}/{len(prompts)}] prompt={prompt!r}")
            cell_images[(0, p_idx, 0)] = _run_cell(prompt, cell)

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

    logger.info("Produced %d cells", len(cell_images))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for (z_idx, y_idx, x_idx), imgs in cell_images.items():
            for n_idx, img in enumerate(imgs):
                save_path = out_dir / f"cell_x{x_idx:02d}_y{y_idx:02d}_z{z_idx:02d}_n{n_idx:02d}.png"
                img.save(save_path)
                print(f"Saved {save_path}")

        if axes:
            x_axis, y_axis, z_axis = axes.get("x"), axes.get("y"), axes.get("z")
            lora_names = [req.lora_name for req in lora_requests]
            col_labels = [_axis_label(x_axis, v, lora_names) for v in x_axis.values] if x_axis else None
            row_labels = [_axis_label(y_axis, v, lora_names) for v in y_axis.values] if y_axis else None
            num_cols = len(x_axis.values) if x_axis else 1
            num_rows = len(y_axis.values) if y_axis else 1

            z_values = z_axis.values if z_axis else [None]
            for z_idx, z_val in enumerate(z_values):
                slice_cells = {(y, x): imgs for (z, y, x), imgs in cell_images.items() if z == z_idx}
                title = f"Z: {_axis_label(z_axis, z_val, lora_names)}" if z_axis else None
                grid = _compose_grid(
                    slice_cells,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    row_labels=row_labels,
                    col_labels=col_labels,
                    title=title,
                )
                fname = f"grid_z{z_idx:02d}.png" if z_axis else "grid.png"
                grid_path = out_dir / fname
                grid.save(grid_path)
                print(f"Saved grid to {grid_path}")
    else:
        # Single-output mode: exactly one cell, one image.
        only_images = next(iter(cell_images.values()))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        only_images[0].save(output_path)
        print(f"Saved generated image to {output_path}")


if __name__ == "__main__":
    main()
