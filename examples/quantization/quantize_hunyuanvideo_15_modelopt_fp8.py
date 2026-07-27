#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize HunyuanVideo-1.5 (480p T2V) to a ModelOpt FP8 Hugging Face checkpoint.

Calibrates the DiT transformer using a small video prompt set and exports a
diffusers-style directory whose transformer carries ModelOpt FP8 metadata.
The exported checkpoint is consumable by vllm-omni's ModelOpt FP8 adapter
(see vllm_omni/diffusion/model_loader/checkpoint_adapters/modelopt.py).

Layers kept full precision match the #2728 / #2795 pattern: modulation,
AdaLayerNorm, entry/exit projections, embeddings, the token refiner path,
and final proj_out. MHA quantizers are off by default; HV-1.5 self-attention
empirically degrades under FP8 (see #2920 ablation).

Example:
    python examples/quantization/quantize_hunyuanvideo_15_modelopt_fp8.py \\
        --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \\
        --output ./hv15-480p-modelopt-fp8 \\
        --overwrite
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline

DEFAULT_PROMPTS = [
    "A dog running across a field of golden wheat.",
    "An astronaut riding a horse across the surface of Mars, red dust swirling, cinematic wide shot.",
    "A hummingbird hovering in front of a vibrant red flower, slow motion, macro shot.",
    "A crackling campfire at night under a starry sky, sparks rising into the dark.",
    "An underwater shot of a coral reef with tropical fish swimming by, sun rays piercing the water.",
    "A close-up of a blooming rose covered in morning dew, soft natural light.",
    "A peaceful mountain village at dawn, mist rolling over the rooftops, cinematic establishing shot.",
    "A skateboarder doing a kickflip in an urban plaza, slow motion, golden hour lighting.",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Input HV-1.5 diffusers directory or HF id.")
    p.add_argument("--output", required=True, help="Output directory for the ModelOpt FP8 checkpoint.")
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument(
        "--num-frames",
        type=int,
        default=33,
        help="Frames per calibration sample. 33 matches the typical short benchmark.",
    )
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument(
        "--calib-steps",
        type=int,
        default=10,
        help="Denoising steps per calibration prompt (10 is enough for amax statistics).",
    )
    p.add_argument("--calib-size", type=int, default=8, help="How many prompts to use for calibration.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Custom calibration prompt. Repeat to provide multiple.",
    )
    p.add_argument(
        "--quantize-mha",
        action="store_true",
        help="Enable FP8 attention K/V/softmax quantizers. Off by default — empirically degrades HV-1.5 video output.",
    )
    p.add_argument(
        "--weight-block-size",
        type=str,
        default=None,
        help="Per-block weight quantization as 'M,N' (e.g. '128,128' for 128x128 tiles). "
        "Default: per-tensor (one scale per linear). Block-wise typically gives tighter quality at "
        "negligible memory cost. Static FP8 is exempt from upstream vLLM's online block-wise gate.",
    )
    p.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return p


def _parse_block_size(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    parts = [int(x) for x in spec.split(",") if x.strip()]
    if len(parts) != 2:
        raise SystemExit(f"--weight-block-size must be 'M,N' (2 ints), got {spec!r}")
    return parts


def _require_modelopt() -> Any:
    try:
        import modelopt.torch.quantization as mtq
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "NVIDIA ModelOpt is not installed. Install with:\n"
            "  pip install 'nvidia-modelopt[all]'\n"
            f"Original error: {exc}"
        ) from exc
    return mtq


def _ensure_paths(args: argparse.Namespace) -> tuple[str, Path]:
    model_path = args.model
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {output_dir}\nPass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    return model_path, output_dir


def _select_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _build_prompts(args: argparse.Namespace) -> list[str]:
    prompts = args.prompt or DEFAULT_PROMPTS
    if args.calib_size <= 0:
        raise SystemExit("--calib-size must be positive.")
    if len(prompts) < args.calib_size:
        repeats = (args.calib_size + len(prompts) - 1) // len(prompts)
        prompts = (prompts * repeats)[: args.calib_size]
    return prompts[: args.calib_size]


# Layers to KEEP at full precision (mirror of the #2920 wiring + #2728/#2795 skip pattern).
# - x_embedder, image_embedder, context_embedder*, time_embed*, cond_type_embed: entry/embedding
# - norm_out, norm1*.linear, norm1_context*.linear, norm2*, norm2_context*: AdaLayerNorm modulation
# - proj_out: final output projection
# - token_refiner*: text-encoder refinement uses diffusers raw nn.Linear
def _filter_func_hv15(name: str) -> bool:
    pattern = re.compile(
        r"(proj_out.*|"
        r".*(x_embedder|image_embedder|context_embedder|context_embedder_2|"
        r"time_embed|cond_type_embed|"
        r"norm_out|norm1\.linear|norm1_context\.linear|norm2|norm2_context|"
        r"token_refiner).*)"
    )
    return pattern.match(name) is not None


def _mha_filter_func(name: str) -> bool:
    pattern = re.compile(
        r".*(q_bmm_quantizer|k_bmm_quantizer|v_bmm_quantizer|softmax_quantizer|bmm2_output_quantizer).*"
    )
    return pattern.match(name) is not None


def _disable_known_problematic_quantizers(mtq: Any, backbone: torch.nn.Module, *, quantize_mha: bool) -> None:
    if not hasattr(mtq, "disable_quantizer"):
        return
    mtq.disable_quantizer(backbone, _filter_func_hv15)
    if not quantize_mha:
        mtq.disable_quantizer(backbone, _mha_filter_func)


def _load_pipeline(model_path: str, dtype: torch.dtype) -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")
    return pipe


def _build_forward_loop(pipe: DiffusionPipeline, args: argparse.Namespace, prompts: list[str]):
    generator = torch.Generator(device="cuda")

    # Try to set guidance on the pipeline's guider object up front (modern
    # diffusers HV-1.5 uses a Guider abstraction, not a per-call kwarg). Falls
    # back silently — calibration uses whatever default the pipeline ships with.
    guider = getattr(pipe, "guider", None)
    if guider is not None and hasattr(guider, "guidance_scale"):
        try:
            guider.guidance_scale = args.guidance_scale
        except Exception:
            pass

    base_kwargs = dict(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.calib_steps,
        output_type="latent",
    )

    def forward_loop(*_unused_args, **_unused_kwargs) -> None:
        with torch.inference_mode():
            for idx, prompt in enumerate(prompts):
                generator.manual_seed(args.seed + idx)
                # Try with guidance_scale first; fall back without on TypeError
                # for pipelines (like HV-1.5) that take CFG via guider config.
                try:
                    pipe(prompt=prompt, generator=generator, guidance_scale=args.guidance_scale, **base_kwargs)
                except TypeError as exc:
                    if "guidance_scale" not in str(exc):
                        raise
                    pipe(prompt=prompt, generator=generator, **base_kwargs)

    return forward_loop


def _summarize_export(output_dir: Path) -> None:
    cfg_path = output_dir / "transformer" / "config.json"
    if not cfg_path.exists():
        print(f"[warn] {cfg_path} missing.", file=sys.stderr)
        return
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict):
        print("[warn] No quantization_config in transformer/config.json.", file=sys.stderr)
        return
    print("Export summary:")
    print(f"  quant_method: {qc.get('quant_method')}")
    print(f"  quant_algo:   {qc.get('quant_algo')}")
    producer = qc.get("producer")
    if isinstance(producer, dict):
        print(f"  producer:     {producer.get('name')} {producer.get('version')}")
    print(f"  config path:  {cfg_path}")


def _force_export_quantized_weights(backbone: torch.nn.Module, dtype: torch.dtype) -> int:
    """Convert in-memory weights of quantized modules to actual FP8 storage.

    `export_hf_checkpoint` skips this step for unknown model types (HV-1.5 isn't
    in ModelOpt's recognized-model registry), so we must call the per-weight
    export helper ourselves. Same workaround as the HunyuanImage-3 calibration
    helper.
    """
    from modelopt.torch.export.quant_utils import (
        QUANTIZATION_NONE,
        get_quantization_format,
        quantizer_attr_names,
        weight_attr_names,
    )
    from modelopt.torch.export.unified_export_hf import _export_quantized_weight

    exported = 0
    for name, module in backbone.named_modules():
        try:
            quantization_format = get_quantization_format(module)
        except Exception as exc:
            print(f"[warn] Could not inspect quantization format for {name}: {exc}", file=sys.stderr)
            continue
        if quantization_format == QUANTIZATION_NONE:
            continue
        for weight_name in weight_attr_names(module):
            quantizer_attrs = quantizer_attr_names(weight_name)
            weight_quantizer = getattr(module, quantizer_attrs.weight_quantizer, None)
            if weight_quantizer is None or not getattr(weight_quantizer, "is_enabled", False):
                continue
            _export_quantized_weight(module, dtype, weight_name)
            exported += 1
    return exported


def _hv15_quant_config_block(weight_block_size: list[int] | None = None) -> dict:
    """Mirror ModelOpt FP8 metadata expected by vllm-omni's adapter (#2913).

    Same shape as the HunyuanImage-3 author's _hunyuan_quant_config(). When
    `weight_block_size` is given, advertise block-wise weight quantization in
    the saved metadata (so consumers know to expect multi-element scale tensors).
    """
    weights_cfg: dict = {"dynamic": False, "num_bits": 8, "type": "float"}
    if weight_block_size is not None:
        weights_cfg["strategy"] = "block"
        weights_cfg["block_structure"] = f"{weight_block_size[0]}x{weight_block_size[1]}"
    return {
        "config_groups": {
            "group_0": {
                "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                "weights": weights_cfg,
                "targets": ["Linear"],
            }
        },
        "ignore": [
            "context_embedder*",
            "context_embedder_2*",
            "cond_type_embed*",
            "image_embedder*",
            "norm1.linear*",
            "norm1_context.linear*",
            "norm2*",
            "norm2_context*",
            "norm_out*",
            "proj_out*",
            "time_embed*",
            "token_refiner*",
            "x_embedder*",
        ],
        "producer": {"name": "modelopt"},
        "quant_algo": "FP8",
        "quant_method": "modelopt",
    }


def _patch_quant_config(output_dir: Path, weight_block_size: list[int] | None = None) -> None:
    """Inject quant_algo: FP8 + config_groups into transformer/config.json so
    vllm-omni's adapter (#2913) recognises the checkpoint as ModelOpt FP8."""
    cfg_path = output_dir / "transformer" / "config.json"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    new_qc = _hv15_quant_config_block(weight_block_size=weight_block_size)
    existing = cfg.get("quantization_config")
    if isinstance(existing, dict):
        producer = existing.get("producer")
        if isinstance(producer, dict):
            new_qc["producer"] = producer

    cfg["quantization_config"] = new_qc
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _save_pipeline_with_fp8_transformer(
    pipe: DiffusionPipeline,
    model_path: str,
    output_dir: Path,
    max_shard_size: str = "5GB",
) -> None:
    """Save the pipeline with the (now FP8) transformer.

    Copies the source directory verbatim except for `transformer/`, then
    saves the transformer with quantizers hidden so the state dict contains
    only the FP8 weights + scale tensors.
    """
    from modelopt.torch.export.diffusers_utils import hide_quantizers_from_state_dict

    src = Path(model_path)
    if not src.exists():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(model_path))

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(src, output_dir, ignore=shutil.ignore_patterns("transformer"))

    transformer_out = output_dir / "transformer"
    # `hide_quantizers_from_state_dict` walks named_modules(); pass the actual
    # nn.Module (transformer), not the diffusers Pipeline wrapper.
    with hide_quantizers_from_state_dict(pipe.transformer):
        pipe.transformer.save_pretrained(
            str(transformer_out),
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )


def main() -> None:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ModelOpt FP8 quantization.")

    mtq = _require_modelopt()
    model_path, output_dir = _ensure_paths(args)
    dtype = _select_dtype(args.dtype)
    prompts = _build_prompts(args)

    print("Quantization plan:")
    weight_block_size = _parse_block_size(args.weight_block_size)

    print(f"  input:           {args.model}")
    print(f"  output:          {output_dir}")
    print(f"  dtype:           {dtype}")
    print(f"  height/width:    {args.height}x{args.width}")
    print(f"  num_frames:      {args.num_frames}")
    print(f"  calib_size:      {len(prompts)}")
    print(f"  calib_steps:     {args.calib_steps}")
    print(f"  quantize_mha:    {args.quantize_mha}")
    print(
        f"  weight strategy: {'block-wise ' + str(weight_block_size) if weight_block_size else 'per-tensor (default)'}"
    )

    pipe = _load_pipeline(model_path, dtype)
    backbone = pipe.transformer

    quant_config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    if weight_block_size is not None:
        # Switch from per-tensor (default) to block-wise weight quantization.
        # ModelOpt's wildcard "*weight_quantizer" matches every linear's weight quantizer.
        quant_config["quant_cfg"]["*weight_quantizer"] = {
            "num_bits": (4, 3),  # E4M3 (FP8 weights, same as default)
            "block_sizes": {-1: weight_block_size[1], -2: weight_block_size[0]},
        }
        print(
            f"  -> overriding weight quantizer with block_sizes={weight_block_size} "
            f"({weight_block_size[0]}x{weight_block_size[1]} tiles)"
        )

    forward_loop = _build_forward_loop(pipe, args, prompts)
    quantized = mtq.quantize(backbone, quant_config, forward_loop)
    if quantized is not None:
        pipe.transformer = quantized
        backbone = quantized

    _disable_known_problematic_quantizers(mtq, backbone, quantize_mha=args.quantize_mha)

    print("\nForcing FP8 weight serialization (HV-1.5 isn't in ModelOpt's recognized-model registry,")
    print("so we have to call the per-weight export helper ourselves)...")
    exported = _force_export_quantized_weights(backbone, dtype)
    print(f"  -> {exported} weights converted to FP8 in memory")
    if exported == 0:
        raise SystemExit(
            "No quantized weights were exported. Calibration may have skipped every layer "
            "(check the disable_quantizer regex) or `mtq.quantize` did not actually wrap any "
            "weight quantizers."
        )

    print("\nSaving pipeline with FP8 transformer...")
    _save_pipeline_with_fp8_transformer(pipe, model_path, output_dir)
    _patch_quant_config(output_dir, weight_block_size=weight_block_size)
    print(f"Saved to: {output_dir}")
    _summarize_export(output_dir)

    print("\nNext: validate the checkpoint with vllm-omni:")
    print(
        "  python examples/offline_inference/text_to_video/text_to_video.py \\\n"
        f"    --model {output_dir} \\\n"
        "    --quantization fp8 \\\n"
        "    --prompt 'A dog running across a field of golden wheat.' \\\n"
        f"    --height {args.height} --width {args.width} --num-frames {args.num_frames} \\\n"
        "    --num-inference-steps 30 --guidance-scale 6.0 --seed 42 \\\n"
        "    --output outputs/hv15_modelopt_fp8.mp4 \\\n"
        "    --enforce-eager"
    )
    print(
        "\n  (--quantization fp8 is auto-upgraded to ModelOpt FP8 at runtime because the "
        "checkpoint's config.json has modelopt metadata.)"
    )


if __name__ == "__main__":
    main()
