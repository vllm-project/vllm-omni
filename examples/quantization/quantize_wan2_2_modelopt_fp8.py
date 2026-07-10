#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize Wan2.2 (TI2V-5B, 704x1280 T2V) to a ModelOpt FP8 Hugging Face checkpoint.

Calibrates the DiT transformer using a small video prompt set and exports a
diffusers-style directory whose transformer carries ModelOpt FP8 metadata.
The exported checkpoint is consumable by vllm-omni's ModelOpt FP8 adapter
(see vllm_omni/diffusion/model_loader/checkpoint_adapters/modelopt.py).

Layers kept full precision match the #2728 / #2795 pattern: condition embedder
(time/text/image), patch embedding, modulation (scale_shift_table), final
norm + proj_out, and sequence-parallel helpers. All attention + FFN linears
are quantized — static calibration handles the numerics that online FP8
couldn't (see #2920 ablation).

Default target is `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, the dense 5B variant that
fits 80GB BF16. The A14B MoE variants need 2+ GPUs and are out of scope here.

Example:
    python examples/quantization/quantize_wan2_2_modelopt_fp8.py \\
        --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \\
        --output ./wan22-ti2v-modelopt-fp8 \\
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
    p.add_argument("--model", required=True, help="Input Wan2.2 diffusers directory or HF id.")
    p.add_argument("--output", required=True, help="Output directory for the ModelOpt FP8 checkpoint.")
    p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    p.add_argument("--height", type=int, default=704, help="Calibration video height (Wan2.2 TI2V-5B native: 704).")
    p.add_argument("--width", type=int, default=1280, help="Calibration video width (Wan2.2 TI2V-5B native: 1280).")
    p.add_argument(
        "--num-frames",
        type=int,
        default=49,
        help="Frames per calibration sample. 49 matches the typical short benchmark; "
        "use 17 to reduce memory pressure during calibration.",
    )
    p.add_argument("--guidance-scale", type=float, default=5.0)
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
        help="Enable FP8 attention K/V/softmax quantizers. Off by default — Wan2.2's long attention "
        "sequences amplified FP8 drift in the online ablation (see #2920).",
    )
    p.add_argument(
        "--weight-block-size",
        type=str,
        default=None,
        help="Per-block weight quantization as 'M,N' (e.g. '128,128'). Default per-tensor. "
        "Note: vllm-omni's ModelOpt adapter may not yet dispatch block-wise scales — check #2924 "
        "for the HV-1.5 investigation status before relying on this.",
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


# Layers to KEEP at full precision. Wan2.2's module naming:
# - condition_embedder: time_embedder, time_proj, text_embedder, image_embedder (I2V)
# - patch_embedding: Conv3dLayer (already not Linear, belt-and-suspenders skip)
# - scale_shift_table: nn.Parameter modulation (not Linear, but pattern guard)
# - norm_out: AdaLayerNorm final
# - proj_out: final nn.Linear
# - timestep_proj_prepare / output_scale_shift_prepare: SP helpers
def _filter_func_wan22(name: str) -> bool:
    pattern = re.compile(
        r"(proj_out.*|"
        r".*(condition_embedder|patch_embedding|"
        r"norm_out|scale_shift_table|"
        r"timestep_proj_prepare|output_scale_shift_prepare).*)"
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
    mtq.disable_quantizer(backbone, _filter_func_wan22)
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

    # Try setting guidance on the pipeline's guider if present (newer diffusers APIs).
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
                # for pipelines that take CFG via guider config only.
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

    `export_hf_checkpoint` skips this step for unknown model types (Wan2.2 isn't
    in ModelOpt's recognized-model registry), so we must call the per-weight
    export helper ourselves. Same workaround as the HunyuanVideo-1.5 / HunyuanImage-3
    calibration helpers.
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


def _wan22_quant_config_block(weight_block_size: list[int] | None = None) -> dict:
    """Mirror ModelOpt FP8 metadata expected by vllm-omni's adapter (#2913)."""
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
            "condition_embedder*",
            "norm_out*",
            "output_scale_shift_prepare*",
            "patch_embedding*",
            "proj_out*",
            "scale_shift_table*",
            "timestep_proj_prepare*",
        ],
        "producer": {"name": "modelopt"},
        "quant_algo": "FP8",
        "quant_method": "modelopt",
    }


def _patch_quant_config(output_dir: Path, weight_block_size: list[int] | None = None) -> None:
    """Inject quant_algo: FP8 + config_groups into each transformer's config.json
    so vllm-omni's adapter (#2913) recognises the checkpoint as ModelOpt FP8.

    Patches both ``transformer`` and ``transformer_2`` (A14B) when present."""
    for sub in ("transformer", "transformer_2"):
        cfg_path = output_dir / sub / "config.json"
        if not cfg_path.exists():
            continue
        with cfg_path.open(encoding="utf-8") as f:
            cfg = json.load(f)

        new_qc = _wan22_quant_config_block(weight_block_size=weight_block_size)
        existing = cfg.get("quantization_config")
        if isinstance(existing, dict):
            producer = existing.get("producer")
            if isinstance(producer, dict):
                new_qc["producer"] = producer

        cfg["quantization_config"] = new_qc
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)


def _list_transformers(pipe: DiffusionPipeline) -> list[tuple[str, torch.nn.Module]]:
    """Return [(attr_name, module)] for every DiT on the pipeline.

    Wan2.2 A14B has both ``transformer`` (low-noise) and ``transformer_2``
    (high-noise). TI2V-5B has only ``transformer``.
    """
    out = []
    for attr in ("transformer", "transformer_2"):
        mod = getattr(pipe, attr, None)
        if mod is not None:
            out.append((attr, mod))
    return out


def _save_pipeline_with_fp8_transformer(
    pipe: DiffusionPipeline,
    model_path: str,
    output_dir: Path,
    max_shard_size: str = "5GB",
) -> None:
    """Copy source dir verbatim minus transformer dirs, then save each quantized DiT.

    Handles A14B's dual transformer (``transformer`` + ``transformer_2``) as well
    as the single-DiT TI2V-5B.
    """
    from modelopt.torch.export.diffusers_utils import hide_quantizers_from_state_dict

    src = Path(model_path)
    if not src.exists():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(model_path))

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(src, output_dir, ignore=shutil.ignore_patterns("transformer", "transformer_2"))

    for attr, backbone in _list_transformers(pipe):
        transformer_out = output_dir / attr
        # Pass the nn.Module (transformer), not the Pipeline wrapper.
        with hide_quantizers_from_state_dict(backbone):
            backbone.save_pretrained(
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
    weight_block_size = _parse_block_size(args.weight_block_size)

    print("Quantization plan:")
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
    transformers = _list_transformers(pipe)
    if not transformers:
        raise SystemExit("Pipeline has no transformer or transformer_2 attribute.")
    print(f"  found {len(transformers)} transformer(s): {', '.join(a for a, _ in transformers)}")

    quant_config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    if weight_block_size is not None:
        quant_config["quant_cfg"]["*weight_quantizer"] = {
            "num_bits": (4, 3),
            "block_sizes": {-1: weight_block_size[1], -2: weight_block_size[0]},
        }
        print(
            f"  -> overriding weight quantizer with block_sizes={weight_block_size} "
            f"({weight_block_size[0]}x{weight_block_size[1]} tiles)"
        )

    forward_loop = _build_forward_loop(pipe, args, prompts)

    # Quantize each DiT. The forward_loop runs the full pipeline, so quantizers
    # on every attached transformer (A14B: transformer + transformer_2) get
    # calibrated during the same passes.
    for attr, backbone in transformers:
        print(f"\nQuantizing {attr}...")
        quantized = mtq.quantize(backbone, quant_config, forward_loop)
        if quantized is not None:
            setattr(pipe, attr, quantized)
        _disable_known_problematic_quantizers(mtq, getattr(pipe, attr), quantize_mha=args.quantize_mha)

    # Export weights to FP8 only AFTER all transformers are calibrated. Exporting
    # inside the loop would convert transformer's weights to real FP8, and the
    # next transformer's forward_loop (full pipeline) would then fake-quantize an
    # already-FP8 weight -> "fake_e4m3fy not implemented for Float8_e4m3fn".
    print("\nForcing FP8 weight serialization (Wan2.2 isn't in ModelOpt's recognized-model registry,")
    print("so we have to call the per-weight export helper ourselves)...")
    total_exported = 0
    for attr, backbone in _list_transformers(pipe):
        exported = _force_export_quantized_weights(backbone, dtype)
        print(f"  -> {exported} weights converted to FP8 in {attr}")
        total_exported += exported

    if total_exported == 0:
        raise SystemExit(
            "No quantized weights were exported. Calibration may have skipped every layer "
            "(check the disable_quantizer regex) or `mtq.quantize` did not actually wrap any "
            "weight quantizers."
        )

    print("\nSaving pipeline with FP8 transformer(s)...")
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
        "    --num-inference-steps 30 --guidance-scale 5.0 --seed 42 \\\n"
        "    --output outputs/wan22_modelopt_fp8.mp4"
    )
    print(
        "\n  (--quantization fp8 is auto-upgraded to ModelOpt FP8 at runtime because the "
        "checkpoint's config.json has modelopt metadata.)"
    )


if __name__ == "__main__":
    main()
