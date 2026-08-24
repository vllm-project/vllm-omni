#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge W8A8 MXFP8 quantized Wan2.2 weights into HF Diffusers format for vllm-omni.

Based on the SGLang convert_wan_to_diffusers approach, extended to:
  - Inject `quantization_config` into each transformer's config.json so that
    vllm-omni's TransformerConfig.from_dict() auto-detects offline MXFP8 mode
    and selects NPUMxfp8LinearMethod (instead of the online/BF16 path).
  - Support MoE models (Wan2.2-T2V-A14B / I2V-A14B) with two transformers:
      quant_path/high_noise_model -> output/transformer
      quant_path/low_noise_model  -> output/transformer_2

Without the quantization_config injection, FP8 weights are passed to the online
quantization path (npu_dynamic_mx_quant), producing garbage output (visual artifacts).

Supported model types:
  - Wan2.2-T2V-A14B   (MoE cascade: transformer + transformer_2)
  - Wan2.2-I2V-A14B   (MoE cascade: transformer + transformer_2)
  - Wan2.2-TI2V-5B    (single transformer)

Usage:
  python vllm_omni/quantization/tools/merge_mxfp8_checkpoint.py \\
      --model-type      Wan2.2-T2V-A14B \\
      --original-model  /path/to/Wan2.2-T2V-A14B-Diffusers \\
      --quant-path      /path/to/quant-output \\
      --output-path     /path/to/merged-Wan2.2-T2V-A14B-MXFP8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from typing import Any

from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# Name mapping: quantization tool naming → diffusers naming
# ---------------------------------------------------------------------------

TRANSFORMER_KEYS_RENAME_DICT: dict[str, str] = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # Norm order swap (original: norm1, norm3, norm2 → diffusers: norm1, norm2, norm3)
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    # Self-attention
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    # Cross-attention
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
    # I2V image embedder
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
}

SUPPORTED_MODEL_TYPES = ["Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B", "Wan2.2-TI2V-5B"]

# MoE cascade models have high_noise_model + low_noise_model
CASCADE_MODEL_TYPES = {"Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B"}

# quantization_config injected into each transformer's config.json.
# is_checkpoint_mxfp8_serialized=True triggers NPUMxfp8LinearMethod (offline path)
# instead of NPUMxfp8OnlineLinearMethod (online/BF16 path).
MXFP8_QUANT_CONFIG: dict[str, Any] = {
    "quant_method": "mxfp8",
    "is_checkpoint_mxfp8_serialized": True,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_transformer_dirs(model_type: str) -> list[str]:
    if model_type in CASCADE_MODEL_TYPES:
        return ["transformer", "transformer_2"]
    return ["transformer"]


def _get_quant_subdir(model_type: str, quant_path: pathlib.Path, transformer_dir: str) -> pathlib.Path:
    if model_type in CASCADE_MODEL_TYPES:
        sub = "high_noise_model" if transformer_dir == "transformer" else "low_noise_model"
        return quant_path / sub
    return quant_path


def _remap_keys(state_dict: dict, quant_meta: dict) -> tuple[dict, dict]:
    """Apply TRANSFORMER_KEYS_RENAME_DICT to both state_dict and quant_meta."""
    new_state: dict = {}
    new_meta: dict = {}

    for key, tensor in state_dict.items():
        new_key = key
        for src, dst in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(src, dst)
        new_state[new_key] = tensor
        if key in quant_meta:
            new_meta[new_key] = quant_meta[key]

    return new_state, new_meta


def _load_safetensors(directory: pathlib.Path, glob: str = "*.safetensors") -> dict:
    candidates = sorted(directory.glob(glob))
    if not candidates:
        raise FileNotFoundError(f"No safetensors matching '{glob}' found in {directory}")
    state_dict = {}
    for f in candidates:
        state_dict.update(load_file(str(f)))
    return state_dict


def _load_quant_safetensors(directory: pathlib.Path) -> dict:
    try:
        return _load_safetensors(directory, "quant_model_weight*.safetensors")
    except FileNotFoundError:
        return _load_safetensors(directory, "*.safetensors")


def _load_quant_meta(directory: pathlib.Path) -> dict:
    """Load quant_model_description*.json (maps tensor name → quant type)."""
    candidates = sorted(directory.glob("quant_model_description*.json"))
    if not candidates:
        print(f"  WARNING: No quant_model_description*.json in {directory}; treating all as FLOAT")
        return {}
    with open(candidates[0]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-transformer conversion
# ---------------------------------------------------------------------------


def _convert_transformer(
    model_type: str,
    quant_subdir: pathlib.Path,
    output_dir: pathlib.Path,
    original_transformer_dir: pathlib.Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original BF16 weights as the merge base so that tensors msModelSlim
    # omits (biases, norms, embeddings) are preserved without gaps.
    print(f"  Loading original BF16 weights from {original_transformer_dir} …")
    base_state_dict = _load_safetensors(original_transformer_dir)
    print(f"  {len(base_state_dict)} BF16 tensors loaded")

    print(f"  Loading quantized weights from {quant_subdir} …")
    quant_state_dict = _load_quant_safetensors(quant_subdir)
    quant_meta = _load_quant_meta(quant_subdir)
    print(f"  {len(quant_state_dict)} quant tensors, {len(quant_meta)} meta entries")

    # Rename keys from msModelSlim convention to diffusers convention
    quant_state_dict, quant_meta = _remap_keys(quant_state_dict, quant_meta)

    # Overlay: start from BF16 base, then apply all msModelSlim tensors on top.
    # Quantized weight tensors + their scale tensors override the BF16 originals.
    # Non-quantized tensors that msModelSlim emits (FLOAT in quant_meta) keep the
    # msModelSlim copy; any tensor msModelSlim omits is retained from the BF16 base.
    merged = {**base_state_dict, **quant_state_dict}

    # Save merged safetensors
    out_weights = output_dir / "diffusion_pytorch_model.safetensors"
    save_file(merged, str(out_weights))
    print(f"  Saved {len(merged)} tensors → {out_weights}")

    # Save renamed quant metadata (optional, for reference)
    out_meta = output_dir / "quant_model_description.json"
    with open(out_meta, "w") as f:
        json.dump(quant_meta, f, indent=2)

    # Copy config.json from original transformer and inject quantization_config.
    # This is the critical step: TransformerConfig.from_dict() reads
    # config["quantization_config"]["quant_method"] to select the quant method,
    # and "is_checkpoint_mxfp8_serialized": true to select NPUMxfp8LinearMethod
    # (offline path) instead of NPUMxfp8OnlineLinearMethod (online/BF16 path).
    src_config = original_transformer_dir / "config.json"
    if src_config.is_file():
        with open(src_config) as f:
            config = json.load(f)
        config["quantization_config"] = MXFP8_QUANT_CONFIG
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Injected quantization_config into config.json → {output_dir / 'config.json'}")
    else:
        print(f"  WARNING: No config.json found at {src_config}")


# ---------------------------------------------------------------------------
# Main repack
# ---------------------------------------------------------------------------


def repack(
    model_type: str,
    original_model_path: pathlib.Path,
    quant_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    transformer_dirs = _get_transformer_dirs(model_type)

    # Step 1: Copy original model, skipping transformer dirs (will be replaced).
    print(f"Copying original model to {output_path} (skipping {transformer_dirs}) …")
    shutil.copytree(
        str(original_model_path),
        str(output_path),
        ignore=shutil.ignore_patterns(*transformer_dirs),
    )

    # Step 2+: Convert each transformer.
    for tdir in transformer_dirs:
        q_subdir = _get_quant_subdir(model_type, quant_path, tdir)
        out_tdir = output_path / tdir
        orig_tdir = original_model_path / tdir
        print(f"\nConverting {tdir} (quant source: {q_subdir.name}) …")
        _convert_transformer(model_type, q_subdir, out_tdir, orig_tdir)

    print(f"\nDone. Merged model → {output_path}")
    print("\nRun inference (no --quantization flag needed; auto-detected from config.json):")
    print(f"  python text_to_video.py --model {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-type", required=True, choices=SUPPORTED_MODEL_TYPES)
    parser.add_argument("--original-model", required=True, help="Original HF Diffusers model directory")
    parser.add_argument("--quant-path", required=True, help="msmodelslim quantized weights directory")
    parser.add_argument("--output-path", required=True, help="Output directory for merged model")
    args = parser.parse_args()

    repack(
        model_type=args.model_type,
        original_model_path=pathlib.Path(args.original_model),
        quant_path=pathlib.Path(args.quant_path),
        output_path=pathlib.Path(args.output_path),
    )


if __name__ == "__main__":
    main()
