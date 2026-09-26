#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge W4A4_MXFP4_DUALSCALE quantized Wan2.2 weights into HF Diffusers format.

msModelSlim produces a checkpoint where each linear layer is either:
  - W4A4_MXFP4_DUALSCALE: quantized weights with fine/coarse scales and mul_scale
  - FLOAT (BF16 fallback): kept as original BF16 weights for precision-sensitive layers

BF16 fallback layers may be interleaved anywhere in the transformer (not just leading
blocks). The merge script detects them from quant_model_description.json and writes
their prefixes into config.json as ignored_layers so the runtime dynamically routes
each layer to the correct quantization method at weight-loading time.

MXFP4_DUALSCALE key structure per linear layer (msModelSlim naming)
---------------------------------------------------------------------
  blocks.N.X.linear.weight             W4A4_MXFP4_DUALSCALE  – int8 (FP4 packed)
  blocks.N.X.linear.weight_scale       W4A4_MXFP4_DUALSCALE  – uint8 (float8_e8m0fnu fine scale, per-32K)
  blocks.N.X.linear.weight_dual_scale  W4A4_MXFP4_DUALSCALE  – float32 (coarse scale, per-512K)
  blocks.N.X.linear.bias               FLOAT                  – bias (if present)
  blocks.N.X.div.mul_scale             FLOAT                  – float32 per-input-channel activation pre-scale

BF16 fallback key structure (no wrappers, plain linear weights):
  blocks.N.X.weight                    FLOAT
  condition_embedder.*.weight          FLOAT  (always BF16 — not quantized)

Self-attention QKV notes
------------------------
Self-attention Q/K/V weights are separate in the checkpoint (self_attn.q/k/v) but fused
into a single to_qkv layer in vllm-omni. The transformer's load_weights() handles this via
stacked_params_mapping. This script keeps Q/K/V keys separate — do NOT pre-fuse them here.

For mul_scale specifically: even though Q/K/V process the same input (same mul_scale value),
they are kept separate. load_weights() routes all three to the same to_qkv.mul_scale parameter
and each overwrites the previous. Since Q=K=V for mul_scale, the final value is correct.

NOTE: Pre-fusing as to_qkv.mul_scale would BREAK loading because ".attn1.to_q" is a
substring of ".attn1.to_qkv", causing load_weights() stacked_params_mapping to produce a
garbage key ("to_qkvkv") that is not in params_dict, triggering a break that skips the
direct-load else branch entirely.

Supported model types:
  - Wan2.2-T2V-A14B  (MoE cascade: transformer + transformer_2)
  - Wan2.2-I2V-A14B  (MoE cascade: transformer + transformer_2)

Note: Wan2.2-TI2V-5B is NOT supported. Its smaller parameter count causes
unacceptable accuracy loss under W4A4 quantization. Use MXFP8 for TI2V-5B.

Usage:
  python merge_mxfp4_dualscale_checkpoint.py \\
      --model-type        Wan2.2-T2V-A14B \\
      --original-model    /path/to/Wan2.2-T2V-A14B-Diffusers \\
      --quant-path        /path/to/msmodelslim-output \\
      --output-path       /path/to/merged-output
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
from typing import Any

import torch
from safetensors.torch import load_file, save_file

# ---------------------------------------------------------------------------
# Key rename: msModelSlim naming → Diffusers / vllm-omni naming
# Identical to merge_mxfp8_checkpoint.py; applied to all blocks uniformly.
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
    # Norm order swap (quant tool: norm1, norm3, norm2 → diffusers: norm1, norm2, norm3)
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

SUPPORTED_MODEL_TYPES = ["Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B"]
CASCADE_MODEL_TYPES = {"Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B"}

# Suffixes that appear inside .linear.* wrapper for MXFP4 tensors.
_LINEAR_ATTRS = ("weight_dual_scale", "weight_scale", "weight", "bias")

_BLOCK_IDX_RE = re.compile(r"^blocks\.(\d+)\.")


# ---------------------------------------------------------------------------
# Key transformation helpers
# ---------------------------------------------------------------------------


def _parse_block_idx(key: str) -> int | None:
    """Extract block index from a key like 'blocks.5.attn1.to_q.weight'."""
    m = _BLOCK_IDX_RE.match(key)
    return int(m.group(1)) if m else None


def _apply_rename_dict(key: str) -> str:
    for src, dst in TRANSFORMER_KEYS_RENAME_DICT.items():
        key = key.replace(src, dst)
    return key


def _strip_mxfp4_wrapper(key: str) -> str:
    """Strip .linear.ATTR or .div.mul_scale wrappers added by msModelSlim.

    MXFP4 tensors are wrapped in sub-modules:
      X.linear.weight / weight_scale / weight_dual_scale / bias
      X.div.mul_scale

    MXFP8 and FLOAT tensors have no wrappers — this function is a no-op for them.

    Examples after apply_rename_dict:
      attn1.to_q.linear.weight      → attn1.to_q.weight
      attn1.to_q.linear.weight_scale → attn1.to_q.weight_scale
      attn1.to_q.div.mul_scale      → attn1.to_q.mul_scale
      attn1.norm_q.weight           → attn1.norm_q.weight  (unchanged)
    """
    # Check longest attribute names first to avoid partial suffix matches.
    for attr in _LINEAR_ATTRS:
        suffix = f".linear.{attr}"
        if key.endswith(suffix):
            return key[: -len(suffix)] + f".{attr}"
    if key.endswith(".div.mul_scale"):
        return key[: -len(".div.mul_scale")] + ".mul_scale"
    return key


# ---------------------------------------------------------------------------
# Quantization metadata helpers
# ---------------------------------------------------------------------------

# Known quantized weight types (FLOAT tensors don't determine block type).
_MXFP8_TYPE = "mxfp8"
_MXFP4_DUALSCALE_TYPE = "mxfp4_dualscale"


def _classify_blocks(quant_meta: dict[str, str]) -> dict[int, str]:
    """Classify each transformer block by quantization type from quant_meta.

    Returns a dict mapping block_idx → 'mxfp8' | 'mxfp4_dualscale'.
    A block's type is determined by the first quantized (non-FLOAT) tensor found for it.
    """
    block_types: dict[int, str] = {}
    for key, qtype in quant_meta.items():
        idx = _parse_block_idx(key)
        if idx is None or idx in block_types:
            continue
        if qtype.startswith("W8A8_MXFP8"):
            block_types[idx] = _MXFP8_TYPE
        elif qtype.startswith("W4A4_MXFP4_DUALSCALE"):
            block_types[idx] = _MXFP4_DUALSCALE_TYPE
    return block_types


def _print_block_summary(block_types: dict[int, str]) -> None:
    """Print a compact run-length summary of the block layout."""
    if not block_types:
        print("  Block layout: (empty)")
        return

    sorted_indices = sorted(block_types)
    runs: list[tuple[int, int, str]] = []
    run_start = sorted_indices[0]
    run_type = block_types[run_start]
    for idx in sorted_indices[1:]:
        if block_types[idx] != run_type:
            runs.append((run_start, idx - 1, run_type))
            run_start = idx
            run_type = block_types[idx]
    runs.append((run_start, sorted_indices[-1], run_type))

    print(f"  Block layout ({len(sorted_indices)} blocks classified):")
    for start, end, btype in runs:
        count = end - start + 1
        range_str = f"{start}" if start == end else f"{start}–{end}"
        print(f"    blocks {range_str:>8}: {btype}  ({count} block{'s' if count > 1 else ''})")


# ---------------------------------------------------------------------------
# Safetensors I/O
# ---------------------------------------------------------------------------


def _load_safetensors_dir(directory: pathlib.Path, glob: str = "*.safetensors") -> dict[str, torch.Tensor]:
    candidates = sorted(directory.glob(glob))
    if not candidates:
        raise FileNotFoundError(f"No safetensors matching '{glob}' found in {directory}")
    state: dict[str, torch.Tensor] = {}
    for f in candidates:
        state.update(load_file(str(f)))
    return state


def _load_quant_safetensors(directory: pathlib.Path) -> dict[str, torch.Tensor]:
    try:
        return _load_safetensors_dir(directory, "quant_model_weight*.safetensors")
    except FileNotFoundError:
        return _load_safetensors_dir(directory)


def _load_quant_meta(directory: pathlib.Path) -> dict[str, str]:
    candidates = sorted(directory.glob("quant_model_description*.json"))
    if not candidates:
        print(f"  WARNING: No quant_model_description*.json in {directory}; treating all tensors as FLOAT.")
        return {}
    with open(candidates[0]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-transformer conversion
# ---------------------------------------------------------------------------


def _is_mxfp4_tensor(key: str, qtype: str) -> bool:
    """Return True for MXFP4 quantized tensors and their companion tensors.

    MXFP4 DualScale layers produce four tensor types:
      .linear.weight / .linear.weight_scale / .linear.weight_dual_scale (W4A4_MXFP4_DUALSCALE)
      .linear.bias / .div.mul_scale (FLOAT but companion to a quantized layer)

    BF16 fallback layers have plain .weight / .bias without .linear. / .div. wrappers.
    """
    if qtype.startswith("W4A4_MXFP4_DUALSCALE"):
        return True
    # Companion tensors: bias and mul_scale that belong to an MXFP4 layer.
    return ".linear." in key or ".div.mul_scale" in key


def _diffusers_to_vllm_ignored(diffusers_ignored: list[str]) -> list[str]:
    """Translate diffusers checkpoint prefixes to vllm-omni model parameter names.

    Three transformations align the naming conventions:

    1. Self-attention QKV fusion (attn1 only, not attn2):
       attn1.to_q + attn1.to_k + attn1.to_v → attn1.to_qkv
       Cross-attention (attn2) to_q/k/v are not fused and remain separate.

    2. FFN naming:
       ffn.net.0.proj  → ffn.net_0.proj
       ffn.net.2       → ffn.net_2

    3. to_out index:
       attn1.to_out.0 / attn2.to_out.0  → attn1.to_out / attn2.to_out
    """
    ignored_set = set(diffusers_ignored)
    result: set[str] = set()

    for name in diffusers_ignored:
        m = re.match(r"^(.*\.attn1)\.to_([qkv])$", name)
        if m:
            prefix = m.group(1)
            if all(f"{prefix}.to_{c}" in ignored_set for c in ("q", "k", "v")):
                result.add(f"{prefix}.to_qkv")
            else:
                present = [f"to_{c}" for c in ("q", "k", "v") if f"{prefix}.to_{c}" in ignored_set]
                missing = [f"to_{c}" for c in ("q", "k", "v") if f"{prefix}.to_{c}" not in ignored_set]
                raise ValueError(
                    f"Partial BF16 fallback for '{prefix}': "
                    f"{', '.join(present)} in ignored_layers but {', '.join(missing)} is not. "
                    f"Self-attention Q/K/V are fused into a single to_qkv layer at runtime; "
                    f"all three must share the same precision. "
                    f"Either quantize all of to_q/to_k/to_v or keep all three in BF16."
                )
            continue

        name = re.sub(r"\.ffn\.net\.0\.proj$", ".ffn.net_0.proj", name)
        name = re.sub(r"\.ffn\.net\.2$", ".ffn.net_2", name)
        name = re.sub(r"\.to_out\.0$", ".to_out", name)
        result.add(name)

    return sorted(result)


def _collect_ignored_layers(merged: dict[str, Any], mxfp4_layer_prefixes: set[str]) -> list[str]:
    """Collect vllm-omni parameter name prefixes for BF16 fallback layers.

    A layer is BF16 if it has a .weight tensor but its prefix is not in
    mxfp4_layer_prefixes. Prefixes are returned in vllm-omni parameter naming
    (QKV-fused, FFN underscored, to_out unindexed) so the list can be written
    directly into ignored_layers in config.json without further translation.
    """
    all_weight_prefixes = {key[: -len(".weight")] for key in merged if key.endswith(".weight")}
    return _diffusers_to_vllm_ignored(sorted(all_weight_prefixes - mxfp4_layer_prefixes))


def _convert_transformer(
    quant_subdir: pathlib.Path,
    output_dir: pathlib.Path,
    original_transformer_dir: pathlib.Path,
) -> None:
    """Convert one transformer directory to the mxfp4_dualscale + BF16 mixed format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # BF16 base: provides the scaffold for non-MXFP4 tensors (norms, embeddings,
    # BF16 fallback linear layers that msModelSlim may omit or keep as FLOAT).
    print(f"  Loading BF16 base from {original_transformer_dir} …")
    base_state = _load_safetensors_dir(original_transformer_dir)
    print(f"  {len(base_state)} BF16 tensors loaded")

    print(f"  Loading quantized weights from {quant_subdir} …")
    quant_state = _load_quant_safetensors(quant_subdir)
    quant_meta = _load_quant_meta(quant_subdir)
    print(f"  {len(quant_state)} quant tensors, {len(quant_meta)} meta entries")

    # Classify blocks for a compact summary (informational only in the new scheme).
    block_types = _classify_blocks(quant_meta)
    _print_block_summary(block_types)

    # Remap MXFP4 quantized tensors only.
    # BF16 fallback tensors (FLOAT in quant_meta, no .linear./.div. wrapper) are
    # skipped here; the base_state provides them unchanged.
    remapped: dict[str, torch.Tensor] = {}
    remapped_meta: dict[str, str] = {}
    mxfp4_layer_prefixes: set[str] = set()
    skipped: list[str] = []

    for key, tensor in quant_state.items():
        qtype = quant_meta.get(key, "FLOAT")

        if not _is_mxfp4_tensor(key, qtype):
            continue  # BF16 fallback — base_state already covers this tensor

        renamed = _apply_rename_dict(key)
        final_key = _strip_mxfp4_wrapper(renamed)

        if final_key.endswith(".quant_type"):
            skipped.append(key)
            continue

        remapped[final_key] = tensor
        remapped_meta[final_key] = qtype

        # Track MXFP4 layer prefixes (via their .weight keys) for ignored_layers.
        if final_key.endswith(".weight") and qtype.startswith("W4A4_MXFP4_DUALSCALE"):
            mxfp4_layer_prefixes.add(final_key[: -len(".weight")])

    if skipped:
        print(f"  Skipped {len(skipped)} metadata keys (quant_type markers): {skipped[:5]}")
    print(f"  {len(remapped)} MXFP4 tensors remapped, {len(mxfp4_layer_prefixes)} quantized layers")

    # Merge: base_state provides BF16 scaffold; MXFP4 tensors override their BF16 counterparts
    # and add the new scale tensors (weight_scale, weight_dual_scale, mul_scale).
    merged = {**base_state, **remapped}

    # Determine ignored_layers: BF16 fallback layer prefixes in vllm-omni parameter naming.
    ignored_layers = _collect_ignored_layers(merged, mxfp4_layer_prefixes)
    print(f"  {len(ignored_layers)} BF16 fallback layers → ignored_layers in config.json")

    # Save weights.
    out_weights = output_dir / "diffusion_pytorch_model.safetensors"
    save_file(merged, str(out_weights))
    print(f"  Saved {len(merged)} tensors → {out_weights}")

    # Save remapped quant metadata (for inspection / debugging).
    out_meta_path = output_dir / "quant_model_description.json"
    with open(out_meta_path, "w") as f:
        json.dump(remapped_meta, f, indent=2)

    # Inject quantization_config into config.json.
    src_config = original_transformer_dir / "config.json"
    if src_config.is_file():
        with open(src_config) as f:
            config = json.load(f)
        config["quantization_config"] = _build_quant_config(ignored_layers)
        out_config = output_dir / "config.json"
        with open(out_config, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Injected quantization_config (mxfp4_dualscale, {len(ignored_layers)} ignored_layers) → {out_config}")
    else:
        print(f"  WARNING: No config.json at {src_config}; quantization_config not injected.")


def _build_quant_config(ignored_layers: list[str]) -> dict[str, Any]:
    return {
        "quant_method": "mxfp4_dualscale",
        "is_checkpoint_serialized": True,
        "ignored_layers": ignored_layers,
    }


# ---------------------------------------------------------------------------
# Model-type helpers
# ---------------------------------------------------------------------------


def _get_transformer_dirs(model_type: str) -> list[str]:
    return ["transformer", "transformer_2"] if model_type in CASCADE_MODEL_TYPES else ["transformer"]


def _get_quant_subdir(model_type: str, quant_path: pathlib.Path, transformer_dir: str) -> pathlib.Path:
    if model_type in CASCADE_MODEL_TYPES:
        sub = "high_noise_model" if transformer_dir == "transformer" else "low_noise_model"
        return quant_path / sub
    return quant_path


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

    print(f"Copying original model to {output_path} (skipping {transformer_dirs}) …")
    shutil.copytree(
        str(original_model_path),
        str(output_path),
        ignore=shutil.ignore_patterns(*transformer_dirs),
    )

    for tdir in transformer_dirs:
        q_subdir = _get_quant_subdir(model_type, quant_path, tdir)
        out_tdir = output_path / tdir
        orig_tdir = original_model_path / tdir
        print(f"\nConverting {tdir} (quant source: {q_subdir.name}) …")
        _convert_transformer(q_subdir, out_tdir, orig_tdir)

    print(f"\nDone. Merged model → {output_path}")
    print("\nRun inference (quantization auto-detected from config.json):")
    print(f"  python text_to_video.py --model {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-type", required=True, choices=SUPPORTED_MODEL_TYPES, help="Model variant.")
    parser.add_argument("--original-model", required=True, help="Original HF Diffusers model directory (BF16).")
    parser.add_argument("--quant-path", required=True, help="msModelSlim quantized weights directory.")
    parser.add_argument("--output-path", required=True, help="Output directory for merged model.")
    args = parser.parse_args()

    repack(
        model_type=args.model_type,
        original_model_path=pathlib.Path(args.original_model),
        quant_path=pathlib.Path(args.quant_path),
        output_path=pathlib.Path(args.output_path),
    )


if __name__ == "__main__":
    main()
