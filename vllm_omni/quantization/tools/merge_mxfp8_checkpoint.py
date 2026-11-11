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
  - MiniMax-H3        (FL2VA partition, transformer at FL2VA/transformer)

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

import torch
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

SUPPORTED_MODEL_TYPES = ["Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B", "Wan2.2-TI2V-5B", "MiniMax-H3"]

# MoE cascade models have high_noise_model + low_noise_model
CASCADE_MODEL_TYPES = {"Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B"}

# MiniMax-H3 keeps the DiT under the FL2VA partition; the other FL2VA
# components (text encoder, VAEs, processor, tokenizer) are reused verbatim
# and only referenced via symlinks so the merge does not copy ~73 GiB.
MINIMAX_H3_TRANSFORMER_DIR = "FL2VA/transformer"
MINIMAX_H3_SYMLINK_DIRS = ("text_encoder", "video_vae", "audio_vae", "processor", "tokenizer")

# Wan checkpoints rename tensors before the runtime sees them (e.g.
# ``ffn.0.weight`` becomes ``ffn.net.0.proj.weight``), and the model's weight
# loader then translates those diffusers-style names into the runtime parameter
# names it actually constructs (``ffn.net_0.proj``, ``ffn.net_2``, ``to_out``).
# Quantization methods are selected *before* that loader runs, so
# ``ignored_layers`` has to carry the runtime spelling already, otherwise the
# layer is not skipped and a BF16 tensor is fed to the MXFP8 loader.
RUNTIME_LAYER_NAME_FIXUPS: dict[str, tuple[tuple[str, str], ...]] = {
    "Wan2.2": (
        (".ffn.net.0.", ".ffn.net_0."),
        (".ffn.net.2", ".ffn.net_2"),
        (".to_out.0", ".to_out"),
    ),
}


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
    if model_type == "MiniMax-H3":
        return [MINIMAX_H3_TRANSFORMER_DIR]
    return ["transformer"]


def _get_rename_dict(model_type: str) -> dict[str, str]:
    """Return the msModelSlim→runtime rename table for a model type.

    MiniMax-H3 shares the vLLM-Omni checkpoint names directly; it only needs
    the generic ``.div.mul_scale`` / ``.linear.`` unwrap performed by
    ``_remap_keys``, so its rename table is empty.
    """
    if model_type == "MiniMax-H3":
        return {}
    return TRANSFORMER_KEYS_RENAME_DICT


def _get_quant_subdir(model_type: str, quant_path: pathlib.Path, transformer_dir: str) -> pathlib.Path:
    if model_type in CASCADE_MODEL_TYPES:
        sub = "high_noise_model" if transformer_dir == "transformer" else "low_noise_model"
        return quant_path / sub
    return quant_path


def _remap_keys(state_dict: dict, quant_meta: dict, rename_dict: dict[str, str] | None = None) -> tuple[dict, dict]:
    """Apply the model rename table to both state_dict and quant_meta."""
    if rename_dict is None:
        rename_dict = TRANSFORMER_KEYS_RENAME_DICT
    renamed_state_keys: dict[str, str] = {}
    for key in state_dict:
        new_key = key
        for src, dst in rename_dict.items():
            new_key = new_key.replace(src, dst)
        renamed_state_keys[key] = new_key

    # msModelSlim wraps a smoothed linear module as ``<prefix>.linear`` and
    # writes the multiplier beside it as ``<prefix>.div.mul_scale``. Only
    # unwrap prefixes that have this marker, so ordinary ``.linear.`` names
    # remain unchanged.
    smooth_prefixes = {
        new_key[: -len(".div.mul_scale")]
        for new_key in renamed_state_keys.values()
        if new_key.endswith(".div.mul_scale")
    }

    def normalize_key(new_key: str) -> str:
        # msModelSlim represents SmoothQuant metadata as a sibling ``div``
        # module. Runtime linear layers consume it as ``mul_scale``.
        if new_key.endswith(".div.mul_scale"):
            new_key = new_key[: -len(".div.mul_scale")] + ".mul_scale"

        for prefix in smooth_prefixes:
            wrapper = f"{prefix}.linear."
            if new_key.startswith(wrapper):
                return prefix + "." + new_key[len(wrapper) :]
        return new_key

    new_state: dict = {}
    new_meta: dict = {}
    for key, tensor in state_dict.items():
        new_key = normalize_key(renamed_state_keys[key])
        new_state[new_key] = tensor
        if key in quant_meta:
            new_meta[new_key] = quant_meta[key]

    return new_state, new_meta


def _has_complete_mxfp8_smooth_scales(state_dict: dict[str, object], quant_meta: dict[str, object]) -> bool:
    """Return whether every serialized MXFP8 weight has a paired smooth scale."""
    mxfp8_layers = {
        key[: -len(".weight")]
        for key, qtype in quant_meta.items()
        if key.endswith(".weight") and isinstance(qtype, str) and qtype.startswith("W8A8_MXFP8")
    }
    return bool(mxfp8_layers) and all(f"{name}.mul_scale" in state_dict for name in mxfp8_layers)


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


def _reverse_qkv_grouped_reorder(
    state_dict: dict,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> dict:
    """Reverse vLLM-Omni's grouped-QKV -> QKV reorder for MiniMax-H3.

    msModelSlim captures the vLLM-Omni loaded model, whose fused QKV weights are
    already in the ``(Q | K | V)`` row order.  The native H3 checkpoint stores
    them grouped per query group (``(q,k,v)`` per group) and omni's load_weights
    reorders them again.  Convert the captured tensors back to the grouped
    layout before overlaying onto the BF16 base so load_weights restores the
    correct order.  Applies to weight and per-row weight_scale (NOT the
    per-input-channel mul_scale).
    """
    for name in list(state_dict):
        if not (name.endswith(".attn.qkv_proj.weight") or name.endswith(".attn.qkv_proj.weight_scale")):
            continue
        weight = state_dict[name]
        per_group = (heads_per_group + 2) * head_dim
        if weight.shape[0] != num_query_groups * per_group:
            continue
        rest = weight.shape[1:]
        q, k, v = torch.split(
            weight,
            [
                num_query_groups * heads_per_group * head_dim,
                num_query_groups * head_dim,
                num_query_groups * head_dim,
            ],
            dim=0,
        )
        q = q.reshape(num_query_groups, heads_per_group * head_dim, *rest)
        k = k.reshape(num_query_groups, head_dim, *rest)
        v = v.reshape(num_query_groups, head_dim, *rest)
        state_dict[name] = torch.cat([q, k, v], dim=1).reshape(-1, *rest).contiguous()
    return state_dict


def _runtime_ignored_layer_name(module_name: str, model_type: str) -> str:
    """Translate a checkpoint module name into the runtime layer prefix.

    ``ignored_layers`` is matched against the prefixes the model builds, not
    against checkpoint tensor names. Model families whose loader renames layers
    declare those renames here; unknown families pass through unchanged.
    """
    for family, fixups in RUNTIME_LAYER_NAME_FIXUPS.items():
        if not model_type.startswith(family):
            continue
        for checkpoint_form, runtime_form in fixups:
            module_name = module_name.replace(checkpoint_form, runtime_form)
    return module_name


def _convert_transformer(
    model_type: str,
    quant_subdir: pathlib.Path,
    output_dir: pathlib.Path,
    original_transformer_dir: pathlib.Path,
    rename_dict: dict[str, str] | None = None,
) -> None:
    if rename_dict is None:
        rename_dict = _get_rename_dict(model_type)
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

    # Rename keys from msModelSlim convention to runtime convention
    quant_state_dict, quant_meta = _remap_keys(quant_state_dict, quant_meta, rename_dict)

    # MiniMax-H3: undo the fused-QKV reorder that omni applies at load time.
    # The modelslim export captures the already-reordered (Q|K|V) layout, so it
    # must be converted back to the checkpoint's grouped layout before the
    # overlay; omni's load_weights then reorders it to the runtime layout.
    if model_type == "MiniMax-H3":
        src_config = original_transformer_dir / "config.json"
        if src_config.is_file():
            with open(src_config) as f:
                arch_cfg = json.load(f)
            num_heads = int(arch_cfg.get("num_attention_heads", 0))
            head_dim = int(arch_cfg.get("attention_head_dim", 0))
            if num_heads and head_dim:
                quant_state_dict = _reverse_qkv_grouped_reorder(
                    quant_state_dict,
                    num_query_groups=num_heads,
                    heads_per_group=1,
                    head_dim=head_dim,
                )

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
        quant_config = dict(MXFP8_QUANT_CONFIG)
        ignored_layers = sorted(
            {
                _runtime_ignored_layer_name(key[: -len(".weight")], model_type)
                for key, qtype in quant_meta.items()
                if key.endswith(".weight") and qtype == "FLOAT" and quant_state_dict[key].ndim == 2
            }
        )
        if ignored_layers:
            quant_config["ignored_layers"] = ignored_layers
        if _has_complete_mxfp8_smooth_scales(quant_state_dict, quant_meta):
            quant_config["require_smooth_scale"] = True
        config["quantization_config"] = quant_config
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Injected quantization_config into config.json → {output_dir / 'config.json'}")
    else:
        print(f"  WARNING: No config.json found at {src_config}")


# ---------------------------------------------------------------------------
# Main repack
# ---------------------------------------------------------------------------


def _repack_minimax_h3(
    original_model_path: pathlib.Path,
    quant_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    """Repack MiniMax-H3 FL2VA: reuse untouched components via symlinks.

    Only the DiT transformer is materialized (BF16 base overlaid with the
    msModelSlim MXFP8 tensors); text encoder, VAEs, processor and tokenizer are
    symlinked so the merge avoids copying ~73 GiB of untouched weights.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original_model_path / "model_index.json", output_path / "model_index.json")

    # Resolve the source so every symlink target is absolute. A relative target
    # is stored verbatim and then interpreted against the link's own parent
    # (output/FL2VA/), which leaves all five component links dangling when the
    # caller passes a relative --original-model.
    fl2va_src = (original_model_path / "FL2VA").resolve()
    fl2va_out = output_path / "FL2VA"
    fl2va_out.mkdir(exist_ok=True)
    shutil.copy2(fl2va_src / "model_index.json", fl2va_out / "model_index.json")

    for sub in MINIMAX_H3_SYMLINK_DIRS:
        dst = fl2va_out / sub
        if not (dst.exists() or dst.is_symlink()):
            dst.symlink_to(fl2va_src / sub, target_is_directory=True)

    _convert_transformer(
        model_type="MiniMax-H3",
        quant_subdir=quant_path,
        output_dir=fl2va_out / "transformer",
        original_transformer_dir=fl2va_src / "transformer",
        rename_dict={},
    )


def repack(
    model_type: str,
    original_model_path: pathlib.Path,
    quant_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    # Normalise to absolute paths so relative CLI arguments cannot leak into
    # generated symlink targets or loader paths.
    original_model_path = original_model_path.resolve()
    quant_path = quant_path.resolve()

    if model_type == "MiniMax-H3":
        _repack_minimax_h3(original_model_path, quant_path, output_path)
        return

    transformer_dirs = _get_transformer_dirs(model_type)

    # Step 1: Copy original model, skipping transformer dirs (will be replaced).
    print(f"Copying original model to {output_path} (skipping {transformer_dirs}) …")
    shutil.copytree(
        str(original_model_path),
        str(output_path),
        ignore=shutil.ignore_patterns(*transformer_dirs),
    )

    # Step 2+: Convert each transformer.
    rename_dict = _get_rename_dict(model_type)
    for tdir in transformer_dirs:
        q_subdir = _get_quant_subdir(model_type, quant_path, tdir)
        out_tdir = output_path / tdir
        orig_tdir = original_model_path / tdir
        print(f"\nConverting {tdir} (quant source: {q_subdir.name}) …")
        _convert_transformer(model_type, q_subdir, out_tdir, orig_tdir, rename_dict)

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
