#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-model scaling_layers maps for Quark SmoothQuant (and, later, rotation).

The ONLY per-model artifact needed to cover a new DiT with the offline Quark MXFP4
flow. The export driver (quantize_wan2_2_quark_mxfp4.py) and the vllm-omni offline
loader (ROCmMxfp4OfflineLinearMethod) are model-agnostic; adding a model = add one
entry to SCALING_MAPS + DECODER_LAYERS_ATTR here.

A scaling_layers entry (schema: quark/torch/algorithm/utils/prepare.py) needs:
  prev_op        : module whose OUTPUT weight absorbs the (inverse) scale
  layers         : the linear(s) whose INPUT is scaled (fused on input dim)
  inp            : REQUIRED - the input-feature key (module name) whose captured
                   activation feeds `layers`; usually the first entry of `layers`.
  module2inspect : optional submodule run for the scale search (e.g. attn/ffn).

Structure derived from vllm_omni/diffusion/models/*/*_transformer.py.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Wan2.2 (WanTransformer3DModel). The A14B cascade uses TWO of these
# (transformer + transformer_2); the map applies to each identically.
# WanTransformerBlock: norm1->attn1(self), norm2->attn2(cross), norm3->ffn.
# ---------------------------------------------------------------------------
WAN_MAP = [
    {"prev_op": "norm1", "layers": ["attn1.to_qkv"], "inp": "attn1.to_qkv", "module2inspect": "attn1"},
    {"prev_op": "attn1.to_qkv", "layers": ["attn1.to_out"], "inp": "attn1.to_out"},
    {"prev_op": "norm2", "layers": ["attn2.to_q"], "inp": "attn2.to_q", "module2inspect": "attn2"},
    {"prev_op": "attn2.to_v", "layers": ["attn2.to_out"], "inp": "attn2.to_out"},
    {"prev_op": "norm3", "layers": ["ffn.net_0"], "inp": "ffn.net_0", "module2inspect": "ffn"},
    {"prev_op": "ffn.net_0", "layers": ["ffn.net_2"], "inp": "ffn.net_2"},
]

# I2V variant adds encoder K/V projections; appended when i2v=True.
WAN_MAP_I2V_EXTRA = [
    {"prev_op": "norm2", "layers": ["attn2.add_k_proj", "attn2.add_v_proj"], "inp": "attn2.add_k_proj"},
]

# R2 rotation (offline, folds into v_proj/o_proj within each block; no runtime op).
# RotationProcessor.get_scaling_layers requires scaling_layers as a dict keyed
# first_layer/middle_layers/last_layer even for R2-only. With r1=False the entries
# only need target_modules (the prev/norm/next requirements are R1-only). The actual
# R2 fold uses the v_proj/o_proj fields below, rotating attention head_dim channels.
_WAN_R2_TARGETS = [{"target_modules": ["blocks.layer_id.attn1.to_v", "blocks.layer_id.attn1.to_out.0"]}]
WAN_R2_ROTATION = {
    "scaling_layers": {"first_layer": _WAN_R2_TARGETS, "middle_layers": _WAN_R2_TARGETS, "last_layer": _WAN_R2_TARGETS},
    "v_proj": "attn1.to_v",
    "o_proj": "attn1.to_out.0",
    "self_attn": "attn1",
    "mlp": "ffn",
}

# Flux - PLACEHOLDER, filled in during the Flux pass. Two block types (dual-stream
# + single-stream) plus a context branch (add_kv_proj / to_add_out) Wan lacks.
FLUX_MAP: list = []  # TODO(flux)

SCALING_MAPS = {
    "WanTransformer3DModel": WAN_MAP,
    "FluxTransformer2DModel": FLUX_MAP,  # placeholder
}

# Structural linears that must stay bf16 (embedders / final projection). These are
# not wrapped as quantizable linears in the vllm-omni model, so quantizing them
# produces packed keys the loader cannot place. Passed to QConfig.exclude and to the
# checkpoint's ignored_layers.
EXCLUDE_MAPS = {
    "WanTransformer3DModel": [
        "*time_embedder*",
        "*time_proj*",
        "*text_embedder*",
        "*condition_embedder*",
        "*patch_embedding*",
        "*norm_out*",
        "*proj_out*",
    ],
}

# R2 rotation configs (dict-form scaling_layers + v/o proj fields), per model.
ROTATION_MAPS = {
    "WanTransformer3DModel": WAN_R2_ROTATION,
}

# Attribute path to the ModuleList of transformer blocks. Quark's processors need
# this (model_decoder_layers); diffusers DiTs are not decoder-style (no model.layers).
DECODER_LAYERS_ATTR = {
    "WanTransformer3DModel": "blocks",
    "FluxTransformer2DModel": "transformer_blocks",  # placeholder (verify at Flux pass)
}


def get_decoder_layers_attr(model) -> str:
    name = type(model).__name__
    if name not in DECODER_LAYERS_ATTR:
        raise NotImplementedError(f"No decoder-layers attr for {name!r}. Add it to DECODER_LAYERS_ATTR.")
    return DECODER_LAYERS_ATTR[name]


def get_exclude_patterns(model) -> list:
    """Structural linears (embedders/proj_out) that must stay bf16, or [] if none."""
    return list(EXCLUDE_MAPS.get(type(model).__name__, []))


def get_rotation_map(model) -> dict:
    """Return the R2 rotation config for a diffusion transformer instance."""
    name = type(model).__name__
    if name not in ROTATION_MAPS:
        raise NotImplementedError(
            f"No Quark R2 rotation map for {name!r}. Add an entry to ROTATION_MAPS. Known: {list(ROTATION_MAPS)}"
        )
    return ROTATION_MAPS[name]


def shim_rotation_config(cfg) -> None:
    """Add the config attrs Quark's rotation processor reads (diffusers DiT configs
    lack the HF-LLM names). R2 needs head_dim + num_hidden_layers; hidden_size is a
    fallback for head_dim."""
    for name, value in (
        ("head_dim", getattr(cfg, "attention_head_dim", None)),
        ("num_hidden_layers", getattr(cfg, "num_layers", None)),
        ("hidden_size", (getattr(cfg, "num_attention_heads", 0) or 0) * (getattr(cfg, "attention_head_dim", 0) or 0)),
    ):
        if getattr(cfg, name, None) is None and value:
            setattr(cfg, name, value)
    if getattr(cfg, "intermediate_size", None) is None and getattr(cfg, "ffn_dim", None):
        cfg.intermediate_size = cfg.ffn_dim


def get_scaling_map(model, i2v: bool = False) -> list:
    """Return the scaling_layers map for a diffusion transformer instance.

    Raises NotImplementedError (not KeyError) for unmapped/placeholder models so the
    export script fails loudly with an actionable message.
    """
    name = type(model).__name__
    if name not in SCALING_MAPS:
        raise NotImplementedError(
            f"No Quark scaling_layers map for {name!r}. Add an entry to SCALING_MAPS. Known: {list(SCALING_MAPS)}"
        )
    m = list(SCALING_MAPS[name])
    if name == "WanTransformer3DModel" and i2v:
        m = m + WAN_MAP_I2V_EXTRA
    if not m:
        raise NotImplementedError(f"Scaling map for {name!r} is a placeholder (empty). Fill it before exporting.")
    return m
