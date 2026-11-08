# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CausVid student ``model.pt`` → Diffusers DiT key remap.

Omni's DiT uses Diffusers names (``attn1``, ``norm2``, …); CausVid's checkpoint
uses Wan-native names (``self_attn``, ``norm3``, …). This module renames tensors
so ``CausVidWanPipeline.load_weights`` can overlay the student after the
Diffusers base load. Without it, only schedule defaults apply, not student weights.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch


def _strip_generator_prefix(name: str) -> str:
    for prefix in ("generator.", "model.", "module.", "dit."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def remap_causvid_key(name: str) -> str | None:
    """Map one CausVid key to Diffusers; ``None`` if not part of the DiT."""
    key = _strip_generator_prefix(name)
    if key.startswith("freqs") or key.endswith(".total_ops") or key.endswith(".total_params"):
        return None

    direct = {
        "patch_embedding.weight": "patch_embedding.weight",
        "patch_embedding.bias": "patch_embedding.bias",
        "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
        "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
        "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
        "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
        "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
        "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
        "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
        "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
        "time_projection.1.weight": "condition_embedder.time_proj.weight",
        "time_projection.1.bias": "condition_embedder.time_proj.bias",
        "head.head.weight": "proj_out.weight",
        "head.head.bias": "proj_out.bias",
        "head.norm.weight": "norm_out.norm.weight",
        "head.norm.bias": "norm_out.norm.bias",
        "head.modulation": "scale_shift_table",
    }
    if key in direct:
        return direct[key]

    if not key.startswith("blocks."):
        return None

    parts = key.split(".", 2)
    if len(parts) < 3:
        return None
    block_id, rest = parts[1], parts[2]
    prefix = f"blocks.{block_id}."

    block_map = {
        "self_attn.q.weight": "attn1.to_q.weight",
        "self_attn.q.bias": "attn1.to_q.bias",
        "self_attn.k.weight": "attn1.to_k.weight",
        "self_attn.k.bias": "attn1.to_k.bias",
        "self_attn.v.weight": "attn1.to_v.weight",
        "self_attn.v.bias": "attn1.to_v.bias",
        "self_attn.o.weight": "attn1.to_out.0.weight",
        "self_attn.o.bias": "attn1.to_out.0.bias",
        "self_attn.norm_q.weight": "attn1.norm_q.weight",
        "self_attn.norm_k.weight": "attn1.norm_k.weight",
        "cross_attn.q.weight": "attn2.to_q.weight",
        "cross_attn.q.bias": "attn2.to_q.bias",
        "cross_attn.k.weight": "attn2.to_k.weight",
        "cross_attn.k.bias": "attn2.to_k.bias",
        "cross_attn.v.weight": "attn2.to_v.weight",
        "cross_attn.v.bias": "attn2.to_v.bias",
        "cross_attn.o.weight": "attn2.to_out.0.weight",
        "cross_attn.o.bias": "attn2.to_out.0.bias",
        "cross_attn.norm_q.weight": "attn2.norm_q.weight",
        "cross_attn.norm_k.weight": "attn2.norm_k.weight",
        "ffn.0.weight": "ffn.net.0.proj.weight",
        "ffn.0.bias": "ffn.net.0.proj.bias",
        "ffn.2.weight": "ffn.net.2.weight",
        "ffn.2.bias": "ffn.net.2.bias",
        "norm3.weight": "norm2.weight",  # Wan-native affine cross-attn LN
        "norm3.bias": "norm2.bias",
        "modulation": "scale_shift_table",
    }
    mapped = block_map.get(rest)
    if mapped is None:
        return None
    return prefix + mapped


def iter_causvid_diffusers_weights(
    state: Mapping[str, Any] | Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield remapped ``(diffusers_name, tensor)`` pairs."""
    if isinstance(state, Mapping):
        items: Iterable[tuple[str, Any]] = state.items()
    else:
        items = state
    for name, tensor in items:
        if not isinstance(tensor, torch.Tensor):
            continue
        mapped = remap_causvid_key(str(name))
        if mapped is None:
            continue
        yield mapped, tensor


def load_causvid_generator_state(path: str | Path) -> dict[str, torch.Tensor]:
    """Load CausVid ``model.pt`` and return the ``generator`` tensors."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "generator" in ckpt:
        raw = ckpt["generator"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        raw = ckpt
    else:
        raise ValueError(f"CausVid checkpoint {path} must be a state dict or contain a 'generator' entry")
    if not isinstance(raw, dict):
        raise TypeError(f"CausVid generator must be a dict, got {type(raw)}")
    return {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}


def causvid_pt_to_diffusers_state(path: str | Path) -> dict[str, torch.Tensor]:
    """Remap a CausVid checkpoint to a Diffusers state dict."""
    raw = load_causvid_generator_state(path)
    out: dict[str, torch.Tensor] = {}
    skipped = 0
    for src, tensor in raw.items():
        mapped = remap_causvid_key(src)
        if mapped is None:
            skipped += 1
            continue
        out[mapped] = tensor
    if not out:
        raise RuntimeError(f"No remappable tensors in CausVid checkpoint {path} (skipped={skipped})")
    return out
