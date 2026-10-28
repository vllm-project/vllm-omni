# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Portable readers and native-layout conversion for standard H3 Lite weights."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class H3LiteDecoderCheckpoint:
    config: dict[str, object]
    kept_layers: tuple[int, ...]
    state_dict: dict[str, torch.Tensor]


def resolve_h3_lightvae_decoder_path(additional: Mapping[str, object]) -> str | None:
    if "lightvae_decoder_path" not in additional:
        return None
    value = additional["lightvae_decoder_path"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("lightvae_decoder_path must be a nonempty local file path")
    if additional.get("video_vae_decoder") is not None:
        raise ValueError("lightvae_decoder_path cannot be combined with video_vae_decoder")
    return value


def _validate_config(config: dict[str, object]) -> tuple[int, ...]:
    if config.get("family", "h3") != "h3":
        raise ValueError("LightVAE family must be h3")
    for field in (
        "latent_channels",
        "decoder_num_layers",
        "decoder_num_attention_heads",
        "decoder_attention_head_dim",
    ):
        value = config.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"LightVAE {field} must be a positive integer")
    kept = config.get("kept_layers")
    depth = config["decoder_num_layers"]
    if (
        not isinstance(kept, list)
        or not kept
        or any(type(index) is not int for index in kept)
        or kept != sorted(set(kept))
        or kept[0] < 0
        or kept[-1] >= depth
    ):
        raise ValueError("LightVAE kept_layers must be ordered unique teacher indices")
    for field in ("latents_mean", "latents_std"):
        values = config.get(field)
        if (
            not isinstance(values, list)
            or len(values) != config["latent_channels"]
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in values)
            or (field == "latents_std" and any(x <= 0 for x in values))
        ):
            raise ValueError(f"LightVAE {field} does not match latent_channels")
    for field, expected in (("clip_length", 17), ("token_drop", 3)):
        if config.get(field) != expected:
            raise ValueError(f"LightVAE {field} must be {expected}")
    return tuple(kept)


def read_h3_lite_decoder_checkpoint(path: str | Path) -> H3LiteDecoderCheckpoint:
    """Read architecture and compact decoder tensors on the CPU."""
    path = Path(path)
    if path.suffix != ".safetensors":
        raise ValueError(f"{path}: expected a standard LightVAE safetensors file")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        try:
            architecture = json.loads((handle.metadata() or {})["architecture"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: missing or invalid architecture metadata") from exc
        if not isinstance(architecture, dict):
            raise ValueError(f"{path}: architecture must be an object")
        config = architecture.get("teacher_config", architecture)
        if not isinstance(config, dict):
            raise ValueError(f"{path}: architecture.teacher_config must be an object")
        config = dict(config)
        config["kept_layers"] = architecture.get("kept_layers")
        kept = _validate_config(config)
        state = {}
        for name in handle.keys():
            if name.startswith(("decoder.", "post_quant_conv.")):
                tensor = handle.get_tensor(name)
                if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise ValueError(f"{path}: unsupported or quantized tensor {name}")
                state[name] = tensor
    indices = {
        int(name.split(".")[2])
        for name in state
        if name.startswith("decoder.transformer_blocks.") and name.split(".")[2].isdigit()
    }
    if indices != set(range(len(kept))):
        raise ValueError(f"{path}: decoder.transformer_blocks must use compact indices")
    return H3LiteDecoderCheckpoint(config, kept, state)


def convert_h3_lite_decoder_state_dict(
    state: Mapping[str, torch.Tensor],
    *,
    num_heads: int,
) -> dict[str, torch.Tensor]:
    """Map Diffusers H3 tensors to the supported native H3 decoder layout."""
    if type(num_heads) is not int or num_heads <= 0:
        raise ValueError("num_heads must be a positive integer")
    converted = {}
    projections: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in state.items():
        parts = name.split(".")
        target = name
        if name.startswith("decoder.transformer_blocks.") and len(parts) >= 4 and parts[2].isdigit():
            suffix = ".".join(parts[3:])
            if len(parts) == 6 and parts[3] == "attn" and parts[4] in ("to_q", "to_k", "to_v"):
                if parts[5] not in ("weight", "bias"):
                    raise ValueError(f"unsupported decoder parameter {name}")
                projection_key = ".".join(parts[:3]) + f".attn.to_qkv.{parts[5]}"
                projections.setdefault(projection_key, {})[parts[4]] = tensor
                continue
            if suffix in ("ff.net.0.proj.weight", "ff.net.0.proj.bias"):
                if tensor.ndim not in (1, 2) or tensor.shape[0] % 2:
                    raise ValueError(f"invalid MLP projection {name}")
                up, gate = tensor.chunk(2, dim=0)
                tensor = torch.cat((gate, up), dim=0)
                target = name.replace(".ff.net.0.proj.", ".ff.w1.")
            elif suffix in ("ff.net.2.weight", "ff.net.2.bias"):
                target = name.replace(".ff.net.2.", ".ff.w2.")
            elif suffix in ("attn.to_out.0.weight", "attn.to_out.0.bias"):
                target = name.replace(".attn.to_out.0.", ".attn.to_out.")
            elif suffix not in ("norm1.weight", "norm2.weight", "scale1", "scale2"):
                raise ValueError(f"unsupported decoder parameter {name}")
        elif name in ("decoder.proj_in.weight", "decoder.proj_in.bias"):
            target = name.replace("decoder.proj_in.", "decoder.x_embedder.")
        elif name not in (
            "decoder.register_tokens",
            "decoder.norm_out.weight",
            "decoder.norm_out.bias",
            "decoder.proj_out.weight",
            "decoder.proj_out.bias",
            "post_quant_conv.weight",
            "post_quant_conv.bias",
        ):
            raise ValueError(f"unsupported decoder parameter {name}")
        converted[target] = tensor
    for target, group in projections.items():
        if set(group) != {"to_q", "to_k", "to_v"}:
            raise ValueError(f"{target}: missing to_q, to_k or to_v")
        tensors = [group[key] for key in ("to_q", "to_k", "to_v")]
        first = tensors[0]
        expected_ndim = 2 if target.endswith("weight") else 1
        if (
            first.ndim != expected_ndim
            or first.shape[0] % num_heads
            or any(t.shape != first.shape or t.dtype != first.dtype for t in tensors)
        ):
            raise ValueError(f"{target}: incompatible to_q/to_k/to_v shapes")
        packed = torch.stack([t.reshape(num_heads, -1, *t.shape[1:]) for t in tensors], dim=1)
        converted[target] = packed.reshape(3 * first.shape[0], *first.shape[1:])
    return converted
