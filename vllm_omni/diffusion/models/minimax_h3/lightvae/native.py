# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Install standard Lite weights into the supported native H3 VAE structure."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import torch
from torch import nn

from .checkpoint import H3LiteDecoderCheckpoint, convert_h3_lite_decoder_state_dict


def install_h3_lite_decoder(
    model: nn.Module,
    checkpoint: H3LiteDecoderCheckpoint,
    *,
    native_config: Mapping[str, object],
) -> None:
    """Validate and replace decoder weights before residency hooks are built."""
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("H3 Lite decoder installation must happen on the CPU")
    decoder = getattr(model, "decoder", None)
    blocks = getattr(decoder, "transformer_blocks", None)
    post_quant = getattr(model, "post_quant_conv", None)
    if (
        not isinstance(decoder, nn.Module)
        or not isinstance(blocks, nn.ModuleList)
        or not isinstance(post_quant, nn.Module)
        or not isinstance(getattr(decoder, "x_embedder", None), nn.Linear)
        or len(blocks) != checkpoint.config["decoder_num_layers"]
    ):
        raise ValueError("unsupported native H3 decoder structure or teacher depth")
    for field in ("latent_channels", "latents_mean", "latents_std"):
        if native_config.get(field) != checkpoint.config[field]:
            raise ValueError(f"H3 Lite configuration disagrees on {field}")
    for field in ("clip_length", "token_drop"):
        actual = native_config.get(f"vae_{field}", native_config.get(field))
        if actual != checkpoint.config[field]:
            raise ValueError(f"H3 Lite configuration disagrees on {field}")
    heads = cast(int, checkpoint.config["decoder_num_attention_heads"])
    head_dim = cast(int, checkpoint.config["decoder_attention_head_dim"])
    hidden = heads * head_dim
    for field in ("decoder_norm_eps", "decoder_rope_theta", "decoder_rope_dim_ratio"):
        value = checkpoint.config.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            or (field in native_config and native_config[field] != value)
        ):
            raise ValueError(f"H3 Lite configuration disagrees on {field}")
    eps = checkpoint.config["decoder_norm_eps"]
    if getattr(getattr(decoder, "norm_out", None), "eps", None) != eps:
        raise ValueError("H3 Lite decoder_norm_eps disagrees with norm_out")
    ratio = cast(float, checkpoint.config["decoder_rope_dim_ratio"])
    rotary_dim = int(head_dim * ratio)
    if ratio > 1 or rotary_dim <= 0 or rotary_dim % 6:
        raise ValueError("unsupported H3 decoder_rope_dim_ratio")
    pos_embed = getattr(decoder, "pos_embed", None)
    inv_freq = getattr(pos_embed, "inv_freq", None)
    expected_freq = 1.0 / cast(float, checkpoint.config["decoder_rope_theta"]) ** torch.arange(
        0, 1, 6 / rotary_dim, dtype=torch.float32, device="cpu"
    )
    if (
        getattr(pos_embed, "n_dim", None) != 3
        or getattr(pos_embed, "angle_scale", None) != 2 * math.pi
        or not isinstance(inv_freq, torch.Tensor)
        or inv_freq.device.type != "cpu"
        or inv_freq.shape != expected_freq.shape
        or not torch.allclose(inv_freq.float(), expected_freq, rtol=1e-6, atol=0)
    ):
        raise ValueError("unsupported H3 decoder rope frequencies")
    if decoder.x_embedder.in_features != checkpoint.config["latent_channels"]:
        raise ValueError("H3 Lite x_embedder latent_channels mismatch")
    for block in blocks:
        attention = getattr(block, "attn", None)
        for owner, name in (
            (block, "norm1"),
            (block, "norm2"),
            (attention, "norm_q"),
            (attention, "norm_k"),
        ):
            if getattr(getattr(owner, name, None), "eps", None) != eps:
                raise ValueError(f"H3 Lite decoder_norm_eps disagrees with {name}")
        qkv = getattr(attention, "to_qkv", None)
        if (
            not isinstance(qkv, nn.Linear)
            or getattr(attention, "dim_head", None) != head_dim
            or qkv.in_features != hidden
            or qkv.out_features != 3 * hidden
        ):
            raise ValueError("unsupported H3 attention structure or head dimensions")
    converted = convert_h3_lite_decoder_state_dict(checkpoint.state_dict, num_heads=heads)
    expected = {}
    for name, tensor in decoder.state_dict().items():
        if name.startswith("transformer_blocks."):
            parts = name.split(".")
            teacher_index = int(parts[1])
            if teacher_index not in checkpoint.kept_layers:
                continue
            parts[1] = str(checkpoint.kept_layers.index(teacher_index))
            name = ".".join(parts)
        expected[f"decoder.{name}"] = tensor
    expected.update({f"post_quant_conv.{key}": value for key, value in post_quant.state_dict().items()})
    if "decoder.mask_token" in expected:
        mask = expected["decoder.mask_token"]
        if tuple(mask.shape) != (1, 1, hidden):
            raise ValueError("unsupported native H3 training mask_token shape")
        # The native implementation retains this unused training parameter.
        converted["decoder.mask_token"] = torch.zeros_like(mask)
    missing, unexpected = (
        expected.keys() - converted.keys(),
        converted.keys() - expected.keys(),
    )
    if missing or unexpected:
        raise ValueError(
            f"H3 Lite decoder parameter mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for name, tensor in converted.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(
                f"H3 Lite tensor shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(expected[name].shape)}"
            )
    decoder.transformer_blocks = nn.ModuleList([blocks[index] for index in checkpoint.kept_layers])
    decoder.to(dtype=torch.float32)
    post_quant.to(dtype=torch.float32)
    decoder.load_state_dict(
        {key.removeprefix("decoder."): value for key, value in converted.items() if key.startswith("decoder.")},
        strict=True,
    )
    post_quant.load_state_dict(
        {
            key.removeprefix("post_quant_conv."): value
            for key, value in converted.items()
            if key.startswith("post_quant_conv.")
        },
        strict=True,
    )
