# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SandAI. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native Stable Audio Open decoder used by MAGI-2.

The MAGI checkpoint embeds Stable Audio's Oobleck VAE under the
``pretransform.model`` prefix and uses the sequential module names from
``stable-audio-tools``. This module converts those names to Diffusers'
``AutoencoderOobleck`` layout and loads only the decoder tensors.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from diffusers import AutoencoderOobleck
from safetensors import safe_open

_CHECKPOINT_PREFIX = "pretransform.model.decoder."
_BLOCK_COMPONENTS = {
    0: "snake1",
    1: "conv_t1",
}
_RESIDUAL_COMPONENTS = {
    0: "snake1",
    1: "conv1",
    2: "snake2",
    3: "conv2",
}


def convert_stable_audio_decoder_key(key: str) -> str | None:
    """Map one Stable Audio decoder key to Diffusers' Oobleck decoder.

    Unrelated checkpoint tensors return ``None``. A key inside the decoder
    namespace that does not match the released Stable Audio layout raises so a
    changed checkpoint cannot be loaded partially by accident.
    """

    if not key.startswith(_CHECKPOINT_PREFIX):
        return None
    suffix = key[len(_CHECKPOINT_PREFIX) :]
    parts = suffix.split(".")
    if len(parts) < 3 or parts[0] != "layers":
        raise ValueError(f"Unsupported Stable Audio decoder key: {key}")

    try:
        layer = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Unsupported Stable Audio decoder key: {key}") from exc
    tail = parts[2:]

    if layer == 0 and len(tail) == 1:
        return f"conv1.{tail[0]}"
    if 1 <= layer <= 5:
        block = layer - 1
        if len(tail) == 3 and tail[0] == "layers":
            try:
                component = _BLOCK_COMPONENTS[int(tail[1])]
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Unsupported Stable Audio decoder key: {key}") from exc
            return f"block.{block}.{component}.{tail[2]}"
        if len(tail) == 5 and tail[0] == "layers" and tail[2] == "layers":
            try:
                residual = int(tail[1]) - 1
                component = _RESIDUAL_COMPONENTS[int(tail[3])]
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Unsupported Stable Audio decoder key: {key}") from exc
            if residual not in (1, 2, 3):
                raise ValueError(f"Unsupported Stable Audio decoder key: {key}")
            return f"block.{block}.res_unit{residual}.{component}.{tail[4]}"
    if layer == 6 and len(tail) == 1:
        return f"snake1.{tail[0]}"
    if layer == 7 and len(tail) == 1:
        return f"conv2.{tail[0]}"
    raise ValueError(f"Unsupported Stable Audio decoder key: {key}")


def convert_stable_audio_decoder_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert and filter a Stable Audio checkpoint state dictionary."""

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        target_key = convert_stable_audio_decoder_key(key)
        if target_key is None:
            continue
        if target_key in converted:
            raise ValueError(f"Duplicate converted Stable Audio key: {target_key}")
        if target_key.endswith((".alpha", ".beta")):
            value = value.reshape(1, -1, 1)
        converted[target_key] = value
    return converted


def _load_oobleck_config(model_path: Path) -> tuple[dict[str, Any], int]:
    config_path = model_path / "model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Stable Audio config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        full_config = json.load(file)

    try:
        pretransform = full_config["model"]["pretransform"]
        config = pretransform["config"]
        decoder = config["decoder"]
        decoder_config = decoder["config"]
        sample_rate = int(full_config["sample_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Stable Audio model config: {config_path}") from exc

    if pretransform.get("type") != "autoencoder" or decoder.get("type") != "oobleck":
        raise ValueError("MAGI-2 requires an Oobleck Stable Audio pretransform")
    if decoder_config.get("use_nearest_upsample", False):
        raise ValueError("Diffusers Oobleck does not support nearest-neighbor decoder upsampling")
    if decoder_config.get("final_tanh", True):
        raise ValueError("MAGI-2 requires the Stable Audio decoder with final_tanh=false")

    strides = list(decoder_config["strides"])
    downsampling_ratio = int(config["downsampling_ratio"])
    if int(torch.tensor(strides).prod().item()) != downsampling_ratio:
        raise ValueError("Stable Audio strides do not match downsampling_ratio")

    kwargs = {
        "encoder_hidden_size": int(config["encoder"]["config"]["channels"]),
        "downsampling_ratios": strides,
        "channel_multiples": list(decoder_config["c_mults"]),
        "decoder_channels": int(decoder_config["channels"]),
        "decoder_input_channels": int(decoder_config["latent_dim"]),
        "audio_channels": int(decoder_config["out_channels"]),
        "sampling_rate": sample_rate,
    }
    return kwargs, downsampling_ratio


class Magi2AudioDecoder(nn.Module):
    """Decode MAGI-2 audio latents with the bundled Stable Audio VAE weights."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise ValueError(f"Stable Audio model path must be a local directory: {model_path}")

        oobleck_kwargs, self.downsampling_ratio = _load_oobleck_config(model_path)
        self.sample_rate = int(oobleck_kwargs["sampling_rate"])
        # Construct the Diffusers implementation for architectural compatibility,
        # then retain only the decoder because MAGI-2 has no audio input encoder.
        autoencoder = AutoencoderOobleck(**oobleck_kwargs)
        self.decoder = autoencoder.decoder.to(device=device, dtype=dtype)
        del autoencoder

        weights_path = model_path / "model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"Stable Audio weights do not exist: {weights_path}")
        selected: dict[str, torch.Tensor] = {}
        with safe_open(weights_path, framework="pt", device=str(device)) as checkpoint:
            for key in checkpoint.keys():
                if key.startswith(_CHECKPOINT_PREFIX):
                    selected[key] = checkpoint.get_tensor(key)
        converted = convert_stable_audio_decoder_state_dict(selected)
        converted = {key: value.to(dtype=dtype) for key, value in converted.items()}
        self.decoder.load_state_dict(converted, strict=True)
        self.eval().requires_grad_(False)

    @property
    def latent_fps(self) -> float:
        return self.sample_rate / self.downsampling_ratio

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        parameter = next(self.decoder.parameters())
        return self.decoder(latents.to(device=parameter.device, dtype=parameter.dtype))

    forward = decode


__all__ = [
    "Magi2AudioDecoder",
    "convert_stable_audio_decoder_key",
    "convert_stable_audio_decoder_state_dict",
]
