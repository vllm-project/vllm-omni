# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared HiggsAudio codec decoder kernel.

This module hosts the parameter-side building blocks for the higgs-audio-v2
audio tokenizer's decoder path:

  audio_codes [B, 8, T]
    -> RVQ codebook lookup + project_out -> sum -> [B, hidden_size, T]
    -> fc2 Linear(hidden_size, 256) -> [B, 256, T]
    -> DAC acoustic decoder (conv-transpose upsampling) -> [B, 1, T*960]
    -> 24 kHz waveform (25 fps x 960 samples/frame)

Originally lived inside vllm_omni/model_executor/models/omnivoice/omnivoice_decoder.py.
Lifted here so the new higgs_audio_v2 integration can reuse the exact same
kernel, while OmniVoice continues to import the symbols via a backward-compatible
re-export shim in omnivoice_decoder.py.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "HiggsAudioVQLayer",
    "HiggsAudioRVQ",
    "adjust_conv_transpose_output_padding",
    "build_higgs_audio_acoustic_decoder",
    "load_higgs_audio_codec",
]


class HiggsAudioVQLayer(nn.Module):
    """Single VQ layer: codebook lookup + project_out."""

    def __init__(self, codebook_size: int = 1024, codebook_dim: int = 64, hidden_size: int = 1024):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.project_out = nn.Linear(codebook_dim, hidden_size)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [B, T] -> [B, hidden_size, T]."""
        quantized = self.codebook(indices)
        quantized = self.project_out(quantized)
        return quantized.permute(0, 2, 1)


class HiggsAudioRVQ(nn.Module):
    """Residual Vector Quantizer with ``num_quantizers`` codebook layers."""

    def __init__(
        self,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 64,
        hidden_size: int = 1024,
    ):
        super().__init__()
        self.quantizers = nn.ModuleList(
            [HiggsAudioVQLayer(codebook_size, codebook_dim, hidden_size) for _ in range(num_quantizers)]
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: [num_quantizers, B, T] -> [B, hidden_size, T]."""
        result = torch.zeros(
            codes.shape[1],
            self.quantizers[0].project_out.out_features,
            codes.shape[2],
            device=codes.device,
            dtype=torch.float32,
        )
        for i, quantizer in enumerate(self.quantizers):
            result = result + quantizer.decode(codes[i])
        return result


def adjust_conv_transpose_output_padding(decoder: nn.Module) -> None:
    """Set ConvTranspose1d output_padding = stride % 2 (HiggsAudioV2 modification).

    The vanilla DAC decoder ships with the default output_padding (0); the
    boson-ai checkpoint expects ``stride % 2`` instead. This is a no-op for
    even strides and adds a single sample for odd strides.
    """
    for module in decoder.modules():
        if isinstance(module, nn.ConvTranspose1d):
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            module.output_padding = (stride % 2,)


def build_higgs_audio_acoustic_decoder(
    tokenizer_config: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Build the DAC acoustic decoder used by the HiggsAudioV2 tokenizer.

    Returns the decoder sub-module of ``transformers.DacModel`` with the
    HiggsAudioV2 output-padding fix already applied. The tanh activation is
    replaced with ``Identity`` so the network matches the boson-ai checkpoint.
    Weights are NOT loaded here; the caller is responsible for copying them in.
    """
    from transformers import DacConfig, DacModel

    dac_cfg = DacConfig(**tokenizer_config["acoustic_model_config"])
    dac_model = DacModel(dac_cfg)
    decoder = dac_model.decoder.to(device)
    adjust_conv_transpose_output_padding(decoder)
    if hasattr(decoder, "tanh"):
        decoder.tanh = nn.Identity()
    return decoder


def load_higgs_audio_codec(
    audio_tokenizer_dir: str,
    device: torch.device,
) -> tuple[HiggsAudioRVQ, nn.Linear, nn.Module, dict[str, Any]]:
    """Load the HiggsAudioV2 RVQ + fc2 + DAC decoder from a checkpoint folder.

    Args:
        audio_tokenizer_dir: Path to a directory containing ``config.json`` and
            ``model.safetensors`` (the boson-ai ``audio_tokenizer/`` layout).
        device: Device to place the loaded modules and state dict on.

    Returns:
        (quantizer, fc2, acoustic_decoder, tokenizer_config)
        - quantizer: ``HiggsAudioRVQ`` with ``num_quantizers`` discovered from the
          state dict (defaults to 8 for boson-ai checkpoints).
        - fc2: ``nn.Linear`` projecting RVQ output (1024) into the DAC encoder's
          hidden dimension (typically 256).
        - acoustic_decoder: DAC decoder with HiggsAudioV2 output-padding fix and
          tanh-replaced-by-Identity, fully initialized.
        - tokenizer_config: The loaded ``config.json`` dict; useful for callers
          that need ``sample_rate`` or other tokenizer metadata.
    """
    from safetensors.torch import load_file

    config_path = os.path.join(audio_tokenizer_dir, "config.json")
    weights_path = os.path.join(audio_tokenizer_dir, "model.safetensors")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Audio tokenizer weights not found at {weights_path}")

    with open(config_path) as f:
        tokenizer_config: dict[str, Any] = json.load(f)

    state_dict = load_file(weights_path, device=str(device))

    codebook_dim = tokenizer_config.get("codebook_dim", 64)
    codebook_size = tokenizer_config.get("codebook_size", 1024)
    hidden_size = state_dict["quantizer.quantizers.0.project_out.weight"].shape[0]
    num_quantizers = sum(
        1 for k in state_dict if k.startswith("quantizer.quantizers.") and k.endswith(".codebook.embed")
    )

    quantizer = HiggsAudioRVQ(
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
        hidden_size=hidden_size,
    ).to(device)
    for i in range(num_quantizers):
        prefix = f"quantizer.quantizers.{i}"
        embed_key = f"{prefix}.codebook.embed"
        if embed_key in state_dict:
            quantizer.quantizers[i].codebook.weight.data.copy_(state_dict[embed_key])
        proj_out_w = f"{prefix}.project_out.weight"
        proj_out_b = f"{prefix}.project_out.bias"
        if proj_out_w in state_dict:
            quantizer.quantizers[i].project_out.weight.data.copy_(state_dict[proj_out_w])
        if proj_out_b in state_dict:
            quantizer.quantizers[i].project_out.bias.data.copy_(state_dict[proj_out_b])

    fc2_w = state_dict["fc2.weight"]
    fc2_b = state_dict["fc2.bias"]
    fc2 = nn.Linear(fc2_w.shape[1], fc2_w.shape[0]).to(device)
    fc2.weight.data.copy_(fc2_w)
    fc2.bias.data.copy_(fc2_b)

    acoustic_decoder = build_higgs_audio_acoustic_decoder(tokenizer_config, device)
    for name, param in acoustic_decoder.named_parameters():
        higgs_name = f"acoustic_decoder.{name}"
        if higgs_name in state_dict:
            param.data.copy_(state_dict[higgs_name])
    acoustic_decoder.eval()

    return quantizer, fc2, acoustic_decoder, tokenizer_config
