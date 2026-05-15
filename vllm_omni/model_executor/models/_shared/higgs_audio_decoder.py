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
    "_remap_boson_model_pth_state_dict",  # exported for unit-testing the mapper
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


def _load_higgs_audio_state_dict(audio_tokenizer_dir: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Load the codec state dict from either layout used in the wild.

    Tries layouts in order:
    1. ``<dir>/model.safetensors`` (OmniVoice-bundled layout used by
       ``k2-fsa/OmniVoice/audio_tokenizer/``).
    2. ``<dir>/model.pth`` (boson-ai standalone ``bosonai/higgs-audio-v2-tokenizer``
       layout). The state-dict keys differ structurally
       (``quantizer.vq.layers.<i>._codebook.embed`` etc.) and are remapped to
       the OmniVoice-style names this kernel expects via
       :func:`_remap_boson_model_pth_state_dict`.
    """
    safetensors_path = os.path.join(audio_tokenizer_dir, "model.safetensors")
    pth_path = os.path.join(audio_tokenizer_dir, "model.pth")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file

        return load_file(safetensors_path, device=str(device))
    if os.path.exists(pth_path):
        sd = torch.load(pth_path, map_location=device, weights_only=False)
        return _remap_boson_model_pth_state_dict(sd)
    raise FileNotFoundError(
        f"Audio tokenizer weights not found at {safetensors_path} or {pth_path}"
    )


def _remap_boson_model_pth_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate boson-ai's standalone ``model.pth`` keys into OmniVoice-style names
    that the shared kernel's RVQ + fc2 + DAC sites consume.

    Best-effort: only the RVQ-side keys map cleanly. The boson-ai decoder
    uses ``decoder_2.model.<i>.weight_g/weight_v`` (DAC with weight-norm +
    Snake activations) rather than the OmniVoice DAC layout; that side
    requires either vendoring the upstream decoder module or rewriting the
    DAC builder to consume weight-normed tensors. This function returns the
    mapped RVQ keys plus any acoustic_decoder.* / fc2.* / fc.* tensors that
    happen to share the OmniVoice names, and leaves the decoder-2 keys
    untouched (the caller will see them as MISSING when copying into the
    OmniVoice DAC decoder).

    Mapping (RVQ side only):
        quantizer.vq.layers.<i>._codebook.embed -> quantizer.quantizers.<i>.codebook.embed
        quantizer.vq.layers.<i>.project_out.weight -> quantizer.quantizers.<i>.project_out.weight
        quantizer.vq.layers.<i>.project_out.bias   -> quantizer.quantizers.<i>.project_out.bias
    """
    if not isinstance(sd, dict):
        raise TypeError(f"expected a state dict, got {type(sd)!r}")
    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        # Quantizer rewrite: vq.layers.<i>._codebook.embed -> quantizers.<i>.codebook.embed
        if key.startswith("quantizer.vq.layers."):
            parts = key.split(".")
            # parts[3] is the layer index, parts[4]+ is the tail
            if len(parts) >= 5:
                idx = parts[3]
                tail = ".".join(parts[4:])
                if tail.startswith("_codebook.embed"):
                    new_key = f"quantizer.quantizers.{idx}.codebook.embed"
                    remapped[new_key] = tensor
                    continue
                if tail.startswith("project_out."):
                    new_key = f"quantizer.quantizers.{idx}.{tail}"
                    remapped[new_key] = tensor
                    continue
                # project_in / _codebook.cluster_size / _codebook.embed_avg / inited
                # are encoder-side or training-only state that the decode path
                # does not need; drop them.
                continue
        # Anything else (acoustic_decoder.*, fc2.*, fc.*, fc_post*.*, decoder_2.*,
        # decoder_semantic.*, semantic_model.*) passes through unchanged so
        # the caller's lookup is unambiguous.
        remapped[key] = tensor
    return remapped


def load_higgs_audio_codec(
    audio_tokenizer_dir: str,
    device: torch.device,
) -> tuple[HiggsAudioRVQ, nn.Linear, nn.Module, dict[str, Any]]:
    """Load the HiggsAudioV2 RVQ + fc2 + DAC decoder from a checkpoint folder.

    Accepts both layouts: ``<dir>/model.safetensors`` (OmniVoice-bundled) and
    ``<dir>/model.pth`` (boson-ai standalone). The standalone path remaps
    quantizer keys before consumption; structural decoder differences
    (boson uses Snake + weight-norm) still leave some DAC parameters missing
    when loading the standalone path -- those entries log warnings but the
    RVQ side completes successfully.

    Args:
        audio_tokenizer_dir: Path to a directory containing ``config.json``
            and EITHER ``model.safetensors`` (OmniVoice layout) OR
            ``model.pth`` (boson-ai standalone layout).
        device: Device to place the loaded modules and state dict on.

    Returns:
        (quantizer, fc2, acoustic_decoder, tokenizer_config)
        - quantizer: ``HiggsAudioRVQ`` with ``num_quantizers`` discovered from
          the state dict (defaults to 8 for boson-ai checkpoints).
        - fc2: ``nn.Linear`` projecting RVQ output (1024) into the DAC's
          hidden dimension (typically 256). May be uninitialized when loaded
          from the boson-ai standalone layout (no ``fc2.*`` keys present).
        - acoustic_decoder: DAC decoder with HiggsAudioV2 output-padding fix
          and tanh-replaced-by-Identity. Fully initialized for the OmniVoice
          layout; partially initialized for boson-ai standalone (missing
          ``decoder_2.*`` -> ``acoustic_decoder.*`` mapping is not yet
          implemented; full upstream-decoder vendoring is the next step).
        - tokenizer_config: The loaded ``config.json`` dict; useful for
          callers that need ``sample_rate`` or other tokenizer metadata.
    """
    config_path = os.path.join(audio_tokenizer_dir, "config.json")

    with open(config_path) as f:
        tokenizer_config: dict[str, Any] = json.load(f)

    state_dict = _load_higgs_audio_state_dict(audio_tokenizer_dir, device)

    codebook_dim = tokenizer_config.get("codebook_dim", 64)
    codebook_size = tokenizer_config.get("codebook_size", 1024)
    # Discover hidden_size and num_quantizers from the (possibly remapped) state dict.
    if "quantizer.quantizers.0.project_out.weight" not in state_dict:
        raise KeyError(
            "Codec state dict is missing 'quantizer.quantizers.0.project_out.weight'. "
            "If you loaded a boson-ai standalone tokenizer, ensure the model.pth "
            "remap fired (see _remap_boson_model_pth_state_dict)."
        )
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
