# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HiggsAudio codec decoder kernel for higgs-audio v2.

This module hosts the parameter-side building blocks for the higgs-audio-v2
audio tokenizer's decoder path:

  audio_codes [B, 8, T]
    -> RVQ codebook lookup + project_out -> sum -> [B, hidden_size, T]
    -> fc2 Linear(hidden_size, 256) -> [B, 256, T]
    -> DAC acoustic decoder (conv-transpose upsampling) -> [B, 1, T*960]
    -> 24 kHz waveform (25 fps x 960 samples/frame)
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
    "BosonDacDecoder",
    "build_boson_dac_decoder",
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


@torch.jit.script
def _snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


class _Snake1d(nn.Module):
    """Snake activation (per-channel learnable alpha). Mirrors the boson-ai DAC
    Snake1d implementation byte-for-byte so the boson ``model.pth`` state-dict
    loads directly into ``self.alpha``.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _snake(x, self.alpha)


def _wn_conv1d(*args: Any, **kwargs: Any) -> nn.Module:
    from torch.nn.utils import weight_norm

    return weight_norm(nn.Conv1d(*args, **kwargs))


def _wn_conv_transpose1d(*args: Any, **kwargs: Any) -> nn.Module:
    from torch.nn.utils import weight_norm

    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class _BosonResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            _Snake1d(dim),
            _wn_conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            _Snake1d(dim),
            _wn_conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class _BosonDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        import math

        self.block = nn.Sequential(
            _Snake1d(input_dim),
            _wn_conv_transpose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            _BosonResidualUnit(output_dim, dilation=1),
            _BosonResidualUnit(output_dim, dilation=3),
            _BosonResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BosonDacDecoder(nn.Module):
    """Vendored copy of ``boson_multimodal``'s standalone DAC decoder, with the
    exact ``decoder_2.model.<i>.{...}`` key layout the boson-ai
    ``bosonai/higgs-audio-v2-tokenizer`` checkpoint expects.

    See ``boson_multimodal/audio_processing/descriptaudiocodec/dac/model/dac.py``
    (``class Decoder``) in the upstream package for the reference implementation
    we mirror.

    Architecture for the v2 24 kHz tokenizer:
      input_channel=256, channels=1024, rates=[8, 5, 4, 2, 3], d_out=1
    Cumulative upsample factor is ``prod(rates) = 960`` (25 fps -> 24 kHz).
    """

    def __init__(
        self,
        input_channel: int = 256,
        channels: int = 1024,
        rates: tuple[int, ...] = (8, 5, 4, 2, 3),
        d_out: int = 1,
    ):
        super().__init__()
        layers: list[nn.Module] = [_wn_conv1d(input_channel, channels, kernel_size=7, padding=3)]
        last_dim = channels
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers.append(_BosonDecoderBlock(input_dim, output_dim, stride))
            last_dim = output_dim
        layers.append(_Snake1d(last_dim))
        layers.append(_wn_conv1d(last_dim, d_out, kernel_size=7, padding=3))
        self.model = nn.Sequential(*layers)
        # Cache hop length (= prod(rates)) so callers don't need to recompute it.
        hop = 1
        for s in rates:
            hop *= int(s)
        self._hop_length = hop

    @property
    def hop_length(self) -> int:
        return self._hop_length

    # Expose conv1 with a ``.in_channels`` attribute so the existing fc2
    # fallback shape inference (acoustic_decoder.conv1.in_channels) works
    # unchanged for the boson decoder too.
    @property
    def conv1(self) -> nn.Module:
        return self.model[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_boson_dac_decoder(device: torch.device) -> BosonDacDecoder:
    """Construct the vendored boson DAC decoder on ``device``.

    The architectural constants (input_channel=256, channels=1024,
    rates=[8,5,4,2,3], d_out=1) are baked in because they're invariant for the
    v2 24 kHz tokenizer ``bosonai/higgs-audio-v2-tokenizer`` checkpoint we
    target. Other channel counts are not supported until we encounter a
    different boson checkpoint in the wild.
    """
    return BosonDacDecoder().to(device)


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
    # Preference order: try model.pth first when present (boson-ai standalone
    # ships the codec there; its model.safetensors is a different artefact).
    # Fall back to model.safetensors (the OmniVoice-bundled layout).
    sd: dict[str, torch.Tensor]
    if os.path.exists(pth_path):
        sd = torch.load(pth_path, map_location=device, weights_only=False)
    elif os.path.exists(safetensors_path):
        from safetensors.torch import load_file

        sd = load_file(safetensors_path, device=str(device))
    else:
        raise FileNotFoundError(f"Audio tokenizer weights not found at {pth_path} or {safetensors_path}")
    # Run the boson-layout remap if needed; for the OmniVoice-bundled layout
    # it is a no-op (no ``quantizer.vq.layers.*`` keys to rewrite).
    if any(k.startswith("quantizer.vq.layers.") for k in sd):
        sd = _remap_boson_model_pth_state_dict(sd)
    return sd


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
                # The actual codebook centroids the decode path uses live at
                # ``_codebook.embed`` (a [codebook_size, codebook_dim] buffer).
                # ``_codebook.embed_avg`` is the un-normalised EMA accumulator
                # used during training (~cluster_size x bigger); MATCHING IT
                # HERE produces a 4-5x amplified codebook and unintelligible
                # PCM downstream. Use an EXACT-match check, not startswith.
                if tail == "_codebook.embed":
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
        # boson decoder + fc_post2 -> our vendored kernel names.
        if key.startswith("decoder_2."):
            # decoder_2.model.<i>.{...} -> acoustic_decoder.model.<i>.{...}
            new_key = "acoustic_decoder." + key[len("decoder_2.") :]
            remapped[new_key] = tensor
            continue
        if key.startswith("fc_post2."):
            # fc_post2.{weight,bias} -> fc2.{weight,bias}
            new_key = "fc2." + key[len("fc_post2.") :]
            remapped[new_key] = tensor
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

    # Decide which acoustic decoder we need. A boson-layout state dict (after
    # _remap_boson_model_pth_state_dict) carries ``acoustic_decoder.model.0.weight_v``
    # (the WNConv first conv) which the OmniVoice DacModel doesn't have; the
    # OmniVoice layout carries ``acoustic_decoder.block.0.*`` instead.
    is_boson_layout = "acoustic_decoder.model.0.weight_v" in state_dict
    if is_boson_layout:
        acoustic_decoder = build_boson_dac_decoder(device)
        boson_sd = {
            k[len("acoustic_decoder.") :]: v for k, v in state_dict.items() if k.startswith("acoustic_decoder.")
        }
        load_report = acoustic_decoder.load_state_dict(boson_sd, strict=False)
        if load_report.missing_keys:
            raise RuntimeError(
                "Boson DAC decoder is missing keys after load: "
                f"{load_report.missing_keys[:5]}... ({len(load_report.missing_keys)} total)"
            )
        # ``unexpected_keys`` is informational only.
    else:
        acoustic_decoder = build_higgs_audio_acoustic_decoder(tokenizer_config, device)
        for name, param in acoustic_decoder.named_parameters():
            higgs_name = f"acoustic_decoder.{name}"
            if higgs_name in state_dict:
                param.data.copy_(state_dict[higgs_name])
    acoustic_decoder.eval()

    if "fc2.weight" in state_dict and "fc2.bias" in state_dict:
        fc2_w = state_dict["fc2.weight"]
        fc2_b = state_dict["fc2.bias"]
        fc2 = nn.Linear(fc2_w.shape[1], fc2_w.shape[0]).to(device)
        fc2.weight.data.copy_(fc2_w)
        fc2.bias.data.copy_(fc2_b)
    else:
        # Last-resort fallback: shape fc2 from the decoder's first conv input
        # channels so the matmul produces compatible shapes. Weights stay
        # uninitialised — PCM quality will be noise on this path.
        first_conv = getattr(acoustic_decoder, "conv1", None)
        if first_conv is None or not hasattr(first_conv, "in_channels"):
            raise RuntimeError("acoustic_decoder has no conv1.in_channels; cannot size fc2 fallback")
        fc2 = nn.Linear(hidden_size, int(first_conv.in_channels)).to(device)

    return quantizer, fc2, acoustic_decoder, tokenizer_config
