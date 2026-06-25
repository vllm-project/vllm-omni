# Copyright 2026 OpenMOSS and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Vendored from OpenMOSS-Team/MOSS-Audio-Tokenizer (configuration_moss_audio_tokenizer.py
# and modeling_moss_audio_tokenizer.py).  Simplified for inference-only use:
#   - Streaming KV-cache infrastructure removed (single-pass batch decode only).
#   - Training-only methods (forward, encode, decode) removed.
#   - Dead branches removed: gating="none" always, weights_per_step=0 always,
#     positional_embedding="rope" always, norm="layer_norm" always in default config.
"""MOSS Audio Tokenizer — inference-only codec (encode waveform ↔ RVQ codes)."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class MossAudioTokenizerConfig(PretrainedConfig):
    model_type = "moss-audio-tokenizer"
    attribute_map = {"sample_rate": "sampling_rate"}

    def __init__(
        self,
        version: str | None = None,
        sampling_rate: int = 24000,
        downsample_rate: int = 1920,
        causal_transformer_context_duration: float = 10.0,
        encoder_kwargs: list[dict[str, Any]] | None = None,
        decoder_kwargs: list[dict[str, Any]] | None = None,
        quantizer_type: str = "rlfq",
        quantizer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("model_type", None)
        self.version = version
        self.sampling_rate = sampling_rate
        self.downsample_rate = downsample_rate
        self.causal_transformer_context_duration = causal_transformer_context_duration
        self.encoder_kwargs = encoder_kwargs or _default_encoder_kwargs()
        self.decoder_kwargs = decoder_kwargs or _default_decoder_kwargs()
        if quantizer_kwargs is None:
            quantizer_kwargs = {
                "input_dim": 768,
                "rvq_dim": 512,
                "output_dim": 768,
                "num_quantizers": 32,
                "codebook_size": 1024,
                "codebook_dim": 8,
                "quantizer_type": "rlfq",
            }
        kw_qtype = quantizer_kwargs.get("quantizer_type")
        self.quantizer_type = kw_qtype if kw_qtype is not None else quantizer_type
        quantizer_kwargs["quantizer_type"] = self.quantizer_type
        self.quantizer_kwargs = quantizer_kwargs
        super().__init__(**kwargs)

    @property
    def num_quantizers(self) -> int:
        return int(self.quantizer_kwargs.get("num_quantizers", 32))

    @property
    def codebook_size(self) -> int:
        return int(self.quantizer_kwargs.get("codebook_size", 1024))

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / self.downsample_rate


def _transformer_block(
    input_dim: int, output_dim: int, d_model: int, num_heads: int, num_layers: int
) -> dict[str, Any]:
    return {
        "module_type": "Transformer",
        "input_dimension": input_dim,
        "output_dimension": output_dim,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dim_feedforward": d_model * 4,
        "causal": True,
        "norm": "layer_norm",
        "positional_embedding": "rope",
        "max_period": 10000,
        "gating": "none",
        "layer_scale": 0.01,
        "conv_layout": True,
    }


def _default_encoder_kwargs() -> list[dict[str, Any]]:
    return [
        {"module_type": "PatchedPretransform", "patch_size": 240},
        _transformer_block(240, 384, 768, 12, 12),
        {"module_type": "PatchedPretransform", "patch_size": 2},
        _transformer_block(768, 384, 768, 12, 12),
        {"module_type": "PatchedPretransform", "patch_size": 2},
        _transformer_block(768, 640, 768, 12, 12),
        {"module_type": "PatchedPretransform", "patch_size": 2},
        _transformer_block(1280, 768, 1280, 20, 32),
    ]


def _default_decoder_kwargs() -> list[dict[str, Any]]:
    return [
        _transformer_block(768, 1280, 1280, 20, 32),
        {"module_type": "PatchedPretransform", "patch_size": 2},
        _transformer_block(640, 768, 768, 12, 12),
        {"module_type": "PatchedPretransform", "patch_size": 2},
        _transformer_block(384, 768, 768, 12, 12),
        {"module_type": "PatchedPretransform", "patch_size": 2},
        _transformer_block(384, 768, 768, 12, 12),
        {"module_type": "PatchedPretransform", "patch_size": 2},
        _transformer_block(384, 240, 768, 12, 12),
        {"module_type": "PatchedPretransform", "patch_size": 240},
    ]


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass
class MossAudioTokenizerEncoderOutput(ModelOutput):
    audio_codes: torch.Tensor | None = None
    audio_codes_lengths: torch.Tensor | None = None


@dataclass
class MossAudioTokenizerDecoderOutput(ModelOutput):
    audio: torch.Tensor | None = None
    audio_lengths: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _LayerScale(nn.Module):
    def __init__(self, channels: int, init: float = 1e-4, device=None, dtype=None) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


def _apply_rope(q: torch.Tensor, k: torch.Tensor, max_period: float = 10_000) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary position embedding over sequence dimension (B, H, T, D).

    Matches upstream MossAudioTokenizer's ``apply_rope``: pair the last dim
    as ``(re, im)`` interleaved (GPT-J style) — *not* GPT-NeoX split-halves.
    The two conventions are not interchangeable with the same checkpoint;
    using the wrong one silently rotates random subspaces of the K/V vectors.
    """
    B, H, T, D = q.shape
    half = D // 2
    ds = torch.arange(half, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = torch.arange(T, device=q.device, dtype=torch.float32).view(1, 1, -1, 1)  # (1, 1, T, 1)
    rotr = torch.cos(freqs * ts)  # (1, 1, T, D/2)
    roti = torch.sin(freqs * ts)

    dims = q.shape[:-1]
    q_pair = q.view(*dims, half, 2)
    k_pair = k.view(*dims, half, 2)
    qr, qi = q_pair[..., 0].float(), q_pair[..., 1].float()
    kr, ki = k_pair[..., 0].float(), k_pair[..., 1].float()

    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    qo = torch.stack([qor.to(q.dtype), qoi.to(q.dtype)], dim=-1).view(*dims, D)
    ko = torch.stack([kor.to(k.dtype), koi.to(k.dtype)], dim=-1).view(*dims, D)
    return qo, ko


class _Attention(nn.Module):
    """Causal multi-head self-attention with RoPE, no streaming KV cache."""

    def __init__(
        self, embed_dim: int, num_heads: int, causal: bool, max_period: float, device=None, dtype=None
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.max_period = max_period
        kw = {"device": device, "dtype": dtype}
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False, **kw)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **kw)
        # Remap legacy weight names produced by the upstream checkpoint saver.
        self._register_load_state_dict_pre_hook(self._remap_weights, with_module=True)

    @staticmethod
    def _remap_weights(module, state_dict, prefix, *_):
        for old in ("in_proj_weight", "in_proj.weight"):
            key = prefix + old
            if key in state_dict and (prefix + "in_proj.weight") not in state_dict:
                state_dict[prefix + "in_proj.weight"] = state_dict.pop(key)
        for old in ("out_proj.weight",):
            key = prefix + old
            # If stored under in_projs.0 / out_projs.0 (multi-module layout), remap.
            for src in (prefix + "in_projs.0.weight", prefix + "out_projs.0.weight"):
                if src in state_dict:
                    dst = src.replace("in_projs.0.", "in_proj.").replace("out_projs.0.", "out_proj.")
                    state_dict[dst] = state_dict.pop(src)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.embed_dim // self.num_heads
        qkv = self.in_proj(x).reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, D)
        q, k = _apply_rope(q, k, self.max_period)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(1, 2).reshape(B, T, self.embed_dim)
        return self.out_proj(out)


class _TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        causal: bool,
        max_period: float,
        layer_scale: float | None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5, **kw)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5, **kw)
        self.attn = _Attention(d_model, num_heads, causal, max_period, device, dtype)
        self.ff1 = nn.Linear(d_model, dim_feedforward, bias=False, **kw)
        self.ff2 = nn.Linear(dim_feedforward, d_model, bias=False, **kw)
        ls_kw = cast(dict[str, object], kw)
        if layer_scale is not None:
            self.ls1 = _LayerScale(d_model, layer_scale, **ls_kw)
            self.ls2 = _LayerScale(d_model, layer_scale, **ls_kw)
        else:
            self.ls1 = nn.Identity()
            self.ls2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.ff2(F.gelu(self.ff1(self.norm2(x)))))
        return x


class _Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        causal: bool,
        max_period: float,
        layer_scale: float | None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TransformerLayer(d_model, num_heads, dim_feedforward, causal, max_period, layer_scale, device, dtype)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _ProjectedTransformer(nn.Module):
    """Input/output projection + causal transformer (conv-layout: B, D, T)."""

    def __init__(
        self, input_dimension: int, output_dimension: int, d_model: int, *, module_type: str, **kwargs: Any
    ) -> None:
        super().__init__()
        self.downsample_ratio: int = 1
        self.in_proj = nn.Linear(input_dimension, d_model, bias=False) if d_model != input_dimension else nn.Identity()
        self.transformer = _Transformer(d_model=d_model, **kwargs)
        self.out_proj = (
            nn.Linear(d_model, output_dimension, bias=False) if d_model != output_dimension else nn.Identity()
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.in_proj(x.transpose(1, 2))  # (B, D, T) → (B, T, d_model)
        x = self.transformer(x)
        x = self.out_proj(x).transpose(1, 2)  # (B, T, D) → (B, D, T)
        return x, lengths


class _PatchedPretransform(nn.Module):
    """Patch-based up/down-sampling (no learned weights)."""

    def __init__(self, patch_size: int, is_downsample: bool, module_type: str, **_: Any) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.downsample_ratio: int = patch_size
        self.is_downsample = is_downsample

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, d, _ = x.shape
        h = self.patch_size
        if self.is_downsample:
            x = x.reshape(b, d, -1, h).permute(0, 1, 3, 2).reshape(b, d * h, -1)
            return x, lengths // h
        else:
            # Upsample by ``h``: split channel dim, then flatten last two dims.
            # Was ``reshape(b, d // h, -1 * h)`` which evaluates to ``-h`` and
            # is rejected by torch (the failing path produced ``[1, 640, -2]``).
            x = x.reshape(b, d // h, h, -1).permute(0, 1, 3, 2).reshape(b, d // h, -1)
            return x, lengths * h


# ---------------------------------------------------------------------------
# Vector Quantization
# ---------------------------------------------------------------------------


def _wn_conv1d(*args: Any, **kwargs: Any) -> nn.Module:
    return nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


class _VQ(nn.Module):
    """Single RVQ codebook (inference: encode + decode)."""

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, **_: Any) -> None:
        super().__init__()
        self.in_proj = _wn_conv1d(input_dim, codebook_dim, 1) if input_dim != codebook_dim else nn.Identity()
        self.out_proj = _wn_conv1d(codebook_dim, input_dim, 1) if input_dim != codebook_dim else nn.Identity()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_e = self.in_proj(z.float())
        enc = z_e.transpose(1, 2).reshape(-1, z_e.shape[1])
        w = self.codebook.weight.float()
        dist = enc.pow(2).sum(1, keepdim=True) - 2 * enc @ w.t() + w.pow(2).sum(1, keepdim=True).t()
        ids = (-dist).max(1)[1].reshape(z.size(0), -1)
        return self.out_proj(self.codebook(ids).transpose(1, 2).float()).float(), ids

    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.codebook(ids).transpose(1, 2).float()).float()


class _LFQ(nn.Module):
    """Single RLFQ codebook (inference: encode + decode)."""

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, **_: Any) -> None:
        super().__init__()
        self.in_proj = _wn_conv1d(input_dim, codebook_dim, 1) if input_dim != codebook_dim else nn.Identity()
        self.out_proj = _wn_conv1d(codebook_dim, input_dim, 1) if input_dim != codebook_dim else nn.Identity()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_e = self.in_proj(z.float())
        enc = F.normalize(z_e.transpose(1, 2).reshape(-1, z_e.shape[1]).float())
        cb = F.normalize(self.codebook.weight.float())
        dist = enc.pow(2).sum(1, keepdim=True) - 2 * enc @ cb.t() + cb.pow(2).sum(1, keepdim=True).t()
        ids = (-dist).max(1)[1].reshape(z.size(0), -1)
        z_q = self.codebook(ids).transpose(1, 2).float()
        z_q = (z_e + (z_q - z_e).detach()).float()
        return self.out_proj(z_q).float(), ids

    def decode(self, ids: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.codebook(ids).transpose(1, 2).float()).float()


class _ResidualQ(nn.Module):
    """Residual VQ or LFQ stack."""

    def __init__(
        self,
        input_dim: int,
        rvq_dim: int | None,
        output_dim: int | None,
        num_quantizers: int,
        codebook_size: int,
        codebook_dim: int,
        quantizer_type: str = "rlfq",
        **_: Any,
    ) -> None:
        super().__init__()
        self.rvq_dim = rvq_dim or input_dim
        self.output_dim = output_dim or input_dim
        QCls = _LFQ if quantizer_type in {"rlfq", "random_prefix_rlfq"} else _VQ
        self.input_proj = _wn_conv1d(input_dim, self.rvq_dim, 1) if input_dim != self.rvq_dim else nn.Identity()
        self.output_proj = (
            _wn_conv1d(self.rvq_dim, self.output_dim, 1) if self.rvq_dim != self.output_dim else nn.Identity()
        )
        self.quantizers = nn.ModuleList(
            [QCls(self.rvq_dim, codebook_size, codebook_dim) for _ in range(num_quantizers)]
        )

    @torch.no_grad()
    def encode(
        self, z: torch.Tensor, lengths: torch.Tensor, n: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.input_proj(z)
        B, _, T = z.shape
        mask = torch.arange(T, device=z.device).expand(B, T) < lengths.unsqueeze(1)
        out = torch.zeros_like(z, dtype=torch.float32)
        residual = z.clone().float()
        codes: list[torch.Tensor] = []
        for i, q in enumerate(self.quantizers[: n or len(self.quantizers)]):
            zq, ids = q.encode(residual * mask.unsqueeze(1))
            out += zq * mask.unsqueeze(1)
            residual -= zq * mask.unsqueeze(1)
            codes.append(ids)
        return self.output_proj(out), torch.stack(codes), lengths

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (NQ, B, T) → (B, rvq_dim, T)."""
        nq, B, T = codes.shape
        emb = torch.zeros(B, self.rvq_dim, T, device=codes.device, dtype=torch.float32)
        for i, q in enumerate(self.quantizers[:nq]):
            emb += q.decode(codes[i])
        return self.output_proj(emb)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _build_modules(
    specs: list[dict[str, Any]], is_downsample: bool, context_duration: float, sampling_rate: float
) -> nn.ModuleList:
    modules: list[nn.Module] = []
    rate = float(sampling_rate)
    for spec in specs:
        spec = dict(spec)
        if spec["module_type"] == "PatchedPretransform":
            m = _PatchedPretransform(**spec, is_downsample=is_downsample)
        else:
            spec.pop("conv_layout", None)
            spec.pop("module_type", None)
            spec.pop("gating", None)
            spec.pop("positional_embedding", None)
            spec.pop("norm", None)
            spec.pop("causal", None)
            m = _ProjectedTransformer(
                module_type="Transformer",
                causal=True,
                max_period=spec.pop("max_period", 10000),
                layer_scale=spec.pop("layer_scale", None),
                **spec,
            )
        modules.append(m)
        rate = rate / m.downsample_ratio if is_downsample else rate * m.downsample_ratio
    return nn.ModuleList(modules)


class MossAudioTokenizerModel(PreTrainedModel):
    """MOSS Audio Tokenizer — inference-only (batch_encode / batch_decode)."""

    config_class = MossAudioTokenizerConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = False

    def __init__(self, config: MossAudioTokenizerConfig) -> None:
        super().__init__(config)
        self.sampling_rate = config.sampling_rate
        self.downsample_rate = config.downsample_rate

        ctx = config.causal_transformer_context_duration
        sr = float(config.sampling_rate)
        self.encoder = _build_modules(config.encoder_kwargs, is_downsample=True, context_duration=ctx, sampling_rate=sr)
        self.decoder = _build_modules(
            copy.deepcopy(config.decoder_kwargs), is_downsample=False, context_duration=ctx, sampling_rate=sr
        )

        kw = dict(config.quantizer_kwargs)
        self.quantizer = _ResidualQ(**kw)
        self.post_init()

    @torch.no_grad()
    def batch_encode(
        self,
        wav_list: list[torch.Tensor],
        num_quantizers: int | None = None,
    ) -> MossAudioTokenizerEncoderOutput:
        """Encode a list of 1-D waveform tensors → RVQ codes (NQ, B, T)."""
        device = wav_list[0].device
        B = len(wav_list)
        max_len = max(w.shape[-1] for w in wav_list)
        x = torch.zeros(B, 1, max_len, device=device)
        lengths = torch.zeros(B, device=device, dtype=torch.long)
        for i, w in enumerate(wav_list):
            x[i, 0, : w.shape[-1]] = w
            lengths[i] = w.shape[-1]
        return self._encode(x, lengths, num_quantizers)

    @torch.no_grad()
    def batch_decode(
        self,
        codes_list: list[torch.Tensor],
        num_quantizers: int | None = None,
    ) -> MossAudioTokenizerDecoderOutput:
        """Decode a list of (NQ, T) code tensors → waveforms."""
        device = codes_list[0].device
        B = len(codes_list)
        nq = num_quantizers or codes_list[0].shape[0]
        max_t = max(c.shape[-1] for c in codes_list)
        codes = torch.zeros(nq, B, max_t, device=device, dtype=torch.long)
        lengths = torch.zeros(B, device=device, dtype=torch.long)
        for i, c in enumerate(codes_list):
            codes[:nq, i, : c.shape[-1]] = c[:nq]
            lengths[i] = c.shape[-1]
        return self._decode(codes, lengths)

    def _encode(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        n_quantizers: int | None = None,
    ) -> MossAudioTokenizerEncoderOutput:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        T = x.shape[-1]
        if T % self.downsample_rate:
            x = F.pad(x, (0, self.downsample_rate - T % self.downsample_rate))
        e, e_len = x, lengths
        for m in self.encoder:
            e, e_len = m(e, e_len)
        _, codes, code_len = self.quantizer.encode(e, e_len, n_quantizers)
        return MossAudioTokenizerEncoderOutput(audio_codes=codes, audio_codes_lengths=code_len)

    def _decode(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> MossAudioTokenizerDecoderOutput:
        z = self.quantizer.decode_codes(codes)
        d, d_len = z, lengths
        for m in self.decoder:
            d, d_len = m(d, d_len)
        return MossAudioTokenizerDecoderOutput(audio=d, audio_lengths=d_len)


__all__ = [
    "MossAudioTokenizerConfig",
    "MossAudioTokenizerModel",
    "MossAudioTokenizerEncoderOutput",
    "MossAudioTokenizerDecoderOutput",
]
