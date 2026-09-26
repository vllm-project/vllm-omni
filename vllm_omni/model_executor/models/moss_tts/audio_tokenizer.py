# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
        number_channels: int = 1,
        enable_channel_interleave: bool = True,
        quantizer_type: str = "rlfq",
        quantizer_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("model_type", None)
        self.version = version
        self.sampling_rate = sampling_rate
        self.downsample_rate = downsample_rate
        self.causal_transformer_context_duration = causal_transformer_context_duration
        self.number_channels = number_channels
        self.enable_channel_interleave = enable_channel_interleave
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


@dataclass(frozen=True)
class _StreamingExecutionContext:
    state_slot_ids: torch.Tensor
    valid_rows: torch.Tensor

    def validate(self, *, batch_size: int, state_capacity: int, device: torch.device) -> None:
        if self.state_slot_ids.shape != (batch_size,):
            raise ValueError(f"Expected state_slot_ids shape ({batch_size},), got {tuple(self.state_slot_ids.shape)}")
        if self.valid_rows.shape != (batch_size,):
            raise ValueError(f"Expected valid_rows shape ({batch_size},), got {tuple(self.valid_rows.shape)}")
        if self.state_slot_ids.device != device or self.valid_rows.device != device:
            raise ValueError("Streaming execution metadata must be on the same device as decoder inputs.")
        if self.state_slot_ids.dtype != torch.long or self.valid_rows.dtype != torch.bool:
            raise TypeError("state_slot_ids must be int64 and valid_rows must be bool.")
        if device.type == "cpu" and self.state_slot_ids.numel() > 0:
            min_slot = int(self.state_slot_ids.min())
            max_slot = int(self.state_slot_ids.max())
            if min_slot < 0 or max_slot >= state_capacity:
                raise ValueError(
                    f"Decoder state slots must be in [0, {state_capacity}), got {self.state_slot_ids.tolist()}"
                )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _LayerScale(nn.Module):
    def __init__(self, channels: int, init: float = 1e-4, device=None, dtype=None) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    offsets: torch.Tensor | None = None,
    max_period: float = 10_000,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    if offsets is None:
        offsets = torch.zeros(B, device=q.device, dtype=torch.long)
    ts = offsets.float().view(B, 1) + torch.arange(T, device=q.device, dtype=torch.float32)
    ts = ts.view(B, 1, T, 1)
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


class _RingKVCache:
    def __init__(
        self,
        state_capacity: int,
        num_heads: int,
        context: int,
        head_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.context = context
        self.cache = torch.zeros(
            (2, state_capacity, num_heads, context, head_dim),
            device=device,
            dtype=dtype,
        )

    def complete(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        offsets: torch.Tensor,
        execution_context: _StreamingExecutionContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_heads, num_frames, head_dim = k.shape
        slots = execution_context.state_slot_ids
        row_cache = self.cache.index_select(1, slots)
        cache_indexes = torch.arange(self.context, device=offsets.device, dtype=torch.long)
        last_offset = offsets.view(-1, 1) - 1
        end_index = last_offset % self.context
        delta = cache_indexes - end_index
        cached_positions = torch.where(
            delta <= 0,
            last_offset + delta,
            last_offset + delta - self.context,
        )
        cached_positions = torch.where(
            (cache_indexes < offsets.view(-1, 1)) & execution_context.valid_rows.view(-1, 1),
            cached_positions,
            torch.full_like(cached_positions, -1),
        )
        current_positions = offsets.view(-1, 1) + torch.arange(num_frames, device=offsets.device, dtype=offsets.dtype)
        current_positions = torch.where(
            execution_context.valid_rows.view(-1, 1),
            current_positions,
            torch.full_like(current_positions, -1),
        )

        # Keep this chunk separate from the pre-write cache so a ring wrap does
        # not overwrite keys that early queries in the chunk still need.
        keys = torch.cat((row_cache[0], k), dim=2)
        values = torch.cat((row_cache[1], v), dim=2)
        positions = torch.cat((cached_positions, current_positions), dim=1)

        cache_frames = min(num_frames, self.context)
        cache_start = offsets.view(-1, 1) + num_frames - cache_frames
        indexes = torch.arange(cache_frames, device=offsets.device, dtype=offsets.dtype)
        indexes = (indexes + cache_start) % self.context
        scatter_indexes = indexes.view(batch_size, 1, cache_frames, 1).expand(-1, num_heads, -1, head_dim)
        row_cache[0].scatter_(2, scatter_indexes, k[:, :, -cache_frames:])
        row_cache[1].scatter_(2, scatter_indexes, v[:, :, -cache_frames:])
        self.cache.index_copy_(1, slots, row_cache)
        return keys, values, positions


@dataclass
class _AttentionStreamingState:
    offsets: torch.Tensor
    kv_cache: _RingKVCache


class _Attention(nn.Module):
    """Causal multi-head self-attention with request-addressed streaming state."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        causal: bool,
        max_period: float,
        context: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.max_period = max_period
        self.context = context
        self._streaming_state: _AttentionStreamingState | None = None
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

    def initialize_streaming_state(self, offsets: torch.Tensor) -> None:
        if self._streaming_state is not None:
            raise RuntimeError("MOSS Audio Tokenizer attention is already streaming.")
        weight = self.in_proj.weight
        head_dim = self.embed_dim // self.num_heads
        self._streaming_state = _AttentionStreamingState(
            offsets=offsets,
            kv_cache=_RingKVCache(
                offsets.shape[0],
                self.num_heads,
                self.context or 1024,
                head_dim,
                device=weight.device,
                dtype=weight.dtype,
            ),
        )

    def close_streaming_state(self) -> None:
        self._streaming_state = None

    def forward(
        self,
        x: torch.Tensor,
        execution_context: _StreamingExecutionContext | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.num_heads, self.embed_dim // self.num_heads
        qkv = self.in_proj(x).reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, D)
        state = self._streaming_state
        if execution_context is None:
            offsets = torch.zeros(B, device=x.device, dtype=torch.long)
        else:
            if state is None:
                raise RuntimeError("MOSS Audio Tokenizer decoder state pool is not initialized.")
            offsets = state.offsets.index_select(0, execution_context.state_slot_ids)
        q, k = _apply_rope(q, k, offsets, self.max_period)

        if execution_context is not None:
            assert state is not None
            k, v, key_positions = state.kv_cache.complete(k, v, offsets, execution_context)
            query_positions = offsets.view(-1, 1, 1) + torch.arange(T, device=x.device).view(1, -1, 1)
            delta = query_positions - key_positions.view(B, 1, -1)
            mask = (key_positions.view(B, 1, -1) >= 0) & (delta >= 0)
            if self.context is not None:
                mask &= delta < self.context
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1))
            next_offsets = torch.where(execution_context.valid_rows, offsets + T, offsets)
            state.offsets.index_copy_(0, execution_context.state_slot_ids, next_offsets)
            out = out.transpose(1, 2).reshape(B, T, self.embed_dim)
            return self.out_proj(out)

        if self.context is not None and self.context < T:
            # Local-windowed causal attention: query i may only see keys in
            # [i - context + 1, i]. Matches upstream's per-stage receptive
            # field — wider context windows are not numerically equivalent.
            #
            # A dense (T, T) mask is O(T^2) memory (e.g. 4 GiB at T=65536) —
            # for long utterances decoded in a single non-streaming call this
            # OOMs. Process queries in blocks against only their reachable
            # key range so peak mask memory is O(block * (block + context))
            # regardless of T.
            block = 4096
            if T <= block:
                positions = torch.arange(T, device=x.device)
                delta = positions.view(-1, 1) - positions.view(1, -1)
                mask = (delta >= 0) & (delta < self.context)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            else:
                outs = []
                for start in range(0, T, block):
                    end = min(start + block, T)
                    k_start = max(0, start - self.context + 1)
                    q_pos = torch.arange(start, end, device=x.device)
                    k_pos = torch.arange(k_start, end, device=x.device)
                    delta = q_pos.view(-1, 1) - k_pos.view(1, -1)
                    blk_mask = (delta >= 0) & (delta < self.context)
                    out_blk = F.scaled_dot_product_attention(
                        q[:, :, start:end], k[:, :, k_start:end], v[:, :, k_start:end], attn_mask=blk_mask
                    )
                    outs.append(out_blk)
                out = torch.cat(outs, dim=2)
        else:
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
        context: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5, **kw)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5, **kw)
        self.attn = _Attention(d_model, num_heads, causal, max_period, context, device, dtype)
        self.ff1 = nn.Linear(d_model, dim_feedforward, bias=False, **kw)
        self.ff2 = nn.Linear(dim_feedforward, d_model, bias=False, **kw)
        ls_kw = cast(dict[str, object], kw)
        if layer_scale is not None:
            self.ls1 = _LayerScale(d_model, layer_scale, **ls_kw)
            self.ls2 = _LayerScale(d_model, layer_scale, **ls_kw)
        else:
            self.ls1 = nn.Identity()
            self.ls2 = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        execution_context: _StreamingExecutionContext | None = None,
    ) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), execution_context))
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
        context: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TransformerLayer(
                    d_model, num_heads, dim_feedforward, causal, max_period, layer_scale, context, device, dtype
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        execution_context: _StreamingExecutionContext | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, execution_context)
        return x


class _ProjectedTransformer(nn.Module):
    """Input/output projection + causal transformer (conv-layout: B, D, T)."""

    def __init__(
        self, input_dimension: int, output_dimension: int, d_model: int, *, module_type: str, **kwargs: Any
    ) -> None:
        super().__init__()
        self.downsample_ratio: int = 1
        self.in_proj = nn.Linear(input_dimension, d_model, bias=False) if input_dimension != d_model else nn.Identity()
        self.transformer = _Transformer(d_model=d_model, **kwargs)
        self.out_proj = (
            nn.Linear(d_model, output_dimension, bias=False) if output_dimension != d_model else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        execution_context: _StreamingExecutionContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.in_proj(x.transpose(1, 2))  # (B, D, T) → (B, T, d_model)
        x = self.transformer(x, execution_context)
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
    specs: list[dict[str, Any]], is_downsample: bool, default_context_duration: float, frame_rate: float
) -> tuple[nn.ModuleList, float]:
    """Build encoder/decoder stage modules.

    ``frame_rate`` tracks the running frames-per-second at the current stage
    (updated after each module by its downsample/upsample ratio, mirroring
    upstream's ``current_frame_rate``). Each ``Transformer`` module's local
    attention window (``context``, in frames) is derived from its own
    ``context_duration`` (seconds) at the rate *at that point in the stack* —
    not a global constant — so it must be computed here, not by the caller.
    Returns the final rate too, since the decoder's starting rate is the
    encoder's bottleneck rate, not the raw sampling rate.
    """
    modules: list[nn.Module] = []
    rate = float(frame_rate)
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
            context_duration = float(spec.pop("context_duration", default_context_duration))
            m = _ProjectedTransformer(
                module_type="Transformer",
                causal=True,
                max_period=spec.pop("max_period", 10000),
                layer_scale=spec.pop("layer_scale", None),
                context=int(round(rate * context_duration)),
                **spec,
            )
        modules.append(m)
        rate = rate / m.downsample_ratio if is_downsample else rate * m.downsample_ratio
    return nn.ModuleList(modules), rate


class MossAudioTokenizerModel(PreTrainedModel):
    """MOSS Audio Tokenizer inference model with stateful decoder streaming."""

    config_class = MossAudioTokenizerConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = False

    def __init__(self, config: MossAudioTokenizerConfig) -> None:
        super().__init__(config)
        self.sampling_rate = config.sampling_rate
        self.downsample_rate = config.downsample_rate
        # Real v1 checkpoints store these as an explicit JSON `null` rather
        # than omitting the key, so `getattr(..., default)` doesn't apply —
        # normalize None to the same defaults here.
        self.number_channels = getattr(config, "number_channels", 1) or 1
        self.enable_channel_interleave = getattr(config, "enable_channel_interleave", True)
        if self.enable_channel_interleave is None:
            self.enable_channel_interleave = True

        ctx = config.causal_transformer_context_duration
        channel_interleave_factor = (
            self.number_channels if (self.enable_channel_interleave and self.number_channels > 1) else 1
        )
        frame_rate = float(config.sampling_rate) * channel_interleave_factor
        self.encoder, bottleneck_rate = _build_modules(
            config.encoder_kwargs, is_downsample=True, default_context_duration=ctx, frame_rate=frame_rate
        )
        self.decoder, _ = _build_modules(
            copy.deepcopy(config.decoder_kwargs),
            is_downsample=False,
            default_context_duration=ctx,
            frame_rate=bottleneck_rate,
        )

        kw = dict(config.quantizer_kwargs)
        self.quantizer = _ResidualQ(**kw)
        self._decoder_streaming_attentions: list[_Attention] = []
        self._decoder_slot_offsets: torch.Tensor | None = None
        self._decoder_state_capacity = 0
        self.post_init()

    def initialize_decoder_state_pool(self, state_capacity: int, scratch_capacity: int = 0) -> None:
        if state_capacity <= 0 or scratch_capacity < 0:
            raise ValueError(f"Invalid decoder state capacities: state={state_capacity}, scratch={scratch_capacity}.")
        if self._decoder_streaming_attentions:
            raise RuntimeError("MOSS Audio Tokenizer decoder state pool is already initialized.")
        attentions = [module for module in self.decoder.modules() if isinstance(module, _Attention)]
        if not attentions:
            raise RuntimeError("MOSS Audio Tokenizer decoder has no streaming attention layers.")
        total_capacity = state_capacity + scratch_capacity
        offsets = torch.zeros(
            (len(attentions), total_capacity),
            device=next(self.parameters()).device,
            dtype=torch.long,
        )
        for index, attention in enumerate(attentions):
            attention.initialize_streaming_state(offsets[index])
        self._decoder_streaming_attentions = attentions
        self._decoder_slot_offsets = offsets
        self._decoder_state_capacity = total_capacity

    def close_decoder_state_pool(self) -> None:
        for attention in self._decoder_streaming_attentions:
            attention.close_streaming_state()
        self._decoder_streaming_attentions = []
        self._decoder_slot_offsets = None
        self._decoder_state_capacity = 0

    def reset_decoder_state_slots(self, state_slot_ids: torch.Tensor) -> None:
        if not self._decoder_streaming_attentions or self._decoder_slot_offsets is None:
            raise RuntimeError("MOSS Audio Tokenizer decoder state pool is not initialized.")
        if state_slot_ids.numel() == 0:
            return
        if state_slot_ids.device.type == "cpu":
            min_slot = int(state_slot_ids.min())
            max_slot = int(state_slot_ids.max())
            if min_slot < 0 or max_slot >= self._decoder_state_capacity:
                raise ValueError(
                    f"Decoder state slots must be in [0, {self._decoder_state_capacity}), got {state_slot_ids.tolist()}"
                )
        slots = state_slot_ids.to(device=self._decoder_slot_offsets.device, dtype=torch.long)
        # A zero offset marks every old ring-cache entry invalid, so the large
        # KV buffers do not need to be cleared when a slot is reused.
        self._decoder_slot_offsets.index_fill_(1, slots, 0)

    def decode_streaming_batch(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor,
        state_slot_ids: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> MossAudioTokenizerDecoderOutput:
        if not self._decoder_streaming_attentions:
            raise RuntimeError("MOSS Audio Tokenizer decoder state pool is not initialized.")
        execution_context = _StreamingExecutionContext(state_slot_ids, valid_rows)
        execution_context.validate(
            batch_size=int(codes.shape[1]),
            state_capacity=self._decoder_state_capacity,
            device=codes.device,
        )
        audio, audio_lengths = self._decode_tensors(codes, codes_lengths, execution_context)
        return MossAudioTokenizerDecoderOutput(audio=audio, audio_lengths=audio_lengths)

    def decode_streaming_tensors(
        self,
        codes: torch.Tensor,
        codes_lengths: torch.Tensor,
        state_slot_ids: torch.Tensor,
        valid_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        execution_context = _StreamingExecutionContext(state_slot_ids, valid_rows)
        return self._decode_tensors(codes, codes_lengths, execution_context)

    @torch.no_grad()
    def batch_encode(
        self,
        wav_list: list[torch.Tensor],
        num_quantizers: int | None = None,
    ) -> MossAudioTokenizerEncoderOutput:
        """Encode a list of waveform tensors → RVQ codes (NQ, B, T).

        Each element of ``wav_list`` is ``(T,)``/``(1, T)`` for a mono codec
        (``number_channels == 1``) or ``(number_channels, T)`` otherwise.
        """
        device = wav_list[0].device
        B = len(wav_list)
        C = self.number_channels
        normalized: list[torch.Tensor] = []
        lengths = torch.zeros(B, device=device, dtype=torch.long)
        for i, w in enumerate(wav_list):
            if C == 1:
                w_i = w.unsqueeze(0) if w.dim() == 1 else w
            else:
                if w.dim() != 2 or w.shape[0] != C:
                    raise ValueError(f"Expected wav_list[{i}] to have shape ({C}, T), got {tuple(w.shape)}.")
                w_i = w
            normalized.append(w_i)
            lengths[i] = w_i.shape[-1]
        max_len = int(lengths.max().item())
        x = torch.zeros(B, C, max_len, device=device, dtype=wav_list[0].dtype)
        for i, w_i in enumerate(normalized):
            x[i, :, : w_i.shape[-1]] = w_i
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

    def _flatten_channels_for_codec(
        self,
        input_values: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad to a ``downsample_rate`` multiple, then interleave channels into one stream.

        Mirrors upstream's ``_flatten_channels_for_codec``: for a stereo
        codec, ``(B, C, T)`` becomes ``(B, 1, T*C)`` with samples interleaved
        frame-by-frame across channels, so the mono encoder/decoder stack can
        process a multi-channel waveform unchanged.
        """
        if input_values.shape[-1] % self.downsample_rate != 0:
            pad_length = self.downsample_rate - (input_values.shape[-1] % self.downsample_rate)
            input_values = F.pad(input_values, (0, pad_length))
        if self.number_channels > 1 and self.enable_channel_interleave:
            input_values = input_values.transpose(1, 2).contiguous().view(input_values.shape[0], 1, -1)
            input_lengths = input_lengths * self.number_channels
        return input_values, input_lengths

    def _restore_channels_from_codec(
        self,
        output_values: torch.Tensor,
        output_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse of ``_flatten_channels_for_codec``: de-interleave back to ``(B, C, T)``."""
        if self.number_channels == 1 or not self.enable_channel_interleave:
            return output_values.float(), output_lengths
        output_values = (
            output_values.squeeze(1)
            .contiguous()
            .view(output_values.shape[0], -1, self.number_channels)
            .transpose(1, 2)
            .contiguous()
            .float()
        )
        output_lengths = torch.div(output_lengths, self.number_channels, rounding_mode="floor")
        return output_values, output_lengths

    def _encode(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        n_quantizers: int | None = None,
    ) -> MossAudioTokenizerEncoderOutput:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x, lengths = self._flatten_channels_for_codec(x, lengths)
        e, e_len = x, lengths
        for m in self.encoder:
            e, e_len = m(e, e_len)
        _, codes, code_len = self.quantizer.encode(e, e_len, n_quantizers)
        return MossAudioTokenizerEncoderOutput(audio_codes=codes, audio_codes_lengths=code_len)

    def _decode(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        execution_context: _StreamingExecutionContext | None = None,
    ) -> MossAudioTokenizerDecoderOutput:
        audio, audio_lengths = self._decode_tensors(codes, lengths, execution_context)
        return MossAudioTokenizerDecoderOutput(audio=audio, audio_lengths=audio_lengths)

    def _decode_tensors(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        execution_context: _StreamingExecutionContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.quantizer.decode_codes(codes)
        d, d_len = z, lengths
        for m in self.decoder:
            if isinstance(m, _ProjectedTransformer):
                d, d_len = m(d, d_len, execution_context)
            else:
                d, d_len = m(d, d_len)
        d, d_len = self._restore_channels_from_codec(d, d_len)
        return d, d_len


__all__ = [
    "MossAudioTokenizerConfig",
    "MossAudioTokenizerModel",
    "MossAudioTokenizerEncoderOutput",
    "MossAudioTokenizerDecoderOutput",
]
