# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The vLLM-Omni team.
# Copyright (c) rednote-hilab. All rights reserved.
# Adapted from:
# https://github.com/rednote-hilab/dots.tts/tree/a393d2e/src/dots_tts/modules/backbone
#
# Single upstream source (preserved as-is, inference behaviour unchanged;
# only the internal `from dots_tts.modules.backbone.layers import ...` line
# was folded to point at our sibling vendor files):
#   - modules/backbone/semantic_encoder.py -> SemanticEncoderDecodeState,
#                                             TransformerEncoderLayer,
#                                             SuperviseEncoder,
#                                             VAESemanticEncoder
#
# Building blocks (Conv1d / Mlp / MultiHeadAttention) are imported from the
# sibling vendor files where they already live:
#   - Conv1d              <- dots_tts_vocoder.py
#   - Mlp                 <- dots_tts_dit.py
#   - MultiHeadAttention  <- dots_tts_dit.py
#
# Nothing else in this file diverges from upstream behaviour.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vllm_omni.model_executor.models.dots_tts.dots_tts_dit import (
    Mlp,
    MultiHeadAttention,
)
from vllm_omni.model_executor.models.dots_tts.dots_tts_vocoder import Conv1d


@dataclass
class SemanticEncoderDecodeState:
    conv_tail: torch.Tensor
    layer_caches: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    seq_len: int


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads=16,
        ffn_hidden_size=4096,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        norm_layer="LayerNorm",
        **kwargs,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            hidden_size,
            num_heads,
            attn_drop=attn_dropout,
            norm_layer=norm_layer,
            **kwargs,
        )
        norm_cls = getattr(nn, norm_layer)
        self.attn_norm = norm_cls(hidden_size)
        self.ffn = Mlp(hidden_size, ffn_hidden_size, dropout=ffn_dropout, act_layer=nn.SiLU)
        self.ffn_norm = norm_cls(hidden_size)
        self.hidden_size = hidden_size

    def _build_causal_mask(self, T: int, device):  # noqa: N803
        return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    def _build_padding_mask(self, x_lens, max_len: int, device):
        B = x_lens.size(0)
        positions = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        return positions < x_lens.unsqueeze(1)

    def _fuse_attn_mask(self, causal_mask, padding_mask):
        if causal_mask is None and padding_mask is None:
            return None
        if causal_mask is None:
            row = padding_mask.unsqueeze(2)
            col = padding_mask.unsqueeze(1)
            return row & col
        if padding_mask is None:
            return causal_mask.unsqueeze(0)

        _B, _T = padding_mask.shape
        causal = causal_mask.unsqueeze(0)
        row = padding_mask.unsqueeze(2)
        col = padding_mask.unsqueeze(1)
        pad_2d = row & col
        return causal & pad_2d

    def forward(
        self,
        x,
        x_lens=None,
        causal=True,
    ):
        _B, T, C = x.shape
        assert self.hidden_size == C
        device = x.device

        causal_mask = self._build_causal_mask(T, device) if causal else None
        if x_lens is not None:
            padding_mask = self._build_padding_mask(x_lens, T, device)
        else:
            padding_mask = None
        fused_mask = self._fuse_attn_mask(causal_mask, padding_mask)

        h = self.attn_norm(x)
        h = self.attn(
            q=h,
            mask=fused_mask,
        )
        x = x + h

        h = self.ffn_norm(x)
        h = self.ffn(h)
        return x + h

    def decode_step(
        self,
        x,
        *,
        cache: tuple[torch.Tensor, torch.Tensor],
        positions: torch.Tensor,
    ):
        if x.size(1) <= 0:
            raise ValueError("TransformerEncoderLayer.decode_step expects a non-empty input.")

        h = self.attn_norm(x)
        h, cache = self.attn.decode_step(h, cache=cache, positions=positions)
        x = x + h

        h = self.ffn_norm(x)
        h = self.ffn(h)
        return x + h, cache


class SuperviseEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 1024)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_size=self.hidden_size,
                    num_heads=config.get("num_heads", 16),
                    ffn_hidden_size=config.get("ffn_hidden_size", 4096),
                    norm_layer=config.get("norm_layer", "LayerNorm"),
                )
                for _ in range(config.get("num_layers", 6))
            ]
        )
        self.causal = config.get("causal", False)

    def forward(self, x, x_lens=None):
        batch_size, seq_len, _ = x.shape
        if x_lens is None:
            x_lens = torch.full((batch_size,), seq_len, device=x.device, dtype=torch.long)
        for layer in self.layers:
            x = layer(x, x_lens=x_lens, causal=self.causal)
        return x

    def init_decode_state(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        layer_caches = []
        for layer in self.layers:
            cache_shape = (
                batch_size,
                layer.attn.num_heads,
                max_seq_len,
                layer.attn.head_dim,
            )
            layer_caches.append(
                (
                    torch.zeros(cache_shape, dtype=dtype, device=device),
                    torch.zeros(cache_shape, dtype=dtype, device=device),
                )
            )
        return tuple(layer_caches)

    def reset_decode_state(
        self,
        layer_caches: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> None:
        if len(layer_caches) != len(self.layers):
            raise ValueError("Layer cache count does not match encoder depth.")
        for key_cache, value_cache in layer_caches:
            key_cache.zero_()
            value_cache.zero_()

    def decode_step(self, x, *, layer_caches, positions: torch.Tensor):
        if len(layer_caches) != len(self.layers):
            raise ValueError("Layer cache count does not match encoder depth.")

        for layer, cache in zip(self.layers, layer_caches, strict=True):
            x, _ = layer.decode_step(x, cache=cache, positions=positions)
        return x


class VAESemanticEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, config):
        super().__init__()
        in_ds_rate = 2
        self.patch_size = int(config.patch_size)
        self.in_ds_rate = in_ds_rate
        self.ds_proj = Conv1d(in_dim, in_dim, kernel_size=in_ds_rate, stride=in_ds_rate, causal=True)
        self.in_proj = nn.Linear(in_dim, config.PatchEncoder.hidden_size)
        self.encoder = SuperviseEncoder(config.PatchEncoder)
        self.out_ds_rate = self.patch_size // in_ds_rate
        self.out_proj = nn.Linear(config.PatchEncoder.hidden_size * self.out_ds_rate, out_dim)

    def forward(self, x, x_lens=None):
        x = self._downsample(x)
        x = self.in_proj(x)
        z = self.encoder(x, x_lens=x_lens)
        return self._project_embeddings(z)

    def init_decode_state(
        self,
        *,
        max_audio_patch_count: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SemanticEncoderDecodeState:
        return SemanticEncoderDecodeState(
            conv_tail=torch.zeros(
                (batch_size, self.ds_proj.in_channels, self.ds_proj.left_padding),
                dtype=dtype,
                device=device,
            ),
            layer_caches=self.encoder.init_decode_state(
                batch_size=batch_size,
                max_seq_len=max_audio_patch_count * self.out_ds_rate,
                device=device,
                dtype=dtype,
            ),
            seq_len=0,
        )

    def reset_decode_state(self, state: SemanticEncoderDecodeState) -> None:
        state.conv_tail.zero_()
        self.encoder.reset_decode_state(state.layer_caches)
        state.seq_len = 0

    def prefill(
        self,
        x,
        state: SemanticEncoderDecodeState,
    ) -> tuple[torch.Tensor, SemanticEncoderDecodeState]:
        if x.ndim != 3:
            raise ValueError(f"VAESemanticEncoder.prefill expects rank-3 input, got {tuple(x.shape)}.")
        if x.size(1) % self.patch_size != 0:
            raise ValueError(f"Prompt latent length {x.size(1)} must be divisible by patch_size={self.patch_size}.")

        if x.size(1) == 0:
            return (
                x.new_zeros((x.size(0), 0, self.out_proj.out_features)),
                state,
            )
        if state.conv_tail.size(0) != x.size(0):
            raise ValueError("VAESemanticEncoder.prefill batch size does not match decode state.")

        step_inputs = self.in_proj(self._downsample(x))
        expected_token_count = (x.size(1) // self.patch_size) * self.out_ds_rate
        if step_inputs.size(1) != expected_token_count:
            raise RuntimeError(
                "Patch encoder prefill produced an unexpected token count: "
                f"expected={expected_token_count} actual={step_inputs.size(1)}."
            )

        current_seq_len = state.seq_len
        next_seq_len = current_seq_len + step_inputs.size(1)
        cache_capacity = state.layer_caches[0][0].size(2)
        if next_seq_len > cache_capacity:
            raise ValueError(
                "Patch encoder prefill exceeds decode-state capacity: "
                f"required={next_seq_len} capacity={cache_capacity}."
            )

        positions = torch.arange(step_inputs.size(1), device=x.device, dtype=torch.long) + current_seq_len
        encoded = self.encoder.decode_step(
            step_inputs,
            layer_caches=state.layer_caches,
            positions=positions,
        )
        embedding = self._project_embeddings(encoded)
        raw = x.transpose(1, 2)
        state.conv_tail.copy_(raw[..., -self.ds_proj.left_padding :])
        state.seq_len = next_seq_len
        return embedding, state

    def decode_patch(
        self,
        latent_patch,
        conv_tail: torch.Tensor,
        layer_caches: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_patch.ndim != 3:
            raise ValueError(f"VAESemanticEncoder.decode_patch expects rank-3 input, got {tuple(latent_patch.shape)}.")
        if latent_patch.size(1) != self.patch_size:
            raise ValueError(f"decode_patch expects patch length {self.patch_size}, got {latent_patch.size(1)}.")
        if positions.ndim != 1 or positions.size(0) != self.out_ds_rate:
            raise ValueError("decode_patch positions must be a rank-1 tensor matching out_ds_rate.")

        step_inputs, conv_tail = self._downsample_step(
            latent_patch,
            conv_tail=conv_tail,
        )
        if step_inputs.size(1) != self.out_ds_rate:
            raise RuntimeError(f"Downsample step produced {step_inputs.size(1)} tokens, expected {self.out_ds_rate}.")

        encoded = self.encoder.decode_step(
            step_inputs,
            layer_caches=layer_caches,
            positions=positions,
        )
        embedding = self._project_embeddings(encoded)
        return embedding, conv_tail

    def _downsample(self, x):
        return self.ds_proj(x.transpose(1, 2)).transpose(1, 2)

    def _project_embeddings(self, z):
        if self.out_ds_rate > 1:
            z = rearrange(z, "b (s d) h -> b s (d h)", d=self.out_ds_rate)
        return self.out_proj(z)

    def _downsample_step(self, latent_patch, *, conv_tail):
        raw = latent_patch.transpose(1, 2)
        conv_input = torch.cat([conv_tail, raw], dim=-1)

        projected = F.conv1d(
            conv_input,
            self.ds_proj.weight,
            self.ds_proj.bias,
            stride=self.ds_proj.stride[0],
            padding=0,
            dilation=self.ds_proj.dilation[0],
            groups=self.ds_proj.groups,
        ).transpose(1, 2)
        new_conv_tail = raw[..., -self.ds_proj.left_padding :]
        return self.in_proj(projected), new_conv_tail
