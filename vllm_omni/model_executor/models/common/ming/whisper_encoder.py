# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright 2025 The vLLM-Omni team.
# Copyright 2024 ANT Group and the HuggingFace Inc. team.
# Copyright (c) 2022 OpenAI
"""Whisper encoder shared by Ming-family models."""

import operator
from collections.abc import Iterable
from itertools import accumulate

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.utils.fa import (
    HAS_FLASH_ATTN,
    flash_attn_varlen_func,
)
from vllm_omni.model_executor.models.whisper_utils import Conv1d, Linear, sinusoids

logger = init_logger(__name__)


class MultiHeadAttention(nn.Module):
    """Whisper attention supporting dense padded and packed inputs."""

    def __init__(self, n_state: int, n_head: int, use_flash_attn: bool = True):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

        if use_flash_attn and not HAS_FLASH_ATTN:
            logger.warning_once("flash-attn is unavailable; packed Ming audio uses a PyTorch fallback")
        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim == 3:
            return self._forward_dense(x)
        if x.ndim != 2 or cu_seqlens is None:
            raise ValueError("Attention expects [batch, time, hidden] or packed [tokens, hidden] with cu_seqlens")
        return self._forward_packed(x, cu_seqlens)

    def _forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_state = x.shape
        head_dim = n_state // self.n_head

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, self.n_head, head_dim).transpose(1, 2)

        q = split_heads(self.query(x))
        k = split_heads(self.key(x))
        v = split_heads(self.value(x))
        output = F.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, n_state)
        return self.out(output)

    def _forward_packed(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        n_ctx, n_state = q.shape
        head_dim = n_state // self.n_head
        q = q.view(n_ctx, self.n_head, head_dim)
        k = k.view(n_ctx, self.n_head, head_dim)
        v = v.view(n_ctx, self.n_head, head_dim)

        if self.use_flash_attn and q.dtype in (torch.float16, torch.bfloat16):
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
            output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
            )
        else:
            output = self._manual_packed_attention(q, k, v, cu_seqlens)

        return self.out(output.contiguous().view(n_ctx, n_state))

    @staticmethod
    def _manual_packed_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        batch_size = len(seqlens)
        max_seqlen = max(seqlens)
        _, n_head, head_dim = q.shape

        q_padded = q.new_zeros(batch_size, max_seqlen, n_head, head_dim)
        k_padded = k.new_zeros(batch_size, max_seqlen, n_head, head_dim)
        v_padded = v.new_zeros(batch_size, max_seqlen, n_head, head_dim)
        for index, seq_len in enumerate(seqlens):
            start = int(cu_seqlens[index])
            end = int(cu_seqlens[index + 1])
            q_padded[index, :seq_len] = q[start:end]
            k_padded[index, :seq_len] = k[start:end]
            v_padded[index, :seq_len] = v[start:end]

        q_padded = q_padded.transpose(1, 2)
        k_padded = k_padded.transpose(1, 2)
        v_padded = v_padded.transpose(1, 2)
        padding_mask = (
            torch.arange(max_seqlen, device=q.device)[None, :] >= torch.tensor(seqlens, device=q.device)[:, None]
        )
        attn_mask = torch.zeros(batch_size, 1, 1, max_seqlen, dtype=q.dtype, device=q.device)
        attn_mask.masked_fill_(padding_mask[:, None, None, :], -torch.finfo(q.dtype).max)
        context = F.scaled_dot_product_attention(q_padded, k_padded, v_padded, attn_mask=attn_mask)
        context = context.transpose(1, 2)
        return torch.cat(
            [context[index, :seq_len] for index, seq_len in enumerate(seqlens)],
            dim=0,
        )


class ResidualAttentionBlock(nn.Module):
    """Whisper residual attention block."""

    def __init__(self, n_state: int, n_head: int, use_flash_attn: bool = True):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head, use_flash_attn=use_flash_attn)
        self.attn_ln = nn.LayerNorm(n_state)
        self.mlp = nn.Sequential(
            Linear(n_state, n_state * 4),
            nn.GELU(),
            Linear(n_state * 4, n_state),
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_ln(x), cu_seqlens=cu_seqlens)
        return x + self.mlp(self.mlp_ln(x))


class WhisperAudioEncoder(nn.Module):
    """Whisper encoder with packed Ming-2 and padded Ming-3 execution paths."""

    def __init__(
        self,
        n_mels: int = 128,
        n_ctx: int = 15000,
        n_state: int = 1280,
        n_head: int = 20,
        n_layer: int = 32,
        use_flash_attn: bool = True,
        learned_positional_embedding: bool = False,
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        positional_embedding = (
            torch.zeros(n_ctx, n_state) if learned_positional_embedding else sinusoids(n_ctx, n_state)
        )
        if learned_positional_embedding:
            self.positional_embedding = nn.Parameter(positional_embedding)
        else:
            self.register_buffer("positional_embedding", positional_embedding)
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, use_flash_attn=use_flash_attn) for _ in range(n_layer)]
        )
        self.ln_post = nn.LayerNorm(n_state)
        self.audio_emb_dim = n_state
        self.n_layer = n_layer
        self.n_mels = n_mels
        self.use_flash_attn = use_flash_attn

    def _convolve(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(x))
        return F.gelu(self.conv2(x))

    def _add_positions(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        if seq_len > self.positional_embedding.shape[0]:
            raise ValueError(
                f"Encoded audio length {seq_len} exceeds positional capacity {self.positional_embedding.shape[0]}"
            )
        return x + self.positional_embedding[:seq_len].to(x.dtype)

    def forward(
        self,
        x_list: list[torch.Tensor],
        audio_lens: list[int],
    ) -> torch.Tensor:
        """Encode variable-sized features in the packed Ming-2 path."""
        del audio_lens
        target_dtype = self.conv1.weight.dtype
        encoded = []
        encoded_lens = []
        for mel_spec in x_list:
            x = self._convolve(mel_spec.to(target_dtype).unsqueeze(0))
            x = self._add_positions(x.squeeze(0).transpose(0, 1))
            encoded.append(x)
            encoded_lens.append(x.shape[0])

        packed = torch.cat(encoded, dim=0)
        cu_seqlens = torch.tensor(
            list(accumulate(encoded_lens, func=operator.add, initial=0)),
            device=packed.device,
            dtype=torch.int32,
        )
        for block in self.blocks:
            packed = block(packed, cu_seqlens=cu_seqlens)
        return self.ln_post(packed)

    def forward_padded(
        self,
        features: torch.Tensor,
        feature_lengths: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        """Encode zero-padded Ming-3 chunks without an attention mask.

        This deliberately preserves the released implementation's behavior:
        the final short chunk attends over positions padded to the longest
        chunk in the batch, and is truncated only after projection.
        """
        if features.ndim != 3:
            raise ValueError("Padded features must have shape [batch, time, n_mels]")
        x = self._convolve(features.to(self.conv1.weight.dtype).transpose(1, 2)).transpose(1, 2)
        x = self._add_positions(x)
        for block in self.blocks:
            x = block(x)
        encoded_lengths = [(length + 1) // 2 for length in feature_lengths]
        return self.ln_post(x), encoded_lengths

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict: dict[str, torch.Tensor] = {
            **dict(self.named_parameters(remove_duplicate=False)),
            **dict(self.named_buffers()),
        }
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                logger.warning("Skipping unknown audio encoder weight: %s", name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
